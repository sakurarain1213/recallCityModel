"""
Recall@K 评估脚本 (v2 - 与 simple_train.py 特征完全对齐)

架构与训练脚本一致:
  base_ready/base_YYYY.feather + city_pair_cache/city_pairs_YYYY.parquet
  动态 join + 交叉特征 → LightGBM predict → Recall@20

运行: cd /data1/wxj/Recall_city_project/ && uv run evaluate.py
      uv run evaluate.py --model output/models/model_v2.txt --years 2019 2020
依赖: output/base_ready/, data/city_pair_cache/, 训练好的模型文件
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── 从 simple_train.py 导入共享的特征定义, 保证对齐 ──
from simple_train import (
    FEATS, CATS, CACHE_FEAT_COLS, CROSS_FEATS,
    load_cache_for_year, build_cross_features,
    BASE_DIR, CACHE_DIR,
)

DEFAULT_MODEL = Path("output/models/model_v2.txt")


def load_year_data(year: int, sample_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    """加载单年数据: base feather + cache join + 交叉特征"""
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        raise FileNotFoundError(f"Base feather not found: {base_path}")

    base = pd.read_feather(base_path)

    # 采样
    if sample_n is not None and sample_n > 0:
        unique_qids = base['qid'].unique()
        if len(unique_qids) > sample_n:
            rng = np.random.default_rng(seed)
            sampled = rng.choice(unique_qids, size=sample_n, replace=False)
            base = base[base['qid'].isin(set(sampled))].copy()

    # join cache
    cache = load_cache_for_year(year)
    base = base.merge(
        cache,
        left_on=['From_City', 'To_City'],
        right_on=['from_city', 'to_city'],
        how='left',
    )
    base.drop(columns=['from_city', 'to_city'], inplace=True)
    del cache
    gc.collect()

    # 交叉特征
    base = build_cross_features(base)

    # 填充缺失
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base


def predict_chunk(model_path: str, chunk_df: pd.DataFrame) -> pd.DataFrame:
    """子进程: 加载模型 + predict, 返回 (qid, To_City, score)"""
    model = lgb.Booster(model_file=model_path)
    X = chunk_df[FEATS].copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    scores = model.predict(X)
    return pd.DataFrame({
        'qid': chunk_df['qid'].values,
        'To_City': chunk_df['To_City'].values,
        'score': scores,
    })


def parallel_predict(model_path: str, df: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    """多进程并行推理"""
    unique_qids = df['qid'].unique()
    qid_splits = np.array_split(unique_qids, n_workers)

    chunks = []
    for qid_subset in qid_splits:
        if len(qid_subset) == 0:
            continue
        mask = df['qid'].isin(set(qid_subset))
        chunks.append(df[mask].copy())

    if len(chunks) == 1:
        return predict_chunk(model_path, chunks[0])

    results = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {executor.submit(predict_chunk, model_path, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            results.append(future.result())

    return pd.concat(results, ignore_index=True)


def compute_recall(df: pd.DataFrame, pred_df: pd.DataFrame, top_infer: int = 20) -> pd.DataFrame:
    """
    Recall@K: 对每个 query 取 score top_infer 个预测, 与 GT 求交集。
    recall = |Top K ∩ GT| / |GT|
    """
    gt = df[df['Label'] == 1].groupby('qid')['To_City'].apply(set).to_dict()

    pred_df = pred_df.sort_values(['qid', 'score'], ascending=[True, False])
    pred_ranked = pred_df.groupby('qid').head(top_infer)

    rows = []
    for qid, group in pred_ranked.groupby('qid'):
        gt_set = gt.get(qid, set())
        n = len(gt_set)
        if n == 0:
            continue
        pred_top_k = set(group['To_City'].values)
        hits = len(pred_top_k & gt_set)
        rows.append({'qid': qid, 'gt_size': n, 'hits': hits, 'recall': hits / n})

    return pd.DataFrame(rows)


def evaluate_year(model_path: str, year: int, sample_n: int | None,
                  n_workers: int, seed: int = 42) -> dict:
    """评估单年"""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[Year {year}] Loading data...")
    df = load_year_data(year, sample_n=sample_n, seed=seed)
    n_queries = df['qid'].nunique()
    print(f"  Loaded {len(df):,} rows, {n_queries:,} queries")

    print(f"[Year {year}] Predicting with {n_workers} workers...")
    pred_df = parallel_predict(model_path, df, n_workers)

    print(f"[Year {year}] Computing Recall@20...")
    recall_df = compute_recall(df, pred_df, top_infer=20)

    avg_recall = recall_df['recall'].mean()
    avg_gt_size = recall_df['gt_size'].mean()
    avg_hits = recall_df['hits'].mean()
    elapsed = time.time() - t0

    print(f"[Year {year}] Done in {elapsed:.1f}s")
    print(f"  Queries evaluated: {len(recall_df)}")
    print(f"  Avg GT cities:     {avg_gt_size:.2f}")
    print(f"  Avg hits:          {avg_hits:.2f}")
    print(f"  Avg Recall@20:     {avg_recall:.4f} ({avg_recall*100:.2f}%)")

    return {
        'year': year, 'n_queries': len(recall_df),
        'avg_gt_size': avg_gt_size, 'avg_hits': avg_hits,
        'recall': avg_recall, 'elapsed_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Recall@K Evaluation (v2)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020])
    parser.add_argument("--sample-n", type=int, default=10000,
                        help="每年采样 query 数, 0=全量")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行 worker 数, 0=auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    n_workers = args.workers if args.workers > 0 else os.cpu_count()
    sample_n = args.sample_n if args.sample_n > 0 else None

    print(f"Model:    {model_path}")
    print(f"Years:    {args.years}")
    print(f"Sample:   {sample_n or 'ALL'} queries/year")
    print(f"Workers:  {n_workers}")
    print(f"Features: {len(FEATS)}")

    results = []
    for year in args.years:
        r = evaluate_year(model_path, year, sample_n, n_workers, args.seed)
        results.append(r)

    # 汇总
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Year':<8} {'Queries':<10} {'AvgGT':<8} {'AvgHits':<10} {'Recall@20':<12} {'Time':<8}")
    print("-" * 60)
    total_queries = 0
    weighted_recall_sum = 0.0
    for r in results:
        print(f"{r['year']:<8} {r['n_queries']:<10} {r['avg_gt_size']:<8.2f} "
              f"{r['avg_hits']:<10.2f} {r['recall']:<12.4f} {r['elapsed_s']:<8.1f}s")
        total_queries += r['n_queries']
        weighted_recall_sum += r['recall'] * r['n_queries']

    if total_queries > 0:
        overall = weighted_recall_sum / total_queries
        print("-" * 60)
        print(f"{'ALL':<8} {total_queries:<10} {'':8} {'':10} {overall:<12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
