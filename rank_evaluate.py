"""
Rank@K 排序评估脚本 (v1 - 针对 LambdaRank 模型的排序质量评估)

核心指标：
  NDCG@K  — 归一化折损累积增益，衡量排序质量的黄金标准
  MAP@K   — 平均精度均值，衡量正样本排在前面的程度
  MRR     — 平均倒数排名，衡量第一个正样本出现的位置
  Precision@K — 前 K 个预测中正样本的比例

架构与训练脚本一致:
  base_ready/base_YYYY.feather + city_pair_cache/city_pairs_YYYY.parquet
  动态 join + 交叉特征 → LightGBM predict → 排序指标

运行: python rank_evaluate.py
      python rank_evaluate.py --model output/models/model_rank.txt --years 2019 2020
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

# ── 从 rank_train.py 导入共享的特征定义, 保证与排序模型对齐 ──
from rank_train import (
    FEATS, CATS, CACHE_FEAT_COLS, CROSS_FEATS,
    load_cache_for_year, build_cross_features,
    BASE_DIR, CACHE_DIR,
)

DEFAULT_MODEL = Path("output/models/model_rank.txt")
K_LIST = [5, 10, 15, 20]


def load_year_data(year: int, sample_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        raise FileNotFoundError(f"Base feather not found: {base_path}")

    base = pd.read_feather(base_path)

    if sample_n is not None and sample_n > 0:
        unique_qids = base['qid'].unique()
        if len(unique_qids) > sample_n:
            rng = np.random.default_rng(seed)
            sampled = rng.choice(unique_qids, size=sample_n, replace=False)
            base = base[base['qid'].isin(set(sampled))].copy()

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

    base = build_cross_features(base)

    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base


def predict_chunk(model_path: str, chunk_df: pd.DataFrame) -> pd.DataFrame:
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


# ================= 排序指标计算 =================

def _dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """计算 DCG@K: sum( (2^rel - 1) / log2(pos + 1) )"""
    rel = relevance[:k]
    positions = np.arange(1, len(rel) + 1)
    return float(np.sum((2.0 ** rel - 1.0) / np.log2(positions + 1.0)))


def _ndcg_at_k(relevance: np.ndarray, ideal_relevance: np.ndarray, k: int) -> float:
    """计算 NDCG@K = DCG@K / IDCG@K"""
    dcg = _dcg_at_k(relevance, k)
    idcg = _dcg_at_k(ideal_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _ap_at_k(hits: np.ndarray, k: int) -> float:
    """计算 AP@K (Average Precision at K)"""
    hits_k = hits[:k]
    if hits_k.sum() == 0:
        return 0.0
    positions = np.arange(1, len(hits_k) + 1)
    precisions = np.cumsum(hits_k) / positions
    return float(np.sum(precisions * hits_k) / hits_k.sum())


def compute_rank_metrics(df: pd.DataFrame, pred_df: pd.DataFrame, k_list: list[int]) -> pd.DataFrame:
    """
    核心：一次推理，计算所有排序指标。

    对每个 query:
      1. 用 GT 的 Rank 构建 relevance (21 - Rank)，与训练标签一致
      2. 按模型 score 降序排列候选城市
      3. 计算 NDCG@K, MAP@K, Precision@K, MRR
    """
    # 构建 GT: qid -> {To_City: Rank}
    gt_rows = df[(df['Rank'] > 0) & (df['Rank'] <= 20)]
    gt_dict = gt_rows.groupby('qid').apply(
        lambda g: dict(zip(g['To_City'].values, g['Rank'].values))
    ).to_dict()

    max_k = max(k_list)
    pred_df = pred_df.sort_values(['qid', 'score'], ascending=[True, False])
    pred_ranked = pred_df.groupby('qid').head(max_k)

    rows = []
    for qid, group in pred_ranked.groupby('qid'):
        gt_ranks = gt_dict.get(qid, {})
        n_gt = len(gt_ranks)
        if n_gt == 0:
            continue

        pred_cities = group['To_City'].values

        # 模型预测顺序下的 relevance 序列
        pred_relevance = np.array([21 - gt_ranks[c] if c in gt_ranks else 0 for c in pred_cities], dtype=np.float64)
        # 命中标记 (是否为正样本)
        pred_hits = np.array([1 if c in gt_ranks else 0 for c in pred_cities], dtype=np.float64)
        # 理想排序: GT 的 relevance 降序排列
        ideal_relevance = np.sort(np.array([21 - r for r in gt_ranks.values()], dtype=np.float64))[::-1]

        # MRR: 第一个正样本的位置
        first_hit_positions = np.where(pred_hits > 0)[0]
        mrr = 1.0 / (first_hit_positions[0] + 1) if len(first_hit_positions) > 0 else 0.0

        row = {'qid': qid, 'gt_size': n_gt, 'mrr': mrr}

        for k in k_list:
            row[f'ndcg@{k}'] = _ndcg_at_k(pred_relevance, ideal_relevance, k)
            row[f'map@{k}'] = _ap_at_k(pred_hits, k)
            row[f'precision@{k}'] = float(pred_hits[:k].sum()) / k
            # 同时保留 recall 作为参考
            row[f'recall@{k}'] = float(pred_hits[:k].sum()) / n_gt

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_year(model_path: str, year: int, sample_n: int | None,
                  n_workers: int, seed: int = 42) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[Year {year}] Loading data...")
    df = load_year_data(year, sample_n=sample_n, seed=seed)
    n_queries = df['qid'].nunique()
    print(f"  Loaded {len(df):,} rows, {n_queries:,} queries")

    print(f"[Year {year}] Predicting with {n_workers} workers...")
    pred_df = parallel_predict(model_path, df, n_workers)

    print(f"[Year {year}] Computing ranking metrics...")
    metrics_df = compute_rank_metrics(df, pred_df, k_list=K_LIST)

    elapsed = time.time() - t0
    n_evaluated = len(metrics_df)
    avg_gt = metrics_df['gt_size'].mean()
    avg_mrr = metrics_df['mrr'].mean()

    print(f"[Year {year}] Done in {elapsed:.1f}s | Queries: {n_evaluated} | Avg GT: {avg_gt:.2f}")
    print(f"  MRR: {avg_mrr:.4f}")

    result = {
        'year': year, 'n_queries': n_evaluated,
        'avg_gt_size': avg_gt, 'mrr': avg_mrr, 'elapsed_s': elapsed,
    }

    for k in K_LIST:
        ndcg = metrics_df[f'ndcg@{k}'].mean()
        map_k = metrics_df[f'map@{k}'].mean()
        prec = metrics_df[f'precision@{k}'].mean()
        recall = metrics_df[f'recall@{k}'].mean()
        result[f'ndcg@{k}'] = ndcg
        result[f'map@{k}'] = map_k
        result[f'precision@{k}'] = prec
        result[f'recall@{k}'] = recall
        print(f"  @{k:>2}: NDCG={ndcg:.4f} | MAP={map_k:.4f} | P={prec:.4f} | R={recall:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Rank Evaluation (v1 - Ranking Quality Metrics)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020])
    parser.add_argument("--sample-n", type=int, default=10000,
                        help="Per-year query sample count, 0=full")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers, 0=auto")
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
    print(f"K values: {K_LIST}")

    results = []
    for year in args.years:
        r = evaluate_year(model_path, year, sample_n, n_workers, args.seed)
        results.append(r)

    # ── 汇总表 ──
    print(f"\n{'='*120}")
    print("Summary (Ranking Quality):")

    # NDCG 行
    header = f"{'Year':<8} {'Queries':<10} {'AvgGT':<8} {'MRR':<10}"
    for k in K_LIST:
        header += f" {'NDCG@'+str(k):<10}"
    print(header)
    print("-" * 120)

    total_q = 0
    weighted = {f'ndcg@{k}': 0.0 for k in K_LIST}
    weighted.update({f'map@{k}': 0.0 for k in K_LIST})
    weighted.update({f'recall@{k}': 0.0 for k in K_LIST})
    weighted['mrr'] = 0.0

    for r in results:
        nq = r['n_queries']
        line = f"{r['year']:<8} {nq:<10} {r['avg_gt_size']:<8.2f} {r['mrr']:<10.4f}"
        for k in K_LIST:
            line += f" {r[f'ndcg@{k}']:<10.4f}"
        print(line)

        total_q += nq
        weighted['mrr'] += r['mrr'] * nq
        for k in K_LIST:
            weighted[f'ndcg@{k}'] += r[f'ndcg@{k}'] * nq
            weighted[f'map@{k}'] += r[f'map@{k}'] * nq
            weighted[f'recall@{k}'] += r[f'recall@{k}'] * nq

    if total_q > 0:
        print("-" * 120)
        line = f"{'ALL':<8} {total_q:<10} {'':8} {weighted['mrr']/total_q:<10.4f}"
        for k in K_LIST:
            line += f" {weighted[f'ndcg@{k}']/total_q:<10.4f}"
        print(line)

    # MAP + Recall 补充表
    print(f"\n{'Year':<8} ", end="")
    for k in K_LIST:
        print(f"{'MAP@'+str(k):<10} ", end="")
    for k in K_LIST:
        print(f"{'R@'+str(k):<10} ", end="")
    print()
    print("-" * 120)
    for r in results:
        line = f"{r['year']:<8} "
        for k in K_LIST:
            line += f"{r[f'map@{k}']:<10.4f} "
        for k in K_LIST:
            line += f"{r[f'recall@{k}']:<10.4f} "
        print(line)
    if total_q > 0:
        print("-" * 120)
        line = f"{'ALL':<8} "
        for k in K_LIST:
            line += f"{weighted[f'map@{k}']/total_q:<10.4f} "
        for k in K_LIST:
            line += f"{weighted[f'recall@{k}']/total_q:<10.4f} "
        print(line)

    print("=" * 120)


if __name__ == "__main__":
    main()
