"""
轻量版 Recall 评估脚本
直接从 feather 文件读取已有特征 + LightGBM predict，多核并行推理。

召回率计算逻辑修正 (Recall@20)：
  1. 对每个 query 统一推理 top20 城市（按 score 降序）
  2. 根据 GT 中实际有去向城市集合 (Label=1 的城市)，
     直接拿 pred 的前 20 个城市与 GT 集合比较命中率
  3. 公式: Recall@20 = |Top 20 ∩ GT| / len(GT)

真正关心的指标其实是 Recall@K（比如 Recall@20），而不是严格的 R-Precision（取与 GT 数量相等的头部预测去求交集）。下游的排序模型（Ranking）接收的是固定数量的候选集（比如 20 或 50），只要这 20 个坑位里包含了用户最终去的真实城市，召回阶段的任务就算圆满完成了。
你之前的代码里，pred_top_n = set(group['To_City'].iloc[:n].values) 这种写法是对模型极大的“不公平”惩罚。平均 GT 有 14 个，意味着你只让模型输出 14 个结果去撞击这 14 个答案，这太难了。放宽到 Top 20
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# ── 与 simple_train.py 完全对齐的特征列表 ──
FEATS = [
    'gender', 'age_group', 'education', 'industry', 'income', 'family',
    'geo_distance', 'dialect_distance',
    'gdp_per_capita_ratio', 'unemployment_rate_ratio', 'housing_price_avg_ratio',
    'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio',
    'education_score_ratio', 'transport_convenience_ratio', 'population_total_ratio',
    'is_same_province',
]

CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']

PROCESSED_DIR = Path("output/processed_ready")

# 核心修改 模型路径 每次更新
DEFAULT_MODEL = Path("output/models/model_iter_3050.txt")


def load_year_data(year: int, sample_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    """从 feather 加载单年数据，可选采样 sample_n 个 query。"""
    fp = PROCESSED_DIR / f"processed_{year}.feather"
    if not fp.exists():
        raise FileNotFoundError(f"Feather not found: {fp}")

    cols_needed = FEATS + ['qid', 'To_City', 'Label', 'Rank']
    df = pd.read_feather(fp, columns=cols_needed)

    if sample_n is not None and sample_n > 0:
        unique_qids = df['qid'].unique()
        if len(unique_qids) > sample_n:
            rng = np.random.default_rng(seed)
            sampled_qids = rng.choice(unique_qids, size=sample_n, replace=False)
            df = df[df['qid'].isin(set(sampled_qids))]

    return df


def predict_chunk(model_path: str, chunk_df: pd.DataFrame) -> pd.DataFrame:
    """在子进程中加载模型并对一个 chunk 进行 predict，返回 (qid, To_City, score)。"""
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
    """多进程并行推理，按 qid 分 chunk 分发到各 worker。"""
    # 按 qid 分组，均匀分配到 n_workers 个 chunk
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
    计算自适应召回率 (Recall@K)。
    对每个 query:
      - GT 集合 = Label==1 的 To_City 集合，大小为 n
      - Pred 集合 = score 降序排列的前 top_infer (默认20) 个 To_City
      - recall = |Pred ∩ GT| / n
    """
    # 构建 GT: 每个 qid 的正样本城市集合
    gt = df[df['Label'] == 1].groupby('qid')['To_City'].apply(set).to_dict()

    # 构建 Pred: 直接截断取 Top K
    pred_df = pred_df.sort_values(['qid', 'score'], ascending=[True, False])
    pred_ranked = pred_df.groupby('qid').head(top_infer)

    rows = []
    for qid, group in pred_ranked.groupby('qid'):
        gt_set = gt.get(qid, set())
        n = len(gt_set)
        if n == 0:
            continue
        
        # 【修改点】：直接用 Top K 集合去命中 GT，不再强行按 n 截断
        pred_top_k = set(group['To_City'].values)
        hits = len(pred_top_k & gt_set)
        
        rows.append({
            'qid': qid,
            'gt_size': n,
            'hits': hits,
            'recall': hits / n,
        })

    return pd.DataFrame(rows)


def evaluate_year(model_path: str, year: int, sample_n: int | None,
                  n_workers: int, seed: int = 42) -> dict:
    """评估单年，返回统计结果字典。"""
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
        'year': year,
        'n_queries': len(recall_df),
        'avg_gt_size': avg_gt_size,
        'avg_hits': avg_hits,
        'recall': avg_recall,
        'elapsed_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="轻量版 Recall 评估")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--years", nargs="+", type=int, default=[2019, 2020])
    parser.add_argument("--sample-n", type=int, default=10000,
                        help="每年采样 query 数，0 表示全量")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行 worker 数，0=自动(CPU核心数)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    n_workers = args.workers if args.workers > 0 else os.cpu_count()
    sample_n = args.sample_n if args.sample_n > 0 else None

    print(f"Model:   {model_path}")
    print(f"Years:   {args.years}")
    print(f"Sample:  {sample_n or 'ALL'} queries/year")
    print(f"Workers: {n_workers}")

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