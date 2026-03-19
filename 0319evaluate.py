"""
Recall@K 评估脚本 (0319 版 - 与 0319build_bin_recall / 0319fast_recall_train 完全对齐)

架构:
  base_ready/base_YYYY.feather + city_pair_cache/city_pairs_YYYY.parquet
  cache_tensor 极速 join + 交叉特征 → LightGBM predict → Recall@5/10/20/30/40

运行: uv run 0319evaluate.py
      uv run 0319evaluate.py --model output/models/0319recall_model_final.txt --years 2019 2020
依赖: output/base_ready/, data/city_pair_cache/, 训练好的模型文件
"""

'''
评估结果
Model:    output/models/0319recall_model_final.txt
Years:    [2019, 2020]
Sample:   10000 queries/year
Workers:  28
Features: 63 (9 categorical)

============================================================
[Year 2019] Loading data...
  Loaded 3,360,000 rows, 10,000 queries
[Year 2019] Predicting with 28 workers...
[Year 2019] Computing Recall@5/10/20/30/40...
[Year 2019] Done in 108.9s
  Queries evaluated: 9792
  Avg GT cities:     13.89
  Recall@5   0.8557 (85.57%) | Avg hits: 4.16
  Recall@10  0.8369 (83.69%) | Avg hits: 7.60
  Recall@20  0.8393 (83.93%) | Avg hits: 12.23
  Recall@30  0.8800 (88.00%) | Avg hits: 12.88
  Recall@40  0.8900 (89.00%) | Avg hits: 13.01

============================================================
[Year 2020] Loading data...
  Loaded 3,360,000 rows, 10,000 queries
[Year 2020] Predicting with 28 workers...
[Year 2020] Computing Recall@5/10/20/30/40...
[Year 2020] Done in 117.5s
  Queries evaluated: 9867
  Avg GT cities:     14.26
  Recall@5   0.8509 (85.09%) | Avg hits: 4.17
  Recall@10  0.8272 (82.72%) | Avg hits: 7.59
  Recall@20  0.7997 (79.97%) | Avg hits: 11.89
  Recall@30  0.8607 (86.07%) | Avg hits: 12.91
  Recall@40  0.8781 (87.81%) | Avg hits: 13.17

==============================================================================================================
Summary:
Year     Queries    AvgGT    R@5        R@10       R@20       R@30       R@40       Time
--------------------------------------------------------------------------------------------------------------
2019     9792       13.89    0.8557     0.8369     0.8393     0.8800     0.8900     108.9   s
2020     9867       14.26    0.8509     0.8272     0.7997     0.8607     0.8781     117.5   s
--------------------------------------------------------------------------------------------------------------
ALL      19659               0.8533     0.8321     0.8194     0.8703     0.8840
==============================================================================================================
'''



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

# ═══════════════════════════════════════════════════════════════
# 特征定义 (从 0319build_bin_recall.py 原样复制, 保证完全对齐)
# 不直接 import 是因为该模块顶层有 shutil.rmtree 副作用,
# Windows spawn 子进程会重新执行导入, 导致灾难性删除。
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path("output/base_ready")
CACHE_DIR = Path("data/city_pair_cache")

PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
RATIO_FEATS = ['gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio', 'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio', 'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio', 'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio', 'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio', 'avg_commute_mins_ratio', 'population_total_ratio', 'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio', 'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio']
DIFF_FEATS = ['gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff', 'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff', 'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff']
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = ['industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio', 'education_x_tier_diff', 'income_x_gdp_ratio', 'age_x_housing_ratio', 'family_x_edu_score_ratio']
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

DEFAULT_MODEL = Path("output/models/0319recall_model_final.txt")


def load_cache_for_year(year: int):
    """与 0319build_bin_recall.load_cache_for_year 完全一致的 tensor 极速加载"""
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)

    unique_cities = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    num_cities = len(unique_cities)
    num_feats = len(CACHE_FEAT_COLS)
    MAX_CITY_ID = 10000

    city_map = np.full(MAX_CITY_ID, num_cities, dtype=np.int32)
    city_map[unique_cities] = np.arange(num_cities)

    cache_tensor = np.zeros((num_cities + 1, num_cities + 1, num_feats), dtype=np.float32)
    from_idx = city_map[df['from_city'].values]
    to_idx = city_map[df['to_city'].values]
    cache_tensor[from_idx, to_idx, :] = df[CACHE_FEAT_COLS].values.astype(np.float32)

    return cache_tensor, city_map


def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """与 0319build_bin_recall.build_cross_features 完全一致"""
    industry = df['industry'].values if hasattr(df['industry'], 'values') else df['industry'].cat.codes.values
    wage_arr = df[['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']].values.astype(np.float32)
    vacancy_arr = df[['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']].values.astype(np.float32)
    ind_idx = np.clip(industry.astype(int), 0, 3)
    rows = np.arange(len(df))
    df['industry_x_matched_wage_ratio'] = wage_arr[rows, ind_idx]
    df['industry_x_matched_vacancy_ratio'] = vacancy_arr[rows, ind_idx]
    edu = df['education'].values if hasattr(df['education'], 'values') else df['education'].cat.codes.values
    df['education_x_tier_diff'] = edu.astype(np.float32) * df['tier_diff'].values.astype(np.float32)
    inc = df['income'].values if hasattr(df['income'], 'values') else df['income'].cat.codes.values
    df['income_x_gdp_ratio'] = inc.astype(np.float32) * df['gdp_per_capita_ratio'].values
    age = df['age_group'].values if hasattr(df['age_group'], 'values') else df['age_group'].cat.codes.values
    df['age_x_housing_ratio'] = age.astype(np.float32) * df['housing_price_avg_ratio'].values
    fam = df['family'].values if hasattr(df['family'], 'values') else df['family'].cat.codes.values
    df['family_x_edu_score_ratio'] = fam.astype(np.float32) * df['education_score_ratio'].values
    return df


# ═══════════════════════════════════════════════════════════════
# 数据加载 + 推理 + 评估
# ═══════════════════════════════════════════════════════════════

def load_year_data(year: int, sample_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    """加载单年数据: base feather + cache tensor join + 交叉特征 (与训练流程完全一致)"""
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        raise FileNotFoundError(f"Base feather not found: {base_path}")

    needed_cols = ['qid', 'From_City', 'To_City', 'Rank'] + PERSON_CATS
    base = pd.read_feather(base_path, columns=needed_cols)

    # 构建 Label: Rank 1-20 为正样本, 与训练完全一致
    rank = base['Rank'].values
    label = np.zeros(len(base), dtype=np.int8)
    label[(rank >= 1) & (rank <= 20)] = 1
    base['Label'] = label

    # 采样
    if sample_n is not None and sample_n > 0:
        unique_qids = base['qid'].unique()
        if len(unique_qids) > sample_n:
            rng = np.random.default_rng(seed)
            sampled = rng.choice(unique_qids, size=sample_n, replace=False)
            base = base[base['qid'].isin(set(sampled))].reset_index(drop=True)

    # 极速 tensor join (与 0319build_bin_recall 完全一致)
    cache_tensor, city_map = load_cache_for_year(year)
    from_idx = city_map[base['From_City'].values]
    to_idx = city_map[base['To_City'].values]
    extracted_feats = cache_tensor[from_idx, to_idx, :]

    for i, col in enumerate(CACHE_FEAT_COLS):
        base[col] = extracted_feats[:, i]

    del cache_tensor, city_map, extracted_feats
    gc.collect()

    # 交叉特征 (与训练完全一致)
    base = build_cross_features(base)

    # 填充缺失 (与训练完全一致)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base


def predict_chunk(model_path: str, feat_names: list[str], cat_names: list[str],
                  chunk_df: pd.DataFrame) -> pd.DataFrame:
    """子进程: 加载模型 + predict, 返回 (qid, To_City, score)"""
    model = lgb.Booster(model_file=model_path)
    X = chunk_df[feat_names].copy()
    for c in cat_names:
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

    feat_names = list(FEATS)
    cat_names = list(CATS)

    if len(chunks) == 1:
        return predict_chunk(model_path, feat_names, cat_names, chunks[0])

    results = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(predict_chunk, model_path, feat_names, cat_names, c): i
            for i, c in enumerate(chunks)
        }
        for future in as_completed(futures):
            results.append(future.result())

    return pd.concat(results, ignore_index=True)


def compute_recall_multi_k(df: pd.DataFrame, pred_df: pd.DataFrame,
                           k_list: list[int] = [5, 10, 20, 30, 40]) -> pd.DataFrame:
    """
    一次推理，计算多个 K 值的 Recall@K。
    公式: Recall@K = |Top-K pred ∩ GT| / min(K, |GT|)
    与训练时 feval 中的 recall 计算逻辑严格一致。
    """
    gt = df[df['Label'] == 1].groupby('qid')['To_City'].apply(set).to_dict()

    max_k = max(k_list)
    pred_df = pred_df.sort_values(['qid', 'score'], ascending=[True, False])
    pred_ranked = pred_df.groupby('qid').head(max_k)

    rows = []
    for qid, group in pred_ranked.groupby('qid'):
        gt_set = gt.get(qid, set())
        n = len(gt_set)
        if n == 0:
            continue

        pred_cities = group['To_City'].values
        row = {'qid': qid, 'gt_size': n}

        for k in k_list:
            pred_top_k = set(pred_cities[:k])
            hits = len(pred_top_k & gt_set)
            row[f'hits@{k}'] = hits
            row[f'recall@{k}'] = hits / min(k, n)

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_year(model_path: str, year: int, sample_n: int | None,
                  n_workers: int, seed: int = 42) -> dict:
    """评估单年，一次推理计算 Recall@5/10/20/30/40"""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[Year {year}] Loading data...")
    df = load_year_data(year, sample_n=sample_n, seed=seed)
    n_queries = df['qid'].nunique()
    print(f"  Loaded {len(df):,} rows, {n_queries:,} queries")

    print(f"[Year {year}] Predicting with {n_workers} workers...")
    pred_df = parallel_predict(model_path, df, n_workers)

    print(f"[Year {year}] Computing Recall@5/10/20/30/40...")
    recall_df = compute_recall_multi_k(df, pred_df, k_list=[5, 10, 20, 30, 40])

    result = {
        'year': year,
        'n_queries': len(recall_df),
        'avg_gt_size': recall_df['gt_size'].mean(),
        'elapsed_s': time.time() - t0,
    }
    for k in [5, 10, 20, 30, 40]:
        result[f'recall@{k}'] = recall_df[f'recall@{k}'].mean()
        result[f'hits@{k}'] = recall_df[f'hits@{k}'].mean()

    print(f"[Year {year}] Done in {result['elapsed_s']:.1f}s")
    print(f"  Queries evaluated: {result['n_queries']}")
    print(f"  Avg GT cities:     {result['avg_gt_size']:.2f}")
    for k in [5, 10, 20, 30, 40]:
        r = result[f'recall@{k}']
        h = result[f'hits@{k}']
        print(f"  Recall@{k:<3d} {r:.4f} ({r*100:.2f}%) | Avg hits: {h:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Recall@K Evaluation (0319 版)")
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
    print(f"Features: {len(FEATS)} ({len(CATS)} categorical)")

    results = []
    for year in args.years:
        r = evaluate_year(model_path, year, sample_n, n_workers, args.seed)
        results.append(r)

    # 汇总
    k_list = [5, 10, 20, 30, 40]
    print(f"\n{'='*110}")
    print("Summary:")
    header = f"{'Year':<8} {'Queries':<10} {'AvgGT':<8}"
    for k in k_list:
        header += f" {'R@'+str(k):<10}"
    header += f" {'Time':<8}"
    print(header)
    print("-" * 110)

    total_queries = 0
    weighted_sums = {k: 0.0 for k in k_list}

    for r in results:
        line = f"{r['year']:<8} {r['n_queries']:<10} {r['avg_gt_size']:<8.2f}"
        for k in k_list:
            line += f" {r[f'recall@{k}']:<10.4f}"
        line += f" {r['elapsed_s']:<8.1f}s"
        print(line)
        total_queries += r['n_queries']
        for k in k_list:
            weighted_sums[k] += r[f'recall@{k}'] * r['n_queries']

    if total_queries > 0:
        print("-" * 110)
        line = f"{'ALL':<8} {total_queries:<10} {'':8}"
        for k in k_list:
            line += f" {weighted_sums[k]/total_queries:<10.4f}"
        print(line)
    print("=" * 110)


if __name__ == "__main__":
    main()
