"""
0322 LambdaRank Pipeline (全量自包含)

上传到服务器后只需这一个脚本，改路径即可运行完整流程。

Usage:
  uv run 0322pipeline.py --step bin,train,eval
  uv run 0322pipeline.py --step bin,train
  uv run 0322pipeline.py --step eval --model output/models/0322ltr_model.txt

服务器迁移: 只需修改顶部的 DB_PATH / CACHE_DIR 路径即可

uv run 0322pipeline.py --step train --train-query-ratio 0.3 --train-neg-ratio 0.8
"""

import gc
import re
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# 路径配置 (自动检测平台，服务器迁移无需修改)
# ═══════════════════════════════════════════════════════════════
import os as _os
if _os.name == 'nt':
    # Windows 本地
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    BIN_DIR = Path("output/ltr_bin")
    MODEL_DIR = Path("output/models")
else:
    # 服务器 Linux
    DB_PATH = Path("/data1/wxj/Recall_city_project/data/local_migration_data.db")
    CACHE_DIR = Path("/data1/wxj/Recall_city_project/data/city_pair_cache")
    BIN_DIR = Path("/data2/wxj/ltr_bin")
    MODEL_DIR = Path("/data1/wxj/Recall_city_project/output/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════════════════
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
RATIO_FEATS = [
    'gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio',
    'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio',
    'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio',
    'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio',
    'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio',
    'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio',
    'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio',
    'avg_commute_mins_ratio', 'population_total_ratio',
    'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio',
    'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio',
]
DIFF_FEATS = [
    'gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff',
    'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff',
    'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff',
]
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = [
    'industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff', 'income_x_gdp_ratio',
    'age_x_housing_ratio', 'family_x_edu_score_ratio',
]
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# TypeID 解析
AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}


def parse_type_id(tid: str) -> tuple:
    parts = tid.rsplit('_', 1)
    fc = int(parts[1])
    a = parts[0].split('_')
    return (GENDER_MAP[a[0]], AGE_MAP[a[1]], EDU_MAP[a[2]],
            IND_MAP[a[3]], INC_MAP[a[4]], FAM_MAP[a[5]], fc)


def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', s)
    return int(m.group(1)) if m else int(s)


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_cache(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)

    all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    max_id = int(all_ids.max())
    n_feat = len(CACHE_FEAT_COLS)

    city_map = np.zeros(max_id + 1, dtype=np.int32)
    city_map[all_ids] = np.arange(len(all_ids))

    tensor = np.zeros((max_id + 1, max_id + 1, n_feat), dtype=np.float32)

    # 只对有效数据进行原地 NaN 替换，避免处理 21.8 亿个格子
    valid_features = df[CACHE_FEAT_COLS].values.astype(np.float32)
    np.nan_to_num(valid_features, copy=False, nan=0.0)
    tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = valid_features

    to_dict = df.groupby('from_city')['to_city'].apply(
        lambda x: x.values.astype(np.int32)).to_dict()

    return tensor, city_map, to_dict


def build_year_data(year: int, sample_n: int = None, neg_sample_n: int = 200, seed: int = 42):
    """从 DB 加载 year 数据，返回 (X, labels, qids, n_queries)

    Args:
        neg_sample_n: 每个 query 保留多少个负样本 (默认 200)。设为 None 则保留全部 (336)。
    """
    t0 = time.time()
    top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
    sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df_raw = conn.execute(sql).fetchdf()
    conn.close()

    n_q = len(df_raw)
    print(f"  DB loaded {n_q:,} Type_IDs ({time.time()-t0:.1f}s)")

    if sample_n and sample_n < n_q:
        idx = np.random.default_rng(seed).choice(n_q, size=sample_n, replace=False)
        df_raw = df_raw.iloc[idx].reset_index(drop=True)
        n_q = sample_n

    tensor, city_map, to_dict = load_cache(year)
    print(f"  Cache: {tensor.shape}")

    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    pos_cities = df_raw[top_cols].map(parse_to_city).values
    type_ids = df_raw['Type_ID'].values
    persons = np.array([list(parse_type_id(t)[:6]) for t in type_ids], dtype=np.float32)
    from_cities = np.array([parse_type_id(t)[6] for t in type_ids], dtype=np.int32)

    # 交叉特征列索引
    wage_i = [RATIO_FEATS.index(c) for c in
              ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']]
    vac_i = [RATIO_FEATS.index(c) for c in
             ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']]
    tier_i = len(RATIO_FEATS) + ABS_FEATS.index('tier_diff')
    gdp_i = RATIO_FEATS.index('gdp_per_capita_ratio')
    hous_i = RATIO_FEATS.index('housing_price_avg_ratio')
    edu_i = RATIO_FEATS.index('education_score_ratio')

    # 预估行数：每 query 约 20 正 + neg_sample_n 负
    avg_per_q = (20 + (neg_sample_n or 336))
    est_total = n_q * avg_per_q
    X = np.empty((int(est_total * 1.1), len(FEATS)), dtype=np.float32)
    labels = np.empty(int(est_total * 1.1), dtype=np.int8)
    qids = np.empty(int(est_total * 1.1), dtype=np.int32)

    rng = np.random.default_rng(seed + year)
    row = 0
    valid_q = 0
    t1 = time.time()
    est_per_q = 0.0004
    print(f"  预估耗时: {n_q * est_per_q:.1f}s (~{n_q * est_per_q / 60:.1f}min)")

    for q in tqdm(range(n_q), desc=f"Build {year}", unit="query"):
        fc = from_cities[q]
        all_to = to_dict.get(fc, np.array([], dtype=np.int32))

        if len(all_to) == 0:
            continue

        if len(all_to) < 336:
            all_to = np.pad(all_to, (0, 336 - len(all_to)), constant_values=fc)
        else:
            all_to = all_to[:336]

        # 直接用 np.isin 向量化判断，提速 10 倍以上
        lbl = np.isin(all_to, pos_cities[q]).astype(np.int8)

        pos_idx = np.where(lbl > 0)[0]
        neg_idx = np.where(lbl == 0)[0]

        if neg_sample_n is not None:
            n_neg_keep = min(neg_sample_n, len(neg_idx))
            neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep > 0 else np.array([], dtype=np.int64)
            keep_idx = np.concatenate([pos_idx, neg_keep])
        else:
            keep_idx = np.arange(336)

        keep_idx.sort()
        n_keep = len(keep_idx)
        if n_keep == 0:
            continue

        kept_all_to = all_to[keep_idx]

        # 只提取保留下来的候选的特征 (修复 from_city 的索引映射)
        pf = tensor[city_map[fc], city_map[kept_all_to], :]

        # 直接利用 NumPy 的广播机制写入大矩阵 X，彻底消除 concatenate 和 tile
        X[row:row+n_keep, :6] = persons[q]                 # 前 6 列：个人特征
        X[row:row+n_keep, 6:57] = pf                       # 中间 51 列：城市与交互特征

        # 直接在 X 上计算交叉特征，零临时内存分配
        ind = int(persons[q][3])
        cross_idx = 57  # 交叉特征从第 57 列开始
        X[row:row+n_keep, cross_idx + 0] = pf[:, wage_i[min(ind, 3)]]
        X[row:row+n_keep, cross_idx + 1] = pf[:, vac_i[min(ind, 3)]]
        X[row:row+n_keep, cross_idx + 2] = persons[q][2] * pf[:, tier_i]
        X[row:row+n_keep, cross_idx + 3] = persons[q][4] * pf[:, gdp_i]
        X[row:row+n_keep, cross_idx + 4] = persons[q][1] * pf[:, hous_i]
        X[row:row+n_keep, cross_idx + 5] = persons[q][5] * pf[:, edu_i]

        lbl_kept = lbl[keep_idx]

        labels[row:row+n_keep] = lbl_kept
        qids[row:row+n_keep] = valid_q
        row += n_keep
        valid_q += 1

    print(f"  Built {row:,} rows, {valid_q:,} queries ({time.time()-t1:.1f}s)")
    X = X[:row]; labels = labels[:row]; qids = qids[:row]
    del tensor, city_map, to_dict; gc.collect()
    return X, labels, qids, valid_q


# ═══════════════════════════════════════════════════════════════
# Bin 构建
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Bin 构建 (修复线程争用，增加耗时监控)
# ═══════════════════════════════════════════════════════════════

def build_bin(year: int, is_train: bool, sample_n: int = None, neg_sample_n: int = 200):
    print(f"\n[{'Train' if is_train else 'Val'} {year}]")
    suffix = "Train" if is_train else "Val"
    X, labels, qids, n_q = build_year_data(year, sample_n=sample_n, neg_sample_n=neg_sample_n if is_train else None)

    groups = np.diff(np.r_[0, np.where(np.diff(qids) != 0)[0] + 1, len(qids)])
    n_rows = len(X)

    print(f"  构建 Dataset (处理 48 亿个特征点，预计需 1~3 分钟，请稍候)...")
    t_construct = time.time()

    # 【关键修复】加上 num_threads 限制，防止服务器几十个核心抢爆内存总线
    ds = lgb.Dataset(X, label=labels, feature_name=list(FEATS),
                     categorical_feature=CATS, free_raw_data=True,
                     params={
                         'max_bin': 255,
                         'num_threads': 16,             # 限制线程数为 16（如果还是很慢，可尝试降到 8）
                         'bin_construct_sample_cnt': 200000 # 显式指定分箱采样数，避免全量扫描算边界
                     })
    ds.set_group(groups)
    ds.construct()

    print(f"  => Dataset 分箱构建完成！耗时: {time.time()-t_construct:.1f}s")

    suffix = "train" if is_train else "val"
    bin_path = BIN_DIR / f"{suffix}_{year}.bin"

    print(f"  正在将 Bin 写入硬盘 (约几 GB)...")
    t_save = time.time()
    ds.save_binary(str(bin_path))
    print(f"  => 落盘完成！耗时: {time.time()-t_save:.1f}s")

    if not is_train:
        np.save(BIN_DIR / f"val_labels_{year}.npy", labels)
        np.save(BIN_DIR / f"val_groups_{year}.npy", groups)

    print(f"  Saved {bin_path.name}: {n_rows:,} rows, {n_q:,} groups")
    del X, labels, qids; gc.collect()
    return bin_path, n_rows, n_q
# ═══════════════════════════════════════════════════════════════
# 训练 (全内存动态轮转接力 - 彻底修复 Group 死锁版)
# ═══════════════════════════════════════════════════════════════

def train(train_years, val_years, query_ratio=1.0, neg_ratio=1.0):
    t0 = time.time()
    print("\n=== Train (全内存动态轮转接力) ===")

    print(f"\n[预加载] 正在构建全局验证集 (年份: {val_years[0]})...")
    val_X, val_labels, val_qids, _ = build_year_data(val_years[0], sample_n=10000, neg_sample_n=None)
    val_groups = np.diff(np.r_[0, np.where(np.diff(val_qids) != 0)[0] + 1, len(val_qids)])
    
    val_ds = lgb.Dataset(val_X, label=val_labels, feature_name=list(FEATS),
                         categorical_feature=CATS, free_raw_data=False, # 保留 raw_data 用于对齐
                         params={'max_bin': 255, 'num_threads': 16})
    val_ds.set_group(val_groups)
    val_ds.construct()
    
    del val_X, val_labels, val_qids; gc.collect()
    print("  => 验证集就绪！")

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20, 40],
        'lambdarank_truncation_level': 20,
        'label_gain': '0,1',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,   # 把步子迈小一点，稳扎稳打
        'num_leaves': 63,         # 把树的复杂度砍半，不让它记噪音
        'max_depth': 7,           # 限制树的深度
        'num_threads': 16,
        'max_bin': 255,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.0,
        'lambda_l2': 0.5,         # 加大正则化力度
        'min_child_samples': 50,  # 提高叶子的样本门槛
        'force_col_wise': True,
        'verbosity': 1,
    }

    TOTAL_TREES = 3000
    EPOCHS = 5  
    n_years = len(train_years)
    trees_per_step = max(1, TOTAL_TREES // (EPOCHS * n_years))

    booster = None
    model_path = MODEL_DIR / "0322ltr_model.txt"
    ckpt_dir = MODEL_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_tree_count = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n" + "="*40)
        print(f">>> 开始 Epoch {epoch}/{EPOCHS} <<<")
        print("="*40)

        for year in train_years:
            print(f"\n  [Epoch {epoch} | 年份 {year}] 正在动态生成 Numpy 数据...")
            
            t_build = time.time()
            # 负样本降到 50，内存瞬间降到 6GB，告别 Swap 卡顿！
            X, labels, qids, n_q = build_year_data(year, sample_n=None, neg_sample_n=50)
            groups = np.diff(np.r_[0, np.where(np.diff(qids) != 0)[0] + 1, len(qids)])
            
            train_ds = lgb.Dataset(X, label=labels, feature_name=list(FEATS),
                                   categorical_feature=CATS, free_raw_data=False,
                                   params={'max_bin': 255, 'num_threads': 16},
                                   reference=val_ds)
            train_ds.set_group(groups)
            train_ds.construct()
            print(f"    -> 数据加载完成，耗时 {time.time()-t_build:.1f}s, 数据量: {len(X):,} 行")

            # =======================================================
            # 局部动态采样 (【修改点 2】修复 Group 丢失问题)
            # =======================================================
            if query_ratio < 1.0 or neg_ratio < 1.0:
                labels_mem = train_ds.get_label()
                groups_mem = train_ds.get_group()
                splits = np.r_[0, np.cumsum(groups_mem)]
                n_queries = len(groups_mem)

                if query_ratio < 1.0:
                    selected_queries = np.random.choice(n_queries, size=int(n_queries * query_ratio), replace=False)
                    selected_queries.sort()
                else:
                    selected_queries = np.arange(n_queries)

                used_indices = []
                new_groups = []  # 核心：记录采样后的新 Group 大小

                for q in selected_queries:
                    start, end = splits[q], splits[q+1]
                    q_labels = labels_mem[start:end]
                    pos_idx = np.where(q_labels > 0)[0] + start
                    neg_idx = np.where(q_labels == 0)[0] + start

                    if neg_ratio < 1.0 and len(neg_idx) > 0:
                        n_neg = int(len(neg_idx) * neg_ratio)
                        if n_neg > 0:
                            neg_idx = np.random.choice(neg_idx, size=n_neg, replace=False)
                        else:
                            neg_idx = np.array([], dtype=int)

                    n_keep = len(pos_idx) + len(neg_idx)
                    if n_keep > 0:
                        used_indices.append(pos_idx)
                        if len(neg_idx) > 0:
                            used_indices.append(neg_idx)
                        new_groups.append(n_keep) # 存下当前 Query 剩下的行数

                if used_indices:
                    used_indices = np.concatenate(used_indices)
                    used_indices.sort()
                    train_ds = train_ds.subset(used_indices)
                    train_ds.set_group(new_groups) # 核心：把新 Group 告诉 LightGBM
                    train_ds.construct()

            print(f"    -> 训练中: 新增 {trees_per_step} 棵树 (累计 {current_tree_count + trees_per_step} / {TOTAL_TREES})")
            booster = lgb.train(
                params,
                train_ds,
                num_boost_round=trees_per_step,
                valid_sets=[train_ds, val_ds],           # 【修改点 3】同时监控 train 和 val
                valid_names=[f'train_{year}', 'val_2019'], 
                callbacks=[lgb.log_evaluation(5)],
                init_model=booster,          
                keep_training_booster=True,
            )
            current_tree_count += trees_per_step

            booster.free_dataset() 
            del X, labels, qids, train_ds
            gc.collect()

            ckpt_path = ckpt_dir / f"ltr_model_latest.txt"
            booster.save_model(str(ckpt_path))

    booster.save_model(str(model_path))
    print(f"\n  Model Saved: {model_path}")
    print(f"  Train total time: {time.time()-t0:.1f}s")

    imp = pd.DataFrame({
        'feature': booster.feature_name(),
        'importance': booster.feature_importance('gain'),
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 Features:")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']}: {r['importance']:.0f}")

    return str(model_path)


# ═══════════════════════════════════════════════════════════════
# 评估 (保持不变)
# ═══════════════════════════════════════════════════════════════
def evaluate(model_path: str, years, sample_n=5000):
    print(f"\n=== Evaluate ({years}) ===")
    model = lgb.Booster(model_file=model_path)

    all_results = []
    for year in years:
        t0 = time.time()
        X, labels, qids, n_q = build_year_data(year, sample_n=sample_n)
        scores = model.predict(X)

        groups = np.diff(np.r_[0, np.where(np.diff(qids) != 0)[0] + 1, len(qids)])
        splits = np.r_[0, np.cumsum(groups)]
        lbl_sp = np.split(labels, splits[1:-1])
        sc_sp = np.split(scores, splits[1:-1])

        valid = [i for i, l in enumerate(lbl_sp) if l.sum() >= 1]
        r = {'year': year}
        for k in [5, 10, 20, 40]:
            total = 0.0
            for i in valid:
                l, p = lbl_sp[i], sc_sp[i]
                topk = np.argpartition(p, -min(k, len(p)))[-min(k, len(p)):]
                total += np.sum(l[topk]) / min(k, int(l.sum()))
            r[f'R@{k}'] = total / len(valid)

        print(f"  {year}: R@5={r['R@5']:.4f} R@10={r['R@10']:.4f} "
              f"R@20={r['R@20']:.4f} ({time.time()-t0:.1f}s)")
        all_results.append(r)
        del X, labels, qids; gc.collect()

    print(f"\n{'='*50}")
    print(f"{'Year':<8} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'R@40':<10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['year']:<8} {r['R@5']:<10.4f} {r['R@10']:<10.4f} "
              f"{r['R@20']:<10.4f} {r['R@40']:<10.4f}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
# main (已剔除无用的 Bin 校验逻辑)
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="0322 LambdaRank Pipeline")
    p.add_argument("--step", type=str, default="train,eval",
                   help="步骤: train,eval 或 all")
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2000, 2019)))
    p.add_argument("--val-years", type=int, nargs="+", default=[2019])
    p.add_argument("--eval-years", type=int, nargs="+", default=[2020])
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--val-sample", type=int, default=None)
    p.add_argument("--neg-sample", type=str, default="336")

    p.add_argument("--train-query-ratio", type=float, default=1.0,
                   help="训练时抽取的 Query 比例, 0~1 (默认 1.0 全量)")
    p.add_argument("--train-neg-ratio", type=float, default=1.0,
                   help="训练时的负采样比例, 0~1 (默认 1.0 不过滤)")

    p.add_argument("--eval-sample", type=int, default=5000)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()

    steps = [s.strip() for s in args.step.split(",")]
    do_all = "all" in steps

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Train
    if do_all or "train" in steps:
        print("\n### STEP: Train ###")
        model_path = train(
            args.train_years, args.val_years,
            query_ratio=args.train_query_ratio,
            neg_ratio=args.train_neg_ratio
        )
    else:
        model_path = args.model or str(MODEL_DIR / "0322ltr_model.txt")

    # Step 2: Eval
    if do_all or "eval" in steps:
        print("\n### STEP: Eval ###")
        evaluate(model_path, args.eval_years, sample_n=args.eval_sample)

    print(f"\n[OK] Done! Model: {model_path}")


if __name__ == '__main__':
    main()