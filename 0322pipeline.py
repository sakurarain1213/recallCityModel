"""
LambdaRank Pipeline (敏捷验证 & 全局大盘版)

Usage 快速验证模式 (每半年抽100个训练, 20个验证):
  uv run 0322pipeline.py --step train --train-n 100 --val-n 20

Usage 放大规模模式 (每年抽1万个训练, 2千个验证):
  uv run 0322pipeline.py --step train --train-n 10000 --val-n 2000
"""

import gc
import re
import time
import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
import os as _os
if _os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    MODEL_DIR = Path("output/models")
else:
    DB_PATH = Path("/data1/wxj/Recall_city_project/data/local_migration_data.db")
    CACHE_DIR = Path("/data1/wxj/Recall_city_project/data/city_pair_cache")
    MODEL_DIR = Path("/data1/wxj/Recall_city_project/output/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

FEATS_COUNT = len(FEATS)

# TypeID 解析映射
AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}

# 交叉特征列索引预计算
wage_i = [RATIO_FEATS.index(c) for c in ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']]
vac_i = [RATIO_FEATS.index(c) for c in ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']]
tier_i = len(RATIO_FEATS) + ABS_FEATS.index('tier_diff')
gdp_i = RATIO_FEATS.index('gdp_per_capita_ratio')
hous_i = RATIO_FEATS.index('housing_price_avg_ratio')
edu_i = RATIO_FEATS.index('education_score_ratio')


def parse_type_id(tid: str) -> tuple:
    parts = tid.rsplit('_', 1)
    fc = int(parts[1])
    a = parts[0].split('_')
    return (GENDER_MAP[a[0]], AGE_MAP[a[1]], EDU_MAP[a[2]],
            IND_MAP[a[3]], INC_MAP[a[4]], FAM_MAP[a[5]], fc)

def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', s)
    return int(m.group(1)) if m else int(s)


def load_cache(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    df = pd.read_parquet(path, columns=['from_city', 'to_city'] + CACHE_FEAT_COLS)
    all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    max_id = int(all_ids.max())
    city_map = np.zeros(max_id + 1, dtype=np.int32)
    city_map[all_ids] = np.arange(len(all_ids))

    tensor = np.zeros((max_id + 1, max_id + 1, len(CACHE_FEAT_COLS)), dtype=np.float32)
    valid_features = df[CACHE_FEAT_COLS].values.astype(np.float32)
    np.nan_to_num(valid_features, copy=False, nan=0.0)
    tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = valid_features

    to_dict = df.groupby('from_city')['to_city'].apply(lambda x: x.values.astype(np.int32)).to_dict()
    return tensor, city_map, to_dict


def _extract_features(df_subset, tensor, city_map, to_dict, neg_sample_n=None, seed=42):
    """将拆分好的小批量 DataFrame 转换为 X, labels, qids"""
    if len(df_subset) == 0:
        return np.empty((0, FEATS_COUNT), dtype=np.float32), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int32), 0

    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    pos_cities = df_subset[top_cols].map(parse_to_city).values
    type_ids = df_subset['Type_ID'].values
    persons = np.array([list(parse_type_id(t)[:6]) for t in type_ids], dtype=np.float32)
    from_cities = np.array([parse_type_id(t)[6] for t in type_ids], dtype=np.int32)

    n_q = len(type_ids)
    est_total = n_q * (20 + (neg_sample_n or 294))
    X = np.empty((int(est_total * 1.2), FEATS_COUNT), dtype=np.float32)
    labels = np.empty(int(est_total * 1.2), dtype=np.int8)
    qids = np.empty(int(est_total * 1.2), dtype=np.int32)

    rng = np.random.default_rng(seed)
    row = 0
    valid_q = 0

    for q in range(n_q):
        fc = from_cities[q]
        all_to = to_dict.get(fc, np.array([], dtype=np.int32))
        if len(all_to) == 0: continue
        
        all_to = all_to[all_to != fc]
        if len(all_to) == 0: continue

        lbl = np.isin(all_to, pos_cities[q]).astype(np.int8)
        pos_idx = np.where(lbl > 0)[0]
        neg_idx = np.where(lbl == 0)[0]

        if neg_sample_n is not None:
            n_neg_keep = min(neg_sample_n, len(neg_idx))
            neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep > 0 else np.array([], dtype=np.int64)
            keep_idx = np.concatenate([pos_idx, neg_keep])
        else:
            keep_idx = np.arange(len(all_to))

        keep_idx.sort()
        n_keep = len(keep_idx)
        if n_keep == 0: continue

        kept_all_to = all_to[keep_idx]
        pf = tensor[city_map[fc], city_map[kept_all_to], :]

        X[row:row+n_keep, :6] = persons[q]
        X[row:row+n_keep, 6:57] = pf

        ind = int(persons[q][3])
        cross_idx = 57
        X[row:row+n_keep, cross_idx + 0] = pf[:, wage_i[min(ind, 3)]]
        X[row:row+n_keep, cross_idx + 1] = pf[:, vac_i[min(ind, 3)]]
        X[row:row+n_keep, cross_idx + 2] = persons[q][2] * pf[:, tier_i]
        X[row:row+n_keep, cross_idx + 3] = persons[q][4] * pf[:, gdp_i]
        X[row:row+n_keep, cross_idx + 4] = persons[q][1] * pf[:, hous_i]
        X[row:row+n_keep, cross_idx + 5] = persons[q][5] * pf[:, edu_i]

        labels[row:row+n_keep] = lbl[keep_idx]
        qids[row:row+n_keep] = valid_q
        row += n_keep
        valid_q += 1

    return X[:row], labels[:row], qids[:row], valid_q


def build_year_train_val_data(year: int, train_n: int, val_n: int, train_neg: int, val_neg: int, seed: int = 42):
    """加载该年份的数据，在保证互斥的情况下，拆分出极小的 Train 和 Val 集合

    Args:
        train_neg: 训练集每个Query的负样本数 (最多274个，因为总共294城-20正样本)
        val_neg: 验证集每个Query的负样本数
    """
    top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
    sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df_raw = conn.execute(sql).fetchdf()
    conn.close()

    total_q = len(df_raw)
    
    # 确保想要采样的总数不超过数据库的总行数
    actual_train_n = min(train_n, total_q)
    actual_val_n = min(val_n, total_q - actual_train_n)

    # 全局打乱并拆分索引，保证 Train 和 Val 绝对不重合！
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(total_q)
    
    train_idx = shuffled_idx[:actual_train_n]
    val_idx = shuffled_idx[actual_train_n : actual_train_n + actual_val_n]

    df_train = df_raw.iloc[train_idx].reset_index(drop=True)
    df_val = df_raw.iloc[val_idx].reset_index(drop=True)

    # 加载底层缓存
    tensor, city_map, to_dict = load_cache(year)

    print(f"  [处理 {year}] 抽取 Train={len(df_train):,} (neg={train_neg}) / Val={len(df_val):,} (neg={val_neg})...")
    tr_X, tr_y, tr_q, tr_nq = _extract_features(df_train, tensor, city_map, to_dict, neg_sample_n=train_neg, seed=seed)
    vl_X, vl_y, vl_q, vl_nq = _extract_features(df_val, tensor, city_map, to_dict, neg_sample_n=val_neg, seed=seed+1)

    del tensor, city_map, to_dict; gc.collect()
    return (tr_X, tr_y, tr_q, tr_nq), (vl_X, vl_y, vl_q, vl_nq)


# ═══════════════════════════════════════════════════════════════
# 敏捷全局训练
# ═══════════════════════════════════════════════════════════════

def train_rapid_global(train_years, train_n, val_n, train_neg, val_neg):
    t0 = time.time()
    print(f"\n=== 🚀 Train (敏捷全局混合模式 | 每年 Train:{train_n}(neg={train_neg}), Val:{val_n}(neg={val_neg})) ===")

    tr_X_list, tr_y_list, tr_q_list = [], [], []
    vl_X_list, vl_y_list, vl_q_list = [], [], []
    
    tr_total_queries = 0
    vl_total_queries = 0

    # 1. 遍历收集所有年份的极小样本
    for year in train_years:
        (tr_X, tr_y, tr_q, tr_nq), (vl_X, vl_y, vl_q, vl_nq) = build_year_train_val_data(year, train_n, val_n, train_neg, val_neg)
        
        # 为了保证不同年份拼在一起时 Query ID 递增不断层
        tr_q += tr_total_queries
        tr_total_queries += tr_nq
        tr_X_list.append(tr_X); tr_y_list.append(tr_y); tr_q_list.append(tr_q)

        vl_q += vl_total_queries
        vl_total_queries += vl_nq
        vl_X_list.append(vl_X); vl_y_list.append(vl_y); vl_q_list.append(vl_q)

    # 2. 全局无缝合并
    print("\n  [合并] 正在拼接全局大盘数据...")
    X_train = np.vstack(tr_X_list)
    y_train = np.concatenate(tr_y_list)
    q_train = np.concatenate(tr_q_list)

    X_val = np.vstack(vl_X_list)
    y_val = np.concatenate(vl_y_list)
    q_val = np.concatenate(vl_q_list)

    del tr_X_list, tr_y_list, tr_q_list, vl_X_list, vl_y_list, vl_q_list; gc.collect()

    groups_train = np.diff(np.r_[0, np.where(np.diff(q_train) != 0)[0] + 1, len(q_train)])
    groups_val = np.diff(np.r_[0, np.where(np.diff(q_val) != 0)[0] + 1, len(q_val)])

    print(f"  [就绪] 全局 Train 矩阵: {X_train.shape} | 全局 Val 矩阵: {X_val.shape}")

    # 3. 构造 LightGBM Dataset
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(FEATS),
                           categorical_feature=CATS, free_raw_data=True,
                           params={'max_bin': 255, 'num_threads': 16})
    train_ds.set_group(groups_train)

    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=list(FEATS),
                         categorical_feature=CATS, free_raw_data=True,
                         reference=train_ds) # 强制对齐训练集的分箱规则
    val_ds.set_group(groups_val)

    del X_train, y_train, q_train, X_val, y_val, q_val; gc.collect()

    # 4. 训练核心配置
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [20],
        'lambdarank_truncation_level': 20,
        'label_gain': '0,1',
        'boosting_type': 'gbdt',
        
        'learning_rate': 0.05, 
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 5, # 小样本测试时，这里必须设小一点，否则不分裂
        
        'feature_pre_filter': False,
        'num_threads': 16,
        'max_bin': 255,
        'force_col_wise': True,
        'verbosity': 1,
    }

    print("\n  [开训] 启动带 Early Stopping 的全局训练...")
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=1000,           # 最大允许树的数量
        valid_sets=[train_ds, val_ds],  # 同时监控 Train 和 Val
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(10),     # 每 10 棵树打印一次日志
            lgb.early_stopping(50)      # 如果 Valid 分数 50 棵树都没涨，自动停止！
        ]
    )

    model_path = MODEL_DIR / "0322ltr_model_rapid.txt"
    booster.save_model(str(model_path))
    print(f"\n[OK] 训练完成！模型已保存: {model_path} (耗时: {time.time()-t0:.1f}s)")

    imp = pd.DataFrame({
        'feature': booster.feature_name(),
        'importance': booster.feature_importance('gain'),
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 核心特征:")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']}: {r['importance']:.0f}")

    return str(model_path)


# ═══════════════════════════════════════════════════════════════
# main 
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, default="train")
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2000, 2021)))
    p.add_argument("--train-n", type=int, default=100, help="每年抽取的训练 Query 数")
    p.add_argument("--val-n", type=int, default=20, help="每年抽取的验证 Query 数")
    p.add_argument("--train-neg", type=int, default=20, help="训练集每个Query的负样本数 (最多274)")
    p.add_argument("--val-neg", type=int, default=50, help="验证集每个Query的负样本数 (最多274)")
    args = p.parse_args()

    steps = [s.strip() for s in args.step.split(",")]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in steps:
        train_rapid_global(args.train_years, args.train_n, args.val_n, args.train_neg, args.val_neg)

if __name__ == '__main__':
    main()