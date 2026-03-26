"""
LambdaRank Pipeline (敏捷验证 & 全局大盘版 - 337 全局城市池修正版)

Usage 快速验证模式 (每半年抽100个训练, 20个验证):
  uv run 0325-recall-pipeline.py --step train --train-n 100 --val-n 20

Usage 放大规模模式 (每年抽5万个训练, 2千个验证) 约1h :
uv run 0325-recall-pipeline.py --step train --train-n 50000 --val-n 2000 --train-neg 100
uv run 0325-recall-pipeline.py --step train --train-n 50000 --val-n 2000 --train-neg 100 --val-neg -1 --num-rounds 3000 --save-every 50


python 0325-recall-pipeline.py --step train --train-n 50000 --val-n 1000 --train-neg -1 --val-neg -1 --num-rounds 6000 --save-every 100
"""

import gc
import re
import json
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
PROJECT_ROOT = Path(__file__).parent.resolve()

if _os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    MODEL_DIR = Path("output/models")
    CITY_NODES_PATH = Path("data/city_nodes.jsonl") # ✅ 新增全局城市池路径
else:
    DB_PATH = Path("/home/lpg/code/recall-train/data/local_migration_data.db")
    CACHE_DIR = Path("/home/lpg/code/recall-train/data/city_pair_cache")
    MODEL_DIR = Path("/home/lpg/code/recall-train/output/models")
    CITY_NODES_PATH = Path("/home/lpg/code/recall-train/data/city_nodes.jsonl") # ✅ 新增全局城市池路径

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════════════════
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
RATIO_FEATS = ['gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio', 'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio', 'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio', 'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio', 'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio', 'avg_commute_mins_ratio', 'population_total_ratio', 'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio', 'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio']
DIFF_FEATS = ['gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff', 'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff', 'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff']
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = ['industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio', 'education_x_tier_diff', 'income_x_gdp_ratio', 'age_x_housing_ratio', 'family_x_edu_score_ratio']
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

FEATS_COUNT = len(FEATS)

AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}

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

# ✅ 新增：加载 337 个全局城市节点
def load_global_cities(path: Path) -> np.ndarray:
    cities = []
    if not path.exists():
        raise FileNotFoundError(f"找不到城市节点文件: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cities.append(int(json.loads(line)['city_id']))
    return np.array(cities, dtype=np.int32)

def load_cache(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    
    # ✅ 改用 duckdb 极速读取，规避 pyarrow 报错
    cols_str = ', '.join(['from_city', 'to_city'] + CACHE_FEAT_COLS)
    df = duckdb.query(f"SELECT {cols_str} FROM '{path}'").to_df()
    
    # 提取 parquet 中实际拥有特征的合法城市 ID，用作后期的安全过滤
    all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    max_id = int(all_ids.max())
    city_map = np.zeros(max_id + 1, dtype=np.int32)
    city_map[all_ids] = np.arange(len(all_ids))

    tensor = np.zeros((max_id + 1, max_id + 1, len(CACHE_FEAT_COLS)), dtype=np.float32)
    valid_features = df[CACHE_FEAT_COLS].values.astype(np.float32)
    np.nan_to_num(valid_features, copy=False, nan=0.0)
    tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = valid_features

    # 返回 all_ids 替代原本的 to_dict
    return tensor, city_map, all_ids


def _extract_features(df_subset, tensor, city_map, valid_ids, global_city_pool, neg_sample_n=None, seed=42):
    """将拆分好的小批量 DataFrame 转换为 X, labels, qids"""
    if len(df_subset) == 0:
        return np.empty((0, FEATS_COUNT), dtype=np.float32), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int32), 0

    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    pos_cities = df_subset[top_cols].map(parse_to_city).values
    type_ids = df_subset['Type_ID'].values
    persons = np.array([list(parse_type_id(t)[:6]) for t in type_ids], dtype=np.float32)
    from_cities = np.array([parse_type_id(t)[6] for t in type_ids], dtype=np.int32)

    n_q = len(type_ids)
    # 因为现在的负样本池固定在 336 左右，所以可以直接预估容量
    est_total = n_q * (20 + (neg_sample_n or len(global_city_pool)))
    X = np.empty((int(est_total * 1.2), FEATS_COUNT), dtype=np.float32)
    labels = np.empty(int(est_total * 1.2), dtype=np.int8)
    qids = np.empty(int(est_total * 1.2), dtype=np.int32)

    rng = np.random.default_rng(seed)
    row = 0
    valid_q = 0

    for q in range(n_q):
        fc = from_cities[q]
        
        # ✅ 核心修正：候选池直接等于 337 全局池 排除 出发地自身
        all_to = global_city_pool[global_city_pool != fc]
        # 加上一道安全锁：只保留在 parquet 里存在特征的城市，防止数组越界
        all_to = np.intersect1d(all_to, valid_ids)

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
    top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
    sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df_raw = conn.execute(sql).fetchdf()
    conn.close()

    total_q = len(df_raw)
    
    actual_train_n = min(train_n, total_q)
    actual_val_n = min(val_n, total_q - actual_train_n)

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(total_q)
    
    train_idx = shuffled_idx[:actual_train_n]
    val_idx = shuffled_idx[actual_train_n : actual_train_n + actual_val_n]

    df_train = df_raw.iloc[train_idx].reset_index(drop=True)
    df_val = df_raw.iloc[val_idx].reset_index(drop=True)

    # ✅ 加载全局的 337 个城市池
    global_city_pool = load_global_cities(CITY_NODES_PATH)

    # 返回有效城市的 IDs
    tensor, city_map, valid_ids = load_cache(year)

    train_neg_str = "全部" if train_neg is None else train_neg
    val_neg_str = "全部" if val_neg is None else val_neg
    print(f"  [处理 {year}] 抽取 Train={len(df_train):,} (neg={train_neg_str}) / Val={len(df_val):,} (neg={val_neg_str})...")
    
    # 传入 valid_ids 和 global_city_pool
    tr_X, tr_y, tr_q, tr_nq = _extract_features(df_train, tensor, city_map, valid_ids, global_city_pool, neg_sample_n=train_neg, seed=seed)
    vl_X, vl_y, vl_q, vl_nq = _extract_features(df_val, tensor, city_map, valid_ids, global_city_pool, neg_sample_n=val_neg, seed=seed+1)

    del tensor, city_map, valid_ids, global_city_pool; gc.collect()
    return (tr_X, tr_y, tr_q, tr_nq), (vl_X, vl_y, vl_q, vl_nq)

class CheckpointCallback:
    def __init__(self, output_dir, freq=50):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq

    def __call__(self, env):
        if (env.iteration + 1) % self.freq == 0:
            ckpt = self.output_dir / f"model_round_{env.iteration+1}.txt"
            env.model.save_model(str(ckpt))


def train_rapid_global(train_years, train_n, val_n, train_neg, val_neg, num_rounds=2000, save_every=100):
    t0 = time.time()
    if train_neg is not None and train_neg < 0: train_neg = None
    if val_neg is not None and val_neg < 0: val_neg = None
    train_neg_str = "全部" if train_neg is None else train_neg
    val_neg_str = "全部" if val_neg is None else val_neg
    print(f"\n=== 🚀 Train (敏捷全局混合模式 | 每年 Train:{train_n}(neg={train_neg_str}), Val:{val_n}(neg={val_neg_str})) ===")
    print(f"    最大轮次: {num_rounds}, Checkpoint间隔: {save_every if save_every > 0 else '关闭'}")

    tr_X_list, tr_y_list = [], []
    vl_X_list, vl_y_list = [], []

    for year in train_years:
        (tr_X, tr_y, _, _), (vl_X, vl_y, _, _) = build_year_train_val_data(year, train_n, val_n, train_neg, val_neg)

        tr_X_list.append(tr_X); tr_y_list.append(tr_y)
        vl_X_list.append(vl_X); vl_y_list.append(vl_y)

    print("\n  [合并] 正在拼接全局大盘数据 (Binary 模式)...")
    X_train = np.vstack(tr_X_list)
    y_train = np.concatenate(tr_y_list)
    X_val = np.vstack(vl_X_list)
    y_val = np.concatenate(vl_y_list)

    del tr_X_list, tr_y_list, vl_X_list, vl_y_list; gc.collect()

    print(f"  [就绪] 全局 Train 矩阵: {X_train.shape} | 全局 Val 矩阵: {X_val.shape}")

    pos_count = np.sum(y_train > 0)
    neg_count = len(y_train) - pos_count
    pos_weight = float(neg_count) / max(1.0, pos_count)
    print(f"  [平衡] 自动计算正样本权重 scale_pos_weight: {pos_weight:.2f}")

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(FEATS), categorical_feature=CATS, free_raw_data=True)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=list(FEATS), categorical_feature=CATS, free_raw_data=True, reference=train_ds)

    del X_train, y_train, X_val, y_val; gc.collect()

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'scale_pos_weight': pos_weight,
        'boosting_type': 'gbdt',

        'learning_rate': 0.1,
        'num_leaves': 511,
        'max_depth': 12,
        'min_child_samples': 500,

        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'min_split_gain': 0.0,

        'feature_pre_filter': False,
        'num_threads': 40,
        'max_bin': 63,
        'force_col_wise': True,
        'verbosity': 1,
    }

    print("\n  [开训] 启动基于 337 全局城市池的二分类过拟合训练...")

    callbacks = [
        lgb.log_evaluation(50),
        lgb.early_stopping(100)
    ]

    if save_every > 0:
        ckpt_cb = CheckpointCallback(MODEL_DIR / "0325ltr_checkpoint", freq=save_every)
        callbacks.append(ckpt_cb)

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=num_rounds,
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    model_path = MODEL_DIR / "0325ltr_model_rapid.txt"
    booster.save_model(str(model_path))
    print(f"\n[OK] 训练完成！模型已保存: {model_path} (耗时: {time.time()-t0:.1f}s)")

    return str(model_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, default="train")
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2000, 2021)))
    p.add_argument("--train-n", type=int, default=100)
    p.add_argument("--val-n", type=int, default=20)
    p.add_argument("--train-neg", type=int, default=-1)
    p.add_argument("--val-neg", type=int, default=-1)
    p.add_argument("--num-rounds", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()

    steps = [s.strip() for s in args.step.split(",")]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in steps:
        train_rapid_global(args.train_years, args.train_n, args.val_n, args.train_neg, args.val_neg,
                          num_rounds=args.num_rounds, save_every=args.save_every)

if __name__ == '__main__':
    main()