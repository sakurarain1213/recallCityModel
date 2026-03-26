"""
0324 LambdaRank Pipeline (精排模型专属 - Hard Negative Mining 困难负样本挖掘版)
核心逻辑: 
读取 Recall 阶段产出的 JSONL 结果。每个 Query 的训练候选集严格限定为 [Recall Top 100 ∪ 真实 GT 20]。
采用梯度相关性得分 (Top1=20分, Top2=19分... 负样本=0分)，专攻"神仙打架"！

Usage:
python 0324-rank-pipeline.py --train-years {2000..2020} --train-n 50000 --val-n 2000 --num-rounds 6000 --save-every 100
"""

import gc
import re
import json
import time
import argparse
import multiprocessing
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

# ═══════════════════════════════════════════════════════════════
# 自动硬件检测
# ═══════════════════════════════════════════════════════════════
def detect_hardware():
    cpu_count = multiprocessing.cpu_count()
    mem_gb = psutil.virtual_memory().total / (1024**3)
    avail_mem_gb = psutil.virtual_memory().available / (1024**3)
    return cpu_count, mem_gb, avail_mem_gb

CPU_COUNT, TOTAL_MEM_GB, AVAIL_MEM_GB = detect_hardware()
print(f"[Hardware] CPU: {CPU_COUNT} 核 | 内存: {TOTAL_MEM_GB:.1f}GB (可用: {AVAIL_MEM_GB:.1f}GB)")

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
import os as _os
if _os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    MODEL_DIR = Path("output/models")
    RECALL_RESULT_DIR = Path("recall_result")
else:
    DB_PATH = Path("/home/lpg/code/recall-train/data/local_migration_data.db")
    CACHE_DIR = Path("/home/lpg/code/recall-train/data/city_pair_cache")
    MODEL_DIR = Path("/home/lpg/code/recall-train/output/models")
    RECALL_RESULT_DIR = Path("/home/lpg/code/recall-train/recall_result")

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
    return (GENDER_MAP[a[0]], AGE_MAP[a[1]], EDU_MAP[a[2]], IND_MAP[a[3]], INC_MAP[a[4]], FAM_MAP[a[5]], fc)

def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', str(s))
    return int(m.group(1)) if m else int(s)

def load_cache(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    
    cols_str = ', '.join(['from_city', 'to_city'] + CACHE_FEAT_COLS)
    df = duckdb.query(f"SELECT {cols_str} FROM '{path}'").to_df()
    
    all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    max_id = int(all_ids.max())
    city_map = np.zeros(max_id + 1, dtype=np.int32)
    city_map[all_ids] = np.arange(len(all_ids))

    tensor = np.zeros((max_id + 1, max_id + 1, len(CACHE_FEAT_COLS)), dtype=np.float32)
    valid_features = df[CACHE_FEAT_COLS].values.astype(np.float32)
    np.nan_to_num(valid_features, copy=False, nan=0.0)
    tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = valid_features

    # 返回 all_ids 提供安全索引范围
    return tensor, city_map, all_ids

def load_recall_dict(year: int, top_k: int = 100) -> dict:
    jsonl_path = RECALL_RESULT_DIR / f"{year}_local_sample.jsonl"
    if not jsonl_path.exists():
        print(f"⚠️ [警告] 找不到召回文件: {jsonl_path}。这部分数据将被跳过。")
        return {}
    
    recall_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for tid, cities in data.items():
                if isinstance(cities, dict):
                    cities = cities.get('top_cities', [])
                recall_dict[tid] = cities[:top_k]
    print(f"  [载入召回池] 成功读取 {jsonl_path.name}, 包含 {len(recall_dict):,} 个 Query")
    return recall_dict

def _extract_features(df_subset, tensor, city_map, valid_ids, recall_dict):
    """将拆分好的小批量 DataFrame 转换为 X, labels, qids"""
    if len(df_subset) == 0:
        return np.empty((0, FEATS_COUNT), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32), 0

    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    pos_cities_matrix = df_subset[top_cols].map(parse_to_city).values
    type_ids = df_subset['Type_ID'].values
    persons = np.array([list(parse_type_id(t)[:6]) for t in type_ids], dtype=np.float32)
    from_cities = np.array([parse_type_id(t)[6] for t in type_ids], dtype=np.int32)

    n_q = len(type_ids)
    est_total = n_q * 120
    X = np.empty((int(est_total * 1.2), FEATS_COUNT), dtype=np.float32)
    labels = np.empty(int(est_total * 1.2), dtype=np.float32) 
    qids = np.empty(int(est_total * 1.2), dtype=np.int32)

    row = 0
    valid_q = 0
    
    # 🎯 梯度相关性得分映射: Top 1 = 20分 ... Top 20 = 1分
    score_mapping = {i: 20 - i for i in range(20)}

    for q in range(n_q):
        tid = type_ids[q]
        fc = from_cities[q]
        pos_c = pos_cities_matrix[q]
        
        recalled_cands = recall_dict.get(tid, [])
        if len(recalled_cands) == 0:
            continue
            
        # 并集：Recall Top 100 ∪ 真实 GT 20
        all_to = np.unique(np.concatenate((recalled_cands, pos_c)))
        
        # 安全过滤：剔除自身，且必须在 valid_ids 特征库内
        all_to = all_to[all_to != fc]
        all_to = np.intersect1d(all_to, valid_ids)
        
        if len(all_to) == 0: continue

        n_keep = len(all_to)
        
        current_labels = np.zeros(n_keep, dtype=np.float32)
        for rank_idx, city_id in enumerate(pos_c):
            match_idx = np.where(all_to == city_id)[0]
            if len(match_idx) > 0:
                current_labels[match_idx[0]] = score_mapping[rank_idx]

        pf = tensor[city_map[fc], city_map[all_to], :]

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

        labels[row:row+n_keep] = current_labels
        qids[row:row+n_keep] = valid_q
        row += n_keep
        valid_q += 1

    return X[:row], labels[:row], qids[:row], valid_q

def build_year_train_val_data(year: int, train_n: int, val_n: int, seed: int = 42):
    top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
    sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df_raw = conn.execute(sql).fetchdf()
    conn.close()

    total_q = len(df_raw)
    
    actual_train_n = min(train_n, total_q) if train_n > 0 else 0
    actual_val_n = min(val_n, total_q - actual_train_n) if val_n > 0 else 0

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(total_q)
    
    train_idx = shuffled_idx[:actual_train_n]
    val_idx = shuffled_idx[actual_train_n : actual_train_n + actual_val_n]

    df_train = df_raw.iloc[train_idx].reset_index(drop=True)
    df_val = df_raw.iloc[val_idx].reset_index(drop=True)

    tensor, city_map, valid_ids = load_cache(year)
    recall_dict = load_recall_dict(year, top_k=100)
    
    if len(recall_dict) == 0:
        del tensor, city_map, df_train, df_val; gc.collect()
        return None

    print(f"  [处理 {year}] 目标抽取 Train={actual_train_n:,} / Val={actual_val_n:,}...")
    
    tr_X, tr_y, tr_q, tr_nq = _extract_features(df_train, tensor, city_map, valid_ids, recall_dict)
    vl_X, vl_y, vl_q, vl_nq = _extract_features(df_val, tensor, city_map, valid_ids, recall_dict)

    del tensor, city_map, valid_ids, recall_dict; gc.collect()
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

def train_rapid_global(train_years, train_n, val_n, num_rounds=2000, save_every=100):
    t0 = time.time()
    
    print(f"\n=== 🚀 Train Rank (硬负样本精排模式 | 每有效年份 Train:{train_n}, Val:{val_n}) ===")

    tr_X_list, tr_y_list, tr_q_list = [], [], []
    vl_X_list, vl_y_list, vl_q_list = [], [], []
    tr_total_queries = 0
    vl_total_queries = 0
    valid_years_processed = 0

    for year in train_years:
        result = build_year_train_val_data(year, train_n, val_n)
        if result is None:
            continue
            
        valid_years_processed += 1
        (tr_X, tr_y, tr_q, tr_nq), (vl_X, vl_y, vl_q, vl_nq) = result
        
        if tr_nq > 0:
            tr_q += tr_total_queries
            tr_total_queries += tr_nq
            tr_X_list.append(tr_X); tr_y_list.append(tr_y); tr_q_list.append(tr_q)

        if vl_nq > 0:
            vl_q += vl_total_queries
            vl_total_queries += vl_nq
            vl_X_list.append(vl_X); vl_y_list.append(vl_y); vl_q_list.append(vl_q)

    if valid_years_processed == 0:
        print("❌ 未能成功处理任何年份的数据，请检查 JSONL 文件是否存在！")
        return None

    print("\n  [合并] 正在拼接全局精排数据...")
    X_train = np.vstack(tr_X_list) if tr_X_list else np.empty((0, FEATS_COUNT))
    y_train = np.concatenate(tr_y_list) if tr_y_list else np.empty(0)
    q_train = np.concatenate(tr_q_list) if tr_q_list else np.empty(0)

    X_val = np.vstack(vl_X_list) if vl_X_list else np.empty((0, FEATS_COUNT))
    y_val = np.concatenate(vl_y_list) if vl_y_list else np.empty(0)
    q_val = np.concatenate(vl_q_list) if vl_q_list else np.empty(0)

    del tr_X_list, tr_y_list, tr_q_list, vl_X_list, vl_y_list, vl_q_list; gc.collect()

    groups_train = np.diff(np.r_[0, np.where(np.diff(q_train) != 0)[0] + 1, len(q_train)])
    groups_val = np.diff(np.r_[0, np.where(np.diff(q_val) != 0)[0] + 1, len(q_val)]) if len(q_val) > 0 else []

    print(f"  [就绪] 混合 Train 矩阵: {X_train.shape} | 混合 Val 矩阵: {X_val.shape}")

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(FEATS), categorical_feature=CATS, free_raw_data=True,
                           params={'max_bin': 255, 'num_threads': CPU_COUNT})
    train_ds.set_group(groups_train)

    valid_sets = [train_ds]
    valid_names = ['train']
    if len(X_val) > 0:
        val_ds = lgb.Dataset(X_val, label=y_val, feature_name=list(FEATS), categorical_feature=CATS, free_raw_data=True, reference=train_ds)
        val_ds.set_group(groups_val)
        valid_sets.append(val_ds)
        valid_names.append('valid')

    del X_train, y_train, q_train, X_val, y_val, q_val; gc.collect()

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [20],
        'lambdarank_truncation_level': 20,
        'label_gain': ','.join([str(i) for i in range(21)]), 
        'boosting_type': 'gbdt',

        'learning_rate': 0.1,          
        'num_leaves': 127,             
        'max_depth': 8,                 
        'min_child_samples': 20,       

        'bagging_fraction': 0.8,        
        'bagging_freq': 1,              
        'feature_fraction': 0.8,        

        'feature_pre_filter': False,
        'num_threads': CPU_COUNT,
        'max_bin': 255,
        'force_col_wise': True,
        'verbosity': 1,
    }

    print("\n  [开训] 启动精排模型训练 (Hard Negative Mining)...")

    callbacks = [
        lgb.log_evaluation(50),     
        lgb.early_stopping(100)      
    ]

    if save_every > 0:
        ckpt_cb = CheckpointCallback(MODEL_DIR / "0326ltr_checkpoint", freq=save_every)
        callbacks.append(ckpt_cb)

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=num_rounds,   
        valid_sets=valid_sets,  
        valid_names=valid_names,
        callbacks=callbacks
    )

    model_path = MODEL_DIR / "0326ltr_model_rank.txt"
    booster.save_model(str(model_path))
    print(f"\n[OK] 训练完成！精排模型已保存: {model_path} (耗时: {time.time()-t0:.1f}s)")

    return str(model_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=str, default="train")
    p.add_argument("--train-years", type=int, nargs="+", default=[2000])
    p.add_argument("--train-n", type=int, default=50000)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--num-rounds", type=int, default=3000)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()

    steps = [s.strip() for s in args.step.split(",")]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in steps:
        train_rapid_global(args.train_years, args.train_n, args.val_n,
                          num_rounds=args.num_rounds, save_every=args.save_every)

if __name__ == '__main__':
    main()