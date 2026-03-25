"""
0325 Two-Stage Cascade Inference (召回 + 精排 工业级级联架构)
核心流程:
1. Recall 阶段: 二分类模型对 294 城市打分，截取 Top 80。
2. Rank 阶段: LambdaRank 模型直接在内存中接管这 80 个城市精准打分，选出最终 Top 20。
3. 评测比对: 打印 Recall@20、Recall@80 和 最终 Rank@20 的平均命中个数。
"""

import os
import re
import json
import time
import argparse
import multiprocessing
import concurrent.futures
import warnings
import sys
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
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

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.resolve()

if os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    # 记得将这里的路径替换成你实际跑出来的模型！
    MODEL_RECALL_PATH = Path("C:/Users/w1625/Desktop/recall_model_round_600.txt") 
    MODEL_RANK_PATH = Path("C:/Users/w1625/Desktop/0326ltr_model_rank.txt")
else:
    DB_PATH = PROJECT_ROOT / "data" / "local_migration_data.db"
    CACHE_DIR = PROJECT_ROOT / "data" / "city_pair_cache"
    MODEL_RECALL_PATH = PROJECT_ROOT / "output" / "models" / "model_recall.txt"
    MODEL_RANK_PATH = PROJECT_ROOT / "output" / "models" / "0326ltr_model_rank.txt"

# ═══════════════════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════════════════
RATIO_FEATS = ['gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio', 'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio', 'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio', 'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio', 'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio', 'avg_commute_mins_ratio', 'population_total_ratio', 'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio', 'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio']
DIFF_FEATS = ['gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff', 'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff', 'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff']
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

FEATS_COUNT = 63 

AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}

WAGE_I = np.array([RATIO_FEATS.index(c) for c in ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']], dtype=np.int32)
VAC_I = np.array([RATIO_FEATS.index(c) for c in ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']], dtype=np.int32)
TIER_I = len(RATIO_FEATS) + ABS_FEATS.index('tier_diff')
GDP_I = RATIO_FEATS.index('gdp_per_capita_ratio')
HOUS_I = RATIO_FEATS.index('housing_price_avg_ratio')
EDU_I = RATIO_FEATS.index('education_score_ratio')

def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', str(s))
    return int(m.group(1)) if m else int(s)

def parse_single_type_id(tid: str) -> tuple:
    parts = tid.split('_')
    fc = int(parts[6])
    person = np.array([GENDER_MAP[parts[0]], AGE_MAP[parts[1]], EDU_MAP[parts[2]], IND_MAP[parts[3]], INC_MAP[parts[4]], FAM_MAP[parts[5]]], dtype=np.float32)
    return person, fc, tid

# ═══════════════════════════════════════════════════════════════
# 级联推理引擎
# ═══════════════════════════════════════════════════════════════
class CascadePredictor:
    def __init__(self, model_recall_path, model_rank_path, db_path, cache_dir, num_threads=1):
        self.num_threads = num_threads
        print(f"[Init] Loading Recall Model (Binary): {model_recall_path.name}")
        self.model_recall = lgb.Booster(model_file=str(model_recall_path))
        print(f"[Init] Loading Rank Model (LambdaRank): {model_rank_path.name}")
        self.model_rank = lgb.Booster(model_file=str(model_rank_path))
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)

    def load_data(self, year: int):
        path = self.cache_dir / f"city_pairs_{year}.parquet"
        cols_str = ', '.join(['from_city', 'to_city'] + CACHE_FEAT_COLS)
        df = duckdb.query(f"SELECT {cols_str} FROM '{path}'").to_df()
        
        all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
        max_id = int(all_ids.max())
        city_map = np.zeros(max_id + 1, dtype=np.int32)
        city_map[all_ids] = np.arange(len(all_ids))

        tensor = np.zeros((max_id + 1, max_id + 1, len(CACHE_FEAT_COLS)), dtype=np.float32)
        v_feats = df[CACHE_FEAT_COLS].values.astype(np.float32)
        np.nan_to_num(v_feats, copy=False, nan=0.0)
        tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = v_feats
        to_dict = df.groupby('from_city')['to_city'].apply(lambda x: x.values.astype(np.int32)).to_dict()
        return tensor, city_map, to_dict

    def run_year(self, year, recall_k=80, rank_k=20, sample_ratio=1.0, seed=42):
        print(f"\n[{year}] 开始读取底层特征与数据库...")
        tensor, city_map, to_dict = self.load_data(year)

        conn = duckdb.connect(str(self.db_path), read_only=True)
        top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
        df_gt = conn.execute(f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}").fetchdf()
        conn.close()

        pos_cities = df_gt[[f'To_Top{i}' for i in range(1, 21)]].map(parse_to_city).values
        type_ids = df_gt['Type_ID'].values.tolist()
        
        n_queries = len(type_ids)
        if sample_ratio < 1.0:
            rng = np.random.default_rng(seed + year)
            keep_n = max(1, int(n_queries * sample_ratio))
            keep_indices = rng.choice(n_queries, keep_n, replace=False)
            type_ids = [type_ids[i] for i in keep_indices]
            pos_cities = [pos_cities[i] for i in keep_indices]

        gt_dict = {tid: set(pos) for tid, pos in zip(type_ids, pos_cities)}
        global_city_pool = np.array(list(set(int(tid.rsplit('_', 1)[-1]) for tid in df_gt['Type_ID'].values)), dtype=np.int32)
        
        fc_groups = defaultdict(list)
        for tid in type_ids:
            person, fc_int, full_tid = parse_single_type_id(tid)
            fc_groups[fc_int].append((person, full_tid))

        cands_cache = {}
        for fc_int in fc_groups:
            cands = global_city_pool[global_city_pool != fc_int]
            reachable = to_dict.get(fc_int, np.array([], dtype=np.int32))
            cands = cands[np.isin(cands, reachable)]
            cands_cache[fc_int] = cands

        t0 = time.time()
        all_pred_recall_20, all_pred_recall_80, all_pred_final_20, all_tids = [], [], [], []

        MAX_BATCH_ROWS = int(AVAIL_MEM_GB * 1024**3 * 0.2 / (FEATS_COUNT * 4))
        MAX_BATCH_ROWS = max(5000, min(MAX_BATCH_ROWS, 1_000_000))  
        print(f"[{year}] 双模型级联推理启动: Recall Top {recall_k} -> Rank Top {rank_k}")

        for fc_int, group in fc_groups.items():
            cands = cands_cache[fc_int]
            if len(cands) == 0: continue

            K, C = len(group), len(cands)
            pf = tensor[city_map[fc_int], city_map[cands], :]
            persons = np.array([g[0] for g in group], dtype=np.float32)
            batch_size = max(1, MAX_BATCH_ROWS // C)

            for batch_start in range(0, K, batch_size):
                batch_end = min(batch_start + batch_size, K)
                K_batch = batch_end - batch_start

                X = np.empty((K_batch, C, FEATS_COUNT), dtype=np.float32)
                X[:, :, :6] = persons[batch_start:batch_end, np.newaxis, :]
                X[:, :, 6:57] = pf[np.newaxis, :, :]

                inds = np.clip(persons[batch_start:batch_end, 3].astype(np.int32), 0, 3)
                X[:, :, 57] = pf.T[WAGE_I[inds]]
                X[:, :, 58] = pf.T[VAC_I[inds]]
                X[:, :, 59] = persons[batch_start:batch_end, 2:3] * pf[np.newaxis, :, TIER_I]
                X[:, :, 60] = persons[batch_start:batch_end, 4:5] * pf[np.newaxis, :, GDP_I]
                X[:, :, 61] = persons[batch_start:batch_end, 1:2] * pf[np.newaxis, :, HOUS_I]
                X[:, :, 62] = persons[batch_start:batch_end, 5:6] * pf[np.newaxis, :, EDU_I]

                # =========================================================
                # 🎯 阶段一：Recall (二分类模型)
                # =========================================================
                scores_recall = self.model_recall.predict(X.reshape(-1, FEATS_COUNT), num_threads=self.num_threads).reshape(K_batch, C)
                
                actual_recall_k = min(C, recall_k)
                top80_idx = np.argpartition(-scores_recall, actual_recall_k - 1, axis=1)[:, :actual_recall_k]
                
                actual_rank_k = min(C, rank_k)
                top20_recall_idx = np.argpartition(-scores_recall, actual_rank_k - 1, axis=1)[:, :actual_rank_k]

                # =========================================================
                # 🎯 阶段二：Rank (精排模型)
                # =========================================================
                # 直接在内存中切出那 80 个被 Recall 选中的城市的特征
                top80_idx_expanded = np.repeat(top80_idx[:, :, np.newaxis], FEATS_COUNT, axis=2)
                X_rank = np.take_along_axis(X, top80_idx_expanded, axis=1)

                scores_rank = self.model_rank.predict(X_rank.reshape(-1, FEATS_COUNT), num_threads=self.num_threads).reshape(K_batch, actual_recall_k)

                top20_rank_rel_idx = np.argpartition(-scores_rank, actual_rank_k - 1, axis=1)[:, :actual_rank_k]
                top20_final_idx = np.take_along_axis(top80_idx, top20_rank_rel_idx, axis=1)

                # =========================================================
                # 归档结果
                # =========================================================
                all_pred_recall_20.extend(cands[top20_recall_idx].tolist())
                all_pred_recall_80.extend(cands[top80_idx].tolist())
                all_pred_final_20.extend(cands[top20_final_idx].tolist())
                all_tids.extend([g[1] for g in group[batch_start:batch_end]])
                
                del X, X_rank, scores_recall, scores_rank

        infer_time = time.time() - t0

        hits_recall_20, hits_recall_80, hits_final_20 = [], [], []

        for i in range(len(all_tids)):
            tid = all_tids[i]
            true_set = gt_dict.get(tid, set())
            if len(true_set) == 0: continue

            hits_recall_20.append(len(set(all_pred_recall_20[i]) & true_set))
            hits_recall_80.append(len(set(all_pred_recall_80[i]) & true_set))
            hits_final_20.append(len(set(all_pred_final_20[i]) & true_set))

        mean_rec20 = np.mean(hits_recall_20) if hits_recall_20 else 0.0
        mean_rec80 = np.mean(hits_recall_80) if hits_recall_80 else 0.0
        mean_fin20 = np.mean(hits_final_20) if hits_final_20 else 0.0

        print(f"\n✅ [{year}] 级联推理完成！耗时: {infer_time:.1f}s | 评估数量 N={len(hits_recall_20):,}")
        print(f"  [对照组] 纯 Recall 模型直接选 Top20 的平均命中: {mean_rec20:.2f}")
        print(f"  [天花板] Recall 模型送给 Rank 的 Top80 的平均命中: {mean_rec80:.2f}")
        print(f"  [最终版] Recall(80) + Rank(20) 精排后的平均命中: {mean_fin20:.2f}  🚀🚀🚀")

        return year, mean_rec20, mean_rec80, mean_fin20

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2000)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Query抽样比例 0-1")
    p.add_argument("--recall-k", type=int, default=80, help="召回阶段送给精排的城市数量")
    p.add_argument("--rank-k", type=int, default=20, help="最终精排选出的城市数量")
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    
    if not MODEL_RECALL_PATH.exists() or not MODEL_RANK_PATH.exists():
        print(f"❌ 找不到模型文件！请确保路径配置正确。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" 🚀 Two-Stage Cascade Inference (硬负样本精排版): {years[0]}-{years[-1]}")
    print(f" ⚙️  参数策略: Query 取 {args.sample_ratio:.1%} | 召回 Top {args.recall_k} -> 精排 Top {args.rank_k}")
    print(f"{'='*60}\n")

    predictor = CascadePredictor(MODEL_RECALL_PATH, MODEL_RANK_PATH, DB_PATH, CACHE_DIR, CPU_COUNT)

    for y in years:
        predictor.run_year(y, recall_k=args.recall_k, rank_k=args.rank_k, sample_ratio=args.sample_ratio)

if __name__ == '__main__':
    main()