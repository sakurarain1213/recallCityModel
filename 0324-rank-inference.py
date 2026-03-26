"""
0326 Pure Rank Inference (纯精排极速推理版)
核心流程:
1. 读取离线召回结果 (recall_result/*.jsonl)，提取每个 Query 的 Top 100 候选池。
2. 【极速矩阵构造】利用 Numpy 向量化索引，秒级铺平并提取特征。
3. 加载精排模型 (LambdaRank) 进行打分并倒序排列。
4. 截取 Top 40 保存至 rank_result/*.jsonl。
5. 评测并打印 Hit20 @ 20, 40, 60, 80。
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

print(f"[Env] Python: {sys.version}")
print(f"[Env] pandas: {pd.__version__}, numpy: {np.__version__}, lightgbm: {lgb.__version__}")
print(f"[Hardware] CPU: {CPU_COUNT} 核 | 内存: {TOTAL_MEM_GB:.1f}GB (可用: {AVAIL_MEM_GB:.1f}GB)")

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.resolve()

if os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    MODEL_RANK_PATH = Path("C:/Users/w1625/Desktop/0326ltr_model_rank.txt") # ✅ 替换为你的精排模型路径
    RECALL_RESULT_DIR = Path("recall_result")
    RANK_RESULT_DIR = Path("rank_result") # ✅ 新的输出目录
else:
    DB_PATH = Path("/home/lpg/code/recall-train/data/local_migration_data.db")
    CACHE_DIR = Path("/home/lpg/code/recall-train/data/city_pair_cache")
    MODEL_RANK_PATH = Path("/home/lpg/code/recall-train/output/models/0326ltr_model_rank.txt")
    RECALL_RESULT_DIR = Path("/home/lpg/code/recall-train/recall_result")
    RANK_RESULT_DIR = Path("/home/lpg/code/recall-train/rank_result")

RANK_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════════════════
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
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
# 推理引擎
# ═══════════════════════════════════════════════════════════════
class RankPredictor:
    def __init__(self, model_rank_path, db_path, cache_dir, num_threads=1):
        self.num_threads = num_threads
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
        valid_features = df[CACHE_FEAT_COLS].values.astype(np.float32)
        np.nan_to_num(valid_features, copy=False, nan=0.0)
        tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = valid_features

        # 返回 valid_ids 提供安全的索引范围
        return tensor, city_map, all_ids

    def load_recall_dict(self, year: int) -> dict:
        jsonl_path = RECALL_RESULT_DIR / f"{year}_local_sample.jsonl"
        if not jsonl_path.exists():
            return {}
        
        recall_dict = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                for tid, cities in data.items():
                    if isinstance(cities, dict):
                        cities = cities.get('top_cities', [])
                    # ⚠️ 明确限制：读取 Recall 的前 100 个
                    recall_dict[tid] = cities[:100]
        return recall_dict

    def run_year(self, year, top_k_eval=[20, 40, 60, 80], save_k=40, sample_ratio=1.0, seed=42):
        out_file = RANK_RESULT_DIR / f"{year}_rank.jsonl"
        
        recall_dict = self.load_recall_dict(year)
        if not recall_dict:
            print(f"⚠️ [{year}] 未找到 Recall JSONL 文件，跳过此年。")
            return None

        print(f"\n[{year}] 开始读取底层特征与数据库...")
        tensor, city_map, valid_ids = self.load_data(year)

        conn = duckdb.connect(str(self.db_path), read_only=True)
        top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
        df_gt = conn.execute(f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}").fetchdf()
        conn.close()

        pos_cities = df_gt[[f'To_Top{i}' for i in range(1, 21)]].map(parse_to_city).values
        type_ids = df_gt['Type_ID'].values.tolist()
        
        # 🎲 抽样逻辑
        n_queries = len(type_ids)
        if sample_ratio < 1.0:
            rng_sample = np.random.default_rng(seed + year)
            keep_n = max(1, int(n_queries * sample_ratio))
            keep_indices = rng_sample.choice(n_queries, keep_n, replace=False)
            type_ids = [type_ids[i] for i in keep_indices]
            pos_cities = [pos_cities[i] for i in keep_indices]

        gt_dict = {tid: set(pos) for tid, pos in zip(type_ids, pos_cities)}
        
        # 🎯 准备提取特征：候选池由 JSONL 决定
        valid_q_indices = []
        valid_cands_list = []
        persons = []
        from_cities = []
        
        for i, tid in enumerate(type_ids):
            person, fc, _ = parse_single_type_id(tid)
            cands = recall_dict.get(tid, [])
            
            # 确保送入特征库的候选都在 parquet 文件内，并且不含自身
            cands = np.intersect1d(cands, valid_ids)
            cands = cands[cands != fc]
            
            if len(cands) == 0:
                continue
                
            valid_q_indices.append(i)
            valid_cands_list.append(np.array(cands, dtype=np.int32))
            persons.append(person)
            from_cities.append(fc)

        n_valid_q = len(valid_q_indices)
        if n_valid_q == 0:
            print(f"[{year}] 有效 Query 数量为 0，跳过。")
            return None

        print(f"[{year}] 数据构建: 参与计算 {n_valid_q:,} 个 Query。")
        t0 = time.time()

        # 分批参数
        BATCH_Q = 100_000
        
        all_pred_cities = [] 
        all_tids = []

        with open(out_file, 'w', encoding='utf-8') as f_out:
            for start_q in range(0, n_valid_q, BATCH_Q):
                end_q = min(start_q + BATCH_Q, n_valid_q)
                batch_q_idx = valid_q_indices[start_q:end_q]
                batch_cands = valid_cands_list[start_q:end_q]
                batch_persons = persons[start_q:end_q]
                batch_fcs = from_cities[start_q:end_q]
                
                # =========================================================
                # 🚀 向量化展开数组，极速构建矩阵
                # =========================================================
                lengths = np.array([len(c) for c in batch_cands], dtype=np.int32)
                total_pairs = lengths.sum()
                
                flat_cands = np.concatenate(batch_cands)
                flat_fcs = np.repeat(batch_fcs, lengths)
                flat_persons = np.repeat(batch_persons, lengths, axis=0)
                
                flat_pf = tensor[city_map[flat_fcs], city_map[flat_cands], :]
                
                X = np.empty((total_pairs, FEATS_COUNT), dtype=np.float32)
                X[:, :6] = flat_persons
                X[:, 6:57] = flat_pf
                
                inds = np.clip(flat_persons[:, 3].astype(np.int32), 0, 3)
                X[:, 57] = flat_pf[np.arange(total_pairs), WAGE_I[inds]]
                X[:, 58] = flat_pf[np.arange(total_pairs), VAC_I[inds]]
                X[:, 59] = flat_persons[:, 2] * flat_pf[:, TIER_I]
                X[:, 60] = flat_persons[:, 4] * flat_pf[:, GDP_I]
                X[:, 61] = flat_persons[:, 1] * flat_pf[:, HOUS_I]
                X[:, 62] = flat_persons[:, 5] * flat_pf[:, EDU_I]

                # =========================================================
                # 💡 精排打分
                # =========================================================
                scores = self.model_rank.predict(X, num_threads=self.num_threads)
                
                # 拆分回 Query
                split_indices = np.cumsum(lengths)[:-1]
                q_scores_list = np.split(scores, split_indices)
                q_cands_list = np.split(flat_cands, split_indices)
                
                for i in range(len(batch_cands)):
                    q_scores = q_scores_list[i]
                    cands = q_cands_list[i]
                    tid = type_ids[batch_q_idx[i]]
                    
                    # 按精排分数倒序排列
                    sort_idx = np.argsort(-q_scores)
                    sorted_cands = cands[sort_idx].tolist()
                    
                    # 记录全部排序结果用于多指标评估
                    all_pred_cities.append(sorted_cands)
                    all_tids.append(tid)
                    
                    # ✅ 保存要求：仅取前 40 个城市写入 JSONL
                    f_out.write(json.dumps({tid: sorted_cands[:save_k]}) + '\n')
                    
                del X, flat_cands, flat_fcs, flat_persons, flat_pf, scores
                
        infer_time = time.time() - t0
        print(f"[{year}] 已导出精排 Top {save_k} 结果至: {out_file.name}")

        # 📊 多维度比对评估
        metrics = {k: [] for k in top_k_eval}
        
        for i in range(len(all_tids)):
            tid = all_tids[i]
            true_set = gt_dict.get(tid, set())
            if len(true_set) == 0: continue
            
            sorted_cands = all_pred_cities[i]
            for k in top_k_eval:
                pred_set_k = set(sorted_cands[:k])
                metrics[k].append(len(pred_set_k & true_set))

        results = {}
        for k in top_k_eval:
            results[k] = np.mean(metrics[k]) if metrics[k] else 0.0

        metrics_str = ' | '.join([f"Hit20@{k}: {results[k]:.2f}" for k in top_k_eval])
        print(f"✅ [{year}] 精排推理完成！耗时: {infer_time:.1f}s | {metrics_str}")

        return year, results, len(all_tids)

def main():
    default_workers = 1
    default_threads = CPU_COUNT 
    TOP_K_EVAL = [20, 40, 60, 80] # 指定评估 K 列表

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=default_workers, help="并行进程数")
    p.add_argument("--threads", type=int, default=default_threads, help="LightGBM 并行线程数")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Query抽样比例 0-1")
    p.add_argument("--save-k", type=int, default=40, help="输出到JSONL保存的前 K 个城市数量")
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    
    if not MODEL_RANK_PATH.exists():
        print(f"❌ 找不到模型文件！请确保精排模型存在: {MODEL_RANK_PATH}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" 🚀 Pure Rank Inference (纯精排直出版): {years[0]}-{years[-1]}")
    print(f" ⚙️  参数策略: Query 取 {args.sample_ratio:.1%} | 保存 Top {args.save_k}")
    print(f" 📊 评估指标: Hit20@{TOP_K_EVAL}")
    print(f" 🏎️  算力配置: LightGBM 使用 {args.threads} 线程运算")
    print(f"{'='*60}\n")

    all_results = {k: [] for k in TOP_K_EVAL}
    total_queries = 0

    predictor = RankPredictor(MODEL_RANK_PATH, DB_PATH, CACHE_DIR, args.threads)

    for y in years:
        res = predictor.run_year(y, top_k_eval=TOP_K_EVAL, save_k=args.save_k, sample_ratio=args.sample_ratio)
        if res is None: continue
        
        _, year_results, n_queries = res
        for k in TOP_K_EVAL:
            all_results[k].append(year_results[k])
        total_queries += n_queries

    if total_queries > 0:
        print(f"\n{'='*60}")
        print(f" 🎯 精排推理任务圆满结束: 指标统计聚合完毕")
        print(f" 📋 有效参与评估的 Query 总数: {total_queries:,}")
        print(f" 📈 最终大盘综合指标 (各年均值):")
        for k in TOP_K_EVAL:
            mean_hit = np.mean(all_results[k])
            print(f"    Mean Hit20@{k}: {mean_hit:.2f}")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True) 
    main()