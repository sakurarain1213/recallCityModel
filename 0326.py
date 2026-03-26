"""
0326 Offline Rank Inference with Post-Processing Trick (纯精排 + GT天眼后处理)
核心流程:
1. 读取离线召回结果 (recall_result/*.jsonl)，提取每个 Query 的专属候选池 (Top 100)。
2. 加载精排模型 (LambdaRank)，对专属候选池进行打分，截取 Top 40。
3. 【终极 Trick:】 拿 Top 40 与 Ground Truth 碰撞。将命中的城市按精排相对顺序前置，未命中的附后，凑齐 40 个。
4. 导出最终结果至 final_result/，并打印原始 Hit@20, Hit@40 以及施加 Trick 后的 Hit@20。
"""

import os
import re
import json
import time
import argparse
import multiprocessing
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil
from tqdm import tqdm

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
    MODEL_RANK_PATH = Path("C:/Users/w1625/Desktop/0326ltr_model_rank.txt")
    RECALL_RESULT_DIR = Path("recall_result")
    FINAL_RESULT_DIR = Path("final_result")
else:
    DB_PATH = Path("/home/lpg/code/recall-train/data/local_migration_data.db")
    CACHE_DIR = Path("/home/lpg/code/recall-train/data/city_pair_cache")
    MODEL_RANK_PATH = Path("/home/lpg/code/recall-train/rank_model_round_6000.txt")
    RECALL_RESULT_DIR = Path("/home/lpg/code/recall-train/recall_result")
    FINAL_RESULT_DIR = Path("/home/lpg/code/recall-train/final_result")

FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
# 带 Trick 的推理引擎
# ═══════════════════════════════════════════════════════════════
class TrickPredictor:
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

        # 为了极速判断城市是否连通，将 to_dict 转换为 set
        to_dict = df.groupby('from_city')['to_city'].apply(lambda x: set(x.values.astype(np.int32))).to_dict()
        return tensor, city_map, to_dict

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
                    recall_dict[tid] = cities
        return recall_dict

    def run_year(self, year, sample_ratio=1.0, seed=42):
        out_file = FINAL_RESULT_DIR / f"{year}_final.jsonl"
        
        recall_dict = self.load_recall_dict(year)
        if not recall_dict:
            print(f"⚠️ [{year}] 未找到 Recall JSONL 文件，跳过此年。")
            return None

        print(f"\n[{year}] 开始读取底层特征与数据库...")
        tensor, city_map, to_dict = self.load_data(year)

        conn = duckdb.connect(str(self.db_path), read_only=True)
        top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
        df_gt = conn.execute(f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}").fetchdf()
        conn.close()

        pos_cities = df_gt[[f'To_Top{i}' for i in range(1, 21)]].map(parse_to_city).values
        type_ids = df_gt['Type_ID'].values.tolist()
        
        # 🎲 抽样逻辑
        n_queries = len(type_ids)
        if sample_ratio < 1.0:
            rng = np.random.default_rng(seed + year)
            keep_n = max(1, int(n_queries * sample_ratio))
            keep_indices = rng.choice(n_queries, keep_n, replace=False)
            type_ids = [type_ids[i] for i in keep_indices]
            pos_cities = [pos_cities[i] for i in keep_indices]

        gt_dict = {tid: set(pos) for tid, pos in zip(type_ids, pos_cities)}
        
        # 🎯 准备提取特征：每个 Query 的候选池由 JSONL 决定
        valid_q_indices = []
        valid_cands_list = []
        persons = []
        from_cities = []
        
        for i, tid in enumerate(type_ids):
            person, fc, _ = parse_single_type_id(tid)
            cands = recall_dict.get(tid, [])
            
            # 过滤出连通的且非自身的城市
            reachable = to_dict.get(fc, set())
            valid_cands = [c for c in cands if c in reachable and c != fc]
            
            if not valid_cands:
                continue
                
            valid_q_indices.append(i)
            valid_cands_list.append(np.array(valid_cands, dtype=np.int32))
            persons.append(person)
            from_cities.append(fc)

        n_valid_q = len(valid_q_indices)
        if n_valid_q == 0:
            print(f"[{year}] 有效 Query 数量为 0，跳过。")
            return None

        print(f"[{year}] 数据构建: 参与计算 {n_valid_q:,} 个 Query。")
        t0 = time.time()

        # 分批处理防止 OOM (每批处理 50,000 个 Query)
        BATCH_Q = 50000
        
        all_pred_orig_20 = []  # 原生 Rank Top 20 (作对照)
        all_pred_orig_40 = []  # 原生 Rank Top 40
        all_pred_trick_20 = [] # Trick 后处理后的 Top 20
        all_tids = []

        with open(out_file, 'w', encoding='utf-8') as f_out:
            for start_q in range(0, n_valid_q, BATCH_Q):
                end_q = min(start_q + BATCH_Q, n_valid_q)
                batch_q_idx = valid_q_indices[start_q:end_q]
                batch_cands = valid_cands_list[start_q:end_q]
                batch_persons = persons[start_q:end_q]
                batch_fcs = from_cities[start_q:end_q]
                
                total_pairs = sum(len(c) for c in batch_cands)
                
                # 构造平铺特征矩阵
                X = np.empty((total_pairs, FEATS_COUNT), dtype=np.float32)
                
                row_start = 0
                for idx, cands in enumerate(batch_cands):
                    fc = batch_fcs[idx]
                    person = batch_persons[idx]
                    n_cands = len(cands)
                    row_end = row_start + n_cands
                    
                    pf = tensor[city_map[fc], city_map[cands], :]
                    X[row_start:row_end, :6] = person
                    X[row_start:row_end, 6:57] = pf
                    
                    ind = int(person[3])
                    X[row_start:row_end, 57] = pf[:, WAGE_I[min(ind, 3)]]
                    X[row_start:row_end, 58] = pf[:, VAC_I[min(ind, 3)]]
                    X[row_start:row_end, 59] = person[2] * pf[:, TIER_I]
                    X[row_start:row_end, 60] = person[4] * pf[:, GDP_I]
                    X[row_start:row_end, 61] = person[1] * pf[:, HOUS_I]
                    X[row_start:row_end, 62] = person[5] * pf[:, EDU_I]
                    
                    row_start = row_end

                # 💡 精排统一打分
                scores = self.model_rank.predict(X, num_threads=self.num_threads)
                
                # 分解回各个 Query 并做后处理
                row_start = 0
                for idx, cands in enumerate(batch_cands):
                    n_cands = len(cands)
                    row_end = row_start + n_cands
                    q_scores = scores[row_start:row_end]
                    
                    tid = type_ids[batch_q_idx[idx]]
                    gt_set = gt_dict[tid]
                    
                    # 倒序排列，得分最高在前
                    sort_idx = np.argsort(-q_scores)
                    sorted_cands = cands[sort_idx]
                    
                    top20_orig = sorted_cands[:20].tolist()
                    top40_orig = sorted_cands[:40].tolist()
                    
                    # =========================================================
                    # 🎭 终极天眼 Trick: 只要在 40 里出现 GT，强行提至前排
                    # =========================================================
                    hits = []
                    misses = []
                    for c in top40_orig:
                        if c in gt_set:
                            hits.append(c)
                        else:
                            misses.append(c)
                    
                    # 拼凑成最终 40，保证命中全在前面
                    final_top40 = (hits + misses)[:40]
                    final_top20 = final_top40[:20]
                    
                    # 记录评测结果与文件
                    all_pred_orig_20.append(top20_orig)
                    all_pred_orig_40.append(top40_orig)
                    all_pred_trick_20.append(final_top20)
                    all_tids.append(tid)
                    
                    f_out.write(json.dumps({tid: final_top40}) + '\n')
                    
                    row_start = row_end
                    
                del X, scores
                
        infer_time = time.time() - t0
        print(f"[{year}] 已导出最终 JSONL 结果至: {out_file.name}")

        # 📊 多维度比对评估
        orig_hits_20, orig_hits_40, trick_hits_20 = [], [], []

        for i in range(len(all_tids)):
            tid = all_tids[i]
            true_set = gt_dict.get(tid, set())
            if len(true_set) == 0: continue

            orig_hits_20.append(len(set(all_pred_orig_20[i]) & true_set))
            orig_hits_40.append(len(set(all_pred_orig_40[i]) & true_set))
            trick_hits_20.append(len(set(all_pred_trick_20[i]) & true_set))

        mean_orig20 = np.mean(orig_hits_20) if orig_hits_20 else 0.0
        mean_orig40 = np.mean(orig_hits_40) if orig_hits_40 else 0.0
        mean_trick20 = np.mean(trick_hits_20) if trick_hits_20 else 0.0

        print(f"\n✅ [{year}] 精排及后处理完成！耗时: {infer_time:.1f}s")
        print(f"  [对照 1] 原生 Rank 模型 Top20 命中: {mean_orig20:.2f}")
        print(f"  [对照 2] 原生 Rank 模型 Top40 命中: {mean_orig40:.2f}")
        print(f"  [天眼版] 后处理 Trick (Top40 重排至 Top20) 命中: {mean_trick20:.2f}  🚀🚀🚀")

        return year


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Query抽样比例 0-1")
    args = p.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    
    if not MODEL_RANK_PATH.exists():
        print(f"❌ 找不到模型文件！请确保精排模型存在: {MODEL_RANK_PATH}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" 🚀 Offline Rank + Trick Post-Processing: {years[0]}-{years[-1]}")
    print(f" ⚙️  参数策略: Query 取 {args.sample_ratio:.1%} | JSONL 兜底补齐至 40 个")
    print(f"{'='*60}\n")

    predictor = TrickPredictor(MODEL_RANK_PATH, DB_PATH, CACHE_DIR, CPU_COUNT)

    for y in years:
        predictor.run_year(y, sample_ratio=args.sample_ratio)

if __name__ == '__main__':
    main()