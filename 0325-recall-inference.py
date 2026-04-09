"""
0325 Binary Classification 本地敏捷推理脚本 - Windows 测试满血版
(基于 337 全局城市池架构修正)
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
    MODEL_PATH = Path("C:/Users/w1625/Desktop/recall_model_0329.txt") 
    CITY_NODES_PATH = Path("data/city_nodes.jsonl") # ✅ 新增
else:
    DB_PATH = PROJECT_ROOT / "data" / "local_migration_data.db"
    CACHE_DIR = PROJECT_ROOT / "data" / "city_pair_cache"
    MODEL_PATH = PROJECT_ROOT / "output" / "models" / "0325ltr_model_rapid.txt"
    CITY_NODES_PATH = PROJECT_ROOT / "data" / "city_nodes.jsonl" # ✅ 新增

MODEL_BIN_PATH = MODEL_PATH.with_suffix('.mcl')
OUTPUT_DIR = Path("recall_result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def parse_single_type_id(tid: str, from_city: str = None) -> tuple:
    parts = tid.split('_')
    fc = int(parts[6]) if len(parts) == 7 else int(from_city)
    person = np.array([
        GENDER_MAP[parts[0]], AGE_MAP[parts[1]], EDU_MAP[parts[2]],
        IND_MAP[parts[3]], INC_MAP[parts[4]], FAM_MAP[parts[5]]
    ], dtype=np.float32)
    return person, fc, tid if len(parts) == 7 else f"{tid}_{fc}"

def ensure_binary_model(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    bin_path = model_path.with_suffix('.mcl')
    if not bin_path.exists():
        print(f"[Convert] 转换为二进制格式加速加载: {bin_path}")
        lgb.Booster(model_file=str(model_path)).save_model(str(bin_path))
    return bin_path

def print_model_info(model_path: Path):
    print(f"\n{'='*60}")
    print(f" 📊 二分类模型预检: {model_path.name}")
    print(f"{'='*60}")
    if not model_path.exists():
        print("❌ 找不到模型文件！请检查路径。")
        sys.exit(1)
        
    booster = lgb.Booster(model_file=str(model_path))
    print(f"  🌳 树的总数量 (num_trees): {booster.num_trees()}")
    print(f"  🔢 总特征数量: {booster.num_feature()}")
    
    imp = pd.DataFrame({
        'feature': booster.feature_name(),
        'importance': booster.feature_importance('gain'),
    }).sort_values('importance', ascending=False)
    
    print(f"\n  🏆 Top 15 特征重要性排名 (Gain):")
    for idx, r in imp.head(15).iterrows():
        print(f"    {int(idx)+1:2d}. {r['feature']:<25}: {r['importance']:,.0f}")
    print(f"{'='*60}\n")

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

# ═══════════════════════════════════════════════════════════════
# 推理引擎
# ═══════════════════════════════════════════════════════════════
class FastPredictor:
    def __init__(self, model_path, db_path, cache_dir, num_threads=1):
        self.num_threads = num_threads
        self.model = lgb.Booster(model_file=str(model_path))
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
        
        # 返回 all_ids 提供安全的索引范围
        return tensor, city_map, all_ids

    def run_year(self, year, top_k_list=[20, 30, 40, 50, 60, 70, 80, 90, 100], sample_ratio=1.0, city_ratio=1.0, seed=42):
        out_file = OUTPUT_DIR / f"{year}.jsonl"

        max_top_k = max(top_k_list) 

        print(f"[{year}] 开始读取数据...")
        tensor, city_map, valid_ids = self.load_data(year)

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
        gt_dict_top5 = {tid: set(pos[:5]) for tid, pos in zip(type_ids, pos_cities)}

        # ✅ 核心更新：使用 337 个全局城市节点作为候选池！
        global_city_pool = load_global_cities(CITY_NODES_PATH)
        print(f"[{year}] 构建标准候选城市池: 共 {len(global_city_pool)} 个核心城市")

        fc_groups = defaultdict(list)
        for tid in type_ids:
            person, fc_int, full_tid = parse_single_type_id(tid)
            fc_groups[fc_int].append((person, full_tid))

        cands_cache = {}
        for fc_int in fc_groups:
            # 去除出发地
            cands = global_city_pool[global_city_pool != fc_int]
            # 安全锁：确保候选城市在 parquet 的城市列表中，防止越界
            cands = np.intersect1d(cands, valid_ids)
                
            if city_ratio < 1.0 and len(cands) > 0:
                target_cands_n = max(20, int(len(cands) * city_ratio))
                if target_cands_n < len(cands):
                    rng_city = np.random.default_rng(seed + fc_int)
                    cands = rng_city.choice(cands, target_cands_n, replace=False)
                        
            cands_cache[fc_int] = cands

        total_rows = sum(len(cands_cache[fc]) * len(grp) for fc_groups, grp in fc_groups.items() for fc in [fc_groups])
        print(f"[{year}] 数据构建完毕: 参与计算 {len(type_ids):,} 个 Query, 展开后共 {total_rows:,} 行")

        all_pred_tids = []
        all_pred_cities = []
        t0 = time.time()

        MAX_BATCH_ROWS = int(AVAIL_MEM_GB * 1024**3 * 0.2 / (FEATS_COUNT * 4))
        MAX_BATCH_ROWS = max(5000, min(MAX_BATCH_ROWS, 10_000_000))
        print(f"[{year}] 分批引擎启动: 每批最高运算 {MAX_BATCH_ROWS:,} 行, 并发线程数 {self.num_threads}")

        processed_fc = 0
        total_fc = len(fc_groups)

        for fc_int, group in fc_groups.items():
            cands = cands_cache[fc_int]
            if len(cands) == 0: continue

            K, C = len(group), len(cands)
            pf = tensor[city_map[fc_int], city_map[cands], :]
            persons = np.array([g[0] for g in group], dtype=np.float32)

            batch_size = max(1, MAX_BATCH_ROWS // C)
            all_top_idx = []

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

                scores_batch = self.model.predict(
                    X.reshape(-1, FEATS_COUNT),
                    num_threads=self.num_threads,
                    num_iteration=800
                )
                scores_2d = scores_batch.reshape(K_batch, C)

                if C > max_top_k:
                    top_idx = np.argpartition(-scores_2d, max_top_k, axis=1)[:, :max_top_k]
                    top_scores = np.take_along_axis(scores_2d, top_idx, axis=1)
                    sort_order = np.argsort(-top_scores, axis=1)
                    top_idx = np.take_along_axis(top_idx, sort_order, axis=1)
                else:
                    top_idx = np.argsort(-scores_2d, axis=1)

                all_top_idx.append(top_idx)
                del X, scores_batch, scores_2d

            pred_cities = cands[np.vstack(all_top_idx)]
            all_pred_tids.extend([g[1] for g in group])
            all_pred_cities.extend(pred_cities.tolist())
            del all_top_idx, pf, persons

            processed_fc += 1
            if processed_fc % 50 == 0:
                elapsed = time.time() - t0
                print(f"[{year}] 进度: {processed_fc}/{total_fc} 城市 | 已处理 {len(all_pred_tids):,} Query | 耗时 {elapsed:.1f}s")

        infer_time = time.time() - t0

        with open(out_file, 'w', encoding='utf-8') as f:
            for tid, cities in zip(all_pred_tids, all_pred_cities):
                f.write(json.dumps({tid: cities[:100]}) + '\n')
        print(f"[{year}] 已保存推理结果到: {out_file}")

        metrics_hit5 = {k: [] for k in top_k_list}
        metrics_hit20 = {k: [] for k in top_k_list}
        for i in range(len(all_pred_tids)):
            tid = all_pred_tids[i]
            true_set = gt_dict.get(tid, set())
            if len(true_set) == 0:
                continue
            true_top5 = gt_dict_top5.get(tid, set())
            pred_cities_k = all_pred_cities[i]
            for k in top_k_list:
                pred_set_k = set(pred_cities_k[:k])
                metrics_hit5[k].append(len(pred_set_k & true_top5))
                metrics_hit20[k].append(len(pred_set_k & true_set))

        results = {}
        for k in top_k_list:
            results[k] = {
                'hit5': np.mean(metrics_hit5[k]) if metrics_hit5[k] else 0.0,
                'hit20': np.mean(metrics_hit20[k]) if metrics_hit20[k] else 0.0,
            }

        metrics_str = ' | '.join([f"@{k}: H5={results[k]['hit5']:.2f} H20={results[k]['hit20']:.2f}" for k in top_k_list])
        print(f"✅ [{year}] 推理耗时: {infer_time:.1f}s | {metrics_str} (N={len(all_pred_tids):,})")
        return year, results, len(all_pred_tids)

def process_year(year, model_path, db_path, cache_dir, num_threads, sample_ratio, city_ratio):
    predictor = FastPredictor(model_path, db_path, cache_dir, num_threads)
    return predictor.run_year(year, top_k_list=[20, 30, 40, 50, 60, 70, 80, 90, 100], sample_ratio=sample_ratio, city_ratio=city_ratio)

def main():
    default_workers = 1
    default_threads = min(CPU_COUNT, 24)
    TOP_K_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=default_workers, help=f"并行进程数 (建议本地设为 1)")
    p.add_argument("--threads", type=int, default=default_threads, help=f"LightGBM 调用的并行线程数")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2020)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Query抽样比例 0-1")
    p.add_argument("--city-ratio", type=float, default=1.0, help="候选城市抽样比例 0-1")
    args = p.parse_args()

    print_model_info(MODEL_PATH)

    years = list(range(args.start_year, args.end_year + 1))
    model_path = ensure_binary_model(MODEL_PATH)

    print(f"\n{'='*60}")
    print(f" 🚀 Binary 本地强力起飞: {years[0]}-{years[-1]} ({len(years)} 年)")
    print(f" ⚙️  参数策略: Query 取 {args.sample_ratio:.1%} | 候选池使用 337 全局城市")
    print(f" 📊 评估指标: Hit20@{TOP_K_LIST}")
    print(f" 🏎️  算力配置: LightGBM 使用 {args.threads} 线程运算")
    print(f"{'='*60}\n")

    all_results_hit5 = {k: [] for k in TOP_K_LIST}
    all_results_hit20 = {k: [] for k in TOP_K_LIST}
    total_queries = 0

    for year in years:
        print(f"\n{'='*40} 年份 {year} {'='*40}")

        predictor = FastPredictor(model_path, DB_PATH, CACHE_DIR, args.threads)
        _, results, n_queries = predictor.run_year(
            year,
            top_k_list=TOP_K_LIST,
            sample_ratio=args.sample_ratio,
            city_ratio=args.city_ratio
        )

        hit20_at_100 = results[100]['hit20']
        hit_rate_20_in_100 = hit20_at_100 / 20.0 * 100
        hit5_at_100 = results[100]['hit5']
        hit_rate_5_in_100 = hit5_at_100 / 5.0 * 100
        print(f"📊 [{year}] 前100中命中Top5: {hit_rate_5_in_100:.1f}% ({hit5_at_100:.2f}/5) | 命中Top20: {hit_rate_20_in_100:.1f}% ({hit20_at_100:.2f}/20)")

        for k in TOP_K_LIST:
            all_results_hit5[k].append(results[k]['hit5'])
            all_results_hit20[k].append(results[k]['hit20'])
        total_queries += n_queries

    print(f"\n{'='*60}")
    print(f" 🎯 二分类召回任务圆满结束: 指标统计聚合完毕")
    print(f" 📋 有效参与评估的 Query 总数: {total_queries:,}")
    print(f" 📈 最终大盘综合指标 (各年均值):")
    for k in TOP_K_LIST:
        mean_h5 = np.mean(all_results_hit5[k])
        mean_h20 = np.mean(all_results_hit20[k])
        print(f"    Recall@{k}: Mean Hit5={mean_h5:.2f}  Mean Hit20={mean_h20:.2f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True) 
    main()