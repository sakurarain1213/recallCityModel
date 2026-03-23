"""
0323 LambdaRank Inference Script
支持单次推理、批量推理、以及年度全量推理并校验 Hit Rate
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# 路径配置 (与 Pipeline 保持一致)
# ═══════════════════════════════════════════════════════════════
if os.name == 'nt':
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    MODEL_PATH = Path("C:/Users/w1625/Desktop/ltr_model_latest.txt")
else:
    DB_PATH = Path("/data1/wxj/Recall_city_project/data/local_migration_data.db")
    CACHE_DIR = Path("/data1/wxj/Recall_city_project/data/city_pair_cache")
    MODEL_PATH = Path("/data1/wxj/Recall_city_project/output/models/checkpoints/ltr_model_latest.txt")

OUTPUT_DIR = Path("recall_result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 特征定义 (严格对齐 Pipeline)
# ═══════════════════════════════════════════════════════════════
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
DIFF_FEATS = ['gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff', 'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff', 'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff']
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

FEATS_COUNT = 63  # 6个人 + 51个缓存特征 + 6个交叉

AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}

# 交叉特征索引提前计算
WAGE_I = [RATIO_FEATS.index(c) for c in ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']]
VAC_I = [RATIO_FEATS.index(c) for c in ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']]
TIER_I = len(RATIO_FEATS) + ABS_FEATS.index('tier_diff')
GDP_I = RATIO_FEATS.index('gdp_per_capita_ratio')
HOUS_I = RATIO_FEATS.index('housing_price_avg_ratio')
EDU_I = RATIO_FEATS.index('education_score_ratio')


def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', str(s))
    return int(m.group(1)) if m else int(s)

def parse_single_type_id(tid: str, from_city: str = None) -> tuple:
    parts = tid.split('_')
    # 处理完整格式: F_20_EduHi_Agri_IncH_Split_3506
    if len(parts) == 7:
        fc = int(parts[6])
    # 处理分离格式: type_id="F_20_...", from_city="3506"
    elif len(parts) == 6 and from_city is not None:
        fc = int(from_city)
    else:
        raise ValueError(f"Invalid TypeID format: {tid}")
    
    person = np.array([
        GENDER_MAP[parts[0]], AGE_MAP[parts[1]], EDU_MAP[parts[2]],
        IND_MAP[parts[3]], INC_MAP[parts[4]], FAM_MAP[parts[5]]
    ], dtype=np.float32)
    return person, fc, tid if len(parts) == 7 else f"{tid}_{fc}"


# ═══════════════════════════════════════════════════════════════
# 推理引擎类
# ═══════════════════════════════════════════════════════════════
class MigrationPredictor:
    def __init__(self, model_path, db_path, cache_dir):
        print(f"[Init] Loading Model: {model_path}")
        self.model = lgb.Booster(model_file=str(model_path))
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache = {}  # year -> (tensor, city_map, to_dict, all_city_ids)

    def load_year_cache(self, year: int):
        if year in self.cache: return
        print(f"[Init] Loading Cache for {year}...")
        path = self.cache_dir / f"city_pairs_{year}.parquet"
        df = pd.read_parquet(path, columns=['from_city', 'to_city'] + CACHE_FEAT_COLS)
        
        all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
        max_id = int(all_ids.max())
        city_map = np.zeros(max_id + 1, dtype=np.int32)
        city_map[all_ids] = np.arange(len(all_ids))
        
        tensor = np.zeros((max_id + 1, max_id + 1, len(CACHE_FEAT_COLS)), dtype=np.float32)
        v_feats = df[CACHE_FEAT_COLS].values.astype(np.float32)
        np.nan_to_num(v_feats, copy=False, nan=0.0)
        tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = v_feats
        
        to_dict = df.groupby('from_city')['to_city'].apply(lambda x: x.values.astype(np.int32)).to_dict()
        self.cache[year] = (tensor, city_map, to_dict, all_ids)

    def _build_batch_features(self, queries, year, candidate_cities=None):
        tensor, city_map, to_dict, _ = self.cache[year]

        # 1. 解析 query 并按 from_city 分组
        cands_cache = {}
        fc_groups = defaultdict(list)  # fc_int -> [(person, full_tid), ...]

        for q in queries:
            tid = q.get('type_id') if isinstance(q, dict) else q
            fc = q.get('from_city') if isinstance(q, dict) else None
            person, fc_int, full_tid = parse_single_type_id(tid, fc)

            if fc_int not in cands_cache:
                cands = to_dict.get(fc_int, np.array([], dtype=np.int32))
                if len(cands) > 0:
                    cands = cands[cands != fc_int]
                    if candidate_cities is not None and len(cands) > 0:
                        cands = cands[np.isin(cands, candidate_cities)]
                cands_cache[fc_int] = cands

            if len(cands_cache[fc_int]) == 0: continue
            fc_groups[fc_int].append((person, full_tid))

        total_candidates = sum(len(cands_cache[fc]) * len(grp) for fc, grp in fc_groups.items())
        if total_candidates == 0:
            return None, [], [], []

        # 2. 按 from_city 分组填充：pf 只查一次，tile 给该城市的所有 query
        X = np.empty((total_candidates, FEATS_COUNT), dtype=np.float32)
        row = 0
        splits = [0]
        valid_tids = []
        all_candidates_flat = []

        for fc_int, group in fc_groups.items():
            cands = cands_cache[fc_int]
            C = len(cands)
            K = len(group)

            # pf 只查表一次 (C, 51)
            pf = tensor[city_map[fc_int], city_map[cands], :]
            pf_tiled = np.tile(pf, (K, 1))  # (K*C, 51)

            persons = np.array([g[0] for g in group], dtype=np.float32)  # (K, 6)
            persons_rep = np.repeat(persons, C, axis=0)  # (K*C, 6)

            n_rows = K * C
            X[row:row+n_rows, :6] = persons_rep
            X[row:row+n_rows, 6:57] = pf_tiled

            # 交叉特征 (向量化)
            inds = np.clip(persons_rep[:, 3].astype(np.int32), 0, 3)
            arange = np.arange(n_rows)
            X[row:row+n_rows, 57] = pf_tiled[arange, np.array(WAGE_I)[inds]]
            X[row:row+n_rows, 58] = pf_tiled[arange, np.array(VAC_I)[inds]]
            X[row:row+n_rows, 59] = persons_rep[:, 2] * pf_tiled[:, TIER_I]
            X[row:row+n_rows, 60] = persons_rep[:, 4] * pf_tiled[:, GDP_I]
            X[row:row+n_rows, 61] = persons_rep[:, 1] * pf_tiled[:, HOUS_I]
            X[row:row+n_rows, 62] = persons_rep[:, 5] * pf_tiled[:, EDU_I]

            for i in range(K):
                valid_tids.append(group[i][1])
                all_candidates_flat.append(cands)
                splits.append(row + (i + 1) * C)

            row += n_rows

        return X, valid_tids, all_candidates_flat, splits

    def batch_predict(self, queries, year, top_k=20, candidate_cities=None):
        self.load_year_cache(year)
        X, valid_tids, all_cands, splits = self._build_batch_features(queries, year, candidate_cities=candidate_cities)
        
        if X is None: return []
        
        scores = self.model.predict(X, num_threads=min(20, os.cpu_count()))
        results = []

        for i in range(len(valid_tids)):
            s, e = splits[i], splits[i+1]
            q_scores = scores[s:e]
            q_cands = all_cands[i]

            # 取 TopK：argpartition 比全排序 argsort 更快
            if len(q_scores) > top_k:
                top_idx = np.argpartition(-q_scores, top_k)[:top_k]
                top_idx = top_idx[np.argsort(-q_scores[top_idx])]
            else:
                top_idx = np.argsort(-q_scores)
            res_cities = q_cands[top_idx].tolist()
            results.append(res_cities)
            
        return valid_tids, results

    def run_full_year_eval(self, year, top_k=20, sample_size=5000):
        out_file = OUTPUT_DIR / f"{year}.jsonl"
        if out_file.exists():
            print(f"\n[Skip] {year} 年结果已存在: {out_file}, 跳过")
            return
        print(f"\n{'='*50}\n开始 {year} 年全量推理与校验\n{'='*50}")
        self.load_year_cache(year)

        # 1. 从 DB 加载当年的 Ground Truth
        print("[1/4] Loading Ground Truth from DB...")
        t0 = time.time()
        top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
        sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"
        conn = duckdb.connect(str(self.db_path), read_only=True)
        df_gt = conn.execute(sql).fetchdf()
        conn.close()

        pos_cities = df_gt[[f'To_Top{i}' for i in range(1, 21)]].map(parse_to_city).values
        type_ids = df_gt['Type_ID'].values.tolist()

        gt_dict = {tid: set(pos) for tid, pos in zip(type_ids, pos_cities)}

        # 从 Type_ID 最后一个字段提取当年所有出发城市作为候选目标城市集合
        gt_from_cities = np.array(list(set(int(tid.rsplit('_', 1)[-1]) for tid in type_ids)), dtype=np.int32)
        print(f"      -> 成功加载 {len(type_ids):,} 个 Query 的 GT, 候选城市数: {len(gt_from_cities)} (耗时: {time.time()-t0:.1f}s)")

        # 2. 按出发城市分组推理（每次 predict ~35万行，L3 缓存友好）
        print(f"[2/4] Running Full Inference (per-city batching)...")
        t1 = time.time()

        tensor, city_map, to_dict, _ = self.cache[year]

        # 按 from_city 分组所有 query
        fc_groups = defaultdict(list)  # fc_int -> [(person, full_tid), ...]
        for tid in type_ids:
            person, fc_int, full_tid = parse_single_type_id(tid)
            fc_groups[fc_int].append((person, full_tid))

        # 预计算每个 from_city 的候选列表
        cands_cache = {}
        for fc_int in fc_groups:
            cands = to_dict.get(fc_int, np.array([], dtype=np.int32))
            if len(cands) > 0:
                cands = cands[cands != fc_int]
                cands = cands[np.isin(cands, gt_from_cities)]
            cands_cache[fc_int] = cands

        all_pred_tids = []
        all_pred_cities = []
        n_threads = min(20, os.cpu_count())

        for fc_int in tqdm(fc_groups, desc="Inferring"):
            cands = cands_cache[fc_int]
            if len(cands) == 0: continue
            group = fc_groups[fc_int]
            K = len(group)
            C = len(cands)

            # pf 只查一次
            pf = tensor[city_map[fc_int], city_map[cands], :]  # (C, 51)
            persons = np.array([g[0] for g in group], dtype=np.float32)  # (K, 6)

            n_rows = K * C
            X = np.empty((n_rows, FEATS_COUNT), dtype=np.float32)
            persons_rep = np.repeat(persons, C, axis=0)  # (K*C, 6)
            pf_tiled = np.tile(pf, (K, 1))  # (K*C, 51)

            X[:, :6] = persons_rep
            X[:, 6:57] = pf_tiled

            # 交叉特征
            inds = np.clip(persons_rep[:, 3].astype(np.int32), 0, 3)
            arange = np.arange(n_rows)
            X[:, 57] = pf_tiled[arange, np.array(WAGE_I)[inds]]
            X[:, 58] = pf_tiled[arange, np.array(VAC_I)[inds]]
            X[:, 59] = persons_rep[:, 2] * pf_tiled[:, TIER_I]
            X[:, 60] = persons_rep[:, 4] * pf_tiled[:, GDP_I]
            X[:, 61] = persons_rep[:, 1] * pf_tiled[:, HOUS_I]
            X[:, 62] = persons_rep[:, 5] * pf_tiled[:, EDU_I]

            scores = self.model.predict(X, num_threads=n_threads)

            # 按 query 切分取 TopK
            for i in range(K):
                s, e = i * C, (i + 1) * C
                q_scores = scores[s:e]
                if len(q_scores) > top_k:
                    top_idx = np.argpartition(-q_scores, top_k)[:top_k]
                    top_idx = top_idx[np.argsort(-q_scores[top_idx])]
                else:
                    top_idx = np.argsort(-q_scores)
                all_pred_tids.append(group[i][1])
                all_pred_cities.append(cands[top_idx].tolist())

        print(f"      -> 推理完成！(耗时: {time.time()-t1:.1f}s)")

        # 3. 落盘为 JSONL
        print(f"[3/4] Saving results to {out_file}...")
        with open(out_file, 'w', encoding='utf-8') as f:
            for tid, cities in tqdm(zip(all_pred_tids, all_pred_cities), total=len(all_pred_tids), desc="Saving"):
                f.write(json.dumps({tid: cities}) + '\n')

        # 4. 采样计算命中率 Hit@20
        print(f"[4/4] Calculating Hit@{top_k} Rate...")
        
        # 挑选采样
        sample_indices = np.random.choice(len(all_pred_tids), size=min(sample_size, len(all_pred_tids)), replace=False)
        
        total_hits = 0
        total_gt = 0
        
        for idx in tqdm(sample_indices, desc="Evaluating"):
            tid = all_pred_tids[idx]
            pred_set = set(all_pred_cities[idx])
            true_set = gt_dict.get(tid, set())
            
            # 计算召回率 Recall (Hit Rate) = 交集数量 / GT数量
            hits = len(pred_set & true_set)
            total_hits += hits
            total_gt += len(true_set)
            
        hit_rate = total_hits / total_gt if total_gt > 0 else 0.0
        print(f"\n✅ 结果报告 ({year}):")
        print(f"   - 总推理 Query 数: {len(all_pred_tids):,}")
        print(f"   - 采样评测 Query 数: {len(sample_indices):,}")
        print(f"   - 平均命中率 (Recall@{top_k}): {hit_rate:.4%}")
        print(f"   - 结果已保存至: {out_file.absolute()}")


# ═══════════════════════════════════════════════════════════════
# 测试执行模块
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, nargs='*', default=None,
                   help="指定年份(可多个), 默认 2000-2020 全部")
    p.add_argument("--sample", type=int, default=5000, help="校验命中率的采样数")
    p.add_argument("--skip-demo", action="store_true", help="跳过示例演示")
    args = p.parse_args()

    years = args.year if args.year else list(range(2000, 2021))

    predictor = MigrationPredictor(
        model_path=MODEL_PATH,
        db_path=DB_PATH,
        cache_dir=CACHE_DIR
    )

    if not args.skip_demo:
        print("\n" + "="*50)
        print(" 示例 API 演示 (热身)")
        print("="*50)

        demo_year = years[0]
        # 示例 1：单次推理（完整 TypeID）
        print("\n[示例 1: Single Prediction (Complete TypeID)]")
        tid_1 = "F_20_EduHi_Agri_IncH_Split_3506"
        tids, res = predictor.batch_predict([tid_1], year=demo_year, top_k=5)
        print(f"Input: {tid_1} \nTop 5: {res[0]}")

        # 示例 2：单次推理（不完整 TypeID）
        print("\n[示例 2: Single Prediction (Incomplete TypeID)]")
        dict_2 = {"type_id": "M_30_EduMid_Mfg_IncM_Unit", "from_city": "1100"}
        tids, res = predictor.batch_predict([dict_2], year=demo_year, top_k=5)
        print(f"Input: {dict_2} \nTop 5: {res[0]}")

        # 示例 3：批量推理
        print("\n[示例 3: Batch Prediction (List)]")
        batch_q = [
            "F_20_EduHi_Agri_IncH_Split_3506",
            {"type_id": "M_30_EduMid_Mfg_IncM_Unit", "from_city": "1100"}
        ]
        tids, res = predictor.batch_predict(batch_q, year=demo_year, top_k=5)
        for t, r in zip(tids, res):
            print(f" -> Query: {t} | Top 5: {r}")

    # 批量全量推理 (自动跳过已存在的年份)
    print(f"\n{'='*60}")
    print(f" 全量推理: {years[0]}-{years[-1]} ({len(years)} 年)")
    print(f"{'='*60}")
    for y in years:
        predictor.run_full_year_eval(year=y, top_k=20, sample_size=args.sample)


if __name__ == '__main__':
    main()