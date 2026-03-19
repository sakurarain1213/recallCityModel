"""
城市迁移召回推理接口 (0319 版 - 与 0319build_bin_recall / 0319fast_recall_train 完全对齐)

最小依赖:
  - lightgbm
  - numpy
  - pandas
  - pyarrow (pandas read_parquet / read_feather 底层依赖)

数据依赖:
  - output/models/0319recall_model_final.txt  (训练好的 LightGBM 模型)
  - data/city_nodes.jsonl                     (城市 ID → Name 映射)
  - data/city_pair_cache/city_pairs_YYYY.parquet (城市对特征缓存)

用法:
  predict(type_id, year, from_city=None, top_k=20)
  batch_predict(queries, year, top_k=20)

运行示例:
  uv run 0319interface.py
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = Path("output/models/0319recall_model_final.txt")
CITY_NODES_PATH = Path("data/city_nodes.jsonl")
CACHE_DIR = Path("data/city_pair_cache")

# ═══════════════════════════════════════════════════════════════
# 特征定义 (从 0319build_bin_recall.py 原样复制, 保证完全对齐)
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
ABS_FEATS = [
    'to_tier', 'to_population_log', 'to_gdp_per_capita',
    'from_tier', 'from_population_log', 'tier_diff',
]
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = [
    'industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff', 'income_x_gdp_ratio',
    'age_x_housing_ratio', 'family_x_edu_score_ratio',
]
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# ═══════════════════════════════════════════════════════════════
# TypeID 解析映射 (与训练时 parse_type_id 一致)
# ═══════════════════════════════════════════════════════════════
AGE_MAPPING = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDUCATION_MAPPING = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
INDUSTRY_MAPPING = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INCOME_MAPPING = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAMILY_MAPPING = {'Split': 0, 'Unit': 1}
GENDER_MAPPING = {'M': 0, 'F': 1}


# ═══════════════════════════════════════════════════════════════
# 基础工具函数
# ═══════════════════════════════════════════════════════════════

def load_city_mapping() -> Dict[int, str]:
    """加载城市 ID → Name 映射"""
    with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
        return {int(json.loads(line.strip())['city_id']): json.loads(line.strip())['name'] for line in f}


def get_all_city_ids() -> List[int]:
    """获取所有城市 ID 列表"""
    with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
        return [int(json.loads(line.strip())['city_id']) for line in f]


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


def parse_type_id(type_id: str, from_city: str = None) -> Dict:
    """解析 TypeID: F_20_EduHi_Agri_IncH_Split_3506 或 F_20_EduHi_Agri_IncH_Split"""
    parts = type_id.split('_')
    if len(parts) == 7:
        gender, age, edu, job, income, family, dialect = parts
        from_city_final = dialect
    elif len(parts) == 6:
        gender, age, edu, job, income, family = parts
        if from_city is None:
            raise ValueError(f"TypeID '{type_id}' 缺少 from_city 参数")
        from_city_final = from_city
    else:
        raise ValueError(f"TypeID 格式错误: '{type_id}'")
    return {
        'gender': GENDER_MAPPING[gender], 'age_group': AGE_MAPPING[age],
        'education': EDUCATION_MAPPING[edu], 'industry': INDUSTRY_MAPPING[job],
        'income': INCOME_MAPPING[income], 'family': FAMILY_MAPPING[family],
        'From_City': int(from_city_final),
    }


# ═══════════════════════════════════════════════════════════════
# 候选构建 + 推理
# ═══════════════════════════════════════════════════════════════

def build_candidate_df(parsed_queries: List[Dict], year: int, all_cities: List[int],
                       cache_tensor=None, city_map_arr=None) -> pd.DataFrame:
    """
    构建候选城市 DataFrame（与训练流程完全一致: tensor join + 交叉特征）
    可传入预加载的 cache_tensor/city_map_arr 避免重复加载。
    """
    rows = [
        {'query_idx': i, 'From_City': pq['From_City'], 'To_City': city,
         'gender': pq['gender'], 'age_group': pq['age_group'], 'education': pq['education'],
         'industry': pq['industry'], 'income': pq['income'], 'family': pq['family']}
        for i, pq in enumerate(parsed_queries) for city in all_cities
    ]
    df = pd.DataFrame(rows)

    # tensor 极速 join (与 0319build_bin_recall / 0319evaluate 完全一致)
    if cache_tensor is None or city_map_arr is None:
        cache_tensor, city_map_arr = load_cache_for_year(year)

    from_idx = city_map_arr[df['From_City'].values]
    to_idx = city_map_arr[df['To_City'].values]
    extracted_feats = cache_tensor[from_idx, to_idx, :]

    for i, col in enumerate(CACHE_FEAT_COLS):
        df[col] = extracted_feats[:, i]

    # 交叉特征 (与训练完全一致)
    df = build_cross_features(df)

    # 填充缺失 (与训练完全一致)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(np.float32)

    return df


def predict(type_id: str, year: int, from_city: str = None, top_k: int = 20,
            model_path: str = str(MODEL_PATH), verbose: bool = True) -> List[Tuple[int, str, float]]:
    """
    单次推理: 给定一个 TypeID，返回 Top-K 推荐城市。

    Args:
        type_id: 如 "F_20_EduHi_Agri_IncH_Split_3506" (完整) 或 "F_20_EduHi_Agri_IncH_Split" (需 from_city)
        year: 使用哪一年的城市对特征
        from_city: 出发城市 ID (TypeID 不含城市时必填)
        top_k: 返回前 K 个城市
        model_path: 模型文件路径
        verbose: 是否打印详细信息

    Returns:
        [(city_id, city_name, score), ...]
    """
    if verbose:
        print(f"Loading model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    city_name_map = load_city_mapping()
    all_cities = get_all_city_ids()

    parsed_query = parse_type_id(type_id, from_city)
    if verbose:
        print(f"\nTypeID: {type_id}")
        print(f"From City: {parsed_query['From_City']} ({city_name_map.get(parsed_query['From_City'], 'Unknown')})")
        print(f"Building candidate features for {len(all_cities)} cities (year={year})...")

    df = build_candidate_df([parsed_query], year, all_cities)

    X = df[FEATS].copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype('category')

    if verbose:
        print(f"Predicting...")
    scores = model.predict(X)
    df['score'] = scores
    df_sorted = df.sort_values('score', ascending=False).head(top_k)

    results = [
        (int(row['To_City']), city_name_map.get(int(row['To_City']), 'Unknown'), row['score'])
        for _, row in df_sorted.iterrows()
    ]

    if verbose:
        print(f"\n{'='*70}\nTop {top_k} Predicted Cities:")
        print(f"{'Rank':<6} {'City ID':<10} {'City Name':<20} {'Score':<10}\n{'-'*70}")
        for rank, (cid, cname, score) in enumerate(results, 1):
            print(f"{rank:<6} {cid:<10} {cname:<20} {score:<10.6f}")
        print("=" * 70)

    return results


def batch_predict(queries: Union[List[str], List[Dict]], year: int, top_k: int = 20,
                  model_path: str = str(MODEL_PATH), verbose: bool = True) -> List[List[Tuple[int, str, float]]]:
    """
    批量推理: 给定多个 TypeID，一次性构建特征 + 推理，返回每个 query 的 Top-K。

    Args:
        queries: TypeID 字符串列表 或 字典列表 [{"type_id": ..., "from_city": ...}]
        year: 使用哪一年的城市对特征
        top_k: 每个 query 返回前 K 个城市
        model_path: 模型文件路径
        verbose: 是否打印详细信息

    Returns:
        [[( city_id, city_name, score), ...], ...]  每个 query 一个列表
    """
    if isinstance(queries[0], str):
        type_ids, from_cities = queries, [None] * len(queries)
    else:
        type_ids = [q['type_id'] for q in queries]
        from_cities = [q.get('from_city', None) for q in queries]

    n_queries = len(type_ids)
    if verbose:
        print(f"Batch predicting {n_queries} queries (year={year}, top_k={top_k})...")

    model = lgb.Booster(model_file=model_path)
    city_name_map = load_city_mapping()
    all_cities = get_all_city_ids()

    # 解析所有 TypeID
    if verbose:
        print(f"Parsing {n_queries} TypeIDs...")
    parsed_queries = []
    for tid, fc in zip(type_ids, from_cities):
        try:
            parsed_queries.append(parse_type_id(tid, fc))
        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to parse '{tid}': {e}")
            parsed_queries.append(None)

    valid_indices = [i for i, pq in enumerate(parsed_queries) if pq is not None]
    valid_parsed = [parsed_queries[i] for i in valid_indices]

    if len(valid_parsed) == 0:
        print("  [Error] No valid queries!")
        return [[] for _ in range(n_queries)]

    # 预加载 cache tensor (所有 query 共享同一年, 只加载一次)
    if verbose:
        print(f"Loading cache tensor for year {year}...")
    cache_tensor, city_map_arr = load_cache_for_year(year)

    # 批量构建特征
    if verbose:
        print(f"Building features for {len(valid_parsed)} queries x {len(all_cities)} cities...")
    df = build_candidate_df(valid_parsed, year, all_cities, cache_tensor, city_map_arr)

    del cache_tensor, city_map_arr

    # 批量预测
    if verbose:
        print(f"Predicting {len(df):,} rows...")
    X = df[FEATS].copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    scores = model.predict(X)
    df['score'] = scores

    # 按 query_idx 分组取 Top K
    if verbose:
        print(f"Extracting Top {top_k} for each query...")
    df_sorted = df.sort_values(['query_idx', 'score'], ascending=[True, False])
    grouped = df_sorted.groupby('query_idx').head(top_k)

    results_by_idx = {}
    for idx, group in grouped.groupby('query_idx'):
        results_by_idx[idx] = [
            (int(row['To_City']), city_name_map.get(int(row['To_City']), 'Unknown'), row['score'])
            for _, row in group.iterrows()
        ]

    # 对解析失败的 query 返回空列表
    final_results = []
    valid_ptr = 0
    for i in range(n_queries):
        if i in valid_indices:
            final_results.append(results_by_idx.get(valid_ptr, []))
            valid_ptr += 1
        else:
            final_results.append([])

    if verbose:
        print(f"Batch prediction completed! {len(valid_parsed)}/{n_queries} succeeded.")
    return final_results


# ═══════════════════════════════════════════════════════════════
# 示例入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 示例 1：单次推理（完整 TypeID）
    print("Example 1: Single Prediction (Complete TypeID)")
    result = predict(type_id="F_20_EduHi_Agri_IncH_Split_3506", year=2019, top_k=20, verbose=True)

    # 示例 2：单次推理（不完整 TypeID）
    print("\n\nExample 2: Single Prediction (Incomplete TypeID)")
    result = predict(type_id="M_30_EduMid_Mfg_IncM_Unit", year=2019, from_city="1100", top_k=10, verbose=True)

    # 示例 3：批量推理（TypeID 列表）
    print("\n\nExample 3: Batch Prediction (TypeID List)")
    batch_type_ids = [
        "F_20_EduHi_Agri_IncH_Split_3506",
        "M_30_EduMid_Mfg_IncM_Unit_1100",
        "F_40_EduHi_Service_IncH_Split_3100",
    ]
    batch_results = batch_predict(queries=batch_type_ids, year=2019, top_k=10, verbose=True)

    city_map = load_city_mapping()
    for i, (tid, res) in enumerate(zip(batch_type_ids, batch_results)):
        print(f"\nQuery {i+1}: {tid}")
        if res:
            print(f"  Top 5: {[(cname, f'{score:.4f}') for _, cname, score in res[:5]]}")

    # 示例 4：批量推理（字典列表）
    print("\n\nExample 4: Batch Prediction (Dict List)")
    batch_queries = [
        {"type_id": "F_20_EduHi_Agri_IncH_Split", "from_city": "3506"},
        {"type_id": "M_30_EduMid_Mfg_IncM_Unit", "from_city": "1100"},
    ]
    batch_results = batch_predict(queries=batch_queries, year=2019, top_k=10, verbose=True)

    for i, (q, res) in enumerate(zip(batch_queries, batch_results)):
        print(f"\nQuery {i+1}: {q['type_id']} (from {q['from_city']})")
        if res:
            print(f"  Top 5: {[(cname, f'{score:.4f}') for _, cname, score in res[:5]]}")
