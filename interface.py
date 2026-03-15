"""
城市迁移推理接口 - 与训练/评估脚本完全一致

依赖: output/models/model_iter_3040.txt, data/city_nodes.jsonl,
      data/city_pair_cache/*.parquet, simple_train.py

用法:
  predict(type_id, year, from_city=None, top_k=20)
  batch_predict(queries, year, top_k=20)
"""
import json, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm
from simple_train import FEATS, CATS, CACHE_FEAT_COLS, CROSS_FEATS, load_cache_for_year, build_cross_features

warnings.filterwarnings('ignore')

MODEL_PATH = Path("output/models/model_iter_3040.txt")
CITY_NODES_PATH = Path("data/city_nodes.jsonl")

# 特征映射（与训练时一致）
AGE_MAPPING = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDUCATION_MAPPING = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
INDUSTRY_MAPPING = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INCOME_MAPPING = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAMILY_MAPPING = {'Split': 0, 'Unit': 1}
GENDER_MAPPING = {'M': 0, 'F': 1}

def load_city_mapping() -> Dict[int, str]:
    """加载城市 ID → Name 映射"""
    with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
        return {int(json.loads(line.strip())['city_id']): json.loads(line.strip())['name'] for line in f}

def get_all_city_ids() -> List[int]:
    """获取所有城市 ID 列表"""
    with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
        return [int(json.loads(line.strip())['city_id']) for line in f]

def parse_type_id(type_id: str, from_city: str = None) -> Dict:
    """解析 TypeID: F_20_EduHi_Agri_IncH_Split_3506 或 F_20_EduHi_Agri_IncH_Split"""
    parts = type_id.split('_')
    if len(parts) == 7:
        gender, age, edu, job, income, family, dialect = parts
        from_city_final = dialect
    elif len(parts) == 6:
        gender, age, edu, job, income, family = parts
        if from_city is None: raise ValueError(f"TypeID '{type_id}' 缺少 from_city")
        from_city_final = from_city
    else:
        raise ValueError(f"TypeID 格式错误: '{type_id}'")
    return {'gender': GENDER_MAPPING[gender], 'age_group': AGE_MAPPING[age], 'education': EDUCATION_MAPPING[edu],
            'industry': INDUSTRY_MAPPING[job], 'income': INCOME_MAPPING[income], 'family': FAMILY_MAPPING[family],
            'From_City': int(from_city_final)}

def build_candidate_df(parsed_queries: List[Dict], year: int, all_cities: List[int]) -> pd.DataFrame:
    """构建候选城市 DataFrame（与训练流程一致）"""
    rows = [{'query_idx': i, 'From_City': pq['From_City'], 'To_City': city, 'gender': pq['gender'],
             'age_group': pq['age_group'], 'education': pq['education'], 'industry': pq['industry'],
             'income': pq['income'], 'family': pq['family']}
            for i, pq in enumerate(parsed_queries) for city in all_cities]
    df = pd.DataFrame(rows)

    # 加载城市对特征并 merge（与训练一致）
    cache = load_cache_for_year(year)
    df = df.merge(cache, left_on=['From_City', 'To_City'], right_on=['from_city', 'to_city'], how='left')
    df.drop(columns=['from_city', 'to_city'], inplace=True)

    # 构建交叉特征（与训练一致）
    df = build_cross_features(df)

    # 填充缺失值（与训练一致）
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in df.columns: df[col] = df[col].fillna(0).astype(np.float32)

    return df

def predict(type_id: str, year: int, from_city: str = None, top_k: int = 20,
            model_path: str = str(MODEL_PATH), verbose: bool = True) -> List[Tuple[int, str, float]]:
    """单次推理"""
    if verbose: print(f"Loading model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    city_map = load_city_mapping()
    all_cities = get_all_city_ids()

    parsed_query = parse_type_id(type_id, from_city)
    if verbose:
        print(f"\nTypeID: {type_id}")
        print(f"From City: {parsed_query['From_City']} ({city_map.get(parsed_query['From_City'], 'Unknown')})")
        print(f"Building candidate features for {len(all_cities)} cities (year={year})...")

    df = build_candidate_df([parsed_query], year, all_cities)

    # 预测（与评估脚本一致：转为 category）
    X = df[FEATS].copy()
    for c in CATS:
        if c in X.columns: X[c] = X[c].astype('category')

    if verbose: print(f"Predicting...")
    scores = model.predict(X)
    df['score'] = scores
    df_sorted = df.sort_values('score', ascending=False).head(top_k)

    results = [(int(row['To_City']), city_map.get(int(row['To_City']), 'Unknown'), row['score'])
               for _, row in df_sorted.iterrows()]

    if verbose:
        print(f"\n{'='*70}\nTop {top_k} Predicted Cities:")
        print(f"{'Rank':<6} {'City ID':<10} {'City Name':<20} {'Score':<10}\n{'-'*70}")
        for rank, (cid, cname, score) in enumerate(results, 1):
            print(f"{rank:<6} {cid:<10} {cname:<20} {score:<10.6f}")
        print("="*70)

    return results

def batch_predict(queries: Union[List[str], List[Dict]], year: int, top_k: int = 20,
                  model_path: str = str(MODEL_PATH), verbose: bool = True) -> List[List[Tuple[int, str, float]]]:
    """批量推理（高效向量化）"""
    # 标准化输入
    if isinstance(queries[0], str):
        type_ids, from_cities = queries, [None] * len(queries)
    else:
        type_ids = [q['type_id'] for q in queries]
        from_cities = [q.get('from_city', None) for q in queries]

    n_queries = len(type_ids)
    if verbose: print(f"Batch predicting {n_queries} queries (year={year}, top_k={top_k})...")

    model = lgb.Booster(model_file=model_path)
    city_map = load_city_mapping()
    all_cities = get_all_city_ids()

    # 解析所有 TypeID
    if verbose: print(f"Parsing {n_queries} TypeIDs...")
    parsed_queries = []
    for tid, fc in zip(type_ids, from_cities):
        try:
            parsed_queries.append(parse_type_id(tid, fc))
        except Exception as e:
            if verbose: print(f"  [Error] Failed to parse '{tid}': {e}")
            parsed_queries.append(None)

    valid_indices = [i for i, pq in enumerate(parsed_queries) if pq is not None]
    valid_parsed_queries = [parsed_queries[i] for i in valid_indices]

    if len(valid_parsed_queries) == 0:
        print("  [Error] No valid queries!")
        return []

    # 批量构建特征
    if verbose: print(f"Building features for {len(valid_parsed_queries)} queries × {len(all_cities)} cities...")
    df = build_candidate_df(valid_parsed_queries, year, all_cities)

    # 批量预测（与评估脚本一致）
    if verbose: print(f"Predicting {len(df)} rows...")
    X = df[FEATS].copy()
    for c in CATS:
        if c in X.columns: X[c] = X[c].astype('category')
    scores = model.predict(X)
    df['score'] = scores

    # 按 query_idx 分组取 Top K
    if verbose: print(f"Extracting Top {top_k} for each query...")
    results = []
    for i in tqdm(range(len(valid_parsed_queries)), disable=not verbose, desc="  Processing"):
        query_df = df[df['query_idx'] == i].sort_values('score', ascending=False).head(top_k)
        query_results = [(int(row['To_City']), city_map.get(int(row['To_City']), 'Unknown'), row['score'])
                         for _, row in query_df.iterrows()]
        results.append(query_results)

    # 对解析失败的 query 返回空列表
    final_results = []
    valid_idx = 0
    for i in range(n_queries):
        if i in valid_indices:
            final_results.append(results[valid_idx])
            valid_idx += 1
        else:
            final_results.append([])

    if verbose: print(f"Batch prediction completed! {len(valid_parsed_queries)}/{n_queries} succeeded.")
    return final_results

if __name__ == '__main__':
    # 示例 1：单次推理（完整 TypeID）
    print("Example 1: Single Prediction (Complete TypeID)")
    result = predict(type_id="F_20_EduHi_Agri_IncH_Split_3506", year=2019, top_k=20, verbose=True)

    # 示例 2：单次推理（不完整 TypeID）
    print("\n\nExample 2: Single Prediction (Incomplete TypeID)")
    result = predict(type_id="M_30_EduMid_Mfg_IncM_Unit", year=2019, from_city="1100", top_k=10, verbose=True)

    # 示例 3：批量推理（TypeID 列表）
    print("\n\nExample 3: Batch Prediction (TypeID List)")
    batch_type_ids = ["F_20_EduHi_Agri_IncH_Split_3506", "M_30_EduMid_Mfg_IncM_Unit_1100",
                      "F_40_EduHi_Service_IncH_Split_3100"]
    batch_results = batch_predict(queries=batch_type_ids, year=2019, top_k=10, verbose=True)

    city_map = load_city_mapping()
    for i, (tid, result) in enumerate(zip(batch_type_ids, batch_results)):
        print(f"\nQuery {i+1}: {tid}")
        if result: print(f"  Top 5: {[(cname, f'{score:.4f}') for _, cname, score in result[:5]]}")

    # 示例 4：批量推理（字典列表）
    print("\n\nExample 4: Batch Prediction (Dict List)")
    batch_queries = [{"type_id": "F_20_EduHi_Agri_IncH_Split", "from_city": "3506"},
                     {"type_id": "M_30_EduMid_Mfg_IncM_Unit", "from_city": "1100"}]
    batch_results = batch_predict(queries=batch_queries, year=2019, top_k=10, verbose=True)

    for i, (q, result) in enumerate(zip(batch_queries, batch_results)):
        print(f"\nQuery {i+1}: {q['type_id']} (from {q['from_city']})")
        if result: print(f"  Top 5: {[(cname, f'{score:.4f}') for _, cname, score in result[:5]]}")
