"""
极速数据合成脚本：纯向量化 + 全量候选集 (336城)
特征: 6维Type_ID + 9维Ratio + 2维Edge + 1维省份
"""
import json
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 你原有的 Config 和解析逻辑
from src.config import Config
from src.feature_eng import DIMENSIONS

RATIO_COLS = [
    'gdp_per_capita_ratio', 'unemployment_rate_ratio',
    'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio',
    'medical_score_ratio', 'education_score_ratio',
    'transport_convenience_ratio', 'population_total_ratio',
]

def load_edges_fast():
    """快速加载地理和方言距离"""
    edges_path = Path(Config.DATA_DIR) / 'city_edges.jsonl'
    data = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append([int(d['source_id']), int(d['target_id']), float(d['w_geo']), float(d['w_dialect'])])
    df = pd.DataFrame(data, columns=['From_City', 'To_City', 'geo_distance', 'dialect_distance'])
    df['From_City'] = df['From_City'].astype('int32')
    df['To_City'] = df['To_City'].astype('int32')
    return df

def load_ratios_fast(year: int):
    """从 ratio_cache 加载 9 维比率"""
    cache_path = Path(Config.RATIO_CACHE_DIR) / f'city_ratios_{year}.jsonl'
    
    # 增加容错：如果某一年的 ratio_cache 不存在，直接报错提示，防止程序崩溃在后续逻辑
    if not cache_path.exists():
        raise FileNotFoundError(f"找不到 {year} 年的 ratio_cache: {cache_path}")
        
    rows = []
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            f_city = int(record['from_city'])
            for t_city_str, ratios in record['to_cities'].items():
                t_city = int(t_city_str)
                row = [f_city, t_city] + [ratios.get(c, 0.0) for c in RATIO_COLS]
                rows.append(row)
    
    df = pd.DataFrame(rows, columns=['From_City', 'To_City'] + RATIO_COLS)
    df['From_City'] = df['From_City'].astype('int32')
    df['To_City'] = df['To_City'].astype('int32')
    for col in RATIO_COLS:
        df[col] = df[col].astype('float32')
    return df

def parse_unique_types(type_series: pd.Series):
    """提速核心：只解析 unique 的 Type_ID"""
    df_types = pd.DataFrame({'Type_ID': type_series.unique()})
    parts = df_types['Type_ID'].str.split('_', expand=True)
    df_types['gender'] = parts[0].map(DIMENSIONS['gender']).fillna(0).astype('int8')
    df_types['age_group'] = parts[1].map(DIMENSIONS['age_group']).fillna(0).astype('int8')
    df_types['education'] = parts[2].map(DIMENSIONS['education']).fillna(0).astype('int8')
    df_types['industry'] = parts[3].map(DIMENSIONS['industry']).fillna(0).astype('int8')
    df_types['income'] = parts[4].map(DIMENSIONS['income']).fillna(0).astype('int8')
    df_types['family'] = parts[5].map(DIMENSIONS['family']).fillna(0).astype('int8')
    return df_types

def generate_year_data(year: int):
    print(f"========== Processing Year {year} ==========")
    
    # 1. 从 DB 加载当年的 Query 和 GT
    db_path = Config.DB_PATH
    with duckdb.connect(db_path, read_only=True) as conn:
        cols = ['Year', 'Type_ID', 'From_City']
        for r in range(1, 21):
            cols.extend([f'To_Top{r}', f'To_Top{r}_Count'])
        query = f"SELECT {', '.join(cols)} FROM migration_data WHERE Year = {year}"
        df_raw = conn.execute(query).df()
    
    if df_raw.empty:
        print(f"No data for {year}")
        return
        
    df_raw['From_City'] = pd.to_numeric(df_raw['From_City'].astype(str).str.extract(r'(\d+)', expand=False)).astype('int32')
    
    # 构建 GT 宽表转长表 (这就是生成 Label 的地方！)
    id_vars = ['Year', 'Type_ID', 'From_City']
    city_cols = [f'To_Top{r}' for r in range(1, 21)]
    gt_df = df_raw[id_vars + city_cols].melt(id_vars=id_vars, value_vars=city_cols, var_name='rank_key', value_name='To_City')
    gt_df['Rank'] = gt_df['rank_key'].str.extract(r'(\d+)').astype('int16')
    gt_df['To_City'] = pd.to_numeric(gt_df['To_City'].astype(str).str.extract(r'(\d+)', expand=False)).fillna(-1).astype('int32')
    gt_df = gt_df[gt_df['To_City'] > 0].copy()
    
    # 正样本 Label 设为 1.0
    gt_df['Label'] = 1.0  
    gt_df = gt_df[['Type_ID', 'From_City', 'To_City', 'Rank', 'Label']]

    # 2. 构建 Query 表
    queries = df_raw[['Type_ID', 'From_City']].drop_duplicates().reset_index(drop=True)
    queries['qid'] = np.arange(len(queries), dtype=np.int32)
    
    # 解析 Type_ID
    type_features = parse_unique_types(queries['Type_ID'])
    queries = queries.merge(type_features, on='Type_ID', how='left')
    
    # 3. 笛卡尔积扩展全量候选集
    print(f"Expanding {len(queries)} queries to full candidates...")
    
    # 修复了这里的 GBK 编码报错问题！！！加了 encoding='utf-8'
    all_cities = np.array([json.loads(line)['city_id'] for line in open(Config.CITY_NODES_PATH, 'r', encoding='utf-8')])
    all_cities = pd.to_numeric(pd.Series(all_cities).astype(str).str.extract(r'(\d+)', expand=False)).astype('int32').values

    # 使用 pd.MultiIndex.from_product 极速构建交叉表
    idx = pd.MultiIndex.from_product([queries['qid'].values, all_cities], names=['qid', 'To_City'])
    df_base = pd.DataFrame(index=idx).reset_index()
    
    # 拼回 Query 属性
    df_base = df_base.merge(queries, on='qid', how='left')
    
    # 剔除 From_City == To_City
    df_base = df_base[df_base['From_City'] != df_base['To_City']].copy()
    
    # 4. Merge 特征与标签
    print("Merging Edges, Ratios and Ground Truth...")
    df_edges = load_edges_fast()
    df_ratios = load_ratios_fast(year)
    
    df_final = df_base.merge(df_edges, on=['From_City', 'To_City'], how='left')
    df_final = df_final.merge(df_ratios, on=['From_City', 'To_City'], how='left')
    
    # 把 GT 合并进来！没命中 GT 的也就是没去过的城市，Label 就会变成 NaN
    df_final = df_final.merge(gt_df, on=['Type_ID', 'From_City', 'To_City'], how='left')
    
    # 填充缺失值：负样本的 Rank 设为 999，Label 设为 0.0 (非常关键)
    df_final['Rank'] = df_final['Rank'].fillna(999).astype('int16')
    df_final['Label'] = df_final['Label'].fillna(0.0).astype('float32')
    
    df_final['geo_distance'] = df_final['geo_distance'].fillna(-1.0).astype('float32')
    df_final['dialect_distance'] = df_final['dialect_distance'].fillna(-1.0).astype('float32')
    df_final['is_same_province'] = (df_final['From_City'] // 100 == df_final['To_City'] // 100).astype('int8')
    
    # 增加 Year 列
    df_final['Year'] = np.int16(year)
    
    # 保存结果
    out_file = Path(Config.PROCESSED_DIR) / f'processed_{year}.feather'
    df_final.reset_index(drop=True).to_feather(out_file)
    print(f"Done! Saved shape: {df_final.shape} to {out_file}")

if __name__ == '__main__':
    # 修复了年份问题：修改为从 2000 开始，左闭右开，涵盖 2000 到 2020 ！！！
    for y in range(2000, 2021):
        generate_year_data(y)