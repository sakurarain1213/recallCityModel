"""
极速评估脚本 (宽表适配修复版)
功能: 加载模型 -> 从宽表 DB 读取并展开 GT -> 构造全量候选集 -> 特征工程 -> 预测 -> 计算 Recall
修复: 解决了 DuckDB 表结构不匹配的问题 (Rank 列不存在)
"""
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import duckdb
import argparse
import re
from pathlib import Path
from src.config import Config
from src.city_data import CityDataLoader
from src.feature_pipeline import FeaturePipeline

# ------------------------------------------------------------------------------
# 1. 核心辅助函数：读取宽表并转为长表 GT
# ------------------------------------------------------------------------------
def load_ground_truth(db_path, year):
    """
    从 DuckDB 读取宽表数据，并展开为长表格式 (Year, Type, From, To, Rank)
    """
    print(f"📥 Querying DuckDB for year {year} (Wide Table)...")
    
    con = duckdb.connect(str(db_path), read_only=True)
    
    # 1. 构造 SQL 查询 Top 20 的列
    # 我们需要 Year, Type_ID, From_City 以及所有的 To_TopX 和 To_TopX_Count
    top_cols = []
    for i in range(1, 21):
        top_cols.append(f"To_Top{i}")
        # top_cols.append(f"To_Top{i}_Count") # 其实评估只需要知道去哪了，Count 可选
    
    cols_str = ", ".join(top_cols)
    
    query = f"""
    SELECT 
        Year, 
        Type_ID, 
        From_City, 
        {cols_str}
    FROM migration_data
    WHERE Year = {year}
    """
    
    try:
        df_wide = con.execute(query).df()
        if df_wide.empty:
            return pd.DataFrame()
            
        print(f"   ✓ Loaded {len(df_wide):,} wide rows. Unpivoting to long format...")
        
        # 2. 清洗 From_City (去除中文，只留 ID)
        # 假设 From_City 可能是 "深圳(4403)" 这种格式
        if df_wide['From_City'].dtype == 'object':
             df_wide['From_City'] = df_wide['From_City'].astype(str).str.extract(r'(\d+)', expand=False)
        df_wide['From_City'] = pd.to_numeric(df_wide['From_City'], errors='coerce').fillna(0).astype('int16')

        # 3. 宽表转长表 (Melt)
        # id_vars = [Year, Type_ID, From_City]
        # value_vars = [To_Top1, ..., To_Top20]
        df_long = pd.melt(
            df_wide, 
            id_vars=['Year', 'Type_ID', 'From_City'], 
            value_vars=[f"To_Top{i}" for i in range(1, 21)],
            var_name='Rank_Str', 
            value_name='To_City_Raw'
        )
        
        # 4. 解析 Rank 和 To_City
        # Rank_Str 是 "To_Top1", "To_Top2"... -> 提取数字作为 Rank
        df_long['Rank'] = df_long['Rank_Str'].str.extract(r'(\d+)').astype(int).astype('int16')
        
        # To_City_Raw 可能是 "上海(3100)" 或 "0" 或 None
        # 我们需要提取其中的数字 ID
        df_long = df_long.dropna(subset=['To_City_Raw'])
        # 转换为字符串处理
        df_long['To_City_Raw'] = df_long['To_City_Raw'].astype(str)
        # 提取数字 (如果本来就是数字字符串也能提取)
        df_long['To_City'] = df_long['To_City_Raw'].str.extract(r'(\d+)', expand=False)
        # 转为数字，非数字变为 NaN
        df_long['To_City'] = pd.to_numeric(df_long['To_City'], errors='coerce')
        
        # 5. 过滤有效数据
        # 去除 To_City 为 0 或 NaN 的行 (表示没有 TopX 数据)
        # 也要去除 To_City == From_City 的行 (虽然理论上 Top 不应该包含自己)
        df_valid = df_long[
            (df_long['To_City'].notna()) & 
            (df_long['To_City'] > 0)
        ].copy()
        
        df_valid['To_City'] = df_valid['To_City'].astype('int16')
        
        # 只保留需要的列
        final_df = df_valid[['Year', 'Type_ID', 'From_City', 'To_City', 'Rank']].reset_index(drop=True)
        
        return final_df

    except Exception as e:
        print(f"❌ DB Error: {e}")
        return pd.DataFrame()
    finally:
        con.close()

# ------------------------------------------------------------------------------
# 2. 核心评估逻辑
# ------------------------------------------------------------------------------
def run_main(year, model_path, sample_size):
    # 1. 加载资源
    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Config.PROCESSED_DIR)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    model = lgb.Booster(model_file=model_path)
    model_feats = model.feature_name()
    print(f"✅ Model loaded: {model_path} ({len(model_feats)} feats)")

    # 2. 获取 GT (使用新函数)
    df_true = load_ground_truth(Config.DB_PATH, year)
    
    if df_true.empty:
        print("❌ No ground truth data found.")
        return
        
    print(f"   ✓ Extracted {len(df_true):,} valid ground truth pairs (Rank <= 20)")

    # 提取 Queries (Year, Type_ID, From_City)
    queries = df_true[['Year', 'Type_ID', 'From_City']].drop_duplicates()
    
    if sample_size and len(queries) > sample_size:
        print(f"⚡ Sampling {sample_size} queries from {len(queries)}...")
        queries = queries.sample(n=sample_size, random_state=42)
    else:
        print(f"📊 Evaluating {len(queries)} queries...")
    
    # 3. 构造候选集 (Query * All_Cities)
    print("🔨 Generating Candidates...")
    all_cities = loader.get_city_ids()
    
    # 笛卡尔积
    queries = queries.copy()
    queries['key'] = 1
    targets = pd.DataFrame({'To_City': all_cities, 'key': 1})
    candidates = pd.merge(queries, targets, on='key').drop('key', axis=1)
    
    # 排除 From == To
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    
    # 4. 特征工程
    print("✨ Feature Engineering...")
    # 为了复用 pipeline，需要 Flow_Count 占位
    candidates['Flow_Count'] = 0 
    
    # Pipeline 变换 (生成特征)
    df_feats = pipeline.transform(candidates.copy(), year, mode='eval', verbose=False)
    
    # 类型处理 (与训练一致)
    if 'Type_ID' in df_feats.columns and df_feats['Type_ID'].dtype == 'object':
        df_feats['Type_Hash'] = pd.util.hash_pandas_object(df_feats['Type_ID'], index=False).astype('int64')
        df_feats.drop(columns=['Type_ID'], inplace=True)
    
    # 准备 X (特征矩阵)
    X = pd.DataFrame(index=df_feats.index)
    for f in model_feats:
        if f in df_feats.columns:
            X[f] = df_feats[f]
        else:
            X[f] = 0
            
    # 转 float32
    for c in X.columns:
        if X[c].dtype == 'float64': X[c] = X[c].astype('float32')

    # 5. 预测
    print("🔮 Predicting...")
    # 将预测分数赋值回 candidates (用于后续排序)
    candidates['score'] = model.predict(X)
    
    # 6. 计算指标 (Recall@K)
    print("📉 Calculating Metrics...")
    
    # 6.1 构造快速查找的 GT 集合
    # 格式: (Type_ID, From_City, To_City) -> True
    gt_set = set(zip(df_true['Type_ID'], df_true['From_City'], df_true['To_City']))
    
    # 6.2 排序: 对每个 (Type_ID, From_City) 分组，按分数降序排列
    candidates['rank'] = candidates.groupby(['Type_ID', 'From_City'])['score'].rank(method='first', ascending=False)
    
    # 6.3 只保留 Top 20 预测结果进行统计
    top_preds = candidates[candidates['rank'] <= 20].copy()
    
    # 6.4 判断是否命中 GT
    top_preds['is_hit'] = top_preds.apply(lambda x: (x['Type_ID'], x['From_City'], x['To_City']) in gt_set, axis=1)
    
    # 6.5 聚合统计
    hits = top_preds.groupby(['Type_ID', 'From_City']).apply(
        lambda x: pd.Series({
            'hit_1': x[x['rank'] <= 1]['is_hit'].sum(),
            'hit_5': x[x['rank'] <= 5]['is_hit'].sum(),
            'hit_10': x[x['rank'] <= 10]['is_hit'].sum(),
            'hit_20': x[x['rank'] <= 20]['is_hit'].sum()
        })
    ).reset_index()
    
    # 获取每个 Query 对应的真实流向总数 (分母)
    gt_counts = df_true.groupby(['Type_ID', 'From_City']).size().reset_index(name='total_true')
    
    # 合并
    res = pd.merge(hits, gt_counts, on=['Type_ID', 'From_City'], how='left').fillna(0)
    
    # 计算平均 Recall
    res['total_true'] = res['total_true'].replace(0, 1)
    
    r1 = (res['hit_1'] / res['total_true']).mean()
    r5 = (res['hit_5'] / res['total_true']).mean()
    r10 = (res['hit_10'] / res['total_true']).mean()
    r20 = (res['hit_20'] / res['total_true']).mean()
    
    print("\n" + "="*40)
    print(f"📊 Evaluation Results for Year {year}")
    print("="*40)
    print(f"Queries Evaluated : {len(res)}")
    print(f"Avg GT per Query: {res['total_true'].mean():.2f}")
    print("-" * 30)
    print(f"Recall@1  : {r1:.2%}")
    print(f"Recall@5  : {r5:.2%}")
    print(f"Recall@10 : {r10:.2%}")
    print(f"Recall@20 : {r20:.2%}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2018, help="Year to evaluate")
    parser.add_argument('--model', type=str, default=None, help="Path to model file")
    parser.add_argument('--sample', type=int, default=1000, help="Number of queries to sample (speed up)")
    args = parser.parse_args()
    
    # 自动查找模型
    if args.model is None:
        p = Path(Config.OUTPUT_DIR) / f"lgb_batch_end_{args.year}.txt"
        if not p.exists():
             models = list(Path(Config.OUTPUT_DIR).glob("lgb_batch_end_*.txt"))
             if models:
                 p = max(models, key=lambda f: f.stat().st_mtime)
        args.model = str(p)

    run_main(args.year, args.model, args.sample)