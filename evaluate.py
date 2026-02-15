"""
极速评估脚本 (宽表适配修复版)
功能: 加载模型 -> 从宽表 DB 读取并展开 GT -> 构造全量候选集 -> 特征工程 -> 预测 -> 计算 Recall
修复: 解决了 DuckDB 表结构不匹配的问题 (Rank 列不存在)
修复: 解决了 ID 类型不匹配导致的 0% Recall 问题
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
from src.feature_eng import extract_city_id  # 复用同一个提取函数

# ------------------------------------------------------------------------------
# 1. 核心辅助函数：读取宽表并转为长表 GT
# ------------------------------------------------------------------------------
def load_ground_truth(db_path, year):
    """
    修复版：严格提取 Int 类型的 ID  尤其注意ground truth是 top10 而不是20个都加载进来。
    """
    print(f"📥 Querying DuckDB for year {year}...")
    con = duckdb.connect(str(db_path), read_only=True)

    # 查询
    # 修改后: 只加载 Top 10 作为 Ground Truth
    # 这样分母(GT总数)就变成了 10，Recall@10 的理论上限就是 100% 了
    top_cols = [f"To_Top{i}" for i in range(1, 11)] 
    cols_str = ", ".join(top_cols)

    query = f"SELECT Year, Type_ID, From_City, {cols_str} FROM migration_data WHERE Year = {year}"

    try:
        df_wide = con.execute(query).df()
        if df_wide.empty: return pd.DataFrame()

        # 1. 清洗 From_City (转 Int)
        df_wide['From_City'] = df_wide['From_City'].apply(extract_city_id).astype('int16')

        # 2. Melt
        df_long = pd.melt(df_wide, id_vars=['Year', 'Type_ID', 'From_City'],
                          value_vars=top_cols, value_name='To_City_Raw')

        # 3. 清洗 To_City (转 Int)
        # 这一步是之前的痛点：To_Top1 可能是 "成都(5101)"
        df_long = df_long.dropna(subset=['To_City_Raw'])
        df_long['To_City'] = df_long['To_City_Raw'].apply(extract_city_id)

        # 过滤无效ID
        df_long = df_long[df_long['To_City'] > 0].copy()
        df_long['To_City'] = df_long['To_City'].astype('int16')

        # 去重 (同一Query可能多个Rank指向同一城市? 一般不会，但防万一)
        final_df = df_long[['Year', 'Type_ID', 'From_City', 'To_City']].drop_duplicates()

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
    # 1. 加载资源 (CityData 会强制 ID 为 Int)
    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Config.PROCESSED_DIR)

    # 【关键修改】 路径检查
    if not model_path:
        print("❌ Error: No model path provided.")
        return

    path_obj = Path(model_path)
    if not path_obj.exists():
        print(f"❌ Error: Model file does not exist at: {model_path}")
        return
        
    print(f"📂 Loading Model from: {path_obj.absolute()}")
    model = lgb.Booster(model_file=str(path_obj)) # 确保转为 str
    model_feats = model.feature_name()
    print(f"✅ Model loaded successfully ({len(model_feats)} feats)")

    # 2. 获取 GT (现在全是 Int)
    df_true = load_ground_truth(Config.DB_PATH, year)
    if df_true.empty:
        print("❌ No ground truth data found.")
        return

    print(f"   ✓ Extracted {len(df_true):,} valid ground truth pairs")

    # 采样
    queries = df_true[['Year', 'Type_ID', 'From_City']].drop_duplicates()
    if sample_size and len(queries) > sample_size:
        print(f"⚡ Sampling {sample_size} queries from {len(queries)}...")
        queries = queries.sample(n=sample_size, random_state=42)
    else:
        print(f"📊 Evaluating {len(queries)} queries...")

    # 3. 构造候选集 (候选 ID 必须是 Int)
    print("🔨 Generating Candidates...")
    all_cities = loader.get_city_ids()  # 这是一个 Int List
    
    # 笛卡尔积
    queries = queries.copy()
    queries['key'] = 1
    targets = pd.DataFrame({'To_City': all_cities, 'key': 1})  # To_City 是 Int
    candidates = pd.merge(queries, targets, on='key').drop('key', axis=1)
    
    # 排除 From == To
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    
    # 4. 特征工程
    print("✨ Feature Engineering...")
    candidates['Flow_Count'] = 0
    df_feats = pipeline.transform(candidates.copy(), year, mode='predict', verbose=False)

    # 准备 X
    X = pd.DataFrame(index=df_feats.index)
    for f in model_feats:
        X[f] = df_feats[f] if f in df_feats.columns else 0.0

    # 转 float32
    for c in X.columns:
        if X[c].dtype == 'float64': X[c] = X[c].astype('float32')

    # 5. 预测
    print("🔮 Predicting...")
    scores = model.predict(X)
    candidates['score'] = scores

    # 6. 计算指标
    print("📉 Calculating Metrics...")
    gt_set = set(zip(df_true['Type_ID'], df_true['From_City'], df_true['To_City']))
    
    candidates['rank'] = candidates.groupby(['Type_ID', 'From_City'])['score'].rank(method='first', ascending=False)
    top_preds = candidates[candidates['rank'] <= 20].copy()
    
    top_preds['is_hit'] = top_preds.apply(lambda x: (x['Type_ID'], x['From_City'], x['To_City']) in gt_set, axis=1)
    
    hits = top_preds.groupby(['Type_ID', 'From_City']).apply(
        lambda x: pd.Series({
            'hit_1': x[x['rank'] <= 1]['is_hit'].sum(),
            'hit_5': x[x['rank'] <= 5]['is_hit'].sum(),
            'hit_10': x[x['rank'] <= 10]['is_hit'].sum(),
            'hit_20': x[x['rank'] <= 20]['is_hit'].sum()
        })
    ).reset_index()
    
    gt_counts = df_true.groupby(['Type_ID', 'From_City']).size().reset_index(name='total_true')
    res = pd.merge(hits, gt_counts, on=['Type_ID', 'From_City'], how='left').fillna(0)
    res['total_true'] = res['total_true'].replace(0, 1)
    
    r1 = (res['hit_1'] / res['total_true']).mean()
    r5 = (res['hit_5'] / res['total_true']).mean()
    r10 = (res['hit_10'] / res['total_true']).mean()
    r20 = (res['hit_20'] / res['total_true']).mean()
    
    print("\n" + "="*40)
    print(f"📊 Evaluation Results for Year {year}")
    print(f"🤖 Model: {Path(model_path).name}")
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
    parser.add_argument('--model', type=str, default=None, help="Specific path to model checkpoint")
    parser.add_argument('--sample', type=int, default=1000, help="Number of queries to sample")
    args = parser.parse_args()
    
    # 自动查找模型 (仅当未指定时)
    if args.model is None:
        print("⚠️ No model path provided, trying to auto-find latest model...")
        p = Path(Config.OUTPUT_DIR) / f"lgb_batch_end_{args.year}.txt"
        if not p.exists():
             models = list(Path(Config.OUTPUT_DIR).glob("lgb_batch_end_*.txt"))
             if models:
                 p = max(models, key=lambda f: f.stat().st_mtime)
        args.model = str(p)

    run_main(args.year, args.model, args.sample)