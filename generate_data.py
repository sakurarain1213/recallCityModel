"""
离线数据生成脚本 (ETL一体化版)
功能: 读取原始数据 -> 特征工程 -> 深度清洗/压缩 -> 保存 Parquet
优势: 直接生成定型的 float32/int16 数据，无需二次处理
"""
import pandas as pd
import numpy as np
import gc
from pathlib import Path
from src.config import Config
from src.city_data import CityDataLoader
from src.data_loader import load_raw_data_fast
from src.feature_pipeline import FeaturePipeline

# 引入 fix_data 中的核心清洗逻辑
def optimize_dataframe(df, verbose=True):
    """深度优化 DataFrame 内存和类型"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # 1. 删除纯 Object 类型的无用列 (保留必要的 ID 用于后续历史特征匹配，如果需要的话)
    # 注意：Type_ID_orig 和 From_City_orig 在生成历史特征时可能需要，
    # 但如果当前是从 pipeline 出来的最终结果，通常可以只保留数值 ID。
    # 这里我们保留 _orig 后缀的列以防万一，但 Hash 化 Type_ID
    
    # 1.1 Type_ID 字符串转 Hash 数值
    if 'Type_ID' in df.columns and df['Type_ID'].dtype == 'object':
        df['Type_Hash'] = pd.util.hash_pandas_object(df['Type_ID'], index=False).astype('int64')
        # 如果有 Type_ID_orig 则不需要 Type_ID 了
        if 'Type_ID_orig' not in df.columns:
            df['Type_ID_orig'] = df['Type_ID'] # 备份用于跨年匹配
        df = df.drop(columns=['Type_ID'])

    # 1.2 城市 ID 标准化 (提取数字并转 int16)
    for col in ['From_City', 'To_City']:
        if col in df.columns:
            # 如果是 "北京(1100)" 格式，提取数字
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.extract(r'(\d+)', expand=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).astype('int16')

    # 2. 数值类型降级 (核心省内存步骤)
    for col in df.columns:
        # 跳过字符串备份列
        if df[col].dtype == 'object': 
            continue
            
        # Float64 -> Float32
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        
        # Int64 -> Int32/Int16
        elif df[col].dtype == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')

    # 3. 删除完全重复的列 (解决 FeaturePipeline 可能产生的冗余)
    # 这是一个 O(N^2) 操作，但列数不多(几十列)时很快
    duplicate_cols = []
    cols = df.columns.tolist()
    # 只检查数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for i in range(len(numeric_cols)):
        col1 = numeric_cols[i]
        for j in range(i + 1, len(numeric_cols)):
            col2 = numeric_cols[j]
            # 快速检查：如果均值不同肯定不同
            if df[col1].mean() != df[col2].mean():
                continue
            # 详细检查
            if df[col1].equals(df[col2]):
                duplicate_cols.append(col2)
    
    if duplicate_cols:
        df = df.drop(columns=list(set(duplicate_cols)))
        if verbose:
            print(f"  ✂️  Removed duplicate cols: {list(set(duplicate_cols))}")

    # 4. 缺失值填充
    # 数值填 0, 字符串填 MISSING
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_number(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("MISSING")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  📉 Memory optimized: {start_mem:.1f}MB -> {end_mem:.1f}MB")
    
    return df

def generate_all_data(start_year, end_year):
    print(f"🚀 开始生成训练数据 (ETL优化版) | {start_year}-{end_year}")
    print(f"📂 目标目录: {Config.PROCESSED_DIR}")

    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Config.PROCESSED_DIR)
    hard_candidates = loader.get_city_ids()

    for year in range(start_year, end_year + 1):
        out_file = Path(Config.PROCESSED_DIR) / f"train_{year}.parquet"
        if out_file.exists():
            print(f"✅ Year {year} 已存在，跳过。")
            continue

        print(f"\n📅 Processing Year {year}...")

        # 1. 加载并采样
        df = load_raw_data_fast(
            Config.DB_PATH, year, hard_candidates, Config.TOTAL_SAMPLES_PER_QUERY
        )
        if df.empty:
            print(f"⚠️  Year {year} 数据为空，跳过。")
            continue
        print(f"  ✅ Raw loaded: {len(df):,} rows")

        # 2. 特征工程
        df = pipeline.transform(df, year, mode='eval', verbose=False)
        
        # 3. 深度清洗与压缩 (新增步骤，替代 fix_data.py)
        print("  🔄 Optimizing data structure...")
        df = optimize_dataframe(df)

        # 4. 保存
        df.to_parquet(out_file, index=False, compression='zstd')
        print(f"💾 Saved {out_file.name}")

        del df
        gc.collect()

    print(f"\n🎉 全部完成!")

if __name__ == "__main__":
    # 生成 2000-2020 的数据
    generate_all_data(2000, 2020)