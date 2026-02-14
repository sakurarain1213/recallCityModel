"""
离线数据生成脚本 - 解耦数据处理与模型训练
运行一次，生成所有年份的 Parquet 文件
"""
import pandas as pd
import gc
from pathlib import Path
from src.config import Config
from src.city_data import CityDataLoader
from src.data_loader import load_raw_data_fast
from src.feature_pipeline import FeaturePipeline


def generate_all_data(start_year, end_year):
    """
    生成所有年份的训练数据并保存为 Parquet

    Args:
        start_year: 起始年份
        end_year: 结束年份
    """
    print(f"🚀 开始离线生成训练数据 ({start_year}-{end_year})...")
    print(f"📂 目标目录: {Config.PROCESSED_DIR}")

    # 初始化
    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Config.PROCESSED_DIR)
    hard_candidates = loader.get_city_ids()

    for year in range(start_year, end_year + 1):
        out_file = Path(Config.PROCESSED_DIR) / f"train_{year}.parquet"
        if out_file.exists():
            print(f"✅ Year {year} 已存在，跳过。")
            continue

        print(f"\n📅 Processing Year {year}...")

        # 1. 加载并采样 (1:4 比例)
        df = load_raw_data_fast(
            Config.DB_PATH,
            year,
            hard_candidates,
            Config.TOTAL_SAMPLES_PER_QUERY
        )
        if df.empty:
            print(f"⚠️  Year {year} 数据为空，跳过。")
            continue

        print(f"  ✅ 采样完成: {len(df):,} rows")

        # 2. 特征工程 (Pipeline 会自动计算同省、距离、历史特征)
        # 使用 mode='eval' 确保生成全量特征 (不Dropout)，保持确定性
        df = pipeline.transform(df, year, mode='eval', verbose=False)
        print(f"  ✅ 特征工程完成: {len(df.columns)} cols")

        # 3. 瘦身 (删除字符串列，只留数值)
        cols_to_drop = ['Type_ID', 'Type_ID_orig', 'From_City_orig']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        # 4. 强制类型转换 (Float64 -> Float32)
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')

        # 5. 保存
        df.to_parquet(out_file, index=False, compression='zstd')
        print(f"💾 Saved {len(df):,} rows x {len(df.columns)} cols to {out_file.name}")

        # 内存清理
        del df
        gc.collect()

    print(f"\n🎉 全部完成! 数据保存在: {Config.PROCESSED_DIR}")


if __name__ == "__main__":
    # 根据你的需求，生成到 2018 (因为 2020 是最后的数据)
    generate_all_data(2001, 2018)
