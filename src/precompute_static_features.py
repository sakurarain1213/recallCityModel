"""
预计算全局城市对特征表 (空间换时间策略)

生成一个包含所有年份、所有城市对的静态特征矩阵,
避免在训练时重复计算距离、比值等特征。

输出: output/global_city_features.parquet
大小: ~238万行 (21年 × 337城市 × 336城市对 ≈ 238万,实际去重后更少
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from city_data import CityDataLoader
from feature_eng import add_cross_features, optimize_dtypes
from config import Config


def generate_global_city_features():
    """预计算所有年份的城市对特征"""
    print("[Start] 预计算全局城市对特征 (2000-2020)...")

    # 1. 加载城市数据
    loader = CityDataLoader(Config.DATA_DIR)
    loader.load_all()
    all_cities = loader.get_city_ids()
    print(f"加载了 {len(all_cities)} 个城市")

    # 2. 生成所有城市对 (排除 From == To)
    print("生成城市对组合...")
    city_pairs = []
    for f in all_cities:
        for t in all_cities:
            if f != t:  # 排除自身
                city_pairs.append({'From_City': f, 'To_City': t})

    base_df = pd.DataFrame(city_pairs)
    base_df['From_City'] = base_df['From_City'].astype('int16')
    base_df['To_City'] = base_df['To_City'].astype('int16')

    print(f"生成了 {len(base_df)} 个城市对")

    # 3. 逐年计算特征
    years = range(2000, 2021)
    all_years_data = []

    for year in tqdm(years, desc="计算各年份特征"):
        df_year = base_df.copy()
        df_year['Year'] = year

        # 临时列用于兼容 add_cross_features
        df_year['From_City_orig'] = df_year['From_City'].astype(str)

        # 获取该年的城市信息
        city_info = loader.get_city_info_for_year(year)

        # 调用特征工程函数 (复用逻辑)
        df_year = add_cross_features(
            df_year,
            city_info,
            loader.city_edges,
            verbose=False
        )

        # 清理: 只保留静态特征列
        static_cols = ['Year', 'From_City', 'To_City', 'geo_distance', 'dialect_distance']
        static_cols += [c for c in df_year.columns if c.endswith('_ratio')]

        df_year = df_year[static_cols]
        df_year = optimize_dtypes(df_year)

        all_years_data.append(df_year)

    # 4. 合并并保存
    print("\n合并所有年份数据...")
    full_matrix = pd.concat(all_years_data, axis=0, ignore_index=True)

    save_path = Path(Config.OUTPUT_DIR) / 'global_city_features.parquet'
    full_matrix.to_parquet(save_path, index=False)
    print(f"[OK] 全局特征表已保存: {save_path}")
    print(f"   行数: {len(full_matrix):,}")
    print(f"   大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return full_matrix


if __name__ == "__main__":
    generate_global_city_features()
