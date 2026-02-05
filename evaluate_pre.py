"""
evaluate_pre.py
预计算所有城市对的静态特征（距离、经济属性比值等）
只需运行一次，生成的特征表可供推理时直接 Merge，速度提升 100倍+
"""
import sys
import os
sys.path.append(os.getcwd()) # 确保能导入 src

import pandas as pd
import numpy as np
from pathlib import Path
from src.city_data import CityDataLoader
from src.feature_eng import add_cross_features, optimize_dtypes
from src.config import Config

def generate_static_matrix():
    print("="*60)
    print("正在生成全量城市对静态特征矩阵 (337 x 337)...")
    print("="*60)

    # 1. 加载基础数据
    city_loader = CityDataLoader(data_dir=Config.DATA_DIR)
    city_loader.load_all()
    all_cities = city_loader.get_city_ids()
    print(f"城市数量: {len(all_cities)}")

    # 2. 构建笛卡尔积 (From_City x To_City)
    # 只要 From != To
    print("构建城市对索引...")
    from_ids = []
    to_ids = []
    
    for f_id in all_cities:
        for t_id in all_cities:
            if f_id != t_id:
                from_ids.append(f_id)
                to_ids.append(t_id)
    
    df_static = pd.DataFrame({
        'From_City': from_ids,
        'To_City': to_ids
    })
    
    # 确保 ID 类型一致 (int16)
    df_static['From_City'] = df_static['From_City'].astype('int16')
    df_static['To_City'] = df_static['To_City'].astype('int16')

    print(f"生成了 {len(df_static):,} 个城市对")

    # 3. 计算交叉特征 (最耗时的部分)
    # 为了复用 add_cross_features，我们需要构造临时的字符串列
    df_static['From_City_orig'] = df_static['From_City'].astype(str)
    # To_City 已经是 int16，但在 add_cross_features 内部可能会被处理
    
    print("计算静态交叉特征 (距离, 经济比值等)...")
    # 调用现有的特征工程函数
    df_static = add_cross_features(
        df_static, 
        city_loader.city_info, # city_nodes
        city_loader.city_edges, 
        verbose=True
    )

    # 4. 清理列
    # 删除临时列和不需要的列
    cols_to_drop = ['From_City_orig', 'geo_distance', 'dialect_distance'] 
    # 注意：保留 geo_distance 和 dialect_distance 的数值列，但在 add_cross_features 里它们可能叫别的
    # 查看 add_cross_features 的实现，它添加了 'geo_distance', 'dialect_distance' 以及各种 '_ratio'
    
    # 我们只需要特征列和索引列
    keep_cols = ['From_City', 'To_City'] + \
                [c for c in df_static.columns if c.endswith('_ratio')] + \
                ['geo_distance', 'dialect_distance']
    
    df_final = df_static[keep_cols].copy()
    
    # 5. 优化类型并保存
    df_final = optimize_dtypes(df_final)
    
    output_path = Path(Config.OUTPUT_DIR) / 'static_city_pairs.parquet'
    df_final.to_parquet(output_path, index=False)
    
    print(f"\n✅ 成功保存静态特征矩阵: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"包含列: {list(df_final.columns)}")

if __name__ == "__main__":
    generate_static_matrix()