"""
统一特征工程 Pipeline

确保训练、评估、预测三个阶段使用完全相同的特征工程流程，
避免向量不一致导致的模型性能下降。

核心原则：
1. 所有阶段必须调用同一个 transform() 方法
2. 特征工程顺序固定：parse_type_id → add_cross_features → add_historical_features
3. 保留必要的中间列（Type_ID_orig, From_City_orig）用于历史特征匹配
4. 最后统一删除中间列，确保输出向量一致
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_eng import parse_type_id, add_cross_features, optimize_dtypes
from src.historical_features import add_historical_features


class FeaturePipeline:
    """
    统一特征工程 Pipeline

    使用方法：
    ```python
    # 初始化
    pipeline = FeaturePipeline(city_data_loader, data_dir='output/processed_data')

    # 训练模式（启用历史特征 Dropout）
    df_train = pipeline.transform(df_train, year=2015, mode='train')

    # 评估模式（不启用 Dropout）
    df_val = pipeline.transform(df_val, year=2018, mode='eval')

    # 预测模式（不启用 Dropout，可能缺少某些列）
    df_pred = pipeline.transform(df_pred, year=2020, mode='predict')
    ```
    """

    def __init__(self, city_data_loader, data_dir='output/processed_data'):
        """
        Args:
            city_data_loader: CityDataLoader 实例
            data_dir: 历史数据目录
        """
        self.city_data_loader = city_data_loader  # 保存完整的 loader
        self.city_info = city_data_loader.city_info  # 年度城市信息字典 {year: DataFrame}
        self.city_edges = city_data_loader.city_edges
        self.city_nodes = city_data_loader.city_info  # 兼容性别名
        self.data_dir = data_dir

    def transform(self, df, year, mode='train', verbose=True):
        """
        统一特征工程入口

        Args:
            df: 输入 DataFrame，必须包含基础列：
                - Year, Type_ID, From_City, To_City
                - 可选：Flow_Count, Rank, Label（训练时需要）
            year: 当前年份（用于加载历史特征和获取对应年份的城市信息）
            mode: 模式，可选值：
                - 'train': 训练模式（启用历史特征 Dropout）
                - 'eval': 评估模式（不启用 Dropout）
                - 'predict': 预测模式（不启用 Dropout，容错处理缺失列）
            verbose: 是否打印详细信息

        Returns:
            df: 添加了所有特征的 DataFrame
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Feature Pipeline - Mode: {mode.upper()}, Year: {year}")
            print(f"{'='*60}")

        # 【步骤1】检查必要的列
        required_cols = ['Year', 'Type_ID', 'From_City', 'To_City']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        # 【步骤2】获取当前年份的城市信息
        # 从年度城市信息字典中获取对应年份的数据
        city_info_for_year = self.city_data_loader.get_city_info_for_year(year)
        if city_info_for_year is None:
            if verbose:
                print(f"  Warning: Year {year} city info not found, trying closest year...")
            # 尝试使用最接近的年份
            available_years = sorted(self.city_info.keys())
            if available_years:
                closest_year = min(available_years, key=lambda y: abs(y - year))
                city_info_for_year = self.city_info.get(closest_year)
                if verbose:
                    print(f"  Using city info from year {closest_year}")
            else:
                raise ValueError(f"No city info data available for any year!")

        # 【步骤3】添加占位列（如果是预测模式且缺少这些列）
        if mode == 'predict':
            if 'Flow_Count' not in df.columns:
                df['Flow_Count'] = 0
            if 'Rank' not in df.columns:
                df['Rank'] = 999
            if 'Label' not in df.columns:
                df['Label'] = 0

        # 【步骤4】解析 Type_ID（拆解为 6 个维度）
        if verbose:
            print(f"[1/4] Parsing Type_ID into 6 dimensions...")
        df, _ = parse_type_id(df, verbose=verbose)

        # 【步骤5】添加城市特征（From_City 和 To_City 的属性 + 差异特征）
        # 使用当前年份的城市信息
        if verbose:
            print(f"[2/4] Adding city features for year {year}...")
        df = add_cross_features(
            df,
            city_info_for_year,  # 使用对应年份的城市信息
            self.city_edges,
            verbose=verbose
        )

        # 【步骤6】添加历史特征
        if verbose:
            print(f"[3/4] Adding historical features from year {year-1}...")
        training_mode = (mode == 'train')
        df = add_historical_features(
            df,
            year,
            self.data_dir,
            verbose=verbose,
            training_mode=training_mode
        )

        # 【步骤7】优化数据类型
        if verbose:
            print(f"[4/4] Optimizing data types...")
        df = optimize_dtypes(df)

        # 【步骤8】特征分离 (Feature Decoupling)
        # 如果是生成训练数据，删掉静态特征以节省空间
        # 静态特征列表 (必须与 static_city_pairs.parquet 一致)
        static_cols = ['geo_distance', 'dialect_distance'] + [c for c in df.columns if c.endswith('_ratio')]

        if mode == 'train':
            # 检查这些列是否存在，存在则删除
            cols_to_drop = [c for c in static_cols if c in df.columns]
            if verbose and cols_to_drop:
                print(f"  [Decoupling] Dropping {len(cols_to_drop)} static features to save disk space...")
            df = df.drop(columns=cols_to_drop)

        # 【步骤9】保留中间列用于历史特征匹配
        # 注意：Type_ID_orig 和 From_City_orig 需要保留在保存的数据中
        # 因为后续年份会读取这些数据来构建历史特征
        # 只在训练/预测时才删除这些列（通过 get_feature_columns 方法）

        if verbose:
            print(f"\n✓ Feature Pipeline Completed")
            print(f"  Final shape: {df.shape}")
            print(f"  Final columns: {list(df.columns)}")
            print(f"{'='*60}\n")

        return df

    def get_feature_columns(self, df):
        """
        获取特征列（排除标签列、ID列和中间列）

        Args:
            df: 经过 transform() 处理后的 DataFrame

        Returns:
            feature_cols: 特征列名列表
        """
        # 排除的列
        exclude_cols = [
            'Year', 'qid',  # ID列
            'To_City',  # 目标城市（不能作为特征）
            'Flow_Count', 'Rank', 'Label',  # 标签列
            'Type_ID_orig', 'From_City_orig'  # 中间列（用于历史特征匹配，不作为模型特征）
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
