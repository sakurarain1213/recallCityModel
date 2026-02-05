import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .config import Config

class MigrationRanker:
    def __init__(self):
        self.model = None

    def prepare_lgbm_data(self, df, feature_cols, target_col):
        """
        LightGBM Ranker 要求数据必须按 Group 排序。
        Group = (Year, Month, From_City, Type_ID) 的组合

        【关键修复】添加随机打乱，防止输入顺序泄露
        """
        # 【关键修复】添加一个随机列，用于在 Group 内部打乱顺序
        # 这一步是为了防止 正样本永远排在负样本前面 导致的虚高分数
        df = df.copy()  # 避免修改原始数据
        df['random_shuffle'] = np.random.random(len(df))

        # 1. 关键：必须排序！先按 Group ID 排，再按随机数排
        df = df.sort_values(by=['Year', 'Month', 'From_City', 'Type_ID', 'random_shuffle'])

        # 2. 计算 Group info (每个 Query 有多少个文档)
        # 获取排序后的 group ID
        df['group_id'] = df['Year'].astype(str) + '_' + df['Month'].astype(str) + '_' + \
                         df['From_City'].astype(str) + '_' + df['Type_ID']

        # 提取 group sizes (必须与 X 的行顺序严格对应)
        q_groups = df.groupby('group_id', sort=False).size().values

        # 移除辅助列
        df = df.drop(columns=['random_shuffle', 'group_id'])

        X = df[feature_cols]
        y = df[target_col]

        return X, y, q_groups, df # 返回df以便后续分析

    def train(self, train_df, val_df, feature_cols, cat_cols):
        print("Preparing Train/Val datasets...")
        X_train, y_train, q_train, _ = self.prepare_lgbm_data(train_df, feature_cols, 'Label')
        X_val, y_val, q_val, _ = self.prepare_lgbm_data(val_df, feature_cols, 'Label')

        print(f"Train Groups: {len(q_train)}, Val Groups: {len(q_val)}")

        train_set = lgb.Dataset(X_train, y_train, group=q_train, categorical_feature=cat_cols)
        val_set = lgb.Dataset(X_val, y_val, group=q_val, categorical_feature=cat_cols, reference=train_set)

        print("Starting LightGBM Training...")
        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=20)
            ]
        )
        return self.model

    def predict(self, df, feature_cols):
        return self.model.predict(df[feature_cols])

    def plot_importance(self, output_dir):
        print("Plotting feature importance...")
        plt.figure(figsize=(12, 6))
        lgb.plot_importance(self.model, max_num_features=20, importance_type='gain')
        plt.title('Feature Importance (Gain)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()
