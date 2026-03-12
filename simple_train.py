"""
极速召回训练脚本
支持：指定训练/验证年份、按比例随机采样 Query (快速测试)
"""
import os
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 基础配置 =================
PROCESSED_DIR = Path("output/processed_ready")
OUTPUT_DIR = Path("output")

FEATS = [
    'gender', 'age_group', 'education', 'industry', 'income', 'family',
    'geo_distance', 'dialect_distance',
    'gdp_per_capita_ratio', 'unemployment_rate_ratio', 'housing_price_avg_ratio',
    'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio',
    'education_score_ratio', 'transport_convenience_ratio', 'population_total_ratio',
    'is_same_province'
]

CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']


def load_data(years: list, sample_ratio: float = 1.0, random_seed: int = 42) -> pd.DataFrame:
    """加载数据，并支持按 Query (qid) 进行采样"""
    print(f"Loading data for years: {years} | Query Sample Ratio: {sample_ratio*100}%")
    dfs = []
    for year in years:
        file_path = PROCESSED_DIR / f'processed_{year}.feather'
        if not file_path.exists():
            print(f"  [Warning] Year {year} file not found: {file_path}")
            continue
            
        df = pd.read_feather(file_path)
        total_queries = df['qid'].nunique()
        
        # 核心：按 Query (qid) 级别进行采样，保证每个 Query 的 336 个城市候选完整
        if 0.0 < sample_ratio < 1.0:
            unique_qids = pd.Series(df['qid'].unique())
            sampled_qids = unique_qids.sample(frac=sample_ratio, random_state=random_seed)
            df = df[df['qid'].isin(sampled_qids)].copy()
            print(f"  - {year}: Sampled {len(sampled_qids):,} queries (out of {total_queries:,}) -> {len(df):,} rows")
        else:
            print(f"  - {year}: Loaded all {total_queries:,} queries -> {len(df):,} rows")
            
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No valid data loaded for years {years}!")
        
    return pd.concat(dfs, axis=0, ignore_index=True)


def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """计算样本权重：让模型重点关注头部正样本"""
    weights = np.ones(len(df), dtype=np.float32)
    pos_mask = df['Label'] == 1.0
    rank = df['Rank'].astype('int16')
    
    weights[pos_mask & (rank <= 3)] = 15.0
    weights[pos_mask & (rank > 3) & (rank <= 10)] = 8.0
    weights[pos_mask & (rank > 10) & (rank <= 20)] = 5.0
    return weights


def main():
    # =====================================================================
    # 🌟 智能控制台：在这里修改你的训练配置！
    # =====================================================================
    TRAIN_YEARS = [2000, 2001, 2002]   # 训练集年份
    VAL_YEARS   = [2003]               # 验证集年份（用于 Early Stopping）
    TEST_YEARS  = [2004]               # 测试集年份（这里仅作记录，模型训练完用 inference.py 测这个年份）
    
    # 采样比例控制 (0.0 ~ 1.0)。例如 0.1 表示随机抽取 10% 的 Query 跑个基线。
    # 如果你想跑全量数据，请改回 1.0
    SAMPLE_RATIO = 0.1 
    # =====================================================================

    print("="*60)
    print(f"🚀 LightGBM Fast Training Session")
    print(f"Train Years : {TRAIN_YEARS}")
    print(f"Val Years   : {VAL_YEARS}")
    print(f"Sample Ratio: {SAMPLE_RATIO}")
    print("="*60)

    # 1. 加载数据
    df_train = load_data(TRAIN_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)
    df_val   = load_data(VAL_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)

    # 2. 计算权重
    print("\nCalculating sample weights...")
    train_weights = calculate_sample_weights(df_train)

    # 3. 构建 LGBM 数据集
    print("Constructing LightGBM Datasets...")
    train_ds = lgb.Dataset(
        df_train[FEATS],
        label=df_train['Label'],
        weight=train_weights,
        categorical_feature=CATS,
        free_raw_data=False
    )
    
    val_ds = lgb.Dataset(
        df_val[FEATS],
        label=df_val['Label'],
        categorical_feature=CATS,
        reference=train_ds,
        free_raw_data=False
    )

    # 4. 训练参数
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': 8,
        'n_estimators': 1000,
        'n_jobs': os.cpu_count() or 8,
        'verbosity': -1,
    }

    # 5. 训练模型
    print("\nTraining started...")
    start_time = time.time()
    model = lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(50)
        ]
    )
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # 6. 保存模型 (用 TEST_YEARS 命名，表示这是用来预测该年份的模型)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_year = TEST_YEARS[0] if TEST_YEARS else VAL_YEARS[0]
    model_path = OUTPUT_DIR / f'model_{target_year}_fast.txt'
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # 7. 打印特征重要性
    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')})
    print("\n🏆 Top Features by Gain:")
    print(imp.sort_values('importance', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()