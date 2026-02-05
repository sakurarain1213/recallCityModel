import gc
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
import argparse
from src.config import Config
# uv run train.py    【一定要在cmd 不要powershell】
# uv run train.py --end_year 2010  【训练到2010年 后面完全不用】
# uv run train.py --gpu  【使用 GPU 训练，需要先安装 GPU 版本】
# uv run train.py --end_year 2010 --gpu  【组合使用】
# uv run train.py --end_year 2020 --gpu  【组合使用】
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_year_config(end_year=None):
    """
    根据截止年份自动划分训练集、验证集、测试集

    Args:
        end_year: 训练截止年份（包含）
                 - None: 使用默认配置（训练到2017，验证2018，测试2019-2020）
                 - 2010: 训练2001-2007，验证2008，测试2009-2010
                 - 其他年份以此类推

    Returns:
        train_years, val_years, test_years
    """
    if end_year is None:
        # 默认配置：使用 parquet 数据到 2020
        train_years = list(range(Config.TRAIN_START_YEAR, Config.TRAIN_END_YEAR + 1))
        val_years = Config.VAL_YEARS
        test_years = Config.TEST_YEARS
        print(f"使用默认配置：")
        print(f"  训练集: {train_years[0]}-{train_years[-1]} ({len(train_years)}年)")
        print(f"  验证集: {val_years}")
        print(f"  测试集: {test_years}")
    else:
        # 自定义配置：根据 end_year 自动划分
        # 训练集：2001 到 (end_year - 3)
        # 验证集：end_year - 2
        # 测试集：end_year - 1 到 end_year

        if end_year < 2004:
            raise ValueError(f"end_year 必须 >= 2004（至少需要3年训练数据 + 1年验证 + 2年测试）")

        train_start = Config.TRAIN_START_YEAR  # 2001
        train_end = end_year - 3
        val_year = end_year - 2
        test_start = end_year - 1
        test_end = end_year

        train_years = list(range(train_start, train_end + 1))
        val_years = [val_year]
        test_years = list(range(test_start, test_end + 1))

        print(f"自定义配置（截止年份={end_year}）：")
        print(f"  训练集: {train_years[0]}-{train_years[-1]} ({len(train_years)}年)")
        print(f"  验证集: {val_years}")
        print(f"  测试集: {test_years}")

    return train_years, val_years, test_years


def load_processed_data(years, data_dir='output/processed_data'):
    """加载处理好的parquet文件（生成器模式，避免内存溢出）"""
    for year in years:
        file_path = Path(data_dir) / f"processed_{year}.parquet"
        if file_path.exists():
            print(f"Loading {year}...")
            df = pd.read_parquet(file_path)
            yield df
            del df
            gc.collect()


def prepare_features(df):
    """准备特征和标签"""
    # 排除的列 - 移除泄露特征、ID列和辅助列
    exclude_cols = [
        'Label',        # 标签
        'To_City',      # 目标城市（不能作为特征）
        'Flow_Count',   # 泄露！
        'Rank',         # 泄露！
        'Total_Count',  # 可能泄露
        'pred_score',   # 预测结果
        'Type_ID_orig', 'From_City_orig', # 中间列

        # 【关键修复】必须排除 ID 类和时间类特征！
        'qid',          # 绝对不能进模型，这是随机ID
        'Year',         # 年份不建议直接进树模型（外推性差），除非做成类别
        'Month'         # 如果Month都是12，也没意义
    ]

    # 特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['Label']

    return X, y, feature_cols


def calculate_ndcg(y_true, y_pred, group_counts, k=20):
    """计算NDCG@k（保留用于兼容性，但二分类模式下不使用）"""
    from sklearn.metrics import ndcg_score

    ndcg_scores = []
    start_idx = 0

    for count in group_counts:
        end_idx = start_idx + count
        y_t = y_true[start_idx:end_idx]
        y_p = y_pred[start_idx:end_idx]

        # 计算NDCG
        if len(y_t) > 0:
            score = ndcg_score([y_t], [y_p], k=k)
            ndcg_scores.append(score)

        start_idx = end_idx

    return np.mean(ndcg_scores)


def train_model(train_years=None, val_years=None, use_gpu=False):
    """训练二分类模型（使用分批加载避免内存溢出）

    Args:
        train_years: 训练年份列表，如 [2001, 2002, ..., 2017]
        val_years: 验证年份列表，如 [2018]
        use_gpu: 是否使用 GPU 训练（需要安装 lightgbm GPU 版本）
    """
    if train_years is None:
        train_years = list(range(Config.TRAIN_START_YEAR, Config.TRAIN_END_YEAR + 1))
    if val_years is None:
        val_years = Config.VAL_YEARS

    print("="*60)
    print(f"Step 1: Preparing training data ({train_years[0]}-{train_years[-1]})...")
    if use_gpu:
        print("🚀 GPU 训练模式已启用")
    print("="*60)

    # 分批策略：每3年一批，避免内存溢出同时保证学习效果
    # === 修改后 ===
    import random
    # 彻底打乱年份，打破时间依赖
    # 例如：Batch 1 可能是 [2015, 2002, 2010]
    # 这样模型每一批都能看到不同时代的特征，不会"遗忘"古代，也不会"过拟合"现代
    shuffled_years = train_years.copy()
    random.shuffle(shuffled_years)

    batch_size = 3
    year_batches = [shuffled_years[i:i+batch_size] for i in range(0, len(shuffled_years), batch_size)]

    print(f"Training strategy: {len(year_batches)} batches (Randomized Years)")
    for i, batch in enumerate(year_batches):
        print(f"  Batch {i+1}: Years {batch}")

    # 第一步：加载第一批数据以获取特征列
    print(f"\nLoading first batch to extract feature columns...")
    first_year_file = Path('output/processed_data') / f"processed_{train_years[0]}.parquet"
    first_df = pd.read_parquet(first_year_file)
    _, _, feature_cols = prepare_features(first_df)
    print(f"Features: {len(feature_cols)}")
    del first_df
    gc.collect()

    print("\n" + "="*60)
    print(f"Step 2: Loading validation data ({val_years[0]})...")
    print("="*60)

    val_file = Path('output/processed_data') / f"processed_{val_years[0]}.parquet"
    val_df = pd.read_parquet(val_file)
    print(f"Validation data: {len(val_df):,} rows")

    X_val, y_val, _ = prepare_features(val_df)

    del val_df
    gc.collect()

    print("\n" + "="*60)
    print("Step 3: Training Binary Classification model (batch mode)...")
    print("="*60)

    # 创建模型保存目录
    output_dir = Path(Config.OUTPUT_DIR) / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练参数
    params = Config.LGBM_PARAMS_GPU.copy() if use_gpu else Config.LGBM_PARAMS.copy()

    # 初始化模型
    current_booster = None

    # 分批训练
    for batch_idx, year_batch in enumerate(year_batches):
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx+1}/{len(year_batches)}: Years {year_batch[0]}-{year_batch[-1]}")
        print(f"{'='*60}")

        # 加载该批次的所有年份数据
        batch_dfs = []
        for year in year_batch:
            year_file = Path('output/processed_data') / f"processed_{year}.parquet"
            if not year_file.exists():
                print(f"  Warning: {year_file} not found, skipping")
                continue

            print(f"  Loading year {year}...")
            year_df = pd.read_parquet(year_file)
            batch_dfs.append(year_df)

        if not batch_dfs:
            print(f"  No data found for batch {batch_idx+1}, skipping")
            continue

        # 合并该批次的数据
        print(f"  Merging {len(batch_dfs)} years...")
        batch_df = pd.concat(batch_dfs, axis=0, ignore_index=True)
        print(f"  Total rows in batch: {len(batch_df):,}")

        del batch_dfs
        gc.collect()

        # 准备特征
        X_train, y_train, _ = prepare_features(batch_df)

        del batch_df
        gc.collect()

        # 创建训练集（二分类不需要 group 参数）
        print(f"  Creating LightGBM Dataset (Binary Mode)...")
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        # 创建验证集（reference 设为当前训练集，保证特征分桶一致）
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        # 训练（如果是第一批，创建新模型；否则继续训练）
        if current_booster is None:
            print(f"  Creating new model...")
            current_booster = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                valid_names=['valid'],
                num_boost_round=200,  # 每批训练200轮
                callbacks=[
                    lgb.log_evaluation(period=50),  # 每50轮打印一次
                    lgb.early_stopping(stopping_rounds=30, verbose=False),  # 30轮不提升则早停
                ],
                keep_training_booster=True  # 允许继续训练
            )
        else:
            print(f"  Continuing training from previous model (current trees: {current_booster.num_trees()})...")
            current_booster = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                valid_names=['valid'],
                num_boost_round=200,  # 每批训练200轮
                init_model=current_booster,  # 从上一批的模型继续
                callbacks=[
                    lgb.log_evaluation(period=50),  # 每50轮打印一次
                    lgb.early_stopping(stopping_rounds=30, verbose=False),  # 30轮不提升则早停
                ],
                keep_training_booster=True  # 允许继续训练
            )

        del X_train, y_train, train_data
        gc.collect()

        print(f"  Batch {batch_idx+1} completed.")
        print(f"  Total trees in model: {current_booster.num_trees()}")
        # 打印 binary_logloss
        if 'binary_logloss' in current_booster.best_score['valid']:
            print(f"  Current best Binary LogLoss: {current_booster.best_score['valid']['binary_logloss']:.6f}")

        # 保存中间 checkpoint
        checkpoint_path = checkpoint_dir / f'model_batch_{batch_idx+1}_years_{year_batch[0]}-{year_batch[-1]}.txt'
        current_booster.save_model(str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")

    print("\n" + "="*60)
    print("Step 4: Final evaluation on validation set...")
    print("="*60)

    # 打印 binary_logloss
    if 'binary_logloss' in current_booster.best_score['valid']:
        print(f"Best Validation Binary LogLoss: {current_booster.best_score['valid']['binary_logloss']:.6f}")
    print(f"Total trees in final model: {current_booster.num_trees()}")

    print("\n" + "="*60)
    print("Step 5: Saving model...")
    print("="*60)

    # 保存最终模型
    model_path = output_dir / 'binary_model.txt'
    current_booster.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # 保存特征列名
    feature_path = output_dir / 'feature_cols.pkl'
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to {feature_path}")

    return current_booster, feature_cols


def plot_feature_importance(model, feature_cols, top_n=20):
    """绘制特征重要性"""
    print("\n" + "="*60)
    print("Step 7: Plotting feature importance...")
    print("="*60)

    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} features:")
    print(feature_importance.head(top_n))

    # 绘图
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 保存图片
    output_dir = Path(Config.OUTPUT_DIR) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved to {plot_path}")

    plt.close()

    return feature_importance


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练迁移排序模型')
    parser.add_argument(
        '--end_year',
        type=int,
        default=None,
        help='训练截止年份（包含）。不填则使用默认配置（训练到2017）。'
             '例如：--end_year 2010 表示训练2001-2007，验证2008，测试2009-2010'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='使用 GPU 训练（需要安装 lightgbm GPU 版本和 CUDA）'
    )
    args = parser.parse_args()

    # 根据参数配置年份划分
    print("="*60)
    print("年份配置")
    print("="*60)
    train_years, val_years, test_years = parse_year_config(args.end_year)
    print()

    # 训练模型
    model, feature_cols = train_model(train_years, val_years, use_gpu=args.gpu)

    # 绘制特征重要性
    feature_importance = plot_feature_importance(model, feature_cols)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
