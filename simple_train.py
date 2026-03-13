import os
import time
import gc  # 引入垃圾回收机制
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

#  cd /data1/wxj/Recall_city_project/ && uv run simple_train.py
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
    """加载数据，预分配内存 + 零拷贝填充，彻底跳过 pd.concat"""
    print(f"Loading data for years: {years} | Query Sample Ratio: {sample_ratio*100}%")

    keep_cols = sorted(set(FEATS + ['Label', 'Rank', 'qid']))

    # ── Pass 1: 只读 qid 列，统计每年采样后的行数 ──
    print("Pass 1: Counting rows per year...")
    year_row_counts = []
    year_files = []
    for year in years:
        file_path = PROCESSED_DIR / f'processed_{year}.feather'
        if not file_path.exists():
            print(f"  [Warning] Year {year} file not found: {file_path}")
            continue
        qid_series = pd.read_feather(file_path, columns=['qid'])['qid']
        if 0.0 < sample_ratio < 1.0:
            unique_qids = pd.Series(qid_series.unique())
            sampled_qids = set(unique_qids.sample(frac=sample_ratio, random_state=random_seed))
            n_rows = int((qid_series.isin(sampled_qids)).sum())
            print(f"  - {year}: {n_rows:,} rows (sampled {len(sampled_qids):,} / {len(unique_qids):,} queries)")
        else:
            n_rows = len(qid_series)
            print(f"  - {year}: {n_rows:,} rows (all queries)")
        year_row_counts.append(n_rows)
        year_files.append((year, file_path))
        del qid_series
        gc.collect()

    total_rows = sum(year_row_counts)
    if total_rows == 0:
        raise ValueError(f"No valid data loaded for years {years}!")
    print(f"Total rows to load: {total_rows:,}")

    # ── Pass 2: 预分配 numpy 数组，逐年填充 ──
    print("Pass 2: Pre-allocating memory and filling...")
    # 确定每列的 dtype
    col_dtypes = {}
    for c in keep_cols:
        if c in CATS:
            col_dtypes[c] = np.int8
        elif c == 'Label':
            col_dtypes[c] = np.int8
        elif c == 'Rank':
            col_dtypes[c] = np.int16
        elif c == 'qid':
            col_dtypes[c] = np.int64
        else:
            col_dtypes[c] = np.float32

    # 预分配
    arrays = {col: np.empty(total_rows, dtype=col_dtypes[col]) for col in keep_cols}
    est_gb = sum(a.nbytes for a in arrays.values()) / 1024**3
    print(f"  Pre-allocated {est_gb:.2f} GB for {len(keep_cols)} columns x {total_rows:,} rows")

    # 逐年读取并填充
    offset = 0
    for (year, file_path), n_rows in zip(year_files, year_row_counts):
        t0 = time.time()
        df = pd.read_feather(file_path, columns=keep_cols)

        # 采样
        if 0.0 < sample_ratio < 1.0:
            unique_qids = pd.Series(df['qid'].unique())
            sampled_qids = set(unique_qids.sample(frac=sample_ratio, random_state=random_seed))
            mask = df['qid'].isin(sampled_qids)
            df = df.loc[mask]

        # 逐列填充到预分配数组
        end = offset + len(df)
        for col in keep_cols:
            arrays[col][offset:end] = df[col].to_numpy(dtype=col_dtypes[col], copy=False)

        offset = end
        del df
        gc.collect()
        print(f"  - {year}: filled [{offset - n_rows:,} : {offset:,}] in {time.time()-t0:.1f}s")

    # 截断（理论上 offset == total_rows，但防御性处理）
    if offset < total_rows:
        arrays = {col: arr[:offset] for col, arr in arrays.items()}

    # ── 一次性构建 DataFrame（零拷贝，numpy 数组直接作为底层 buffer）──
    print("Building final DataFrame from pre-allocated arrays...")
    final_df = pd.DataFrame(arrays)
    del arrays
    gc.collect()

    # 转 category（此时只有一张表，int8 -> category 极快）
    print("Converting categoricals...")
    for col in CATS:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('category')

    print(f"Data loaded: {final_df.shape}, Memory: {final_df.memory_usage(deep=True).sum()/1024**3:.2f} GB")
    return final_df

def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """计算样本权重：让模型重点关注头部正样本"""
    weights = np.ones(len(df), dtype=np.float32)
    pos_mask = df['Label'] == 1
    rank = df['Rank']
    
    weights[pos_mask & (rank <= 3)] = 15.0
    weights[pos_mask & (rank > 3) & (rank <= 10)] = 8.0
    weights[pos_mask & (rank > 10) & (rank <= 20)] = 5.0
    return weights

class CheckpointCallback:
    """自定义 LightGBM 回调函数：用于定期保存 Checkpoint 模型文件"""
    def __init__(self, output_dir, freq=10):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        # LightGBM 机制要求的属性，确保在日志之后执行，且是在迭代完成后触发
        self.order = 30  
        self.before_iteration = False

    def __call__(self, env):
        # env.iteration 是从 0 开始的
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            ckpt_path = self.output_dir / f"model_iter_{current_round}.txt"
            env.model.save_model(str(ckpt_path))
            print(f" 💾 [Checkpoint] Model saved at iteration {current_round} -> {ckpt_path}")

def main():
    TRAIN_YEARS = list(range(2000, 2017))
    VAL_YEARS   = [2017, 2018]
    TEST_YEARS  = [2019, 2020]

    SAMPLE_RATIO = 0.1

    print("="*60)
    print(f"🚀 LightGBM Fast & Memory-Optimized Training Session")
    print("="*60)

    # 1. 加载数据
    df_train = load_data(TRAIN_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)
    df_val   = load_data(VAL_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)

    # 2. 计算权重
    print("\nCalculating sample weights...")
    train_weights = calculate_sample_weights(df_train)

    # 🌟 修复点：将 params 提前定义！
    # 因为 max_bin 是构建 Dataset 直方图时的底层参数，必须在 construct() 之前让 Dataset 知道。
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': 8,
        'n_estimators': 2000,     # 全局最大轮次 目前测试 1k轮=60min 且没有触发早停 因此依然有提升空间
        
        # 🚀 性能与速度优化
        'n_jobs': 24,             # 降至 24 线程，减少调度开销
        'num_threads': 24,        # 明确传递底层 OpenMP 线程数
        'max_bin': 63,            # 直方图压缩，提升构建速度，减少缓存未命中
        'bagging_fraction': 0.8,  # 行采样
        'bagging_freq': 1,        # 配合 bagging_fraction 使用，每 1 轮进行一次采样
        'feature_fraction': 0.8,  # 列采样
        
        'verbosity': -1,
        # 'two_round': True,      # 如果依然 OOM，把这行注释打开（两阶段加载，能再降内存，但稍慢）
    }

    # 3. 构建 LGBM 数据集
    print("Constructing LightGBM Datasets...")
    train_ds = lgb.Dataset(
        df_train[FEATS],
        label=df_train['Label'],
        weight=train_weights,
        categorical_feature=CATS,
        params=params,          # 🌟 修复点：传入 params，告诉底层按照 max_bin=63 进行数据分箱
        free_raw_data=True      # 🚀 内存优化 2：允许 LGBM 在构建后丢弃原始数据
    )
    
    val_ds = lgb.Dataset(
        df_val[FEATS],
        label=df_val['Label'],
        categorical_feature=CATS,
        reference=train_ds,
        params=params,          # 🌟 修复点：验证集同样传入 params
        free_raw_data=True
    )

    # 🚀 内存优化 3：手动强制 LGBM 提前构建底层直方图数据
    print("Forcing LightGBM Dataset construction and clearing Pandas memory...")
    train_ds.construct()
    val_ds.construct()

    # 构建完成后，Pandas 原数据彻底没用了，直接暴力删除并回收内存！
    del df_train
    del df_val
    del train_weights
    gc.collect()

    # 4. 训练模型
    print("\nTraining started (Utilizing 24 Cores)...")
    start_time = time.time()
    
    # 实例化自定义的 checkpoint 回调，每 10 轮保存一次
    ckpt_callback = CheckpointCallback(output_dir=OUTPUT_DIR, freq=10)

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],          # 彻底移除 train_ds 评估，跳过无意义的大量数据打分和排序
        valid_names=['val'],          # 同步修改评估集名称
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(10),  # 每 10 轮打印一次
            ckpt_callback            # 加入 Checkpoint 回调
        ]
    )
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # 5. 保存模型
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_year = TEST_YEARS[0] if TEST_YEARS else VAL_YEARS[0]
    model_path = OUTPUT_DIR / f'model_{target_year}_fast.txt'
    model.save_model(str(model_path))
    print(f"Final Model saved to {model_path}")

    # 6. 打印特征重要性
    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')})
    print("\n🏆 Top Features by Gain:")
    print(imp.sort_values('importance', ascending=False).to_string(index=False))

if __name__ == '__main__':
    main()