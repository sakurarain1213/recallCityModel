"""
分批训练模式 + 内存极致优化版 (Numpy-First Strategy)
"""
import gc
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import Config
from src.city_data import CityDataLoader
from src.data_loader_v2 import load_raw_data_fast
from src.feature_pipeline import FeaturePipeline
from evaluate import evaluate_year, EvalContext

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_year_data(year, pipeline, hard_candidates, mode='train'):
    """
    获取一年的数据
    """
    cache_file = Path(Config.OUTPUT_DIR) / 'cache' / f"processed_{year}.parquet"

    # 1. 尝试读取缓存
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file, engine='pyarrow')
        except:
            pass

    print(f"  [Processing] Generating data for Year {year}...")

    # 2. 加载原始数据
    df = load_raw_data_fast(Config.DB_PATH, year, hard_candidates, Config.NEG_SAMPLE_RATE)
    if df.empty:
        return None

    # 3. 特征工程
    df = pipeline.transform(df, year, mode=mode, verbose=False)

    # 4. 写入缓存 (优化类型)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('category')
    
    # 强制 float32
    f_cols = df.select_dtypes(include=['float64']).columns
    if len(f_cols) > 0:
        df[f_cols] = df[f_cols].astype('float32')

    table = pa.Table.from_pandas(df, nthreads=4)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(cache_file), compression='snappy', use_dictionary=True, write_statistics=False)

    return df

def generate_batches(end_year, start_year=2001):
    val_years = [end_year - 2, end_year - 1]
    train_end = end_year - 3

    # 【内存关键】每批训练多少年的数据 如果内存依然紧张，将此处的 3 改为 2
    BATCH_SIZE = 5
    
    batches = []
    current = start_year
    batch_idx = 1

    while current <= train_end:
        batch_years = []
        for _ in range(BATCH_SIZE):
            if current <= train_end:
                batch_years.append(current)
                current += 1

        if batch_years:
            batches.append({
                'name': f'batch_{batch_idx}_{min(batch_years)}-{max(batch_years)}',
                'train_years': batch_years,
                'val_years': val_years
            })
            batch_idx += 1

    return batches, val_years

def train_dynamic(target_end_year, use_gpu=False):
    print(f"🚀 启动动态分批训练 (Numpy优化版) | 目标预测年份: {target_end_year}")

    # 1. 初始化资源
    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Path(Config.OUTPUT_DIR)/'cache')
    hard_candidates = loader.get_city_ids()

    # 2. 生成 Batches
    batches, val_years = generate_batches(target_end_year, start_year=Config.DATA_START_YEAR + 1)
    
    print(f"📅 验证集: {val_years}")
    for b in batches:
        print(f"  - {b['name']}: {b['train_years']}")

    # 3. 预加载验证集 (带采样优化)
    print(f"\n📦 预加载验证集 {val_years}...")
    val_dfs = []
    for yr in val_years:
        df = get_year_data(yr, pipeline, hard_candidates, mode='eval')
        if df is not None: 
            val_dfs.append(df)
            
    if not val_dfs:
        print("❌ 验证集为空")
        return

    full_val = pd.concat(val_dfs, axis=0, ignore_index=True)
    feature_cols = pipeline.get_feature_columns(full_val)
    
    # 【内存优化 A】验证集采样
    # 3000万验证集太大，限制在 500 万行以内足以评估
    MAX_VAL_SIZE = 5000000 
    if len(full_val) > MAX_VAL_SIZE:
        print(f"  ⚠️ 验证集过大 ({len(full_val):,})，采样至 {MAX_VAL_SIZE:,} 行以节省内存...")
        full_val = full_val.sample(n=MAX_VAL_SIZE, random_state=42)
    
    print(f"  ⚡ 转换验证集为 Numpy Float32...")
    # 显式转换为 numpy float32，避免隐式 float64
    val_X = full_val[feature_cols].values.astype(np.float32)
    val_y = full_val['Label'].values.astype(np.float32)
    
    # 立即释放 DataFrame
    del full_val, val_dfs
    gc.collect()

    print(f"✅ 验证集就绪: {len(val_X):,} 行")

    # 【修复 Batch 2 报错的关键】
    # 验证集必须保留 Raw Data (False)，因为它要被多个 Batch 重复使用
    # 训练集使用 True (节省内存)，验证集使用 False (兼容多轮训练)

    # 【修正】定义类别特征列表
    categorical_feats = ['From_City', 'is_same_province']
    categorical_feats = [c for c in categorical_feats if c in feature_cols]

    val_ds = lgb.Dataset(
        val_X,
        label=val_y,
        feature_name=feature_cols,        # 关键：传入特征名列表
        categorical_feature=categorical_feats, # 关键：指定类别特征
        free_raw_data=False
    )

    # 注意：因为 free_raw_data=False，val_ds 会持有 val_X 的引用
    # 所以这里不能删除 val_X，否则 val_ds 也会失效
    # LightGBM 会自动管理这部分内存（约 0.7GB，在可接受范围内）

    # 4. 逐 Batch 训练
    model = None
    model_save_path = Path(Config.OUTPUT_DIR) / 'models' / f'lgb_end_{target_end_year}.txt'
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    params = Config.LGBM_PARAMS_GPU if use_gpu else Config.LGBM_PARAMS
    evals_result = {}

    for i, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"🏃 Training {batch['name']} ({i+1}/{len(batches)})")

        # 【内存优化 B】逐年加载并转 Numpy，不进行 Pandas Concat
        train_arrays = []
        train_labels = []
        total_rows = 0

        for yr in batch['train_years']:
            print(f"  📖 Loading Year {yr}...")
            df = get_year_data(yr, pipeline, hard_candidates, mode='train')
            if df is None or df.empty: continue
            
            # 补齐列
            for col in feature_cols:
                if col not in df.columns: df[col] = 0
            
            # 立即转为 float32 numpy array
            # 这步是关键：防止 int 和 float 混合导致 concat 后变成 float64
            arr = df[feature_cols].values.astype(np.float32)
            lbl = df['Label'].values.astype(np.float32)
            
            train_arrays.append(arr)
            train_labels.append(lbl)
            total_rows += len(arr)
            
            # 立即释放 DataFrame
            del df
            gc.collect()

        if total_rows == 0:
            print("  ⚠️ 跳过空Batch")
            continue

        print(f"  ⚡ Merging into single Float32 matrix ({total_rows:,} rows)...")
        # 使用 numpy vstack (比 pandas concat 省内存且类型可控)
        X_train = np.vstack(train_arrays)
        y_train = np.concatenate(train_labels)

        # 释放临时列表
        del train_arrays, train_labels
        gc.collect()

        # 【修正】定义类别特征列表 (LightGBM 需要知道哪些列是类别)
        # 注意：这里使用的是特征名，必须确保这些列在 feature_cols 中
        categorical_feats = ['From_City', 'is_same_province']
        # 确保只包含存在的列
        categorical_feats = [c for c in categorical_feats if c in feature_cols]

        print(f"  📦 Constructing LGBM Dataset (Categorical: {categorical_feats})...")
        # 【修正】显式传入 feature_name 和 categorical_feature
        train_ds = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_cols,        # 关键：传入特征名列表
            categorical_feature=categorical_feats, # 关键：指定类别特征
            free_raw_data=True
        )
        
        # 立即释放巨大的 Numpy 数组
        del X_train, y_train
        gc.collect()

        # 训练
        print(f"  🔥 Fitting model...")
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=params['n_estimators'],
            valid_sets=[train_ds, val_ds],
            valid_names=['train', 'val'],
            init_model=model,
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(50),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        del train_ds
        gc.collect()

    if model:
        model.save_model(str(model_save_path))
        print(f"\n💾 模型已保存: {model_save_path}")

        # 【新增】打印并保存特征重要性列表
        print_and_plot_importance(model, target_end_year)
        plot_history(evals_result, target_end_year)

        # 【快速评估】在测试集上评估
        print(f"\n{'='*60}")
        print(f"📈 在测试集 {Config.TEST_YEARS} 上快速评估...")

        # 初始化评估上下文
        ctx = EvalContext()
        # 加载刚才训练好的模型
        ctx.load_resources(model_save_path)

        # 评估配置中定义的测试年份 (通常是 target_end_year - 1 或 target_end_year)
        # 这里为了演示，我们评估 target_end_year 这一年
        # 注意：cache_dir 指向训练生成的数据目录
        CACHE_DIR = Path(Config.OUTPUT_DIR) / 'cache'

        # 使用 evaluate_year (这是 evaluate.py 中的主函数)
        evaluate_year(target_end_year, ctx, sample_size=50000, cache_dir=CACHE_DIR)

def print_and_plot_importance(model, year):
    """
    【新增】打印文本版特征重要性并保存图表
    """
    # 1. 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    names = model.feature_name()

    # 2. 构建 DataFrame
    df_imp = pd.DataFrame({'feature': names, 'gain': importance})
    df_imp = df_imp.sort_values(by='gain', ascending=False).reset_index(drop=True)

    # 3. 打印 Top 20 到控制台
    print(f"\n📊 Feature Importance (Top 20) - End {year}")
    print("-" * 60)
    print(f"{'Rank':<5} {'Feature':<30} {'Gain':<15} {'Share':<10}")
    print("-" * 60)
    total_gain = df_imp['gain'].sum()
    for i, row in df_imp.head(20).iterrows():
        share = row['gain'] / total_gain
        print(f"{i+1:<5} {row['feature']:<30} {row['gain']:.2f}          {share:.1%}")
    print("-" * 60)

    # 4. 画图 (带名字)
    print("\n📊 生成特征重要性图表...")
    plt.figure(figsize=(12, 10))
    lgb.plot_importance(model, max_num_features=30, importance_type='gain',
                        height=0.5, title=f'Feature Importance (Gain) - End {year}', grid=False)
    plt.tight_layout()
    plt.savefig(Path(Config.OUTPUT_DIR) / f'feature_importance_{year}.png')

def plot_feature_importance(model, year):
    # (保持不变，已废弃)
    print("\n📊 生成特征重要性图表...")
    plt.figure(figsize=(12, 10))
    lgb.plot_importance(model, max_num_features=30, importance_type='gain',
                        height=0.5, title=f'Feature Importance (Gain) - End {year}', grid=False)
    plt.tight_layout()
    plt.savefig(Path(Config.OUTPUT_DIR) / f'feature_importance_{year}.png')

def plot_history(evals, year):
    # (保持不变)
    if not evals: return
    plt.figure(figsize=(10, 6))
    for k in ['binary_logloss', 'auc']:
        if k in evals.get('train', {}):
            plt.plot(evals['train'][k], label=f'Train {k}')
        if k in evals.get('val', {}):
            plt.plot(evals['val'][k], label=f'Val {k}')
    plt.title(f'Training Metrics - End {year}')
    plt.legend()
    plt.savefig(Path(Config.OUTPUT_DIR) / f'training_history_{year}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_year', type=int, default=2012)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    train_dynamic(args.end_year, args.gpu)