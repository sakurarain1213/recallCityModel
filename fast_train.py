"""
极速训练脚本 (调试增强版)
功能: 包含详细耗时监控、内存监控、显式 Dataset 构建
"""
import lightgbm as lgb
import pandas as pd
import gc
import time
import argparse
import sys
from pathlib import Path
from src.config import Config

# 极速配置
FAST_PARAMS = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'goss',      # 核心提速
    'top_rate': 0.2,
    'other_rate': 0.1,
    'num_leaves': 63,
    'max_depth': 8,
    'max_bin': 63,                # 核心提速
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'colsample_bytree': 0.8,
    'min_child_samples': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': 24,
    'verbosity': -1
}

def print_log(msg):
    """打印带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_data_silent(years, data_dir):
    dfs = []
    total_rows = 0
    t0 = time.time()
    
    # 预分配列表以减少内存碎片
    print_log(f"正在加载 {len(years)} 个年份的数据...")
    
    for year in years:
        p = Path(data_dir) / f"train_{year}.parquet"
        if p.exists():
            # 只读取需要的列? (暂不优化，假设已有内存足够)
            df = pd.read_parquet(p)
            
            # 防御性转换 (确保 float32/int16)
            for c in ['From_City', 'To_City']:
                if c in df.columns and df[c].dtype != 'int16': 
                    df[c] = df[c].astype('int16')
            if 'Label' in df.columns and df['Label'].dtype != 'float32': 
                df['Label'] = df['Label'].astype('float32')
            
            dfs.append(df)
            total_rows += len(df)
            # print(f"  -> Loaded {year} ({len(df):,} rows)")
    
    if not dfs: return None
    
    print_log(f"合并 {len(dfs)} 个 DataFrame...")
    res = pd.concat(dfs, axis=0, ignore_index=True)
    print_log(f"加载完成: {total_rows:,} 行, 耗时 {time.time()-t0:.1f}s")
    return res

def train_fast(target_end_year):
    total_start = time.time()
    print("="*60)
    print(f"🚀 Training Task: End Year {target_end_year}")
    print("="*60)
    
    # 1. 划分数据集
    train_years = list(range(2001, target_end_year - 2))
    val_years = [target_end_year - 2, target_end_year - 1]
    
    # 2. 加载数据
    print_log("📦 Loading Training Data...")
    df_train = load_data_silent(train_years, Config.PROCESSED_DIR)
    
    # 内存监控
    mem_usage = df_train.memory_usage(deep=True).sum() / 1024**3
    print_log(f"📊 Training Data Memory: {mem_usage:.2f} GB")
    
    print_log("📦 Loading Validation Data...")
    df_val = load_data_silent(val_years, Config.PROCESSED_DIR)
    
    # 3. 验证集瘦身
    if len(df_val) > 2000000:
        print_log(f"⚡ Sampling Val: {len(df_val):,} -> 2,000,000")
        df_val = df_val.sample(n=2000000, random_state=42).reset_index(drop=True)

    # 4. 准备特征
    excludes = ['Year', 'From_City', 'To_City', 'Label', 'Rank', 'Flow_Count', 'qid']
    feats = [c for c in df_train.columns if c not in excludes and not c.endswith('_orig')]
    cats = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']
    cats = [c for c in cats if c in feats]
    
    print_log(f"✨ Features: {len(feats)} (Cats: {len(cats)})")

    # 5. 构建 Dataset (显式 Construct)
    print_log("🔨 Init Train Dataset...")
    t_ds = time.time()
    
    # free_raw_data=False: 2亿行数据建议保留在内存中(如果够大)，否则每次迭代重新读取会有开销
    # 但如果内存不够(>100GB占用)，这里会OOM，此时需改为 True
    train_ds = lgb.Dataset(
        df_train[feats], 
        label=df_train['Label'], 
        categorical_feature=cats, 
        params=FAST_PARAMS, 
        free_raw_data=False 
    )
    
    print_log("🔨 Constructing Train Binning (这将花费一些时间)...")
    # 显式调用 construct() 以便我们知道这一步花了多久
    train_ds.construct()
    print_log(f"✅ Train DS Constructed. Time: {time.time()-t_ds:.1f}s")
    
    # 验证集
    print_log("🔨 Init Val Dataset...")
    val_ds = lgb.Dataset(
        df_val[feats], 
        label=df_val['Label'], 
        categorical_feature=cats, 
        reference=train_ds, 
        params=FAST_PARAMS,
        free_raw_data=False
    )
    val_ds.construct() # 显式构建
    
    # 释放 Pandas 内存 (Dataset 如果设置了 free_raw_data=False，它会拷贝/引用数据，这里释放 df_train 安全吗？)
    # 如果 free_raw_data=False，LightGBM 会持有数据引用或副本。
    # 为了保险，先删除 df_train 看看内存变化。
    del df_train, df_val
    gc.collect()
    print_log("🗑️  Pandas DataFrames deleted.")

    # 6. 训练
    print_log("🔥 Start Training Loop...")
    
    def log_callback(env):
        # 强制每10轮打印时间，监测是否卡顿
        if env.iteration % 10 == 0:
            elapsed = time.time() - total_start
            print(f"   [Iter {env.iteration}] {elapsed:.1f}s elapsed")

    model = lgb.train(
        FAST_PARAMS,
        train_ds,
        num_boost_round=FAST_PARAMS['n_estimators'],
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(10),
            log_callback 
        ]
    )

    # 7. 保存
    out_path = Path(Config.OUTPUT_DIR) / f'lgb_fast_end_{target_end_year}.txt'
    model.save_model(str(out_path))
    print_log(f"✅ Finished! Total: {(time.time() - total_start)/60:.1f} min. Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_year', type=int, default=2016, help='Target End Year')
    args = parser.parse_args()
    
    train_fast(args.end_year)