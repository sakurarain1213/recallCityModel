"""
分批训练脚本 (最终修复版)
修复了 init_model 导致的 "Cannot set predictor after freed raw data" 错误
"""
import lightgbm as lgb
import pandas as pd
import gc
import time
import argparse
from pathlib import Path
from src.config import Config

# 极速配置
FAST_PARAMS = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'goss',
    'top_rate': 0.2,
    'other_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 8,
    'max_bin': 63,
    'learning_rate': 0.15,
    'n_estimators': 1000,
    'colsample_bytree': 0.8,
    'min_child_samples': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': 24,
    'verbosity': -1
}

def print_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_data_batch(years, shuffle=True):
    """
    加载指定年份的数据作为一个Batch
    """
    dfs = []
    print_log(f"   📥 Loading parquet for years: {years}")
    for year in years:
        p = Path(Config.PROCESSED_DIR) / f"train_{year}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            # 简单的防御性类型转换
            for c in ['From_City', 'To_City']:
                if c in df.columns: df[c] = df[c].astype('int16')
            if 'Label' in df.columns: df['Label'] = df['Label'].astype('float32')
            dfs.append(df)
    
    if not dfs: return None
    
    # 合并
    df_batch = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Batch内部打乱
    if shuffle:
        print_log(f"   🔀 Shuffling {len(df_batch):,} rows...")
        df_batch = df_batch.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return df_batch

def train_batch_mode(target_end_year, batch_size_years=3):
    total_start = time.time()
    print("="*60)
    print(f"🚀 Batch Training Task: End Year {target_end_year}")
    print(f"📦 Batch Size: {batch_size_years} Years (Sequential Order)")
    print("="*60)

    # 1. 规划 Batches
    all_train_years = list(range(2001, target_end_year - 2))
    val_years = [target_end_year - 2, target_end_year - 1]

    batches = [all_train_years[i:i + batch_size_years] for i in range(0, len(all_train_years), batch_size_years)]
    
    print_log(f"📅 Training Sequence: {batches}")
    print_log(f"📅 Validation Years: {val_years}")

    # 2. 准备验证集 (固定)
    print_log("\n📦 Loading Validation Data (Global)...")
    df_val = load_data_batch(val_years, shuffle=False)
    
    if len(df_val) > 2000000:
        print_log(f"⚡ Sampling Val: {len(df_val):,} -> 2,000,000")
        df_val = df_val.sample(n=2000000, random_state=42).reset_index(drop=True)

    # 特征识别
    excludes = ['Year', 'From_City', 'To_City', 'Label', 'Rank', 'Flow_Count', 'qid']
    feats = [c for c in df_val.columns if c not in excludes and not c.endswith('_orig')]
    cats = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']
    cats = [c for c in cats if c in feats]
    
    print_log(f"✨ Features: {len(feats)} | Categorical: {len(cats)}")

    # 预构建验证集 Dataset
    print_log("🔨 Constructing Validation Dataset...")
    val_ds = lgb.Dataset(
        df_val[feats], 
        label=df_val['Label'], 
        categorical_feature=cats, 
        params=FAST_PARAMS, 
        free_raw_data=False 
    )
    val_ds.construct()
    del df_val
    gc.collect()

    # 3. 循环训练 (Incremental Learning)
    model = None
    
    for i, batch_years in enumerate(batches):
        print("\n" + "-"*40)
        print_log(f"🔄 Processing Batch {i+1}/{len(batches)}: Years {batch_years}")
        print("-"*40)
        
        # 加载 -> 打乱
        df_train = load_data_batch(batch_years, shuffle=True)
        if df_train is None: continue
            
        print_log(f"   Rows: {len(df_train):,} | Memory: {df_train.memory_usage(deep=True).sum()/1024**3:.2f} GB")
        
        # 构建 Dataset
        t_build = time.time()
        
        # 【核心修复点】: 设置 free_raw_data=False
        # LightGBM 增量训练需要原始数据来重新计算残差
        train_ds = lgb.Dataset(
            df_train[feats], 
            label=df_train['Label'], 
            categorical_feature=cats, 
            params=FAST_PARAMS,
            free_raw_data=False  # <--- 必须为 False
        )
        train_ds.construct()
        print_log(f"   Dataset Built: {time.time()-t_build:.1f}s")
        
        # 虽然 train_ds 持有数据引用，但 df_train 变量本身可以删了以减少引用计数
        del df_train
        gc.collect()
        
        # 训练
        print_log("   🔥 Training...")
        try:
            model = lgb.train(
                FAST_PARAMS,
                train_ds,
                num_boost_round=1000, 
                valid_sets=[train_ds, val_ds],
                valid_names=['train', 'val'],
                init_model=model,            # 继承上一轮的模型
                keep_training_booster=True,  # 允许下一轮继续训练
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=True),
                    lgb.log_evaluation(50)
                ]
            )
        except Exception as e:
            print_log(f"❌ Training failed at batch {i+1}: {e}")
            raise e
        finally:
            # 【内存释放】训练完一个Batch后，手动释放 Dataset
            del train_ds
            gc.collect()

    # 4. 保存
    out_path = Path(Config.OUTPUT_DIR) / f'lgb_batch_end_{target_end_year}.txt'
    if model:
        model.save_model(str(out_path))
        print_log(f"\n✅ All Batches Finished! Total time: {(time.time() - total_start)/60:.1f} min")
        print_log(f"💾 Model saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_year', type=int, default=2020, help='Target End Year')
    parser.add_argument('--batch_size', type=int, default=3, help='Years per batch')
    args = parser.parse_args()
    
    train_batch_mode(args.end_year, args.batch_size)