
"""
分批训练脚本 (Checkpoint & 提速优化版)
功能: 
1. 支持每 N 轮保存 Checkpoint
2. 使用 Mini-Validation Set 加速训练过程中的评估
3. 移除训练集实时评估，大幅提速
"""
import lightgbm as lgb
import pandas as pd
import gc
import time
import argparse
import os
from pathlib import Path
from src.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

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
        p = Path(Config.PROCESSED_DIR) / f"processed_{year}.parquet"
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

# === 新增：Checkpoint 回调函数 ===
def save_checkpoint_callback(save_freq, output_dir, year_prefix):
    """
    每 save_freq 轮保存一次模型
    """
    def callback(env):
        # env.iteration 从 0 开始
        iteration = env.iteration + 1
        if iteration % save_freq == 0:
            # 构造文件名: checkpoints/model_2010_round_50.txt
            ckpt_dir = Path(output_dir) / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            path = ckpt_dir / f"model_{year_prefix}_round_{iteration}.txt"
            env.model.save_model(str(path))
            print_log(f"   💾 Checkpoint saved: {path.name}")
    return callback

def train_batch_mode(target_end_year, batch_size_years=3, checkpoint_freq=50):
    total_start = time.time()
    print("="*60)
    print(f"🚀 Batch Training Task: End Year {target_end_year}")
    print(f"📦 Batch Size: {batch_size_years} Years")
    print(f"⏱️ Checkpoint Frequency: Every {checkpoint_freq} rounds")
    print("="*60)

    # 1. 规划 Batches
    all_train_years = list(range(2001, target_end_year - 2))
    val_years = [target_end_year - 2, target_end_year - 1]

    batches = [all_train_years[i:i + batch_size_years] for i in range(0, len(all_train_years), batch_size_years)]
    
    print_log(f"📅 Training Sequence: {batches}")
    print_log(f"📅 Validation Years: {val_years}")

    # 2. 准备验证集
    print_log("\n📦 Loading Validation Data (Global)...")
    df_val = load_data_batch(val_years, shuffle=False)
    
    # 【提速优化核心】
    # 构造一个极小的验证集 (20万) 专门用于 Early Stopping 和 实时打印
    # 原始 200万 太大了，每轮评估太慢
    WATCH_SIZE = 200000 
    
    if len(df_val) > WATCH_SIZE:
        print_log(f"⚡ Creating Mini-Validation Set for Speed: {WATCH_SIZE:,} rows")
        # 分离出 mini set
        df_val_watch = df_val.sample(n=WATCH_SIZE, random_state=42).reset_index(drop=True)
        # 释放原始大表 (如果内存紧张) - 或者保留用于最后 Full Evaluate (这里为了省内存先释放)
        del df_val
        gc.collect()
    else:
        df_val_watch = df_val
        del df_val

    # 特征识别
    excludes = ['Year', 'From_City', 'To_City', 'Label', 'Rank', 'Flow_Count', 'qid']
    feats = [c for c in df_val_watch.columns if c not in excludes and not c.endswith('_orig')]
    cats = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']
    cats = [c for c in cats if c in feats]
    
    print_log(f"✨ Features: {len(feats)} | Categorical: {len(cats)}")

    # 构建 Mini 验证集 Dataset
    print_log("🔨 Constructing Watch Dataset...")
    val_ds_watch = lgb.Dataset(
        df_val_watch[feats], 
        label=df_val_watch['Label'], 
        categorical_feature=cats, 
        params=FAST_PARAMS, 
        free_raw_data=False 
    )
    val_ds_watch.construct()
    
    # 释放 Pandas 对象
    del df_val_watch
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

        # ======================= 【特征对齐】 =======================
        # 🛡️ 确保训练集包含所有验证集有的特征
        # 如果早期年份缺少 5y_avg 等特征，手动补上并填 -1
        missing_cols = [c for c in feats if c not in df_train.columns]
        if missing_cols:
            print_log(f"   ⚠️ Aligning features: Filling {len(missing_cols)} missing cols (e.g., {missing_cols[0]}) with -1")
            for c in missing_cols:
                df_train[c] = -1.0
                df_train[c] = df_train[c].astype('float32')
        # ======================= 【特征对齐结束】 =======================

        print_log(f"   Rows: {len(df_train):,} | Memory: {df_train.memory_usage(deep=True).sum()/1024**3:.2f} GB")
        
        # 构建 Dataset
        t_build = time.time()
        train_ds = lgb.Dataset(
            df_train[feats], 
            label=df_train['Label'], 
            categorical_feature=cats, 
            params=FAST_PARAMS,
            free_raw_data=False 
        )
        train_ds.construct()
        print_log(f"   Dataset Built: {time.time()-t_build:.1f}s")
        
        del df_train
        gc.collect()
        
        # 训练
        print_log("   🔥 Training...")
        try:
            # 回调列表
            callbacks_list = [
                lgb.early_stopping(stopping_rounds=20, verbose=True),
                lgb.log_evaluation(50), # 减少打印频率到 50
                # 添加 Checkpoint 回调
                save_checkpoint_callback(checkpoint_freq, Config.OUTPUT_DIR, target_end_year)
            ]

            model = lgb.train(
                FAST_PARAMS,
                train_ds,
                num_boost_round=1000, 
                # 【提速关键】valid_sets 只放 mini 验证集，且不放训练集
                valid_sets=[val_ds_watch], 
                valid_names=['val_mini'], # 改个名字区分
                init_model=model,            
                keep_training_booster=True,  
                callbacks=callbacks_list
            )
        except Exception as e:
            print_log(f"❌ Training failed at batch {i+1}: {e}")
            raise e
        finally:
            del train_ds
            gc.collect()

    # 4. 保存最终模型
    out_path = Path(Config.OUTPUT_DIR) / f'lgb_batch_end_{target_end_year}.txt'
    if model:
        model.save_model(str(out_path))
        print_log(f"\n✅ All Batches Finished! Total time: {(time.time() - total_start)/60:.1f} min")
        print_log(f"💾 Model saved to: {out_path}")

        # 绘图
        print("\n" + "="*40)
        print("📊 Feature Importance (Gain)")
        print("="*40)
        
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        fi_df = fi_df.sort_values(by='importance', ascending=False)
        
        print(fi_df.head(20).to_string(index=False))
        
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=fi_df.head(30))
        plt.title(f'LightGBM Feature Importance (Gain) - End Year {target_end_year}')
        plt.tight_layout()
        plt.savefig(Path(Config.OUTPUT_DIR) / f'feature_importance_{target_end_year}.png')
        print(f"\n🖼️ Feature importance plot saved to output/feature_importance_{target_end_year}.png")
        
        geo_rank = fi_df[fi_df['feature'] == 'geo_distance'].index
        if len(geo_rank) > 0 and geo_rank[0] > 5:
            print("\n⚠️ 警告: 地理距离 (geo_distance) 权重过低！请检查 city_edges 是否正确 Merge！")

        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_year', type=int, default=2020, help='Target End Year')
    parser.add_argument('--batch_size', type=int, default=3, help='Years per batch')
    parser.add_argument('--ckpt_freq', type=int, default=50, help='Checkpoint frequency')
    args = parser.parse_args()
    
    train_batch_mode(args.end_year, args.batch_size, args.ckpt_freq)