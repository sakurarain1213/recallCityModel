
"""
分批训练脚本 (Checkpoint & 提速优化版)
功能: 
1. 支持每 N 轮保存 Checkpoint
2. 使用 Mini-Validation Set 加速训练过程中的评估
3. 移除训练集实时评估，大幅提速
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc
import time
import argparse
import os
from pathlib import Path
from src.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

def print_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def calculate_sample_weights(df, end_year, decay_rate=0.9):
    """
    计算样本权重：时间衰减 + 差异化样本权重

    参数:
    - df: 训练数据 DataFrame (必须包含 Rank, Label, Year 列)
    - end_year: 目标预测年份
    - decay_rate: 时间衰减率 (默认 0.9，即每年衰减 10%)

    返回:
    - weights: 样本权重数组
    """
    # 1. 时间衰减权重
    year_diff = end_year - df['Year']
    time_weights = decay_rate ** year_diff

    # 2. 样本类型权重
    base_weights = np.ones(len(df), dtype=np.float32)

    # 正样本 (Rank 1-10, Label=1)
    pos_mask = (df['Label'] == 1.0)
    if pos_mask.any():
        rank = df['Rank'].copy()
        # 头部保护 (Rank 1-3)
        top3_mask = pos_mask & (rank <= 3)
        base_weights[top3_mask] = 20.0
        # 其他正样本 (Rank 4-10)
        other_pos_mask = pos_mask & (rank > 3) & (rank <= 10)
        base_weights[other_pos_mask] = 10.0

    # 困难负样本 (Rank 11-20, Label=0) - Hard Negative Mining
    hard_neg_mask = (df['Label'] == 0.0) & (df['Rank'] > 10) & (df['Rank'] <= 20)
    base_weights[hard_neg_mask] = 5.0

    # 普通负样本 (Rank > 20 或 Rank 为 97/98/99) 权重保持为 1.0

    # 3. 组合权重
    final_weights = base_weights * time_weights

    return final_weights

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

    # 生成 qid (Query ID) 如果不存在
    # qid 用于按 Query 完整采样,确保验证集的 Recall 指标准确
    if 'qid' not in df_batch.columns:
        print_log("   🆔 Generating qid (Query ID) for batch...")
        # parquet 文件中 Type_ID 被转为 Type_Hash，使用它来区分不同类型
        if 'Type_Hash' in df_batch.columns:
            df_batch['qid'] = (
                df_batch['Year'].astype('int64') * 100000 +
                df_batch['Type_Hash'].astype('int64') % 1000 +  # 取模避免数值过大
                df_batch['From_City'].astype('int64')
            ).astype('int64')
        else:
            # 降级方案：只用 Year + From_City
            df_batch['qid'] = (
                df_batch['Year'].astype('int64') * 100000 +
                df_batch['From_City'].astype('int64')
            ).astype('int64')

    # 确保 Rank 列存在 (用于计算权重)
    if 'Rank' not in df_batch.columns:
        print_log("   ⚠️ Warning: Rank column not found, weights will be uniform")

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

def train_batch_mode(target_end_year, batch_size_years=5, checkpoint_freq=50):
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

    # 【精度优化】按 Query 完整采样,不随机拆分行
    # 构造一个极小的验证集 (20万行) 专门用于 Early Stopping 和 实时打印
    # 关键: 按 qid 分组,确保一个 Query 的所有样本都在验证集中
    WATCH_SIZE = 200000

    if len(df_val) > WATCH_SIZE:
        print_log(f"⚡ Creating Mini-Validation Set for Speed: ~{WATCH_SIZE:,} rows")
        print_log(f"   📊 Sampling by complete queries (qid) to preserve Recall metric...")

        # 计算需要的 query 数量
        avg_samples_per_query = len(df_val) / df_val['qid'].nunique()
        n_queries_needed = int(WATCH_SIZE / avg_samples_per_query)

        # 随机采样完整的 query
        unique_qids = df_val['qid'].unique()
        sampled_qids = pd.Series(unique_qids).sample(n=n_queries_needed, random_state=42).values

        # 保留这些 query 的所有样本
        df_val_watch = df_val[df_val['qid'].isin(sampled_qids)].reset_index(drop=True)
        print_log(f"   ✅ Sampled {len(sampled_qids):,} queries -> {len(df_val_watch):,} rows")

        # 释放原始大表
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
        params=Config.LGBM_PARAMS, 
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

        # ======================= 【样本权重计算】 =======================
        # 计算样本权重：时间衰减 + 差异化权重
        print_log("   🎯 Calculating sample weights (Time Decay + Reweighting)...")
        weights = calculate_sample_weights(df_train, target_end_year, decay_rate=0.9)

        # 打印权重统计
        print_log(f"   📊 Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        # ======================= 【权重计算结束】 =======================

        # 构建 Dataset
        t_build = time.time()
        train_ds = lgb.Dataset(
            df_train[feats],
            label=df_train['Label'],
            weight=weights,  # 应用样本权重
            categorical_feature=cats,
            params=Config.LGBM_PARAMS,
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
                # 核心早停参数 修改后：增加耐心到 50 或 100，或者直接注释掉，让它跑满 1000 轮
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(50), # 减少打印频率到 50
                # 添加 Checkpoint 回调
                save_checkpoint_callback(checkpoint_freq, Config.OUTPUT_DIR, target_end_year)
            ]

            model = lgb.train(
                Config.LGBM_PARAMS,
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
    parser.add_argument('--batch_size', type=int, default=5, help='Years per batch')
    parser.add_argument('--ckpt_freq', type=int, default=50, help='Checkpoint frequency')
    args = parser.parse_args()
    
    train_batch_mode(args.end_year, args.batch_size, args.ckpt_freq)