"""
快速训练脚本 (最终修复版) - 修复 evals_result 报错

修复内容:
1. 修复 'Booster' object has no attribute 'evals_result_' 报错
2. 正确使用 lgb.train 的 evals_result 参数来捕获 loss
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

from src.config import Config
from src.city_data import CityDataLoader
from src.data_loader_v2 import load_raw_data_fast
from src.feature_eng import parse_type_id, optimize_dtypes
from src.historical_features import add_historical_features

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 缓存目录
CACHE_DIR = Path(Config.OUTPUT_DIR) / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_year_data(year, global_features, hard_candidates, neg_sample_rate, is_training=True):
    """
    获取一年的数据：
    1. 优先检查硬盘缓存 (output/cache/train_20xx.parquet)
    2. 如果没有缓存，则从 DuckDB 加载并处理，然后保存缓存
    """
    cache_file = CACHE_DIR / f"processed_{year}.parquet"
    
    # A. 命中缓存，直接读取
    if cache_file.exists():
        # print(f"  [缓存命中] 从硬盘加载 {year} 年数据...")
        try:
            df = pd.read_parquet(cache_file)
            if 'Label' in df.columns:
                return df
        except Exception as e:
            print(f"  [缓存损坏] 读取失败: {e}，将重新生成...")
            try: cache_file.unlink() 
            except: pass

    # B. 无缓存，重新生成
    print(f"  [生成数据] 处理 {year} 年原始数据...")
    
    # 1. 加载宽表
    df = load_raw_data_fast(
        Config.DB_PATH,
        year,
        hard_candidates,
        neg_sample_rate=neg_sample_rate
    )
    if df.empty: return None

    # 2. Merge 全局特征
    df = df.merge(global_features, on=['Year', 'From_City', 'To_City'], how='left')

    # 3. 解析 Type_ID
    df, _ = parse_type_id(df, verbose=False)

    # 4. 添加历史特征
    df = add_historical_features(
        df,
        year,
        Path(Config.OUTPUT_DIR) / 'processed_data',
        verbose=False,
        training_mode=is_training
    )

    # 5. 优化类型
    df = optimize_dtypes(df)

    # 6. 保存缓存到硬盘
    print(f"  [写入缓存] 保存至 {cache_file} ...")
    df.to_parquet(cache_file, index=False)
    
    return df

def fast_train(end_year=None, use_gpu=False):
    # 0. 参数配置 (建议保持 10)
    CURRENT_NEG_RATE = 10
    
    if end_year is None:
        train_years = list(range(Config.TRAIN_START_YEAR, Config.TRAIN_END_YEAR + 1))
        val_years = Config.VAL_YEARS
    else:
        val_year = end_year - 1
        train_years = list(range(Config.TRAIN_START_YEAR, val_year))
        val_years = [val_year]

    print(f"训练年份序列: {train_years}")
    print(f"验证年份: {val_years}")
    print(f"缓存目录: {CACHE_DIR}")

    # 1. 准备基础数据
    print("\nStep 1: 加载基础配置...")
    city_loader = CityDataLoader(Config.DATA_DIR)
    city_loader.load_all()
    
    city_info_2010 = city_loader.get_city_info_for_year(2010)
    if city_info_2010 is None:
        avail = sorted(city_loader.city_info.keys())
        city_info_2010 = city_loader.get_city_info_for_year(avail[0])
        
    tier_cities = city_info_2010[city_info_2010['tier'] <= 2].index.tolist()
    core_cities = [
        1100, 1200, 1300, 1400, 1500, 2100, 2200, 2300, 3100, 3200, 
        3300, 3400, 3500, 3600, 3700, 4100, 4200, 4300, 4400, 4500, 
        4600, 5000, 5100, 5200, 5300, 5400, 6100, 6200, 6300, 6400, 6500
    ]
    hard_candidates = list(set([int(c) for c in tier_cities] + core_cities))

    # 2. 加载全局特征表
    print("\nStep 2: 加载全局特征表...")
    global_features_path = Path(Config.OUTPUT_DIR) / 'global_city_features.parquet'
    if not global_features_path.exists():
        print("❌ 未找到 output/global_city_features.parquet")
        return
    global_features = pd.read_parquet(global_features_path)
    global_features['From_City'] = global_features['From_City'].astype('int16')
    global_features['To_City'] = global_features['To_City'].astype('int16')

    # 3. 准备验证集
    print(f"\nStep 3: 准备验证集 {val_years}...")
    val_dfs = []
    for yr in val_years:
        df = get_year_data(yr, global_features, hard_candidates, CURRENT_NEG_RATE, is_training=False)
        if df is not None:
            val_dfs.append(df)
            
    if not val_dfs:
        print("❌ 验证集为空！")
        return

    df_val = pd.concat(val_dfs, axis=0, ignore_index=True)
    exclude_cols = ['Year', 'qid', 'Type_ID_orig', 'From_City_orig', 'To_City', 'Flow_Count', 'Rank', 'Label']
    feature_cols = [c for c in df_val.columns if c not in exclude_cols]
    
    print(f"验证集大小: {len(df_val):,} 行")
    print(f"特征数量: {len(feature_cols)}")
    
    X_val = df_val[feature_cols]
    y_val = df_val['Label']
    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False).construct()
    
    del df_val, X_val, y_val
    gc.collect()

    # 4. 增量训练循环
    print("\nStep 4: 开始增量训练...")
    
    params = Config.LGBM_PARAMS_GPU if use_gpu else Config.LGBM_PARAMS
    params.update({
        'n_estimators': 100,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'verbosity': -1,
        'keep_training_booster': True 
    })
    
    model = None
    loss_history = []
    evals_result = {}  # 在循环外部定义，用于累积所有年份的评估结果

    pbar = tqdm(train_years, desc="Training Years")

    for year in pbar:
        pbar.set_description(f"Training {year}")

        # A. 加载当年数据
        df_train = get_year_data(year, global_features, hard_candidates, CURRENT_NEG_RATE, is_training=True)

        if df_train is None:
            continue

        # B. 构造 Dataset
        train_ds = lgb.Dataset(
            df_train[feature_cols],
            label=df_train['Label'],
            free_raw_data=True
        )

        # C. 训练 (使用回调记录评估结果)
        model = lgb.train(
            params,
            train_set=train_ds,
            valid_sets=[val_data],
            valid_names=['val'],
            init_model=model,
            keep_training_booster=True,
            callbacks=[
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result)  # 记录到 evals_result 字典
            ]
        )

        # 从字典里读取验证集损失
        if 'val' in evals_result and 'binary_logloss' in evals_result['val']:
            current_loss = evals_result['val']['binary_logloss'][-1]
            loss_history.append(current_loss)
            pbar.set_postfix({'val_loss': f"{current_loss:.4f}"})

        del df_train, train_ds
        gc.collect()

    # 5. 保存最终模型
    print("\nStep 5: 保存最终模型...")
    output_path = Path(Config.OUTPUT_DIR) / 'fast_model.txt'
    model.save_model(str(output_path))
    print(f"✓ 模型已保存至: {output_path}")
    
    # 绘制 Loss 曲线
    if loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(train_years, loss_history, marker='o')
        plt.title(f'Incremental Training Loss (End Year: {train_years[-1]})')
        plt.xlabel('Year')
        plt.ylabel('Validation LogLoss')
        plt.grid(True)
        plt.savefig(Path(Config.OUTPUT_DIR) / 'fast_training_history.png')
        print("✓ 训练历史图表已保存")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_year', type=int, default=None)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    fast_train(end_year=args.end_year, use_gpu=args.gpu)