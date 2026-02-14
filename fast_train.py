"""
极速训练脚本 - 读 Parquet -> 喂给 LightGBM
解耦后的训练脚本只负责 I/O 和模型训练
"""
import lightgbm as lgb
import pandas as pd
import gc
from pathlib import Path
from src.config import Config


def train_fast(target_end_year):
    """
    极速训练流程

    Args:
        target_end_year: 目标预测年份，例如 2012
                        训练集: 2001 到 target_end_year-3
                        验证集: target_end_year-2, target_end_year-1
    """
    # 1. 定义时间窗口 (累积训练)
    train_years = list(range(2001, target_end_year - 2))  # 例如 2001-2009
    val_years = [target_end_year - 2, target_end_year - 1]  # 例如 2010, 2011

    print(f"🚀 极速训练 (Read Parquet -> Train)")
    print(f"Train Years: {train_years}")
    print(f"Val Years:   {val_years}")

    # 2. 极速加载
    def load_parquet_years(years):
        files = [Path(Config.PROCESSED_DIR) / f"train_{y}.parquet" for y in years]
        files = [f for f in files if f.exists()]
        if not files:
            return None
        return pd.read_parquet(files)  # pandas支持直接读文件列表

    print("📦 Loading Train Set...")
    df_train = load_parquet_years(train_years)

    print("📦 Loading Val Set...")
    df_val = load_parquet_years(val_years)

    if df_train is None or df_val is None:
        print("❌ 数据加载失败，请先运行 generate_data.py 生成数据!")
        return

    print(f"  Train: {len(df_train):,} rows")
    print(f"  Val:   {len(df_val):,} rows")

    # 3. 准备 Dataset
    # 排除非特征列
    excludes = ['Year', 'From_City', 'To_City', 'Label', 'Rank', 'Flow_Count', 'qid']
    feats = [c for c in df_train.columns if c not in excludes]
    print(f"Features ({len(feats)}): {feats[:10]}...")

    # 类别特征
    cats = ['gender', 'age_group', 'education', 'industry', 'income', 'family', 'is_same_province']
    cats = [c for c in cats if c in feats]

    train_ds = lgb.Dataset(df_train[feats], label=df_train['Label'], categorical_feature=cats)
    val_ds = lgb.Dataset(df_val[feats], label=df_val['Label'], categorical_feature=cats, reference=train_ds)

    del df_train, df_val
    gc.collect()

    # 4. 训练
    print("🔥 Start Training...")
    model = lgb.train(
        Config.LGBM_PARAMS,
        train_ds,
        num_boost_round=Config.LGBM_PARAMS['n_estimators'],
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )

    # 5. 保存
    out_path = Path(Config.OUTPUT_DIR) / f'lgb_fast_end_{target_end_year}.txt'
    model.save_model(str(out_path))
    print(f"✅ Model Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='极速训练脚本')
    parser.add_argument('--end_year', type=int, default=2012, help='目标预测年份')
    args = parser.parse_args()

    train_fast(args.end_year)
