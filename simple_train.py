"""
LightGBM 召回模型训练脚本 (v2 丰富特征版)

架构:
  base_ready/base_YYYY.feather  — 人口统计 + 标签 (精简, ~150MB/年)
  city_pair_cache/city_pairs_YYYY.parquet — 城市对特征 (~50列, ~5MB/年)
  训练时动态 join + 构建交叉特征

运行: cd /data1/wxj/Recall_city_project/ && uv run simple_train.py
依赖: output/base_ready/, data/city_pair_cache/

LightGBM 召回模型训练脚本 (v3 - 工业级流式构建版)
解决亿级别数据 OOM 假死问题，内存占用严格控制在安全范围。
"""

import os
import time
import gc
import psutil
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= 路径配置 =================
BASE_DIR = Path("output/base_ready")
CACHE_DIR = Path("data/city_pair_cache")
OUTPUT_DIR = Path("output")

# ================= 特征定义 =================
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
RATIO_FEATS = [
    'gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio',
    'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio',
    'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio',
    'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio',
    'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio',
    'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio',
    'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio',
    'avg_commute_mins_ratio', 'population_total_ratio',
    'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio',
    'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio',
]
DIFF_FEATS = [
    'gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff',
    'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff',
    'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff',
]
ABS_FEATS = [
    'to_tier', 'to_population_log', 'to_gdp_per_capita',
    'from_tier', 'from_population_log', 'tier_diff',
]
NET_DIST_FEATS = [
    'migrant_stock_from_to',
    'geo_distance', 'dialect_distance', 'is_same_province',
]
CROSS_FEATS = [
    'industry_x_matched_wage_ratio',
    'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff',
    'income_x_gdp_ratio',
    'age_x_housing_ratio',
    'family_x_edu_score_ratio',
]
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# ================= 负样本城市池 =================
HARD_NEG_CITIES_SET = {
    1100, 3100, 4401, 4403,
    1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301, 
    4406, 4419, 5000, 5101, 6101,
    1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206,
    3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701,
    3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501
}

N_HARD_NEG = 30
N_RAND_NEG = 30

def load_cache_for_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)
    df['from_city'] = df['from_city'].astype(np.int32)
    df['to_city'] = df['to_city'].astype(np.int32)
    return df

def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    industry = df['industry'].values if hasattr(df['industry'], 'values') else df['industry'].cat.codes.values
    wage_cols = ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']
    vacancy_cols = ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']
    wage_arr = df[wage_cols].values.astype(np.float32)
    vacancy_arr = df[vacancy_cols].values.astype(np.float32)
    ind_idx = np.clip(industry.astype(int), 0, 3)
    rows = np.arange(len(df))
    df['industry_x_matched_wage_ratio'] = wage_arr[rows, ind_idx]
    df['industry_x_matched_vacancy_ratio'] = vacancy_arr[rows, ind_idx]
    edu = df['education'].values if hasattr(df['education'], 'values') else df['education'].cat.codes.values
    df['education_x_tier_diff'] = edu.astype(np.float32) * df['tier_diff'].values.astype(np.float32)
    inc = df['income'].values if hasattr(df['income'], 'values') else df['income'].cat.codes.values
    df['income_x_gdp_ratio'] = inc.astype(np.float32) * df['gdp_per_capita_ratio'].values
    age = df['age_group'].values if hasattr(df['age_group'], 'values') else df['age_group'].cat.codes.values
    df['age_x_housing_ratio'] = age.astype(np.float32) * df['housing_price_avg_ratio'].values
    fam = df['family'].values if hasattr(df['family'], 'values') else df['family'].cat.codes.values
    df['family_x_edu_score_ratio'] = fam.astype(np.float32) * df['education_score_ratio'].values
    return df

def _vectorized_neg_sample(base: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    label = base['Label'].values
    to_city = base['To_City'].values
    is_pos = (label == 1)
    is_neg = (label != 1)
    is_hard_city = np.isin(to_city, list(HARD_NEG_CITIES_SET))
    is_hard_neg = is_neg & is_hard_city
    is_rand_neg = is_neg & (np.logical_not(is_hard_city))
    keep = is_pos.copy()
    hard_idx = np.where(is_hard_neg)[0]
    if len(hard_idx) > 0:
        qids_hard = base['qid'].values[hard_idx]
        rand_rank = rng.random(len(hard_idx))
        hard_df = pd.DataFrame({'qid': qids_hard, 'rank': rand_rank, 'orig_idx': hard_idx})
        hard_df['grp_rank'] = hard_df.groupby('qid')['rank'].rank(method='first')
        selected_hard = hard_df[hard_df['grp_rank'] <= N_HARD_NEG]['orig_idx'].values
        keep[selected_hard] = True
    rand_idx = np.where(is_rand_neg)[0]
    if len(rand_idx) > 0:
        qids_rand = base['qid'].values[rand_idx]
        rand_rank = rng.random(len(rand_idx))
        rand_df = pd.DataFrame({'qid': qids_rand, 'rank': rand_rank, 'orig_idx': rand_idx})
        rand_df['grp_rank'] = rand_df.groupby('qid')['rank'].rank(method='first')
        selected_rand = rand_df[rand_df['grp_rank'] <= N_RAND_NEG]['orig_idx'].values
        keep[selected_rand] = True
    return base[keep].reset_index(drop=True)

def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float32)
    pos_mask = df['Label'] == 1
    rank = df['Rank']
    weights[pos_mask & (rank <= 3)]  = 50.0
    weights[pos_mask & (rank > 3)  & (rank <= 10)] = 35.0
    weights[pos_mask & (rank > 10) & (rank <= 20)] = 24.0
    return weights

def _process_single_year(year: int, sample_ratio: float, random_seed: int, neg_sample: bool = True):
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        return None
    base = pd.read_feather(base_path)
    if 0.0 < sample_ratio < 1.0:
        unique_qids = base['qid'].unique()
        rng = np.random.default_rng(random_seed)
        n_sample = int(len(unique_qids) * sample_ratio)
        sampled = set(rng.choice(unique_qids, size=n_sample, replace=False))
        base = base[base['qid'].isin(sampled)]
    if neg_sample:
        base = _vectorized_neg_sample(base, random_seed + year)
        gc.collect()
    cache = load_cache_for_year(year)
    base = base.merge(
        cache, left_on=['From_City', 'To_City'], right_on=['from_city', 'to_city'], how='left'
    )
    base.drop(columns=['from_city', 'to_city'], inplace=True)
    del cache
    base = build_cross_features(base)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)
    return base

def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3


def load_data_to_numpy(years: list, sample_ratio: float, random_seed: int, 
                       neg_sample: bool, max_expected_rows: int, split_name: str):
    """
    【核心优化】零硬盘消耗，直接将每年的 Pandas 数据提取为极其紧凑的纯数字 Numpy 矩阵。
    """
    print(f"\n[{split_name}] Pre-allocating Numpy Memory Matrix for max {max_expected_rows:,} rows...")
    t_start = time.time()
    
    # 在内存中预分配连续空间 (极其省内存)
    X = np.empty((max_expected_rows, len(FEATS)), dtype=np.float32)
    y = np.empty(max_expected_rows, dtype=np.int8)
    w = np.empty(max_expected_rows, dtype=np.float32)
    
    current_idx = 0

    for year in tqdm(years, desc=f"  Loading {split_name}", unit="yr"):
        t0 = time.time()
        
        df = _process_single_year(year, sample_ratio, random_seed, neg_sample=neg_sample)
        if df is None:
            continue

        # 将 Category 转为 LightGBM 喜欢的整型数字编码，消除所有 string/object 开销
        for col in CATS:
            if str(df[col].dtype) == 'category':
                df[col] = df[col].cat.codes

        # 强制转换为浮点型以保证 `.values` 输出完美的 C-Contiguous 矩阵
        df_feats = df[FEATS].astype(np.float32)
        weights = calculate_sample_weights(df)
        rows = len(df)

        if current_idx + rows > max_expected_rows:
            raise ValueError(f"Max rows {max_expected_rows} exceeded! Increase `max_expected_rows` in main.")

        # 将数据像搬砖一样填入预先留好的 Numpy 空白矩阵中
        X[current_idx : current_idx + rows] = df_feats.values
        y[current_idx : current_idx + rows] = df['Label'].values.astype(np.int8)
        w[current_idx : current_idx + rows] = weights

        current_idx += rows

        # 瞬间清空 Pandas 的所有痕迹
        del df, df_feats, weights
        gc.collect()
        
        tqdm.write(f"  {year} done ({time.time()-t0:.1f}s) | Rows: {rows:,} | Total Loaded: {current_idx:,} | Python RSS: {_get_mem_gb():.1f} GB")

    print(f"[{split_name}] Memory Matrix Ready! Total {current_idx:,} rows. (Time: {(time.time()-t_start)/60:.1f} min)\n")
    
    # 裁剪掉预分配但没用完的尾部空白部分，返回切片引用
    return X[:current_idx], y[:current_idx], w[:current_idx]


class CheckpointCallback:
    def __init__(self, output_dir, freq=10):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        self.order = 30
        self.before_iteration = False

    def __call__(self, env):
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            ckpt_path = self.output_dir / f"model_iter_{current_round}.txt"
            env.model.save_model(str(ckpt_path))
            print(f"  [Checkpoint] iter {current_round} -> {ckpt_path}")

class TimingCallback:
    def __init__(self, freq=10):
        self.freq = freq
        self.start_time = None
        self.order = 25
        self.before_iteration = False

    def __call__(self, env):
        if self.start_time is None:
            self.start_time = time.time()
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            elapsed = time.time() - self.start_time
            speed = current_round / elapsed * 60
            print(f"  [Timer] iter {current_round} | elapsed {elapsed/60:.1f}min | {speed:.1f} iter/min")


def main():
    TRAIN_YEARS = list(range(2000, 2017))
    VAL_YEARS   = [2017, 2018]
    SAMPLE_RATIO = 0.8  

    print("=" * 60)
    print("LightGBM Recall Training (v4 - 0 Disk & RAM Optimization)")
    print(f"Features: {len(FEATS)} total")
    print(f"RSS at start: {_get_mem_gb():.1f} GB")
    print("=" * 60)

    # 我们根据你之前的报错推算出: 17年训练集大约 3.5 亿行，这里预留 3.8 亿的最高界限。验证集预留 4500 万。
    TRAIN_MAX_ROWS = 380_000_000
    VAL_MAX_ROWS   = 45_000_000

    # 1. 加载数据至 Numpy
    X_train, y_train, w_train = load_data_to_numpy(TRAIN_YEARS, SAMPLE_RATIO, 42, True, TRAIN_MAX_ROWS, "Train")
    X_val, y_val, w_val       = load_data_to_numpy(VAL_YEARS, SAMPLE_RATIO, 42, True, VAL_MAX_ROWS, "Val")

    params = {
        'objective': 'binary',
        'metric': ['auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 255,
        'max_depth': 11,
        'n_estimators': 5000,
        'num_threads': 20,
        'max_bin': 63,
        'feature_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 2000,
        'bin_construct_sample_cnt': 5000000,
        'force_col_wise': True,
        'verbosity': -1,
    }

    # 2. 构建 LightGBM 数据集
    print(f"\nHanding over to LightGBM C++ Core... (Python RSS: {_get_mem_gb():.1f} GB)")
    t0 = time.time()
    
    # 【修改这里】：增加了 feature_name=list(FEATS)
    train_ds = lgb.Dataset(
        X_train, 
        label=y_train, 
        weight=w_train, 
        feature_name=list(FEATS), 
        categorical_feature=CATS, 
        params=params, 
        free_raw_data=True
    )
    
    # 主动解除 Python 的引用，为 LightGBM C++ 腾出空间
    del X_train, y_train, w_train
    gc.collect()

    print("Forcing C++ Train Dataset construction (This will take a few minutes)...")
    train_ds.construct()
    print(f"  Train Dataset construct done in {time.time()-t0:.1f}s | Python RSS drops to: {_get_mem_gb():.1f} GB")
    
    t1 = time.time()

    val_ds = lgb.Dataset(
        X_val, 
        label=y_val, 
        reference=train_ds, 
        feature_name=list(FEATS), 
        categorical_feature=CATS, 
        params=params, 
        free_raw_data=True
    )

    del X_val, y_val, w_val
    gc.collect()

    print("Forcing C++ Val Dataset construction...")
    val_ds.construct()
    print(f"  Val Dataset construct done in {time.time()-t1:.1f}s | Python RSS drops to: {_get_mem_gb():.1f} GB")

    # 3. 训练
    print(f"\nTraining started...")
    start_time = time.time()

    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
    timing_cb = TimingCallback(freq=10)
    
    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(10),
            timing_cb,
            ckpt_cb,
        ],
    )
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # 4. 保存模型
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'model_v2.txt'
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\nTop 20 Features by Gain:")
    print(imp.head(20).to_string(index=False))

if __name__ == '__main__':
    main()