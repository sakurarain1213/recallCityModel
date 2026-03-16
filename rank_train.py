"""
LightGBM 召回模型训练脚本 (v8 - rank_xendcg 全局排序优化版)

核心改动 (v7 → v8):
1. 目标函数: lambdarank → rank_xendcg (Cross-Entropy NDCG)。
   - lambdarank 通过 pairwise 交换计算梯度，天然偏执于头部位置；
   - rank_xendcg 基于 softmax 交叉熵近似 NDCG，对整个列表的全局排序更均衡，
     既能保住 Top-5 的精准度，又能兼顾 6-20 名的顺序质量。
2. lambdarank_truncation_level = 20：
   - 控制优化时关注列表的前多少个位置。GT 最多 20 个正样本，
     优化超过 20 的位置毫无意义（20 名之后全是负样本），设为 20 刚好覆盖全部正样本。
3. 负采样扩容: 50+50 → 80+80。
   - 推理时候选集 336 城市，训练时只见 100 个负样本会导致"中间难度"城市在推理时被错误排高；
     扩容到 160 个负样本，让模型见到更多边界样本，提升 Recall@20。
4. TRAIN_SAMPLE_RATIO: 0.7 → 0.6，配合负采样扩容控制总内存在 200GB 安全线内。

不变项: 标签映射 Relevance = 21 - Rank (绝对映射方案 A)。
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

# ================= 负样本池及参数 =================
HARD_NEG_CITIES_SET = {
    1100, 3100, 4401, 4403,
    1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301, 
    4406, 4419, 5000, 5101, 6101,
    1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206,
    3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701,
    3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501
}
# v8: 扩容负采样 50→80，让模型见到更多"中间难度"的负样本城市，
# 缩小训练时(160负样本)与推理时(316负样本)的分布差距，提升 Recall@20。
N_HARD_NEG = 80
N_RAND_NEG = 80

# ================= 回调函数恢复 =================
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


def load_cache_for_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
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
    is_pos = (base['Relevance'] > 0).values
    is_neg = ~is_pos
    to_city = base['To_City'].values
    is_hard_city = np.isin(to_city, list(HARD_NEG_CITIES_SET))
    
    is_hard_neg = is_neg & is_hard_city
    is_rand_neg = is_neg & (~is_hard_city)
    keep = is_pos.copy()  # 保留所有正样本
    
    hard_idx = np.where(is_hard_neg)[0]
    if len(hard_idx) > 0:
        qids_hard = base['qid'].values[hard_idx]
        hard_df = pd.DataFrame({'qid': qids_hard, 'rank': rng.random(len(hard_idx)), 'orig_idx': hard_idx})
        hard_df['grp_rank'] = hard_df.groupby('qid')['rank'].rank(method='first')
        keep[hard_df[hard_df['grp_rank'] <= N_HARD_NEG]['orig_idx'].values] = True
        
    rand_idx = np.where(is_rand_neg)[0]
    if len(rand_idx) > 0:
        qids_rand = base['qid'].values[rand_idx]
        rand_df = pd.DataFrame({'qid': qids_rand, 'rank': rng.random(len(rand_idx)), 'orig_idx': rand_idx})
        rand_df['grp_rank'] = rand_df.groupby('qid')['rank'].rank(method='first')
        keep[rand_df[rand_df['grp_rank'] <= N_RAND_NEG]['orig_idx'].values] = True
        
    return base[keep].reset_index(drop=True)

def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3

def _process_single_year(year: int, sample_ratio: float, random_seed: int, is_train: bool):
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        return None, None
    base = pd.read_feather(base_path)
    
    base['Relevance'] = np.where((base['Rank'] > 0) & (base['Rank'] <= 20), 21 - base['Rank'], 0).astype(np.int8)
    
    valid_qids_mask = base.groupby('qid')['Relevance'].transform('max') > 0
    base = base[valid_qids_mask].copy()

    if 0.0 < sample_ratio < 1.0:
        unique_qids = base['qid'].unique()
        rng = np.random.default_rng(random_seed)
        n_sample = int(len(unique_qids) * sample_ratio)
        sampled = set(rng.choice(unique_qids, size=n_sample, replace=False))
        base = base[base['qid'].isin(sampled)]
        
    if is_train:
        base = _vectorized_neg_sample(base, random_seed + year)
        gc.collect()
        
    group_counts = base.groupby('qid', sort=False).size().values
        
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
            
    return base, group_counts


def load_data_to_numpy_ltr(years: list, sample_ratio: float, random_seed: int, 
                           is_train: bool, max_expected_rows: int, split_name: str):
    print(f"\n[{split_name}] Pre-allocating Memory Matrix (Max {max_expected_rows:,} rows)...")
    X = np.empty((max_expected_rows, len(FEATS)), dtype=np.float32)
    y = np.empty(max_expected_rows, dtype=np.int8)
    all_groups = []
    current_idx = 0

    for year in tqdm(years, desc=f"  Loading {split_name}", unit="yr"):
        t0 = time.time()
        df, group_counts = _process_single_year(year, sample_ratio, random_seed, is_train)
        if df is None:
            continue

        for col in CATS:
            if str(df[col].dtype) == 'category':
                df[col] = df[col].cat.codes

        df_feats = df[FEATS].astype(np.float32)
        rows = len(df)
        if current_idx + rows > max_expected_rows:
            raise ValueError(f"Max rows exceeded! Increase `max_expected_rows`.")

        X[current_idx : current_idx + rows] = df_feats.values
        y[current_idx : current_idx + rows] = df['Relevance'].values 
        all_groups.extend(group_counts) 

        current_idx += rows
        del df, df_feats
        gc.collect()
        tqdm.write(f"  {year} | Rows: {rows:,} | Total Loaded: {current_idx:,} | Python RSS: {_get_mem_gb():.1f} GB")

    return X[:current_idx], y[:current_idx], np.array(all_groups, dtype=np.int32)


def main():
    TRAIN_YEARS = list(range(2000, 2017))
    VAL_YEARS   = [2017, 2018]
    
    # v8: 负采样扩容到 80+80，配合 TRAIN_SAMPLE_RATIO 降至 0.6 控制总内存。
    # 估算: 17年 × 329K QID × 0.6 × 170行/QID ≈ 5.7亿行 × 60特征 × 4B ≈ 137GB，安全。
    TRAIN_SAMPLE_RATIO = 0.6
    VAL_SAMPLE_RATIO   = 0.2

    print("=" * 60)
    print("LightGBM Recall (v8 - rank_xendcg 全局排序优化版)")
    print(f"RSS at start: {_get_mem_gb():.1f} GB")
    print("=" * 60)

    # v8: 负采样扩容后每 QID 约 170 行，上调预留空间至 6 亿
    TRAIN_MAX_ROWS = 600_000_000
    VAL_MAX_ROWS   = 60_000_000

    X_train, y_train, group_train = load_data_to_numpy_ltr(TRAIN_YEARS, TRAIN_SAMPLE_RATIO, 42, is_train=True, max_expected_rows=TRAIN_MAX_ROWS, split_name="Train")
    X_val, y_val, group_val       = load_data_to_numpy_ltr(VAL_YEARS, VAL_SAMPLE_RATIO, 42, is_train=False, max_expected_rows=VAL_MAX_ROWS, split_name="Val")

    params = {
        # v8 核心改动: lambdarank → rank_xendcg
        # rank_xendcg 基于 softmax 交叉熵近似 NDCG，相比 lambdarank 的 pairwise 交换梯度，
        # 它对整个列表做全局概率建模，不会过度偏执于头部，能更均衡地优化 1-20 名的排序质量。
        'objective': 'rank_xendcg',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        # v8: 限制优化关注前 20 个位置。GT 最多 20 个正样本，
        # 超过 20 的位置全是负样本，优化它们的排序没有收益。
        'lambdarank_truncation_level': 20,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 9,
        'n_estimators': 5000,
        'num_threads': 20,
        'max_bin': 127,
        'feature_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 500,
        'force_col_wise': True,
        'verbosity': -1,
    }

    print(f"\nHanding over to C++... (Python RSS: {_get_mem_gb():.1f} GB)")
    train_ds = lgb.Dataset(X_train, label=y_train, group=group_train, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
    del X_train, y_train, group_train; gc.collect()
    train_ds.construct()
    
    val_ds = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_ds, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
    del X_val, y_val, group_val; gc.collect()
    val_ds.construct()

    print(f"\nTraining started... RSS: {_get_mem_gb():.1f} GB")
    
    # 决断 5：恢复 Checkpoint 与 Timing
    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
    timing_cb = TimingCallback(freq=10)

    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True), # 决断 4：50轮早停防过拟合
            lgb.log_evaluation(10),
            timing_cb,
            ckpt_cb,
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_DIR / 'model_v8_xendcg.txt'))
    
    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')}).sort_values('importance', ascending=False)
    print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")

if __name__ == '__main__':
    main()



'''

老版本 采用  lambdarank  只关注top5的顺序的准确性
"""
LightGBM 召回模型训练脚本 (v7 - 内存安全与极致排序终极版)
核心定调：
1. 绝对映射：Relevance = 21 - Rank。
2. 负采样甜点：50 Hard + 50 Random (正负比约 1:10，极致平衡内存与信息增益)。
3. 内存安全锁：Train(0.7), Val(0.2)，确保 Mac Mini M4 200GB 内存绝对不爆。
4. 锐利评估：NDCG@[5, 10, 15]，早期停止 50 轮。
5. 监控护航：恢复 Checkpoint 与 Timing 机制。
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
# ================= 负样本池及参数 =================
HARD_NEG_CITIES_SET = {
    1100, 3100, 4401, 4403,
    1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301,
    4406, 4419, 5000, 5101, 6101,
    1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206,
    3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701,
    3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501
}
# 决断 2：采用 1:10 黄金法则比例
N_HARD_NEG = 50
N_RAND_NEG = 50
# ================= 回调函数恢复 =================
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
def load_cache_for_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
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
    is_pos = (base['Relevance'] > 0).values
    is_neg = ~is_pos
    to_city = base['To_City'].values
    is_hard_city = np.isin(to_city, list(HARD_NEG_CITIES_SET))
    is_hard_neg = is_neg & is_hard_city
    is_rand_neg = is_neg & (~is_hard_city)
    keep = is_pos.copy()  # 保留所有正样本
    hard_idx = np.where(is_hard_neg)[0]
    if len(hard_idx) > 0:
        qids_hard = base['qid'].values[hard_idx]
        hard_df = pd.DataFrame({'qid': qids_hard, 'rank': rng.random(len(hard_idx)), 'orig_idx': hard_idx})
        hard_df['grp_rank'] = hard_df.groupby('qid')['rank'].rank(method='first')
        keep[hard_df[hard_df['grp_rank'] <= N_HARD_NEG]['orig_idx'].values] = True
    rand_idx = np.where(is_rand_neg)[0]
    if len(rand_idx) > 0:
        qids_rand = base['qid'].values[rand_idx]
        rand_df = pd.DataFrame({'qid': qids_rand, 'rank': rng.random(len(rand_idx)), 'orig_idx': rand_idx})
        rand_df['grp_rank'] = rand_df.groupby('qid')['rank'].rank(method='first')
        keep[rand_df[rand_df['grp_rank'] <= N_RAND_NEG]['orig_idx'].values] = True
    return base[keep].reset_index(drop=True)
def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3
def _process_single_year(year: int, sample_ratio: float, random_seed: int, is_train: bool):
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        return None, None
    base = pd.read_feather(base_path)
    base['Relevance'] = np.where((base['Rank'] > 0) & (base['Rank'] <= 20), 21 - base['Rank'], 0).astype(np.int8)
    valid_qids_mask = base.groupby('qid')['Relevance'].transform('max') > 0
    base = base[valid_qids_mask].copy()
    if 0.0 < sample_ratio < 1.0:
        unique_qids = base['qid'].unique()
        rng = np.random.default_rng(random_seed)
        n_sample = int(len(unique_qids) * sample_ratio)
        sampled = set(rng.choice(unique_qids, size=n_sample, replace=False))
        base = base[base['qid'].isin(sampled)]
    if is_train:
        base = _vectorized_neg_sample(base, random_seed + year)
        gc.collect()
    group_counts = base.groupby('qid', sort=False).size().values
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
    return base, group_counts
def load_data_to_numpy_ltr(years: list, sample_ratio: float, random_seed: int,
                           is_train: bool, max_expected_rows: int, split_name: str):
    print(f"\n[{split_name}] Pre-allocating Memory Matrix (Max {max_expected_rows:,} rows)...")
    X = np.empty((max_expected_rows, len(FEATS)), dtype=np.float32)
    y = np.empty(max_expected_rows, dtype=np.int8)
    all_groups = []
    current_idx = 0
    for year in tqdm(years, desc=f"  Loading {split_name}", unit="yr"):
        t0 = time.time()
        df, group_counts = _process_single_year(year, sample_ratio, random_seed, is_train)
        if df is None:
            continue
        for col in CATS:
            if str(df[col].dtype) == 'category':
                df[col] = df[col].cat.codes
        df_feats = df[FEATS].astype(np.float32)
        rows = len(df)
        if current_idx + rows > max_expected_rows:
            raise ValueError(f"Max rows exceeded! Increase `max_expected_rows`.")
        X[current_idx : current_idx + rows] = df_feats.values
        y[current_idx : current_idx + rows] = df['Relevance'].values
        all_groups.extend(group_counts)
        current_idx += rows
        del df, df_feats
        gc.collect()
        tqdm.write(f"  {year} | Rows: {rows:,} | Total Loaded: {current_idx:,} | Python RSS: {_get_mem_gb():.1f} GB")
    return X[:current_idx], y[:current_idx], np.array(all_groups, dtype=np.int32)
def main():
    TRAIN_YEARS = list(range(2000, 2017))
    VAL_YEARS   = [2017, 2018]
    # 决断 1：内存保安，下调采样率
    TRAIN_SAMPLE_RATIO = 0.7  
    VAL_SAMPLE_RATIO   = 0.2  
    print("=" * 60)
    print("LightGBM Recall (v7 - LambdaRank 黄金比例闭环版)")
    print(f"RSS at start: {_get_mem_gb():.1f} GB")
    print("=" * 60)
    # 配合采样率和负样本数量下降，安全下调预期最大行数
    TRAIN_MAX_ROWS = 550_000_000
    VAL_MAX_ROWS   = 60_000_000
    X_train, y_train, group_train = load_data_to_numpy_ltr(TRAIN_YEARS, TRAIN_SAMPLE_RATIO, 42, is_train=True, max_expected_rows=TRAIN_MAX_ROWS, split_name="Train")
    X_val, y_val, group_val       = load_data_to_numpy_ltr(VAL_YEARS, VAL_SAMPLE_RATIO, 42, is_train=False, max_expected_rows=VAL_MAX_ROWS, split_name="Val")
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 15],  # 决断 3：更锐利的评估
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,        
        'num_leaves': 127,            
        'max_depth': 9,              
        'n_estimators': 5000,
        'num_threads': 20,
        'max_bin': 127,
        'feature_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 500,
        'force_col_wise': True,
        'verbosity': -1,
    }
    print(f"\nHanding over to C++... (Python RSS: {_get_mem_gb():.1f} GB)")
    train_ds = lgb.Dataset(X_train, label=y_train, group=group_train, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
    del X_train, y_train, group_train; gc.collect()
    train_ds.construct()
    val_ds = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_ds, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
    del X_val, y_val, group_val; gc.collect()
    val_ds.construct()
    print(f"\nTraining started... RSS: {_get_mem_gb():.1f} GB")
    # 决断 5：恢复 Checkpoint 与 Timing
    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
    timing_cb = TimingCallback(freq=10)
    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True), # 决断 4：50轮早停防过拟合
            lgb.log_evaluation(10),
            timing_cb,
            ckpt_cb,
        ],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_DIR / 'model_v7_lambdarank.txt'))
    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')}).sort_values('importance', ascending=False)
    print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")
if __name__ == '__main__':
    main()
'''