"""
LightGBM 召回模型训练脚本 (v2 - 丰富特征版)

架构:
  base_ready/base_YYYY.feather  — 人口统计 + 标签 (精简, ~150MB/年)
  city_pair_cache/city_pairs_YYYY.parquet — 城市对特征 (~50列, ~5MB/年)
  训练时动态 join + 构建交叉特征

运行: cd /data1/wxj/Recall_city_project/ && uv run simple_train.py
依赖: output/base_ready/, data/city_pair_cache/
"""

import os
import time
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 路径配置 =================
BASE_DIR = Path("output/base_ready")
CACHE_DIR = Path("data/city_pair_cache")
OUTPUT_DIR = Path("output")

# ================= 特征定义 (训练/评估共享) =================
# 人口统计特征 (类别型)
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']

# 城市对 — 比率特征
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

# 城市对 — 差值特征
DIFF_FEATS = [
    'gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff',
    'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff',
    'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff',
]

# 城市对 — 绝对特征
ABS_FEATS = [
    'to_tier', 'to_population_log', 'to_gdp_per_capita',
    'from_tier', 'from_population_log', 'tier_diff',
]

# 社会网络 + 距离
NET_DIST_FEATS = [
    'migrant_stock_from_to',
    'geo_distance', 'dialect_distance', 'is_same_province',
]

# 交叉特征名 (动态构建)
CROSS_FEATS = [
    'industry_x_matched_wage_ratio',
    'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff',
    'income_x_gdp_ratio',
    'age_x_housing_ratio',
    'family_x_edu_score_ratio',
]

# 所有类别型特征
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']

# 最终特征列表 (训练 & 评估共用)
FEATS = (
    PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
)

# cache parquet 中需要 join 的列 (排除 from_city, to_city)
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS


def load_cache_for_year(year: int) -> pd.DataFrame:
    """加载单年城市对 cache, 返回 (from_city, to_city, ...features) DataFrame"""
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)
    df['from_city'] = df['from_city'].astype(np.int32)
    df['to_city'] = df['to_city'].astype(np.int32)
    return df


def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建人×城市交叉特征 (向量化, 就地修改)。
    行业匹配: 根据人的 industry 编码选择对应行业的 wage/vacancy ratio。
    """
    industry = df['industry'].values if hasattr(df['industry'], 'values') else df['industry'].cat.codes.values

    # industry: 0=Agri, 1=Mfg, 2=TradSvc, 3=ModSvc
    wage_cols = ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']
    vacancy_cols = ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']

    wage_arr = df[wage_cols].values.astype(np.float32)    # (N, 4)
    vacancy_arr = df[vacancy_cols].values.astype(np.float32)

    ind_idx = np.clip(industry.astype(int), 0, 3)
    rows = np.arange(len(df))

    df['industry_x_matched_wage_ratio'] = wage_arr[rows, ind_idx]
    df['industry_x_matched_vacancy_ratio'] = vacancy_arr[rows, ind_idx]

    # education × tier_diff
    edu = df['education'].values if hasattr(df['education'], 'values') else df['education'].cat.codes.values
    df['education_x_tier_diff'] = edu.astype(np.float32) * df['tier_diff'].values.astype(np.float32)

    # income × gdp_ratio
    inc = df['income'].values if hasattr(df['income'], 'values') else df['income'].cat.codes.values
    df['income_x_gdp_ratio'] = inc.astype(np.float32) * df['gdp_per_capita_ratio'].values

    # age × housing_ratio
    age = df['age_group'].values if hasattr(df['age_group'], 'values') else df['age_group'].cat.codes.values
    df['age_x_housing_ratio'] = age.astype(np.float32) * df['housing_price_avg_ratio'].values

    # family × education_score_ratio
    fam = df['family'].values if hasattr(df['family'], 'values') else df['family'].cat.codes.values
    df['family_x_edu_score_ratio'] = fam.astype(np.float32) * df['education_score_ratio'].values

    return df


# ================= 困难负样本城市池 (Tier 1+2, 高吸引力城市) =================
HARD_NEG_CITIES_SET = {
    # 一线城市 (Tier 1)
    1100, 3100, 4401, 4403,
    
    # 新一线城市 (New Tier 1)
    1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301, 
    4406, 4419, 5000, 5101, 6101,
    
    # 二线核心城市 (Tier 2)
    1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206,
    3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701,
    3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501
}

# 负采样参数 (可调)
N_HARD_NEG = 30   # 每个 query 最多保留的困难负样本数
N_RAND_NEG = 30   # 每个 query 的随机负样本数


def _vectorized_neg_sample(base: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """
    向量化负采样: 不用 groupby, 用布尔 mask 实现。
    策略: 保留全部正样本 + 困难负样本(随机N_HARD_NEG个) + 随机负样本(N_RAND_NEG个)
    """
    rng = np.random.default_rng(random_seed)

    label = base['Label'].values
    to_city = base['To_City'].values

    is_pos = (label == 1)
    is_neg = (label != 1)
    is_hard_city = np.isin(to_city, list(HARD_NEG_CITIES_SET))
    is_hard_neg = is_neg & is_hard_city
    is_rand_neg = is_neg & (np.logical_not(is_hard_city))

    # 正样本: 全部保留
    keep = is_pos.copy()

    # 困难负样本: 每个 qid 最多 N_HARD_NEG 个
    # 用随机数 + 组内排名实现, 避免 groupby
    hard_idx = np.where(is_hard_neg)[0]
    if len(hard_idx) > 0:
        qids_hard = base['qid'].values[hard_idx]
        # 给每个困难负样本一个随机排名
        rand_rank = rng.random(len(hard_idx))
        # 按 qid 分组, 取每组前 N_HARD_NEG 个
        hard_df = pd.DataFrame({'qid': qids_hard, 'rank': rand_rank, 'orig_idx': hard_idx})
        hard_df['grp_rank'] = hard_df.groupby('qid')['rank'].rank(method='first')
        selected_hard = hard_df[hard_df['grp_rank'] <= N_HARD_NEG]['orig_idx'].values
        keep[selected_hard] = True

    # 随机负样本: 每个 qid N_RAND_NEG 个
    rand_idx = np.where(is_rand_neg)[0]
    if len(rand_idx) > 0:
        qids_rand = base['qid'].values[rand_idx]
        rand_rank = rng.random(len(rand_idx))
        rand_df = pd.DataFrame({'qid': qids_rand, 'rank': rand_rank, 'orig_idx': rand_idx})
        rand_df['grp_rank'] = rand_df.groupby('qid')['rank'].rank(method='first')
        selected_rand = rand_df[rand_df['grp_rank'] <= N_RAND_NEG]['orig_idx'].values
        keep[selected_rand] = True

    return base[keep].reset_index(drop=True)


def _process_single_year(year: int, sample_ratio: float, random_seed: int,
                         neg_sample: bool = True):
    """加载单年: base feather → query采样 → 负采样 → merge cache → 交叉特征"""
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists():
        return None

    base = pd.read_feather(base_path)

    # query 级采样
    if 0.0 < sample_ratio < 1.0:
        unique_qids = base['qid'].unique()
        rng = np.random.default_rng(random_seed)
        n_sample = int(len(unique_qids) * sample_ratio)
        sampled = set(rng.choice(unique_qids, size=n_sample, replace=False))
        base = base[base['qid'].isin(sampled)]

    # 负采样 (训练时启用, 评估时不启用)
    if neg_sample:
        base = _vectorized_neg_sample(base, random_seed + year)
        gc.collect()

    # merge cache (cache 只有 11.3 万行, 很小)
    cache = load_cache_for_year(year)
    base = base.merge(
        cache,
        left_on=['From_City', 'To_City'],
        right_on=['from_city', 'to_city'],
        how='left',
    )
    base.drop(columns=['from_city', 'to_city'], inplace=True)
    del cache

    # 交叉特征
    base = build_cross_features(base)

    # 填充缺失
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base


def load_data(years: list, sample_ratio: float = 1.0, random_seed: int = 42,
              neg_sample: bool = True) -> pd.DataFrame:
    """
    逐年加载 + 负采样 + merge cache + 交叉特征, 最后 concat。
    负采样后每年行数从 ~9500万 降到 ~2000万, 17年总计 ~3.4亿行, 内存 ~80GB。
    """
    print(f"Loading data for years: {years} | Sample Ratio: {sample_ratio*100:.0f}%")
    if neg_sample:
        print(f"  Neg sampling: {N_HARD_NEG} hard + {N_RAND_NEG} random per query")

    all_dfs = []
    total_rows = 0

    for year in years:
        t0 = time.time()

        df = _process_single_year(year, sample_ratio, random_seed, neg_sample=neg_sample)
        if df is None:
            print(f"  [Warning] {year} not found, skip")
            continue

        n_q = df['qid'].nunique()
        total_rows += len(df)
        print(f"  {year}: {len(df):,} rows, {n_q:,} queries ({time.time()-t0:.1f}s)")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No data loaded for years {years}!")

    print(f"Concatenating {len(all_dfs)} years, total {total_rows:,} rows...")
    final = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    # 转 category
    for col in CATS:
        if col in final.columns:
            final[col] = final[col].astype('category')

    print(f"Data loaded: {final.shape}, Memory: {final.memory_usage(deep=True).sum()/1024**3:.2f} GB")
    return final


def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """正负比 ~1:23, 正样本按 Rank 分级加权"""
    weights = np.ones(len(df), dtype=np.float32)
    pos_mask = df['Label'] == 1
    rank = df['Rank']
    weights[pos_mask & (rank <= 3)]  = 50.0
    weights[pos_mask & (rank > 3)  & (rank <= 10)] = 35.0
    weights[pos_mask & (rank > 10) & (rank <= 20)] = 24.0
    return weights


class CheckpointCallback:
    """定期保存 Checkpoint"""
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


def main():
    TRAIN_YEARS = list(range(2000, 2017))
    VAL_YEARS   = [2017, 2018]

    SAMPLE_RATIO = 0.8  # 可选采样比例 (0.0-1.0), 用于快速迭代

    print("=" * 60)
    print("LightGBM Recall Training (v2 - Rich Features)")
    print(f"Features: {len(FEATS)} total")
    print("=" * 60)

    # 1. 加载数据
    df_train = load_data(TRAIN_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)
    df_val   = load_data(VAL_YEARS, sample_ratio=SAMPLE_RATIO, random_seed=42)

    # 2. 权重
    print("\nCalculating sample weights...")
    train_weights = calculate_sample_weights(df_train)

    # 3. 参数
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt', 
        'learning_rate': 0.1,   # 或者0.05
        'num_leaves': 255,  # 或者127
        'max_depth': 10,
        'n_estimators': 5000,
        'n_jobs': os.cpu_count(),
        'num_threads': os.cpu_count(),
        'max_bin': 255,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 100,
        'verbosity': -1,
    }

    # 4. 构建 Dataset
    print("Constructing LightGBM Datasets...")
    train_ds = lgb.Dataset(
        df_train[FEATS], label=df_train['Label'], weight=train_weights,
        categorical_feature=CATS, params=params, free_raw_data=True,
    )
    val_ds = lgb.Dataset(
        df_val[FEATS], label=df_val['Label'],
        categorical_feature=CATS, reference=train_ds,
        params=params, free_raw_data=True,
    )

    print("Forcing Dataset construction...")
    train_ds.construct()
    val_ds.construct()
    del df_train, df_val, train_weights
    gc.collect()

    # 5. 训练
    n_cores = os.cpu_count()
    print(f"\nTraining started ({n_cores} cores)...")
    start_time = time.time()

    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(10),
            ckpt_cb,
        ],
    )
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")

    # 6. 保存
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'model_v2.txt'
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # 7. 特征重要性
    imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')})
    imp = imp.sort_values('importance', ascending=False)
    print(f"\nTop 20 Features by Gain:")
    print(imp.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
