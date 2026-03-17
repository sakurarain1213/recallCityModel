import os
import gc
import time
import shutil
import psutil
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= 极速存储路径 =================
BASE_DIR = Path("output/base_ready")
CACHE_DIR = Path("data/city_pair_cache")
BIN_OUTPUT_DIR = Path("/data2/wxj/recall_bin")

# ================= 特征定义 =================
PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
RATIO_FEATS = ['gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio', 'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio', 'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio', 'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio', 'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio', 'avg_commute_mins_ratio', 'population_total_ratio', 'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio', 'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio']
DIFF_FEATS = ['gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff', 'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff', 'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff']
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = ['industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio', 'education_x_tier_diff', 'income_x_gdp_ratio', 'age_x_housing_ratio', 'family_x_edu_score_ratio']
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

HARD_NEG_CITIES_SET = {1100, 3100, 4401, 4403, 1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301, 4406, 4419, 5000, 5101, 6101, 1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206, 3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701, 3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501}

# 🚀 核心内存优化：随机负样本降低至 50，省下 50GB 内存！
N_RAND_NEG = 50

def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3

def load_cache_for_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)
    df['from_city'] = df['from_city'].astype(np.int32)
    df['to_city'] = df['to_city'].astype(np.int32)
    return df

def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    industry = df['industry'].values if hasattr(df['industry'], 'values') else df['industry'].cat.codes.values
    wage_arr = df[['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']].values.astype(np.float32)
    vacancy_arr = df[['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']].values.astype(np.float32)
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

def _process_single_year(year: int, sample_ratio: float, random_seed: int, is_train: bool):
    base_path = BASE_DIR / f"base_{year}.feather"
    if not base_path.exists(): return None, None, None

    needed_cols = ['qid', 'From_City', 'To_City', 'Rank'] + PERSON_CATS
    base = pd.read_feather(base_path, columns=needed_cols)
    rank = base['Rank'].values

    # =====================================================================
    # 🎯 算法核心大招：Top 1-10 全阶梯指数落差与权重倾斜
    # =====================================================================
    relevance = np.zeros_like(rank, dtype=np.int16) 
    weight = np.ones_like(rank, dtype=np.float32)

    # Top 1-5：核心优先级（Relevance=2^n-1，权重梯度递减）
    relevance[rank == 1] = 127; weight[rank == 1] = 15.0  
    relevance[rank == 2] = 63;  weight[rank == 2] = 14.0
    relevance[rank == 3] = 31;  weight[rank == 3] = 13.0
    relevance[rank == 4] = 15;  weight[rank == 4] = 12.0
    relevance[rank == 5] = 7;   weight[rank == 5] = 11.0

    # Top 6-10：次要优先级（Relevance=2^n-1，权重快速递减）
    relevance[rank == 6] = 3;   weight[rank == 6] = 8.0
    relevance[rank == 7] = 2;   weight[rank == 7] = 6.0
    relevance[rank == 8] = 1;   weight[rank == 8] = 4.0
    relevance[rank == 9] = 1;   weight[rank == 9] = 3.0
    relevance[rank == 10]= 1;   weight[rank == 10]= 2.0
    
    # Top 11-20：相对准确即可，只充当及格线的正样本基石
    mask_11_20 = (rank >= 11) & (rank <= 20)
    relevance[mask_11_20] = 1;   weight[mask_11_20] = 1.0

    base['Relevance'] = relevance
    base['Weight'] = weight
    qids = base['qid'].values

    # NumPy 极速查找边界
    change_indices = np.nonzero(qids[:-1] != qids[1:])[0] + 1
    split_indices = np.r_[0, change_indices, len(qids)]
    group_counts = np.diff(split_indices)

    # 过滤无效 QID
    max_rel_per_group = np.maximum.reduceat(relevance, split_indices[:-1])
    valid_group_mask = max_rel_per_group > 0
    if not valid_group_mask.all():
        valid_row_mask = np.repeat(valid_group_mask, group_counts)
        base = base[valid_row_mask].reset_index(drop=True)
        qids = base['qid'].values
        relevance = base['Relevance'].values
        group_counts = group_counts[valid_group_mask]

    # QID 极速采样
    if 0.0 < sample_ratio < 1.0:
        rng = np.random.default_rng(random_seed)
        unique_qids = qids[np.r_[0, np.nonzero(qids[:-1] != qids[1:])[0] + 1]]
        n_sample = int(len(unique_qids) * sample_ratio)
        sampled_qids = rng.choice(unique_qids, size=n_sample, replace=False)
        sampled_row_mask = np.isin(qids, sampled_qids)
        base = base[sampled_row_mask].reset_index(drop=True)
        qids = base['qid'].values
        relevance = base['Relevance'].values

    # 动态负采样
    if is_train:
        rng = np.random.default_rng(random_seed + year)
        is_pos = relevance > 0
        is_neg = ~is_pos
        to_city = base['To_City'].values
        from_city = base['From_City'].values

        keep = is_pos.copy()
        is_static_hard = np.isin(to_city, list(HARD_NEG_CITIES_SET))
        keep[is_neg & is_static_hard] = True

        to_province = to_city // 100
        from_province = from_city // 100
        is_dynamic_hard = is_neg & (to_province == from_province) & (~is_static_hard)
        keep[is_dynamic_hard] = True

        remaining_neg = is_neg & (~keep)
        if remaining_neg.sum() > 0:
            qids_rem = qids[remaining_neg]
            change_indices_rem = np.nonzero(qids_rem[:-1] != qids_rem[1:])[0] + 1
            split_indices_rem = np.r_[0, change_indices_rem, len(qids_rem)]
            rem_counts = np.diff(split_indices_rem)

            probs = np.minimum(1.0, N_RAND_NEG / rem_counts)
            row_probs = np.repeat(probs, rem_counts)
            rand_mask = rng.random(len(row_probs)) < row_probs
            rand_idx = np.where(remaining_neg)[0]
            keep[rand_idx[rand_mask]] = True

        base = base[keep].reset_index(drop=True)
        gc.collect()

    qids_final = base['qid'].values
    change_indices_final = np.nonzero(qids_final[:-1] != qids_final[1:])[0] + 1
    split_indices_final = np.r_[0, change_indices_final, len(qids_final)]
    group_counts = np.diff(split_indices_final).astype(np.int32)
    weights_final = base['Weight'].values

    del qids, relevance, qids_final; gc.collect()

    cache = load_cache_for_year(year)
    base = base.merge(cache, left_on=['From_City', 'To_City'], right_on=['from_city', 'to_city'], how='left', sort=False)
    base.drop(columns=['from_city', 'to_city'], inplace=True)
    del cache; gc.collect()

    base = build_cross_features(base)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base, group_counts, weights_final

# 🎯 核心改动点：增加 reference_ds 参数，解决 Validation 报错
def build_and_save_bin(years, split_name, max_rows, sample_ratio, is_train, reference_ds=None):
    print(f"\n[{split_name}] 分配极速内存矩阵 (Max {max_rows:,} rows)... 当前 RSS: {_get_mem_gb():.1f} GB")
    X = np.empty((max_rows, len(FEATS)), dtype=np.float32)
    y = np.empty(max_rows, dtype=np.int16) # 对齐 int16
    w = np.empty(max_rows, dtype=np.float32)
    all_groups = []
    current_idx = 0

    t_start = time.time()
    for year in tqdm(years, desc=f"  打包 {split_name}", unit="yr"):
        t0 = time.time()
        df, group_counts, weights = _process_single_year(year, sample_ratio, 42, is_train)
        if df is None: continue

        for col in CATS:
            if str(df[col].dtype) == 'category':
                df[col] = df[col].cat.codes

        rows = len(df)
        if current_idx + rows > max_rows:
            raise ValueError(f"溢出！请调大 {split_name} 的预留行数 max_rows。")

        df_feats = df[FEATS].astype(np.float32)
        X[current_idx : current_idx + rows] = df_feats.values
        y[current_idx : current_idx + rows] = df['Relevance'].values
        w[current_idx : current_idx + rows] = weights
        all_groups.extend(group_counts)

        current_idx += rows
        
        del df, df_feats, weights; gc.collect()
        tqdm.write(f"  {year} | Rows: {rows:,} | Total: {current_idx:,} | RSS: {_get_mem_gb():.1f} GB | Time: {time.time()-t0:.1f}s")

    X, y, w = X[:current_idx], y[:current_idx], w[:current_idx]
    all_groups = np.array(all_groups, dtype=np.int32)

    print(f"\n📦 数据处理完毕！耗时 {(time.time()-t_start)/60:.1f} 分钟。")
    print(f"正在构建 LightGBM 底层二进制文件（该过程会压缩数据并计算特征直方图，请耐心等待）...")
    
    # 🎯 核心改动点：强行传入 params={'max_bin': 255} 和 reference_ds 保证对齐
    ds = lgb.Dataset(
        X, label=y, weight=w, group=all_groups, 
        feature_name=list(FEATS), categorical_feature=CATS, 
        free_raw_data=True,
        reference=reference_ds,       # 对齐验证集的桶边界
        params={'max_bin': 255}       # 对齐训练脚本的 max_bin
    )
    
    bin_path = BIN_OUTPUT_DIR / f"{split_name.lower()}_v10_top10.bin"
    ds.construct()
    ds.save_binary(str(bin_path))
    
    print(f"✅ {split_name} 二进制文件已安全落地至高速盘: {bin_path}")
    
    # 🎯 核心改动点：把构建好的 Dataset 对象返回，作为验证集的 reference
    return ds

if __name__ == '__main__':
    # 🎯 彻底清空旧的二进制缓存，保证数据绝对干净
    if BIN_OUTPUT_DIR.exists():
        print(f"🧹 正在彻底清空旧缓存目录: {BIN_OUTPUT_DIR} ...")
        shutil.rmtree(BIN_OUTPUT_DIR)
    BIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_MAX_ROWS = 400_000_000
    VAL_MAX_ROWS   = 50_000_000

    # 先构建 Train，并获取其 Dataset 对象
    train_dataset = build_and_save_bin(list(range(2000, 2017)), "Train", TRAIN_MAX_ROWS, 0.6, True, reference_ds=None)
    
    # 将 Train 的 Dataset 对象作为 reference 传给 Val 构建过程
    build_and_save_bin([2017, 2018], "Val", VAL_MAX_ROWS, 0.2, False, reference_ds=train_dataset)
    
    print("\n🎉 全部打包完成！现在可以直接去跑 rank_train.py ")