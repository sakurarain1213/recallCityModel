"""
LightGBM 召回专属数据构建脚本 (二分类扁平权重特化版 + 极速硬盘映射防 Swap 假死)
"""

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

if BIN_OUTPUT_DIR.exists():
    print(f"🧹 正在彻底清空旧缓存目录: {BIN_OUTPUT_DIR} ...")
    shutil.rmtree(BIN_OUTPUT_DIR)
BIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# 🎯 控制内存与样本均衡的黄金法则
N_RAND_NEG = 120

def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3

def load_cache_for_year(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)

    unique_cities = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    num_cities = len(unique_cities)
    num_feats = len(CACHE_FEAT_COLS)
    MAX_CITY_ID = 10000

    city_map = np.full(MAX_CITY_ID, num_cities, dtype=np.int32)
    city_map[unique_cities] = np.arange(num_cities)

    cache_tensor = np.zeros((num_cities + 1, num_cities + 1, num_feats), dtype=np.float32)
    from_idx = city_map[df['from_city'].values]
    to_idx = city_map[df['to_city'].values]
    cache_tensor[from_idx, to_idx, :] = df[CACHE_FEAT_COLS].values.astype(np.float32)

    return cache_tensor, city_map

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
    if not base_path.exists(): return None, None, None, None

    needed_cols = ['qid', 'From_City', 'To_City', 'Rank'] + PERSON_CATS
    base = pd.read_feather(base_path, columns=needed_cols)
    rank = base['Rank'].values

    # =====================================================================
    # 🎯 纯二分类召回法则：所有前 20 的都是正样本 (1)，其余全为负样本 (0)
    # =====================================================================
    label = np.zeros_like(rank, dtype=np.int8)
    label[(rank >= 1) & (rank <= 20)] = 1
    
    # 🎯 扁平化极权：保证头部同时避免 loss 淹没
    weight = np.ones_like(rank, dtype=np.float32)
    weight[(rank >= 1) & (rank <= 5)] = 40.0
    weight[(rank >= 6) & (rank <= 20)] = 30.0

    base['Label'] = label
    base['Weight'] = weight
    qids = base['qid'].values

    change_indices = np.nonzero(qids[:-1] != qids[1:])[0] + 1
    split_indices = np.r_[0, change_indices, len(qids)]
    group_counts = np.diff(split_indices)

    max_label_per_group = np.maximum.reduceat(label, split_indices[:-1])
    valid_group_mask = max_label_per_group > 0
    if not valid_group_mask.all():
        valid_row_mask = np.repeat(valid_group_mask, group_counts)
        base = base[valid_row_mask].reset_index(drop=True)
        qids = base['qid'].values
        label = base['Label'].values
        weight = base['Weight'].values
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
        label = base['Label'].values
        weight = base['Weight'].values

    # 动态负采样
    if is_train:
        rng = np.random.default_rng(random_seed + year)
        is_pos = label == 1
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

    del qids, label, weight, qids_final; gc.collect()

    cache_tensor, city_map = load_cache_for_year(year)
    from_idx = city_map[base['From_City'].values]
    to_idx = city_map[base['To_City'].values]
    extracted_feats = cache_tensor[from_idx, to_idx, :]

    for i, col in enumerate(CACHE_FEAT_COLS):
        base[col] = extracted_feats[:, i]

    del cache_tensor, city_map, extracted_feats; gc.collect()

    base = build_cross_features(base)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base, group_counts

def build_and_save_bin(years, split_name, max_rows, sample_ratio, is_train, reference_ds=None):
    print(f"\n[{split_name}] 硬盘映射矩阵模式 (Max {max_rows:,} rows)... 当前 Python RSS: {_get_mem_gb():.1f} GB")
    
    # 🎯 核心改动：用 memmap 替代 np.empty，直接在硬盘读写，绕开 Linux Swap！
    temp_X_path = BIN_OUTPUT_DIR / f"{split_name}_temp_X.dat"
    temp_y_path = BIN_OUTPUT_DIR / f"{split_name}_temp_y.dat"
    temp_w_path = BIN_OUTPUT_DIR / f"{split_name}_temp_w.dat"
    
    X = np.memmap(temp_X_path, dtype=np.float32, mode='w+', shape=(max_rows, len(FEATS)))
    y = np.memmap(temp_y_path, dtype=np.int8, mode='w+', shape=(max_rows,))
    w = np.memmap(temp_w_path, dtype=np.float32, mode='w+', shape=(max_rows,))

    all_groups = []
    current_idx = 0

    t_start = time.time()
    for year in tqdm(years, desc=f"  打包 {split_name}", unit="yr"):
        t0 = time.time()
        df, group_counts = _process_single_year(year, sample_ratio, 42, is_train)
        if df is None: continue

        for col in CATS:
            if str(df[col].dtype) == 'category':
                df[col] = df[col].cat.codes

        rows = len(df)
        if current_idx + rows > max_rows:
            raise ValueError(f"溢出！请调大 {split_name} 的预留行数 max_rows。")

        # 将每年处理好的数据推入硬盘
        X[current_idx : current_idx + rows] = df[FEATS].astype(np.float32).values
        y[current_idx : current_idx + rows] = df['Label'].values
        w[current_idx : current_idx + rows] = df['Weight'].values
        
        # 🎯 强制刷入硬盘，释放页缓存
        X.flush(); y.flush(); w.flush()
        
        all_groups.extend(group_counts)
        current_idx += rows

        del df; gc.collect()
        tqdm.write(f"  {year} | Rows: {rows:,} | Total: {current_idx:,} | Python RSS 稳如泰山: {_get_mem_gb():.1f} GB | Time: {time.time()-t0:.1f}s")

    # 裁剪到实际大小的视图
    X_view = X[:current_idx]
    y_view = y[:current_idx]
    w_view = w[:current_idx]
    all_groups = np.array(all_groups, dtype=np.int32)

    if not is_train:
        print("正在物理隔离保存验证集结构以便实现极速 Recall 评测...")
        # 注意：这里我们存了一份到 npy 文件里供验证用
        np.save(BIN_OUTPUT_DIR / "val_labels.npy", np.array(y_view))
        np.save(BIN_OUTPUT_DIR / "val_groups.npy", all_groups)

    print(f"\n📦 数据处理完毕！耗时 {(time.time()-t_start)/60:.1f} 分钟。")
    print(f"正在交给底层 C++ 构建 LightGBM 二进制文件 (它将直接从硬盘读取矩阵)...")

    ds = lgb.Dataset(
        X_view, label=y_view, weight=w_view,
        feature_name=list(FEATS), categorical_feature=CATS,
        free_raw_data=True,
        reference=reference_ds,
        params={'max_bin': 255}
    )

    bin_path = BIN_OUTPUT_DIR / f"{split_name.lower()}_recall.bin"
    ds.construct()
    ds.save_binary(str(bin_path))

    print(f"✅ {split_name} 二进制文件已安全落地: {bin_path}")

    # 🎯 阅后即焚：删除临时占据大量硬盘空间的 .dat 映射文件
    del X, y, w, X_view, y_view, w_view
    gc.collect()
    time.sleep(2) # 等待操作系统彻底释放文件句柄
    
    if temp_X_path.exists(): temp_X_path.unlink()
    if temp_y_path.exists(): temp_y_path.unlink()
    if temp_w_path.exists(): temp_w_path.unlink()
    
    print(f"🧹 已自动清理硬盘上的临时缓存映射文件。")
    return ds

if __name__ == '__main__':
    TRAIN_MAX_ROWS = 400_000_000
    VAL_MAX_ROWS   = 50_000_000

    train_dataset = build_and_save_bin(list(range(2000, 2017)), "Train", TRAIN_MAX_ROWS, 0.3, True, reference_ds=None)
    build_and_save_bin([2017, 2018], "Val", VAL_MAX_ROWS, 0.05, False, reference_ds=train_dataset)
    
    print("\n🎉 全部打包完成！现在可以去跑 fast_recall_train.py 了！")