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

# 🎯 方案 B：彻底清空旧的二进制缓存，保证数据绝对干净
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

# 🚀 负样本数量：提升到 120，增加模型见识（Hard Neg 已经帮我们省内存了）
N_RAND_NEG = 120

def _get_mem_gb():
    return psutil.Process().memory_info().rss / 1024**3

def load_cache_for_year(year: int):
    """加载城市对缓存，返回 3D 张量用于 O(1) 向量化查找（带防越界保护）"""
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)

    # 找到所有唯一城市
    unique_cities = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    num_cities = len(unique_cities)
    num_feats = len(CACHE_FEAT_COLS)

    # 🛡️ 防御 1：固定容量（中国城市 ID < 10000）
    MAX_CITY_ID = 10000

    # 🛡️ 防御 2：默认指向"防空洞"（张量最后一行，全 0）
    city_map = np.full(MAX_CITY_ID, num_cities, dtype=np.int32)
    city_map[unique_cities] = np.arange(num_cities)

    # 🛡️ 防御 3：张量多分配一行作为兜底
    cache_tensor = np.zeros((num_cities + 1, num_cities + 1, num_feats), dtype=np.float32)

    # 向量化填充
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
    if not base_path.exists(): return None, None, None

    needed_cols = ['qid', 'From_City', 'To_City', 'Rank'] + PERSON_CATS
    base = pd.read_feather(base_path, columns=needed_cols)
    rank = base['Rank'].values

    # =====================================================================
    # 🎯 改进版：连续整数 Relevance (0-7)，利用框架默认 2^rel-1 增益
    # =====================================================================
    relevance = np.zeros_like(rank, dtype=np.int8)

    # Top 1-5：死磕精度 (Rel 7->Gain 127, Rel 6->Gain 63, ...)
    relevance[rank == 1] = 7
    relevance[rank == 2] = 6
    relevance[rank == 3] = 5
    relevance[rank == 4] = 4
    relevance[rank == 5] = 3

    # Top 6-15：平滑过渡（用户平均选14个城市，这些都是真样本）
    relevance[(rank >= 6) & (rank <= 10)] = 2   # Gain 3
    relevance[(rank >= 11) & (rank <= 15)] = 1  # Gain 1

    # Top 16-20：兜底（只要排在负样本前即可）
    relevance[(rank >= 16) & (rank <= 20)] = 1

    base['Relevance'] = relevance
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

    del qids, relevance, qids_final; gc.collect()

    # 🚀 极速特征提取：用 3D 张量替代 Pandas merge
    cache_tensor, city_map = load_cache_for_year(year)
    from_idx = city_map[base['From_City'].values]
    to_idx = city_map[base['To_City'].values]
    extracted_feats = cache_tensor[from_idx, to_idx, :]

    # 原地赋值，避免创建新 DataFrame
    for i, col in enumerate(CACHE_FEAT_COLS):
        base[col] = extracted_feats[:, i]

    del cache_tensor, city_map, extracted_feats; gc.collect()

    base = build_cross_features(base)
    for col in CACHE_FEAT_COLS + CROSS_FEATS:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(np.float32)

    return base, group_counts

# 🎯 核心改动点：删除 weight 参数，只保留 label 和 group
def build_and_save_bin(years, split_name, max_rows, sample_ratio, is_train, reference_ds=None):
    print(f"\n[{split_name}] 分配极速内存矩阵 (Max {max_rows:,} rows)... 当前 RSS: {_get_mem_gb():.1f} GB")
    X = np.empty((max_rows, len(FEATS)), dtype=np.float32)
    y = np.empty(max_rows, dtype=np.int8)  # 改为 int8，Relevance 只有 0-7
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

        df_feats = df[FEATS].astype(np.float32)
        X[current_idx : current_idx + rows] = df_feats.values
        y[current_idx : current_idx + rows] = df['Relevance'].values
        all_groups.extend(group_counts)

        current_idx += rows

        del df, df_feats; gc.collect()
        tqdm.write(f"  {year} | Rows: {rows:,} | Total: {current_idx:,} | RSS: {_get_mem_gb():.1f} GB | Time: {time.time()-t0:.1f}s")

    X, y = X[:current_idx], y[:current_idx]
    all_groups = np.array(all_groups, dtype=np.int32)

    print(f"\n📦 数据处理完毕！耗时 {(time.time()-t_start)/60:.1f} 分钟。")
    print(f"正在构建 LightGBM 底层二进制文件（该过程会压缩数据并计算特征直方图，请耐心等待）...")

    # 🎯 删除 weight 参数
    ds = lgb.Dataset(
        X, label=y, group=all_groups,
        feature_name=list(FEATS), categorical_feature=CATS,
        free_raw_data=True,
        reference=reference_ds,
        params={'max_bin': 255}
    )

    bin_path = BIN_OUTPUT_DIR / f"{split_name.lower()}_v11_clean.bin"
    ds.construct()
    ds.save_binary(str(bin_path))

    print(f"✅ {split_name} 二进制文件已安全落地至高速盘: {bin_path}")

    return ds

if __name__ == '__main__':
    TRAIN_MAX_ROWS = 400_000_000
    VAL_MAX_ROWS   = 50_000_000

    # 先构建 Train，并获取其 Dataset 对象
    train_dataset = build_and_save_bin(list(range(2000, 2017)), "Train", TRAIN_MAX_ROWS, 0.3, True, reference_ds=None)
    
    # 将 Train 的 Dataset 对象作为 reference 传给 Val 构建过程
    build_and_save_bin([2017, 2018], "Val", VAL_MAX_ROWS, 0.2, False, reference_ds=train_dataset)
    
    print("\n🎉 全部打包完成！现在可以直接去跑 rank_train.py ")