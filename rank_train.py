
"""
LightGBM 极速训练脚本 (v10 - O(1) 二进制加载 + LambdaRank Top-10 专杀版)

核心改动：
1. 彻底删除 Pandas 数据加载与特征拼接逻辑，直接以 O(1) 复杂度从 NVMe 盘加载 .bin 缓存。
2. 目标函数: rank_xendcg → lambdarank。配合我们在 build_bin 中定制的指数级 Relevance 标签。
3. 截断层级: lambdarank_truncation_level = 10。让模型算力死磕前 10 名。
"""

import time
import lightgbm as lgb
import pandas as pd
from pathlib import Path

# ================= 极速存储路径 =================
# 🎯 指向我们刚刚用 build_bin.py 生成好的高速盘目录
BIN_DIR = Path("/data2/wxj/recall_bin") 
OUTPUT_DIR = Path("output")

# ================= 回调函数 =================
class CheckpointCallback:
    def __init__(self, output_dir, freq=10):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
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
    print("=" * 60)
    print("🚀 LightGBM O(1) 极速训练模式启动 (v10 - LambdaRank Top-10 版)")
    print("=" * 60)

    t0 = time.time()
    
    # 🔥 魔法时刻：O(1) 极速加载，跳过所有特征直方图的预计算
    print("正在从 /data2 极速加载 Train 数据...")
    train_ds = lgb.Dataset(str(BIN_DIR / "train_v10_top10.bin"))
    
    print("正在从 /data2 极速加载 Val 数据...")
    # 验证集必须设置 reference=train_ds，保证特征分桶边界一致
    val_ds = lgb.Dataset(str(BIN_DIR / "val_v10_top10.bin"), reference=train_ds)
    
    print(f"✅ 数据加载完毕！耗时仅 {(time.time() - t0):.2f} 秒。")

    params = {
        # 🎯 核心算法回调 1：换回 lambdarank，天然专一于头部排序
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],
        
        # 🎯 核心算法回调 2：告诉模型算梯度时只看前 10 名
        'lambdarank_truncation_level': 10, 
        
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 95,
        'max_depth': 9,
        'n_estimators': 5000,
        'num_threads': 38,
        'max_bin': 127,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 1000,
        'force_col_wise': True,
        'verbosity': -1,
    }

    print(f"\n💥 Training started...")
    
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = OUTPUT_DIR / 'model_v10_lambdarank_top10.txt'
    model.save_model(str(final_model_path))
    print(f"\n✅ 模型训练完成！已保存至 {final_model_path}")

    # 动态获取模型内固化的特征名称，避免在脚本里写死一长串列表
    feats = model.feature_name()
    imp = pd.DataFrame({
        'feature': feats, 
        'importance': model.feature_importance('gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")

if __name__ == '__main__':
    main()

















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

# import os
# import time
# import gc
# import psutil
# import lightgbm as lgb
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm

# # ================= 路径配置 =================
# BASE_DIR = Path("output/base_ready")
# CACHE_DIR = Path("data/city_pair_cache")
# OUTPUT_DIR = Path("output")

# # ================= 特征定义 =================
# PERSON_CATS = ['gender', 'age_group', 'education', 'industry', 'income', 'family']
# RATIO_FEATS = [
#     'gdp_per_capita_ratio', 'cpi_index_ratio', 'unemployment_rate_ratio',
#     'agri_share_ratio', 'agri_wage_ratio', 'agri_vacancy_ratio',
#     'mfg_share_ratio', 'mfg_wage_ratio', 'mfg_vacancy_ratio',
#     'trad_svc_share_ratio', 'trad_svc_wage_ratio', 'trad_svc_vacancy_ratio',
#     'mod_svc_share_ratio', 'mod_svc_wage_ratio', 'mod_svc_vacancy_ratio',
#     'housing_price_avg_ratio', 'rent_avg_ratio', 'daily_cost_index_ratio',
#     'medical_score_ratio', 'education_score_ratio', 'transport_convenience_ratio',
#     'avg_commute_mins_ratio', 'population_total_ratio',
#     'age_0_17_ratio', 'age_18_34_ratio', 'age_35_54_ratio',
#     'age_55_64_ratio', 'age_65_plus_ratio', 'sex_ratio_ratio', 'area_sqkm_ratio',
# ]
# DIFF_FEATS = [
#     'gdp_per_capita_diff', 'housing_price_avg_diff', 'rent_avg_diff',
#     'agri_wage_diff', 'mfg_wage_diff', 'trad_svc_wage_diff', 'mod_svc_wage_diff',
#     'agri_vacancy_diff', 'mfg_vacancy_diff', 'trad_svc_vacancy_diff', 'mod_svc_vacancy_diff',
# ]
# ABS_FEATS = [
#     'to_tier', 'to_population_log', 'to_gdp_per_capita',
#     'from_tier', 'from_population_log', 'tier_diff',
# ]
# NET_DIST_FEATS = [
#     'migrant_stock_from_to',
#     'geo_distance', 'dialect_distance', 'is_same_province',
# ]
# CROSS_FEATS = [
#     'industry_x_matched_wage_ratio',
#     'industry_x_matched_vacancy_ratio',
#     'education_x_tier_diff',
#     'income_x_gdp_ratio',
#     'age_x_housing_ratio',
#     'family_x_edu_score_ratio',
# ]
# CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
# FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
# CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# # ================= 负样本池及参数 =================
# # 静态困难负样本城市池：一二线城市（共 50 个）
# # 这些城市是热门迁移目标，作为困难负样本帮助模型区分真实偏好与热门城市的干扰
# HARD_NEG_CITIES_SET = {
#     1100, 3100, 4401, 4403,
#     1200, 3201, 3205, 3301, 3302, 3401, 3702, 4101, 4201, 4301,
#     4406, 4419, 5000, 5101, 6101,
#     1301, 1401, 2101, 2102, 2201, 2301, 3202, 3203, 3204, 3206,
#     3303, 3304, 3306, 3307, 3310, 3501, 3502, 3505, 3601, 3701,
#     3706, 3707, 3713, 4404, 4413, 4420, 4501, 4601, 5201, 5301, 6501
# }

# # v8 负采样策略：
# # 1. 保留所有静态困难负样本（HARD_NEG_CITIES_SET 中的负样本城市）
# # 2. 保留所有动态困难负样本（与出发城市同省份的负样本，前两位相同）
# # 3. 从剩余负样本中随机采样 N_RAND_NEG 个，保证模型见到全局视野
# N_RAND_NEG = 100

# # ================= 回调函数恢复 =================
# class CheckpointCallback:
#     def __init__(self, output_dir, freq=10):
#         self.output_dir = Path(output_dir) / "checkpoints"
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.freq = freq
#         self.order = 30
#         self.before_iteration = False

#     def __call__(self, env):
#         current_round = env.iteration + 1
#         if current_round % self.freq == 0:
#             ckpt_path = self.output_dir / f"model_iter_{current_round}.txt"
#             env.model.save_model(str(ckpt_path))
#             print(f"  [Checkpoint] iter {current_round} -> {ckpt_path}")

# class TimingCallback:
#     def __init__(self, freq=10):
#         self.freq = freq
#         self.start_time = None
#         self.order = 25
#         self.before_iteration = False

#     def __call__(self, env):
#         if self.start_time is None:
#             self.start_time = time.time()
#         current_round = env.iteration + 1
#         if current_round % self.freq == 0:
#             elapsed = time.time() - self.start_time
#             speed = current_round / elapsed * 60
#             print(f"  [Timer] iter {current_round} | elapsed {elapsed/60:.1f}min | {speed:.1f} iter/min")


# def load_cache_for_year(year: int) -> pd.DataFrame:
#     path = CACHE_DIR / f"city_pairs_{year}.parquet"
#     if not path.exists():
#         raise FileNotFoundError(f"Cache not found: {path}")
#     cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
#     # 优化：直接指定 dtype 避免类型推断开销
#     df = pd.read_parquet(path, columns=cols)
#     df['from_city'] = df['from_city'].astype(np.int32)
#     df['to_city'] = df['to_city'].astype(np.int32)
#     return df

# def build_cross_features(df: pd.DataFrame) -> pd.DataFrame:
#     industry = df['industry'].values if hasattr(df['industry'], 'values') else df['industry'].cat.codes.values
#     wage_cols = ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']
#     vacancy_cols = ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']
#     wage_arr = df[wage_cols].values.astype(np.float32)
#     vacancy_arr = df[vacancy_cols].values.astype(np.float32)
#     ind_idx = np.clip(industry.astype(int), 0, 3)
#     rows = np.arange(len(df))
#     df['industry_x_matched_wage_ratio'] = wage_arr[rows, ind_idx]
#     df['industry_x_matched_vacancy_ratio'] = vacancy_arr[rows, ind_idx]
#     edu = df['education'].values if hasattr(df['education'], 'values') else df['education'].cat.codes.values
#     df['education_x_tier_diff'] = edu.astype(np.float32) * df['tier_diff'].values.astype(np.float32)
#     inc = df['income'].values if hasattr(df['income'], 'values') else df['income'].cat.codes.values
#     df['income_x_gdp_ratio'] = inc.astype(np.float32) * df['gdp_per_capita_ratio'].values
#     age = df['age_group'].values if hasattr(df['age_group'], 'values') else df['age_group'].cat.codes.values
#     df['age_x_housing_ratio'] = age.astype(np.float32) * df['housing_price_avg_ratio'].values
#     fam = df['family'].values if hasattr(df['family'], 'values') else df['family'].cat.codes.values
#     df['family_x_edu_score_ratio'] = fam.astype(np.float32) * df['education_score_ratio'].values
#     return df

# def _get_mem_gb():
#     return psutil.Process().memory_info().rss / 1024**3

# def _process_single_year(year: int, sample_ratio: float, random_seed: int, is_train: bool):
#     base_path = BASE_DIR / f"base_{year}.feather"
#     if not base_path.exists():
#         return None, None

#     # 1. 精简读取
#     needed_cols = ['qid', 'From_City', 'To_City', 'Rank'] + PERSON_CATS
#     base = pd.read_feather(base_path, columns=needed_cols)

#     # 提前计算 Relevance
#     base['Relevance'] = np.where((base['Rank'] > 0) & (base['Rank'] <= 20), 21 - base['Rank'], 0).astype(np.int8)

#     # 提取纯 Numpy 视图以供极速运算
#     qids = base['qid'].values
#     relevance = base['Relevance'].values

#     # =====================================================================
#     # 🚀 极致优化 1：利用排序特性，光速计算 Group 边界 (消灭 groupby size)
#     # =====================================================================
#     # np.nonzero 错位比较，瞬间找出 QID 切换的行索引
#     change_indices = np.nonzero(qids[:-1] != qids[1:])[0] + 1
#     split_indices = np.r_[0, change_indices, len(qids)]
#     group_counts = np.diff(split_indices) # 每个 QID 的行数

#     # =====================================================================
#     # 🚀 极致优化 2：光速过滤无正样本的无效 QID (消灭 transform max)
#     # =====================================================================
#     # reduceat：基于刚刚找出的边界，瞬间求出每个 QID 组内的 max Relevance
#     max_rel_per_group = np.maximum.reduceat(relevance, split_indices[:-1])
#     valid_group_mask = max_rel_per_group > 0

#     if not valid_group_mask.all():
#         # np.repeat 将组级别的 True/False 还原成行级别的 Mask
#         valid_row_mask = np.repeat(valid_group_mask, group_counts)
#         base = base[valid_row_mask].reset_index(drop=True)
#         # 刷新 Numpy 视图与统计
#         qids = base['qid'].values
#         relevance = base['Relevance'].values
#         group_counts = group_counts[valid_group_mask]

#     # =====================================================================
#     # 🚀 极致优化 3：极速 QID 采样
#     # =====================================================================
#     if 0.0 < sample_ratio < 1.0:
#         rng = np.random.default_rng(random_seed)
#         # 从提取好的连续边界中直接取出唯一的 QID (比 np.unique 快)
#         unique_qids = qids[np.r_[0, np.nonzero(qids[:-1] != qids[1:])[0] + 1]]
#         n_sample = int(len(unique_qids) * sample_ratio)
#         sampled_qids = rng.choice(unique_qids, size=n_sample, replace=False)

#         # 过滤并刷新
#         sampled_row_mask = np.isin(qids, sampled_qids)
#         base = base[sampled_row_mask].reset_index(drop=True)
#         qids = base['qid'].values
#         relevance = base['Relevance'].values

#     # =====================================================================
#     # 🚀 极致优化 4：纯 NumPy 极简动态负采样 (消灭复杂的 return_inverse)
#     # =====================================================================
#     if is_train:
#         rng = np.random.default_rng(random_seed + year)
#         is_pos = relevance > 0
#         is_neg = ~is_pos
#         to_city = base['To_City'].values
#         from_city = base['From_City'].values

#         keep = is_pos.copy() # 1. 绝对保留正样本

#         # 2. 静态困难负样本
#         is_static_hard = np.isin(to_city, list(HARD_NEG_CITIES_SET))
#         keep[is_neg & is_static_hard] = True

#         # 3. 动态困难负样本 (同省竞争)
#         to_province = to_city // 100
#         from_province = from_city // 100
#         is_dynamic_hard = is_neg & (to_province == from_province) & (~is_static_hard)
#         keep[is_dynamic_hard] = True

#         # 4. 随机负样本：极简概率法
#         remaining_neg = is_neg & (~keep)
#         if remaining_neg.sum() > 0:
#             qids_rem = qids[remaining_neg]
#             # 过滤后的 qids_rem 依然是严格排序的！再次用错位比较法提速
#             change_indices_rem = np.nonzero(qids_rem[:-1] != qids_rem[1:])[0] + 1
#             split_indices_rem = np.r_[0, change_indices_rem, len(qids_rem)]
#             rem_counts = np.diff(split_indices_rem)

#             # 既然所有 QID 负样本都大于 100，直接求概率：目标数量 / 当前组剩余数量
#             probs = np.minimum(1.0, N_RAND_NEG / rem_counts)
#             row_probs = np.repeat(probs, rem_counts) # 完美映射回行级别

#             # 向量化抛硬币
#             rand_mask = rng.random(len(row_probs)) < row_probs
#             rand_idx = np.where(remaining_neg)[0]
#             keep[rand_idx[rand_mask]] = True

#         # 执行截断
#         base = base[keep].reset_index(drop=True)
#         gc.collect()

#     # =====================================================================
#     # 🚀 必须重算一次 Group Counts (LightGBM 需要最新的 group 规模)
#     # =====================================================================
#     qids_final = base['qid'].values
#     change_indices_final = np.nonzero(qids_final[:-1] != qids_final[1:])[0] + 1
#     split_indices_final = np.r_[0, change_indices_final, len(qids_final)]
#     group_counts = np.diff(split_indices_final).astype(np.int32)

#     # 释放引用
#     del qids, relevance, qids_final; gc.collect()

#     # =====================================================================
#     # 优化 5：合并特征 (利用 sort=False 防止 Pandas 内部重新排序打乱 QID 顺序)
#     # =====================================================================
#     cache = load_cache_for_year(year)
#     base = base.merge(
#         cache, left_on=['From_City', 'To_City'], right_on=['from_city', 'to_city'],
#         how='left', sort=False
#     )
#     base.drop(columns=['from_city', 'to_city'], inplace=True)
#     del cache; gc.collect()

#     # 构建交叉特征
#     base = build_cross_features(base)

#     # 批量填充
#     for col in CACHE_FEAT_COLS + CROSS_FEATS:
#         if col in base.columns:
#             base[col] = base[col].fillna(0).astype(np.float32)

#     return base, group_counts


# def load_data_to_numpy_ltr(years: list, sample_ratio: float, random_seed: int,
#                            is_train: bool, max_expected_rows: int, split_name: str):
#     print(f"\n[{split_name}] Pre-allocating Memory Matrix (Max {max_expected_rows:,} rows)...")
#     t_start = time.time()

#     # 预分配连续内存空间
#     X = np.empty((max_expected_rows, len(FEATS)), dtype=np.float32)
#     y = np.empty(max_expected_rows, dtype=np.int8)
#     all_groups = []
#     current_idx = 0

#     for year in tqdm(years, desc=f"  Loading {split_name}", unit="yr"):
#         t0 = time.time()
#         df, group_counts = _process_single_year(year, sample_ratio, random_seed, is_train)
#         if df is None:
#             continue

#         # 优化：批量转换 category 为整型编码
#         for col in CATS:
#             if str(df[col].dtype) == 'category':
#                 df[col] = df[col].cat.codes

#         rows = len(df)
#         if current_idx + rows > max_expected_rows:
#             raise ValueError(f"Max rows exceeded! Increase `max_expected_rows`.")

#         # 优化：强制转换为 float32，确保 C-Contiguous 内存布局
#         df_feats = df[FEATS].astype(np.float32)
#         X[current_idx : current_idx + rows] = df_feats.values
#         y[current_idx : current_idx + rows] = df['Relevance'].values
#         all_groups.extend(group_counts)

#         current_idx += rows

#         # 立即释放 Pandas 内存
#         del df, df_feats
#         gc.collect()

#         tqdm.write(f"  {year} | Rows: {rows:,} | Total: {current_idx:,} | RSS: {_get_mem_gb():.1f} GB | Time: {time.time()-t0:.1f}s")

#     print(f"[{split_name}] Memory Matrix Ready! Total {current_idx:,} rows. (Time: {(time.time()-t_start)/60:.1f} min)\n")
#     return X[:current_idx], y[:current_idx], np.array(all_groups, dtype=np.int32)

# def main():
#     TRAIN_YEARS = list(range(2000, 2017))
#     VAL_YEARS   = [2017, 2018]
    
#     # v8: 动态负采样策略（静态hard + 动态同省hard + 100 random）
#     # 估算: 平均每 QID 约 172 行（10正 + 50静态hard + 12动态hard + 100random）
#     # 17年 × 329K QID × 0.6 × 172 ≈ 5.8亿行 × 60特征 × 4B ≈ 139GB，预留 6.5亿确保安全
#     TRAIN_SAMPLE_RATIO = 0.6
#     VAL_SAMPLE_RATIO   = 0.2

#     print("=" * 60)
#     print("LightGBM Recall (v8 - rank_xendcg 全局排序优化版)")
#     print(f"RSS at start: {_get_mem_gb():.1f} GB")
#     print("=" * 60)

#     # v8: 动态负采样后预留 6.5 亿行空间
#     TRAIN_MAX_ROWS = 650_000_000
#     VAL_MAX_ROWS   = 60_000_000

#     X_train, y_train, group_train = load_data_to_numpy_ltr(TRAIN_YEARS, TRAIN_SAMPLE_RATIO, 42, is_train=True, max_expected_rows=TRAIN_MAX_ROWS, split_name="Train")
#     X_val, y_val, group_val       = load_data_to_numpy_ltr(VAL_YEARS, VAL_SAMPLE_RATIO, 42, is_train=False, max_expected_rows=VAL_MAX_ROWS, split_name="Val")

#     params = {
#         # v8 核心改动: lambdarank → rank_xendcg
#         # rank_xendcg 基于 softmax 交叉熵近似 NDCG，相比 lambdarank 的 pairwise 交换梯度，
#         # 它对整个列表做全局概率建模，不会过度偏执于头部，能更均衡地优化 1-20 名的排序质量。
#         'objective': 'rank_xendcg',
#         'metric': 'ndcg',
#         'ndcg_eval_at': [5, 10, 20],
#         # v8: 限制优化关注前 20 个位置。GT 最多 20 个正样本，
#         # 超过 20 的位置全是负样本，优化它们的排序没有收益。
#         'lambdarank_truncation_level': 20,
#         'boosting_type': 'gbdt',
#         'learning_rate': 0.05,
#         'num_leaves': 95,               # v8 加速优化：127→95，减少树复杂度，加速训练
#         'max_depth': 9,
#         'n_estimators': 5000,
#         'num_threads': 38,              # v8 加速优化：双路 40 核服务器，留 2 核给系统，避免 NUMA 跨节点瓶颈
#         'max_bin': 127,                 # 保持 127，平衡速度与精度
#         'feature_fraction': 0.7,        # 特征采样：每棵树随机使用 70% 特征
#         'bagging_fraction': 0.8,        # v8 加速优化：数据采样，每棵树使用 80% 数据（按 group 采样）
#         'bagging_freq': 1,              # v8 加速优化：每轮迭代都进行 bagging，加速 20% 且增强泛化
#         'lambda_l1': 0.1,
#         'lambda_l2': 1.0,
#         'min_child_samples': 1000,      # v8 加速优化：500→1000，减少过拟合，加速训练
#         'force_col_wise': True,
#         'verbosity': -1,
#     }

#     print(f"\nHanding over to C++... (Python RSS: {_get_mem_gb():.1f} GB)")
#     train_ds = lgb.Dataset(X_train, label=y_train, group=group_train, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
#     del X_train, y_train, group_train; gc.collect()
#     train_ds.construct()
    
#     val_ds = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_ds, feature_name=list(FEATS), categorical_feature=CATS, params=params, free_raw_data=True)
#     del X_val, y_val, group_val; gc.collect()
#     val_ds.construct()

#     print(f"\nTraining started... RSS: {_get_mem_gb():.1f} GB")
    
#     # 决断 5：恢复 Checkpoint 与 Timing
#     ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
#     timing_cb = TimingCallback(freq=10)

#     model = lgb.train(
#         params, train_ds,
#         valid_sets=[val_ds], valid_names=['val'],
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=50, verbose=True), # 决断 4：50轮早停防过拟合
#             lgb.log_evaluation(10),
#             timing_cb,
#             ckpt_cb,
#         ],
#     )

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     model.save_model(str(OUTPUT_DIR / 'model_v8_xendcg.txt'))
    
#     imp = pd.DataFrame({'feature': FEATS, 'importance': model.feature_importance('gain')}).sort_values('importance', ascending=False)
#     print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")

# if __name__ == '__main__':
#     main()

