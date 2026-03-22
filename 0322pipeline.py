"""
0322 LambdaRank Pipeline (全量自包含)

上传到服务器后只需这一个脚本，改路径即可运行完整流程。

Usage:
  uv run 0322pipeline.py --step bin,train,eval
  uv run 0322pipeline.py --step bin,train
  uv run 0322pipeline.py --step eval --model output/models/0322ltr_model.txt

服务器迁移: 只需修改顶部的 DB_PATH / CACHE_DIR 路径即可

uv run 0322pipeline.py --step train --train-query-ratio 0.3 --train-neg-ratio 0.8
"""

import gc
import re
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# 路径配置 (自动检测平台，服务器迁移无需修改)
# ═══════════════════════════════════════════════════════════════
import os as _os
if _os.name == 'nt':
    # Windows 本地
    DB_PATH = Path("C:/Users/w1625/Desktop/local_migration_data.db")
    CACHE_DIR = Path("data/city_pair_cache")
    BIN_DIR = Path("output/ltr_bin")
    MODEL_DIR = Path("output/models")
else:
    # 服务器 Linux
    DB_PATH = Path("/data1/wxj/Recall_city_project/data/local_migration_data.db")
    CACHE_DIR = Path("/data1/wxj/Recall_city_project/data/city_pair_cache")
    BIN_DIR = Path("/data2/wxj/ltr_bin")
    MODEL_DIR = Path("/data1/wxj/Recall_city_project/output/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════

class CheckpointCallback:
    """每 N 轮保存一次 checkpoint"""
    def __init__(self, output_dir, freq=20):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        self.before_iteration = False

    def __call__(self, env):
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            ckpt_path = self.output_dir / f"ltr_model_iter_{current_round}.txt"
            env.model.save_model(str(ckpt_path))
            print(f"    [Checkpoint] iter {current_round} -> {ckpt_path.name}")


class TimingCallback:
    """计时器：显示速度和预估剩余时间"""
    def __init__(self, freq=20, total_iters=2000):
        self.freq = freq
        self.total_iters = total_iters
        self.start_time = None
        self.before_iteration = False

    def __call__(self, env):
        if self.start_time is None:
            self.start_time = time.time()
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            elapsed = time.time() - self.start_time
            speed = current_round / elapsed * 60
            remaining = (self.total_iters - current_round) / speed if speed > 0 else 0
            print(f"    [Timer] iter {current_round}/{self.total_iters} | "
                  f"elapsed {elapsed/60:.1f}min | "
                  f"speed {speed:.1f} iter/min | "
                  f"ETA ~{remaining:.1f}min")


# ═══════════════════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════════════════
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
ABS_FEATS = ['to_tier', 'to_population_log', 'to_gdp_per_capita', 'from_tier', 'from_population_log', 'tier_diff']
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = [
    'industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff', 'income_x_gdp_ratio',
    'age_x_housing_ratio', 'family_x_edu_score_ratio',
]
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# TypeID 解析
AGE_MAP = {'20': 0, '30': 1, '40': 2, '55': 3, '65': 4}
EDU_MAP = {'EduLo': 0, 'EduMid': 1, 'EduHi': 2}
IND_MAP = {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3}
INC_MAP = {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4}
FAM_MAP = {'Split': 0, 'Unit': 1}
GENDER_MAP = {'M': 0, 'F': 1}


def parse_type_id(tid: str) -> tuple:
    parts = tid.rsplit('_', 1)
    fc = int(parts[1])
    a = parts[0].split('_')
    return (GENDER_MAP[a[0]], AGE_MAP[a[1]], EDU_MAP[a[2]],
            IND_MAP[a[3]], INC_MAP[a[4]], FAM_MAP[a[5]], fc)


def parse_to_city(s: str) -> int:
    m = re.search(r'\((\d+)\)', s)
    return int(m.group(1)) if m else int(s)


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_cache(year: int):
    path = CACHE_DIR / f"city_pairs_{year}.parquet"
    cols = ['from_city', 'to_city'] + CACHE_FEAT_COLS
    df = pd.read_parquet(path, columns=cols)

    all_ids = np.unique(np.concatenate([df['from_city'].values, df['to_city'].values]))
    max_id = int(all_ids.max())
    n_feat = len(CACHE_FEAT_COLS)

    city_map = np.zeros(max_id + 1, dtype=np.int32)
    city_map[all_ids] = np.arange(len(all_ids))

    tensor = np.zeros((max_id + 1, max_id + 1, n_feat), dtype=np.float32)
    tensor[city_map[df['from_city'].values], city_map[df['to_city'].values]] = \
        df[CACHE_FEAT_COLS].values.astype(np.float32)

    to_dict = df.groupby('from_city')['to_city'].apply(
        lambda x: x.values.astype(np.int32)).to_dict()

    return tensor, city_map, to_dict


def build_year_data(year: int, sample_n: int = None, seed: int = 42):
    """从 DB 加载 year 数据，返回 (X, labels, qids, n_queries)"""
    t0 = time.time()
    top_cols = ', '.join([f'To_Top{i}' for i in range(1, 21)])
    sql = f"SELECT Type_ID, {top_cols} FROM migration_data WHERE Year = {year}"

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    df_raw = conn.execute(sql).fetchdf()
    conn.close()

    n_q = len(df_raw)
    print(f"  DB loaded {n_q:,} Type_IDs ({time.time()-t0:.1f}s)")

    if sample_n and sample_n < n_q:
        idx = np.random.default_rng(seed).choice(n_q, size=sample_n, replace=False)
        df_raw = df_raw.iloc[idx].reset_index(drop=True)
        n_q = sample_n

    tensor, city_map, to_dict = load_cache(year)
    print(f"  Cache: {tensor.shape}")

    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    pos_cities = df_raw[top_cols].map(parse_to_city).values
    type_ids = df_raw['Type_ID'].values
    persons = np.array([list(parse_type_id(t)[:6]) for t in type_ids], dtype=np.float32)
    from_cities = np.array([parse_type_id(t)[6] for t in type_ids], dtype=np.int32)

    # 交叉特征列索引
    wage_i = [RATIO_FEATS.index(c) for c in
              ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']]
    vac_i = [RATIO_FEATS.index(c) for c in
             ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']]
    tier_i = len(RATIO_FEATS) + ABS_FEATS.index('tier_diff')
    gdp_i = RATIO_FEATS.index('gdp_per_capita_ratio')
    hous_i = RATIO_FEATS.index('housing_price_avg_ratio')
    edu_i = RATIO_FEATS.index('education_score_ratio')

    total = n_q * 336
    X = np.zeros((total, len(FEATS)), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int8)
    qids = np.zeros(total, dtype=np.int32)

    row = 0
    t1 = time.time()
    # 预估时间
    est_time_per_q = 0.0005  # 每query预估秒数，根据数据规模调整
    est_total = n_q * est_time_per_q
    print(f"  预估耗时: {est_total:.1f}s (~{est_total/60:.1f}min)")

    for q in tqdm(range(n_q), desc=f"Build {year}", unit="query"):
        fc = from_cities[q]
        pos = set(pos_cities[q])
        all_to = to_dict.get(fc, np.array([], dtype=np.int32))

        if len(all_to) == 0:
            n_q -= 1
            continue

        if len(all_to) < 336:
            all_to = np.pad(all_to, (0, 336 - len(all_to)), constant_values=fc)
        else:
            all_to = all_to[:336]

        pf = tensor[fc, city_map[all_to], :]
        pf = np.nan_to_num(pf, nan=0.0)
        pf_base = np.concatenate([np.tile(persons[q], (336, 1)), pf], axis=1)

        ind = int(persons[q][3])
        cross = np.zeros((336, 6), dtype=np.float32)
        cross[:, 0] = pf[:, wage_i[min(ind, 3)]]
        cross[:, 1] = pf[:, vac_i[min(ind, 3)]]
        cross[:, 2] = persons[q][2] * pf[:, tier_i]
        cross[:, 3] = persons[q][4] * pf[:, gdp_i]
        cross[:, 4] = persons[q][1] * pf[:, hous_i]
        cross[:, 5] = persons[q][5] * pf[:, edu_i]

        feat = np.ascontiguousarray(np.concatenate([pf_base, cross], axis=1), dtype=np.float32)
        lbl = np.array([1 if tc in pos else 0 for tc in all_to], dtype=np.int8)

        X[row:row+336] = feat
        labels[row:row+336] = lbl
        qids[row:row+336] = q
        row += 336

    X = X[:row]; labels = labels[:row]; qids = qids[:row]
    print(f"  Built {len(X):,} rows, {row//336:,} queries ({time.time()-t1:.1f}s)")
    del tensor, city_map, to_dict; gc.collect()
    return X, labels, qids, row // 336


# ═══════════════════════════════════════════════════════════════
# Bin 构建
# ═══════════════════════════════════════════════════════════════

def build_bin(year: int, is_train: bool, sample_n: int = None):
    print(f"\n[{'Train' if is_train else 'Val'} {year}]")
    suffix = "Train" if is_train else "Val"
    X, labels, qids, n_q = build_year_data(year, sample_n=sample_n)

    print(f"  构建 Dataset...")
    groups = np.full(n_q, 336, dtype=np.int32)
    n_rows = len(X)

    ds = lgb.Dataset(X, label=labels, feature_name=list(FEATS),
                     categorical_feature=CATS, free_raw_data=True,
                     params={'max_bin': 255})
    ds.set_group(groups)
    ds.construct()

    suffix = "train" if is_train else "val"
    bin_path = BIN_DIR / f"{suffix}_{year}.bin"
    ds.save_binary(str(bin_path))

    if not is_train:
        np.save(BIN_DIR / f"val_labels_{year}.npy", labels)
        np.save(BIN_DIR / f"val_groups_{year}.npy", groups)

    print(f"  Saved {bin_path.name}: {n_rows:,} rows, {n_q:,} groups")
    del X, labels, qids; gc.collect()
    return bin_path, n_rows, n_q


# ═══════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════

def train(train_bin_info, val_years, query_ratio=1.0, neg_ratio=1.0):
    """train_bin_info: list of (bin_path, n_rows, n_q)"""
    t0 = time.time()
    print("\n=== Train ===")

    # 核心技巧：用逗号拼接多个 bin 文件路径，LightGBM C++ 底层自动合并
    bin_paths = [str(p) for p, _, _ in train_bin_info]
    train_ds = lgb.Dataset(",".join(bin_paths))

    # 必须先 construct 载入内存，否则无法获取 label 和 group 进行动态切分
    print("  Constructing Dataset (加载合并的 bin 文件中，请稍候)...")
    train_ds.construct()
    total_train = train_ds.num_data()
    print(f"  Original Train Dataset: {total_train:,} rows")

    # ═══════════════════════════════════════════════════════════════
    # 动态采样逻辑 (Query 级抽样 + 组内负采样)
    # ═══════════════════════════════════════════════════════════════
    if query_ratio < 1.0 or neg_ratio < 1.0:
        print(f"  => 动态采样中 (Query Ratio: {query_ratio:.2f}, Neg Ratio: {neg_ratio:.2f})...")
        labels = train_ds.get_label()
        groups = train_ds.get_group()
        splits = np.r_[0, np.cumsum(groups)]
        n_queries = len(groups)

        # 1. 抽取特定比例的 Query
        if query_ratio < 1.0:
            n_select = int(n_queries * query_ratio)
            selected_queries = np.random.choice(n_queries, size=n_select, replace=False)
            selected_queries.sort()
        else:
            selected_queries = np.arange(n_queries)

        # 2. 负采样 (遍历选中的 Query，保证正样本全保留)
        used_indices = []
        for q in tqdm(selected_queries, desc="  Sampling rows", unit="query"):
            start, end = splits[q], splits[q+1]
            q_labels = labels[start:end]

            # 正样本索引 (Label > 0)
            pos_idx = np.where(q_labels > 0)[0] + start
            # 负样本索引 (Label == 0)
            neg_idx = np.where(q_labels == 0)[0] + start

            if neg_ratio < 1.0 and len(neg_idx) > 0:
                n_neg = int(len(neg_idx) * neg_ratio)
                if n_neg > 0:
                    neg_idx = np.random.choice(neg_idx, size=n_neg, replace=False)
                else:
                    neg_idx = np.array([], dtype=int)

            used_indices.append(pos_idx)
            if len(neg_idx) > 0:
                used_indices.append(neg_idx)

        if used_indices:
            used_indices = np.concatenate(used_indices)
            used_indices.sort()  # LightGBM subset 要求索引有序
            # subset 会根据我们抽取的行，自动重新计算并设置 group
            train_ds = train_ds.subset(used_indices).construct()
            total_train = train_ds.num_data()
            print(f"  => 采样完成! Sampled Train Dataset: {total_train:,} rows")
        else:
            raise ValueError("采样后没有任何样本留下，请检查采样比例参数！")

    # 训练时用 train 自身监控，不用 val（避免 bin mapper 不一致问题）
    # Recall@K 在 evaluate() 中单独计算
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20, 40],
        'lambdarank_truncation_level': 20,
        'label_gain': '0,1',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': 10,
        'n_estimators': 2000,
        'num_threads': 16,
        'max_bin': 255,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 2.0,
        'min_child_samples': 200,
        'force_col_wise': True,
        'verbosity': -1,
    }

    # 预估训练时间
    n_estimators = 2000
    print(f"  预估训练时间: ~{total_train/1000000*3:.0f}-{total_train/1000000*5:.0f}分钟 (基于{n_estimators}轮, lr=0.03)")

    ckpt_cb = CheckpointCallback(output_dir=MODEL_DIR, freq=20)
    timing_cb = TimingCallback(freq=20, total_iters=n_estimators)

    model = lgb.train(
        params, train_ds,
        valid_sets=[train_ds], valid_names=['train'],
        callbacks=[
            lgb.log_evaluation(20),
            ckpt_cb,
            timing_cb,
        ],
    )

    model_path = MODEL_DIR / "0322ltr_model.txt"
    model.save_model(str(model_path))
    print(f"\n  Model: {model_path}")
    print(f"  Train time: {time.time()-t0:.1f}s")

    imp = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance('gain'),
    }).sort_values('importance', ascending=False)
    print(f"\nTop 10 Features:")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']}: {r['importance']:.0f}")

    return model_path


# ═══════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════

def evaluate(model_path: str, years, sample_n=5000):
    print(f"\n=== Evaluate ({years}) ===")
    model = lgb.Booster(model_file=model_path)

    all_results = []
    for year in years:
        t0 = time.time()
        X, labels, qids, n_q = build_year_data(year, sample_n=sample_n)
        scores = model.predict(X)

        groups = np.full(n_q, 336, dtype=np.int32)
        splits = np.r_[0, np.cumsum(groups)]
        lbl_sp = np.split(labels, splits[1:-1])
        sc_sp = np.split(scores, splits[1:-1])

        valid = [i for i, l in enumerate(lbl_sp) if l.sum() >= 1]
        r = {'year': year}
        for k in [5, 10, 20, 40]:
            total = 0.0
            for i in valid:
                l, p = lbl_sp[i], sc_sp[i]
                topk = np.argpartition(p, -min(k, len(p)))[-min(k, len(p)):]
                total += np.sum(l[topk]) / min(k, int(l.sum()))
            r[f'R@{k}'] = total / len(valid)

        print(f"  {year}: R@5={r['R@5']:.4f} R@10={r['R@10']:.4f} "
              f"R@20={r['R@20']:.4f} ({time.time()-t0:.1f}s)")
        all_results.append(r)
        del X, labels, qids; gc.collect()

    # 汇总
    print(f"\n{'='*50}")
    print(f"{'Year':<8} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'R@40':<10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['year']:<8} {r['R@5']:<10.4f} {r['R@10']:<10.4f} "
              f"{r['R@20']:<10.4f} {r['R@40']:<10.4f}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="0322 LambdaRank Pipeline")
    p.add_argument("--step", type=str, default="all",
                   help="步骤: bin,train,eval 或 all")
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2000, 2019)))
    p.add_argument("--val-years", type=int, nargs="+", default=[2019])
    p.add_argument("--eval-years", type=int, nargs="+", default=[2020])

    # Bin构建：默认全量，不采样
    p.add_argument("--sample", type=int, default=None,
                   help="训练集每 year 采样 query 数 (默认 None 即全量)")
    p.add_argument("--val-sample", type=int, default=None,
                   help="验证集每 year 采样 query 数 (默认 None 即全量)")

    # 训练时的动态比例配置
    p.add_argument("--train-query-ratio", type=float, default=1.0,
                   help="训练时抽取的 Query 比例, 0~1 (默认 1.0 全量)")
    p.add_argument("--train-neg-ratio", type=float, default=1.0,
                   help="训练时的负采样比例, 0~1 (默认 1.0 不过滤，正样本永远100%%保留)")

    p.add_argument("--eval-sample", type=int, default=5000)
    p.add_argument("--model", type=str, default=None,
                   help="指定模型路径 (eval 用)")
    args = p.parse_args()

    steps = [s.strip() for s in args.step.split(",")]
    do_all = "all" in steps

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if do_all or "bin" in steps:
        if BIN_DIR.exists() and any(BIN_DIR.iterdir()):
            shutil.rmtree(BIN_DIR)
            BIN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[CLEAN] {BIN_DIR}")

    # Step 1: Bin
    if do_all or "bin" in steps:
        print("\n### STEP: Build Bin ###")
        train_bins = []
        for y in args.train_years:
            bp = build_bin(y, is_train=True, sample_n=args.sample)
            train_bins.append(bp)
        val_bins = []
        for y in args.val_years:
            bp = build_bin(y, is_train=False, sample_n=args.val_sample)
            val_bins.append(bp)
    else:
        train_bins = [(BIN_DIR / f"train_{y}.bin", 0, 0) for y in args.train_years]
        val_bins = [(BIN_DIR / f"val_{y}.bin", 0, 0) for y in args.val_years]

    # Step 2: Train
    if do_all or "train" in steps:
        print("\n### STEP: Train ###")
        model_path = train(
            train_bins, args.val_years,
            query_ratio=args.train_query_ratio,
            neg_ratio=args.train_neg_ratio
        )
    else:
        model_path = args.model or str(MODEL_DIR / "0322ltr_model.txt")

    # Step 3: Eval
    if do_all or "eval" in steps:
        print("\n### STEP: Eval ###")
        evaluate(model_path, args.eval_years, sample_n=args.eval_sample)

    print(f"\n[OK] Done! Model: {model_path}")


if __name__ == '__main__':
    main()
