"""
从 recall_result/{year}.jsonl 直接评估 Recall@K (无推理)
用法: python temp.py
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
YEAR = 2009                # 手动指定年份 (2000-2007)
SAMPLE_N = 10000                # 随机采样 type 数
SEED = 42
K_LIST = [5, 10, 20, 30, 40]

JSONL_DIR = Path("recall_result")
FEATHER_DIR = Path("output/base_ready")

# type_id 编码映射 (与 0319build-final-json.py 一致)
GENDER_REV = {0: 'M', 1: 'F'}
AGE_REV = {0: '20', 1: '30', 2: '40', 3: '55', 4: '65'}
EDU_REV = {0: 'EduLo', 1: 'EduMid', 2: 'EduHi'}
IND_REV = {0: 'Agri', 1: 'Mfg', 2: 'Service', 3: 'Wht'}
INC_REV = {0: 'IncL', 1: 'IncML', 2: 'IncM', 3: 'IncMH', 4: 'IncH'}
FAM_REV = {0: 'Split', 1: 'Unit'}


def build_gt_dict(year: int) -> dict[str, set[int]]:
    """从 feather 构建 GT: type_id -> set of cities (Rank 1-20), 向量化构建"""
    print(f"  Loading feather for year {year}...")
    df = pd.read_feather(
        FEATHER_DIR / f"base_{year}.feather",
        columns=['qid', 'From_City', 'To_City', 'Rank',
                 'gender', 'age_group', 'education', 'industry', 'income', 'family'],
    )

    # 向量化构建 type_id (避免 iterrows)
    person = df.drop_duplicates(subset=['qid']).copy()
    g_map = person['gender'].map(GENDER_REV)
    a_map = person['age_group'].map(AGE_REV)
    e_map = person['education'].map(EDU_REV)
    i_map = person['industry'].map(IND_REV)
    inc_map = person['income'].map(INC_REV)
    f_map = person['family'].map(FAM_REV)
    city_str = person['From_City'].astype(int).astype(str)
    person['type_id'] = g_map + '_' + a_map + '_' + e_map + '_' + i_map + '_' + inc_map + '_' + f_map + '_' + city_str
    qid_to_tid = dict(zip(person['qid'], person['type_id']))

    # GT: Rank 1-20, 按 qid 聚合
    gt_df = df[df['Rank'].between(1, 20)]
    gt_grouped = gt_df.groupby('qid')['To_City'].apply(set).to_dict()

    gt_dict = {qid_to_tid[qid]: cities for qid, cities in gt_grouped.items() if qid in qid_to_tid}
    print(f"  GT built: {len(gt_dict)} types, avg GT size: {np.mean([len(v) for v in gt_dict.values()]):.1f}")
    return gt_dict


def load_predictions(year: int) -> dict[str, list[int]]:
    """从 jsonl 加载预测结果: type_id -> list of 40 cities (有序)"""
    path = JSONL_DIR / f"{year}.jsonl"
    print(f"  Loading predictions from {path}...")
    preds = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            for tid, cities in obj.items():
                preds[tid] = cities
    print(f"  Loaded {len(preds)} predictions")
    return preds


def compute_recall(gt_dict: dict[str, set[int]], preds: dict[str, list[int]],
                   sample_tids: list[str]) -> None:
    """计算并打印 Recall@K"""
    results = {k: [] for k in K_LIST}
    n_eval = 0
    skipped = 0

    for tid in sample_tids:
        gt_set = gt_dict.get(tid)
        pred_list = preds.get(tid)
        if gt_set is None or pred_list is None:
            skipped += 1
            continue
        n_gt = len(gt_set)
        if n_gt == 0:
            skipped += 1
            continue
        n_eval += 1
        for k in K_LIST:
            top_k = set(pred_list[:k])
            hits = len(top_k & gt_set)
            recall = hits / min(k, n_gt)
            results[k].append(recall)

    print(f"\n{'='*60}")
    print(f"Year: {YEAR} | Sampled: {len(sample_tids)} | Evaluated: {n_eval} | Skipped: {skipped}")
    print(f"{'='*60}")
    for k in K_LIST:
        vals = results[k]
        if vals:
            avg = np.mean(vals)
            print(f"  Recall@{k:<3d} {avg:.4f} ({avg*100:.2f}%)")
    print(f"{'='*60}")


def main():
    t0 = time.time()
    random.seed(SEED)
    print(f"Year: {YEAR} | Sample: {SAMPLE_N} | Seed: {SEED}")

    # 1. 加载 GT
    gt_dict = build_gt_dict(YEAR)

    # 2. 加载预测
    preds = load_predictions(YEAR)

    # 3. 取交集后随机采样
    common_tids = list(set(gt_dict.keys()) & set(preds.keys()))
    print(f"  Common types: {len(common_tids)}")

    if len(common_tids) > SAMPLE_N:
        sample_tids = random.sample(common_tids, SAMPLE_N)
    else:
        sample_tids = common_tids
        print(f"  Warning: only {len(common_tids)} common types, using all")

    # 4. 计算 Recall
    compute_recall(gt_dict, preds, sample_tids)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
