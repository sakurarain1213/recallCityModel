"""
0319 全量召回推理导出脚本 — 高性能版 (tl2cgen 编译推理 + C 级内存拼接)

从 output/base_ready/base_{year}.feather 读取预组装数据进行推理，
对每年 (2000-2020) 使用 0319 recall 模型推理 Top40 城市，
输出到 recall_result/{year}.jsonl。

每行格式: {"M_20_EduLo_Agri_IncL_Split_4210": [1100, 3100, ...]}

运行: uv run 0319build-final-json.py
      uv run 0319build-final-json.py --years 2019 2020
   
      uv run 0319build-final-json.py --year-workers 5 --lgb-threads 8


uv run tmp.py --model output/models/0319recall_model_final.so --year-workers 5 --lgb-threads 8 --max-mem-gb 90

"""

import os
# 必须在所有库导入之前! 强行限制底层 OpenMP 线程, 防止多进程 + LightGBM 死锁
# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import ctypes
import multiprocessing as mp
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# tl2cgen 优先, LightGBM 兜底
try:
    import tl2cgen
    HAS_TL2CGEN = True
except ImportError:
    HAS_TL2CGEN = False

import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = Path("output/models/0319recall_model_final.txt")
MODEL_SO_PATH = Path("output/models/0319recall_model_final.so")
CACHE_DIR = Path("data/city_pair_cache")
FEATHER_DIR = Path("output/base_ready")
OUTPUT_DIR = Path("recall_result")

# ═══════════════════════════════════════════════════════════════
# 特征定义 (与训练完全一致)
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
ABS_FEATS = [
    'to_tier', 'to_population_log', 'to_gdp_per_capita',
    'from_tier', 'from_population_log', 'tier_diff',
]
NET_DIST_FEATS = ['migrant_stock_from_to', 'geo_distance', 'dialect_distance', 'is_same_province']
CROSS_FEATS = [
    'industry_x_matched_wage_ratio', 'industry_x_matched_vacancy_ratio',
    'education_x_tier_diff', 'income_x_gdp_ratio',
    'age_x_housing_ratio', 'family_x_edu_score_ratio',
]
CATS = PERSON_CATS + ['is_same_province', 'to_tier', 'from_tier']
FEATS = PERSON_CATS + RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS + CROSS_FEATS
CACHE_FEAT_COLS = RATIO_FEATS + DIFF_FEATS + ABS_FEATS + NET_DIST_FEATS

# 预计算交叉特征列索引
_WAGE_COLS = [CACHE_FEAT_COLS.index(c) for c in
              ['agri_wage_ratio', 'mfg_wage_ratio', 'trad_svc_wage_ratio', 'mod_svc_wage_ratio']]
_VACANCY_COLS = [CACHE_FEAT_COLS.index(c) for c in
                 ['agri_vacancy_ratio', 'mfg_vacancy_ratio', 'trad_svc_vacancy_ratio', 'mod_svc_vacancy_ratio']]
_TIER_DIFF_COL = CACHE_FEAT_COLS.index('tier_diff')
_GDP_RATIO_COL = CACHE_FEAT_COLS.index('gdp_per_capita_ratio')
_HOUSING_RATIO_COL = CACHE_FEAT_COLS.index('housing_price_avg_ratio')
_EDU_SCORE_RATIO_COL = CACHE_FEAT_COLS.index('education_score_ratio')

# TypeID 反向映射
GENDER_REV = {0: 'M', 1: 'F'}
AGE_REV = {0: '20', 1: '30', 2: '40', 3: '55', 4: '65'}
EDU_REV = {0: 'EduLo', 1: 'EduMid', 2: 'EduHi'}
IND_REV = {0: 'Agri', 1: 'Mfg', 2: 'Service', 3: 'Wht'}
INC_REV = {0: 'IncL', 1: 'IncML', 2: 'IncM', 3: 'IncMH', 4: 'IncH'}
FAM_REV = {0: 'Split', 1: 'Unit'}

TOP_K = 40
MEM_PER_WORKER_GB = 3.0

# ═══════════════════════════════════════════════════════════════
# 子进程全局状态
# ═══════════════════════════════════════════════════════════════
_shared_counter = None
_shared_phase = None    # 共享数组: 每个 worker 的阶段 (0=idle, 1=loading, 2=inferring, 3=done)
_shared_cities = None   # 共享数组: 每个 worker 已完成的 city 数
_shared_years = None    # 共享数组: 每个 worker 正在处理的年份
_worker_slot = None     # 当前 worker 在共享数组中的槽位


def _worker_init(counter, phase_arr, cities_arr, years_arr):
    global _shared_counter, _shared_phase, _shared_cities, _shared_years, _worker_slot
    _shared_counter = counter
    _shared_phase = phase_arr
    _shared_cities = cities_arr
    _shared_years = years_arr
    _worker_slot = None  # 在第一次调用 process_year_worker 时分配


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


# ═══════════════════════════════════════════════════════════════
# 核心: C 级连续内存拼接推理
# ═══════════════════════════════════════════════════════════════

def infer_single_city(
    model,
    from_city: int,
    person_arr: np.ndarray,
    to_city_arr: np.ndarray,
    cache_tensor: np.ndarray,
    city_map: np.ndarray,
    lgb_threads: int = 0,
    is_debug: bool = False,
    use_tl2cgen: bool = False,
) -> list[str]:
    """
    C 级连续内存拼接推理。支持 tl2cgen (编译机器码) 和 lgb.Booster (原生) 双模式。
    person_arr: (N, 6) float32, to_city_arr: (C,) int32
    """
    t_start = time.time()
    N = len(person_arr)
    C = len(to_city_arr)

    if is_debug:
        mode = "tl2cgen" if use_tl2cgen else "lgb"
        print(f"\n[Debug] city={from_city} | N={N}, C={C} | matrix={N*C} rows | mode={mode}", flush=True)

    from_mapped = city_map[from_city]
    to_mapped = city_map[to_city_arr]
    c_f = cache_tensor[from_mapped, to_mapped, :]   # (C, 51) float32
    p_f = person_arr                                 # (N, 6) float32

    # 1. 极速大块复制 (底层 C 级 memcpy)
    p_exp = np.repeat(p_f, C, axis=0)       # (N*C, 6)
    c_exp = np.tile(c_f, (N, 1))            # (N*C, 51)

    # 2. 向量化交叉特征
    ind_idx = np.clip(p_f[:, 3].astype(np.intp), 0, 3)

    cr_0 = c_f[:, _WAGE_COLS].T[ind_idx].ravel()
    cr_1 = c_f[:, _VACANCY_COLS].T[ind_idx].ravel()
    cr_2 = (p_f[:, 2:3] * c_f[:, _TIER_DIFF_COL]).ravel()
    cr_3 = (p_f[:, 4:5] * c_f[:, _GDP_RATIO_COL]).ravel()
    cr_4 = (p_f[:, 1:2] * c_f[:, _HOUSING_RATIO_COL]).ravel()
    cr_5 = (p_f[:, 5:6] * c_f[:, _EDU_SCORE_RATIO_COL]).ravel()

    cross_exp = np.column_stack([cr_0, cr_1, cr_2, cr_3, cr_4, cr_5])

    # 3. 合并特征大表 (C 连续 float32)
    X_all = np.concatenate([p_exp, c_exp, cross_exp], axis=1)
    X_all = np.ascontiguousarray(np.nan_to_num(X_all, copy=False, nan=0.0), dtype=np.float32)

    t_build = time.time()
    if is_debug:
        print(f"[Debug] 1.特征构建: {t_build - t_start:.4f}s | shape={X_all.shape}", flush=True)

    # 4. 推理 (tl2cgen 编译机器码 vs LightGBM 原生)
    if use_tl2cgen:
        dmat = tl2cgen.DMatrix(X_all)
        scores = model.predict(dmat).reshape(N, C)
    else:
        predict_kwargs = {'num_threads': lgb_threads} if lgb_threads > 0 else {}
        scores = model.predict(X_all, **predict_kwargs).reshape(N, C)

    t_infer = time.time()
    if is_debug:
        print(f"[Debug] 2.推理: {t_infer - t_build:.4f}s", flush=True)

    # 5. 向量化 Top-K
    if C <= TOP_K:
        top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :TOP_K]
    else:
        top_indices = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
        row_scores = np.take_along_axis(scores, top_indices, axis=1)
        sort_within = np.argsort(row_scores, axis=1)[:, ::-1]
        top_indices = np.take_along_axis(top_indices, sort_within, axis=1)

    top_cities_2d = to_city_arr[top_indices]

    t_sort = time.time()
    if is_debug:
        print(f"[Debug] 3.Top-K: {t_sort - t_infer:.4f}s", flush=True)

    # 6. 手工 JSON 序列化
    lines = []
    g_rev, a_rev, e_rev = GENDER_REV, AGE_REV, EDU_REV
    i_rev, inc_rev, f_rev = IND_REV, INC_REV, FAM_REV
    _map_str = map
    _str = str

    for i in range(N):
        p = person_arr[i]
        tid = (f"{g_rev[int(p[0])]}_{a_rev[int(p[1])]}_{e_rev[int(p[2])]}_"
               f"{i_rev[int(p[3])]}_{inc_rev[int(p[4])]}_{f_rev[int(p[5])]}_{from_city}")
        city_str = ",".join(_map_str(_str, top_cities_2d[i]))
        lines.append(f'{{"{tid}":[{city_str}]}}\n')

    t_json = time.time()
    if is_debug:
        print(f"[Debug] 4.JSON: {t_json - t_sort:.4f}s | 总计: {t_json - t_start:.4f}s\n", flush=True)

    return lines


def load_feather_groups(year: int) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], int]:
    """神速解析 Feather: drop_duplicates 秒速提取, 杜绝 sort_values OOM"""
    feather_path = FEATHER_DIR / f"base_{year}.feather"

    df = pd.read_feather(feather_path, columns=['qid', 'From_City', 'To_City',
                                                  'gender', 'age_group', 'education',
                                                  'industry', 'income', 'family'])

    person_cols = ['gender', 'age_group', 'education', 'industry', 'income', 'family']

    # 直接去重提取人口基准数据 (几百毫秒)
    person_df = df.drop_duplicates(subset=['qid'])

    # 提取候选城市池: 每个 from_city 取第一个 qid 的所有 to_city
    first_qids_per_city = person_df.groupby('From_City', sort=False)['qid'].first().values
    to_city_df = df[df['qid'].isin(set(first_qids_per_city))][['From_City', 'To_City']]

    groups = []
    n_total = len(person_df)

    for from_city, group in person_df.groupby('From_City', sort=False):
        p_arr = group[person_cols].values.astype(np.float32)
        t_arr = to_city_df[to_city_df['From_City'] == from_city]['To_City'].values.astype(np.int32)
        groups.append((int(from_city), p_arr, t_arr))

    del df, person_df, to_city_df
    return groups, n_total


# ═══════════════════════════════════════════════════════════════
# 子进程 worker (独立进程, 绕过 GIL)
# ═══════════════════════════════════════════════════════════════

def process_year_worker(
    year: int,
    model_path: str,
    lgb_threads: int,
    worker_slot: int,
) -> dict:
    """子进程: 加载数据 → 逐 city 推理 → 写临时文件 → 原子 rename"""
    global _shared_counter, _shared_phase, _shared_cities, _shared_years
    t0 = time.time()
    out_path = OUTPUT_DIR / f"{year}.jsonl"
    n_cities_done = 0

    # 更新共享状态: 正在加载
    _shared_years[worker_slot] = year
    _shared_phase[worker_slot] = 1  # loading
    _shared_cities[worker_slot] = 0

    # 根据模型文件后缀自动选择: .so → tl2cgen 编译推理, .txt → LightGBM 原生
    use_tl2cgen = model_path.endswith('.so') and HAS_TL2CGEN
    if use_tl2cgen:
        model = tl2cgen.Predictor(model_path, nthread=lgb_threads)
    else:
        model = lgb.Booster(model_file=model_path)
    t_model = time.time()

    feather_groups, n_total = load_feather_groups(year)
    t_feather = time.time()

    cache_tensor, city_map = load_cache_for_year(year)
    t_cache = time.time()
    n_cities_total = len(feather_groups)

    if worker_slot == 0:
        mode_str = "tl2cgen" if use_tl2cgen else "lgb"
        print(f"[Year {year} slot-0] mode={mode_str} model={t_model-t0:.2f}s feather={t_feather-t_model:.2f}s "
              f"cache={t_cache-t_feather:.2f}s | {n_total} types, {n_cities_total} cities", flush=True)

    # 更新共享状态: 开始推理
    _shared_phase[worker_slot] = 2  # inferring

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix='.jsonl.tmp', prefix=f'{year}_', dir=str(OUTPUT_DIR))
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            for from_city, person_arr, to_city_arr in feather_groups:
                debug_flag = (worker_slot == 0 and n_cities_done == 0)

                lines = infer_single_city(
                    model, from_city, person_arr, to_city_arr,
                    cache_tensor, city_map, lgb_threads=lgb_threads,
                    is_debug=debug_flag,
                    use_tl2cgen=use_tl2cgen,
                )
                f.writelines(lines)

                n_cities_done += 1
                if _shared_counter is not None:
                    with _shared_counter.get_lock():
                        _shared_counter.value += len(person_arr)
                _shared_cities[worker_slot] = n_cities_done

        # 原子替换
        if out_path.exists():
            out_path.unlink()
        os.rename(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    del cache_tensor, city_map, feather_groups

    # 更新共享状态: 完成
    _shared_phase[worker_slot] = 3  # done

    elapsed = time.time() - t0
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    return {
        'year': year, 'n_total': n_total,
        'n_cities': n_cities_total,
        'elapsed_s': elapsed, 'file_size_mb': file_size_mb,
        'output_path': str(out_path),
    }


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="0319 全量召回推理导出 (C 级连续内存拼接版)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2000, 2021)))
    parser.add_argument("--year-workers", type=int, default=0,
                        help="并行进程数, 0=自动 (CPU/4, 受内存限制)")
    parser.add_argument("--lgb-threads", type=int, default=0,
                        help="每个 worker 内 LightGBM 线程数, 0=自动")
    parser.add_argument("--max-mem-gb", type=int, default=90,
                        help="最大允许内存 (GB), 用于限制 worker 数量")
    args = parser.parse_args()

    # 自动选择模型: 优先 .so (tl2cgen 编译), 回退 .txt (LightGBM 原生)
    model_path = Path(args.model)
    if not model_path.exists():
        # 如果传的是 .txt 但 .so 存在, 自动切换
        if model_path.suffix == '.txt' and MODEL_SO_PATH.exists() and HAS_TL2CGEN:
            model_path = MODEL_SO_PATH
            print(f"Auto-detected compiled model: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {args.model}")

    # 如果传的是 .txt 但 .so 也存在, 提示用户
    if model_path.suffix == '.txt' and MODEL_SO_PATH.exists() and HAS_TL2CGEN:
        model_path = MODEL_SO_PATH
        print(f"Switching to compiled model: {model_path}")
    elif model_path.suffix == '.txt' and not MODEL_SO_PATH.exists():
        print("Tip: run 'uv run compile_model.py' to compile model for 3-10x faster inference")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 去重 + 检查文件
    seen_years = set()
    valid_years = []
    for year in args.years:
        if year in seen_years:
            continue
        seen_years.add(year)
        feather_path = FEATHER_DIR / f"base_{year}.feather"
        cache_path = CACHE_DIR / f"city_pairs_{year}.parquet"
        if not feather_path.exists():
            print(f"[Year {year}] SKIP - feather not found: {feather_path}")
        elif not cache_path.exists():
            print(f"[Year {year}] SKIP - cache not found: {cache_path}")
        else:
            valid_years.append(year)

    if not valid_years:
        print("No valid years to process.")
        return

    # 2. 并行配置 (CPU + 内存双重约束)
    cpu_total = os.cpu_count() or 1
    max_by_mem = max(1, int(args.max_mem_gb / MEM_PER_WORKER_GB))

    if args.year_workers > 0:
        year_workers = min(args.year_workers, len(valid_years), max_by_mem)
    else:
        year_workers = min(len(valid_years), max(1, cpu_total // 4), max_by_mem)

    if args.lgb_threads > 0:
        lgb_threads = args.lgb_threads
    else:
        lgb_threads = max(1, cpu_total // year_workers)

    # 预读 TypeID 总数
    sample_df = pd.read_feather(FEATHER_DIR / f"base_{valid_years[0]}.feather", columns=['qid'])
    per_year_total = sample_df['qid'].nunique()
    del sample_df
    grand_total = per_year_total * len(valid_years)

    infer_mode = "tl2cgen (compiled C++)" if str(model_path).endswith('.so') and HAS_TL2CGEN else "LightGBM (native)"
    print(f"Model: {model_path} | Inference: {infer_mode}")
    print(f"Years: {len(valid_years)} | {per_year_total:,} types/year | Total: {grand_total:,}")
    print(f"Features: {len(FEATS)} | C-level memcpy + drop_duplicates")
    print(f"CPU: {cpu_total} | Workers: {year_workers} (processes) | LGB threads/worker: {lgb_threads}")
    print(f"Memory budget: {args.max_mem_gb} GB | ~{MEM_PER_WORKER_GB:.0f} GB/worker")

    # 3. 共享计数器 + 共享状态数组
    counter = mp.Value(ctypes.c_long, 0)
    # 每个 worker 一个槽位: phase / cities_done / year
    phase_arr = mp.Array(ctypes.c_int, year_workers, lock=False)
    cities_arr = mp.Array(ctypes.c_int, year_workers, lock=False)
    years_arr = mp.Array(ctypes.c_int, year_workers, lock=False)

    t_total = time.time()
    results = []

    def build_status_line():
        """构建实时状态行: 显示每个 worker 在干什么"""
        n_loading = 0
        n_inferring = 0
        for i in range(year_workers):
            p = phase_arr[i]
            if p == 1:
                n_loading += 1
            elif p == 2:
                n_inferring += 1
        elapsed = time.time() - t_total
        current = counter.value
        speed = current / elapsed if elapsed > 0 else 0
        remaining = (grand_total - current) / speed if speed > 0 else 0
        return (
            f"{len(results)}/{len(valid_years)}yr done | "
            f"{n_loading}w loading {n_inferring}w inferring | "
            f"{speed:,.0f} type/s | ETA {format_seconds(remaining)}"
        )

    bar = tqdm(total=grand_total, desc="Total", unit="type", unit_scale=True,
               dynamic_ncols=True, mininterval=0.3)

    if year_workers <= 1:
        _worker_init(counter, phase_arr, cities_arr, years_arr)
        for year in valid_years:
            result = process_year_worker(year, str(model_path), lgb_threads, worker_slot=0)
            results.append(result)
            bar.n = counter.value
            bar.set_postfix_str(f"Year {year} done ({format_seconds(result['elapsed_s'])})")
            bar.refresh()
    else:
        with ProcessPoolExecutor(
            max_workers=year_workers,
            initializer=_worker_init,
            initargs=(counter, phase_arr, cities_arr, years_arr),
        ) as executor:
            futures = {}
            for slot, year in enumerate(valid_years[:year_workers]):
                f = executor.submit(
                    process_year_worker, year, str(model_path), lgb_threads,
                    worker_slot=slot,
                )
                futures[f] = (year, slot)

            next_year_idx = year_workers
            free_slots = []

            pending = set(futures)
            while pending:
                time.sleep(0.3)
                bar.n = counter.value
                bar.set_postfix_str(build_status_line())
                bar.refresh()

                done = [f for f in pending if f.done()]
                for f in done:
                    pending.remove(f)
                    year, slot = futures[f]
                    result = f.result()
                    results.append(result)
                    free_slots.append(slot)

                    tqdm.write(
                        f"  [Year {result['year']}] {format_seconds(result['elapsed_s'])} | "
                        f"{result['n_cities']} cities | {result['file_size_mb']:.1f} MB | "
                        f"{len(results)}/{len(valid_years)} years done"
                    )

                # 把空闲槽位分配给剩余年份
                while free_slots and next_year_idx < len(valid_years):
                    slot = free_slots.pop()
                    year = valid_years[next_year_idx]
                    next_year_idx += 1
                    phase_arr[slot] = 0
                    cities_arr[slot] = 0
                    years_arr[slot] = 0
                    f = executor.submit(
                        process_year_worker, year, str(model_path), lgb_threads,
                        worker_slot=slot,
                    )
                    futures[f] = (year, slot)
                    pending.add(f)

            bar.n = counter.value
            bar.set_postfix_str(build_status_line())
            bar.refresh()

    bar.close()

    results.sort(key=lambda r: r['year'])

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print("Per-year summary:")
    for result in results:
        print(
            f"  [Year {result['year']}] {format_seconds(result['elapsed_s'])} | "
            f"{result['file_size_mb']:.1f} MB | {result['output_path']}"
        )
    print(f"\nAll done! {len(results)} years in {format_seconds(total_elapsed)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
