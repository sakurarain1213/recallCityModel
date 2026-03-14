"""
城市对特征缓存生成脚本
从 cities_YYYY.jsonl + city_edges.jsonl 生成每年的城市对特征 Parquet 文件。

输出: data/city_pair_cache/city_pairs_YYYY.parquet
每行 = 一个 (from_city, to_city) 对，包含 ~50 个特征。

运行: cd /data1/wxj/Recall_city_project/ && uv run generate_city_pair_cache.py
依赖: data/cities_2000-2020/cities_YYYY.jsonl, data/city_edges.jsonl
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

DATA_DIR = Path("data")
CITY_INFO_DIR = DATA_DIR / "cities_2000-2020"
EDGES_PATH = DATA_DIR / "city_edges.jsonl"
OUTPUT_DIR = DATA_DIR / "city_pair_cache"

# 安全比率: clamp 分母避免除零, clamp 结果避免极值
RATIO_CLIP_MIN = 0.01
RATIO_CLIP_MAX = 100.0


def safe_ratio(to_vals: np.ndarray, from_vals: np.ndarray) -> np.ndarray:
    """安全比率计算: to / from, clamp 避免极值"""
    denom = np.where(np.abs(from_vals) < 1e-9, 1e-9, from_vals)
    ratio = to_vals / denom
    return np.clip(ratio, RATIO_CLIP_MIN, RATIO_CLIP_MAX).astype(np.float32)


def safe_diff(to_vals: np.ndarray, from_vals: np.ndarray) -> np.ndarray:
    """安全差值计算"""
    diff = (to_vals - from_vals).astype(np.float32)
    return np.clip(diff, -1e8, 1e8)


def load_city_info_for_year(year: int) -> pd.DataFrame:
    """加载单年城市信息, 展平为 DataFrame, city_id 为 int index"""
    path = CITY_INFO_DIR / f"cities_{year}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"City info not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sectors = item["economy"].get("industry_sectors", {})
            age = item["demographics"].get("age_structure", {})
            msd = item.get("social_context", {}).get("migrant_stock_distribution", {})

            rec = {
                "city_id": int(item["city_id"]),
                # 基本
                "tier": item["basic_info"]["tier"],
                "area_sqkm": item["basic_info"]["area_sqkm"],
                # 经济
                "gdp_per_capita": item["economy"]["gdp_per_capita"],
                "cpi_index": item["economy"]["cpi_index"],
                "unemployment_rate": item["economy"]["unemployment_rate"],
                # 产业-农业
                "agri_share": sectors.get("agriculture", {}).get("share", 0),
                "agri_wage": sectors.get("agriculture", {}).get("avg_wage", 0),
                "agri_vacancy": sectors.get("agriculture", {}).get("vacancy_rate", 0),
                # 产业-制造
                "mfg_share": sectors.get("manufacturing", {}).get("share", 0),
                "mfg_wage": sectors.get("manufacturing", {}).get("avg_wage", 0),
                "mfg_vacancy": sectors.get("manufacturing", {}).get("vacancy_rate", 0),
                # 产业-传统服务
                "trad_svc_share": sectors.get("traditional_services", {}).get("share", 0),
                "trad_svc_wage": sectors.get("traditional_services", {}).get("avg_wage", 0),
                "trad_svc_vacancy": sectors.get("traditional_services", {}).get("vacancy_rate", 0),
                # 产业-现代服务
                "mod_svc_share": sectors.get("modern_services", {}).get("share", 0),
                "mod_svc_wage": sectors.get("modern_services", {}).get("avg_wage", 0),
                "mod_svc_vacancy": sectors.get("modern_services", {}).get("vacancy_rate", 0),
                # 人口结构
                "age_0_17": age.get("0_17", 0),
                "age_18_34": age.get("18_34", 0),
                "age_35_54": age.get("35_54", 0),
                "age_55_64": age.get("55_64", 0),
                "age_65_plus": age.get("65_plus", 0),
                "sex_ratio": item["demographics"].get("sex_ratio", 100),
                # 生活成本
                "housing_price_avg": item["living_cost"]["housing_price_avg"],
                "rent_avg": item["living_cost"]["rent_avg"],
                "daily_cost_index": item["living_cost"]["daily_cost_index"],
                # 公共服务
                "medical_score": item["public_services"]["medical_score"],
                "education_score": item["public_services"]["education_score"],
                "transport_convenience": item["public_services"]["transport_convenience"],
                "avg_commute_mins": item["public_services"]["avg_commute_mins"],
                # 人口总量
                "population_total": item["social_context"]["population_total"],
            }
            # migrant_stock_distribution 存为 dict, 后续按需查询
            rec["_migrant_stock"] = msd
            records.append(rec)

    df = pd.DataFrame(records)
    df["city_id"] = df["city_id"].astype(np.int32)
    df.set_index("city_id", inplace=True)
    return df


def load_edges() -> pd.DataFrame:
    """加载城市边: (source_id, target_id, w_geo, w_dialect)"""
    records = []
    with open(EDGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            records.append({
                "from_city": int(e["source_id"]),
                "to_city": int(e["target_id"]),
                "geo_distance": float(e["w_geo"]),
                "dialect_distance": float(e["w_dialect"]),
            })
    df = pd.DataFrame(records)
    df["from_city"] = df["from_city"].astype(np.int32)
    df["to_city"] = df["to_city"].astype(np.int32)
    df["geo_distance"] = df["geo_distance"].astype(np.float32)
    df["dialect_distance"] = df["dialect_distance"].astype(np.float32)
    return df


def build_city_pairs_for_year(year: int, city_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建单年的城市对特征 DataFrame。
    每行 = (from_city, to_city), from_city != to_city。
    """
    city_ids = city_df.index.values  # int32 array
    n = len(city_ids)

    # --- 1. 构建全连接对 (排除自身) ---
    from_ids = np.repeat(city_ids, n - 1)
    to_ids_list = []
    for i in range(n):
        to_ids_list.append(np.concatenate([city_ids[:i], city_ids[i+1:]]))
    to_ids = np.concatenate(to_ids_list)

    pairs = pd.DataFrame({"from_city": from_ids, "to_city": to_ids})
    pairs["from_city"] = pairs["from_city"].astype(np.int32)
    pairs["to_city"] = pairs["to_city"].astype(np.int32)
    print(f"  [{year}] Total pairs: {len(pairs):,} ({n} cities, {n-1} targets each)")

    # --- 2. 准备城市属性 lookup (不含 _migrant_stock) ---
    migrant_stocks = city_df["_migrant_stock"].to_dict()  # {city_id: {province: ratio}}
    numeric_cols = [c for c in city_df.columns if c != "_migrant_stock"]
    city_arr = city_df[numeric_cols]

    # 用 from/to index 做向量化 lookup
    from_vals = city_arr.loc[pairs["from_city"].values]
    to_vals = city_arr.loc[pairs["to_city"].values]
    from_vals.index = pairs.index
    to_vals.index = pairs.index

    # --- 3. 比率特征 (to / from) ---
    ratio_cols = [
        "gdp_per_capita", "cpi_index", "unemployment_rate",
        "agri_share", "agri_wage", "agri_vacancy",
        "mfg_share", "mfg_wage", "mfg_vacancy",
        "trad_svc_share", "trad_svc_wage", "trad_svc_vacancy",
        "mod_svc_share", "mod_svc_wage", "mod_svc_vacancy",
        "housing_price_avg", "rent_avg", "daily_cost_index",
        "medical_score", "education_score", "transport_convenience",
        "avg_commute_mins", "population_total",
        "age_0_17", "age_18_34", "age_35_54", "age_55_64", "age_65_plus",
        "sex_ratio", "area_sqkm",
    ]
    for col in ratio_cols:
        pairs[f"{col}_ratio"] = safe_ratio(
            to_vals[col].values.astype(np.float64),
            from_vals[col].values.astype(np.float64),
        )

    # --- 4. 差值特征 (to - from), 选择经济学意义明确的 ---
    diff_cols = [
        "gdp_per_capita", "housing_price_avg", "rent_avg",
        "agri_wage", "mfg_wage", "trad_svc_wage", "mod_svc_wage",
        "agri_vacancy", "mfg_vacancy", "trad_svc_vacancy", "mod_svc_vacancy",
    ]
    for col in diff_cols:
        pairs[f"{col}_diff"] = safe_diff(
            to_vals[col].values.astype(np.float64),
            from_vals[col].values.astype(np.float64),
        )

    # --- 5. 目的地绝对特征 ---
    pairs["to_tier"] = to_vals["tier"].values.astype(np.int8)
    pairs["to_population_log"] = np.log1p(to_vals["population_total"].values).astype(np.float32)
    pairs["to_gdp_per_capita"] = to_vals["gdp_per_capita"].values.astype(np.float32)

    # --- 6. 出发地绝对特征 ---
    pairs["from_tier"] = from_vals["tier"].values.astype(np.int8)
    pairs["from_population_log"] = np.log1p(from_vals["population_total"].values).astype(np.float32)

    # --- 7. 城市等级差 ---
    pairs["tier_diff"] = (to_vals["tier"].values - from_vals["tier"].values).astype(np.int8)

    # --- 8. 老乡网络特征: 目的地已有来自出发地省份的迁移人口比例 ---
    # from_city 的省份 = from_city // 100 * 100 (如 5101 -> 5100)
    from_provinces = (pairs["from_city"].values // 100 * 100).astype(str)
    to_city_arr = pairs["to_city"].values

    migrant_vals = np.zeros(len(pairs), dtype=np.float32)
    for i in range(len(pairs)):
        to_cid = int(to_city_arr[i])
        from_prov = from_provinces[i]
        stock = migrant_stocks.get(to_cid, {})
        if isinstance(stock, dict):
            migrant_vals[i] = stock.get(from_prov, 0.0)
    pairs["migrant_stock_from_to"] = migrant_vals

    # --- 9. 合并距离特征 (from city_edges, 不随年份变化) ---
    pairs = pairs.merge(edges_df, on=["from_city", "to_city"], how="left")
    pairs["geo_distance"] = pairs["geo_distance"].fillna(0).astype(np.float32)
    pairs["dialect_distance"] = pairs["dialect_distance"].fillna(0).astype(np.float32)

    # --- 10. 是否同省 ---
    pairs["is_same_province"] = (
        (pairs["from_city"] // 100) == (pairs["to_city"] // 100)
    ).astype(np.int8)

    # 清理临时列
    pairs.drop(columns=["_migrant_stock"], errors="ignore", inplace=True)

    return pairs


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """压缩 float64 -> float32, 确保无 inf"""
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        # 替换 inf
        if df[col].dtype in [np.float32, np.float64]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(2000, 2021))
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]

    print(f"Loading edges from {EDGES_PATH}...")
    edges_df = load_edges()
    print(f"  Loaded {len(edges_df):,} edges")

    for year in years:
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Processing year {year}...")

        city_df = load_city_info_for_year(year)
        print(f"  Loaded {len(city_df)} cities")

        pairs_df = build_city_pairs_for_year(year, city_df, edges_df)
        pairs_df = optimize_dtypes(pairs_df)

        out_path = OUTPUT_DIR / f"city_pairs_{year}.parquet"
        pairs_df.to_parquet(out_path, index=False, engine="pyarrow")

        elapsed = time.time() - t0
        n_feats = len([c for c in pairs_df.columns if c not in ("from_city", "to_city")])
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
        print(f"  Shape: {pairs_df.shape}, Features: {n_feats}")
        print(f"  Time: {elapsed:.1f}s")

    print(f"\nAll done! Cache files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
