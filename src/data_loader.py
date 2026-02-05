import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_raw_data_from_duckdb(db_path, year_filter=None):
    """
    从 DuckDB 加载数据（加载全部数据，不采样）

    Args:
        db_path: 数据库路径
        year_filter: 年份过滤，例如 [2015, 2016, 2017] 只加载这些年份的数据
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    try:
        with duckdb.connect(db_path, read_only=True) as con:
            # 构建查询
            query_parts = ["SELECT * FROM migration_data"]

            # 添加年份过滤
            if year_filter is not None:
                if isinstance(year_filter, list):
                    years_str = ','.join(map(str, year_filter))
                    query_parts.append(f"WHERE Year IN ({years_str})")
                else:
                    query_parts.append(f"WHERE Year = {year_filter}")

            query = " ".join(query_parts)
            df = con.execute(query).df()
            return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_long_format_with_negatives(df_wide, all_city_ids, hard_candidates, neg_sample_rate=20, is_test_set=False):
    """
    将宽表转换为长表格式，并添加困难负样本（二分类模式 + 软标签）

    软标签策略：
    1. Top 1-10：Label = 1.0（绝对正样本）
    2. Top 11-20：Label = 0.1（灰度样本，比负样本强但不如 Top 10）
    3. 困难负样本：Label = 0.0（纯负样本）

    这样可以避免标签噪声，让模型学到更平滑的特征梯度。

    Args:
        df_wide: 宽表格式的数据
        all_city_ids: 所有可能的目标城市列表（保留用于兼容性，实际不使用）
        hard_candidates: 困难负样本候选池（省会+一二线城市的ID列表）
        neg_sample_rate: 负采样数量（固定20）
        is_test_set: 是否是测试集（保留用于兼容性）
    """
    # 1. 定义 ID 列
    id_cols = ['Year', 'Type_ID', 'From_City', 'Total_Count']
    id_cols = [col for col in id_cols if col in df_wide.columns]

    # ==========================================
    # A. 构建正样本和负样本 (Top20)
    # ==========================================
    top_n = 20
    city_cols = [f'To_Top{i}' for i in range(1, top_n + 1)]
    count_cols = [f'To_Top{i}_Count' for i in range(1, top_n + 1)]

    existing_city_cols = [c for c in city_cols if c in df_wide.columns]
    existing_count_cols = [c for c in count_cols if c in df_wide.columns]

    if not existing_city_cols:
        return pd.DataFrame()

    # 使用 NumPy 数组操作，避免两次 melt
    n_rows = len(df_wide)
    n_cols = len(existing_city_cols)

    # 提取城市ID和计数（转为NumPy数组）
    city_values = df_wide[existing_city_cols].values
    count_values = df_wide[existing_count_cols].values

    # 展平为长格式
    city_flat = city_values.ravel()
    count_flat = count_values.ravel()

    # 创建ID列（重复n_cols次）
    id_data = df_wide[id_cols].values
    id_repeated = np.repeat(id_data, n_cols, axis=0)

    # 创建Rank列
    rank_array = np.tile(np.arange(1, n_cols + 1), n_rows)

    # 构建DataFrame
    df_pos = pd.DataFrame(id_repeated, columns=id_cols)
    df_pos['To_City'] = city_flat
    df_pos['Flow_Count'] = count_flat
    df_pos['Rank'] = rank_array

    # 过滤掉空值
    df_pos = df_pos[pd.notna(df_pos['To_City'])].copy()

    # 【软标签打标逻辑】
    # Rank 1-10: Label = 1.0 (绝对正样本)
    # Rank 11-20: Label = 0.1 (灰度样本，避免标签噪声)
    df_pos['Label'] = np.where(
        df_pos['Rank'] <= 10,
        1.0,  # Top 10 是绝对正样本
        0.1   # Top 11-20 是"灰度"样本，比负样本强
    ).astype('float32')  # 必须是 float，不能是 int

    # 提取城市ID
    from src.feature_eng import extract_city_id
    df_pos['To_City'] = df_pos['To_City'].apply(extract_city_id)
    # 过滤掉无法提取城市ID的记录（空字符串或无效值）
    df_pos = df_pos[pd.notna(df_pos['To_City'])].copy()
    df_pos['To_City'] = df_pos['To_City'].astype('int16')

    # ==========================================
    # B. 构建困难负样本（只使用困难负样本）
    # ==========================================
    df_base = df_wide[id_cols].copy()
    n_queries = len(df_base)
    n_neg = neg_sample_rate  # 固定20个

    # 从困难候选池中随机抽样
    if hard_candidates is not None and len(hard_candidates) > 0:
        hard_pool = np.array(hard_candidates)
        chosen_neg_cities = np.random.choice(hard_pool, size=n_queries * n_neg)
    else:
        # 如果没有提供困难候选池，报错（召回模式必须有困难负样本）
        raise ValueError("召回模式必须提供 hard_candidates（困难负样本候选池）")

    # 构造负样本
    df_neg = df_base.loc[df_base.index.repeat(n_neg)].reset_index(drop=True)
    df_neg['To_City'] = chosen_neg_cities.astype(str)

    # 转换为 int16
    df_neg['To_City'] = df_neg['To_City'].astype('int16')
    df_neg['From_City_Int'] = df_neg['From_City'].apply(extract_city_id).astype('int16')

    # 排除 To_City == From_City
    df_neg = df_neg[df_neg['To_City'] != df_neg['From_City_Int']]
    df_neg = df_neg.drop(columns=['From_City_Int'])

    # 添加标签
    df_neg['Flow_Count'] = 0
    df_neg['Rank'] = 999
    df_neg['Label'] = 0.0  # 纯负样本
    df_neg['Label'] = df_neg['Label'].astype('float32')

    # ==========================================
    # C. 合并与去重
    # ==========================================
    df_long = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)

    # 【关键】如果同一个 Query 下同一个 To_City 既是正又是负，保留 Label 大的（即保留正样本）
    df_long = df_long.sort_values('Label', ascending=False)
    df_long = df_long.drop_duplicates(subset=['Year', 'Type_ID', 'From_City', 'To_City'], keep='first')

    # 转换类型
    df_long['Label'] = df_long['Label'].astype('float32')  # 保持 float32
    df_long['Rank'] = df_long['Rank'].astype('int16')
    df_long['To_City'] = df_long['To_City'].astype('int16')

    if 'Type_ID' in df_long.columns:
        df_long['Type_ID'] = df_long['Type_ID'].astype('category')

    return df_long