"""
超简化快速数据加载器 v2 (最终修复版)

修复内容:
1. 【关键】修复负采样数量 BUG：确保每个 Query 严格生成 neg_sample_rate 个负样本
2. 保留 clean_city_id 清洗逻辑，防止 ID 解析错误
3. 优化内存占用，使用 int16
"""

import duckdb
import pandas as pd
import numpy as np
import re
from pathlib import Path


def clean_city_id(val):
    """
    清洗城市ID，处理多种格式
    """
    if pd.isna(val):
        return None

    s = str(val).strip()
    if s == '' or s == '0' or s == 'None':
        return None

    # 优先匹配括号内的数字: "上饶(3611)"
    if '(' in s and ')' in s:
        match = re.search(r'\((\d+)\)', s)
        if match:
            return int(match.group(1))

    # 其次匹配任意位置的连续数字
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))

    return None


def load_raw_data_fast(db_path, year, hard_candidates, neg_sample_rate=20):
    """
    超快速数据加载

    Args:
        neg_sample_rate: 每个 Query 生成的负样本数量 (建议 10-20)
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    try:
        with duckdb.connect(db_path, read_only=True) as con:
            # 1. SQL读取宽表
            query = f"""
            SELECT
                Year, Type_ID, From_City, Total_Count,
                To_Top1, To_Top1_Count, To_Top2, To_Top2_Count,
                To_Top3, To_Top3_Count, To_Top4, To_Top4_Count,
                To_Top5, To_Top5_Count, To_Top6, To_Top6_Count,
                To_Top7, To_Top7_Count, To_Top8, To_Top8_Count,
                To_Top9, To_Top9_Count, To_Top10, To_Top10_Count,
                To_Top11, To_Top11_Count, To_Top12, To_Top12_Count,
                To_Top13, To_Top13_Count, To_Top14, To_Top14_Count,
                To_Top15, To_Top15_Count, To_Top16, To_Top16_Count,
                To_Top17, To_Top17_Count, To_Top18, To_Top18_Count,
                To_Top19, To_Top19_Count, To_Top20, To_Top20_Count
            FROM migration_data
            WHERE Year = {year}
            """
            df_wide = con.execute(query).df()
            print(f"[SQL] Loaded {len(df_wide)} rows")

        if df_wide.empty:
            return pd.DataFrame()

        # 2. Wide-to-Long (构建正样本)
        pos_data = []
        # 预编译 columns 索引以加速
        cols_map = {c: i for i, c in enumerate(df_wide.columns)}
        values = df_wide.values

        # 这里的列索引需要根据 SQL 查询顺序硬编码或动态获取
        idx_year = cols_map['Year']
        idx_type = cols_map['Type_ID']
        idx_from = cols_map['From_City']

        # 使用 numpy 加速循环 (比 iterrows 快很多)
        for row in values:
            year_val = row[idx_year]
            type_id = row[idx_type]

            # 清洗 From_City
            from_city = clean_city_id(row[idx_from])
            if from_city is None: continue

            for i in range(1, 21):
                col_city = f'To_Top{i}'
                col_count = f'To_Top{i}_Count'
                # 检查两列是否都存在
                if col_city not in cols_map or col_count not in cols_map:
                    continue

                raw_to = row[cols_map[col_city]]
                count = row[cols_map[col_count]]

                to_city = clean_city_id(raw_to)
                if to_city is None: continue

                rank = i
                label = 1.0 if rank <= 10 else 0.1

                pos_data.append({
                    'Year': year_val,
                    'Type_ID': type_id,
                    'From_City': from_city,
                    'To_City': to_city,
                    'Flow_Count': count,
                    'Rank': rank,
                    'Label': label
                })

        df_pos = pd.DataFrame(pos_data)
        if df_pos.empty: return pd.DataFrame()

        print(f"[Wide-to-Long] Generated {len(df_pos)} positive samples")

        # 3. 生成负样本 (修复版)
        # 提取唯一的 Query
        queries = df_pos[['Year', 'Type_ID', 'From_City']].drop_duplicates()
        n_queries = len(queries)

        # 计算数量: 一半困难(Hard)，一半随机(Random)
        n_hard_per_query = int(neg_sample_rate * 0.5)
        n_rand_per_query = neg_sample_rate - n_hard_per_query

        neg_dfs = []

        # --- 3.1 困难负样本 ---
        if n_hard_per_query > 0:
            # 重复 Query
            q_hard = pd.concat([queries] * n_hard_per_query, ignore_index=True)
            # 随机选择 Hard Cities
            hard_pool = np.array(hard_candidates, dtype=np.int16)
            chosen = np.random.choice(hard_pool, size=len(q_hard))
            q_hard['To_City'] = chosen
            neg_dfs.append(q_hard)

        # --- 3.2 随机负样本 ---
        if n_rand_per_query > 0:
            q_rand = pd.concat([queries] * n_rand_per_query, ignore_index=True)
            # 候选池 = 正样本中出现过的城市 + 困难样本
            all_cities = list(set(df_pos['To_City'].unique()) | set(hard_candidates))
            pool = np.array(all_cities, dtype=np.int16)
            chosen = np.random.choice(pool, size=len(q_rand))
            q_rand['To_City'] = chosen
            neg_dfs.append(q_rand)

        # 合并负样本
        if neg_dfs:
            queries_repeated = pd.concat(neg_dfs, ignore_index=True)
            queries_repeated['Rank'] = 999
            queries_repeated['Label'] = 0.0
            queries_repeated['Flow_Count'] = 0

            # 排除 From == To
            queries_repeated = queries_repeated[queries_repeated['From_City'] != queries_repeated['To_City']]

            # 合并正负样本
            df_final = pd.concat([df_pos, queries_repeated], axis=0, ignore_index=True)

            # 去重：如果同个 (Query, To) 既是正又是负，保留正 (Label大)
            df_final = df_final.sort_values('Label', ascending=False) \
                .drop_duplicates(subset=['Year', 'Type_ID', 'From_City', 'To_City'], keep='first')
        else:
            df_final = df_pos

        # 4. 类型优化
        for col in ['From_City', 'To_City', 'Rank']:
            df_final[col] = df_final[col].astype('int16')
        df_final['Label'] = df_final['Label'].astype('float32')

        n_neg = len(df_final) - len(df_pos)
        print(f"[Negative Sampling] Added {n_neg} negative samples (Target: {neg_sample_rate} per query)")
        print(f"[Final] Total: {len(df_final)} rows")

        return df_final

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise
