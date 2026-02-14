"""
终极加速版 - NumPy 矩阵操作 + Anti-Join 去重
消除 sort_values 和 apply 瓶颈
速度: ~10-20秒处理单年 30万 Query (1200万行)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.config import Config


def clean_city_id_vectorized(series):
    """
    向量化清洗城市ID (比 apply 快 100 倍)
    处理: 1100 (int/str), "1100", "北京(1100)"
    """
    # 1. 尝试直接转数字 (处理纯数字字符串和数字类型)
    s_numeric = pd.to_numeric(series, errors='coerce')

    # 2. 如果有 NaN (说明是 "北京(1100)" 这种格式)，用正则提取
    if s_numeric.isna().any():
        # 提取括号内或字符串中的数字
        # \d+ 匹配数字
        extracted = series.astype(str).str.extract(r'(\d+)', expand=False)
        s_final = pd.to_numeric(extracted, errors='coerce')
        # 合并结果：优先用直接转的，失败的用正则结果
        return s_numeric.fillna(s_final).astype('Int32') # Int32 支持 NaN (虽然我们最后会 dropna)

    return s_numeric.astype('Int32')


def load_raw_data_fast(db_path, year, hard_candidates, neg_sample_rate=40):
    """
    【终极加速版】NumPy 矩阵操作 + Anti-Join
    消除 sort_values 和 apply 瓶颈
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    # --- 1. 极速读取 ---
    with duckdb.connect(db_path, read_only=True) as con:
        # 只读取需要的列，减少 I/O
        cols_sql = "Year, Type_ID, From_City, " + ", ".join([f"To_Top{i}, To_Top{i}_Count" for i in range(1, 21)])
        df_wide = con.execute(f"SELECT {cols_sql} FROM migration_data WHERE Year = {year}").df()
        if df_wide.empty:
            return pd.DataFrame()

    print(f"  [{year}] Step 1: NumPy 矩阵化重塑 (Wide->Long)...")

    # --- 2. NumPy Flatten (替代 Melt) ---
    # 这种方法比 pd.melt 快非常多，因为它是纯内存视图操作
    n_rows = len(df_wide)
    n_top = 20

    # 提取基础列
    # repeat: [A, B] -> [A, A, ..., B, B, ...] (按行重复)
    base_year = np.repeat(df_wide['Year'].values, n_top)
    base_type = np.repeat(df_wide['Type_ID'].values, n_top)
    base_from = np.repeat(df_wide['From_City'].values, n_top)

    # 提取 Top 列 (Matrix)
    top_cols = [f'To_Top{i}' for i in range(1, 21)]
    count_cols = [f'To_Top{i}_Count' for i in range(1, 21)]

    # Values: (N_rows, 20) -> Flatten -> (N_rows * 20,)
    # 注意: flatten() 默认是按行优先 (C-style)，这正是我们要的 [Row1_Top1, Row1_Top2...]
    vals_to = df_wide[top_cols].values.flatten()
    vals_count = df_wide[count_cols].values.flatten()

    # 生成 Rank: [1, 2, ..., 20] 重复 N 次
    # tile: [1, 2] -> [1, 2, 1, 2] (整体重复)
    vals_rank = np.tile(np.arange(1, 21), n_rows)

    # 构建 DataFrame (此时包含所有 Top 1-20，包含空值)
    df_long = pd.DataFrame({
        'Year': base_year,
        'Type_ID': base_type,
        'From_City': base_from,
        'To_City': vals_to,
        'Flow_Count': vals_count,
        'Rank': vals_rank
    })

    # --- 3. 向量化清洗 ---
    # 清洗 From 和 To (耗时点优化)
    df_long['From_City'] = clean_city_id_vectorized(df_long['From_City'])
    df_long['To_City'] = clean_city_id_vectorized(df_long['To_City'])

    # 丢弃无效行 (空城市)
    df_long = df_long.dropna(subset=['From_City', 'To_City'])

    # 类型转换
    df_long['From_City'] = df_long['From_City'].astype('int16')
    df_long['To_City'] = df_long['To_City'].astype('int16')
    df_long['Rank'] = df_long['Rank'].astype('int16')
    df_long['Flow_Count'] = df_long['Flow_Count'].fillna(0).astype('int32')

    # 生成 Label
    df_long['Label'] = np.where(df_long['Rank'] <= 10, 1.0, 0.0).astype('float32')

    # 正样本与 Top20 负样本
    df_pos_hard = df_long  # Rank 1-20

    print(f"  [{year}] Step 2: 矩阵化生成负样本...")

    # --- 4. 极速负采样 ---
    # 唯一 Queries
    queries = df_pos_hard[['Year', 'Type_ID', 'From_City']].drop_duplicates()
    n_queries = len(queries)

    # 候选池
    all_cities = np.array(list(set(hard_candidates) | set(df_pos_hard['To_City'])), dtype=np.int16)
    global_hot = np.array(Config.POPULAR_CITIES, dtype=np.int16)

    # 预计算省份 Mask
    city_prov_map = pd.Series(index=all_cities, data=all_cities // 100)

    neg_dfs = []

    # 策略: 同省(15) + 热门(15) + 随机(15) -> 总 45 -> 去重/截断 -> 40
    n_prov, n_global, n_rand = 15, 15, 15

    # A. 同省负样本 (Groupby 加速)
    queries['Prov_ID'] = queries['From_City'] // 100
    # 预先构建每个省的候选数组 (Dict: int -> Array)
    prov_pool = {pid: city_prov_map[city_prov_map == pid].index.values for pid in queries['Prov_ID'].unique()}

    prov_neg_list = []
    with tqdm(total=len(queries.groupby('Prov_ID')), desc=f"  [{year}] 同省负样本", leave=False) as pbar:
        for pid, group in queries.groupby('Prov_ID'):
            cands = prov_pool.get(pid, all_cities)
            if len(cands) == 0:
                cands = all_cities

            # 随机采样矩阵
            chosen = np.random.choice(cands, size=(len(group), n_prov))

            # 扩展 Group 索引
            idx_rep = np.repeat(group.index, n_prov)
            vals_rep = queries.loc[idx_rep] # 直接用 queries (比 group.loc 快)

            prov_neg_list.append(pd.DataFrame({
                'Year': vals_rep['Year'].values,
                'Type_ID': vals_rep['Type_ID'].values,
                'From_City': vals_rep['From_City'].values,
                'To_City': chosen.flatten(),
                'Rank': 99,
                'Label': 0.0,
                'Flow_Count': 0
            }))
            pbar.update(1)

    if prov_neg_list:
        neg_dfs.append(pd.concat(prov_neg_list))

    # B. 全局热门
    if len(global_hot) > 0:
        chosen_glob = np.random.choice(global_hot, size=(n_queries, n_global))
        idx_rep = np.repeat(queries.index, n_global)
        vals_rep = queries.loc[idx_rep]

        neg_dfs.append(pd.DataFrame({
            'Year': vals_rep['Year'].values,
            'Type_ID': vals_rep['Type_ID'].values,
            'From_City': vals_rep['From_City'].values,
            'To_City': chosen_glob.flatten(),
            'Rank': 98,
            'Label': 0.0,
            'Flow_Count': 0
        }))

    # C. 随机
    chosen_rand = np.random.choice(all_cities, size=(n_queries, n_rand))
    idx_rep = np.repeat(queries.index, n_rand)
    vals_rep = queries.loc[idx_rep]

    neg_dfs.append(pd.DataFrame({
        'Year': vals_rep['Year'].values,
        'Type_ID': vals_rep['Type_ID'].values,
        'From_City': vals_rep['From_City'].values,
        'To_City': chosen_rand.flatten(),
        'Rank': 97,
        'Label': 0.0,
        'Flow_Count': 0
    }))

    # 合并负样本
    df_all_neg = pd.concat(neg_dfs, ignore_index=True)
    df_all_neg['To_City'] = df_all_neg['To_City'].astype('int16')

    print(f"  [{year}] Step 3: Anti-Join 极速去重 (替代Sort)...")

    # --- 5. Anti-Join 去重 (关键加速点) ---
    # 我们不进行排序，而是直接从负样本中剔除那些"其实是正样本"的行
    # 原理: 正样本(Rank 1-20) 已经存在于 df_pos_hard 中
    # 我们只需要保留 df_all_neg 中不存在于 df_pos_hard 的行

    # 定义连接键 (Type_ID 是字符串，可能会慢，但比 sort 快)
    join_keys = ['Year', 'Type_ID', 'From_City', 'To_City']

    # 标记哪些负样本其实在 Top 20 里出现过
    # indicator=True 会生成一个 '_merge' 列
    merged = df_all_neg.merge(
        df_pos_hard[join_keys],
        on=join_keys,
        how='left',
        indicator=True
    )

    # 只保留 left_only (即只在负样本表中出现，不在 Top 20 中出现的)
    # 这样就完美剔除了冲突样本，且不需要排序！
    df_true_neg = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # 恢复列 (Merge 可能会丢失非 Key 列的数据，如果 df_pos_hard 只有 Key)
    # 但这里 df_all_neg 本身包含所有数据，所以没问题

    # --- 6. 截断与合并 ---
    # 目标负样本数
    target_neg = int(n_queries * neg_sample_rate)

    # 如果生成的真负样本太多，随机采样截断
    if len(df_true_neg) > target_neg:
        # 使用 numpy 随机索引替代 DataFrame.sample (更快)
        indices = np.random.choice(len(df_true_neg), target_neg, replace=False)
        df_true_neg = df_true_neg.iloc[indices]

    # 最终合并
    df_final = pd.concat([df_pos_hard, df_true_neg], ignore_index=True)

    # 最终类型优化
    df_final['Label'] = df_final['Label'].astype('float32')

    print(f"  [{year}] 完成! 总样本: {len(df_final):,} (正/Hard: {len(df_pos_hard):,}, 补充负: {len(df_true_neg):,})")

    return df_final
