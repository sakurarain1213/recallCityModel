"""
严格 1:4 采样版本 - Top10正样本 + Top11-20困难负样本 + 补充负样本
核心优化: 预计算省份池 + 分省批量采样 (既严格又快)
"""
import duckdb
import pandas as pd
import numpy as np
from src.config import Config


def clean_city_id_vectorized(series):
    """向量化清洗 CityID"""
    # 强制转为 numeric，处理可能的错误字符串
    s = pd.to_numeric(series, errors='coerce')
    # 处理 "北京(1100)" 这种格式
    if s.isna().any():
        # 提取括号内或单纯的数字
        extracted = series.astype(str).str.extract(r'(\d+)', expand=False)
        s = s.fillna(pd.to_numeric(extracted, errors='coerce'))
    return s.astype('Int32')


def load_raw_data_fast(db_path, year, hard_candidates, total_samples=50):
    """
    加载原始数据并按严格比例构建样本。
    - 正样本: Rank 1-10 (Label=1.0)
    - 困难负样本: Rank 11-20 (Label=0.0)
    - 补充负样本: 严格同省(Label=0.0) + 热门 + 随机
    """
    # 1. 极速读取 Top 1-20 数据
    with duckdb.connect(db_path, read_only=True) as con:
        # 动态生成列名 To_Top1 ... To_Top20
        cols = ["Year", "Type_ID", "From_City"] + [f"To_Top{i}" for i in range(1, 21)]
        query = f"SELECT {', '.join(cols)} FROM migration_data WHERE Year = {year}"
        df_wide = con.execute(query).df()

    if df_wide.empty:
        return pd.DataFrame()

    # 2. Wide 转 Long (矩阵操作，极快)
    # 目标：生成 Rank 1-20 的所有行
    id_vars = ['Year', 'Type_ID', 'From_City']
    df_long = df_wide.melt(id_vars=id_vars, value_name='To_City', var_name='Rank_Str')

    # 提取 Rank 数值 (To_Top5 -> 5)
    df_long['Rank'] = df_long['Rank_Str'].str.extract(r'(\d+)').astype(int)
    df_long = df_long.drop(columns=['Rank_Str'])

    # 清洗 ID
    df_long['From_City'] = clean_city_id_vectorized(df_long['From_City'])
    df_long['To_City'] = clean_city_id_vectorized(df_long['To_City'])
    df_long = df_long.dropna(subset=['From_City', 'To_City'])

    # --- 3. 核心：Label 定义 ---
    # Rank 1-10 为正，Rank 11-20 为负
    df_long['Label'] = np.where(df_long['Rank'] <= 10, 1.0, 0.0).astype('float32')

    # 这是一个基础集 (Base Set)，包含了所有的 Top 20
    df_base = df_long.copy()

    # --- 4. 生成补充负样本 (Supplement) ---
    # 我们已有 20 个样本 (Top 1-20)，还需要补 (50 - 20) = 30 个
    n_needed = total_samples - 20

    if n_needed > 0:
        queries = df_wide[['Year', 'Type_ID', 'From_City']].copy()
        # 确保 From_City 清洗过并转为 int
        queries['From_City'] = clean_city_id_vectorized(queries['From_City']).astype(int)
        queries = queries.dropna()
        n_queries = len(queries)

        # 准备候选池
        all_cities = np.array(hard_candidates, dtype=np.int32)  # 确保是 int
        hot_cities = np.array(Config.POPULAR_CITIES, dtype=np.int32)

        # 定义采样数量
        n_hot = 10
        n_prov = 10  # 同省
        n_rand = n_needed - n_hot - n_prov

        neg_dfs = []

        # A. 热门城市负样本 (10个)
        if len(hot_cities) > 0 and n_hot > 0:
            chosen = np.random.choice(hot_cities, size=(n_queries, n_hot))
            neg_dfs.append(pd.DataFrame({
                'Year': np.repeat(queries['Year'].values, n_hot),
                'Type_ID': np.repeat(queries['Type_ID'].values, n_hot),
                'From_City': np.repeat(queries['From_City'].values, n_hot),
                'To_City': chosen.flatten(),
                'Rank': 98, 'Label': 0.0
            }))

        # B. 严格同省负样本 (10个) - 【优化重点】
        if n_prov > 0:
            # 1. 预计算：构建 {Prov_ID: [City_IDs]} 字典
            # 使用 numpy 极速构建
            all_provs = all_cities // 100
            unique_provs = np.unique(all_provs)

            prov_pool = {}
            for pid in unique_provs:
                prov_pool[pid] = all_cities[all_provs == pid]

            # 2. 计算 Query 的省份
            queries['Prov_ID'] = queries['From_City'] // 100

            # 3. 分省批量采样
            # groupby 只有 ~34 个组 (省份数)，循环非常快
            prov_samples = []

            for pid, group in queries.groupby('Prov_ID'):
                # 获取该省的所有候选城市
                candidates = prov_pool.get(pid, all_cities)  # 兜底：如果没找到，用全量

                # 如果该省城市太少(比如只有自己)，允许重复采样(replace=True)
                # 后面会有去重逻辑处理掉 From==To 的情况
                n_grp = len(group)

                if len(candidates) > 0:
                    chosen = np.random.choice(candidates, size=(n_grp, n_prov), replace=True)
                else:
                    # 极端情况：该省没有候选城市，退化为随机采样
                    chosen = np.random.choice(all_cities, size=(n_grp, n_prov))

                # 构建该省的负样本 DF
                # 注意：使用 np.repeat 扩展 metadata
                df_prov = pd.DataFrame({
                    'Year': np.repeat(group['Year'].values, n_prov),
                    'Type_ID': np.repeat(group['Type_ID'].values, n_prov),
                    'From_City': np.repeat(group['From_City'].values, n_prov),
                    'To_City': chosen.flatten(),
                    'Rank': 99, 'Label': 0.0
                })
                prov_samples.append(df_prov)

            if prov_samples:
                neg_dfs.append(pd.concat(prov_samples, ignore_index=True))

        # C. 随机负样本 (剩余所有)
        if n_rand > 0:
            chosen = np.random.choice(all_cities, size=(n_queries, n_rand))
            neg_dfs.append(pd.DataFrame({
                'Year': np.repeat(queries['Year'].values, n_rand),
                'Type_ID': np.repeat(queries['Type_ID'].values, n_rand),
                'From_City': np.repeat(queries['From_City'].values, n_rand),
                'To_City': chosen.flatten(),
                'Rank': 97, 'Label': 0.0
            }))

        # --- 合并 ---
        if neg_dfs:
            df_supp = pd.concat(neg_dfs, ignore_index=True)
            df_supp['From_City'] = df_supp['From_City'].astype('Int32')  # 对齐类型

            # --- 5. 清洗与去重 ---

            # 5.1 排除 From == To (同省采样很容易采到自己)
            df_supp = df_supp[df_supp['From_City'] != df_supp['To_City']]

            # 5.2 合并到 Base
            df_final = pd.concat([df_base, df_supp], ignore_index=True)

            # 5.3 去重：保留 df_base 的 Rank (因为 base 在前，keep='first')
            # 这一步会剔除掉那些"不小心采到了 Top20"的负样本
            df_final = df_final.drop_duplicates(
                subset=['Year', 'Type_ID', 'From_City', 'To_City'],
                keep='first'
            )
        else:
            df_final = df_base
    else:
        df_final = df_base

    # 最终类型转换
    for col in ['From_City', 'To_City']:
        df_final[col] = df_final[col].astype('int16')

    # 填充 Flow_Count
    if 'Flow_Count' not in df_final.columns:
        df_final['Flow_Count'] = 0

    return df_final
