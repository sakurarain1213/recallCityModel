import pandas as pd
import numpy as np
import re

# Type_ID 维度定义（使用序数编码，保留顺序信息）
DIMENSIONS = {
    'gender': {
        'M': 0,
        'F': 1
    },
    'age_group': {
        '16': 0,  # 16-24岁
        '20': 1,  # 25-34岁
        '35': 2,  # 35-49岁
        '50': 3,  # 50-60岁
        '60': 4   # 60+岁
    },
    'education': {
        'EduLo': 0,   # 低学历
        'EduMid': 1,  # 中学历
        'EduHi': 2    # 高学历
    },
    'industry': {
        'Agri': 0,      # 农业
        'Mfg': 1,       # 制造业
        'Service': 2,   # 传统服务业
        'Wht': 3        # 现代服务业（白领）
    },
    'income': {
        'IncL': 0,   # 低收入
        'IncML': 1,  # 中低收入
        'IncM': 2,   # 中等收入
        'IncMH': 3,  # 中高收入
        'IncH': 4    # 高收入
    },
    'family': {
        'Split': 0,  # 分居/单身
        'Unit': 1    # 家庭单元
    }
}


def parse_type_id(df, verbose=True):
    """
    拆解 Type_ID 为 6 个维度的序数特征（Ordinal Encoding）

    Type_ID 格式: M_20_EduHi_Agri_IncH_Split_1100
    拆解为: 性别、年龄段、学历、行业、收入、家庭状态

    使用序数编码而非 One-Hot，大幅减少特征数量：
    - One-Hot: 21 个特征
    - Ordinal: 6 个特征（减少 71%）

    注意：保留原始 Type_ID 列作为 Type_ID_orig，用于历史特征匹配
    """
    if verbose:
        print("Parsing Type_ID into 6 ordinal features...")

    if 'Type_ID' not in df.columns:
        return df, []

    # 【关键】保留原始 Type_ID 列，用于历史特征匹配
    df['Type_ID_orig'] = df['Type_ID']

    # 拆解 Type_ID
    type_parts = df['Type_ID'].str.split('_', expand=True)

    # 提取各维度并进行序数编码
    df['gender'] = type_parts[0].map(DIMENSIONS['gender']).fillna(0).astype('int8')
    df['age_group'] = type_parts[1].map(DIMENSIONS['age_group']).fillna(0).astype('int8')
    df['education'] = type_parts[2].map(DIMENSIONS['education']).fillna(0).astype('int8')
    df['industry'] = type_parts[3].map(DIMENSIONS['industry']).fillna(0).astype('int8')
    df['income'] = type_parts[4].map(DIMENSIONS['income']).fillna(0).astype('int8')
    df['family'] = type_parts[5].map(DIMENSIONS['family']).fillna(0).astype('int8')

    # 删除原始 Type_ID 列（但保留 Type_ID_orig）
    df = df.drop(columns=['Type_ID'])

    if verbose:
        print(f"  Generated 6 ordinal features from Type_ID (gender, age_group, education, industry, income, family)")

    return df, ['gender', 'age_group', 'education', 'industry', 'income', 'family']


def extract_city_id(val):
    """
    提取城市ID（处理可能的格式：1100, "1100", "北京(1100)"）
    """
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    # 如果是空字符串，返回 None
    if not val_str:
        return None
    # 如果是 "北京(1100)" 格式，提取括号内的数字
    if '(' in val_str and ')' in val_str:
        match = re.search(r'\((\d+)\)', val_str)
        if match:
            return match.group(1)
    # 否则直接返回字符串形式
    return val_str


def add_cross_features(df, city_nodes, city_edges, verbose=True):
    """
    添加城市特征（极简版）：
    1. From_City 保留为分类特征（337个城市，一个维度）
    2. From 和 To 城市的属性差异 ratio（21个属性）
    3. 地理距离和方言距离

    注意：保留原始 From_City 列作为 From_City_orig，用于历史特征匹配
    """
    if verbose:
        print("Generating City features (minimal version)...")

    # 【关键】保留原始 From_City 列，用于历史特征匹配
    if 'From_City' in df.columns:
        # 提取并清洗城市ID
        df['From_City_orig'] = df['From_City'].apply(extract_city_id)
        # 【优化】From_City 转换为 int16（城市 ID 是 4 位数）
        if verbose:
            print("  - Converting From_City to int16...")
        df['From_City'] = df['From_City_orig'].astype(str).astype('int16')

    # 【优化】清洗 To_City（已经在 data_loader 中转换为 int16）
    if 'To_City' in df.columns:
        # To_City 可能已经是 int16，需要先转为 str 再提取 ID
        df['To_City'] = df['To_City'].apply(lambda x: extract_city_id(x) if not isinstance(x, (int, np.integer)) else str(x)).astype('int16')

    # 1. 添加边特征 (距离信息)
    if verbose:
        print("  - Adding edge features (geo distance, dialect distance)...")

    if not city_edges.empty:
        # 需要使用原始的 From_City_orig 进行 merge
        edge_lookup = city_edges[['source_id', 'target_id', 'w_geo', 'w_dialect']].copy()
        edge_lookup.columns = ['From_City_orig', 'To_City', 'geo_distance', 'dialect_distance']

        # 【关键修复】强制两边都转为 string 类型，防止 int vs str 导致匹配失败
        df['From_City_orig'] = df['From_City_orig'].astype(str)
        df['To_City'] = df['To_City'].astype(str)
        edge_lookup['From_City_orig'] = edge_lookup['From_City_orig'].astype(str)
        edge_lookup['To_City'] = edge_lookup['To_City'].astype(str)

        df = df.merge(
            edge_lookup,
            on=['From_City_orig', 'To_City'],
            how='left'
        )

        # 【调试输出】打印匹配率
        if verbose:
            match_rate = (df['geo_distance'].notna()).mean()
            print(f"    > Edge Match Rate: {match_rate:.2%}")

        df['geo_distance'] = df['geo_distance'].fillna(-1).astype('float32')
        df['dialect_distance'] = df['dialect_distance'].fillna(-1).astype('float32')
    else:
        df['geo_distance'] = -1.0
        df['dialect_distance'] = -1.0

    # 2. 计算 From 和 To 城市的属性差异 ratio（不添加绝对属性）
    if verbose:
        print("  - Calculating From-To attribute ratios...")

    if not city_nodes.empty:
        # 定义要使用的城市属性（从 city_info 中提取）
        city_attrs = [
            'tier', 'area_sqkm', 'gdp_per_capita', 'cpi_index', 'unemployment_rate',
            'agriculture_share', 'agriculture_wage',
            'manufacturing_share', 'manufacturing_wage',
            'traditional_services_share', 'traditional_services_wage',
            'modern_services_share', 'modern_services_wage',
            'housing_price_avg', 'rent_avg', 'daily_cost_index',
            'medical_score', 'education_score', 'transport_convenience', 'avg_commute_mins',
            'population_total'
        ]

        available_attrs = [attr for attr in city_attrs if attr in city_nodes.columns]

        if available_attrs:
            # 重置索引，将 city_id 从索引变为列
            city_data = city_nodes.reset_index()[['city_id'] + available_attrs].copy()

            # 【关键修复】强制转换为 string 类型
            city_data['city_id'] = city_data['city_id'].astype(str)

            # Merge From_City 属性（临时使用）
            df = df.merge(
                city_data.add_prefix('from_'),
                left_on='From_City_orig',
                right_on='from_city_id',
                how='left'
            )

            # 【调试输出】打印匹配率
            if verbose:
                from_match_rate = (df['from_tier'].notna()).mean() if 'from_tier' in df.columns else 0
                print(f"    > From_City Match Rate: {from_match_rate:.2%}")

            # Merge To_City 属性（临时使用）
            df = df.merge(
                city_data.add_prefix('to_'),
                left_on='To_City',
                right_on='to_city_id',
                how='left'
            )

            # 【调试输出】打印匹配率
            if verbose:
                to_match_rate = (df['to_tier'].notna()).mean() if 'to_tier' in df.columns else 0
                print(f"    > To_City Match Rate: {to_match_rate:.2%}")

            # 计算 ratio 特征（To / From）
            for attr in available_attrs:
                from_col = f'from_{attr}'
                to_col = f'to_{attr}'

                if from_col in df.columns and to_col in df.columns:
                    # 【修改点 1】加上平滑项，防止除以 0
                    # 【修改点 2】使用截断（Clipping），防止极端值
                    # 原始代码:
                    # df[f'{attr}_ratio'] = np.where(df[from_col] != 0, df[to_col] / df[from_col], 1.0)

                    # === 新代码 Start ===
                    # 1. 先计算原始比值，并处理除零异常
                    raw_ratio = np.where(
                        df[from_col] > 1e-6,  # 避免除以极小值
                        df[to_col] / (df[from_col] + 1e-6),  # 分母加平滑项
                        1.0
                    )

                    # 2. 【关键】强制截断 (Clipping)
                    # 将比值限制在 [0.1, 10] 范围内，防止出现 1000 倍的极端差异
                    # 经济指标差异超过 10 倍通常是异常或量纲错误
                    clipped_ratio = np.clip(raw_ratio, 0.1, 10.0)

                    df[f'{attr}_ratio'] = clipped_ratio.astype('float32')
                    # === 新代码 End ===

            # 删除临时的 from_* 和 to_* 列
            cols_to_drop = [f'from_{attr}' for attr in available_attrs] + \
                          [f'to_{attr}' for attr in available_attrs] + \
                          ['from_city_id', 'to_city_id']
            df = df.drop(cols_to_drop, axis=1, errors='ignore')

            if verbose:
                feature_count = len(available_attrs) + 2 + 1  # ratio + 2 distances + From_City categorical
                print(f"  - Generated {feature_count} city features (From_City as int + {len(available_attrs)} ratios + 2 distances)")
        else:
            if verbose:
                print("  - Warning: No city attributes found")
    else:
        if verbose:
            print("  - Warning: city_nodes is empty")

    return df


def optimize_dtypes(df):
    """
    优化数据类型以减少内存使用
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # 整数列优化
        if col_type in ['int64', 'int32']:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # 浮点数列优化
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
    
    return df


def batch_process_chunks(chunks, city_nodes, city_edges, max_workers=4):
    """
    使用多进程并行处理chunks
    """
    from multiprocessing import Pool
    from functools import partial
    
    def process_single_chunk(chunk_data):
        chunk, nodes, edges = chunk_data
        chunk, cats = parse_type_id(chunk)
        chunk = add_cross_features(chunk, nodes, edges)
        chunk = optimize_dtypes(chunk)
        return chunk, cats
    
    # 准备数据
    chunk_data_list = [(chunk, city_nodes, city_edges) for chunk in chunks]
    
    # 并行处理
    with Pool(max_workers) as pool:
        results = pool.map(process_single_chunk, chunk_data_list)
    
    # 合并结果
    processed_chunks = [r[0] for r in results]
    all_cats = results[0][1] if results else []
    
    return processed_chunks, all_cats