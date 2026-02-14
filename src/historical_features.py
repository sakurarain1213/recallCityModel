import pandas as pd
import numpy as np
from pathlib import Path
import gc


def add_historical_features(df, year, data_dir='output/processed_data', verbose=True, training_mode=False):
    """
    【内存优化版】添加历史特征

    优化策略：
    避免使用字符串列（Type_ID_orig, From_City_orig）进行 Merge。
    改为使用整数列（From_City）和哈希值（Type_Hash）进行连接。
    内存消耗减少 90%，彻底解决 1400万行 Merge 时的 OOM 问题。

    特征包括：
    1. Hist_Flow_Log: log(去年的流量 + 1)
    2. Hist_Rank: 去年的排名（1-20，50表示不在Top20）
    3. Hist_Label: 去年的断层Label（30-21 for Top10, 10-1 for Rank11-20, 0 for others）
    4. Hist_Share: 去年该路线占出发城市总流出的比例
    5. Hist_Label_1y: 近1年的Label（同 Hist_Label）
    6. Hist_Label_3y_avg: 近3年的平均Label（-1表示数据不足）
    7. Hist_Label_5y_avg: 近5年的平均Label（-1表示数据不足）

    防止过拟合策略：
    - training_mode=True 时，随机 Dropout 20% 的历史特征，强迫模型学习属性特征
    - training_mode=False 时（评估/预测），使用全部历史特征

    Args:
        df: 当前年份的数据，必须包含 ['Year', 'Type_ID_orig', 'From_City', 'To_City']
        year: 当前年份
        data_dir: 历史数据目录
        verbose: 是否打印详细信息
        training_mode: 是否为训练模式（训练时启用 Dropout）

    Returns:
        df: 添加了历史特征的 DataFrame
    """
    if verbose:
        mode_str = "Training (with Dropout)" if training_mode else "Evaluation (no Dropout)"
        print(f"  Adding historical features from year {year-1} to {year-5} [{mode_str}]...")

    # 1. 检查必要的列
    # 注意：我们需要 int 类型的 From_City
    required_cols = ['Year', 'Type_ID_orig', 'From_City', 'To_City']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        if verbose:
            print(f"  Warning: Missing columns {missing_cols}, skipping historical features")
        return _fill_empty_history(df)

    # ==========================================================================
    # 核心优化：生成连接键的 Hash (避免字符串 Merge)
    # ==========================================================================
    # 1. 确保 From_City 是 int16 (已在 pipeline 中处理，这里兜底)
    if not pd.api.types.is_integer_dtype(df['From_City']):
        df['From_City'] = pd.to_numeric(df['From_City'], errors='coerce').fillna(0).astype('int16')

    # 2. 对 Type_ID_orig 进行 Hash，生成 uint64 整数
    # pandas.util.hash_pandas_object 速度极快且碰撞率极低
    df['Type_Hash'] = pd.util.hash_pandas_object(df['Type_ID_orig'], index=False)

    # 定义纯数字的连接键
    # Type_Hash(uint64), From_City(int16), To_City(int16) -> 极其省内存
    merge_keys = ['Type_Hash', 'From_City', 'To_City']
    # ==========================================================================

    # 2. 加载多年历史数据
    historical_years = [year - i for i in range(1, 6)]
    yearly_features = {}

    for hist_year in historical_years:
        hist_data_path = Path(data_dir) / f"processed_{hist_year}.parquet"

        if not hist_data_path.exists():
            continue

        # 只读取需要的列，减少内存占用
        # 注意：读取 int 类型的 From_City，而非 orig
        cols_to_read = ['Type_ID_orig', 'From_City', 'To_City', 'Flow_Count', 'Rank', 'Label']
        try:
            df_hist = pd.read_parquet(hist_data_path, columns=cols_to_read)
        except Exception:
            # 兼容旧数据可能没有 From_City int列的情况
            df_hist = pd.read_parquet(hist_data_path)
            if 'From_City' not in df_hist.columns and 'From_City_orig' in df_hist.columns:
                 df_hist['From_City'] = pd.to_numeric(df_hist['From_City_orig'], errors='coerce').fillna(0).astype('int16')

        # 过滤正样本
        df_hist_pos = df_hist[df_hist['Label'] > 0].copy()
        del df_hist # 立即释放
        gc.collect()

        if len(df_hist_pos) == 0:
            continue

        # 【核心优化】对历史数据也做同样的 Hash 处理
        df_hist_pos['Type_Hash'] = pd.util.hash_pandas_object(df_hist_pos['Type_ID_orig'], index=False)

        # 丢弃字符串列，释放内存
        df_hist_pos = df_hist_pos.drop(columns=['Type_ID_orig'], errors='ignore')
        if 'From_City_orig' in df_hist_pos.columns:
            df_hist_pos = df_hist_pos.drop(columns=['From_City_orig'])

        yearly_features[hist_year] = df_hist_pos

    # 如果无历史数据
    if not yearly_features:
        df = df.drop(columns=['Type_Hash']) # 清理临时列
        return _fill_empty_history(df)

    # 3. 构造 1 年历史特征（T-1）
    if year - 1 in yearly_features:
        hist_1y = yearly_features[year - 1].copy()

        # 3.1 基础特征
        hist_1y['Hist_Flow_Log'] = np.log1p(hist_1y['Flow_Count']).astype('float32')
        hist_1y['Hist_Rank'] = hist_1y['Rank'].astype('int16')
        hist_1y['Hist_Label'] = hist_1y['Label'].astype('float32')
        hist_1y['Hist_Label_1y'] = hist_1y['Label'].astype('int16')

        # 3.2 计算 Hist_Share (使用 Hash 键进行 GroupBy，速度更快)
        # Total_Out: 某类人从某城市出发的总流量
        total_out = hist_1y.groupby(['Type_Hash', 'From_City'])['Flow_Count'].sum().reset_index()
        total_out.rename(columns={'Flow_Count': 'Total_Out'}, inplace=True)

        hist_1y = hist_1y.merge(total_out, on=['Type_Hash', 'From_City'], how='left')

        # 贝叶斯平滑
        alpha, beta = 1, 10
        hist_1y['Hist_Share'] = ((hist_1y['Flow_Count'] + alpha) / (hist_1y['Total_Out'] + beta)).astype('float32')

        # 选列 (只保留数字列)
        cols_1y = merge_keys + ['Hist_Flow_Log', 'Hist_Rank', 'Hist_Label', 'Hist_Share', 'Hist_Label_1y']
        hist_1y_features = hist_1y[cols_1y].copy()

        del hist_1y, total_out
        gc.collect()
    else:
        hist_1y_features = None

    # 4. 构造 3 年平均特征
    years_3 = [y for y in [year-1, year-2, year-3] if y in yearly_features]
    if years_3:
        # 只取需要的列进行 concat
        hist_3y_list = [yearly_features[y][merge_keys + ['Label']] for y in years_3]
        hist_3y = pd.concat(hist_3y_list, axis=0, ignore_index=True)

        # GroupBy Hash keys
        hist_3y_avg = hist_3y.groupby(merge_keys)['Label'].mean().reset_index()
        hist_3y_avg.rename(columns={'Label': 'Hist_Label_3y_avg'}, inplace=True)
        hist_3y_avg['Hist_Label_3y_avg'] = hist_3y_avg['Hist_Label_3y_avg'].astype('float32')

        del hist_3y, hist_3y_list
        gc.collect()
    else:
        hist_3y_avg = None

    # 5. 构造 5 年平均特征
    years_5 = [y for y in historical_years if y in yearly_features]
    if years_5:
        hist_5y_list = [yearly_features[y][merge_keys + ['Label']] for y in years_5]
        hist_5y = pd.concat(hist_5y_list, axis=0, ignore_index=True)

        hist_5y_avg = hist_5y.groupby(merge_keys)['Label'].mean().reset_index()
        hist_5y_avg.rename(columns={'Label': 'Hist_Label_5y_avg'}, inplace=True)
        hist_5y_avg['Hist_Label_5y_avg'] = hist_5y_avg['Hist_Label_5y_avg'].astype('float32')

        del hist_5y, hist_5y_list
        gc.collect()
    else:
        hist_5y_avg = None

    # 6. Merge 回主表 (全部是数字列 Merge，极快)
    if hist_1y_features is not None:
        df = df.merge(hist_1y_features, on=merge_keys, how='left')
        del hist_1y_features

    if hist_3y_avg is not None:
        df = df.merge(hist_3y_avg, on=merge_keys, how='left')
        del hist_3y_avg

    if hist_5y_avg is not None:
        df = df.merge(hist_5y_avg, on=merge_keys, how='left')
        del hist_5y_avg

    # 7. 清理临时 Hash 列
    df = df.drop(columns=['Type_Hash'])
    gc.collect()

    # 8. 填充默认值 & Dropout
    return _post_process_history(df, training_mode, verbose)


def _fill_empty_history(df):
    """辅助函数：填充全空的特征"""
    for col in ['Hist_Flow_Log', 'Hist_Rank', 'Hist_Label', 'Hist_Share', 'Hist_Label_1y']:
        df[col] = 0
    df['Hist_Rank'] = 50
    df['Hist_Label_3y_avg'] = -1.0
    df['Hist_Label_5y_avg'] = -1.0
    return df


def _post_process_history(df, training_mode, verbose):
    """后处理：Dropout 和 缺失值填充 (带容错)"""

    # 1. 确保所有历史特征列都存在 (容错核心)
    # 如果因为某种原因(如只有3年前数据而没有1年前数据)导致某些列缺失，这里补齐
    expected_cols = {
        'Hist_Flow_Log': 0.0,
        'Hist_Rank': 50,
        'Hist_Label': 0.0,
        'Hist_Share': 0.0,
        'Hist_Label_1y': 0,
        'Hist_Label_3y_avg': -1.0,
        'Hist_Label_5y_avg': -1.0
    }

    for col, default_val in expected_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # 2. Dropout (仅训练模式)
    if training_mode:
        dropout_prob = 0.2
        mask = np.random.rand(len(df)) < dropout_prob

        if verbose:
             print(f"    [Dropout] Masking {mask.sum():,} rows")

        # 重置为默认值
        cols_zero = ['Hist_Flow_Log', 'Hist_Label', 'Hist_Share', 'Hist_Label_1y']
        df.loc[mask, cols_zero] = 0.0
        df.loc[mask, 'Hist_Rank'] = 50
        df.loc[mask, ['Hist_Label_3y_avg', 'Hist_Label_5y_avg']] = -1.0

    # 3. 填充 NaN 并转换类型
    # 注意：fillna 之前确保列已存在(由步骤1保证)
    df['Hist_Flow_Log'] = df['Hist_Flow_Log'].fillna(0.0).astype('float32')
    df['Hist_Rank'] = df['Hist_Rank'].fillna(50).astype('int16')
    df['Hist_Label'] = df['Hist_Label'].fillna(0.0).astype('float32')
    df['Hist_Share'] = df['Hist_Share'].fillna(0.0).astype('float32')
    df['Hist_Label_1y'] = df['Hist_Label_1y'].fillna(0).astype('int16')
    df['Hist_Label_3y_avg'] = df['Hist_Label_3y_avg'].fillna(-1.0).astype('float32')
    df['Hist_Label_5y_avg'] = df['Hist_Label_5y_avg'].fillna(-1.0).astype('float32')

    # 4. 统计信息
    if verbose:
        has_hist_1y = (df['Hist_Rank'] != 50).sum()
        has_hist_3y = (df['Hist_Label_3y_avg'] >= 0).sum()  # >= 0 表示有数据
        has_hist_5y = (df['Hist_Label_5y_avg'] >= 0).sum()  # >= 0 表示有数据
        print(f"    Historical features added:")
        print(f"      - 1-year:  {has_hist_1y:,} / {len(df):,} rows ({has_hist_1y/len(df):.1%})")
        print(f"      - 3-year:  {has_hist_3y:,} / {len(df):,} rows ({has_hist_3y/len(df):.1%})")
        print(f"      - 5-year:  {has_hist_5y:,} / {len(df):,} rows ({has_hist_5y/len(df):.1%})")

    return df
