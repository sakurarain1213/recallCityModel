import pandas as pd
import numpy as np
from pathlib import Path


def add_historical_features(df, year, data_dir='output/processed_data', verbose=True, training_mode=False):
    """
    添加历史特征（使用 T-1, T-3, T-5 年的数据预测 T 年）

    特征包括：
    1. Hist_Flow_Log: log(去年的流量 + 1)
    2. Hist_Rank: 去年的排名（1-20，50表示不在Top20）
    3. Hist_Label: 去年的断层Label（30-21 for Top10, 10-1 for Rank11-20, 0 for others）
    4. Hist_Share: 去年该路线占出发城市总流出的比例
    5. Hist_Label_1y: 近1年的Label（同 Hist_Label）
    6. Hist_Label_3y_avg: 近3年的平均Label（-1表示数据不足）
    7. Hist_Label_5y_avg: 近5年的平均Label（-1表示数据不足）

    注意：
    - Hist_Label_3y_avg 和 Hist_Label_5y_avg 使用 -1 表示"历史数据不足"
    - 这与 0（表示"历史排名很差，不在Top20"）区分开来

    防止过拟合策略：
    - training_mode=True 时，随机 Dropout 20% 的历史特征，强迫模型学习属性特征
    - training_mode=False 时（评估/预测），使用全部历史特征

    Args:
        df: 当前年份的数据，必须包含 ['Year', 'Month', 'Type_ID_orig', 'From_City_orig', 'To_City']
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
    required_cols = ['Year', 'Type_ID_orig', 'From_City_orig', 'To_City']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        if verbose:
            print(f"  Warning: Missing columns {missing_cols}, skipping historical features")
        # 添加空的历史特征列
        df['Hist_Flow_Log'] = 0.0
        df['Hist_Rank'] = 50
        df['Hist_Label'] = 0
        df['Hist_Share'] = 0.0
        df['Hist_Label_1y'] = 0
        df['Hist_Label_3y_avg'] = -1.0  # -1 表示数据不足
        df['Hist_Label_5y_avg'] = -1.0  # -1 表示数据不足
        return df

    # 2. 加载多年历史数据（1年、3年、5年）
    historical_years = [year - i for i in range(1, 6)]  # [year-1, year-2, ..., year-5]

    # 用于存储每年的历史特征
    yearly_features = {}

    for hist_year in historical_years:
        hist_data_path = Path(data_dir) / f"processed_{hist_year}.parquet"

        if not hist_data_path.exists():
            if verbose and hist_year == year - 1:
                print(f"  Warning: {hist_data_path} not found")
            continue

        # 读取该年的正样本数据（现在包含 Rank 1-20）
        df_hist = pd.read_parquet(hist_data_path)
        # 【修复】不再依赖 Is_Positive 列，直接通过 Label > 0 判断正样本
        df_hist_pos = df_hist[df_hist['Label'] > 0].copy()

        if len(df_hist_pos) == 0:
            continue

        # 提取需要的列
        hist_cols = ['Type_ID_orig', 'From_City_orig', 'To_City',
                     'Flow_Count', 'Rank', 'Label']
        hist_data = df_hist_pos[hist_cols].copy()

        # 存储该年的数据
        yearly_features[hist_year] = hist_data

    # 如果没有任何历史数据，填充默认值
    if not yearly_features:
        if verbose:
            print(f"  Warning: No historical data found, filling with default values")
        df['Hist_Flow_Log'] = 0.0
        df['Hist_Rank'] = 50
        df['Hist_Label'] = 0
        df['Hist_Share'] = 0.0
        df['Hist_Label_1y'] = 0
        df['Hist_Label_3y_avg'] = -1.0  # -1 表示数据不足
        df['Hist_Label_5y_avg'] = -1.0  # -1 表示数据不足
        return df

    # 3. 构造 1 年历史特征（T-1）
    if year - 1 in yearly_features:
        hist_1y = yearly_features[year - 1].copy()

        # 3.1 计算基础特征
        hist_1y['Hist_Flow_Log'] = np.log1p(hist_1y['Flow_Count']).astype('float32')
        hist_1y['Hist_Rank'] = hist_1y['Rank'].astype('int16')
        hist_1y['Hist_Label'] = hist_1y['Label'].astype('int16')  # 使用断层Label
        hist_1y['Hist_Label_1y'] = hist_1y['Label'].astype('int16')

        # 3.2 计算 Hist_Share（该路线占出发城市总流出的比例）
        total_out = hist_1y.groupby(['Type_ID_orig', 'From_City_orig'])['Flow_Count'].sum().reset_index()
        total_out.columns = ['Type_ID_orig', 'From_City_orig', 'Total_Out']
        hist_1y = hist_1y.merge(total_out, on=['Type_ID_orig', 'From_City_orig'], how='left')

        # 贝叶斯平滑
        alpha, beta = 1, 10
        hist_1y['Hist_Share'] = ((hist_1y['Flow_Count'] + alpha) / (hist_1y['Total_Out'] + beta)).astype('float32')

        # 只保留需要的列
        hist_1y_features = hist_1y[['Type_ID_orig', 'From_City_orig', 'To_City',
                                     'Hist_Flow_Log', 'Hist_Rank', 'Hist_Label', 'Hist_Share', 'Hist_Label_1y']].copy()
    else:
        # 如果没有 T-1 数据，创建空 DataFrame
        hist_1y_features = pd.DataFrame(columns=['Type_ID_orig', 'From_City_orig', 'To_City',
                                                  'Hist_Flow_Log', 'Hist_Rank', 'Hist_Label', 'Hist_Share', 'Hist_Label_1y'])

    # 4. 构造 3 年平均历史特征（T-1, T-2, T-3）
    years_3 = [y for y in [year-1, year-2, year-3] if y in yearly_features]
    if years_3:
        hist_3y_list = [yearly_features[y][['Type_ID_orig', 'From_City_orig', 'To_City', 'Label']].copy()
                        for y in years_3]
        hist_3y = pd.concat(hist_3y_list, axis=0, ignore_index=True)

        # 计算平均 Label
        hist_3y_avg = hist_3y.groupby(['Type_ID_orig', 'From_City_orig', 'To_City'])['Label'].mean().reset_index()
        hist_3y_avg.columns = ['Type_ID_orig', 'From_City_orig', 'To_City', 'Hist_Label_3y_avg']
        hist_3y_avg['Hist_Label_3y_avg'] = hist_3y_avg['Hist_Label_3y_avg'].astype('float32')
    else:
        hist_3y_avg = pd.DataFrame(columns=['Type_ID_orig', 'From_City_orig', 'To_City', 'Hist_Label_3y_avg'])

    # 5. 构造 5 年平均历史特征（T-1 到 T-5）
    years_5 = [y for y in historical_years if y in yearly_features]
    if years_5:
        hist_5y_list = [yearly_features[y][['Type_ID_orig', 'From_City_orig', 'To_City', 'Label']].copy()
                        for y in years_5]
        hist_5y = pd.concat(hist_5y_list, axis=0, ignore_index=True)

        # 计算平均 Label
        hist_5y_avg = hist_5y.groupby(['Type_ID_orig', 'From_City_orig', 'To_City'])['Label'].mean().reset_index()
        hist_5y_avg.columns = ['Type_ID_orig', 'From_City_orig', 'To_City', 'Hist_Label_5y_avg']
        hist_5y_avg['Hist_Label_5y_avg'] = hist_5y_avg['Hist_Label_5y_avg'].astype('float32')
    else:
        hist_5y_avg = pd.DataFrame(columns=['Type_ID_orig', 'From_City_orig', 'To_City', 'Hist_Label_5y_avg'])

    # 6. Merge 所有历史特征到当前数据
    merge_keys = ['Type_ID_orig', 'From_City_orig', 'To_City']

    # 6.1 Merge 1年特征
    df = df.merge(hist_1y_features, on=merge_keys, how='left')

    # 6.2 Merge 3年平均特征
    df = df.merge(hist_3y_avg, on=merge_keys, how='left')

    # 6.3 Merge 5年平均特征
    df = df.merge(hist_5y_avg, on=merge_keys, how='left')

    # 7. 训练模式：随机 Dropout 历史特征
    if training_mode:
        dropout_prob = 0.2  # 20% 的概率抹除历史特征

        # 生成随机掩码
        mask = np.random.rand(len(df)) < dropout_prob

        if verbose:
            print(f"    [Dropout] Masking {mask.sum():,} rows ({dropout_prob:.0%}) to force learning of attribute features")

        # 将被选中的行的历史特征重置为"冷启动"状态
        df.loc[mask, 'Hist_Flow_Log'] = 0.0
        df.loc[mask, 'Hist_Rank'] = 50
        df.loc[mask, 'Hist_Label'] = 0
        df.loc[mask, 'Hist_Share'] = 0.0
        df.loc[mask, 'Hist_Label_1y'] = 0
        df.loc[mask, 'Hist_Label_3y_avg'] = -1.0  # -1 表示数据不足
        df.loc[mask, 'Hist_Label_5y_avg'] = -1.0  # -1 表示数据不足

    # 8. 填充缺失值（原本就没有历史记录的路线）
    df['Hist_Flow_Log'] = df['Hist_Flow_Log'].fillna(0.0).astype('float32')
    df['Hist_Rank'] = df['Hist_Rank'].fillna(50).astype('int16')
    df['Hist_Label'] = df['Hist_Label'].fillna(0).astype('int16')
    df['Hist_Share'] = df['Hist_Share'].fillna(0.0).astype('float32')
    df['Hist_Label_1y'] = df['Hist_Label_1y'].fillna(0).astype('int16')

    # 【关键修改】多年平均特征：用 -1 表示"数据不足"，而不是 0
    # 这样模型可以区分"没有历史数据"（-1）和"历史排名很差"（0）
    df['Hist_Label_3y_avg'] = df['Hist_Label_3y_avg'].fillna(-1.0).astype('float32')
    df['Hist_Label_5y_avg'] = df['Hist_Label_5y_avg'].fillna(-1.0).astype('float32')

    # 9. 统计信息
    if verbose:
        has_hist_1y = (df['Hist_Rank'] != 50).sum()
        has_hist_3y = (df['Hist_Label_3y_avg'] >= 0).sum()  # >= 0 表示有数据
        has_hist_5y = (df['Hist_Label_5y_avg'] >= 0).sum()  # >= 0 表示有数据
        print(f"    Historical features added:")
        print(f"      - 1-year:  {has_hist_1y:,} / {len(df):,} rows ({has_hist_1y/len(df):.1%})")
        print(f"      - 3-year:  {has_hist_3y:,} / {len(df):,} rows ({has_hist_3y/len(df):.1%})")
        print(f"      - 5-year:  {has_hist_5y:,} / {len(df):,} rows ({has_hist_5y/len(df):.1%})")

    return df
