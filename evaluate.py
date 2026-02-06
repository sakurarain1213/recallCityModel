# uv run evaluate.py    【一定要在cmd 不要powershell】

import gc
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
import pickle
from sklearn.metrics import ndcg_score
from src.config import Config
from src.city_data import CityDataLoader
from src.feature_eng import parse_type_id

# ==============================================================================
# 全局缓存 (单例模式)
# ==============================================================================
class FastContext:
    """
    高速上下文缓存
    存储：
    1. 静态城市对特征 (Static Pair Features)
    2. 历史年份的查找表 (Historical Lookup)
    """
    _instance = None
    
    def __init__(self):
        self.static_features = None
        self.history_lookup = {} # key: year, value: DataFrame
        self.city_ids = []
        self.feature_cols = []
        self.model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FastContext()
        return cls._instance

    def load_static_data(self):
        """加载静态特征矩阵"""
        path = Path(Config.OUTPUT_DIR) / 'static_city_pairs.parquet'
        if not path.exists():
            raise FileNotFoundError(f"请先运行 evaluate_pre.py 生成 {path}")

        print(f"[FastContext] Loading static city pairs from {path}...")
        self.static_features = pd.read_parquet(path)

        # 【修复】统一类型为 int16
        self.static_features['From_City'] = self.static_features['From_City'].astype('int16')
        self.static_features['To_City'] = self.static_features['To_City'].astype('int16')

        # 加载城市ID列表
        self.city_ids = self.static_features['To_City'].unique().tolist()
        print(f"[FastContext] Loaded {len(self.static_features):,} pairs.")

    def load_history_year(self, year):
        """
        加载特定年份的历史数据到内存
        """
        target_hist_year = year - 1
        if target_hist_year in self.history_lookup:
            return

        path = Path(Config.OUTPUT_DIR) / 'processed_data' / f"processed_{target_hist_year}.parquet"
        print(f"[FastContext] Pre-loading history data from {path}...")
        
        if not path.exists():
            print("Warning: History file not found, using empty.")
            self.history_lookup[target_hist_year] = pd.DataFrame()
            return

        # 只读取需要的列
        cols = ['Type_ID_orig', 'From_City_orig', 'To_City', 'Flow_Count', 'Rank', 'Label']
        df = pd.read_parquet(path, columns=cols)
        
        # 过滤正样本 (Label > 0) 减小内存占用
        df = df[df['Label'] > 0].copy()
        
        # 重命名以匹配特征工程逻辑
        df = df.rename(columns={
            'Flow_Count': 'Hist_Flow_LastYear', # 暂存名字
            'Rank': 'Hist_Rank',
            'Label': 'Hist_Label',
            'Type_ID_orig': 'Type_ID',   # 对齐
            'From_City_orig': 'From_City' # 对齐
        })

        # 【修复】转换类型以便 Merge，确保与静态特征一致
        df['From_City'] = df['From_City'].astype(str).astype('int16')
        df['To_City'] = df['To_City'].astype(str).astype('int16')
        
        # 计算 Log Flow
        df['Hist_Flow_Log'] = np.log1p(df['Hist_Flow_LastYear']).astype('float32')
        df['Hist_Label_1y'] = df['Hist_Label'].astype('int16')
        
        # 只需要保留特征列
        keep_cols = ['Type_ID', 'From_City', 'To_City', 
                     'Hist_Flow_Log', 'Hist_Rank', 'Hist_Label', 'Hist_Label_1y']
        
        self.history_lookup[target_hist_year] = df[keep_cols]
        print(f"[FastContext] History loaded: {len(df):,} records")


def load_model_fast(model_path):
    """加载模型和特征列"""
    ctx = FastContext.get_instance()
    
    if ctx.model is None:
        model_path = Path(model_path)
        feature_path = model_path.parent / 'feature_cols.pkl'
        
        ctx.model = lgb.Booster(model_file=str(model_path))
        with open(feature_path, 'rb') as f:
            ctx.feature_cols = pickle.load(f)
        print(f"Model loaded: {len(ctx.feature_cols)} features")
        
    return ctx.model, ctx.feature_cols


def fast_batch_predict(df_queries, year, ctx):
    """
    极速批量预测

    Args:
        df_queries: DataFrame, 包含 ['Year', 'Month', 'Type_ID', 'From_City']
        year: 当前年份
        ctx: FastContext 对象
    """
    print(f"  [1/6] 构造候选集 ({len(df_queries)} 个查询 × 337 个城市)...")

    # 1. 构造候选集 (Cross Join)
    # 每一行 Query 都要扩展出 337 个 To_City
    # 使用 merge 'key' 的方式做笛卡尔积

    # 这里的 Type_ID 必须是原始字符串，用于匹配历史特征
    # 同时我们需要解析后的 Type_ID (int) 用于模型输入

    # A. 准备 Query 数据
    df_queries['key'] = 1
    # 只需要 From_City (int16)
    df_queries['From_City'] = df_queries['From_City'].astype('int16')

    # B. 准备 Target Cities (从静态表里取唯一值)
    df_targets = pd.DataFrame({'To_City': ctx.city_ids, 'key': 1})
    df_targets['To_City'] = df_targets['To_City'].astype('int16')

    # C. 笛卡尔积 -> 生成所有候选
    # (Num_Queries * 337) 行
    candidates = pd.merge(df_queries, df_targets, on='key').drop('key', axis=1)

    # 排除 From == To
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    print(f"     生成了 {len(candidates):,} 个候选样本")

    # 【修复】添加 qid 列（查询ID），确保特征数量与训练时一致
    query_cols = ['Year', 'Type_ID', 'From_City']
    candidates['qid'] = candidates.groupby(query_cols).ngroup()

    print(f"  [2/6] 合并静态特征（城市属性、距离）...")
    # 2. Merge 静态特征 (Distance, Ratios)
    # 这一步极快，因为是 Int16 Join
    candidates = pd.merge(candidates, ctx.static_features, on=['From_City', 'To_City'], how='left')

    print(f"  [3/6] 合并历史特征（去年的流量、排名）...")
    # 3. Merge 历史特征 (Memory Lookup)
    if (year - 1) in ctx.history_lookup:
        hist_df = ctx.history_lookup[year - 1]
        # Merge Key: Type_ID(str), From_City(int), To_City(int)
        candidates = pd.merge(candidates, hist_df, on=['Type_ID', 'From_City', 'To_City'], how='left')

        # 填充缺失值 (NaNs means no history)
        candidates['Hist_Flow_Log'] = candidates['Hist_Flow_Log'].fillna(0.0)
        candidates['Hist_Rank'] = candidates['Hist_Rank'].fillna(50)
        candidates['Hist_Label'] = candidates['Hist_Label'].fillna(0)
        candidates['Hist_Label_1y'] = candidates['Hist_Label_1y'].fillna(0)
        # 注意：这里简化了 3y 和 5y 平均，为了速度暂时只用 1y，或者你可以预计算好 3y 放进去
        candidates['Hist_Label_3y_avg'] = -1.0
        candidates['Hist_Label_5y_avg'] = -1.0
        candidates['Hist_Share'] = 0.0 # 简化，如果需要可以预计算

    else:
        # 没有历史数据，全填默认值
        candidates['Hist_Flow_Log'] = 0.0
        candidates['Hist_Rank'] = 50
        candidates['Hist_Label'] = 0
        candidates['Hist_Label_1y'] = 0
        candidates['Hist_Label_3y_avg'] = -1.0
        candidates['Hist_Label_5y_avg'] = -1.0
        candidates['Hist_Share'] = 0.0

    print(f"  [4/6] 解析 Type_ID（性别、年龄、学历等）...")
    # 4. 解析 Type_ID (String -> 6 Ints)
    # 因为 parse_type_id 比较慢，我们已经有 Type_ID 列，直接 apply 可能会慢
    # 优化：Type_ID 是重复的，对 unique Type_ID 解析一次，然后 merge 回去
    unique_types = candidates[['Type_ID']].drop_duplicates().copy()
    unique_types_parsed, _ = parse_type_id(unique_types, verbose=False)

    # 【修复】parse_type_id 会删除 Type_ID 列并创建 Type_ID_orig
    # 我们需要保留原始的 Type_ID 用于 merge
    if 'Type_ID_orig' in unique_types_parsed.columns and 'Type_ID' not in unique_types_parsed.columns:
        # 恢复 Type_ID 列用于 merge
        unique_types_parsed['Type_ID'] = unique_types['Type_ID'].values

    # 删掉 Type_ID_orig 避免重复
    if 'Type_ID_orig' in unique_types_parsed.columns:
        unique_types_parsed = unique_types_parsed.drop(columns=['Type_ID_orig'])

    # Merge 回去（只保留解析后的特征列）
    type_feature_cols = [col for col in unique_types_parsed.columns if col != 'Type_ID']
    candidates = pd.merge(candidates, unique_types_parsed[['Type_ID'] + type_feature_cols], on='Type_ID', how='left')

    print(f"  [5/6] 准备特征矩阵（{len(ctx.feature_cols)} 个特征）...")
    # 5. 准备预测特征矩阵 X
    # 确保列存在且顺序一致
    for col in ctx.feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0

    X = candidates[ctx.feature_cols]

    print(f"  [6/6] LightGBM 批量预测...")
    # 6. 预测
    preds = ctx.model.predict(X)
    candidates['pred_score'] = preds
    print(f"     预测完成！")

    return candidates

# ==============================================================================
# 评估逻辑
# ==============================================================================

def calculate_metrics_fast(candidates, df_pos):
    """
    向量化计算 Recall, Normalized Hits 等指标 (极速版)
    完全移除 for 循环，利用 pandas merge 进行集合运算
    一次性计算 @5, @10, @20
    """
    print("     [Metrics] 正在进行向量化评估...")

    # 1. 准备标准答案 (Ground Truth)
    # 筛选正样本
    true_df = df_pos[df_pos['Label'] >= 0.99].copy()

    # 对齐列名 (df_pos 中通常是 Type_ID_orig，candidates 中是 Type_ID)
    if 'Type_ID_orig' in true_df.columns:
        true_df = true_df.rename(columns={'Type_ID_orig': 'Type_ID'})

    # 确保类型一致 (int16) 以便快速 Merge
    true_df['From_City'] = true_df['From_City'].astype('int16')
    true_df['To_City'] = true_df['To_City'].astype('int16')

    # 只保留需要的列：Key(Year, Type, From) + Target(To_City)
    truth_set = true_df[['Year', 'Type_ID', 'From_City', 'To_City']].drop_duplicates()

    # 计算每个 Query 的真实正样本数 (Total True)
    # 限制 Truth 范围在当前的评估集内
    unique_queries = candidates[['Year', 'Type_ID', 'From_City']].drop_duplicates()
    relevant_truth = pd.merge(truth_set, unique_queries, on=['Year', 'Type_ID', 'From_City'], how='inner')
    truth_counts = relevant_truth.groupby(['Year', 'Type_ID', 'From_City']).size().reset_index(name='total_true')

    # 2. 提取预测 Top 20
    # 先按分数降序排列
    candidates = candidates.sort_values(
        by=['Year', 'Type_ID', 'From_City', 'pred_score'],
        ascending=[True, True, True, False]
    )

    # 计算排名 (1-based)
    candidates['rank'] = candidates.groupby(['Year', 'Type_ID', 'From_City']).cumcount() + 1
    
    # 截取 Top 20 用于计算
    top20_preds = candidates[candidates['rank'] <= 20].copy()

    # 3. 计算命中数 (Hits)
    # 通过 Inner Merge 找出既在 Top20 预测中，又在 Truth 中的记录
    hits = pd.merge(
        top20_preds,
        truth_set,
        on=['Year', 'Type_ID', 'From_City', 'To_City'],
        how='inner'
    )

    # 统计不同 K 下的命中数
    hits['hit_5'] = (hits['rank'] <= 5).astype(int)
    hits['hit_10'] = (hits['rank'] <= 10).astype(int)
    hits['hit_20'] = (hits['rank'] <= 20).astype(int)

    # 聚合每个 Query 的 Hit 数
    hits_metrics = hits.groupby(['Year', 'Type_ID', 'From_City'])[['hit_5', 'hit_10', 'hit_20']].sum().reset_index()

    # 4. 合并分母和分子
    results_df = pd.merge(truth_counts, hits_metrics, on=['Year', 'Type_ID', 'From_City'], how='left')

    # 没命中的填 0
    results_df[['hit_5', 'hit_10', 'hit_20']] = results_df[['hit_5', 'hit_10', 'hit_20']].fillna(0)

    # 5. 计算指标
    # 防止除零异常 (理论上 total_true >= 1)
    results_df['total_true'] = results_df['total_true'].replace(0, 1)

    # 标准 Recall (Hits / Total True)
    results_df['recall_5'] = results_df['hit_5'] / results_df['total_true']
    results_df['recall_10'] = results_df['hit_10'] / results_df['total_true']
    results_df['recall_20'] = results_df['hit_20'] / results_df['total_true']

    # 归一化命中率 (Hits / min(Total True, K))
    # 解决了"由于正样本过多导致 Recall 虚低"的问题
    results_df['norm_hit_5'] = results_df['hit_5'] / np.minimum(results_df['total_true'], 5)
    results_df['norm_hit_10'] = results_df['hit_10'] / np.minimum(results_df['total_true'], 10)
    results_df['norm_hit_20'] = results_df['hit_20'] / np.minimum(results_df['total_true'], 20)

    return results_df

def evaluate_fast(model_path, year, sample_size=10):
    # 1. 初始化环境
    print("初始化环境...")
    ctx = FastContext.get_instance()
    ctx.load_static_data() # 加载静态表
    ctx.load_history_year(year) # 加载历史
    load_model_fast(model_path) # 加载模型

    # 2. 加载测试集的"问题"（Queries）和"答案"（Positives）
    print(f"加载测试数据 {year}...")
    path = Path(Config.OUTPUT_DIR) / 'processed_data' / f"processed_{year}.parquet"
    df_orig = pd.read_parquet(path)

    # 正样本 (Label >= 0.99)
    df_pos = df_orig[df_orig['Label'] >= 0.99].copy()

    # 提取唯一的 Queries
    unique_queries = df_pos[['Year', 'Type_ID_orig', 'From_City']].drop_duplicates()
    unique_queries = unique_queries.rename(columns={'Type_ID_orig': 'Type_ID'})

    # 抽样
    if sample_size:
        unique_queries = unique_queries.sample(n=sample_size, random_state=42)

    print(f"评估 {len(unique_queries)} 个查询...")

    # 补充 Month 列 (如果模型训练时用了Month)
    unique_queries['Month'] = 12

    # 3. 批量预测（添加进度条）
    print("批量生成候选集并预测...")
    candidates_with_scores = fast_batch_predict(unique_queries, year, ctx)

    # 4. 计算指标（添加进度条）
    print("计算评估指标...")
    metrics = calculate_metrics_fast(candidates_with_scores, df_pos)

    print("\n" + "="*40)
    print(f"快速评估结果 ({year})")
    print("="*40)
    print(f"查询数量 (Queries): {len(metrics)}")
    print(f"平均正样本数 (Avg GT Size): {metrics['total_true'].mean():.2f}")
    
    print("-" * 30)
    print("Standard Recall (Hits / Total_True):")
    print(f"  Recall@5 : {metrics['recall_5'].mean():.4f}")
    print(f"  Recall@10: {metrics['recall_10'].mean():.4f}")
    print(f"  Recall@20: {metrics['recall_20'].mean():.4f}")
    
    print("-" * 30)
    print("Normalized Hits (Hits / min(Total, K)):")
    print("  (解决了正样本 > K 时 Recall 虚低的问题)")
    print(f"  Norm@5   : {metrics['norm_hit_5'].mean():.4f}")
    print(f"  Norm@10  : {metrics['norm_hit_10'].mean():.4f}")
    print(f"  Norm@20  : {metrics['norm_hit_20'].mean():.4f}")

    return metrics

# ==============================================================================
# 单次推理接口
# ==============================================================================

def predict_one(year, type_id, from_city, model_path):
    """
    极速单次推理
    """
    ctx = FastContext.get_instance()
    # 懒加载
    if ctx.static_features is None:
        ctx.load_static_data()
        ctx.load_history_year(year)
        load_model_fast(model_path)

    # 构造单行 Query
    query_df = pd.DataFrame([{
        'Year': year,
        'Month': 12, # 默认
        'Type_ID': type_id,
        'From_City': int(from_city)  # 【修复】确保是整数
    }])

    # 预测
    result = fast_batch_predict(query_df, year, ctx)

    # 排序取 Top 10
    top10 = result.nlargest(10, 'pred_score')

    return top10[['To_City', 'pred_score']].values.tolist()

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\w1625\Desktop\reranker-train\output\models\binary_model.txt"

    '''
output\models\binary_model.txt
    '''

    # 1. 采样评估
    print("\n" + "="*80)
    print("开始采样评估...")
    print("="*80)
    evaluate_fast(MODEL_PATH, 2020, sample_size=100000)

    # 2. 快速推理示例
    print("\n" + "="*80)
    print("开始快速推理示例...")
    print("="*80)
    import time
    t0 = time.time()

    top10 = predict_one(2020, 'F_20_EduHi_Service_IncH_Split', 3301, MODEL_PATH)

    t1 = time.time()
    print(f"\n推理时间: {(t1-t0)*1000:.2f} ms")

    # 加载城市名称映射
    import json
    city_id_to_name = {}
    city_nodes_path = Path(Config.DATA_DIR) / 'city_nodes.jsonl'
    if city_nodes_path.exists():
        with open(city_nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                node = json.loads(line)
                city_id_to_name[str(node['city_id'])] = node['name']

    # 打印 Top 10 城市（ID + 名称）
    print("\nTop 10 城市:")
    for i, (city_id, score) in enumerate(top10, 1):
        city_id_str = str(int(city_id))
        city_name = city_id_to_name.get(city_id_str, "未知")
        print(f"  {i}. {city_id_str} ({city_name}) - 分数: {score:.4f}")