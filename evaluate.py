"""
evaluate.py
极速评估脚本 - 复用训练缓存，无需额外生成步骤
"""
# uv run evaluate.py --year 2010    运行评估（例如评估 2010 年）
# uv run evaluate.py --year 2010 --predict 运行评估并演示单次推理
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt

# 导入 src 模块
from src.config import Config
from src.city_data import CityDataLoader
from src.data_loader_v2 import load_raw_data_fast
from src.feature_eng import parse_type_id
from src.historical_features import add_historical_features

# ==============================================================================
# 全局资源管理
# ==============================================================================
class EvalContext:
    def __init__(self):
        self.model = None
        self.global_features = None
        self.city_ids = None
        self.feature_cols = None

    def load_resources(self, model_path):
        print("正在加载评估资源...")
        
        # 1. 加载模型
        try:
            self.model = lgb.Booster(model_file=str(model_path))
            self.feature_cols = self.model.feature_name()
            print(f"✓ 模型已加载，特征数量: {len(self.feature_cols)}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

        # 2. 加载全局特征表 (与训练时一致)
        global_feat_path = Path(Config.OUTPUT_DIR) / 'global_city_features.parquet'
        if not global_feat_path.exists():
            raise FileNotFoundError(f"未找到全局特征表: {global_feat_path}\n请先运行: python src/precompute_static_features.py")
        
        # 读取并优化内存
        self.global_features = pd.read_parquet(global_feat_path)
        self.global_features['From_City'] = self.global_features['From_City'].astype('int16')
        self.global_features['To_City'] = self.global_features['To_City'].astype('int16')
        
        # 获取所有目标城市ID (用于构造候选集)
        self.city_ids = self.global_features['To_City'].unique().astype('int16')
        print(f"✓ 全局特征表已加载: {len(self.global_features):,} 行 (涵盖 {len(self.city_ids)} 个城市)")

# ==============================================================================
# 评估核心逻辑
# ==============================================================================
def evaluate_year(year, ctx, sample_size=None, cache_dir='output/cache'):
    """
    对指定年份进行全量召回评估
    """
    print(f"\n{'='*40}")
    print(f"开始评估年份: {year}")
    print(f"{'='*40}")
    
    # 1. 获取 Ground Truth (真实流向)
    # 使用 load_raw_data_fast 获取原始正样本
    # neg_sample_rate 设为 1 即可，因为我们只筛选正样本
    print("Step 1: 加载测试集 Ground Truth...")
    df_raw = load_raw_data_fast(Config.DB_PATH, year, hard_candidates=[], neg_sample_rate=1)
    
    if df_raw.empty:
        print("❌ 该年份无数据")
        return None

    # 筛选正样本 (Label > 0 表示它是真实存在的流向，包括 Rank 1-20)
    df_pos = df_raw[df_raw['Label'] > 0].copy()
    
    # 提取唯一的 Queries (Year, Type, From)
    queries = df_pos[['Year', 'Type_ID', 'From_City']].drop_duplicates().reset_index(drop=True)
    
    # 采样 (如果配置了)
    if sample_size and len(queries) > sample_size:
        print(f"⚠️ 进行采样评估: {sample_size}/{len(queries)}")
        queries = queries.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"评估全量查询: {len(queries)} 个")

    # 2. 构造全量候选集 (Query x 337 Cities)
    # 这是 Recall 评估的关键：对每个出发的人群，我们要对全国所有城市打分
    print(f"Step 2: 生成候选集 ({len(queries)} Queries x {len(ctx.city_ids)} Cities)...")
    
    # 使用 Cross Join 构造
    # 技巧：给两边都加一个常数 key 进行 merge
    queries['key'] = 1
    targets = pd.DataFrame({'To_City': ctx.city_ids, 'key': 1})
    
    # 笛卡尔积 (可能很大，注意内存)
    candidates = pd.merge(queries, targets, on='key').drop('key', axis=1)
    
    # 排除 From == To 的情况 (自己不能流向自己)
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    
    print(f"候选集大小: {len(candidates):,} 行")
    
    # 3. 特征工程 (复用训练时的逻辑)
    print("Step 3: 特征工程...")
    
    # A. 合并静态特征 (Year, From, To)
    # 注意：global_features 包含 Year 列，会自动对齐
    candidates = candidates.merge(
        ctx.global_features, 
        on=['Year', 'From_City', 'To_City'], 
        how='left'
    )
    
    # B. 解析 Type_ID (Gender, Age, etc.)
    # 对 unique Type_ID 解析一次，然后 merge 回去 (比直接 apply 快 100倍)
    unique_types = candidates[['Type_ID']].drop_duplicates()
    unique_types_parsed, _ = parse_type_id(unique_types, verbose=False)
    
    # 如果 parse_type_id 删除了 Type_ID 列，需要恢复以便 merge
    if 'Type_ID' not in unique_types_parsed.columns and 'Type_ID_orig' in unique_types_parsed.columns:
         unique_types_parsed['Type_ID'] = unique_types_parsed['Type_ID_orig'] # 恢复用于Merge
    elif 'Type_ID' not in unique_types_parsed.columns:
         # 兜底：如果 parse_type_id 实现改变
         unique_types_parsed['Type_ID'] = unique_types['Type_ID'].values
         
    # 移除 Type_ID_orig 避免重复
    if 'Type_ID_orig' in unique_types_parsed.columns:
        unique_types_parsed = unique_types_parsed.drop(columns=['Type_ID_orig'])
        
    candidates = candidates.merge(unique_types_parsed, on='Type_ID', how='left')

    # C. 添加历史特征
    # 关键：指向 output/cache，因为 fast_train 把处理好的历史数据存在那里
    # training_mode=False (不进行 Dropout，使用全部历史信息)
    candidates = add_historical_features(
        candidates, 
        year, 
        data_dir=Path(cache_dir), 
        verbose=False, 
        training_mode=False
    )
    
    # D. 准备特征矩阵 X
    # 确保列顺序与模型一致，缺失列填 0
    for col in ctx.feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0
            
    X = candidates[ctx.feature_cols]
    
    # 4. 预测
    print("Step 4: 模型打分...")
    candidates['pred_score'] = ctx.model.predict(X)
    
    # 5. 计算指标
    print("Step 5: 计算评估指标...")
    metrics = calculate_metrics(candidates, df_pos)
    
    # 清理内存
    del candidates, X, df_raw, df_pos
    gc.collect()
    
    return metrics

def calculate_metrics(candidates, ground_truth):
    """
    向量化计算 Recall@K
    """
    # 1. 准备 Ground Truth 集合 (Year, Type, From, To)
    gt_set = ground_truth[['Year', 'Type_ID', 'From_City', 'To_City']].copy()
    gt_set['is_true'] = 1
    
    # 2. 对每个 Query 内部按分数排序
    # 使用 groupby + rank (method='first' 保证排名连续)
    # ascending=False 表示分数越高排名越前 (1 是第一名)
    candidates['rank'] = candidates.groupby(['Year', 'Type_ID', 'From_City'])['pred_score'] \
                                   .rank(method='first', ascending=False)
    
    # 3. 只保留 Top 20 的预测结果用于评估 (节省 Join 资源)
    top_preds = candidates[candidates['rank'] <= 20].copy()
    
    # 4. 标记命中情况
    # Left Join Truth: 如果预测的 (Query, To) 在 Truth 里，is_true 就是 1
    merged = pd.merge(
        top_preds, 
        gt_set, 
        on=['Year', 'Type_ID', 'From_City', 'To_City'], 
        how='left'
    )
    merged['is_hit'] = merged['is_true'].fillna(0)
    
    # 5. 聚合计算每个 Query 的命中数
    # 技巧：直接判断 rank <= K 且 is_hit == 1
    hits = merged.groupby(['Year', 'Type_ID', 'From_City']).apply(
        lambda x: pd.Series({
            'hit_1': ((x['rank'] <= 1) & (x['is_hit'] == 1)).sum(),
            'hit_5': ((x['rank'] <= 5) & (x['is_hit'] == 1)).sum(),
            'hit_10': ((x['rank'] <= 10) & (x['is_hit'] == 1)).sum(),
            'hit_20': ((x['rank'] <= 20) & (x['is_hit'] == 1)).sum()
        })
    ).reset_index()
    
    # 6. 计算每个 Query 的真实正样本总数 (分母)
    gt_counts = gt_set.groupby(['Year', 'Type_ID', 'From_City']).size().reset_index(name='total_true')
    
    # 7. 合并分子分母
    eval_df = pd.merge(gt_counts, hits, on=['Year', 'Type_ID', 'From_City'], how='left').fillna(0)
    
    # 8. 计算 Recall (平均值)
    # 防止除以 0 (虽然理论上 total_true >= 1)
    eval_df['total_true'] = eval_df['total_true'].replace(0, 1)
    
    recall_1 = (eval_df['hit_1'] / eval_df['total_true']).mean()
    recall_5 = (eval_df['hit_5'] / eval_df['total_true']).mean()
    recall_10 = (eval_df['hit_10'] / eval_df['total_true']).mean()
    recall_20 = (eval_df['hit_20'] / eval_df['total_true']).mean()
    
    return {
        'recall_1': recall_1,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'recall_20': recall_20,
        'avg_gt_size': eval_df['total_true'].mean(),
        'num_queries': len(eval_df)
    }

# ==============================================================================
# 单次推理接口 (用于演示)
# ==============================================================================
def predict_one(year, type_id, from_city, ctx):
    """
    单次推理：预测某个人群从某城市出发，最可能去的 Top 10 城市
    """
    print(f"\n🔮 单次推理: {year} | {type_id} | From: {from_city}")
    
    # 1. 构造 Query DataFrame
    query = pd.DataFrame([{
        'Year': year,
        'Type_ID': type_id,
        'From_City': int(from_city)
    }])
    
    # 2. 构造 Candidates (1 Query x 337 Cities)
    targets = pd.DataFrame({'To_City': ctx.city_ids})
    targets['key'] = 1
    query['key'] = 1
    candidates = pd.merge(query, targets, on='key').drop('key', axis=1)
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    
    # 3. 特征工程 (简化版)
    candidates = candidates.merge(ctx.global_features, on=['Year', 'From_City', 'To_City'], how='left')
    
    types, _ = parse_type_id(candidates[['Type_ID']].drop_duplicates(), verbose=False)
    # 兼容列名
    if 'Type_ID' not in types.columns and 'Type_ID_orig' in types.columns:
        types['Type_ID'] = types['Type_ID_orig']
        
    candidates = candidates.merge(types, on='Type_ID', how='left')
    
    candidates = add_historical_features(candidates, year, data_dir=Path(Config.OUTPUT_DIR)/'cache', verbose=False)
    
    for col in ctx.feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0
            
    # 4. 预测
    scores = ctx.model.predict(candidates[ctx.feature_cols])
    candidates['score'] = scores
    
    # 5. 排序输出
    top10 = candidates.nlargest(10, 'score')[['To_City', 'score']]
    
    # 尝试加载城市名
    city_map = {}
    try:
        loader = CityDataLoader(Config.DATA_DIR)
        loader.load_city_nodes()
        city_map = loader.get_city_id_to_name()
    except:
        pass
        
    print(f"{'Rank':<5} {'City ID':<10} {'Name':<15} {'Score':<10}")
    print("-" * 45)
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        city_id = int(row['To_City'])
        name = city_map.get(city_id, "Unknown")
        print(f"{i:<5} {city_id:<10} {name:<15} {row['score']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2010, help="评估年份")
    parser.add_argument('--sample', type=int, default=1000, help="采样Query数，0为全量")
    parser.add_argument('--predict', action='store_true', help="运行单次推理演示")
    args = parser.parse_args()
    
    # 初始化上下文
    ctx = EvalContext()
    model_path = Path(Config.OUTPUT_DIR) / 'fast_model.txt'
    
    if not model_path.exists():
        print(f"❌ 未找到模型文件: {model_path}")
    else:
        ctx.load_resources(model_path)
        
        # 缓存目录 (fast_train.py 的输出目录)
        CACHE_DIR = Path(Config.OUTPUT_DIR) / 'cache'
        
        # 运行评估
        metrics = evaluate_year(args.year, ctx, sample_size=args.sample if args.sample > 0 else None, cache_dir=CACHE_DIR)
        
        if metrics:
            print("\n" + "="*40)
            print(f"📊 评估结果报告 ({args.year})")
            print("="*40)
            print(f"Query样本数 : {metrics['num_queries']}")
            print(f"平均正样本数 : {metrics['avg_gt_size']:.2f}")
            print("-" * 30)
            print(f"Recall@1   : {metrics['recall_1']:.2%}")
            print(f"Recall@5   : {metrics['recall_5']:.2%}")
            print(f"Recall@10  : {metrics['recall_10']:.2%}")
            print(f"Recall@20  : {metrics['recall_20']:.2%}")
            print("="*40)
            
        # 运行演示
        if args.predict:
            # 找一个存在的 Type_ID 和 City 演示
            predict_one(args.year, 'F_30_EduHi_Service_IncML_Unit_5119', 5119, ctx)