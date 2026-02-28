"""
真实场景推理脚本
模拟线上业务调用：给定一个请求 (年份, 人群类型, 出发城市)，
利用模型预测其最可能流向的 Top K 目标城市。

【依赖文件说明】 (请确保以下文件存在于项目根目录的相对路径中)
1. 模型文件:
   - output/lgb_batch_90.txt (或在 main 中指定的其他模型路径)

2. 数据文件 (位于 data/ 目录下):
   - data/cities_2000-2020/cities_{year}.jsonl (例如: data/cities_2000-2020/cities_2020.jsonl)
   - data/city_edges.jsonl
   - data/city_nodes.jsonl
"""
import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb

# 导入项目中原有的模块（需确保在项目根目录下运行）
from src.config import Config
from src.city_data import CityDataLoader
from src.feature_pipeline import FeaturePipeline

def predict_top_k_cities(model_path, year, type_id, from_city, top_k=20):
    """
    真实场景下的推理函数
    
    参数:
        model_path: LightGBM 模型路径
        year: 当前年份 (如 2020)
        type_id: 人群类型 (如 "F_40_EduLo_Wht_IncMH_Split_4453")
        from_city: 出发城市 ID (如 4453)
        top_k: 返回前 K 个预测结果
    """
    print(f"=====================================")
    print(f"🚀 开始真实场景推理任务")
    print(f"📅 请求年份: {year}")
    print(f"🧑‍🤝‍🧑 人群类型: {type_id}")
    print(f"📍 出发城市: {from_city}")
    print(f"=====================================")

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    print(f"📂 加载模型: {model_path}...")
    bst = lgb.Booster(model_file=model_path)
    model_feats = bst.feature_name()
    print(f"✅ 模型加载成功，期望特征数: {len(model_feats)}")

    # 2. 初始化数据加载器和特征管道
    print("\n⏳ 初始化基础数据 (加载城市属性、边关系)...")
    loader = CityDataLoader(Config.DATA_DIR).load_all()
    pipeline = FeaturePipeline(loader, data_dir=Config.PROCESSED_DIR)
    city_id_to_name = loader.get_city_id_to_name()
    
    from_city_name = city_id_to_name.get(from_city, "未知城市")
    print(f"   出发城市确认为: {from_city_name} ({from_city})")

    # 3. 构造候选集 (Candidate Generation)
    # 获取全国所有候选城市 ID
    all_cities = loader.get_city_ids()
    print(f"\n🔨 构造候选集: 全国共 {len(all_cities)} 个城市候选")

    # 创建基础 DataFrame，包含一个 Query 到所有目标城市的笛卡尔积
    candidates = pd.DataFrame({
        'Year': year,
        'Type_ID': type_id,
        'From_City': from_city,
        'To_City': all_cities
    })
    
    # 排除出发城市本身 (人不会流向出发地本身)
    candidates = candidates[candidates['From_City'] != candidates['To_City']].copy()
    
    # 4. 特征工程 (Feature Engineering)
    # 调用与训练时完全一致的 transform 方法，提取城市距离、经济差异 Ratio、历史特征
    print("✨ 执行统一特征工程流水线 (抽取静态+动态差异特征)...")
    start_time = time.time()
    
    # 补充在 'predict' 模式下需要的占位列，防止底层报错
    candidates['Flow_Count'] = 0
    candidates['Rank'] = 999
    candidates['Label'] = 0.0
    
    df_feats = pipeline.transform(candidates.copy(), year=year, mode='predict', verbose=False)
    
    # 提取模型所需的特征列并对齐
    X = pd.DataFrame(index=df_feats.index)
    for f in model_feats:
        # 如果特征工程生成的列存在则使用，否则用 0.0 填充（安全容错处理）
        X[f] = df_feats[f] if f in df_feats.columns else 0.0

    # 确保数据类型与训练时一致 (主要是 float32 以提速和防溢出)
    for c in X.columns:
        if X[c].dtype == 'float64': 
            X[c] = X[c].astype('float32')
            
    feat_time = time.time() - start_time
    print(f"✅ 特征工程完成，耗时 {feat_time:.2f} 秒")

    # 5. 模型推理 (Scoring)
    print("\n🔮 模型批量打分中...")
    start_time = time.time()
    scores = bst.predict(X)
    infer_time = time.time() - start_time
    
    candidates['Score'] = scores
    print(f"✅ 推理完成，耗时 {infer_time:.4f} 秒")

    # 6. 排序并提取 Top K (Ranking)
    # 根据预测召回概率分数降序排列
    top_preds = candidates.sort_values(by='Score', ascending=False).head(top_k).copy()
    
    # 添加城市名称以方便可视化阅读
    top_preds['From_City_Name'] = top_preds['From_City'].map(city_id_to_name)
    top_preds['To_City_Name'] = top_preds['To_City'].map(city_id_to_name)
    top_preds['Rank'] = range(1, len(top_preds) + 1)
    
    print(f"\n🏆 推理结果 (Top {top_k} 召回城市):")
    print("-" * 55)
    print(f"{'排名':<5} | {'目标城市':<15} | {'模型得分':<15}")
    print("-" * 55)
    
    for _, row in top_preds.iterrows():
        to_city_str = f"{row['To_City_Name']}({row['To_City']})"
        print(f"Top {row['Rank']:<2} | {to_city_str:<15} | {row['Score']:.6f}")
        
    print("-" * 55)
    return top_preds

if __name__ == "__main__":
    # 使用相对路径指向您生成的模型检查点
    MODEL_PATH = os.path.join('output', 'lgb_batch_90.txt')
    
    # 模拟真实业务请求 
    # 从真实 GT 数据中挑选的一条样本：
    # "2020 12 F_40_EduLo_Wht_IncMH_Split_4453 4453 云浮(4453)"
    QUERY_YEAR = 2018
    QUERY_TYPE_ID = "F_40_EduLo_Wht_IncMH_Split_4453"
    QUERY_FROM_CITY = 4453
    
    predict_top_k_cities(
        model_path=MODEL_PATH, 
        year=QUERY_YEAR, 
        type_id=QUERY_TYPE_ID, 
        from_city=QUERY_FROM_CITY,
        top_k=20
    )