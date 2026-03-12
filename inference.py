"""
极简在线推理服务
直接调用预计算好的 ratio_cache 和 city_edges，极速返回 Top K
"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

# 配置路径
DATA_DIR = Path("data")
RATIO_CACHE_DIR = DATA_DIR / ".ratio_cache"
CITY_NODES_PATH = DATA_DIR / "city_nodes.jsonl"
CITY_EDGES_PATH = DATA_DIR / "city_edges.jsonl"
MODEL_PATH = Path("output/model_2020_fast.txt")

# 18维特征严格对齐
FEATS = [
    'gender', 'age_group', 'education', 'industry', 'income', 'family',
    'geo_distance', 'dialect_distance',
    'gdp_per_capita_ratio', 'unemployment_rate_ratio', 'housing_price_avg_ratio',
    'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio',
    'education_score_ratio', 'transport_convenience_ratio', 'population_total_ratio',
    'is_same_province'
]

DIMENSIONS = {
    'gender': {'M': 0, 'F': 1},
    'age_group': {'16': 0, '20': 1, '35': 2, '50': 3, '60': 4},
    'education': {'EduLo': 0, 'EduMid': 1, 'EduHi': 2},
    'industry': {'Agri': 0, 'Mfg': 1, 'Service': 2, 'Wht': 3},
    'income': {'IncL': 0, 'IncML': 1, 'IncM': 2, 'IncMH': 3, 'IncH': 4},
    'family': {'Split': 0, 'Unit': 1}
}

class FastRecallPredictor:
    def __init__(self, model_path: str):
        print(f"Loading Model from {model_path}...")
        self.model = lgb.Booster(model_file=str(model_path))
        
        # 1. 加载所有候选城市 ID 与名称
        self.city_id_to_name = {}
        with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                # 兼容不同格式的 jsonl
                cid = int(d.get('city_id', d.get('id', 0)))
                name = d.get('name', d.get('city_name', str(cid)))
                if cid > 0:
                    self.city_id_to_name[cid] = name
        self.all_cities = np.array(list(self.city_id_to_name.keys()), dtype=np.int32)
        
        # 2. 预加载所有城市之间的边特征 (O(1) 查询)
        print("Loading City Edges...")
        self.edges_dict = {}
        with open(CITY_EDGES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                self.edges_dict[(int(d['source_id']), int(d['target_id']))] = (float(d['w_geo']), float(d['w_dialect']))

    def parse_type_id_vectorized(self, type_id: str, n_repeats: int) -> dict:
        """解析 type_id 并广播到 n_repeats 行"""
        parts = type_id.split('_')
        # 如果长度超出6（例如带有 _1100 后缀），截断即可
        return {
            'gender': np.full(n_repeats, DIMENSIONS['gender'].get(parts[0], 0), dtype=np.int8),
            'age_group': np.full(n_repeats, DIMENSIONS['age_group'].get(parts[1], 0), dtype=np.int8),
            'education': np.full(n_repeats, DIMENSIONS['education'].get(parts[2], 0), dtype=np.int8),
            'industry': np.full(n_repeats, DIMENSIONS['industry'].get(parts[3], 0), dtype=np.int8),
            'income': np.full(n_repeats, DIMENSIONS['income'].get(parts[4], 0), dtype=np.int8),
            'family': np.full(n_repeats, DIMENSIONS['family'].get(parts[5], 0), dtype=np.int8),
        }

    def predict_single(self, year: int, type_id: str, from_city: int, top_k: int = 20):
        # 1. 剔除出发城市，生成候选 To_City 列表
        candidate_cities = self.all_cities[self.all_cities != from_city]
        n_cands = len(candidate_cities)
        
        # 2. 构建特征 DataFrame
        df_pred = pd.DataFrame({'To_City': candidate_cities})
        
        # 插入 Type_ID 6维特征
        for k, v in self.parse_type_id_vectorized(type_id, n_cands).items():
            df_pred[k] = v
            
        # 插入 is_same_province
        df_pred['is_same_province'] = (from_city // 100 == candidate_cities // 100).astype(np.int8)
        
        # 插入 Geo 和 Dialect 距离
        geo_dists, dialect_dists = [], []
        for t_city in candidate_cities:
            geo, dialect = self.edges_dict.get((from_city, t_city), (-1.0, -1.0))
            geo_dists.append(geo)
            dialect_dists.append(dialect)
        df_pred['geo_distance'] = np.array(geo_dists, dtype=np.float32)
        df_pred['dialect_distance'] = np.array(dialect_dists, dtype=np.float32)
        
        # 3. 动态加载当年的 Ratio 特征（仅提取需要的 from_city 那一行）
        ratio_file = RATIO_CACHE_DIR / f'city_ratios_{year}.jsonl'
        ratio_features = {c: -1.0 for c in candidate_cities} # 默认值字典，保存每个 to_city 的 9个维度的 list
        
        if ratio_file.exists():
            with open(ratio_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    if int(record['from_city']) == from_city:
                        for t_str, ratios in record['to_cities'].items():
                            t_city = int(t_str)
                            if t_city in candidate_cities:
                                ratio_features[t_city] = [
                                    ratios.get('gdp_per_capita_ratio', 0.0),
                                    ratios.get('unemployment_rate_ratio', 0.0),
                                    ratios.get('housing_price_avg_ratio', 0.0),
                                    ratios.get('rent_avg_ratio', 0.0),
                                    ratios.get('daily_cost_index_ratio', 0.0),
                                    ratios.get('medical_score_ratio', 0.0),
                                    ratios.get('education_score_ratio', 0.0),
                                    ratios.get('transport_convenience_ratio', 0.0),
                                    ratios.get('population_total_ratio', 0.0)
                                ]
                        break # 找到了就直接跳出

        # 展开 Ratio 特征并拼接入 df
        ratio_matrix = np.array([
            ratio_features[tc] if isinstance(ratio_features[tc], list) else [0.0]*9 
            for tc in candidate_cities
        ], dtype=np.float32)
        
        ratio_col_names = [
            'gdp_per_capita_ratio', 'unemployment_rate_ratio', 'housing_price_avg_ratio',
            'rent_avg_ratio', 'daily_cost_index_ratio', 'medical_score_ratio',
            'education_score_ratio', 'transport_convenience_ratio', 'population_total_ratio'
        ]
        
        for i, col in enumerate(ratio_col_names):
            df_pred[col] = ratio_matrix[:, i]
            
        # 4. 预测打分
        X = df_pred[FEATS]
        scores = self.model.predict(X)
        df_pred['Score'] = scores
        
        # 5. 降序排序取 Top K
        df_topk = df_pred.sort_values(by='Score', ascending=False).head(top_k).copy()
        df_topk['Rank'] = np.arange(1, len(df_topk) + 1)
        df_topk['To_City_Name'] = df_topk['To_City'].map(self.city_id_to_name)
        df_topk['From_City'] = from_city
        df_topk['From_City_Name'] = self.city_id_to_name.get(from_city, str(from_city))
        
        return df_topk[['Rank', 'From_City_Name', 'To_City_Name', 'To_City', 'Score']]

if __name__ == '__main__':
    predictor = FastRecallPredictor(MODEL_PATH)
    
    # 你要求的单条请求测试
    res = predictor.predict_single(
        year=2020, 
        type_id='F_20_EduHi_Service_IncM_Unit', 
        from_city=3301, 
        top_k=10
    )
    
    print("\n========= Recall Result =========")
    print(res.to_string(index=False))