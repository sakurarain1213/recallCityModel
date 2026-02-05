"""
城市数据加载模块
从 JSONL 文件加载城市信息、边关系和节点信息
"""
import json
import pandas as pd
from pathlib import Path


class CityDataLoader:
    """城市数据加载器"""

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.city_info = None
        self.city_edges = None
        self.city_nodes = None
        self.city_ids = None

    def load_all(self):
        """加载所有城市数据"""
        self.load_city_info()
        self.load_city_edges()
        self.load_city_nodes()
        return self

    def load_city_info(self):
        """
        加载城市详细信息（静态属性）
        包括：坐标、产业结构、经济指标、人口结构等
        """
        print("Loading city info from cities_data.jsonl...")
        city_info_path = self.data_dir / 'cities_data.jsonl'

        if not city_info_path.exists():
            print(f"Warning: {city_info_path} not found, using empty data")
            self.city_info = pd.DataFrame()
            return self

        data = []
        with open(city_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # 展平嵌套的 JSON 结构
        flattened_data = []
        for item in data:
            flat_item = {
                'city_id': item['city_id'],
                'city_name': item['city_name'],
                # 基本信息
                'tier': item['basic_info']['tier'],
                'area_sqkm': item['basic_info']['area_sqkm'],
                # 经济指标
                'gdp_per_capita': item['economy']['gdp_per_capita'],
                'cpi_index': item['economy']['cpi_index'],
                'unemployment_rate': item['economy']['unemployment_rate'],
                # 产业结构
                'agriculture_share': item['economy']['industry_sectors']['agriculture']['share'],
                'agriculture_wage': item['economy']['industry_sectors']['agriculture']['avg_wage'],
                'manufacturing_share': item['economy']['industry_sectors']['manufacturing']['share'],
                'manufacturing_wage': item['economy']['industry_sectors']['manufacturing']['avg_wage'],
                'traditional_services_share': item['economy']['industry_sectors']['traditional_services']['share'],
                'traditional_services_wage': item['economy']['industry_sectors']['traditional_services']['avg_wage'],
                'modern_services_share': item['economy']['industry_sectors']['modern_services']['share'],
                'modern_services_wage': item['economy']['industry_sectors']['modern_services']['avg_wage'],
                # 生活成本
                'housing_price_avg': item['living_cost']['housing_price_avg'],
                'rent_avg': item['living_cost']['rent_avg'],
                'daily_cost_index': item['living_cost']['daily_cost_index'],
                # 公共服务
                'medical_score': item['public_services']['medical_score'],
                'education_score': item['public_services']['education_score'],
                'transport_convenience': item['public_services']['transport_convenience'],
                'avg_commute_mins': item['public_services']['avg_commute_mins'],
                # 人口
                'population_total': item['social_context']['population_total'],
            }
            flattened_data.append(flat_item)

        self.city_info = pd.DataFrame(flattened_data)
        self.city_info.set_index('city_id', inplace=True)
        print(f"Loaded {len(self.city_info)} cities info")
        return self

    def load_city_edges(self):
        """
        加载城市边关系（距离信息）
        包括：地理距离、方言距离
        """
        print("Loading city edges from city_edges.jsonl...")
        edges_path = self.data_dir / 'city_edges.jsonl'

        if not edges_path.exists():
            print(f"Warning: {edges_path} not found, using empty data")
            self.city_edges = pd.DataFrame()
            return self

        data = []
        with open(edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                edge = json.loads(line)
                data.append({
                    'source_id': edge['source_id'],
                    'target_id': edge['target_id'],
                    'w_geo': edge['w_geo'],
                    'w_dialect': edge['w_dialect'],
                })

        self.city_edges = pd.DataFrame(data)
        print(f"Loaded {len(self.city_edges)} city edges")
        return self

    def load_city_nodes(self):
        """
        加载城市节点信息（基础映射）
        包括：城市ID和名称
        """
        print("Loading city nodes from city_nodes.jsonl...")
        nodes_path = self.data_dir / 'city_nodes.jsonl'

        if not nodes_path.exists():
            print(f"Warning: {nodes_path} not found, using empty data")
            self.city_nodes = pd.DataFrame()
            self.city_ids = []
            return self

        data = []
        with open(nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                node = json.loads(line)
                data.append({
                    'city_id': node['city_id'],
                    'city_name': node['name'],
                })

        self.city_nodes = pd.DataFrame(data)
        self.city_ids = self.city_nodes['city_id'].tolist()
        print(f"Loaded {len(self.city_nodes)} city nodes")
        return self

    def get_city_ids(self):
        """获取所有城市ID列表"""
        return self.city_ids

    def get_city_id_to_name(self):
        """获取城市ID到名称的映射"""
        if self.city_nodes is None or self.city_nodes.empty:
            return {}
        return dict(zip(self.city_nodes['city_id'], self.city_nodes['city_name']))

    def get_city_name_to_id(self):
        """获取城市名称到ID的映射"""
        if self.city_nodes is None or self.city_nodes.empty:
            return {}
        return dict(zip(self.city_nodes['city_name'], self.city_nodes['city_id']))

    def get_edge_features(self, source_id, target_id):
        """
        获取两个城市之间的边特征
        返回: (w_geo, w_dialect) 或 (None, None)
        """
        if self.city_edges is None or self.city_edges.empty:
            return None, None

        edge = self.city_edges[
            (self.city_edges['source_id'] == source_id) &
            (self.city_edges['target_id'] == target_id)
        ]

        if edge.empty:
            return None, None

        return edge.iloc[0]['w_geo'], edge.iloc[0]['w_dialect']

    def get_city_attributes(self, city_id):
        """
        获取城市的所有属性
        返回: dict 或 None
        """
        if self.city_info is None or self.city_info.empty:
            return None

        if city_id not in self.city_info.index:
            return None

        return self.city_info.loc[city_id].to_dict()
