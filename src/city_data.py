"""
城市数据加载模块
从 JSONL 文件加载城市信息、边关系和节点信息
"""
import json
import pandas as pd
from pathlib import Path


class CityDataLoader:
    """城市数据加载器"""

    def __init__(self, data_dir='/data1/wxj/Recall_city_project/data'):
        self.data_dir = Path(data_dir)
        self.city_info_dir = self.data_dir / 'cities_2000-2020'  # 年度城市信息目录
        self.city_info = {}  # 改为字典,按年份存储: {year: DataFrame}
        self.city_edges = None
        self.city_nodes = None
        self.city_ids = None

    def _safe_int(self, val):
        """辅助函数:安全转换为int"""
        try:
            return int(val)
        except:
            return -1

    def load_all(self, year=None):
        """
        加载所有城市数据

        Args:
            year: 指定年份的城市数据,如果为None则加载所有年份
        """
        self.load_city_info(year)
        self.load_city_edges()
        self.load_city_nodes()
        return self

    def load_city_info(self, year=None):
        """
        加载城市详细信息(年度动态属性)

        Args:
            year: 指定年份(2000-2020),如果为None则加载所有年份
                  数据目录: data/cities_2000-2020/cities_{year}.jsonl
        """
        # 确定要加载的年份
        if year is not None:
            years_to_load = [year]
            print(f"Loading city info for year {year}...")
        else:
            # 加载所有年份(2000-2020)
            years_to_load = range(2000, 2021)
            print("Loading city info for all years (2000-2020)...")

        # 加载指定年份的数据
        for yr in years_to_load:
            city_info_path = self.city_info_dir / f'cities_{yr}.jsonl'

            if not city_info_path.exists():
                print(f"Warning: {city_info_path} not found, skipping year {yr}")
                continue

            data = []
            with open(city_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            # 展平嵌套的 JSON 结构
            flattened_data = []
            for item in data:
                # 产业结构安全获取
                sectors = item['economy'].get('industry_sectors', {})

                flat_item = {
                    'city_id': self._safe_int(item['city_id']),  # 强制转 Int
                    'city_name': item['city_name'],
                    # 基本信息
                    'tier': item['basic_info']['tier'],
                    'area_sqkm': item['basic_info']['area_sqkm'],
                    # 经济指标
                    'gdp_per_capita': item['economy']['gdp_per_capita'],
                    'cpi_index': item['economy']['cpi_index'],
                    'unemployment_rate': item['economy']['unemployment_rate'],
                    # 产业结构（使用安全获取）
                    'agriculture_share': sectors.get('agriculture', {}).get('share', 0),
                    'agriculture_wage': sectors.get('agriculture', {}).get('avg_wage', 0),
                    'manufacturing_share': sectors.get('manufacturing', {}).get('share', 0),
                    'manufacturing_wage': sectors.get('manufacturing', {}).get('avg_wage', 0),
                    'traditional_services_share': sectors.get('traditional_services', {}).get('share', 0),
                    'traditional_services_wage': sectors.get('traditional_services', {}).get('avg_wage', 0),
                    'modern_services_share': sectors.get('modern_services', {}).get('share', 0),
                    'modern_services_wage': sectors.get('modern_services', {}).get('avg_wage', 0),
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

            df = pd.DataFrame(flattened_data)
            # 确保索引是 Int
            df['city_id'] = df['city_id'].astype(int)
            df.set_index('city_id', inplace=True)
            self.city_info[yr] = df
            print(f"Loaded {len(df)} cities info for year {yr}")

        return self

    def get_city_info_for_year(self, year):
        """
        获取指定年份的城市信息DataFrame

        Args:
            year: 年份(2000-2020)

        Returns:
            DataFrame or None(如果该年份数据不存在)
        """
        return self.city_info.get(year)

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
                    'source_id': self._safe_int(edge['source_id']),  # 强制转 Int
                    'target_id': self._safe_int(edge['target_id']),  # 强制转 Int
                    'w_geo': edge['w_geo'],
                    'w_dialect': edge['w_dialect'],
                })

        self.city_edges = pd.DataFrame(data)
        # 确保列类型为 Int
        self.city_edges['source_id'] = self.city_edges['source_id'].astype(int)
        self.city_edges['target_id'] = self.city_edges['target_id'].astype(int)
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
                    'city_id': self._safe_int(node['city_id']),  # 强制转 Int
                    'city_name': node['name'],
                })

        self.city_nodes = pd.DataFrame(data)
        if not self.city_nodes.empty:
            self.city_nodes['city_id'] = self.city_nodes['city_id'].astype(int)
            self.city_ids = self.city_nodes['city_id'].tolist()  # 这里输出的就是 Int List 了
        else:
            self.city_ids = []
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

    def get_city_attributes(self, city_id, year=None):
        """
        获取城市的所有属性

        Args:
            city_id: 城市ID
            year: 年份(2000-2020),如果为None则返回2010年的数据作为默认

        Returns:
            dict 或 None
        """
        # 默认使用2010年数据
        if year is None:
            year = 2010

        # 获取该年份的城市信息
        city_info_df = self.city_info.get(year)
        if city_info_df is None or city_info_df.empty:
            # 如果指定年份不存在,尝试使用最接近的年份
            available_years = sorted(self.city_info.keys())
            if available_years:
                # 找最接近的年份
                year = min(available_years, key=lambda y: abs(y - year))
                city_info_df = self.city_info.get(year)
            else:
                return None

        if city_id not in city_info_df.index:
            return None

        return city_info_df.loc[city_id].to_dict()
