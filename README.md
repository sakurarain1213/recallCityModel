# 迁移排序模型 (Migration Ranker)

基于 LightGBM LambdaRank 的人口迁移预测排序模型，用于预测各城市人群的迁移目的地 Top 20。

## 项目结构

```
reranker-train/
├── data/                       # 数据目录
│   ├── migration.db           # DuckDB 数据库文件（迁移数据）
│   ├── cities_data.jsonl      # 城市详细信息（经济、产业、人口等）
│   ├── city_edges.jsonl       # 城市边关系（地理距离、方言距离）
│   └── city_nodes.jsonl       # 城市节点（ID和名称映射）
├── output/                     # 输出目录
│   └── feature_importance.png # 特征重要性图
├── src/
│   ├── __init__.py
│   ├── city_data.py           # 城市数据加载器（JSONL）
│   ├── config.py              # 全局配置参数
│   ├── data_loader.py         # 迁移数据加载与转换
│   ├── feature_eng.py         # 特征工程（Type解析、历史特征、交叉特征）
│   └── model.py               # LightGBM Ranker 模型
├── main.py                    # 主训练脚本
├── FEATURES.md                # 特征详细说明文档
└── README.md                  # 本文件
```

## 快速开始

### 1. 环境配置

```bash
# 初始化虚拟环境
uv venv

# 安装依赖
uv pip install pandas numpy lightgbm duckdb matplotlib scikit-learn
```

### 2. 数据准备

#### 2.1 迁移数据（DuckDB）

将迁移数据库文件放置到 `data/migration.db`。

数据格式要求（宽表）：
- `Year`: 年份
- `Month`: 月份
- `Type_ID`: 人群类型ID（格式：`F_20_EduHi_Service_IncML_Unit_1100`）
- `From_City`: 出发城市（格式：`城市名(城市代码)`）
- `Total_Count`: 总迁徙人数
- `To_Top1` ~ `To_Top20`: Top 20 目的地城市
- `To_Top1_Count` ~ `To_Top20_Count`: 对应的迁徙人数

#### 2.2 城市数据（JSONL）

确保 `data/` 目录下有以下三个 JSONL 文件：

**1. cities_data.jsonl** - 城市详细信息
```json
{
  "city_id": "1301",
  "city_name": "石家庄",
  "basic_info": {"tier": 3, "coordinates": [114.51, 38.04], "area_sqkm": 3738.0},
  "economy": {
    "gdp_per_capita": 61000.4,
    "cpi_index": 102.5,
    "unemployment_rate": 0.046,
    "industry_sectors": {
      "agriculture": {"share": 0.175, "avg_wage": 3117, "vacancy_rate": 0.08},
      "manufacturing": {"share": 0.38, "avg_wage": 6833, "vacancy_rate": 0.11},
      "traditional_services": {"share": 0.302, "avg_wage": 3531, "vacancy_rate": 0.07},
      "modern_services": {"share": 0.143, "avg_wage": 8968, "vacancy_rate": 0.04}
    }
  },
  "living_cost": {"housing_price_avg": 13485.0, "rent_avg": 747.0, "daily_cost_index": 1.03},
  "public_services": {"medical_score": 0.73, "education_score": 0.68, "transport_convenience": 0.6, "avg_commute_mins": 35},
  "social_context": {"population_total": 5823559}
}
```

**2. city_edges.jsonl** - 城市边关系
```json
{"source_id": "1301", "target_id": "1302", "w_geo": 97.2, "w_dialect": 1.35, "w_admin": 0.0, "w_transport": 0.0}
```
注：只使用 `source_id`, `target_id`, `w_geo`（地理距离）, `w_dialect`（方言距离）

**3. city_nodes.jsonl** - 城市节点
```json
{"city_id": "1301", "name": "石家庄"}
```

### 3. 运行训练

```bash
# Windows
.venv\Scripts\python.exe main.py

# Linux/Mac
.venv/bin/python main.py
```

## 特征说明

### 输入特征（33个）

#### 1. 类别特征（6个，从 Type_ID 解析）
- `Feat_Gender`: 性别 (F/M)
- `Feat_Age`: 年龄段
- `Feat_Edu`: 教育水平
- `Feat_Ind`: 行业类型
- `Feat_Inc`: 收入水平
- `Feat_Fam`: 家庭状况

#### 2. 历史特征（4个，核心特征）
- `Hist_LastYear_Count`: 去年该路径的迁徙人数
- `Hist_LastYear_Score`: 去年该路径的标签值
- `Hist_LastYear_Rank`: 去年该路径的排名
- `Hist_LastYear_LogCount`: 去年迁徙人数的对数

#### 3. 交叉特征（22个，城市属性差值）

**边特征（2个）：**
- `Cross_Geo_Distance`: 地理距离
- `Cross_Dialect_Distance`: 方言距离

**城市属性差值（20个）：**
- `Cross_Diff_tier`: 城市等级差
- `Cross_Diff_gdp_per_capita`: 人均GDP差
- `Cross_Diff_cpi_index`: CPI指数差
- `Cross_Diff_unemployment_rate`: 失业率差
- `Cross_Diff_agriculture_share`: 农业占比差
- `Cross_Diff_manufacturing_share`: 制造业占比差
- `Cross_Diff_traditional_services_share`: 传统服务业占比差
- `Cross_Diff_modern_services_share`: 现代服务业占比差
- `Cross_Diff_agriculture_wage`: 农业工资差
- `Cross_Diff_manufacturing_wage`: 制造业工资差
- `Cross_Diff_traditional_services_wage`: 传统服务业工资差
- `Cross_Diff_modern_services_wage`: 现代服务业工资差
- `Cross_Diff_housing_price_avg`: 房价差
- `Cross_Diff_rent_avg`: 租金差
- `Cross_Diff_daily_cost_index`: 生活成本指数差
- `Cross_Diff_medical_score`: 医疗评分差
- `Cross_Diff_education_score`: 教育评分差
- `Cross_Diff_transport_convenience`: 交通便利度差
- `Cross_Diff_avg_commute_mins`: 平均通勤时间差
- `Cross_Diff_population_total`: 总人口差

#### 4. 其他特征（1个）
- `Total_Count`: 该出发城市的总迁徙人数

### 输出标签

- **Label**: 整数排名标签
  - Top1 → 20, Top2 → 19, ..., Top20 → 1
  - 负样本 → 0

详细特征说明请查看 [FEATURES.md](FEATURES.md)

## 数据集划分

针对 20 年数据（2001-2020）：

- **训练集**: 2001-2017 (17年)
- **验证集**: 2018 (1年)
- **测试集**: 2019-2020 (2年)

## 模型配置

- **算法**: LightGBM LambdaRank
- **目标函数**: lambdarank（直接优化排序）
- **评估指标**: NDCG@10, NDCG@20
- **负采样**: 每个查询采样 30 个负样本

主要超参数：
```python
{
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

## 输出结果

训练完成后，会在 `output/` 目录生成：

1. **feature_importance.png**: 特征重要性图
2. 控制台输出：
   - 数据加载统计（337个城市，113232条边）
   - 数据集划分统计
   - 训练/验证 NDCG 指标
   - 测试集样例预测结果

## 使用真实数据

修改 [main.py]

```python
# 注释掉模拟数据部分
# 取消注释以下行
df_raw = load_raw_data_from_duckdb(Config.DB_PATH)
```

确保 DuckDB 中的表名与 [src/data_loader.py](src/data_loader.py:11) 中的查询一致：

```python
df = con.query("SELECT * FROM migration_table").df()  # 替换为实际表名
```

## 核心改进

### v2.0 更新（当前版本）

   - 改用 JSONL 文件动态加载城市数据

2. **完整的交叉特征**
   - 添加地理距离和方言距离（从 city_edges.jsonl）
   - 添加20个城市属性差值特征（从 cities_data.jsonl）
   - 特征数量从 12 个增加到 33 个

3. **模块化城市数据加载**
   - 新增 `CityDataLoader` 类
   - 支持城市信息、边关系、节点信息的独立加载
   - 便于扩展和维护

## 扩展功能

### 1. 添加更多交叉特征

在 [src/feature_eng.py](src/feature_eng.py:100) 的 `diff_attributes` 列表中添加新属性：

```python
diff_attributes = [
    'tier', 'gdp_per_capita', 'cpi_index',
    # 添加新属性
    'your_new_attribute',
]
```

### 2. 调整负采样率

修改 [src/config.py](src/config.py:31)：

```python
NEG_SAMPLE_RATE = 50  # 增加负样本数量
```

### 3. 全量预测（测试集）

修改 [src/data_loader.py](src/data_loader.py:76-79)，在测试集使用全量城市：

```python
# 测试集使用全量城市，不采样
if is_test_set:
    neg_samples = neg_candidates
else:
    # 训练集和验证集采样
    if len(neg_candidates) > Config.NEG_SAMPLE_RATE:
        neg_samples = random.sample(neg_candidates, Config.NEG_SAMPLE_RATE)
```

## 性能优化建议

1. **大规模数据**: 使用 Spark 进行宽表到长表的转换
2. **特征缓存**: 预计算历史特征和交叉特征并保存
3. **并行计算**: 使用 `multiprocessing` 加速特征计算
4. **模型调优**: 使用 Optuna 进行超参数搜索

## 常见问题

### Q: 训练时 NDCG 达到 1.0 是否正常？

A: 在模拟数据或数据量较小时可能出现。使用真实数据后，NDCG 通常在 0.7-0.9 之间。

### Q: 如何处理新出现的城市？

A: 模型会依赖交叉特征（距离、GDP、产业结构等）来预测新路径。确保 JSONL 文件包含所有城市信息。

### Q: 历史特征缺失怎么办？

A: 代码已处理缺失值填充（填充为 0 或 999）。第一年数据无历史特征是正常的。

### Q: JSONL 文件格式错误怎么办？

A: 确保每行是一个有效的 JSON 对象，使用 UTF-8 编码。可以用 `jq` 工具验证：
```bash
cat data/cities_data.jsonl | jq . > /dev/null
```
