```
服务器版本
本地： ssh -p 2023 -R 10080:127.0.0.1:7890 -N wxj@10.82.1.210
服务器：
cd /data1/wxj/Recall_city_project
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export http_proxy="http://127.0.0.1:10080"
export https_proxy="http://127.0.0.1:10080"
export HTTP_PROXY="http://127.0.0.1:10080"
export HTTPS_PROXY="http://127.0.0.1:10080"
curl -v https://api.github.com  
```

# 人口流动预测模型

# TODO
核心改进策略
可视化增强 (User Req 1): 在训练结束时输出 Feature Importance 图片。

训练策略重构 (User Req 2): 放弃逐年增量训练，改为 Batch 训练。
原因: 逐年训练 (init_model) 在 LightGBM 中会导致"遗忘灾难"。2001年的规律到2010年虽然有漂移，但基础规律（如距离衰减、省内流动）是不变的。我的需求是只指定结束年份 比如--2012  那么2001-2003 每三年一个batch进行训练  验证集永远是结束年份的前2年即2010全量和2011全量  然后2012这一年全量作为evaluate的测试集。

负样本策略革命 (User Req 5 - 最关键):
引入"省内干扰": 绝大多数流动发生在省内。模型必须学会区分"省会城市"和"省内其他城市"。
引入"排名靠后干扰" (Rank 11-20): 你希望 Recall@10 高，那么 Rank 11-20 就是最强的干扰项（Hard Negative）。把它们标记为负样本（Label=0），强迫模型把它们排在 Top 10 之外。  然后也需要本省内非出发城市非热门城市的所有省内其它城市 作为负样本
最后 保证随机从剩下的全国城市随机采样作为负样本
最后保证训练的时候 一个query top10作为正样本  剩下大约40个城市作为负样本


特征工程优化 (User Req 4): 增加 is_same_province (是否同省) 特征。这是一个强特征。

模型参数调整 (User Req 3): 增加树的深度，因为人口流动是非线性的复杂关系。





基于 LightGBM 的人口流动预测模型，用于预测不同人群类型在城市间的流动方向和规模。采用召回模式(Recall Mode)，结合动态城市特征和历史流动数据训练模型。


```
其中训练集加载.db每年的数据schema都是Year Month Type_ID Birth_Region From_City Total_Count Stay_Prob Outflow_Count To_Top1 To_Top1_Count To_Top2 To_Top2_Count To_Top3 To_Top3_Count To_Top4 To_Top4_Count To_Top5 To_Top5_Count To_Top6 To_Top6_Count To_Top7 To_Top7_Count To_Top8 To_Top8_Count To_Top9 To_Top9_Count To_Top10 To_Top10_Count To_Top11 To_Top11_Count To_Top12 To_Top12_Count To_Top13 To_Top13_Count To_Top14 To_Top14_Count To_Top15 To_Top15_Count To_Top16 To_Top16_Count To_Top17 To_Top17_Count To_Top18 To_Top18_Count To_Top19 To_Top19_Count To_Top20 To_Top20_Count 2002 12 F_30_EduHi_Service_IncML_Unit_5119 5119 巴中(5119) 6 0.990736 6 成都(5101) 2 重庆(5000) 0 达州(5117) 0 广元(5108) 0 绵阳(5107) 0 上海(3100) 0 南充(5113) 0 北京(1100) 0 遂宁(5109) 0 乌鲁木齐(6501) 0 攀枝花(5104) 0 德阳(5106) 0 广安(5116) 0 昆明(5301) 0 宜宾(5115) 0 天津(1200) 0 西安(6101) 0 泸州(5105) 0 汉中(6107) 0 贵阳(5201) 0 2020 12 F_40_EduLo_Wht_IncMH_Split_4453 4453 云浮(4453) 249 0.959849 239 广州(4401) 59 深圳(4403) 54 佛山(4406) 39 肇庆(4412) 20 江门(4407) 7 惠州(4413) 7 湛江(4408) 5 清远(4418) 2 上海(3100) 2 海口(4601) 2 珠海(4404) 2 阳江(4417) 2 韶关(4402) 2 重庆(5000) 2 北京(1100) 2 茂名(4409) 0 南宁(4501) 0 汕头(4405) 0 梧州(4504) 0 河源(4416) 0 2012 12 F_40_EduLo_Wht_IncH_Unit_4502 4502 柳州(4502) 69 0.812853 56 南宁(4501) 17 桂林(4503) 15 重庆(5000) 2 上海(3100) 2 梧州(4504) 2 百色(4510) 2 玉林(4509) 2 贵港(4508) 2 海口(4601) 0 北京(1100) 0 北海(4505) 0 深圳(4403) 0 贵阳(5201) 0 贺州(4511) 0 钦州(4507) 0 昆明(5301) 0 广州(4401) 0 乌鲁木齐(6501) 0 防城港(4506) 0 天津(1200) 0





然后关于城市信息 每年都有337个城市属性 例如{"city_id": "1301", "city_name": "石家庄", "basic_info": {"tier": 3, "coordinates": [114.51, 38.04], "area_sqkm": 3738.0}, "economy": {"gdp_per_capita": 61000.4, "cpi_index": 102.5, "unemployment_rate": 0.046, "industry_sectors": {"agriculture": {"share": 0.175, "avg_wage": 3117, "vacancy_rate": 0.08}, "manufacturing": {"share": 0.38, "avg_wage": 6833, "vacancy_rate": 0.11}, "traditional_services": {"share": 0.302, "avg_wage": 3531, "vacancy_rate": 0.07}, "modern_services": {"share": 0.143, "avg_wage": 8968, "vacancy_rate": 0.04}}}, "demographics": {"age_structure": {"16_24": 0.093, "25_34": 0.145, "35_49": 0.264, "50_60": 0.207, "60_plus": 0.291}, "sex_ratio": 105.1}, "living_cost": {"housing_price_avg": 13485.0, "rent_avg": 747.0, "daily_cost_index": 1.03}, "public_services": {"medical_score": 0.73, "education_score": 0.68, "transport_convenience": 0.6, "avg_commute_mins": 35}, "social_context": {"population_total": 5823559, "migrant_stock_distribution": {"5000": 0.021, "3100": 0.021, "3500": 0.007}}, "ground_truth_cache": {"inflow_index_last_year": 3.29}}





然后city edge的jsonl例如{"source_id": "1301", "target_id": "1302", "w_geo": 97.2, "w_dialect": 1.35, "w_admin": 0.0, "w_transport": 0.0} 是固定的 也只需要w_geo和w_dialect距离。 然后城市节点就是{"city_id": "1301", "name": "石家庄"}。然后特征人的属性就是DIMENSIONS = {

    'D1': {'name': '性别', 'values': ['M', 'F']},

    'D2': {'name': '生命周期', 'values': ['16-24', '25-34', '35-49', '50-60', '60+']},

    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi']},

    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht']},

    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH']},

    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit']},

}

```






## ✨ 主要特点

- **年度动态城市特征**: 支持2000-2020年每年城市属性更新
- **历史流动特征**: 利用历史数据提升预测准确性
- **混合负样本策略**: 困难负样本(大城市) + 随机负样本，比例1:5
- **二分类召回模型**: 使用 binary_logloss 训练，优化 Recall@K
- **CPU/GPU支持**: 灵活选择训练后端

---

## 🚀 快速开始

### 1. 环境设置
```bash
# 安装依赖
uv pip install -e .
```

### 2. 生成训练数据
```bash
uv run main.py
```

### 3. 训练模型
```bash
# CPU训练
uv run train.py

# GPU训练
uv run train.py --gpu

# 快速测试(只训练到2010年)
uv run train.py --end_year 2010
```

### 4. 评估模型
```bash
uv run evaluate.py
```

详细使用说明请查看 [使用指南.md](使用指南.md)

---

## 📊 特征说明

### 特征Schema

保存到 Parquet 的列(~36个):

| 类别 | 列名 | 类型 | 说明 |
|------|------|------|------|
| **基础标识** | Year | int16 | 年份 |
| | Type_ID_orig | str | 原始Type_ID(用于历史特征匹配) |
| | From_City_orig | str | 原始出发城市 |
| | From_City | int16 | 出发城市ID |
| | To_City | int16 | 目标城市ID |
| | qid | int32 | 查询组ID |
| **标签** | Rank | int16 | 原始排名 |
| | Label | float32 | 二分类标签(Top10=1.0, Top11-20=0.1, 负样本=0.0) |
| | Flow_Count | int32 | 实际流量 |
| **Type_ID特征** | gender, age_group, education, industry, income, family | int8 | 6维人口类型特征 |
| **城市特征** | geo_distance, dialect_distance | float32 | 地理距离和方言距离 |
| | *_ratio (21个) | float32 | 城市属性差异ratio(To/From) |
| **历史特征** | Hist_Flow_Log | float32 | log(去年流量+1) |
| | Hist_Rank | int16 | 去年排名 |
| | Hist_Label | float32 | 去年Label |
| | Hist_Share | float32 | 去年流量占比 |
| | Hist_Label_1y | int16 | 近1年Label |
| | Hist_Label_3y_avg | float32 | 近3年平均Label |
| | Hist_Label_5y_avg | float32 | 近5年平均Label |

### 训练特征(~36个)

排除以下列后用于训练:
```python
exclude_cols = [
    'Label',        # 标签
    'To_City',      # 目标城市
    'Flow_Count',   # 泄露特征
    'Rank',         # 泄露特征
    'Total_Count',   # 可能泄露
    'qid',          # 查询组ID
    'Type_ID_orig', 'From_City_orig'  # 中间列
]
```

实际使用的特征:
- Year (1个)
- Type_ID拆解特征 (6个)
- From_City (1个, int16)
- 城市特征 (23个: 2个距离 + 21个ratio)
- 历史特征 (7个)

---

## 📈 模型架构

### 训练流程
1. 从DuckDB加载原始人口流动数据
2. 为每年加载对应的城市特征数据(如2015年用cities_2015.jsonl)
3. 生成混合负样本(50个困难负样本 + 50个随机负样本)
4. 进行特征工程(人口类型、城市特征、历史特征)
5. 使用LightGBM二分类训练(binary_logloss)

### 预测流程
给定一个查询(Year, Type_ID, From_City)，对337个候选城市全部打分:
1. 生成查询-城市对特征
2. 使用LightGBM模型预测分数
3. 按分数排序取Top20
4. 与真实Top20对比评估

### 评估指标
- **Recall@K**: Top K预测中实际正样本的比例(最重要指标)
- **NDCG@K**: 归一化折损累计增益(排序质量)
- **Precision@K**: Top K预测的精确率

---

## 🗂️ 数据文件说明

### 数据结构
```
recall/
├── data/
│   ├── local_migration_data.db              # 人口流动数据库(DuckDB格式)
│   ├── cities_2000-2020/                   # 年度城市特征目录
│   │   ├── cities_2000.jsonl              # 2000年城市特征
│   │   ├── cities_2001.jsonl              # 2001年城市特征
│   │   ├── ...
│   │   └── cities_2020.jsonl              # 2020年城市特征
│   ├── city_edges.jsonl                     # 城市间距离关系
│   └── city_nodes.jsonl                     # 城市基本信息
├── output/                                   # 输出目录
│   └──processed_ready                      # 有2000-2020预处理后的parquet
└── src/                                      # 源代码
```

### 数据库路径配置
已在 `src/config.py` 中配置为:
- **数据库**: `/data1/wxj/Recall_city_project/data/local_migration_data.db`
- **年度城市数据**: `/data1/wxj/Recall_city_project/data/cities_2000-2020/`

---

## 🚀 从零开始完整流程

### 步骤1: 环境设置 ✅

已完成:
```bash
# 1. 创建虚拟环境
uv venv

# 2. 安装依赖
uv pip install -e .
```

如果需要重新激活虚拟环境:
```bash
# Windows (CMD)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 步骤2: 生成训练数据

运行主程序生成处理后的训练数据:

```bash
uv run main.py
```

**这一步会做什么:**
1. 从 DuckDB 加载原始人口流动数据(2000-2020年)
2. 为每年加载对应的城市特征数据(例如处理2015年数据时使用cities_2015.jsonl)
3. 生成混合负样本(30个困难负样本 + 30个随机负样本)
4. 进行特征工程(人口类型特征、城市特征、历史特征)
5. 保存处理后的数据到 `output/processed_data/` 目录

**输出文件:**
```
output/processed_data/
├── processed_2000.parquet    # 2000年数据(仅用于历史特征)
├── processed_2001.parquet
├── ...
└── processed_2020.parquet
```

**预计时间:** 10-30分钟(取决于CPU性能)

### 步骤3: 训练模型

训练 LightGBM 模型:

```bash
# CPU训练(完整数据: 2001-2017训练, 2018验证, 2019-2020测试)
uv run train.py

# 快速测试(训练到2010年)
uv run train.py --end_year 2010

# GPU训练(需要安装 lightgbm-gpu)
uv run train.py --gpu

# GPU训练 + 快速测试
uv run train.py --end_year 2010 --gpu
```

**训练参数说明:**
- `--end_year`: 训练截止年份(默认2020)
  - 自动划分: 训练集(2001 到 end_year-3), 验证集(end_year-2), 测试集(end_year-1 到 end_year)
- `--gpu`: 启用GPU加速训练

**输出文件:**
```
output/
├── lgb_model_2017.txt                 # 训练好的模型
├── training_history.png               # 训练曲线图
├── feature_importance.png             # 特征重要性图
└── training_metrics.txt              # 训练指标记录
```

**预计时间:**
- CPU: 30-60分钟
- GPU: 10-20分钟

### 步骤4: 评估模型

评估模型性能并查看召回效果:

```bash
uv run evaluate.py
```

**这一步会做什么:**
1. 加载训练好的模型
2. 在测试集(2019-2020年)上评估
3. 计算召回率@K、NDCG@K等指标
4. 生成详细的评估报告

**输出:**
- 控制台打印评估指标
- 各年份的详细性能分析

**关键指标:**
- **Recall@K**: Top K预测中实际正样本的比例
- **NDCG@K**: 归一化折损累计增益
- **Precision@K**: Top K预测的精确率

### 步骤5: (可选) 预处理静态特征

如果要使用静态特征优化(可选):

```bash
uv run evaluate_pre.py
```

---

## 📊 查看Recall效果

### 评估指标说明

运行 `uv run evaluate.py` 后，你会看到类似以下输出:

```
========================================
开始评估年份: 2010
========================================
Step 1: 加载测试集 Ground Truth...      
[SQL] Loaded 357600 rows
[Wide-to-Long] Generated 3153120 positive samples
[Negative Sampling] Added 292616 negative samples (Target: 1 per query)
[Final] Total: 3445736 rows
⚠️ 进行采样评估: 1000/302873
Step 2: 生成候选集 (1000 Queries x 337 Cities)...
候选集大小: 336,000 行
Step 3: 特征工程...
Step 4: 模型打分...
Step 5: 计算评估指标...

========================================
📊 评估结果报告 (2010)
========================================
Query样本数 : 302873
平均正样本数 : 10.41
------------------------------
Recall@1   : 0.01%
Recall@5   : 0.06%
Recall@10  : 0.10%
Recall@20  : 0.16%
========================================
```

### 指标解读

1. **Recall@K**: 模型预测的Top K个城市中，有多少是真实的目标城市
   - Recall@10 = 0.85 表示: 在预测的前10个城市中，平均能找到85%的真实目标城市
   - 这是**最重要的指标**，衡量召回能力

2. **NDCG@K**: 考虑排名位置的质量指标
   - 越接近1越好，表示排序质量高
   - 真实目标城市排在越前面，NDCG越高

3. **Precision@K**: Top K预测的准确率
   - Precision@10 = 0.086 表示: Top 10预测中，平均有8.6%是正确的

### 可视化结果

训练完成后，查看生成的图表:

1. **训练曲线** (`output/training_history.png`)
   - 显示训练集和验证集的LogLoss变化
   - 用于判断是否过拟合

2. **特征重要性** (`output/feature_importance.png`)
   - 显示哪些特征对预测最重要
   - 帮助理解模型决策

---

## 🔧 常见问题

### Q1: 如何只处理某几年的数据?

修改 `src/config.py` 中的配置:
```python
DATA_START_YEAR = 2000  # 起始年份
TRAIN_START_YEAR = 2001  # 训练起始年份
TRAIN_END_YEAR = 2017    # 训练结束年份
VAL_YEARS = [2018]       # 验证年份
TEST_YEARS = [2019, 2020]  # 测试年份
```

或使用命令行参数:
```bash
uv run train.py --end_year 2015  # 只训练到2015年
```

### Q2: 内存不足怎么办?

如果内存不足，可以:
1. 减少负样本数量(`Config.NEG_SAMPLE_RATE`从60改为30)
2. 减少训练年份范围
3. 使用 `--end_year` 参数减少数据量

### Q3: 如何使用GPU训练?

1. 确保安装了GPU版本的LightGBM:
```bash
uv pip install lightgbm --gpu
```

2. 添加 `--gpu` 参数:
```bash
uv run train.py --gpu
```

### Q4: 数据已经处理过了，如何跳过?

`main.py` 会自动跳过已存在的年份:
```
Year 2015: Already processed, skipping
```

如需重新处理，删除对应文件:
```bash
rm output/processed_data/processed_2015.parquet
```

### Q5: 年度城市数据加载失败?

确保以下路径正确:
- 城市数据目录: `/data1/wxj/Recall_city_project/data/cities_2000-2020/`
- 年度文件格式: `cities_2000.jsonl`, `cities_2001.jsonl`, ..., `cities_2020.jsonl`

检查文件是否存在:
```bash
ls data/cities_2000-2020/
```

---

## 📝 代码变更说明

### 修改的文件

1. **src/config.py**
   - 更新数据库路径为绝对路径
   - 添加年度城市数据目录配置 `CITY_INFO_DIR`

2. **src/city_data.py**
   - `CityDataLoader.city_info` 从 DataFrame 改为字典 `{year: DataFrame}`
   - `load_city_info()` 支持按年份加载
   - 新增 `get_city_info_for_year(year)` 方法
   - `get_city_attributes()` 支持指定年份

3. **src/feature_pipeline.py**
   - `FeaturePipeline` 在transform时根据Year列自动获取对应年份的城市信息
   - 确保特征工程使用正确年份的城市特征

4. **main.py**
   - 加载所有年份(2000-2020)的城市数据
   - 使用2010年数据作为困难负样本候选池基准

### 不需要修改的文件

- `src/data_loader.py` - 数据加载逻辑不变
- `src/feature_eng.py` - 特征工程逻辑不变
- `src/historical_features.py` - 历史特征逻辑不变
- `train.py` - 训练逻辑不变
- `evaluate.py` - 评估逻辑不变

---

## 🎯 下一步

1. **运行完整流程:**
   ```bash
   uv run main.py      # 生成数据(首次运行)
   uv run train.py     # 训练模型
   uv run evaluate.py  # 评估效果
   ```

2. **查看结果:**
   - 检查 `output/` 目录下的模型和图表
   - 查看 Recall@10 指标是否达到预期(>0.8)

3. **优化模型:**
   - 调整 `Config.LGBM_PARAMS` 中的超参数
   - 尝试不同的特征组合
   - 增加训练数据量

---

## 📞 技术支持

如有问题，检查:
1. Python版本 >= 3.11
2. 所有依赖已安装: `uv pip list`
3. 数据文件路径正确
4. 虚拟环境已激活

祝训练顺利! 🎉
