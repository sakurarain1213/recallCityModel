# 模型部署指南

本指南介绍如何将训练好的 LightGBM 排序模型部署到生产环境进行纯推理（Inference）。

## 1. 服务器文件清单

生产环境服务器只需要上传以下文件和目录。保持相对路径结构如下：

```text
/app/
├── predict.py                # 推理脚本 (基于 evaluate.py 修改或直接使用其中的 API)
├── src/                      # 源代码依赖
│   ├── __init__.py
│   ├── config.py             # 确保 OUTPUT_DIR 路径配置正确
│   ├── feature_eng.py        # 特征工程逻辑
│   └── ...                   # 其他 src 下的文件
├── output/
│   ├── models/
│   │   ├── binary_model.txt  # 训练好的模型文件
│   │   └── feature_cols.pkl  # 模型对应的特征列名列表
│   ├── static_city_pairs.parquet               # 静态特征表 (必须)
│   └── processed_data/                         # 历史数据文件夹
│       └── processed_{YEAR}.parquet            # 需要预测年份的前一年数据 (例: 预测2020年需2019数据)
└── data/
    └── city_nodes.jsonl      # (可选) 用于将 ID 转换为城市名称
```

> **注意**: 如果预测年份是 `2024`，则必须确保 `output/processed_data/processed_2023.parquet` 存在，用于提取历史特征（上一年流量、排名等）。

## 2. 环境依赖

安装必要的 Python 包：

```bash
pip install pandas numpy lightgbm scikit-learn pyarrow fastparquet
```

## 3. 推理接口定义

### 输入 (Batch Input)

推荐使用 Pandas DataFrame 或 JSON 列表格式作为输入。核心字段如下：

| 字段名 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `Year` | Int | 预测年份 | `2024` |
| `Type_ID` | String | 人群类型ID | `"F_20_EduHi_Service_IncH_Split"` |
| `From_City` | Int/Str | 出发城市ID | `3301` |
| `Month` | Int | (可选) 月份 | `12` (模型通常对月份不敏感也可设默认值) |

### 输出 (Batch Output)

返回每个查询（Query）推荐的 Top N 目标城市列表。

```json
[
  {
    "Year": 2024,
    "Type_ID": "F_20...",
    "From_City": 3301,
    "Recommendations": [
      {"To_City": 1101, "Score": 3.45, "Rank": 1},
      {"To_City": 3101, "Score": 2.12, "Rank": 2},
      ...
    ]
  },
  ...
]
```

## 4. 推理脚本示例 (`predict.py`)

这是一个最小化的生产环境推理脚本示例：

```python
import pandas as pd
from pathlib import Path
from evaluate import FastContext, load_model_fast, fast_batch_predict
# 确保 evaluate.py 在同级目录下，或者将相关类提取到单独的 inference.py 中

# 1. 初始化配置
MODEL_PATH = "output/models/binary_model.txt"
PREDICT_YEAR = 2024

# 2. 加载资源 (单例模式，常驻内存)
print("Loading model and context...")
ctx = FastContext.get_instance()
ctx.load_static_data()           # 加载 output/static_city_pairs.parquet
ctx.load_history_year(PREDICT_YEAR) # 加载 output/processed_data/processed_{YEAR-1}.parquet
model, feats = load_model_fast(MODEL_PATH)

def batch_inference(queries):
    """
    Args:
        queries: List of dicts, e.g. [{'Year': 2024, 'Type_ID': '...', 'From_City': 3301}]
    Returns:
        List of results with Top 10 recommendations
    """
    # 转为 DataFrame
    df_q = pd.DataFrame(queries)
    
    # 确保类型正确
    df_q['Year'] = df_q['Year'].astype(int)
    df_q['From_City'] = df_q['From_City'].astype(int)
    if 'Month' not in df_q.columns:
        df_q['Month'] = 12
        
    # 批量预测
    candidates = fast_batch_predict(df_q, PREDICT_YEAR, ctx)
    
    # 获取 Top 10
    results = []
    # 按查询分组取 Top K
    top_k = candidates.sort_values('pred_score', ascending=False).groupby(['Year', 'Type_ID', 'From_City']).head(10)
    
    for (year, type_id, from_city), group in top_k.groupby(['Year', 'Type_ID', 'From_City']):
        recs = []
        for _, row in group.iterrows():
            recs.append({
                "To_City": int(row['To_City']),
                "Score": float(row['pred_score'])
            })
        
        results.append({
            "Year": int(year),
            "Type_ID": type_id,
            "From_City": int(from_city),
            "Recommendations": recs
        })
        
    return results

if __name__ == "__main__":
    # 测试输入
    sample_queries = [
        {"Year": 2024, "Type_ID": "F_20_EduHi_Service_IncH_Split", "From_City": 3301},
        {"Year": 2024, "Type_ID": "M_35_EduMid_Mfg_IncM_Split", "From_City": 1101}
    ]
    
    print("Running inference...")
    result = batch_inference(sample_queries)
    print(result)
```

## 5. 性能优化提示

1.  **内存常驻**: `FastContext` 加载的静态特征和历史特征应常驻内存，不要每次请求都重新加载。
2.  **批处理**: 不要一次预测一条，尽量凑齐一批（如 100-1000 条 Query）调用一次 `fast_batch_predict`，以利用向量化计算优势。


```
简单的评估结论
(reranker-train) C:\Users\w1625\Desktop\reranker-train>uv run evaluate.py  

================================================================================
开始采样评估...
================================================================================
初始化环境...
[FastContext] Loading static city pairs from output\static_city_pairs.parquet...
[FastContext] Loaded 113,232 pairs.
[FastContext] Pre-loading history data from output\processed_data\processed_2019.parquet...
[FastContext] History loaded: 7,133,676 records
Model loaded: 37 features
加载测试数据 2020...
评估 100000 个查询...
批量生成候选集并预测...
  [1/6] 构造候选集 (100000 个查询 × 337 个城市)...
     生成了 33,600,000 个候选样本
  [2/6] 合并静态特征（城市属性、距离）...
  [3/6] 合并历史特征（去年的流量、排名）...
  [4/6] 解析 Type_ID（性别、年龄、学历等）...
  [5/6] 准备特征矩阵（37 个特征）...
  [6/6] LightGBM 批量预测...
     预测完成！
计算评估指标...
     [Metrics] 正在进行向量化评估...

========================================
快速评估结果 (2020)
========================================
查询数量 (Queries): 100000
平均正样本数 (Avg GT Size): 10.00
------------------------------
Standard Recall (Hits / Total_True):
  Recall@5 : 0.3271
  Recall@10: 0.5237
  Recall@20: 0.7900
------------------------------
Normalized Hits (Hits / min(Total, K)):
  (解决了正样本 > K 时 Recall 虚低的问题)
  Norm@5   : 0.6543
  Norm@10  : 0.5237
  Norm@20  : 0.7900

================================================================================
开始快速推理示例...
================================================================================
  [1/6] 构造候选集 (1 个查询 × 337 个城市)...
     生成了 336 个候选样本
  [2/6] 合并静态特征（城市属性、距离）...
  [3/6] 合并历史特征（去年的流量、排名）...
  [4/6] 解析 Type_ID（性别、年龄、学历等）...
  [5/6] 准备特征矩阵（37 个特征）...
  [6/6] LightGBM 批量预测...
     预测完成！

推理时间: 1390.76 ms

Top 10 城市:
  1. 3303 (温州) - 分数: 1.0000
  2. 3307 (金华) - 分数: 1.0000
  3. 3100 (上海) - 分数: 0.9999
  4. 3306 (绍兴) - 分数: 0.9999
  5. 3202 (无锡) - 分数: 0.9999
  6. 3305 (湖州) - 分数: 0.9997
  7. 3302 (宁波) - 分数: 0.9995
  8. 3204 (常州) - 分数: 0.9994
  9. 3311 (丽水) - 分数: 0.9989
  10. 3310 (台州) - 分数: 0.9988
```
