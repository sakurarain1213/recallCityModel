LambdaRank训练的输入输出
输入：

优化后的最终 Schema
保存到 Parquet 的列（~36 个）
类别	列名	类型	说明
基础标识	Year	int16	年份
Type_ID	category	原始 Type_ID（用于历史特征匹配）
From_City	int16	出发城市（降级）
To_City	int16	目标城市（降级）
Total_Count	int32	总流量（保留但不训练）
qid	int32	查询组 ID
标签	Rank	int16	原始排名
Label	int8	断层 Label（降级）
Flow_Count	int32	实际流量（保留但不训练）
Type_ID 特征	gender	int8	性别
age_group	int8	年龄段
education	int8	学历
industry	int8	行业
income	int8	收入
family	int8	家庭状态
城市特征	geo_distance	float32	地理距离
dialect_distance	float32	方言距离
*_ratio (21个)	float32	城市属性差异 ratio
历史特征	Hist_Flow_Log	float32	log(去年流量+1)
Hist_Rank	int16	去年排名
Hist_Label	int16	去年断层 Label
Hist_Share	float32	去年流量占比
Hist_Label_1y	int16	近 1 年 Label
Hist_Label_3y_avg	float32	近 3 年平均 Label
Hist_Label_5y_avg	float32	近 5 年平均 Label
🎯 训练时使用的特征（~36 个）
排除的列（7 个）

exclude_cols = [
    'Label',        # 标签
    'To_City',      # 目标城市
    'Flow_Count',   # 泄露特征
    'Rank',         # 泄露特征
    'Total_Count',  # 可能泄露
    'qid',          # 查询组 ID
    'pred_score'    # 预测结果
]



实际训练特征（~36 个）
Year (1 个)
Type_ID (1 个，category)
Type_ID 拆解特征 (6 个)
From_City (1 个，int16)
城市特征 (23 个：2 个距离 + 21 个 ratio)
历史特征 (7 个)








输出：

训练好的LightGBM模型，能对任意(From_City, To_City)对打分
预测时：

给定一个查询，对337个候选城市全部打分
按分数排序，取Top20
能否得到Top20效果？
可以！ 训练数据已经包含了真实的Top20（Label=20到1）和负样本（Label=0）。模型学习后能够：

对337个城市打分
排序后输出Top20
与真实Top20对比评估
评估指标
使用NDCG@10和NDCG@20评估排序质量：

NDCG越高，说明预测的Top20与真实Top20越吻合
在验证集(2018)和测试集(2019-2020)上评估
特征权重图代码
有！ LightGBM自带特征重要性可视化。