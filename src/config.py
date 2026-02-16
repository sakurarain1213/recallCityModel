import os

class Config:
    # --- 路径配置 ---
    DB_PATH = '/data1/wxj/Recall_city_project/data/local_migration_data.db'
    DATA_DIR = '/data1/wxj/Recall_city_project/data'
    CITY_INFO_DIR = '/data1/wxj/Recall_city_project/data/cities_2000-2020'
    OUTPUT_DIR = 'output'

    # 预处理数据的存储目录 (新)
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed_ready')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # JSONL 文件路径
    # 城市详细信息现在是年度文件: cities_2000.jsonl 到 cities_2020.jsonl
    CITY_INFO_DIR = '/data1/wxj/Recall_city_project/data/cities_2000-2020'  # 年度城市信息文件目录
    CITY_EDGES_PATH = '/data1/wxj/Recall_city_project/data/city_edges.jsonl'    # 城市边关系（距离）
    CITY_NODES_PATH = '/data1/wxj/Recall_city_project/data/city_nodes.jsonl'    # 城市节点（ID和名称）

    # --- 采样策略 (严格 1:4) ---
    # 一个 Query 总样本数 = 50
    # 构成: 10个正样本 (Rank 1-10) + 10个困难负样本 (Rank 11-20) + 30个补充负样本
    TOTAL_SAMPLES_PER_QUERY = 50

    # 困难负样本池 (热门城市 ID)
    POPULAR_CITIES = [
        1100, 3100, 4401, 4403,  # 一线
        5101, 3301, 5000, 4201, 6101, 3205, 3201, 1200, 4101, 4301, 4419, 4406 # 新一线
    ]

    # --- 训练策略配置 ---
    # 分批训练：每批使用几年数据
    TRAIN_BATCH_SIZE_YEARS = 3

    # 早停轮数（Early Stopping）
    EARLY_STOPPING_ROUNDS = 100

    # Checkpoint 保存频率
    CHECKPOINT_FREQ = 50

    # Mini-Validation Set 大小（用于 Early Stopping 和实时监控）
    # 按完整 Query 采样，确保 Recall 指标准确
    MINI_VAL_SIZE = 500000

    # 日志打印频率（每 N 轮打印一次）
    LOG_EVALUATION_FREQ = 50

    # --- LightGBM 参数 (CPU 极速版) ---
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],

        # 【核心修改 1】切换为 GOSS (亿级数据提速神器)
        'boosting_type': 'goss',
        'top_rate': 0.2,      # 保留梯度最大的 20% 样本
        'other_rate': 0.1,    # 从剩余样本中随机采样 10%
        # 结果：相当于只用 30% 的数据量训练，但保留了核心信息

        # 【精度优化】增加模型容量以提升 Recall
        'num_leaves': 127,        # 原 63 -> 127 (增加模型容量)
        'max_depth': 10,          # 原 8 -> 10 (配合叶子数增加)

        # 【速度优化核心 2】CPU 训练的神器：减少分桶
        # 默认 255，降为 63 可提升 3-5 倍速度，精度损失极小  255可以更高的精度进行学习
        'max_bin': 63,

        # 【精度优化】降低学习率，增加树数量,提升精度
        'learning_rate': 0.1,    # 原 0.1 -> 0.05 (更细致的学习)
        'n_estimators': 3000,     # 原 1000 -> 3000 (配合低学习率)

        # 【核心修改 2】GOSS 模式下必须移除 subsample (行采样)
        # 'subsample': 0.8,         # GOSS 不兼容，注释掉
        # 'subsample_freq': 5,      # GOSS 不兼容，注释掉

        'colsample_bytree': 0.6,  # 原 0.8 -> 0.6 (增加随机性,防止过拟合)
        'min_child_samples': 100, # 防止过拟合
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,

        # 【关键修改 1】限制线程数，不要用 -1
        # 建议设置为 16-32 之间，不要超过物理核数
        'n_jobs': 24,            # 原 -1 -> 24 (避免过度竞争)
        'verbosity': -1
    }

    # GPU 参数
    LGBM_PARAMS_GPU = LGBM_PARAMS.copy()
    LGBM_PARAMS_GPU.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': True,
    })
