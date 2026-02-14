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

    # 训练参数
    SEED = 42

    # 数据集划分
    DATA_START_YEAR = 2000

    # 【分批训练策略】三年一个 Batch，适应 14GB 内存限制
    # 每批使用 3 年训练数据，内存需求约 8-10GB ✅
    #
    # Batch 划分:
    #   Batch 1: 训练 2001-2003，验证 2008
    #   Batch 2: 训练 2004-2006，验证 2009
    #   Batch 3: 训练 2007，验证 2010
    #   最终测试: 2010
    #
    # 优势:
    #   1. 时间顺序训练，更符合实际预测场景
    #   2. 每批内存占用可控（~8-10GB）
    #   3. 充分利用历史数据，避免浪费
    TRAIN_BATCHES = [
        {
            'name': 'batch1_2001-2003',
            'train_years': [2001, 2002, 2003],
            'val_year': 2008
        },
        {
            'name': 'batch2_2004-2006',
            'train_years': [2004, 2005, 2006],
            'val_year': 2009
        },
        {
            'name': 'batch3_2007',
            'train_years': [2007],
            'val_year': 2010
        }
    ]

    # 最终测试年份
    TEST_YEARS = [2010]

    # 向后兼容的旧配置（用于单批训练）
    TRAIN_START_YEAR = 2001
    TRAIN_END_YEAR = 2003
    VAL_YEARS = [2008]

    # --- LightGBM 参数 (CPU 极速版) ---
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],

        # 【核心修改 1】切换为 GOSS (亿级数据提速神器)
        'boosting_type': 'goss',
        'top_rate': 0.2,      # 保留梯度最大的 20% 样本
        'other_rate': 0.1,    # 从剩余样本中随机采样 10%
        # 结果：相当于只用 30% 的数据量训练，但保留了核心信息

        # 【速度优化核心 1】大幅减少树的复杂度
        # Recall 任务不需要拟合太细的残差，63 叶子 + 8 深足够区分 Top10 和 Top100
        'num_leaves': 63,         # 原 255 -> 63
        'max_depth': 8,           # 原 12 -> 8

        # 【速度优化核心 2】CPU 训练的神器：减少分桶
        # 默认 255，降为 63 可提升 3-5 倍速度，精度损失极小
        'max_bin': 63,            # 必须确认是 63

        # 【速度优化核心 3】提高学习率，减少树数量
        'learning_rate': 0.1,     # 原 0.05 -> 0.1 (加速收敛)
        'n_estimators': 1000,      # 原 2000 -> 1000 (配合早停)

        # 【核心修改 2】GOSS 模式下必须移除 subsample (行采样)
        # 'subsample': 0.8,         # GOSS 不兼容，注释掉
        # 'subsample_freq': 5,      # GOSS 不兼容，注释掉

        'colsample_bytree': 0.8,  # 列采样可以保留
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
