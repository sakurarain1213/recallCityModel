import os

class Config:
    # 路径配置
    DB_PATH = 'C:/Users/w1625/Desktop/recall/data/local_migration_data.db'
    DATA_DIR = 'C:/Users/w1625/Desktop/recall/data'  # JSONL 文件目录
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSONL 文件路径
    # 城市详细信息现在是年度文件: cities_2000.jsonl 到 cities_2020.jsonl
    CITY_INFO_DIR = 'C:/Users/w1625/Desktop/recall/data/cities_2000-2020'  # 年度城市信息文件目录
    CITY_EDGES_PATH = 'C:/Users/w1625/Desktop/recall/data/city_edges.jsonl'    # 城市边关系（距离）
    CITY_NODES_PATH = 'C:/Users/w1625/Desktop/recall/data/city_nodes.jsonl'    # 城市节点（ID和名称）

    # 训练参数
    SEED = 42

    # 【User Req 5】负样本目标数量
    NEG_SAMPLE_RATE = 40  # 1个正样本 : 40个负样本 (Top10正样本共10个，所以比例约 1:4)

    # 【User Req 5】全局热门城市 (Hard Global Negatives)
    # 这些城市如果在Ground Truth Top10之外，就是极强的干扰项
    # 包含: 北上广深 + 成都/杭州/重庆/武汉/西安/苏州/南京/天津/郑州/长沙/东莞/佛山
    POPULAR_CITIES = [
        1100, 3100, 4401, 4403,  # 一线
        5101, 3301, 5000, 4201, 6101, 3205, 3201, 1200, 4101, 4301, 4419, 4406 # 新一线
    ]

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

    # 【User Req 3】模型参数调整 (增加深度，处理非线性)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'n_estimators': 3000,     # 增加树数量
        'learning_rate': 0.03,    # 降低学习率以适应Batch训练
        'num_leaves': 255,        # 【User Req 3】大幅增加叶子节点 (原本127)
        'max_depth': 12,          # 【User Req 3】限制深度防止过拟合 (配合num_leaves)
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'n_jobs': -1,
        'verbosity': -1,
        'first_metric_only': True
    }

    # GPU 参数
    LGBM_PARAMS_GPU = LGBM_PARAMS.copy()
    LGBM_PARAMS_GPU.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': True,
    })
