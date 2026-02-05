import os

class Config:
    # 路径配置
    DB_PATH = 'data/local_migration_data.db'
    DATA_DIR = 'data'  # JSONL 文件目录
    OUTPUT_DIR = 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSONL 文件路径
    CITY_INFO_PATH = 'data/cities_data.jsonl'    # 城市详细信息
    CITY_EDGES_PATH = 'data/city_edges.jsonl'    # 城市边关系（距离）
    CITY_NODES_PATH = 'data/city_nodes.jsonl'    # 城市节点（ID和名称）

    # 训练参数
    SEED = 42

    # 【召回模式】负样本采样率：固定20个困难负样本
    # 结果：每个Query有 20个正样本(Top1-20) + 20个困难负样本 = 40条数据
    NEG_SAMPLE_RATE = 20

    # 数据集划分（21年数据：2000-2020）
    # 2000年仅用于提供历史特征，不参与训练
    # 训练集：2001-2017 (17年)
    # 验证集：2018 (1年)
    # 测试集：2019-2020 (2年)
    DATA_START_YEAR = 2000  # 从2000年开始处理（用于历史特征）
    TRAIN_START_YEAR = 2001
    TRAIN_END_YEAR = 2017
    VAL_YEARS = [2018]
    TEST_YEARS = [2019, 2020]

    # LGBM 参数（二分类模式：Top 10 = 1，其他 = 0）
    LGBM_PARAMS = {
        'objective': 'binary',          # 改为二分类
        'metric': 'binary_logloss',     # 监控 LogLoss
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,

        # 【优化】增加叶子节点数，让树更深，捕捉复杂关系
        'num_leaves': 63,

        'max_depth': -1,

        # 【修复】从 10 改为 500，对于千万级数据，叶子节点至少要有几百个样本才具有统计意义
        'min_child_samples': 500,

        'subsample': 0.8,

        # 【优化】特征采样率 0.8，每次只看 80% 特征，强迫模型使用更多特征
        'colsample_bytree': 0.8,

        # 【优化】增加 L1/L2 正则化，防止过拟合单一特征
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,

        # 【新增】增加 Hessian 阈值，防止在弱信号上强行分裂
        'min_sum_hessian_in_leaf': 1.0,

        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,  # 减少警告输出
    }

    # GPU 训练参数（需要安装 lightgbm-gpu 版本）
    LGBM_PARAMS_GPU = {
        'objective': 'binary',          # 改为二分类
        'metric': 'binary_logloss',     # 监控 LogLoss
        'boosting_type': 'gbdt',

        # 1. 降低学习率，增加树，让学习过程更平滑
        'n_estimators': 2000,   # 原 1000 -> 2000
        'learning_rate': 0.03,  # 原 0.05 -> 0.03

        # GPU 特定参数
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,

        # 树结构参数
        'num_leaves': 63,
        'max_depth': -1,        # 不限制深度

        # 【关键】增加叶子节点所需样本数，防止过拟合
        'min_child_samples': 800,  # 原 500 -> 800

        # 【关键】降低采样率，强迫模型使用更多样化的特征
        'subsample': 0.7,           # 原 0.8 -> 0.7
        'colsample_bytree': 0.6,    # 原 0.8 -> 0.6

        # 【关键】增加正则化，防止过拟合单一特征
        'lambda_l1': 1.0,       # 原 0.1 -> 1.0
        'lambda_l2': 1.0,       # 原 0.1 -> 1.0

        'min_sum_hessian_in_leaf': 1.0,

        'random_state': 42,
        'verbosity': -1,
    }
