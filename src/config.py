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

    # 【召回模式】负样本采样率：混合采样策略
    # 策略：30个困难负样本（大城市）+ 30个随机负样本（任意城市）
    # 结果：每个Query有 20个正样本(Top1-20) + 60个混合负样本 = 80条数据
    # 目的：解决小城市高分问题，让模型学会区分"距离近但无关"的城市
    NEG_SAMPLE_RATE = 60

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
        'learning_rate': 0.1,

        # 【优化】增加叶子节点数，让树更深，捕捉复杂关系
        'num_leaves': 63,

        'max_depth': -1,

        # 【修复】从 10 改为 100，对于千万级数据，叶子节点至少要有几百个样本才具有统计意义
        'min_child_samples': 100,

        'subsample': 0.8,

        # 【优化】特征采样率 0.8，每次只看 80% 特征，强迫模型使用更多特征
        'colsample_bytree': 0.8,

        # 【优化】增加 L1/L2 正则化，防止过拟合单一特征
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,

        # 【新增】增加 Hessian 阈值，防止在弱信号上强行分裂
        'min_sum_hessian_in_leaf': 0.001,

        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,  # 减少警告输出
    }

    # 第二套！  GPU 训练参数（需要安装 lightgbm-gpu 版本）
    LGBM_PARAMS_GPU = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',

        # 基础参数
        'n_estimators': 3000,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbosity': 1,

        # GPU 特定配置
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,

        # 【核心修复 1】开启双精度 (Double Precision)
        # 单精度(False)会导致直方图计算误差，引发 left_count > 0 崩溃
        # 虽然会慢一点点，但这是 RTX 显卡跑 OpenCL 的唯一稳定解
        'gpu_use_dp': True,

        # 【核心修复 2】降低直方图箱数 (Max Bin)
        # 默认 255 在 GPU 上容易出现空箱子，降到 63 可以显著提升稳定性
        'max_bin': 63,

        # 树结构参数 (保守设置)
        'num_leaves': 31,       # 保持适中
        'max_depth': -1,        # 让 num_leaves 控制
        'min_child_samples': 50,# 恢复到 50，20 太敏感，1000 太难满足
        'min_child_weight': 0.001,

        # 采样与特征
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8, # 每次只用 80% 特征

        # 正则化 (适度)
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'min_gain_to_split': 0.0,
        'min_sum_hessian_in_leaf': 0.001,

        # 强制列式直方图 (通常更稳)
        'force_col_wise': True,
    }
