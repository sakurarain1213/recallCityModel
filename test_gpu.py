"""测试 LightGBM GPU 支持"""
import lightgbm as lgb
import numpy as np

print(f"LightGBM version: {lgb.__version__}")

# 创建简单的测试数据
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)

# 尝试使用 GPU
try:
    params_gpu = {
        'objective': 'binary',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbosity': 1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    print("\n尝试使用 GPU 训练...")
    model = lgb.train(params_gpu, train_data, num_boost_round=10)
    print("[SUCCESS] GPU 训练成功！")

except Exception as e:
    print(f"[FAILED] GPU 训练失败: {e}")
    print("\n尝试使用 CPU 训练...")
    params_cpu = {
        'objective': 'binary',
        'device': 'cpu',
        'verbosity': 1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params_cpu, train_data, num_boost_round=10)
    print("[SUCCESS] CPU 训练成功")
