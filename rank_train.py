
"""
LightGBM 极速训练脚本 (v10 - O(1) 二进制加载 + LambdaRank Top-10 专杀版)

核心改动：
1. 彻底删除 Pandas 数据加载与特征拼接逻辑，直接以 O(1) 复杂度从 NVMe 盘加载 .bin 缓存。
2. 目标函数: rank_xendcg → lambdarank。配合我们在 build_bin 中定制的指数级 Relevance 标签。
3. 截断层级: lambdarank_truncation_level = 10。让模型算力死磕前 10 名。
"""

import time
import lightgbm as lgb
import pandas as pd
from pathlib import Path

# ================= 极速存储路径 =================
# 🎯 指向我们刚刚用 build_bin.py 生成好的高速盘目录
BIN_DIR = Path("/data2/wxj/recall_bin") 
OUTPUT_DIR = Path("output")

# ================= 回调函数 =================
class CheckpointCallback:
    def __init__(self, output_dir, freq=10):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        self.before_iteration = False

    def __call__(self, env):
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            ckpt_path = self.output_dir / f"model_iter_{current_round}.txt"
            env.model.save_model(str(ckpt_path))
            print(f"  [Checkpoint] iter {current_round} -> {ckpt_path}")

class TimingCallback:
    def __init__(self, freq=10):
        self.freq = freq
        self.start_time = None
        self.before_iteration = False

    def __call__(self, env):
        if self.start_time is None:
            self.start_time = time.time()
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            elapsed = time.time() - self.start_time
            speed = current_round / elapsed * 60
            print(f"  [Timer] iter {current_round} | elapsed {elapsed/60:.1f}min | {speed:.1f} iter/min")


def main():
    print("=" * 60)
    print("🚀 LightGBM O(1) 极速训练模式启动 (v10 - LambdaRank Top-10 版)")
    print("=" * 60)

    t0 = time.time()
    
    # 🔥 魔法时刻：O(1) 极速加载，跳过所有特征直方图的预计算
    print("正在从 /data2 极速加载 Train 数据...")
    train_ds = lgb.Dataset(str(BIN_DIR / "train_v10_top10.bin"))
    
    print("正在从 /data2 极速加载 Val 数据...")
    # 验证集必须设置 reference=train_ds，保证特征分桶边界一致
    val_ds = lgb.Dataset(str(BIN_DIR / "val_v10_top10.bin"), reference=train_ds)
    
    print(f"✅ 数据加载完毕！耗时仅 {(time.time() - t0):.2f} 秒。")

    # ==============================================================================
    # 🎯 工业级黑魔法：指数级梯度换算表 (强保 Top-10)
    # ==============================================================================
    custom_label_gain = [0] * 128

    # Top 1-5：百万/十万级梯度（核心中的核心，生死红线）
    custom_label_gain[127] = (1 << 20) - 1  # 1,048,575  (Top 1)
    custom_label_gain[63]  = (1 << 19) - 1  # 524,287    (Top 2)
    custom_label_gain[31]  = (1 << 18) - 1  # 262,143    (Top 3)
    custom_label_gain[15]  = (1 << 17) - 1  # 131,071    (Top 4)
    custom_label_gain[7]   = (1 << 16) - 1  # 65,535     (Top 5)

    # Top 6-7：万级梯度（次要核心）
    custom_label_gain[3]   = (1 << 15) - 1  # 32,767     (Top 6)
    custom_label_gain[2]   = (1 << 14) - 1  # 16,383     (Top 7)

    # Top 8-20：千级梯度（温和兜底，只求召回不求绝对顺序）
    custom_label_gain[1]   = (1 << 12) - 1  # 4,095      (Top 8-20)

    params = {
        # 🎯 核心算法回调 1：换回 lambdarank，天然专一于头部排序
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10, 20],

        # 🎯 核心破解：注入自定义换算表，完美解决 31 级溢出报错
        'label_gain': custom_label_gain,

        # 🎯 视野锁死前 20，配合强梯度实现算力极致聚焦
        'lambdarank_truncation_level': 20,

        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 95,
        'max_depth': 9,
        'n_estimators': 5000,
        'num_threads': 38,
        'max_bin': 255,  # 必须与 build_bin 时的参数一致
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_child_samples': 1000,
        'force_col_wise': True,
        'verbosity': -1,
    }

    print(f"\n💥 Training started...")
    
    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=20)
    timing_cb = TimingCallback(freq=10)

    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(10),
            timing_cb,
            ckpt_cb,
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = OUTPUT_DIR / 'model_v10_lambdarank_top10.txt'
    model.save_model(str(final_model_path))
    print(f"\n✅ 模型训练完成！已保存至 {final_model_path}")

    # 动态获取模型内固化的特征名称，避免在脚本里写死一长串列表
    feats = model.feature_name()
    imp = pd.DataFrame({
        'feature': feats, 
        'importance': model.feature_importance('gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")

if __name__ == '__main__':
    main()
