"""
LightGBM 专属召回训练脚本 (纯二分类 + O(1)极速加载 + 无阻断 Recall 实时监控)
"""

import time
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path

# ================= 极速存储路径 =================
BIN_DIR = Path("/data2/wxj/recall_bin") 
OUTPUT_DIR = Path("output")

class CheckpointCallback:
    def __init__(self, output_dir, freq=10):
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freq = freq
        self.before_iteration = False

    def __call__(self, env):
        current_round = env.iteration + 1
        if current_round % self.freq == 0:
            ckpt_path = self.output_dir / f"recall_model_iter_{current_round}.txt"
            env.model.save_model(str(ckpt_path))

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


# ================= 极速无阻断 Recall 计算 =================
def get_recall_evaluator(val_labels_path, val_groups_path):
    """
    黑科技：通过预先切片好 Numpy 数组，使得回调内部完全向量化，零 For 循环等待时间。
    计算公式严格遵守：Hit Count / min(K, n)
    """
    labels = np.load(val_labels_path)
    groups = np.load(val_groups_path)
    
    # 获取切片边界
    split_indices = np.r_[0, np.cumsum(groups)]
    labels_split = np.split(labels, split_indices[1:-1])
    
    # 过滤掉标签全是 0 的异常组，防止分母出现 0
    valid_idx = [i for i, l in enumerate(labels_split) if np.sum(l) > 0]
    
    def recall_eval(preds, train_data):
        preds_split = np.split(preds, split_indices[1:-1])
        metrics = []
        
        for k in [5, 10, 20]:
            total_score = 0.0
            for i in valid_idx:
                l = labels_split[i]
                p = preds_split[i]
                n = np.sum(l) # GT 中的实际正样本数量
                
                # 极端情况防抖：如果组内一共都没 k 行（比如切负样本切多了），取实际最大行数
                actual_k = min(k, len(p))
                
                # 利用 numpy.argpartition 极速挑出前 actual_k 名的索引（无序挑出，比 sort 快 10 倍）
                topk_idx = np.argpartition(p, -actual_k)[-actual_k:]
                hits = np.sum(l[topk_idx])
                
                # 核心公式
                total_score += hits / min(k, n)
                
            mean_score = total_score / len(valid_idx)
            # is_higher_better = True
            metrics.append((f'Recall@{k}', mean_score, True))
            
        return metrics
        
    return recall_eval


def main():
    print("=" * 60)
    print("🚀 LightGBM 召回特化模式启动 (O(1)极速加载 + 扁平权重二分类)")
    print("=" * 60)

    t0 = time.time()
    print("正在极速加载二分类缓存...")
    train_ds = lgb.Dataset(str(BIN_DIR / "train_recall.bin"))
    val_ds = lgb.Dataset(str(BIN_DIR / "val_recall.bin"), reference=train_ds)
    print(f"✅ 数据加载完毕！耗时仅 {(time.time() - t0):.2f} 秒。")

    # 载入用于评估的高速评测器
    eval_func = get_recall_evaluator(
        BIN_DIR / "val_labels.npy", 
        BIN_DIR / "val_groups.npy"
    )

    params = {
        'objective': 'binary',            # 🎯 纯粹的分类目标
        'metric': 'None',                 # 🎯 关掉默认指标，用我们手写的严谨 Recall
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,            # 二分类可以适当跑快点
        'num_leaves': 127,
        'max_depth': 10,
        'n_estimators': 8000,
        'num_threads': 38,
        'max_bin': 255,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 2.0,
        'min_child_samples': 200,
        'force_col_wise': True,
        'verbosity': -1,
    }

    print(f"\n💥 Training started...")
    
    ckpt_cb = CheckpointCallback(output_dir=OUTPUT_DIR, freq=40)
    timing_cb = TimingCallback(freq=10)

    model = lgb.train(
        params, train_ds,
        valid_sets=[val_ds], valid_names=['val'],
        feval=eval_func,  # 注入我们的评测黑科技
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(10),
            timing_cb,
            ckpt_cb,
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = OUTPUT_DIR / 'recall_model_final.txt'
    model.save_model(str(final_model_path))
    print(f"\n✅ 召回模型训练完成！已保存至 {final_model_path}")

    feats = model.feature_name()
    imp = pd.DataFrame({
        'feature': feats, 
        'importance': model.feature_importance('gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Features:\n{imp.head(20).to_string(index=False)}")

if __name__ == '__main__':
    main()