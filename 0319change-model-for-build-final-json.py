import lightgbm as lgb
import treelite
import tl2cgen
import time

# 注意只能在Linux上运行 
orig_path = "output/models/0319recall_model_final.txt"
trunc_path = "output/models/0319recall_model_trunc.txt"
so_path = "output/models/0319recall_model_trunc.so"

print("1. 加载并分析原始模型...")
bst = lgb.Booster(model_file=orig_path)
total_trees = bst.num_trees()

# 核心魔法：只保留前 25% 的树（对于召回 Top40 完全足够，保底 150 棵）
keep_trees = max(int(total_trees * 0.25), 150)
print(f"--> 原模型共有 {total_trees} 棵树。")
print(f"--> 为实现极速召回，强制截取最核心的前 {keep_trees} 棵树！")

bst.save_model(trunc_path, num_iteration=keep_trees)

print("\n2. 使用 treelite 加载截断后的模型...")
tl_model = treelite.frontend.load_lightgbm_model(trunc_path)

print("3. 使用 tl2cgen 编译为极速 C++ 动态链接库...")
t0 = time.time()
tl2cgen.export_lib(
    tl_model, 
    toolchain="gcc", 
    libpath=so_path, 
    params={'parallel_comp': 40}
)
print(f"编译完成！耗时: {time.time() - t0:.2f} 秒。输出文件: {so_path}")