"""
预分配内存逐年填入：解决每年越来越慢 + 内存峰值问题
- 第一步：扫描所有年份，估算总行数上限
- 第二步：一次性 np.empty 申请虚拟地址空间
- 第三步：逐年生成数据，直接填入预分配的大矩阵切片
- 第四步：裁剪到实际大小，构建 LightGBM Dataset 并保存

用法: uv run 0322merge-memory.py
"""
import gc
import sys
import time
import numpy as np
import lightgbm as lgb
import importlib.util
from tqdm import tqdm

# 动态导入你原有的 0322pipeline.py
try:
    spec = importlib.util.spec_from_file_location("pipeline", "0322pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
except Exception as e:
    print(f"导入 0322pipeline.py 失败，请确保它在同级目录下！报错: {e}")
    sys.exit(1)


def main():
    train_years = list(range(2000, 2019))
    BIN_DIR = pipeline.BIN_DIR
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    NEG_SAMPLE_N = 200  # 每个 query 最多保留 200 个负样本

    # ============================================================
    # Step 1: 估算总行数上限（先只跑 build_year_data 拿 query 数，不生成特征）
    # ============================================================
    print("=== Step 1: 扫描所有年份，估算总行数 ===")
    query_counts = []
    for y in tqdm(train_years, desc="扫描年份", unit="年"):
        # 临时 patch: monkey-patch build_year_data 返回 (None, None, None, n_q)
        # 直接从 DB 拿 query 数量，避免重复加载 tensor
        import duckdb
        conn = duckdb.connect(str(pipeline.DB_PATH), read_only=True)
        n_q = conn.execute(
            f"SELECT COUNT(*) FROM migration_data WHERE Year = {y}"
        ).fetchone()[0]
        conn.close()
        query_counts.append(n_q)
        del conn
        gc.collect()

    total_queries = sum(query_counts)
    # 上限估算：每个 query = 全正样本 (~20) + 最多 200 负样本
    max_rows_estimate = sum(
        n_q * (20 + NEG_SAMPLE_N) for n_q in query_counts
    )
    print(f"  总 queries: {total_queries:,}")
    print(f"  预估总行数上限: {max_rows_estimate:,} (~{max_rows_estimate/1e6:.0f}M)")

    # ============================================================
    # Step 2: 预分配大矩阵（虚拟地址占坑，不填物理内存）
    # ============================================================
    print("\n=== Step 2: 预分配全局特征矩阵 ===")
    n_feats = len(pipeline.FEATS)
    X_all = np.empty((max_rows_estimate, n_feats), dtype=np.float32)
    labels_all = np.empty(max_rows_estimate, dtype=np.int8)
    # qid 用 int32 存行索引，方便后续构建 group
    # 由于负采样后每组大小不一，不预先填 group，改为逐年记录 group_sizes
    group_sizes = []

    current_idx = 0

    # ============================================================
    # Step 3: 逐年生成并填入
    # ============================================================
    print("\n=== Step 3: 逐年生成并填入大矩阵 ===")
    for year, n_q in zip(train_years, query_counts):
        t_year = time.time()
        print(f"\n--- {year} (共 {n_q:,} queries) ---")

        # build_year_data 已内置负采样逻辑，neg_sample_n=200
        X, labels, qids, _ = pipeline.build_year_data(
            year, sample_n=None, neg_sample_n=NEG_SAMPLE_N, seed=42
        )

        rows = len(X)
        if current_idx + rows > max_rows_estimate:
            raise RuntimeError(
                f"行数超预估！{year}: {current_idx}+{rows} > {max_rows_estimate}"
            )

        # 填入预分配的大矩阵
        X_all[current_idx: current_idx + rows] = X
        labels_all[current_idx: current_idx + rows] = labels

        # 计算这一年的 group sizes
        splits = np.r_[0, np.where(np.diff(qids) != 0)[0] + 1, len(qids)]
        yearly_groups = np.diff(splits).astype(np.int32)
        group_sizes.append(yearly_groups)

        n_queries_year = len(yearly_groups)
        print(f"  生成 {rows:,} 行, {n_queries_year:,} queries, "
              f"平均 {rows/n_queries_year:.1f} 行/query, "
              f"耗时 {time.time()-t_year:.1f}s")

        current_idx += rows

        # 立即释放当年临时变量
        del X, labels, qids, yearly_groups
        gc.collect()

    # ============================================================
    # Step 4: 裁剪到实际大小
    # ============================================================
    print(f"\n=== Step 4: 裁剪到实际大小 ===")
    X_final = X_all[:current_idx]
    labels_final = labels_all[:current_idx]
    groups_final = np.concatenate(group_sizes).astype(np.int32)

    del X_all, labels_all, group_sizes
    gc.collect()

    print(f"  最终: {len(X_final):,} 行, {len(groups_final):,} groups")
    print(f"  压缩比: {total_queries * 336 / current_idx:.2f}x "
          f"(原始 {total_queries * 336 / 1e6:.0f}M -> 采样后 {current_idx / 1e6:.0f}M)")

    # ============================================================
    # Step 5: 构建 LightGBM Dataset 并保存
    # ============================================================
    print("\n=== Step 5: 构建 LightGBM Dataset ===")
    ds = lgb.Dataset(
        X_final,
        label=labels_final,
        feature_name=list(pipeline.FEATS),
        categorical_feature=pipeline.CATS,
        free_raw_data=True,
        params={
            'max_bin': 255,
            'num_threads': 40,
        }
    )
    ds.set_group(groups_final)

    print("  Constructing (计算全局特征分箱)...")
    t_construct = time.time()
    ds.construct()
    print(f"  Dataset 构建耗时: {time.time()-t_construct:.1f}s")

    out_path = BIN_DIR / "train_all.bin"
    print(f"\n=== 保存 {out_path.name} ===")
    ds.save_binary(str(out_path))

    print(f"\n  大功告成！bin 文件大小: {out_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"  后续训练: train_ds = lgb.Dataset(str(BIN_DIR / 'train_all.bin'))")


if __name__ == '__main__':
    main()
