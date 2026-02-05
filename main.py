import gc
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.city_data import CityDataLoader
from src.data_loader import load_raw_data_from_duckdb, create_long_format_with_negatives
from src.feature_pipeline import FeaturePipeline  # 【召回模式】使用统一 Pipeline
from src.config import Config
# 1 用main处理生成训练数据  2 用train.py训练模型  3 用evaluate.py评估模型
# uv run main.py    【一定要在cmd 不要powershell】

pd.options.mode.copy_on_write = True

# 数据格式版本（用于检测旧数据）
DATA_FORMAT_VERSION = "v2_recall_mode"  # 召回模式版本


def check_data_format_compatibility(output_dir):
    """
    检查已存在的数据是否与当前代码兼容

    如果发现旧格式数据，提示用户删除
    """
    processed_files = list(Path(output_dir).glob("processed_*.parquet"))

    if not processed_files:
        return True  # 没有旧数据，可以继续

    # 检查第一个文件的格式
    first_file = processed_files[0]
    df_sample = pd.read_parquet(first_file)

    # 检查是否有必需的列
    required_cols = ['Type_ID_orig', 'From_City_orig']
    missing_cols = [col for col in required_cols if col not in df_sample.columns]

    if missing_cols:
        print("\n" + "="*60)
        print("⚠️  检测到旧格式数据！")
        print("="*60)
        print(f"发现文件: {first_file}")
        print(f"缺少列: {missing_cols}")
        print("\n当前代码需要新格式数据（召回模式 v2）")
        print("旧数据与新代码不兼容，需要删除后重新生成。")
        print("\n解决方案：")
        print(f"  rm -rf {output_dir}/*.parquet")
        print(f"  或者手动删除 {output_dir} 目录下的所有 .parquet 文件")
        print("\n然后重新运行: uv run main.py")
        print("="*60)
        return False

    return True


def process_single_year(year, city_data_loader, pipeline, output_dir, hard_candidates, save_to_file=True):
    """
    处理单个年份的数据（召回模式 - 使用统一 Pipeline）

    Args:
        year: 年份
        city_data_loader: 城市数据加载器
        pipeline: FeaturePipeline 实例（统一特征工程）
        output_dir: 输出目录
        hard_candidates: 困难负样本候选池（省会+一二线城市）
        save_to_file: 是否保存到文件（2000年不保存，只用于提供历史特征）

    Returns:
        如果save_to_file=True，返回(output_file, total_rows)
        如果save_to_file=False，返回None
    """
    print(f"\n{'='*60}")
    print(f"Processing Year {year}")
    print(f"{'='*60}")

    # 1. 加载原始数据
    print(f"[{year}] Step 1/4: Loading raw data from DuckDB...")
    df_wide = load_raw_data_from_duckdb(Config.DB_PATH, year_filter=[year])

    if df_wide.empty:
        print(f"[{year}] Warning: No data found for year {year}")
        return None

    print(f"[{year}] Loaded {len(df_wide):,} records")

    # 2. Wide to Long 转换（召回模式：20正样本 + 20困难负样本）
    print(f"[{year}] Step 2/4: Converting to long format (Recall Mode: 20 Pos + 20 Hard Neg)...")
    df_long = create_long_format_with_negatives(
        df_wide,
        city_data_loader.get_city_ids(),
        hard_candidates=hard_candidates,  # 【召回模式】只使用困难负样本
        neg_sample_rate=Config.NEG_SAMPLE_RATE,  # 固定20个
        is_test_set=False
    )

    del df_wide
    gc.collect()
    print(f"[{year}] Generated {len(df_long):,} samples (20 pos + 20 hard neg per query)")

    # 3. 创建查询组ID（在特征工程之前）
    print(f"[{year}] Step 3/4: Creating query groups...")
    query_cols = ['Year', 'Type_ID', 'From_City']
    df_long['qid'] = df_long.groupby(query_cols).ngroup()

    # 4. 【召回模式】使用统一的 FeaturePipeline 进行特征工程
    print(f"[{year}] Step 4/4: Feature Engineering (Unified Pipeline)...")
    is_training = (year >= Config.TRAIN_START_YEAR and year <= Config.TRAIN_END_YEAR)
    mode = 'train' if is_training else 'eval'

    df_long = pipeline.transform(
        df_long,
        year=year,
        mode=mode,
        verbose=True
    )

    # 打印最终 Schema
    print(f"[{year}] Final schema: {list(df_long.columns)}")
    print(f"[{year}] Sample dtypes: {dict(list(df_long.dtypes.items())[:10])}")

    # 注意：保留 Type_ID_orig 和 From_City_orig 用于后续年份的历史特征匹配
    # 这些列会在训练时通过 prepare_features() 函数排除

    # 保存文件
    output_file = Path(output_dir) / f"processed_{year}.parquet"
    print(f"[{year}] Saving to {output_file}...")
    df_long.to_parquet(output_file, index=False, engine='pyarrow')

    total_processed = len(df_long)
    memory_mb = df_long.memory_usage(deep=True).sum() / 1024 / 1024

    if save_to_file:
        print(f"[{year}] ✓ Completed: {total_processed:,} rows, {memory_mb:.1f} MB")
    else:
        print(f"[{year}] ✓ Completed: Saved for historical features only")

    del df_long
    gc.collect()

    if save_to_file:
        return output_file, total_processed
    else:
        return None


def main():
    import sys

    # 【召回模式】设置输出目录
    output_dir = Path(Config.OUTPUT_DIR) / 'processed_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 【重要】检查数据格式兼容性
    if not check_data_format_compatibility(output_dir):
        sys.exit(1)

    # 初始化数据加载器
    print("Loading city data...")
    city_data_loader = CityDataLoader(data_dir=Config.DATA_DIR)
    city_data_loader.load_all()

    # 【召回模式】创建统一的 FeaturePipeline
    print("Initializing Feature Pipeline...")
    pipeline = FeaturePipeline(city_data_loader, data_dir=str(output_dir))

    # ==========================================
    # 生成困难负样本候选池（省会 + 一二线城市）
    # ==========================================
    print("\nGenerating hard negative candidates pool...")
    city_info = city_data_loader.city_info

    # 1. 获取一二线城市（Tier <= 2）
    tier_cities = city_info[city_info['tier'] <= 2].index.tolist()

    # 2. 获取省会城市（假设 city_info 中有 is_capital 列，如果没有则跳过）
    if 'is_capital' in city_info.columns:
        capital_cities = city_info[city_info['is_capital'] == 1].index.tolist()
    else:
        # 如果没有省会标记，手动列出主要省会城市ID
        # 这里列出中国31个省会+直辖市的城市代码（前4位）
        capital_city_codes = [
            '1100',  # 北京
            '1200',  # 天津
            '1300',  # 石家庄（河北）
            '1400',  # 太原（山西）
            '1500',  # 呼和浩特（内蒙古）
            '2100',  # 沈阳（辽宁）
            '2200',  # 长春（吉林）
            '2300',  # 哈尔滨（黑龙江）
            '3100',  # 上海
            '3200',  # 南京（江苏）
            '3300',  # 杭州（浙江）
            '3400',  # 合肥（安徽）
            '3500',  # 福州（福建）
            '3600',  # 南昌（江西）
            '3700',  # 济南（山东）
            '4100',  # 郑州（河南）
            '4200',  # 武汉（湖北）
            '4300',  # 长沙（湖南）
            '4400',  # 广州（广东）
            '4500',  # 南宁（广西）
            '4600',  # 海口（海南）
            '5000',  # 重庆
            '5100',  # 成都（四川）
            '5200',  # 贵阳（贵州）
            '5300',  # 昆明（云南）
            '5400',  # 拉萨（西藏）
            '6100',  # 西安（陕西）
            '6200',  # 兰州（甘肃）
            '6300',  # 西宁（青海）
            '6400',  # 银川（宁夏）
            '6500',  # 乌鲁木齐（新疆）
        ]
        capital_cities = [cid for cid in capital_city_codes if cid in city_info.index]

    # 3. 合并并去重
    hard_candidates = list(set(tier_cities + capital_cities))

    print(f"  - Tier 1-2 cities: {len(tier_cities)}")
    print(f"  - Capital cities: {len(capital_cities)}")
    print(f"  - Total hard candidates (unique): {len(hard_candidates)}")
    print(f"  - Examples: {hard_candidates[:10]}")

    # 处理年份 (2000-2020，共21年)
    # 2000年用于提供2001年的历史特征
    years = list(range(Config.DATA_START_YEAR, Config.TEST_YEARS[-1] + 1))

    print(f"\n{'='*60}")
    print(f"Sequential Data Processing (Recall Mode)")
    print(f"{'='*60}")
    print(f"Processing {len(years)} years (2000-2020)...")
    print("Note: 2000 is processed for historical features only")
    print(f"{'='*60}\n")

    total_rows_processed = 0
    successful_years = []
    failed_years = []

    for year in years:
        try:
            # 检查文件是否已存在
            output_file = Path(output_dir) / f"processed_{year}.parquet"
            if output_file.exists():
                print(f"\nYear {year}: Already processed, skipping")
                continue

            # 处理年份
            # 2000年不计入训练数据统计，只用于历史特征
            save_to_file = (year >= Config.TRAIN_START_YEAR)
            result = process_single_year(
                year,
                city_data_loader,
                pipeline,  # 【召回模式】传入统一 Pipeline
                output_dir,
                hard_candidates,  # 传入困难负样本候选池
                save_to_file
            )

            if result:
                output_file, total_rows = result
                total_rows_processed += total_rows
                successful_years.append(year)
            elif year == Config.DATA_START_YEAR:
                # 2000年特殊处理
                successful_years.append(year)

            gc.collect()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nYear {year}: Error - {e}")
            import traceback
            traceback.print_exc()
            failed_years.append(year)
            continue

    print(f"\n{'='*60}")
    print("Processing Summary (Recall Mode)")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_years)} years")
    print(f"Failed: {len(failed_years)} years")
    if failed_years:
        print(f"Failed years: {failed_years}")
    print(f"Total training rows generated: {total_rows_processed:,}")
    print(f"\nDataset split:")
    print(f"  Historical: 2000 (1 year, for features only)")
    print(f"  Training: 2001-2017 (17 years)")
    print(f"  Validation: 2018 (1 year)")
    print(f"  Test: 2019-2020 (2 years)")
    print(f"\n【召回模式】每个Query: 20正样本 + 20困难负样本 = 40条数据")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
