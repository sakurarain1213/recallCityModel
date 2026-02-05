#!/usr/bin/env python
"""
数据库和程序诊断脚本
用于排查 KeyboardInterrupt 错误的根本原因
"""

import sys
import os
from pathlib import Path
import signal

def check_database_file(db_path):
    """检查数据库文件"""
    print("\n" + "=" * 60)
    print("1. 检查数据库文件")
    print("=" * 60)
    
    if not Path(db_path).exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"   请检查路径是否正确")
        return False
    
    file_size = Path(db_path).stat().st_size
    print(f"✅ 数据库文件存在: {db_path}")
    print(f"   文件大小: {file_size / 1024**2:.2f} MB")
    
    return True


def check_duckdb_installation():
    """检查 DuckDB 是否正确安装"""
    print("\n" + "=" * 60)
    print("2. 检查 DuckDB 安装")
    print("=" * 60)
    
    try:
        import duckdb
        print(f"✅ DuckDB 已安装")
        print(f"   版本: {duckdb.__version__}")
        return True
    except ImportError:
        print("❌ DuckDB 未安装")
        print("   请运行: pip install duckdb")
        return False


def test_database_connection(db_path):
    """测试数据库连接"""
    print("\n" + "=" * 60)
    print("3. 测试数据库连接")
    print("=" * 60)
    
    try:
        import duckdb
        
        # 设置超时处理
        def timeout_handler(signum, frame):
            raise TimeoutError("数据库连接超时（10秒）")
        
        # 在Unix系统上设置超时（Windows不支持signal.alarm）
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10秒超时
        
        try:
            print("正在连接数据库...")
            with duckdb.connect(db_path, read_only=True) as con:
                print("✅ 连接成功")
                
                # 获取表列表
                print("\n查询可用的表...")
                tables = con.execute("SHOW TABLES").fetchall()
                print(f"可用的表: {[t[0] for t in tables]}")
                
                # 检查 migration_data 表
                if any('migration_data' in str(t).lower() for t in tables):
                    print("\n✅ 找到 migration_data 表")
                    
                    # 快速计数测试
                    print("正在统计行数（这可能需要一些时间）...")
                    count = con.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
                    print(f"总行数: {count:,}")
                    
                    # 查看表结构
                    print("\n表结构:")
                    schema = con.execute("DESCRIBE migration_data").fetchall()
                    for col in schema[:10]:  # 只显示前10列
                        print(f"  - {col[0]}: {col[1]}")
                    if len(schema) > 10:
                        print(f"  ... 还有 {len(schema) - 10} 列")
                    
                    return True
                else:
                    print("❌ 未找到 migration_data 表")
                    print("   请检查表名是否正确")
                    return False
                    
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 取消超时
                
    except TimeoutError as e:
        print(f"❌ {e}")
        print("   数据库响应太慢，可能文件损坏或数据量过大")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory_usage():
    """检查内存使用情况"""
    print("\n" + "=" * 60)
    print("4. 检查系统资源")
    print("=" * 60)
    
    try:
        import psutil
        
        # 内存信息
        mem = psutil.virtual_memory()
        print(f"总内存: {mem.total / 1024**3:.2f} GB")
        print(f"可用内存: {mem.available / 1024**3:.2f} GB")
        print(f"使用率: {mem.percent}%")
        
        if mem.percent > 90:
            print("⚠️ 警告: 内存使用率过高，可能导致程序卡顿")
        else:
            print("✅ 内存充足")
            
        return True
    except ImportError:
        print("⚠️ psutil 未安装，跳过内存检查")
        print("   可选安装: pip install psutil")
        return True


def suggest_fixes():
    """提供修复建议"""
    print("\n" + "=" * 60)
    print("修复建议")
    print("=" * 60)
    
    print("""
常见问题和解决方案:

1️⃣ 如果数据库连接超时或程序卡死:
   - 检查数据库文件是否损坏
   - 尝试减少数据量: 使用 limit 参数测试
   - 增加年份过滤: 只加载必要的年份

2️⃣ 如果出现 KeyboardInterrupt:
   - 问题: 程序长时间无响应，你按了 Ctrl+C
   - 原因: 可能是数据加载太慢或内存不足
   - 解决: 使用我提供的修复版本，它包含:
     * 正确的资源管理（context manager）
     * 进度提示
     * 超时处理
     * 更好的错误信息

3️⃣ 如果内存不足:
   - 使用采样: sample_rate=0.1 (加载10%数据)
   - 限制行数: limit=10000
   - 分批处理: 按年份分别加载

4️⃣ 推荐的加载方式:
   
   # 先测试少量数据
   df = load_raw_data_from_duckdb(db_path, limit=100)
   
   # 如果成功，增加到1000
   df = load_raw_data_from_duckdb(db_path, limit=1000)
   
   # 然后按年份加载
   df = load_raw_data_from_duckdb(db_path, year_filter=[2018])
   
   # 最后全量加载（如果内存足够）
   df = load_raw_data_from_duckdb(db_path)

5️⃣ 使用修复后的代码:
   替换你的 src/data_loader.py 为我提供的 data_loader_fixed.py
   
   主要改进:
   - ✅ 使用 context manager 管理连接
   - ✅ 添加进度提示
   - ✅ 更好的错误处理
   - ✅ 防止资源泄漏
   - ✅ 支持 Ctrl+C 优雅退出
""")


def main():
    """主诊断流程"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     数据库问题诊断工具                                    ║
║     Database Diagnostic Tool                             ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # 数据库路径（请根据实际情况修改）
    db_path = "data/local_migration_data.db"
    
    # 如果命令行提供了路径，使用命令行参数
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print(f"数据库路径: {db_path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 执行诊断
    checks = [
        check_database_file(db_path),
        check_duckdb_installation(),
        test_database_connection(db_path),
        check_memory_usage(),
    ]
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断结果汇总")
    print("=" * 60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ 所有检查通过 ({passed}/{total})")
        print("\n如果程序仍然有问题，请:")
        print("1. 使用我提供的 data_loader_fixed.py 替换原文件")
        print("2. 从小数据量开始测试 (limit=10)")
        print("3. 检查是否有其他代码问题")
    else:
        print(f"⚠️ 发现问题 ({passed}/{total} 通过)")
    
    suggest_fixes()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 诊断被用户中断 (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)