"""
重新训练并运行80/20评估，添加详细跟踪日志
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_with_tracking():
    """运行训练和评估，并跟踪进度"""
    print("="*80)
    print("重新训练和80/20评估 - 详细跟踪模式")
    print("="*80)
    print(f"开始时间: {datetime.now()}")
    
    project_root = Path(__file__).resolve().parent.parent
    subset_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet"
    
    if not subset_file.exists():
        print(f"[ERROR] 子集文件不存在: {subset_file}")
        return False
    
    print(f"\n子集文件: {subset_file}")
    print(f"文件大小: {subset_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 步骤1: 训练
    print("\n" + "="*80)
    print("[步骤1] 开始训练")
    print("="*80)
    print(f"时间: {datetime.now()}")
    
    train_cmd = [
        sys.executable,
        "scripts/train_full_dataset.py",
        "--train-data", str(subset_file),
        "--top-n", "50",
        "--log-level", "INFO"
    ]
    
    print(f"\n执行命令: {' '.join(train_cmd)}")
    print("\n开始训练...")
    print("注意: 训练过程中会输出详细的fold跟踪信息")
    
    train_start = time.time()
    try:
        result = subprocess.run(
            train_cmd,
            cwd=project_root,
            check=True,
            capture_output=False
        )
        train_time = time.time() - train_start
        print(f"\n[OK] 训练完成! 耗时: {train_time/60:.1f}分钟")
    except subprocess.CalledProcessError as e:
        train_time = time.time() - train_start
        print(f"\n[ERROR] 训练失败 (耗时: {train_time/60:.1f}分钟): {e}")
        print("\n请检查日志中的详细跟踪信息，查找阻塞位置:")
        print("  - 查找 'Fold X/Y 开始处理' 来确定最后一个处理的fold")
        print("  - 查找 '训练窗天数' 来检查是否有fold被跳过")
        print("  - 查找 '数据分割完成' 来检查数据准备是否成功")
        return False
    
    # 步骤2: 80/20评估
    print("\n" + "="*80)
    print("[步骤2] 开始80/20评估")
    print("="*80)
    print(f"时间: {datetime.now()}")
    
    eval_cmd = [
        sys.executable,
        "scripts/time_split_80_20_oos_eval.py",
        "--data-file", str(subset_file),
        "--horizon-days", "10",
        "--split", "0.8",
        "--top-n", "20",
        "--log-level", "INFO"
    ]
    
    print(f"\n执行命令: {' '.join(eval_cmd)}")
    print("\n开始80/20评估...")
    
    eval_start = time.time()
    try:
        result = subprocess.run(
            eval_cmd,
            cwd=project_root,
            check=True,
            capture_output=False
        )
        eval_time = time.time() - eval_start
        print(f"\n[OK] 80/20评估完成! 耗时: {eval_time/60:.1f}分钟")
        return True
    except subprocess.CalledProcessError as e:
        eval_time = time.time() - eval_start
        print(f"\n[ERROR] 80/20评估失败 (耗时: {eval_time/60:.1f}分钟): {e}")
        return False

if __name__ == "__main__":
    success = run_with_tracking()
    
    if success:
        print("\n" + "="*80)
        print("[OK] 训练和评估都已完成!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("[ERROR] 训练或评估过程中出现错误")
        print("="*80)
        print("\n请检查日志中的详细跟踪信息:")
        print("  - 查找跟踪日志标记")
        print("  - 查找最后一个成功处理的fold")
        print("  - 查找错误或警告信息")
        sys.exit(1)
