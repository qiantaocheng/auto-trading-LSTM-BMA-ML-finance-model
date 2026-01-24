"""
使用1/5 ticker子集进行训练和80/20 OOS评估
"""
import subprocess
import sys
from pathlib import Path

def run_training_and_eval(subset_file: str):
    """
    使用子集文件进行训练和80/20评估
    
    Args:
        subset_file: 子集parquet文件路径
    """
    subset_path = Path(subset_file)
    if not subset_path.exists():
        raise FileNotFoundError(f"子集文件不存在: {subset_file}")
    
    print("="*80)
    print("使用1/5 Ticker子集进行训练和80/20评估")
    print("="*80)
    print(f"\n子集文件: {subset_path}")
    
    # 步骤1: 训练
    print("\n" + "="*80)
    print("[步骤1] 使用子集进行训练")
    print("="*80)
    
    train_cmd = [
        sys.executable,
        "scripts/train_full_dataset.py",
        "--train-data", str(subset_path),
        "--top-n", "50",
        "--log-level", "INFO"
    ]
    
    print(f"\n执行命令: {' '.join(train_cmd)}")
    print("\n开始训练...")
    
    try:
        result = subprocess.run(
            train_cmd,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=False
        )
        print("\n[OK] 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 训练失败: {e}")
        return False
    
    # 步骤2: 80/20 OOS评估
    print("\n" + "="*80)
    print("[步骤2] 使用子集进行80/20 OOS评估")
    print("="*80)
    
    eval_cmd = [
        sys.executable,
        "scripts/time_split_80_20_oos_eval.py",
        "--data-file", str(subset_path),
        "--horizon-days", "10",
        "--split", "0.8",
        "--top-n", "20",
        "--log-level", "INFO"
    ]
    
    print(f"\n执行命令: {' '.join(eval_cmd)}")
    print("\n开始80/20评估...")
    
    try:
        result = subprocess.run(
            eval_cmd,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=False
        )
        print("\n[OK] 80/20评估完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 80/20评估失败: {e}")
        return False

if __name__ == "__main__":
    subset_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet"
    
    success = run_training_and_eval(subset_file)
    
    if success:
        print("\n" + "="*80)
        print("[OK] 训练和评估都已完成!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("[ERROR] 训练或评估过程中出现错误")
        print("="*80)
        sys.exit(1)
