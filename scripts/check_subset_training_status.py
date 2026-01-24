"""
检查子集训练和80/20评估的状态
"""
import sys
from pathlib import Path
from datetime import datetime

def check_status():
    """检查训练和评估状态"""
    print("="*80)
    print("检查子集训练和80/20评估状态")
    print("="*80)
    
    project_root = Path(__file__).resolve().parent.parent
    
    # 检查训练状态
    print("\n[训练状态]")
    train_dir = project_root / "results" / "full_dataset_training"
    if train_dir.exists():
        runs = sorted(train_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            latest_run = runs[0]
            print(f"  最新训练运行: {latest_run.name}")
            print(f"  创建时间: {datetime.fromtimestamp(latest_run.stat().st_mtime)}")
            
            snapshot_file = latest_run / "snapshot_id.txt"
            if snapshot_file.exists():
                snapshot_id = snapshot_file.read_text(encoding='utf-8').strip()
                print(f"  [OK] Snapshot ID: {snapshot_id}")
                print(f"  状态: 训练完成")
            else:
                print(f"  [IN PROGRESS] 训练进行中...")
                # 检查是否有日志文件
                log_files = list(latest_run.glob("*.log")) + list(latest_run.glob("*.txt"))
                if log_files:
                    print(f"  找到 {len(log_files)} 个日志文件")
        else:
            print(f"  [WARN] 没有找到训练运行目录")
    else:
        print(f"  [WARN] 训练目录不存在: {train_dir}")
    
    # 检查80/20评估状态
    print("\n[80/20评估状态]")
    eval_dir = project_root / "results" / "t10_time_split_80_20_final"
    if eval_dir.exists():
        runs = sorted(eval_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            latest_run = runs[0]
            print(f"  最新评估运行: {latest_run.name}")
            print(f"  创建时间: {datetime.fromtimestamp(latest_run.stat().st_mtime)}")
            
            report_file = latest_run / "report_df.csv"
            if report_file.exists():
                print(f"  [OK] 报告文件存在: {report_file}")
                print(f"  状态: 评估完成")
                # 读取报告的前几行
                try:
                    import pandas as pd
                    df = pd.read_csv(report_file)
                    print(f"  报告形状: {df.shape}")
                    print(f"  报告列: {list(df.columns)}")
                    if len(df) > 0:
                        print(f"\n  报告摘要:")
                        # Show key metrics
                    key_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['sharpe', 'return', 'cum', 'drawdown', 'win'])]
                    if key_cols:
                        print(f"  关键指标列: {key_cols}")
                        for col in key_cols[:5]:  # Show first 5 key columns
                            if col in df.columns:
                                print(f"    {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
                    print(f"\n  完整报告前5行:")
                    print(df.head().to_string())
                except Exception as e:
                    print(f"  [WARN] 无法读取报告: {e}")
            else:
                print(f"  [IN PROGRESS] 评估进行中...")
                # 检查是否有其他文件
                other_files = list(latest_run.glob("*"))
                if other_files:
                    print(f"  找到 {len(other_files)} 个文件")
                    for f in other_files[:5]:
                        print(f"    - {f.name}")
        else:
            print(f"  [WARN] 没有找到评估运行目录")
    else:
        print(f"  [WARN] 评估目录不存在: {eval_dir}")
    
    # 检查子集文件
    print("\n[子集文件]")
    subset_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet"
    if subset_file.exists():
        print(f"  [OK] 子集文件存在: {subset_file}")
        print(f"  文件大小: {subset_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  修改时间: {datetime.fromtimestamp(subset_file.stat().st_mtime)}")
    else:
        print(f"  [WARN] 子集文件不存在: {subset_file}")
    
    print("\n" + "="*80)
    print("状态检查完成")
    print("="*80)

if __name__ == "__main__":
    check_status()
