#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动全量训练并等待完成，确保snapshot更新到Direct Predict
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def wait_for_training_completion(max_wait_hours=2):
    """等待训练完成"""
    print("=" * 80)
    print("等待全量训练完成...")
    print("=" * 80)
    
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    output_dir = project_root / "results" / "full_dataset_training"
    
    # 获取初始snapshot ID（如果有）
    initial_snapshot = None
    if latest_snapshot_file.exists():
        initial_snapshot = latest_snapshot_file.read_text(encoding="utf-8").strip()
        print(f"初始snapshot ID: {initial_snapshot}")
    
    # 找到最新的训练运行目录
    if output_dir.exists():
        run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            latest_run = run_dirs[0]
            print(f"监控训练运行: {latest_run.name}")
            snapshot_file = latest_run / "snapshot_id.txt"
        else:
            snapshot_file = None
    else:
        snapshot_file = None
    
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    check_interval = 60  # 每60秒检查一次
    
    print(f"\n开始监控（最多等待{max_wait_hours}小时）...")
    print("每60秒检查一次训练状态\n")
    
    while True:
        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600
        
        if elapsed > max_wait_seconds:
            print(f"\n[WARN] 等待超时（{max_wait_hours}小时），停止监控")
            break
        
        # 检查snapshot是否更新
        if latest_snapshot_file.exists():
            current_snapshot = latest_snapshot_file.read_text(encoding="utf-8").strip()
            if current_snapshot and current_snapshot != initial_snapshot:
                print(f"\n[OK] Snapshot已更新！")
                print(f"  新snapshot ID: {current_snapshot}")
                print(f"  更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  训练耗时: {elapsed_hours:.2f}小时")
                return True, current_snapshot
        
        # 检查训练运行目录中的snapshot
        if snapshot_file and snapshot_file.exists():
            run_snapshot = snapshot_file.read_text(encoding="utf-8").strip()
            if run_snapshot:
                # 更新latest_snapshot_id.txt
                latest_snapshot_file.write_text(run_snapshot, encoding="utf-8")
                print(f"\n[OK] 训练完成！Snapshot已更新到latest_snapshot_id.txt")
                print(f"  Snapshot ID: {run_snapshot}")
                print(f"  训练耗时: {elapsed_hours:.2f}小时")
                return True, run_snapshot
        
        # 显示进度
        if int(elapsed) % 300 == 0:  # 每5分钟显示一次
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练进行中... ({elapsed_hours:.2f}小时)")
        
        time.sleep(check_interval)
    
    return False, None

def main():
    """主函数"""
    print("=" * 80)
    print("全量训练启动脚本")
    print("=" * 80)
    print("\n训练配置:")
    print("  数据文件: D:\\trade\\data\\factor_exports\\polygon_factors_all_filtered_clean_final_v2.parquet")
    print("  Top N: 50")
    print("  Snapshot Tag: FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS")
    print("\n训练完成后:")
    print("  - Snapshot自动保存到latest_snapshot_id.txt")
    print("  - Direct Predict自动使用这个snapshot")
    print("=" * 80)
    
    # 检查是否已有训练在运行
    output_dir = project_root / "results" / "full_dataset_training"
    if output_dir.exists():
        run_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            latest_run = run_dirs[0]
            snapshot_file = latest_run / "snapshot_id.txt"
            
            # 检查训练是否已完成
            if snapshot_file.exists():
                snapshot_id = snapshot_file.read_text(encoding="utf-8").strip()
                print(f"\n[INFO] 发现已有训练运行: {latest_run.name}")
                print(f"[INFO] Snapshot ID: {snapshot_id}")
                
                # 更新latest_snapshot_id.txt
                latest_snapshot_file = project_root / "latest_snapshot_id.txt"
                latest_snapshot_file.write_text(snapshot_id, encoding="utf-8")
                print(f"[OK] Snapshot已更新到latest_snapshot_id.txt")
                print(f"[OK] Direct Predict将使用snapshot: {snapshot_id}")
                return 0
            else:
                print(f"\n[INFO] 发现训练运行: {latest_run.name}（可能仍在进行中）")
                print("[INFO] 等待训练完成...")
                
                # 等待训练完成
                success, snapshot_id = wait_for_training_completion()
                if success:
                    print(f"\n[OK] 训练完成！")
                    print(f"[OK] Snapshot已更新: {snapshot_id}")
                    print(f"[OK] Direct Predict将自动使用这个snapshot")
                    return 0
                else:
                    print(f"\n[WARN] 训练可能仍在进行中，请稍后检查")
                    return 1
    
    # 启动新训练
    print("\n启动新训练...")
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_full_dataset.py"),
        "--train-data", r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet",
        "--top-n", "50",
        "--log-level", "INFO"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("\n训练开始，等待完成...")
    
    # 运行训练
    try:
        result = subprocess.run(cmd, cwd=str(project_root), check=True)
        
        # 训练完成后，确保snapshot已更新
        latest_snapshot_file = project_root / "latest_snapshot_id.txt"
        if latest_snapshot_file.exists():
            snapshot_id = latest_snapshot_file.read_text(encoding="utf-8").strip()
            print(f"\n[OK] 训练完成！")
            print(f"[OK] Snapshot已更新: {snapshot_id}")
            print(f"[OK] Direct Predict将自动使用这个snapshot")
            return 0
        else:
            print(f"\n[ERROR] 训练完成但snapshot未更新")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n[WARN] 训练被中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())
