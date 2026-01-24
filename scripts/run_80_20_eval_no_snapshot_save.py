#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行80/20评估，训练但不保存snapshot到latest_snapshot_id.txt
确保不会覆盖正在进行的全量训练snapshot
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """运行80/20评估，不保存snapshot到latest_snapshot_id.txt"""
    print("=" * 80)
    print("80/20评估 - 不覆盖latest_snapshot_id.txt")
    print("=" * 80)
    
    # 备份当前的latest_snapshot_id.txt（如果存在）
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    backup_file = project_root / "latest_snapshot_id.txt.backup_before_80_20"
    
    if latest_snapshot_file.exists():
        snapshot_id_backup = latest_snapshot_file.read_text(encoding="utf-8").strip()
        backup_file.write_text(snapshot_id_backup, encoding="utf-8")
        print(f"[INFO] 已备份latest_snapshot_id.txt: {snapshot_id_backup}")
        print(f"[INFO] 备份文件: {backup_file}")
    else:
        print(f"[INFO] latest_snapshot_id.txt不存在，无需备份")
    
    print("\n[INFO] 开始80/20评估...")
    print("[INFO] 训练将使用:")
    print("  - 数据文件: polygon_factors_all_filtered_clean_final_v2.parquet")
    print("  - 时间分割: 80%训练，20%测试（包含purge gap）")
    print("  - start_date和end_date: 正确传递，确保无时间泄露")
    print("[INFO] Snapshot将保存到运行目录，不会覆盖latest_snapshot_id.txt")
    print("=" * 80)
    
    # 运行80/20评估
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "time_split_80_20_oos_eval.py"),
        "--data-file", r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet",
        "--split", "0.8",
        "--models", "catboost", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--log-level", "INFO"
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("\n开始评估...\n")
    
    try:
        result = subprocess.run(cmd, cwd=str(project_root), check=True)
        
        # 检查latest_snapshot_id.txt是否被修改
        if latest_snapshot_file.exists():
            current_snapshot = latest_snapshot_file.read_text(encoding="utf-8").strip()
            if backup_file.exists():
                backup_snapshot = backup_file.read_text(encoding="utf-8").strip()
                if current_snapshot != backup_snapshot:
                    print(f"\n[WARN] latest_snapshot_id.txt被修改了！")
                    print(f"  备份值: {backup_snapshot}")
                    print(f"  当前值: {current_snapshot}")
                    print(f"[INFO] 恢复备份...")
                    latest_snapshot_file.write_text(backup_snapshot, encoding="utf-8")
                    print(f"[OK] 已恢复latest_snapshot_id.txt")
                else:
                    print(f"\n[OK] latest_snapshot_id.txt未被修改")
            else:
                print(f"\n[INFO] 无法验证latest_snapshot_id.txt是否被修改（无备份）")
        
        print(f"\n[OK] 80/20评估完成！")
        print(f"[OK] Snapshot保存在运行目录，未覆盖latest_snapshot_id.txt")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 80/20评估失败: {e}")
        
        # 恢复备份（如果存在）
        if backup_file.exists():
            backup_snapshot = backup_file.read_text(encoding="utf-8").strip()
            latest_snapshot_file.write_text(backup_snapshot, encoding="utf-8")
            print(f"[INFO] 已恢复latest_snapshot_id.txt备份")
        
        return 1
    except KeyboardInterrupt:
        print(f"\n[WARN] 评估被中断")
        
        # 恢复备份（如果存在）
        if backup_file.exists():
            backup_snapshot = backup_file.read_text(encoding="utf-8").strip()
            latest_snapshot_file.write_text(backup_snapshot, encoding="utf-8")
            print(f"[INFO] 已恢复latest_snapshot_id.txt备份")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
