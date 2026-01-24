#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全运行80/20评估
- 训练使用80%数据（包含purge gap，无时间泄露）
- Snapshot只保存到运行目录，不覆盖latest_snapshot_id.txt
- 确保不会影响正在进行的全量训练
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """安全运行80/20评估"""
    print("=" * 80)
    print("80/20评估 - 安全模式（不覆盖latest_snapshot_id.txt）")
    print("=" * 80)
    
    # 设置环境变量，禁用train_from_document的自动snapshot保存到latest_snapshot_id.txt
    # 注意：train_from_document仍会保存snapshot到数据库，但不会更新latest_snapshot_id.txt
    os.environ['BMA_SKIP_LATEST_SNAPSHOT_UPDATE'] = '1'
    
    # 备份当前的latest_snapshot_id.txt
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    backup_file = project_root / "latest_snapshot_id.txt.backup_before_80_20_eval"
    
    if latest_snapshot_file.exists():
        snapshot_id_backup = latest_snapshot_file.read_text(encoding="utf-8").strip()
        backup_file.write_text(snapshot_id_backup, encoding="utf-8")
        print(f"[OK] 已备份latest_snapshot_id.txt: {snapshot_id_backup}")
    else:
        print(f"[INFO] latest_snapshot_id.txt不存在，无需备份")
    
    print("\n[INFO] 配置:")
    print("  - 数据文件: polygon_factors_all_filtered_clean_final_v2.parquet")
    print("  - 时间分割: 80%训练，20%测试")
    print("  - Purge gap: 10天（防止时间泄露）")
    print("  - start_date/end_date: 正确传递")
    print("  - Snapshot: 只保存到运行目录，不覆盖latest_snapshot_id.txt")
    print("=" * 80)
    
    # 导入并运行80/20评估
    try:
        from scripts.time_split_80_20_oos_eval import main as eval_main
        import argparse
        
        # 创建参数对象
        class Args:
            data_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet"
            train_data = r"D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet"
            split = 0.8
            horizon_days = 10
            models = ["catboost", "lambdarank", "ridge_stacking"]
            top_n = 20
            output_dir = "results/t10_time_split_80_20_final"
            snapshot_id = None
            ridge_base_cols = None
            cost_bps = 0.0
            benchmark = "QQQ"
            hac_method = "newey-west"
            hac_lag = None
            ema_top_n = -1
            ema_min_days = 3
            log_level = "INFO"
        
        args = Args()
        
        print("\n开始80/20评估...\n")
        
        # 运行评估
        eval_main(args)
        
        # 验证latest_snapshot_id.txt未被修改
        if latest_snapshot_file.exists() and backup_file.exists():
            current_snapshot = latest_snapshot_file.read_text(encoding="utf-8").strip()
            backup_snapshot = backup_file.read_text(encoding="utf-8").strip()
            
            if current_snapshot != backup_snapshot:
                print(f"\n[WARN] latest_snapshot_id.txt被修改了！")
                print(f"  备份值: {backup_snapshot}")
                print(f"  当前值: {current_snapshot}")
                print(f"[INFO] 恢复备份...")
                latest_snapshot_file.write_text(backup_snapshot, encoding="utf-8")
                print(f"[OK] 已恢复latest_snapshot_id.txt")
            else:
                print(f"\n[OK] latest_snapshot_id.txt未被修改（安全）")
        
        print(f"\n[OK] 80/20评估完成！")
        print(f"[OK] Snapshot保存在运行目录，未覆盖latest_snapshot_id.txt")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] 80/20评估失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 恢复备份（如果存在）
        if backup_file.exists():
            backup_snapshot = backup_file.read_text(encoding="utf-8").strip()
            latest_snapshot_file.write_text(backup_snapshot, encoding="utf-8")
            print(f"[INFO] 已恢复latest_snapshot_id.txt备份")
        
        return 1
    finally:
        # 清理环境变量
        os.environ.pop('BMA_SKIP_LATEST_SNAPSHOT_UPDATE', None)

if __name__ == "__main__":
    sys.exit(main())
