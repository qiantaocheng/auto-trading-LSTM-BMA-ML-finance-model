#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查排除SBET后的80/20评估状态
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """检查排除SBET后的80/20评估状态"""
    print("=" * 80)
    print("排除SBET后的80/20评估状态检查")
    print("=" * 80)
    
    # 查找最新的运行目录
    results_dir = project_root / "results" / "t10_time_split_80_20_final"
    
    if not results_dir.exists():
        print(f"[INFO] 结果目录不存在: {results_dir}")
        return
    
    # 查找所有运行目录
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        print(f"[INFO] 未找到运行目录")
        return
    
    latest_run = run_dirs[0]
    print(f"\n[INFO] 最新运行目录: {latest_run.name}")
    print(f"[INFO] 修改时间: {datetime.fromtimestamp(latest_run.stat().st_mtime)}")
    
    # 检查snapshot_id.txt
    snapshot_file = latest_run / "snapshot_id.txt"
    if snapshot_file.exists():
        snapshot_id = snapshot_file.read_text(encoding="utf-8").strip()
        print(f"\n[OK] Snapshot ID: {snapshot_id}")
    else:
        print(f"\n[WARN] Snapshot ID文件不存在")
    
    # 检查结果文件
    oos_metrics_file = latest_run / "oos_metrics.json"
    if oos_metrics_file.exists():
        import json
        with open(oos_metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"\n[OK] 评估结果文件存在")
        print(f"\n[结果摘要]")
        print(f"  - 训练期: {metrics.get('train_start')} 至 {metrics.get('train_end')}")
        print(f"  - 测试期: {metrics.get('test_start')} 至 {metrics.get('test_end')}")
        print(f"  - IC: {metrics.get('IC', 0):.4f}")
        print(f"  - Rank IC: {metrics.get('Rank_IC', 0):.4f}")
        print(f"  - 平均收益: {metrics.get('avg_top_return', 0)*100:.2f}%")
        print(f"  - 胜率: {metrics.get('win_rate', 0)*100:.1f}%")
    else:
        print(f"\n[INFO] 评估结果文件尚未生成（可能仍在运行）")
    
    # 检查CSV结果
    csv_files = list(latest_run.glob("*_top20_vs_qqq.csv"))
    if csv_files:
        print(f"\n[OK] 找到 {len(csv_files)} 个CSV结果文件")
        for csv_file in csv_files[:3]:
            print(f"  - {csv_file.name}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
