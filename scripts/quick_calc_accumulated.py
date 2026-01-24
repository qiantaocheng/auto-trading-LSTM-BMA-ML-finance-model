#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速计算 CatBoost 和 LambdaRank 的累计收益（基于非重叠回测）
从最新的评估运行中提取预测数据
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import glob

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.time_split_80_20_oos_eval import calculate_group_returns_hold10d_nonoverlap

def main():
    # Find latest run directory
    result_dirs = glob.glob(str(project_root / "results" / "t10_time_split_80_20" / "run_*"))
    if not result_dirs:
        print("No results directory found")
        return
    
    latest_dir = max(result_dirs, key=lambda x: Path(x).stat().st_mtime)
    latest_path = Path(latest_dir)
    print(f"Latest run: {latest_path.name}")
    
    # Check for snapshot_id
    snapshot_file = latest_path / "snapshot_id.txt"
    if snapshot_file.exists():
        snapshot_id = snapshot_file.read_text().strip()
        print(f"Snapshot ID: {snapshot_id}")
    
    # We need to re-run evaluation with --models catboost lambdarank
    # Or extract from existing predictions if available
    print("\nTo calculate accumulated returns for CatBoost and LambdaRank:")
    print("Run the evaluation with --models parameter:")
    print(f"python scripts/time_split_80_20_oos_eval.py \\")
    print(f"    --horizon-days 10 --top-n 20 --cost-bps 10 \\")
    print(f"    --output-dir results/t10_time_split_80_20 \\")
    print(f"    --models catboost lambdarank \\")
    print(f"    --snapshot-id {snapshot_id if snapshot_file.exists() else '<snapshot_id>'}")

if __name__ == "__main__":
    main()
