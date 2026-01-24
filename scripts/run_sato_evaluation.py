#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 80/20 time split evaluation with Sato factors and save results to results/sato/
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run 80/20 evaluation with Sato factors"""
    
    # Ensure results/sato directory exists
    output_dir = project_root / "results" / "sato"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {output_dir}")
    
    # Data file with Sato factors
    data_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean.parquet"
    
    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        return 1
    
    print(f"[INFO] Data file: {data_file}")
    print(f"[INFO] Output directory: {output_dir}")
    print("\n[START] Starting 80/20 evaluation with Sato factors...")
    print("=" * 80)
    
    # Run the evaluation script
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "time_split_80_20_oos_eval.py"),
        "--data-file", str(data_file),
        "--horizon-days", "10",
        "--split", "0.8",
        "--models", "catboost", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--output-dir", str(output_dir),
        "--log-level", "INFO"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the command
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("[SUCCESS] Evaluation completed successfully!")
        print(f"[INFO] Results saved to: {output_dir}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"[ERROR] Evaluation failed with exit code: {result.returncode}")
        print("=" * 80)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
