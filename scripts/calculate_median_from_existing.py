#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate what median bucket returns would be from existing predictions (if available)"""

import pandas as pd
from pathlib import Path
import sys

def calculate_median_buckets_from_predictions(predictions_file: Path):
    """Calculate median bucket returns from predictions DataFrame"""
    if not predictions_file.exists():
        return None
    
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    print(f"Predictions shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'prediction' not in df.columns or 'actual' not in df.columns:
        print("ERROR: Missing 'prediction' or 'actual' columns")
        return None
    
    # Calculate median bucket returns
    top_buckets = [(1, 10), (11, 20), (21, 30)]
    bottom_buckets = [(1, 10)]
    
    results = []
    
    for date, date_group in df.groupby('date'):
        valid = date_group.dropna(subset=['prediction', 'actual'])
        if len(valid) < 30:
            continue
        
        sorted_group = valid.sort_values('prediction', ascending=False).reset_index(drop=True)
        n = len(sorted_group)
        
        row = {'date': pd.to_datetime(date), 'n_stocks': n}
        
        # Top buckets (MEDIAN)
        for a, b in top_buckets:
            if a <= n:
                s = sorted_group.iloc[a-1:b]['actual']
                row[f'top_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
        
        # Bottom buckets (MEDIAN)
        for a, b in bottom_buckets:
            if a <= n:
                start = max(0, n - b)
                end = n - (a - 1)
                s = sorted_group.iloc[start:end]['actual']
                row[f'bottom_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
        
        results.append(row)
    
    if not results:
        return None
    
    median_df = pd.DataFrame(results).sort_values('date')
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("MEDIAN BUCKET RETURNS SUMMARY")
    print("=" * 100)
    
    for col in median_df.columns:
        if col.endswith('_median') and median_df[col].notna().any():
            avg_median = median_df[col].mean()
            print(f"\n{col}:")
            print(f"  Average (mean of medians): {avg_median:.4f}%")
            print(f"  Median of medians: {median_df[col].median():.4f}%")
            print(f"  Std dev: {median_df[col].std():.4f}%")
    
    return median_df

def main():
    # Check for predictions files in the latest run
    base_dir = Path(r"D:\trade\results\t10_time_split_80_20_bucket_test_filtered")
    
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(run_dirs) == 0:
        print("ERROR: No run directories found")
        return 1
    
    run_dir = run_dirs[0]
    print(f"Checking run directory: {run_dir}")
    
    # Look for predictions files (they might be saved with different names)
    # The predictions are usually stored in the model snapshots or in memory during evaluation
    # Let's check if there's a way to access them
    
    print("\nNOTE: Predictions data may not be saved separately.")
    print("To get median bucket returns, you need to:")
    print("  1. Retrain models with the updated code (using median)")
    print("  2. Or recalculate from saved predictions if available")
    
    # Check bucket files to show current (mean-based) results
    print("\n" + "=" * 100)
    print("CURRENT BUCKET RESULTS (using MEAN - from previous run)")
    print("=" * 100)
    
    models = ['elastic_net', 'xgboost', 'catboost', 'lightgbm_ranker', 'lambdarank', 'ridge_stacking']
    
    for model in models:
        bucket_file = run_dir / f"{model}_bucket_returns.csv"
        if bucket_file.exists():
            df = pd.read_csv(bucket_file)
            print(f"\n{model.upper()}:")
            if 'top_1_10_return' in df.columns:
                print(f"  Top 1-10 avg (mean): {df['top_1_10_return'].mean():.4f}%")
            if 'top_11_20_return' in df.columns:
                print(f"  Top 11-20 avg (mean): {df['top_11_20_return'].mean():.4f}%")
            if 'top_21_30_return' in df.columns:
                print(f"  Top 21-30 avg (mean): {df['top_21_30_return'].mean():.4f}%")
    
    print("\n" + "=" * 100)
    print("TO GET MEDIAN RESULTS:")
    print("=" * 100)
    print("Run: python scripts/time_split_80_20_oos_eval.py --data-file ... --output-dir results/t10_time_split_80_20_median")
    print("=" * 100)
    
    return 0

if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
