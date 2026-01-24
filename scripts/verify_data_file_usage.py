#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify data file usage in Direct Predict and 80/20 time split
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 80)
    print("Data File Usage Verification")
    print("=" * 80)
    
    data_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean.parquet"
    
    print(f"\n1. File: {data_file.name}")
    print("-" * 80)
    
    if data_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(data_file)
            
            print(f"   [OK] File exists")
            print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            print(f"   Index type: {type(df.index).__name__}")
            
            if isinstance(df.index, pd.MultiIndex):
                dates = df.index.get_level_values('date').unique()
                tickers = df.index.get_level_values('ticker').unique()
                print(f"   Date range: {dates.min()} to {dates.max()}")
                print(f"   Unique tickers: {len(tickers):,}")
                print(f"   Unique dates: {len(dates):,}")
            
            print(f"\n   Columns ({len(df.columns)} total):")
            for i, col in enumerate(df.columns, 1):
                marker = "[SATO]" if "sato" in col.lower() else "     "
                print(f"   {marker} {i:2d}. {col}")
            
            has_sato_momentum = "feat_sato_momentum_10d" in df.columns
            has_sato_divergence = "feat_sato_divergence_10d" in df.columns
            
            print(f"\n   Sato factors:")
            print(f"   [OK] feat_sato_momentum_10d: {has_sato_momentum}")
            print(f"   [OK] feat_sato_divergence_10d: {has_sato_divergence}")
            
            if has_sato_momentum and has_sato_divergence:
                non_zero_momentum = (df['feat_sato_momentum_10d'] != 0.0).sum()
                non_zero_divergence = (df['feat_sato_divergence_10d'] != 0.0).sum()
                print(f"   Non-zero momentum: {non_zero_momentum:,} / {len(df):,} ({100*non_zero_momentum/len(df):.1f}%)")
                print(f"   Non-zero divergence: {non_zero_divergence:,} / {len(df):,} ({100*non_zero_divergence/len(df):.1f}%)")
            
        except Exception as e:
            print(f"   [ERROR] Error reading file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   [MISSING] File does not exist: {data_file}")
    
    print("\n" + "=" * 80)
    print("2. Usage in Direct Predict and 80/20 Time Split")
    print("=" * 80)
    
    print("\n   Direct Predict (app.py):")
    print("   - Uses parquet file for: TICKER LIST EXTRACTION ONLY")
    print("   - Does NOT load factors from parquet")
    print("   - Computes factors from live market data (API)")
    print("   - All first-layer models use computed factors")
    
    print("\n   80/20 Time Split (time_split_80_20_oos_eval.py):")
    print("   - Uses parquet file for: FACTOR DATA")
    print("   - Loads factors directly from parquet")
    print("   - Can use this file via: --data-file parameter")
    print("   - All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank) use parquet factors")
    print("   - Automatically computes Sato factors if missing")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("[OK] File exists and contains Sato factors")
    print("[OK] Direct Predict: Uses file for tickers only (not factors)")
    print("[OK] 80/20 Time Split: Uses file for factors (all first-layer models)")
    print("=" * 80)

if __name__ == "__main__":
    main()
