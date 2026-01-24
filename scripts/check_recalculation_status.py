#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check status of factor recalculation
"""

from pathlib import Path
import pandas as pd

input_file = Path("D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet")
output_file = Path("D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet")

print("=" * 80)
print("Factor Recalculation Status Check")
print("=" * 80)

print(f"\nInput file exists: {input_file.exists()}")
if input_file.exists():
    try:
        df_in = pd.read_parquet(input_file)
        print(f"  Shape: {df_in.shape}")
        print(f"  Columns: {len(df_in.columns)}")
    except Exception as e:
        print(f"  Error reading: {e}")

print(f"\nOutput file exists: {output_file.exists()}")
if output_file.exists():
    try:
        df_out = pd.read_parquet(output_file)
        print(f"  Shape: {df_out.shape}")
        print(f"  Columns: {len(df_out.columns)}")
        
        # Check new factors
        print("\nNew Factors Check:")
        new_factors = ['obv_momentum_40d', 'feat_vol_price_div_30d', 'vol_ratio_30d', 
                      'ret_skew_30d', 'ivol_30', 'blowoff_ratio_30d']
        for f in new_factors:
            status = "[YES]" if f in df_out.columns else "[NO]"
            print(f"  {status} {f}")
        
        # Check old factors removed
        print("\nOld Factors Check (should be removed):")
        old_factors = ['obv_divergence', 'feat_sato_momentum_10d', 'feat_sato_divergence_10d',
                      'vol_ratio_20d', 'ret_skew_20d', 'ivol_20', 'blowoff_ratio']
        for f in old_factors:
            status = "[REMOVED]" if f not in df_out.columns else "[STILL PRESENT]"
            print(f"  {status} {f}")
        
        # Check other columns kept
        print("\nOther Columns Check (should be kept):")
        other_cols = ['downside_beta_252', 'momentum_60d', 'obv_momentum_60d', 
                     'ebit', 'making_new_low_5d', 'roa', 'target']
        for col in other_cols:
            status = "[KEPT]" if col in df_out.columns else "[MISSING]"
            print(f"  {status} {col}")
            
    except Exception as e:
        print(f"  Error reading: {e}")
else:
    print("  [INFO] Output file not found - script may still be running")
    print("  [INFO] This is a large dataset (4M+ rows), recalculation may take 30-60+ minutes")

print("\n" + "=" * 80)
