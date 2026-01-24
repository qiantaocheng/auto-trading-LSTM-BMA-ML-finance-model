#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check factor values in the updated dataset"""

import pandas as pd
import numpy as np

df = pd.read_parquet('data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet')

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumns: {list(df.columns)}")

print("\n" + "=" * 80)
print("Factor Value Analysis")
print("=" * 80)

factors = ['momentum_10d', 'ivol_30', 'feat_vol_price_div_30d', 'rsi_21', 'liquid_momentum', 'obv_momentum_40d']

for f in factors:
    if f in df.columns:
        non_zero = ((df[f] != 0) & df[f].notna()).sum()
        total_nan = df[f].isna().sum()
        total_zero = (df[f] == 0).sum()
        total_not_nan = df[f].notna().sum()
        
        print(f"\n{f}:")
        print(f"  Non-zero values: {non_zero:,} ({non_zero/len(df)*100:.2f}%)")
        print(f"  Zero values: {total_zero:,} ({total_zero/len(df)*100:.2f}%)")
        print(f"  NaN values: {total_nan:,} ({total_nan/len(df)*100:.2f}%)")
        print(f"  Non-NaN values: {total_not_nan:,} ({total_not_nan/len(df)*100:.2f}%)")
        
        if total_not_nan > 0:
            non_null_values = df[f].dropna()
            print(f"  Mean: {non_null_values.mean():.6f}")
            print(f"  Std: {non_null_values.std():.6f}")
            print(f"  Min: {non_null_values.min():.6f}")
            print(f"  Max: {non_null_values.max():.6f}")
            print(f"  Sample (first 10): {non_null_values.head(10).tolist()}")
        else:
            print(f"  [WARN] All values are NaN!")
