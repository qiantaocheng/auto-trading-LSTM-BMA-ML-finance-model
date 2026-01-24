#!/usr/bin/env python3
import pandas as pd
import sys

data_file = r'D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet'

print("Checking data file columns...")
# Read just first row to get column names
df = pd.read_parquet(data_file)
cols = list(df.columns)
df = None  # Free memory

print(f"\nTotal columns: {len(cols)}")
print(f"\nret_skew_20d in data: {'ret_skew_20d' in cols}")
print(f"ret_skew_30d in data: {'ret_skew_30d' in cols}")

print("\nAll columns:")
for i, c in enumerate(cols, 1):
    print(f"{i:2d}. {c}")

if 'ret_skew_20d' in cols:
    print("\n⚠️ WARNING: ret_skew_20d still exists in data file!")
    print("   This means the model might still be using it during training!")
