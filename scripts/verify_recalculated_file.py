#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify recalculated multiindex file
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS

output_file = "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet"

print("=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

df = pd.read_parquet(output_file)

print(f"\nShape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
print(f"Index: {df.index.names}")

print(f"\nT10_ALPHA_FACTORS ({len(T10_ALPHA_FACTORS)} factors):")
missing = [f for f in T10_ALPHA_FACTORS if f not in df.columns]
if missing:
    print(f"  MISSING: {missing}")
else:
    print("  [OK] All T10_ALPHA_FACTORS present")

print(f"\nAll columns ({len(df.columns)}):")
for i, col in enumerate(sorted(df.columns), 1):
    marker = ""
    if col in T10_ALPHA_FACTORS:
        marker = "[T10]"
    elif col in ['target', 'Close']:
        marker = "[DATA]"
    else:
        marker = "[OTHER]"
    print(f"  {marker} {i:2d}. {col}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"Total columns: {len(df.columns)}")
print(f"T10_ALPHA_FACTORS: {len([c for c in df.columns if c in T10_ALPHA_FACTORS])}")
print(f"Other columns: {len([c for c in df.columns if c not in T10_ALPHA_FACTORS and c not in ['target', 'Close']])}")
print("=" * 80)
