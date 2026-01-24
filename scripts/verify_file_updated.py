#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify main file has been updated with new factors
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS

main_file = "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet"

print("=" * 80)
print("Main File Verification")
print("=" * 80)

df = pd.read_parquet(main_file)

print(f"\nFile: {main_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

print(f"\nNew factors present:")
new_factors = ['obv_momentum_40d', 'feat_vol_price_div_30d', 'vol_ratio_30d', 
               'ret_skew_30d', 'ivol_30', 'blowoff_ratio_30d']
for f in new_factors:
    status = "[YES]" if f in df.columns else "[NO]"
    print(f"  {status} {f}")

print(f"\nOld factors removed:")
old_factors = ['obv_divergence', 'feat_sato_momentum_10d', 'feat_sato_divergence_10d',
               'vol_ratio_20d', 'ret_skew_20d', 'ivol_20', 'blowoff_ratio']
for f in old_factors:
    status = "[REMOVED]" if f not in df.columns else "[STILL PRESENT]"
    print(f"  {status} {f}")

print(f"\nT10_ALPHA_FACTORS check:")
missing = [f for f in T10_ALPHA_FACTORS if f not in df.columns]
if missing:
    print(f"  [MISSING] {missing}")
else:
    print("  [OK] All T10_ALPHA_FACTORS present")

print("\n" + "=" * 80)
