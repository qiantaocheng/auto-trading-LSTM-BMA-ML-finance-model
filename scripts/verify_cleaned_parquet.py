#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证parquet文件清理结果
"""

import pandas as pd
from pathlib import Path

files = [
    "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet",
    "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
]

print("=" * 80)
print("验证清理结果")
print("=" * 80)

for file_path in files:
    if not Path(file_path).exists():
        print(f"\n❌ File not found: {file_path}")
        continue
    
    df = pd.read_parquet(file_path)
    
    print(f"\n文件: {Path(file_path).name}")
    print(f"  列数: {len(df.columns)}")
    
    # 检查是否还有以_new结尾的列（备份列）
    new_cols = [c for c in df.columns if c.endswith('_new')]
    if new_cols:
        print(f"  [FAIL] 仍有_new结尾的列: {new_cols}")
    else:
        print(f"  [OK] 无_new结尾的列")
    
    # 检查是否还有roa/ebit
    bad_cols = [c for c in df.columns if c in ['roa', 'ebit', 'target_new', 'Close_new']]
    if bad_cols:
        print(f"  [FAIL] 仍有不需要的列: {bad_cols}")
    else:
        print(f"  [OK] 无不需要的列")
    
    # 列出所有列
    print(f"  所有列 ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
