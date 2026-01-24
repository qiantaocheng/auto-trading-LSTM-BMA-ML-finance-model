#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the two data files to see if they're the same"""

import pandas as pd
from pathlib import Path

f1 = Path("data/factor_exports/polygon_factors_all_filtered.parquet")
f2 = Path("data/factor_exports/polygon_factors_all_filtered_clean.parquet")

print("=" * 80)
print("Data File Comparison")
print("=" * 80)

# File 1
print(f"\n1. {f1.name}")
print(f"   Exists: {f1.exists()}")
if f1.exists():
    df1 = pd.read_parquet(f1)
    if isinstance(df1.index, pd.MultiIndex):
        dates1 = df1.index.get_level_values('date').unique()
        tickers1 = df1.index.get_level_values('ticker').unique()
    else:
        dates1 = df1['date'].unique() if 'date' in df1.columns else []
        tickers1 = df1['ticker'].unique() if 'ticker' in df1.columns else []
    
    print(f"   Dates: {len(dates1)}")
    if len(dates1) > 0:
        print(f"   Date range: {pd.Timestamp(min(dates1)).strftime('%Y-%m-%d')} to {pd.Timestamp(max(dates1)).strftime('%Y-%m-%d')}")
    print(f"   Tickers: {len(tickers1)}")
    print(f"   Shape: {df1.shape}")

# File 2
print(f"\n2. {f2.name}")
print(f"   Exists: {f2.exists()}")
if f2.exists():
    df2 = pd.read_parquet(f2)
    if isinstance(df2.index, pd.MultiIndex):
        dates2 = df2.index.get_level_values('date').unique()
        tickers2 = df2.index.get_level_values('ticker').unique()
    else:
        dates2 = df2['date'].unique() if 'date' in df2.columns else []
        tickers2 = df2['ticker'].unique() if 'ticker' in df2.columns else []
    
    print(f"   Dates: {len(dates2)}")
    if len(dates2) > 0:
        print(f"   Date range: {pd.Timestamp(min(dates2)).strftime('%Y-%m-%d')} to {pd.Timestamp(max(dates2)).strftime('%Y-%m-%d')}")
    print(f"   Tickers: {len(tickers2)}")
    print(f"   Shape: {df2.shape}")

# Comparison
if f1.exists() and f2.exists():
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    
    same_dates = len(dates1) == len(dates2) and (len(dates1) == 0 or (min(dates1) == min(dates2) and max(dates1) == max(dates2)))
    same_tickers = len(tickers1) == len(tickers2)
    same_shape = df1.shape == df2.shape
    
    print(f"Same date range: {same_dates}")
    print(f"Same number of tickers: {same_tickers}")
    print(f"Same shape: {same_shape}")
    
    if same_dates and same_tickers and same_shape:
        print("\n✅ Files appear to have the same data structure")
    else:
        print("\n⚠️ Files have different data")
