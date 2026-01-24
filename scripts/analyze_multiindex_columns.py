#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze multiindex parquet file columns to identify:
1. Old factors (to be replaced)
2. Unused columns (not factors, not market data)
3. Required columns (to keep)
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS

def analyze_columns(input_file: str):
    """Analyze columns in multiindex file"""
    
    print("=" * 80)
    print("MultiIndex File Column Analysis")
    print("=" * 80)
    
    # Load file
    df = pd.read_parquet(input_file)
    all_columns = sorted(df.columns)
    
    print(f"\nFile: {input_file}")
    print(f"Shape: {df.shape}")
    print(f"Total columns: {len(all_columns)}")
    
    # Categorize columns
    required_market_data = ['Close', 'Volume', 'High', 'Low', 'Open']
    new_factors = list(T10_ALPHA_FACTORS)
    
    old_factors = [
        'obv_divergence',  # Replaced by obv_momentum_40d
        'feat_sato_momentum_10d',  # Replaced by feat_vol_price_div_30d
        'feat_sato_divergence_10d',  # Replaced by feat_vol_price_div_30d
        'vol_ratio_20d',  # Replaced by vol_ratio_30d
        'ret_skew_20d',  # Replaced by ret_skew_30d
        'ivol_20',  # Replaced by ivol_30
        'blowoff_ratio',  # Replaced by blowoff_ratio_30d
    ]
    
    # Categorize each column
    categories = {
        'required_market_data': [],
        'new_factors': [],
        'old_factors': [],
        'other_factors': [],
        'metadata': [],
        'unknown': []
    }
    
    for col in all_columns:
        if col in required_market_data:
            categories['required_market_data'].append(col)
        elif col in new_factors:
            categories['new_factors'].append(col)
        elif col in old_factors:
            categories['old_factors'].append(col)
        elif col in ['target', 'date', 'ticker']:
            categories['metadata'].append(col)
        elif any(x in col.lower() for x in ['momentum', 'vol', 'rsi', 'beta', 'ratio', 'skew', 'divergence', 'sato', 'feat']):
            categories['other_factors'].append(col)
        else:
            categories['unknown'].append(col)
    
    # Print analysis
    print("\n" + "=" * 80)
    print("Column Categories:")
    print("=" * 80)
    
    print(f"\n1. Required Market Data ({len(categories['required_market_data'])}):")
    for col in categories['required_market_data']:
        print(f"   - {col}")
    
    print(f"\n2. New Factors (T10_ALPHA_FACTORS) ({len(categories['new_factors'])}):")
    for col in categories['new_factors']:
        print(f"   - {col}")
    
    print(f"\n3. Old Factors (TO BE REPLACED) ({len(categories['old_factors'])}):")
    for col in categories['old_factors']:
        replacement = {
            'obv_divergence': 'obv_momentum_40d',
            'feat_sato_momentum_10d': 'feat_vol_price_div_30d',
            'feat_sato_divergence_10d': 'feat_vol_price_div_30d',
            'vol_ratio_20d': 'vol_ratio_30d',
            'ret_skew_20d': 'ret_skew_30d',
            'ivol_20': 'ivol_30',
            'blowoff_ratio': 'blowoff_ratio_30d',
        }.get(col, '?')
        print(f"   - {col} -> {replacement}")
    
    print(f"\n4. Other Factors (NOT IN T10_ALPHA_FACTORS) ({len(categories['other_factors'])}):")
    for col in categories['other_factors']:
        in_new = '[YES]' if col in new_factors else '[NO]'
        print(f"   {in_new} {col}")
    
    print(f"\n5. Metadata ({len(categories['metadata'])}):")
    for col in categories['metadata']:
        print(f"   - {col}")
    
    print(f"\n6. Unknown Columns ({len(categories['unknown'])}):")
    for col in categories['unknown']:
        print(f"   - {col}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total columns: {len(all_columns)}")
    print(f"Required market data: {len(categories['required_market_data'])}")
    print(f"New factors (will be added): {len(new_factors)}")
    print(f"Old factors (will be replaced): {len(categories['old_factors'])}")
    print(f"Other factors (not in T10_ALPHA_FACTORS): {len(categories['other_factors'])}")
    print(f"Metadata: {len(categories['metadata'])}")
    print(f"Unknown: {len(categories['unknown'])}")
    
    # Questions for user
    print("\n" + "=" * 80)
    print("Questions for User:")
    print("=" * 80)
    print("\n1. Old Factors - Should these be DELETED after replacement?")
    for col in categories['old_factors']:
        print(f"   - {col}")
    
    print("\n2. Other Factors - Should these be KEPT or DELETED?")
    for col in categories['other_factors']:
        print(f"   - {col}")
    
    print("\n3. Unknown Columns - Should these be KEPT or DELETED?")
    for col in categories['unknown']:
        print(f"   - {col}")
    
    print("\n" + "=" * 80)
    
    return categories

if __name__ == "__main__":
    input_file = "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet"
    analyze_columns(input_file)
