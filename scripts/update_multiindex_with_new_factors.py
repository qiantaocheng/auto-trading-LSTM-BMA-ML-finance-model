#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update multiindex parquet file with new factors:
- Replace feat_sato_momentum_10d and feat_sato_divergence_10d with feat_vol_price_div_30d
- Update factor names: vol_ratio_20d→30d, ret_skew_20d→30d, ivol_20→30, blowoff_ratio→30d
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine

def add_robust_divergence_feature(df):
    """
    计算基于Rank归一化的30天价量背离因子。
    
    逻辑:
    1. 计算价格动量 (30d Return)
    2. 计算成交量趋势 (MA10 vs MA30)
    3. 将两者在当天截面上转化为 Rank (0~1)
    4. 背离 = Price_Rank - Volume_Rank
    
    Args:
        df: MultiIndex (date, ticker) DataFrame with 'close' and 'volume' columns
        
    Returns:
        df: 新增列 'feat_vol_price_div_30d'
    """
    work_df = df.copy()
    
    # 1. 计算原始指标 (Time-Series Step)
    # A. 价格变化 (30天动量)
    work_df['raw_price_chg'] = work_df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change(periods=30)
    )
    
    # B. 成交量变化 (趋势强度: 短期均量 vs 长期均量)
    def calc_vol_trend(x):
        ma10 = x.rolling(window=10, min_periods=5).mean()
        ma30 = x.rolling(window=30, min_periods=15).mean()
        return (ma10 - ma30) / (ma30 + 1e-6)
    
    work_df['raw_vol_chg'] = work_df.groupby('ticker')['volume'].transform(calc_vol_trend)
    
    # 2. 横截面 Rank 归一化 (Cross-Sectional Step)
    if isinstance(work_df.index, pd.MultiIndex):
        groupby_col = work_df.index.get_level_values('date')
    else:
        groupby_col = work_df['date'] if 'date' in work_df.columns else work_df.index
    
    work_df['rank_price'] = work_df.groupby(groupby_col)['raw_price_chg'].rank(pct=True)
    work_df['rank_vol'] = work_df.groupby(groupby_col)['raw_vol_chg'].rank(pct=True)
    
    # 3. 计算最终背离因子
    df['feat_vol_price_div_30d'] = (work_df['rank_price'] - work_df['rank_vol']).fillna(0)
    
    return df

def update_multiindex_file(input_file: str, output_file: str = None):
    """
    Update multiindex parquet file with new factors
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return
    
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_updated{input_path.suffix}")
    
    print(f"[INFO] Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"[INFO] Original shape: {df.shape}")
    print(f"[INFO] Original columns: {len(df.columns)}")
    
    # Check if MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        print("[ERROR] File does not have MultiIndex. Expected (date, ticker)")
        return
    
    # Rename columns to match expected format (lowercase)
    column_mapping = {}
    if 'Close' in df.columns:
        column_mapping['Close'] = 'close'
    if 'Volume' in df.columns:
        column_mapping['Volume'] = 'volume'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Ensure we have close and volume
    if 'close' not in df.columns:
        print("[ERROR] Missing 'close' column")
        return
    if 'volume' not in df.columns:
        print("[ERROR] Missing 'volume' column")
        return
    
    # Remove old Sato factors if they exist
    old_sato_cols = ['feat_sato_momentum_10d', 'feat_sato_divergence_10d']
    for col in old_sato_cols:
        if col in df.columns:
            print(f"[INFO] Removing old factor: {col}")
            df = df.drop(columns=[col])
    
    # Rename old factor columns to new names
    rename_map = {
        'vol_ratio_20d': 'vol_ratio_30d',
        'ret_skew_20d': 'ret_skew_30d',
        'ivol_20': 'ivol_30',
        'blowoff_ratio': 'blowoff_ratio_30d',
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            print(f"[INFO] Renaming {old_name} -> {new_name}")
            df = df.rename(columns={old_name: new_name})
    
    # Add new divergence factor
    if 'feat_vol_price_div_30d' not in df.columns:
        print("[INFO] Computing feat_vol_price_div_30d...")
        df = add_robust_divergence_feature(df)
    else:
        print("[INFO] feat_vol_price_div_30d already exists, recomputing...")
        df = df.drop(columns=['feat_vol_price_div_30d'])
        df = add_robust_divergence_feature(df)
    
    # Restore original column names if needed
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    if reverse_mapping:
        df = df.rename(columns=reverse_mapping)
    
    print(f"[INFO] Updated shape: {df.shape}")
    print(f"[INFO] Updated columns: {len(df.columns)}")
    
    # Save updated file
    print(f"[INFO] Saving to {output_file}...")
    df.to_parquet(output_file, index=True)
    print(f"[SUCCESS] File saved: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Removed factors: {old_sato_cols}")
    print(f"Renamed factors: {list(rename_map.keys())} -> {list(rename_map.values())}")
    print(f"Added factor: feat_vol_price_div_30d")
    print("=" * 80)

if __name__ == "__main__":
    input_file = "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet"
    update_multiindex_file(input_file)
