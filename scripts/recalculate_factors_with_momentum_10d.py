#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalculate all factors including NEW momentum_10d factor and update multiindex parquet file.
This script ensures momentum_10d is calculated and added to the dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine, T10_ALPHA_FACTORS

def recalculate_with_momentum_10d(
    input_file: str,
    output_file: str = None,
    lookback_days: int = 120,
    backup: bool = True
):
    """
    Recalculate all factors including momentum_10d and update multiindex parquet file
    
    Args:
        input_file: Path to input multiindex parquet file
        output_file: Path to output file (default: overwrites input with backup)
        lookback_days: Lookback days for factor calculation (default: 120)
        backup: Whether to create backup of original file (default: True)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return False
    
    if output_file is None:
        output_file = str(input_path)
    
    print("=" * 80)
    print("Recalculate Factors with momentum_10d - Update MultiIndex File")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Lookback days: {lookback_days}")
    print(f"Expected factors: {len(T10_ALPHA_FACTORS)} factors")
    print(f"Factor list: {T10_ALPHA_FACTORS}")
    print("=" * 80)
    
    # Step 1: Backup original file if requested
    if backup and input_file == output_file:
        backup_file = str(input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{input_path.suffix}")
        print(f"\n[STEP 0] Creating backup: {backup_file}")
        try:
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"   ✅ Backup created successfully")
        except Exception as e:
            print(f"   ⚠️ Backup failed: {e}")
            response = input("   Continue without backup? (y/n): ")
            if response.lower() != 'y':
                return False
    
    # Step 2: Load existing data
    print("\n[STEP 1] Loading existing multiindex data...")
    try:
        df = pd.read_parquet(input_file)
        print(f"   ✅ Original shape: {df.shape}")
        print(f"   ✅ Original columns: {len(df.columns)}")
        print(f"   ✅ Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        
        if not isinstance(df.index, pd.MultiIndex):
            print("[ERROR] File does not have MultiIndex. Expected (date, ticker)")
            return False
        
        if df.index.names != ['date', 'ticker']:
            print(f"[WARN] Index names: {df.index.names}, expected ['date', 'ticker']")
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Prepare data for Simple17FactorEngine
    print("\n[STEP 2] Preparing data for Simple17FactorEngine...")
    try:
        # Reset index to get date and ticker as columns
        df_reset = df.reset_index()
        
        # Ensure required columns exist
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df_reset.columns]
        if missing_cols:
            print(f"[ERROR] Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(df_reset.columns)}")
            return False
        
        # Handle Volume column
        if 'Volume' not in df_reset.columns:
            print("   [WARN] Volume column not found, attempting to estimate...")
            if 'vol_ratio_30d' in df_reset.columns:
                print("   [INFO] Estimating Volume from vol_ratio_30d...")
                base_volume = 1_000_000
                df_reset['Volume'] = base_volume * (1 + df_reset['vol_ratio_30d'].fillna(0).clip(-0.9, 10))
            else:
                print("   [WARN] Using default volume (1M shares)")
                df_reset['Volume'] = 1_000_000
        
        # Ensure date is datetime
        df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None).dt.normalize()
        
        # Sort by date and ticker
        df_reset = df_reset.sort_values(['date', 'ticker'])
        
        print(f"   ✅ Prepared data shape: {df_reset.shape}")
        print(f"   ✅ Date range: {df_reset['date'].min()} to {df_reset['date'].max()}")
        print(f"   ✅ Unique tickers: {df_reset['ticker'].nunique()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Initialize Simple17FactorEngine
    print("\n[STEP 3] Initializing Simple17FactorEngine...")
    try:
        engine = Simple17FactorEngine(
            lookback_days=lookback_days,
            mode='predict',
            horizon=10  # T+10 horizon
        )
        print(f"   ✅ Engine initialized with horizon: {engine.horizon}")
        print(f"   ✅ Alpha factors: {len(engine.alpha_factors)} factors")
        print(f"   ✅ Factor list: {engine.alpha_factors}")
        
        # Verify momentum_10d is in the list
        if 'momentum_10d' not in engine.alpha_factors:
            print(f"   ⚠️ WARNING: momentum_10d not in alpha_factors list!")
            print(f"   ⚠️ This may indicate the code needs to be updated.")
        else:
            print(f"   ✅ momentum_10d is in alpha_factors list")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Recalculate factors
    print("\n[STEP 4] Recalculating factors using Simple17FactorEngine...")
    print("   This may take a while for large datasets...")
    
    try:
        # Prepare market_data format
        market_data = df_reset[['date', 'ticker', 'Close']].copy()
        
        # Add Volume
        if 'Volume' in df_reset.columns:
            market_data['Volume'] = df_reset['Volume']
        
        # Add High and Low if available
        for col in ['High', 'Low', 'Open']:
            if col in df_reset.columns:
                market_data[col] = df_reset[col]
            elif col.lower() in df_reset.columns:
                market_data[col] = df_reset[col.lower()]
        
        # Fill missing High/Low with Close if needed
        if 'High' not in market_data.columns:
            market_data['High'] = market_data['Close']
        if 'Low' not in market_data.columns:
            market_data['Low'] = market_data['Close']
        if 'Open' not in market_data.columns:
            market_data['Open'] = market_data['Close']
        
        print(f"   ✅ Market data prepared: {market_data.shape}")
        print(f"   ✅ Computing factors for all dates...")
        
        # Compute all factors
        factors_df = engine.compute_all_17_factors(market_data, mode='predict')
        
        print(f"   ✅ Factors computed: {factors_df.shape}")
        print(f"   ✅ Factor columns: {list(factors_df.columns)}")
        
        # Verify momentum_10d was computed
        if 'momentum_10d' in factors_df.columns:
            print(f"   ✅ momentum_10d computed successfully")
            print(f"      Coverage: {(factors_df['momentum_10d'] != 0).sum() / len(factors_df) * 100:.1f}%")
            print(f"      Mean: {factors_df['momentum_10d'].mean():.4f}")
        else:
            print(f"   ⚠️ WARNING: momentum_10d not found in computed factors!")
            print(f"   ⚠️ Available factors: {list(factors_df.columns)}")
        
        # Ensure factors_df has MultiIndex
        if not isinstance(factors_df.index, pd.MultiIndex):
            if 'date' in factors_df.columns and 'ticker' in factors_df.columns:
                factors_df = factors_df.set_index(['date', 'ticker'])
            else:
                factors_df.index = pd.MultiIndex.from_arrays(
                    [market_data['date'], market_data['ticker']],
                    names=['date', 'ticker']
                )
        
        # Remove Close column if present (keep original)
        if 'Close' in factors_df.columns:
            factors_df = factors_df.drop(columns=['Close'])
        
    except Exception as e:
        print(f"[ERROR] Failed to recalculate factors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Merge factors back into original dataframe
    print("\n[STEP 5] Merging recalculated factors into original data...")
    try:
        # Keep original non-factor columns
        non_factor_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        non_factor_cols = [col for col in non_factor_cols if col in df.columns]
        
        # Get expected factors from engine
        expected_factors = engine.alpha_factors
        
        # Create new dataframe
        df_new = df[non_factor_cols].copy()
        
        # Align factors_df index with df_new index
        factors_df_aligned = factors_df.reindex(df_new.index)
        
        # Add recalculated factors
        added_factors = []
        missing_factors = []
        for factor in expected_factors:
            if factor in factors_df_aligned.columns:
                df_new[factor] = factors_df_aligned[factor]
                added_factors.append(factor)
            else:
                missing_factors.append(factor)
        
        print(f"   ✅ Added {len(added_factors)} factors")
        if missing_factors:
            print(f"   ⚠️ Missing factors: {missing_factors}")
        
        # Fill missing factors with zeros (should not happen, but safety check)
        for factor in missing_factors:
            df_new[factor] = 0.0
            print(f"   ⚠️ Filled missing factor {factor} with zeros")
        
        # Keep other non-factor columns from original
        old_factors_to_remove = [
            'obv_divergence',
            'feat_sato_momentum_10d',
            'feat_sato_divergence_10d',
            'vol_ratio_20d',
            'ret_skew_20d',
            'ivol_20',
            'blowoff_ratio',
        ]
        
        other_cols = [col for col in df.columns 
                     if col not in non_factor_cols 
                     and col not in expected_factors
                     and col not in old_factors_to_remove]
        
        for col in other_cols:
            if col not in df_new.columns:
                df_new[col] = df[col]
        
        # Remove old factors explicitly
        for old_factor in old_factors_to_remove:
            if old_factor in df_new.columns:
                df_new = df_new.drop(columns=[old_factor])
                print(f"   ✅ Removed old factor: {old_factor}")
        
        print(f"   ✅ Final dataframe shape: {df_new.shape}")
        print(f"   ✅ Final columns: {len(df_new.columns)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to merge factors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Save updated file
    print("\n[STEP 6] Saving updated file...")
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_new.to_parquet(output_file, compression='snappy', index=True)
        
        print(f"   ✅ File saved: {output_file}")
        print(f"   ✅ Final shape: {df_new.shape}")
        print(f"   ✅ Factor columns: {len([c for c in df_new.columns if c in expected_factors])}")
        
        # Verify momentum_10d is in saved file
        verify_df = pd.read_parquet(output_file)
        if 'momentum_10d' in verify_df.columns:
            print(f"   ✅ Verified: momentum_10d is in saved file")
            print(f"      Coverage: {(verify_df['momentum_10d'] != 0).sum() / len(verify_df) * 100:.1f}%")
        else:
            print(f"   ⚠️ WARNING: momentum_10d not found in saved file!")
        
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ Recalculation completed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate factors with momentum_10d")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/factor_exports/polygon_factors_all_filtered_clean.parquet",
        help="Input multiindex parquet file path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: overwrites input with backup)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Lookback days for factor calculation (default: 120)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup of original file"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive mode (skip confirmations)"
    )
    
    args = parser.parse_args()
    
    success = recalculate_with_momentum_10d(
        input_file=args.input_file,
        output_file=args.output_file,
        lookback_days=args.lookback_days,
        backup=not args.no_backup
    )
    
    sys.exit(0 if success else 1)
