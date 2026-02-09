#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalculate all factors using Simple17FactorEngine and update multiindex parquet file
This ensures all factors match the updated calculation methods.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine

def recalculate_and_update_multiindex(input_file: str, output_file: str = None,
                                      lookback_days: int = 120, horizon: int = 5):
    """
    Recalculate all factors using Simple17FactorEngine and update multiindex parquet file
    
    Args:
        input_file: Path to input multiindex parquet file
        output_file: Path to output file (default: adds '_recalculated' suffix)
        lookback_days: Lookback days for factor calculation (default: 120)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return False
    
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_recalculated{input_path.suffix}")
    
    print("=" * 80)
    print("Recalculate Factors and Update MultiIndex File")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Lookback days: {lookback_days}")
    print("=" * 80)
    
    # Step 1: Load existing data
    print("\n[STEP 1] Loading existing multiindex data...")
    try:
        df = pd.read_parquet(input_file)
        print(f"   Original shape: {df.shape}")
        print(f"   Original columns: {len(df.columns)}")
        print(f"   Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        
        if not isinstance(df.index, pd.MultiIndex):
            print("[ERROR] File does not have MultiIndex. Expected (date, ticker)")
            return False
        
        if df.index.names != ['date', 'ticker']:
            print(f"[WARN] Index names: {df.index.names}, expected ['date', 'ticker']")
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return False
    
    # Step 2: Prepare data for Simple17FactorEngine
    print("\n[STEP 2] Preparing data for Simple17FactorEngine...")
    try:
        # Reset index to get date and ticker as columns
        df_reset = df.reset_index()
        
        # Ensure required columns exist
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df_reset.columns]
        if missing_cols:
            # Try alternative column names
            col_mapping = {
                'close': 'Close',
                'Close': 'Close',
            }
            for alt_col in missing_cols:
                for key, val in col_mapping.items():
                    if key in df_reset.columns and val not in df_reset.columns:
                        df_reset[val] = df_reset[key]
                        print(f"   Mapped {key} -> {val}")
        
        # Check again for Close
        if 'Close' not in df_reset.columns:
            print(f"[ERROR] Missing required column: Close")
            print(f"   Available columns: {list(df_reset.columns)}")
            return False
        
        # Handle Volume column - estimate from vol_ratio if available
        if 'Volume' not in df_reset.columns:
            print("   [WARN] Volume column not found, attempting to estimate...")
            # Try to estimate Volume from vol_ratio_20d if available
            if 'vol_ratio_20d' in df_reset.columns:
                print("   [INFO] Estimating Volume from vol_ratio_20d...")
                # Estimate: Volume = base_volume * (1 + vol_ratio_20d)
                # Use a reasonable base volume (e.g., 1M shares)
                base_volume = 1_000_000
                df_reset['Volume'] = base_volume * (1 + df_reset['vol_ratio_20d'].fillna(0).clip(-0.9, 10))
                print("   [INFO] Volume estimated from vol_ratio_20d")
            else:
                # Use a constant default volume (not ideal, but allows calculation to proceed)
                print("   [WARN] No vol_ratio available, using default volume (1M shares)")
                df_reset['Volume'] = 1_000_000
        else:
            # Volume exists, check for alternative names
            if 'volume' in df_reset.columns and 'Volume' not in df_reset.columns:
                df_reset['Volume'] = df_reset['volume']
                print("   Mapped volume -> Volume")
        
        # Ensure date is datetime
        df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None).dt.normalize()
        
        # Sort by date and ticker
        df_reset = df_reset.sort_values(['date', 'ticker'])
        
        print(f"   Prepared data shape: {df_reset.shape}")
        print(f"   Date range: {df_reset['date'].min()} to {df_reset['date'].max()}")
        print(f"   Unique tickers: {df_reset['ticker'].nunique()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Initialize Simple17FactorEngine
    print("\n[STEP 3] Initializing Simple17FactorEngine...")
    try:
        engine = Simple17FactorEngine(
            lookback_days=lookback_days,
            mode='predict',  # Use predict mode to compute all factors
            horizon=horizon
        )
        print(f"   Engine initialized with horizon: {engine.horizon}")
        print(f"   Alpha factors: {len(engine.alpha_factors)} factors")
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Recalculate factors using Simple17FactorEngine
    print("\n[STEP 4] Recalculating factors using Simple17FactorEngine...")
    print("   This may take a while for large datasets...")
    
    try:
        # Prepare market_data format for Simple17FactorEngine
        # Need: date, ticker, Close, Volume, High, Low (if available)
        market_data = df_reset[['date', 'ticker', 'Close']].copy()
        
        # Ensure Volume exists (should have been handled above)
        if 'Volume' in df_reset.columns:
            market_data['Volume'] = df_reset['Volume']
        else:
            print("   [ERROR] Volume column still missing after estimation")
            return False
        
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
        
        print(f"   Market data prepared: {market_data.shape}")
        print(f"   Computing factors for all dates at once...")
        
        # Compute all factors using Simple17FactorEngine
        # This will process all dates together (more efficient)
        factors_df = engine.compute_all_17_factors(market_data, mode='predict')
        
        print(f"   Factors computed: {factors_df.shape}")
        
        # Ensure factors_df has MultiIndex
        if not isinstance(factors_df.index, pd.MultiIndex):
            if 'date' in factors_df.columns and 'ticker' in factors_df.columns:
                factors_df = factors_df.set_index(['date', 'ticker'])
            else:
                # Try to align with market_data
                factors_df.index = pd.MultiIndex.from_arrays(
                    [market_data['date'], market_data['ticker']],
                    names=['date', 'ticker']
                )
        
        target_cols = [col for col in ['target', 'target_excess_qqq'] if col in factors_df.columns]

        # Remove Close column if present (we'll keep original)
        if 'Close' in factors_df.columns:
            factors_df = factors_df.drop(columns=['Close'])
        
        print(f"   Final factors shape: {factors_df.shape}")
        print(f"   Factor columns: {list(factors_df.columns)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to recalculate factors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Merge factors back into original dataframe
    print("\n[STEP 6] Merging recalculated factors into original data...")
    try:
        # Keep original Close and Volume (and other non-factor columns)
        non_factor_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        non_factor_cols = [col for col in non_factor_cols if col in df.columns]
        
        # Get factor columns from engine
        expected_factors = engine.alpha_factors
        
        # Create new dataframe with original structure
        df_new = df[non_factor_cols].copy()
        
        # Align factors_df index with df_new index
        factors_df_aligned = factors_df.reindex(df_new.index)
        
        # Add recalculated factors
        for factor in expected_factors:
            if factor in factors_df_aligned.columns:
                df_new[factor] = factors_df_aligned[factor]
                print(f"   Added factor: {factor}")
            else:
                print(f"   [WARN] Factor {factor} not found in recalculated factors")
        
        # Fill missing factors with zeros
        for factor in expected_factors:
            if factor not in df_new.columns:
                df_new[factor] = 0.0
                print(f"   [WARN] Filled missing factor {factor} with zeros")
        
        # Add target-style columns if available
        for tgt in target_cols:
            if tgt in factors_df_aligned.columns:
                df_new[tgt] = factors_df_aligned[tgt]
                print(f"   Added column: {tgt}")

        # Keep any other columns from original that aren't factors
        # But exclude old factors that are being replaced
        old_factors_to_remove = [
            'downside_beta_252',
            'downside_beta_ewm_21',
            'rsrs_beta_18',
            'ret_skew_20d',
            'bollinger_squeeze',
            'feat_vol_price_div_30d',
            'obv_momentum_40d',
            'vol_ratio_30d',
            'ret_skew_30d',
            'ivol_30',
            'blowoff_ratio_30d',
            'blowoff_ratio',
            'roa',
            'ebit',
            'price_ma60_deviation',
            'making_new_low_5d',
        ]
        
        other_cols = [col for col in df.columns 
                     if col not in non_factor_cols 
                     and col not in expected_factors
                     and col not in old_factors_to_remove]  # Exclude old factors
        
        for col in other_cols:
            if col not in df_new.columns:
                df_new[col] = df[col]
                print(f"   Kept column: {col}")
        
        # Explicitly remove old factors (if they exist)
        for old_factor in old_factors_to_remove:
            if old_factor in df_new.columns:
                df_new = df_new.drop(columns=[old_factor])
                print(f"   Removed old factor: {old_factor}")
        
        print(f"   Final shape: {df_new.shape}")
        print(f"   Final columns: {len(df_new.columns)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to merge factors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Save updated file
    print("\n[STEP 7] Saving updated file...")
    try:
        df_new.to_parquet(output_file, index=True)
        print(f"   [SUCCESS] File saved: {output_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"Removed factors (legacy/deleted):")
        print("  - downside_beta_252, downside_beta_ewm_21, rsrs_beta_18, ret_skew_20d, bollinger_squeeze")
        print("  - roa, ebit, price_ma60_deviation, making_new_low_5d (if present)")
        print(f"\nCurrent factor set ({len(expected_factors)} factors):")
        for i, factor in enumerate(expected_factors, 1):
            print(f"  {i:2d}. {factor}")
        print(f"\nAll factors recalculated using Simple17FactorEngine")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Recalculate factors using Simple17FactorEngine')
    parser.add_argument('--input', type=str, 
                       default="D:/trade/data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet",
                       help='Input multiindex parquet file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: adds _recalculated suffix)')
    parser.add_argument('--lookback', type=int, default=120,
                       help='Lookback days for factor calculation (default: 120)')
    parser.add_argument('--horizon', type=int, default=5,
                       help='Forward horizon (days) for target/excess calculation (default: 5)')
    parser.add_argument('--yes', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Factor Recalculation Script")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Load the multiindex parquet file")
    print("2. Recalculate ALL factors using Simple17FactorEngine")
    print("3. Update factor names (remove old, add new)")
    print("4. Save updated file")
    print("\nWARNING: This will recalculate factors for the entire dataset.")
    print("This may take a significant amount of time for large datasets.")
    print("=" * 80)
    
    if not args.yes:
        try:
            response = input("\nContinue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Aborted.")
                sys.exit(0)
        except EOFError:
            print("\n[INFO] Non-interactive mode, proceeding...")
    
    success = recalculate_and_update_multiindex(
        args.input,
        args.output,
        lookback_days=args.lookback,
        horizon=args.horizon,
    )
    
    if success:
        print("\n[SUCCESS] Factor recalculation completed!")
    else:
        print("\n[ERROR] Factor recalculation failed!")
        sys.exit(1)
