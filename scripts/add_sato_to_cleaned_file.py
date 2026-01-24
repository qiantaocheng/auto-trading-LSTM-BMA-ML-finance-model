#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Sato factors to the existing polygon_factors_all_filtered_clean.parquet file.

This script:
1. Loads the existing cleaned parquet file
2. Computes Sato factors (momentum + divergence) if missing
3. Saves the updated file with Sato factors included
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # File paths
    data_dir = Path("data/factor_exports")
    input_file = data_dir / "polygon_factors_all_filtered_clean.parquet"
    backup_file = data_dir / f"polygon_factors_all_filtered_clean_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    
    logger.info("=" * 80)
    logger.info("Adding Sato Factors to Cleaned Parquet File")
    logger.info("=" * 80)
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return 1
    
    logger.info(f"📂 Loading data from: {input_file}")
    
    # Load data
    try:
        df = pd.read_parquet(input_file)
        logger.info(f"✅ Loaded data: {df.shape}")
        logger.info(f"   Index type: {type(df.index)}")
        if isinstance(df.index, pd.MultiIndex):
            logger.info(f"   Index names: {df.index.names}")
            dates = df.index.get_level_values('date').unique()
            tickers = df.index.get_level_values('ticker').unique()
            logger.info(f"   Date range: {pd.Timestamp(min(dates)).strftime('%Y-%m-%d')} to {pd.Timestamp(max(dates)).strftime('%Y-%m-%d')}")
            logger.info(f"   Tickers: {len(tickers)}")
        else:
            logger.warning("⚠️ Data is not MultiIndex format")
            return 1
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return 1
    
    # Check if Sato factors already exist
    has_sato_momentum = 'feat_sato_momentum_10d' in df.columns
    has_sato_divergence = 'feat_sato_divergence_10d' in df.columns
    
    if has_sato_momentum and has_sato_divergence:
        logger.info("✅ Sato factors already exist in the file")
        logger.info("   Checking if they need recalculation...")
        
        # Check if factors are all zeros (might need recalculation)
        momentum_all_zero = (df['feat_sato_momentum_10d'] == 0.0).all()
        divergence_all_zero = (df['feat_sato_divergence_10d'] == 0.0).all()
        
        if momentum_all_zero and divergence_all_zero:
            logger.warning("⚠️ Sato factors exist but are all zeros - will recalculate")
        else:
            logger.info("✅ Sato factors exist and have non-zero values")
            logger.info(f"   Momentum: min={df['feat_sato_momentum_10d'].min():.6f}, max={df['feat_sato_momentum_10d'].max():.6f}, mean={df['feat_sato_momentum_10d'].mean():.6f}")
            logger.info(f"   Divergence: min={df['feat_sato_divergence_10d'].min():.6f}, max={df['feat_sato_divergence_10d'].max():.6f}, mean={df['feat_sato_divergence_10d'].mean():.6f}")
            
            response = input("\nSato factors already exist. Recalculate anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("Skipping recalculation")
                return 0
    
    # Create backup
    logger.info(f"💾 Creating backup: {backup_file.name}")
    try:
        df.to_parquet(backup_file)
        logger.info("✅ Backup created")
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {e}")
        return 1
    
    # Compute Sato factors
    logger.info("\n🔥 Computing Sato Square Root Factors...")
    try:
        from scripts.sato_factor_calculation import calculate_sato_factors
        
        # Prepare data for Sato calculation
        sato_data = df.copy()
        
        # Ensure we have adj_close (use Close if not available)
        if 'adj_close' not in sato_data.columns:
            if 'Close' in sato_data.columns:
                sato_data['adj_close'] = sato_data['Close']
                logger.info("   Using 'Close' as 'adj_close'")
            else:
                logger.error("❌ No 'Close' or 'adj_close' column found")
                return 1
        
        # Check if vol_ratio_20d exists
        has_vol_ratio = 'vol_ratio_20d' in sato_data.columns
        
        # If Volume doesn't exist, estimate from vol_ratio_20d
        if 'Volume' not in sato_data.columns:
            if has_vol_ratio:
                base_volume = 1_000_000
                sato_data['Volume'] = base_volume * sato_data['vol_ratio_20d'].fillna(1.0).clip(lower=0.1, upper=10.0)
                use_vol_ratio = True
                logger.info("   Estimated 'Volume' from 'vol_ratio_20d'")
            else:
                logger.error("❌ No 'Volume' or 'vol_ratio_20d' column found")
                return 1
        else:
            use_vol_ratio = has_vol_ratio
        
        # Calculate Sato factors
        logger.info("   Computing factors (this may take a few minutes)...")
        sato_factors_df = calculate_sato_factors(
            df=sato_data,
            price_col='adj_close',
            volume_col='Volume',
            vol_ratio_col='vol_ratio_20d',
            lookback_days=10,
            vol_window=20,
            use_vol_ratio_directly=use_vol_ratio
        )
        
        logger.info("✅ Sato factors computed")
        logger.info(f"   Result shape: {sato_factors_df.shape}")
        logger.info(f"   Columns: {list(sato_factors_df.columns)}")
        
        # Add Sato factors to df
        df['feat_sato_momentum_10d'] = sato_factors_df['feat_sato_momentum_10d'].reindex(df.index).fillna(0.0)
        df['feat_sato_divergence_10d'] = sato_factors_df['feat_sato_divergence_10d'].reindex(df.index).fillna(0.0)
        
        # Verify factors were added
        logger.info("\n📊 Factor Statistics:")
        logger.info(f"   feat_sato_momentum_10d:")
        logger.info(f"      min={df['feat_sato_momentum_10d'].min():.6f}")
        logger.info(f"      max={df['feat_sato_momentum_10d'].max():.6f}")
        logger.info(f"      mean={df['feat_sato_momentum_10d'].mean():.6f}")
        logger.info(f"      std={df['feat_sato_momentum_10d'].std():.6f}")
        logger.info(f"      non-zero: {(df['feat_sato_momentum_10d'] != 0.0).sum():,} / {len(df):,}")
        
        logger.info(f"   feat_sato_divergence_10d:")
        logger.info(f"      min={df['feat_sato_divergence_10d'].min():.6f}")
        logger.info(f"      max={df['feat_sato_divergence_10d'].max():.6f}")
        logger.info(f"      mean={df['feat_sato_divergence_10d'].mean():.6f}")
        logger.info(f"      std={df['feat_sato_divergence_10d'].std():.6f}")
        logger.info(f"      non-zero: {(df['feat_sato_divergence_10d'] != 0.0).sum():,} / {len(df):,}")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute Sato factors: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save updated file
    logger.info(f"\n💾 Saving updated file: {input_file}")
    try:
        df.to_parquet(input_file)
        logger.info("✅ File saved successfully")
        
        # Verify file was saved correctly
        logger.info("\n🔍 Verifying saved file...")
        verify_df = pd.read_parquet(input_file)
        if 'feat_sato_momentum_10d' in verify_df.columns and 'feat_sato_divergence_10d' in verify_df.columns:
            logger.info("✅ Verification passed: Sato factors are in the saved file")
        else:
            logger.error("❌ Verification failed: Sato factors missing from saved file")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS: Sato factors added to cleaned parquet file")
    logger.info("=" * 80)
    logger.info(f"   Updated file: {input_file}")
    logger.info(f"   Backup file: {backup_file}")
    logger.info(f"   File modification date will be updated to: {datetime.now().strftime('%Y-%m-%d')}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
