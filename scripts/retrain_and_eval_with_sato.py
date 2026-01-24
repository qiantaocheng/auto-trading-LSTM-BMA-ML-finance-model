#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain models with Sato factors and run 80/20 time split evaluation.

This script:
1. Verifies Sato factors are in the cleaned parquet file
2. Trains models with full dataset (including Sato factors)
3. Runs 80/20 time split evaluation
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def verify_sato_factors():
    """Verify Sato factors are in the cleaned parquet file"""
    logger.info("=" * 80)
    logger.info("Step 1: Verifying Sato Factors in Cleaned File")
    logger.info("=" * 80)
    
    data_file = Path("data/factor_exports/polygon_factors_all_filtered_clean.parquet")
    
    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return False
    
    logger.info(f"📂 Loading: {data_file}")
    df = pd.read_parquet(data_file)
    
    logger.info(f"✅ Data loaded: {df.shape}")
    logger.info(f"   Index type: {type(df.index)}")
    
    if isinstance(df.index, pd.MultiIndex):
        dates = df.index.get_level_values('date').unique()
        tickers = df.index.get_level_values('ticker').unique()
        logger.info(f"   Date range: {pd.Timestamp(min(dates)).strftime('%Y-%m-%d')} to {pd.Timestamp(max(dates)).strftime('%Y-%m-%d')}")
        logger.info(f"   Tickers: {len(tickers)}")
    
    # Check for Sato factors
    has_momentum = 'feat_sato_momentum_10d' in df.columns
    has_divergence = 'feat_sato_divergence_10d' in df.columns
    
    logger.info(f"\n📊 Sato Factor Check:")
    logger.info(f"   feat_sato_momentum_10d: {'✅' if has_momentum else '❌'}")
    logger.info(f"   feat_sato_divergence_10d: {'✅' if has_divergence else '❌'}")
    
    if has_momentum and has_divergence:
        logger.info(f"\n📈 Sato Factor Statistics:")
        logger.info(f"   Momentum: min={df['feat_sato_momentum_10d'].min():.6f}, max={df['feat_sato_momentum_10d'].max():.6f}, mean={df['feat_sato_momentum_10d'].mean():.6f}")
        logger.info(f"   Divergence: min={df['feat_sato_divergence_10d'].min():.6f}, max={df['feat_sato_divergence_10d'].max():.6f}, mean={df['feat_sato_divergence_10d'].mean():.6f}")
        logger.info(f"   Non-zero momentum: {(df['feat_sato_momentum_10d'] != 0.0).sum():,} / {len(df):,}")
        logger.info(f"   Non-zero divergence: {(df['feat_sato_divergence_10d'] != 0.0).sum():,} / {len(df):,}")
        return True
    else:
        logger.error("❌ Sato factors missing from cleaned file!")
        return False


def main():
    logger.info("=" * 80)
    logger.info("Retrain and Evaluate with Sato Factors")
    logger.info("=" * 80)
    
    # Step 1: Verify Sato factors
    if not verify_sato_factors():
        logger.error("❌ Verification failed. Please run add_sato_to_cleaned_file.py first.")
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Training Instructions")
    logger.info("=" * 80)
    logger.info("To train models with Sato factors, run:")
    logger.info("  python scripts/train_full_dataset.py --train-data \"data/factor_exports/polygon_factors_all_filtered_clean.parquet\" --top-n 50")
    
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: 80/20 Time Split Evaluation Instructions")
    logger.info("=" * 80)
    logger.info("To run 80/20 time split evaluation, run:")
    logger.info("  python scripts/time_split_80_20_oos_eval.py --data-file \"data/factor_exports/polygon_factors_all_filtered_clean.parquet\" --horizon-days 10 --split 0.8 --models catboost lambdarank ridge_stacking --top-n 20")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Verification complete!")
    logger.info("=" * 80)
    logger.info("Sato factors are present in the cleaned file.")
    logger.info("You can now proceed with training and evaluation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
