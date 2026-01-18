#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalculate downside_beta_ewm_21 in existing parquet file and replace downside_beta_252"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bma_models.simple_25_factor_engine import Simple17FactorEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def recalculate_downside_beta_ewm_21(
    parquet_path: str,
    output_path: Optional[str] = None,
    backup: bool = True
) -> None:
    """
    Recalculate downside_beta_ewm_21 in existing parquet file.
    
    Args:
        parquet_path: Path to input parquet file (MultiIndex format)
        output_path: Optional output path (default: overwrite input)
        backup: Whether to create backup before overwriting
    """
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    logger.info(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Ensure MultiIndex format
    if not isinstance(df.index, pd.MultiIndex):
        if 'date' in df.columns and 'ticker' in df.columns:
            logger.info("Converting to MultiIndex format...")
            df = df.set_index(['date', 'ticker']).sort_index()
        else:
            raise ValueError("DataFrame must have MultiIndex or 'date'/'ticker' columns")
    
    logger.info(f"Loaded data: {len(df)} rows, {df.index.nlevels} index levels")
    logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    logger.info(f"Tickers: {df.index.get_level_values('ticker').nunique()}")
    
    # Check if downside_beta_252 exists
    has_old = 'downside_beta_252' in df.columns
    has_new = 'downside_beta_ewm_21' in df.columns
    
    logger.info(f"Current columns: downside_beta_252={has_old}, downside_beta_ewm_21={has_new}")
    
    # Create backup if requested
    if backup and output_path is None:
        backup_path = parquet_file.parent / f"{parquet_file.stem}_backup_before_ewm21.parquet"
        logger.info(f"Creating backup: {backup_path}")
        df.to_parquet(backup_path, compression='snappy', index=True)
        logger.info("Backup created successfully")
    
    # Reset index to work with Simple17FactorEngine
    logger.info("Resetting index for factor calculation...")
    df_reset = df.reset_index()
    
    # Ensure required columns exist
    required_cols = ['date', 'ticker', 'Close']
    missing_cols = [c for c in required_cols if c not in df_reset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize factor engine
    logger.info("Initializing Simple17FactorEngine...")
    engine = Simple17FactorEngine(mode='predict', lookback_days=252)
    
    # Set alpha_factors to only include downside_beta_ewm_21
    engine.alpha_factors = ['downside_beta_ewm_21']
    
    # Group by ticker for calculation
    grouped = df_reset.groupby('ticker')
    
    # Calculate new factor
    logger.info("Calculating downside_beta_ewm_21...")
    try:
        beta_results = engine._compute_downside_beta_ewm_21(df_reset, grouped, benchmark='QQQ')
        logger.info(f"Calculated {len(beta_results)} rows of downside_beta_ewm_21")
        
        # Merge back into dataframe
        df_reset['downside_beta_ewm_21'] = beta_results['downside_beta_ewm_21'].values
        
        # Remove old downside_beta_252 if it exists
        if 'downside_beta_252' in df_reset.columns:
            logger.info("Removing old downside_beta_252 column...")
            df_reset = df_reset.drop(columns=['downside_beta_252'])
        
        # Restore MultiIndex
        df_reset = df_reset.set_index(['date', 'ticker']).sort_index()
        
        # Verify calculation
        new_col = df_reset['downside_beta_ewm_21']
        logger.info(f"New factor statistics:")
        logger.info(f"  Non-null values: {new_col.notna().sum()} / {len(new_col)}")
        logger.info(f"  Mean: {new_col.mean():.6f}")
        logger.info(f"  Std: {new_col.std():.6f}")
        logger.info(f"  Min: {new_col.min():.6f}")
        logger.info(f"  Max: {new_col.max():.6f}")
        
    except Exception as e:
        logger.error(f"Error calculating downside_beta_ewm_21: {e}", exc_info=True)
        raise
    
    # Save updated dataframe
    output_file = Path(output_path) if output_path else parquet_file
    logger.info(f"Saving updated parquet file: {output_file}")
    df_reset.to_parquet(output_file, compression='snappy', index=True)
    logger.info("File saved successfully!")
    
    # Summary
    logger.info("=" * 80)
    logger.info("RECALCULATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input file: {parquet_path}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Rows processed: {len(df_reset)}")
    logger.info(f"Tickers: {df_reset.index.get_level_values('ticker').nunique()}")
    logger.info(f"Date range: {df_reset.index.get_level_values('date').min()} to {df_reset.index.get_level_values('date').max()}")
    logger.info(f"downside_beta_252 removed: {has_old}")
    logger.info(f"downside_beta_ewm_21 added: True")
    logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recalculate downside_beta_ewm_21 in existing parquet file"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=r"D:\trade\data\factor_exports\over_100m_5y_t10_end20260115_nocfo\polygon_factors_all.parquet",
        help='Input parquet file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet file path (default: overwrite input)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    
    args = parser.parse_args()
    
    try:
        recalculate_downside_beta_ewm_21(
            parquet_path=args.input,
            output_path=args.output,
            backup=not args.no_backup
        )
        logger.info("Recalculation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Recalculation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
