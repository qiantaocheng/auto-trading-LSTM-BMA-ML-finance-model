#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter out tickers with extreme volatility (hist_vol_40d > P99.5 or blowoff_ratio > P99.5)"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_extreme_volatility_tickers(
    parquet_path: str,
    output_path: Optional[str] = None,
    percentile: float = 99.5,
    backup: bool = True
) -> pd.DataFrame:
    """
    Filter out tickers with extreme volatility from MultiIndex parquet file.
    
    If ANY day for a ticker has hist_vol_40d > P99.5 OR blowoff_ratio > P99.5,
    remove ALL data for that ticker.
    
    Args:
        parquet_path: Path to input parquet file (MultiIndex format)
        output_path: Optional output path (default: overwrite input)
        percentile: Percentile threshold (default: 99.5)
        backup: Whether to create backup before overwriting
    
    Returns:
        Filtered DataFrame with MultiIndex
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
    
    logger.info(f"Loaded data: {len(df)} rows")
    logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    logger.info(f"Tickers: {df.index.get_level_values('ticker').nunique()}")
    
    # Check required columns
    required_cols = ['hist_vol_40d', 'blowoff_ratio']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create backup if requested
    if backup and output_path is None:
        backup_path = parquet_file.parent / f"{parquet_file.stem}_backup_before_vol_filter.parquet"
        logger.info(f"Creating backup: {backup_path}")
        df.to_parquet(backup_path, compression='snappy', index=True)
        logger.info("Backup created successfully")
    
    # Calculate percentiles across all data
    logger.info(f"Calculating {percentile}th percentiles...")
    hist_vol_p99_5 = df['hist_vol_40d'].quantile(percentile / 100.0)
    blowoff_p99_5 = df['blowoff_ratio'].quantile(percentile / 100.0)
    
    logger.info(f"  hist_vol_40d P{percentile}: {hist_vol_p99_5:.6f}")
    logger.info(f"  blowoff_ratio P{percentile}: {blowoff_p99_5:.6f}")
    
    # Find tickers with extreme values on ANY day
    logger.info("Identifying tickers with extreme volatility...")
    
    # Mark rows with extreme values
    extreme_hist_vol = df['hist_vol_40d'] > hist_vol_p99_5
    extreme_blowoff = df['blowoff_ratio'] > blowoff_p99_5
    extreme_any = extreme_hist_vol | extreme_blowoff
    
    # Get tickers that have at least one extreme day
    extreme_tickers = set(df.index[extreme_any].get_level_values('ticker').unique())
    
    logger.info(f"Found {len(extreme_tickers)} tickers with extreme volatility:")
    logger.info(f"  Tickers with hist_vol_40d > P{percentile}: {extreme_hist_vol.sum()} rows")
    logger.info(f"  Tickers with blowoff_ratio > P{percentile}: {extreme_blowoff.sum()} rows")
    logger.info(f"  Total tickers to remove: {len(extreme_tickers)}")
    
    if len(extreme_tickers) > 0:
        # Show some examples
        logger.info(f"\nSample extreme tickers (first 10):")
        for i, ticker in enumerate(sorted(extreme_tickers)[:10]):
            ticker_data = df.xs(ticker, level='ticker', drop_level=False)
            extreme_rows = ticker_data[extreme_any.xs(ticker, level='ticker', drop_level=False)]
            logger.info(f"  {ticker}: {len(extreme_rows)} extreme days out of {len(ticker_data)} total days")
            if len(extreme_rows) > 0:
                logger.info(f"    Max hist_vol_40d: {extreme_rows['hist_vol_40d'].max():.6f}")
                logger.info(f"    Max blowoff_ratio: {extreme_rows['blowoff_ratio'].max():.6f}")
    
    # Filter out all data for extreme tickers
    logger.info(f"\nFiltering out {len(extreme_tickers)} tickers...")
    df_filtered = df[~df.index.get_level_values('ticker').isin(extreme_tickers)].copy()
    
    # Statistics
    rows_before = len(df)
    rows_after = len(df_filtered)
    rows_removed = rows_before - rows_after
    
    tickers_before = df.index.get_level_values('ticker').nunique()
    tickers_after = df_filtered.index.get_level_values('ticker').nunique()
    tickers_removed = tickers_before - tickers_after
    
    logger.info("=" * 80)
    logger.info("FILTERING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Rows before: {rows_before:,}")
    logger.info(f"Rows after:  {rows_after:,}")
    logger.info(f"Rows removed: {rows_removed:,} ({rows_removed/rows_before*100:.2f}%)")
    logger.info(f"\nTickers before: {tickers_before}")
    logger.info(f"Tickers after:  {tickers_after}")
    logger.info(f"Tickers removed: {tickers_removed} ({tickers_removed/tickers_before*100:.2f}%)")
    
    # Verify filtered data
    if len(df_filtered) > 0:
        logger.info(f"\nFiltered data statistics:")
        logger.info(f"  Date range: {df_filtered.index.get_level_values('date').min()} to {df_filtered.index.get_level_values('date').max()}")
        logger.info(f"  hist_vol_40d max: {df_filtered['hist_vol_40d'].max():.6f} (threshold: {hist_vol_p99_5:.6f})")
        logger.info(f"  blowoff_ratio max: {df_filtered['blowoff_ratio'].max():.6f} (threshold: {blowoff_p99_5:.6f})")
    
    # Save filtered data
    output_file = Path(output_path) if output_path else parquet_file
    logger.info(f"\nSaving filtered parquet file: {output_file}")
    df_filtered.to_parquet(output_file, compression='snappy', index=True)
    logger.info("File saved successfully!")
    
    # Save removed tickers list
    removed_tickers_file = output_file.parent / f"{output_file.stem}_removed_tickers.txt"
    with open(removed_tickers_file, 'w', encoding='utf-8') as f:
        for ticker in sorted(extreme_tickers):
            f.write(f"{ticker}\n")
    logger.info(f"Removed tickers list saved to: {removed_tickers_file}")
    
    logger.info("=" * 80)
    
    return df_filtered


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter out tickers with extreme volatility (hist_vol_40d > P99.5 or blowoff_ratio > P99.5)"
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
        '--percentile',
        type=float,
        default=99.5,
        help='Percentile threshold (default: 99.5)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    
    args = parser.parse_args()
    
    try:
        df_filtered = filter_extreme_volatility_tickers(
            parquet_path=args.input,
            output_path=args.output,
            percentile=args.percentile,
            backup=not args.no_backup
        )
        logger.info("Filtering completed successfully!")
        logger.info(f"Filtered DataFrame shape: {df_filtered.shape}")
        return 0
    except Exception as e:
        logger.error(f"Filtering failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
