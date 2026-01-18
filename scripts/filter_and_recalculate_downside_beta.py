#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter multiindex file by removing tickers that meet exclusion criteria,
then recalculate downside_beta_ewm_21 for the filtered data.

Exclusion criteria:
- hist_vol_40d > P99.9
- close price on 2026/01/16 < 5 USD (exclude low-priced stocks)

If a ticker meets ANY of these criteria, remove ALL its data from the multiindex.
"""

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


def filter_tickers_by_criteria(
    df: pd.DataFrame,
    filter_date: str = '2026-01-16',
    min_price: float = 5.0
) -> pd.DataFrame:
    """
    Filter out tickers that meet exclusion criteria.
    
    Args:
        df: MultiIndex DataFrame (date, ticker)
        filter_date: Date to check close price (YYYY-MM-DD)
        min_price: Minimum price threshold (USD)
    
    Returns:
        Filtered DataFrame with excluded tickers removed
    """
    logger.info("=" * 100)
    logger.info("FILTERING TICKERS BY EXCLUSION CRITERIA")
    logger.info("=" * 100)
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (date, ticker)")
    
    original_tickers = df.index.get_level_values('ticker').nunique()
    original_rows = len(df)
    
    logger.info(f"Original data: {original_rows:,} rows, {original_tickers:,} tickers")
    
    # Get date level
    dates = df.index.get_level_values('date')
    tickers = df.index.get_level_values('ticker')
    
    # Check required columns
    required_cols = ['hist_vol_40d', 'Close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    excluded_tickers = set()
    
    # 1. Filter by hist_vol_40d > P99.9
    logger.info("\n1. Checking hist_vol_40d > P99.9...")
    hist_vol = df['hist_vol_40d'].dropna()
    if len(hist_vol) > 0:
        p99_9_hist_vol = hist_vol.quantile(0.999)
        logger.info(f"   P99.9 of hist_vol_40d: {p99_9_hist_vol:.6f}")
        
        high_vol_mask = df['hist_vol_40d'] > p99_9_hist_vol
        high_vol_tickers = set(df.index[high_vol_mask].get_level_values('ticker').unique())
        logger.info(f"   Found {len(high_vol_tickers)} tickers with hist_vol_40d > P99.9")
        excluded_tickers.update(high_vol_tickers)
    
    # 2. Filter by close price on filter_date < min_price (exclude low-priced stocks)
    logger.info(f"\n3. Checking close price on {filter_date} < {min_price} USD...")
    filter_date_dt = pd.to_datetime(filter_date)
    
    # Find closest available date if exact date not found
    available_dates = df.index.get_level_values('date').unique()
    if filter_date_dt not in available_dates:
        # Find closest date (prefer later dates, then earlier)
        later_dates = [d for d in available_dates if d >= filter_date_dt]
        if later_dates:
            closest_date = min(later_dates)
            logger.info(f"   Date {filter_date} not found, using closest later date: {closest_date.date()}")
        else:
            closest_date = max(available_dates)
            logger.info(f"   Date {filter_date} not found, using latest available date: {closest_date.date()}")
        filter_date_dt = closest_date
    
    try:
        date_data = df.xs(filter_date_dt, level='date', drop_level=False)
        if len(date_data) > 0:
            low_price_mask = date_data['Close'] < min_price
            low_price_tickers = set(date_data.index[low_price_mask].get_level_values('ticker').unique())
            logger.info(f"   Found {len(low_price_tickers)} tickers with Close < {min_price} on {filter_date_dt.date()}")
            excluded_tickers.update(low_price_tickers)
        else:
            logger.warning(f"   No data found for date {filter_date_dt.date()}")
    except KeyError:
        logger.warning(f"   Date {filter_date_dt.date()} not found in index")
    
    logger.info(f"\nTotal excluded tickers: {len(excluded_tickers)}")
    if excluded_tickers:
        logger.info(f"   Excluded tickers (first 20): {sorted(list(excluded_tickers))[:20]}")
        if len(excluded_tickers) > 20:
            logger.info(f"   ... and {len(excluded_tickers) - 20} more")
    
    # Remove excluded tickers
    if excluded_tickers:
        keep_mask = ~df.index.get_level_values('ticker').isin(excluded_tickers)
        df_filtered = df[keep_mask].copy()
        
        filtered_tickers = df_filtered.index.get_level_values('ticker').nunique()
        filtered_rows = len(df_filtered)
        
        logger.info(f"\nFiltered data: {filtered_rows:,} rows, {filtered_tickers:,} tickers")
        logger.info(f"Removed: {original_rows - filtered_rows:,} rows ({original_rows - filtered_rows:.1f}%)")
        logger.info(f"Removed: {original_tickers - filtered_tickers:,} tickers")
        
        return df_filtered
    else:
        logger.info("\nNo tickers excluded - returning original data")
        return df.copy()


def recalculate_downside_beta_ewm_21(
    df: pd.DataFrame,
    benchmark: str = 'QQQ'
) -> pd.DataFrame:
    """
    Recalculate downside_beta_ewm_21 for the filtered DataFrame.
    
    Args:
        df: MultiIndex DataFrame (date, ticker) - already filtered
        benchmark: Benchmark symbol (default: QQQ)
    
    Returns:
        DataFrame with recalculated downside_beta_ewm_21
    """
    logger.info("=" * 100)
    logger.info("RECALCULATING downside_beta_ewm_21")
    logger.info("=" * 100)
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (date, ticker)")
    
    # Reset index for factor calculation
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
    logger.info(f"Calculating downside_beta_ewm_21 vs {benchmark}...")
    try:
        beta_results = engine._compute_downside_beta_ewm_21(df_reset, grouped, benchmark=benchmark)
        logger.info(f"Calculated {len(beta_results)} rows of downside_beta_ewm_21")
        
        # Merge back into dataframe
        df_reset['downside_beta_ewm_21'] = beta_results['downside_beta_ewm_21'].values
        
        # Restore MultiIndex
        df_reset = df_reset.set_index(['date', 'ticker']).sort_index()
        
        # Verify calculation
        new_col = df_reset['downside_beta_ewm_21']
        logger.info(f"\nNew factor statistics:")
        logger.info(f"  Non-null values: {new_col.notna().sum():,} / {len(new_col):,}")
        logger.info(f"  Coverage: {new_col.notna().sum() / len(new_col) * 100:.2f}%")
        logger.info(f"  Mean: {new_col.mean():.6f}")
        logger.info(f"  Std: {new_col.std():.6f}")
        logger.info(f"  Min: {new_col.min():.6f}")
        logger.info(f"  Max: {new_col.max():.6f}")
        
        return df_reset
        
    except Exception as e:
        logger.error(f"Error calculating downside_beta_ewm_21: {e}", exc_info=True)
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter multiindex file and recalculate downside_beta_ewm_21"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=r"D:\trade\data\factor_exports\over_100m_5y_t10_end20260115_nocfo\polygon_factors_all_backup.parquet",
        help='Input parquet file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet file path (default: overwrite input)'
    )
    parser.add_argument(
        '--filter-date',
        type=str,
        default='2026-01-16',
        help='Date to check close price (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=5.0,
        help='Minimum price threshold (USD)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='QQQ',
        help='Benchmark symbol for downside beta calculation'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    output_file = Path(args.output) if args.output else input_file
    
    try:
        # Load data
        logger.info("=" * 100)
        logger.info("LOADING DATA")
        logger.info("=" * 100)
        logger.info(f"Input file: {input_file}")
        
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded: {len(df):,} rows, {df.index.nlevels} index levels")
        
        # Ensure MultiIndex format
        if not isinstance(df.index, pd.MultiIndex):
            if 'date' in df.columns and 'ticker' in df.columns:
                logger.info("Converting to MultiIndex format...")
                df = df.set_index(['date', 'ticker']).sort_index()
            else:
                raise ValueError("DataFrame must have MultiIndex or 'date'/'ticker' columns")
        
        logger.info(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        logger.info(f"Tickers: {df.index.get_level_values('ticker').nunique():,}")
        
        # Create backup if requested
        if not args.no_backup and args.output is None:
            backup_path = input_file.parent / f"{input_file.stem}_backup_before_filter.parquet"
            logger.info(f"\nCreating backup: {backup_path}")
            df.to_parquet(backup_path, compression='snappy', index=True)
            logger.info("Backup created successfully")
        
        # Filter tickers
        df_filtered = filter_tickers_by_criteria(
            df,
            filter_date=args.filter_date,
            min_price=args.min_price
        )
        
        # Recalculate downside_beta_ewm_21
        df_final = recalculate_downside_beta_ewm_21(
            df_filtered,
            benchmark=args.benchmark
        )
        
        # Save updated dataframe
        logger.info("=" * 100)
        logger.info("SAVING RESULTS")
        logger.info("=" * 100)
        logger.info(f"Output file: {output_file}")
        df_final.to_parquet(output_file, compression='snappy', index=True)
        logger.info("File saved successfully!")
        
        # Summary
        logger.info("=" * 100)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 100)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Original rows: {len(df):,}")
        logger.info(f"Filtered rows: {len(df_final):,}")
        logger.info(f"Rows removed: {len(df) - len(df_final):,} ({(len(df) - len(df_final))/len(df)*100:.2f}%)")
        logger.info(f"Original tickers: {df.index.get_level_values('ticker').nunique():,}")
        logger.info(f"Filtered tickers: {df_final.index.get_level_values('ticker').nunique():,}")
        logger.info(f"Tickers removed: {df.index.get_level_values('ticker').nunique() - df_final.index.get_level_values('ticker').nunique():,}")
        logger.info(f"downside_beta_ewm_21 recalculated: Yes")
        logger.info("=" * 100)
        
        logger.info("\n[SUCCESS] Filtering and recalculation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
