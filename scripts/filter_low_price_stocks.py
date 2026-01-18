#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter out stocks with Close < $5 on 2025-01-14 from MultiIndex dataset"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import time
import requests
from typing import Set, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Polygon API configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"

# Rate limiting
_last_request_time = 0.0
_rate_limit_delay = 0.2  # 200ms between requests

def rate_limited_request():
    """Enforce rate limiting for Polygon API"""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _rate_limit_delay:
        time.sleep(_rate_limit_delay - elapsed)
    _last_request_time = time.time()

def get_polygon_price(ticker: str, date: str, api_key: str) -> Optional[float]:
    """
    Get close price for a ticker on a specific date using Polygon API
    
    Args:
        ticker: Stock ticker symbol
        date: Date in YYYY-MM-DD format
        api_key: Polygon API key
    
    Returns:
        Close price or None if not found/error
    """
    if not api_key:
        logger.warning(f"No API key provided, skipping Polygon verification for {ticker}")
        return None
    
    try:
        rate_limited_request()
        
        # Convert date to timestamp format for Polygon API
        # Polygon API expects dates in YYYY-MM-DD format for /v2/aggs endpoint
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 1,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
            result = data["results"][0]
            close_price = result.get("c")  # 'c' is close price
            return float(close_price) if close_price is not None else None
        else:
            logger.debug(f"No data found for {ticker} on {date}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"API error for {ticker} on {date}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error getting price for {ticker} on {date}: {e}")
        return None

def filter_low_price_stocks(
    input_file: str,
    output_file: str,
    filter_date: str = "2025-01-14",
    min_price: float = 5.0,
    use_polygon_verify: bool = True,
    api_key: Optional[str] = None
) -> int:
    """
    Filter out stocks with Close < min_price on filter_date
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        filter_date: Date to check prices (YYYY-MM-DD)
        min_price: Minimum price threshold (default $5)
        use_polygon_verify: Whether to verify with Polygon API
        api_key: Polygon API key (if None, uses env var)
    
    Returns:
        Number of tickers removed
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return -1
    
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    logger.info(f"Original dataset shape: {df.shape}")
    logger.info(f"Original unique tickers: {df.index.get_level_values('ticker').nunique()}")
    
    # Ensure MultiIndex format
    if not isinstance(df.index, pd.MultiIndex):
        logger.error("Dataset does not have MultiIndex format")
        return -1
    
    if 'date' not in df.index.names or 'ticker' not in df.index.names:
        logger.error("MultiIndex must have 'date' and 'ticker' levels")
        return -1
    
    # Normalize dates
    dates = pd.to_datetime(df.index.get_level_values('date')).normalize()
    filter_date_dt = pd.Timestamp(filter_date).normalize()
    
    # Get data for filter date
    date_mask = dates == filter_date_dt
    filter_date_data = df[date_mask].copy()
    
    if len(filter_date_data) == 0:
        logger.warning(f"No data found for filter date {filter_date}")
        logger.info("Checking available dates...")
        unique_dates = pd.to_datetime(df.index.get_level_values('date')).normalize().unique()
        logger.info(f"Date range: {unique_dates.min()} to {unique_dates.max()}")
        logger.info(f"Sample dates: {sorted(unique_dates)[:5]}")
        return -1
    
    logger.info(f"Found {len(filter_date_data)} stocks on {filter_date}")
    
    # Identify stocks with Close < min_price
    if 'Close' not in filter_date_data.columns:
        logger.error("'Close' column not found in dataset")
        return -1
    
    # Get tickers with low prices
    low_price_mask = filter_date_data['Close'] < min_price
    low_price_tickers = set(filter_date_data[low_price_mask].index.get_level_values('ticker').unique())
    
    logger.info(f"Found {len(low_price_tickers)} tickers with Close < ${min_price} on {filter_date}")
    
    # Verify with Polygon API if requested
    if use_polygon_verify and api_key:
        logger.info("Verifying prices with Polygon API...")
        verified_low_price_tickers = set()
        
        for ticker in sorted(low_price_tickers):
            polygon_price = get_polygon_price(ticker, filter_date, api_key)
            if polygon_price is not None:
                if polygon_price < min_price:
                    verified_low_price_tickers.add(ticker)
                    logger.debug(f"Verified {ticker}: ${polygon_price:.2f} < ${min_price}")
                else:
                    logger.info(f"{ticker}: Dataset shows ${filter_date_data.loc[filter_date_data.index.get_level_values('ticker') == ticker, 'Close'].iloc[0]:.2f}, but Polygon shows ${polygon_price:.2f} >= ${min_price} - keeping")
            else:
                # If API fails, use dataset value (conservative - remove if dataset says low)
                verified_low_price_tickers.add(ticker)
                logger.warning(f"Could not verify {ticker} via API, using dataset value")
        
        low_price_tickers = verified_low_price_tickers
        logger.info(f"After Polygon verification: {len(low_price_tickers)} tickers to remove")
    
    if len(low_price_tickers) == 0:
        logger.info("No tickers to remove!")
        # Still save a copy to output location
        df.to_parquet(output_path, index=True)
        logger.info(f"Saved unchanged dataset to: {output_path}")
        return 0
    
    # Show sample of tickers to be removed
    logger.info(f"\nSample tickers to be removed (first 20):")
    for i, ticker in enumerate(sorted(low_price_tickers)[:20], 1):
        ticker_data = filter_date_data[filter_date_data.index.get_level_values('ticker') == ticker]
        if len(ticker_data) > 0:
            close_price = ticker_data['Close'].iloc[0]
            logger.info(f"  {i:2d}. {ticker}: ${close_price:.4f}")
    
    # Remove these tickers from entire dataset (all dates)
    logger.info(f"\nRemoving {len(low_price_tickers)} tickers from entire dataset...")
    
    all_tickers = set(df.index.get_level_values('ticker').unique())
    keep_tickers = all_tickers - low_price_tickers
    
    logger.info(f"Keeping {len(keep_tickers)} tickers, removing {len(low_price_tickers)} tickers")
    
    # Filter dataset
    ticker_mask = df.index.get_level_values('ticker').isin(keep_tickers)
    df_filtered = df[ticker_mask].copy()
    
    logger.info(f"Filtered dataset shape: {df_filtered.shape}")
    logger.info(f"Filtered unique tickers: {df_filtered.index.get_level_values('ticker').nunique()}")
    logger.info(f"Rows removed: {len(df) - len(df_filtered):,}")
    
    # Save filtered dataset
    logger.info(f"\nSaving filtered dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_path, index=True)
    
    logger.info("=" * 80)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original tickers: {len(all_tickers)}")
    logger.info(f"Removed tickers: {len(low_price_tickers)}")
    logger.info(f"Remaining tickers: {len(keep_tickers)}")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 80)
    
    # Save list of removed tickers
    removed_file = output_path.parent / f"{output_path.stem}_removed_tickers.txt"
    removed_file.write_text("\n".join(sorted(low_price_tickers)) + "\n", encoding="utf-8")
    logger.info(f"Removed tickers list saved to: {removed_file}")
    
    return len(low_price_tickers)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter stocks with Close < $5 on 2025-01-14")
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\trade\data\factor_exports\over_100m_5y_t10_end20260115_nocfo\polygon_factors_all.parquet",
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\trade\data\factor_exports\over_100m_5y_t10_end20260115_nocfo\polygon_factors_all_filtered.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--filter-date",
        type=str,
        default="2025-01-14",
        help="Date to check prices (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="Minimum price threshold (default: 5.0)"
    )
    parser.add_argument(
        "--no-polygon-verify",
        action="store_true",
        help="Skip Polygon API verification (use dataset values only)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Polygon API key (or use POLYGON_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("POLYGON_API_KEY", "")
    if not api_key and not args.no_polygon_verify:
        logger.warning("No Polygon API key provided. Set POLYGON_API_KEY env var or use --no-polygon-verify")
        logger.info("Proceeding without Polygon verification...")
        use_polygon = False
    else:
        use_polygon = not args.no_polygon_verify
    
    result = filter_low_price_stocks(
        input_file=args.input,
        output_file=args.output,
        filter_date=args.filter_date,
        min_price=args.min_price,
        use_polygon_verify=use_polygon,
        api_key=api_key
    )
    
    return 0 if result >= 0 else 1

if __name__ == "__main__":
    sys.exit(main())
