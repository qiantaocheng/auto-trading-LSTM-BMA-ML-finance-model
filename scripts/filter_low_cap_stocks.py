#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter stocks based on:
1. Close price < 5 on 2026-01-26
2. Market cap < 500 million

Uses Polygon API to get:
- Close price from aggregates
- Weighted shares outstanding from ticker details
- Market cap = Close Price × Weighted Shares Outstanding
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
import os
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('D:/trade/results/filter_low_cap_stocks.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Polygon API key
API_KEY = os.environ.get('POLYGON_API_KEY', '')
if not API_KEY:
    # Try to load from config
    try:
        import sys
        sys.path.insert(0, 'D:/trade')
        from polygon_client import polygon_client
        API_KEY = polygon_client.api_key
    except:
        pass

BASE_URL = "https://api.polygon.io"
RATE_LIMIT_DELAY = 0.15  # seconds between requests

# Filter criteria
TARGET_DATE = "2026-01-26"
MIN_CLOSE_PRICE = 5.0
MIN_MARKET_CAP = 500_000_000  # 500 million

# Output files
OUTPUT_DIR = Path("D:/trade/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "filtered_low_cap_stocks.csv"
KEEP_TICKERS_CSV = OUTPUT_DIR / "valid_tickers_after_filter.csv"


def get_close_price(ticker: str, date: str, session: requests.Session) -> float:
    """Get close price for a ticker on a specific date using aggregates API."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
    params = {"apiKey": API_KEY, "adjusted": "true"}

    try:
        response = session.get(url, params=params, timeout=30)
        time.sleep(RATE_LIMIT_DELAY)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results and len(results) > 0:
                return float(results[0].get("c", 0))  # 'c' is close price
        elif response.status_code == 403:
            # Try previous trading day
            pass
        return np.nan
    except Exception as e:
        logger.debug(f"Error getting close price for {ticker}: {e}")
        return np.nan


def get_ticker_details(ticker: str, session: requests.Session) -> dict:
    """Get ticker details including weighted_shares_outstanding."""
    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    params = {"apiKey": API_KEY}

    try:
        response = session.get(url, params=params, timeout=30)
        time.sleep(RATE_LIMIT_DELAY)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {})
            return {
                "weighted_shares_outstanding": results.get("weighted_shares_outstanding"),
                "market_cap": results.get("market_cap"),
                "name": results.get("name", ""),
                "type": results.get("type", ""),
            }
        return {}
    except Exception as e:
        logger.debug(f"Error getting ticker details for {ticker}: {e}")
        return {}


def main():
    logger.info("=" * 80)
    logger.info("FILTER LOW CAP STOCKS")
    logger.info("=" * 80)
    logger.info(f"Target date: {TARGET_DATE}")
    logger.info(f"Min close price: ${MIN_CLOSE_PRICE}")
    logger.info(f"Min market cap: ${MIN_MARKET_CAP:,.0f}")

    # Load parquet file
    parquet_file = Path("D:/trade/data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet")

    if not parquet_file.exists():
        logger.error(f"Parquet file not found: {parquet_file}")
        return

    logger.info(f"\nLoading: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    logger.info(f"Shape: {df.shape}")

    # Get unique tickers from MultiIndex
    all_tickers = df.index.get_level_values('ticker').unique().tolist()
    logger.info(f"Unique tickers: {len(all_tickers)}")

    # Create session
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {API_KEY}'})

    # Process each ticker
    results = []
    filtered_out = []
    valid_tickers = []

    logger.info(f"\nProcessing {len(all_tickers)} tickers...")

    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(all_tickers)} ({(i + 1) / len(all_tickers) * 100:.1f}%)")

        # Get close price
        close_price = get_close_price(ticker, TARGET_DATE, session)

        # Get ticker details
        details = get_ticker_details(ticker, session)
        shares = details.get("weighted_shares_outstanding")
        api_market_cap = details.get("market_cap")

        # Calculate market cap
        if shares and close_price and not np.isnan(close_price):
            calculated_market_cap = close_price * shares
        else:
            calculated_market_cap = api_market_cap  # Use API market cap as fallback

        # Determine filter reason
        filter_reasons = []
        if close_price and not np.isnan(close_price) and close_price < MIN_CLOSE_PRICE:
            filter_reasons.append(f"close<{MIN_CLOSE_PRICE}")
        if calculated_market_cap and calculated_market_cap < MIN_MARKET_CAP:
            filter_reasons.append(f"mcap<{MIN_MARKET_CAP/1e6:.0f}M")

        is_filtered = len(filter_reasons) > 0

        result = {
            "ticker": ticker,
            "close_price": close_price,
            "weighted_shares": shares,
            "api_market_cap": api_market_cap,
            "calculated_market_cap": calculated_market_cap,
            "name": details.get("name", ""),
            "type": details.get("type", ""),
            "is_filtered": is_filtered,
            "filter_reason": ", ".join(filter_reasons) if filter_reasons else ""
        }
        results.append(result)

        if is_filtered:
            filtered_out.append(ticker)
        else:
            valid_tickers.append(ticker)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save all results
    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\nAll results saved to: {OUTPUT_CSV}")

    # Save valid tickers
    pd.DataFrame({"ticker": valid_tickers}).to_csv(KEEP_TICKERS_CSV, index=False)
    logger.info(f"Valid tickers saved to: {KEEP_TICKERS_CSV}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tickers processed: {len(all_tickers)}")
    logger.info(f"Tickers to FILTER OUT: {len(filtered_out)}")
    logger.info(f"Tickers to KEEP: {len(valid_tickers)}")

    # Show filtered tickers
    filtered_df = results_df[results_df['is_filtered']]
    if not filtered_df.empty:
        logger.info(f"\nFiltered out tickers ({len(filtered_df)}):")
        for _, row in filtered_df.iterrows():
            close_str = f"${row['close_price']:.2f}" if pd.notna(row['close_price']) else "N/A"
            mcap_str = f"${row['calculated_market_cap']/1e6:.1f}M" if pd.notna(row['calculated_market_cap']) else "N/A"
            logger.info(f"  {row['ticker']:<8} Close={close_str:<10} MCap={mcap_str:<12} Reason: {row['filter_reason']}")

    logger.info("\n" + "=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)
    logger.info(f"Results CSV: {OUTPUT_CSV}")
    logger.info(f"Valid tickers CSV: {KEEP_TICKERS_CSV}")


if __name__ == "__main__":
    main()
