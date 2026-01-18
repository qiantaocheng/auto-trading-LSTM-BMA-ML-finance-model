#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add sector/industry column to factor dataset
Fetch sector info from yfinance once and save to factors_all.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict
import yfinance as yf
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sector_info(ticker: str, cache: Dict[str, str] = None) -> str:
    """Get sector information for a ticker using yfinance"""
    if cache is None:
        cache = {}
    
    if ticker in cache:
        return cache[ticker]
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', info.get('industry', 'Unknown'))
        if not sector or sector == 'None':
            sector = 'Unknown'
        cache[ticker] = sector
        # Rate limiting
        time.sleep(0.1)
        return sector
    except Exception as e:
        logger.debug(f"Failed to get sector for {ticker}: {e}")
        cache[ticker] = 'Unknown'
        return 'Unknown'

def add_sector_column(factors_path: Path, output_path: Path = None) -> pd.DataFrame:
    """Add sector column to factors dataset"""
    
    logger.info(f"Loading factors from {factors_path}")
    df = pd.read_parquet(factors_path)
    
    # Check if sector already exists
    if 'sector' in df.columns:
        logger.info("✅ Sector column already exists!")
        return df
    
    # Ensure MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.set_index(['date', 'ticker'])
        else:
            raise ValueError("Factor data must have MultiIndex (date, ticker) or columns (date, ticker)")
    
    # Get unique tickers
    tickers = df.index.get_level_values('ticker').unique().tolist()
    logger.info(f"Found {len(tickers)} unique tickers")
    
    # Fetch sector info for all tickers
    logger.info("Fetching sector information from yfinance...")
    sector_cache = {}
    sector_map = {}
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(tickers)} tickers...")
        sector = get_sector_info(ticker, sector_cache)
        sector_map[ticker] = sector
    
    logger.info(f"✅ Fetched sector info for {len(sector_map)} tickers")
    
    # Map sectors to dataframe
    logger.info("Adding sector column to dataframe...")
    df['sector'] = df.index.get_level_values('ticker').map(sector_map).fillna('Unknown')
    
    # Save sector mapping for future use
    sector_df = pd.DataFrame([
        {'ticker': ticker, 'sector': sector}
        for ticker, sector in sector_map.items()
    ])
    mapping_path = factors_path.parent / "ticker_sector_mapping.csv"
    sector_df.to_csv(mapping_path, index=False)
    logger.info(f"✅ Saved sector mapping to {mapping_path}")
    
    # Save updated factors
    if output_path is None:
        output_path = factors_path
    
    logger.info(f"Saving updated factors to {output_path}...")
    df.to_parquet(output_path, index=True)
    logger.info(f"✅ Saved {len(df)} rows with sector column")
    
    # Print sector distribution
    sector_counts = df['sector'].value_counts()
    logger.info("\nSector distribution:")
    for sector, count in sector_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {sector}: {count:,} ({pct:.1f}%)")
    
    return df

def main():
    factors_path = Path("data/factor_exports/factors/factors_all.parquet")
    
    if not factors_path.exists():
        logger.error(f"Factors file not found: {factors_path}")
        return 1
    
    # Add sector column
    df = add_sector_column(factors_path)
    
    logger.info("✅ Done! Sector column added to factors dataset.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

