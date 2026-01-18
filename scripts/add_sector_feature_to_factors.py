#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add sector/industry feature to factors_all.parquet
One-time operation: fetch sector info for all unique tickers and add as a column
With progress saving to resume if interrupted
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict
import yfinance as yf
import time
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sector_cache(cache_file: Path) -> Dict[str, str]:
    """Load sector cache from file"""
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {}

def save_sector_cache(cache: Dict[str, str], cache_file: Path):
    """Save sector cache to file"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def get_sector_info(ticker: str, cache: Dict[str, str] = None, max_retries: int = 3) -> str:
    """Get sector information for a ticker using yfinance"""
    if cache is None:
        cache = {}
    
    if ticker in cache:
        return cache[ticker]
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', info.get('industry', 'Unknown'))
            if not sector or sector == 'None':
                sector = 'Unknown'
            cache[ticker] = sector
            # Rate limiting
            time.sleep(0.05)
            return sector
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            logger.debug(f"Failed to get sector for {ticker} after {max_retries} attempts: {e}")
            cache[ticker] = 'Unknown'
            return 'Unknown'

def main():
    factors_path = Path("data/factor_exports/factors/factors_all.parquet")
    cache_file = Path("data/factor_exports/factors/sector_cache.json")
    
    if not factors_path.exists():
        logger.error(f"Factors file not found: {factors_path}")
        return 1
    
    logger.info(f"Loading factors from {factors_path}...")
    df = pd.read_parquet(factors_path)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check if sector already exists
    if 'sector' in df.columns:
        logger.info("Sector column already exists!")
        logger.info(f"Sector distribution:\n{df['sector'].value_counts()}")
        return 0
    
    # Load existing cache
    sector_cache = load_sector_cache(cache_file)
    logger.info(f"Loaded {len(sector_cache)} tickers from cache")
    
    # Get unique tickers
    if isinstance(df.index, pd.MultiIndex):
        tickers = df.index.get_level_values('ticker').unique().tolist()
    elif 'ticker' in df.columns:
        tickers = df['ticker'].unique().tolist()
    else:
        logger.error("Cannot find ticker information")
        return 1
    
    logger.info(f"Found {len(tickers)} unique tickers")
    
    # Filter to tickers not in cache
    tickers_to_fetch = [t for t in tickers if t not in sector_cache]
    logger.info(f"Need to fetch {len(tickers_to_fetch)} tickers ({len(tickers) - len(tickers_to_fetch)} already cached)")
    
    # Fetch sector info for remaining tickers
    if tickers_to_fetch:
        logger.info("Fetching sector information from yfinance...")
        logger.info("This may take a while. Progress is saved every 50 tickers.")
        logger.info("You can stop and resume - it will continue from where it left off.")
        
        for i, ticker in enumerate(tickers_to_fetch):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(tickers_to_fetch)} tickers... ({i+1}/{len(tickers_to_fetch)} = {(i+1)/len(tickers_to_fetch)*100:.1f}%)")
                # Save progress every 50 tickers
                save_sector_cache(sector_cache, cache_file)
                logger.info(f"  Progress saved to {cache_file}")
            
            get_sector_info(ticker, sector_cache)
        
        # Final save
        save_sector_cache(sector_cache, cache_file)
        logger.info(f"Fetched sector info for {len(tickers_to_fetch)} tickers")
    
    logger.info(f"Total sector info: {len(sector_cache)} tickers")
    
    # Add sector column
    if isinstance(df.index, pd.MultiIndex):
        # Map sector to MultiIndex
        df['sector'] = df.index.get_level_values('ticker').map(lambda t: sector_cache.get(t, 'Unknown'))
    elif 'ticker' in df.columns:
        df['sector'] = df['ticker'].map(lambda t: sector_cache.get(t, 'Unknown'))
    else:
        logger.error("Cannot map sector to dataframe")
        return 1
    
    # Show sector distribution
    logger.info("\nSector distribution:")
    sector_counts = df['sector'].value_counts()
    for sector, count in sector_counts.items():
        logger.info(f"  {sector}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Save updated factors
    backup_path = factors_path.with_suffix('.parquet.backup')
    if not backup_path.exists():
        logger.info(f"\nCreating backup: {backup_path}")
        pd.read_parquet(factors_path).to_parquet(backup_path)
    else:
        logger.info(f"Backup already exists: {backup_path}")
    
    logger.info(f"Saving updated factors with sector column...")
    df.to_parquet(factors_path)
    
    # Also save sector mapping as CSV for reference
    sector_mapping_path = factors_path.parent / "sector_mapping.csv"
    sector_df = pd.DataFrame([
        {'ticker': ticker, 'sector': sector}
        for ticker, sector in sector_cache.items()
    ])
    sector_df.to_csv(sector_mapping_path, index=False)
    logger.info(f"Saved sector mapping to {sector_mapping_path}")
    
    logger.info("\n✅ Sector feature added successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
