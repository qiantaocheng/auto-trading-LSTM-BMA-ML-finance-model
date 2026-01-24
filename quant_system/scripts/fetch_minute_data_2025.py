"""
Fetch minute-level data from Polygon API for 2025 (one year)
Store as MultiIndex DataFrame (Symbol, DateTime) -> OHLCV columns
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional
import pickle

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from polygon_client import PolygonClient, polygon_client
from scripts.extract_tickers_from_list import get_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinuteDataFetcher:
    """Fetches minute-level data from Polygon API and stores in MultiIndex format."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize fetcher with Polygon client."""
        if polygon_client is not None:
            self.client = polygon_client
        elif api_key:
            self.client = PolygonClient(api_key, delayed_data_mode=True)
        else:
            raise ValueError("Either polygon_client must be available or api_key must be provided")
        
        # Cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_cache/minute_data_2025")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MinuteDataFetcher initialized with cache_dir={self.cache_dir}")
    
    def fetch_symbol_minutes(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch minute-level data for a single symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with DateTime index and OHLCV columns, or None if failed
        """
        # Check cache
        if use_cache:
            cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_minute.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                        logger.debug(f"Cache hit for {symbol}")
                        return df
                except Exception as e:
                    logger.debug(f"Cache load failed for {symbol}: {e}")
        
        # Fetch from API
        try:
            # Use get_historical_bars with minute timespan
            df = self.client.get_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timespan='minute',
                multiplier=1
            )
            
            if df is None or df.empty:
                logger.warning(f"Empty minute data for {symbol}")
                return None
            
            # Ensure proper column names
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.warning(f"{symbol}: Missing columns {missing}")
                return None
            
            # Ensure DateTime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index.name = 'DateTime'
            
            # Validate and clean data
            df = self._validate_data(df, symbol)
            
            # Cache
            if use_cache:
                try:
                    cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_minute.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f)
                except Exception as e:
                    logger.debug(f"Cache save failed for {symbol}: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching minute data for {symbol}: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean minute-level data."""
        if df.empty:
            return df
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill missing values
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid = (
                (df['High'] < df[['Open', 'Close']].max(axis=1)) |
                (df['Low'] > df[['Open', 'Close']].min(axis=1))
            )
            if invalid.any():
                logger.warning(f"{symbol}: {invalid.sum()} invalid OHLC rows removed")
                df = df[~invalid]
        
        # Remove zero/negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid = (df[col] <= 0) | df[col].isna()
                if invalid.any():
                    logger.warning(f"{symbol}: {invalid.sum()} invalid {col} values")
                    df = df[~invalid]
        
        return df
    
    def fetch_universe_minutes(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Fetch minute-level data for multiple symbols and return MultiIndex DataFrame.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cache
            batch_size: Number of symbols to process before saving checkpoint
            
        Returns:
            MultiIndex DataFrame with columns: Open, High, Low, Close, Volume
            Index: (Symbol, DateTime)
        """
        logger.info(f"Fetching minute data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_data = []
        successful = 0
        failed = []
        
        # Check for existing checkpoint
        checkpoint_file = self.cache_dir / f"checkpoint_{start_date}_{end_date}.pkl"
        processed_symbols = set()
        
        if use_cache and checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    processed_symbols = checkpoint_data.get('processed', set())
                    if checkpoint_data.get('data'):
                        all_data = checkpoint_data['data']
                        successful = len(processed_symbols)
                        logger.info(f"Loaded checkpoint: {successful} symbols already processed")
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
        
        # Process each symbol
        for i, symbol in enumerate(symbols, 1):
            if symbol in processed_symbols:
                logger.info(f"[{i}/{len(symbols)}] Skipping {symbol} (already processed)")
                continue
            
            try:
                logger.info(f"[{i}/{len(symbols)}] Fetching {symbol}...")
                
                df = self.fetch_symbol_minutes(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                
                if df is not None and not df.empty:
                    # Add symbol to index for MultiIndex
                    df['Symbol'] = symbol
                    df = df.reset_index().set_index(['Symbol', 'DateTime'])
                    all_data.append(df)
                    successful += 1
                    processed_symbols.add(symbol)
                    logger.info(f"  {symbol}: {len(df)} minutes")
                else:
                    logger.warning(f"  No data returned for {symbol}")
                    failed.append(symbol)
                
                # Rate limiting
                time.sleep(0.3)
                
                # Save checkpoint every batch_size symbols
                if successful % batch_size == 0:
                    self._save_checkpoint(checkpoint_file, all_data, processed_symbols, start_date, end_date)
                    logger.info(f"Checkpoint saved: {successful} symbols processed")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                failed.append(symbol)
                continue
        
        if not all_data:
            logger.error("No minute data fetched for any symbols")
            return pd.DataFrame()
        
        # Combine all dataframes
        logger.info(f"Combining data: {successful} successful, {len(failed)} failed")
        combined_df = pd.concat(all_data, axis=0)
        
        # Sort by Symbol and DateTime
        combined_df = combined_df.sort_index()
        
        # Save final checkpoint
        self._save_checkpoint(checkpoint_file, all_data, processed_symbols, start_date, end_date)
        
        logger.info(f"Minute data fetch complete: {len(combined_df)} total rows")
        logger.info(f"Date range: {combined_df.index.get_level_values(1).min()} to {combined_df.index.get_level_values(1).max()}")
        
        if failed:
            logger.warning(f"Failed symbols: {failed}")
        
        return combined_df
    
    def _save_checkpoint(
        self,
        checkpoint_file: Path,
        all_data: List[pd.DataFrame],
        processed_symbols: set,
        start_date: str,
        end_date: str
    ):
        """Save checkpoint data."""
        try:
            checkpoint_data = {
                'data': all_data,
                'processed': processed_symbols,
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
    
    def save_to_parquet(self, df: pd.DataFrame, filepath: str):
        """Save MultiIndex DataFrame to Parquet format."""
        try:
            df.to_parquet(filepath, engine='pyarrow')
            logger.info(f"Minute data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save to parquet: {e}")
            # Fallback to pickle
            pickle_path = filepath.replace('.parquet', '.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Saved to pickle instead: {pickle_path}")


def main():
    """Main function to fetch minute data for all tickers."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch minute-level data from Polygon API for 2025'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='List of tickers (if not provided, uses all from extract_tickers_from_list.py)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Start date (YYYY-MM-DD, default: 2025-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-12-31',
        help='End date (YYYY-MM-DD, default: 2025-12-31)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='minute_data_2025.parquet',
        help='Output file path (default: minute_data_2025.parquet)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data_cache/minute_data_2025',
        help='Cache directory'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for checkpoint saving (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = get_all_tickers()
        logger.info(f"Using all {len(tickers)} tickers from extract_tickers_from_list.py")
    
    logger.info(f"Fetching minute data for {len(tickers)} tickers")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Initialize fetcher
    fetcher = MinuteDataFetcher(cache_dir=args.cache_dir)
    
    # Fetch data
    try:
        combined_df = fetcher.fetch_universe_minutes(
            symbols=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=not args.no_cache,
            batch_size=args.batch_size
        )
        
        if combined_df.empty:
            logger.error("No data fetched!")
            return
        
        # Save to parquet
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fetcher.save_to_parquet(combined_df, str(output_path))
        
        logger.info("\n" + "="*80)
        logger.info("FETCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Total rows: {len(combined_df):,}")
        logger.info(f"Symbols: {len(combined_df.index.get_level_values(0).unique())}")
        logger.info(f"Date range: {combined_df.index.get_level_values(1).min()} to {combined_df.index.get_level_values(1).max()}")
        logger.info(f"Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Fetch failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
