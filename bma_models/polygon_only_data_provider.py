#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon-Only Data Provider
仅使用Polygon API的数据提供器 - 无模拟数据，无其他数据源

This module uses the existing polygon_client.py for ALL data needs.
NO Yahoo Finance, NO random data, NO simulation, NO fake data.
If Polygon data is unavailable, returns None/NaN - never generates fake data.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import Dict, Optional, List, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path to import polygon_client
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the existing polygon_client
from polygon_client import PolygonClient, polygon_client

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality indicators"""
    POLYGON_REALTIME = "polygon_realtime"     # Real-time from Polygon
    POLYGON_DELAYED = "polygon_delayed"       # Delayed data from Polygon
    POLYGON_HISTORICAL = "polygon_historical" # Historical data from Polygon
    NOT_AVAILABLE = "not_available"          # No data available


class PolygonOnlyDataProvider:
    """
    Data provider that ONLY uses Polygon API through polygon_client.py.
    NO mock data, NO random generation, NO other sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with Polygon API client
        
        Args:
            api_key: Polygon API key (uses existing client if None)
        """
        if api_key:
            # Create new client with provided key
            self.client = PolygonClient(api_key)
        else:
            # Use existing global client from polygon_client.py
            self.client = polygon_client
        
        # Cache settings
        self.cache_dir = Path("cache/polygon_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = 24
        
        logger.info("PolygonOnlyDataProvider initialized using polygon_client.py")
    
    def get_market_data(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timespan: str = 'day'
    ) -> Optional[pd.DataFrame]:
        """
        Get market data using polygon_client
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            timespan: Time interval (day, hour, minute)
            
        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        try:
            # Convert dates to string if datetime
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Use polygon_client's get_historical_bars method
            df = self.client.get_historical_bars(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
                timespan=timespan
            )
            
            if df is not None and not df.empty:
                logger.info(f"Retrieved {len(df)} bars for {ticker} from Polygon")
                return df
            else:
                logger.warning(f"No data available for {ticker} from Polygon")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None
    
    def get_realtime_price(self, ticker: str) -> Optional[float]:
        """
        Get real-time price using polygon_client
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Current price or None if unavailable
        """
        try:
            price = self.client.get_current_price(ticker)
            if price and price > 0:
                return price
            else:
                logger.warning(f"No real-time price for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error fetching real-time price for {ticker}: {e}")
            return None
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        Get ticker details using polygon_client
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Ticker information dict or None
        """
        try:
            info = self.client.get_ticker_details(ticker)
            if info:
                return info
            else:
                logger.warning(f"No ticker details for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error fetching ticker details for {ticker}: {e}")
            return None
    
    def get_fundamentals(
        self,
        ticker: str,
        limit: int = 4
    ) -> Optional[pd.DataFrame]:
        """
        Get fundamental/financial data using polygon_client
        
        Args:
            ticker: Stock ticker
            limit: Number of periods to retrieve
            
        Returns:
            DataFrame with fundamental data or None
        """
        try:
            # Get financials from polygon_client
            data = self.client.get_financials(ticker, limit=limit)
            
            if data and data.get('results'):
                results = data['results']
                
                # Process into DataFrame
                processed = []
                for result in results:
                    financials = result.get('financials', {})
                    
                    # Income statement
                    income = financials.get('income_statement', {})
                    revenue = income.get('revenues', {}).get('value')
                    net_income = income.get('net_income_loss', {}).get('value')
                    
                    # Balance sheet
                    balance = financials.get('balance_sheet', {})
                    total_assets = balance.get('assets', {}).get('value')
                    total_liabilities = balance.get('liabilities', {}).get('value')
                    total_equity = balance.get('equity', {}).get('value')
                    
                    # Cash flow
                    cashflow = financials.get('cash_flow_statement', {})
                    operating_cf = cashflow.get('net_cash_flow_from_operating_activities', {}).get('value')
                    
                    # Calculate key ratios
                    roe = None
                    if total_equity and total_equity > 0 and net_income:
                        roe = net_income / total_equity
                    
                    debt_to_equity = None
                    if total_equity and total_equity > 0 and total_liabilities:
                        debt_to_equity = total_liabilities / total_equity
                    
                    # Get shares outstanding for per-share metrics
                    shares = balance.get('shares_outstanding', {}).get('value', 1)
                    if shares == 0:
                        shares = 1  # Avoid division by zero
                    
                    earnings_per_share = net_income / shares if net_income else None
                    book_value_per_share = total_equity / shares if total_equity else None
                    
                    processed.append({
                        'ticker': ticker,
                        'period': result.get('period', ''),
                        'fiscal_year': result.get('fiscal_year'),
                        'fiscal_quarter': result.get('fiscal_quarter'),
                        'revenue': revenue,
                        'net_income': net_income,
                        'earnings_per_share': earnings_per_share,
                        'total_assets': total_assets,
                        'total_liabilities': total_liabilities,
                        'total_equity': total_equity,
                        'operating_cash_flow': operating_cf,
                        'roe': roe,
                        'debt_to_equity': debt_to_equity,
                        'book_value_per_share': book_value_per_share,
                        'shares_outstanding': shares
                    })
                
                if processed:
                    df = pd.DataFrame(processed)
                    
                    # Add calculated ratios that might be missing
                    if 'book_to_market' not in df.columns:
                        # We need market cap for this - get current price
                        current_price = self.get_realtime_price(ticker)
                        if current_price and 'book_value_per_share' in df.columns:
                            df['book_to_market'] = df['book_value_per_share'] / current_price
                        else:
                            df['book_to_market'] = np.nan
                    
                    if 'pe_ratio' not in df.columns:
                        current_price = self.get_realtime_price(ticker) if 'current_price' not in locals() else current_price
                        if current_price and 'earnings_per_share' in df.columns:
                            df['pe_ratio'] = current_price / df['earnings_per_share']
                        else:
                            df['pe_ratio'] = np.nan
                    
                    # Add market cap
                    if 'market_cap' not in df.columns:
                        current_price = self.get_realtime_price(ticker) if 'current_price' not in locals() else current_price
                        if current_price and 'shares_outstanding' in df.columns:
                            df['market_cap'] = current_price * df['shares_outstanding']
                        else:
                            df['market_cap'] = np.nan
                    
                    logger.info(f"Retrieved {len(df)} periods of fundamentals for {ticker}")
                    return df
                else:
                    logger.warning(f"No fundamental data processed for {ticker}")
                    return None
            else:
                logger.warning(f"No fundamental data available for {ticker} from Polygon")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return None
    
    def get_news(
        self,
        ticker: str,
        limit: int = 10,
        start_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get news using polygon_client
        
        Args:
            ticker: Stock ticker
            limit: Number of news items
            start_date: Start date for news
            
        Returns:
            List of news items or None
        """
        try:
            news_data = self.client.get_ticker_news(ticker, limit=limit, published_utc_gte=start_date)
            
            if news_data and news_data.get('results'):
                return news_data['results']
            else:
                logger.warning(f"No news available for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return None
    
    def get_intraday_data(
        self,
        ticker: str,
        date: Optional[str] = None,
        multiplier: int = 5,
        timespan: str = 'minute'
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday data using polygon_client
        
        Args:
            ticker: Stock ticker
            date: Date for intraday data (None for today)
            multiplier: Bar size multiplier
            timespan: Time unit (minute, hour)
            
        Returns:
            DataFrame with intraday data or None
        """
        try:
            if date is None:
                # Get today's intraday data
                df = self.client.get_today_intraday(ticker, timespan)
            else:
                # Get specific date's intraday data
                df = self.client.get_intraday_bars(ticker, multiplier, timespan, date)
            
            if df is not None and not df.empty:
                logger.info(f"Retrieved {len(df)} intraday bars for {ticker}")
                return df
            else:
                logger.warning(f"No intraday data for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return None
    
    def get_options_chain(
        self,
        ticker: str,
        expiration_date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get options chain using polygon_client
        
        Args:
            ticker: Stock ticker
            expiration_date: Options expiration date
            
        Returns:
            Options chain data or None
        """
        try:
            options_data = self.client.get_options_chain(ticker, expiration_date)
            
            if options_data and options_data.get('results'):
                return options_data
            else:
                logger.warning(f"No options data for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {e}")
            return None
    
    def get_market_status(self) -> Optional[Dict]:
        """
        Get market status using polygon_client
        
        Returns:
            Market status dict or None
        """
        try:
            status = self.client.get_market_status()
            if status:
                return status
            else:
                logger.warning("No market status available")
                return None
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return None
    
    def download_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers using polygon_client
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            Dictionary of DataFrames by ticker
        """
        results = {}
        
        for ticker in tickers:
            try:
                df = self.client.download(ticker, start_date, end_date, interval=interval)
                if df is not None and not df.empty:
                    results[ticker] = df
                    logger.info(f"Downloaded {len(df)} bars for {ticker}")
                else:
                    logger.warning(f"No data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation metrics dictionary
        """
        if df is None or df.empty:
            return {
                'valid': False,
                'reason': 'No data',
                'rows': 0,
                'missing_values': 0,
                'data_source': 'polygon'
            }
        
        metrics = {
            'valid': True,
            'rows': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'date_range': f"{df.index.min()} to {df.index.max()}" if not df.empty else "N/A",
            'data_source': 'polygon'
        }
        
        # Check for critical columns
        critical_cols = ['Close', 'Volume'] if 'Close' in df.columns else []
        for col in critical_cols:
            if col in df.columns and df[col].isnull().all():
                metrics['valid'] = False
                metrics['reason'] = f'All {col} values are null'
                break
        
        return metrics


# Factory function for backward compatibility
def create_polygon_provider(api_key: Optional[str] = None) -> PolygonOnlyDataProvider:
    """
    Factory function to create PolygonOnlyDataProvider
    
    Args:
        api_key: Polygon API key (uses existing client if None)
        
    Returns:
        PolygonOnlyDataProvider instance
    """
    return PolygonOnlyDataProvider(api_key=api_key)


# Example usage and testing
if __name__ == "__main__":
    # Create provider using existing polygon_client
    provider = PolygonOnlyDataProvider()
    
    print("Testing PolygonOnlyDataProvider with polygon_client.py...")
    
    # Test market data
    print("\n1. Testing market data:")
    df = provider.get_market_data('AAPL', '2024-01-01', '2024-01-31')
    if df is not None:
        print(f"   Retrieved {len(df)} days of data")
        print(f"   Columns: {list(df.columns)}")
    else:
        print("   No data retrieved")
    
    # Test real-time price
    print("\n2. Testing real-time price:")
    price = provider.get_realtime_price('AAPL')
    print(f"   AAPL current price: ${price}")
    
    # Test ticker info
    print("\n3. Testing ticker info:")
    info = provider.get_ticker_info('AAPL')
    if info:
        print(f"   Company: {info.get('name')}")
        print(f"   Market Cap: ${info.get('market_cap', 'N/A')}")
    
    # Test fundamentals
    print("\n4. Testing fundamentals:")
    fundamentals = provider.get_fundamentals('AAPL', limit=1)
    if fundamentals is not None and not fundamentals.empty:
        latest = fundamentals.iloc[0]
        print(f"   Revenue: ${latest.get('revenue', 'N/A')}")
        print(f"   ROE: {latest.get('roe', 'N/A')}")
    
    # Validate data quality
    print("\n5. Data quality check:")
    if df is not None:
        quality = provider.validate_data_quality(df)
        print(f"   Valid: {quality['valid']}")
        print(f"   Source: {quality['data_source']}")
        print(f"   Missing: {quality['missing_pct']:.1%}")