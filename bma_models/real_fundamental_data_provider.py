#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Fundamental Data Provider
真实基本面数据提供器 - 使用Polygon API专用版本

This file has been updated to use ONLY Polygon API.
NO Yahoo Finance, NO mock data, NO random generation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from enum import Enum

# Import Polygon-only data provider
from polygon_only_data_provider import PolygonOnlyDataProvider

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types"""
    POLYGON = "polygon"  # Only Polygon API
    CACHED = "cached"
    UNAVAILABLE = "unavailable"


@dataclass
class FundamentalData:
    """Container for fundamental data"""
    ticker: str
    date: datetime
    book_to_market: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    pe_ratio: Optional[float] = None
    earnings_per_share: Optional[float] = None
    market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    data_source: DataSource = DataSource.UNAVAILABLE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ticker': self.ticker,
            'date': self.date,
            'book_to_market': self.book_to_market,
            'roe': self.roe,
            'debt_to_equity': self.debt_to_equity,
            'pe_ratio': self.pe_ratio,
            'earnings_per_share': self.earnings_per_share,
            'market_cap': self.market_cap,
            'revenue_growth': self.revenue_growth,
            'profit_margin': self.profit_margin,
            'data_source': self.data_source.value
        }


class RealFundamentalDataProvider:
    """
    Provides real fundamental data from Polygon API ONLY.
    NO mock data, NO random generation, NO other sources.
    """
    
    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        cache_dir: str = "cache/fundamentals",
        cache_hours: int = 24
    ):
        """
        Initialize data provider with Polygon API only
        
        Args:
            polygon_api_key: Polygon API key
            cache_dir: Directory for caching data
            cache_hours: Hours to keep cached data
        """
        self.polygon_provider = PolygonOnlyDataProvider(api_key=polygon_api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours
        
        logger.info("RealFundamentalDataProvider initialized with Polygon API only")
    
    def get_fundamentals(
        self,
        ticker: str,
        date: Optional[datetime] = None
    ) -> FundamentalData:
        """
        Get fundamental data for a ticker using ONLY Polygon API
        
        Args:
            ticker: Stock ticker
            date: Date for fundamentals (uses latest if None)
            
        Returns:
            FundamentalData object (may have None values if data unavailable)
        """
        if date is None:
            date = datetime.now()
        
        # Try cache first
        cached_data = self._get_cached_data(ticker, date)
        if cached_data:
            logger.info(f"Using cached data for {ticker}")
            return cached_data
        
        # Get from Polygon API
        fund_data = FundamentalData(ticker=ticker, date=date)
        
        try:
            # Get fundamentals from Polygon
            df = self.polygon_provider.get_fundamentals(ticker, limit=1)
            
            if df is not None and not df.empty:
                latest = df.iloc[0]
                
                # Map Polygon data to our structure
                fund_data.book_to_market = latest.get('book_to_market', np.nan)
                fund_data.roe = latest.get('roe', np.nan)
                fund_data.debt_to_equity = latest.get('debt_to_equity', np.nan)
                fund_data.pe_ratio = latest.get('pe_ratio', np.nan)
                fund_data.earnings_per_share = latest.get('earnings_per_share', np.nan)
                fund_data.market_cap = latest.get('market_cap', np.nan)
                fund_data.revenue_growth = latest.get('revenue_growth', np.nan)
                fund_data.profit_margin = latest.get('profit_margin', np.nan)
                fund_data.data_source = DataSource.POLYGON
                
                # Cache the data
                self._cache_data(ticker, date, fund_data)
                
                logger.info(f"Retrieved Polygon data for {ticker}")
            else:
                logger.warning(f"No Polygon data available for {ticker}")
                fund_data.data_source = DataSource.UNAVAILABLE
                
        except Exception as e:
            logger.error(f"Error fetching Polygon data for {ticker}: {e}")
            fund_data.data_source = DataSource.UNAVAILABLE
        
        return fund_data
    
    def get_batch_fundamentals(
        self,
        tickers: List[str],
        date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get fundamental data for multiple tickers
        
        Args:
            tickers: List of tickers
            date: Date for fundamentals
            
        Returns:
            DataFrame with fundamental data
        """
        all_data = []
        
        for ticker in tickers:
            fund_data = self.get_fundamentals(ticker, date)
            all_data.append(fund_data.to_dict())
        
        df = pd.DataFrame(all_data)
        df.set_index('ticker', inplace=True)
        
        # Remove unavailable data columns if all NaN
        df = df.dropna(axis=1, how='all')
        
        return df
    
    def prepare_fundamental_features(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Prepare fundamental features for a ticker's data
        
        Args:
            data: Price/volume data
            ticker: Stock ticker
            
        Returns:
            DataFrame with fundamental features added
        """
        # Get fundamentals
        fund_data = self.get_fundamentals(ticker)
        
        # Add to dataframe
        if fund_data.data_source != DataSource.UNAVAILABLE:
            data['book_to_market'] = fund_data.book_to_market
            data['roe'] = fund_data.roe
            data['debt_to_equity'] = fund_data.debt_to_equity
            data['pe_ratio'] = fund_data.pe_ratio
            data['earnings_per_share'] = fund_data.earnings_per_share
            
            # Fill NaN with neutral values (NOT random)
            data['book_to_market'] = data['book_to_market'].fillna(1.0)
            data['roe'] = data['roe'].fillna(0.0)
            data['debt_to_equity'] = data['debt_to_equity'].fillna(1.0)
            data['pe_ratio'] = data['pe_ratio'].fillna(15.0)  # Market average
            data['earnings_per_share'] = data['earnings_per_share'].fillna(0.0)
        else:
            # No data available - use NaN (NOT random values)
            logger.warning(f"No fundamental data for {ticker}, using NaN")
            data['book_to_market'] = np.nan
            data['roe'] = np.nan
            data['debt_to_equity'] = np.nan
            data['pe_ratio'] = np.nan
            data['earnings_per_share'] = np.nan
        
        return data
    
    def _get_cached_data(
        self,
        ticker: str,
        date: datetime
    ) -> Optional[FundamentalData]:
        """Get cached fundamental data if available and fresh"""
        cache_file = self.cache_dir / f"{ticker}_{date.strftime('%Y%m')}.pkl"
        
        if cache_file.exists():
            try:
                mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - mod_time).total_seconds() < self.cache_hours * 3600:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    return FundamentalData(**data)
            except Exception as e:
                logger.error(f"Error loading cache for {ticker}: {e}")
        
        return None
    
    def _cache_data(
        self,
        ticker: str,
        date: datetime,
        data: FundamentalData
    ) -> None:
        """Cache fundamental data"""
        cache_file = self.cache_dir / f"{ticker}_{date.strftime('%Y%m')}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data.to_dict(), f)
            logger.debug(f"Cached data for {ticker}")
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {e}")
    
    def validate_data_quality(
        self,
        data: FundamentalData
    ) -> Dict[str, Any]:
        """
        Validate data quality and return metrics
        
        Args:
            data: Fundamental data to validate
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {
            'ticker': data.ticker,
            'data_source': data.data_source.value,
            'completeness': 0.0,
            'has_critical_fields': False,
            'warnings': []
        }
        
        # Count non-None fields
        fields = ['book_to_market', 'roe', 'debt_to_equity', 'pe_ratio', 
                 'earnings_per_share', 'market_cap', 'revenue_growth', 'profit_margin']
        
        non_none_count = sum(1 for field in fields if getattr(data, field) is not None)
        metrics['completeness'] = non_none_count / len(fields)
        
        # Check critical fields
        critical_fields = ['roe', 'pe_ratio', 'book_to_market']
        has_critical = all(getattr(data, field) is not None for field in critical_fields)
        metrics['has_critical_fields'] = has_critical
        
        # Generate warnings
        if data.data_source == DataSource.UNAVAILABLE:
            metrics['warnings'].append("No data available from Polygon API")
        
        if metrics['completeness'] < 0.5:
            metrics['warnings'].append(f"Low data completeness: {metrics['completeness']:.1%}")
        
        # Check for suspicious values
        if data.pe_ratio and (data.pe_ratio < 0 or data.pe_ratio > 100):
            metrics['warnings'].append(f"Unusual P/E ratio: {data.pe_ratio}")
        
        if data.roe and (data.roe < -0.5 or data.roe > 1.0):
            metrics['warnings'].append(f"Unusual ROE: {data.roe}")
        
        return metrics


def create_fundamental_provider(
    polygon_api_key: Optional[str] = None
) -> RealFundamentalDataProvider:
    """
    Factory function to create fundamental data provider
    
    Args:
        polygon_api_key: Polygon API key
        
    Returns:
        RealFundamentalDataProvider instance
    """
    return RealFundamentalDataProvider(
        polygon_api_key=polygon_api_key
    )


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    polygon_key = os.getenv('POLYGON_API_KEY')
    
    if not polygon_key:
        print("WARNING: No Polygon API key found. Set POLYGON_API_KEY environment variable.")
        print("Data will be unavailable without API key.")
    
    # Create provider
    provider = create_fundamental_provider(polygon_key)
    
    # Test single ticker
    print("\nTesting single ticker fundamental data:")
    fund_data = provider.get_fundamentals('AAPL')
    print(f"Ticker: {fund_data.ticker}")
    print(f"Data Source: {fund_data.data_source.value}")
    print(f"ROE: {fund_data.roe}")
    print(f"P/E Ratio: {fund_data.pe_ratio}")
    print(f"Debt/Equity: {fund_data.debt_to_equity}")
    
    # Validate data quality
    print("\nData quality validation:")
    metrics = provider.validate_data_quality(fund_data)
    print(f"Completeness: {metrics['completeness']:.1%}")
    print(f"Has Critical Fields: {metrics['has_critical_fields']}")
    if metrics['warnings']:
        print("Warnings:")
        for warning in metrics['warnings']:
            print(f"  - {warning}")
    
    # Test batch
    print("\nTesting batch fundamental data:")
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    df = provider.get_batch_fundamentals(tickers)
    print(df)