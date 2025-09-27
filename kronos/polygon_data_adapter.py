#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon Data Adapter for Kronos
Replaces yfinance with Polygon API for better data quality and real-time access
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import sys
import os

# Add parent directory to path to import polygon client
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

class PolygonDataAdapter:
    """Adapter to use Polygon API for fetching stock data instead of yfinance"""

    def __init__(self):
        self.client = None
        self._init_polygon_client()

    def _init_polygon_client(self):
        """Initialize Polygon client from existing autotrader system"""
        try:
            from polygon_client import polygon_client
            self.client = polygon_client
            logger.info("Polygon client initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import polygon_client: {e}")
            self.client = None

    def get_stock_data(self,
                      symbol: str,
                      period: str = "3mo",
                      interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get stock data using Polygon API

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk')

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.client:
            logger.error("Polygon client not available. Polygon-only mode requires a valid client.")
            return None

        try:
            # Convert period to days
            period_days = self._period_to_days(period)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Get data based on interval (map to Polygon historical bars)
            if interval in ['1d', '1wk']:
                df = self._get_daily_data(symbol, start_date, end_date, interval)
            else:
                df = self._get_intraday_data(symbol, start_date, end_date, interval)

            if df is not None and not df.empty:
                # Ensure consistent column names
                df = self._standardize_columns(df)
                logger.info(f"Retrieved {len(df)} data points for {symbol} from Polygon")
                return df
            else:
                logger.warning(f"No data returned from Polygon for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching data from Polygon for {symbol}: {e}")
            return None

    def _get_daily_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Get daily or weekly data from Polygon"""
        try:
            if hasattr(self.client, 'get_historical_bars'):
                # Use historical bars method
                df = self.client.get_historical_bars(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    timespan='day' if interval == '1d' else 'week'
                )
            elif hasattr(self.client, 'get_daily_bars'):
                # Use daily bars method
                df = self.client.get_daily_bars(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            else:
                logger.warning("No suitable method found in Polygon client for daily data")
                return None

            return df

        except Exception as e:
            logger.error(f"Error getting daily data: {e}")
            return None

    def _get_intraday_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """Get intraday data via historical bars (Polygon-only, DataFrame return)."""
        try:
            interval_map = {
                '1m': ('minute', 1),
                '5m': ('minute', 5),
                '15m': ('minute', 15),
                '30m': ('minute', 30),
                '1h': ('hour', 1)
            }

            if interval not in interval_map:
                logger.warning(f"Unsupported interval {interval}, using 1d")
                return self._get_daily_data(symbol, start_date, end_date, '1d')

            timespan, multiplier = interval_map[interval]

            if hasattr(self.client, 'get_historical_bars'):
                df = self.client.get_historical_bars(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    timespan=timespan,
                    multiplier=multiplier
                )
                return df
            else:
                logger.warning("Polygon client missing get_historical_bars for intraday data")
                return None

        except Exception as e:
            logger.error(f"Error getting intraday data: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match expected format"""
        # Common column name mappings
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in data")
                # If volume is missing, estimate it
                if col == 'volume':
                    df[col] = 1000000  # Default volume
                else:
                    logger.error(f"Required price column {col} missing")
                    return pd.DataFrame()

        # Select only the required columns
        df = df[required_cols]

        # Clean data
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df[df['volume'] > 0]  # Remove zero volume bars

        return df

    def _period_to_days(self, period: str) -> int:
        """Convert period string to number of days"""
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        return period_map.get(period, 90)  # Default to 3 months

    # yfinance fallback removed to enforce Polygon-only data source
    def _fallback_to_yfinance(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        logger.error("yfinance fallback is disabled in Polygon-only mode")
        return None

    def is_us_equity(self, symbol: str) -> bool:
        """Validate via Polygon reference endpoint when possible; fallback to heuristic."""
        try:
            s = (symbol or "").strip().upper()
            if not s:
                return False
            # Prefer Polygon reference check if client/session is available
            if hasattr(self.client, 'base_url') and hasattr(self.client, 'session') and hasattr(self.client, 'api_key'):
                import requests
                url = f"{self.client.base_url}/v3/reference/tickers/{s}"
                params = {'apikey': self.client.api_key}
                r = self.client.session.get(url, params=params)
                if r.status_code == 200:
                    data = r.json()
                    res = data.get('results')
                    if res:
                        locale = res.get('locale', 'us').lower()
                        market = res.get('market', '').lower()
                        active = res.get('active', True)
                        type_code = res.get('type', 'CS').upper()
                        return (locale == 'us') and (market in ['stocks', 'otc']) and active and type_code in ['CS', 'ADR', 'ETF']
            # Fallback heuristic
            allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.-")
            if not set(s) <= allowed:
                return False
            return any(ch.isalpha() for ch in s)
        except Exception:
            return False

    def get_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time/latest data for a symbol"""
        if not self.client:
            return None

        try:
            # Try different methods to get current price
            if hasattr(self.client, 'get_current_price'):
                price = self.client.get_current_price(symbol)
                if price and price > 0:
                    return {
                        'symbol': symbol,
                        'price': price,
                        'timestamp': datetime.now()
                    }

            if hasattr(self.client, 'get_realtime_snapshot'):
                snapshot = self.client.get_realtime_snapshot(symbol)
                if snapshot and 'last_trade' in snapshot:
                    return {
                        'symbol': symbol,
                        'price': snapshot['last_trade'].get('price', 0),
                        'timestamp': datetime.now()
                    }

            return None

        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {e}")
            return None

# Global instance
polygon_adapter = PolygonDataAdapter()