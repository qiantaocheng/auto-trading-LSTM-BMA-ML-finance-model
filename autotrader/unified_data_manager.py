#!/usr/bin/env python3
"""
Unified Data Manager
Consolidates all Polygon data access across AutoTrader and BMA systems
Eliminates redundant data access patterns found in 3+ files
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import time
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Polygon client
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from polygon_client import polygon_client, download, Ticker
    POLYGON_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Polygon client not available: {e}")
    polygon_client = None
    POLYGON_AVAILABLE = False

@dataclass
class DataRequest:
    """Standardized data request structure"""
    symbol: str
    start_date: str
    end_date: str
    timeframe: str = "1Day"
    adjusted: bool = True
    data_type: str = "bars"  # bars, quotes, trades, aggregates

@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    data: Any
    timestamp: float
    ttl: float = 300.0  # 5 minutes default
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

class UnifiedDataManager:
    """
    Unified data manager to eliminate redundant Polygon access
    Consolidates data access from:
    - autotrader/unified_polygon_factors.py
    - autotrader/ibkr_auto_trader.py  
    - bma_models/bma_walkforward_enhanced.py
    """
    
    def __init__(self, cache_ttl: float = 300.0, max_cache_size: int = 1000):
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'last_reset': time.time()
        }
        
        logger.info(f"UnifiedDataManager initialized with {cache_ttl}s cache TTL")
    
    def _make_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request"""
        return f"{request.symbol}_{request.start_date}_{request.end_date}_{request.timeframe}_{request.adjusted}_{request.data_type}"
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        with self.cache_lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            for key in expired_keys:
                del self.cache[key]
            
            # Limit cache size
            if len(self.cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1].timestamp)
                to_remove = len(self.cache) - self.max_cache_size
                for key, _ in sorted_items[:to_remove]:
                    del self.cache[key]
    
    def get_market_data(self, symbol: str, days: int = 30, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Unified market data retrieval
        Replaces multiple scattered Polygon calls
        """
        self.stats['total_requests'] += 1
        
        try:
            # Prepare request
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - 
                         timedelta(days=days)).strftime('%Y-%m-%d')
            
            request = DataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Day",
                adjusted=True,
                data_type="bars"
            )
            
            cache_key = self._make_cache_key(request)
            
            # Check cache first
            with self.cache_lock:
                if cache_key in self.cache and not self.cache[cache_key].is_expired:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {symbol} market data")
                    return self.cache[cache_key].data
                
                self.stats['cache_misses'] += 1
            
            # Fetch from Polygon
            if not POLYGON_AVAILABLE or not polygon_client:
                logger.warning("Polygon client not available, returning empty DataFrame")
                return pd.DataFrame()
            
            self.stats['api_calls'] += 1
            logger.debug(f"Fetching {symbol} data from Polygon: {start_date} to {end_date}")
            
            # Use polygon_client download function  
            data = download(
                tickers=symbol,
                start=start_date,
                end=end_date
            )
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = CacheEntry(
                    data=data.copy(),
                    timestamp=time.time(),
                    ttl=self.cache_ttl
                )
            
            # Periodic cache cleanup
            if len(self.cache) > self.max_cache_size * 1.2:
                self._cleanup_cache()
            
            logger.debug(f"Successfully fetched {len(data)} bars for {symbol}")
            return data
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            # Try to get from recent data first
            data = self.get_market_data(symbol, days=5)
            if not data.empty and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def get_historical_prices(self, symbol: str, days: int = 60) -> List[float]:
        """Get historical closing prices as list"""
        try:
            data = self.get_market_data(symbol, days=days)
            if data.empty or 'Close' not in data.columns:
                return []
            return data['Close'].tolist()
        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            return []
    
    def get_ohlcv_data(self, symbol: str, days: int = 30) -> Dict[str, List[float]]:
        """Get OHLCV data as dictionary of lists"""
        try:
            data = self.get_market_data(symbol, days=days)
            if data.empty:
                return {}
            
            result = {}
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    result[col.lower()] = data[col].tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
            return {}
    
    def get_multiple_symbols(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Efficiently fetch data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_market_data(symbol, days=days)
                if not data.empty:
                    results[symbol] = data
                    
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data manager statistics"""
        with self.cache_lock:
            cache_info = {
                'cache_size': len(self.cache),
                'cache_hit_rate': (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100,
                'expired_entries': sum(1 for v in self.cache.values() if v.is_expired)
            }
        
        return {
            **self.stats,
            **cache_info,
            'uptime_seconds': time.time() - self.stats['last_reset']
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Data cache cleared")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'last_reset': time.time()
        }
        logger.info("Data manager statistics reset")

# Global instance
_data_manager = None
_data_manager_lock = threading.RLock()

def get_unified_data_manager(cache_ttl: float = 300.0) -> UnifiedDataManager:
    """Get global unified data manager instance"""
    global _data_manager
    with _data_manager_lock:
        if _data_manager is None:
            _data_manager = UnifiedDataManager(cache_ttl=cache_ttl)
        return _data_manager

def create_data_manager(cache_ttl: float = 300.0) -> UnifiedDataManager:
    """Create a new data manager instance"""
    return UnifiedDataManager(cache_ttl=cache_ttl)

# Convenience functions for backward compatibility
def get_market_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get market data (convenience function)"""
    return get_unified_data_manager().get_market_data(symbol, days)

def get_latest_price(symbol: str) -> Optional[float]:
    """Get latest price (convenience function)"""
    return get_unified_data_manager().get_latest_price(symbol)

def get_historical_prices(symbol: str, days: int = 60) -> List[float]:
    """Get historical prices (convenience function)"""
    return get_unified_data_manager().get_historical_prices(symbol, days)