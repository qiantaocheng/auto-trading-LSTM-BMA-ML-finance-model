"""
Polygon.io API client for real-time and historical stock data
Replaces all stock data sources with Polygon.io API
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)

class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # 付费订阅功能标识
        self.is_premium = True  # 29.99订阅
        self.rate_limit_delay = 0.1  # 付费版更宽松的限制
        
    def get_ticker_details(self, symbol: str) -> Dict:
        """Get ticker details (replaces yf.Ticker().info)"""
        url = f"{self.base_url}/v3/reference/tickers/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                # Convert to compatible format with fallbacks
                return {
                    'symbol': result.get('ticker', symbol),
                    'shortName': result.get('name', symbol),
                    'longName': result.get('name', symbol),
                    'name': result.get('name', symbol),
                    'market': result.get('market', 'stocks'),
                    'locale': result.get('locale', 'us'),
                    'type': result.get('type', 'CS'),
                    'active': result.get('active', True),
                    'currency_name': result.get('currency_name', 'USD'),
                    'description': result.get('description', ''),
                    'homepage_url': result.get('homepage_url', ''),
                    'total_employees': result.get('total_employees', 0),
                    'market_cap': result.get('market_cap', 1000000000),  # Default 1B
                    'share_class_shares_outstanding': result.get('share_class_shares_outstanding', 1000000),
                    'weighted_shares_outstanding': result.get('weighted_shares_outstanding', 1000000),
                    # Add common yfinance fields with defaults
                    'marketCap': result.get('market_cap', 1000000000),
                    'sector': 'Technology',  # Default sector since Polygon doesn't provide
                    'industry': 'Software',  # Default industry
                    'country': result.get('locale', 'us').upper()
                }
            else:
                logger.warning(f"No data found for {symbol} in Polygon response")
                return self._get_fallback_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {e}")
            return self._get_fallback_info(symbol)
            
    def _get_fallback_info(self, symbol: str) -> Dict:
        """Provide fallback info when API fails"""
        return {
            'symbol': symbol,
            'shortName': symbol,
            'longName': symbol,
            'name': symbol,
            'market': 'stocks',
            'locale': 'us',
            'type': 'CS',
            'active': True,
            'currency_name': 'USD',
            'description': '',
            'homepage_url': '',
            'total_employees': 0,
            'market_cap': 1000000000,
            'share_class_shares_outstanding': 1000000,
            'weighted_shares_outstanding': 1000000,
            'marketCap': 1000000000,
            'sector': 'Technology',
            'industry': 'Software',
            'country': 'US'
        }
    
    # ===============================
    # 付费订阅专属功能
    # ===============================
    
    def get_real_time_trades(self, symbol: str, limit: int = 1000) -> Dict:
        """获取实时交易数据（付费功能）"""
        if not self.is_premium:
            logger.warning("实时交易数据需要付费订阅")
            return {}
            
        url = f"{self.base_url}/v3/trades/{symbol}"
        params = {
            'apikey': self.api_key,
            'limit': limit,
            'timestamp.gte': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching real-time trades for {symbol}: {e}")
            return {}
    
    def get_options_chain(self, underlying_symbol: str, expiration_date: str = None) -> Dict:
        """获取期权链数据（付费功能）"""
        if not self.is_premium:
            logger.warning("期权数据需要付费订阅")
            return {}
            
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            'apikey': self.api_key,
            'underlying_ticker': underlying_symbol,
            'limit': 1000
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching options chain for {underlying_symbol}: {e}")
            return {}
    
    def get_ticker_news(self, symbol: str, limit: int = 10, 
                       published_utc_gte: str = None) -> Dict:
        """获取股票相关新闻（付费功能）"""
        if not self.is_premium:
            logger.warning("新闻数据需要付费订阅")
            return {}
            
        url = f"{self.base_url}/v2/reference/news"
        params = {
            'apikey': self.api_key,
            'ticker': symbol,
            'limit': limit
        }
        
        if published_utc_gte:
            params['published_utc.gte'] = published_utc_gte
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return {}
    
    def get_sma_indicator(self, symbol: str, timespan: str = "day", 
                         window: int = 50, limit: int = 100) -> Dict:
        """获取SMA技术指标数据"""
        url = f"{self.base_url}/v1/indicators/sma/{symbol}"
        params = {
            'apikey': self.api_key,
            'timespan': timespan,
            'window': window,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching SMA for {symbol}: {e}")
            return {}
    
    def get_intraday_bars(self, symbol: str, multiplier: int = 5, 
                         timespan: str = "minute", date: str = None) -> Dict:
        """获取日内分钟级数据（付费功能）"""
        if not self.is_premium:
            logger.warning("日内数据需要付费订阅")
            return {}
            
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{date}/{date}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 5000
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching intraday bars for {symbol}: {e}")
            return {}
            
    def get_last_trade(self, symbol: str) -> Dict:
        """Get last trade data (current price)"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                return {
                    'price': result.get('p', 0),
                    'size': result.get('s', 0),
                    'timestamp': result.get('t', 0),
                    'exchange': result.get('x', ''),
                    'conditions': result.get('c', [])
                }
        except Exception as e:
            logger.error(f"Error fetching last trade for {symbol}: {e}")
            return {'price': 0}
            
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        trade_data = self.get_last_trade(symbol)
        return trade_data.get('price', 0.0)
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          timespan: str = "day", multiplier: int = 1) -> pd.DataFrame:
        """
        Get historical data (replaces yf.download())
        timespan: minute, hour, day, week, month, quarter, year
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                results = data['results']
                if not results:
                    logger.warning(f"Empty results for {symbol} from {start_date} to {end_date}")
                    return pd.DataFrame()
                    
                df_data = []
                
                for bar in results:
                    try:
                        df_data.append({
                            'Date': pd.to_datetime(bar.get('t', 0), unit='ms'),
                            'Open': float(bar.get('o', 0)),
                            'High': float(bar.get('h', 0)),
                            'Low': float(bar.get('l', 0)),
                            'Close': float(bar.get('c', 0)),
                            'Volume': int(bar.get('v', 0)),
                            'Adj Close': float(bar.get('c', 0))  # Polygon provides adjusted data
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid data point for {symbol}: {e}")
                        continue
                
                if not df_data:
                    logger.warning(f"No valid data points for {symbol}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.index.name = 'Date'
                
                # Add symbol column for multi-symbol compatibility
                df['Symbol'] = symbol
                
                return df
            else:
                error_msg = data.get('error', 'Unknown error')
                logger.warning(f"API error for {symbol}: {error_msg}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    def download(self, symbols: Union[str, List[str]], start: str, end: str, 
                interval: str = "1d", **kwargs) -> pd.DataFrame:
        """
        Replaces yfinance download function
        interval mapping: 1m->minute, 1h->hour, 1d->day, 1wk->week, 1mo->month
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Convert interval to Polygon timespan
        interval_mapping = {
            '1m': ('minute', 1),
            '5m': ('minute', 5),
            '15m': ('minute', 15),
            '30m': ('minute', 30),
            '1h': ('hour', 1),
            '1d': ('day', 1),
            '1wk': ('week', 1),
            '1mo': ('month', 1)
        }
        
        timespan, multiplier = interval_mapping.get(interval, ('day', 1))
        
        all_data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start, end, timespan, multiplier)
            if not df.empty:
                all_data[symbol] = df
            time.sleep(0.1)  # Rate limiting
            
        if len(all_data) == 1:
            return list(all_data.values())[0]
        elif len(all_data) > 1:
            # Combine multiple symbols
            combined = pd.concat(all_data.values(), keys=all_data.keys())
            return combined
        else:
            return pd.DataFrame()
            
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote data"""
        url = f"{self.base_url}/v2/last/nbbo/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results']
                return {
                    'bid': result.get('P', 0),
                    'ask': result.get('p', 0),
                    'bid_size': result.get('S', 0),
                    'ask_size': result.get('s', 0),
                    'timestamp': result.get('t', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return {}
            
    def get_market_status(self) -> Dict:
        """Get market status"""
        url = f"{self.base_url}/v1/marketstatus/now"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {}
            
    def get_financials(self, symbol: str, limit: int = 4) -> Dict:
        """Get financial data for a symbol"""
        url = f"{self.base_url}/v3/reference/tickers/{symbol}/financials"
        params = {
            'apikey': self.api_key,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {}
            
# Global instance
polygon_client = PolygonClient("FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1")

# Compatibility functions
def download(tickers, start=None, end=None, **kwargs):
    """Drop-in replacement for yf.download()"""
    return polygon_client.download(tickers, start, end, **kwargs)
    
class Ticker:
    """Drop-in replacement for yf.Ticker()"""
    def __init__(self, symbol):
        self.symbol = symbol
        self._info = None
        
    @property
    def info(self):
        if self._info is None:
            self._info = polygon_client.get_ticker_details(self.symbol)
        return self._info
        
    def history(self, start=None, end=None, period="1y", interval="1d"):
        if start is None or end is None:
            if period == "1y":
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            elif period == "6mo":
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            elif period == "3mo":
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            elif period == "1mo":
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                
        return polygon_client.download(self.symbol, start, end, interval=interval)
        
    def get_info(self):
        return self.info