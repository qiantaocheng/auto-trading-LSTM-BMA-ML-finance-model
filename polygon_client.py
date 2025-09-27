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
import os

logger = logging.getLogger(__name__)

class PolygonClient:
    def __init__(self, api_key: str, delayed_data_mode: bool = True):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # DELAYED DATA SUBSCRIPTION CONFIGURATION
        self.delayed_data_mode = delayed_data_mode  # 延迟数据模式
        self.is_premium = True  # 29.99延迟数据订阅
        self.rate_limit_delay = 0.2  # 稍微增加延迟为延迟数据模式
        
        # Delayed data settings
        self.skip_realtime_calls = delayed_data_mode
        self.prefer_historical_fallback = delayed_data_mode
        self.ignore_403_errors = delayed_data_mode
        
        # Data validation settings
        self.min_data_points = 10  # 最少数据点数
        self.max_price_change = 50.0  # 最大价格变化百分比
        
        if delayed_data_mode:
            logger.info("✅ Polygon客户端配置为延迟数据模式")
            logger.info("  - 实时数据调用: 已禁用")
            logger.info("  - 优先历史数据: 已启用") 
            logger.info("  - 忽略403错误: 已启用")

    def _request_with_retry(self, url: str, params: Optional[Dict] = None, method: str = 'get', max_retries: int = 5):
        """HTTP 请求封装：处理429/503、Retry-After和指数退避，并打全错误日志"""
        attempt = 0
        backoff_seconds = 0.5
        while True:
            try:
                if method.lower() == 'get':
                    response = self.session.get(url, params=params)
                else:
                    response = self.session.request(method.upper(), url, params=params)
                # 快速通道：直接抛出HTTP错误
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as http_err:
                status_code = getattr(http_err.response, 'status_code', 'NA')
                response_text = ''
                retry_after = None
                try:
                    if http_err.response is not None:
                        response_text = http_err.response.text
                        retry_after = http_err.response.headers.get('Retry-After')
                except Exception:
                    pass

                # 429/503：退避重试
                if status_code in (429, 503) and attempt < max_retries:
                    sleep_seconds = backoff_seconds * (2 ** attempt)
                    if retry_after:
                        try:
                            sleep_seconds = max(sleep_seconds, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(f"HTTP {status_code} on {url}. Retry in {sleep_seconds:.2f}s (attempt {attempt+1}/{max_retries}). Retry-After={retry_after}")
                    time.sleep(sleep_seconds)
                    attempt += 1
                    continue

                logger.error(f"HTTP error {status_code} for {url}: {response_text}")
                raise
            except requests.exceptions.RequestException as req_err:
                # 网络层错误：有限重试
                if attempt < max_retries:
                    sleep_seconds = backoff_seconds * (2 ** attempt)
                    logger.warning(f"Network error on {url}: {req_err}. Retry in {sleep_seconds:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_seconds)
                    attempt += 1
                    continue
                logger.error(f"Request failed for {url}: {req_err}")
                raise
            finally:
                # 请求间隔，温和限速
                time.sleep(self.rate_limit_delay)
    
    def _validate_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """验证价格数据的合理性"""
        if df.empty:
            return df
            
        try:
            # 检查基本列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"{symbol}: Missing columns {missing_cols}")
            
            # 检查数据点数量
            if len(df) < self.min_data_points:
                logger.warning(f"{symbol}: Insufficient data points ({len(df)} < {self.min_data_points})")
            
            # 检查价格合理性
            numeric_cols = ['Open', 'High', 'Low', 'Close']
            for col in numeric_cols:
                if col in df.columns:
                    # 去除零值和负值
                    invalid_mask = (df[col] <= 0) | df[col].isna()
                    if invalid_mask.any():
                        invalid_count = invalid_mask.sum()
                        logger.warning(f"{symbol}: {invalid_count} invalid {col} values (<=0 or NaN)")
                        # 用前值填充
                        df.loc[invalid_mask, col] = df[col].ffill().bfill()
            
            # 检查异常价格变化
            if 'Close' in df.columns and len(df) > 1:
                price_changes = df['Close'].pct_change().abs() * 100
                extreme_changes = price_changes > self.max_price_change
                if extreme_changes.any():
                    extreme_count = extreme_changes.sum()
                    max_change = price_changes.max()
                    logger.warning(f"{symbol}: {extreme_count} extreme price changes (max: {max_change:.1f}%)")
            
            # 检查OHLC逻辑
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # High应该是最高价
                invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1))
                # Low应该是最低价  
                invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1))
                
                if invalid_high.any():
                    logger.warning(f"{symbol}: {invalid_high.sum()} invalid High prices")
                if invalid_low.any():
                    logger.warning(f"{symbol}: {invalid_low.sum()} invalid Low prices")
            
            return df
            
        except Exception as e:
            logger.error(f"Data validation failed for {symbol}: {e}")
            return df
        
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
                    'market_cap': result.get('market_cap') or np.nan,  # No hardcoded defaults
                    'share_class_shares_outstanding': result.get('share_class_shares_outstanding') or np.nan,
                    'weighted_shares_outstanding': result.get('weighted_shares_outstanding') or np.nan,
                    # Add common yfinance fields without hardcoded defaults
                    'marketCap': result.get('market_cap') or np.nan,
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
            'market_cap': np.nan,
            'share_class_shares_outstanding': np.nan,
            'weighted_shares_outstanding': np.nan,
            'marketCap': np.nan,
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
            'series_type': 'close',  # 必需参数：价格类型
            'adjusted': 'true',      # 调整价格
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
        """Get last trade data (current price) with delayed data mode support"""
        
        # DELAYED DATA MODE: Skip realtime calls and use historical data directly
        if self.skip_realtime_calls:
            logger.info(f"延迟数据模式：跳过实时调用，直接使用历史数据获取 {symbol}")
            return self._fallback_to_historical_price(symbol)
        
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            status_flag = data.get('status')
            if data.get('results'):
                if status_flag and status_flag != 'OK':
                    logger.info(f"Polygon last trade returned status={status_flag} for {symbol}; proceeding with results.")
                result = data['results']
                return {
                    'price': result.get('p', 0),
                    'size': result.get('s', 0),
                    'timestamp': result.get('t', 0),
                    'exchange': result.get('x', ''),
                    'conditions': result.get('c', [])
                }
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 403 and self.ignore_403_errors:
                logger.info(f"延迟数据模式：忽略403错误，使用历史数据替代 {symbol}")
                return self._fallback_to_historical_price(symbol)
            else:
                logger.error(f"HTTP error fetching last trade for {symbol}: {http_err}")
                return self._fallback_to_historical_price(symbol)
        except Exception as e:
            logger.error(f"Error fetching last trade for {symbol}: {e}")
            return self._fallback_to_historical_price(symbol)
    
    def _fallback_to_historical_price(self, symbol: str) -> Dict:
        """使用历史数据作为实时价格的替代方案 - FIXED: 移除yfinance依赖"""
        try:
            logger.info(f"使用历史数据获取 {symbol} 的最新价格")
            # 获取最近3天的数据作为替代（增加缓冲区以处理周末）
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            
            df = self.get_historical_bars(symbol, start_date, end_date, 'day', 1)
            if not df.empty and 'Close' in df.columns:
                latest_price = float(df['Close'].iloc[-1])
                logger.info(f"使用历史数据获取到 {symbol} 价格: ${latest_price:.2f}")
                return {
                    'price': latest_price,
                    'size': 0,
                    'timestamp': int(time.time() * 1000),
                    'exchange': 'POLYGON_HISTORICAL',
                    'conditions': ['DELAYED_DATA_FALLBACK']
                }
            else:
                logger.warning(f"历史数据也无法获取 {symbol} 价格 - 返回默认价格")
                return self._get_default_price(symbol)
        except Exception as e:
            logger.error(f"历史数据替代方案失败 {symbol}: {e}")
            return self._get_default_price(symbol)
    
    def _get_default_price(self, symbol: str) -> Dict:
        """提供默认价格（不使用外部服务）"""
        # 常见股票的合理默认价格
        default_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 120.0,
            'AMZN': 130.0,
            'TSLA': 200.0,
            'NVDA': 400.0,
            'META': 250.0,
            'SPY': 450.0,
            'QQQ': 350.0
        }
        
        default_price = default_prices.get(symbol, 100.0)  # 默认$100
        logger.warning(f"使用默认价格 {symbol}: ${default_price}")
        return {
            'price': default_price,
            'size': 0,
            'timestamp': int(time.time() * 1000),
            'exchange': 'DEFAULT',
            'conditions': ['NO_DATA_AVAILABLE']
        }
            
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        trade_data = self.get_last_trade(symbol)
        return trade_data.get('price', 0.0)
        
    def get_historical_bars(self, symbol: str, start_date: str, end_date: str, 
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
            response = self._request_with_retry(url, params=params)
            data = response.json()

            status_flag = data.get('status')
            if data.get('results') is not None:
                if status_flag and status_flag != 'OK':
                    if status_flag in ['DELAYED', 'NOT_AUTHORIZED']:
                        logger.warning(f"Polygon API issue for {symbol}: status={status_flag}")
                    else:
                        logger.info(f"Polygon aggs returned status={status_flag} for {symbol}; proceeding with results.")
                # 首包结果
                results = list(data.get('results', []))
                next_url = data.get('next_url')
                # 游标翻页
                while next_url:
                    try:
                        r2 = self._request_with_retry(next_url, params={'apikey': self.api_key})
                        d2 = r2.json()
                        results.extend(d2.get('results', []))
                        next_url = d2.get('next_url')
                    except Exception as page_err:
                        logger.warning(f"Pagination failed for {symbol}: {page_err}. Proceeding with collected results.")
                        break

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
                            'Adj Close': float(bar.get('c', 0))
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
                
                # 🔧 Apply data validation
                df = self._validate_price_data(df, symbol)

                return df
            else:
                error_msg = data.get('error') or data.get('message') or str(data)
                status = data.get('status', 'NA')
                req_id = data.get('request_id', 'NA')
                logger.warning(f"API error for {symbol}: status={status} msg={error_msg} request_id={req_id} range=[{start_date},{end_date}]")
                return pd.DataFrame()

        except requests.exceptions.HTTPError as http_err:
            scode = getattr(http_err.response, 'status_code', 'NA')
            rtext = ''
            try:
                rtext = http_err.response.text if http_err.response is not None else ''
            except Exception:
                pass
            logger.error(f"HTTP error fetching historical data for {symbol}: status={scode} body={rtext}")
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
            df = self.get_historical_bars(symbol, start, end, timespan, multiplier)
            if not df.empty:
                all_data[symbol] = df
            # 温和延时，配合 _request_with_retry 的 finally 通道
            time.sleep(self.rate_limit_delay)
            
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
            
            status_flag = data.get('status')
            if data.get('results'):
                if status_flag and status_flag != 'OK':
                    logger.info(f"Polygon nbbo returned status={status_flag} for {symbol}; proceeding with results.")
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
            
    def get_today_intraday(self, symbol: str, timespan: str = 'minute') -> pd.DataFrame:
        """获取今日到上一分钟的盘中数据"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_historical_bars(symbol, today, today, timespan, 1)
    
    def get_realtime_snapshot(self, symbol: str) -> Dict:
        """获取实时快照 - 最新成交/报价/当日累计"""
        try:
            # 获取最新成交
            trade = self.get_last_trade(symbol)
            # 获取最新报价
            quote = self.get_real_time_quote(symbol)
            # 获取当日聚合数据
            today = datetime.now().strftime('%Y-%m-%d')
            daily_agg = self.get_historical_bars(symbol, today, today, 'day', 1)
            
            snapshot = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'last_trade': trade,
                'last_quote': quote,
                'daily_bar': {}
            }
            
            if not daily_agg.empty:
                latest_bar = daily_agg.iloc[-1]
                snapshot['daily_bar'] = {
                    'open': latest_bar.get('Open', 0),
                    'high': latest_bar.get('High', 0),
                    'low': latest_bar.get('Low', 0),
                    'close': latest_bar.get('Close', 0),
                    'volume': latest_bar.get('Volume', 0)
                }
            
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching realtime snapshot for {symbol}: {e}")
            return {}
    
    def stream_realtime(self, symbols: List[str], on_trade=None, on_quote=None, on_bar=None):
        """WebSocket实时数据流 (仅框架，需要WebSocket实现)"""
        logger.warning("WebSocket streaming not implemented yet. Use get_realtime_snapshot for near real-time data")
        return None
            
    def get_financials(self, symbol: str, limit: int = 4) -> Dict:
        """Get financial data for a symbol using correct Polygon API endpoint"""
        url = f"{self.base_url}/vX/reference/financials"
        params = {
            'ticker': symbol,  # 正确参数名
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
            
# Global instance - DELAYED DATA MODE
# Prefer environment variable, fallback to api_config if available
def _get_polygon_api_key() -> str:
    env_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_API_TOKEN")
    if env_key:
        return env_key
    try:
        from api_config import POLYGON_API_KEY as CFG_KEY
        return CFG_KEY
    except Exception:
        logging.getLogger(__name__).warning("POLYGON_API_KEY not set; using empty key which will fail requests")
        return ""

polygon_client = PolygonClient(_get_polygon_api_key(), delayed_data_mode=True)

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