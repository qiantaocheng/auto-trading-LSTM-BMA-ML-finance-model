#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR自动交易策略增强版 - 基于BMA模型输出的均值回归策略
结合BMA量化模型的Top10推荐和统计套利原理

核心改进：
1. 完整的买入/卖出/止盈止损逻辑
2. 修复技术指标计算bug
3. 组合层面风险控制
4. 批量数据下载优化
5. 模块化架构

Authors: AI Assistant
Version: 2.0
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
from collections import defaultdict, deque

# IBKR API相关导入
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import BarData
    IBKR_AVAILABLE = True
    print("IBKR API已加载")
except ImportError as e:
    print(f"警告: IBKR API导入失败 ({e})，尝试使用ib_insync")
    IBKR_AVAILABLE = False
    # 创建占位符类以避免NameError
    class Contract:
        def __init__(self):
            self.symbol = ""
            self.secType = "STK"
            self.exchange = "SMART"
            self.currency = "USD"

# 备用导入 ib_insync
try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
    if not IBKR_AVAILABLE:
        print("使用ib_insync作为IBKR接口")
        IBKR_AVAILABLE = True  # 设置为可用
except ImportError:
    IB_INSYNC_AVAILABLE = False
    if not IBKR_AVAILABLE:
        print("错误: 没有可用的IBKR接口")

# 尝试导入talib和yfinance
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("警告: talib不可用，将使用自定义指标计算")

try:
    import yfinance as yf
    YF_AVAILABLE = True
    if not IBKR_AVAILABLE:
        print("警告: 将使用yfinance作为数据源")
except ImportError:
    YF_AVAILABLE = False


class TradingSignal:
    """交易信号类"""
    
    def __init__(self, symbol: str, action: str, quantity: int, 
                 signal_strength: float, reason: str, timestamp: datetime = None):
        self.symbol = symbol
        self.action = action  # BUY, SELL, HOLD
        self.quantity = quantity
        self.signal_strength = signal_strength  # 0-1
        self.reason = reason
        self.timestamp = timestamp or datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'signal_strength': self.signal_strength,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


class TechnicalIndicators:
    """技术指标计算模块"""
    
    @staticmethod
    def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI - 使用Wilder指数加权移动平均"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # 默认中性值
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: pd.Series, window: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        sma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    @staticmethod
    def calculate_zscore(close_prices: pd.Series, window: int = 20) -> pd.Series:
        """计算Z-Score序列"""
        rolling_mean = close_prices.rolling(window=window).mean()
        rolling_std = close_prices.rolling(window=window).std()
        zscore = (close_prices - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore.fillna(0)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算ATR (Average True Range)"""
        if TALIB_AVAILABLE:
            return talib.ATR(high.values, low.values, close.values, timeperiod=window)
        else:
            # 手动计算ATR
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr.fillna(true_range.mean())
    
    @staticmethod
    def calculate_ma(close_prices: pd.Series, window: int = 200) -> pd.Series:
        """计算移动平均线"""
        return close_prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算ADX (Average Directional Index)"""
        if TALIB_AVAILABLE:
            return talib.ADX(high.values, low.values, close.values, timeperiod=window)
        else:
            # 简化版ADX计算
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
            
            tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
            
            plus_di = 100 * (plus_dm.rolling(window).mean() / tr.rolling(window).mean())
            minus_di = 100 * (minus_dm.rolling(window).mean() / tr.rolling(window).mean())
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window).mean()
            
            return adx.fillna(25)  # 默认中性值

if IBKR_AVAILABLE and not IB_INSYNC_AVAILABLE:
    # 使用原生IBKR API
    class IBKRDataApp(EWrapper, EClient):
        """IBKR数据获取应用"""
        
        def __init__(self):
            EClient.__init__(self, self)
            self.nextOrderId = None
            self.historical_data = {}
            self.market_data = {}
            self.data_ready = {}
            self.account_info = {}
            self.positions_data = []
            
        def nextValidId(self, orderId: int):
            self.nextOrderId = orderId
            
        def historicalData(self, reqId: int, bar):
            """接收历史数据"""
            if reqId not in self.historical_data:
                self.historical_data[reqId] = []
            self.historical_data[reqId].append({
                'Date': bar.date,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume
            })
            
        def historicalDataEnd(self, reqId: int, start: str, end: str):
            """历史数据结束"""
            self.data_ready[reqId] = True
            
        def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
            """实时价格数据"""
            if reqId not in self.market_data:
                self.market_data[reqId] = {}
            self.market_data[reqId][tickType] = price
            
        def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
            """账户信息"""
            self.account_info[tag] = {'value': value, 'currency': currency}
            
        def position(self, account: str, contract: Contract, position: float, avgCost: float):
            """持仓信息"""
            self.positions_data.append({
                'symbol': contract.symbol,
                'position': position,
                'avgCost': avgCost
            })
else:
    # 使用ib_insync或者创建占位符
    class IBKRDataApp:
        """IBKR数据获取应用（备选版本）"""
        
        def __init__(self):
            self.nextOrderId = None
            self.historical_data = {}
            self.market_data = {}
            self.data_ready = {}
            self.account_info = {}
            self.positions_data = []
            self.ib = None
            
        def connect(self, host, port, client_id):
            if IB_INSYNC_AVAILABLE:
                self.ib = IB()
                try:
                    self.ib.connect(host, port, clientId=client_id)
                    return True
                except:
                    return False
            return False
            
        def isConnected(self):
            if self.ib:
                return self.ib.isConnected()
            return False

class DataService:
    """数据服务模块 - 使用IBKR API"""
    
    def __init__(self, logger):
        self.logger = logger
        self._data_cache = {}
        self.ibkr_app = None
        self.ibkr_connected = False
        
    def connect_ibkr(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 50310) -> bool:
        """连接IBKR"""
        try:
            if IBKR_AVAILABLE:
                self.ibkr_app = IBKRDataApp()
                
                if IB_INSYNC_AVAILABLE:
                    # 使用ib_insync
                    success = self.ibkr_app.connect(host, port, client_id)
                    if success:
                        self.ibkr_connected = True
                        self.logger.info(f"IBKR数据连接成功 (ib_insync): {host}:{port}")
                        return True
                else:
                    # 使用原生API
                    self.ibkr_app.connect(host, port, client_id)
                    
                    # 启动消息循环
                    import threading
                    thread = threading.Thread(target=self.ibkr_app.run, daemon=True)
                    thread.start()
                    
                    # 等待连接
                    import time
                    time.sleep(2)
                    
                    if self.ibkr_app.isConnected():
                        self.ibkr_connected = True
                        self.logger.info(f"IBKR数据连接成功 (原生API): {host}:{port}")
                        return True
                        
            self.logger.warning("IBKR不可用，将使用yfinance作为数据源")
            return YF_AVAILABLE  # 如果yfinance可用，则返回True
                
        except Exception as e:
            self.logger.error(f"IBKR数据连接异常: {e}")
            if YF_AVAILABLE:
                self.logger.info("降级使用yfinance作为数据源")
                return True
            return False
    
    def create_contract(self, symbol: str):
        """创建股票合约"""
        try:
            if IBKR_AVAILABLE:
                from ibapi.contract import Contract as IBContract
                contract = IBContract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                return contract
            else:
                # 使用占位符Contract类
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                return contract
        except ImportError:
            # 如果导入失败，使用占位符
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            return contract
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """获取当前IBKR持仓"""
        try:
            if not self.ibkr_connected:
                return {}
                
            self.ibkr_app.reqPositions()
            
            # 等待持仓数据
            import time
            time.sleep(3)
            
            positions = {}
            for pos in self.ibkr_app.positions_data:
                if pos['position'] != 0:  # 只返回非零持仓
                    positions[pos['symbol']] = {
                        'shares': int(pos['position']),
                        'avg_price': float(pos['avgCost'])
                    }
            
            # 清空数据，避免累积
            self.ibkr_app.positions_data = []
            
            return positions
            
        except Exception as e:
            self.logger.error(f"获取IBKR持仓失败: {e}")
            return {}
    
    def get_account_info(self, account: str = "c2dvdongg") -> Dict:
        """获取IBKR账户信息"""
        try:
            if not self.ibkr_connected:
                return {}
            
            if IB_INSYNC_AVAILABLE and self.ibkr_app.ib:
                # 使用ib_insync获取账户信息
                account_values = self.ibkr_app.ib.accountValues()
                positions = self.ibkr_app.ib.positions()
                
                # 过滤指定账户
                if account:
                    account_values = [av for av in account_values if av.account == account]
                    positions = [pos for pos in positions if pos.account == account]
                
                # 提取余额信息
                balance_info = {}
                for av in account_values:
                    if av.tag in ['NetLiquidation', 'TotalCashValue', 'AvailableFunds', 'BuyingPower']:
                        balance_info[av.tag] = {
                            'value': float(av.value),
                            'currency': av.currency
                        }
                
                return {
                    'account_values': account_values,
                    'positions': positions,
                    'balance': balance_info,
                    'account': account,
                    'client_id': 50310,
                    'connected': True
                }
            else:
                # 使用原生API
                self.ibkr_app.reqAccountSummary(4001, account, "$LEDGER")
                
                # 等待账户数据
                import time
                time.sleep(3)
                
                # 提取余额信息
                balance_info = {}
                for tag, info in self.ibkr_app.account_info.items():
                    if tag in ['NetLiquidation', 'TotalCashValue', 'AvailableFunds', 'BuyingPower']:
                        try:
                            balance_info[tag] = {
                                'value': float(info['value']),
                                'currency': info['currency']
                            }
                        except:
                            continue
                
                return {
                    'account_info': self.ibkr_app.account_info,
                    'balance': balance_info,
                    'account': account,
                    'client_id': 50310,
                    'connected': True
                }
            
        except Exception as e:
            self.logger.error(f"获取IBKR账户信息失败: {e}")
            return {'connected': False, 'error': str(e)}
    
    def _download_via_ibkr(self, tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
        """通过IBKR API下载数据"""
        try:
            # 解析period参数
            if period == '3mo':
                days = 90
            elif period == '1mo': 
                days = 30
            elif period == '6mo':
                days = 180
            elif period == '1y':
                days = 365
            else:
                days = 90  # 默认3个月
                
            result = {}
            import time
            
            for i, ticker in enumerate(tickers):
                try:
                    # 创建合约
                    contract = self.create_contract(ticker)
                    req_id = 2000 + i
                    
                    # 清空之前的数据
                    if req_id in self.ibkr_app.historical_data:
                        del self.ibkr_app.historical_data[req_id]
                    if req_id in self.ibkr_app.data_ready:
                        del self.ibkr_app.data_ready[req_id]
                    
                    # 请求历史数据
                    duration = f"{days} D"
                    self.ibkr_app.reqHistoricalData(
                        req_id, contract, "", duration, "1 day", "TRADES", 1, 1, False, []
                    )
                    
                    # 等待数据
                    timeout = 10
                    start_time = time.time()
                    while req_id not in self.ibkr_app.data_ready and (time.time() - start_time) < timeout:
                        time.sleep(0.1)
                    
                    if req_id in self.ibkr_app.historical_data and self.ibkr_app.historical_data[req_id]:
                        # 转换为DataFrame
                        data = self.ibkr_app.historical_data[req_id]
                        df = pd.DataFrame(data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        
                        result[ticker] = df
                        self.logger.debug(f"IBKR成功下载 {ticker}: {len(df)} 天数据")
                    else:
                        self.logger.warning(f"IBKR无法获取 {ticker} 的数据")
                        
                    # 避免请求过快
                    time.sleep(0.2)
                        
                except Exception as e:
                    self.logger.error(f"IBKR下载 {ticker} 数据时出错: {e}")
                    continue
            
            return result
            
        except Exception as e:
            self.logger.error(f"IBKR批量下载失败: {e}")
            return {}
    
    def _download_via_yfinance(self, tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
        """通过yfinance下载数据"""
        try:
            result = {}
            
            if len(tickers) == 1:
                # 单只股票
                data = yf.download(tickers[0], period=period, progress=False)
                if not data.empty:
                    result[tickers[0]] = data
            else:
                # 批量下载
                data = yf.download(tickers, period=period, group_by='ticker', progress=False)
                
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            ticker_data = data[ticker]
                            if not ticker_data.empty:
                                result[ticker] = ticker_data
                    except:
                        # 如果批量失败，尝试单独下载
                        try:
                            single_data = yf.download(ticker, period=period, progress=False)
                            if not single_data.empty:
                                result[ticker] = single_data
                        except:
                            continue
            
            return result
            
        except Exception as e:
            self.logger.error(f"yfinance下载失败: {e}")
            return {}
        
    def load_bma_recommendations(self) -> List[Dict]:
        """加载BMA推荐数据 - 支持周度BMA和传统BMA"""
        # 首先尝试加载周度BMA数据
        weekly_data = self._load_weekly_bma_data()
        if weekly_data:
            return weekly_data
        
        # 如果没有周度数据，回退到传统方法
        return self._load_legacy_bma_data()
    
    def _load_weekly_bma_data(self) -> List[Dict]:
        """加载周度BMA推荐数据"""
        try:
            bma_file = 'weekly_bma_trading.json'
            
            if not os.path.exists(bma_file):
                return []  # 静默返回，不记录警告
            
            with open(bma_file, 'r', encoding='utf-8') as f:
                bma_data = json.load(f)
            
            recommendations = bma_data.get('recommendations', [])
            config = bma_data.get('config', {})
            
            # 应用价格过滤和评级过滤
            filtered_recommendations = []
            for rec in recommendations:
                # 只选择BUY和STRONG_BUY推荐
                if rec.get('rating') in ['BUY', 'STRONG_BUY']:
                    current_price = rec.get('current_price', 0)
                    min_price = config.get('min_price_threshold', 0)
                    max_price = config.get('max_price_threshold', float('inf'))
                    
                    if min_price <= current_price <= max_price:
                        # 转换格式以兼容现有代码
                        formatted_rec = {
                            'ticker': rec.get('ticker'),
                            'rating': rec.get('rating'),
                            'predicted_return': rec.get('predicted_return', 0),
                            'confidence': rec.get('confidence', 0),
                            'current_price': current_price,
                            'target_price': rec.get('target_price', current_price),
                            'stop_loss': rec.get('stop_loss', current_price * 0.95),
                            'ranking': rec.get('ranking', 0)
                        }
                        filtered_recommendations.append(formatted_rec)
            
            self.logger.info(f"✅ 加载周度BMA推荐: {len(filtered_recommendations)} 只股票")
            if config.get('timestamp'):
                self.logger.info(f"📅 BMA数据时间: {config['timestamp']}")
            
            return filtered_recommendations
            
        except Exception as e:
            self.logger.debug(f"周度BMA数据加载失败: {e}")
            return []
    
    def _load_legacy_bma_data(self) -> List[Dict]:
        """读取传统BMA模型的Top10推荐（向后兼容）"""
        try:
            ibkr_dir = "ibkr_trading"
            if not os.path.exists(ibkr_dir):
                self.logger.error("未找到IBKR交易目录")
                return []
            
            # 使用正则表达式匹配文件名格式
            pattern = r'top_10_stocks_(\d{8}_\d{6})\.json'
            json_files = []
            
            for filename in os.listdir(ibkr_dir):
                match = re.match(pattern, filename)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        json_files.append((filename, timestamp))
                    except ValueError:
                        continue
            
            if not json_files:
                self.logger.error("未找到符合格式的BMA模型输出文件")
                return []
            
            # 按时间戳排序，取最新的
            latest_file = max(json_files, key=lambda x: x[1])[0]
            file_path = os.path.join(ibkr_dir, latest_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                recommendations = json.load(f)
            
            # 只选择BUY推荐
            buy_recommendations = [stock for stock in recommendations if stock['rating'] == 'BUY']
            
            self.logger.info(f"加载传统BMA推荐: {len(buy_recommendations)} 个BUY信号 (文件: {latest_file})")
            return buy_recommendations
            
        except Exception as e:
            self.logger.error(f"加载传统BMA推荐失败: {e}")
            return []
    
    def download_stock_data_batch(self, tickers: List[str], period: str = '3mo') -> Dict[str, pd.DataFrame]:
        """批量下载股票数据"""
        try:
            self.logger.info(f"批量下载 {len(tickers)} 只股票数据...")
            
            if self.ibkr_connected and IBKR_AVAILABLE:
                # 使用IBKR API
                result = self._download_via_ibkr(tickers, period)
                self.logger.info(f"IBKR批量下载完成: {len(result)}/{len(tickers)} 成功")
                return result
            elif YF_AVAILABLE:
                # 使用yfinance
                result = self._download_via_yfinance(tickers, period)
                self.logger.info(f"yfinance批量下载完成: {len(result)}/{len(tickers)} 成功")
                return result
            else:
                self.logger.error("没有可用的数据源")
                return {}
            
        except Exception as e:
            self.logger.error(f"批量下载失败: {e}")
            # 降级到单独下载
            return self._download_individually(tickers, period)
    
    def _download_individually(self, tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
        """降级：单独下载每只股票"""
        if self.ibkr_connected and IBKR_AVAILABLE:
            self.logger.info("使用IBKR API单独下载模式...")
            result = {}
            
            for ticker in tickers:
                try:
                    single_result = self._download_via_ibkr([ticker], period)
                    if ticker in single_result:
                        result[ticker] = single_result[ticker]
                except Exception as e:
                    self.logger.error(f"IBKR下载 {ticker} 失败: {e}")
                    continue
            
            return result
        elif YF_AVAILABLE:
            self.logger.info("使用yfinance单独下载模式...")
            return self._download_via_yfinance(tickers, period)
        else:
            self.logger.error("没有可用的数据源")
            return {}

class SignalGenerator:
    """信号生成模块"""
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        
        # 优化的技术指标参数
        self.rsi_period = config.get('rsi_period', 3)  # 缩短RSI周期到3
        self.rsi_oversold = config.get('rsi_oversold', 10)  # 更严格的超卖阈值
        self.rsi_overbought = config.get('rsi_overbought', 90)  # 更严格的超买阈值
        self.bb_std = config.get('bollinger_std', 2.0)
        self.lookback_days = config.get('lookback_days', 20)
        self.zscore_threshold = config.get('zscore_threshold', 3.0)  # 提高到3.0
        
        # BMA + 技术分析组合参数
        self.bma_score_threshold = config.get('bma_score_threshold', 0.65)  # BMA评分阈值
        self.use_trend_filter = config.get('use_trend_filter', True)  # 启用趋势过滤
        self.trend_deviation_threshold = config.get('trend_deviation_threshold', 0.15)  # 15%趋势偏差阈值
        self.adx_trend_threshold = config.get('adx_trend_threshold', 25)  # ADX趋势强度阈值
        self.min_conditions_met = config.get('min_conditions_met', 2)  # 最少满足条件数
        self.require_technical_confirmation = config.get('require_technical_confirmation', True)  # 要求技术确认
        self.min_signal_count = config.get('min_signal_count', 2)  # 最少信号计数
        
        # ATR动态止盈止损参数
        self.atr_period = config.get('atr_period', 14)  # ATR计算周期
        self.stop_loss_multiplier = config.get('stop_loss_multiplier', 1.0)  # 止损ATR倍数
        self.take_profit_multiplier = config.get('take_profit_multiplier', 0.5)  # 止盈ATR倍数
        self.use_trailing_stop = config.get('use_trailing_stop', True)  # 启用追踪止损
        self.trailing_atr_multiplier = config.get('trailing_atr_multiplier', 0.5)  # 追踪止损ATR倍数
        
        # 固定止盈止损比率优化
        self.profit_target_pct = config.get('profit_target_pct', 0.04)  # 固定止盈4%
        self.stop_loss_pct = config.get('stop_loss_pct', 0.08)  # 固定止损8%（原6%）
        self.risk_reward_ratio = config.get('risk_reward_ratio', 0.5)  # 风险收益比
        
        # 信号强度参数
        self.signal_stability_days = config.get('signal_stability_days', 3)  # 信号稳定性检查
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标 - 修复pandas Series比较问题"""
        try:
            close = data['Close']
            
            # 确保有足够的数据
            min_data_points = max(14, self.lookback_days, self.signal_stability_days)
            if len(close) < min_data_points:
                self.logger.warning(f"数据不足，需要至少{min_data_points}天数据，当前{len(close)}天")
                return {}
            
            # 计算技术指标
            rsi = TechnicalIndicators.calculate_rsi(close, window=self.rsi_period)
            bb_upper, bb_lower, sma = TechnicalIndicators.calculate_bollinger_bands(close, self.lookback_days, self.bb_std)
            zscore = TechnicalIndicators.calculate_zscore(close, self.lookback_days)
            
            # 计算趋势过滤指标
            ma_200 = close.rolling(window=200).mean() if len(close) >= 200 else close.rolling(window=len(close)).mean()
            if len(data) >= 14 and all(col in data.columns for col in ['High', 'Low']):
                adx = TechnicalIndicators.calculate_adx(data['High'], data['Low'], close, window=14)
                atr = TechnicalIndicators.calculate_atr(data['High'], data['Low'], close, window=self.atr_period)
            else:
                adx = pd.Series([25.0] * len(close), index=close.index)  # 默认值
                atr = pd.Series([close.std() * 0.1] * len(close), index=close.index)  # 基于波动率的估算
            
            # 安全获取最新值 - 使用.iloc[-1]然后转换为标量
            def safe_get_scalar(series, default=0):
                try:
                    value = series.iloc[-1]
                    return float(value) if pd.notna(value) else default
                except (IndexError, ValueError, TypeError):
                    return default
            
            current_price = safe_get_scalar(close)
            current_rsi = safe_get_scalar(rsi, 50.0)
            current_sma = safe_get_scalar(sma, current_price)
            current_bb_upper = safe_get_scalar(bb_upper, current_price)
            current_bb_lower = safe_get_scalar(bb_lower, current_price)
            current_zscore = safe_get_scalar(zscore, 0.0)
            current_ma_200 = safe_get_scalar(ma_200, current_price)
            current_adx = safe_get_scalar(adx, 25.0)
            current_atr = safe_get_scalar(atr, current_price * 0.02)
            
            # 计算其他指标
            volatility = safe_get_scalar(close.rolling(window=self.lookback_days).std(), 0.0)
            price_deviation = (current_price - current_sma) / current_sma if current_sma != 0 else 0.0
            volatility_ratio = volatility / current_sma if current_sma != 0 else 0.0
            
            # 计算信号稳定性 - 检查近几日信号一致性
            stability_window = min(self.signal_stability_days, len(close))
            rsi_stable = (rsi.iloc[-stability_window:] < self.rsi_oversold).sum() >= stability_window // 2
            zscore_stable = (zscore.iloc[-stability_window:] < -self.zscore_threshold).sum() >= stability_window // 2
            
            indicators = {
                'current_price': current_price,
                'rsi': current_rsi,
                'bb_upper': current_bb_upper,
                'bb_lower': current_bb_lower,
                'sma': current_sma,
                'zscore': current_zscore,
                'ma_200': current_ma_200,
                'adx': current_adx,
                'atr': current_atr,
                'price_deviation': price_deviation,
                'volatility': volatility,
                'volatility_ratio': volatility_ratio,
                'rsi_stable': rsi_stable,
                'zscore_stable': zscore_stable,
                
                # 历史序列（用于进一步分析）
                'rsi_series': rsi,
                'zscore_series': zscore,
                'close_series': close,
                'atr_series': atr
            }
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}
    
    def analyze_buy_signals(self, ticker: str, stock_data: pd.DataFrame) -> Dict:
        """分析买入信号 - 增强版过滤条件"""
        try:
            indicators = self.calculate_technical_indicators(stock_data)
            if not indicators:
                return {'action': 'HOLD', 'confidence': 0, 'reason': '技术指标计算失败'}
            
            # 趋势过滤 - 优先检查
            trend_filter_result = self._apply_trend_filter(indicators)
            if trend_filter_result['blocked']:
                return {'action': 'HOLD', 'confidence': 0, 'reason': trend_filter_result['reason']}
            
            # 技术条件检查 - 需要满足多个条件
            technical_conditions = self._evaluate_technical_conditions(indicators)
            signal_count = technical_conditions['count']
            confidence = technical_conditions['confidence']
            signals = technical_conditions['signals']
            
            # 检查最小信号数量要求
            if signal_count < self.min_signal_count:
                return {
                    'action': 'HOLD', 
                    'confidence': confidence, 
                    'reason': f'技术条件不足({signal_count}/{self.min_signal_count}): {"; ".join(signals)}',
                    'signal_count': signal_count,
                    'required_count': self.min_signal_count
                }
            
            # 综合判断 - 使用严格的阈值
            if confidence >= self.buy_confidence_threshold:
                action = 'BUY'
                reason = '; '.join(signals)
            elif confidence >= 0.5:
                action = 'STRONG_WATCH'  # 新增强监控状态
                reason = f'强监控信号: {"; ".join(signals)}'
            elif confidence >= 0.3:
                action = 'WATCH'
                reason = f'弱信号: {"; ".join(signals)}'
            else:
                action = 'HOLD'
                reason = '技术条件不满足'
            
            return {
                'action': action,
                'confidence': min(confidence, 1.0),
                'reason': reason,
                'indicators': indicators,
                'signals': signals,
                'signal_count': signal_count,
                'required_count': self.min_signal_count,
                'trend_filter_passed': True  # 已通过趋势过滤
            }
            
        except Exception as e:
            self.logger.error(f"买入信号分析失败 {ticker}: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': '分析失败'}
    
    def analyze_sell_signals(self, ticker: str, stock_data: pd.DataFrame, entry_price: float, current_price: float) -> Dict:
        """分析卖出信号"""
        try:
            indicators = self.calculate_technical_indicators(stock_data)
            if not indicators:
                return {'action': 'HOLD', 'confidence': 0, 'reason': '技术指标计算失败'}
            
            signals = []
            confidence = 0
            
            # 计算盈亏比例
            pnl_ratio = (current_price - entry_price) / entry_price
            
            # 止盈信号
            take_profit_threshold = self.config.get('take_profit_pct', 0.03)
            if pnl_ratio >= take_profit_threshold:
                signals.append(f'触发止盈({pnl_ratio:.1%})')
                confidence += 0.6
            
            # 止损信号
            stop_loss_threshold = -self.config.get('stop_loss_pct', 0.05)
            if pnl_ratio <= stop_loss_threshold:
                signals.append(f'触发止损({pnl_ratio:.1%})')
                confidence += 0.8  # 止损优先级更高
            
            # 技术反转信号
            # 信号1: RSI超买
            if indicators['rsi'] > self.rsi_overbought:
                signals.append(f'RSI超买({indicators["rsi"]:.1f})')
                confidence += 0.3
            
            # 信号2: 价格突破布林带上轨
            if indicators['current_price'] >= indicators['bb_upper'] * 0.99:
                signals.append('突破布林带上轨')
                confidence += 0.3
            
            # 信号3: Z-Score正向过度偏离
            if indicators['zscore'] > self.zscore_threshold:
                signals.append(f'Z-Score正向偏离({indicators["zscore"]:.2f})')
                confidence += 0.4
            
            # 信号4: 价格相对均值过度上涨
            if indicators['price_deviation'] > 0.08:  # 相对均值上涨超过8%
                signals.append(f'价格过度上涨({indicators["price_deviation"]:.1%})')
                confidence += 0.2
            
            # 信号5: 高波动率环境
            if indicators['volatility_ratio'] > 0.05:  # 波动率相对价格>5%
                signals.append('高波动环境')
                confidence += 0.1
            
            # 综合判断
            if confidence >= 0.6:  # 60%置信度即可卖出
                action = 'SELL'
                reason = '; '.join(signals)
            elif confidence >= 0.3:
                action = 'WATCH'
                reason = f'弱卖出信号: {"; ".join(signals)}'
            else:
                action = 'HOLD'
                reason = '无卖出信号'
            
            return {
                'action': action,
                'confidence': min(confidence, 1.0),
                'reason': reason,
                'pnl_ratio': pnl_ratio,
                'indicators': indicators,
                'signals': signals
            }
            
        except Exception as e:
            self.logger.error(f"卖出信号分析失败 {ticker}: {e}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': '分析失败'}
    
    def _apply_trend_filter(self, indicators: Dict) -> Dict:
        """应用趋势过滤逻辑"""
        if not self.use_trend_filter:
            return {'blocked': False, 'reason': ''}
        
        current_price = indicators['current_price']
        ma_200 = indicators['ma_200']
        adx = indicators['adx']
        
        # 200日均线趋势过滤
        trend_deviation = abs(current_price - ma_200) / ma_200 if ma_200 != 0 else 0
        min_price_threshold = ma_200 * (1 - self.trend_deviation_threshold)
        
        if current_price < min_price_threshold:
            return {
                'blocked': True, 
                'reason': f'逆势交易风险（价格{current_price:.2f} < 200MA阈值{min_price_threshold:.2f}，偏差{trend_deviation:.1%}）'
            }
        
        # ADX趋势强度过滤
        if adx > self.adx_trend_threshold:
            return {
                'blocked': True,
                'reason': f'ADX趋势过强({adx:.1f} > {self.adx_trend_threshold})，不适合均值回归策略'
            }
        
        return {'blocked': False, 'reason': f'趋势过滤通过（价格适合，ADX={adx:.1f}）'}
    
    def _evaluate_technical_conditions(self, indicators: Dict) -> Dict:
        """评估技术条件 - RSI、布林带、Z-Score三大指标"""
        signals = []
        confidence = 0.0
        condition_count = 0
        
        # 条件1: RSI超卖条件
        if indicators['rsi'] < self.rsi_oversold:
            signals.append(f'RSI超卖({indicators["rsi"]:.1f} < {self.rsi_oversold})')
            confidence += 0.35  # RSI权重35%
            condition_count += 1
            
            # RSI信号稳定性加分
            if indicators.get('rsi_stable', False):
                confidence += 0.05
                signals.append('RSI信号稳定')
        
        # 条件2: 布林带下轨条件
        bb_lower_threshold = indicators['bb_lower'] * 1.01  # 允许1%缓冲
        if indicators['current_price'] <= bb_lower_threshold:
            signals.append(f'触及布林带下轨({indicators["current_price"]:.2f} ≤ {bb_lower_threshold:.2f})')
            confidence += 0.30  # 布林带权重30%
            condition_count += 1
        
        # 条件3: Z-Score过度偏离条件
        if indicators['zscore'] < -self.zscore_threshold:
            signals.append(f'Z-Score过度偏离({indicators["zscore"]:.2f} < -{self.zscore_threshold})')
            confidence += 0.35  # Z-Score权重35%
            condition_count += 1
            
            # Z-Score信号稳定性加分
            if indicators.get('zscore_stable', False):
                confidence += 0.05
                signals.append('Z-Score信号稳定')
        
        # 附加条件: 高质量信号加分
        # 价格相对均值过度下跌
        if indicators['price_deviation'] < -0.06:  # 提高阈值到6%
            signals.append(f'价格过度偏离均值({indicators["price_deviation"]:.1%})')
            confidence += 0.10
        
        # 低波动环境加分
        if indicators['volatility_ratio'] < 0.015:  # 更严格的波动率要求
            signals.append(f'低波动环境(vol_ratio={indicators["volatility_ratio"]:.3f})')
            confidence += 0.05
        
        return {
            'count': condition_count,
            'confidence': confidence,
            'signals': signals,
            'rsi_triggered': indicators['rsi'] < self.rsi_oversold,
            'bb_triggered': indicators['current_price'] <= bb_lower_threshold,
            'zscore_triggered': indicators['zscore'] < -self.zscore_threshold
        }
    
    def analyze_combined_signals(self, ticker: str, stock_data: pd.DataFrame, bma_score: float = 0.0) -> Dict:
        """综合BMA评分和技术分析的买入信号分析"""
        try:
            # 首先进行技术分析
            technical_result = self.analyze_buy_signals(ticker, stock_data)
            
            # BMA评分检查
            bma_passed = bma_score >= self.bma_score_threshold
            technical_passed = technical_result['action'] in ['BUY', 'STRONG_WATCH']
            
            # 综合判断逻辑
            if self.require_technical_confirmation:
                # 要求BMA + 技术面双重确认
                if bma_passed and technical_passed:
                    if technical_result['action'] == 'BUY':
                        final_action = 'BUY'
                        final_confidence = min(technical_result['confidence'] + 0.1, 1.0)  # BMA加分
                        final_reason = f"BMA评分{bma_score:.3f} + 技术面确认: {technical_result['reason']}"
                    else:  # STRONG_WATCH
                        final_action = 'STRONG_WATCH'
                        final_confidence = technical_result['confidence']
                        final_reason = f"BMA评分{bma_score:.3f} + 技术面强监控: {technical_result['reason']}"
                elif bma_passed and not technical_passed:
                    final_action = 'WATCH'
                    final_confidence = 0.4
                    final_reason = f"BMA评分{bma_score:.3f}通过，但技术面不确认: {technical_result['reason']}"
                elif not bma_passed and technical_passed:
                    final_action = 'WATCH'
                    final_confidence = 0.3
                    final_reason = f"BMA评分{bma_score:.3f}低于BMA阈值{self.bma_score_threshold}，但技术面好: {technical_result['reason']}"
                else:
                    final_action = 'HOLD'
                    final_confidence = 0.0
                    final_reason = f"BMA评分{bma_score:.3f}不足且技术面不确认"
            else:
                # 仅使用BMA评分
                if bma_passed:
                    final_action = 'BUY'
                    final_confidence = min(bma_score * 1.2, 1.0)
                    final_reason = f"BMA评分{bma_score:.3f}通过阈值{self.bma_score_threshold}"
                else:
                    final_action = 'HOLD'
                    final_confidence = 0.0
                    final_reason = f"BMA评分{bma_score:.3f}低于阈值{self.bma_score_threshold}"
            
            return {
                'action': final_action,
                'confidence': final_confidence,
                'reason': final_reason,
                'bma_score': bma_score,
                'bma_passed': bma_passed,
                'technical_passed': technical_passed,
                'technical_result': technical_result,
                'indicators': technical_result.get('indicators', {})
            }
            
        except Exception as e:
            self.logger.error(f"综合信号分析失败 {ticker}: {e}")
            return {
                'action': 'HOLD', 
                'confidence': 0, 
                'reason': f'综合分析失败: {e}',
                'bma_score': bma_score,
                'bma_passed': False,
                'technical_passed': False
            }

class RiskManager:
    """增强版风险管理器"""
    
    def __init__(self, config: Dict = None, logger=None, max_position_size: float = 0.05, 
                 max_portfolio_risk: float = 0.20, stop_loss_pct: float = 0.05, 
                 take_profit_pct: float = 0.10):
        # 兼容原有配置系统
        if config:
            self.config = config
            self.logger = logger
            self.total_capital = config.get('total_capital', 100000)
            self.max_position_size = config.get('max_position_size', max_position_size)
            self.max_portfolio_exposure = config.get('max_portfolio_exposure', 0.5)
            self.commission_rate = config.get('commission_rate', 0.001)
            self.max_portfolio_risk = config.get('max_portfolio_risk', max_portfolio_risk)
            self.stop_loss_pct = config.get('stop_loss_pct', stop_loss_pct)
            self.take_profit_pct = config.get('take_profit_pct', take_profit_pct)
        else:
            # 新的独立配置
            self.max_position_size = max_position_size  # 单个持仓最大占比
            self.max_portfolio_risk = max_portfolio_risk  # 组合最大风险敞口
            self.stop_loss_pct = stop_loss_pct  # 止损比例
            self.take_profit_pct = take_profit_pct  # 止盈比例
            self.total_capital = 100000
            self.max_portfolio_exposure = 0.5
            self.commission_rate = 0.001
            self.logger = logger
        
        # 风险指标
        self.current_risk = 0.0
        self.position_risks = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = -0.05  # 日内最大亏损5%
        
        # 动态跟踪
        self.used_capital = 0.0
        self.total_positions = 0
        
    def calculate_position_size(self, ticker: str, current_price: float, confidence: float, 
                               existing_positions: Dict) -> Tuple[int, Dict]:
        """计算仓位大小 - 考虑组合层面限制"""
        try:
            # 计算当前已用资金
            current_used = sum(pos['shares'] * pos['avg_price'] for pos in existing_positions.values())
            available_capital = self.total_capital - current_used
            
            # 单个股票最大投资额
            max_single_investment = self.total_capital * self.max_position_size
            
            # 组合层面限制
            max_total_investment = self.total_capital * self.max_portfolio_exposure
            max_additional_investment = max_total_investment - current_used
            
            # 基于置信度调整
            confidence_multiplier = min(confidence * 1.2, 1.0)
            target_investment = max_single_investment * confidence_multiplier
            
            # 取最小值确保不超限
            final_investment = min(
                target_investment,
                max_additional_investment,
                available_capital * 0.95  # 保留5%缓冲
            )
            
            if final_investment <= 0:
                return 0, {
                    'reason': '资金不足或达到组合限制',
                    'available_capital': available_capital,
                    'used_capital': current_used,
                    'target_investment': target_investment
                }
            
            # 考虑手续费
            final_investment_after_commission = final_investment / (1 + self.commission_rate)
            shares = int(final_investment_after_commission / current_price)
            
            actual_cost = shares * current_price * (1 + self.commission_rate)
            
            info = {
                'shares': shares,
                'actual_cost': actual_cost,
                'target_investment': target_investment,
                'confidence_multiplier': confidence_multiplier,
                'portfolio_exposure': (current_used + actual_cost) / self.total_capital,
                'reason': '正常计算'
            }
            
            self.logger.info(f"{ticker} 仓位计算: 置信度={confidence:.1%}, 股数={shares}, "
                           f"成本=${actual_cost:.0f}, 组合暴露={info['portfolio_exposure']:.1%}")
            
            return shares, info
            
        except Exception as e:
            self.logger.error(f"仓位计算失败 {ticker}: {e}")
            return 0, {'reason': f'计算错误: {e}'}
    
    def check_risk_limits(self, positions: Dict) -> Dict:
        """检查风险限制"""
        try:
            total_value = sum(pos['shares'] * pos['current_price'] for pos in positions.values() 
                            if 'current_price' in pos)
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
            
            portfolio_exposure = total_value / self.total_capital
            total_pnl_ratio = total_pnl / self.total_capital
            
            warnings = []
            
            # 检查组合暴露
            if portfolio_exposure > self.max_portfolio_exposure:
                warnings.append(f'组合暴露过高: {portfolio_exposure:.1%}')
            
            # 检查总亏损
            max_portfolio_loss = self.config.get('max_portfolio_loss', 0.10)
            if total_pnl_ratio < -max_portfolio_loss:
                warnings.append(f'组合亏损过大: {total_pnl_ratio:.1%}')
            
            # 检查单个仓位
            for ticker, pos in positions.items():
                position_ratio = (pos['shares'] * pos.get('current_price', pos['avg_price'])) / self.total_capital
                if position_ratio > self.max_position_size * 1.1:  # 允许10%缓冲
                    warnings.append(f'{ticker}仓位过大: {position_ratio:.1%}')
            
            return {
                'portfolio_exposure': portfolio_exposure,
                'total_pnl_ratio': total_pnl_ratio,
                'total_positions': len(positions),
                'warnings': warnings,
                'risk_level': 'HIGH' if warnings else 'NORMAL'
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"风险检查失败: {e}")
            return {'risk_level': 'ERROR', 'warnings': [f'风险检查错误: {e}']}
    
    def check_position_size(self, symbol: str, quantity: int, price: float, 
                           portfolio_value: float) -> Tuple[bool, str]:
        """检查仓位大小"""
        position_value = quantity * price
        position_pct = position_value / portfolio_value
        
        if position_pct > self.max_position_size:
            return False, f"仓位过大: {position_pct:.2%} > {self.max_position_size:.2%}"
        
        return True, "仓位检查通过"
    
    def check_portfolio_risk(self, new_risk: float) -> Tuple[bool, str]:
        """检查组合风险"""
        total_risk = self.current_risk + new_risk
        
        if total_risk > self.max_portfolio_risk:
            return False, f"组合风险过高: {total_risk:.2%} > {self.max_portfolio_risk:.2%}"
        
        return True, "风险检查通过"
    
    def check_daily_loss_limit(self) -> Tuple[bool, str]:
        """检查日内亏损限制"""
        if self.daily_pnl < self.max_daily_loss:
            return False, f"日内亏损超限: {self.daily_pnl:.2%} < {self.max_daily_loss:.2%}"
        
        return True, "日内亏损检查通过"
    
    def calculate_stop_loss_price(self, entry_price: float, action: str) -> float:
        """计算止损价格"""
        if action == "BUY":
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit_price(self, entry_price: float, action: str) -> float:
        """计算止盈价格"""
        if action == "BUY":
            return entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            return entry_price * (1 - self.take_profit_pct)


class MarketDataProcessor:
    """市场数据处理器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.volume_history = defaultdict(lambda: deque(maxlen=window_size))
        self.tick_data = defaultdict(dict)
        self.indicators = defaultdict(dict)
        
        self.logger = logging.getLogger('MarketDataProcessor')
    
    def update_tick_data(self, symbol: str, tick_type: int, price: float):
        """更新tick数据"""
        self.tick_data[symbol][tick_type] = {
            'price': price,
            'timestamp': datetime.now()
        }
        
        # 更新价格历史（使用最新成交价）
        if tick_type == 4:  # LAST_PRICE
            self.price_history[symbol].append(price)
            self._update_indicators(symbol)
    
    def _update_indicators(self, symbol: str):
        """更新技术指标"""
        if len(self.price_history[symbol]) < 20:
            return
        
        prices = np.array(list(self.price_history[symbol]))
        
        # 计算移动平均
        if len(prices) >= 20:
            self.indicators[symbol]['sma20'] = np.mean(prices[-20:])
        if len(prices) >= 50:
            self.indicators[symbol]['sma50'] = np.mean(prices[-50:])
        
        # 计算RSI
        if len(prices) >= 14:
            self.indicators[symbol]['rsi'] = self._calculate_rsi(prices)
        
        # 计算布林带
        if len(prices) >= 20:
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(prices)
            self.indicators[symbol].update({
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle
            })
    
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """计算RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, lower, sma
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        if symbol in self.tick_data and 4 in self.tick_data[symbol]:
            return self.tick_data[symbol][4]['price']
        return None
    
    def get_indicators(self, symbol: str) -> Dict:
        """获取技术指标"""
        return self.indicators.get(symbol, {})


class EnhancedSignalGenerator:
    """增强版信号生成器"""
    
    def __init__(self, data_processor: MarketDataProcessor):
        self.data_processor = data_processor
        self.bma_recommendations = {}  # BMA模型推荐
        self.lstm_predictions = {}  # LSTM预测
        
        self.logger = logging.getLogger('EnhancedSignalGenerator')
    
    def load_bma_recommendations(self, file_path: str):
        """加载BMA模型推荐"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, sheet_name='Top10买入推荐')
                for _, row in df.iterrows():
                    symbol = row['股票代码']
                    self.bma_recommendations[symbol] = {
                        'rating': row['评级'],
                        'prediction': row['加权预测收益率(%)'] / 100,
                        'confidence': row.get('置信度评分', 0.8)
                    }
                self.logger.info(f"加载{len(self.bma_recommendations)}个BMA推荐")
            
        except Exception as e:
            self.logger.error(f"加载BMA推荐失败: {e}")
    
    def load_lstm_predictions(self, file_path: str):
        """加载LSTM预测"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, sheet_name='多日分析结果')
                for _, row in df.iterrows():
                    symbol = row['股票代码']
                    self.lstm_predictions[symbol] = {
                        'day1_pred': row['第1天预测(%)'] / 100,
                        'day2_pred': row['第2天预测(%)'] / 100,
                        'weighted_pred': row['加权预测收益率(%)'] / 100,
                        'confidence': row['置信度评分']
                    }
                self.logger.info(f"加载{len(self.lstm_predictions)}个LSTM预测")
            
        except Exception as e:
            self.logger.error(f"加载LSTM预测失败: {e}")
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """生成交易信号"""
        indicators = self.data_processor.get_indicators(symbol)
        current_price = self.data_processor.get_latest_price(symbol)
        
        if not indicators or current_price is None:
            return None
        
        # 获取模型推荐
        bma_rec = self.bma_recommendations.get(symbol, {})
        lstm_pred = self.lstm_predictions.get(symbol, {})
        
        # 信号强度计算
        signal_strength = 0.0
        reasons = []
        
        # BMA模型信号
        if bma_rec:
            if bma_rec['rating'] in ['BUY', 'STRONG_BUY']:
                signal_strength += 0.3 * bma_rec['confidence']
                reasons.append(f"BMA:{bma_rec['rating']}")
        
        # LSTM预测信号
        if lstm_pred:
            if lstm_pred['weighted_pred'] > 0.02:  # 预期收益率>2%
                signal_strength += 0.3 * lstm_pred['confidence']
                reasons.append(f"LSTM:{lstm_pred['weighted_pred']:.1%}")
        
        # 技术指标信号
        tech_signal = self._generate_technical_signal(indicators, current_price)
        signal_strength += tech_signal
        
        if tech_signal > 0:
            reasons.append("技术面看涨")
        elif tech_signal < 0:
            reasons.append("技术面看跌")
        
        # 决定操作
        if signal_strength > 0.6:
            action = "BUY"
            quantity = self._calculate_position_size(symbol, signal_strength)
        elif signal_strength < -0.6:
            action = "SELL"
            quantity = self._calculate_position_size(symbol, abs(signal_strength))
        else:
            action = "HOLD"
            quantity = 0
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            quantity=quantity,
            signal_strength=abs(signal_strength),
            reason="; ".join(reasons)
        )
    
    def _generate_technical_signal(self, indicators: Dict, current_price: float) -> float:
        """生成技术指标信号"""
        signal = 0.0
        
        # RSI信号
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                signal += 0.2  # 超卖
            elif rsi > 70:
                signal -= 0.2  # 超买
        
        # 移动平均信号
        if 'sma20' in indicators and 'sma50' in indicators:
            sma20 = indicators['sma20']
            sma50 = indicators['sma50']
            if sma20 > sma50 and current_price > sma20:
                signal += 0.15  # 上升趋势
            elif sma20 < sma50 and current_price < sma20:
                signal -= 0.15  # 下降趋势
        
        # 布林带信号
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            if current_price < indicators['bb_lower']:
                signal += 0.15  # 价格接近下轨
            elif current_price > indicators['bb_upper']:
                signal -= 0.15  # 价格接近上轨
        
        return signal
    
    def _calculate_position_size(self, symbol: str, signal_strength: float) -> int:
        """计算建议仓位大小"""
        # 基础仓位（假设10万资金，最大5%单一持仓）
        base_position_value = 100000 * 0.05 * signal_strength
        
        # 获取当前价格
        current_price = self.data_processor.get_latest_price(symbol)
        if not current_price:
            return 0
        
        quantity = int(base_position_value / current_price)
        return max(100, quantity)  # 最小100股


class EnhancedMeanReversionStrategy:
    """增强版均值回归交易策略"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # 初始化模块
        self.data_service = DataService(self.logger)
        
        # BMA周度数据配置
        self.bma_json_file = config.get('bma_json_file', 'weekly_bma_trading.json')
        self.use_bma_recommendations = config.get('use_bma_recommendations', True)
        self.bma_data = None
        self.bma_load_time = None
        self.signal_generator = SignalGenerator(config, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        
        # 新增增强功能组件
        self.market_data_processor = MarketDataProcessor()
        self.enhanced_signal_generator = EnhancedSignalGenerator(self.market_data_processor)
        
        # 事件驱动相关
        self.running = False
        self.strategy_thread = None
        self.pending_orders = {}
        self.active_positions = {}
        self.subscribed_symbols = set()
        
        # IBKR连接参数
        self.ibkr_host = config.get('ibkr_host', '127.0.0.1')
        self.ibkr_port = config.get('ibkr_port', 4002)  # 统一使用4002端口
        self.ibkr_client_id = config.get('ibkr_client_id', 50310)
        
        # 数据存储
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易历史
        self.ib = None  # IBKR连接
        
        # 线程锁
        self._position_lock = threading.Lock()
        
    def setup_logging(self):
        """设置日志"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"enhanced_trading_strategy_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_ibkr(self) -> bool:
        """连接IBKR"""
        if not IBKR_AVAILABLE:
            self.logger.warning("IBKR功能不可用，将运行模拟模式")
            return False
            
        try:
            self.ib = IB()
            self.ib.connect(self.ibkr_host, self.ibkr_port, clientId=self.ibkr_client_id)
            self.logger.info(f"成功连接IBKR: {self.ibkr_host}:{self.ibkr_port}")
            return True
        except Exception as e:
            self.logger.error(f"IBKR连接失败: {e}")
            return False
    
    def disconnect_ibkr(self):
        """断开IBKR连接"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("已断开IBKR连接")
    
    def execute_trade(self, ticker: str, action: str, shares: int, price: float, reason: str = "") -> bool:
        """执行交易 - 包含手续费计算"""
        try:
            commission = shares * price * self.risk_manager.commission_rate
            total_cost = shares * price + commission
            
            if not self.ib or not self.ib.isConnected():
                # 模拟交易
                self.logger.info(f"[模拟交易] {action} {shares} shares of {ticker} at ${price:.2f}")
                self.logger.info(f"  手续费: ${commission:.2f}, 总成本: ${total_cost:.2f}")
                if reason:
                    self.logger.info(f"  交易原因: {reason}")
                
                # 记录到交易历史
                trade_record = {
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'commission': commission,
                    'total_cost': total_cost,
                    'reason': reason,
                    'type': 'SIMULATION'
                }
                self.trade_history.append(trade_record)
                
                # 更新模拟持仓
                with self._position_lock:
                    self._update_position(ticker, action, shares, price)
                
                return True
            
            else:
                # 实际IBKR交易
                try:
                    contract = Stock(ticker, 'SMART', 'USD')
                    order = MarketOrder(action, shares)
                    
                    trade = self.ib.placeOrder(contract, order)
                    self.ib.sleep(2)
                    
                    self.logger.info(f"[实际交易] {action} {shares} shares of {ticker}")
                    
                    trade_record = {
                        'timestamp': datetime.now(),
                        'ticker': ticker,
                        'action': action,
                        'shares': shares,
                        'price': price,
                        'commission': commission,
                        'total_cost': total_cost,
                        'reason': reason,
                        'type': 'REAL',
                        'order_id': trade.order.orderId
                    }
                    self.trade_history.append(trade_record)
                    
                    with self._position_lock:
                        self._update_position(ticker, action, shares, price)
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"IBKR交易执行失败 {ticker}: {e}")
                    return False
            
        except Exception as e:
            self.logger.error(f"交易执行失败 {ticker}: {e}")
            return False
    
    def _update_position(self, ticker: str, action: str, shares: int, price: float):
        """更新持仓信息"""
        if ticker not in self.positions:
            self.positions[ticker] = {
                'shares': 0, 
                'avg_price': 0, 
                'entry_time': datetime.now(),
                'total_cost': 0
            }
        
        pos = self.positions[ticker]
        
        if action == 'BUY':
            old_cost = pos['shares'] * pos['avg_price']
            new_cost = shares * price
            new_shares = pos['shares'] + shares
            
            if new_shares > 0:
                pos['avg_price'] = (old_cost + new_cost) / new_shares
                pos['shares'] = new_shares
                pos['total_cost'] = old_cost + new_cost
            
        elif action == 'SELL':
            pos['shares'] = max(0, pos['shares'] - shares)
            if pos['shares'] == 0:
                # 完全平仓，重置信息
                pos['avg_price'] = 0
                pos['total_cost'] = 0
                pos['entry_time'] = datetime.now()
    
    def update_positions_with_current_prices(self, market_data: Dict[str, pd.DataFrame]):
        """更新持仓的当前价格和盈亏"""
        with self._position_lock:
            for ticker, pos in self.positions.items():
                if pos['shares'] > 0 and ticker in market_data:
                    try:
                        current_price = float(market_data[ticker]['Close'].iloc[-1])
                        pos['current_price'] = current_price
                        pos['market_value'] = pos['shares'] * current_price
                        pos['unrealized_pnl'] = pos['market_value'] - pos['total_cost']
                        pos['pnl_ratio'] = pos['unrealized_pnl'] / pos['total_cost'] if pos['total_cost'] > 0 else 0
                    except Exception as e:
                        self.logger.error(f"更新 {ticker} 价格失败: {e}")
    
    def run_strategy(self):
        """运行交易策略 - 包含完整买入卖出逻辑"""
        self.logger.info("=" * 80)
        self.logger.info("启动增强版BMA均值回归交易策略")
        self.logger.info("=" * 80)
        
        try:
            # 1. 连接IBKR数据源
            if not self.data_service.connect_ibkr(
                self.config.get('ibkr_host', '127.0.0.1'),
                self.config.get('ibkr_port', 4002), 
                self.config.get('ibkr_client_id', 50310)
            ):
                self.logger.error("IBKR数据连接失败，策略退出")
                return
            
            # 2. 获取当前IBKR账户信息和持仓
            account = self.config.get('ibkr_account', 'c2dvdongg')
            account_info = self.data_service.get_account_info(account)
            
            if account_info.get('connected'):
                self.logger.info(f"成功连接到IBKR账户: {account}")
                
                # 显示账户余额信息
                if 'account_values' in account_info:
                    for av in account_info['account_values']:
                        if 'NetLiquidation' in av.tag:
                            self.logger.info(f"账户净值: {av.value} {av.currency}")
                        elif 'TotalCashValue' in av.tag:
                            self.logger.info(f"现金余额: {av.value} {av.currency}")
                        elif 'AvailableFunds' in av.tag:
                            self.logger.info(f"可用资金: {av.value} {av.currency}")
                
                # 获取持仓信息
                ibkr_positions = self.data_service.get_current_positions()
                if ibkr_positions:
                    self.logger.info(f"IBKR当前持仓: {len(ibkr_positions)} 只股票")
                    # 更新策略持仓记录
                    for ticker, pos_info in ibkr_positions.items():
                        self.positions[ticker] = {
                            'shares': pos_info['shares'],
                            'avg_price': pos_info['avg_price'],
                            'entry_time': datetime.now(),
                            'total_cost': pos_info['shares'] * pos_info['avg_price']
                        }
            else:
                self.logger.warning(f"无法连接到IBKR账户 {account}: {account_info.get('error', '未知错误')}")
            
            # 3. 连接IBKR交易接口
            if self.config.get('enable_real_trading', False):
                if not self.connect_ibkr():
                    self.logger.warning("IBKR交易连接失败，转为模拟模式")
            else:
                self.logger.info("运行模拟模式")
            
            # 4. 加载BMA推荐作为候选股票
            recommendations = self.data_service.load_bma_recommendations()
            if not recommendations:
                self.logger.warning("无BMA推荐，仅分析当前持仓")
                recommendations = []
            
            # 5. 准备完整的股票考虑范围
            candidate_tickers = [stock['ticker'] for stock in recommendations]
            current_holdings = list(self.positions.keys()) if self.positions else []
            all_tickers = list(set(candidate_tickers + current_holdings))
            
            if not all_tickers:
                self.logger.warning("没有需要分析的股票")
                return
            
            self.logger.info(f"股票考虑范围: 当前持仓 {len(current_holdings)} 只 + BMA推荐 {len(candidate_tickers)} 只 = 总计 {len(all_tickers)} 只")
            
            # 6. 批量下载股票数据（通过IBKR API）
            market_data = self.data_service.download_stock_data_batch(all_tickers)
            if not market_data:
                self.logger.error("无法获取市场数据，策略退出")
                return
            
            # 7. 更新持仓价格
            self.update_positions_with_current_prices(market_data)
            
            # 8. 风险检查
            risk_status = self.risk_manager.check_risk_limits(self.positions)
            self.logger.info(f"风险状态: {risk_status['risk_level']}")
            if risk_status['warnings']:
                for warning in risk_status['warnings']:
                    self.logger.warning(f"风险提醒: {warning}")
            
            # 9. 处理卖出信号（优先处理持仓）
            sell_executed = 0
            for ticker, position in list(self.positions.items()):
                if position['shares'] > 0 and ticker in market_data:
                    current_price = position.get('current_price', position['avg_price'])
                    
                    # 分析卖出信号
                    sell_analysis = self.signal_generator.analyze_sell_signals(
                        ticker, market_data[ticker], position['avg_price'], current_price
                    )
                    
                    self.logger.info(f"\n{ticker} 卖出分析:")
                    self.logger.info(f"  当前价格: ${current_price:.2f} (成本: ${position['avg_price']:.2f})")
                    self.logger.info(f"  盈亏: {sell_analysis.get('pnl_ratio', 0):.1%}")
                    self.logger.info(f"  动作: {sell_analysis['action']}")
                    self.logger.info(f"  置信度: {sell_analysis['confidence']:.1%}")
                    self.logger.info(f"  原因: {sell_analysis['reason']}")
                    
                    # 执行卖出
                    if sell_analysis['action'] == 'SELL' and sell_analysis['confidence'] >= 0.6:
                        success = self.execute_trade(
                            ticker, 'SELL', position['shares'], current_price, 
                            sell_analysis['reason']
                        )
                        if success:
                            sell_executed += 1
                            self.logger.info(f"✅ 成功卖出 {ticker}: {position['shares']} 股")
            
            # 10. 处理BMA推荐买入信号
            buy_candidates = []
            for stock in recommendations:
                ticker = stock['ticker']
                if ticker in market_data:
                    # 如果已经持仓，跳过
                    if ticker in self.positions and self.positions[ticker]['shares'] > 0:
                        continue
                    
                    # 直接使用BMA评分，不进行额外技术分析
                    bma_score = stock.get('final_score', 0.5)
                    predicted_return = stock.get('predicted_return', 0.0)
                    current_price = market_data[ticker]['Close'].iloc[-1] if not market_data[ticker].empty else 0
                    
                    self.logger.info(f"\n{ticker} BMA推荐分析:")
                    self.logger.info(f"  BMA评分: {bma_score:.3f}")
                    self.logger.info(f"  预测收益: {predicted_return:.1%}")
                    self.logger.info(f"  当前价格: ${current_price:.2f}")
                    
                    # 基于BMA评分进行交易决策
                    if bma_score >= 0.6:  # BMA评分阈值
                        buy_candidates.append({
                            'ticker': ticker,
                            'bma_data': stock,
                            'bma_score': bma_score,
                            'predicted_return': predicted_return,
                            'current_price': current_price,
                            'reason': f"BMA推荐 (评分: {bma_score:.3f}, 预测收益: {predicted_return:.1%})"
                        })
            
            # 11. 执行买入交易
            buy_candidates.sort(key=lambda x: x['bma_score'], reverse=True)
            buy_executed = 0
            
            self.logger.info(f"\n找到 {len(buy_candidates)} 个买入机会")
            
            for candidate in buy_candidates:
                ticker = candidate['ticker']
                current_price = candidate['current_price']
                bma_score = candidate['bma_score']
                
                # 基于BMA评分计算仓位
                shares, position_info = self.risk_manager.calculate_position_size(
                    ticker, current_price, bma_score, self.positions
                )
                
                if shares > 0:
                    success = self.execute_trade(
                        ticker, 'BUY', shares, current_price, 
                        candidate['reason']
                    )
                    if success:
                        buy_executed += 1
                        self.logger.info(f"✅ 成功买入 {ticker}: {shares} 股 @ ${current_price:.2f} (BMA评分: {bma_score:.3f})")
                else:
                    self.logger.info(f"⏭️ 跳过 {ticker}: {position_info['reason']}")
            
            # 10. 策略总结
            self.logger.info(f"\n" + "="*60)
            self.logger.info(f"策略执行完成")
            self.logger.info(f"卖出执行: {sell_executed} 笔")
            self.logger.info(f"买入执行: {buy_executed} 笔")
            self.logger.info(f"当前持仓: {len([p for p in self.positions.values() if p['shares'] > 0])} 只")
            
            # 11. 保存交易记录
            self.save_trading_results()
            
        except Exception as e:
            self.logger.error(f"策略运行失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            
        finally:
            if self.ib:
                self.disconnect_ibkr()
    
    def save_trading_results(self):
        """保存交易结果"""
        try:
            results_dir = "trading_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存交易历史
            if self.trade_history:
                trades_df = pd.DataFrame(self.trade_history)
                trades_file = os.path.join(results_dir, f"enhanced_trades_{timestamp}.csv")
                trades_df.to_csv(trades_file, index=False, encoding='utf-8')
                self.logger.info(f"交易历史已保存: {trades_file}")
            
            # 保存持仓信息
            active_positions = {k: v for k, v in self.positions.items() if v['shares'] > 0}
            if active_positions:
                positions_data = []
                total_value = 0
                total_pnl = 0
                
                for ticker, pos in active_positions.items():
                    market_value = pos.get('market_value', pos['shares'] * pos['avg_price'])
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                    
                    positions_data.append({
                        'ticker': ticker,
                        'shares': pos['shares'],
                        'avg_price': pos['avg_price'],
                        'current_price': pos.get('current_price', pos['avg_price']),
                        'total_cost': pos['total_cost'],
                        'market_value': market_value,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_ratio': pos.get('pnl_ratio', 0),
                        'entry_time': pos['entry_time']
                    })
                    
                    total_value += market_value
                    total_pnl += unrealized_pnl
                
                positions_df = pd.DataFrame(positions_data)
                positions_file = os.path.join(results_dir, f"enhanced_positions_{timestamp}.csv")
                positions_df.to_csv(positions_file, index=False, encoding='utf-8')
                
                # 组合汇总
                portfolio_summary = {
                    'timestamp': timestamp,
                    'total_positions': len(active_positions),
                    'total_market_value': total_value,
                    'total_unrealized_pnl': total_pnl,
                    'portfolio_return': total_pnl / self.risk_manager.total_capital,
                    'portfolio_exposure': total_value / self.risk_manager.total_capital
                }
                
                summary_file = os.path.join(results_dir, f"portfolio_summary_{timestamp}.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(portfolio_summary, f, ensure_ascii=False, indent=2, default=str)
                
                self.logger.info(f"持仓信息已保存: {positions_file}")
                self.logger.info(f"组合汇总已保存: {summary_file}")
                self.logger.info(f"组合总价值: ${total_value:.2f}")
                self.logger.info(f"未实现盈亏: ${total_pnl:.2f} ({total_pnl/self.risk_manager.total_capital:.1%})")
            
        except Exception as e:
            self.logger.error(f"保存交易结果失败: {e}")
    
    def start_enhanced_trading(self):
        """启动增强版交易策略"""
        if not self._initialize_enhanced_components():
            self.logger.error("增强组件初始化失败，无法启动")
            return False
        
        self.running = True
        self.logger.info("增强版交易策略已启动")
        
        # 加载最新的模型数据
        self._load_model_data()
        
        # 启动策略线程
        self.strategy_thread = threading.Thread(target=self._enhanced_strategy_loop, daemon=True)
        self.strategy_thread.start()
        
        return True
    
    def stop_enhanced_trading(self):
        """停止增强版交易策略"""
        self.running = False
        self.logger.info("正在停止增强版交易策略...")
        
        if self.ib:
            try:
                self.ib.disconnect()
            except:
                pass
        
        if self.strategy_thread:
            self.strategy_thread.join(timeout=5)
        
        self.logger.info("增强版交易策略已停止")
    
    def _initialize_enhanced_components(self) -> bool:
        """初始化增强组件"""
        try:
            # 连接IBKR（如果启用）
            if self.config.get('enable_real_trading', False):
                if not self.connect_ibkr():
                    self.logger.warning("IBKR连接失败，转为模拟模式")
            
            # 设置事件监听
            self._setup_event_listeners()
            
            return True
            
        except Exception as e:
            self.logger.error(f"增强组件初始化失败: {e}")
            return False
    
    def _load_model_data(self):
        """加载模型数据"""
        try:
            # 查找最新的BMA文件
            import glob
            bma_pattern = 'result/*bma_quantitative_analysis_*.xlsx'
            bma_files = glob.glob(bma_pattern)
            if bma_files:
                latest_bma = max(bma_files, key=os.path.getmtime)
                self.enhanced_signal_generator.load_bma_recommendations(latest_bma)
                self.logger.info(f"加载BMA推荐: {latest_bma}")
            
            # 查找最新的LSTM文件
            lstm_pattern = 'result/*lstm_analysis_*.xlsx'
            lstm_files = glob.glob(lstm_pattern)
            if lstm_files:
                latest_lstm = max(lstm_files, key=os.path.getmtime)
                self.enhanced_signal_generator.load_lstm_predictions(latest_lstm)
                self.logger.info(f"加载LSTM预测: {latest_lstm}")
                
        except Exception as e:
            self.logger.error(f"模型数据加载失败: {e}")
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        # 这里可以添加IBKR事件监听器的设置
        # 由于当前使用ib_insync，事件处理方式不同
        pass
    
    def _enhanced_strategy_loop(self):
        """增强版策略主循环"""
        self.logger.info("增强版策略主循环已启动")
        
        while self.running:
            try:
                # 定期检查和更新
                self._periodic_enhanced_check()
                
                # 暂停30秒
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"增强策略循环错误: {e}")
                time.sleep(60)
    
    def _periodic_enhanced_check(self):
        """定期增强检查"""
        try:
            # 检查连接状态
            if self.ib and not self.ib.isConnected():
                self.logger.warning("IBKR连接断开，尝试重连...")
                if not self.connect_ibkr():
                    return
            
            # 获取候选股票列表
            recommendations = self.data_service.load_bma_recommendations()
            if not recommendations:
                return
            
            candidate_tickers = [stock['ticker'] for stock in recommendations[:10]]  # 限制前10个
            
            # 批量获取实时数据
            market_data = self.data_service.download_stock_data_batch(candidate_tickers)
            if not market_data:
                return
            
            # 更新市场数据处理器
            for ticker, data in market_data.items():
                if len(data) > 0:
                    latest_price = float(data['Close'].iloc[-1])
                    self.market_data_processor.update_tick_data(ticker, 4, latest_price)
            
            # 生成交易信号
            for ticker in candidate_tickers:
                signal = self.enhanced_signal_generator.generate_signal(ticker)
                if signal and signal.action != "HOLD":
                    self.logger.info(f"交易信号: {signal.symbol} {signal.action} {signal.quantity} "
                                   f"(强度: {signal.signal_strength:.2f}, 原因: {signal.reason})")
                    
                    # 执行交易（如果启用）
                    if self.config.get('enable_real_trading', False):
                        self._execute_enhanced_trade(signal)
            
            # 检查持仓和风险
            self._check_positions_and_risk()
            
        except Exception as e:
            self.logger.error(f"定期检查失败: {e}")
    
    def _execute_enhanced_trade(self, signal: TradingSignal) -> bool:
        """执行增强版交易"""
        try:
            # 风险检查
            portfolio_value = self.risk_manager.total_capital
            current_price = self.market_data_processor.get_latest_price(signal.symbol)
            
            if not current_price:
                return False
            
            # 检查仓位大小
            can_trade, msg = self.risk_manager.check_position_size(
                signal.symbol, signal.quantity, current_price, portfolio_value
            )
            
            if not can_trade:
                self.logger.warning(f"仓位检查失败 {signal.symbol}: {msg}")
                return False
            
            # 检查日内亏损
            can_trade, msg = self.risk_manager.check_daily_loss_limit()
            if not can_trade:
                self.logger.warning(f"日内亏损检查失败: {msg}")
                return False
            
            # 执行交易
            success = self.execute_trade(signal.symbol, signal.action, signal.quantity, current_price, signal.reason)
            
            if success:
                # 更新持仓记录
                if signal.symbol not in self.active_positions:
                    self.active_positions[signal.symbol] = {
                        'shares': 0,
                        'avg_price': 0,
                        'entry_time': datetime.now()
                    }
                
                position = self.active_positions[signal.symbol]
                if signal.action == "BUY":
                    new_shares = position['shares'] + signal.quantity
                    new_avg_price = ((position['shares'] * position['avg_price']) + 
                                   (signal.quantity * current_price)) / new_shares
                    position['shares'] = new_shares
                    position['avg_price'] = new_avg_price
                else:  # SELL
                    position['shares'] = max(0, position['shares'] - signal.quantity)
                
                # 记录订单
                order_id = f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.pending_orders[order_id] = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'price': current_price,
                    'timestamp': datetime.now()
                }
            
            return success
            
        except Exception as e:
            self.logger.error(f"增强版交易执行失败: {e}")
            return False
    
    def _check_positions_and_risk(self):
        """检查持仓和风险"""
        try:
            # 获取当前价格并更新持仓
            for symbol in list(self.active_positions.keys()):
                current_price = self.market_data_processor.get_latest_price(symbol)
                if not current_price:
                    continue
                
                position = self.active_positions[symbol]
                if position['shares'] <= 0:
                    continue
                
                # 计算盈亏
                avg_cost = position['avg_price']
                pnl_pct = (current_price - avg_cost) / avg_cost
                
                # 检查止损止盈
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    self.logger.warning(f"触发止损 {symbol}: {pnl_pct:.2%}")
                    self._execute_stop_loss(symbol, position['shares'])
                elif pnl_pct >= self.risk_manager.take_profit_pct:
                    self.logger.info(f"触发止盈 {symbol}: {pnl_pct:.2%}")
                    self._execute_take_profit(symbol, position['shares'])
            
        except Exception as e:
            self.logger.error(f"持仓风险检查失败: {e}")
    
    def _execute_stop_loss(self, symbol: str, quantity: int):
        """执行止损"""
        try:
            current_price = self.market_data_processor.get_latest_price(symbol)
            if current_price:
                success = self.execute_trade(symbol, "SELL", quantity, current_price, "止损")
                if success:
                    self.logger.warning(f"止损执行成功: SELL {quantity} {symbol} @ ${current_price:.2f}")
        except Exception as e:
            self.logger.error(f"止损执行失败 {symbol}: {e}")
    
    def _execute_take_profit(self, symbol: str, quantity: int):
        """执行止盈"""
        try:
            current_price = self.market_data_processor.get_latest_price(symbol)
            if current_price:
                success = self.execute_trade(symbol, "SELL", quantity, current_price, "止盈")
                if success:
                    self.logger.info(f"止盈执行成功: SELL {quantity} {symbol} @ ${current_price:.2f}")
        except Exception as e:
            self.logger.error(f"止盈执行失败 {symbol}: {e}")
    
    def get_enhanced_status(self) -> Dict:
        """获取增强版策略状态"""
        return {
            'running': self.running,
            'connected': self.ib.isConnected() if self.ib else False,
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'subscribed_symbols': len(self.subscribed_symbols),
            'bma_recommendations': len(self.enhanced_signal_generator.bma_recommendations),
            'lstm_predictions': len(self.enhanced_signal_generator.lstm_predictions),
            'last_update': datetime.now().isoformat()
        }
    
    def run_strategy(self):
        """运行策略 - 兼容原有接口"""
        try:
            # 如果配置了增强模式，启动增强版
            if self.config.get('enable_enhanced_mode', False):
                if self.start_enhanced_trading():
                    print("增强版交易策略已启动，按Ctrl+C停止...")
                    try:
                        while self.running:
                            status = self.get_enhanced_status()
                            print(f"\r状态: 运行={status['running']}, 连接={status['connected']}, "
                                  f"持仓={status['active_positions']}, 订单={status['pending_orders']}", end="")
                            time.sleep(5)
                    except KeyboardInterrupt:
                        print("\n正在停止增强版交易策略...")
                        self.stop_enhanced_trading()
                        print("增强版交易策略已停止")
                return
            
            # 原有的策略运行逻辑
            self.logger.info("启动增强版BMA均值回归交易策略")
            
            # 1. 获取账户信息
            account = self.config.get('ibkr_account', 'DU12345')
            account_info = self.get_account_info(account)
            
            if account_info.get('success', False):
                self.logger.info(f"IBKR账户连接成功: {account}")
                self.logger.info(f"账户净值: ${account_info.get('net_liquidation', 'N/A')}")
                
                # 2. 获取当前持仓
                ibkr_positions = self.get_current_positions(account)
                if ibkr_positions:
                    self.logger.info(f"IBKR当前持仓: {len(ibkr_positions)} 只股票")
                    # 更新策略持仓记录
                    for ticker, pos_info in ibkr_positions.items():
                        self.positions[ticker] = {
                            'shares': pos_info['shares'],
                            'avg_price': pos_info['avg_price'],
                            'entry_time': datetime.now(),
                            'total_cost': pos_info['shares'] * pos_info['avg_price']
                        }
            else:
                self.logger.warning(f"无法连接到IBKR账户 {account}: {account_info.get('error', '未知错误')}")
            
            # 3. 连接IBKR交易接口
            if self.config.get('enable_real_trading', False):
                if not self.connect_ibkr():
                    self.logger.warning("IBKR交易连接失败，转为模拟模式")
            else:
                self.logger.info("运行模拟模式")
            
            # 4. 加载BMA推荐作为候选股票
            recommendations = self.data_service.load_bma_recommendations()
            if not recommendations:
                self.logger.warning("无BMA推荐，仅分析当前持仓")
                recommendations = []
            
            # 5. 准备完整的股票考虑范围
            candidate_tickers = [stock['ticker'] for stock in recommendations]
            current_holdings = list(self.positions.keys()) if self.positions else []
            all_tickers = list(set(candidate_tickers + current_holdings))
            
            if not all_tickers:
                self.logger.warning("没有需要分析的股票")
                return
            
            self.logger.info(f"股票考虑范围: 当前持仓 {len(current_holdings)} 只 + BMA推荐 {len(candidate_tickers)} 只 = 总计 {len(all_tickers)} 只")
            
            # 6. 批量下载股票数据（通过IBKR API）
            market_data = self.data_service.download_stock_data_batch(all_tickers)
            if not market_data:
                self.logger.error("无法获取市场数据，策略退出")
                return
            
            # 7. 更新持仓价格
            self.update_positions_with_current_prices(market_data)
            
            # 8. 风险检查
            risk_status = self.risk_manager.check_risk_limits(self.positions)
            self.logger.info(f"风险状态: {risk_status['risk_level']}")
            if risk_status['warnings']:
                for warning in risk_status['warnings']:
                    self.logger.warning(f"风险提醒: {warning}")
            
            # 9. 分析交易机会
            trading_decisions = []
            for ticker in all_tickers:
                decision = self.analyze_trading_opportunity(ticker, market_data.get(ticker), recommendations)
                if decision['action'] != 'HOLD':
                    trading_decisions.append(decision)
            
            # 10. 执行交易决策
            if trading_decisions:
                self.logger.info(f"发现 {len(trading_decisions)} 个交易机会")
                for decision in trading_decisions:
                    if self.execute_trade(decision['ticker'], decision['action'], 
                                        decision['shares'], decision['price'], decision['reason']):
                        self.trade_history.append({
                            'timestamp': datetime.now(),
                            'ticker': decision['ticker'],
                            'action': decision['action'],
                            'shares': decision['shares'],
                            'price': decision['price'],
                            'reason': decision['reason']
                        })
            else:
                self.logger.info("当前没有合适的交易机会")
            
            # 11. 保存交易结果和持仓信息
            self.save_trading_results()
            
            self.logger.info("策略运行完成")
            
        except KeyboardInterrupt:
            self.logger.info("用户中断，策略停止")
        except Exception as e:
            self.logger.error(f"策略运行失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数 - 支持命令行参数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强版BMA均值回归交易策略')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--real-trading', action='store_true', 
                       help='启用实盘交易（默认模拟模式）')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--bma-file', type=str, help='指定BMA推荐文件路径')
    parser.add_argument('--weekly-bma', type=str, default='weekly_bma_trading.json', 
                       help='周度BMA推荐文件路径')
    parser.add_argument('--use-weekly-bma', action='store_true', 
                       help='优先使用周度BMA推荐（如果可用）')
    parser.add_argument('--min-price', type=float, help='最低股价过滤(美元)')
    parser.add_argument('--max-price', type=float, help='最高股价过滤(美元)')
    parser.add_argument('--port', type=int, default=7497, help='IBKR端口')
    parser.add_argument('--client-id', type=int, default=1, help='IBKR客户端ID')
    parser.add_argument('--enhanced-mode', action='store_true', 
                       help='启用增强模式（事件驱动+实时监控）')
    
    args = parser.parse_args()
    
    # 增强版策略配置 - 优化参数
    config = {
        # 日志配置
        'log_level': args.log_level,
        'debug_mode': args.debug,
        
        # 优化后的技术指标参数
        'rsi_period': 3,  # 缩短RSI周期到3
        'rsi_oversold': 10,  # 更严格的超卖阈值
        'rsi_overbought': 90,  # 更严格的超买阈值
        
        # 布林带参数优化
        'bb_period': 20,  # 布林带周期
        'bb_stddev': 2.0,  # 布林带标准差倍数
        
        # Z-Score参数优化
        'lookback_days': 20,
        'zscore_threshold': 3.0,  # 提高Z-Score阈值到3.0
        'signal_stability_days': 3,  # 信号稳定性检查
        
        # 信号质量提升参数
        'buy_confidence_threshold': 0.8,  # 提高买入置信度阈值
        'min_signal_count': 2,  # 至少满足2个技术指标（RSI、布林带、Z-Score）
        
        # 趋势过滤参数
        'use_trend_filter': True,  # 启用趋势过滤避免逆势交易
        'trend_ma_period': 200,  # 200日均线趋势判断
        'trend_deviation_threshold': 0.02,  # 允许价格低于200MA的最大偏差2%
        'adx_trend_threshold': 30,  # ADX趋势强度阈值，超过则不适合均值回归
        
        # BMA与技术分析统合参数
        'bma_score_threshold': 0.65,  # BMA评分阈值
        'require_technical_confirmation': True,  # 要求BMA+技术面双重确认
        
        # 风控参数 - 调整后的参数
        'max_position_size': 0.1,  # 单个股票最大仓位10%
        'max_portfolio_exposure': 0.80,  # 组合最大暴露80%
        'max_portfolio_loss': 0.10,  # 组合最大亏损10%
        'stop_loss_pct': 0.08,  # 8%止损
        'take_profit_pct': 0.04,  # 4%止盈
        'max_consecutive_losses': 3,  # 最大连续亏损次数
        'max_new_positions_per_day': 3,  # 每日最大新开仓数量控制
        
        # ATR动态止损止盈
        'use_atr_stops': True,  # 启用ATR止损
        'atr_period': 14,  # ATR周期
        'stop_loss_atr_multiplier': 1.0,  # 止损ATR倍数
        'take_profit_atr_multiplier': 0.5,  # 止盈ATR倍数
        'use_trailing_stop': True,  # 启用追踪止损
        'trailing_atr_multiplier': 0.5,  # 追踪止损ATR倍数
        
        # 交易参数
        'total_capital': 100000,  # 总资金10万
        'commission_rate': 0.001,  # 0.1%手续费
        'risk_reward_ratio': 0.5,  # 风险收益比
        
        # 数据获取配置
        'min_data_days': 60,  # 最小数据天数
        'data_retry_attempts': 2,  # 数据重试次数
        'use_threading': True,  # 启用多线程下载
        'thread_pool_size': 5,  # 线程池大小
        
        # BMA文件配置
        'bma_filepath': args.bma_file,  # 手动指定文件路径
        'bma_search_dirs': ['ibkr_trading', '.', 'data', 'bma_output'],
        'bma_file_pattern': r'top_10_stocks_(\d{8}_\d{6})\.json',
        
        # 周度BMA配置
        'bma_json_file': args.weekly_bma,
        'use_bma_recommendations': args.use_weekly_bma,
        'min_price_threshold': args.min_price,
        'max_price_threshold': args.max_price,
        
        # IBKR连接设置
        'enable_real_trading': args.real_trading,
        'ibkr_host': '127.0.0.1',
        'ibkr_port': args.port,
        'ibkr_client_id': args.client_id,
        'ibkr_account': 'c2dvdongg',
        
        # 增强模式设置
        'enable_enhanced_mode': args.enhanced_mode
    }
    
    # 输出配置信息
    print("\n策略配置:")
    print(f"    交易模式: {'实盘' if config['enable_real_trading'] else '模拟'}")
    print(f"    增强模式: {'ON' if config['enable_enhanced_mode'] else 'OFF'} (事件驱动+实时监控)")
    print(f"    日志级别: {config['log_level']}")
    print(f"    RSI参数: 周期{config['rsi_period']}, 超卖{config['rsi_oversold']}, 超买{config['rsi_overbought']}")
    print(f"    Z-Score阈值: ±{config['zscore_threshold']} (提高精度)")
    print(f"    技术条件: 至少{config['min_signal_count']}/3个指标确认")
    print(f"    趋势过滤: {'ON' if config['use_trend_filter'] else 'OFF'} (200MA+ADX)")
    print(f"    BMA+技术: {'ON' if config['require_technical_confirmation'] else 'OFF'} (BMA阈值{config['bma_score_threshold']})")
    print(f"    最大仓位: {config['max_position_size']:.1%}")
    print(f"    组合暴露: {config['max_portfolio_exposure']:.1%}")
    print(f"    信号阈值: 买入{config['buy_confidence_threshold']:.0%}")
    if config['bma_filepath']:
        print(f"    BMA文件: {config['bma_filepath']}")
    print(f"    IBKR端口: {config['ibkr_port']}")
    print()
    if config['enable_enhanced_mode']:
        print(" 增强特性: 事件驱动架构 + 实时市场数据 + 智能信号生成 + 订单生命周期跟踪")
    else:
        print(" 增强特性: 多条件确认 + 趋势过滤 + BMA技术统合")
    print()
    
    # 创建并运行增强版策略
    strategy = EnhancedMeanReversionStrategy(config)
    strategy.run_strategy()

if __name__ == "__main__":
    main()