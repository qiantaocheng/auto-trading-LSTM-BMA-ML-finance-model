#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时市场数据处理系统
实现reqMktData()订阅和tickPrice回调的事件驱动架构
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from collections import defaultdict, deque
import pandas as pd
import numpy as np

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False


class RealTimeDataProcessor:
    """实时数据处理器 - 管理市场数据订阅和事件处理"""
    
    def __init__(self, ib_connection, logger=None):
        self.ib = ib_connection
        self.logger = logger or logging.getLogger(__name__)
        
        # 数据存储
        self.tick_data = defaultdict(lambda: {
            'bid': None, 'ask': None, 'last': None, 'volume': None,
            'bid_size': None, 'ask_size': None, 'last_size': None,
            'high': None, 'low': None, 'close': None, 'open': None,
            'timestamp': None
        })
        
        # 历史价格队列（用于技术指标计算）
        self.price_history = defaultdict(lambda: deque(maxlen=200))
        self.volume_history = defaultdict(lambda: deque(maxlen=200))
        
        # 技术指标缓存
        self.indicators = defaultdict(dict)
        
        # 订阅状态
        self.subscribed_symbols = set()
        self.active_contracts = {}
        
        # 事件回调
        self.tick_callbacks = []
        self.indicator_callbacks = []
        self.bar_callbacks = []
        
        # 计算标志
        self.calculate_indicators = True
        self.indicator_update_interval = 5  # 每5个tick更新一次指标
        self.tick_count = defaultdict(int)
        
        # 数据质量监控
        self.data_quality = defaultdict(lambda: {
            'last_update': None,
            'update_count': 0,
            'missing_ticks': 0,
            'stale_data_threshold': 60  # 60秒无更新视为过期
        })
    
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """订阅市场数据"""
        if not self.ib or not self.ib.isConnected():
            self.logger.error("No valid IB connection for market data subscription")
            return False
        
        success_count = 0
        
        for symbol in symbols:
            try:
                if symbol in self.subscribed_symbols:
                    self.logger.debug(f"Already subscribed to {symbol}")
                    continue
                
                # 创建合约
                contract = self._create_contract(symbol)
                if not contract:
                    continue
                
                # 订阅实时数据
                ticker = self.ib.reqMktData(contract, '', False, False)
                
                if ticker:
                    self.active_contracts[symbol] = {
                        'contract': contract,
                        'ticker': ticker,
                        'subscribed_at': datetime.now()
                    }
                    
                    self.subscribed_symbols.add(symbol)
                    success_count += 1
                    
                    self.logger.info(f"Subscribed to market data for {symbol}")
                else:
                    self.logger.error(f"Failed to subscribe to {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error subscribing to {symbol}: {e}")
        
        # 设置事件处理器
        if success_count > 0:
            self._setup_event_handlers()
        
        self.logger.info(f"Successfully subscribed to {success_count}/{len(symbols)} symbols")
        return success_count > 0
    
    def unsubscribe_market_data(self, symbols: List[str] = None):
        """取消订阅市场数据"""
        if symbols is None:
            symbols = list(self.subscribed_symbols)
        
        for symbol in symbols:
            try:
                if symbol in self.active_contracts:
                    contract = self.active_contracts[symbol]['contract']
                    self.ib.cancelMktData(contract)
                    
                    del self.active_contracts[symbol]
                    self.subscribed_symbols.discard(symbol)
                    
                    self.logger.info(f"Unsubscribed from market data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {symbol}: {e}")
    
    def _create_contract(self, symbol: str) -> Optional[Contract]:
        """创建交易合约"""
        try:
            # 处理不同的股票代码格式
            if '.' in symbol:
                # 可能是港股或其他市场
                base_symbol = symbol.split('.')[0]
                if symbol.endswith('.HK'):
                    contract = Stock(base_symbol, 'SEHK', 'HKD')
                else:
                    contract = Stock(symbol, 'SMART', 'USD')
            else:
                # 美股
                contract = Stock(symbol, 'SMART', 'USD')
            
            return contract
            
        except Exception as e:
            self.logger.error(f"Error creating contract for {symbol}: {e}")
            return None
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 设置tick数据回调
        self.ib.pendingTickersEvent.clear()  # 清除现有事件
        self.ib.pendingTickersEvent += self._on_pending_tickers
        
        # 设置错误处理
        self.ib.errorEvent += self._on_error
        
        self.logger.info("Event handlers set up")
    
    def _on_pending_tickers(self, tickers):
        """处理待处理的ticker数据"""
        for ticker in tickers:
            symbol = ticker.contract.symbol
            
            # 更新tick数据
            self._update_tick_data(symbol, ticker)
            
            # 更新技术指标
            if self.calculate_indicators:
                self._update_indicators(symbol)
            
            # 触发回调
            self._trigger_tick_callbacks(symbol, ticker)
    
    def _update_tick_data(self, symbol: str, ticker):
        """更新tick数据"""
        now = datetime.now()
        
        # 更新基础数据
        tick_data = self.tick_data[symbol]
        
        if ticker.bid and not np.isnan(ticker.bid):
            tick_data['bid'] = ticker.bid
        if ticker.ask and not np.isnan(ticker.ask):
            tick_data['ask'] = ticker.ask
        if ticker.last and not np.isnan(ticker.last):
            tick_data['last'] = ticker.last
            # 添加到价格历史
            self.price_history[symbol].append(ticker.last)
        
        if ticker.bidSize and not np.isnan(ticker.bidSize):
            tick_data['bid_size'] = ticker.bidSize
        if ticker.askSize and not np.isnan(ticker.askSize):
            tick_data['ask_size'] = ticker.askSize
        if ticker.lastSize and not np.isnan(ticker.lastSize):
            tick_data['last_size'] = ticker.lastSize
        
        if ticker.high and not np.isnan(ticker.high):
            tick_data['high'] = ticker.high
        if ticker.low and not np.isnan(ticker.low):
            tick_data['low'] = ticker.low
        if ticker.close and not np.isnan(ticker.close):
            tick_data['close'] = ticker.close
        if ticker.open and not np.isnan(ticker.open):
            tick_data['open'] = ticker.open
        
        if ticker.volume and not np.isnan(ticker.volume):
            tick_data['volume'] = ticker.volume
            self.volume_history[symbol].append(ticker.volume)
        
        tick_data['timestamp'] = now
        
        # 更新数据质量监控
        quality = self.data_quality[symbol]
        quality['last_update'] = now
        quality['update_count'] += 1
        
        # 增加tick计数
        self.tick_count[symbol] += 1
        
        self.logger.debug(f"Updated tick data for {symbol}: last={tick_data['last']}, bid={tick_data['bid']}, ask={tick_data['ask']}")
    
    def _update_indicators(self, symbol: str):
        """更新技术指标"""
        # 每N个tick更新一次指标
        if self.tick_count[symbol] % self.indicator_update_interval != 0:
            return
        
        prices = np.array(self.price_history[symbol])
        if len(prices) < 20:  # 需要足够的数据点
            return
        
        try:
            indicators = {}
            
            # RSI
            if len(prices) >= 14:
                indicators['rsi'] = self._calculate_rsi(prices, 14)
            
            # 布林带
            if len(prices) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                # 布林带位置
                current_price = prices[-1]
                indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            # 移动平均
            if len(prices) >= 20:
                indicators['ma_20'] = np.mean(prices[-20:])
            if len(prices) >= 50:
                indicators['ma_50'] = np.mean(prices[-50:])
            if len(prices) >= 200:
                indicators['ma_200'] = np.mean(prices[-200:])
            
            # 价格变化率
            if len(prices) >= 2:
                indicators['price_change'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
                indicators['price_change_pct'] = indicators['price_change'] * 100
            
            # Z-Score (相对于20期均值的标准化偏差)
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                if std_price > 0:
                    indicators['zscore'] = (prices[-1] - mean_price) / std_price
            
            # 成交量指标
            volumes = np.array(self.volume_history[symbol])
            if len(volumes) >= 20:
                indicators['volume_ma_20'] = np.mean(volumes[-20:])
                if len(volumes) >= 2:
                    indicators['volume_ratio'] = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
            
            # 更新指标缓存
            self.indicators[symbol] = indicators
            
            # 触发指标回调
            self._trigger_indicator_callbacks(symbol, indicators)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _trigger_tick_callbacks(self, symbol: str, ticker):
        """触发tick数据回调"""
        for callback in self.tick_callbacks:
            try:
                callback(symbol, ticker, self.tick_data[symbol])
            except Exception as e:
                self.logger.error(f"Error in tick callback: {e}")
    
    def _trigger_indicator_callbacks(self, symbol: str, indicators: Dict):
        """触发指标回调"""
        for callback in self.indicator_callbacks:
            try:
                callback(symbol, indicators)
            except Exception as e:
                self.logger.error(f"Error in indicator callback: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """处理错误事件"""
        self.logger.error(f"IB Error - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
        
        # 处理特定错误
        if errorCode == 200:  # No security definition found
            symbol = contract.symbol if contract else "Unknown"
            self.logger.error(f"No security definition found for {symbol}")
            
        elif errorCode == 354:  # Requested market data is not subscribed
            symbol = contract.symbol if contract else "Unknown"
            self.logger.warning(f"Market data not subscribed for {symbol}")
    
    def add_tick_callback(self, callback: Callable):
        """添加tick数据回调函数"""
        self.tick_callbacks.append(callback)
    
    def add_indicator_callback(self, callback: Callable):
        """添加指标回调函数"""
        self.indicator_callbacks.append(callback)
    
    def remove_tick_callback(self, callback: Callable):
        """移除tick数据回调函数"""
        if callback in self.tick_callbacks:
            self.tick_callbacks.remove(callback)
    
    def remove_indicator_callback(self, callback: Callable):
        """移除指标回调函数"""
        if callback in self.indicator_callbacks:
            self.indicator_callbacks.remove(callback)
    
    def get_latest_data(self, symbol: str) -> Dict:
        """获取最新的市场数据"""
        return dict(self.tick_data[symbol])
    
    def get_indicators(self, symbol: str) -> Dict:
        """获取技术指标"""
        return dict(self.indicators[symbol])
    
    def get_price_history(self, symbol: str, count: int = None) -> List[float]:
        """获取价格历史"""
        history = list(self.price_history[symbol])
        if count:
            return history[-count:] if len(history) >= count else history
        return history
    
    def is_data_stale(self, symbol: str) -> bool:
        """检查数据是否过期"""
        quality = self.data_quality[symbol]
        if not quality['last_update']:
            return True
        
        time_diff = (datetime.now() - quality['last_update']).total_seconds()
        return time_diff > quality['stale_data_threshold']
    
    def get_data_quality_report(self) -> Dict:
        """获取数据质量报告"""
        report = {}
        now = datetime.now()
        
        for symbol in self.subscribed_symbols:
            quality = self.data_quality[symbol]
            
            time_since_update = None
            if quality['last_update']:
                time_since_update = (now - quality['last_update']).total_seconds()
            
            report[symbol] = {
                'subscribed': True,
                'last_update': quality['last_update'].isoformat() if quality['last_update'] else None,
                'time_since_update_seconds': time_since_update,
                'update_count': quality['update_count'],
                'is_stale': self.is_data_stale(symbol),
                'tick_count': self.tick_count[symbol]
            }
        
        return report
    
    def cleanup(self):
        """清理资源"""
        self.unsubscribe_market_data()
        self.tick_callbacks.clear()
        self.indicator_callbacks.clear()
        self.logger.info("Real-time data processor cleaned up")


class EventDrivenStrategyEngine:
    """事件驱动的策略引擎"""
    
    def __init__(self, data_processor: RealTimeDataProcessor, logger=None):
        self.data_processor = data_processor
        self.logger = logger or logging.getLogger(__name__)
        
        # 策略状态
        self.active_strategies = {}
        self.positions = {}
        self.pending_orders = {}
        
        # 事件处理
        self.strategy_callbacks = {}
        
        # 注册数据处理器回调
        self.data_processor.add_tick_callback(self._on_tick_data)
        self.data_processor.add_indicator_callback(self._on_indicator_update)
    
    def register_strategy(self, strategy_name: str, symbols: List[str], callback: Callable):
        """注册策略"""
        self.active_strategies[strategy_name] = {
            'symbols': symbols,
            'callback': callback,
            'active': True,
            'registered_at': datetime.now()
        }
        
        self.strategy_callbacks[strategy_name] = callback
        self.logger.info(f"Strategy '{strategy_name}' registered for symbols: {symbols}")
    
    def deregister_strategy(self, strategy_name: str):
        """注销策略"""
        if strategy_name in self.active_strategies:
            del self.active_strategies[strategy_name]
            del self.strategy_callbacks[strategy_name]
            self.logger.info(f"Strategy '{strategy_name}' deregistered")
    
    def _on_tick_data(self, symbol: str, ticker, tick_data: Dict):
        """处理tick数据事件"""
        # 触发相关策略
        for strategy_name, strategy_info in self.active_strategies.items():
            if symbol in strategy_info['symbols'] and strategy_info['active']:
                try:
                    callback = strategy_info['callback']
                    callback('tick', symbol, {
                        'ticker': ticker,
                        'tick_data': tick_data,
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy_name} tick handler: {e}")
    
    def _on_indicator_update(self, symbol: str, indicators: Dict):
        """处理指标更新事件"""
        # 触发相关策略
        for strategy_name, strategy_info in self.active_strategies.items():
            if symbol in strategy_info['symbols'] and strategy_info['active']:
                try:
                    callback = strategy_info['callback']
                    callback('indicator', symbol, {
                        'indicators': indicators,
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy_name} indicator handler: {e}")
    
    def pause_strategy(self, strategy_name: str):
        """暂停策略"""
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name]['active'] = False
            self.logger.info(f"Strategy '{strategy_name}' paused")
    
    def resume_strategy(self, strategy_name: str):
        """恢复策略"""
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name]['active'] = True
            self.logger.info(f"Strategy '{strategy_name}' resumed")
    
    def get_strategy_status(self) -> Dict:
        """获取策略状态"""
        return {
            name: {
                'symbols': info['symbols'],
                'active': info['active'],
                'registered_at': info['registered_at'].isoformat()
            }
            for name, info in self.active_strategies.items()
        }


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 模拟连接（实际使用时需要真实的IB连接）
    class MockIB:
        def isConnected(self):
            return True
        
        def reqMktData(self, contract, genericTickList, snapshot, regulatorySnapshot):
            return None
    
    mock_ib = MockIB()
    
    # 创建实时数据处理器
    data_processor = RealTimeDataProcessor(mock_ib)
    
    # 创建事件驱动引擎
    strategy_engine = EventDrivenStrategyEngine(data_processor)
    
    # 示例策略回调
    def sample_strategy(event_type: str, symbol: str, data: Dict):
        print(f"Strategy triggered - Type: {event_type}, Symbol: {symbol}")
        if event_type == 'tick':
            tick_data = data['tick_data']
            print(f"  Last price: {tick_data['last']}")
        elif event_type == 'indicator':
            indicators = data['indicators']
            print(f"  RSI: {indicators.get('rsi', 'N/A')}")
    
    # 注册策略
    strategy_engine.register_strategy('sample_strategy', ['AAPL', 'MSFT'], sample_strategy)
    
    print("Event-driven system initialized successfully")