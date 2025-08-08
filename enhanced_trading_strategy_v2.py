#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版交易策略 v2.0
基于持久化IBKR连接的事件驱动交易系统

核心改进：
1. 持久化连接与自动重连
2. 事件驱动的市场数据订阅
3. 完整的订单生命周期跟踪
4. 高级风险管理
5. 实时监控与告警

Authors: AI Assistant
Version: 2.0
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# 导入持久化客户端
from persistent_ibkr_client import PersistentIBKRClient, IBKR_AVAILABLE

# 导入数据分析库
try:
    import yfinance as yf
    YF_AVAILABLE = True
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


class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size: float = 0.05, max_portfolio_risk: float = 0.20,
                 stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10):
        self.max_position_size = max_position_size  # 单个持仓最大占比
        self.max_portfolio_risk = max_portfolio_risk  # 组合最大风险敞口
        self.stop_loss_pct = stop_loss_pct  # 止损比例
        self.take_profit_pct = take_profit_pct  # 止盈比例
        
        # 风险指标
        self.current_risk = 0.0
        self.position_risks = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = -0.05  # 日内最大亏损5%
        
        self.logger = logging.getLogger('RiskManager')
    
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


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self, data_processor: MarketDataProcessor):
        self.data_processor = data_processor
        self.bma_recommendations = {}  # BMA模型推荐
        self.lstm_predictions = {}  # LSTM预测
        
        self.logger = logging.getLogger('SignalGenerator')
    
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
        
        if action != "HOLD":
            return TradingSignal(
                symbol=symbol,
                action=action,
                quantity=quantity,
                signal_strength=abs(signal_strength),
                reason="; ".join(reasons)
            )
        
        return None
    
    def _generate_technical_signal(self, indicators: Dict, current_price: float) -> float:
        """生成技术分析信号"""
        signal = 0.0
        
        # RSI信号
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:  # 超卖
                signal += 0.2
            elif rsi > 70:  # 超买
                signal -= 0.2
        
        # 布林带信号
        if 'bb_lower' in indicators and 'bb_upper' in indicators:
            if current_price < indicators['bb_lower']:  # 跌破下轨
                signal += 0.15
            elif current_price > indicators['bb_upper']:  # 突破上轨
                signal -= 0.15
        
        # 移动平均信号
        if 'sma20' in indicators and 'sma50' in indicators:
            if indicators['sma20'] > indicators['sma50']:  # 金叉
                signal += 0.1
            else:  # 死叉
                signal -= 0.1
        
        return signal
    
    def _calculate_position_size(self, symbol: str, signal_strength: float) -> int:
        """计算仓位大小"""
        # 基础仓位：100股
        base_size = 100
        
        # 根据信号强度调整
        size_multiplier = min(signal_strength * 2, 3.0)  # 最大3倍
        
        return int(base_size * size_multiplier)


class EnhancedTradingStrategy:
    """增强版交易策略"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config = self._load_config(config_file)
        
        # 设置日志
        self.logger = self._setup_logging()
        
        # 初始化组件
        self.ibkr_client = None
        self.risk_manager = RiskManager(**self.config.get('risk_management', {}))
        self.data_processor = MarketDataProcessor()
        self.signal_generator = SignalGenerator(self.data_processor)
        
        # 交易状态
        self.active_positions = {}  # symbol -> position_info
        self.pending_orders = {}  # order_id -> order_info
        self.trade_history = []
        self.subscribed_symbols = set()
        
        # 运行控制
        self.running = False
        self.strategy_thread = None
        
        self.logger.info("增强版交易策略初始化完成")
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置"""
        default_config = {
            'ibkr': {
                'host': '127.0.0.1',
                'port': 4002,
                'client_id': 50310
            },
            'risk_management': {
                'max_position_size': 0.05,
                'max_portfolio_risk': 0.20,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10
            },
            'trading': {
                'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'signal_threshold': 0.6,
                'max_positions': 10
            },
            'data_sources': {
                'bma_file': 'result/bma_quantitative_analysis_*.xlsx',
                'lstm_file': 'result/*lstm_analysis_*.xlsx'
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('EnhancedTradingStrategy')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 文件日志
            file_handler = logging.FileHandler(
                f'logs/enhanced_trading_{datetime.now().strftime("%Y%m%d")}.log',
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # 控制台日志
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def initialize(self) -> bool:
        """初始化系统"""
        self.logger.info("正在初始化交易系统...")
        
        try:
            # 初始化IBKR连接
            if not self._initialize_ibkr():
                return False
            
            # 加载模型数据
            self._load_model_data()
            
            # 设置事件监听
            self._setup_event_listeners()
            
            self.logger.info("交易系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def _initialize_ibkr(self) -> bool:
        """初始化IBKR连接"""
        if not IBKR_AVAILABLE:
            self.logger.error("IBKR API不可用")
            return False
        
        try:
            ibkr_config = self.config['ibkr']
            self.ibkr_client = PersistentIBKRClient(
                host=ibkr_config['host'],
                port=ibkr_config['port'],
                client_id=ibkr_config['client_id']
            )
            
            # 等待连接
            timeout = 10
            start_time = time.time()
            while not self.ibkr_client.is_connected() and (time.time() - start_time) < timeout:
                time.sleep(0.5)
            
            if self.ibkr_client.is_connected():
                self.logger.info("IBKR连接成功")
                return True
            else:
                self.logger.error("IBKR连接超时")
                return False
                
        except Exception as e:
            self.logger.error(f"IBKR连接失败: {e}")
            return False
    
    def _load_model_data(self):
        """加载模型数据"""
        try:
            # 查找最新的BMA文件
            import glob
            bma_pattern = self.config['data_sources']['bma_file']
            bma_files = glob.glob(bma_pattern)
            if bma_files:
                latest_bma = max(bma_files, key=os.path.getmtime)
                self.signal_generator.load_bma_recommendations(latest_bma)
            
            # 查找最新的LSTM文件
            lstm_pattern = self.config['data_sources']['lstm_file']
            lstm_files = glob.glob(lstm_pattern)
            if lstm_files:
                latest_lstm = max(lstm_files, key=os.path.getmtime)
                self.signal_generator.load_lstm_predictions(latest_lstm)
                
        except Exception as e:
            self.logger.error(f"模型数据加载失败: {e}")
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        if not self.ibkr_client:
            return
        
        # 连接事件
        self.ibkr_client.add_event_listener('connection_restored', self._on_connection_restored)
        self.ibkr_client.add_event_listener('connection_lost', self._on_connection_lost)
        
        # 交易事件
        self.ibkr_client.add_event_listener('order_filled', self._on_order_filled)
        self.ibkr_client.add_event_listener('order_cancelled', self._on_order_cancelled)
        
        # 市场数据事件
        self.ibkr_client.add_event_listener('market_data_update', self._on_market_data_update)
        self.ibkr_client.add_event_listener('position_update', self._on_position_update)
    
    def _on_connection_restored(self):
        """连接恢复事件"""
        self.logger.info("IBKR连接已恢复")
        
        # 重新订阅市场数据
        self._subscribe_market_data()
    
    def _on_connection_lost(self):
        """连接断开事件"""
        self.logger.warning("IBKR连接已断开")
    
    def _on_order_filled(self, data: Dict):
        """订单成交事件"""
        order_id = data['orderId']
        filled = data['filled']
        avg_price = data['avgFillPrice']
        
        self.logger.info(f"订单成交: {order_id}, 数量: {filled}, 价格: {avg_price}")
        
        # 更新持仓记录
        if order_id in self.pending_orders:
            order_info = self.pending_orders[order_id]
            symbol = order_info['symbol']
            action = order_info['action']
            
            if symbol not in self.active_positions:
                self.active_positions[symbol] = {
                    'shares': 0,
                    'avg_price': 0,
                    'entry_time': datetime.now()
                }
            
            position = self.active_positions[symbol]
            
            if action == "BUY":
                new_shares = position['shares'] + filled
                new_avg_price = ((position['shares'] * position['avg_price']) + 
                               (filled * avg_price)) / new_shares
                position['shares'] = new_shares
                position['avg_price'] = new_avg_price
            else:  # SELL
                position['shares'] -= filled
            
            # 如果持仓归零，移除记录
            if position['shares'] == 0:
                del self.active_positions[symbol]
            
            # 移除已完成订单
            del self.pending_orders[order_id]
    
    def _on_order_cancelled(self, data: Dict):
        """订单取消事件"""
        order_id = data['orderId']
        self.logger.info(f"订单已取消: {order_id}")
        
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def _on_market_data_update(self, data: Dict):
        """市场数据更新事件"""
        req_id = data['reqId']
        tick_type = data['tickType']
        price = data['price']
        contract = data.get('contract')
        
        if contract:
            symbol = contract.symbol
            self.data_processor.update_tick_data(symbol, tick_type, price)
            
            # 检查是否需要生成信号
            if tick_type == 4:  # LAST_PRICE
                self._check_trading_signals(symbol)
    
    def _on_position_update(self, data: Dict):
        """持仓更新事件"""
        symbol = data['symbol']
        position = data['position']
        avg_cost = data['avgCost']
        
        self.logger.info(f"持仓更新: {symbol}, 数量: {position}, 成本: {avg_cost}")
    
    def _subscribe_market_data(self):
        """订阅市场数据"""
        watchlist = self.config['trading']['watchlist']
        
        for symbol in watchlist:
            if symbol not in self.subscribed_symbols:
                contract = self.ibkr_client.create_stock_contract(symbol)
                req_id = self.ibkr_client.subscribe_market_data(contract)
                
                if req_id > 0:
                    self.subscribed_symbols.add(symbol)
                    self.logger.info(f"订阅市场数据: {symbol}")
    
    def _check_trading_signals(self, symbol: str):
        """检查交易信号"""
        if not self.running:
            return
        
        signal = self.signal_generator.generate_signal(symbol)
        
        if signal and signal.signal_strength >= self.config['trading']['signal_threshold']:
            self._execute_signal(signal)
    
    def _execute_signal(self, signal: TradingSignal):
        """执行交易信号"""
        try:
            # 风险检查
            if not self._risk_check(signal):
                return
            
            # 创建订单
            contract = self.ibkr_client.create_stock_contract(signal.symbol)
            order = self.ibkr_client.create_market_order(signal.action, signal.quantity)
            
            # 提交订单
            order_id = self.ibkr_client.place_order(contract, order)
            
            if order_id > 0:
                # 记录订单
                self.pending_orders[order_id] = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'signal': signal,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"提交订单: {signal.action} {signal.quantity} {signal.symbol} "
                               f"(信号强度: {signal.signal_strength:.2f}, 原因: {signal.reason})")
            
        except Exception as e:
            self.logger.error(f"执行交易信号失败: {e}")
    
    def _risk_check(self, signal: TradingSignal) -> bool:
        """风险检查"""
        try:
            # 获取账户信息
            account_info = self.ibkr_client.get_account_summary()
            if not account_info:
                self.logger.warning("无法获取账户信息，跳过风险检查")
                return True
            
            # 估算当前价格
            current_price = self.data_processor.get_latest_price(signal.symbol)
            if not current_price:
                self.logger.warning(f"无法获取{signal.symbol}当前价格")
                return False
            
            # 估算组合价值
            portfolio_value = 100000  # 默认值，实际应从账户信息获取
            
            # 检查仓位大小
            ok, msg = self.risk_manager.check_position_size(
                signal.symbol, signal.quantity, current_price, portfolio_value
            )
            if not ok:
                self.logger.warning(f"风险检查失败: {msg}")
                return False
            
            # 检查日内亏损限制
            ok, msg = self.risk_manager.check_daily_loss_limit()
            if not ok:
                self.logger.warning(f"风险检查失败: {msg}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return False
    
    def start(self):
        """启动交易策略"""
        if not self.initialize():
            self.logger.error("系统初始化失败，无法启动")
            return False
        
        self.running = True
        self.logger.info("交易策略已启动")
        
        # 订阅市场数据
        self._subscribe_market_data()
        
        # 启动策略线程
        self.strategy_thread = threading.Thread(target=self._strategy_loop, daemon=True)
        self.strategy_thread.start()
        
        return True
    
    def stop(self):
        """停止交易策略"""
        self.running = False
        self.logger.info("正在停止交易策略...")
        
        if self.ibkr_client:
            self.ibkr_client.disconnect()
        
        if self.strategy_thread:
            self.strategy_thread.join(timeout=5)
        
        self.logger.info("交易策略已停止")
    
    def _strategy_loop(self):
        """策略主循环"""
        self.logger.info("策略主循环已启动")
        
        while self.running:
            try:
                # 定期检查和更新
                self._periodic_check()
                
                # 暂停1秒
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"策略循环错误: {e}")
                time.sleep(5)
    
    def _periodic_check(self):
        """定期检查"""
        # 检查连接状态
        if not self.ibkr_client or not self.ibkr_client.is_connected():
            return
        
        # 每60秒检查一次持仓和风险
        current_time = datetime.now()
        if not hasattr(self, '_last_check_time'):
            self._last_check_time = current_time
        
        if (current_time - self._last_check_time).seconds >= 60:
            self._check_positions_and_risk()
            self._last_check_time = current_time
    
    def _check_positions_and_risk(self):
        """检查持仓和风险"""
        try:
            # 获取当前持仓
            positions = self.ibkr_client.get_current_positions()
            
            for position in positions:
                symbol = position['symbol']
                shares = position['position']
                avg_cost = position['avgCost']
                
                # 获取当前价格
                current_price = self.data_processor.get_latest_price(symbol)
                if not current_price:
                    continue
                
                # 计算盈亏
                if shares > 0:  # 多头
                    pnl_pct = (current_price - avg_cost) / avg_cost
                else:  # 空头
                    pnl_pct = (avg_cost - current_price) / avg_cost
                
                # 检查止损止盈
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    self._execute_stop_loss(symbol, abs(shares))
                elif pnl_pct >= self.risk_manager.take_profit_pct:
                    self._execute_take_profit(symbol, abs(shares))
            
        except Exception as e:
            self.logger.error(f"持仓风险检查失败: {e}")
    
    def _execute_stop_loss(self, symbol: str, quantity: int):
        """执行止损"""
        try:
            action = "SELL"  # 简化处理，实际需要根据持仓方向确定
            
            contract = self.ibkr_client.create_stock_contract(symbol)
            order = self.ibkr_client.create_market_order(action, quantity)
            
            order_id = self.ibkr_client.place_order(contract, order)
            
            if order_id > 0:
                self.logger.warning(f"执行止损: {action} {quantity} {symbol}")
            
        except Exception as e:
            self.logger.error(f"止损执行失败: {e}")
    
    def _execute_take_profit(self, symbol: str, quantity: int):
        """执行止盈"""
        try:
            action = "SELL"  # 简化处理
            
            contract = self.ibkr_client.create_stock_contract(symbol)
            order = self.ibkr_client.create_market_order(action, quantity)
            
            order_id = self.ibkr_client.place_order(contract, order)
            
            if order_id > 0:
                self.logger.info(f"执行止盈: {action} {quantity} {symbol}")
            
        except Exception as e:
            self.logger.error(f"止盈执行失败: {e}")
    
    def get_status(self) -> Dict:
        """获取策略状态"""
        return {
            'running': self.running,
            'connected': self.ibkr_client.is_connected() if self.ibkr_client else False,
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'subscribed_symbols': len(self.subscribed_symbols),
            'last_update': datetime.now().isoformat()
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版交易策略')
    parser.add_argument('--config', default='trading_config.json', help='配置文件路径')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # 创建并启动策略
    strategy = EnhancedTradingStrategy(args.config)
    
    try:
        if strategy.start():
            print("交易策略已启动，按Ctrl+C停止...")
            
            while True:
                time.sleep(1)
                
                # 显示状态
                status = strategy.get_status()
                print(f"\\r状态: 运行={status['running']}, 连接={status['connected']}, "
                      f"持仓={status['active_positions']}, 订单={status['pending_orders']}", end="")
        
    except KeyboardInterrupt:
        print("\\n正在停止交易策略...")
        strategy.stop()
        print("交易策略已停止")


if __name__ == "__main__":
    main()