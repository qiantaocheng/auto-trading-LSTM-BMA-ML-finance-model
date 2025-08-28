#!/usr/bin/env python3

# Enhanced error handling
try:
    from .error_handling_system import (
        get_error_handler, with_error_handling, error_handling_context,
        ErrorSeverity, ErrorCategory, ErrorContext
    )
except ImportError:
    from error_handling_system import (
        get_error_handler, with_error_handling, error_handling_context,
        ErrorSeverity, ErrorCategory, ErrorContext
    )

"""
统一交易核心 - 整合所有AutoTrader功能
替代多个重复的管理器和引擎
"""

import asyncio
import logging
import math
import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from threading import RLock, Lock
from enum import Enum
import json
import os
import sys

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from polygon_client import polygon_client, download, Ticker
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

# Import new enhanced modules
try:
    from .labeling import EnhancedLabelingPipeline, create_enhanced_labeling_pipeline
    from .factors_pit import EnhancedFactorPipeline, create_enhanced_factor_pipeline
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False


@dataclass
class Quote:
    """报价数据"""
    bid: float
    ask: float
    bidSize: float = 0.0
    askSize: float = 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class AccountSnapshot:
    """账户快照"""
    timestamp: float = field(default_factory=time.time)
    account_id: str = ""
    total_cash: float = 0.0
    available_funds: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    account_values: Dict[str, str] = field(default_factory=dict)
    currency: str = "USD"
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class OrderState:
    """订单状态"""
    order_id: int
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    order_type: str  # MKT/LMT
    status: str = "PendingSubmit"
    filled: int = 0
    remaining: int = 0
    avg_fill_price: float = 0.0
    last_fill_price: float = 0.0
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # 智能头寸计算相关字段
    position_calculation: Optional[Dict[str, Any]] = None
    signal_strength: Optional[float] = None
    signal_confidence: Optional[float] = None


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING_SUBMIT = "PendingSubmit"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    ERROR = "Error"


class UnifiedTradingCore:
    """统一交易核心 - 集成所有交易功能"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedTradingCore")
        
        # 核心组件锁
        self.data_lock = RLock()
        self.order_lock = RLock()
        self.account_lock = Lock()
        
        # 数据存储
        self.tickers: Dict[str, Any] = {}
        self.quotes: Dict[str, Quote] = {}
        self.account_snapshot: Optional[AccountSnapshot] = None
        self.orders: Dict[int, OrderState] = {}
        self.positions: Dict[str, int] = {}
        
        # 历史数据
        self.price_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.order_history: List[OrderState] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Polygon集成
        self.polygon_available = POLYGON_AVAILABLE
        if self.polygon_available:
            self.polygon_client = polygon_client
            self.logger.info("Polygon数据源已集成")
        
        # Enhanced prediction modules
        self.enhanced_modules_available = ENHANCED_MODULES_AVAILABLE
        self.labeling_pipeline = None
        self.factor_pipeline = None
        
        if self.enhanced_modules_available:
            try:
                # Initialize labeling pipeline
                labeling_config = self.config.get('labeling', {})
                self.labeling_pipeline = create_enhanced_labeling_pipeline(labeling_config)
                
                # Initialize factor pipeline  
                factor_config = self.config.get('factors', {})
                self.factor_pipeline = create_enhanced_factor_pipeline(factor_config)
                
                self.logger.info("Enhanced prediction modules initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced modules: {e}")
                self.enhanced_modules_available = False
        
        # 配置参数
        self.max_orders_per_symbol = self.config.get('max_orders_per_symbol', 5)
        self.max_position_value = self.config.get('max_position_value', 100000)
        self.min_order_value = self.config.get('min_order_value', 1000)
        self.alloc_ratio = self.config.get('alloc_ratio', 0.03)
        
        # 性能监控
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = time.time()
        
        self.logger.info("统一交易核心初始化完成")
    
    # =============================================================================
    # 数据管理功能 (替代 data_source_manager.py)
    # =============================================================================
    
    def get_universe(self) -> List[str]:
        """获取股票池"""
        try:
            # 优先从filtered_stocks读取
            filtered_file = Path("../filtered_stocks_20250817_002928.txt")
            if filtered_file.exists():
                with open(filtered_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                self.logger.info(f"从过滤文件加载 {len(tickers)} 只股票")
                return tickers
            
            # 备选stocks.txt
            stocks_file = Path("../stocks.txt")
            if stocks_file.exists():
                with open(stocks_file, 'r') as f:
                    tickers = [line.strip() for line in f if line.strip()]
                self.logger.info(f"从stocks.txt加载 {len(tickers)} 只股票")
                return tickers
                
            # 🔒 移除硬编码默认股票池
            default_tickers = ['SPY']  # 使用ETF作为安全默认值
            self.logger.warning("使用安全默认股票池: SPY")
            return default_tickers
            
        except Exception as e:
            self.logger.error(f"加载股票池失败: {e}")
            # 🔒 移除硬编码，返回安全的默认值
            return ['SPY']  # 使用ETF作为安全默认值
    
    def get_polygon_factors(self, symbol: str) -> Dict[str, float]:
        """获取Polygon因子数据"""
        if not self.polygon_available:
            return {}
        
        try:
            # 这里应该调用具体的Polygon API
            # 暂时返回模拟数据
            return {
                'sma_20': 150.0,
                'rsi_14': 65.0,
                'volume_ratio': 1.2,
                'price_momentum': 0.05
            }
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return {}
    
    def get_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """获取交易信号"""
        try:
            quote = self.quotes.get(symbol)
            if not quote:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No quote data'}
            
            # 获取因子数据
            factors = self.get_polygon_factors(symbol)
            if not factors:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No factor data'}
            
            # 简单信号逻辑
            rsi = factors.get('rsi_14', 50.0)
            momentum = factors.get('price_momentum', 0.0)
            
            if rsi < 30 and momentum > 0.02:
                return {'signal': 'BUY', 'confidence': 0.8, 'reason': 'Oversold with momentum'}
            elif rsi > 70 and momentum < -0.02:
                return {'signal': 'SELL', 'confidence': 0.8, 'reason': 'Overbought with negative momentum'}
            else:
                return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Neutral conditions'}
                
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    # =============================================================================
    # 账户管理功能 (替代 account_data_manager.py)
    # =============================================================================
    
    async def refresh_account_data(self, ib_client) -> AccountSnapshot:
        """刷新账户数据"""
        with self.account_lock:
            try:
                # 获取账户信息
                account_values = {}
                if hasattr(ib_client, 'accountSummary'):
                    for item in ib_client.accountSummary():
                        account_values[item.tag] = item.value
                
                # 获取持仓
                positions = {}
                if hasattr(ib_client, 'positions'):
                    for pos in ib_client.positions():
                        if pos.position != 0:
                            positions[pos.contract.symbol] = int(pos.position)
                
                # 创建快照
                snapshot = AccountSnapshot(
                    timestamp=time.time(),
                    account_id=getattr(ib_client, 'account_id', ''),
                    total_cash=float(account_values.get('TotalCashValue', 0)),
                    available_funds=float(account_values.get('AvailableFunds', 0)),
                    positions=positions,
                    account_values=account_values,
                    is_valid=True
                )
                
                self.account_snapshot = snapshot
                self.positions.update(positions)
                
                self.logger.debug(f"账户数据已刷新，现金: ${snapshot.total_cash:,.2f}")
                return snapshot
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"刷新账户数据失败: {e}")
                
                error_snapshot = AccountSnapshot(
                    is_valid=False,
                    validation_errors=[str(e)]
                )
                return error_snapshot
    
    def get_available_cash(self) -> float:
        """获取可用现金"""
        if self.account_snapshot and self.account_snapshot.is_valid:
            return self.account_snapshot.available_funds
        return 0.0
    
    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        return self.positions.get(symbol, 0)
    
    # =============================================================================
    # 订单管理功能 (替代 order_state_machine.py + enhanced_order_execution.py)
    # =============================================================================
    
    def create_order(self, symbol: str, action: str, quantity: int, 
                    order_type: str = "MKT", price: float = None) -> Optional[OrderState]:
        """创建订单"""
        with self.order_lock:
            try:
                # 风险检查
                if not self._validate_order(symbol, action, quantity, price):
                    return None
                
                # 生成订单ID
                order_id = len(self.orders) + 1000
                
                # 创建订单状态
                order_state = OrderState(
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    order_type=order_type,
                    remaining=quantity
                )
                
                self.orders[order_id] = order_state
                
                self.logger.info(f"创建订单: {action} {quantity} {symbol} @ {order_type}")
                return order_state
                
            except Exception as e:
                context = ErrorContext(
                    operation="unified_trading_core",
                    component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
                )
                get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
                return None
    
    def create_smart_order(self, symbol: str, action: str, 
                          signal_strength: float = 0.5,
                          signal_confidence: float = 0.8,
                          order_type: str = "MKT", 
                          price: float = None,
                          target_allocation_pct: float = None) -> Optional[OrderState]:
        """
        创建智能订单 - 基于资金百分比和信号强度动态计算股数
        
        Args:
            symbol: 股票代码
            action: 买卖方向 ("BUY"/"SELL")
            signal_strength: 信号强度 (-1 到 1)
            signal_confidence: 信号置信度 (0 到 1)
            order_type: 订单类型
            price: 价格 (如果为None则使用当前市价)
            target_allocation_pct: 目标资金分配百分比 (如果为None则使用默认5%)
            
        Returns:
            OrderState对象或None
        """
        from .position_size_calculator import create_position_calculator
        
        try:
            # 获取当前价格
            if price is None:
                quote = self.quotes.get(symbol)
                if quote:
                    price = quote.ask if action == "BUY" else quote.bid
                else:
                    self.logger.warning(f"{symbol}无报价数据，无法创建智能订单")
                    return None
            
            if price <= 0:
                self.logger.warning(f"{symbol}价格无效: {price}")
                return None
            
            # 获取可用资金
            available_cash = self.get_available_cash()
            if available_cash <= 0:
                self.logger.warning(f"可用资金不足: ${available_cash:,.2f}")
                return None
            
            # 创建头寸计算器
            calculator = create_position_calculator(
                target_percentage=target_allocation_pct or 0.05,  # 默认5%
                min_percentage=0.04,    # 4%最小
                max_percentage=0.10,    # 10%最大
                method="signal_strength"  # 使用信号强度调整方法
            )
            
            # 计算动态股数
            position_result = calculator.calculate_position_size(
                symbol=symbol,
                current_price=price,
                signal_strength=signal_strength,
                available_cash=available_cash,
                signal_confidence=signal_confidence
            )
            
            if not position_result.get('valid', False):
                self.logger.warning(f"{symbol}头寸计算失败: {position_result.get('reason', 'Unknown error')}")
                return None
            
            # 获取计算后的股数
            smart_quantity = position_result['shares']
            actual_allocation_pct = position_result['actual_percentage']
            
            if smart_quantity <= 0:
                self.logger.warning(f"{symbol}计算股数为0，跳过交易")
                return None
            
            # 记录头寸计算详情
            self.logger.info(f"{symbol}智能头寸计算: {smart_quantity}股, "
                           f"${position_result['actual_value']:,.2f} ({actual_allocation_pct:.1%}), "
                           f"信号强度{signal_strength:.2f}, 置信度{signal_confidence:.2f}")
            
            # 创建订单
            order_state = self.create_order(
                symbol=symbol,
                action=action,
                quantity=smart_quantity,
                order_type=order_type,
                price=price
            )
            
            # 添加头寸计算信息到订单状态
            if order_state:
                order_state.position_calculation = position_result
                order_state.signal_strength = signal_strength
                order_state.signal_confidence = signal_confidence
                
                self.logger.info(f"智能订单创建成功: {action} {smart_quantity} {symbol} "
                               f"(资金占比{actual_allocation_pct:.1%})")
            
            return order_state
            
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return None
    
    def update_order_status(self, order_id: int, status: str, 
                          filled: int = None, avg_price: float = None):
        """更新订单状态"""
        with self.order_lock:
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = status
                order.last_update = time.time()
                
                if filled is not None:
                    order.filled = filled
                    order.remaining = order.quantity - filled
                
                if avg_price is not None:
                    order.avg_fill_price = avg_price
                
                self.logger.debug(f"订单{order_id}状态更新: {status}")
    
    def cancel_order(self, order_id: int) -> bool:
        """取消订单"""
        try:
            if order_id in self.orders:
                self.update_order_status(order_id, OrderStatus.CANCELLED.value)
                self.logger.info(f"订单{order_id}已取消")
                return True
            return False
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return False
    
    def _validate_order(self, symbol: str, action: str, quantity: int, price: float) -> bool:
        """订单验证"""
        # 检查可用资金
        available_cash = self.get_available_cash()
        if action == "BUY":
            quote = self.quotes.get(symbol)
            estimated_cost = quantity * (price or quote.ask if quote else 100)
            # 对于智能订单，使用更宽松的验证（最大10%分配）
            max_allowed = available_cash * 0.15  # 允许最大15%用于单笔订单
            if estimated_cost > max_allowed:
                self.logger.warning(f"资金不足: 需要${estimated_cost:,.2f}, 允许${max_allowed:,.2f} (可用${available_cash:,.2f})")
                return False
        
        # 检查订单数量限制
        symbol_orders = [o for o in self.orders.values() 
                        if o.symbol == symbol and o.status not in ['Filled', 'Cancelled']]
        if len(symbol_orders) >= self.max_orders_per_symbol:
            self.logger.warning(f"{symbol}订单数量超限: {len(symbol_orders)}")
            return False
        
        return True
    
    # =============================================================================
    # 性能监控功能 (替代 performance_optimizer.py + resource_monitor.py)
    # =============================================================================
    
    def update_performance_metrics(self):
        """更新性能指标"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        self.performance_metrics.update({
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'active_orders': len([o for o in self.orders.values() 
                                if o.status not in ['Filled', 'Cancelled']]),
            'total_positions': len(self.positions),
            'last_update': current_time
        })
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        self.update_performance_metrics()
        
        # 计算健康分数
        error_rate = self.performance_metrics.get('error_rate', 0)
        health_score = max(0, min(100, 100 * (1 - error_rate * 10)))
        
        status = "HEALTHY"
        if health_score < 50:
            status = "CRITICAL"
        elif health_score < 80:
            status = "WARNING"
        
        return {
            'status': status,
            'health_score': health_score,
            'metrics': self.performance_metrics,
            'account_valid': self.account_snapshot.is_valid if self.account_snapshot else False,
            'polygon_available': self.polygon_available,
            'enhanced_modules_available': self.enhanced_modules_available
        }
    
    # =============================================================================
    # Enhanced Prediction Methods
    # =============================================================================
    
    def generate_calibrated_signal(self, symbol: str, features: pd.DataFrame, 
                                  raw_prediction: float, raw_confidence: float,
                                  reference_price: float) -> Dict[str, Any]:
        """
        🚀 P0 Generate calibrated trading signal using OOF calibration
        
        Args:
            symbol: Stock symbol
            features: Feature data for prediction
            raw_prediction: Raw model prediction (return)
            raw_confidence: Raw model confidence
            reference_price: Current reference price
            
        Returns:
            Dict: Calibrated signal compatible with plan_and_place_with_rr
        """
        try:
            # P0 OOF等值校准：使用IsotonicRegression校准器
            from .oof_calibration import calibrate_signal, get_oof_calibrator
            
            # 记录此次预测到OOF数据库（用于后续校准训练）
            calibrator = get_oof_calibrator()
            calibrator.record_oof_prediction(
                symbol=symbol,
                raw_prediction=raw_prediction,
                raw_confidence=raw_confidence,
                reference_price=reference_price,
                model_version="unified_trading_core"
            )
            
            # 使用校准器获取等值alpha和置信度
            expected_alpha_bps, calibrated_confidence = calibrate_signal(
                raw_prediction, raw_confidence
            )
            
            # 构造校准后的信号
            calibrated_signal = {
                "symbol": symbol,
                "side": "BUY" if raw_prediction > 0 else "SELL",
                "expected_alpha_bps": expected_alpha_bps,  # 校准后的期望alpha(bps)
                "confidence": calibrated_confidence,       # 校准后的置信度
                "reference_price": reference_price,
                "raw_prediction": raw_prediction,          # 保留原始预测用于调试
                "raw_confidence": raw_confidence,          # 保留原始置信度用于调试
                "signal_source": "oof_calibrated"
            }
            
            self.logger.debug(f"OOF校准 {symbol}: raw_pred={raw_prediction:.4f} -> "
                            f"alpha_bps={expected_alpha_bps:.1f}, "
                            f"raw_conf={raw_confidence:.3f} -> conf={calibrated_confidence:.3f}")
            
            return calibrated_signal
            
        except ImportError:
            # 如果OOF校准模块不可用，回退到简单校准
            self.logger.warning("OOF校准模块不可用，使用简单校准")
            return self._generate_simple_calibrated_signal(
                symbol, raw_prediction, raw_confidence, reference_price
            )
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return self._generate_simple_calibrated_signal(
                symbol, raw_prediction, raw_confidence, reference_price
            )
    
    def _generate_simple_calibrated_signal(self, symbol: str, raw_prediction: float, 
                                         raw_confidence: float, reference_price: float) -> Dict[str, Any]:
        """简单校准回退方案"""
        return {
            "symbol": symbol,
            "side": "BUY" if raw_prediction > 0 else "SELL",
            "expected_alpha_bps": abs(raw_prediction * 10000),
            "confidence": max(0.01, min(0.99, raw_confidence)),
            "reference_price": reference_price,
            "signal_source": "simple_fallback"
        }
        
        # 注释：原有的enhanced labeling pipeline代码已移除，使用OOF校准作为主要方案
    
    def process_enhanced_factors(self, market_data: pd.DataFrame, 
                               financial_data: Optional[pd.DataFrame] = None,
                               industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Process market data through enhanced factor pipeline
        
        Args:
            market_data: Market data with adj_close, market_cap, etc.
            financial_data: Optional financial statements data
            industry_data: Optional industry classification data
            
        Returns:
            DataFrame: Processed factors for model input
        """
        if not self.enhanced_modules_available or not self.factor_pipeline:
            self.logger.warning("Enhanced factor pipeline not available")
            return market_data  # Return original data as fallback
        
        try:
            if financial_data is not None and not financial_data.empty:
                # Use full PIT factor pipeline
                trading_dates = pd.to_datetime(market_data.index.get_level_values(0).unique())
                
                pit_factors = self.factor_pipeline.compute_all_pit_factors(
                    financial_data, market_data, trading_dates
                )
                
                if industry_data is not None and not industry_data.empty:
                    # Apply neutralization
                    neutralized_factors = self.factor_pipeline.neutralize_factors(
                        pit_factors, industry_data, market_data
                    )
                    
                    # Integrate with existing factors
                    final_factors = self.factor_pipeline.integrate_with_existing_factors(
                        neutralized_factors, market_data, integration_method='concat'
                    )
                    
                    self.logger.info(f"Enhanced factors processed: {len(final_factors.columns)} features")
                    return final_factors
                else:
                    self.logger.info("No industry data, skipping neutralization")
                    return pit_factors
            else:
                self.logger.info("No financial data, using market data only")
                return market_data
                
        except Exception as e:
            context = ErrorContext(
                operation="unified_trading_core",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return market_data  # Return original data as fallback
    
    def batch_generate_signals(self, signal_data: List[Dict]) -> List[Dict]:
        """
        Generate calibrated signals for multiple symbols in batch
        
        Args:
            signal_data: List of dicts with symbol, features, prediction, confidence, price
            
        Returns:
            List[Dict]: Calibrated signals ready for execution
        """
        calibrated_signals = []
        
        for signal_info in signal_data:
            try:
                symbol = signal_info.get('symbol')
                features = signal_info.get('features')
                raw_pred = signal_info.get('raw_prediction', 0.0)
                raw_conf = signal_info.get('raw_confidence', 0.5)
                ref_price = signal_info.get('reference_price', 0.0)
                
                if not symbol or ref_price <= 0:
                    self.logger.warning(f"Invalid signal data: {signal_info}")
                    continue
                
                calibrated_signal = self.generate_calibrated_signal(
                    symbol, features, raw_pred, raw_conf, ref_price
                )
                
                calibrated_signals.append(calibrated_signal)
                
            except Exception as e:
                self.logger.error(f"Failed to process signal for {signal_info.get('symbol', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Generated {len(calibrated_signals)} calibrated signals from {len(signal_data)} inputs")
        return calibrated_signals
    
    # =============================================================================
    # 数据订阅功能 (整合 engine.py 的 DataFeed)
    # =============================================================================
    
    async def subscribe_market_data(self, symbols: List[str], ib_client):
        """订阅市场数据"""
        try:
            for symbol in symbols:
                # 这里应该调用IB API订阅
                # 暂时创建模拟数据
                self.quotes[symbol] = Quote(bid=100.0, ask=100.1, bidSize=100, askSize=100)
                self.logger.debug(f"已订阅{symbol}市场数据")
            
            self.logger.info(f"已订阅 {len(symbols)} 个标的市场数据")
            
        except Exception as e:
            self.logger.error(f"订阅市场数据失败: {e}")
    
    async def unsubscribe_all_data(self, ib_client):
        """取消所有数据订阅"""
        try:
            # 清空报价数据
            symbols_count = len(self.quotes)
            self.quotes.clear()
            
            self.logger.info(f"已取消 {symbols_count} 个标的数据订阅")
            
        except Exception as e:
            self.logger.error(f"取消数据订阅失败: {e}")
    
    def update_quote(self, symbol: str, bid: float, ask: float, 
                    bid_size: float = 0, ask_size: float = 0):
        """更新报价数据"""
        with self.data_lock:
            quote = Quote(bid=bid, ask=ask, bidSize=bid_size, askSize=ask_size)
            self.quotes[symbol] = quote
            
            # 更新价格历史
            mid_price = quote.mid_price
            self.price_history[symbol].append((time.time(), mid_price))
            
            self.request_count += 1
    
    def get_best_quote(self, symbol: str) -> Optional[Quote]:
        """获取最佳报价"""
        return self.quotes.get(symbol)
    
    # =============================================================================
    # 公共接口
    # =============================================================================
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'core_status': 'RUNNING',
            'subscribed_symbols': len(self.quotes),
            'active_orders': len([o for o in self.orders.values() 
                                if o.status not in ['Filled', 'Cancelled']]),
            'positions': len(self.positions),
            'account_valid': self.account_snapshot.is_valid if self.account_snapshot else False,
            'available_cash': self.get_available_cash(),
            'performance': self.performance_metrics,
            'health': self.get_health_status()
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            self.quotes.clear()
            self.price_history.clear()
            self.logger.info("统一交易核心已清理")
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")


# =============================================================================
# 工厂函数
# =============================================================================

def create_unified_trading_core(config: Dict[str, Any] = None) -> UnifiedTradingCore:
    """创建统一交易核心实例"""
    return UnifiedTradingCore(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建核心实例
    core = create_unified_trading_core({
        'alloc_ratio': 0.05,
        'max_orders_per_symbol': 3
    })
    
    # 测试功能
    print("=== 统一交易核心测试 ===")
    
    # 测试股票池
    universe = core.get("scanner.universe", ["SPY"])
    print(f"股票池: {universe[:5]}... (共{len(universe)}只)")
    
    # 测试报价更新
    core.update_quote('AAPL', 150.0, 150.1, 100, 100)
    quote = core.get_best_quote('AAPL')
    print(f"AAPL报价: {quote}")
    
    # 测试交易信号
    signal = core.get_trading_signal('AAPL')
    print(f"AAPL信号: {signal}")
    
    # 测试订单创建
    order = core.create_order('AAPL', 'BUY', 100)
    print(f"创建订单: {order}")
    
    # 测试状态摘要
    status = core.get_status_summary()
    print(f"状态摘要: {status}")
    
    print("统一交易核心测试完成")