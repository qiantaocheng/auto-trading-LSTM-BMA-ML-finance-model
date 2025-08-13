#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成风控收益平衡器
支持GUI一键启用/禁用，集成Polygon数据源和IBKR交易系统
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ==================== 数据结构定义 ====================

class TradeAction(Enum):
    APPROVE = "APPROVE"
    DEGRADE = "DEGRADE" 
    REJECT = "REJECT"

class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Quote:
    """报价数据"""
    last: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    tick_size: float = 0.01
    source: str = "DELAYED"
    timestamp: Optional[datetime] = None
    
    @property
    def mid(self) -> float:
        """中间价"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        """价差基点"""
        if self.spread > 0 and self.mid > 0:
            return (self.spread / self.mid) * 10000
        return 0.0

@dataclass
class Metrics:
    """股票指标数据"""
    prev_close: float
    atr_14: float = 0.0
    adv_usd_20: float = 0.0  # 20日平均成交金额
    adv_shares_20: float = 0.0  # 20日平均成交股数
    median_spread_bps_20: Optional[float] = None
    sigma_15m: Optional[float] = None
    market_cap: Optional[float] = None
    
@dataclass
class Signal:
    """交易信号"""
    symbol: str
    side: TradeSide
    expected_alpha_bps: float  # 预期超额收益(基点)
    model_price: float  # 模型估值
    confidence: float  # 信心度 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class Decision:
    """交易决策"""
    action: TradeAction
    reason: str
    shrink_to_pct: Optional[float] = None  # 降级时缩减比例
    
@dataclass
class RealtimeGuards:
    """实时风控参数"""
    max_spread_bps_largecap: float = 30.0  # 大盘股最大价差
    max_spread_bps_smallcap: float = 80.0  # 小盘股最大价差
    k_spread: float = 1.5  # 价差容忍倍数
    k_atr: float = 0.5  # ATR容忍倍数
    min_dollar_depth: float = 5000.0  # 最小美元深度
    min_lot: int = 100  # 最小手数
    largecap_threshold: float = 10e9  # 大盘股门槛(100亿美元)

@dataclass
class SizingConfig:
    """仓位配置"""
    max_weight: float = 0.03  # 单票最大权重3%
    max_child_adv_pct: float = 0.10  # 单笔不超过ADV的10%
    max_child_book_pct: float = 0.10  # 单笔不超过盘口的10%
    min_child_shares: int = 50  # 最小交易股数

@dataclass
class DegradePolicy:
    """降级策略"""
    on_wide_spread_shrink_to_pct: float = 0.5  # 价差过宽时缩减到50%
    on_thin_liquidity_shrink_to_pct: float = 0.3  # 流动性不足时缩减到30%
    on_price_drift_reject: bool = True  # 价格漂移时直接拒绝

@dataclass
class ThrottleConfig:
    """节流配置"""
    min_interval_seconds: int = 300  # 同标的最小间隔5分钟
    max_orders_per_minute: int = 10  # 每分钟最大订单数

@dataclass
class Config:
    """完整配置"""
    # 流动性筛选
    min_price: float = 5.0
    min_adv_usd: float = 500000.0  # 50万美元
    max_median_spread_bps: float = 50.0
    
    # 信号门槛
    min_alpha_bps: float = 50.0  # 0.5%
    min_alpha_vs_15m_sigma: float = 2.0
    
    # 组合权重
    top_n_boost: int = 10
    top_n_boost_multiplier: float = 1.25
    
    # 子配置
    guards: RealtimeGuards = field(default_factory=RealtimeGuards)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    degrade: DegradePolicy = field(default_factory=DegradePolicy)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)

# ==================== 节流状态管理 ====================

class ThrottleState:
    """节流状态管理"""
    
    def __init__(self):
        self.last_sent = {}  # symbol -> timestamp
        self.orders_per_minute = deque()  # timestamp queue
        self.lock = threading.Lock()
    
    def can_send(self, config: ThrottleConfig, symbol: str) -> bool:
        """检查是否可以发送订单"""
        with self.lock:
            now = time.time()
            
            # 检查单标的间隔
            if symbol in self.last_sent:
                if now - self.last_sent[symbol] < config.min_interval_seconds:
                    return False
            
            # 检查全局频率限制
            # 清除1分钟前的记录
            while self.orders_per_minute and now - self.orders_per_minute[0] > 60:
                self.orders_per_minute.popleft()
            
            if len(self.orders_per_minute) >= config.max_orders_per_minute:
                return False
                
            return True
    
    def mark_sent(self, symbol: str):
        """标记订单已发送"""
        with self.lock:
            now = time.time()
            self.last_sent[symbol] = now
            self.orders_per_minute.append(now)

# ==================== 核心决策函数 ====================

def passes_static_liquidity_filters(config: Config, metrics: Metrics) -> Tuple[bool, str]:
    """静态流动性筛选"""
    if metrics.prev_close < config.min_price:
        return False, f"价格过低: {metrics.prev_close:.2f} < {config.min_price}"
    
    if metrics.adv_usd_20 < config.min_adv_usd:
        return False, f"成交金额不足: {metrics.adv_usd_20:.0f} < {config.min_adv_usd}"
    
    if (metrics.median_spread_bps_20 and 
        metrics.median_spread_bps_20 > config.max_median_spread_bps):
        return False, f"历史价差过大: {metrics.median_spread_bps_20:.1f}bps > {config.max_median_spread_bps}bps"
    
    return True, "通过流动性筛选"

def passes_signal_thresholds(config: Config, signal: Signal, metrics: Metrics) -> Tuple[bool, str]:
    """信号门槛检查"""
    if signal.expected_alpha_bps < config.min_alpha_bps:
        return False, f"Alpha不足: {signal.expected_alpha_bps:.1f}bps < {config.min_alpha_bps}bps"
    
    if (metrics.sigma_15m and 
        signal.expected_alpha_bps < config.min_alpha_vs_15m_sigma * metrics.sigma_15m):
        required = config.min_alpha_vs_15m_sigma * metrics.sigma_15m
        return False, f"Alpha vs sigma不足: {signal.expected_alpha_bps:.1f}bps < {required:.1f}bps"
    
    return True, "通过信号门槛"

def build_limit_price(config: Config, signal: Signal, quote_delayed: Quote, 
                     metrics: Metrics, quote_rt: Optional[Quote] = None) -> float:
    """构建限价"""
    # 优先使用实时报价
    if quote_rt and quote_rt.source == "REALTIME":
        quote = quote_rt
    else:
        quote = quote_delayed
    
    # 基础价格
    if quote.bid and quote.ask:
        base_price = quote.mid
    else:
        base_price = quote.last
    
    # 计算价格带宽度
    atr_band = metrics.atr_14 * 0.01 if metrics.atr_14 > 0 else 0
    min_band = base_price * 0.003  # 最小30bps
    band = max(atr_band, min_band)
    
    # 根据信号方向调整
    if signal.side == TradeSide.BUY:
        # 买单: 不高追，在mid到bid之间
        if quote.bid:
            limit_price = min(base_price, quote.bid + band)
        else:
            limit_price = base_price - band
    else:
        # 卖单: 不低砍，在mid到ask之间  
        if quote.ask:
            limit_price = max(base_price, quote.ask - band)
        else:
            limit_price = base_price + band
    
    # 对齐tick_size
    tick_size = quote.tick_size
    limit_price = round(limit_price / tick_size) * tick_size
    
    return max(limit_price, tick_size)  # 确保价格为正

def should_trade(config: Config, signal: Signal, quote_delayed: Quote,
                metrics: Metrics, quote_rt: Optional[Quote] = None) -> Decision:
    """核心交易决策"""
    
    # 1. 流动性筛选
    liquidity_ok, liquidity_reason = passes_static_liquidity_filters(config, metrics)
    if not liquidity_ok:
        return Decision(TradeAction.REJECT, liquidity_reason)
    
    # 2. 信号门槛
    signal_ok, signal_reason = passes_signal_thresholds(config, signal, metrics)
    if not signal_ok:
        return Decision(TradeAction.REJECT, signal_reason)
    
    # 3. 实时风控检查(如果有实时数据)
    if quote_rt and quote_rt.source == "REALTIME":
        # 价差检查
        is_largecap = (metrics.market_cap and 
                      metrics.market_cap > config.guards.largecap_threshold)
        max_spread = (config.guards.max_spread_bps_largecap if is_largecap 
                     else config.guards.max_spread_bps_smallcap)
        
        if quote_rt.spread_bps > max_spread:
            if config.degrade.on_wide_spread_shrink_to_pct:
                return Decision(TradeAction.DEGRADE, 
                               f"价差过宽: {quote_rt.spread_bps:.1f}bps > {max_spread}bps",
                               config.degrade.on_wide_spread_shrink_to_pct)
            else:
                return Decision(TradeAction.REJECT, 
                               f"价差过宽: {quote_rt.spread_bps:.1f}bps > {max_spread}bps")
        
        # 价格漂移检查
        if abs(quote_rt.last - metrics.prev_close) > metrics.atr_14 * config.guards.k_atr:
            if config.degrade.on_price_drift_reject:
                return Decision(TradeAction.REJECT, 
                               f"价格漂移过大: {abs(quote_rt.last - metrics.prev_close):.2f}")
        
        # 流动性深度检查  
        if quote_rt.bid_size and quote_rt.ask_size:
            min_depth = min(quote_rt.bid_size * quote_rt.bid if quote_rt.bid else 0,
                           quote_rt.ask_size * quote_rt.ask if quote_rt.ask else 0)
            if min_depth < config.guards.min_dollar_depth:
                return Decision(TradeAction.DEGRADE,
                               f"流动性不足: {min_depth:.0f} < {config.guards.min_dollar_depth}",
                               config.degrade.on_thin_liquidity_shrink_to_pct)
    
    return Decision(TradeAction.APPROVE, "通过所有检查")

def allocate_portfolio(config: Config, signals: List[Signal]) -> Dict[str, float]:
    """组合权重分配"""
    if not signals:
        return {}
    
    # 计算原始权重 (alpha * confidence)
    weights = {}
    for signal in signals:
        raw_weight = signal.expected_alpha_bps * signal.confidence
        weights[signal.symbol] = max(raw_weight, 0)
    
    if not weights or sum(weights.values()) == 0:
        return {}
    
    # 按权重排序，给Top-N加权
    sorted_symbols = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)
    top_n_symbols = sorted_symbols[:config.top_n_boost]
    
    for symbol in top_n_symbols:
        weights[symbol] *= config.top_n_boost_multiplier
    
    # 归一化
    total_weight = sum(weights.values())
    for symbol in weights:
        weights[symbol] /= total_weight
    
    # 应用单票上限
    for symbol in weights:
        weights[symbol] = min(weights[symbol], config.sizing.max_weight)
    
    # 重新归一化
    total_weight = sum(weights.values())
    if total_weight > 0:
        for symbol in weights:
            weights[symbol] /= total_weight
    
    return weights

def compute_child_qty(target_delta: int, adv_shares: float, top_of_book_shares: Optional[int],
                     config: SizingConfig) -> int:
    """计算子订单数量"""
    if target_delta == 0:
        return 0
    
    # 基于ADV限制
    max_by_adv = int(adv_shares * config.max_child_adv_pct)
    
    # 基于盘口限制
    max_by_book = float('inf')
    if top_of_book_shares:
        max_by_book = int(top_of_book_shares * config.max_child_book_pct)
    
    # 取最小值
    max_qty = min(max_by_adv, max_by_book)
    child_qty = min(abs(target_delta), max_qty)
    
    # 应用最小数量限制
    if child_qty < config.min_child_shares:
        return 0
    
    return child_qty if target_delta > 0 else -child_qty

# ==================== 集成风控收益平衡器类 ====================

class RiskRewardBalancer:
    """
    集成风控收益平衡器
    可通过GUI一键启用/禁用，集成Polygon数据源和IBKR交易系统
    """
    
    def __init__(self, polygon_client=None, ibkr_trader=None, config: Optional[Config] = None):
        """
        初始化风控收益平衡器
        
        Args:
            polygon_client: Polygon数据客户端
            ibkr_trader: IBKR交易客户端
            config: 配置参数
        """
        self.enabled = False  # 默认不启用
        self.config = config or Config()
        self.throttle = ThrottleState()
        
        # 外部依赖
        self.polygon_client = polygon_client
        self.ibkr_trader = ibkr_trader
        
        # 缓存
        self.metrics_cache = {}
        self.quote_cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        # 统计信息
        self.stats = {
            'total_signals': 0,
            'approved': 0,
            'degraded': 0,
            'rejected': 0,
            'orders_sent': 0,
            'last_reset': datetime.now()
        }
        
        logger.info("风控收益平衡器初始化完成")
    
    def enable(self):
        """启用风控收益平衡器"""
        self.enabled = True
        logger.info("风控收益平衡器已启用")
    
    def disable(self):
        """禁用风控收益平衡器"""
        self.enabled = False
        logger.info("风控收益平衡器已禁用")
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled
    
    def update_config(self, new_config: Config):
        """更新配置"""
        self.config = new_config
        logger.info("风控收益平衡器配置已更新")
    
    def get_polygon_metrics(self, symbol: str) -> Optional[Metrics]:
        """从Polygon获取股票指标"""
        if not self.polygon_client:
            logger.warning("Polygon客户端未配置")
            return None
        
        # 检查缓存
        cache_key = f"{symbol}_metrics"
        if cache_key in self.metrics_cache:
            cached_time, cached_data = self.metrics_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            # 获取基础信息
            ticker_details = self.polygon_client.get_ticker_details(symbol)
            if not ticker_details:
                return None
            
            # 获取历史数据计算指标
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            hist_data = self.polygon_client.download(
                symbol, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if hist_data is None or len(hist_data) < 20:
                logger.warning(f"历史数据不足: {symbol}")
                return None
            
            # 计算指标
            prev_close = float(hist_data['Close'].iloc[-1])
            
            # ATR计算
            high_low = hist_data['High'] - hist_data['Low']
            high_close = abs(hist_data['High'] - hist_data['Close'].shift(1))
            low_close = abs(hist_data['Low'] - hist_data['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            
            # 成交量指标
            volume_20 = hist_data['Volume'].rolling(20).mean().iloc[-1]
            adv_usd_20 = volume_20 * prev_close
            
            # 价差估算(基于波动率)
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            estimated_spread_bps = volatility * 10000 * 0.5  # 估算价差
            
            metrics = Metrics(
                prev_close=prev_close,
                atr_14=atr_14 if pd.notna(atr_14) else prev_close * 0.02,
                adv_usd_20=adv_usd_20,
                adv_shares_20=volume_20,
                median_spread_bps_20=estimated_spread_bps,
                market_cap=ticker_details.get('market_cap', 1e9)
            )
            
            # 缓存结果
            self.metrics_cache[cache_key] = (time.time(), metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"获取{symbol}指标失败: {e}")
            return None
    
    def get_polygon_quote(self, symbol: str, realtime: bool = False) -> Optional[Quote]:
        """从Polygon获取报价"""
        if not self.polygon_client:
            return None
        
        cache_key = f"{symbol}_quote_{'rt' if realtime else 'delayed'}"
        if cache_key in self.quote_cache:
            cached_time, cached_data = self.quote_cache[cache_key]
            # 实时报价缓存时间更短
            ttl = 30 if realtime else 300
            if time.time() - cached_time < ttl:
                return cached_data
        
        try:
            if realtime:
                # 尝试获取实时报价
                rt_quote = self.polygon_client.get_real_time_quote(symbol)
                if rt_quote:
                    quote = Quote(
                        last=rt_quote.get('last', 0),
                        bid=rt_quote.get('bid'),
                        ask=rt_quote.get('ask'),
                        bid_size=rt_quote.get('bid_size'),
                        ask_size=rt_quote.get('ask_size'),
                        tick_size=0.01,  # 默认tick
                        source="REALTIME",
                        timestamp=datetime.now()
                    )
                    self.quote_cache[cache_key] = (time.time(), quote)
                    return quote
            
            # 获取延时报价
            delayed_data = self.polygon_client.download(symbol, period="1d", interval="1m")
            if delayed_data is not None and len(delayed_data) > 0:
                last_close = float(delayed_data['Close'].iloc[-1])
                
                quote = Quote(
                    last=last_close,
                    tick_size=0.01,
                    source="DELAYED",
                    timestamp=datetime.now()
                )
                
                self.quote_cache[cache_key] = (time.time(), quote)
                return quote
                
        except Exception as e:
            logger.error(f"获取{symbol}报价失败: {e}")
            
        return None
    
    def process_signals(self, signals: List[Signal], current_positions: Dict[str, int] = None,
                       portfolio_nav: float = 1000000) -> List[Dict]:
        """
        处理交易信号，返回订单列表
        
        Args:
            signals: 交易信号列表
            current_positions: 当前持仓 {symbol: shares}
            portfolio_nav: 组合净值
            
        Returns:
            订单列表
        """
        if not self.enabled:
            logger.info("风控收益平衡器未启用，跳过信号处理")
            return []
        
        if not signals:
            return []
        
        current_positions = current_positions or {}
        orders = []
        
        # 更新统计
        self.stats['total_signals'] += len(signals)
        
        # 1. 组合权重分配
        weights = allocate_portfolio(self.config, signals)
        
        for signal in signals:
            try:
                # 获取市场数据
                metrics = self.get_polygon_metrics(signal.symbol)
                if not metrics:
                    logger.warning(f"无法获取{signal.symbol}的市场数据，跳过")
                    continue
                
                quote_delayed = self.get_polygon_quote(signal.symbol, realtime=False)
                quote_rt = self.get_polygon_quote(signal.symbol, realtime=True)
                
                if not quote_delayed:
                    logger.warning(f"无法获取{signal.symbol}的报价，跳过")
                    continue
                
                # 2. 交易决策
                decision = should_trade(self.config, signal, quote_delayed, metrics, quote_rt)
                
                if decision.action == TradeAction.REJECT:
                    logger.info(f"拒绝交易{signal.symbol}: {decision.reason}")
                    self.stats['rejected'] += 1
                    continue
                
                if decision.action == TradeAction.DEGRADE:
                    logger.info(f"降级交易{signal.symbol}: {decision.reason}")
                    self.stats['degraded'] += 1
                else:
                    self.stats['approved'] += 1
                
                # 3. 计算目标仓位
                target_weight = weights.get(signal.symbol, 0)
                target_value = target_weight * portfolio_nav
                target_shares = int(target_value / metrics.prev_close) if metrics.prev_close > 0 else 0
                
                current_shares = current_positions.get(signal.symbol, 0)
                target_delta = target_shares - current_shares
                
                # 4. 应用降级
                if decision.action == TradeAction.DEGRADE and decision.shrink_to_pct:
                    target_delta = int(target_delta * decision.shrink_to_pct)
                
                if abs(target_delta) < self.config.sizing.min_child_shares:
                    continue
                
                # 5. 节流检查
                if not self.throttle.can_send(self.config.throttle, signal.symbol):
                    logger.info(f"节流限制，跳过{signal.symbol}")
                    continue
                
                # 6. 计算订单数量
                child_qty = compute_child_qty(
                    target_delta,
                    metrics.adv_shares_20,
                    None,  # 暂时不使用盘口深度
                    self.config.sizing
                )
                
                if child_qty == 0:
                    continue
                
                # 7. 计算限价
                limit_price = build_limit_price(self.config, signal, quote_delayed, metrics, quote_rt)
                
                # 8. 构造订单
                side = "BUY" if child_qty > 0 else "SELL"
                order = {
                    'symbol': signal.symbol,
                    'side': side,
                    'quantity': abs(child_qty),
                    'limit_price': limit_price,
                    'order_type': 'LMT',
                    'time_in_force': 'DAY',
                    'signal_alpha': signal.expected_alpha_bps,
                    'signal_confidence': signal.confidence,
                    'decision_action': decision.action.value,
                    'decision_reason': decision.reason
                }
                
                orders.append(order)
                self.throttle.mark_sent(signal.symbol)
                self.stats['orders_sent'] += 1
                
                logger.info(f"生成订单: {signal.symbol} {side} {abs(child_qty)}@{limit_price:.2f}")
                
            except Exception as e:
                logger.error(f"处理信号{signal.symbol}时出错: {e}")
                continue
        
        logger.info(f"处理{len(signals)}个信号，生成{len(orders)}个订单")
        return orders
    
    def send_orders_to_ibkr(self, orders: List[Dict]) -> bool:
        """发送订单到IBKR"""
        if not self.ibkr_trader:
            logger.warning("IBKR交易客户端未配置")
            return False
        
        if not orders:
            return True
        
        try:
            # 调用IBKR交易接口
            success_count = 0
            for order in orders:
                try:
                    # 这里调用你现有的IBKR交易接口
                    # 假设有place_limit_order方法
                    if hasattr(self.ibkr_trader, 'place_limit_order'):
                        result = self.ibkr_trader.place_limit_order(
                            symbol=order['symbol'],
                            side=order['side'],
                            quantity=order['quantity'],
                            limit_price=order['limit_price']
                        )
                        if result:
                            success_count += 1
                            logger.info(f"订单提交成功: {order['symbol']}")
                        else:
                            logger.error(f"订单提交失败: {order['symbol']}")
                    
                except Exception as e:
                    logger.error(f"提交订单{order['symbol']}失败: {e}")
                    
            logger.info(f"成功提交{success_count}/{len(orders)}个订单")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"批量提交订单失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_signals': 0,
            'approved': 0,
            'degraded': 0,
            'rejected': 0,
            'orders_sent': 0,
            'last_reset': datetime.now()
        }
        logger.info("统计信息已重置")
    
    def clear_cache(self):
        """清理缓存"""
        self.metrics_cache.clear()
        self.quote_cache.clear()
        logger.info("缓存已清理")