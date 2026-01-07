#!/usr/bin/env python3
"""
交易系统共享数据类型
Shared Data Types for Trading System

这个模块包含engine和ibkr_auto_trader共享的数据类，
用于打破循环依赖
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Quote:
    """市场报价数据"""
    bid: float                    # 买入价
    ask: float                    # 卖出价
    bidSize: float = 0.0         # 买单量
    askSize: float = 0.0         # 卖单量
    last: Optional[float] = None  # 最新成交价
    close: Optional[float] = None # 收盘价

    @property
    def mid(self) -> float:
        """中间价"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last or self.close or 0.0

    @property
    def spread(self) -> float:
        """买卖价差"""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """买卖价差（基点）"""
        mid = self.mid
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0


@dataclass
class Signal:
    """交易信号"""
    symbol: str                              # 股票代码
    side: str                                # 方向 "BUY" 或 "SELL"
    expected_alpha_bps: float = 0.0         # 预期Alpha（基点）
    model_price: Optional[float] = None      # 模型预测价格
    confidence: float = 1.0                  # 信号置信度 [0-1]
    signal_strength: float = 0.0             # 信号强度
    timestamp: float = 0.0                   # 时间戳
    source: str = "unknown"                  # 信号来源
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def __post_init__(self):
        """验证数据"""
        if self.side not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"Invalid side: {self.side}")

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    @property
    def is_buy(self) -> bool:
        return self.side == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.side == "SELL"

    @property
    def is_strong(self) -> bool:
        """是否是强信号（高置信度 + 高强度）"""
        return self.confidence >= 0.7 and abs(self.signal_strength) >= 0.6


@dataclass
class Metrics:
    """交易指标"""
    symbol: str                              # 股票代码

    # 信号指标
    signal_count: int = 0                    # 信号数量
    avg_signal_strength: float = 0.0         # 平均信号强度
    avg_confidence: float = 0.0              # 平均置信度

    # 执行指标
    orders_placed: int = 0                   # 已下订单数
    orders_filled: int = 0                   # 已成交订单数
    fill_rate: float = 0.0                   # 成交率

    # 盈亏指标
    realized_pnl: float = 0.0                # 已实现盈亏
    unrealized_pnl: float = 0.0              # 未实现盈亏
    total_pnl: float = 0.0                   # 总盈亏

    # 风险指标
    current_position: int = 0                # 当前仓位
    max_position: int = 0                    # 最大仓位
    position_value: float = 0.0              # 仓位市值

    # 性能指标
    sharpe_ratio: Optional[float] = None     # 夏普比率
    max_drawdown: Optional[float] = None     # 最大回撤
    win_rate: Optional[float] = None         # 胜率

    # 时间戳
    last_update: Optional[datetime] = None   # 最后更新时间

    def update_pnl(self, realized: float = 0.0, unrealized: float = 0.0):
        """更新盈亏"""
        if realized != 0.0:
            self.realized_pnl += realized
        if unrealized != 0.0:
            self.unrealized_pnl = unrealized
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.last_update = datetime.now()

    def update_fill_rate(self):
        """更新成交率"""
        if self.orders_placed > 0:
            self.fill_rate = self.orders_filled / self.orders_placed
        else:
            self.fill_rate = 0.0


@dataclass
class OrderRequest:
    """订单请求（engine → broker）"""
    symbol: str                              # 股票代码
    side: str                                # 方向 "BUY" 或 "SELL"
    quantity: int                            # 数量
    order_type: str = "LMT"                  # 订单类型 "MKT" "LMT" "STP"
    limit_price: Optional[float] = None      # 限价
    stop_price: Optional[float] = None       # 止损价

    # 止损止盈
    stop_loss_price: Optional[float] = None  # 止损价
    take_profit_price: Optional[float] = None # 止盈价

    # 元数据
    signal: Optional[Signal] = None          # 原始信号
    timestamp: float = 0.0                   # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """验证订单请求"""
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")

        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")

        if self.order_type == "LMT" and self.limit_price is None:
            raise ValueError("Limit price required for LMT orders")

        if self.order_type == "STP" and self.stop_price is None:
            raise ValueError("Stop price required for STP orders")


@dataclass
class OrderStatus:
    """订单状态（broker → engine）"""
    order_id: int                            # 订单ID
    symbol: str                              # 股票代码
    side: str                                # 方向
    quantity: int                            # 数量
    filled_quantity: int = 0                 # 已成交数量
    status: str = "PENDING"                  # 状态
    avg_fill_price: float = 0.0              # 平均成交价
    commission: float = 0.0                  # 佣金

    # 时间戳
    submitted_at: Optional[datetime] = None   # 提交时间
    filled_at: Optional[datetime] = None      # 成交时间

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        """是否已完全成交"""
        return self.filled_quantity >= self.quantity

    @property
    def is_partial(self) -> bool:
        """是否部分成交"""
        return 0 < self.filled_quantity < self.quantity

    @property
    def is_pending(self) -> bool:
        """是否待成交"""
        return self.status == "PENDING" and self.filled_quantity == 0


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str                              # 股票代码
    quantity: int                            # 数量
    avg_cost: float                          # 平均成本
    current_price: float = 0.0               # 当前价格
    unrealized_pnl: float = 0.0              # 未实现盈亏
    realized_pnl: float = 0.0                # 已实现盈亏

    # 时间
    opened_at: Optional[datetime] = None     # 开仓时间
    last_update: Optional[datetime] = None   # 最后更新

    @property
    def market_value(self) -> float:
        """市值"""
        return abs(self.quantity) * self.current_price

    @property
    def total_cost(self) -> float:
        """总成本"""
        return abs(self.quantity) * self.avg_cost

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.avg_cost > 0:
            return ((self.current_price / self.avg_cost) - 1) * 100
        return 0.0

    @property
    def is_long(self) -> bool:
        """是否多头"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """是否空头"""
        return self.quantity < 0


# 导出所有类型
__all__ = [
    'Quote',
    'Signal',
    'Metrics',
    'OrderRequest',
    'OrderStatus',
    'PositionInfo',
]
