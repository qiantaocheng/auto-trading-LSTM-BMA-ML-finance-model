#!/usr/bin/env python3
"""
交易系统抽象接口
Trading System Abstract Interfaces

使用Protocol定义抽象接口，打破engine和broker之间的循环依赖
"""

from typing import Protocol, Optional, Dict, List, Any
from .trading_types import Quote, Signal, OrderRequest, OrderStatus, PositionInfo


class IBrokerInterface(Protocol):
    """
    Broker接口定义

    Engine通过这个接口与Broker交互，不直接依赖具体的IbkrAutoTrader实现
    """

    # 连接管理
    async def connect(self) -> bool:
        """连接到broker"""
        ...

    async def disconnect(self) -> None:
        """断开连接"""
        ...

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        ...

    # 账户信息
    @property
    def net_liq(self) -> float:
        """账户净值"""
        ...

    @property
    def cash_balance(self) -> float:
        """现金余额"""
        ...

    @property
    def buying_power(self) -> float:
        """购买力"""
        ...

    # 行情数据
    async def subscribe(self, symbol: str) -> None:
        """订阅实时行情"""
        ...

    def unsubscribe(self, symbol: str) -> None:
        """取消订阅"""
        ...

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        ...

    # 订单管理
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "LMT",
        limit_price: Optional[float] = None,
        **kwargs
    ) -> int:
        """下单"""
        ...

    async def cancel_order(self, order_id: int) -> bool:
        """撤单"""
        ...

    def get_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """获取订单状态"""
        ...

    @property
    def open_orders(self) -> Dict[int, Any]:
        """未完成订单"""
        ...

    # 持仓管理
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """获取持仓"""
        ...

    @property
    def positions(self) -> Dict[str, Any]:
        """所有持仓"""
        ...

    # 数据访问（兼容性）
    @property
    def tickers(self) -> Dict[str, Any]:
        """Ticker数据（IBKR特有）"""
        ...


class ISignalProvider(Protocol):
    """
    信号提供者接口

    定义信号生成器的标准接口
    """

    def generate_signal(self, symbol: str, threshold: float = 0.3) -> Signal:
        """生成交易信号"""
        ...

    async def generate_signals_batch(self, symbols: List[str]) -> List[Signal]:
        """批量生成信号"""
        ...


class IRiskManager(Protocol):
    """
    风险管理器接口

    定义风险检查和管理的标准接口
    """

    async def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        account_value: float
    ) -> Any:  # 返回RiskValidationResult
        """验证订单风险"""
        ...

    def update_position(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        entry_price: Optional[float] = None
    ) -> None:
        """更新持仓风险指标"""
        ...

    def check_emergency_stop_conditions(self) -> tuple:
        """检查紧急停止条件"""
        ...


class IPositionManager(Protocol):
    """
    持仓管理器接口

    定义持仓管理的标准接口
    """

    def update_position(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        entry_price: Optional[float] = None
    ) -> None:
        """更新持仓"""
        ...

    def get_quantity(self, symbol: str) -> int:
        """获取持仓数量"""
        ...

    def get_all_positions(self) -> Dict[str, Any]:
        """获取所有持仓"""
        ...

    def close_position(self, symbol: str) -> None:
        """关闭持仓"""
        ...


class IOrderValidator(Protocol):
    """
    订单验证器接口

    定义订单验证的标准接口
    """

    async def validate_order_unified(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        account_value: float
    ) -> Any:  # 返回ValidationResult
        """统一订单验证"""
        ...


class IDataProvider(Protocol):
    """
    数据提供者接口

    定义历史数据和实时数据的标准接口
    """

    async def get_historical_bars(
        self,
        symbol: str,
        lookback_days: int = 60
    ) -> List[Any]:
        """获取历史K线"""
        ...

    async def get_market_data(
        self,
        symbol: str,
        days: int = 30
    ) -> Any:  # 返回DataFrame
        """获取市场数据"""
        ...

    def get_realtime_quote(self, symbol: str) -> Optional[Quote]:
        """获取实时报价"""
        ...


class ILogger(Protocol):
    """
    日志接口

    定义日志记录的标准接口
    """

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Debug日志"""
        ...

    def info(self, msg: str, *args, **kwargs) -> None:
        """Info日志"""
        ...

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Warning日志"""
        ...

    def error(self, msg: str, *args, **kwargs) -> None:
        """Error日志"""
        ...

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Critical日志"""
        ...


# 辅助函数：类型检查
def is_valid_broker(obj: Any) -> bool:
    """检查对象是否实现了IBrokerInterface"""
    required_attrs = [
        'connect', 'disconnect', 'is_connected',
        'net_liq', 'cash_balance', 'buying_power',
        'subscribe', 'unsubscribe', 'get_quote',
        'place_order', 'cancel_order', 'get_order_status',
        'get_position', 'positions', 'tickers'
    ]
    return all(hasattr(obj, attr) for attr in required_attrs)


def is_valid_signal_provider(obj: Any) -> bool:
    """检查对象是否实现了ISignalProvider"""
    required_attrs = ['generate_signal', 'generate_signals_batch']
    return all(hasattr(obj, attr) for attr in required_attrs)


# 导出所有接口
__all__ = [
    'IBrokerInterface',
    'ISignalProvider',
    'IRiskManager',
    'IPositionManager',
    'IOrderValidator',
    'IDataProvider',
    'ILogger',
    'is_valid_broker',
    'is_valid_signal_provider',
]
