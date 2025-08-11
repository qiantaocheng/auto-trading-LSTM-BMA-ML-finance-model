"""
IBKR 自动交易系统
专业级量化交易平台，支持策略信号生成、风险管理、订单执行
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"

# 导入核心模块
# 已改用统一配置管理器
# from .config import HotConfig
from .unified_config import UnifiedConfigManager
from .engine import Engine, RiskEngine, SignalHub, OrderRouter
from .ibkr_auto_trader import IbkrAutoTrader

__all__ = [
    "UnifiedConfigManager", 
    "Engine", 
    "RiskEngine", 
    "SignalHub", 
    "OrderRouter",
    "IbkrAutoTrader"
]