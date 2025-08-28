"""
IBKR 自动交易系统
专业级量化交易平台，支持策略信号生成、风险管理、订单执行
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"

# 仅暴露轻量入口，避免in包导入when触发重模块加载，降低冷startwhen延and循环导入风险。
from .config_manager import get_config_manager

__all__ = [
    "UnifiedConfigManager",
]