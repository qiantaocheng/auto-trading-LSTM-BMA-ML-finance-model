"""
IBKR 自动交易系统
专业级量化交易平台，支持策略信号生成、风险管理、订单执行
"""

__version__ = "1.0.0"
__author__ = "Trading System Team"

# 仅暴露轻量入口，避免在包导入时触发重模块加载，降低冷启动时延与循环导入风险。
from .unified_config import UnifiedConfigManager  # 轻量，安全导出

__all__ = [
    "UnifiedConfigManager",
]