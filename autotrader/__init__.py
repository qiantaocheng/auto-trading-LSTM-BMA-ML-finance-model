"""
IBKR 自动交易系统
专业级量化交易平台，支持策略信号生成、风险管理、订单执行
"""

__version__ = "2.0.0"
__author__ = "Trading System Team"

# 配置管理器导入
try:
    from bma_models.unified_config_loader import get_unified_config
except ImportError:
    # 配置加载失败时的回退
    def get_unified_config():
        return None

# 导入核心组件（懒加载以避免循环导入）
def get_autotrader_gui():
    """懒加载AutoTrader GUI"""
    from .app import AutoTraderGUI
    return AutoTraderGUI

def get_ibkr_trader():
    """懒加载IBKR交易器"""
    from .ibkr_auto_trader import IbkrAutoTrader
    return IbkrAutoTrader

def get_engine():
    """懒加载交易引擎"""
    from .engine import Engine
    return Engine

def get_database():
    """懒加载数据库"""
    from .database import StockDatabase
    return StockDatabase

# 向后兼容的风险平衡器
def get_risk_balancer_adapter(enable_balancer=False):
    """获取风险平衡器适配器"""
    try:
        from .real_risk_balancer import get_risk_balancer_adapter as get_real_balancer
        return get_real_balancer(enable_balancer)
    except ImportError:
        # Fallback到Mock实现
        class MockRiskBalancer:
            def __init__(self, enabled=False):
                self.enabled = enabled
            def balance_portfolio(self, positions):
                return positions
            def is_enabled(self):
                return self.enabled
        return MockRiskBalancer(enable_balancer)

# 向后兼容的监控接口
def get_monitoring_interface():
    """获取监控接口"""
    try:
        from .unified_monitoring_interface import get_monitoring_interface
        return get_monitoring_interface()
    except ImportError:
        try:
            from .unified_monitoring_system import UnifiedMonitoringSystem
            return UnifiedMonitoringSystem()
        except ImportError:
            return None

# 向后兼容的错误处理
def get_error_handler():
    """获取错误处理器"""
    try:
        from .error_handling_system import get_error_handler
        return get_error_handler()
    except ImportError:
        return None

__all__ = [
    "get_config_manager",
    "get_autotrader_gui", 
    "get_ibkr_trader",
    "get_engine",
    "get_database",
    "get_risk_balancer_adapter",
    "get_monitoring_interface",
    "get_error_handler"
]