
# 监控系统适配器
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class UnifiedMonitoringAdapter:
    """统一监控系统适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('trading_monitor')
        self.performance_metrics = {}
        self.error_counts = {}
        
    def log_trade(self, trade_info: Dict[str, Any]):
        """记录交易信息"""
        self.logger.info(f"交易执行: {trade_info}")
        
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        error_key = f"{type(error).__name__}_{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.logger.error(f"错误 [{context}]: {error}")
        
    def update_performance(self, metric: str, value: float):
        """更新性能指标"""
        self.performance_metrics[metric] = {
            'value': value,
            'timestamp': datetime.now()
        }
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'performance_metrics': self.performance_metrics,
            'error_counts': self.error_counts,
            'system_health': 'healthy' if len(self.error_counts) < 10 else 'degraded'
        }

# 全局监控实例
_monitoring_adapter = None

def get_monitoring_adapter():
    """获取监控适配器实例"""
    global _monitoring_adapter
    if _monitoring_adapter is None:
        _monitoring_adapter = UnifiedMonitoringAdapter()
    return _monitoring_adapter
