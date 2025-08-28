
# 增强日志系统适配器
import logging
import os
from datetime import datetime
from typing import Any, Dict

class EnhancedTradingLogger:
    """增强交易日志器"""
    
    def __init__(self):
        self.setup_loggers()
        
    def setup_loggers(self):
        """设置日志器"""
        # 创建日志目录
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 交易日志
        self.trade_logger = logging.getLogger('trading')
        trade_handler = logging.FileHandler(
            f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
        
        # 错误日志
        self.error_logger = logging.getLogger('trading_errors')
        error_handler = logging.FileHandler(
            f"{log_dir}/errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)
        
    def log_order_submission(self, order_data: Dict[str, Any]):
        """记录订单提交"""
        self.trade_logger.info(f"订单提交: {order_data}")
        
    def log_order_fill(self, fill_data: Dict[str, Any]):
        """记录订单成交"""
        self.trade_logger.info(f"订单成交: {fill_data}")
        
    def log_position_update(self, position_data: Dict[str, Any]):
        """记录仓位更新"""
        self.trade_logger.info(f"仓位更新: {position_data}")
        
    def log_risk_alert(self, alert_message: str):
        """记录风险警报"""
        self.trade_logger.warning(f"风险警报: {alert_message}")
        
    def log_system_error(self, error: Exception, context: str = ""):
        """记录系统错误"""
        self.error_logger.error(f"系统错误 [{context}]: {error}")

# 全局日志实例
_enhanced_logger = None

def get_enhanced_logger():
    """获取增强日志器实例"""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedTradingLogger()
    return _enhanced_logger
