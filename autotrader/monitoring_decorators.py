#!/usr/bin/env python3
"""
监控装饰器 - 简化性能跟踪和错误监控
"""

import time
import functools
import logging
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

# 简化的监控实现，不依赖external模块
class AlertLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType:
    COUNTER = "counter"
    TIMER = "timer"
    GAUGE = "gauge"

class SimpleMonitor:
    def track_performance(self, name, duration, success):
        logger.info(f"Performance: {name} took {duration:.2f}s, success={success}")

    def record_metric(self, name, value, metric_type):
        logger.debug(f"Metric: {name}={value}")

    def emit_alert(self, title, message, level, category):
        logger.warning(f"Alert [{level}]: {title} - {message}")

def get_enhanced_monitor():
    return SimpleMonitor()

def monitor_performance(operation_name: Optional[str] = None, 
                       alert_on_error: bool = True,
                       alert_threshold_seconds: Optional[float] = None):
    """
    性能监控装饰器
    
    Args:
        operation_name: 操作名称，默认使用函数名
        alert_on_error: 是否在错误时发送告警
        alert_threshold_seconds: 响应时间告警阈值
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        monitor = get_enhanced_monitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            success = True
            result = None
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                
                # 记录性能指标
                monitor.track_performance(op_name, duration, success)
                
                # 检查响应时间告警
                if alert_threshold_seconds and duration > alert_threshold_seconds:
                    monitor.emit_alert(
                        f"{op_name}响应时间过长",
                        f"操作{op_name}耗时{duration:.2f}秒，超过阈值{alert_threshold_seconds:.2f}秒",
                        AlertLevel.WARNING,
                        "performance"
                    )
                
                # 错误告警
                if not success and alert_on_error:
                    monitor.emit_alert(
                        f"{op_name}执行失败",
                        f"操作{op_name}执行失败: {str(error)}",
                        AlertLevel.ERROR,
                        "operation"
                    )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            success = True
            result = None
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration = time.time() - start_time
                
                # 记录性能指标
                monitor.track_performance(op_name, duration, success)
                
                # 检查响应时间告警
                if alert_threshold_seconds and duration > alert_threshold_seconds:
                    monitor.emit_alert(
                        f"{op_name}响应时间过长",
                        f"操作{op_name}耗时{duration:.2f}秒，超过阈值{alert_threshold_seconds:.2f}秒",
                        AlertLevel.WARNING,
                        "performance"
                    )
                
                # 错误告警
                if not success and alert_on_error:
                    monitor.emit_alert(
                        f"{op_name}执行失败",
                        f"操作{op_name}执行失败: {str(error)}",
                        AlertLevel.ERROR,
                        "operation"
                    )
        
        # 根据函数类型返回合适的装饰器
        if hasattr(func, '__code__') and 'async' in str(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

def monitor_api_call(api_name: Optional[str] = None,
                    timeout_seconds: float = 30.0):
    """
    API调用监控装饰器
    
    Args:
        api_name: API名称
        timeout_seconds: 超时阈值
    """
    def decorator(func: Callable) -> Callable:
        name = api_name or f"api_{func.__name__}"
        monitor = get_enhanced_monitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录API调用指标
                monitor.record_metric(f"{name}_calls", 1, MetricType.COUNTER)
                monitor.record_metric(f"{name}_response_time", duration, MetricType.TIMER)
                
                # 超时检查
                if duration > timeout_seconds:
                    monitor.emit_alert(
                        f"API调用超时",
                        f"{name} API调用耗时{duration:.2f}秒，超过预期{timeout_seconds:.2f}秒",
                        AlertLevel.WARNING,
                        "api"
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_metric(f"{name}_errors", 1, MetricType.COUNTER)
                
                monitor.emit_alert(
                    f"API调用失败",
                    f"{name} API调用失败: {str(e)}",
                    AlertLevel.ERROR,
                    "api"
                )
                raise
        
        return wrapper
    return decorator

def monitor_trading_operation(operation_type: str = "trading"):
    """
    交易操作监控装饰器
    
    Args:
        operation_type: 操作类型
    """
    def decorator(func: Callable) -> Callable:
        monitor = get_enhanced_monitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # 记录操作开始
            monitor.record_metric(f"{operation_type}_started", 1, MetricType.COUNTER)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功指标
                monitor.record_metric(f"{operation_type}_success", 1, MetricType.COUNTER)
                monitor.record_metric(f"{operation_type}_duration", duration, MetricType.TIMER)
                
                # 关键操作成功告警
                if operation_type in ['order_placement', 'position_close']:
                    monitor.emit_alert(
                        f"{operation_type}操作成功",
                        f"{operation_type}操作在{duration:.2f}秒内成功完成",
                        AlertLevel.INFO,
                        "trading"
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败指标
                monitor.record_metric(f"{operation_type}_failed", 1, MetricType.COUNTER)
                
                # 交易操作失败是严重问题
                monitor.emit_alert(
                    f"{operation_type}操作失败",
                    f"{operation_type}操作失败: {str(e)}",
                    AlertLevel.CRITICAL,
                    "trading"
                )
                raise
        
        return wrapper
    return decorator

def monitor_connection(connection_name: str = "ibkr"):
    """
    连接监控装饰器
    
    Args:
        connection_name: 连接名称
    """
    def decorator(func: Callable) -> Callable:
        monitor = get_enhanced_monitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                
                # 连接成功
                monitor.record_metric(f"{connection_name}_connection_success", 1, MetricType.COUNTER)
                monitor.emit_alert(
                    f"{connection_name}连接成功",
                    f"{connection_name}连接已建立",
                    AlertLevel.INFO,
                    "connection"
                )
                
                return result
                
            except Exception as e:
                # 连接失败
                monitor.record_metric(f"{connection_name}_connection_failed", 1, MetricType.COUNTER)
                monitor.emit_alert(
                    f"{connection_name}连接失败",
                    f"{connection_name}连接失败: {str(e)}",
                    AlertLevel.CRITICAL,
                    "connection"
                )
                raise
        
        return wrapper
    return decorator

class MonitoringContext:
    """监控上下文管理器"""
    
    def __init__(self, operation_name: str, alert_on_error: bool = True):
        self.operation_name = operation_name
        self.alert_on_error = alert_on_error
        self.monitor = get_enhanced_monitor()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.record_metric(f"{self.operation_name}_started", 1, MetricType.COUNTER)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            self.monitor.track_performance(self.operation_name, duration, success)
            
            if not success and self.alert_on_error:
                self.monitor.emit_alert(
                    f"{self.operation_name}执行失败",
                    f"操作{self.operation_name}执行失败: {str(exc_val)}",
                    AlertLevel.ERROR,
                    "context"
                )
    
    def record_checkpoint(self, checkpoint_name: str, value: float = 1.0):
        """记录检查点"""
        self.monitor.record_metric(
            f"{self.operation_name}_{checkpoint_name}",
            value,
            MetricType.COUNTER
        )

# 便捷的监控上下文函数
def monitor_context(operation_name: str, alert_on_error: bool = True):
    """创建监控上下文"""
    return MonitoringContext(operation_name, alert_on_error)