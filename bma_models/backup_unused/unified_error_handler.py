#!/usr/bin/env python3
"""
统一异常处理机制
标准化错误处理、日志记录和恢复策略
"""

import logging
import traceback
import functools
import warnings
from typing import Optional, Callable, Any, Dict, Type, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

# 配置统一的日志格式
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorContext:
    """错误上下文信息"""
    function_name: str
    module_name: str
    error_type: str
    error_message: str
    timestamp: datetime
    stack_trace: str
    recovery_attempted: bool
    recovery_successful: bool
    additional_info: Dict[str, Any]

class UnifiedErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, 
                 logger_name: str = "BMA_System",
                 default_log_level: LogLevel = LogLevel.INFO,
                 enable_recovery: bool = True):
        
        self.logger = self._setup_logger(logger_name, default_log_level)
        self.enable_recovery = enable_recovery
        self.error_history = []
        self.recovery_strategies = {}
        
    def _setup_logger(self, name: str, level: LogLevel) -> logging.Logger:
        """设置统一的日志记录器"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:  # 避免重复添加处理器
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 设置日志级别
            logger.setLevel(getattr(logging, level.value))
        
        return logger
    
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: Callable) -> None:
        """注册错误恢复策略"""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, 
                    error: Exception,
                    context: Dict[str, Any] = None,
                    severity: LogLevel = LogLevel.ERROR,
                    should_reraise: bool = False,
                    recovery_data: Any = None) -> Optional[Any]:
        """
        统一错误处理
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            severity: 错误严重程度
            should_reraise: 是否重新抛出异常
            recovery_data: 恢复策略所需的数据
            
        Returns:
            恢复策略的结果(如果适用)
        """
        context = context or {}
        
        # 创建错误上下文
        error_context = ErrorContext(
            function_name=context.get('function', 'unknown'),
            module_name=context.get('module', 'unknown'),
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            recovery_attempted=False,
            recovery_successful=False,
            additional_info=context
        )
        
        # 记录错误
        self._log_error(error_context, severity)
        
        # 尝试恢复
        recovery_result = None
        if self.enable_recovery:
            recovery_result = self._attempt_recovery(error, error_context, recovery_data)
        
        # 保存错误历史
        self.error_history.append(error_context)
        
        # 重新抛出异常(如果需要)
        if should_reraise and not error_context.recovery_successful:
            raise error
        
        return recovery_result
    
    def _log_error(self, error_context: ErrorContext, severity: LogLevel) -> None:
        """记录错误信息"""
        log_message = (
            f"🚨 {error_context.function_name}() 发生 {error_context.error_type}: "
            f"{error_context.error_message}"
        )
        
        # 根据严重程度选择日志方法
        if severity == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif severity == LogLevel.INFO:
            self.logger.info(log_message)
        elif severity == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif severity == LogLevel.ERROR:
            self.logger.error(log_message)
            self.logger.error(f"堆栈跟踪:\n{error_context.stack_trace}")
        elif severity == LogLevel.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"堆栈跟踪:\n{error_context.stack_trace}")
    
    def _attempt_recovery(self, error: Exception, 
                         error_context: ErrorContext,
                         recovery_data: Any = None) -> Optional[Any]:
        """尝试错误恢复"""
        error_context.recovery_attempted = True
        
        error_type = type(error)
        
        # 查找恢复策略
        recovery_strategy = self.recovery_strategies.get(error_type)
        if not recovery_strategy:
            # 查找父类的恢复策略
            for registered_type, strategy in self.recovery_strategies.items():
                if isinstance(error, registered_type):
                    recovery_strategy = strategy
                    break
        
        if recovery_strategy:
            try:
                self.logger.info(f"尝试恢复策略: {recovery_strategy.__name__}")
                result = recovery_strategy(error, error_context, recovery_data)
                error_context.recovery_successful = True
                self.logger.info("✅ 错误恢复成功")
                return result
            except Exception as recovery_error:
                self.logger.error(f"恢复策略失败: {recovery_error}")
                error_context.recovery_successful = False
        
        return None
    
    def suppress_warnings(self, category: Type[Warning] = None) -> None:
        """抑制特定类型的警告"""
        if category:
            warnings.filterwarnings("ignore", category=category)
        else:
            warnings.filterwarnings("ignore")
        
        self.logger.debug(f"已抑制警告: {category.__name__ if category else 'All'}")

# 全局错误处理器实例
_global_error_handler = UnifiedErrorHandler()

def get_error_handler() -> UnifiedErrorHandler:
    """获取全局错误处理器"""
    return _global_error_handler

def safe_execute(fallback_value: Any = None,
                recovery_data: Any = None,
                log_level: LogLevel = LogLevel.ERROR,
                suppress_reraise: bool = True):
    """
    安全执行装饰器
    
    Args:
        fallback_value: 异常时的返回值
        recovery_data: 传递给恢复策略的数据
        log_level: 日志级别
        suppress_reraise: 是否抑制重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                result = _global_error_handler.handle_error(
                    e,
                    context=context,
                    severity=log_level,
                    should_reraise=not suppress_reraise,
                    recovery_data=recovery_data
                )
                
                return result if result is not None else fallback_value
        
        return wrapper
    return decorator

def critical_section(operation_name: str = "操作"):
    """
    关键区段装饰器 - 严格错误处理
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                _global_error_handler.logger.info(f"开始执行关键操作: {operation_name}")
                result = func(*args, **kwargs)
                _global_error_handler.logger.info(f"✅ 关键操作完成: {operation_name}")
                return result
            except Exception as e:
                _global_error_handler.logger.critical(
                    f"❌ 关键操作失败: {operation_name} - {e}"
                )
                raise  # 关键区段始终重新抛出异常
        
        return wrapper
    return decorator

# 标准恢复策略
def default_data_recovery_strategy(error: Exception, 
                                 error_context: ErrorContext,
                                 recovery_data: Any) -> Any:
    """数据相关错误的默认恢复策略"""
    if "数据为空" in error_context.error_message or "empty" in error_context.error_message.lower():
        return recovery_data if recovery_data is not None else []
    
    if "索引" in error_context.error_message or "index" in error_context.error_message.lower():
        return recovery_data if recovery_data is not None else None
    
    return None

def memory_error_recovery_strategy(error: Exception,
                                 error_context: ErrorContext,
                                 recovery_data: Any) -> Any:
    """内存错误恢复策略"""
    import gc
    gc.collect()  # 强制垃圾回收
    _global_error_handler.logger.info("执行内存清理恢复策略")
    return None

# 注册默认恢复策略
_global_error_handler.register_recovery_strategy(ValueError, default_data_recovery_strategy)
_global_error_handler.register_recovery_strategy(IndexError, default_data_recovery_strategy)
_global_error_handler.register_recovery_strategy(KeyError, default_data_recovery_strategy)
_global_error_handler.register_recovery_strategy(MemoryError, memory_error_recovery_strategy)

# 抑制常见的无害警告
_global_error_handler.suppress_warnings(UserWarning)
_global_error_handler.suppress_warnings(FutureWarning)