#!/usr/bin/env python3
"""
🔧 统一错误处理策略
===================

实现标准化的错误处理机制，确保系统错误的一致性处理和恰当降级
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Union, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
import json
import os
from pathlib import Path


class ErrorSeverity(Enum):
    """错误严重级别"""
    LOW = "low"           # 可忽略的警告
    MEDIUM = "medium"     # 需要关注但不阻止运行
    HIGH = "high"         # 严重错误，可能影响功能
    CRITICAL = "critical" # 关键错误，影响核心功能


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"                 # 系统级错误
    DATA = "data"                     # 数据相关错误
    NETWORK = "network"               # 网络连接错误
    TRADING = "trading"               # 交易逻辑错误
    VALIDATION = "validation"         # 验证错误
    CONFIGURATION = "configuration"   # 配置错误
    EXTERNAL_API = "external_api"     # 外部API错误
    PERFORMANCE = "performance"       # 性能问题


class ErrorAction(Enum):
    """错误处理动作"""
    LOG_ONLY = "log_only"           # 仅记录日志
    RETRY = "retry"                 # 重试操作
    FALLBACK = "fallback"           # 降级处理
    ALERT = "alert"                 # 发送告警
    STOP = "stop"                   # 停止操作
    RESTART = "restart"             # 重启组件


@dataclass
class ErrorContext:
    """错误上下文信息"""
    operation: str
    component: str
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: ErrorContext
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    handled: bool = False
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorHandlingRule:
    """错误处理规则"""
    category: ErrorCategory
    severity: ErrorSeverity
    actions: List[ErrorAction]
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_function: Optional[Callable] = None
    custom_handler: Optional[Callable] = None


class UnifiedErrorHandler:
    """统一错误处理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("UnifiedErrorHandler")
        self._lock = threading.RLock()

        # 错误记录存储
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 1000

        # 错误统计
        self.error_counts: Dict[str, int] = {}
        self.retry_counts: Dict[str, int] = {}

        # 处理规则映射
        self.handling_rules: Dict[tuple, ErrorHandlingRule] = {}

        # 初始化默认规则
        self._initialize_default_rules()

        # 加载自定义配置
        if config_path:
            self._load_config(config_path)

    def _initialize_default_rules(self):
        """初始化默认错误处理规则"""
        default_rules = [
            # 系统错误规则
            ErrorHandlingRule(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.ALERT],
                max_retries=0
            ),
            ErrorHandlingRule(
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.HIGH,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.RETRY],
                max_retries=2,
                retry_delay=2.0
            ),

            # 数据错误规则
            ErrorHandlingRule(
                category=ErrorCategory.DATA,
                severity=ErrorSeverity.HIGH,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.FALLBACK],
                max_retries=1,
                retry_delay=0.5
            ),
            ErrorHandlingRule(
                category=ErrorCategory.DATA,
                severity=ErrorSeverity.MEDIUM,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.FALLBACK],
                max_retries=3,
                retry_delay=1.0
            ),

            # 网络错误规则
            ErrorHandlingRule(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.RETRY],
                max_retries=5,
                retry_delay=2.0
            ),

            # 交易错误规则
            ErrorHandlingRule(
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.CRITICAL,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.ALERT, ErrorAction.STOP],
                max_retries=0
            ),
            ErrorHandlingRule(
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.HIGH,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.RETRY],
                max_retries=2,
                retry_delay=1.0
            ),

            # 外部API错误规则
            ErrorHandlingRule(
                category=ErrorCategory.EXTERNAL_API,
                severity=ErrorSeverity.HIGH,
                actions=[ErrorAction.LOG_ONLY, ErrorAction.RETRY, ErrorAction.FALLBACK],
                max_retries=3,
                retry_delay=3.0
            )
        ]

        for rule in default_rules:
            key = (rule.category, rule.severity)
            self.handling_rules[key] = rule

    def _load_config(self, config_path: str):
        """加载错误处理配置"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # 解析自定义规则
                custom_rules = config.get('error_handling_rules', [])
                for rule_config in custom_rules:
                    rule = self._parse_rule_config(rule_config)
                    if rule:
                        key = (rule.category, rule.severity)
                        self.handling_rules[key] = rule

                self.logger.info(f"加载了{len(custom_rules)}条自定义错误处理规则")

        except Exception as e:
            self.logger.warning(f"加载错误处理配置失败: {e}")

    def _parse_rule_config(self, rule_config: Dict[str, Any]) -> Optional[ErrorHandlingRule]:
        """解析规则配置"""
        try:
            category = ErrorCategory(rule_config['category'])
            severity = ErrorSeverity(rule_config['severity'])
            actions = [ErrorAction(action) for action in rule_config['actions']]

            return ErrorHandlingRule(
                category=category,
                severity=severity,
                actions=actions,
                max_retries=rule_config.get('max_retries', 3),
                retry_delay=rule_config.get('retry_delay', 1.0)
            )
        except Exception as e:
            self.logger.warning(f"解析规则配置失败: {e}")
            return None

    def handle_error(self,
                    error: Union[Exception, str],
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    custom_metadata: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """统一错误处理入口"""

        with self._lock:
            # 创建错误记录
            error_id = self._generate_error_id(context, category, severity)

            error_record = ErrorRecord(
                error_id=error_id,
                severity=severity,
                category=category,
                message=str(error),
                context=context,
                exception=error if isinstance(error, Exception) else None,
                stack_trace=traceback.format_exc() if isinstance(error, Exception) else None,
                metadata=custom_metadata or {}
            )

            # 添加到历史记录
            self.error_history.append(error_record)
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]

            # 更新统计
            self._update_error_stats(category, severity)

            # 获取处理规则
            rule = self._get_handling_rule(category, severity)

            # 执行处理动作
            self._execute_handling_actions(error_record, rule)

            return error_record

    def _generate_error_id(self, context: ErrorContext, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """生成错误ID"""
        timestamp = int(time.time() * 1000)
        return f"{category.value}_{severity.value}_{context.component}_{timestamp}"

    def _update_error_stats(self, category: ErrorCategory, severity: ErrorSeverity):
        """更新错误统计"""
        key = f"{category.value}_{severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    def _get_handling_rule(self, category: ErrorCategory, severity: ErrorSeverity) -> Optional[ErrorHandlingRule]:
        """获取处理规则"""
        # 精确匹配
        key = (category, severity)
        if key in self.handling_rules:
            return self.handling_rules[key]

        # 降级匹配（同类别，低严重级别）
        for sev in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]:
            if sev.value != severity.value:
                key = (category, sev)
                if key in self.handling_rules:
                    return self.handling_rules[key]

        # 默认规则
        return ErrorHandlingRule(
            category=category,
            severity=severity,
            actions=[ErrorAction.LOG_ONLY],
            max_retries=1
        )

    def _execute_handling_actions(self, error_record: ErrorRecord, rule: Optional[ErrorHandlingRule]):
        """执行错误处理动作"""
        if not rule:
            return

        for action in rule.actions:
            try:
                if action == ErrorAction.LOG_ONLY:
                    self._log_error(error_record)

                elif action == ErrorAction.ALERT:
                    self._send_alert(error_record)

                elif action == ErrorAction.RETRY:
                    self._schedule_retry(error_record, rule)

                elif action == ErrorAction.FALLBACK:
                    self._execute_fallback(error_record, rule)

                elif action == ErrorAction.STOP:
                    self._execute_stop(error_record)

                elif action == ErrorAction.RESTART:
                    self._execute_restart(error_record)

            except Exception as e:
                self.logger.error(f"执行错误处理动作失败 {action}: {e}")

    def _log_error(self, error_record: ErrorRecord):
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_record.severity, logging.ERROR)

        log_msg = f"[{error_record.category.value.upper()}] {error_record.message}"
        if error_record.context.symbol:
            log_msg += f" (Symbol: {error_record.context.symbol})"
        if error_record.context.order_id:
            log_msg += f" (Order: {error_record.context.order_id})"

        self.logger.log(log_level, log_msg)

        if error_record.stack_trace and error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.log(log_level, f"Stack trace:\n{error_record.stack_trace}")

    def _send_alert(self, error_record: ErrorRecord):
        """发送告警（占位实现）"""
        alert_msg = f"CRITICAL ERROR: {error_record.message} in {error_record.context.component}"
        self.logger.critical(f"ALERT: {alert_msg}")
        # 这里可以集成实际的告警系统

    def _schedule_retry(self, error_record: ErrorRecord, rule: ErrorHandlingRule):
        """调度重试"""
        retry_key = f"{error_record.context.operation}_{error_record.context.component}"
        current_retries = self.retry_counts.get(retry_key, 0)

        if current_retries < rule.max_retries:
            self.retry_counts[retry_key] = current_retries + 1
            self.logger.info(f"调度重试 {retry_key} (尝试 {current_retries + 1}/{rule.max_retries})")
            # 实际重试逻辑需要在调用方实现
        else:
            self.logger.warning(f"重试次数已达上限 {retry_key}")

    def _execute_fallback(self, error_record: ErrorRecord, rule: ErrorHandlingRule):
        """执行降级处理"""
        if rule.fallback_function:
            try:
                result = rule.fallback_function(error_record)
                self.logger.info(f"降级处理完成: {error_record.context.operation}")
                return result
            except Exception as e:
                self.logger.error(f"降级处理失败: {e}")
        else:
            self.logger.warning(f"没有配置降级处理函数: {error_record.context.operation}")

    def _execute_stop(self, error_record: ErrorRecord):
        """执行停止操作"""
        self.logger.critical(f"执行停止操作: {error_record.context.operation}")
        # 实际停止逻辑需要在调用方实现

    def _execute_restart(self, error_record: ErrorRecord):
        """执行重启操作"""
        self.logger.critical(f"执行重启操作: {error_record.context.component}")
        # 实际重启逻辑需要在调用方实现

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        with self._lock:
            return {
                'error_counts': self.error_counts.copy(),
                'retry_counts': self.retry_counts.copy(),
                'total_errors': len(self.error_history),
                'recent_errors': len([e for e in self.error_history if time.time() - e.timestamp < 3600]),  # 近1小时
                'critical_errors': len([e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL])
            }

    def clear_retry_count(self, operation_key: str):
        """清除重试计数"""
        with self._lock:
            if operation_key in self.retry_counts:
                del self.retry_counts[operation_key]


# 全局错误处理器实例
_global_error_handler: Optional[UnifiedErrorHandler] = None


def get_unified_error_handler(config_path: Optional[str] = None) -> UnifiedErrorHandler:
    """获取全局统一错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = UnifiedErrorHandler(config_path)
    return _global_error_handler


def with_error_handling(category: ErrorCategory = ErrorCategory.SYSTEM,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       component: str = "unknown",
                       fallback_value: Any = None):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=func.__name__,
                    component=component
                )

                error_handler = get_unified_error_handler()
                error_handler.handle_error(e, context, severity, category)

                return fallback_value
        return wrapper
    return decorator


def error_handling_context(operation: str,
                          component: str,
                          category: ErrorCategory = ErrorCategory.SYSTEM,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """错误处理上下文管理器"""
    class ErrorHandlingContext:
        def __init__(self):
            self.error_handler = get_unified_error_handler()
            self.context = ErrorContext(operation=operation, component=component)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.error_handler.handle_error(exc_val, self.context, severity, category)
                return False  # 不压制异常

    return ErrorHandlingContext()


if __name__ == "__main__":
    # 测试统一错误处理
    logging.basicConfig(level=logging.INFO)

    handler = UnifiedErrorHandler()

    # 测试不同类型的错误
    context1 = ErrorContext(operation="test_operation", component="test_component")
    handler.handle_error("测试系统错误", context1, ErrorSeverity.HIGH, ErrorCategory.SYSTEM)

    context2 = ErrorContext(operation="data_processing", component="data_loader", symbol="AAPL")
    handler.handle_error("数据加载失败", context2, ErrorSeverity.MEDIUM, ErrorCategory.DATA)

    # 获取统计信息
    stats = handler.get_error_statistics()
    print(f"错误统计: {json.dumps(stats, indent=2)}")