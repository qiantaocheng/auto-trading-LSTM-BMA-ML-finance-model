#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强异常处理模块

提供结构化的异常处理、详细的错误日志、以及错误恢复机制
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Type, Callable
from functools import wraps
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重性级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    CONFIG_ERROR = "config_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

class EnhancedExceptionHandler:
    """增强的异常处理器"""

    def __init__(self):
        self.error_log = []
        self.error_counts = {}
        self.recovery_strategies = {}
        self._setup_recovery_strategies()

    def _setup_recovery_strategies(self):
        """设置错误恢复策略"""
        self.recovery_strategies = {
            ValueError: self._handle_value_error,
            KeyError: self._handle_key_error,
            TypeError: self._handle_type_error,
            MemoryError: self._handle_memory_error,
            ImportError: self._handle_import_error,
            Exception: self._handle_generic_error
        }

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """分类错误类型"""
        error_type = type(error)
        error_msg = str(error).lower()

        if error_type in [ValueError, TypeError]:
            if 'data' in error_msg or 'index' in error_msg or 'shape' in error_msg:
                return ErrorCategory.DATA_ERROR
            elif 'model' in error_msg or 'predict' in error_msg:
                return ErrorCategory.MODEL_ERROR
            else:
                return ErrorCategory.COMPUTATION_ERROR

        elif error_type == KeyError:
            return ErrorCategory.DATA_ERROR

        elif error_type == MemoryError:
            return ErrorCategory.MEMORY_ERROR

        elif error_type == ImportError:
            return ErrorCategory.CONFIG_ERROR

        elif 'timeout' in error_msg:
            return ErrorCategory.TIMEOUT_ERROR

        else:
            return ErrorCategory.UNKNOWN_ERROR

    def assess_severity(self, error: Exception, context: str = "") -> ErrorSeverity:
        """评估错误严重性"""
        error_type = type(error)
        error_msg = str(error).lower()

        # 关键错误
        if error_type in [MemoryError, SystemError]:
            return ErrorSeverity.CRITICAL

        # 高级错误
        if any(keyword in error_msg for keyword in ['data leakage', 'temporal', 'cv', 'validation']):
            return ErrorSeverity.HIGH

        if any(keyword in context.lower() for keyword in ['training', 'model', 'prediction']):
            if error_type in [ValueError, TypeError]:
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.MEDIUM

        # 中级错误
        if error_type in [KeyError, AttributeError]:
            return ErrorSeverity.MEDIUM

        # 低级错误
        return ErrorSeverity.LOW

    def handle_exception(
        self,
        error: Exception,
        context: str = "",
        operation: str = "",
        data_info: Optional[Dict] = None,
        raise_on_critical: bool = True
    ) -> Dict[str, Any]:
        """处理异常"""
        error_type = type(error)
        category = self.categorize_error(error)
        severity = self.assess_severity(error, context)

        # 记录错误统计
        error_key = f"{error_type.__name__}:{category.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # 构建错误记录
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type.__name__,
            'error_message': str(error),
            'category': category.value,
            'severity': severity.value,
            'context': context,
            'operation': operation,
            'data_info': data_info or {},
            'traceback': traceback.format_exc(),
            'count': self.error_counts[error_key]
        }

        self.error_log.append(error_record)

        # 选择日志级别
        log_message = f"[{severity.value.upper()}] {category.value}: {str(error)}"
        if context:
            log_message += f" (Context: {context})"

        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # 记录详细堆栈（仅对高级和关键错误）
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # 尝试恢复
        recovery_result = self._attempt_recovery(error, context, data_info)

        # 决定是否重新抛出异常
        if severity == ErrorSeverity.CRITICAL and raise_on_critical:
            logger.critical("Critical error detected - raising exception")
            raise error
        elif severity == ErrorSeverity.HIGH and not recovery_result.get('recovered', False):
            logger.error("High severity error without recovery - raising exception")
            raise error

        return {
            'error_record': error_record,
            'recovery_result': recovery_result,
            'should_continue': recovery_result.get('recovered', False) or severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        }

    def _attempt_recovery(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """尝试错误恢复"""
        error_type = type(error)

        # 查找恢复策略
        recovery_func = None
        for exc_type, func in self.recovery_strategies.items():
            if issubclass(error_type, exc_type):
                recovery_func = func
                break

        if recovery_func:
            try:
                return recovery_func(error, context, data_info)
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy failed: {recovery_error}")
                return {'recovered': False, 'recovery_error': str(recovery_error)}
        else:
            return {'recovered': False, 'reason': 'No recovery strategy available'}

    def _handle_value_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理值错误"""
        error_msg = str(error).lower()

        if 'empty' in error_msg:
            return {
                'recovered': False,
                'strategy': 'empty_data',
                'suggestion': 'Check data source and filtering logic'
            }
        elif 'shape' in error_msg or 'dimension' in error_msg:
            return {
                'recovered': False,
                'strategy': 'shape_mismatch',
                'suggestion': 'Verify data alignment and preprocessing steps'
            }
        else:
            return {
                'recovered': False,
                'strategy': 'generic_value_error',
                'suggestion': 'Review input validation and data quality'
            }

    def _handle_key_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理键错误"""
        missing_key = str(error).strip("'\"")

        return {
            'recovered': False,
            'strategy': 'missing_key',
            'missing_key': missing_key,
            'suggestion': f'Ensure key "{missing_key}" exists in data structure'
        }

    def _handle_type_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理类型错误"""
        return {
            'recovered': False,
            'strategy': 'type_mismatch',
            'suggestion': 'Check data types and conversions'
        }

    def _handle_memory_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理内存错误"""
        return {
            'recovered': False,
            'strategy': 'memory_exhaustion',
            'suggestion': 'Reduce data size, optimize processing, or increase system memory'
        }

    def _handle_import_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理导入错误"""
        missing_module = str(error).replace('No module named ', '').strip("'\"")

        return {
            'recovered': True,  # 导入错误通常可以通过降级功能来恢复
            'strategy': 'fallback_implementation',
            'missing_module': missing_module,
            'suggestion': f'Install missing module: pip install {missing_module}'
        }

    def _handle_generic_error(self, error: Exception, context: str, data_info: Optional[Dict]) -> Dict[str, Any]:
        """处理通用错误"""
        return {
            'recovered': False,
            'strategy': 'generic_handling',
            'suggestion': 'Review operation logic and error conditions'
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_log:
            return {'total_errors': 0, 'summary': 'No errors recorded'}

        # 统计各类错误
        category_counts = {}
        severity_counts = {}

        for record in self.error_log:
            category = record['category']
            severity = record['severity']

            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # 最近的错误
        recent_errors = self.error_log[-5:] if len(self.error_log) > 5 else self.error_log

        return {
            'total_errors': len(self.error_log),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recent_errors': [
                {
                    'timestamp': err['timestamp'],
                    'type': err['error_type'],
                    'message': err['error_message'][:100] + '...' if len(err['error_message']) > 100 else err['error_message'],
                    'severity': err['severity']
                }
                for err in recent_errors
            ],
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 分析错误模式
        if self.error_counts:
            most_common = max(self.error_counts.items(), key=lambda x: x[1])
            error_type, count = most_common

            if count > 3:
                recommendations.append(f"频繁错误 '{error_type}' 出现了 {count} 次，建议重点排查")

        # 检查严重错误
        critical_errors = [err for err in self.error_log if err['severity'] == 'critical']
        if critical_errors:
            recommendations.append(f"发现 {len(critical_errors)} 个关键错误，需要立即处理")

        # 检查数据错误
        data_errors = [err for err in self.error_log if err['category'] == 'data_error']
        if len(data_errors) > 2:
            recommendations.append("数据错误较多，建议检查数据质量和预处理流程")

        return recommendations

    def save_error_log(self, filepath: str):
        """保存错误日志到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': self.get_error_summary(),
                    'detailed_log': self.error_log
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Error log saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")


def enhanced_exception_decorator(
    context: str = "",
    raise_on_critical: bool = True,
    log_data_info: bool = False
):
    """增强异常处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = EnhancedExceptionHandler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 收集数据信息
                data_info = {}
                if log_data_info:
                    for arg in args:
                        if hasattr(arg, 'shape'):
                            data_info[f'arg_shape'] = getattr(arg, 'shape')
                        elif hasattr(arg, '__len__'):
                            data_info[f'arg_length'] = len(arg)

                # 处理异常
                result = handler.handle_exception(
                    error=e,
                    context=context or func.__name__,
                    operation=func.__name__,
                    data_info=data_info,
                    raise_on_critical=raise_on_critical
                )

                if not result['should_continue']:
                    raise e

                # 返回错误信息而不是崩溃
                return {
                    'success': False,
                    'error': str(e),
                    'error_details': result['error_record']
                }

        return wrapper
    return decorator


# 全局异常处理器实例
global_exception_handler = EnhancedExceptionHandler()


def apply_enhanced_exception_handling(bma_model):
    """将增强异常处理应用到BMA模型"""

    # 添加异常处理器
    bma_model._exception_handler = global_exception_handler

    # 包装关键方法
    original_prepare = bma_model._prepare_standard_data_format
    original_clean = bma_model._clean_training_data
    original_train = bma_model._unified_model_training
    original_stacking = bma_model._train_stacking_models_modular

    @enhanced_exception_decorator(
        context="data_preparation",
        raise_on_critical=True,
        log_data_info=True
    )
    def wrapped_prepare(feature_data):
        return original_prepare(feature_data)

    @enhanced_exception_decorator(
        context="data_cleaning",
        raise_on_critical=True,
        log_data_info=True
    )
    def wrapped_clean(X, y, dates, tickers):
        return original_clean(X, y, dates, tickers)

    @enhanced_exception_decorator(
        context="model_training",
        raise_on_critical=False,  # 允许部分模型失败
        log_data_info=True
    )
    def wrapped_train(X, y, dates, tickers):
        return original_train(X, y, dates, tickers)

    @enhanced_exception_decorator(
        context="stacking",
        raise_on_critical=False,  # 堆叠失败时使用单模型
        log_data_info=True
    )
    def wrapped_stacking(training_results, X, y, dates, tickers):
        return original_stacking(training_results, X, y, dates, tickers)

    # 替换方法
    bma_model._prepare_standard_data_format = wrapped_prepare
    bma_model._clean_training_data = wrapped_clean
    bma_model._unified_model_training = wrapped_train
    bma_model._train_stacking_models_modular = wrapped_stacking

    # 添加错误报告方法
    def get_error_report():
        return bma_model._exception_handler.get_error_summary()

    def save_error_log(filepath: str = None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"D:/trade/logs/bma_errors_{timestamp}.json"
        bma_model._exception_handler.save_error_log(filepath)
        return filepath

    bma_model.get_error_report = get_error_report
    bma_model.save_error_log = save_error_log

    logger.info("✅ 增强异常处理已应用到BMA模型")

    return bma_model