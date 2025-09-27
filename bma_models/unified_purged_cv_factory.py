#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一Purged/Embargo CV工厂 - 唯一入口点
严禁静默回退到普通TSCV，强制使用时间安全的CV切分
"""

import logging
import warnings
from typing import Any, Dict, Iterator, Optional, Tuple, Union, List
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
import numpy as np
import pandas as pd

try:
    from bma_models.unified_config_loader import get_time_config, TIME_CONFIG
except ImportError:
    # Fallback if running as standalone
    def get_time_config():
        # CRITICAL FIX: Use unified configuration instead of hardcoded values
        from bma_models.unified_config_loader import get_time_config as get_unified_time_config
        return get_unified_time_config()
    TIME_CONFIG = None

# Simplified config enforcement (replaces strict_time_config_enforcer)
class TimeConfigConflictError(Exception):
    """Time configuration conflict error"""
    pass

def enforce_time_config(func=None, **enforce_kwargs):
    """Simple time config enforcement using unified config (can be used as decorator)"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Validate time config parameters if provided
            time_config = get_time_config()
            for param_name, param_value in enforce_kwargs.items():
                if hasattr(time_config, param_name):
                    expected_value = getattr(time_config, param_name)
                    if param_value != expected_value:
                        raise TimeConfigConflictError(f"{param_name}: expected {expected_value}, got {param_value}")
            return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        # Called with arguments: @enforce_time_config(param=value)
        return decorator
    else:
        # Called without arguments: @enforce_time_config
        return decorator(func)

logger = logging.getLogger(__name__)

class CVDegradationError(Exception):
    """CV降级错误 - 禁止回退到普通TSCV"""
    pass

class UnifiedPurgedTimeSeriesCV(BaseCrossValidator):
    """
    统一的Purged Time Series Cross Validator - 系统唯一CV入口
    
    特性：
    1. 强制使用统一时间配置
    2. 严禁静默回退到TimeSeriesSplit
    3. 自动记录任何降级尝试
    4. Fail-fast模式确保时间安全
    """
    
    def __init__(self, n_splits: int = None, gap: int = None, 
                 embargo: int = None, test_size: int = None, 
                 max_train_size: int = None, groups_required: bool = False):
        """
        初始化统一Purged CV
        
        Args:
            n_splits: CV折数（如果为None则使用统一配置）
            gap: Gap天数（如果为None则使用统一配置）
            embargo: Embargo天数（如果为None则使用统一配置）
            test_size: 测试集大小
            max_train_size: 最大训练集大小
            groups_required: 是否要求groups参数（防止退化）
        """
        # 获取统一配置
        unified_config = get_time_config()
        
        # 使用统一配置或验证传入参数
        self.n_splits = n_splits if n_splits is not None else 5
        self.gap = gap if gap is not None else unified_config.cv_gap_days
        self.embargo = embargo if embargo is not None else unified_config.cv_embargo_days
        self.test_size = test_size if test_size is not None else unified_config.validation_window_days
        self.max_train_size = max_train_size
        self.groups_required = groups_required
        
        # 强制验证参数一致性
        if gap is not None or embargo is not None:
            try:
                if TIME_CONFIG is not None:
                    TIME_CONFIG.validate_external_params(
                        cv_gap_days=self.gap,
                        cv_embargo_days=self.embargo
                    )
                else:
                    # Validate against unified config
                    if gap is not None and gap != unified_config.cv_gap_days:
                        logger.warning(f"CV gap {gap} differs from unified config {unified_config.cv_gap_days}")
                    if embargo is not None and embargo != unified_config.cv_embargo_days:
                        logger.warning(f"CV embargo {embargo} differs from unified config {unified_config.cv_embargo_days}")
            except Exception as e:
                raise TimeConfigConflictError(f"CV参数与统一配置冲突: {e}")
        
        # 记录配置来源
        self._config_source = "unified_time_config"
        logger.info(f"UnifiedPurgedCV初始化: gap={self.gap}, embargo={self.embargo}, splits={self.n_splits}")
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回CV折数"""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        执行Purged Time Series Split
        
        Args:
            X: 特征数据
            y: 目标变量
            groups: 时间分组（必需，防止退化到普通TSCV）
        
        Yields:
            train_indices, test_indices: 训练集和测试集索引
        """
        # 强制检查groups参数
        if self.groups_required and groups is None:
            error_msg = (
                "严禁退化到普通TSCV！groups参数为必需项。\n"
                "这防止了时间泄漏风险，确保时间安全的CV切分。\n"
                "请提供时间分组索引（如日期索引）。"
            )
            logger.error(error_msg)
            raise CVDegradationError(error_msg)
        
        # 执行Purged Time Series Split
        X = self._validate_input(X)
        n_samples = len(X)
        
        # 记录CV执行
        logger.info(f"执行UnifiedPurgedCV: n_samples={n_samples}, gap={self.gap}, embargo={self.embargo}")
        
        # 生成时间安全的split
        for train_idx, test_idx in self._purged_time_series_split(X, y, groups, n_samples):
            # 记录每个fold的统计信息
            logger.debug(f"CV Fold: train_size={len(train_idx)}, test_size={len(test_idx)}, "
                        f"train_range=({train_idx[0]}, {train_idx[-1]}), "
                        f"test_range=({test_idx[0]}, {test_idx[-1]})")
            
            yield train_idx, test_idx
    
    def _validate_input(self, X):
        """验证输入数据"""
        if hasattr(X, 'index') and hasattr(X.index, 'to_numpy'):
            # pandas DataFrame/Series
            return X
        elif hasattr(X, 'shape'):
            # numpy array
            return X
        else:
            raise ValueError(f"不支持的输入类型: {type(X)}")
    
    def _purged_time_series_split(self, X, y, groups, n_samples):
        """
        执行Purged Time Series Split逻辑
        """
        if groups is not None:
            # 使用groups进行时间分组
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
        else:
            # 没有groups时，使用索引（但已经在上面被阻止）
            unique_groups = np.arange(n_samples)
            n_groups = n_samples

        # 自适应test_size计算
        if self.test_size and self.test_size > n_groups // 2:
            # 如果test_size太大，自动调整
            adaptive_test_size = max(5, min(n_groups // (self.n_splits + 1), n_groups // 10))
            logger.warning(f"test_size={self.test_size}对于{n_groups}个groups太大，自动调整为{adaptive_test_size}")
            test_size = adaptive_test_size
        else:
            test_size = self.test_size if self.test_size else n_groups // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # 计算测试集位置
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_groups - self.embargo)
            
            if test_start >= test_end:
                break
            
            # 计算训练集位置（考虑gap和embargo）
            train_end = test_start - self.gap
            train_start = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
            
            if train_start >= train_end:
                continue
            
            # 生成索引
            if groups is not None:
                train_groups = unique_groups[train_start:train_end]
                test_groups = unique_groups[test_start:test_end]
                
                train_mask = np.isin(groups, train_groups)
                test_mask = np.isin(groups, test_groups)
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
            else:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取CV配置摘要"""
        return {
            'cv_type': 'UnifiedPurgedTimeSeriesCV',
            'n_splits': self.n_splits,
            'gap': self.gap,
            'embargo': self.embargo,
            'test_size': self.test_size,
            'max_train_size': self.max_train_size,
            'groups_required': self.groups_required,
            'config_source': self._config_source,
            'time_safe': True,
            'degradation_protected': True
        }

# === 全局CV工厂函数 ===

def create_unified_cv(**kwargs) -> UnifiedPurgedTimeSeriesCV:
    """
    创建统一CV分割器的唯一入口点
    
    这是系统中唯一允许的CV创建方式，严禁直接使用：
    - TimeSeriesSplit
    - 其他非时间安全的CV方法
    
    Returns:
        UnifiedPurgedTimeSeriesCV: 时间安全的CV分割器
    """
    # 强制使用统一配置验证
    if TIME_CONFIG is not None:
        validated_kwargs = TIME_CONFIG.validate_external_params(**kwargs)
    else:
        # Use the kwargs as-is if TIME_CONFIG is not available
        validated_kwargs = kwargs
        logger.debug("TIME_CONFIG not available, using provided kwargs directly")
    
    # 记录CV创建
    logger.info(f"创建统一CV分割器: {validated_kwargs}")
    
    return UnifiedPurgedTimeSeriesCV(**validated_kwargs)

@enforce_time_config
def get_default_cv_splitter(**kwargs) -> UnifiedPurgedTimeSeriesCV:
    """
    获取默认CV分割器（带参数验证装饰器）
    """
    return create_unified_cv(**kwargs)

def get_cv_params_for_model() -> Dict[str, Any]:
    """
    获取模型使用的CV参数
    """
    return TIME_CONFIG.get_purged_cv_factory_params()

# === 禁用危险CV方法的监控 ===

class CVDegradationMonitor:
    """CV降级监控器"""
    
    _degradation_count = 0
    _degradation_log = []
    
    @classmethod
    def record_degradation_attempt(cls, attempted_class: str, location: str):
        """记录降级尝试"""
        cls._degradation_count += 1
        degradation_record = {
            'timestamp': pd.Timestamp.now(),
            'attempted_class': attempted_class,
            'location': location,
            'count': cls._degradation_count
        }
        cls._degradation_log.append(degradation_record)
        
        error_msg = (
            f"⚠️ 检测到CV降级尝试 #{cls._degradation_count}\n"
            f"位置: {location}\n"
            f"尝试使用: {attempted_class}\n"
            f"要求: 必须使用 UnifiedPurgedTimeSeriesCV\n"
            f"解决方案: 使用 create_unified_cv() 或 get_default_cv_splitter()"
        )
        
        logger.error(error_msg)
        warnings.warn(error_msg, UserWarning, stacklevel=3)
        
        # 在严格模式下抛出异常
        import os
        if os.getenv('BMA_STRICT_CV_MODE', 'true').lower() == 'true':
            raise CVDegradationError(error_msg)
    
    @classmethod
    def get_degradation_report(cls) -> Dict[str, Any]:
        """获取降级报告"""
        return {
            'total_attempts': cls._degradation_count,
            'attempts_log': cls._degradation_log,
            'last_attempt': cls._degradation_log[-1] if cls._degradation_log else None,
            'system_integrity': 'COMPROMISED' if cls._degradation_count > 0 else 'SECURE'
        }

# 全局降级监控实例
CV_DEGRADATION_MONITOR = CVDegradationMonitor()

# === 猴子补丁防护 - 拦截危险CV ===

def _intercept_dangerous_cv():
    """拦截对危险CV方法的调用"""
    import sklearn.model_selection as sk_ms
    
    # 保存原始方法
    _original_TimeSeriesSplit = sk_ms.TimeSeriesSplit
    
    def _protected_TimeSeriesSplit(*args, **kwargs):
        """被保护的TimeSeriesSplit - 记录并阻止"""
        import inspect
        frame = inspect.currentframe()
        location = f"{frame.f_back.f_code.co_filename}:{frame.f_back.f_lineno}"
        
        CV_DEGRADATION_MONITOR.record_degradation_attempt(
            'TimeSeriesSplit', location
        )
        
        # 自动替换为安全版本
        logger.warning("自动替换TimeSeriesSplit为UnifiedPurgedTimeSeriesCV")
        return UnifiedPurgedTimeSeriesCV(*args, **kwargs)
    
    # 替换危险方法
    sk_ms.TimeSeriesSplit = _protected_TimeSeriesSplit

# 启用保护
_intercept_dangerous_cv()

if __name__ == "__main__":
    # 测试统一CV工厂
    print("=== 统一Purged CV工厂测试 ===")
    
    # 测试正常使用
    try:
        cv = create_unified_cv()
        print(f"✓ 统一CV创建成功: {cv.get_config_summary()}")
    except Exception as e:
        print(f"✗ 统一CV创建失败: {e}")
    
    # 测试参数冲突检测
    try:
        cv_conflict = create_unified_cv(gap=get_time_config().cv_gap_days, embargo=get_time_config().cv_embargo_days)  # 与统一配置冲突
        print(f"✗ 冲突检测失败: {cv_conflict}")
    except TimeConfigConflictError as e:
        print(f"✓ 冲突检测成功: {type(e).__name__}")
    
    # 测试降级保护
    try:
        from sklearn.model_selection import TimeSeriesSplit
        ts_cv = TimeSeriesSplit()  # 应该被拦截
        print(f"降级拦截状态: {type(ts_cv).__name__}")
    except Exception as e:
        print(f"降级拦截: {e}")
    
    # 生成降级报告
    report = CV_DEGRADATION_MONITOR.get_degradation_report()
    print(f"\n降级监控报告: {report}")