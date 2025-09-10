#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一时间配置中心 - 唯一真源的时间安全参数管理
防止时间泄漏的配置冲突和参数不一致问题
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class UnifiedTimeConfig:
    """统一时间配置 - 不可变的时间安全参数"""
    
    # === 核心时间参数 ===
    feature_lag_days: int = 1          # T-1特征滞后
    prediction_horizon_days: int = 10   # T+10预测目标
    safety_gap_days: int = 1           # 基础安全间隔
    
    # === Purged CV 专用参数 ===
    cv_gap_days: int = 9               # CV gap = T+10-1 = 9天
    cv_embargo_days: int = 10          # CV embargo = T+10 = 10天
    
    # === 高级配置 ===
    max_lookback_days: int = 252       # 最大回溯窗口(1年交易日)
    min_train_days: int = 126          # 最小训练窗口(6个月)
    validation_window_days: int = 63   # 验证窗口(3个月)
    
    # === 特征工程时间配置 ===
    moving_average_windows: tuple = (5, 20, 60)
    volatility_windows: tuple = (20, 60)
    momentum_windows: tuple = (21, 126, 252)
    
    def __post_init__(self):
        """初始化后验证配置一致性"""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """验证时间配置的逻辑一致性"""
        errors = []
        
        # 基本范围检查
        if self.feature_lag_days < 1:
            errors.append("feature_lag_days必须>=1")
        
        if self.prediction_horizon_days < 1:
            errors.append("prediction_horizon_days必须>=1")
        
        # 🔧 CRITICAL FIX: 与IndexAligner统一策略保持一致
        # IndexAligner已改为不剪尾，因此CV参数要求可以放宽，但仍需保证时间安全
        if self.cv_gap_days < self.prediction_horizon_days - 1:
            errors.append(f"CV gap应该>=prediction_horizon-1: {self.prediction_horizon_days-1}, 当前: {self.cv_gap_days}")
        
        if self.cv_embargo_days < self.prediction_horizon_days:
            errors.append(f"CV embargo应该>=prediction_horizon: {self.prediction_horizon_days}, 当前: {self.cv_embargo_days}")
        
        if self.min_train_days < 30:
            errors.append("最小训练窗口不能小于30天")
        
        if self.validation_window_days > self.min_train_days:
            errors.append("验证窗口不能大于最小训练窗口")
        
        if errors:
            error_msg = "时间配置验证失败:\n" + "\n".join(f"- {err}" for err in errors)
            raise ValueError(error_msg)
        
        logger.info(f"时间配置验证通过: T-{self.feature_lag_days} -> T+{self.prediction_horizon_days}, Gap={self.cv_gap_days}, Embargo={self.cv_embargo_days}")
    
    def get_cv_params(self) -> Dict[str, int]:
        """获取标准化的CV参数"""
        return {
            'gap': self.cv_gap_days,
            'embargo': self.cv_embargo_days,
            'max_train_size': None,  # 使用全部可用数据
        }
    
    def get_temporal_validation_params(self) -> Dict[str, Any]:
        """获取时间验证参数"""
        return {
            'feature_lag': self.feature_lag_days,
            'prediction_horizon': self.prediction_horizon_days,
            'safety_gap': self.safety_gap_days,
            'strict_mode': True  # 强制严格模式
        }

class TimeConfigManager:
    """时间配置管理器 - 系统唯一实例"""
    
    _instance: Optional['TimeConfigManager'] = None
    _config: Optional[UnifiedTimeConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = UnifiedTimeConfig()
            logger.info("时间配置管理器初始化完成")
    
    @property
    def config(self) -> UnifiedTimeConfig:
        """获取统一时间配置（只读）"""
        if self._config is None:
            raise RuntimeError("时间配置未初始化")
        return self._config
    
    def override_config(self, **kwargs) -> None:
        """覆盖配置（仅用于测试，生产环境禁用）"""
        import os
        if os.getenv('BMA_ALLOW_CONFIG_OVERRIDE') != 'true':
            raise RuntimeError("生产环境不允许覆盖时间配置")
        
        logger.warning(f"覆盖时间配置: {kwargs}")
        current_dict = self._config.__dict__.copy()
        current_dict.update(kwargs)
        self._config = UnifiedTimeConfig(**current_dict)
    
    def validate_external_params(self, **params) -> bool:
        """验证外部传入的时间参数是否与统一配置一致"""
        config = self.config
        
        conflicts = []
        for param_name, param_value in params.items():
            if hasattr(config, param_name):
                expected_value = getattr(config, param_name)
                if param_value != expected_value:
                    conflicts.append(f"{param_name}: 期望{expected_value}, 实际{param_value}")
        
        if conflicts:
            error_msg = "时间参数冲突:\n" + "\n".join(f"- {conflict}" for conflict in conflicts)
            logger.error(error_msg)
            return False
        
        return True
    
    def get_purged_cv_factory_params(self) -> Dict[str, Any]:
        """获取PurgedTimeSeriesCV的标准参数"""
        config = self.config
        return {
            'n_splits': 5,
            'gap': config.cv_gap_days,
            'embargo': config.cv_embargo_days,
            'test_size': config.validation_window_days,
            'max_train_size': None
        }

# === 全局时间配置单例 ===
TIME_CONFIG = TimeConfigManager()

def get_time_config() -> UnifiedTimeConfig:
    """获取全局时间配置"""
    return TIME_CONFIG.config

def validate_temporal_configuration(**kwargs) -> bool:
    """验证时间配置一致性（替代原有的validate_temporal_configuration）"""
    return TIME_CONFIG.validate_external_params(**kwargs)

def get_cv_params() -> Dict[str, Any]:
    """获取标准化CV参数"""
    return TIME_CONFIG.get_purged_cv_factory_params()

# === 向后兼容别名 ===
def get_unified_constants():
    """向后兼容：获取统一常量"""
    config = get_time_config()
    return {
        'UNIFIED_FEATURE_LAG_DAYS': config.feature_lag_days,
        'UNIFIED_SAFETY_GAP_DAYS': config.safety_gap_days,
        'UNIFIED_CV_GAP_DAYS': config.cv_gap_days,
        'UNIFIED_CV_EMBARGO_DAYS': config.cv_embargo_days,
        'UNIFIED_PREDICTION_HORIZON_DAYS': config.prediction_horizon_days,
    }

if __name__ == "__main__":
    # 测试时间配置
    config = get_time_config()
    print("统一时间配置:")
    print(f"  特征滞后: T-{config.feature_lag_days}")
    print(f"  预测目标: T+{config.prediction_horizon_days}")
    print(f"  CV Gap: {config.cv_gap_days}天")
    print(f"  CV Embargo: {config.cv_embargo_days}天")
    
    print("\nCV参数:")
    print(get_cv_params())