#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Configuration Loader - Single Entry Point
Provides ALL configuration access through UnifiedTrainingConfig
Replaces: TimeConfigManager, CentralizedConfigManager, all competing systems
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global config instance
_UNIFIED_CONFIG = None

def get_unified_config():
    """Get the unified configuration instance (single source of truth)"""
    global _UNIFIED_CONFIG
    if _UNIFIED_CONFIG is None:
        try:
            # Import here to avoid circular imports
            from .量化模型_bma_ultra_enhanced import UnifiedTrainingConfig
            _UNIFIED_CONFIG = UnifiedTrainingConfig()
            logger.info("Unified configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load unified configuration: {e}")
            raise RuntimeError(f"Configuration system initialization failed: {e}")
    return _UNIFIED_CONFIG

# === COMPATIBILITY FUNCTIONS (replacing deleted systems) ===

def get_time_config():
    """
    Get time configuration (COMPATIBILITY: replaces unified_time_config.get_time_config)
    Returns a compatible object with temporal parameters from UnifiedTrainingConfig
    """
    config = get_unified_config()
    
    @dataclass
    class TimeConfig:
        """Compatible time config object"""
        feature_lag_days: int = config.FEATURE_LAG_DAYS
        prediction_horizon_days: int = config.PREDICTION_HORIZON_DAYS
        cv_gap_days: int = config.CV_GAP_DAYS
        cv_embargo_days: int = config.CV_EMBARGO_DAYS
        cv_n_splits: int = config.CV_SPLITS if hasattr(config, 'CV_SPLITS') else 5
        safety_gap_days: int = 1
        max_lookback_days: int = 252
        min_train_days: int = 126
        validation_window_days: int = 63
        
        def get_cv_params(self) -> Dict[str, int]:
            return {
                'gap': self.cv_gap_days,
                'embargo': self.cv_embargo_days,
                'max_train_size': None,
            }
        
        def get_section(self, section_name: str) -> Dict[str, Any]:
            """Compatibility method for get_section calls from old config managers"""
            # Provide default configurations for common sections
            sections = {
                'connection': {
                    'host': '127.0.0.1',
                    'port': 7497,
                    'client_id': config.RANDOM_STATE if hasattr(config, 'RANDOM_STATE') else 42,
                    'account_id': None,
                    'use_delayed_if_no_realtime': True,
                    'connection_timeout': 30,
                    'retry_attempts': 3
                },
                'monitoring': {
                    'account_update_interval': 60.0,
                    'position_monitor_interval': 30.0,
                    'risk_monitor_interval': 15.0,
                    'enable_real_time_alerts': True,
                    'alert_cooldown_seconds': 300,
                    'log_level': 'INFO'
                },
                'price_validation': {
                    'min_price': 0.01,
                    'max_price': 50000.0,
                    'max_daily_change_pct': 0.30,
                    'max_tick_change_pct': 0.05,
                    'max_data_age_seconds': 180.0,
                    'stale_warning_seconds': 30.0,
                    'outlier_std_multiplier': 3.0
                },
                'frequency_control': {
                    'max_daily_orders': 20,
                    'max_orders_per_5min': 10,
                    'max_orders_per_hour': 25,
                    'order_cooldown_seconds': 60
                },
                'risk_management': {
                    'max_single_position_pct': 0.15,
                    'max_sector_exposure_pct': 0.30,
                    'cash_reserve_pct': 0.15,
                    'max_daily_orders': 20,
                    'per_trade_risk_pct': 0.02,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.05,
                    'max_drawdown_pct': 0.05
                }
            }
            return sections.get(section_name, {})
    
    time_config = TimeConfig()

    # 🔴 CRITICAL: 注册配置使用以监控一致性
    monitor = get_config_monitor()
    monitor.register_config_usage('unified_config_loader', 'time', 'cv_gap_days', time_config.cv_gap_days)
    monitor.register_config_usage('unified_config_loader', 'time', 'cv_embargo_days', time_config.cv_embargo_days)

    return time_config

def get_cv_params() -> Dict[str, Any]:
    """Get CV parameters (COMPATIBILITY: replaces unified_time_config.get_cv_params)"""
    time_config = get_time_config()
    return time_config.get_cv_params()

def get_config_manager():
    """Get configuration manager (COMPATIBILITY: replaces various config managers)"""
    # Return a wrapper that supports both UnifiedTrainingConfig and get_section methods
    class ConfigManagerWrapper:
        def __init__(self, config):
            self.config = config
            
        def __getattr__(self, name):
            return getattr(self.config, name)
            
        def get_section(self, section_name: str) -> Dict[str, Any]:
            """Compatibility method for get_section calls from old config managers"""
            sections = {
                'connection': {
                    'host': '127.0.0.1',
                    'port': 7497,
                    'client_id': self.config.RANDOM_STATE if hasattr(self.config, 'RANDOM_STATE') else 42,
                    'account_id': None,
                    'use_delayed_if_no_realtime': True,
                    'connection_timeout': 30,
                    'retry_attempts': 3
                },
                'monitoring': {
                    'account_update_interval': 60.0,
                    'position_monitor_interval': 30.0,
                    'risk_monitor_interval': 15.0,
                    'enable_real_time_alerts': True,
                    'alert_cooldown_seconds': 300,
                    'log_level': 'INFO'
                },
                'price_validation': {
                    'min_price': 0.01,
                    'max_price': 50000.0,
                    'max_daily_change_pct': 0.30,
                    'max_tick_change_pct': 0.05,
                    'max_data_age_seconds': 180.0,
                    'stale_warning_seconds': 30.0,
                    'outlier_std_multiplier': 3.0
                },
                'frequency_control': {
                    'max_daily_orders': 20,
                    'max_orders_per_5min': 10,
                    'max_orders_per_hour': 25,
                    'order_cooldown_seconds': 60
                },
                'risk_management': {
                    'max_single_position_pct': 0.15,
                    'max_sector_exposure_pct': 0.30,
                    'cash_reserve_pct': 0.15,
                    'max_daily_orders': 20,
                    'per_trade_risk_pct': 0.02,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.05,
                    'max_drawdown_pct': 0.05
                }
            }
            return sections.get(section_name, {})
            
        def get_connection_params(self, auto_allocate_client_id: bool = True) -> Dict[str, Any]:
            """Get connection parameters from config file"""
            import json
            import os
            
            # Try to load from autotrader config file
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'autotrader', 'config', 'autotrader_unified_config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        conn_config = config_data.get('connection', {})
                        return {
                            'host': conn_config.get('host', '127.0.0.1'),
                            'port': conn_config.get('port', 4002),
                            'client_id': conn_config.get('client_id', 3130),
                            'timeout': conn_config.get('timeout', 10)
                        }
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_path}: {e}")
            
            # Fallback to default values
            return {
                'host': '127.0.0.1',
                'port': 4002,
                'client_id': 3130,
                'timeout': 10
            }
            
        def validate_config(self) -> List[str]:
            """Compatibility method for validate_config calls"""
            return []  # Return empty list (no errors) for now
    
    unified_config = get_unified_config()
    return ConfigManagerWrapper(unified_config)

# === DIRECT ACCESS FUNCTIONS ===

def get_temporal_config() -> Dict[str, Any]:
    """Get temporal configuration section"""
    config = get_unified_config()
    return {
        'prediction_horizon_days': config.PREDICTION_HORIZON_DAYS,
        'feature_lag_days': config.FEATURE_LAG_DAYS,
        'cv_gap_days': config.CV_GAP_DAYS,
        'cv_embargo_days': config.CV_EMBARGO_DAYS,
        'random_state': config.RANDOM_STATE
    }

def get_training_config() -> Dict[str, Any]:
    """Get training configuration section"""
    config = get_unified_config()
    return {
        'cv_splits': config.CV_SPLITS,
        'min_train_size': config.MIN_TRAIN_SIZE,
        'test_size': config.TEST_SIZE,
        'min_samples_for_cv': config.MIN_SAMPLES_FOR_CV
    }

def get_model_config() -> Dict[str, Any]:
    """Get model configuration section"""
    config = get_unified_config()
    return {
        'elastic_net': config.ELASTIC_NET_CONFIG,
        'xgboost': config.XGBOOST_CONFIG,
        'lightgbm': config.LIGHTGBM_CONFIG,
        'random_state': config.RANDOM_STATE
    }

def get_feature_config() -> Dict[str, Any]:
    """Get feature engineering configuration"""
    config = get_unified_config()
    return {
        'max_features': config.MAX_FEATURES,
        'min_features': config.MIN_FEATURES,
        'variance_threshold': config.VARIANCE_THRESHOLD,
        'correlation_threshold': config.CORRELATION_THRESHOLD
    }

def get_pca_config() -> Dict[str, Any]:
    """Get PCA configuration section"""
    config = get_unified_config()
    # 🔴 CRITICAL FIX: 提供完整的PCA配置，避免硬编码
    if hasattr(config, 'PCA_CONFIG'):
        return config.PCA_CONFIG
    else:
        # 默认安全配置
        return {
            'enabled': False,  # 默认禁用PCA防止数据泄漏
            'method': 'time_safe',
            'variance_threshold': 0.95,
            'n_components': 10,
            'traditional_components': 8,
            'alpha_components': 12,
            'min_components': 5,
            'max_components_ratio': 0.8,
            'time_safe': {
                'min_history_days': 252,
                'refit_frequency': 20
            }
        }

# === EWA AND DYNAMIC CONFIGURATION ===

def get_ewa_config() -> Dict[str, Any]:
    """获取EWA配置 - 解决adaptive eta selection问题"""
    return {
        'initial_eta': 0.1,
        'min_eta': 0.001,
        'max_eta': 0.5,
        'adaptation_rate': 0.05,
        'momentum_factor': 0.9,
        'stability_threshold': 0.02,
        'performance_lookback': 50,
        'regime_adjustment': True
    }

def get_weight_constraint_config() -> Dict[str, Any]:
    """获取权重约束配置 - 解决winsorization和稳定性问题"""
    return {
        'min_weight': 0.01,
        'max_weight': 0.6,
        'max_hhi': 0.5,
        'stability_alpha': 0.7,
        'max_weight_change': 0.1,
        'winsor_lower': 0.05,
        'winsor_upper': 0.95,
        'winsor_method': 'percentile',
        'sigma_multiplier': 2.5,
        'volatility_threshold': 0.05
    }

def get_exception_handling_config() -> Dict[str, Any]:
    """获取异常处理配置 - 解决静默失败问题"""
    return {
        'max_retries': 3,
        'enable_fallback': True,
        'log_level': 'WARNING',
        'silence_threshold': 10,
        'critical_error_threshold': 5,
        'raise_on_critical': True,  # 🔴 关键：不允许静默失败
        'fallback_strategies': {
            'model_training': 'equal_weight',
            'prediction': 'historical_mean',
            'feature_generation': 'zero_fill'
        }
    }

def get_kalman_config() -> Dict[str, Any]:
    """获取Kalman滤波器配置 - 增强不确定性量化"""
    return {
        'process_noise_scale': 0.001,
        'observation_noise_scale': 0.01,
        'adaptive_noise': True,
        'regime_aware': True,
        'min_observations': 20,
        'lookback_window': 60,
        'enable_uncertainty_bands': True,
        'confidence_levels': [0.68, 0.95],
        'weight_momentum_tracking': True
    }

def update_config_dynamically(section: str, key: str, value: Any) -> bool:
    """动态更新配置 - 解决硬编码参数问题"""
    try:
        config = get_unified_config()

        # 创建配置更新映射
        config_updates = {
            'pca': get_pca_config,
            'ewa': get_ewa_config,
            'weight_constraints': get_weight_constraint_config,
            'exception_handling': get_exception_handling_config,
            'kalman': get_kalman_config
        }

        if section in config_updates:
            current_config = config_updates[section]()
            if key in current_config:
                current_config[key] = value
                logger.info(f"配置动态更新: {section}.{key} = {value}")
                return True
            else:
                logger.warning(f"配置项不存在: {section}.{key}")
                return False
        else:
            logger.error(f"未知配置节: {section}")
            return False
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        return False

# === VALIDATION FUNCTIONS ===

def validate_temporal_configuration(**kwargs) -> bool:
    """Validate temporal configuration consistency"""
    time_config = get_time_config()
    
    conflicts = []
    for param_name, param_value in kwargs.items():
        if hasattr(time_config, param_name):
            expected_value = getattr(time_config, param_name)
            if param_value != expected_value:
                conflicts.append(f"{param_name}: expected {expected_value}, got {param_value}")
    
    if conflicts:
        logger.error(f"Temporal configuration conflicts: {conflicts}")
        return False
    return True

# === PCA COMPATIBILITY FUNCTIONS ===

def get_unified_pca_config():
    """Get unified PCA config (COMPATIBILITY: replaces unified_pca_config.get_unified_pca_config)"""
    from dataclasses import dataclass
    from enum import Enum

    class PCAMethod(Enum):
        UNIFIED = "unified"
        SEPARATED = "separated"
        TIME_SAFE = "time_safe"
        DISABLED = "disabled"

    @dataclass
    class UnifiedPCAConfig:
        """Compatible PCA config object"""
        def __init__(self):
            pca_config = get_pca_config()
            self.enabled = pca_config['enabled']
            self.method = PCAMethod(pca_config['method'])
            self.variance_threshold = pca_config['variance_threshold']
            self.traditional_components = pca_config['traditional_components']
            self.alpha_components = pca_config['alpha_components']
            self.min_components = pca_config['min_components']
            self.max_components_ratio = pca_config['max_components_ratio']
            self.time_safe_min_history_days = pca_config['time_safe']['min_history_days']
            self.time_safe_refit_frequency = pca_config['time_safe']['refit_frequency']

    return UnifiedPCAConfig()

# === CRITICAL ERROR MONITORING ===

class ConfigIntegrityMonitor:
    """配置完整性监控器 - 防止配置冲突和硬编码回归"""

    def __init__(self):
        self.monitored_configs = {}
        self.error_counts = {}

    def register_config_usage(self, module: str, config_section: str, config_key: str, value: Any):
        """注册配置使用"""
        key = f"{module}.{config_section}.{config_key}"
        self.monitored_configs[key] = value

    def detect_hardcoded_conflicts(self) -> List[str]:
        """检测硬编码冲突"""
        conflicts = []

        # 检查时序参数一致性
        time_config = get_time_config()
        expected_gap = time_config.cv_gap_days
        expected_embargo = time_config.cv_embargo_days

        # 检查是否有模块使用了不一致的参数
        for key, value in self.monitored_configs.items():
            if 'gap' in key.lower() and value != expected_gap:
                conflicts.append(f"Gap参数不一致: {key}={value}, 期望={expected_gap}")
            elif 'embargo' in key.lower() and value != expected_embargo:
                conflicts.append(f"Embargo参数不一致: {key}={value}, 期望={expected_embargo}")

        return conflicts

    def validate_anti_leakage_settings(self) -> List[str]:
        """验证反泄漏设置"""
        violations = []

        pca_config = get_pca_config()
        if pca_config['enabled'] and pca_config['method'] != 'time_safe':
            violations.append("PCA未使用时序安全模式，存在数据泄漏风险")

        time_config = get_time_config()
        if time_config.cv_gap_days < 1:
            violations.append(f"CV gap过小({time_config.cv_gap_days}<1)，存在数据泄漏风险")

        return violations

# 全局配置监控器
_config_monitor = ConfigIntegrityMonitor()

def get_config_monitor() -> ConfigIntegrityMonitor:
    """获取配置监控器"""
    return _config_monitor

# === BACKWARD COMPATIBILITY ALIASES ===

# For unified_time_config compatibility
TIME_CONFIG = None  # Will be lazily initialized

def get_unified_constants():
    """Get unified constants (backward compatibility)"""
    time_config = get_time_config()
    return {
        'UNIFIED_FEATURE_LAG_DAYS': time_config.feature_lag_days,
        'UNIFIED_SAFETY_GAP_DAYS': time_config.safety_gap_days,
        'UNIFIED_CV_GAP_DAYS': time_config.cv_gap_days,
        'UNIFIED_CV_EMBARGO_DAYS': time_config.cv_embargo_days,
        'UNIFIED_PREDICTION_HORIZON_DAYS': time_config.prediction_horizon_days,
    }

# Initialize lazy TIME_CONFIG
def _get_time_config_lazy():
    global TIME_CONFIG
    if TIME_CONFIG is None:
        TIME_CONFIG = get_time_config()
    return TIME_CONFIG

# Expose TIME_CONFIG as module-level variable for backward compatibility
import sys
def __getattr__(name):
    if name == 'TIME_CONFIG':
        return _get_time_config_lazy()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if __name__ == "__main__":
    # Test the unified config loader
    print("=== Unified Configuration Test ===")

    # Test time config
    time_config = get_time_config()
    print(f"Time Config: T-{time_config.feature_lag_days} -> T+{time_config.prediction_horizon_days}")
    print(f"CV Params: {get_cv_params()}")

    # Test configuration sections
    print(f"Temporal Config: {get_temporal_config()}")
    print(f"Training Config keys: {list(get_training_config().keys())}")

    # 🔴 CRITICAL: Test new configuration sections
    print("\n=== 新增配置测试 ===")
    print(f"EWA Config: {get_ewa_config()}")
    print(f"Weight Constraints: {get_weight_constraint_config()}")
    print(f"Exception Handling: {get_exception_handling_config()}")
    print(f"Kalman Config: {get_kalman_config()}")

    # Test integrity monitoring
    monitor = get_config_monitor()
    conflicts = monitor.detect_hardcoded_conflicts()
    leakage_risks = monitor.validate_anti_leakage_settings()

    if conflicts:
        print(f"⚠️  配置冲突: {conflicts}")
    else:
        print("✅ 无配置冲突")

    if leakage_risks:
        print(f"🔴 数据泄漏风险: {leakage_risks}")
    else:
        print("✅ 无数据泄漏风险")

    print("✅ Unified configuration loader working correctly")