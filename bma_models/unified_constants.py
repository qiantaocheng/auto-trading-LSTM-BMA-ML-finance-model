#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一常量管理模块

集中管理所有硬编码的魔数和配置常量
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ValidationConstants:
    """验证相关常量"""
    # 样本数量阈值
    MIN_CORRELATION_SAMPLES: int = 30
    MIN_DAILY_SAMPLES: int = 5
    MIN_VALIDATION_SAMPLES: int = 30
    MIN_CV_FOLD_SIZE: int = 30

    # 数值阈值
    MIN_STD_DEVIATION: float = 1e-12
    MIN_VARIANCE_THRESHOLD: float = 1e-6
    MAX_CORRELATION_THRESHOLD: float = 0.95
    OUTLIER_PERCENTILE_LOW: float = 1.0
    OUTLIER_PERCENTILE_HIGH: float = 99.0

    # IC和性能阈值
    MIN_IC_THRESHOLD: float = 0.01
    MIN_R2_THRESHOLD: float = 0.0
    MAX_IC_FOR_WARNING: float = 0.02
    IC_STABILITY_THRESHOLD: float = 0.5


@dataclass
class MemoryConstants:
    """内存管理常量"""
    # 内存阈值 (MB)
    SMALL_DATAFRAME_MB: float = 10.0
    LARGE_DATAFRAME_MB: float = 100.0
    MEMORY_WARNING_MB: float = 100.0
    MEMORY_THRESHOLD_MB: float = 1000.0

    # 批处理大小
    DEFAULT_BATCH_SIZE: int = 10
    MAX_CONCAT_SIZE_MB: float = 500.0


@dataclass
class ModelConstants:
    """模型相关常量"""
    # 早停参数
    EARLY_STOPPING_ROUNDS: int = 100
    EARLY_STOPPING_PATIENCE: int = 100

    # 模型参数限制
    MAX_ITERATIONS: int = 2000
    MIN_ESTIMATORS: int = 50
    MAX_ESTIMATORS: int = 5000

    # 特征选择
    MAX_FEATURES_RATIO: float = 0.8
    MIN_FEATURE_IMPORTANCE: float = 1e-6

    # CV参数
    MIN_CV_SPLITS: int = 2
    MAX_CV_SPLITS: int = 10


@dataclass
class WeightConstants:
    """权重优化常量"""
    # 权重约束
    MIN_WEIGHT_FLOOR: float = 0.01
    MAX_WEIGHT_CAP: float = 0.8
    MAX_WEIGHT_CHANGE_PER_STEP: float = 0.1

    # EWA参数
    DEFAULT_ETA: float = 0.8
    CONSERVATIVE_ETA: float = 0.5
    ETA_GRID_SIZE: int = 21

    # 稳定性阈值
    HIGH_UNCERTAINTY_THRESHOLD: float = 0.5
    WEIGHT_VOLATILITY_THRESHOLD: float = 0.3


@dataclass
class TimeConstants:
    """时间相关常量"""
    # 默认时间参数（天）
    DEFAULT_PREDICTION_HORIZON: int = 10
    DEFAULT_FEATURE_LAG: int = 1
    DEFAULT_SAFETY_GAP: int = 1
    DEFAULT_CV_GAP: int = 5
    DEFAULT_CV_EMBARGO: int = 1

    # 最小时间要求
    MIN_TRAIN_SIZE_DAYS: int = 252
    MIN_TEST_SIZE_DAYS: int = 63
    MIN_TOTAL_DAYS: int = 400

    # IC计算参数
    IC_LOOKBACK_DAYS: int = 120
    IC_SMOOTHING_WINDOW: int = 21


@dataclass
class NumericalConstants:
    """数值计算常量"""
    # 数值稳定性
    EPSILON: float = 1e-12
    SMALL_NUMBER: float = 1e-8
    LARGE_NUMBER: float = 1e8

    # 截断限制
    CLIP_LOWER_BOUND: float = -10.0
    CLIP_UPPER_BOUND: float = 10.0
    FISHER_Z_CLIP: float = 0.999

    # 正则化参数
    RIDGE_LAMBDA_DEFAULT: float = 0.01
    L1_RATIO_DEFAULT: float = 0.5


@dataclass
class LoggingConstants:
    """日志相关常量"""
    # 日志级别映射
    LOG_LEVEL_MAPPING: Dict[str, str] = None

    # 默认日志配置
    DEFAULT_LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_LOG_LEVEL: str = 'INFO'

    # 性能日志阈值
    SLOW_OPERATION_THRESHOLD: float = 10.0  # 秒
    MEMORY_LOG_THRESHOLD: float = 50.0  # MB

    def __post_init__(self):
        if self.LOG_LEVEL_MAPPING is None:
            self.LOG_LEVEL_MAPPING = {
                'critical': 'CRITICAL',
                'high': 'ERROR',
                'medium': 'WARNING',
                'low': 'INFO'
            }


@dataclass
class PathConstants:
    """路径相关常量"""
    # 默认路径
    DEFAULT_LOG_DIR: str = "D:/trade/logs"
    DEFAULT_OUTPUT_DIR: str = "D:/trade/predictions"
    DEFAULT_CACHE_DIR: str = "D:/trade/cache"

    # 文件扩展名
    EXCEL_EXTENSION: str = ".xlsx"
    JSON_EXTENSION: str = ".json"
    CSV_EXTENSION: str = ".csv"
    LOG_EXTENSION: str = ".log"


class ModelNames:
    """模型名称常量"""
    # 标准模型名称
    ELASTIC_NET = "elastic_net"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"

    # 模型别名映射
    ALIASES = {
        'elastic': ELASTIC_NET,
        'elasticnet': ELASTIC_NET,
        'en': ELASTIC_NET,

        'xgb': XGBOOST,
        'xgb_regressor': XGBOOST,

        'cat': CATBOOST,
        'cb': CATBOOST
    }

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """规范化模型名称"""
        name_lower = name.lower()
        return cls.ALIASES.get(name_lower, name_lower)

    @classmethod
    def get_all_standard_names(cls) -> List[str]:
        """获取所有标准模型名称"""
        return [cls.ELASTIC_NET, cls.XGBOOST, cls.CATBOOST]


class ColumnNames:
    """列名常量"""
    # 标准列名
    DATE = "date"
    TICKER = "ticker"
    TARGET = "target"
    CLOSE = "Close"

    # 索引名称
    STANDARD_INDEX = [DATE, TICKER]

    # 排除的列名（不作为特征）
    EXCLUDED_COLUMNS = [DATE, TICKER, TARGET, CLOSE]


class ConfigurationKeys:
    """配置键名常量"""
    # YAML配置节
    TEMPORAL = "temporal"
    TRAINING = "training"
    DATA = "data"
    FEATURES = "features"
    PCA = "pca"
    RISK_MANAGEMENT = "risk_management"

    # 配置文件路径
    DEFAULT_CONFIG_PATH = "bma_models/unified_config.yaml"


# 创建全局常量实例
VALIDATION = ValidationConstants()
MEMORY = MemoryConstants()
MODEL = ModelConstants()
WEIGHT = WeightConstants()
TIME = TimeConstants()
NUMERICAL = NumericalConstants()
LOGGING = LoggingConstants()
PATH = PathConstants()

# 常量字典，便于动态访问
CONSTANTS_DICT = {
    'validation': VALIDATION,
    'memory': MEMORY,
    'model': MODEL,
    'weight': WEIGHT,
    'time': TIME,
    'numerical': NUMERICAL,
    'logging': LOGGING,
    'path': PATH
}


def get_constant(category: str, name: str, default=None):
    """
    获取常量值

    Args:
        category: 常量类别
        name: 常量名称
        default: 默认值

    Returns:
        常量值或默认值
    """
    if category in CONSTANTS_DICT:
        constant_obj = CONSTANTS_DICT[category]
        return getattr(constant_obj, name.upper(), default)
    return default


def update_constant(category: str, name: str, value):
    """
    更新常量值（用于配置覆盖）

    Args:
        category: 常量类别
        name: 常量名称
        value: 新值
    """
    if category in CONSTANTS_DICT:
        constant_obj = CONSTANTS_DICT[category]
        if hasattr(constant_obj, name.upper()):
            setattr(constant_obj, name.upper(), value)
            return True
    return False


def get_all_constants() -> Dict[str, Dict[str, Any]]:
    """获取所有常量的字典表示"""
    result = {}
    for category, constant_obj in CONSTANTS_DICT.items():
        result[category] = {}
        for attr_name in dir(constant_obj):
            if not attr_name.startswith('_') and not callable(getattr(constant_obj, attr_name)):
                result[category][attr_name.lower()] = getattr(constant_obj, attr_name)
    return result


# 向后兼容的别名
THRESHOLDS = VALIDATION  # 向后兼容
MEMORY_THRESHOLDS = MEMORY  # 向后兼容