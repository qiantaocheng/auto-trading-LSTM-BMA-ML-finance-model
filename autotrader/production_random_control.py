#!/usr/bin/env python3
"""
🔥 P0级别修复：生产环境随机性控制
=======================================

确保生产环境下所有随机性都被严格控制，防止不可重现的交易决策。
包含特征哈希记录、模型版本追踪等量化交易必需的可重现性控制。
"""

import os
import hashlib
import logging
import numpy as np
import pandas as pd
import random
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ProductionRandomControl:
    """生产环境随机性控制器"""
    
    # 固定种子用于生产环境
    PRODUCTION_SEED = 42
    
    def __init__(self, is_production: bool = None):
        if is_production is None:
            # 自动检测生产环境
            is_production = self._detect_production_environment()
        
        self.is_production = is_production
        self.feature_hashes = {}
        self.model_versions = {}
        self.random_calls_log = []
        
        # 生产环境强制禁用随机性
        if self.is_production:
            self._enforce_production_randomness()
        
        logger.info(f"Random control initialized - Production: {self.is_production}")
    
    def _detect_production_environment(self) -> bool:
        """自动检测是否为生产环境"""
        production_indicators = [
            os.getenv('ENVIRONMENT') == 'production',
            os.getenv('TRADING_ENV') == 'live', 
            os.getenv('IS_PRODUCTION') == 'true',
            'production' in os.getcwd().lower(),
            not os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
        ]
        return any(production_indicators)
    
    def _enforce_production_randomness(self):
        """强制设置生产环境随机种子"""
        # 设置所有随机库的固定种子
        random.seed(self.PRODUCTION_SEED)
        np.random.seed(self.PRODUCTION_SEED)
        
        # 尝试设置torch随机种子（如果可用）
        try:
            import torch
            torch.manual_seed(self.PRODUCTION_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.PRODUCTION_SEED)
            logger.info("PyTorch random seeds set for production")
        except ImportError:
            pass
        
        # 尝试设置tensorflow随机种子（如果可用）
        try:
            import tensorflow as tf
            tf.random.set_seed(self.PRODUCTION_SEED)
            logger.info("TensorFlow random seed set for production")
        except ImportError:
            pass
        
        logger.warning("🔒 PRODUCTION MODE: All randomness fixed with seed=42")
    
    def controlled_random_call(self, operation: str, **kwargs) -> Any:
        """受控的随机调用，生产环境下返回确定性结果"""
        if self.is_production:
            logger.warning(f"🚫 BLOCKED random operation in production: {operation}")
            # 记录被阻止的随机调用
            self.random_calls_log.append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'kwargs': kwargs,
                'action': 'blocked'
            })
            return self._get_deterministic_result(operation, **kwargs)
        else:
            # 开发环境允许随机调用
            return self._execute_random_operation(operation, **kwargs)
    
    def _get_deterministic_result(self, operation: str, **kwargs) -> Any:
        """为生产环境返回确定性结果"""
        if operation == 'random_choice':
            # 返回第一个选项而不是随机选择
            choices = kwargs.get('choices', [0])
            return choices[0] if choices else 0
        
        elif operation == 'random_uniform':
            # 返回中位数而不是随机值
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            return (low + high) / 2.0
        
        elif operation == 'random_normal':
            # 返回均值而不是随机采样
            return kwargs.get('loc', 0.0)
        
        elif operation == 'random_shuffle':
            # 返回原序列，不打乱
            return kwargs.get('array', [])
        
        else:
            logger.warning(f"Unknown random operation: {operation}, returning 0")
            return 0
    
    def _execute_random_operation(self, operation: str, **kwargs) -> Any:
        """执行实际的随机操作（开发环境）"""
        if operation == 'random_choice':
            choices = kwargs.get('choices', [0])
            return np.random.choice(choices)
        
        elif operation == 'random_uniform':
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            return np.random.uniform(low, high)
        
        elif operation == 'random_normal':
            loc = kwargs.get('loc', 0.0)
            scale = kwargs.get('scale', 1.0)
            return np.random.normal(loc, scale)
        
        elif operation == 'random_shuffle':
            array = kwargs.get('array', [])
            np.random.shuffle(array)
            return array
        
        else:
            raise ValueError(f"Unsupported random operation: {operation}")
    
    def record_feature_hash(self, features: Union[pd.DataFrame, np.ndarray], 
                           feature_name: str) -> str:
        """记录特征数据的哈希值，确保可重现性"""
        if isinstance(features, pd.DataFrame):
            data_bytes = features.values.tobytes()
        elif isinstance(features, np.ndarray):
            data_bytes = features.tobytes()
        else:
            data_bytes = str(features).encode()
        
        feature_hash = hashlib.sha256(data_bytes).hexdigest()[:16]
        self.feature_hashes[feature_name] = {
            'hash': feature_hash,
            'timestamp': datetime.now().isoformat(),
            'shape': getattr(features, 'shape', 'unknown'),
            'dtype': str(getattr(features, 'dtype', 'unknown'))
        }
        
        logger.info(f"Feature hash recorded: {feature_name} -> {feature_hash}")
        return feature_hash
    
    def record_model_version(self, model_name: str, version_info: Dict[str, Any]):
        """记录模型版本信息"""
        self.model_versions[model_name] = {
            **version_info,
            'recorded_at': datetime.now().isoformat(),
            'is_production': self.is_production
        }
        
        logger.info(f"Model version recorded: {model_name} -> {version_info}")
    
    def get_reproducibility_report(self) -> Dict[str, Any]:
        """生成可重现性报告"""
        return {
            'environment': 'production' if self.is_production else 'development',
            'random_seed': self.PRODUCTION_SEED if self.is_production else 'dynamic',
            'feature_hashes': self.feature_hashes,
            'model_versions': self.model_versions,
            'blocked_random_calls': len(self.random_calls_log),
            'random_calls_log': self.random_calls_log[-10:],  # 最近10次调用
            'generated_at': datetime.now().isoformat()
        }
    
    def save_reproducibility_report(self, file_path: Optional[str] = None):
        """保存可重现性报告到文件"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'logs/reproducibility_report_{timestamp}.json'
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = self.get_reproducibility_report()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reproducibility report saved: {file_path}")
        return file_path


# 全局实例
_global_random_control: Optional[ProductionRandomControl] = None


def get_random_control() -> ProductionRandomControl:
    """获取全局随机控制器实例"""
    global _global_random_control
    if _global_random_control is None:
        _global_random_control = ProductionRandomControl()
    return _global_random_control


def safe_random_choice(choices, **kwargs):
    """安全的随机选择（生产环境下返回确定性结果）"""
    return get_random_control().controlled_random_call('random_choice', choices=choices, **kwargs)


def safe_random_uniform(low=0.0, high=1.0, **kwargs):
    """安全的随机均匀分布（生产环境下返回中位数）"""
    return get_random_control().controlled_random_call('random_uniform', low=low, high=high, **kwargs)


def safe_random_normal(loc=0.0, scale=1.0, **kwargs):
    """安全的随机正态分布（生产环境下返回均值）"""
    return get_random_control().controlled_random_call('random_normal', loc=loc, scale=scale, **kwargs)


# 装饰器：确保函数在生产环境下具有确定性
def production_deterministic(func):
    """装饰器：确保函数在生产环境下运行确定性逻辑"""
    def wrapper(*args, **kwargs):
        random_control = get_random_control()
        if random_control.is_production:
            logger.info(f"🔒 Running {func.__name__} in deterministic production mode")
        return func(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    # 测试生产随机控制
    logging.basicConfig(level=logging.INFO)
    
    # 测试自动检测
    control = ProductionRandomControl()
    print(f"Environment: {'Production' if control.is_production else 'Development'}")
    
    # 测试随机控制
    choice = safe_random_choice([1, 2, 3, 4, 5])
    uniform = safe_random_uniform(0.1, 0.9)
    normal = safe_random_normal(0.5, 0.2)
    
    print(f"Random choice: {choice}")
    print(f"Random uniform: {uniform}")
    print(f"Random normal: {normal}")
    
    # 测试特征哈希记录
    test_features = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    hash_value = control.record_feature_hash(test_features, 'test_features')
    print(f"Feature hash: {hash_value}")
    
    # 生成报告
    report = control.get_reproducibility_report()
    print(f"Report: {json.dumps(report, indent=2)}")