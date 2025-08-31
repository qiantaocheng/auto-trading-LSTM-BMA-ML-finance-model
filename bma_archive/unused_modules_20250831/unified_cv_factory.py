#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一CV工厂 - 机构级单一真相源
提供唯一的Purged CV实现，消除各模块重复CV定义
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UnifiedCVFactory:
    """
    统一CV工厂 - 单一真相源
    
    职责：
    1. 提供唯一的PurgedGroupTimeSeriesSplit实现
    2. 强制使用T10_CONFIG的统一参数
    3. 确保所有训练头使用相同的CV策略
    4. 防止CV参数被各模块私自修改
    """
    
    def __init__(self, config_source='t10_config'):
        """
        初始化CV工厂
        
        Args:
            config_source: 配置来源 ('t10_config' | 'manual')
        """
        self.config_source = config_source
        self._load_unified_config()
        self._cv_cache = {}  # CV实例缓存，避免重复创建
        
        logger.info(f"统一CV工厂初始化 - 配置源: {config_source}")
        logger.info(f"统一参数: isolation={self.isolation_days}, embargo={self.embargo_days}, splits={self.cv_n_splits}")
    
    def _load_unified_config(self):
        """从统一配置源加载CV参数"""
        try:
            if self.config_source == 't10_config':
                from .t10_config import T10_CONFIG
                
                # 🔥 单一真相源：只从T10_CONFIG读取，禁止各模块覆盖
                self.prediction_horizon = T10_CONFIG.PREDICTION_HORIZON
                self.isolation_days = T10_CONFIG.ISOLATION_DAYS  
                self.embargo_days = T10_CONFIG.EMBARGO_DAYS
                self.cv_n_splits = T10_CONFIG.CV_N_SPLITS
                
                # 验证参数合理性
                self._validate_config()
                
            else:
                # 手动配置模式（用于测试）
                self.prediction_horizon = 10
                self.isolation_days = 21
                self.embargo_days = 15
                self.cv_n_splits = 5
                
        except ImportError:
            logger.warning("T10_CONFIG不可用，使用默认参数")
            self.prediction_horizon = 10
            self.isolation_days = 21
            self.embargo_days = 15
            self.cv_n_splits = 5
    
    def _validate_config(self):
        """验证CV配置参数"""
        if self.isolation_days < 1:
            raise ValueError(f"ISOLATION_DAYS必须>=1，当前值: {self.isolation_days}")
        
        if self.embargo_days < 1:
            raise ValueError(f"EMBARGO_DAYS必须>=1，当前值: {self.embargo_days}")
        
        if self.cv_n_splits < 2:
            raise ValueError(f"CV_N_SPLITS必须>=2，当前值: {self.cv_n_splits}")
        
        # 检查embargo >= 持有期+1的要求
        holding_period = 1  # T+1持有
        required_embargo = holding_period + 1
        if self.embargo_days < required_embargo:
            logger.warning(f"EMBARGO_DAYS({self.embargo_days}) < 推荐值({required_embargo})，可能存在微小泄露风险")
    
    def create_cv_splitter(self, dates: pd.Series, strict_validation: bool = True) -> Callable:
        """
        创建统一的CV分割器
        
        Args:
            dates: 日期序列
            strict_validation: 是否启用严格验证
            
        Returns:
            CV分割函数
        """
        # 生成缓存键
        if hasattr(dates, 'iloc'):
            date_hash = hash(tuple(dates.iloc[::100]))  # 采样hash避免过长
        else:
            date_hash = hash(tuple(dates[::100]))  # DatetimeIndex使用直接索引
        cache_key = f"{date_hash}_{strict_validation}_{self.cv_n_splits}"
        
        if cache_key in self._cv_cache:
            logger.debug(f"使用缓存的CV分割器: {cache_key}")
            return self._cv_cache[cache_key]
        
        # 创建groups（从dates派生）
        groups = self._create_groups_from_dates(dates)
        
        # 创建配置
        from .fixed_purged_time_series_cv import ValidationConfig
        
        config = ValidationConfig(
            n_splits=self.cv_n_splits,  # 🔥 统一折数
            test_size=63,  # 约3个月测试集
            gap=self.isolation_days,   # 🔥 统一隔离期
            embargo=self.embargo_days, # 🔥 统一embargo
            min_train_size=252,  # 最小1年训练集
            group_freq='D',
            strict_validation=strict_validation
        )
        
        # 创建CV分割器
        from .fixed_purged_time_series_cv import FixedPurgedGroupTimeSeriesSplit
        
        cv_splitter = FixedPurgedGroupTimeSeriesSplit(config)
        
        # 🔥 关键修复：创建统一的分割函数，确保groups必传
        def unified_split_function(X, y=None, **kwargs):
            """
            统一的CV分割函数
            强制传入groups，防止各模块绕过group约束
            """
            # 验证输入
            if len(X) != len(groups):
                raise ValueError(f"特征长度({len(X)})与日期长度({len(groups)})不匹配")
            
            # 🔥 CRITICAL: groups必传，禁止退化到TimeSeriesSplit
            if 'groups' in kwargs and kwargs['groups'] is not None:
                logger.warning("检测到外部传入groups，将被统一groups覆盖")
            
            logger.info(f"执行统一CV分割: {len(groups)}个样本, {self.cv_n_splits}折")
            logger.info(f"CV参数: isolation={self.isolation_days}, embargo={self.embargo_days}")
            
            # 执行分割
            splits = list(cv_splitter.split(X, y, groups=groups))
            
            if len(splits) == 0:
                raise ValueError("CV分割失败，未生成任何有效分割")
            
            logger.info(f"✅ CV分割成功: {len(splits)}个有效分割")
            
            return splits
        
        # 缓存分割器
        self._cv_cache[cache_key] = unified_split_function
        
        return unified_split_function
    
    def _create_groups_from_dates(self, dates: pd.Series) -> np.ndarray:
        """从日期序列创建groups"""
        try:
            # 确保是datetime类型
            dt_dates = pd.to_datetime(dates)
            
            # 生成YYYYMMDD格式的groups
            groups = dt_dates.dt.strftime("%Y%m%d").values
            
            logger.debug(f"生成groups: {len(groups)}个, 范围: {groups[0]} - {groups[-1]}")
            
            return groups
            
        except Exception as e:
            logger.error(f"groups生成失败: {e}")
            # 回退方案：使用索引生成伪groups
            logger.warning("使用索引回退方案生成groups")
            return np.arange(len(dates)).astype(str)
    
    def create_cv_factory(self) -> Callable:
        """
        创建CV工厂函数（主要接口）
        
        Returns:
            cv_factory函数，接受dates参数并返回CV分割器
        """
        def cv_factory(dates: pd.Series, strict_validation: bool = True) -> Callable:
            """
            CV工厂函数
            
            Args:
                dates: 日期序列
                strict_validation: 是否启用严格验证
                
            Returns:
                CV分割函数
            """
            return self.create_cv_splitter(dates, strict_validation)
        
        return cv_factory
    
    def validate_cv_integrity(self, dates: pd.Series, X: pd.DataFrame = None) -> dict:
        """
        验证CV完整性（类似原系统的validate_timesplit_integrity）
        
        Args:
            dates: 日期序列
            X: 特征数据（可选）
            
        Returns:
            验证结果
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            # 基本统计
            validation_result['stats'] = {
                'total_samples': len(dates),
                'date_range_days': (dates.max() - dates.min()).days,
                'unique_dates': dates.nunique(),
                'cv_splits': self.cv_n_splits,
                'isolation_days': self.isolation_days,
                'embargo_days': self.embargo_days
            }
            
            # 数据充足性检查
            min_required_days = (self.cv_n_splits + 2) * 63 + self.isolation_days + self.embargo_days
            if validation_result['stats']['date_range_days'] < min_required_days:
                validation_result['warnings'].append(
                    f"数据时间跨度({validation_result['stats']['date_range_days']}天) < 推荐值({min_required_days}天)"
                )
            
            # CV分割测试
            cv_splitter = self.create_cv_splitter(dates, strict_validation=False)
            
            if X is not None:
                test_splits = cv_splitter(X)
                validation_result['stats']['actual_splits'] = len(test_splits)
                
                if len(test_splits) != self.cv_n_splits:
                    validation_result['warnings'].append(
                        f"实际分割数({len(test_splits)}) != 期望分割数({self.cv_n_splits})"
                    )
                
                # 检查分割大小
                for i, (train_idx, test_idx) in enumerate(test_splits):
                    if len(train_idx) < 252:  # 最小训练集要求
                        validation_result['warnings'].append(
                            f"第{i+1}折训练集过小: {len(train_idx)}样本"
                        )
                    if len(test_idx) < 20:  # 最小测试集要求
                        validation_result['warnings'].append(
                            f"第{i+1}折测试集过小: {len(test_idx)}样本"
                        )
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"CV验证失败: {str(e)}")
        
        # 判断总体有效性
        if validation_result['errors']:
            validation_result['valid'] = False
        
        return validation_result
    
    def get_config_summary(self) -> dict:
        """获取配置摘要"""
        return {
            'config_source': self.config_source,
            'prediction_horizon': self.prediction_horizon,
            'isolation_days': self.isolation_days,
            'embargo_days': self.embargo_days,
            'cv_n_splits': self.cv_n_splits,
            'cache_size': len(self._cv_cache),
            'is_production_config': self.config_source == 't10_config'
        }
    
    def get_fingerprint(self, dates: pd.Series, X: pd.DataFrame = None) -> dict:
        """
        生成CV分割指纹（用于split_fingerprint.json）
        
        Args:
            dates: 日期序列
            X: 特征数据（可选，用于数据hash）
            
        Returns:
            CV分割指纹字典
        """
        import hashlib
        from datetime import datetime
        
        try:
            # 基本配置指纹
            fingerprint = {
                'cv_config': {
                    'n_splits': self.cv_n_splits,
                    'isolation_days': self.isolation_days,
                    'embargo_days': self.embargo_days,
                    'prediction_horizon': self.prediction_horizon,
                    'cv_type': 'FixedPurgedGroupTimeSeriesSplit'
                },
                'data_info': {
                    'total_samples': len(dates),
                    'date_range': {
                        'start': str(dates.min()),
                        'end': str(dates.max()),
                        'days': (dates.max() - dates.min()).days
                    },
                    'unique_dates': dates.nunique()
                },
                'generation_timestamp': datetime.now().isoformat(),
                'config_source': self.config_source
            }
            
            # Groups生成
            groups = self._create_groups_from_dates(dates)
            fingerprint['groups_info'] = {
                'total_groups': len(groups),
                'unique_groups': len(set(groups)),
                'first_group': str(groups[0]),
                'last_group': str(groups[-1])
            }
            
            # 数据hash（如果提供了X）
            if X is not None:
                data_str = f"{X.shape}_{X.columns.tolist()}_{X.fillna(0).sum().sum()}"
                data_hash = hashlib.md5(data_str.encode()).hexdigest()[:16]
                fingerprint['data_hash'] = data_hash
            
            # Git信息（如果可用）
            try:
                import subprocess
                git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8]
                fingerprint['git_sha'] = git_sha
            except:
                fingerprint['git_sha'] = 'unknown'
            
            # 种子信息（如果需要）
            fingerprint['seed_info'] = {
                'numpy_seed_state': 'not_set',  # CV分割通常不使用随机种子
                'deterministic': True  # Purged CV是确定性的
            }
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"生成CV指纹失败: {e}")
            return {
                'error': str(e),
                'generation_timestamp': datetime.now().isoformat(),
                'cv_config': {
                    'n_splits': self.cv_n_splits,
                    'isolation_days': self.isolation_days,
                    'embargo_days': self.embargo_days
                }
            }
    
    def save_split_fingerprint(self, dates: pd.Series, X: pd.DataFrame = None, 
                             output_path: str = "split_fingerprint.json") -> bool:
        """
        保存CV分割指纹到文件
        
        Args:
            dates: 日期序列
            X: 特征数据（可选）
            output_path: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            import json
            
            fingerprint = self.get_fingerprint(dates, X)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ CV分割指纹已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存CV分割指纹失败: {e}")
            return False


# 全局CV工厂实例
_global_cv_factory = None

def get_unified_cv_factory() -> UnifiedCVFactory:
    """获取全局统一CV工厂实例"""
    global _global_cv_factory
    if _global_cv_factory is None:
        _global_cv_factory = UnifiedCVFactory('t10_config')
    return _global_cv_factory

def create_cv_for_training(dates: pd.Series) -> Callable:
    """
    便捷函数：为训练创建统一CV
    
    Args:
        dates: 日期序列
        
    Returns:
        CV分割函数
    """
    factory = get_unified_cv_factory()
    return factory.create_cv_splitter(dates, strict_validation=True)


if __name__ == "__main__":
    # 测试统一CV工厂
    logger.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    X = pd.DataFrame(np.random.randn(1000, 10))
    
    print("测试统一CV工厂")
    
    # 创建工厂
    factory = UnifiedCVFactory('manual')
    
    # 配置摘要
    config = factory.get_config_summary()
    print(f"配置摘要: {config}")
    
    # 创建CV分割器
    cv_splitter = factory.create_cv_splitter(dates)
    
    # 测试分割
    splits = cv_splitter(X)
    print(f"生成分割: {len(splits)}个")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  分割{i+1}: 训练{len(train_idx)}, 测试{len(test_idx)}")
    
    # 验证完整性
    validation = factory.validate_cv_integrity(dates, X)
    print(f"验证结果: {validation['valid']}")
    if validation['warnings']:
        print("警告:")
        for warning in validation['warnings']:
            print(f"  - {warning}")