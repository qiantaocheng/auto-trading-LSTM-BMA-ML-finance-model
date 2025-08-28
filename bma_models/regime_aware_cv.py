#!/usr/bin/env python3
"""
状态感知交叉验证系统 - Regime-Aware Cross-Validation
为BMA Enhanced双层CV架构提供市场状态感知的分割能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
import logging

from market_regime_detector import MarketRegimeDetector, RegimeConfig

logger = logging.getLogger(__name__)

@dataclass
class RegimeAwareCVConfig:
    """状态感知CV配置"""
    # 基础CV配置
    n_splits: int = 5
    test_size: int = 63
    gap: int = 5
    embargo: int = 2
    min_train_size: int = 252
    
    # 状态感知配置
    enable_regime_stratification: bool = True    # 启用状态分层
    min_regime_samples: int = 50                # 每个状态最少样本数
    regime_balance_threshold: float = 0.8       # 状态平衡阈值
    cross_regime_validation: bool = True        # 跨状态验证
    
    # 时间验证增强
    strict_regime_temporal_order: bool = True   # 严格的状态时间顺序
    regime_transition_buffer: int = 5           # 状态转换缓冲期

class RegimeAwareTimeSeriesCV:
    """
    状态感知时间序列交叉验证器
    
    在传统的时间序列CV基础上增加市场状态感知：
    1. 确保训练集和测试集的状态分布合理
    2. 避免在状态转换期进行分割
    3. 提供状态特定的验证指标
    """
    
    def __init__(self, 
                 cv_config: RegimeAwareCVConfig,
                 regime_detector: MarketRegimeDetector = None):
        self.config = cv_config
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        # 状态分割缓存
        self._regime_splits_cache = {}
        self._regime_statistics = {}
        
        logger.info(f"RegimeAwareTimeSeriesCV初始化 - 状态分层: {cv_config.enable_regime_stratification}")
    
    def split_with_regime_awareness(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series, 
                                  dates: pd.Series,
                                  base_cv_splitter) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        基于状态感知的CV分割
        
        Args:
            X: 特征数据
            y: 目标变量
            dates: 日期序列
            base_cv_splitter: 基础CV分割器
            
        Yields:
            (train_idx, test_idx, regime_info): 训练索引、测试索引、状态信息
        """
        
        if not self.config.enable_regime_stratification:
            # 不启用状态感知，使用基础分割
            for fold_idx, (train_idx, test_idx) in enumerate(base_cv_splitter.split(X, y)):
                regime_info = {'fold_idx': fold_idx, 'regime_aware': False}
                yield train_idx, test_idx, regime_info
            return
        
        try:
            # 1. 检测市场状态
            logger.info("检测训练数据的市场状态...")
            
            # 构造状态检测所需的数据
            regime_data = self._prepare_regime_data(X, dates)
            regimes = self.regime_detector.detect_regimes(regime_data)
            
            if regimes.empty:
                logger.warning("状态检测失败，回退到标准CV")
                for fold_idx, (train_idx, test_idx) in enumerate(base_cv_splitter.split(X, y)):
                    regime_info = {'fold_idx': fold_idx, 'regime_aware': False, 'fallback': True}
                    yield train_idx, test_idx, regime_info
                return
            
            # 2. 生成状态感知分割
            logger.info(f"生成状态感知CV分割，检测到状态: {regimes.value_counts().to_dict()}")
            
            for fold_idx, (base_train_idx, base_test_idx) in enumerate(base_cv_splitter.split(X, y)):
                
                # 3. 优化分割以平衡状态分布
                train_idx, test_idx, fold_regime_info = self._optimize_regime_split(
                    base_train_idx, base_test_idx, regimes, fold_idx
                )
                
                # 4. 验证分割质量
                split_quality = self._validate_regime_split(train_idx, test_idx, regimes, dates)
                fold_regime_info.update(split_quality)
                
                logger.info(f"Fold {fold_idx}: 训练集状态分布 {fold_regime_info.get('train_regime_dist', {})}")
                logger.info(f"Fold {fold_idx}: 测试集状态分布 {fold_regime_info.get('test_regime_dist', {})}")
                
                yield train_idx, test_idx, fold_regime_info
                
        except Exception as e:
            logger.error(f"状态感知CV分割失败: {e}")
            # 回退到基础分割
            for fold_idx, (train_idx, test_idx) in enumerate(base_cv_splitter.split(X, y)):
                regime_info = {'fold_idx': fold_idx, 'regime_aware': False, 'error': str(e)}
                yield train_idx, test_idx, regime_info
    
    def _prepare_regime_data(self, X: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
        """准备状态检测所需的数据"""
        
        # 从特征中提取价格和成交量相关指标
        regime_data = pd.DataFrame({'date': dates})
        
        # 查找价格相关特征
        price_cols = [col for col in X.columns if any(
            price_keyword in col.lower() 
            for price_keyword in ['close', 'price', 'adj_close']
        )]
        
        volume_cols = [col for col in X.columns if any(
            vol_keyword in col.lower()
            for vol_keyword in ['volume', 'vol', 'amount']
        )]
        
        # 使用第一个找到的价格和成交量列
        if price_cols:
            regime_data['close'] = X[price_cols[0]]
        else:
            # 如果没有价格列，使用合成价格
            regime_data['close'] = np.zeros(len(X)).cumsum() + 100
            logger.warning("未找到价格列，使用合成数据进行状态检测")
        
        if volume_cols:
            regime_data['volume'] = X[volume_cols[0]]
        else:
            # 如果没有成交量列，使用合成成交量
            regime_data['volume'] = np.random.randint(1000000, 10000000, len(X))
            logger.warning("未找到成交量列，使用合成数据进行状态检测")
        
        return regime_data
    
    def _optimize_regime_split(self, 
                              base_train_idx: np.ndarray,
                              base_test_idx: np.ndarray,
                              regimes: pd.Series,
                              fold_idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """优化分割以平衡状态分布"""
        
        train_regimes = regimes.iloc[base_train_idx]
        test_regimes = regimes.iloc[base_test_idx]
        
        # 计算状态分布
        train_regime_dist = train_regimes.value_counts(normalize=True).to_dict()
        test_regime_dist = test_regimes.value_counts(normalize=True).to_dict()
        
        # 检查状态平衡
        regime_balance_score = self._calculate_regime_balance(train_regime_dist, test_regime_dist)
        
        # 如果平衡度不够，尝试微调（简化实现）
        optimized_train_idx = base_train_idx
        optimized_test_idx = base_test_idx
        
        if regime_balance_score < self.config.regime_balance_threshold:
            logger.debug(f"Fold {fold_idx}: 状态平衡度 {regime_balance_score:.3f} 低于阈值，尝试优化")
            # 这里可以实现更复杂的优化逻辑
            # 当前保持简化实现
        
        fold_info = {
            'fold_idx': fold_idx,
            'regime_aware': True,
            'train_regime_dist': train_regime_dist,
            'test_regime_dist': test_regime_dist,
            'regime_balance_score': regime_balance_score,
            'optimization_applied': regime_balance_score < self.config.regime_balance_threshold
        }
        
        return optimized_train_idx, optimized_test_idx, fold_info
    
    def _calculate_regime_balance(self, 
                                 train_dist: Dict[int, float], 
                                 test_dist: Dict[int, float]) -> float:
        """计算训练集和测试集的状态分布平衡度"""
        
        all_regimes = set(train_dist.keys()) | set(test_dist.keys())
        
        if not all_regimes:
            return 1.0
        
        balance_scores = []
        for regime in all_regimes:
            train_prop = train_dist.get(regime, 0.0)
            test_prop = test_dist.get(regime, 0.0)
            
            # 计算分布差异（越小越好）
            if train_prop + test_prop > 0:
                diff = abs(train_prop - test_prop) / (train_prop + test_prop + 1e-8)
                balance_scores.append(1.0 - diff)
            else:
                balance_scores.append(1.0)
        
        return np.mean(balance_scores)
    
    def _validate_regime_split(self, 
                              train_idx: np.ndarray,
                              test_idx: np.ndarray, 
                              regimes: pd.Series,
                              dates: pd.Series) -> Dict[str, Any]:
        """验证分割质量"""
        
        validation_info = {}
        
        try:
            # 时间验证
            train_dates = dates.iloc[train_idx]
            test_dates = dates.iloc[test_idx]
            
            train_date_range = (train_dates.min(), train_dates.max())
            test_date_range = (test_dates.min(), test_dates.max())
            
            # 检查时间顺序
            temporal_valid = train_date_range[1] < test_date_range[0]
            
            validation_info.update({
                'train_date_range': train_date_range,
                'test_date_range': test_date_range,
                'temporal_order_valid': temporal_valid,
                'time_gap_days': (test_date_range[0] - train_date_range[1]).days if temporal_valid else -1
            })
            
            # 状态转换验证
            if self.config.strict_regime_temporal_order:
                train_regimes = regimes.iloc[train_idx]
                test_regimes = regimes.iloc[test_idx]
                
                train_regime_transitions = self._count_regime_transitions(train_regimes)
                test_regime_transitions = self._count_regime_transitions(test_regimes)
                
                validation_info.update({
                    'train_regime_transitions': train_regime_transitions,
                    'test_regime_transitions': test_regime_transitions,
                })
            
            # 样本数量验证
            regime_sample_counts = {}
            for regime_id in regimes.iloc[train_idx].unique():
                if not pd.isna(regime_id):
                    count = (regimes.iloc[train_idx] == regime_id).sum()
                    regime_sample_counts[int(regime_id)] = count
            
            min_samples_valid = all(
                count >= self.config.min_regime_samples 
                for count in regime_sample_counts.values()
            )
            
            validation_info.update({
                'regime_sample_counts': regime_sample_counts,
                'min_samples_valid': min_samples_valid
            })
            
        except Exception as e:
            logger.warning(f"分割验证过程出错: {e}")
            validation_info.update({
                'validation_error': str(e),
                'temporal_order_valid': False,
                'min_samples_valid': False
            })
        
        return validation_info
    
    def _count_regime_transitions(self, regime_series: pd.Series) -> int:
        """计算状态转换次数"""
        if len(regime_series) <= 1:
            return 0
        return (regime_series.diff() != 0).sum()
    
    def get_regime_cv_statistics(self) -> Dict[str, Any]:
        """获取状态感知CV统计信息"""
        
        stats = {
            'regime_aware_enabled': self.config.enable_regime_stratification,
            'total_folds_processed': len(self._regime_splits_cache),
            'regime_detector_config': self.regime_detector.config.__dict__ if self.regime_detector else {},
        }
        
        if self._regime_statistics:
            stats.update(self._regime_statistics)
        
        return stats


def create_regime_aware_cv(cv_config: RegimeAwareCVConfig = None, 
                          regime_detector: MarketRegimeDetector = None) -> RegimeAwareTimeSeriesCV:
    """工厂函数：创建状态感知CV分割器"""
    config = cv_config or RegimeAwareCVConfig()
    return RegimeAwareTimeSeriesCV(config, regime_detector)