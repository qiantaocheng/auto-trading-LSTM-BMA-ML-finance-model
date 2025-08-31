#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳健的IC计算模块
统一的Information Coefficient计算，具有异常处理和数据验证
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RobustICCalculator:
    """稳健的IC计算器"""
    
    def __init__(self, min_samples: int = 10, outlier_threshold: float = 3.0):
        """
        初始化IC计算器
        
        Args:
            min_samples: 最小样本数要求
            outlier_threshold: 异常值检测阈值（标准差倍数）
        """
        self.min_samples = min_samples
        self.outlier_threshold = outlier_threshold
    
    def calculate_ic(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        remove_outliers: bool = True
    ) -> Dict[str, float]:
        """
        计算稳健的IC指标
        
        Args:
            predictions: 预测值
            targets: 目标值
            remove_outliers: 是否移除异常值
            
        Returns:
            包含IC指标的字典
        """
        result = {
            'pearson_ic': 0.0,
            'pearson_pvalue': 1.0,
            'spearman_ic': 0.0,
            'spearman_pvalue': 1.0,
            'kendall_ic': 0.0,
            'kendall_pvalue': 1.0,
            'n_samples': 0,
            'n_valid_samples': 0,
            'outliers_removed': 0,
            'valid': False,
            'error': None
        }
        
        try:
            # 输入验证
            if not self._validate_inputs(predictions, targets):
                result['error'] = 'Invalid input data'
                return result
            
            # 转换为numpy数组
            pred_array = np.asarray(predictions, dtype=float)
            target_array = np.asarray(targets, dtype=float)
            
            result['n_samples'] = len(pred_array)
            
            # 移除NaN和Inf值
            valid_mask = (
                np.isfinite(pred_array) & 
                np.isfinite(target_array) & 
                ~np.isnan(pred_array) & 
                ~np.isnan(target_array)
            )
            
            pred_clean = pred_array[valid_mask]
            target_clean = target_array[valid_mask]
            
            result['n_valid_samples'] = len(pred_clean)
            
            # 检查最小样本数
            if len(pred_clean) < self.min_samples:
                result['error'] = f'Insufficient samples: {len(pred_clean)} < {self.min_samples}'
                return result
            
            # 异常值处理
            if remove_outliers:
                outlier_mask = self._detect_outliers(pred_clean, target_clean)
                pred_clean = pred_clean[~outlier_mask]
                target_clean = target_clean[~outlier_mask]
                result['outliers_removed'] = np.sum(outlier_mask)
                result['n_valid_samples'] = len(pred_clean)
                
                # 再次检查样本数
                if len(pred_clean) < self.min_samples:
                    result['error'] = f'Too many outliers removed, samples: {len(pred_clean)}'
                    return result
            
            # 检查方差
            if np.var(pred_clean) == 0 or np.var(target_clean) == 0:
                result['error'] = 'Zero variance in predictions or targets'
                return result
            
            # 计算Pearson相关系数
            try:
                pearson_ic, pearson_pvalue = stats.pearsonr(pred_clean, target_clean)
                result['pearson_ic'] = float(pearson_ic) if np.isfinite(pearson_ic) else 0.0
                result['pearson_pvalue'] = float(pearson_pvalue) if np.isfinite(pearson_pvalue) else 1.0
            except Exception as e:
                logger.warning(f"Pearson计算失败: {e}")
                result['pearson_ic'] = 0.0
                result['pearson_pvalue'] = 1.0
            
            # 计算Spearman相关系数  
            try:
                spearman_ic, spearman_pvalue = stats.spearmanr(pred_clean, target_clean)
                result['spearman_ic'] = float(spearman_ic) if np.isfinite(spearman_ic) else 0.0
                result['spearman_pvalue'] = float(spearman_pvalue) if np.isfinite(spearman_pvalue) else 1.0
            except Exception as e:
                logger.warning(f"Spearman计算失败: {e}")
                result['spearman_ic'] = 0.0
                result['spearman_pvalue'] = 1.0
            
            # 计算Kendall相关系数（更稳健但计算较慢）
            try:
                if len(pred_clean) <= 1000:  # 只对小样本计算Kendall
                    kendall_ic, kendall_pvalue = stats.kendalltau(pred_clean, target_clean)
                    result['kendall_ic'] = float(kendall_ic) if np.isfinite(kendall_ic) else 0.0
                    result['kendall_pvalue'] = float(kendall_pvalue) if np.isfinite(kendall_pvalue) else 1.0
            except Exception as e:
                logger.warning(f"Kendall计算失败: {e}")
                result['kendall_ic'] = 0.0
                result['kendall_pvalue'] = 1.0
            
            result['valid'] = True
            
        except Exception as e:
            logger.error(f"IC计算异常: {e}")
            result['error'] = str(e)
        
        return result
    
    def _validate_inputs(self, predictions: np.ndarray, targets: np.ndarray) -> bool:
        """验证输入数据"""
        try:
            if predictions is None or targets is None:
                return False
            
            if len(predictions) == 0 or len(targets) == 0:
                return False
            
            if len(predictions) != len(targets):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _detect_outliers(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """检测异常值"""
        try:
            # 使用Z-score方法检测异常值
            pred_z = np.abs(stats.zscore(predictions))
            target_z = np.abs(stats.zscore(targets))
            
            outlier_mask = (
                (pred_z > self.outlier_threshold) | 
                (target_z > self.outlier_threshold)
            )
            
            return outlier_mask
            
        except Exception as e:
            logger.warning(f"异常值检测失败: {e}")
            return np.zeros(len(predictions), dtype=bool)
    
    def calculate_rolling_ic(
        self, 
        predictions: pd.Series, 
        targets: pd.Series, 
        window: int = 252,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        计算滚动IC
        
        Args:
            predictions: 预测值时间序列
            targets: 目标值时间序列
            window: 滚动窗口大小
            min_periods: 最小观测数
            
        Returns:
            滚动IC时间序列
        """
        if min_periods is None:
            min_periods = max(self.min_samples, window // 2)
        
        def rolling_spearman(pred_window, target_window):
            try:
                if len(pred_window) < self.min_samples:
                    return np.nan
                
                ic_result = self.calculate_ic(
                    pred_window.values, 
                    target_window.values, 
                    remove_outliers=True
                )
                
                return ic_result['spearman_ic'] if ic_result['valid'] else np.nan
                
            except Exception:
                return np.nan
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'target': targets
        }).dropna()
        
        if len(aligned_data) < min_periods:
            return pd.Series(index=predictions.index, dtype=float)
        
        # 计算滚动IC
        rolling_ic = aligned_data.rolling(
            window=window, 
            min_periods=min_periods
        ).apply(lambda x: rolling_spearman(x['pred'], x['target']), raw=False)['pred']
        
        # 重新索引到原始索引
        return rolling_ic.reindex(predictions.index)


def create_robust_ic_calculator(min_samples: int = 10, outlier_threshold: float = 3.0) -> RobustICCalculator:
    """创建稳健IC计算器的工厂函数"""
    return RobustICCalculator(min_samples=min_samples, outlier_threshold=outlier_threshold)


# 便捷函数
def calculate_ic(predictions: np.ndarray, targets: np.ndarray, **kwargs) -> Dict[str, float]:
    """快速计算IC的便捷函数"""
    calculator = create_robust_ic_calculator()
    return calculator.calculate_ic(predictions, targets, **kwargs)


def calculate_rolling_ic(predictions: pd.Series, targets: pd.Series, window: int = 252, **kwargs) -> pd.Series:
    """快速计算滚动IC的便捷函数"""
    calculator = create_robust_ic_calculator()
    return calculator.calculate_rolling_ic(predictions, targets, window=window, **kwargs)