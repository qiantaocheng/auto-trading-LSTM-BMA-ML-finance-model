#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha IC加权处理器 - 核心依赖修复
用于基于IC值对Alpha因子进行加权处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ICWeightedAlphaProcessor:
    """IC加权Alpha处理器"""
    
    def __init__(self, lookback_window: int = 60, min_ic_threshold: float = 0.01):
        """
        初始化IC加权处理器
        
        Args:
            lookback_window: IC计算的回望窗口
            min_ic_threshold: 最小IC阈值
        """
        self.lookback_window = lookback_window
        self.min_ic_threshold = min_ic_threshold
        self.ic_history = {}
        self.weights = {}
        
    def calculate_ic(self, predictions: pd.Series, targets: pd.Series, 
                     dates: pd.Series = None) -> float:
        """
        计算信息系数(IC)
        
        Args:
            predictions: 预测值
            targets: 真实值
            dates: 日期序列（可选）
            
        Returns:
            IC值
        """
        try:
            # 对齐数据
            valid_idx = ~(predictions.isna() | targets.isna())
            if valid_idx.sum() < 10:
                return 0.0
            
            pred_clean = predictions[valid_idx]
            target_clean = targets[valid_idx]
            
            # 计算Rank IC
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred_clean, target_clean)
            
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            logger.warning(f"IC计算失败: {e}")
            return 0.0
    
    def calculate_rolling_ic(self, predictions: pd.DataFrame, 
                            targets: pd.Series,
                            dates: pd.Series) -> pd.DataFrame:
        """
        计算滚动IC
        
        Args:
            predictions: 多个模型的预测值
            targets: 真实值
            dates: 日期序列
            
        Returns:
            滚动IC DataFrame
        """
        rolling_ics = {}
        
        for col in predictions.columns:
            daily_ics = []
            unique_dates = dates.unique()
            
            for date in unique_dates:
                date_mask = dates == date
                if date_mask.sum() > 5:  # 至少5个样本
                    ic = self.calculate_ic(
                        predictions.loc[date_mask, col],
                        targets[date_mask]
                    )
                    daily_ics.append(ic)
                else:
                    daily_ics.append(np.nan)
            
            rolling_ics[col] = daily_ics
        
        return pd.DataFrame(rolling_ics, index=unique_dates)
    
    def calculate_weights(self, ic_scores: Dict[str, float], 
                         method: str = 'exponential') -> Dict[str, float]:
        """
        基于IC计算权重
        
        Args:
            ic_scores: 各模型的IC分数
            method: 权重计算方法
            
        Returns:
            归一化的权重字典
        """
        if not ic_scores:
            return {}
        
        # 过滤低IC模型
        filtered_scores = {k: max(v, 0) for k, v in ic_scores.items() 
                          if abs(v) >= self.min_ic_threshold}
        
        if not filtered_scores:
            # 如果都不满足阈值，使用等权重
            n = len(ic_scores)
            return {k: 1.0/n for k in ic_scores.keys()}
        
        if method == 'exponential':
            # 指数加权
            exp_scores = {k: np.exp(v * 10) for k, v in filtered_scores.items()}
            total = sum(exp_scores.values())
            weights = {k: v/total for k, v in exp_scores.items()}
        elif method == 'linear':
            # 线性加权
            total = sum(filtered_scores.values())
            if total > 0:
                weights = {k: v/total for k, v in filtered_scores.items()}
            else:
                n = len(filtered_scores)
                weights = {k: 1.0/n for k in filtered_scores.keys()}
        else:
            # 平方根加权
            sqrt_scores = {k: np.sqrt(abs(v)) * np.sign(v) 
                          for k, v in filtered_scores.items()}
            total = sum(abs(v) for v in sqrt_scores.values())
            weights = {k: abs(v)/total for k, v in sqrt_scores.items()}
        
        # 补充未包含的模型（权重为0）
        for k in ic_scores:
            if k not in weights:
                weights[k] = 0.0
        
        return weights
    
    def process_alpha_signals(self, signals: pd.DataFrame,
                             targets: pd.Series,
                             dates: pd.Series) -> pd.Series:
        """
        处理Alpha信号并返回IC加权组合
        
        Args:
            signals: Alpha信号DataFrame
            targets: 目标值
            dates: 日期
            
        Returns:
            IC加权的组合信号
        """
        # 计算各信号的IC
        ic_scores = {}
        for col in signals.columns:
            ic = self.calculate_ic(signals[col], targets, dates)
            ic_scores[col] = ic
            logger.info(f"Signal {col} IC: {ic:.4f}")
        
        # 计算权重
        weights = self.calculate_weights(ic_scores)
        self.weights = weights
        
        # 生成加权组合
        weighted_signal = pd.Series(0, index=signals.index)
        for col, weight in weights.items():
            if weight > 0 and col in signals.columns:
                weighted_signal += signals[col] * weight
        
        return weighted_signal
    
    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'weights': self.weights,
            'ic_history': self.ic_history,
            'lookback_window': self.lookback_window,
            'min_ic_threshold': self.min_ic_threshold
        }


class AlphaSignalProcessor:
    """Alpha信号处理器 - 用于信号清洗和标准化"""
    
    def __init__(self):
        self.scaler = None
        
    def clean_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """清洗Alpha信号"""
        # 移除异常值
        for col in signals.columns:
            # 3-sigma规则
            mean = signals[col].mean()
            std = signals[col].std()
            signals[col] = signals[col].clip(mean - 3*std, mean + 3*std)
        
        # 填充缺失值
        signals = signals.fillna(method='ffill').fillna(0)
        
        return signals
    
    def standardize_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """标准化Alpha信号"""
        from sklearn.preprocessing import RobustScaler
        
        if self.scaler is None:
            self.scaler = RobustScaler()
            return pd.DataFrame(
                self.scaler.fit_transform(signals),
                index=signals.index,
                columns=signals.columns
            )
        else:
            return pd.DataFrame(
                self.scaler.transform(signals),
                index=signals.index,
                columns=signals.columns
            )


def create_ic_weighted_processor(config: Dict = None) -> ICWeightedAlphaProcessor:
    """工厂函数：创建IC加权处理器"""
    if config is None:
        config = {
            'lookback_window': 60,
            'min_ic_threshold': 0.01
        }
    
    return ICWeightedAlphaProcessor(
        lookback_window=config.get('lookback_window', 60),
        min_ic_threshold=config.get('min_ic_threshold', 0.01)
    )