#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda Percentile一致性转换器
确保训练和预测时Lambda percentile计算的一致性
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from typing import Optional

logger = logging.getLogger(__name__)


class LambdaPercentileTransformer:
    """
    Lambda预测到Percentile的一致性转换器

    解决训练-预测不对称问题：
    - 训练时：Lambda OOF预测来自不同fold，存在fold间方差
    - 预测时：Lambda预测来自单一模型，分布更一致

    转换器在训练时学习OOF预测的统计特性，预测时使用相同的映射方法
    """

    def __init__(self, method='quantile'):
        """
        初始化转换器

        Args:
            method: 转换方法
                - 'quantile': 使用分位数映射（推荐，更稳健）
                - 'zscore': 使用Z-score标准化后映射
                - 'rank': 直接排名（兼容旧方法）
        """
        self.method = method
        self.oof_mean_ = None
        self.oof_std_ = None
        self.oof_quantiles_ = None
        self.fitted_ = False

    def fit(self, lambda_oof_predictions: pd.Series):
        """
        从OOF预测中学习percentile转换

        Args:
            lambda_oof_predictions: Lambda模型的OOF预测（训练集）
        """
        logger.info(f"🔧 学习Lambda Percentile转换器 (方法={self.method})")

        # 保存OOF预测的统计特性
        self.oof_mean_ = float(lambda_oof_predictions.mean())
        self.oof_std_ = float(lambda_oof_predictions.std())

        # 计算OOF的分位数（0-100）
        self.oof_quantiles_ = lambda_oof_predictions.quantile(
            [i/100 for i in range(101)]
        ).values

        self.fitted_ = True

        logger.info(f"   OOF统计: mean={self.oof_mean_:.4f}, std={self.oof_std_:.4f}")
        logger.info(f"   OOF范围: [{self.oof_quantiles_[0]:.4f}, {self.oof_quantiles_[-1]:.4f}]")
        logger.info(f"   分位数已保存: {len(self.oof_quantiles_)} 个")

        return self

    def transform(self, lambda_predictions: pd.Series) -> pd.Series:
        """
        将新预测转换为percentile，保持与OOF一致的分布

        Args:
            lambda_predictions: Lambda模型的预测（新数据）

        Returns:
            percentile值（0-100）
        """
        if not self.fitted_:
            raise RuntimeError("转换器未拟合，请先调用fit()")

        if self.method == 'quantile':
            return self._transform_quantile(lambda_predictions)
        elif self.method == 'zscore':
            return self._transform_zscore(lambda_predictions)
        elif self.method == 'rank':
            return self._transform_rank(lambda_predictions)
        else:
            raise ValueError(f"未知的转换方法: {self.method}")

    def _transform_quantile(self, predictions: pd.Series) -> pd.Series:
        """方法1：使用训练时的分位数映射（推荐）"""
        # 将每个预测值映射到最近的OOF分位数
        percentiles = predictions.apply(
            lambda x: np.searchsorted(self.oof_quantiles_, x)
        ).astype(float)

        # 限制在0-100范围内
        percentiles = np.clip(percentiles, 0, 100)

        logger.info(f"✓ Quantile转换: 均值={percentiles.mean():.1f}, 范围=[{percentiles.min():.1f}, {percentiles.max():.1f}]")

        return pd.Series(percentiles, index=predictions.index, name='lambda_percentile')

    def _transform_zscore(self, predictions: pd.Series) -> pd.Series:
        """方法2：Z-score标准化后映射（更稳健）"""
        # 使用训练时的均值和标准差进行标准化
        z_scores = (predictions - self.oof_mean_) / (self.oof_std_ + 1e-8)

        # 使用正态分布CDF将Z-score映射到0-1，再乘以100
        percentiles = norm.cdf(z_scores) * 100

        # 限制在0-100范围内
        percentiles = np.clip(percentiles, 0, 100)

        logger.info(f"✓ Z-score转换: 均值={percentiles.mean():.1f}, 范围=[{percentiles.min():.1f}, {percentiles.max():.1f}]")

        return pd.Series(percentiles, index=predictions.index, name='lambda_percentile')

    def _transform_rank(self, predictions: pd.Series) -> pd.Series:
        """方法3：直接排名（兼容旧方法）"""
        # 按日期分组计算排名（如果是MultiIndex）
        if isinstance(predictions.index, pd.MultiIndex) and 'date' in predictions.index.names:
            percentiles = predictions.groupby(level='date').rank(pct=True) * 100
        else:
            percentiles = predictions.rank(pct=True) * 100

        logger.info(f"✓ Rank转换: 均值={percentiles.mean():.1f}, 范围=[{percentiles.min():.1f}, {percentiles.max():.1f}]")

        return percentiles

    def fit_transform(self, lambda_oof_predictions: pd.Series) -> pd.Series:
        """拟合并转换（训练时使用）"""
        self.fit(lambda_oof_predictions)
        return self.transform(lambda_oof_predictions)

    def get_params(self):
        """获取转换器参数"""
        if not self.fitted_:
            return None

        return {
            'method': self.method,
            'oof_mean': self.oof_mean_,
            'oof_std': self.oof_std_,
            'oof_quantiles_min': float(self.oof_quantiles_[0]),
            'oof_quantiles_max': float(self.oof_quantiles_[-1]),
            'fitted': self.fitted_
        }
