#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction-Only Blender - 纯预测导向融合器
专注于最大化预测性能，移除所有研究性组件
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class PredictionBlender:
    """
    纯预测导向融合器

    设计原则：
    1. 最大化预测性能 - 不是研究分析
    2. 最小化计算开销 - 使用numpy而非pandas操作
    3. 直接优化target correlation - 不搞复杂指标
    4. 零多余输出 - 只返回融合预测
    """

    def __init__(self, ridge_weight: float = 0.5):
        """
        Args:
            ridge_weight: Ridge权重，Lambda权重 = 1 - ridge_weight
        """
        self.ridge_weight = np.clip(ridge_weight, 0.0, 1.0)
        self.lambda_weight = 1.0 - self.ridge_weight

    def blend(self, ridge_scores: np.ndarray, lambda_scores: np.ndarray) -> np.ndarray:
        """
        核心融合函数 - 最简单最快

        Args:
            ridge_scores: Ridge预测分数
            lambda_scores: Lambda预测分数

        Returns:
            融合后的预测分数
        """
        # 直接加权平均，不做任何额外处理
        return self.ridge_weight * ridge_scores + self.lambda_weight * lambda_scores

    def blend_with_targets(self, ridge_scores: np.ndarray, lambda_scores: np.ndarray,
                          targets: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        基于目标优化权重的融合

        Args:
            ridge_scores: Ridge预测分数
            lambda_scores: Lambda预测分数
            targets: 真实目标值

        Returns:
            (融合分数, 最优权重)
        """
        # 直接搜索最优权重
        best_weight = self._find_optimal_weight(ridge_scores, lambda_scores, targets)
        self.ridge_weight = best_weight
        self.lambda_weight = 1.0 - best_weight

        # 使用最优权重融合
        blended = self.blend(ridge_scores, lambda_scores)
        return blended, best_weight

    def _find_optimal_weight(self, ridge_scores: np.ndarray, lambda_scores: np.ndarray,
                            targets: np.ndarray) -> float:
        """
        找到最大化correlation的权重
        """
        best_weight = 0.5
        best_corr = -1.0

        # 网格搜索最优权重
        for w in np.arange(0.1, 1.0, 0.1):
            blended = w * ridge_scores + (1 - w) * lambda_scores

            # 计算与target的correlation
            corr = np.corrcoef(blended, targets)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_weight = w

        return best_weight


class FastDataFrameBlender:
    """
    处理DataFrame但优化了性能的版本
    """

    def __init__(self, ridge_weight: float = 0.5):
        self.blender = PredictionBlender(ridge_weight)

    def blend_dataframes(self, ridge_df: pd.DataFrame, lambda_df: pd.DataFrame,
                        targets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        处理DataFrame格式的数据，但内部使用优化的numpy操作
        """
        # 快速对齐和提取数据
        common_idx = ridge_df.index.intersection(lambda_df.index)

        ridge_values = ridge_df.reindex(common_idx).iloc[:, 0].values
        lambda_values = lambda_df.reindex(common_idx).iloc[:, 0].values

        if targets_df is not None:
            # 有target时优化权重
            target_values = targets_df.reindex(common_idx).iloc[:, 0].values

            # 按日期分组优化（如果是MultiIndex）
            if isinstance(common_idx, pd.MultiIndex) and 'date' in common_idx.names:
                blended_values = self._blend_by_date_groups(
                    ridge_values, lambda_values, target_values, common_idx
                )
            else:
                blended_values, _ = self.blender.blend_with_targets(
                    ridge_values, lambda_values, target_values
                )
        else:
            # 没有target时使用固定权重
            blended_values = self.blender.blend(ridge_values, lambda_values)

        # 返回最简结果
        return pd.DataFrame({'prediction': blended_values}, index=common_idx)

    def _blend_by_date_groups(self, ridge_values: np.ndarray, lambda_values: np.ndarray,
                             target_values: np.ndarray, index: pd.MultiIndex) -> np.ndarray:
        """
        按日期分组优化权重（如果有MultiIndex）
        """
        # 获取日期分组
        dates = index.get_level_values('date')
        unique_dates = dates.unique()

        blended_values = np.zeros_like(ridge_values)

        for date in unique_dates:
            mask = dates == date

            if mask.sum() > 10:  # 只在样本足够时优化
                date_ridge = ridge_values[mask]
                date_lambda = lambda_values[mask]
                date_target = target_values[mask]

                date_blended, _ = self.blender.blend_with_targets(
                    date_ridge, date_lambda, date_target
                )
                blended_values[mask] = date_blended
            else:
                # 样本不足时使用默认权重
                blended_values[mask] = self.blender.blend(
                    ridge_values[mask], lambda_values[mask]
                )

        return blended_values


def quick_blend(ridge_pred: pd.DataFrame, lambda_pred: pd.DataFrame,
               targets: Optional[pd.DataFrame] = None, ridge_weight: float = 0.5) -> pd.DataFrame:
    """
    一行代码融合 - 最简接口

    Args:
        ridge_pred: Ridge预测
        lambda_pred: Lambda预测
        targets: 目标值（可选）
        ridge_weight: Ridge权重

    Returns:
        融合结果
    """
    blender = FastDataFrameBlender(ridge_weight)
    return blender.blend_dataframes(ridge_pred, lambda_pred, targets)


# 极简版本 - 直接numpy操作
def ultra_fast_blend(ridge_scores: np.ndarray, lambda_scores: np.ndarray,
                    ridge_weight: float = 0.5) -> np.ndarray:
    """
    最快的融合函数 - 纯numpy
    """
    return ridge_weight * ridge_scores + (1 - ridge_weight) * lambda_scores


def optimize_weight_for_target(ridge_scores: np.ndarray, lambda_scores: np.ndarray,
                              targets: np.ndarray) -> float:
    """
    找到最大化target correlation的权重
    """
    best_weight = 0.5
    best_corr = -1.0

    for w in np.linspace(0.0, 1.0, 21):  # 0.05步长
        blended = w * ridge_scores + (1 - w) * lambda_scores
        corr = np.corrcoef(blended, targets)[0, 1]

        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_weight = w

    return best_weight