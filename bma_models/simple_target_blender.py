#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Target-Oriented Blender - 简单目标导向融合器
直接consolidate预测结果，不做复杂的自适应和去相关
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SimpleTargetBlender:
    """
    简单目标导向融合器

    核心原则：
    1. 不做去相关处理 - 只是consolidate预测
    2. 不算IC等错误指标 - 直接基于目标优化
    3. 简单权重组合 - 可选固定或基于历史表现
    4. 专注于产生稳定的最终预测
    """

    def __init__(self,
                 ridge_weight: float = 0.5,
                 lambda_weight: float = 0.5,
                 use_rank_consolidation: bool = True,
                 use_score_clipping: bool = True,
                 clip_quantile: float = 0.02):
        """
        初始化简单融合器

        Args:
            ridge_weight: Ridge固定权重
            lambda_weight: Lambda固定权重
            use_rank_consolidation: 是否使用排名consolidation
            use_score_clipping: 是否裁剪极值
            clip_quantile: 裁剪分位数
        """
        self.ridge_weight = ridge_weight
        self.lambda_weight = lambda_weight
        self.use_rank_consolidation = use_rank_consolidation
        self.use_score_clipping = use_score_clipping
        self.clip_quantile = clip_quantile

        # 确保权重和为1
        total_weight = self.ridge_weight + self.lambda_weight
        self.ridge_weight = self.ridge_weight / total_weight
        self.lambda_weight = self.lambda_weight / total_weight

        logger.info("🎯 Simple Target Blender 初始化")
        logger.info(f"   权重配置: Ridge={self.ridge_weight:.3f}, Lambda={self.lambda_weight:.3f}")
        logger.info(f"   Rank Consolidation: {self.use_rank_consolidation}")
        logger.info(f"   Score Clipping: {self.use_score_clipping}")

    def blend_predictions(self,
                         ridge_predictions: pd.DataFrame,
                         lambda_predictions: pd.DataFrame,
                         targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        简单融合预测 - 直接consolidate

        Args:
            ridge_predictions: Ridge预测
            lambda_predictions: Lambda预测
            targets: 目标值（可选，用于验证）

        Returns:
            融合后的预测
        """
        logger.info("🔄 开始简单目标导向融合...")

        # 1. 对齐数据
        combined_df = self._align_predictions(ridge_predictions, lambda_predictions)

        # 2. 标准化分数（按日期组内）
        combined_df = self._standardize_scores(combined_df)

        # 3. 简单加权融合
        combined_df['blended_score'] = (
            self.ridge_weight * combined_df['ridge_z'] +
            self.lambda_weight * combined_df['lambda_z']
        )

        # 4. Rank Consolidation（可选）
        if self.use_rank_consolidation:
            combined_df = self._apply_rank_consolidation(combined_df)

        # 5. 裁剪极值（可选）
        if self.use_score_clipping:
            combined_df = self._clip_extreme_scores(combined_df)

        # 6. 计算最终排名
        combined_df = self._calculate_final_ranks(combined_df)

        # 7. 输出统计
        self._log_blend_statistics(combined_df, targets)

        return combined_df

    def _align_predictions(self, ridge_pred: pd.DataFrame, lambda_pred: pd.DataFrame) -> pd.DataFrame:
        """对齐预测数据"""
        # 确保索引一致
        common_index = ridge_pred.index.intersection(lambda_pred.index)

        if len(common_index) == 0:
            raise ValueError("Ridge和Lambda预测没有共同样本")

        # 提取需要的列
        combined_df = pd.DataFrame(index=common_index)
        combined_df['ridge_score'] = ridge_pred.reindex(common_index).get('score',
                                                                          ridge_pred.reindex(common_index).get('score_z', np.nan))
        combined_df['lambda_score'] = lambda_pred.reindex(common_index).get('lambda_score', np.nan)

        # 删除缺失值
        combined_df = combined_df.dropna()

        logger.info(f"   对齐样本数: {len(combined_df)}")

        return combined_df

    def _standardize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期组内标准化"""

        def standardize_group(group):
            """组内z-score标准化"""
            for col in ['ridge_score', 'lambda_score']:
                if col in group.columns:
                    scores = group[col]
                    if len(scores) > 1:
                        mean = scores.mean()
                        std = scores.std()
                        if std > 1e-8:
                            group[col.replace('score', 'z')] = (scores - mean) / std
                        else:
                            group[col.replace('score', 'z')] = 0.0
                    else:
                        group[col.replace('score', 'z')] = 0.0
            return group

        df = df.groupby(level='date', group_keys=False).apply(standardize_group)

        return df

    def _apply_rank_consolidation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank Consolidation - 结合分数和排名信息
        使用排名信息调整融合分数，提高稳定性
        """

        def consolidate_group(group):
            """组内rank consolidation"""
            n = len(group)
            if n <= 1:
                return group

            # 计算原始排名
            ridge_ranks = group['ridge_z'].rank(ascending=False, pct=True)
            lambda_ranks = group['lambda_z'].rank(ascending=False, pct=True)
            blended_ranks = group['blended_score'].rank(ascending=False, pct=True)

            # Rank consolidation: 如果两个模型的排名一致性高，增强信号
            rank_agreement = 1 - abs(ridge_ranks - lambda_ranks)  # 0到1，1表示完全一致

            # 调整融合分数
            group['blended_consolidated'] = group['blended_score'] * (1 + 0.2 * rank_agreement)

            return group

        df = df.groupby(level='date', group_keys=False).apply(consolidate_group)

        # 如果应用了consolidation，使用新分数
        if 'blended_consolidated' in df.columns:
            df['blended_score'] = df['blended_consolidated']
            df = df.drop('blended_consolidated', axis=1)

        return df

    def _clip_extreme_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """裁剪极值，提高稳定性"""

        def clip_group(group):
            """组内裁剪极值"""
            scores = group['blended_score']

            if len(scores) > 10:  # 只在样本足够多时裁剪
                lower = scores.quantile(self.clip_quantile)
                upper = scores.quantile(1 - self.clip_quantile)
                group['blended_score'] = scores.clip(lower, upper)

            return group

        df = df.groupby(level='date', group_keys=False).apply(clip_group)

        return df

    def _calculate_final_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终排名和标准化分数"""

        def rank_group(group):
            """组内排名"""
            # 最终排名
            group['final_rank'] = group['blended_score'].rank(method='average', ascending=False)

            # 最终标准化分数
            scores = group['blended_score']
            if len(scores) > 1:
                mean = scores.mean()
                std = scores.std()
                if std > 1e-8:
                    group['final_z'] = (scores - mean) / std
                else:
                    group['final_z'] = 0.0
            else:
                group['final_z'] = 0.0

            return group

        df = df.groupby(level='date', group_keys=False).apply(rank_group)

        return df

    def _log_blend_statistics(self, df: pd.DataFrame, targets: Optional[pd.DataFrame] = None):
        """输出融合统计"""

        logger.info("✅ 融合完成")
        logger.info(f"   总样本数: {len(df)}")
        logger.info(f"   融合权重: Ridge={self.ridge_weight:.3f}, Lambda={self.lambda_weight:.3f}")

        # 基础统计
        mean_score = df['blended_score'].mean()
        std_score = df['blended_score'].std()
        logger.info(f"   融合分数: mean={mean_score:.6f}, std={std_score:.6f}")

        # 如果有目标值，计算简单的方向一致性
        if targets is not None:
            try:
                common_idx = df.index.intersection(targets.index)
                if len(common_idx) > 100:
                    target_values = targets.reindex(common_idx).iloc[:, 0]
                    blend_values = df.reindex(common_idx)['blended_score']

                    # 方向一致性（同为正或同为负）
                    direction_agreement = (np.sign(target_values) == np.sign(blend_values)).mean()
                    logger.info(f"   预测方向一致率: {direction_agreement:.1%}")

                    # Top-K命中率
                    for k in [10, 50, 100]:
                        if len(common_idx) >= k * 2:
                            # 实际top-k
                            actual_top_k = set(target_values.nlargest(k).index)
                            # 预测top-k
                            pred_top_k = set(blend_values.nlargest(k).index)
                            # 命中率
                            hit_rate = len(actual_top_k.intersection(pred_top_k)) / k
                            logger.info(f"   Top-{k}命中率: {hit_rate:.1%}")
                            break
            except Exception as e:
                logger.debug(f"目标值验证失败: {e}")

    def blend_with_dynamic_weights(self,
                                  ridge_predictions: pd.DataFrame,
                                  lambda_predictions: pd.DataFrame,
                                  targets: pd.DataFrame) -> pd.DataFrame:
        """
        基于目标的动态权重融合
        根据历史预测准确度调整权重
        """
        logger.info("🎯 基于目标的动态权重融合...")

        # 计算简单的历史表现
        ridge_weight, lambda_weight = self._calculate_target_based_weights(
            ridge_predictions, lambda_predictions, targets
        )

        # 更新权重
        self.ridge_weight = ridge_weight
        self.lambda_weight = lambda_weight

        # 执行融合
        return self.blend_predictions(ridge_predictions, lambda_predictions, targets)

    def _calculate_target_based_weights(self,
                                       ridge_pred: pd.DataFrame,
                                       lambda_pred: pd.DataFrame,
                                       targets: pd.DataFrame) -> tuple:
        """
        基于目标计算权重
        使用简单的预测准确度，不用IC
        """
        try:
            # 对齐数据
            common_idx = ridge_pred.index.intersection(lambda_pred.index).intersection(targets.index)

            if len(common_idx) < 100:
                logger.warning("样本不足，使用默认权重")
                return 0.5, 0.5

            ridge_scores = ridge_pred.reindex(common_idx).iloc[:, 0]
            lambda_scores = lambda_pred.reindex(common_idx).iloc[:, 0]
            target_values = targets.reindex(common_idx).iloc[:, 0]

            # 计算简单的预测准确度（方向一致性）
            ridge_accuracy = (np.sign(ridge_scores) == np.sign(target_values)).mean()
            lambda_accuracy = (np.sign(lambda_scores) == np.sign(target_values)).mean()

            # 基于准确度分配权重
            total_accuracy = ridge_accuracy + lambda_accuracy + 1e-8
            ridge_weight = ridge_accuracy / total_accuracy
            lambda_weight = lambda_accuracy / total_accuracy

            # 限制权重范围
            ridge_weight = np.clip(ridge_weight, 0.3, 0.7)
            lambda_weight = 1 - ridge_weight

            logger.info(f"   Ridge准确度: {ridge_accuracy:.1%}")
            logger.info(f"   Lambda准确度: {lambda_accuracy:.1%}")
            logger.info(f"   动态权重: Ridge={ridge_weight:.3f}, Lambda={lambda_weight:.3f}")

            return ridge_weight, lambda_weight

        except Exception as e:
            logger.warning(f"权重计算失败，使用默认: {e}")
            return 0.5, 0.5

    def get_info(self) -> Dict[str, Any]:
        """获取融合器信息"""
        return {
            'type': 'SimpleTargetBlender',
            'ridge_weight': self.ridge_weight,
            'lambda_weight': self.lambda_weight,
            'use_rank_consolidation': self.use_rank_consolidation,
            'use_score_clipping': self.use_score_clipping,
            'clip_quantile': self.clip_quantile
        }