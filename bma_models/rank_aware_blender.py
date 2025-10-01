#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank-aware Blending - 智能融合Ridge和LambdaRank预测

核心思路：
- Ridge回归：连续预测，保留刻度信息
- LambdaRank：排序优化，提升Top-K性能
- 自适应权重：基于历史RankIC和NDCG动态调整

融合策略：
1. 按交易日组内标准化
2. 历史60d窗口计算性能指标
3. 自适应权重：wR ∝ RankIC@K, wL ∝ NDCG@K
4. Copula正态化增强鲁棒性
5. 融合分数：s* = wR·zR + wL·zL
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from scipy import stats
from scipy.special import ndtr, ndtri  # 标准正态分布CDF和逆CDF
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RankGateConfig:
    """
    门控增益配置类

    实现LTR专注排名门控，Ridge专注幅度刻度的分离设计
    """

    def __init__(self,
                 tau_long: float = 0.7,          # 长准入阈值
                 tau_short: float = 0.3,         # 短准入阈值
                 alpha_long: float = 0.2,        # 长侧增益系数
                 alpha_short: float = 0.2,       # 短侧增益系数
                 min_coverage: float = 0.3,      # 最小覆盖率兜底
                 neutral_band: bool = True,      # 是否启用中性带置零
                 top_k_list: list = None,        # Top-K监控列表
                 ewma_beta: float = 0.1,         # EWMA平滑系数
                 max_gain: float = 1.3):         # 最大增益上限

        self.tau_long = tau_long
        self.tau_short = tau_short
        self.alpha_long = alpha_long
        self.alpha_short = alpha_short
        self.min_coverage = min_coverage
        self.neutral_band = neutral_band
        # 优化K值设置：适应2600只股票的投资宇宙
        # 分层策略：精选(5,10,20) + 投资组合(50,100) + 风险分散(200)
        self.top_k_list = top_k_list or [5, 10, 20, 50, 100, 200]
        self.ewma_beta = ewma_beta
        self.max_gain = max_gain

        # 运行时状态
        self.coverage_history = []
        self.gain_stats_history = []

        logger.info(f"🚪 门控配置初始化: 长准入≥{tau_long}, 短准入≤{tau_short}")
        logger.info(f"   增益系数: α_long={alpha_long}, α_short={alpha_short}")
        logger.info(f"   最小覆盖: {min_coverage}, 中性带: {neutral_band}")

class RankAwareBlender:
    """
    Rank-aware智能融合器

    核心优势：
    - 自适应权重，根据历史表现动态调整
    - Copula正态化，对重尾分布更鲁棒
    - 按交易日组内处理，符合实际交易场景
    - 平滑权重变化，避免极端漂移
    """

    def __init__(self,
                 lookback_window: int = 60,  # 历史窗口天数
                 min_weight: float = 0.3,   # 最小权重（防极端）
                 max_weight: float = 0.7,   # 最大权重（防极端）
                 weight_smoothing: float = 0.3,  # 权重平滑系数
                 use_copula: bool = True,    # 是否使用Copula正态化
                 use_decorrelation: bool = True,  # 是否使用去相关融合
                 top_k_list: list = None):   # Top-K评估列表
        """
        初始化Rank-aware Blender

        Args:
            lookback_window: 历史性能计算窗口
            min_weight: LambdaRank最小权重
            max_weight: LambdaRank最大权重
            weight_smoothing: 权重EWMA平滑系数
            use_copula: 是否使用Copula正态化
            top_k_list: Top-K评估指标
        """
        self.lookback_window = lookback_window
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_smoothing = weight_smoothing
        self.use_copula = use_copula
        self.use_decorrelation = use_decorrelation

        if top_k_list is None:
            # 优化K值设置：适应2600只股票的投资宇宙
            # 分层策略：精选(5,10,20) + 投资组合(50,100) + 风险分散(200)
            self.top_k_list = [5, 10, 20, 50, 100, 200]
        else:
            self.top_k_list = top_k_list

        # 历史权重记录
        self.weight_history = []
        self.current_lambda_weight = 0.5  # 初始权重

        # 启用高级特性
        self.enable_advanced_blending = True  # 启用高级融合
        self.enable_insightful_metrics = True  # 启用深度指标

        # 门控+残差微融合参数（2600股票建议参数）
        self.tau_long = 0.65
        self.tau_short = 0.35
        self.alpha_long = 0.15
        self.alpha_short = 0.15
        self.max_gain = 1.25
        self.min_coverage = 0.30

        # 残差微融合参数管理
        self.current_beta = 0.08  # β初始值
        self.beta_range = [0.0, 0.15]  # β取值范围
        self.beta_ewma_alpha = 0.3  # EWMA平滑系数
        self.beta_history = []  # β历史记录

        # 性能监控
        self._recent_performance_improved = True  # 性能改善标志
        self._flip_ratio_history = []  # 方向翻转历史
        self._coverage_history = []  # 覆盖率历史
        self._ndcg_history = []  # NDCG历史

        logger.info("🤝 智能Rank-aware Blender V2.0 初始化完成")
        logger.info(f"   历史窗口: {self.lookback_window}天")
        logger.info(f"   权重范围: [{self.min_weight}, {self.max_weight}]")
        logger.info(f"   Top-K评估: {self.top_k_list}")
        logger.info(f"   特性配置:")
        logger.info(f"     - Copula正态化: {self.use_copula}")
        logger.info(f"     - 智能去相关: {self.use_decorrelation}")
        logger.info(f"     - 高级融合: {self.enable_advanced_blending}")
        logger.info(f"     - 深度指标: {self.enable_insightful_metrics}")
        logger.info(f"   门控参数: 长侧≥{self.tau_long}, 短侧≤{self.tau_short}, 增益上限{self.max_gain}")

    def blend_predictions(self,
                         ridge_predictions: pd.DataFrame,
                         lambda_predictions: pd.DataFrame,
                         targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        融合Ridge和LambdaRank预测

        Args:
            ridge_predictions: Ridge预测结果，包含'score'列
            lambda_predictions: LambdaRank预测结果，包含'lambda_score'列
            targets: 历史目标变量（用于计算权重），可选

        Returns:
            融合后的预测结果
        """
        logger.info("🔄 开始智能Rank-aware融合...")
        logger.info(f"   融合模式: {'Advanced' if self.enable_advanced_blending else 'Standard'}")

        # 验证输入
        if not isinstance(ridge_predictions.index, pd.MultiIndex):
            raise ValueError("预测数据必须有MultiIndex(date, ticker)")

        # 对齐两个预测结果
        ridge_aligned = ridge_predictions.reindex(ridge_predictions.index)
        lambda_aligned = lambda_predictions.reindex(ridge_predictions.index)

        # 合并数据
        combined_df = pd.DataFrame(index=ridge_predictions.index)
        # 处理Ridge预测的多列输出 - 只提取'score'列（安全处理Series和DataFrame）
        if hasattr(ridge_aligned, 'columns') and 'score' in ridge_aligned.columns:
            combined_df['ridge_score'] = ridge_aligned['score']
        elif hasattr(ridge_aligned, 'columns') and 'score_z' in ridge_aligned.columns:
            combined_df['ridge_score'] = ridge_aligned['score_z']
        else:
            # 如果ridge_aligned本身是Series或单列DataFrame，直接使用
            if isinstance(ridge_aligned, pd.Series):
                combined_df['ridge_score'] = ridge_aligned
            elif len(ridge_aligned.columns) == 1:
                combined_df['ridge_score'] = ridge_aligned.iloc[:, 0]
            else:
                combined_df['ridge_score'] = ridge_aligned.get('score', np.nan)
        # 安全处理lambda_aligned的lambda_score列
        if isinstance(lambda_aligned, pd.Series):
            combined_df['lambda_score'] = lambda_aligned
        elif hasattr(lambda_aligned, 'columns') and 'lambda_score' in lambda_aligned.columns:
            combined_df['lambda_score'] = lambda_aligned['lambda_score']
        else:
            combined_df['lambda_score'] = lambda_aligned.get('lambda_score', np.nan) if hasattr(lambda_aligned, 'get') else np.nan

        # 删除任一模型缺失的样本
        valid_mask = combined_df['ridge_score'].notna() & combined_df['lambda_score'].notna()
        total_samples = len(combined_df)
        ridge_valid = combined_df['ridge_score'].notna().sum()
        lambda_valid = combined_df['lambda_score'].notna().sum()
        both_valid = valid_mask.sum()

        logger.info(f"   预测样本统计: 总数={total_samples}, Ridge有效={ridge_valid}, Lambda有效={lambda_valid}, 双方有效={both_valid}")

        combined_df = combined_df[valid_mask]

        if len(combined_df) == 0:
            # 🔧 FIX: 优雅处理单模型情况 - 使用有效的单一模型
            if ridge_valid > 0 and lambda_valid == 0:
                logger.warning("LambdaRank预测全为NaN，退化为纯Ridge预测")
                result_df = pd.DataFrame(index=ridge_predictions.index)
                if 'score' in ridge_predictions.columns:
                    result_df['blended_score'] = ridge_predictions['score']
                else:
                    result_df['blended_score'] = ridge_predictions.iloc[:, 0]
                result_df['blended_rank'] = result_df['blended_score'].rank(ascending=False)
                # 防止除以0：std() 返回标量，使用max进行下限裁剪
                result_df['blended_z'] = (result_df['blended_score'] - result_df['blended_score'].mean()) / max(result_df['blended_score'].std(), 1e-8)
                return result_df
            elif lambda_valid > 0 and ridge_valid == 0:
                logger.warning("Ridge预测全为NaN，退化为纯LambdaRank预测")
                result_df = pd.DataFrame(index=lambda_predictions.index)
                if 'lambda_score' in lambda_predictions.columns:
                    result_df['blended_score'] = lambda_predictions['lambda_score']
                else:
                    result_df['blended_score'] = lambda_predictions.iloc[:, 0]
                result_df['blended_rank'] = result_df['blended_score'].rank(ascending=False)
                # 防止除以0：std() 返回标量，使用max进行下限裁剪
                result_df['blended_z'] = (result_df['blended_score'] - result_df['blended_score'].mean()) / max(result_df['blended_score'].std(), 1e-8)
                return result_df
            else:
                error_msg = f"两个模型预测都无效 (总数={total_samples}, Ridge有效={ridge_valid}, Lambda有效={lambda_valid})"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(f"   有效样本: {len(combined_df)}")

        # 计算自适应权重（如果有历史目标数据）
        if targets is not None:
            lambda_weight = self._calculate_adaptive_weight(combined_df, targets)
        else:
            lambda_weight = self.current_lambda_weight
            logger.info(f"   使用当前权重: λ={lambda_weight:.3f}")

        # 更新当前权重
        self.current_lambda_weight = lambda_weight
        ridge_weight = 1.0 - lambda_weight

        # 计算原始信号统计
        original_corr = combined_df['ridge_score'].corr(combined_df['lambda_score'])
        logger.info(f"   原始信号相关性: {original_corr:.4f}")

        # 按交易日组内标准化
        if self.use_copula:
            # Copula正态化
            combined_df = self._apply_copula_normalization(combined_df)
            ridge_col, lambda_col = 'ridge_norm', 'lambda_norm'
        else:
            # 普通z-score标准化
            combined_df = self._apply_zscore_normalization(combined_df)
            ridge_col, lambda_col = 'ridge_z', 'lambda_z'

        # 去相关处理（保留有价值的相关性）
        if self.use_decorrelation:
            pre_decorr_corr = combined_df[ridge_col].corr(combined_df[lambda_col])
            combined_df = self._apply_decorrelation(combined_df, ridge_col, lambda_col)
            lambda_col = lambda_col + '_ortho'  # 使用正交化后的LambdaRank信号
            post_decorr_corr = combined_df[ridge_col].corr(combined_df[lambda_col])
            logger.info(f"   标准化后相关性: {pre_decorr_corr:.4f} → {post_decorr_corr:.4f}")

        # 智能融合分数
        if hasattr(self, 'enable_advanced_blending') and self.enable_advanced_blending:
            # 高级融合：动态权重 + 门控
            blended_scores = self._apply_advanced_blending(
                combined_df, ridge_col, lambda_col, ridge_weight, lambda_weight
            )
            # 确保赋值正确 - 处理 Series/DataFrame 返回值
            if isinstance(blended_scores, pd.DataFrame):
                # 取第一列作为blended分数，并发出提示
                try:
                    first_col = blended_scores.columns[0]
                    combined_df['blended_score'] = blended_scores[first_col].reindex(combined_df.index)
                    logger.warning(f"高级融合返回多列，使用第一列 '{first_col}' 作为blended_score")
                except Exception:
                    # 回退：按行均值
                    combined_df['blended_score'] = blended_scores.mean(axis=1).reindex(combined_df.index)
                    logger.warning("高级融合返回多列，使用行均值作为blended_score")
            elif isinstance(blended_scores, pd.Series):
                combined_df['blended_score'] = blended_scores.reindex(combined_df.index)
            else:
                # 非pandas对象，尝试转换为Series
                combined_df['blended_score'] = pd.Series(blended_scores, index=combined_df.index)
        else:
            # 标准门控融合
            blended_scores = self._apply_gated_blending(
                combined_df, ridge_col, lambda_col, ridge_weight, lambda_weight
            )
            # 确保赋值正确 - 处理 Series/DataFrame 返回值
            if isinstance(blended_scores, pd.DataFrame):
                try:
                    first_col = blended_scores.columns[0]
                    combined_df['blended_score'] = blended_scores[first_col].reindex(combined_df.index)
                    logger.warning(f"标准门控融合返回多列，使用第一列 '{first_col}' 作为blended_score")
                except Exception:
                    combined_df['blended_score'] = blended_scores.mean(axis=1).reindex(combined_df.index)
                    logger.warning("标准门控融合返回多列，使用行均值作为blended_score")
            elif isinstance(blended_scores, pd.Series):
                combined_df['blended_score'] = blended_scores.reindex(combined_df.index)
            else:
                combined_df['blended_score'] = pd.Series(blended_scores, index=combined_df.index)

        # 计算最终排名
        def _rank_by_date(group):
            scores = group['blended_score']
            return scores.rank(method='average', ascending=False)

        combined_df['blended_rank'] = combined_df.groupby(level='date').apply(_rank_by_date).values

        # 仅对最终融合阶段做稳健后处理（不改上游）：
        # 1) 日内winsorize(1%-99%) 2) tanh压缩 3) 日内去均值与定尺

        def _postprocess_final_by_date(group):
            s = group['blended_score'].astype(float)
            if len(s) <= 1:
                group['blended_score_pp'] = s
                group['blended_z'] = 0.0
                return group
            # winsorize
            lo, hi = np.percentile(s, [1, 99])
            s_w = s.clip(lower=lo, upper=hi)
            # tanh 压缩到 [-1,1]
            s_c = np.tanh(s_w / 2.0)
            # 日内去均值与定尺（目标std≈1）
            std = s_c.std()
            if std < 1e-8:
                z = s_c * 0.0
            else:
                z = (s_c - s_c.mean()) / std
            group['blended_score_pp'] = s_c
            group['blended_z'] = z
            return group

        combined_df = combined_df.groupby(level='date', group_keys=False).apply(_postprocess_final_by_date)

        # 记录权重历史
        self.weight_history.append({
            'date': combined_df.index.get_level_values('date').max(),
            'ridge_weight': ridge_weight,
            'lambda_weight': lambda_weight
        })

        # 融合统计
        blend_stats = self._calculate_blend_statistics(combined_df)

        # 输出深度性能洞察
        self._log_performance_insights(combined_df, ridge_weight, lambda_weight)

        logger.info(f"✅ 融合完成: Ridge权重={ridge_weight:.3f}, Lambda权重={lambda_weight:.3f}")
        logger.info(f"   融合样本: {len(combined_df)}")
        logger.info(f"   融合统计: mean={blend_stats['mean']:.6f}, std={blend_stats['std']:.6f}")
        logger.info(f"   信号对比: Ridge与Lambda正相关率={blend_stats['agreement_rate']:.1%}")

        # 对外仍暴露 blended_score（兼容），并提供压缩后的 blended_score_pp
        return combined_df[['ridge_score', 'lambda_score', 'blended_score', 'blended_score_pp', 'blended_rank', 'blended_z']]

    def _log_performance_insights(self, df: pd.DataFrame, ridge_weight: float, lambda_weight: float):
        """输出深度性能洞察"""
        if not self.enable_insightful_metrics:
            return

        logger.info("📊 深度性能分析:")

        # 权重分析
        weight_ratio = lambda_weight / (ridge_weight + 1e-8)
        if weight_ratio > 1.5:
            insight = "LambdaRank主导 (排序优先)"
        elif weight_ratio < 0.67:
            insight = "Ridge主导 (幅度优先)"
        else:
            insight = "均衡融合 (协同优化)"

        logger.info(f"   权重策略: {insight}")
        logger.info(f"   权重比: {weight_ratio:.2f}")

        # 信号分析
        if 'ridge_score' in df.columns and 'lambda_score' in df.columns:
            correlation = df['ridge_score'].corr(df['lambda_score'])
            if correlation > 0.7:
                signal_insight = "高度一致 (增强信心)"
            elif correlation > 0.3:
                signal_insight = "中度一致 (互补增益)"
            elif correlation > 0:
                signal_insight = "低度一致 (分散风险)"
            else:
                signal_insight = "负相关 (对冲信号)"

            logger.info(f"   信号关系: {signal_insight}")
            logger.info(f"   相关系数: {correlation:.3f}")

        # 风险分析
        if 'blended_score' in df.columns:
            vol = df['blended_score'].std()
            skew = df['blended_score'].skew()
            if abs(skew) < 0.5:
                risk_insight = "对称分布 (风险均衡)"
            elif skew > 0.5:
                risk_insight = "右偏分布 (上行倾向)"
            else:
                risk_insight = "左偏分布 (下行风险)"

            logger.info(f"   分布特征: {risk_insight}")
            logger.info(f"   波动率: {vol:.3f}, 偏度: {skew:.3f}")

        logger.info("🎯 融合效果评估完成")

    def blend_with_gate(self,
                       ridge_predictions: pd.DataFrame,
                       lambda_predictions: pd.DataFrame,
                       targets: Optional[pd.DataFrame] = None,
                       cfg: Optional[RankGateConfig] = None) -> pd.DataFrame:
        """
        门控增益融合 - LTR专注排名门控，Ridge专注幅度刻度

        核心策略：
        - 只用LTR的当日组内百分位lambda_pct做准入/分档，绝不与Ridge分数线性加权
        - 最终信号只来自Ridge幅度，经LTR的"门控+增益"调制
        - score_final = score_ridge_z × gain(lambda_pct) × gate(lambda_pct)

        Args:
            ridge_predictions: Ridge预测结果，包含'score'列
            lambda_predictions: LambdaRank预测结果，包含'lambda_score'或'lambda_pct'列
            targets: 历史目标变量（用于监控），可选
            cfg: 门控配置，使用默认配置如果为None

        Returns:
            门控融合后的预测结果
        """
        if cfg is None:
            cfg = RankGateConfig()

        logger.info("🚪 开始门控增益融合...")
        logger.info(f"   门控阈值: 长≥{cfg.tau_long}, 短≤{cfg.tau_short}")
        logger.info(f"   增益系数: α_long={cfg.alpha_long}, α_short={cfg.alpha_short}")

        # 验证输入
        if not isinstance(ridge_predictions.index, pd.MultiIndex):
            raise ValueError("预测数据必须有MultiIndex(date, ticker)")

        # 对齐两个预测结果
        combined_df = pd.DataFrame(index=ridge_predictions.index)
        # 处理Ridge预测的多列输出 - 只提取'score'列（安全处理Series和DataFrame）
        if hasattr(ridge_predictions, 'columns') and 'score' in ridge_predictions.columns:
            combined_df['ridge_score'] = ridge_predictions['score']
        elif hasattr(ridge_predictions, 'columns') and 'score_z' in ridge_predictions.columns:
            combined_df['ridge_score'] = ridge_predictions['score_z']
        else:
            # 如果ridge_predictions本身是Series或单列DataFrame，直接使用
            if isinstance(ridge_predictions, pd.Series):
                combined_df['ridge_score'] = ridge_predictions
            elif len(ridge_predictions.columns) == 1:
                combined_df['ridge_score'] = ridge_predictions.iloc[:, 0]
            else:
                combined_df['ridge_score'] = ridge_predictions.get('score', np.nan)

        # 获取LambdaRank百分位（已由LambdaRankStacker.predict产出）
        if hasattr(lambda_predictions, 'columns') and 'lambda_pct' in lambda_predictions.columns:
            combined_df['lambda_pct'] = lambda_predictions['lambda_pct']
        elif hasattr(lambda_predictions, 'columns') and 'lambda_score' in lambda_predictions.columns:
            # 如果只有lambda_score，需要计算当日组内百分位
            logger.info("   从lambda_score计算组内百分位...")
            combined_df['lambda_score'] = lambda_predictions['lambda_score']
            combined_df = self._calculate_daily_percentiles(combined_df)
        elif isinstance(lambda_predictions, pd.Series):
            # 如果lambda_predictions是Series，将其作为lambda_score处理
            combined_df['lambda_score'] = lambda_predictions
            combined_df = self._calculate_daily_percentiles(combined_df)
        else:
            raise ValueError("LambdaRank预测必须包含'lambda_pct'或'lambda_score'列，或者是包含预测值的Series")

        # 删除任一模型缺失的样本
        valid_mask = combined_df['ridge_score'].notna() & combined_df['lambda_pct'].notna()
        total_samples = len(combined_df)
        ridge_valid = combined_df['ridge_score'].notna().sum()
        lambda_valid = combined_df['lambda_pct'].notna().sum()
        both_valid = valid_mask.sum()

        logger.info(f"   预测样本统计: 总数={total_samples}, Ridge有效={ridge_valid}, Lambda有效={lambda_valid}, 双方有效={both_valid}")

        combined_df = combined_df[valid_mask]

        if len(combined_df) == 0:
            # 🔧 FIX: 优雅处理单模型情况 - 使用有效的单一模型
            if ridge_valid > 0 and lambda_valid == 0:
                logger.warning("LambdaRank预测全为NaN，门控退化为纯Ridge预测")
                result_df = pd.DataFrame(index=ridge_predictions.index)
                if 'score' in ridge_predictions.columns:
                    result_df['blended_score'] = ridge_predictions['score']
                else:
                    result_df['blended_score'] = ridge_predictions.iloc[:, 0]
                result_df['blended_rank'] = result_df['blended_score'].rank(ascending=False)
                # 防止除以0：std() 返回标量，使用max进行下限裁剪
                result_df['blended_z'] = (result_df['blended_score'] - result_df['blended_score'].mean()) / max(result_df['blended_score'].std(), 1e-8)
                result_df['gate'] = 1.0  # 全部通过门控
                result_df['gain'] = 1.0  # 无增益
                return result_df
            elif lambda_valid > 0 and ridge_valid == 0:
                logger.warning("Ridge预测全为NaN，门控退化为纯LambdaRank预测")
                result_df = pd.DataFrame(index=lambda_predictions.index)
                if 'lambda_score' in lambda_predictions.columns:
                    result_df['blended_score'] = lambda_predictions['lambda_score']
                elif 'lambda_pct' in lambda_predictions.columns:
                    # 将百分位转换为分数
                    result_df['blended_score'] = lambda_predictions['lambda_pct'] - 0.5
                else:
                    result_df['blended_score'] = lambda_predictions.iloc[:, 0]
                result_df['blended_rank'] = result_df['blended_score'].rank(ascending=False)
                # 防止除以0：std() 返回标量，使用max进行下限裁剪
                result_df['blended_z'] = (result_df['blended_score'] - result_df['blended_score'].mean()) / max(result_df['blended_score'].std(), 1e-8)
                result_df['gate'] = 1.0  # 全部通过门控
                result_df['gain'] = 1.0  # 无增益
                return result_df
            else:
                error_msg = f"两个模型预测都无效 (总数={total_samples}, Ridge有效={ridge_valid}, Lambda有效={lambda_valid})"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info(f"   有效样本: {len(combined_df)}")

        # 按交易日组内标准化Ridge分数
        combined_df = self._standardize_ridge_scores(combined_df)

        # 计算门控与增益
        combined_df = self._apply_rank_gate_and_gain(combined_df, cfg)

        # 最终门控融合: score_final = ridge_z × gain × gate
        combined_df['gated_score'] = (
            combined_df['ridge_z'] *
            combined_df['gain'] *
            combined_df['gate']
        )

        # 计算最终排名和标准化分数
        combined_df = self._finalize_gated_results(combined_df)

        # 监控统计
        self._log_gate_monitoring(combined_df, cfg)

        # 覆盖率兜底检查
        coverage = self._check_coverage_fallback(combined_df, cfg)

        logger.info(f"✅ 门控融合完成: 覆盖率={coverage:.1%}")

        # 🔧 FIX: 为了API一致性，添加blended_score列作为gated_score的别名
        combined_df['blended_score'] = combined_df['gated_score']
        combined_df['blended_rank'] = combined_df['gated_rank']
        combined_df['blended_z'] = combined_df['gated_z']

        return combined_df[['ridge_score', 'ridge_z', 'lambda_pct', 'gate', 'gain',
                           'gated_score', 'gated_rank', 'gated_z',
                           'blended_score', 'blended_rank', 'blended_z']]

    def _calculate_daily_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日组内计算LambdaRank百分位"""

        def _pct_by_date(group):
            lambda_scores = group['lambda_score']
            # 使用rank方法计算百分位（0-1范围）
            percentiles = lambda_scores.rank(method='average') / len(lambda_scores)
            group['lambda_pct'] = percentiles
            return group

        return df.groupby(level='date', group_keys=False).apply(_pct_by_date)

    def _standardize_ridge_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日组内标准化Ridge分数"""

        def _zscore_ridge_by_date(group):
            ridge_scores = group['ridge_score']
            if len(ridge_scores) > 1:
                z_scores = (ridge_scores - ridge_scores.mean()) / (ridge_scores.std() + 1e-8)
                group['ridge_z'] = z_scores
            else:
                group['ridge_z'] = 0.0
            return group

        return df.groupby(level='date', group_keys=False).apply(_zscore_ridge_by_date)

    def _apply_rank_gate_and_gain(self, df: pd.DataFrame, cfg: RankGateConfig) -> pd.DataFrame:
        """应用排名门控与分档增益"""

        # 初始化门控和增益
        df['gate'] = 0.0  # 默认不通过门控
        df['gain'] = 1.0  # 默认无增益

        # 长侧门控与增益
        long_mask = df['lambda_pct'] >= cfg.tau_long
        if long_mask.any():
            # 长侧门控通过
            df.loc[long_mask, 'gate'] = 1.0
            # 长侧分档增益: gain = 1 + α_long × ((lambda_pct - τ_long)/(1-τ_long))
            long_gain_factor = (df.loc[long_mask, 'lambda_pct'] - cfg.tau_long) / (1 - cfg.tau_long)
            df.loc[long_mask, 'gain'] = 1.0 + cfg.alpha_long * long_gain_factor
            df.loc[long_mask, 'gain'] = np.clip(df.loc[long_mask, 'gain'], 1.0, cfg.max_gain)

        # 短侧门控与增益
        short_mask = df['lambda_pct'] <= cfg.tau_short
        if short_mask.any():
            # 短侧门控通过
            df.loc[short_mask, 'gate'] = 1.0
            # 短侧分档增益: gain = 1 + α_short × ((τ_short - lambda_pct)/τ_short)
            short_gain_factor = (cfg.tau_short - df.loc[short_mask, 'lambda_pct']) / cfg.tau_short
            df.loc[short_mask, 'gain'] = 1.0 + cfg.alpha_short * short_gain_factor
            df.loc[short_mask, 'gain'] = np.clip(df.loc[short_mask, 'gain'], 1.0, cfg.max_gain)

        # 中性带处理（可选置零）
        if cfg.neutral_band:
            neutral_mask = (df['lambda_pct'] > cfg.tau_short) & (df['lambda_pct'] < cfg.tau_long)
            # 中性带既不通过门控，也无增益（已在初始化时设置）
            pass

        return df

    def _finalize_gated_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终排名和标准化分数"""

        # 按日计算最终排名
        def _rank_by_date(group):
            gated_scores = group['gated_score']
            # 排名：分数越高排名越靠前
            group['gated_rank'] = gated_scores.rank(method='average', ascending=False)
            return group

        df = df.groupby(level='date', group_keys=False).apply(_rank_by_date)

        # 按日标准化最终分数
        def _zscore_final_by_date(group):
            gated_scores = group['gated_score']
            if len(gated_scores) > 1 and gated_scores.std() > 1e-8:
                z_scores = (gated_scores - gated_scores.mean()) / gated_scores.std()
                group['gated_z'] = z_scores
            else:
                group['gated_z'] = 0.0
            return group

        df = df.groupby(level='date', group_keys=False).apply(_zscore_final_by_date)

        return df

    def _log_gate_monitoring(self, df: pd.DataFrame, cfg: RankGateConfig):
        """记录门控监控统计"""

        total_samples = len(df)
        if total_samples == 0:
            return

        # 覆盖率统计
        gated_samples = (df['gate'] > 0).sum()
        coverage = gated_samples / total_samples

        # 长短侧分布
        long_samples = (df['lambda_pct'] >= cfg.tau_long).sum()
        short_samples = (df['lambda_pct'] <= cfg.tau_short).sum()
        neutral_samples = total_samples - long_samples - short_samples

        # 增益统计
        gain_mean = df['gain'].mean()
        gain_max = df['gain'].max()
        gain_top_rate = (df['gain'] >= cfg.max_gain * 0.95).sum() / total_samples

        logger.info(f"📊 门控统计:")
        logger.info(f"   覆盖率: {coverage:.1%} ({gated_samples}/{total_samples})")
        logger.info(f"   分布: 长侧{long_samples}, 短侧{short_samples}, 中性{neutral_samples}")
        logger.info(f"   增益: 均值{gain_mean:.3f}, 最大{gain_max:.3f}, 触顶率{gain_top_rate:.1%}")

        # 更新历史记录
        cfg.coverage_history.append(coverage)
        cfg.gain_stats_history.append({
            'mean': gain_mean,
            'max': gain_max,
            'top_rate': gain_top_rate
        })

    def _check_coverage_fallback(self, df: pd.DataFrame, cfg: RankGateConfig) -> float:
        """检查覆盖率并在必要时回退到Ridge"""

        coverage = (df['gate'] > 0).sum() / len(df) if len(df) > 0 else 0.0

        if coverage < cfg.min_coverage:
            logger.warning(f"⚠️ 覆盖率过低 ({coverage:.1%} < {cfg.min_coverage:.1%})，回退到Ridge分数")
            # 回退策略：使用Ridge分数，无门控增益
            df['gated_score'] = df['ridge_z']
            df['gate'] = 1.0  # 全部通过
            df['gain'] = 1.0  # 无增益
            # 重新计算排名和标准化
            df = self._finalize_gated_results(df)
            coverage = 1.0

        return coverage

    def _calculate_adaptive_weight(self, combined_df: pd.DataFrame, targets: pd.DataFrame) -> float:
        """
        基于多维度性能计算智能自适应权重

        核心改进：
        1. 使用多个性能指标综合评估
        2. 动态调整权重对比基准
        3. 考虑近期 vs 远期性能趋势
        4. 加入风险调整因子

        Args:
            combined_df: 当前预测结果
            targets: 历史目标变量

        Returns:
            LambdaRank权重
        """
        logger.info("📊 计算智能自适应权重...")

        target_aligned = targets.reindex(combined_df.index)
        ridge_metrics = {'ic': [], 'rankic': [], 'top_return': [], 'volatility': []}
        lambda_metrics = {'ic': [], 'rankic': [], 'ndcg': [], 'precision': []}

        # 多尺度评估参数
        eval_k_list = self.top_k_list[:2] if len(self.top_k_list) > 1 else [5, 10]
        recent_window = min(10, self.lookback_window // 3)  # 近期窗口

        try:
            # 合并数据用于评估
            target_values = target_aligned.iloc[:, 0] if len(target_aligned.columns) > 0 else target_aligned
            eval_df = pd.DataFrame({
                'ridge_score': combined_df['ridge_score'],
                'lambda_score': combined_df['lambda_score'],
                'target': target_values
            }, index=combined_df.index).dropna()

            if len(eval_df) == 0:
                logger.warning("评估数据为空，使用当前权重")
                return self.current_lambda_weight

            # 按日期分组计算性能指标
            def _calculate_group_rankic(group_data, score_col):
                """计算组内RankIC"""
                min_samples = eval_k_list[0] if eval_k_list else 5  # 使用最小的k值作为最小样本数
                if len(group_data) < min_samples:
                    return 0.0
                try:
                    score_ranks = group_data[score_col].rank(ascending=False)
                    target_ranks = group_data['target'].rank(ascending=False)
                    correlation = score_ranks.corr(target_ranks, method='spearman')
                    return correlation if not np.isnan(correlation) else 0.0
                except:
                    return 0.0

            def _calculate_group_ndcg_at_k(group_data, score_col, k):
                """计算组内NDCG@K"""
                if len(group_data) < k:
                    return 0.0
                try:
                    # 按分数降序排序，取Top-K
                    sorted_group = group_data.sort_values(score_col, ascending=False).head(k)

                    # 简化NDCG计算：使用目标值作为相关性
                    relevance = sorted_group['target'].values

                    # DCG计算
                    dcg = 0.0
                    for i, rel in enumerate(relevance):
                        dcg += rel / np.log2(i + 2)  # i+2 because log2(1)=0

                    # IDCG计算（理想排序）
                    ideal_relevance = sorted(group_data['target'].values, reverse=True)[:k]
                    idcg = 0.0
                    for i, rel in enumerate(ideal_relevance):
                        idcg += rel / np.log2(i + 2)

                    # NDCG计算
                    if idcg == 0:
                        return 0.0
                    return dcg / idcg
                except:
                    return 0.0

            # 按日期分组评估
            date_list = eval_df.index.get_level_values('date').unique()
            for idx, date in enumerate(date_list):
                group_data = eval_df.loc[date]
                is_recent = idx >= len(date_list) - recent_window
                weight_mult = 1.5 if is_recent else 1.0  # 近期数据更重要

                # Ridge多维度评估
                ridge_ic = group_data['ridge_score'].corr(group_data['target'])
                ridge_rankic = _calculate_group_rankic(group_data, 'ridge_score')

                # 计算Top-K平均收益
                if len(group_data) >= eval_k_list[0]:
                    top_k_indices = group_data['ridge_score'].nlargest(eval_k_list[0]).index
                    ridge_top_return = group_data.loc[top_k_indices, 'target'].mean()
                    ridge_volatility = group_data.loc[top_k_indices, 'target'].std()
                else:
                    ridge_top_return = 0.0
                    ridge_volatility = 1.0

                ridge_metrics['ic'].append(ridge_ic * weight_mult if not np.isnan(ridge_ic) else 0)
                ridge_metrics['rankic'].append(max(0.0, ridge_rankic) * weight_mult)
                ridge_metrics['top_return'].append(ridge_top_return * weight_mult)
                ridge_metrics['volatility'].append(ridge_volatility)

                # Lambda多维度评估
                lambda_ic = group_data['lambda_score'].corr(group_data['target'])
                lambda_rankic = _calculate_group_rankic(group_data, 'lambda_score')

                # NDCG和Precision@K
                for k in eval_k_list:
                    if len(group_data) >= k:
                        lambda_ndcg = _calculate_group_ndcg_at_k(group_data, 'lambda_score', k)
                        # Precision@K: Top-K中正收益的比例
                        top_k_indices = group_data['lambda_score'].nlargest(k).index
                        lambda_precision = (group_data.loc[top_k_indices, 'target'] > 0).mean()
                        break
                else:
                    lambda_ndcg = 0.0
                    lambda_precision = 0.5

                lambda_metrics['ic'].append(lambda_ic * weight_mult if not np.isnan(lambda_ic) else 0)
                lambda_metrics['rankic'].append(max(0.0, lambda_rankic) * weight_mult)
                lambda_metrics['ndcg'].append(lambda_ndcg * weight_mult)
                lambda_metrics['precision'].append(lambda_precision * weight_mult)

        except Exception as e:
            logger.warning(f"权重计算失败，使用默认权重: {e}")
            return self.current_lambda_weight

        # 综合性能计算
        if all(ridge_metrics[k] for k in ridge_metrics) and all(lambda_metrics[k] for k in lambda_metrics):
            # Ridge综合得分
            ridge_ic_score = np.mean(ridge_metrics['ic'])
            ridge_rankic_score = np.mean(ridge_metrics['rankic'])
            ridge_return_score = np.mean(ridge_metrics['top_return'])
            ridge_vol_penalty = 1.0 / (1.0 + np.mean(ridge_metrics['volatility']))  # 波动率惩罚

            # Ridge综合性能（加权平均）
            ridge_performance = (
                0.3 * (ridge_ic_score + 1) / 2 +  # IC贡献30%
                0.3 * (ridge_rankic_score + 1) / 2 +  # RankIC贡献30%
                0.2 * np.tanh(ridge_return_score * 10) +  # 收益贡献20%
                0.2 * ridge_vol_penalty  # 稳定性贡献20%
            )

            # Lambda综合得分
            lambda_ic_score = np.mean(lambda_metrics['ic'])
            lambda_rankic_score = np.mean(lambda_metrics['rankic'])
            lambda_ndcg_score = np.mean(lambda_metrics['ndcg'])
            lambda_precision_score = np.mean(lambda_metrics['precision'])

            # Lambda综合性能（更重视排序指标）
            lambda_performance = (
                0.2 * (lambda_ic_score + 1) / 2 +  # IC贡献20%
                0.2 * (lambda_rankic_score + 1) / 2 +  # RankIC贡献20%
                0.4 * lambda_ndcg_score +  # NDCG贡献40%（核心指标）
                0.2 * lambda_precision_score  # Precision贡献20%
            )

            # 动态调整基准
            performance_ratio = lambda_performance / (ridge_performance + 1e-8)

            # 非线性映射：使权重更敏感于性能差异
            if performance_ratio > 1.2:  # Lambda明显更好
                raw_lambda_weight = 0.6 + 0.1 * min((performance_ratio - 1.2), 1.0)
            elif performance_ratio < 0.8:  # Ridge明显更好
                raw_lambda_weight = 0.4 - 0.1 * min((0.8 - performance_ratio), 0.3)
            else:  # 性能接近
                raw_lambda_weight = 0.45 + 0.1 * (performance_ratio - 0.95)

            # 加入趋势调整
            if len(ridge_metrics['ic']) >= recent_window:
                recent_ridge_trend = np.mean(ridge_metrics['ic'][-recent_window:]) - np.mean(ridge_metrics['ic'][:-recent_window])
                recent_lambda_trend = np.mean(lambda_metrics['ndcg'][-recent_window:]) - np.mean(lambda_metrics['ndcg'][:-recent_window])

                # 趋势调整权重
                if recent_lambda_trend > recent_ridge_trend + 0.02:  # Lambda趋势更好
                    raw_lambda_weight = min(raw_lambda_weight + 0.05, 0.75)
                elif recent_ridge_trend > recent_lambda_trend + 0.02:  # Ridge趋势更好
                    raw_lambda_weight = max(raw_lambda_weight - 0.05, 0.25)

            # 应用约束和智能平滑
            constrained_lambda_weight = np.clip(raw_lambda_weight, self.min_weight, self.max_weight)

            # 动态平滑系数：权重变化大时加强平滑
            weight_change = abs(constrained_lambda_weight - self.current_lambda_weight)
            dynamic_smoothing = self.weight_smoothing * (1 + weight_change)  # 变化大时更平滑
            dynamic_smoothing = min(dynamic_smoothing, 0.7)  # 上限为0.7

            # EWMA平滑
            smoothed_lambda_weight = (
                (1 - dynamic_smoothing) * constrained_lambda_weight +
                dynamic_smoothing * self.current_lambda_weight
            )

            logger.info(f"   Ridge综合性能: {ridge_performance:.4f}")
            logger.info(f"     - IC: {ridge_ic_score:.4f}, RankIC: {ridge_rankic_score:.4f}")
            logger.info(f"     - Top收益: {ridge_return_score:.4f}, 波动惩罚: {ridge_vol_penalty:.4f}")
            logger.info(f"   Lambda综合性能: {lambda_performance:.4f}")
            logger.info(f"     - IC: {lambda_ic_score:.4f}, RankIC: {lambda_rankic_score:.4f}")
            logger.info(f"     - NDCG: {lambda_ndcg_score:.4f}, Precision: {lambda_precision_score:.4f}")
            logger.info(f"   性能比: {performance_ratio:.3f}")
            logger.info(f"   原始权重: λ={raw_lambda_weight:.3f}")
            logger.info(f"   约束权重: λ={constrained_lambda_weight:.3f}")
            logger.info(f"   平滑权重(系数{dynamic_smoothing:.2f}): λ={smoothed_lambda_weight:.3f}")

            return smoothed_lambda_weight

        else:
            logger.warning("无法计算性能指标，使用当前权重")
            return self.current_lambda_weight

    def _apply_copula_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用Copula正态化：秩百分位 → 正态分位数

        Args:
            df: 包含预测分数的DataFrame

        Returns:
            添加正态化列的DataFrame
        """
        logger.info("🔄 应用Copula正态化...")

        df_norm = df.copy()

        def _copula_transform_by_date(group):
            """按日期组内进行Copula正态化"""
            for col in ['ridge_score', 'lambda_score']:
                if col in group.columns:
                    scores = group[col].dropna()
                    if len(scores) > 1:
                        # 计算秩百分位
                        ranks_pct = scores.rank(pct=True)
                        # 避免极值（0和1）
                        ranks_pct = np.clip(ranks_pct, 1e-6, 1-1e-6)
                        # 正态逆变换
                        norm_scores = ndtri(ranks_pct)
                        # 映射回原索引
                        full_norm = pd.Series(0.0, index=group.index)
                        full_norm.loc[scores.index] = norm_scores
                        group[col.replace('score', 'norm')] = full_norm
                    else:
                        group[col.replace('score', 'norm')] = 0.0
            return group

        df_norm = df_norm.groupby(level='date', group_keys=False).apply(_copula_transform_by_date)

        logger.info("✅ Copula正态化完成")
        return df_norm

    def _apply_zscore_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用普通z-score标准化

        Args:
            df: 包含预测分数的DataFrame

        Returns:
            添加标准化列的DataFrame
        """
        logger.info("🔄 应用Z-score标准化...")

        df_norm = df.copy()

        def _zscore_by_date(group):
            """按日期组内进行z-score标准化"""
            for col in ['ridge_score', 'lambda_score']:
                if col in group.columns:
                    scores = group[col]
                    if len(scores) > 1:
                        z_scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                        group[col.replace('score', 'z')] = z_scores
                    else:
                        group[col.replace('score', 'z')] = 0.0
            return group

        df_norm = df_norm.groupby(level='date', group_keys=False).apply(_zscore_by_date)

        logger.info("✅ Z-score标准化完成")
        return df_norm

    def _apply_decorrelation(self, df: pd.DataFrame, ridge_col: str, lambda_col: str) -> pd.DataFrame:
        """
        应用智能去相关融合 - 保留有价值的相关性，移除冗余共振

        核心改进：
        1. 部分去相关：保留30-40%有益相关性
        2. 自适应调整去相关强度
        3. 使用软阈值而非硬消除
        4. 保护极值信号不被过度修正

        Args:
            df: 包含标准化分数的DataFrame
            ridge_col: Ridge分数列名
            lambda_col: LambdaRank分数列名

        Returns:
            添加正交化列的DataFrame
        """
        logger.info("🔧 应用智能去相关融合...")

        df_ortho = df.copy()

        # 目标相关性范围（保留适度相关性）
        target_corr_range = (0.15, 0.35)  # 保留15%-35%的相关性
        decorr_strength = 0.7  # 去相关强度（0=不去相关, 1=完全去相关）

        def _decorrelate_by_date(group):
            """按日期组内智能去相关"""
            z_ridge = group[ridge_col]
            z_lambda = group[lambda_col]

            # 检查有效数据
            valid_mask = z_ridge.notna() & z_lambda.notna()
            if valid_mask.sum() < 10:  # 需要足够样本才去相关
                # 数据不足，使用原始信号
                group[lambda_col + '_ortho'] = z_lambda
                return group

            z_ridge_valid = z_ridge[valid_mask]
            z_lambda_valid = z_lambda[valid_mask]

            try:
                # 计算原始相关性
                original_corr = z_ridge_valid.corr(z_lambda_valid)

                # 判断是否需要去相关
                if abs(original_corr) < 0.5:  # 相关性不高时不需要处理
                    group[lambda_col + '_ortho'] = z_lambda
                    return group

                # OLS回归：z_L = β * z_R + ε
                cov_lr = (z_ridge_valid * z_lambda_valid).mean() - z_ridge_valid.mean() * z_lambda_valid.mean()
                var_r = ((z_ridge_valid - z_ridge_valid.mean()) ** 2).mean()

                if var_r > 1e-8:
                    beta = cov_lr / var_r

                    # 自适应调整去相关强度
                    if abs(original_corr) > 0.8:  # 高度相关时增强去相关
                        adjusted_strength = min(1.0, decorr_strength * 1.2)
                    elif abs(original_corr) > 0.6:
                        adjusted_strength = decorr_strength
                    else:
                        adjusted_strength = decorr_strength * 0.6  # 中度相关时减弱去相关

                    # 部分去相关：保留一定比例的共同信号
                    beta_adjusted = beta * adjusted_strength
                else:
                    beta_adjusted = 0.0

                # 计算软去相关信号
                z_lambda_ortho = z_lambda - beta_adjusted * z_ridge

                # 保护极值信号（top/bottom 5%不过度修正）
                extreme_mask = (abs(z_lambda) > np.percentile(abs(z_lambda_valid), 95))
                if extreme_mask.any():
                    # 极值位置使用较弱的去相关
                    z_lambda_ortho[extreme_mask] = z_lambda[extreme_mask] - 0.3 * beta_adjusted * z_ridge[extreme_mask]

                group[lambda_col + '_ortho'] = z_lambda_ortho

                # 记录去相关效果
                if len(z_ridge_valid) > 1:
                    z_lambda_ortho_valid = z_lambda_ortho[valid_mask]
                    new_corr = z_ridge_valid.corr(z_lambda_ortho_valid) if z_ridge_valid.std() > 0 and z_lambda_ortho_valid.std() > 0 else 0.0

                    # 确保相关性在目标范围内
                    if abs(new_corr) < target_corr_range[0] and abs(original_corr) > 0.3:
                        # 相关性过低，减弱去相关
                        correction_factor = 0.5
                        z_lambda_ortho = z_lambda - beta_adjusted * correction_factor * z_ridge
                        group[lambda_col + '_ortho'] = z_lambda_ortho

                    # 存储调试信息
                    group._decorr_info = {
                        'beta': beta_adjusted,
                        'original_corr': original_corr,
                        'ortho_corr': new_corr,
                        'strength': adjusted_strength,
                        'n_valid': len(z_ridge_valid)
                    }

            except Exception as e:
                # 回归失败，使用原始信号
                logger.debug(f"去相关处理跳过: {e}")
                group[lambda_col + '_ortho'] = z_lambda

            return group

        # 按日期分组执行去相关
        df_ortho = df_ortho.groupby(level='date', group_keys=False).apply(_decorrelate_by_date)

        # 统计去相关效果
        try:
            # 计算全局相关性变化
            z_ridge_all = df_ortho[ridge_col].dropna()
            z_lambda_all = df_ortho[lambda_col].dropna()
            z_lambda_ortho_all = df_ortho[lambda_col + '_ortho'].dropna()

            if len(z_ridge_all) > 1 and len(z_lambda_all) > 1:
                # 计算对齐的数据
                common_idx = z_ridge_all.index.intersection(z_lambda_all.index).intersection(z_lambda_ortho_all.index)
                if len(common_idx) > 1:
                    z_r = z_ridge_all.reindex(common_idx)
                    z_l = z_lambda_all.reindex(common_idx)
                    z_l_ortho = z_lambda_ortho_all.reindex(common_idx)

                    original_corr = z_r.corr(z_l) if z_r.std() > 0 and z_l.std() > 0 else 0.0
                    ortho_corr = z_r.corr(z_l_ortho) if z_r.std() > 0 and z_l_ortho.std() > 0 else 0.0

                    logger.info(f"   原始相关性: {original_corr:.4f}")
                    logger.info(f"   调整后相关性: {ortho_corr:.4f} (目标范围: 0.15-0.35)")
                    logger.info(f"   去相关降幅: {abs(original_corr - ortho_corr):.4f}")

                    # 验证是否达到理想效果
                    if abs(ortho_corr) < 0.05:
                        logger.warning("   ⚠️ 相关性过低，可能损失协同信号")
                    elif abs(ortho_corr) > 0.5:
                        logger.warning("   ⚠️ 相关性仍较高，去相关效果有限")

        except Exception as e:
            logger.warning(f"去相关统计失败: {e}")

        logger.info("✅ 智能去相关完成")
        return df_ortho

    def _apply_gated_blending(self, df: pd.DataFrame, ridge_col: str, lambda_col: str,
                             ridge_weight: float, lambda_weight: float) -> pd.Series:
        """门控+残差微融合：保留协同红利，避免排序-幅度错配"""

        def _gate_with_residual_fusion(group):
            ridge_scores = group[ridge_col]
            lambda_scores = group[lambda_col]
            n_samples = len(ridge_scores)

            if n_samples < 10:  # 样本太少时回退到简单组合
                return ridge_weight * ridge_scores + lambda_weight * lambda_scores

            # Step 1: 按日标准化得到 z_ridge, z_lambda
            z_ridge = (ridge_scores - ridge_scores.mean()) / (ridge_scores.std() + 1e-8)
            z_lambda = (lambda_scores - lambda_scores.mean()) / (lambda_scores.std() + 1e-8)

            # Step 2: 计算 lambda_pct (LambdaRank 的百分位排名)
            lambda_pct = lambda_scores.rank(method='average') / n_samples

            # Step 3: 门控判断 - 基于 lambda_pct 阈值
            tau_long = getattr(self, 'tau_long', 0.65)
            tau_short = getattr(self, 'tau_short', 0.35)
            alpha_long = getattr(self, 'alpha_long', 0.15)
            alpha_short = getattr(self, 'alpha_short', 0.15)
            max_gain = getattr(self, 'max_gain', 1.25)

            # 计算门控和增益
            gate = np.zeros(n_samples, dtype=float)
            gain = np.ones(n_samples, dtype=float)

            # 长侧门控与增益
            long_mask = lambda_pct >= tau_long
            if long_mask.any():
                gate[long_mask] = 1.0
                long_gain_factor = (lambda_pct[long_mask] - tau_long) / (1 - tau_long)
                gain[long_mask] = 1.0 + alpha_long * long_gain_factor
                gain[long_mask] = np.clip(gain[long_mask], 1.0, max_gain)

            # 短侧门控与增益
            short_mask = lambda_pct <= tau_short
            if short_mask.any():
                gate[short_mask] = 1.0
                short_gain_factor = (tau_short - lambda_pct[short_mask]) / tau_short
                gain[short_mask] = 1.0 + alpha_short * short_gain_factor
                gain[short_mask] = np.clip(gain[short_mask], 1.0, max_gain)

            # Step 4: 计算覆盖率并检查触发条件
            coverage = gate.sum() / n_samples
            min_coverage = getattr(self, 'min_coverage', 0.30)

            # Step 5: 去相关残差计算（仅在门内样本上）
            gated_mask = gate > 0
            if coverage >= min_coverage and gated_mask.any():
                # 使用门内样本计算去相关回归
                z_ridge_gated = z_ridge[gated_mask]
                z_lambda_gated = z_lambda[gated_mask]

                if len(z_ridge_gated) > 5 and z_ridge_gated.std() > 1e-6:
                    # 计算回归系数 β_reg = Cov(z_λ, z_r) / Var(z_r)
                    cov_lr = np.cov(z_lambda_gated, z_ridge_gated)[0, 1]
                    var_r = np.var(z_ridge_gated)
                    beta_reg = cov_lr / (var_r + 1e-8)

                    # 计算去相关残差：z_λ⊥ = z_lambda - β_reg * z_ridge
                    z_lambda_ortho = z_lambda - beta_reg * z_ridge
                else:
                    z_lambda_ortho = z_lambda.copy()
            else:
                z_lambda_ortho = z_lambda.copy()

            # Step 6: 残差微融合参数 β 管理
            current_beta = getattr(self, 'current_beta', 0.08)
            beta_range = getattr(self, 'beta_range', [0.0, 0.15])

            # 触发条件检查
            enable_micro_fusion = self._check_micro_fusion_trigger(coverage)

            if enable_micro_fusion:
                # 启用微融合时，根据性能调整β
                new_beta = self._update_beta_with_ewma(current_beta)
            else:
                # 降级到纯门控，β 指数衰减至0
                ewma_alpha = getattr(self, 'beta_ewma_alpha', 0.3)
                new_beta = current_beta * (1 - ewma_alpha)  # EWMA衰减
                new_beta = max(0.0, new_beta)

            # 确保β在允许范围内
            new_beta = np.clip(new_beta, beta_range[0], beta_range[1])
            setattr(self, 'current_beta', new_beta)

            # 记录β历史（用于监控）
            if not hasattr(self, 'beta_history'):
                setattr(self, 'beta_history', [])
            self.beta_history.append(new_beta)
            if len(self.beta_history) > 100:
                self.beta_history = self.beta_history[-100:]

            current_beta = new_beta

            # Step 7: 计算最终分数
            # score = z_ridge × gain(lambda_pct) × gate(lambda_pct) × (1 + β × clip(z_λ⊥, p2, p98))

            # 计算残差裁剪阈值 (p2, p98)
            p2, p98 = np.percentile(z_lambda_ortho[gated_mask], [2, 98]) if gated_mask.any() else [-2.0, 2.0]
            z_lambda_ortho_clipped = np.clip(z_lambda_ortho, p2, p98)

            # 残差微融合项
            residual_tilt = 1.0 + current_beta * z_lambda_ortho_clipped

            # 基础门控分数
            base_score = z_ridge * gain * gate

            # 仅对门内样本应用残差微融合
            final_score = base_score.copy()
            if gated_mask.any():
                final_score[gated_mask] = base_score[gated_mask] * residual_tilt[gated_mask]

            # Step 8: 方向约束 - 确保不翻方向
            sign_flips = (np.sign(final_score) != np.sign(z_ridge)) & (np.abs(z_ridge) > 1e-6)
            if sign_flips.any():
                # 发生翻方向时，回退到原始门控分数
                final_score[sign_flips] = base_score[sign_flips]

            # 记录统计信息
            flip_ratio = sign_flips.sum() / n_samples if n_samples > 0 else 0.0
            if not hasattr(self, '_flip_ratio_history'):
                self._flip_ratio_history = []
            self._flip_ratio_history.append(flip_ratio)
            if len(self._flip_ratio_history) > 100:
                self._flip_ratio_history = self._flip_ratio_history[-100:]

            return pd.Series(final_score, index=ridge_scores.index)

        # 按日期分组应用门控+残差微融合
        return df.groupby(level='date', group_keys=False).apply(_gate_with_residual_fusion)

    def _check_micro_fusion_trigger(self, coverage: float) -> bool:
        """检查残差微融合触发条件"""

        # 条件1：覆盖率满足最小要求
        min_coverage = getattr(self, 'min_coverage', 0.30)
        if coverage < min_coverage:
            return False

        # 条件2：最近窗口NDCG有提升（简化版本检查）
        ndcg_history = getattr(self, '_ndcg_history', [])
        if len(ndcg_history) >= 5:
            # 计算最近5次的NDCG趋势
            recent_ndcg = ndcg_history[-5:]
            if len(recent_ndcg) >= 3:
                # 简单趋势检查：最近3次的平均是否优于前面
                recent_avg = np.mean(recent_ndcg[-3:])
                earlier_avg = np.mean(recent_ndcg[:-3]) if len(recent_ndcg) > 3 else recent_avg
                performance_improved = recent_avg >= earlier_avg
            else:
                performance_improved = True  # 数据不足时保守启用
        else:
            performance_improved = True  # 初始阶段默认启用

        # 记录性能改善状态
        setattr(self, '_recent_performance_improved', performance_improved)

        return performance_improved

    def _update_beta_with_ewma(self, current_beta: float) -> float:
        """使用EWMA平滑更新β参数"""

        ewma_alpha = getattr(self, 'beta_ewma_alpha', 0.3)
        beta_range = getattr(self, 'beta_range', [0.0, 0.15])

        # 基于性能指标调整β的目标值
        performance_improved = getattr(self, '_recent_performance_improved', True)
        flip_ratio_history = getattr(self, '_flip_ratio_history', [])

        # 计算目标β
        if performance_improved:
            # 性能改善时，略微增加β以获得更多LTR红利
            recent_flip_ratio = np.mean(flip_ratio_history[-10:]) if len(flip_ratio_history) >= 10 else 0.0

            if recent_flip_ratio < 0.05:  # 翻方向比例很低，可以增加β
                target_beta = min(beta_range[1], current_beta * 1.05)
            elif recent_flip_ratio < 0.1:  # 翻方向比例可接受，保持β
                target_beta = current_beta
            else:  # 翻方向比例过高，减少β
                target_beta = max(beta_range[0], current_beta * 0.95)
        else:
            # 性能未改善时，保守减少β
            target_beta = max(beta_range[0], current_beta * 0.9)

        # EWMA平滑更新
        new_beta = (1 - ewma_alpha) * current_beta + ewma_alpha * target_beta

        # 确保在合理范围内
        new_beta = np.clip(new_beta, beta_range[0], beta_range[1])

        return new_beta

    def update_performance_metrics(self, ridge_predictions: pd.DataFrame,
                                 lambda_predictions: pd.DataFrame,
                                 actual_returns: pd.DataFrame) -> None:
        """更新性能指标用于触发条件判断"""

        try:
            # 计算NDCG@K指标
            for k in self.top_k_list:
                if k <= len(actual_returns):
                    # 简化的NDCG计算（实际应用中需要更完整的实现）
                    ndcg_score = self._calculate_ndcg_k(ridge_predictions, actual_returns, k)

                    if not hasattr(self, '_ndcg_history'):
                        self._ndcg_history = []
                    self._ndcg_history.append(ndcg_score)
                    if len(self._ndcg_history) > 50:
                        self._ndcg_history = self._ndcg_history[-50:]
                    break  # 只使用第一个可计算的K值

        except Exception as e:
            logger.warning(f"更新性能指标失败: {e}")

    def _calculate_ndcg_k(self, predictions: pd.DataFrame,
                         actual_returns: pd.DataFrame, k: int) -> float:
        """计算NDCG@K指标（简化版本）"""

        try:
            # 确保数据对齐
            common_index = predictions.index.intersection(actual_returns.index)
            if len(common_index) < k:
                return 0.0

            pred_aligned = predictions.loc[common_index].squeeze()
            ret_aligned = actual_returns.loc[common_index].squeeze()

            # 按预测排序，取top-k
            sorted_indices = pred_aligned.argsort()[::-1][:k]
            top_k_returns = ret_aligned.iloc[sorted_indices]

            # 计算DCG
            dcg = np.sum(top_k_returns.values / np.log2(np.arange(2, k + 2)))

            # 计算IDCG（理想情况）
            ideal_returns = ret_aligned.sort_values(ascending=False)[:k]
            idcg = np.sum(ideal_returns.values / np.log2(np.arange(2, k + 2)))

            # 计算NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0

            return ndcg

        except Exception as e:
            logger.warning(f"NDCG计算失败: {e}")
            return 0.0

    def get_weight_history(self) -> pd.DataFrame:
        """获取权重历史"""
        if not self.weight_history:
            return pd.DataFrame()

        return pd.DataFrame(self.weight_history)

    def _calculate_blend_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算融合统计信息"""
        stats = {}

        if 'blended_score' in df.columns:
            stats['mean'] = df['blended_score'].mean()
            stats['std'] = df['blended_score'].std()
            stats['skew'] = df['blended_score'].skew()
            stats['kurt'] = df['blended_score'].kurtosis()

        # 计算Ridge和Lambda信号一致性
        if 'ridge_score' in df.columns and 'lambda_score' in df.columns:
            # 正相关率：两个信号同向的比例
            same_sign = (np.sign(df['ridge_score']) == np.sign(df['lambda_score']))
            stats['agreement_rate'] = same_sign.mean()

            # Top-K重叠率
            k = 100
            if len(df) >= k:
                ridge_top_k = set(df.nlargest(k, 'ridge_score').index)
                lambda_top_k = set(df.nlargest(k, 'lambda_score').index)
                overlap = len(ridge_top_k.intersection(lambda_top_k))
                stats['top_k_overlap'] = overlap / k

        return stats

    def _apply_advanced_blending(self, df: pd.DataFrame, ridge_col: str, lambda_col: str,
                                ridge_weight: float, lambda_weight: float) -> pd.Series:
        """高级融合策略：结合动态权重、门控和非线性融合"""

        def _advanced_blend_by_date(group):
            ridge_scores = group[ridge_col]
            lambda_scores = group[lambda_col]
            n_samples = len(ridge_scores)

            if n_samples < 10:
                # 样本太少，使用简单线性组合
                return ridge_weight * ridge_scores + lambda_weight * lambda_scores

            # 1. 计算分位数
            ridge_pct = ridge_scores.rank(pct=True)
            lambda_pct = lambda_scores.rank(pct=True)

            # 2. 动态权重调整：极端位置使用不同权重
            dynamic_weights = np.ones(n_samples)

            # 极端多头（top 5%）：Lambda权重更高（排序更准）
            extreme_long = (ridge_pct > 0.95) | (lambda_pct > 0.95)
            dynamic_weights[extreme_long] = lambda_weight + 0.1

            # 极端空头（bottom 5%）：Ridge权重更高（幅度更准）
            extreme_short = (ridge_pct < 0.05) | (lambda_pct < 0.05)
            dynamic_weights[extreme_short] = ridge_weight + 0.1

            # 中间区域：标准权重
            middle_zone = ~(extreme_long | extreme_short)
            dynamic_weights[middle_zone] = lambda_weight

            # 归一化权重
            dynamic_weights = np.clip(dynamic_weights, 0, 1)
            ridge_dynamic = 1 - dynamic_weights

            # 3. 非线性融合：考虑信号一致性
            signal_agreement = np.sign(ridge_scores) == np.sign(lambda_scores)

            # 信号一致时加强，不一致时减弱
            boost_factor = np.where(signal_agreement, 1.1, 0.9)

            # 最终融合
            blended = (
                ridge_dynamic * ridge_scores +
                dynamic_weights * lambda_scores
            ) * boost_factor

            return pd.Series(blended, index=ridge_scores.index)

        # 按日期分组应用高级融合
        return df.groupby(level='date', group_keys=False).apply(_advanced_blend_by_date)

    def get_blender_info(self) -> Dict[str, Any]:
        """获取融合器信息"""
        return {
            'lookback_window': self.lookback_window,
            'current_lambda_weight': self.current_lambda_weight,
            'current_ridge_weight': 1.0 - self.current_lambda_weight,
            'weight_constraints': [self.min_weight, self.max_weight],
            'use_copula': self.use_copula,
            'weight_smoothing': self.weight_smoothing,
            'n_weight_records': len(self.weight_history),
            # 门控+残差微融合参数
            'gate_params': {
                'tau_long': getattr(self, 'tau_long', 0.65),
                'tau_short': getattr(self, 'tau_short', 0.35),
                'alpha_long': getattr(self, 'alpha_long', 0.15),
                'alpha_short': getattr(self, 'alpha_short', 0.15),
                'max_gain': getattr(self, 'max_gain', 1.25),
                'min_coverage': getattr(self, 'min_coverage', 0.30)
            },
            'residual_fusion': {
                'current_beta': getattr(self, 'current_beta', 0.08),
                'beta_range': getattr(self, 'beta_range', [0.0, 0.15]),
                'beta_ewma_alpha': getattr(self, 'beta_ewma_alpha', 0.3),
                'recent_performance_improved': getattr(self, '_recent_performance_improved', True)
            },
            'performance_stats': self._get_performance_stats()
        }

    def _get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""

        flip_ratio_history = getattr(self, '_flip_ratio_history', [])
        coverage_history = getattr(self, '_coverage_history', [])
        ndcg_history = getattr(self, '_ndcg_history', [])
        beta_history = getattr(self, 'beta_history', [])

        stats = {}

        # 方向翻转统计
        if flip_ratio_history:
            stats['flip_ratio'] = {
                'recent_mean': np.mean(flip_ratio_history[-10:]) if len(flip_ratio_history) >= 10 else np.mean(flip_ratio_history),
                'overall_mean': np.mean(flip_ratio_history),
                'max': np.max(flip_ratio_history),
                'count': len(flip_ratio_history)
            }

        # 覆盖率统计
        if coverage_history:
            stats['coverage'] = {
                'recent_mean': np.mean(coverage_history[-10:]) if len(coverage_history) >= 10 else np.mean(coverage_history),
                'overall_mean': np.mean(coverage_history),
                'min': np.min(coverage_history),
                'count': len(coverage_history)
            }

        # NDCG统计
        if ndcg_history:
            stats['ndcg'] = {
                'recent_mean': np.mean(ndcg_history[-10:]) if len(ndcg_history) >= 10 else np.mean(ndcg_history),
                'overall_mean': np.mean(ndcg_history),
                'trend': 'improving' if len(ndcg_history) >= 5 and np.mean(ndcg_history[-3:]) > np.mean(ndcg_history[-5:-3]) else 'stable',
                'count': len(ndcg_history)
            }

        # β参数统计
        if beta_history:
            stats['beta'] = {
                'current': beta_history[-1] if beta_history else 0.0,
                'recent_mean': np.mean(beta_history[-10:]) if len(beta_history) >= 10 else np.mean(beta_history),
                'trend': 'increasing' if len(beta_history) >= 5 and beta_history[-1] > np.mean(beta_history[-5:-1]) else 'stable',
                'count': len(beta_history)
            }

        return stats

    def calculate_acceptance_metrics(self, oof_predictions: pd.DataFrame,
                                   online_ridge_predictions: pd.DataFrame,
                                   actual_returns: pd.DataFrame,
                                   top_k_list: list = None) -> Dict[str, float]:
        """计算验收指标"""

        if top_k_list is None:
            top_k_list = self.top_k_list

        metrics = {}

        try:
            # 1. Top-K 命中率和NDCG提升
            for k in top_k_list:
                if k <= len(actual_returns):
                    # 计算OOF预测的NDCG@K
                    oof_ndcg = self._calculate_ndcg_k(oof_predictions, actual_returns, k)
                    # 计算在线Ridge预测的NDCG@K
                    ridge_ndcg = self._calculate_ndcg_k(online_ridge_predictions, actual_returns, k)

                    metrics[f'ndcg@{k}_oof'] = oof_ndcg
                    metrics[f'ndcg@{k}_ridge'] = ridge_ndcg
                    metrics[f'ndcg@{k}_improvement'] = oof_ndcg - ridge_ndcg

                    # 计算Top-K命中率
                    top_k_hit_rate = self._calculate_top_k_hit_rate(oof_predictions, actual_returns, k)
                    metrics[f'top{k}_hit_rate'] = top_k_hit_rate

            # 2. OOS Information Ratio (简化版本)
            if len(oof_predictions) > 20:
                oos_ir = self._calculate_oos_ir(oof_predictions, actual_returns)
                metrics['oos_ir'] = oos_ir

            # 3. KS检验（OOF vs 线上Ridge输入）
            ks_stat = self._calculate_ks_test(oof_predictions, online_ridge_predictions)
            metrics['ks_statistic'] = ks_stat

            # 4. 方向翻转比例
            flip_ratio_history = getattr(self, '_flip_ratio_history', [])
            if flip_ratio_history:
                metrics['flip_ratio'] = np.mean(flip_ratio_history[-10:]) if len(flip_ratio_history) >= 10 else np.mean(flip_ratio_history)

            # 5. 覆盖率统计
            coverage_history = getattr(self, '_coverage_history', [])
            if coverage_history:
                metrics['coverage'] = np.mean(coverage_history[-10:]) if len(coverage_history) >= 10 else np.mean(coverage_history)

        except Exception as e:
            logger.error(f"验收指标计算失败: {e}")

        return metrics

    def _calculate_top_k_hit_rate(self, predictions: pd.DataFrame,
                                actual_returns: pd.DataFrame, k: int) -> float:
        """计算Top-K命中率"""

        try:
            common_index = predictions.index.intersection(actual_returns.index)
            if len(common_index) < k:
                return 0.0

            pred_aligned = predictions.loc[common_index].squeeze()
            ret_aligned = actual_returns.loc[common_index].squeeze()

            # 按预测排序，取top-k
            top_k_pred_indices = pred_aligned.argsort()[::-1][:k]
            # 按实际收益排序，取top-k
            top_k_actual_indices = ret_aligned.argsort()[::-1][:k]

            # 计算命中数量
            hits = len(set(top_k_pred_indices).intersection(set(top_k_actual_indices)))
            hit_rate = hits / k

            return hit_rate

        except Exception as e:
            logger.warning(f"Top-K命中率计算失败: {e}")
            return 0.0

    def _calculate_oos_ir(self, predictions: pd.DataFrame,
                         actual_returns: pd.DataFrame) -> float:
        """计算样本外信息比率"""

        try:
            common_index = predictions.index.intersection(actual_returns.index)
            if len(common_index) < 20:
                return 0.0

            pred_aligned = predictions.loc[common_index].squeeze()
            ret_aligned = actual_returns.loc[common_index].squeeze()

            # 计算IC
            ic = pred_aligned.corr(ret_aligned)

            # 计算IC的标准差（滚动窗口）
            if len(pred_aligned) >= 50:
                rolling_ics = []
                window_size = min(20, len(pred_aligned) // 3)
                for i in range(window_size, len(pred_aligned)):
                    window_pred = pred_aligned.iloc[i-window_size:i]
                    window_ret = ret_aligned.iloc[i-window_size:i]
                    window_ic = window_pred.corr(window_ret)
                    if not np.isnan(window_ic):
                        rolling_ics.append(window_ic)

                if rolling_ics:
                    ic_std = np.std(rolling_ics)
                    ir = ic / (ic_std + 1e-8)
                else:
                    ir = ic / 0.1  # 默认分母
            else:
                ir = ic / 0.1  # 样本不足时使用默认分母

            return ir

        except Exception as e:
            logger.warning(f"OOS IR计算失败: {e}")
            return 0.0

    def _calculate_ks_test(self, oof_predictions: pd.DataFrame,
                          online_predictions: pd.DataFrame) -> float:
        """计算KS检验统计量"""

        try:
            from scipy.stats import ks_2samp

            oof_values = oof_predictions.dropna().squeeze().values
            online_values = online_predictions.dropna().squeeze().values

            if len(oof_values) < 10 or len(online_values) < 10:
                return 1.0  # 数据不足时返回最大值

            # 进行KS双样本检验
            ks_stat, p_value = ks_2samp(oof_values, online_values)

            return ks_stat

        except Exception as e:
            logger.warning(f"KS检验计算失败: {e}")
            return 1.0

    def get_acceptance_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """生成验收结果摘要"""

        summary = {}

        # NDCG提升检查
        ndcg_improvements = [v for k, v in metrics.items() if 'ndcg' in k and 'improvement' in k]
        if ndcg_improvements:
            avg_improvement = np.mean(ndcg_improvements)
            summary['ndcg_status'] = 'PASS' if avg_improvement > 0 else 'FAIL'
            summary['ndcg_avg_improvement'] = f"{avg_improvement:.4f}"

        # OOS IR检查
        if 'oos_ir' in metrics:
            summary['oos_ir_status'] = 'PASS' if metrics['oos_ir'] > 0.5 else 'FAIL'  # 简化阈值
            summary['oos_ir_value'] = f"{metrics['oos_ir']:.4f}"

        # KS检验检查
        if 'ks_statistic' in metrics:
            summary['ks_status'] = 'PASS' if metrics['ks_statistic'] < 0.1 else 'FAIL'
            summary['ks_value'] = f"{metrics['ks_statistic']:.4f}"

        # 方向翻转检查
        if 'flip_ratio' in metrics:
            summary['flip_ratio_status'] = 'PASS' if metrics['flip_ratio'] < 0.1 else 'FAIL'
            summary['flip_ratio_value'] = f"{metrics['flip_ratio']:.4f}"

        # 覆盖率检查
        if 'coverage' in metrics:
            min_coverage = getattr(self, 'min_coverage', 0.30)
            summary['coverage_status'] = 'PASS' if metrics['coverage'] >= min_coverage else 'FAIL'
            summary['coverage_value'] = f"{metrics['coverage']:.4f}"

        # 总体验收状态
        all_checks = [v for k, v in summary.items() if k.endswith('_status')]
        summary['overall_status'] = 'PASS' if all([check == 'PASS' for check in all_checks]) else 'FAIL'

        return summary