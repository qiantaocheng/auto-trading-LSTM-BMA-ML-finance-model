#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Head Late Fusion - 双头晚融合系统
====================================

专业级双头融合架构，针对"最大化预测收益率"优化：
- 回归头（Ridge stacking）：预测连续收益率magnitude
- 排序头（LambdaRank）：优化Top-K选股排序

核心特性：
1. 独立校准：Isotonic（回归） + Quantile-Normal（排序）
2. 横截面标准化：统一刻度（z-score）
3. 固定线性加权：S = α * z_reg + β * z_rank
4. OOF调参：自动选择最优α/β
5. 防数据泄漏：严格时间隔离

作者: BMA Enhanced System
日期: 2025-01-XX
版本: 1.0.0 (T+5 Optimized)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DualHeadLateFusion:
    """
    双头晚融合系统 - 固定线性加权基线（Strong Stable Baseline）

    架构：
    ┌──────────────────────────────────────────────────────────┐
    │  第一层：基础模型（PurgedCV OOF）                          │
    │  ├─ ElasticNet, XGBoost, CatBoost                        │
    │  └─ 生成OOF predictions                                  │
    └──────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
    ┌───────────────────────┐  ┌──────────────────────┐
    │  回归头（Ridge）       │  │  排序头（LambdaRank） │
    │  目标：预测收益率      │  │  目标：Top-K排序     │
    │  输入：3个OOF          │  │  输入：Alpha Factors │
    │  输出：ŷ_reg          │  │  输出：s_rank        │
    └───────────┬───────────┘  └──────────┬───────────┘
                │                         │
                │  Isotonic校准           │  Quantile映射
                │  ↓                      │  ↓
                │  横截面z-score          │  正态化z-score
                │  ↓                      │  ↓
                │  z_reg                  │  z_rank
                └───────────┬─────────────┘
                            ▼
              ┌─────────────────────────┐
              │  固定线性加权融合        │
              │  S = α*z_reg + β*z_rank │
              │  (α=0.7, β=0.3 默认)    │
              └─────────────────────────┘

    使用场景：
    - 量化选股：需要预测收益率 + 排序
    - 投资组合构建：sizing用回归，排序用LambdaRank
    - 风险管理：连续预测值用于风控计算
    """

    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 auto_tune_weights: bool = True,
                 alpha_grid: List[float] = None,
                 tanh_clip_c: float = 2.5,
                 use_isotonic_calibration: bool = False,  # 🔧 默认关闭：效果小且增加过拟合风险
                 random_state: int = 42):
        """
        初始化双头晚融合系统

        Args:
            alpha: 回归头权重（默认0.7，回归为主）
            beta: 排序头权重（默认0.3，排序为辅）
            auto_tune_weights: 是否使用OOF自动调参（推荐True）
            alpha_grid: 调参网格（默认[0.5, 0.6, 0.7, 0.8, 0.9]）
            tanh_clip_c: tanh限幅参数（防止极端值，默认2.5）
            use_isotonic_calibration: 是否对回归头使用Isotonic校准
            random_state: 随机种子
        """
        # 权重配置
        self.alpha = alpha
        self.beta = beta
        if abs(alpha + beta - 1.0) > 0.01:
            logger.warning(f"权重之和不为1: α={alpha}, β={beta}，将自动归一化")
            total = alpha + beta
            self.alpha = alpha / total
            self.beta = beta / total

        self.auto_tune_weights = auto_tune_weights
        self.alpha_grid = alpha_grid or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.tanh_clip_c = tanh_clip_c
        self.use_isotonic_calibration = use_isotonic_calibration
        self.random_state = random_state

        # 校准模型（训练后初始化）
        self.isotonic_reg = None  # 回归头校准器
        self.rank_quantile_map = None  # 排序头分位数映射

        # 训练状态
        self.fitted_ = False
        self.best_alpha_ = alpha
        self.best_beta_ = beta
        self.tuning_results_ = None

        logger.info("✅ 双头晚融合系统初始化完成")
        logger.info(f"   回归头权重 α: {self.alpha:.2f}")
        logger.info(f"   排序头权重 β: {self.beta:.2f}")
        logger.info(f"   自动调参: {self.auto_tune_weights}")
        logger.info(f"   Isotonic校准: {self.use_isotonic_calibration}")
        logger.info(f"   Tanh限幅: c={self.tanh_clip_c}")

    def calibrate_regression_head(self, y_pred_reg_oof: pd.Series,
                                   y_true_oof: pd.Series) -> None:
        """
        校准回归头：Isotonic Regression（单调映射，保留排序）

        为什么使用Isotonic？
        - 单调性：保留原始排序信息
        - 非参数：自适应数据分布
        - 校准性：修正系统性偏差

        Args:
            y_pred_reg_oof: 回归头OOF预测（连续值）
            y_true_oof: 真实标签（连续收益率）
        """
        if not self.use_isotonic_calibration:
            logger.info("⏭️ Isotonic校准已禁用，跳过")
            return

        logger.info("🔧 开始校准回归头（Isotonic Regression）...")

        # 移除NaN
        valid_mask = y_pred_reg_oof.notna() & y_true_oof.notna()
        y_pred_clean = y_pred_reg_oof[valid_mask].values
        y_true_clean = y_true_oof[valid_mask].values

        if len(y_pred_clean) < 30:
            logger.warning(f"⚠️ 有效样本过少({len(y_pred_clean)}), 跳过Isotonic校准")
            return

        # 训练Isotonic Regression
        self.isotonic_reg = IsotonicRegression(
            y_min=np.percentile(y_true_clean, 1),
            y_max=np.percentile(y_true_clean, 99),
            increasing=True,
            out_of_bounds='clip'
        )
        self.isotonic_reg.fit(y_pred_clean, y_true_clean)

        # 评估校准效果
        y_calibrated = self.isotonic_reg.predict(y_pred_clean)
        mse_before = mean_squared_error(y_true_clean, y_pred_clean)
        mse_after = mean_squared_error(y_true_clean, y_calibrated)

        logger.info(f"✅ Isotonic校准完成")
        logger.info(f"   MSE: {mse_before:.6f} → {mse_after:.6f} (改善{(1-mse_after/mse_before)*100:.1f}%)")

    def calibrate_ranking_head(self, s_rank_oof: pd.Series,
                               y_true_oof: pd.Series,
                               dates: pd.Series) -> None:
        """
        校准排序头：简化版 - 直接使用rank percentile

        修复原因：
        - 原版计算的是"平均真实收益"而非"收益分位数"
        - 导致映射范围倒挂（最大值 < 最小值）
        - 简化为直接使用Lambda分数的rank percentile

        重要说明：
        - 如果Lambda OOF与收益负相关，会输出ERROR级别警告
        - 不应在预测时自动反转分数（会掩盖训练问题）
        - 建议：使用lambda_pct（百分位）而非lambda_score（原始分数）
          因为lambda_pct通过rank(pct=True, ascending=True)计算，
          保证了分数高→百分位高的一致性

        Args:
            s_rank_oof: 排序头OOF分数（LambdaRank输出的原始分数）
            y_true_oof: 真实标签
            dates: 日期序列（用于横截面分位数计算）
        """
        logger.info("🔧 开始校准排序头（简化版：直接使用Rank Percentile）...")

        # 移除NaN
        valid_mask = s_rank_oof.notna() & y_true_oof.notna()
        s_rank_clean = s_rank_oof[valid_mask]
        y_true_clean = y_true_oof[valid_mask]
        dates_clean = dates[valid_mask]

        if len(s_rank_clean) < 30:
            logger.warning(f"⚠️ 有效样本过少({len(s_rank_clean)}), 跳过排序校准")
            return

        # 验证Lambda与收益的相关性（仅诊断，不反转）
        from scipy.stats import spearmanr
        avg_rank_ic = 0.0

        try:
            df_temp = pd.DataFrame({
                's_rank': s_rank_clean.values,
                'y_true': y_true_clean.values,
                'date': dates_clean.values
            })

            # 计算横截面Spearman相关系数
            corr_scores = []
            for date, group in df_temp.groupby('date'):
                if len(group) >= 10:  # 至少10个样本才计算相关性
                    corr, _ = spearmanr(group['s_rank'], group['y_true'])
                    if not np.isnan(corr):
                        corr_scores.append(corr)

            avg_rank_ic = np.mean(corr_scores) if corr_scores else 0.0

            # 🔧 诊断警告（不自动修正，用户需检查训练逻辑）
            if avg_rank_ic < -0.01:  # 负相关阈值
                logger.error("="*80)
                logger.error(f"❌ 严重问题：Lambda与收益呈负相关({avg_rank_ic:.4f})！")
                logger.error("   这表明LambdaRank训练可能有问题：")
                logger.error("   1. 检查特征与收益的关系是否正确")
                logger.error("   2. 检查是否有数据泄漏")
                logger.error("   3. 检查LambdaRank是否过拟合")
                logger.error("   建议：使用lambda_pct（百分位）而非lambda_score（原始分数）")
                logger.error("="*80)
            elif avg_rank_ic < 0:
                logger.warning(f"⚠️ Lambda与收益相关性接近零({avg_rank_ic:.4f})")
        except Exception as e:
            logger.warning(f"⚠️ 相关性验证失败: {e}")

        # 🔧 保存Lambda OOF的统计信息，用于预测时标准化
        self.rank_quantile_map = {
            'method': 'simple_rank_pct',
            'lambda_mean': float(s_rank_clean.mean()),
            'lambda_std': float(s_rank_clean.std()),
            'lambda_min': float(s_rank_clean.min()),
            'lambda_max': float(s_rank_clean.max()),
            'rank_ic': float(avg_rank_ic)
        }

        logger.info(f"✅ 排序头校准完成（简化版）")
        logger.info(f"   Lambda统计: mean={self.rank_quantile_map['lambda_mean']:.4f}, "
                   f"std={self.rank_quantile_map['lambda_std']:.4f}")
        logger.info(f"   Lambda范围: [{self.rank_quantile_map['lambda_min']:.4f}, "
                   f"{self.rank_quantile_map['lambda_max']:.4f}]")
        logger.info(f"   平均RankIC: {avg_rank_ic:.4f} (横截面相关性)")

    def _apply_calibration(self, y_pred_reg: np.ndarray,
                          s_rank: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用校准：Isotonic（回归） + Rank Percentile（排序）

        修复：简化排序头校准，直接使用rank percentile

        Args:
            y_pred_reg: 回归头原始预测
            s_rank: 排序头原始分数

        Returns:
            校准后的 (y_pred_reg_calibrated, s_rank_calibrated)
        """
        # 校准回归头
        if self.isotonic_reg is not None:
            y_pred_reg_cal = self.isotonic_reg.predict(y_pred_reg)
        else:
            y_pred_reg_cal = y_pred_reg

        # 🔧 简化排序头校准 - 直接使用rank percentile
        if self.rank_quantile_map is not None and self.rank_quantile_map.get('method') == 'simple_rank_pct':
            # 计算rank percentile（0-100）
            # argsort两次得到排名，除以(n-1)得到分位数
            # 注意：默认ascending=True，分数高→percentile高
            rank_pct = np.argsort(np.argsort(s_rank)) / max(len(s_rank) - 1, 1)
            s_rank_cal = rank_pct * 100  # 转为0-100范围
        else:
            # 兜底：直接使用原始分数
            s_rank_cal = s_rank

        return y_pred_reg_cal, s_rank_cal

    def _cross_sectional_zscore(self, values: pd.Series, dates: pd.Series) -> pd.Series:
        """
        横截面z-score标准化（每日内标准化）

        为什么横截面标准化？
        - 消除市场整体波动影响
        - 统一不同日期的预测刻度
        - 使回归头和排序头在同一标尺上

        Args:
            values: 待标准化的值
            dates: 日期序列

        Returns:
            z-score标准化后的值（长度与输入相同）
        """
        # Extract raw values and ensure proper alignment
        vals = values.values if hasattr(values, 'values') else values
        dts = dates.values if hasattr(dates, 'values') else dates

        input_len = len(vals)

        # Create DataFrame with explicit integer index
        df_temp = pd.DataFrame({
            'value': vals,
            'date': dts
        }, index=range(input_len))

        # Initialize result array with zeros
        z_scores_array = np.zeros(input_len)

        # Group by date and calculate z-scores
        for date_val, group in df_temp.groupby('date'):
            indices = group.index
            group_values = group['value'].values

            mean_val = np.mean(group_values)
            std_val = np.std(group_values)

            if std_val < 1e-10:
                # If std is too small, set to zero
                z_scores_array[indices] = 0.0
            else:
                # Calculate z-score
                z_scores_array[indices] = (group_values - mean_val) / std_val

        # Return as Series to maintain compatibility
        return pd.Series(z_scores_array, index=range(input_len))

    def _tanh_clip(self, z: np.ndarray, c: float = None) -> np.ndarray:
        """
        Tanh限幅：防止极端值破坏融合

        ẑ = tanh(z / c)

        Args:
            z: z-score
            c: 限幅参数（默认使用self.tanh_clip_c）

        Returns:
            限幅后的z-score
        """
        if c is None:
            c = self.tanh_clip_c
        return np.tanh(z / c)

    def tune_weights_on_oof(self, y_pred_reg_oof: pd.Series,
                            s_rank_oof: pd.Series,
                            y_true_oof: pd.Series,
                            dates_oof: pd.Series) -> Tuple[float, float]:
        """
        使用OOF数据网格搜索最优权重α, β

        优化目标：RankIC（Spearman相关系数）
        - 为什么用RankIC？既考虑magnitude又考虑ranking
        - 简单稳定，无需复杂的分层评估

        Args:
            y_pred_reg_oof: 回归头OOF预测
            s_rank_oof: 排序头OOF分数
            y_true_oof: 真实标签
            dates_oof: 日期序列

        Returns:
            (best_alpha, best_beta)
        """
        logger.info("🎯 开始OOF权重调参...")

        # 先校准
        valid_mask = (y_pred_reg_oof.notna() &
                     s_rank_oof.notna() &
                     y_true_oof.notna())

        y_pred_reg_val = y_pred_reg_oof[valid_mask].values
        s_rank_val = s_rank_oof[valid_mask].values
        y_true_val = y_true_oof[valid_mask].values
        dates_val = dates_oof[valid_mask]

        # 应用校准
        y_pred_reg_cal, s_rank_cal = self._apply_calibration(y_pred_reg_val, s_rank_val)

        # 横截面z-score - pass as Series to maintain compatibility
        z_reg = self._cross_sectional_zscore(
            pd.Series(y_pred_reg_cal),
            pd.Series(dates_val.values if hasattr(dates_val, 'values') else dates_val)
        ).values
        z_rank = self._cross_sectional_zscore(
            pd.Series(s_rank_cal),
            pd.Series(dates_val.values if hasattr(dates_val, 'values') else dates_val)
        ).values

        # Tanh限幅
        z_reg = self._tanh_clip(z_reg)
        z_rank = self._tanh_clip(z_rank)

        # 网格搜索
        best_alpha = self.alpha
        best_score = -999
        tuning_results = []

        for alpha_candidate in self.alpha_grid:
            beta_candidate = 1.0 - alpha_candidate

            # 融合
            S = alpha_candidate * z_reg + beta_candidate * z_rank

            # 评估RankIC
            try:
                rank_ic, _ = spearmanr(S, y_true_val)
                if np.isnan(rank_ic):
                    rank_ic = 0.0
            except:
                rank_ic = 0.0

            tuning_results.append({
                'alpha': alpha_candidate,
                'beta': beta_candidate,
                'rank_ic': rank_ic
            })

            logger.info(f"   α={alpha_candidate:.2f}, β={beta_candidate:.2f}: RankIC={rank_ic:.4f}")

            if rank_ic > best_score:
                best_score = rank_ic
                best_alpha = alpha_candidate

        best_beta = 1.0 - best_alpha

        self.tuning_results_ = pd.DataFrame(tuning_results)

        logger.info(f"✅ 最优权重: α={best_alpha:.2f}, β={best_beta:.2f}, RankIC={best_score:.4f}")

        return best_alpha, best_beta

    def fit(self, ridge_oof_preds: pd.Series,
            lambda_oof_preds: pd.Series,
            y_true: pd.Series,
            dates: pd.Series) -> 'DualHeadLateFusion':
        """
        训练双头融合系统

        Args:
            ridge_oof_preds: Ridge回归头OOF预测
            lambda_oof_preds: Lambda排序头OOF预测
            y_true: 真实标签
            dates: 日期序列

        Returns:
            self
        """
        logger.info("="*80)
        logger.info("🚀 开始训练双头晚融合系统")
        logger.info("="*80)

        # Step 1: 校准两个头
        self.calibrate_regression_head(ridge_oof_preds, y_true)
        self.calibrate_ranking_head(lambda_oof_preds, y_true, dates)

        # Step 2: 自动调参（如果启用）
        if self.auto_tune_weights:
            self.best_alpha_, self.best_beta_ = self.tune_weights_on_oof(
                ridge_oof_preds, lambda_oof_preds, y_true, dates
            )
        else:
            self.best_alpha_ = self.alpha
            self.best_beta_ = self.beta
            logger.info(f"使用固定权重: α={self.best_alpha_:.2f}, β={self.best_beta_:.2f}")

        self.fitted_ = True

        logger.info("="*80)
        logger.info("✅ 双头晚融合系统训练完成")
        logger.info(f"   最终权重: α={self.best_alpha_:.2f} (回归), β={self.best_beta_:.2f} (排序)")
        logger.info("="*80)

        return self

    def predict(self, ridge_preds: pd.Series,
                lambda_preds: pd.Series,
                dates: pd.Series) -> pd.Series:
        """
        使用双头融合系统进行预测

        Args:
            ridge_preds: Ridge回归头预测
            lambda_preds: Lambda排序头预测
            dates: 日期序列

        Returns:
            融合后的最终分数 S
        """
        if not self.fitted_:
            raise RuntimeError("模型未训练，请先调用fit()")

        logger.info("🔮 双头融合预测开始...")

        # 移除NaN
        valid_mask = ridge_preds.notna() & lambda_preds.notna()
        y_reg = ridge_preds[valid_mask].values
        s_rank = lambda_preds[valid_mask].values
        dates_val = dates[valid_mask]

        # Step 1: 校准
        y_reg_cal, s_rank_cal = self._apply_calibration(y_reg, s_rank)

        # Step 2: 横截面z-score - pass as Series to maintain compatibility
        z_reg = self._cross_sectional_zscore(
            pd.Series(y_reg_cal),
            pd.Series(dates_val.values if hasattr(dates_val, 'values') else dates_val)
        ).values
        z_rank = self._cross_sectional_zscore(
            pd.Series(s_rank_cal),
            pd.Series(dates_val.values if hasattr(dates_val, 'values') else dates_val)
        ).values

        # Step 3: Tanh限幅
        z_reg = self._tanh_clip(z_reg)
        z_rank = self._tanh_clip(z_rank)

        # Step 4: 固定线性加权融合
        S = self.best_alpha_ * z_reg + self.best_beta_ * z_rank

        # Verify lengths match before assignment
        n_valid = valid_mask.sum()
        if len(S) != n_valid:
            logger.warning(f"⚠️ 长度不匹配: S长度={len(S)}, valid_mask计数={n_valid}")
            logger.warning(f"   输入: y_reg={len(y_reg)}, dates_val={len(dates_val)}")
            logger.warning(f"   z-score后: z_reg={len(z_reg)}, z_rank={len(z_rank)}")
            # Truncate or pad to match
            if len(S) > n_valid:
                S = S[:n_valid]
            else:
                S_padded = np.full(n_valid, np.nan)
                S_padded[:len(S)] = S
                S = S_padded

        # 构建完整结果（包括NaN）
        result = pd.Series(np.nan, index=ridge_preds.index)
        result[valid_mask] = S

        logger.info(f"✅ 融合预测完成")
        logger.info(f"   有效样本: {len(S)}/{len(ridge_preds)} ({len(S)/len(ridge_preds)*100:.1f}%)")
        logger.info(f"   分数范围: [{S.min():.4f}, {S.max():.4f}]")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'fitted': self.fitted_,
            'best_alpha': float(self.best_alpha_),
            'best_beta': float(self.best_beta_),
            'isotonic_calibration': self.isotonic_reg is not None,
            'rank_quantile_mapping': self.rank_quantile_map is not None,
            'tanh_clip_c': float(self.tanh_clip_c),
            'tuning_results': self.tuning_results_.to_dict() if self.tuning_results_ is not None else None
        }
