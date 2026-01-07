#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Head Integration Module - 双头融合集成模块
==============================================

无缝集成双头晚融合到现有BMA系统，最小化代码改动。

核心功能：
1. 提取Ridge和Lambda的OOF预测
2. 训练双头融合模型
3. 在预测时应用融合
4. 与现有ExcelExporter兼容

作者: BMA Enhanced System
日期: 2025-01-XX
版本: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

try:
    from bma_models.dual_head_late_fusion import DualHeadLateFusion
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

logger = logging.getLogger(__name__)


class DualHeadFusionManager:
    """
    双头融合管理器 - 集成到BMA系统的桥接类

    职责：
    1. 从训练结果中提取OOF预测
    2. 训练双头融合模型
    3. 管理融合模型的保存和加载
    4. 在预测时应用融合
    """

    def __init__(self,
                 enable_fusion: bool = True,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 auto_tune: bool = True):
        """
        初始化双头融合管理器

        Args:
            enable_fusion: 是否启用双头融合
            alpha: 回归头权重（默认0.7）
            beta: 排序头权重（默认0.3）
            auto_tune: 是否自动调参
        """
        if not FUSION_AVAILABLE:
            logger.warning("⚠️ DualHeadLateFusion模块不可用，将禁用双头融合")
            self.enable_fusion = False
        else:
            self.enable_fusion = enable_fusion

        self.alpha = alpha
        self.beta = beta
        self.auto_tune = auto_tune

        # 融合模型
        self.fusion_model: Optional[DualHeadLateFusion] = None

        # 训练状态
        self.fitted_ = False

        if self.enable_fusion:
            logger.info("✅ 双头融合管理器初始化完成")
            logger.info(f"   初始权重: α={alpha:.2f}, β={beta:.2f}")
            logger.info(f"   自动调参: {auto_tune}")
        else:
            logger.info("⏭️ 双头融合已禁用")

    def extract_oof_predictions(self, training_results: Dict[str, Any], y_true: Optional[pd.Series] = None, dates: Optional[pd.Series] = None) -> Optional[Dict[str, pd.Series]]:
        """
        从训练结果中提取OOF预测

        Args:
            training_results: 训练结果字典（来自_unified_parallel_training）

        Returns:
            包含ridge_oof和lambda_oof的字典，如果提取失败返回None
        """
        if not self.enable_fusion:
            return None

        logger.info("📊 开始提取OOF预测...")

        try:
            import pandas as pd
            from typing import Dict as _Dict
            ridge_oof = None

            # 1) 优先从标准位置获取第一层OOF
            oof_preds: _Dict[str, pd.Series] = {}
            if 'oof_predictions' in training_results and isinstance(training_results['oof_predictions'], dict):
                oof_preds = training_results['oof_predictions']
            elif 'traditional_models' in training_results and isinstance(training_results['traditional_models'], dict):
                oof_preds = training_results['traditional_models'].get('oof_predictions', {}) or {}

            # 2) 若可用，基于第一层OOF和时间CV生成 Ridge 的 OOF（无泄漏）
            if ridge_oof is None and oof_preds and y_true is not None and dates is not None:
                try:
                    # 仅使用三回归头
                    base_keys = ['elastic_net', 'xgboost', 'catboost']
                    missing = [k for k in base_keys if k not in oof_preds]
                    if not missing:
                        # 组装特征矩阵（保持索引）
                        X_df = pd.DataFrame({
                            'pred_elastic': pd.Series(oof_preds['elastic_net']).reindex(y_true.index),
                            'pred_xgb': pd.Series(oof_preds['xgboost']).reindex(y_true.index),
                            'pred_catboost': pd.Series(oof_preds['catboost']).reindex(y_true.index),
                        })
                        y_vec = pd.Series(y_true).reindex(y_true.index)

                        # 时间安全CV
                        try:
                            from bma_models.unified_purged_cv_factory import create_unified_cv
                            cv = create_unified_cv(n_splits=6, gap=5, embargo=5)
                            # 将日期映射为组索引
                            date_index = y_true.index.get_level_values('date') if isinstance(y_true.index, pd.MultiIndex) else pd.to_datetime(dates)
                            date_codes = pd.Series(date_index).astype('category').cat.codes.values
                            splits = list(cv.split(X_df.values, y_vec.values, groups=date_codes))
                        except Exception:
                            # 回退：简单KFold（非理想，但保证不崩）
                            from sklearn.model_selection import KFold
                            kf = KFold(n_splits=6, shuffle=False)
                            splits = list(kf.split(X_df.values, y_vec.values))

                        # 逐折训练Ridge并产出OOF
                        import numpy as np
                        from sklearn.linear_model import Ridge
                        ridge_oof_arr = np.full(len(X_df), np.nan)
                        for tr_idx, va_idx in splits:
                            X_tr, X_va = X_df.values[tr_idx], X_df.values[va_idx]
                            y_tr = y_vec.values[tr_idx]
                            model = Ridge(alpha=1.0, fit_intercept=False)
                            model.fit(X_tr, y_tr)
                            ridge_oof_arr[va_idx] = model.predict(X_va)
                        ridge_oof = pd.Series(ridge_oof_arr, index=y_true.index, name='ridge_oof')
                        logger.info(f"   ✓ Ridge OOF(derived): {ridge_oof.notna().sum()} samples")
                    else:
                        logger.warning(f"   ⚠️ 缺少第一层OOF: {missing}，无法派生Ridge OOF")
                except Exception as e:
                    logger.warning(f"   ⚠️ 基于第一层OOF生成Ridge OOF失败: {e}")

            # 提取Lambda OOF（优先使用统一OOF字典）
            lambda_oof = None
            if oof_preds and 'lambdarank' in oof_preds:
                lambda_oof = pd.Series(oof_preds['lambdarank'], name='lambda_oof')
                logger.info(f"   ✓ Lambda OOF: {len(lambda_oof)} samples")
            else:
                # 次选：从models结构或模型内存取
                if 'models' in training_results and 'lambdarank' in training_results['models']:
                    lambda_info = training_results['models']['lambdarank']
                    if 'oof_predictions' in lambda_info:
                        lambda_oof = pd.Series(lambda_info['oof_predictions'], name='lambda_oof')
                        logger.info(f"   ✓ Lambda OOF: {len(lambda_oof)} samples")
                    elif 'model' in lambda_info:
                        model = lambda_info['model']
                        if hasattr(model, '_oof_predictions') and model._oof_predictions is not None:
                            lambda_oof = pd.Series(model._oof_predictions, name='lambda_oof')
                            logger.info(f"   ✓ Lambda OOF (from model): {len(lambda_oof)} samples")

            # 验证
            if ridge_oof is None or lambda_oof is None:
                logger.error("❌ 无法提取完整的OOF预测")
                logger.error(f"   Ridge OOF: {'✓' if ridge_oof is not None else '✗'}")
                logger.error(f"   Lambda OOF: {'✓' if lambda_oof is not None else '✗'}")
                return None

            # 确保索引对齐
            if not isinstance(ridge_oof, pd.Series):
                ridge_oof = pd.Series(ridge_oof, name='ridge_oof')
            if not isinstance(lambda_oof, pd.Series):
                lambda_oof = pd.Series(lambda_oof, name='lambda_oof')

            logger.info("✅ OOF预测提取成功")
            return {
                'ridge_oof': ridge_oof,
                'lambda_oof': lambda_oof
            }

        except Exception as e:
            logger.error(f"❌ 提取OOF预测时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fit_fusion_model(self,
                         ridge_oof: pd.Series,
                         lambda_oof: pd.Series,
                         y_true: pd.Series,
                         dates: pd.Series) -> bool:
        """
        训练双头融合模型

        Args:
            ridge_oof: Ridge回归头OOF预测
            lambda_oof: Lambda排序头OOF预测
            y_true: 真实标签
            dates: 日期序列

        Returns:
            训练是否成功
        """
        if not self.enable_fusion:
            return False

        logger.info("="*80)
        logger.info("🔥 开始训练双头融合模型")
        logger.info("="*80)

        try:
            # 初始化融合模型
            self.fusion_model = DualHeadLateFusion(
                alpha=self.alpha,
                beta=self.beta,
                auto_tune_weights=self.auto_tune,
                use_isotonic_calibration=False,  # 🔧 关闭Isotonic：改善小(1%)且后续z-score会覆盖
                tanh_clip_c=2.5
            )

            # 训练
            self.fusion_model.fit(
                ridge_oof_preds=ridge_oof,
                lambda_oof_preds=lambda_oof,
                y_true=y_true,
                dates=dates
            )

            self.fitted_ = True

            # 输出融合模型信息
            fusion_info = self.fusion_model.get_model_info()
            logger.info("="*80)
            logger.info("✅ 双头融合模型训练完成")
            logger.info(f"   最优权重: α={fusion_info['best_alpha']:.3f} (回归), β={fusion_info['best_beta']:.3f} (排序)")
            logger.info(f"   Isotonic校准: {'启用' if fusion_info['isotonic_calibration'] else '禁用'}")
            logger.info(f"   分位数映射: {'启用' if fusion_info['rank_quantile_mapping'] else '禁用'}")
            logger.info("="*80)

            return True

        except Exception as e:
            logger.error(f"❌ 训练双头融合模型时出错: {e}")
            import traceback
            traceback.print_exc()
            self.fitted_ = False
            return False

    def apply_fusion(self,
                     ridge_preds: pd.Series,
                     lambda_preds: pd.Series,
                     dates: pd.Series) -> Optional[pd.Series]:
        """
        应用融合：在预测时使用

        Args:
            ridge_preds: Ridge预测
            lambda_preds: Lambda预测
            dates: 日期序列

        Returns:
            融合后的最终分数，如果失败返回None
        """
        if not self.enable_fusion or not self.fitted_:
            logger.warning("⚠️ 融合模型未训练或已禁用，返回Ridge预测")
            return ridge_preds

        try:
            fused_scores = self.fusion_model.predict(
                ridge_preds=ridge_preds,
                lambda_preds=lambda_preds,
                dates=dates
            )

            logger.info(f"✅ 双头融合预测完成: {fused_scores.notna().sum()}/{len(fused_scores)} 有效样本")
            return fused_scores

        except Exception as e:
            logger.error(f"❌ 应用融合时出错: {e}")
            logger.warning("   降级到Ridge预测")
            return ridge_preds

    def integrate_with_training(self, training_results: Dict[str, Any],
                                y_true: pd.Series,
                                dates: pd.Series) -> Dict[str, Any]:
        """
        与现有训练流程集成的主要接口

        在训练完成后调用，自动提取OOF并训练融合模型

        Args:
            training_results: 训练结果字典
            y_true: 真实标签
            dates: 日期序列

        Returns:
            更新后的训练结果（添加fusion_model字段）
        """
        if not self.enable_fusion:
            logger.info("⏭️ 双头融合已禁用，跳过集成")
            return training_results

        logger.info("="*80)
        logger.info("🔗 集成双头融合到训练流程")
        logger.info("="*80)

        # Step 1: 提取OOF预测
        oof_dict = self.extract_oof_predictions(training_results, y_true=y_true, dates=dates)
        if oof_dict is None:
            logger.error("❌ OOF提取失败，无法训练融合模型")
            return training_results

        ridge_oof = oof_dict['ridge_oof']
        lambda_oof = oof_dict['lambda_oof']

        # Step 2: 训练融合模型
        success = self.fit_fusion_model(
            ridge_oof=ridge_oof,
            lambda_oof=lambda_oof,
            y_true=y_true,
            dates=dates
        )

        if success:
            # 将融合模型添加到训练结果中
            training_results['fusion_model'] = {
                'model': self.fusion_model,
                'manager': self,
                'fitted': True,
                'info': self.fusion_model.get_model_info()
            }
            logger.info("✅ 双头融合已成功集成到训练结果")
        else:
            logger.error("❌ 双头融合集成失败")

        return training_results

    def get_info(self) -> Dict[str, Any]:
        """获取管理器信息"""
        return {
            'enable_fusion': self.enable_fusion,
            'fitted': self.fitted_,
            'alpha': self.alpha,
            'beta': self.beta,
            'auto_tune': self.auto_tune,
            'fusion_model_info': self.fusion_model.get_model_info() if self.fusion_model else None
        }


def enable_dual_head_fusion_for_model(model_instance,
                                      enable: bool = True,
                                      alpha: float = 0.7,
                                      beta: float = 0.3,
                                      auto_tune: bool = True):
    """
    为现有模型实例启用双头融合

    这是一个便捷函数，用于快速为BMA模型添加双头融合功能

    Args:
        model_instance: BMA模型实例（量化模型_bma_ultra_enhanced的实例）
        enable: 是否启用融合
        alpha: 回归头权重
        beta: 排序头权重
        auto_tune: 是否自动调参

    Example:
        >>> model = QuantitativeModel(...)
        >>> enable_dual_head_fusion_for_model(model, enable=True)
        >>> # 之后训练会自动使用双头融合
    """
    fusion_manager = DualHeadFusionManager(
        enable_fusion=enable,
        alpha=alpha,
        beta=beta,
        auto_tune=auto_tune
    )

    # 将管理器附加到模型实例
    model_instance.dual_head_fusion_manager = fusion_manager

    logger.info("✅ 双头融合已启用")
    logger.info("   在run_complete_analysis或train_models完成后，融合模型将自动训练")

    return model_instance
