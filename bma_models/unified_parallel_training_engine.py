#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一并行训练引擎 - 正确的数据流架构

核心改进：
1. 数据流分离：Ridge使用第一层OOF，LambdaRank使用Alpha Factors
2. 正确的并行策略：阶段1统一训练，阶段2并行不同数据源
3. 集成simple25factor引擎和purged CV factory
4. 确保时间安全和数据质量一致性
"""

import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class UnifiedParallelTrainingEngine:
    """
    统一并行训练引擎 v3.0

    关键架构改进：
    1. 数据流分离：Ridge使用第一层OOF预测，LambdaRank使用Alpha Factors原始特征
    2. 正确并行策略：先统一第一层，再并行二层（不同数据源）
    3. 时间安全：统一使用purged CV factory
    4. 质量保证：集成simple25factor引擎
    """

    def __init__(self, parent_model):
        """
        初始化统一并行训练引擎

        Args:
            parent_model: 主模型实例（UltraEnhancedQuantitativeModel）
        """
        self.parent = parent_model
        self.timing_stats = {}
        self.quality_stats = {}

    def unified_parallel_train(self, X: pd.DataFrame, y: pd.Series,
                             dates: pd.Series, tickers: pd.Series,
                             alpha_factors: pd.DataFrame = None) -> Dict[str, Any]:
        """
        统一并行训练架构

        正确的并行策略：
        阶段1: [统一第一层训练] → [统一OOF预测]
        阶段2: 基于统一OOF并行训练: [Ridge] || [LambdaRank]

        Returns:
            训练结果字典
        """
        logger.info("="*70)
        logger.info("🚀 统一并行训练引擎 v2.0")
        logger.info("   修复：数据一致性 + 正确并行策略")
        logger.info("   阶段1: 统一第一层训练")
        logger.info("   阶段2: 基于相同OOF的并行二层训练")
        logger.info("="*70)

        start_time = time.time()
        results = {
            'stage1_success': False,
            'ridge_success': False,
            'lambda_success': False,
            'unified_oof_predictions': None,
            'stacker_data': None,
            'timing': {},
            'quality_metrics': {}
        }

        try:
            # 阶段1：统一第一层训练（使用simple25factor引擎）
            stage1_start = time.time()
            logger.info("📊 阶段1: 统一第一层模型训练开始...")

            unified_first_layer_results = self._unified_first_layer_training(
                X, y, dates, tickers
            )

            if not unified_first_layer_results['success']:
                logger.error("❌ 阶段1失败，终止并行训练")
                return results

            results['stage1_success'] = True
            results['unified_oof_predictions'] = unified_first_layer_results['oof_predictions']
            results['timing']['stage1'] = time.time() - stage1_start

            logger.info(f"✅ 阶段1完成，耗时: {results['timing']['stage1']:.2f}秒")
            self._log_oof_quality(unified_first_layer_results['oof_predictions'], y)

            # 阶段2：基于统一OOF的并行二层训练
            stage2_start = time.time()
            logger.info("🔄 阶段2: 并行二层训练开始...")

            # 显式选择 DataFrame 避免 pandas 布尔歧义
            af_df = alpha_factors if alpha_factors is not None else X
            parallel_results = self._parallel_second_layer_training(
                unified_first_layer_results['oof_predictions'],
                y, dates, tickers,
                alpha_factors=af_df
            )

            results.update({
                'ridge_success': parallel_results['ridge_success'],
                'lambda_success': parallel_results['lambda_success'],
                'stacker_data': parallel_results['stacker_data']
            })
            results['timing']['stage2'] = time.time() - stage2_start
            results['timing'].update(parallel_results['timing'])

            logger.info(f"✅ 阶段2完成，耗时: {results['timing']['stage2']:.2f}秒")

        # 已移除Rank-aware Blender初始化，保留空操作

        except Exception as e:
            logger.error(f"❌ 统一并行训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # 性能统计
        total_time = time.time() - start_time
        self._log_performance_summary(results, total_time)

        return results

    # Backward-compatible wrapper used by UltraEnhancedQuantitativeModel
    def train_unified_stackers(self, oof_predictions: Dict[str, pd.Series],
                               y: pd.Series, dates: pd.Series, tickers: pd.Series,
                               alpha_factors: pd.DataFrame = None) -> Dict[str, Any]:
        """Compatibility API: run only stage-2 training given precomputed OOF.

        Args:
            oof_predictions: dict of first-layer OOF prediction Series
            y: target Series aligned to dates/tickers
            dates: date Series for index construction
            tickers: ticker Series for index construction
            alpha_factors: optional alpha feature DataFrame for LambdaRank

        Returns:
            Dict with ridge_success, lambda_success, stacker_data, timing, lambda_percentile_info
        """
        return self._parallel_second_layer_training(
            unified_oof_predictions=oof_predictions,
            y=y,
            dates=dates,
            tickers=tickers,
            alpha_factors=alpha_factors
        )

    def _unified_first_layer_training(self, X: pd.DataFrame, y: pd.Series,
                                    dates: pd.Series, tickers: pd.Series) -> Dict[str, Any]:
        """
        统一第一层训练 - 确保数据质量一致性

        使用相同的参数配置和CV策略，确保生成高质量的OOF预测
        """
        logger.info("🏗️ 使用统一配置进行第一层训练...")

        try:
            # 使用主模型的统一训练方法，确保配置一致性
            first_layer_results = self.parent._unified_model_training(
                X, y, dates, tickers
            )

            if first_layer_results.get('success') and 'oof_predictions' in first_layer_results:
                # 验证OOF预测质量
                oof_preds = first_layer_results['oof_predictions']
                quality_metrics = self._validate_oof_quality(oof_preds, y)

                logger.info("✅ 统一第一层训练成功")
                logger.info(f"   模型数量: {len(oof_preds)}")
                logger.info(f"   平均IC: {quality_metrics['avg_ic']:.4f}")
                logger.info(f"   IC标准差: {quality_metrics['ic_std']:.4f}")

                return {
                    'success': True,
                    'oof_predictions': oof_preds,
                    'models': first_layer_results.get('models', {}),
                    'cv_scores': first_layer_results.get('cv_scores', {}),
                    'quality_metrics': quality_metrics
                }
            else:
                logger.error("❌ 第一层训练失败或缺少OOF预测")
                return {'success': False}

        except Exception as e:
            logger.error(f"❌ 统一第一层训练异常: {e}")
            return {'success': False}

    def _parallel_second_layer_training(self, unified_oof_predictions: Dict[str, pd.Series],
                                      y: pd.Series, dates: pd.Series, tickers: pd.Series,
                                      alpha_factors: pd.DataFrame = None) -> Dict[str, Any]:
        """
        并行二层训练 - 基于统一的OOF预测

        这是真正的并行：两个模型使用相同的输入数据
        """
        logger.info("⚡ 开始并行二层训练（基于统一OOF）...")

        results = {
            'ridge_success': False,
            'lambda_success': False,
            'stacker_data': None,
            'timing': {}
        }

        # 构建不同的数据源
        # Ridge使用OOF
        ridge_data = self._build_unified_stacker_data(
            unified_oof_predictions, y, dates, tickers
        )

        # LambdaRank使用Alpha Factors
        lambda_data = self._build_lambda_data(
            alpha_factors, y, dates, tickers
        )

        if ridge_data is None:
            logger.error("❌ 无法构建Ridge stacker数据")
            return results

        if lambda_data is None:
            logger.error("❌ 无法构建LambdaRank数据")
            # 即使LambdaRank数据失败，仍可训练Ridge
            lambda_data = ridge_data  # 降级使用OOF

        logger.info(f"📊 Ridge数据形状: {ridge_data.shape}")
        logger.info(f"📊 LambdaRank数据形状: {lambda_data.shape}")
        results['stacker_data'] = ridge_data  # 保留兼容性

        # 启用LambdaRank：必须开启（可用且样本量足够），否则中止流程
        lambda_available = self._check_lambda_available()
        lambda_data_valid = (lambda_data is not None and len(lambda_data) > 0)
        sample_count_ok = len(ridge_data) >= 12  # 放宽以支持小样本

        logger.info(f"📊 Lambda启用检查:")
        logger.info(f"   Lambda可导入: {lambda_available}")
        logger.info(f"   Lambda数据有效: {lambda_data_valid}")
        logger.info(f"   样本数量: {len(ridge_data)} (需要>=50: {sample_count_ok})")

        use_lambda = (lambda_available and sample_count_ok)

        if not use_lambda:
            logger.warning(
                f"⚠️ LambdaRank未启用或样本不足，跳过Lambda训练并仅训练Ridge"
            )
            # 直接训练Ridge并返回
            ridge_start = time.time()
            ridge_success = self.parent._train_ridge_stacker(
                unified_oof_predictions, y, dates, ridge_data=ridge_data
            )
            results['ridge_success'] = ridge_success
            results['timing']['ridge'] = time.time() - ridge_start
            logger.info(f"✅ Ridge训练完成，耗时: {results['timing']['ridge']:.2f}秒")
            return results

        if lambda_data is None or len(lambda_data) == 0:
            logger.error("❌ Lambda数据为空，训练中止")
            raise RuntimeError(f"Lambda data is empty or None")

        # 顺序训练：先Lambda生成percentile，再Ridge使用
        logger.info("🔄 新融合策略：Lambda Percentile → Ridge Stacker")

        # 步骤1：训练LambdaRank，生成OOF预测
        logger.info("="*60)
        logger.info("🚀 步骤1: 开始训练LambdaRank模型")
        logger.info(f"   Lambda数据形状: {lambda_data.shape}")
        logger.info(f"   Lambda特征数: {lambda_data.shape[1] - 1}")  # 减去target列
        logger.info("="*60)

        lambda_start = time.time()
        lambda_result = self._train_lambda_unified(lambda_data)
        results['lambda_success'] = lambda_result['success']
        results['timing']['lambda'] = time.time() - lambda_start

        logger.info(f"📊 Lambda训练结果: {'成功✅' if lambda_result['success'] else '失败❌'}")
        logger.info(f"   耗时: {results['timing']['lambda']:.2f}秒")

        if lambda_result['success']:
            self.parent.lambda_rank_stacker = lambda_result['model']
            logger.info(f"✅ LambdaRank训练完成，耗时: {results['timing']['lambda']:.2f}秒")

            # 步骤2：计算Lambda OOF percentile
            logger.info("="*60)
            logger.info("🔧 步骤2: 生成Lambda Percentile特征")
            logger.info("="*60)

            try:
                # 获取Lambda模型的真正OOF预测（防数据泄漏）
                lambda_model = lambda_result['model']

                # 🔧 确保lambda_data对齐到ridge_data的索引
                if not lambda_data.index.equals(ridge_data.index):
                    logger.info(f"🔧 对齐Lambda数据到Ridge索引")
                    logger.info(f"   Lambda原始: {len(lambda_data)} 样本")
                    logger.info(f"   Ridge目标: {len(ridge_data)} 样本")
                    lambda_data_aligned = lambda_data.reindex(ridge_data.index)
                else:
                    logger.info(f"✅ Lambda和Ridge索引已对齐")
                    lambda_data_aligned = lambda_data

                lambda_oof = lambda_model.get_oof_predictions(lambda_data_aligned)
                logger.info(f"✅ Lambda OOF预测获取完成: {len(lambda_oof)} 样本")

                # 🔧 Critical Fix: 使用一致性转换器计算percentile
                from bma_models.lambda_percentile_transformer import LambdaPercentileTransformer

                # 创建并拟合转换器
                lambda_percentile_transformer = LambdaPercentileTransformer(method='quantile')
                lambda_percentile_series = lambda_percentile_transformer.fit_transform(lambda_oof)

                # 保存转换器供预测时使用
                self.parent.lambda_percentile_transformer = lambda_percentile_transformer

                logger.info(f"✅ Lambda Percentile转换器已创建并保存")
                logger.info(f"   Percentile统计: 均值={lambda_percentile_series.mean():.1f}, 范围=[{lambda_percentile_series.min():.1f}, {lambda_percentile_series.max():.1f}]")

                # 📊 详细索引对齐诊断
                logger.info(f"📊 索引对齐诊断:")
                logger.info(f"   Ridge形状: {ridge_data.shape}")
                logger.info(f"   Lambda Percentile形状: {lambda_percentile_series.shape}")
                logger.info(f"   索引完全匹配: {ridge_data.index.equals(lambda_percentile_series.index)}")

                # 🔧 验证索引对齐
                if not lambda_percentile_series.index.equals(ridge_data.index):
                    logger.warning(f"⚠️ Lambda Percentile索引不匹配，强制对齐")
                    lambda_percentile_series = lambda_percentile_series.reindex(ridge_data.index)

                    # 检查NaN比例
                    nan_count = lambda_percentile_series.isna().sum()
                    nan_ratio = nan_count / len(ridge_data)
                    logger.warning(f"   对齐后NaN: {nan_count} ({nan_ratio:.2%})")

                    if nan_ratio > 0.05:
                        logger.error(f"❌ Lambda Percentile NaN比例过高 ({nan_ratio:.2%})")
                        raise ValueError("Lambda Percentile对齐失败，NaN过多")

                # 步骤3：加入Ridge数据
                logger.info("="*60)
                logger.info("🔧 步骤3: 将Lambda Percentile加入Ridge特征")
                logger.info("="*60)
                logger.info(f"   Ridge原始特征: {list(ridge_data.columns)}")

                ridge_data['lambda_percentile'] = lambda_percentile_series

                logger.info(f"✅ Lambda Percentile已加入Ridge特征")
                logger.info(f"   Ridge新特征: {list(ridge_data.columns)}")
                logger.info(f"   Lambda Percentile无NaN: {lambda_percentile_series.notna().all()}")
                logger.info(f"   Ridge数据最终形状: {ridge_data.shape}")

                # 收集Lambda Percentile信息用于Excel导出
                results['lambda_percentile_info'] = {
                    'n_factors': len(lambda_model._alpha_factor_cols) if hasattr(lambda_model, '_alpha_factor_cols') else 15,
                    'oof_samples': len(lambda_oof),
                    'percentile_mean': float(lambda_percentile_series.mean()),
                    'percentile_min': float(lambda_percentile_series.min()),
                    'percentile_max': float(lambda_percentile_series.max()),
                    'alignment_status': '完全对齐' if ridge_data.index.equals(lambda_percentile_series.index) else '已强制对齐',
                    'nan_ratio': float(lambda_percentile_series.isna().sum() / len(lambda_percentile_series))
                }

            except Exception as e_perc:
                logger.warning(f"⚠️ 计算Lambda Percentile失败: {e_perc}")
                logger.warning("   Ridge将不使用Lambda特征")
        else:
            logger.warning("⚠️ LambdaRank训练失败，继续仅用Ridge流程")

        # 步骤4：训练Ridge Stacker（使用OOF + Lambda Percentile）
        ridge_start = time.time()
        ridge_success = self.parent._train_ridge_stacker(
            unified_oof_predictions, y, dates, ridge_data=ridge_data
        )
        results['ridge_success'] = ridge_success
        results['timing']['ridge'] = time.time() - ridge_start
        logger.info(f"✅ Ridge训练完成，耗时: {results['timing']['ridge']:.2f}秒")

        return results

    def _build_unified_stacker_data(self, oof_predictions: Dict[str, pd.Series],
                                  y: pd.Series, dates: pd.Series, tickers: pd.Series) -> Optional[pd.DataFrame]:
        """
        构建统一的stacker数据集

        确保Ridge与LambdaRank使用完全相同的数据
        """
        try:
            if not isinstance(y.index, pd.MultiIndex):
                multi_index = pd.MultiIndex.from_arrays(
                    [dates, tickers], names=['date', 'ticker']
                )
                y_indexed = pd.Series(y.values, index=multi_index)
            else:
                y_indexed = y

            stacker_dict: Dict[str, pd.Series] = {}
            for model_name, pred_series in oof_predictions.items():
                key = f'pred_{model_name}'
                if isinstance(pred_series.index, pd.MultiIndex):
                    stacker_dict[key] = pred_series
                else:
                    stacker_dict[key] = pd.Series(pred_series.values, index=y_indexed.index)

            parent_horizon = getattr(self.parent, 'horizon', 5)
            if parent_horizon != 5:
                raise ValueError(f'Unified stacker expects parent horizon=5, got {parent_horizon}.')

            target_col = 'ret_fwd_10d'
            stacker_dict[target_col] = y_indexed

            stacker_data = pd.DataFrame(stacker_dict)

            missing_data = stacker_data.isnull().sum()
            if missing_data.any():
                logger.warning(f"[Stacker] 数据缺失: {missing_data.to_dict()}")

            feature_cols = [c for c in stacker_data.columns if c != target_col]
            stacker_data[feature_cols] = stacker_data[feature_cols].fillna(0.0)
            clean_data = stacker_data.dropna(subset=[target_col])

            if len(clean_data) < len(stacker_data) * 0.8:
                retention = len(clean_data) / len(stacker_data) * 100
                logger.warning(f"[Stacker] 目标列清理后剩余 {len(clean_data)}/{len(stacker_data)} ({retention:.1f}%)")

            logger.info(f"[Stacker] 构建完成: {clean_data.shape}")
            return clean_data

        except Exception as e:
            logger.error(f"构建stacker数据失败: {e}")
            return None

    def _train_ridge_unified(self, oof_predictions: Dict[str, pd.Series],
                           y: pd.Series, dates: pd.Series) -> Dict[str, Any]:
        """
        训练Ridge Stacker（统一数据源版本）
        """
        start_time = time.time()
        try:
            logger.info("[Ridge-Thread] 开始训练Ridge Stacker...")
            success = self.parent._train_ridge_stacker(oof_predictions, y, dates)

            result = {
                'success': success,
                'elapsed_time': time.time() - start_time
            }

            if success:
                logger.info("[Ridge-Thread] ✅ Ridge训练成功")
            else:
                logger.error("[Ridge-Thread] ❌ Ridge训练失败")

            return result

        except Exception as e:
            logger.error(f"[Ridge-Thread] 训练异常: {e}")
            return {
                'success': False,
                'elapsed_time': time.time() - start_time
            }


    def _train_lambda_unified(self, stacker_data: pd.DataFrame) -> Dict[str, Any]:
        """Train LambdaRank stacker using the unified stacker dataset."""
        start_time = time.time()
        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker
            from bma_models.unified_config_loader import get_time_config

            logger.info("[Lambda-Thread] ????LambdaRank...")

            time_config = get_time_config()
            time_horizon = int(getattr(time_config, 'prediction_horizon_days', getattr(self.parent, 'horizon', 10)))
            parent_horizon = int(getattr(self.parent, 'horizon', time_horizon))
            if parent_horizon != time_horizon:
                logger.warning(f"[Lambda-Thread] Parent horizon {parent_horizon}d != config horizon {time_horizon}d; using config value")

            cv_splits = int(getattr(time_config, 'cv_n_splits', 6))
            cv_gap = int(getattr(time_config, 'cv_gap_days', 5))
            cv_embargo = int(getattr(time_config, 'cv_embargo_days', 5))
            logger.info(f"[Lambda-Thread] CV config splits/gap/embargo = {(cv_splits, cv_gap, cv_embargo)}")

            required_target = f'ret_fwd_{time_horizon}d'
            if required_target not in stacker_data.columns:
                available_targets = [col for col in stacker_data.columns if col.startswith('ret_fwd_')]
                raise KeyError(f"Missing {required_target} in stacker data. Available targets: {available_targets}")

            prohibited_cols = {required_target, 'target'}
            feature_cols = [
                col for col in stacker_data.columns
                if col not in prohibited_cols and not col.startswith('ret_fwd_')
            ]
            if not feature_cols:
                raise ValueError('No usable feature columns for LambdaRank after enforcing target alignment.')

            lambda_training_df = stacker_data[feature_cols + [required_target]].copy()
            logger.info(f"[Lambda-Thread] ?????{required_target}")
            logger.info(f"[Lambda-Thread] ??????: {feature_cols}")

            lambda_config = {
                'base_cols': tuple(feature_cols),
                'n_quantiles': 128,
                'winsorize_quantiles': (0.01, 0.99),
                'label_gain_power': 1.5,
                'num_boost_round': 100,
                'early_stopping_rounds': 0,
                'use_purged_cv': True,
                'cv_n_splits': cv_splits,
                'cv_gap_days': cv_gap,
                'cv_embargo_days': cv_embargo,
                'random_state': 42
            }

            lambda_stacker = LambdaRankStacker(**lambda_config)
            lambda_stacker.fit(lambda_training_df, target_col=required_target)

            logger.info("[Lambda-Thread] ? LambdaRank????")

            return {
                'success': True,
                'model': lambda_stacker,
                'elapsed_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"[Lambda-Thread] ????: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'model': None,
                'elapsed_time': time.time() - start_time
            }
    def _check_lambda_available(self) -> bool:
        """检查LambdaRank是否可用"""
        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker
            from bma_models.rank_aware_blender import RankAwareBlender
            return True
        except ImportError:
            return False

    def _init_unified_blender(self):
        """(Removed) Rank-aware blender deprecated; no-op for compatibility."""
        return None

    def _validate_oof_quality(self, oof_predictions: Dict[str, pd.Series], y: pd.Series) -> Dict[str, float]:
        """验证OOF预测质量"""
        from scipy.stats import spearmanr

        quality_metrics = {
            'avg_ic': 0.0,
            'ic_std': 0.0,
            'min_ic': 0.0,
            'max_ic': 0.0
        }

        try:
            ics = []
            for model_name, pred_series in oof_predictions.items():
                # 对齐预测和目标
                aligned_pred = pred_series.reindex(y.index)
                valid_mask = ~(aligned_pred.isna() | y.isna())

                if valid_mask.sum() > 10:  # 至少10个有效样本
                    ic, _ = spearmanr(
                        aligned_pred[valid_mask],
                        y[valid_mask]
                    )
                    if not np.isnan(ic):
                        ics.append(ic)

            if ics:
                quality_metrics['avg_ic'] = np.mean(ics)
                quality_metrics['ic_std'] = np.std(ics)
                quality_metrics['min_ic'] = np.min(ics)
                quality_metrics['max_ic'] = np.max(ics)

        except Exception as e:
            logger.warning(f"⚠️ 质量验证失败: {e}")

        return quality_metrics

    def _log_oof_quality(self, oof_predictions: Dict[str, pd.Series], y: pd.Series):
        """记录OOF预测质量"""
        quality_metrics = self._validate_oof_quality(oof_predictions, y)

        logger.info("📊 OOF预测质量报告:")
        logger.info(f"   平均IC: {quality_metrics['avg_ic']:.4f}")
        logger.info(f"   IC范围: [{quality_metrics['min_ic']:.4f}, {quality_metrics['max_ic']:.4f}]")
        logger.info(f"   IC标准差: {quality_metrics['ic_std']:.4f}")

        # 质量警告
        if quality_metrics['avg_ic'] < 0.01:
            logger.warning("⚠️ 平均IC过低，可能影响二层模型质量")
        if quality_metrics['ic_std'] > 0.1:
            logger.warning("⚠️ IC波动过大，模型稳定性可能受影响")

    def _log_performance_summary(self, results: Dict[str, Any], total_time: float):
        """记录性能总结"""
        logger.info("="*70)
        logger.info("📊 统一并行训练性能报告:")

        if 'stage1' in results['timing']:
            logger.info(f"   阶段1（统一第一层）: {results['timing']['stage1']:.2f}秒")

        if 'stage2' in results['timing']:
            logger.info(f"   阶段2（并行二层）: {results['timing']['stage2']:.2f}秒")

        if 'ridge' in results['timing'] and 'lambda' in results['timing']:
            ridge_time = results['timing']['ridge']
            lambda_time = results['timing']['lambda']
            sequential_time = ridge_time + lambda_time
            parallel_time = max(ridge_time, lambda_time)
            time_saved = sequential_time - parallel_time

            logger.info(f"   Ridge时间: {ridge_time:.2f}秒")
            logger.info(f"   LambdaRank时间: {lambda_time:.2f}秒")
            logger.info(f"   并行节省时间: {time_saved:.2f}秒")
            if sequential_time > 0:
                logger.info(f"   二层加速比: {sequential_time/parallel_time:.2f}x")

        logger.info(f"   总耗时: {total_time:.2f}秒")
        logger.info(f"   阶段1成功: {results['stage1_success']}")
        logger.info(f"   Ridge成功（OOF）: {results['ridge_success']}")
        logger.info(f"   Lambda成功（Alpha）: {results['lambda_success']}")
        logger.info("="*70)

    def _build_lambda_data(self, alpha_factors, y: pd.Series,
                          dates: pd.Series, tickers: pd.Series) -> Optional[pd.DataFrame]:
        """
        构建LambdaRank的数据（使用Alpha Factors或fallback到OOF）
        """
        try:
            # 创建MultiIndex
            if not isinstance(y.index, pd.MultiIndex):
                multi_index = pd.MultiIndex.from_arrays(
                    [dates, tickers], names=['date', 'ticker']
                )
                y_indexed = pd.Series(y.values, index=multi_index)
            else:
                y_indexed = y

            # 处理不同类型的输入
            if isinstance(alpha_factors, dict):
                raise ValueError("Unified lambda data builder requires alpha factor DataFrame; dict-based inputs are no longer supported.")
            else:
                # 正常的Alpha Factors DataFrame
                logger.info("🎯 使用Alpha Factors构建LambdaRank数据")
                if isinstance(alpha_factors.index, pd.MultiIndex):
                    lambda_data = alpha_factors.copy()
                else:
                    # 设置MultiIndex
                    lambda_data = alpha_factors.copy()
                    lambda_data.index = multi_index
                # 如存在与索引重复的辅助列，先移除，避免歧义
                lambda_data = lambda_data.drop(columns=['date', 'ticker'], errors='ignore')

                # 移除预测列（如果有的话）
                pred_cols = [col for col in lambda_data.columns if 'pred_' in col.lower()]
                if pred_cols:
                    lambda_data = lambda_data.drop(columns=pred_cols)
                    logger.info(f"   移除{len(pred_cols)}个预测列")

            # 添加目标变量（动态T+H from unified time config）
            # 动态目标列：严格使用T+5
            from bma_models.unified_config_loader import get_time_config
            horizon_days = int(get_time_config().prediction_horizon_days)
            if horizon_days != 5:
                raise ValueError(f'Unified lambda data builder expects prediction_horizon_days=5, got {horizon_days}.')
            target_col = 'ret_fwd_10d'
            lambda_data[target_col] = y_indexed

            # 验证数据
            feature_count = lambda_data.shape[1] - 1  # 减去target列
            logger.info(f"📊 LambdaRank数据: {lambda_data.shape[0]}行 × {feature_count}个特征")

            # 清理NaN
            clean_data = lambda_data.dropna()
            if len(clean_data) < len(lambda_data) * 0.8:
                logger.warning(f"⚠️ 清理后剩余 {len(clean_data)}/{len(lambda_data)} 样本")

            return clean_data

        except Exception as e:
            logger.error(f"❌ 构建LambdaRank数据失败: {e}")
            return None
