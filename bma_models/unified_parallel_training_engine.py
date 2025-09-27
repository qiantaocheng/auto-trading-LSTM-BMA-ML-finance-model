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

            parallel_results = self._parallel_second_layer_training(
                unified_first_layer_results['oof_predictions'],
                y, dates, tickers,
                alpha_factors=alpha_factors or X  # 如果没有提供alpha_factors，使用X
            )

            results.update({
                'ridge_success': parallel_results['ridge_success'],
                'lambda_success': parallel_results['lambda_success'],
                'stacker_data': parallel_results['stacker_data']
            })
            results['timing']['stage2'] = time.time() - stage2_start
            results['timing'].update(parallel_results['timing'])

            logger.info(f"✅ 阶段2完成，耗时: {results['timing']['stage2']:.2f}秒")

            # 初始化Blender (如果两个模型都成功)
            if results['ridge_success'] and results['lambda_success']:
                self._init_unified_blender()

        except Exception as e:
            logger.error(f"❌ 统一并行训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # 性能统计
        total_time = time.time() - start_time
        self._log_performance_summary(results, total_time)

        return results

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
            alpha_factors if alpha_factors is not None else X, y, dates, tickers
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

        # 检查是否使用并行训练
        use_lambda = (hasattr(self.parent, 'use_rank_aware_blending') and
                     self.parent.use_rank_aware_blending and
                     self._check_lambda_available() and
                     len(ridge_data) >= 200)  # 数据量检查

        if not use_lambda or lambda_data is None:
            logger.info("🔄 只训练Ridge Stacker（LambdaRank不可用或数据不足）")
            ridge_start = time.time()
            ridge_success = self.parent._train_ridge_stacker(
                unified_oof_predictions, y, dates
            )
            results['ridge_success'] = ridge_success
            results['timing']['ridge_only'] = time.time() - ridge_start
            return results

        # 并行训练Ridge和LambdaRank
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="Unified-Parallel") as executor:
            futures = {}

            # 任务1：Ridge Stacker
            ridge_future = executor.submit(
                self._train_ridge_unified,
                unified_oof_predictions, y, dates
            )
            futures[ridge_future] = 'ridge'

            # 任务2：LambdaRank Stacker（使用Alpha Factors）
            lambda_future = executor.submit(
                self._train_lambda_unified,
                lambda_data  # Alpha Factors数据
            )
            futures[lambda_future] = 'lambda'

            # 等待完成
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    task_result = future.result(timeout=1800)  # 30分钟超时

                    if task_name == 'ridge':
                        results['ridge_success'] = task_result['success']
                        results['timing']['ridge'] = task_result['elapsed_time']
                        logger.info(f"✅ Ridge完成，耗时: {task_result['elapsed_time']:.2f}秒")

                    elif task_name == 'lambda':
                        results['lambda_success'] = task_result['success']
                        results['timing']['lambda'] = task_result['elapsed_time']
                        if task_result['success']:
                            self.parent.lambda_rank_stacker = task_result['model']
                            logger.info(f"✅ LambdaRank完成，耗时: {task_result['elapsed_time']:.2f}秒")

                except Exception as e:
                    logger.error(f"❌ {task_name} 训练失败: {e}")

        return results

    def _build_unified_stacker_data(self, oof_predictions: Dict[str, pd.Series],
                                  y: pd.Series, dates: pd.Series, tickers: pd.Series) -> Optional[pd.DataFrame]:
        """
        构建统一的stacker输入数据

        确保Ridge和LambdaRank使用完全相同的数据
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

            # 构建stacker DataFrame
            stacker_dict = {}
            for model_name, pred_series in oof_predictions.items():
                # 确保预测series有正确的索引
                if isinstance(pred_series.index, pd.MultiIndex):
                    stacker_dict[f'pred_{model_name}'] = pred_series
                else:
                    # 如果没有MultiIndex，使用y的索引
                    stacker_dict[f'pred_{model_name}'] = pd.Series(
                        pred_series.values, index=y_indexed.index
                    )

            # 添加目标变量
            stacker_dict['ret_fwd_5d'] = y_indexed

            stacker_data = pd.DataFrame(stacker_dict)

            # 验证数据完整性
            missing_data = stacker_data.isnull().sum()
            if missing_data.any():
                logger.warning(f"⚠️ Stacker数据缺失: {missing_data.to_dict()}")

            # 移除包含NaN的行
            clean_data = stacker_data.dropna()
            if len(clean_data) < len(stacker_data) * 0.8:
                logger.warning(f"⚠️ 数据清理后剩余 {len(clean_data)}/{len(stacker_data)} ({len(clean_data)/len(stacker_data)*100:.1f}%)")

            logger.info(f"📊 统一stacker数据构建完成: {clean_data.shape}")
            return clean_data

        except Exception as e:
            logger.error(f"❌ 构建stacker数据失败: {e}")
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
        """
        训练LambdaRank Stacker（统一数据源版本）
        """
        start_time = time.time()
        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker

            logger.info("[Lambda-Thread] 开始训练LambdaRank...")

            # 使用统一的时间配置
            from bma_models.unified_config_loader import get_time_config
            time_config = get_time_config()

            # 动态确定特征列（根据实际数据）
            feature_cols = [col for col in stacker_data.columns if col != 'ret_fwd_5d']
            logger.info(f"[Lambda-Thread] 使用特征列: {feature_cols}")

            # 配置LambdaRank（使用purged CV factory）
            lambda_config = {
                'base_cols': tuple(feature_cols),  # 动态使用实际可用的特征列
                'n_quantiles': 64,
                'winsorize_quantiles': (0.01, 0.99),
                'label_gain_power': 1.5,
                'num_boost_round': 100,
                'early_stopping_rounds': 0,
                'use_purged_cv': True,
                'cv_n_splits': time_config.cv_n_splits,
                'cv_gap_days': time_config.cv_gap_days,
                'cv_embargo_days': time_config.cv_embargo_days,
                'random_state': 42
            }

            lambda_stacker = LambdaRankStacker(**lambda_config)
            lambda_stacker.fit(stacker_data)

            logger.info("[Lambda-Thread] ✅ LambdaRank训练成功")

            return {
                'success': True,
                'model': lambda_stacker,
                'elapsed_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"[Lambda-Thread] 训练异常: {e}")
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
        """初始化统一Blender"""
        try:
            from bma_models.rank_aware_blender import RankAwareBlender

            self.parent.rank_aware_blender = RankAwareBlender(
                lookback_window=60,
                min_weight=0.3,
                max_weight=0.7,
                weight_smoothing=0.3,
                use_copula=True,
                use_decorrelation=True,
                top_k_list=[5, 10, 20]
            )
            logger.info("✅ 统一Rank-aware Blender初始化成功")
        except Exception as e:
            logger.error(f"❌ Blender初始化失败: {e}")

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
                # 如果是OOF predictions dict，转换为DataFrame
                logger.info("🔄 使用OOF预测构建LambdaRank数据（fallback模式）")
                lambda_dict = {}
                for model_name, pred_series in alpha_factors.items():
                    if isinstance(pred_series.index, pd.MultiIndex):
                        lambda_dict[f'pred_{model_name}'] = pred_series
                    else:
                        lambda_dict[f'pred_{model_name}'] = pd.Series(
                            pred_series.values, index=y_indexed.index
                        )
                lambda_data = pd.DataFrame(lambda_dict)
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

            # 添加目标变量
            lambda_data['ret_fwd_5d'] = y_indexed

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