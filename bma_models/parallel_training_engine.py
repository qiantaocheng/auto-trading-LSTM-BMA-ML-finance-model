#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行训练引擎 - 实现第一层模型+Ridge Stacking 与 LambdaRank 并行训练

架构：
特征计算 → [第一层(XGB/CatBoost/ElasticNet)+Ridge] || [LambdaRank] → 融合
"""

import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ParallelTrainingEngine:
    """
    并行训练引擎

    核心功能：
    1. 并行训练第一层模型+Ridge和LambdaRank
    2. 性能监控和时间统计
    3. 异常处理和回退机制
    """

    def __init__(self, parent_model):
        """
        初始化并行训练引擎

        Args:
            parent_model: 主模型实例（BMA_Ultra_Enhanced）
        """
        self.parent = parent_model
        self.timing_stats = {}

    def parallel_train_all(self, X: pd.DataFrame, y: pd.Series,
                          dates: pd.Series, tickers: pd.Series) -> Dict[str, Any]:
        """
        并行训练所有模型

        Returns:
            训练结果字典
        """
        logger.info("="*70)
        logger.info("🚀 启动并行训练架构 v2.0")
        logger.info("   Thread 1: 第一层模型(XGBoost/CatBoost/ElasticNet) + Ridge Stacking")
        logger.info("   Thread 2: LambdaRank (如果可用)")
        logger.info("="*70)

        start_time = time.time()
        results = {
            'first_layer_success': False,
            'ridge_success': False,
            'lambda_success': False,
            'oof_predictions': None,
            'stacker_data': None,
            'lambda_model': None,
            'timing': {}
        }

        # 准备共享训练数据
        training_data = {
            'X': X,
            'y': y,
            'dates': dates,
            'tickers': tickers
        }

        # 检查是否需要并行训练LambdaRank
        use_lambda = (hasattr(self.parent, 'use_rank_aware_blending') and
                     self.parent.use_rank_aware_blending and
                     self._check_lambda_available())

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="BMA-Parallel") as executor:
            futures = {}

            # Thread 1: 第一层模型 + Ridge
            first_layer_future = executor.submit(
                self._train_first_layer_and_ridge,
                training_data
            )
            futures[first_layer_future] = 'first_layer_ridge'

            # Thread 2: LambdaRank (如果可用)
            lambda_future = None
            if use_lambda:
                lambda_future = executor.submit(
                    self._train_lambda_rank_direct,
                    training_data
                )
                futures[lambda_future] = 'lambda_rank'

            # 等待所有任务完成
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    task_result = future.result(timeout=1800)  # 30分钟超时

                    if task_name == 'first_layer_ridge':
                        results.update({
                            'first_layer_success': task_result['first_layer_success'],
                            'ridge_success': task_result['ridge_success'],
                            'oof_predictions': task_result['oof_predictions'],
                            'stacker_data': task_result['stacker_data'],
                            'first_layer_models': task_result.get('models', {}),
                            'cv_scores': task_result.get('cv_scores', {})
                        })
                        results['timing']['first_layer_ridge'] = task_result['elapsed_time']
                        logger.info(f"✅ 第一层+Ridge完成，耗时: {task_result['elapsed_time']:.2f}秒")

                    elif task_name == 'lambda_rank':
                        results['lambda_success'] = task_result['success']
                        results['lambda_model'] = task_result.get('model')
                        results['timing']['lambda_rank'] = task_result['elapsed_time']
                        if task_result['success']:
                            self.parent.lambda_rank_stacker = task_result['model']
                            logger.info(f"✅ LambdaRank完成，耗时: {task_result['elapsed_time']:.2f}秒")

                except Exception as e:
                    logger.error(f"❌ {task_name} 训练失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # 计算性能统计
        total_time = time.time() - start_time
        sequential_time = sum(results['timing'].values())
        time_saved = max(0, sequential_time - total_time)

        # 打印性能报告
        logger.info("="*70)
        logger.info("📊 并行训练性能报告:")
        logger.info(f"   第一层+Ridge时间: {results['timing'].get('first_layer_ridge', 0):.2f}秒")
        if use_lambda:
            logger.info(f"   LambdaRank时间: {results['timing'].get('lambda_rank', 0):.2f}秒")
        logger.info(f"   总耗时: {total_time:.2f}秒")
        logger.info(f"   节省时间: {time_saved:.2f}秒")
        if sequential_time > 0:
            logger.info(f"   加速比: {sequential_time/total_time:.2f}x")
        logger.info("="*70)

        # 初始化Blender (如果两个模型都成功)
        if results['ridge_success'] and results['lambda_success'] and use_lambda:
            self._init_rank_aware_blender()

        return results

    def _check_lambda_available(self) -> bool:
        """检查LambdaRank是否可用"""
        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker
            from bma_models.rank_aware_blender import RankAwareBlender
            return True
        except ImportError:
            return False

    def _train_first_layer_and_ridge(self, training_data: Dict) -> Dict:
        """
        Thread 1: 训练第一层模型 + Ridge Stacking
        """
        start_time = time.time()
        result = {
            'first_layer_success': False,
            'ridge_success': False,
            'oof_predictions': None,
            'stacker_data': None,
            'models': {},
            'cv_scores': {},
            'elapsed_time': 0
        }

        try:
            # Step 1: 训练第一层模型
            logger.info("[Thread-1] 开始训练第一层模型...")
            first_layer_results = self.parent._unified_model_training(
                training_data['X'],
                training_data['y'],
                training_data['dates'],
                training_data['tickers']
            )

            if first_layer_results.get('success') and 'oof_predictions' in first_layer_results:
                result['first_layer_success'] = True
                result['oof_predictions'] = first_layer_results['oof_predictions']
                result['models'] = first_layer_results.get('models', {})
                result['cv_scores'] = first_layer_results.get('cv_scores', {})
                logger.info("[Thread-1] ✅ 第一层模型训练成功")

                # Step 2: 训练Ridge Stacker
                logger.info("[Thread-1] 开始训练Ridge Stacker...")
                ridge_success = self.parent._train_ridge_stacker(
                    first_layer_results['oof_predictions'],
                    training_data['y'],
                    training_data['dates']
                )

                result['ridge_success'] = ridge_success
                # 获取stacker_data
                if hasattr(self.parent, '_last_stacker_data'):
                    result['stacker_data'] = self.parent._last_stacker_data

                if ridge_success:
                    logger.info("[Thread-1] ✅ Ridge Stacker训练成功")
                else:
                    logger.error("[Thread-1] ❌ Ridge Stacker训练失败")
            else:
                logger.error("[Thread-1] ❌ 第一层模型训练失败")

        except Exception as e:
            logger.error(f"[Thread-1] 训练异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

        result['elapsed_time'] = time.time() - start_time
        return result

    def _train_lambda_rank_direct(self, training_data: Dict) -> Dict:
        """
        Thread 2: 直接训练LambdaRank模型
        """
        start_time = time.time()
        result = {
            'success': False,
            'model': None,
            'elapsed_time': 0
        }

        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker
            from sklearn.linear_model import ElasticNet
            from xgboost import XGBRegressor
            from catboost import CatBoostRegressor

            logger.info("[Thread-2] 开始训练LambdaRank...")

            X = training_data['X']
            y = training_data['y']
            dates = training_data['dates']
            tickers = training_data['tickers']

            # 快速训练基础模型获取预测
            logger.info("[Thread-2] 生成快速基础预测...")

            # 简化的模型配置（快速版）
            quick_models = {
                'elastic': ElasticNet(alpha=0.001, l1_ratio=0.05, max_iter=100, random_state=42),
                'xgb': XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=1),
                'catboost': CatBoostRegressor(iterations=50, depth=3, random_state=42, verbose=0)
            }

            quick_preds = {}
            for name, model in quick_models.items():
                try:
                    model.fit(X, y)
                    pred = model.predict(X)
                    quick_preds[f'pred_{name}'] = pred
                    logger.info(f"[Thread-2] {name} 快速预测完成")
                except Exception as e:
                    logger.error(f"[Thread-2] {name} 预测失败: {e}")
                    quick_preds[f'pred_{name}'] = np.zeros(len(y))

            # 构建LambdaRank输入数据
            multi_index = pd.MultiIndex.from_arrays(
                [dates, tickers],
                names=['date', 'ticker']
            )

            lambda_input = pd.DataFrame(quick_preds, index=multi_index)
                # 动态目标列名（默认T+1）
                horizon_days = getattr(self.parent, 'horizon', 1)
                target_col = f'ret_fwd_{horizon_days}d'
                lambda_input[target_col] = y.values

            # 训练LambdaRank
            if len(lambda_input) >= 200:
                lambda_config = {
                    'base_cols': tuple(quick_preds.keys()),
                    'n_quantiles': 64,
                    'winsorize_quantiles': (0.01, 0.99),
                    'label_gain_power': 1.5,
                    'num_boost_round': 100,
                    'early_stopping_rounds': 0,
                    'use_purged_cv': True,
                    'cv_n_splits': 5,
                    'cv_gap_days': 6,
                    'cv_embargo_days': 5,
                    'random_state': 42
                }

                lambda_stacker = LambdaRankStacker(**lambda_config)
                lambda_stacker.fit(lambda_input)

                result['success'] = True
                result['model'] = lambda_stacker
                logger.info("[Thread-2] ✅ LambdaRank训练成功")
            else:
                logger.warning(f"[Thread-2] 数据量不足({len(lambda_input)} < 200)，跳过LambdaRank")

        except Exception as e:
            logger.error(f"[Thread-2] LambdaRank训练异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

        result['elapsed_time'] = time.time() - start_time
        return result

    def _init_rank_aware_blender(self):
        """初始化Rank-aware Blender"""
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
            logger.info("✅ Rank-aware Blender初始化成功")
        except Exception as e:
            logger.error(f"❌ Blender初始化失败: {e}")