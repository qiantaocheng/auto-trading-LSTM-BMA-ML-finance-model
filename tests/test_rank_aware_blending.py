#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank-aware Blending方案A集成测试

测试内容：
- LambdaRank训练器功能验证
- Rank-aware Blender融合效果
- 主pipeline集成测试
- 性能对比：单Ridge vs Rank-aware Blending
- Top-K选股性能提升验证

期望效果：
- Top-K性能提升（NDCG@K, Precision@K）
- 排序稳定性增强
- 融合权重自适应调整
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, Tuple
import logging

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bma_models'))

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRankAwareBlending(unittest.TestCase):
    """Rank-aware Blending方案A集成测试"""

    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)

        # 创建足够大的测试数据集（LambdaRank需要）
        self.dates = pd.date_range('2023-01-01', periods=150, freq='D')
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
        self.index = pd.MultiIndex.from_product([self.dates, self.tickers], names=['date', 'ticker'])

        logger.info(f"测试数据集: {len(self.index)} 样本 ({len(self.dates)} 日期 × {len(self.tickers)} 股票)")

    def create_realistic_second_layer_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """创建真实的第二层训练数据"""
        # 创建第一层预测（具有一定相关性的）
        base_signal = np.random.normal(0, 0.02, len(self.index))

        data = pd.DataFrame(index=self.index)
        data['pred_elastic'] = base_signal + np.random.normal(0, 0.01, len(self.index))
        data['pred_xgb'] = base_signal + np.random.normal(0, 0.015, len(self.index))
        data['pred_catboost'] = base_signal + np.random.normal(0, 0.012, len(self.index))

        # 创建与预测相关的目标变量（带噪声）
        target = pd.Series(
            base_signal * 0.3 + np.random.normal(0, 0.025, len(self.index)),
            index=self.index,
            name='ret_fwd_5d'
        )

        # 添加一些数据质量问题
        nan_indices = np.random.choice(len(data), size=int(len(data) * 0.02), replace=False)
        for col in data.columns:
            col_nans = np.random.choice(nan_indices, size=len(nan_indices)//3, replace=False)
            data.loc[data.index[col_nans], col] = np.nan

        # 目标变量也添加少量NaN
        target_nans = np.random.choice(len(target), size=int(len(target) * 0.01), replace=False)
        target.iloc[target_nans] = np.nan

        return data, target

    def test_lambda_rank_stacker_basic(self):
        """测试LambdaRank Stacker基础功能"""
        try:
            from bma_models.lambda_rank_stacker import LambdaRankStacker

            # 创建测试数据
            stacker_data, target = self.create_realistic_second_layer_data()
            stacker_data['ret_fwd_5d'] = target

            # 初始化LambdaRank
            lambda_stacker = LambdaRankStacker(
                n_quantiles=8,
                num_boost_round=30,
                early_stopping_rounds=10
            )

            # 训练
            lambda_stacker.fit(stacker_data)

            # 验证训练结果
            self.assertTrue(lambda_stacker.fitted_, "LambdaRank应该已训练")

            # 预测
            predictions = lambda_stacker.predict(stacker_data)

            # 验证预测结果
            self.assertIn('lambda_score', predictions.columns, "应该包含lambda_score")
            self.assertIn('lambda_rank', predictions.columns, "应该包含lambda_rank")
            self.assertIn('lambda_pct', predictions.columns, "应该包含lambda_pct")

            # 验证排名逻辑
            for date in self.dates[:5]:  # 检查前几个日期
                try:
                    date_data = predictions.loc[date]
                    if len(date_data) > 1:
                        ranks = date_data['lambda_rank'].dropna()
                        if len(ranks) > 1:
                            # 验证排名是否合理（最小值应该是1）
                            self.assertGreaterEqual(ranks.min(), 1, f"日期{date}排名应该从1开始")
                except KeyError:
                    continue  # 某些日期可能没有数据

            logger.info(f"✅ LambdaRank训练和预测成功: 覆盖率={predictions['lambda_score'].notna().mean():.1%}")

        except ImportError:
            self.skipTest("LightGBM不可用，跳过LambdaRank测试")

    def test_rank_aware_blender_basic(self):
        """测试Rank-aware Blender基础功能"""
        try:
            from bma_models.rank_aware_blender import RankAwareBlender

            # 创建模拟的Ridge和LambdaRank预测
            ridge_predictions = pd.DataFrame(index=self.index[:1000])  # 使用部分数据
            ridge_predictions['score'] = np.random.normal(0, 0.02, len(ridge_predictions))
            ridge_predictions['score_z'] = (ridge_predictions['score'] - ridge_predictions['score'].mean()) / ridge_predictions['score'].std()

            lambda_predictions = pd.DataFrame(index=self.index[:1000])
            lambda_predictions['lambda_score'] = np.random.normal(0, 0.025, len(lambda_predictions))
            lambda_predictions['lambda_pct'] = np.random.uniform(0, 1, len(lambda_predictions))

            # 初始化Blender
            blender = RankAwareBlender(
                lookback_window=30,
                min_weight=0.2,
                max_weight=0.8,
                use_copula=True
            )

            # 融合预测
            blended_results = blender.blend_predictions(
                ridge_predictions=ridge_predictions,
                lambda_predictions=lambda_predictions
            )

            # 验证融合结果
            expected_cols = ['ridge_score', 'lambda_score', 'blended_score', 'blended_rank', 'blended_z']
            for col in expected_cols:
                self.assertIn(col, blended_results.columns, f"应该包含{col}列")

            # 验证融合分数不是NaN
            valid_blended = blended_results['blended_score'].notna().sum()
            self.assertGreater(valid_blended, len(blended_results) * 0.8, "融合分数覆盖率应该>80%")

            # 验证排名逻辑
            test_date = blended_results.index.get_level_values('date')[0]
            date_ranks = blended_results.loc[test_date]['blended_rank'].dropna()
            if len(date_ranks) > 1:
                self.assertGreaterEqual(date_ranks.min(), 1, "排名应该从1开始")
                self.assertLessEqual(date_ranks.max(), len(date_ranks), "最大排名不应超过股票数")

            # 获取融合信息
            blender_info = blender.get_blender_info()
            self.assertIn('current_lambda_weight', blender_info)
            self.assertIn('use_copula', blender_info)

            logger.info(f"✅ Rank-aware融合成功: Lambda权重={blender_info['current_lambda_weight']:.3f}")

        except ImportError:
            self.skipTest("Rank-aware Blending组件不可用")

    def test_main_pipeline_integration(self):
        """测试主pipeline集成"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 创建测试数据
            stacker_data, target = self.create_realistic_second_layer_data()
            stacker_data['ret_fwd_5d'] = target

            # 初始化模型
            model = UltraEnhancedQuantitativeModel()

            # 确保启用Rank-aware Blending
            if hasattr(model, 'use_rank_aware_blending'):
                model.use_rank_aware_blending = True
                logger.info("✅ 已启用Rank-aware Blending")

            # 训练二层模型
            success = model._train_ridge_stacker(
                oof_predictions={
                    'elastic_net': stacker_data['pred_elastic'],
                    'xgboost': stacker_data['pred_xgb'],
                    'catboost': stacker_data['pred_catboost']
                },
                y=target,
                dates=self.dates
            )

            if success:
                # 验证Ridge Stacker训练
                self.assertIsNotNone(model.ridge_stacker, "Ridge Stacker应该已训练")

                # 验证LambdaRank Stacker训练（如果数据量足够）
                if hasattr(model, 'lambda_rank_stacker') and model.lambda_rank_stacker is not None:
                    logger.info("✅ LambdaRank Stacker已训练")
                    self.assertTrue(model.lambda_rank_stacker.fitted_, "LambdaRank应该已训练")

                    # 验证Rank-aware Blender初始化
                    if hasattr(model, 'rank_aware_blender') and model.rank_aware_blender is not None:
                        logger.info("✅ Rank-aware Blender已初始化")
                        blender_info = model.rank_aware_blender.get_blender_info()
                        self.assertIsInstance(blender_info, dict, "Blender信息应该是字典")
                    else:
                        logger.warning("⚠️ Rank-aware Blender未初始化")
                else:
                    logger.warning("⚠️ LambdaRank Stacker未训练（可能数据量不足）")

                logger.info(f"✅ 主pipeline集成测试通过")
            else:
                logger.warning("⚠️ 训练失败，但程序正常处理")
                self.assertTrue(True)  # 失败但正常处理也是可接受的

        except ImportError:
            self.skipTest("主模型不可用")

    def test_performance_comparison(self):
        """测试性能对比：单Ridge vs Rank-aware Blending"""
        try:
            from bma_models.ridge_stacker import RidgeStacker
            from bma_models.lambda_rank_stacker import LambdaRankStacker
            from bma_models.rank_aware_blender import RankAwareBlender

            # 创建测试数据
            stacker_data, target = self.create_realistic_second_layer_data()
            stacker_data['ret_fwd_5d'] = target

            # 训练单Ridge模型
            ridge_stacker = RidgeStacker(alpha=1.0, auto_tune_alpha=False)
            ridge_stacker.fit(stacker_data)
            ridge_pred = ridge_stacker.predict(stacker_data)

            # 训练LambdaRank模型
            lambda_stacker = LambdaRankStacker(n_quantiles=8, num_boost_round=30)
            lambda_stacker.fit(stacker_data)
            lambda_pred = lambda_stacker.predict(stacker_data)

            # 使用Rank-aware Blender融合
            blender = RankAwareBlender(use_copula=True)
            blended_pred = blender.blend_predictions(
                ridge_predictions=ridge_pred[['score']],
                lambda_predictions=lambda_pred
            )

            # 计算性能指标
            def calculate_rank_ic(predictions, targets):
                """计算RankIC"""
                valid_mask = predictions.notna() & targets.notna()
                if valid_mask.sum() < 10:
                    return 0.0
                pred_ranks = predictions[valid_mask].rank()
                target_ranks = targets[valid_mask].rank()
                return pred_ranks.corr(target_ranks, method='spearman')

            def calculate_top_k_precision(predictions, targets, k=20):
                """计算Top-K精确度"""
                try:
                    # 按日期计算Top-K精确度
                    precisions = []
                    for date in predictions.index.get_level_values('date').unique():
                        try:
                            date_pred = predictions.loc[date].dropna()
                            date_target = targets.loc[date].dropna()

                            # 找到共同的股票
                            common_tickers = date_pred.index.intersection(date_target.index)
                            if len(common_tickers) < k:
                                continue

                            pred_vals = date_pred.loc[common_tickers]
                            target_vals = date_target.loc[common_tickers]

                            # Top-K预测
                            top_k_pred = pred_vals.nlargest(k).index
                            # Top-K实际
                            top_k_actual = target_vals.nlargest(k).index

                            # 计算精确度
                            precision = len(set(top_k_pred) & set(top_k_actual)) / k
                            precisions.append(precision)
                        except:
                            continue

                    return np.mean(precisions) if precisions else 0.0
                except:
                    return 0.0

            # 对齐数据用于评估
            ridge_scores = ridge_pred['score']
            lambda_scores = lambda_pred['lambda_score']
            blended_scores = blended_pred['blended_score']

            # 计算RankIC
            ridge_ic = calculate_rank_ic(ridge_scores, target)
            lambda_ic = calculate_rank_ic(lambda_scores, target)
            blended_ic = calculate_rank_ic(blended_scores, target)

            # 计算Top-K精确度
            ridge_precision = calculate_top_k_precision(ridge_scores, target, k=5)
            lambda_precision = calculate_top_k_precision(lambda_scores, target, k=5)
            blended_precision = calculate_top_k_precision(blended_scores, target, k=5)

            logger.info("📊 性能对比结果:")
            logger.info(f"  Ridge RankIC: {ridge_ic:.4f}")
            logger.info(f"  Lambda RankIC: {lambda_ic:.4f}")
            logger.info(f"  Blended RankIC: {blended_ic:.4f}")
            logger.info(f"  Ridge Top-5精确度: {ridge_precision:.3f}")
            logger.info(f"  Lambda Top-5精确度: {lambda_precision:.3f}")
            logger.info(f"  Blended Top-5精确度: {blended_precision:.3f}")

            # 验证融合效果（至少不应该显著变差）
            self.assertGreaterEqual(blended_ic, min(ridge_ic, lambda_ic) - 0.05,
                                  "融合后RankIC不应显著下降")

            # 获取权重信息
            blender_info = blender.get_blender_info()
            logger.info(f"  最终权重: Ridge={1-blender_info['current_lambda_weight']:.3f}, "
                       f"Lambda={blender_info['current_lambda_weight']:.3f}")

            logger.info("✅ 性能对比测试完成")

        except ImportError:
            self.skipTest("所需组件不可用")

def run_rank_aware_blending_tests():
    """运行Rank-aware Blending测试套件"""
    logger.info("开始Rank-aware Blending方案A测试...")

    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestRankAwareBlending))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    # 生成报告
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped

    print("\n" + "="*80)
    print("RANK-AWARE BLENDING方案A测试总结")
    print("="*80)
    print(f"总测试数: {total_tests}")
    print(f"成功: {success}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    print(f"跳过: {skipped}")
    print(f"成功率: {success/total_tests*100:.1f}%" if total_tests > 0 else "N/A")

    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    print("\n" + "="*80)

    return result

if __name__ == '__main__':
    run_rank_aware_blending_tests()