#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking集成测试 - 验证第一层到第二层完整流程
Integration test for first-to-second layer stacking pipeline

测试内容：
- 完整的第一层到第二层数据流
- 新对齐逻辑与现有系统的集成
- Ridge Stacker与新对齐逻辑的兼容性
- 端到端性能和稳定性测试
- 实际使用场景模拟
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

class TestStackingIntegration(unittest.TestCase):
    """Stacking集成测试"""

    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)

        # 创建较大的测试数据集（模拟真实场景）
        self.dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 一年交易日
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']  # 8只股票
        self.index = pd.MultiIndex.from_product([self.dates, self.tickers], names=['date', 'ticker'])

        logger.info(f"测试数据集大小: {len(self.index)} 样本 ({len(self.dates)} 日期 × {len(self.tickers)} 股票)")

    def create_realistic_oof_predictions(self) -> Dict[str, pd.Series]:
        """创建现实的OOF预测数据"""

        # 模拟三个模型的不同特性
        oof_predictions = {}

        # ElasticNet: 较小的预测值，较低的波动性
        oof_predictions['elastic_net'] = pd.Series(
            np.random.normal(0, 0.015, len(self.index)),
            index=self.index
        )

        # XGBoost: 中等预测值，中等波动性
        oof_predictions['xgboost'] = pd.Series(
            np.random.normal(0, 0.025, len(self.index)),
            index=self.index
        )

        # CatBoost: 较大的预测值，较高的波动性，偶尔有异常值
        catboost_preds = np.random.normal(0, 0.03, len(self.index))
        # 添加一些异常值（模拟模型在某些条件下的异常表现）
        outlier_indices = np.random.choice(len(catboost_preds), size=10, replace=False)
        catboost_preds[outlier_indices] *= 3

        oof_predictions['catboost'] = pd.Series(catboost_preds, index=self.index)

        # 添加一些缺失值（模拟实际情况）
        for name, pred in oof_predictions.items():
            missing_indices = np.random.choice(len(pred), size=int(len(pred) * 0.02), replace=False)
            pred.iloc[missing_indices] = np.nan

        return oof_predictions

    def create_realistic_target(self) -> pd.Series:
        """创建现实的目标变量"""

        # 创建具有时间和截面相关性的目标变量
        target = pd.Series(index=self.index, dtype=float)

        for date in self.dates:
            date_mask = self.index.get_level_values('date') == date
            n_stocks = date_mask.sum()

            # 市场因子（影响所有股票）
            market_factor = np.random.normal(0, 0.02)

            # 个股特质收益
            idiosyncratic = np.random.normal(0, 0.025, n_stocks)

            # 组合收益
            daily_returns = market_factor + idiosyncratic

            target.loc[date_mask] = daily_returns

        # 添加一些缺失值
        missing_indices = np.random.choice(len(target), size=int(len(target) * 0.01), replace=False)
        target.iloc[missing_indices] = np.nan

        return target

    def test_robust_alignment_integration(self):
        """测试健壮对齐逻辑集成"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            logger.info("测试健壮对齐逻辑集成...")

            # 创建测试数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_realistic_target()

            # 创建对齐引擎
            engine = create_robust_alignment_engine(
                strict_validation=False,  # 允许一些数据质量问题
                auto_fix=True,
                backup_strategy='intersection'
            )

            # 执行对齐
            start_time = datetime.now()
            stacker_data, alignment_report = engine.align_data(oof_predictions, target)
            end_time = datetime.now()

            # 验证对齐结果
            self.assertTrue(alignment_report['success'], "对齐应该成功")
            self.assertIsInstance(stacker_data, pd.DataFrame, "结果应该是DataFrame")
            self.assertIsInstance(stacker_data.index, pd.MultiIndex, "索引应该是MultiIndex")
            self.assertEqual(stacker_data.index.names, ['date', 'ticker'], "索引名称应该正确")

            # 验证列
            expected_columns = ['pred_elastic', 'pred_xgb', 'pred_catboost', 'ret_fwd_10d']
            for col in expected_columns:
                self.assertIn(col, stacker_data.columns, f"应该包含列: {col}")

            # 验证数据质量
            self.assertGreater(len(stacker_data), 1000, "应该有足够的样本")
            self.assertFalse(stacker_data.isna().all().any(), "不应该有全NaN列")

            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"对齐完成: {len(stacker_data)} 样本, 耗时: {processing_time:.2f}秒")

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_ridge_stacker_integration(self):
        """测试Ridge Stacker集成"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine
            from bma_models.ridge_stacker import RidgeStacker

            logger.info("测试Ridge Stacker集成...")

            # 创建数据并对齐
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_realistic_target()

            engine = create_robust_alignment_engine(auto_fix=True)
            stacker_data, _ = engine.align_data(oof_predictions, target)

            # 初始化Ridge Stacker
            ridge_stacker = RidgeStacker(
                base_cols=('pred_catboost', 'pred_elastic', 'pred_xgb'),
                alpha=1.0,
                fit_intercept=False,
                auto_tune_alpha=False,  # 简化测试
                random_state=42
            )

            # 训练Ridge Stacker
            start_time = datetime.now()
            ridge_stacker.fit(stacker_data)
            training_time = (datetime.now() - start_time).total_seconds()

            # 验证训练结果
            self.assertTrue(ridge_stacker.fitted_, "Ridge Stacker应该已训练")
            self.assertIsNotNone(ridge_stacker.ridge_model, "Ridge模型应该已初始化")
            self.assertIsNotNone(ridge_stacker.scaler, "标准化器应该已初始化")

            # 测试预测
            predictions = ridge_stacker.predict(stacker_data)

            # 验证预测结果
            self.assertIsInstance(predictions, pd.DataFrame, "预测结果应该是DataFrame")
            self.assertTrue('score' in predictions.columns, "应该包含score列")
            self.assertTrue('score_rank' in predictions.columns, "应该包含score_rank列")
            self.assertTrue('score_z' in predictions.columns, "应该包含score_z列")

            # 验证预测质量
            valid_predictions = predictions['score'].dropna()
            self.assertGreater(len(valid_predictions), len(stacker_data) * 0.8, "预测覆盖率应该足够高")

            # 获取模型信息
            model_info = ridge_stacker.get_model_info()
            self.assertIn('train_score', model_info, "应该包含训练分数")
            self.assertIn('feature_importance', model_info, "应该包含特征重要性")

            logger.info(f"Ridge训练完成: 耗时{training_time:.2f}秒, R²={model_info.get('train_score', 'N/A'):.4f}")

        except ImportError as e:
            self.skipTest(f"Ridge Stacker模块不可用: {e}")


    def test_ridge_replace_ewa_interface(self):
        """RidgeStacker should expose legacy replace_ewa_in_pipeline helper."""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine
            from bma_models.ridge_stacker import RidgeStacker

            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_realistic_target()

            engine = create_robust_alignment_engine(strict_validation=False, auto_fix=True)
            stacker_data, _ = engine.align_data(oof_predictions, target)

            ridge_stacker = RidgeStacker(
                base_cols=('pred_catboost', 'pred_elastic', 'pred_xgb'),
                alpha=1.0,
                fit_intercept=False,
                auto_tune_alpha=False,
                random_state=42
            )

            ridge_stacker.fit(stacker_data)

            feature_cols = ['pred_catboost', 'pred_elastic', 'pred_xgb']
            valid_mask = ~stacker_data[feature_cols].isna().any(axis=1)

            compat_predictions = ridge_stacker.replace_ewa_in_pipeline(stacker_data)
            direct_predictions = ridge_stacker.predict(stacker_data.loc[valid_mask])

            valid_scores = compat_predictions.loc[valid_mask, 'score']
            pd.testing.assert_series_equal(
                valid_scores,
                direct_predictions['score'],
                check_names=False
            )

            if not valid_mask.all():
                self.assertTrue(compat_predictions['score'].isna().any())
        except ImportError as exc:
            self.skipTest(f"RidgeStacker dependencies unavailable: {exc}")

    def test_end_to_end_pipeline(self):
        """测试端到端流程"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine
            from bma_models.ridge_stacker import RidgeStacker

            logger.info("测试端到端Stacking流程...")

            # Step 1: 创建第一层预测数据（模拟第一层模型输出）
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_realistic_target()

            # Step 2: 数据对齐
            engine = create_robust_alignment_engine(
                strict_validation=False,
                auto_fix=True
            )

            stacker_data, alignment_report = engine.align_data(oof_predictions, target)

            # Step 3: 第二层模型训练
            ridge_stacker = RidgeStacker(
                base_cols=('pred_catboost', 'pred_elastic', 'pred_xgb'),
                alpha=2.0,  # 使用较强的正则化
                fit_intercept=False,
                auto_tune_alpha=False,
                random_state=42
            )

            ridge_stacker.fit(stacker_data)

            # Step 4: 预测
            final_predictions = ridge_stacker.predict(stacker_data)

            # Step 5: 验证完整流程
            self.assertTrue(alignment_report['success'], "数据对齐应该成功")
            self.assertTrue(ridge_stacker.fitted_, "模型应该已训练")
            self.assertIsInstance(final_predictions, pd.DataFrame, "最终预测应该是DataFrame")

            # 验证数据一致性
            self.assertEqual(len(final_predictions), len(stacker_data), "预测长度应该与输入一致")
            self.assertTrue(final_predictions.index.equals(stacker_data.index), "索引应该一致")

            # 验证预测质量
            valid_scores = final_predictions['score'].dropna()
            self.assertGreater(len(valid_scores), 0, "应该有有效的预测分数")

            # 计算信息比率（简单版本）
            if len(valid_scores) > 10:
                target_aligned = stacker_data['ret_fwd_10d'].reindex(valid_scores.index).dropna()
                if len(target_aligned) > 10:
                    correlation = np.corrcoef(valid_scores.reindex(target_aligned.index), target_aligned)[0, 1]
                    if not np.isnan(correlation):
                        logger.info(f"预测相关性: {correlation:.4f}")

            # 生成流程报告
            pipeline_report = {
                'alignment_method': alignment_report['method'],
                'samples_processed': len(stacker_data),
                'prediction_coverage': len(valid_scores) / len(final_predictions),
                'auto_fixes_applied': len(alignment_report.get('auto_fixes_applied', [])),
                'warnings': len(alignment_report.get('warnings', [])),
                'model_alpha': ridge_stacker.alpha,
                'model_score': ridge_stacker.train_score_
            }

            logger.info("端到端流程完成:")
            for key, value in pipeline_report.items():
                logger.info(f"  {key}: {value}")

            return stacker_data, final_predictions, pipeline_report

        except ImportError as e:
            self.skipTest(f"必需模块不可用: {e}")

    def test_error_recovery(self):
        """测试错误恢复机制"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            logger.info("测试错误恢复机制...")

            # 创建严重损坏的数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_realistic_target()

            # 损坏数据
            # 1. 大量NaN
            for name, pred in oof_predictions.items():
                nan_indices = np.random.choice(len(pred), size=int(len(pred) * 0.3), replace=False)
                pred.iloc[nan_indices] = np.nan

            # 2. 无穷值
            oof_predictions['xgboost'].iloc[100:110] = np.inf
            oof_predictions['catboost'].iloc[200:210] = -np.inf

            # 3. 重复索引
            duplicate_data = oof_predictions['elastic_net'].iloc[:50].copy()
            oof_predictions['elastic_net'] = pd.concat([oof_predictions['elastic_net'], duplicate_data])

            # 尝试对齐（应该通过自动修复处理）
            engine = create_robust_alignment_engine(
                strict_validation=False,
                auto_fix=True,
                backup_strategy='intersection'
            )

            stacker_data, report = engine.align_data(oof_predictions, target)

            # 验证错误恢复
            self.assertTrue(report['success'], "即使有数据问题也应该成功")
            self.assertGreater(len(report['auto_fixes_applied']), 0, "应该应用了自动修复")
            self.assertGreater(len(stacker_data), 500, "应该恢复了足够的数据")

            logger.info(f"错误恢复成功: 应用了{len(report['auto_fixes_applied'])}个修复")

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_performance_benchmark(self):
        """测试性能基准"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine
            from bma_models.ridge_stacker import RidgeStacker

            logger.info("测试性能基准...")

            # 创建更大的数据集
            large_dates = pd.date_range('2023-01-01', periods=500, freq='D')
            large_tickers = [f'STOCK_{i:03d}' for i in range(50)]  # 50只股票
            large_index = pd.MultiIndex.from_product([large_dates, large_tickers], names=['date', 'ticker'])

            # 创建大数据集
            large_oof_predictions = {
                'elastic_net': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index),
                'xgboost': pd.Series(np.random.normal(0, 0.025, len(large_index)), index=large_index),
                'catboost': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index)
            }
            large_target = pd.Series(np.random.normal(0, 0.03, len(large_index)), index=large_index)

            # 性能测试
            engine = create_robust_alignment_engine(strict_validation=False)

            # 对齐性能
            start_time = datetime.now()
            stacker_data, _ = engine.align_data(large_oof_predictions, large_target)
            alignment_time = (datetime.now() - start_time).total_seconds()

            # 训练性能
            ridge_stacker = RidgeStacker(
                base_cols=('pred_catboost', 'pred_elastic', 'pred_xgb'),
                alpha=1.0,
                auto_tune_alpha=False  # 禁用调参以加快速度
            )

            start_time = datetime.now()
            ridge_stacker.fit(stacker_data)
            training_time = (datetime.now() - start_time).total_seconds()

            # 预测性能
            start_time = datetime.now()
            predictions = ridge_stacker.predict(stacker_data)
            prediction_time = (datetime.now() - start_time).total_seconds()

            # 性能要求
            total_samples = len(large_index)

            self.assertLess(alignment_time, 60.0, "对齐时间应该少于60秒")
            self.assertLess(training_time, 30.0, "训练时间应该少于30秒")
            self.assertLess(prediction_time, 10.0, "预测时间应该少于10秒")

            # 计算吞吐量
            alignment_throughput = total_samples / alignment_time
            training_throughput = total_samples / training_time
            prediction_throughput = total_samples / prediction_time

            logger.info(f"性能基准测试完成 ({total_samples} 样本):")
            logger.info(f"  对齐: {alignment_time:.2f}秒 ({alignment_throughput:.0f} 样本/秒)")
            logger.info(f"  训练: {training_time:.2f}秒 ({training_throughput:.0f} 样本/秒)")
            logger.info(f"  预测: {prediction_time:.2f}秒 ({prediction_throughput:.0f} 样本/秒)")

        except ImportError:
            self.skipTest("必需模块不可用")
        except MemoryError:
            self.skipTest("内存不足，跳过性能测试")

def run_stacking_integration_tests():
    """运行Stacking集成测试"""
    logger.info("开始Stacking集成测试...")

    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestStackingIntegration))

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
    print("STACKING集成测试总结")
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
    run_stacking_integration_tests()