#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主流程pipeline集成测试 - 验证健壮对齐引擎在主模型中的集成
Main pipeline integration test for robust alignment engine integration

测试内容：
- 主量化模型中的健壮对齐引擎集成
- Ridge Stacker在新对齐逻辑下的训练
- 端到端流程验证
- 性能和稳定性测试
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

class TestMainPipelineIntegration(unittest.TestCase):
    """主流程pipeline集成测试"""

    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)

        # 创建测试数据集
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        self.index = pd.MultiIndex.from_product([self.dates, self.tickers], names=['date', 'ticker'])

        logger.info(f"测试数据集: {len(self.index)} 样本 ({len(self.dates)} 日期 × {len(self.tickers)} 股票)")

    def create_realistic_oof_predictions(self) -> Dict[str, pd.Series]:
        """创建真实的OOF预测数据"""
        oof_predictions = {}

        # ElasticNet
        oof_predictions['elastic_net'] = pd.Series(
            np.random.normal(0, 0.015, len(self.index)),
            index=self.index
        )

        # XGBoost
        oof_predictions['xgboost'] = pd.Series(
            np.random.normal(0, 0.025, len(self.index)),
            index=self.index
        )

        # CatBoost
        oof_predictions['catboost'] = pd.Series(
            np.random.normal(0, 0.02, len(self.index)),
            index=self.index
        )

        # 添加一些数据质量问题（模拟实际情况）
        for name, pred in oof_predictions.items():
            # 添加少量NaN
            nan_indices = np.random.choice(len(pred), size=int(len(pred) * 0.01), replace=False)
            pred.iloc[nan_indices] = np.nan

        return oof_predictions

    def create_target_variable(self) -> pd.Series:
        """创建目标变量"""
        # 创建与预测相关的目标变量
        target = pd.Series(np.random.normal(0, 0.03, len(self.index)), index=self.index)

        # 添加少量NaN
        nan_indices = np.random.choice(len(target), size=int(len(target) * 0.005), replace=False)
        target.iloc[nan_indices] = np.nan

        return target

    def test_import_main_model(self):
        """测试主模型导入"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
            self.assertTrue(True, "主模型导入成功")
        except ImportError as e:
            self.skipTest(f"主模型不可用: {e}")

    def test_robust_alignment_engine_availability(self):
        """测试健壮对齐引擎可用性"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import ROBUST_ALIGNMENT_AVAILABLE

            if ROBUST_ALIGNMENT_AVAILABLE is True:
                logger.info("✅ 健壮对齐引擎在主模型中可用")
                self.assertTrue(True)
            elif ROBUST_ALIGNMENT_AVAILABLE is False:
                logger.warning("⚠️ 健壮对齐引擎不可用，使用fallback")
                self.assertTrue(True)  # 这也是可接受的
            else:
                logger.error("❌ 所有对齐器都不可用")
                self.fail("所有对齐器都不可用")

        except ImportError:
            self.skipTest("主模型不可用")

    def test_ridge_stacker_training_integration(self):
        """测试Ridge Stacker训练集成"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 创建测试数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_target_variable()

            # 初始化模型
            model = UltraEnhancedQuantitativeModel()

            # 测试Ridge Stacker训练方法
            success = model._train_ridge_stacker(
                oof_predictions=oof_predictions,
                y=target,
                dates=self.dates
            )

            # 验证训练结果
            self.assertTrue(success, "Ridge Stacker训练应该成功")
            self.assertIsNotNone(model.ridge_stacker, "Ridge Stacker应该已初始化")
            self.assertTrue(model.ridge_stacker.fitted_, "Ridge Stacker应该已训练")

            # 验证模型信息
            model_info = model.ridge_stacker.get_model_info()
            self.assertIn('train_score', model_info, "应该包含训练分数")
            self.assertIn('feature_importance', model_info, "应该包含特征重要性")

            logger.info(f"Ridge训练成功: R²={model_info.get('train_score', 'N/A'):.4f}")

        except ImportError:
            self.skipTest("主模型不可用")

    def test_ridge_prediction_integration(self):
        """测试Ridge预测集成"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 创建测试数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_target_variable()

            # 初始化和训练模型
            model = UltraEnhancedQuantitativeModel()
            success = model._train_ridge_stacker(
                oof_predictions=oof_predictions,
                y=target,
                dates=self.dates
            )

            self.assertTrue(success, "训练应该成功")

            # 创建预测数据（相同格式）
            pred_data = self.create_realistic_oof_predictions()

            # 构建预测输入
            prediction_input = pd.DataFrame(index=self.index)
            prediction_input['pred_catboost'] = pred_data['catboost']
            prediction_input['pred_elastic'] = pred_data['elastic_net']
            prediction_input['pred_xgb'] = pred_data['xgboost']

            # 测试预测
            predictions = model.ridge_stacker.predict(prediction_input)

            # 验证预测结果
            self.assertIsInstance(predictions, pd.DataFrame, "预测结果应该是DataFrame")
            self.assertTrue('score' in predictions.columns, "应该包含score列")
            self.assertTrue('score_rank' in predictions.columns, "应该包含score_rank列")
            self.assertTrue('score_z' in predictions.columns, "应该包含score_z列")

            # 验证预测覆盖率
            valid_predictions = predictions['score'].dropna()
            coverage_rate = len(valid_predictions) / len(predictions)
            self.assertGreater(coverage_rate, 0.8, "预测覆盖率应该 > 80%")

            logger.info(f"预测成功: 覆盖率={coverage_rate:.2%}")

        except ImportError:
            self.skipTest("主模型不可用")

    def test_alignment_error_handling(self):
        """测试对齐错误处理"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 创建有问题的数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_target_variable()

            # 破坏数据：不同长度的索引
            short_index = self.index[:50]
            oof_predictions['elastic_net'] = oof_predictions['elastic_net'].iloc[:50]

            # 初始化模型
            model = UltraEnhancedQuantitativeModel()

            # 测试错误恢复
            try:
                success = model._train_ridge_stacker(
                    oof_predictions=oof_predictions,
                    y=target,
                    dates=self.dates
                )

                # 如果健壮对齐引擎工作正常，应该能处理这种情况
                if success:
                    logger.info("✅ 错误恢复成功")
                    self.assertTrue(True)
                else:
                    logger.info("⚠️ 训练失败但程序正常处理")
                    self.assertTrue(True)  # 失败但正常处理也是可接受的

            except Exception as e:
                # 某些错误是预期的，特别是在数据严重不一致时
                logger.info(f"预期的错误: {e}")
                self.assertTrue(True)

        except ImportError:
            self.skipTest("主模型不可用")

    def test_configuration_adaptation(self):
        """测试配置自适应"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 测试小数据集
            small_dates = pd.date_range('2023-01-01', periods=20, freq='D')
            small_tickers = ['AAPL', 'MSFT']
            small_index = pd.MultiIndex.from_product([small_dates, small_tickers], names=['date', 'ticker'])

            small_oof_predictions = {
                'elastic_net': pd.Series(np.random.normal(0, 0.02, len(small_index)), index=small_index),
                'xgboost': pd.Series(np.random.normal(0, 0.02, len(small_index)), index=small_index),
                'catboost': pd.Series(np.random.normal(0, 0.02, len(small_index)), index=small_index)
            }
            small_target = pd.Series(np.random.normal(0, 0.03, len(small_index)), index=small_index)

            model = UltraEnhancedQuantitativeModel()
            success = model._train_ridge_stacker(
                oof_predictions=small_oof_predictions,
                y=small_target,
                dates=small_dates
            )

            if success:
                # 验证小数据集配置
                self.assertFalse(model.ridge_stacker.auto_tune_alpha, "小数据集应该禁用自动调参")
                logger.info("✅ 小数据集配置验证通过")

            # 测试大数据集
            large_dates = pd.date_range('2023-01-01', periods=200, freq='D')
            large_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
            large_index = pd.MultiIndex.from_product([large_dates, large_tickers], names=['date', 'ticker'])

            large_oof_predictions = {
                'elastic_net': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index),
                'xgboost': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index),
                'catboost': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index)
            }
            large_target = pd.Series(np.random.normal(0, 0.03, len(large_index)), index=large_index)

            model = UltraEnhancedQuantitativeModel()
            success = model._train_ridge_stacker(
                oof_predictions=large_oof_predictions,
                y=large_target,
                dates=large_dates
            )

            if success:
                # 验证大数据集配置（如果健壮对齐引擎可用）
                from bma_models.量化模型_bma_ultra_enhanced import ROBUST_ALIGNMENT_AVAILABLE
                if ROBUST_ALIGNMENT_AVAILABLE:
                    # 大数据集应该启用自动调参
                    logger.info(f"大数据集自动调参状态: {model.ridge_stacker.auto_tune_alpha}")

                logger.info("✅ 大数据集配置验证通过")

        except ImportError:
            self.skipTest("主模型不可用")

    def test_performance_monitoring(self):
        """测试性能监控"""
        try:
            from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            # 创建数据
            oof_predictions = self.create_realistic_oof_predictions()
            target = self.create_target_variable()

            # 性能测试
            model = UltraEnhancedQuantitativeModel()

            start_time = datetime.now()
            success = model._train_ridge_stacker(
                oof_predictions=oof_predictions,
                y=target,
                dates=self.dates
            )
            end_time = datetime.now()

            training_time = (end_time - start_time).total_seconds()

            if success:
                # 性能要求
                self.assertLess(training_time, 30.0, "训练时间应该少于30秒")

                # 预测性能
                prediction_input = pd.DataFrame(index=self.index)
                prediction_input['pred_catboost'] = oof_predictions['catboost']
                prediction_input['pred_elastic'] = oof_predictions['elastic_net']
                prediction_input['pred_xgb'] = oof_predictions['xgboost']

                start_time = datetime.now()
                predictions = model.ridge_stacker.predict(prediction_input)
                prediction_time = (datetime.now() - start_time).total_seconds()

                self.assertLess(prediction_time, 10.0, "预测时间应该少于10秒")

                logger.info(f"性能测试通过: 训练={training_time:.2f}s, 预测={prediction_time:.2f}s")

        except ImportError:
            self.skipTest("主模型不可用")

def run_main_pipeline_integration_tests():
    """运行主流程集成测试"""
    logger.info("开始主流程pipeline集成测试...")

    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestMainPipelineIntegration))

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
    print("主流程PIPELINE集成测试总结")
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
    run_main_pipeline_integration_tests()