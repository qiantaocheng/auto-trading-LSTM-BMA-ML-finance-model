#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健壮对齐功能测试
Test suite for robust alignment functionality

测试内容：
- 简化数据对齐器测试
- 强化数据验证器测试
- 健壮对齐引擎测试
- 边界情况和错误处理测试
- 性能和一致性测试
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bma_models'))

# 忽略警告
warnings.filterwarnings('ignore')

class TestDataAlignmentBase(unittest.TestCase):
    """对齐测试基类"""

    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        self.index = pd.MultiIndex.from_product([self.dates, self.tickers], names=['date', 'ticker'])

    def create_clean_data(self):
        """创建干净的测试数据"""
        oof_predictions = {
            'elastic_net': pd.Series(np.random.normal(0, 0.02, len(self.index)), index=self.index),
            'xgboost': pd.Series(np.random.normal(0, 0.025, len(self.index)), index=self.index),
            'catboost': pd.Series(np.random.normal(0, 0.02, len(self.index)), index=self.index)
        }
        target = pd.Series(np.random.normal(0, 0.03, len(self.index)), index=self.index)
        return oof_predictions, target

    def create_problematic_data(self):
        """创建有问题的测试数据"""
        oof_predictions, target = self.create_clean_data()

        # 添加数据质量问题
        # 1. NaN值
        oof_predictions['xgboost'].iloc[10:15] = np.nan
        target.iloc[20:25] = np.nan

        # 2. 无穷值
        oof_predictions['catboost'].iloc[30] = np.inf
        oof_predictions['catboost'].iloc[31] = -np.inf

        # 3. 重复索引
        duplicate_index = self.index[:5]
        oof_predictions['elastic_net'] = pd.concat([
            oof_predictions['elastic_net'],
            pd.Series(np.random.normal(0, 0.01, 5), index=duplicate_index)
        ])

        return oof_predictions, target

class TestSimplifiedDataAligner(TestDataAlignmentBase):
    """简化数据对齐器测试"""

    def test_import_aligner(self):
        """测试对齐器导入"""
        try:
            from bma_models.simplified_data_aligner import SimplifiedDataAligner, create_simple_aligner
            self.assertTrue(True, "简化对齐器导入成功")
        except ImportError:
            self.skipTest("简化对齐器模块不可用")

    def test_clean_data_alignment(self):
        """测试干净数据对齐"""
        try:
            from bma_models.simplified_data_aligner import create_simple_aligner

            oof_predictions, target = self.create_clean_data()
            aligner = create_simple_aligner(strict_mode=True)

            stacker_data, report = aligner.align_first_to_second_layer(oof_predictions, target)

            # 验证结果
            self.assertIsInstance(stacker_data, pd.DataFrame)
            self.assertIsInstance(stacker_data.index, pd.MultiIndex)
            self.assertEqual(stacker_data.index.names, ['date', 'ticker'])
            self.assertTrue('ret_fwd_10d' in stacker_data.columns)

            # 验证预测列
            expected_pred_cols = ['pred_elastic', 'pred_xgb', 'pred_catboost']
            for col in expected_pred_cols:
                self.assertTrue(col in stacker_data.columns, f"缺少列: {col}")

            # 验证报告
            self.assertTrue(report['success'])
            self.assertGreater(report['sample_retention_rate'], 0.95)

        except ImportError:
            self.skipTest("简化对齐器模块不可用")

    def test_partial_alignment(self):
        """测试部分对齐模式"""
        try:
            from bma_models.simplified_data_aligner import create_simple_aligner

            oof_predictions, target = self.create_clean_data()

            # 创建不同长度的数据
            short_index = self.index[:100]  # 只取前100个样本
            oof_predictions['elastic_net'] = oof_predictions['elastic_net'].iloc[:100]

            aligner = create_simple_aligner(strict_mode=False)  # 非严格模式允许部分对齐

            stacker_data, report = aligner.align_first_to_second_layer(oof_predictions, target)

            # 验证使用了交集对齐
            self.assertTrue(report['success'])
            self.assertEqual(len(stacker_data), 100)  # 应该是交集的大小

        except ImportError:
            self.skipTest("简化对齐器模块不可用")

    def test_ridge_input_validation(self):
        """测试Ridge输入验证"""
        try:
            from bma_models.simplified_data_aligner import create_simple_aligner

            oof_predictions, target = self.create_clean_data()
            aligner = create_simple_aligner(strict_mode=True)

            stacker_data, _ = aligner.align_first_to_second_layer(oof_predictions, target)

            # 验证Ridge输入格式
            validation_result = aligner.validate_ridge_input(stacker_data)
            self.assertTrue(validation_result)

        except ImportError:
            self.skipTest("简化对齐器模块不可用")

class TestEnhancedDataValidator(TestDataAlignmentBase):
    """强化数据验证器测试"""

    def test_import_validator(self):
        """测试验证器导入"""
        try:
            from bma_models.enhanced_data_validator import EnhancedDataValidator, create_enhanced_validator
            self.assertTrue(True, "强化验证器导入成功")
        except ImportError:
            self.skipTest("强化验证器模块不可用")

    def test_multiindex_validation(self):
        """测试MultiIndex验证"""
        try:
            from bma_models.enhanced_data_validator import create_enhanced_validator

            oof_predictions, target = self.create_clean_data()
            validator = create_enhanced_validator()

            # 测试有效的MultiIndex
            result = validator.validate_multiindex_structure(target, "target")
            self.assertTrue(result['valid'])
            self.assertTrue(result['is_multiindex'])
            self.assertEqual(result['levels'], 2)
            self.assertEqual(result['names'], ['date', 'ticker'])

        except ImportError:
            self.skipTest("强化验证器模块不可用")

    def test_data_quality_validation(self):
        """测试数据质量验证"""
        try:
            from bma_models.enhanced_data_validator import create_enhanced_validator

            oof_predictions, target = self.create_clean_data()
            validator = create_enhanced_validator(min_coverage_rate=0.8)

            # 测试高质量数据
            result = validator.validate_data_quality(target, "target")
            self.assertTrue(result['valid'])
            self.assertGreater(result['coverage_rate'], 0.95)
            self.assertEqual(result['inf_count'], 0)

        except ImportError:
            self.skipTest("强化验证器模块不可用")

    def test_temporal_safety_validation(self):
        """测试时间安全性验证"""
        try:
            from bma_models.enhanced_data_validator import create_enhanced_validator

            oof_predictions, target = self.create_clean_data()
            validator = create_enhanced_validator(temporal_safety=True)

            # 测试时间序列安全性
            result = validator.validate_temporal_safety(target, "target")
            if not result.get('skipped', False):
                self.assertTrue(result['valid'])
                self.assertEqual(result['future_dates'], 0)  # 不应该有未来日期
                self.assertIsNotNone(result['date_range'])

        except ImportError:
            self.skipTest("强化验证器模块不可用")

    def test_comprehensive_validation(self):
        """测试综合验证"""
        try:
            from bma_models.enhanced_data_validator import create_enhanced_validator

            oof_predictions, target = self.create_clean_data()
            validator = create_enhanced_validator()

            # 测试综合验证
            validation_report = validator.comprehensive_validation(oof_predictions, target)

            self.assertTrue(validation_report['validation_passed'])
            self.assertGreater(validation_report['passed_checks'], 0)
            self.assertEqual(validation_report['failed_checks'], 0)

        except ImportError:
            self.skipTest("强化验证器模块不可用")

class TestRobustAlignmentEngine(TestDataAlignmentBase):
    """健壮对齐引擎测试"""

    def test_import_engine(self):
        """测试引擎导入"""
        try:
            from bma_models.robust_alignment_engine import RobustAlignmentEngine, create_robust_alignment_engine
            self.assertTrue(True, "健壮对齐引擎导入成功")
        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_clean_data_alignment(self):
        """测试干净数据对齐"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            oof_predictions, target = self.create_clean_data()
            engine = create_robust_alignment_engine(strict_validation=True, auto_fix=True)

            stacker_data, report = engine.align_data(oof_predictions, target)

            # 验证结果
            self.assertTrue(report['success'])
            self.assertIsInstance(stacker_data, pd.DataFrame)
            self.assertGreater(len(stacker_data), 100)
            self.assertTrue('ret_fwd_10d' in stacker_data.columns)

            # 验证报告结构
            self.assertIn('method', report)
            self.assertIn('performance', report)
            self.assertIn('auto_fixes_applied', report)

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_problematic_data_alignment(self):
        """测试有问题数据的对齐"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            oof_predictions, target = self.create_problematic_data()
            engine = create_robust_alignment_engine(strict_validation=False, auto_fix=True)

            stacker_data, report = engine.align_data(oof_predictions, target)

            # 验证自动修复
            self.assertTrue(report['success'])
            self.assertGreater(len(report['auto_fixes_applied']), 0)

            # 验证数据质量
            self.assertFalse(stacker_data.isna().all().any())  # 不应该有全NaN列
            self.assertFalse(np.isinf(stacker_data.select_dtypes(include=[np.number])).any().any())  # 不应该有无穷值

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_fallback_mechanism(self):
        """测试fallback机制"""
        try:
            from bma_models.robust_alignment_engine import RobustAlignmentEngine

            oof_predictions, target = self.create_clean_data()

            # 创建一个没有自定义模块的引擎（强制使用fallback）
            engine = RobustAlignmentEngine()
            engine.validator = None
            engine.aligner = None

            stacker_data, report = engine.align_data(oof_predictions, target)

            # 验证fallback工作
            self.assertTrue(report['success'])
            self.assertEqual(report['method'], 'fallback_basic')

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_alignment_summary(self):
        """测试对齐总结功能"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            oof_predictions, target = self.create_clean_data()
            engine = create_robust_alignment_engine()

            # 执行几次对齐
            for i in range(3):
                engine.align_data(oof_predictions, target)

            summary = engine.get_alignment_summary()

            # 验证总结
            self.assertEqual(summary['total_alignments'], 3)
            self.assertEqual(summary['successful_alignments'], 3)
            self.assertEqual(summary['success_rate'], 1.0)
            self.assertGreater(summary['average_samples'], 0)

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

class TestErrorHandling(TestDataAlignmentBase):
    """错误处理测试"""

    def test_invalid_input_handling(self):
        """测试无效输入处理"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            engine = create_robust_alignment_engine(strict_validation=True)

            # 测试空预测字典
            with self.assertRaises(Exception):
                engine.align_data({}, pd.Series([1, 2, 3]))

            # 测试非Series目标
            oof_predictions, _ = self.create_clean_data()
            with self.assertRaises(Exception):
                engine.align_data(oof_predictions, [1, 2, 3])

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

    def test_index_mismatch_handling(self):
        """测试索引不匹配处理"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            oof_predictions, target = self.create_clean_data()

            # 创建不匹配的目标变量索引
            different_dates = pd.date_range('2023-06-01', periods=50, freq='D')
            different_index = pd.MultiIndex.from_product([different_dates, self.tickers], names=['date', 'ticker'])
            different_target = pd.Series(np.random.normal(0, 0.03, len(different_index)), index=different_index)

            engine = create_robust_alignment_engine(strict_validation=False, backup_strategy='intersection')

            # 在非严格模式下应该能处理索引不匹配
            with self.assertRaises(Exception):
                # 即使在非严格模式下，完全不匹配的索引也应该失败
                engine.align_data(oof_predictions, different_target)

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

class TestPerformanceAndConsistency(TestDataAlignmentBase):
    """性能和一致性测试"""

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            # 创建较大的数据集
            large_dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 一年的交易日
            large_tickers = [f'STOCK_{i:03d}' for i in range(100)]  # 100只股票
            large_index = pd.MultiIndex.from_product([large_dates, large_tickers], names=['date', 'ticker'])

            large_oof_predictions = {
                'elastic_net': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index),
                'xgboost': pd.Series(np.random.normal(0, 0.025, len(large_index)), index=large_index),
                'catboost': pd.Series(np.random.normal(0, 0.02, len(large_index)), index=large_index)
            }
            large_target = pd.Series(np.random.normal(0, 0.03, len(large_index)), index=large_index)

            engine = create_robust_alignment_engine(strict_validation=False)  # 放宽验证以提高速度

            start_time = datetime.now()
            stacker_data, report = engine.align_data(large_oof_predictions, large_target)
            end_time = datetime.now()

            processing_time = (end_time - start_time).total_seconds()

            # 验证性能
            self.assertTrue(report['success'])
            self.assertEqual(len(stacker_data), len(large_index))
            self.assertLess(processing_time, 30.0, "处理时间应该少于30秒")  # 性能要求

            # 验证内存使用
            memory_usage = report['performance']['memory_usage_mb']
            self.assertLess(memory_usage, 500, "内存使用应该少于500MB")  # 内存要求

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")
        except MemoryError:
            self.skipTest("内存不足，跳过大数据集测试")

    def test_consistency_across_runs(self):
        """测试跨运行一致性"""
        try:
            from bma_models.robust_alignment_engine import create_robust_alignment_engine

            oof_predictions, target = self.create_clean_data()

            # 多次运行相同的对齐
            results = []
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                engine = create_robust_alignment_engine(strict_validation=True)
                stacker_data, report = engine.align_data(oof_predictions, target)
                results.append((stacker_data, report))

            # 验证一致性
            first_data, first_report = results[0]
            for i, (data, report) in enumerate(results[1:], 1):
                # 形状应该一致
                self.assertEqual(data.shape, first_data.shape, f"运行 {i+1} 形状不一致")

                # 索引应该一致
                self.assertTrue(data.index.equals(first_data.index), f"运行 {i+1} 索引不一致")

                # 列名应该一致
                self.assertEqual(list(data.columns), list(first_data.columns), f"运行 {i+1} 列名不一致")

                # 成功状态应该一致
                self.assertEqual(report['success'], first_report['success'], f"运行 {i+1} 成功状态不一致")

        except ImportError:
            self.skipTest("健壮对齐引擎模块不可用")

def run_comprehensive_alignment_tests():
    """运行综合对齐测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestSimplifiedDataAligner,
        TestEnhancedDataValidator,
        TestRobustAlignmentEngine,
        TestErrorHandling,
        TestPerformanceAndConsistency
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)

    # 生成测试报告
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped

    print("\n" + "="*80)
    print("健壮对齐功能测试总结")
    print("="*80)
    print(f"总测试数: {total_tests}")
    print(f"成功: {success}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    print(f"跳过: {skipped}")
    print(f"成功率: {success/total_tests*100:.1f}%" if total_tests > 0 else "N/A")

    if result.failures:
        print(f"\n失败的测试 ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split(chr(10))[0]}")

    if result.errors:
        print(f"\n错误的测试 ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception: ')[-1].split(chr(10))[0]}")

    print("\n" + "="*80)

    return result

if __name__ == '__main__':
    # 运行所有测试
    run_comprehensive_alignment_tests()