#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一层模型训练单元测试
测试ElasticNet, XGBoost, LightGBM三个基础模型的训练和预测功能
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from unittest.mock import patch, MagicMock

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bma_models'))

# 忽略警告
warnings.filterwarnings('ignore')

class TestFirstLayerModels(unittest.TestCase):
    """第一层模型测试基类"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.test_data = self._create_ml_test_data()
        self.feature_cols = [col for col in self.test_data.columns 
                           if col not in ['date', 'ticker', 'target']]
        self.X = self.test_data[self.feature_cols]
        self.y = self.test_data['target']
        self.dates = self.test_data['date']
        self.tickers = self.test_data['ticker']
        
    def _create_ml_test_data(self):
        """创建用于ML测试的数据"""
        n_samples = 500
        dates = pd.date_range(start='2023-01-01', periods=100)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        data = []
        for ticker in tickers:
            for date in dates:
                # 创建有一定可预测性的数据
                base_signal = np.random.normal(0, 0.1)
                noise = np.random.normal(0, 0.02)
                
                row = {
                    'date': date,
                    'ticker': ticker,
                    'open': np.random.uniform(100, 200),
                    'high': np.random.uniform(100, 200),
                    'low': np.random.uniform(100, 200),
                    'close': np.random.uniform(100, 200),
                    'volume': np.random.randint(1000000, 10000000),
                    'returns': base_signal + noise,
                    'sma_20': np.random.uniform(100, 200),
                    'sma_50': np.random.uniform(100, 200),
                    'rsi': np.random.uniform(20, 80),
                    'macd': np.random.normal(0, 1),
                    'bollinger_upper': np.random.uniform(100, 200),
                    'bollinger_lower': np.random.uniform(100, 200),
                    'pe_ratio': np.random.uniform(10, 30),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'alpha_momentum': base_signal * 0.5 + np.random.normal(0, 0.05),
                    'alpha_value': -base_signal * 0.3 + np.random.normal(0, 0.03),
                    'alpha_quality': np.random.normal(0, 0.05),
                    'alpha_sentiment': base_signal * 0.7 + np.random.normal(0, 0.04),
                    # 目标变量有一些可预测性
                    'target': base_signal * 0.8 + noise * 2
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['date', 'ticker']).reset_index(drop=True)

class TestTraditionalMLModels(TestFirstLayerModels):
    """传统ML模型测试"""
    
    def test_elastic_net_training(self):
        """测试ElasticNet模型训练"""
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.model_selection import cross_val_score
            
            # 创建ElasticNet模型
            model = ElasticNet(alpha=0.1, random_state=42)
            
            # 训练模型
            model.fit(self.X, self.y)
            
            # 验证模型属性
            self.assertTrue(hasattr(model, 'coef_'))
            self.assertTrue(hasattr(model, 'intercept_'))
            self.assertEqual(len(model.coef_), self.X.shape[1])
            
            # 测试预测
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            self.assertTrue(np.isfinite(predictions).all())
            
            # 测试交叉验证
            cv_scores = cross_val_score(model, self.X, self.y, cv=3, scoring='r2')
            self.assertEqual(len(cv_scores), 3)
            self.assertTrue(np.isfinite(cv_scores).all())
            
        except ImportError:
            self.skipTest("scikit-learn不可用")
        except Exception as e:
            self.fail(f"ElasticNet训练失败: {e}")
    
    def test_xgboost_training(self):
        """测试XGBoost模型训练"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
            
            # 创建XGBoost模型
            model = xgb.XGBRegressor(
                n_estimators=50,  # 减少估算器数量用于测试
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            # 训练模型
            model.fit(self.X, self.y)
            
            # 验证模型属性
            self.assertTrue(hasattr(model, 'feature_importances_'))
            self.assertEqual(len(model.feature_importances_), self.X.shape[1])
            
            # 测试预测
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            self.assertTrue(np.isfinite(predictions).all())
            
            # 测试特征重要性
            importances = model.feature_importances_
            self.assertTrue((importances >= 0).all())
            self.assertAlmostEqual(importances.sum(), 1.0, places=3)
            
        except ImportError:
            self.skipTest("XGBoost不可用")
        except Exception as e:
            self.fail(f"XGBoost训练失败: {e}")
    
    def test_lightgbm_training(self):
        """测试LightGBM模型训练"""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import cross_val_score
            
            # 创建LightGBM模型
            model = lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
            
            # 训练模型
            model.fit(self.X, self.y)
            
            # 验证模型属性
            self.assertTrue(hasattr(model, 'feature_importances_'))
            self.assertEqual(len(model.feature_importances_), self.X.shape[1])
            
            # 测试预测
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            self.assertTrue(np.isfinite(predictions).all())
            
            # 测试特征重要性
            importances = model.feature_importances_
            self.assertTrue((importances >= 0).all())
            
        except ImportError:
            self.skipTest("LightGBM不可用")
        except Exception as e:
            self.fail(f"LightGBM训练失败: {e}")

class TestFirstLayerIntegration(TestFirstLayerModels):
    """第一层模型集成测试"""
    
    def test_traditional_models_training_method(self):
        """测试传统模型训练方法"""
        try:
            from 量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
            model = UltraEnhancedQuantitativeModel()
            
            # 测试传统模型训练方法
            result = model._train_standard_models(self.X, self.y, self.dates, self.tickers)
            
            # 验证结果结构
            self.assertIsInstance(result, dict)
            self.assertIn('models', result)
            self.assertIn('best_model', result)
            self.assertIn('best_score', result)
            
            # 验证模型结果
            models = result['models']
            self.assertIsInstance(models, dict)
            
            # 验证每个模型的结果
            for model_name, model_result in models.items():
                self.assertIsInstance(model_result, dict)
                # 每个模型应该有模型对象和预测结果
                self.assertIn('model', model_result)
                self.assertIn('predictions', model_result)
                self.assertIn('cv_score', model_result)
                
                # 验证预测结果
                predictions = model_result['predictions']
                if isinstance(predictions, str):
                    # 如果是字符串，应该是numpy数组的字符串表示
                    self.assertTrue(len(predictions) > 0)
                else:
                    # 如果是数组，验证形状
                    self.assertTrue(hasattr(predictions, '__len__'))
            
            # 验证最佳模型选择
            best_model = result['best_model']
            self.assertIn(best_model, models.keys())
            
        except Exception as e:
            self.fail(f"传统模型训练方法测试失败: {e}")
    
    def test_model_cross_validation(self):
        """测试模型交叉验证"""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.linear_model import ElasticNet
            from sklearn.metrics import r2_score
            
            # 创建时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=3)
            
            model = ElasticNet(alpha=0.1, random_state=42)
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                cv_scores.append(score)
            
            # 验证交叉验证结果
            self.assertEqual(len(cv_scores), 3)
            self.assertTrue(all(np.isfinite(score) for score in cv_scores))
            
            # 计算平均得分
            avg_score = np.mean(cv_scores)
            self.assertTrue(np.isfinite(avg_score))
            
        except Exception as e:
            self.fail(f"模型交叉验证测试失败: {e}")
    
    def test_model_predictions_consistency(self):
        """测试模型预测一致性"""
        model_configs = [
            ("ElasticNet", {"alpha": 0.1, "random_state": 42}),
            ("XGBoost", {"n_estimators": 10, "random_state": 42}),
            ("LightGBM", {"n_estimators": 10, "random_state": 42, "verbosity": -1})
        ]
        
        predictions_dict = {}
        
        for model_name, config in model_configs:
            try:
                if model_name == "ElasticNet":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet(**config)
                elif model_name == "XGBoost":
                    import xgboost as xgb
                    model = xgb.XGBRegressor(**config)
                elif model_name == "LightGBM":
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(**config)
                
                # 训练和预测
                model.fit(self.X, self.y)
                predictions = model.predict(self.X)
                
                # 验证预测结果
                self.assertEqual(len(predictions), len(self.y))
                self.assertTrue(np.isfinite(predictions).all())
                
                predictions_dict[model_name] = predictions
                
            except ImportError:
                self.skipTest(f"{model_name}不可用")
        
        # 如果有多个模型的预测结果，验证它们不完全相同（说明模型有差异）
        if len(predictions_dict) > 1:
            pred_values = list(predictions_dict.values())
            for i in range(len(pred_values)):
                for j in range(i + 1, len(pred_values)):
                    # 预测结果应该有差异（不完全相同）
                    self.assertFalse(np.allclose(pred_values[i], pred_values[j], rtol=1e-10))

class TestModelPerformanceMetrics(TestFirstLayerModels):
    """模型性能指标测试"""
    
    def test_model_evaluation_metrics(self):
        """测试模型评估指标"""
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from sklearn.linear_model import ElasticNet
            
            # 训练简单模型
            model = ElasticNet(alpha=0.1, random_state=42)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            
            # 计算各种指标
            r2 = r2_score(self.y, predictions)
            mse = mean_squared_error(self.y, predictions)
            mae = mean_absolute_error(self.y, predictions)
            
            # 验证指标有效性
            self.assertTrue(np.isfinite(r2))
            self.assertTrue(np.isfinite(mse))
            self.assertTrue(np.isfinite(mae))
            
            self.assertGreaterEqual(mse, 0)  # MSE应该 >= 0
            self.assertGreaterEqual(mae, 0)  # MAE应该 >= 0
            
            # 对于随机数据，R2可能为负，这是正常的
            self.assertTrue(-np.inf < r2 <= 1.0)
            
        except Exception as e:
            self.fail(f"模型评估指标测试失败: {e}")
    
    def test_feature_importance_calculation(self):
        """测试特征重要性计算"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(n_estimators=10, random_state=42)
            model.fit(self.X, self.y)
            
            # 获取特征重要性
            importances = model.feature_importances_
            
            # 验证特征重要性
            self.assertEqual(len(importances), self.X.shape[1])
            self.assertTrue((importances >= 0).all())
            self.assertAlmostEqual(importances.sum(), 1.0, places=3)
            
            # 创建重要性排名
            feature_importance_pairs = list(zip(self.feature_cols, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 验证排序
            sorted_importances = [pair[1] for pair in feature_importance_pairs]
            self.assertEqual(sorted_importances, sorted(sorted_importances, reverse=True))
            
        except ImportError:
            self.skipTest("XGBoost不可用")
        except Exception as e:
            self.fail(f"特征重要性计算测试失败: {e}")

if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTest(unittest.makeSuite(TestTraditionalMLModels))
    suite.addTest(unittest.makeSuite(TestFirstLayerIntegration))
    suite.addTest(unittest.makeSuite(TestModelPerformanceMetrics))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print(f"\n=== 第一层模型测试结果 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback[:200]}...")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback[:200]}...")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")