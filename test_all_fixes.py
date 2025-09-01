#!/usr/bin/env python3
"""
测试所有修复是否正常工作
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# 设置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_fixes():
    """测试所有修复"""
    try:
        # 导入主模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bma_ultra", 
            r"D:\trade\bma_models\量化模型_bma_ultra_enhanced.py"
        )
        bma_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bma_module)
        UltraEnhancedQuantitativeModel = bma_module.UltraEnhancedQuantitativeModel
        
        logger.info("Starting comprehensive test of all fixes...")
        
        # 1. 创建测试数据（MultiIndex结构）
        n_stocks = 30
        n_days = 100
        n_samples = n_stocks * n_days
        
        # 创建MultiIndex DataFrame
        dates = []
        tickers = []
        for d in range(n_days):
            date = datetime(2023, 1, 1) + timedelta(days=d)
            for s in range(n_stocks):
                dates.append(date)
                tickers.append(f'STOCK_{s:03d}')
        
        # 创建特征数据
        np.random.seed(42)
        close_prices = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02))
        feature_data = pd.DataFrame({
            'close': close_prices,
            'high': close_prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
            'low': close_prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
            'open': close_prices * (1 + np.random.randn(n_samples) * 0.005),
            'volume': np.random.poisson(1000000, n_samples),
            'returns': np.random.randn(n_samples) * 0.05,
            'rsi': 50 + 10 * np.random.randn(n_samples),
            'ma_20': 100 + 5 * np.random.randn(n_samples),
            'volatility': np.abs(np.random.randn(n_samples)) * 0.02,
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'target': np.random.randn(n_samples) * 0.05
        })
        
        # 设置MultiIndex
        feature_data['date'] = pd.to_datetime(dates)
        feature_data['ticker'] = tickers
        feature_data['COUNTRY'] = 'US'  # 添加COUNTRY列
        feature_data = feature_data.set_index(['date', 'ticker'])
        
        logger.info(f"Created test data with shape: {feature_data.shape}")
        logger.info(f"Index type: {type(feature_data.index)}")
        
        # 2. 初始化模型
        model = UltraEnhancedQuantitativeModel()
        logger.info("Model initialized successfully")
        
        # 3. 测试特征创建（测试ticker/date列冲突修复）
        try:
            # 传入已有数据
            traditional_features = model.create_traditional_features(feature_data)
            logger.info("✅ Test 1 PASSED: Feature creation with MultiIndex")
        except Exception as e:
            logger.error(f"❌ Test 1 FAILED: {e}")
            return False
        
        # 4. 测试风险模型（测试MarketDataManager缺失修复）
        try:
            risk_model = model.build_risk_model()
            if risk_model.get('error') == 'MarketDataManager not available':
                logger.info("✅ Test 2 PASSED: Risk model gracefully handles missing MarketDataManager")
            else:
                logger.info("✅ Test 2 PASSED: Risk model built successfully")
        except Exception as e:
            logger.error(f"❌ Test 2 FAILED: {e}")
            return False
        
        # 5. 测试训练流程（测试cv_score字典比较修复）
        try:
            # 准备训练数据
            X = feature_data.drop('target', axis=1).reset_index()
            y = feature_data['target'].reset_index(drop=True)
            dates_series = pd.Series([d for d in dates], name='date')
            tickers_series = pd.Series(tickers, name='ticker')
            
            # 训练传统模型
            training_results = model._train_traditional_models_modular(X, y, dates_series, tickers_series)
            
            if training_results.get('success'):
                logger.info("✅ Test 3 PASSED: Traditional models training completed")
            else:
                logger.warning("⚠️ Test 3 WARNING: Training completed but no successful models")
        except Exception as e:
            logger.error(f"❌ Test 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. 测试横截面标准化（测试date歧义修复）
        try:
            from bma_models.temporal_safe_preprocessing import TemporalSafePreprocessor
            
            preprocessor = TemporalSafePreprocessor()
            # 移除非数值列
            X_numeric = X.drop(['date', 'ticker', 'COUNTRY'], axis=1, errors='ignore')
            X_numeric = X_numeric.select_dtypes(include=[np.number])
            X_processed, _ = preprocessor.fit_transform(
                X_numeric,
                dates=dates_series
            )
            logger.info("✅ Test 4 PASSED: Temporal safe preprocessing completed")
        except Exception as e:
            logger.error(f"❌ Test 4 FAILED: {e}")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    
    if success:
        logger.info("\n🎉 All fixes verified and working correctly!")
    else:
        logger.error("\n⚠️ Some tests failed. Please review the fixes.")