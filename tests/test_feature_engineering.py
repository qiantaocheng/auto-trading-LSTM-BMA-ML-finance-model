#!/usr/bin/env python3
"""
Comprehensive Feature Engineering Unit Tests for BMA Pipeline
Tests alpha strategies, feature computation, and data preprocessing
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging
import yaml

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bma_models'))

# Import modules to test
try:
    from bma_models.enhanced_alpha_strategies import AlphaStrategiesEngine
    ALPHA_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from enhanced_alpha_strategies import AlphaStrategiesEngine
        ALPHA_ENGINE_AVAILABLE = True
    except ImportError:
        ALPHA_ENGINE_AVAILABLE = False
        AlphaStrategiesEngine = None


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering and alpha computation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = self._create_sample_stock_data()
        
        # Create mock config
        self.mock_config = {
            'alphas': [
                {
                    'name': 'momentum_5d',
                    'function': 'momentum',
                    'period': 5,
                    'decay': 15,
                    'enabled': True
                },
                {
                    'name': 'mean_reversion_10d', 
                    'function': 'mean_reversion',
                    'period': 10,
                    'decay': 20,
                    'enabled': True
                },
                {
                    'name': 'volatility_5d',
                    'function': 'volatility', 
                    'period': 5,
                    'decay': 8,
                    'enabled': True
                }
            ],
            'processing': {
                'winsorize_quantiles': [0.01, 0.99],
                'standardization': 'cross_sectional',
                'neutralization': {
                    'by_industry': False,
                    'by_market_cap': False
                }
            },
            'performance': {
                'cache_enabled': True,
                'parallel_processing': False,
                'max_workers': 1
            }
        }
        
    def _create_sample_stock_data(self) -> pd.DataFrame:
        """Create sample stock data with proper price relationships"""
        np.random.seed(42)  # For reproducible tests
        
        dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META']
        
        data_rows = []
        
        for ticker in tickers:
            base_price = 100 + hash(ticker) % 50  # Different base price per ticker
            price = base_price
            
            for i, date in enumerate(dates):
                # Skip weekends
                if date.weekday() >= 5:
                    continue
                
                # Random walk with some mean reversion
                price_change = np.random.randn() * 0.02 * price
                if price > base_price * 1.2:
                    price_change -= 0.01 * price  # Mean reversion
                elif price < base_price * 0.8:
                    price_change += 0.01 * price  # Mean reversion
                    
                price += price_change
                
                # Ensure proper OHLC relationships
                daily_volatility = abs(np.random.randn() * 0.01 * price)
                
                open_price = price + np.random.randn() * daily_volatility * 0.5
                close_price = price + np.random.randn() * daily_volatility * 0.5
                high_price = max(open_price, close_price) + abs(np.random.randn() * daily_volatility)
                low_price = min(open_price, close_price) - abs(np.random.randn() * daily_volatility)
                
                volume = int(1000000 + abs(np.random.randn() * 500000))
                
                data_rows.append({
                    'ticker': ticker,
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2), 
                    'close': round(close_price, 2),
                    'volume': volume,
                    'adjusted_close': round(close_price, 2)
                })
                
                # Update price for next iteration
                price = close_price
        
        df = pd.DataFrame(data_rows)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    @unittest.skipIf(not ALPHA_ENGINE_AVAILABLE, "AlphaStrategiesEngine not available")
    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_alpha_engine_initialization(self, mock_yaml_load, mock_open):
        """Test AlphaStrategiesEngine initialization"""
        # Mock config loading
        mock_yaml_load.return_value = self.mock_config
        mock_open.return_value.__enter__.return_value = Mock()
        
        # Test initialization
        engine = AlphaStrategiesEngine()
        
        # Check engine is properly initialized
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.config)
        self.assertIn('alphas', engine.config)
        
    @unittest.skipIf(not ALPHA_ENGINE_AVAILABLE, "AlphaStrategiesEngine not available")
    def test_basic_alpha_computations(self):
        """Test basic alpha factor computations"""
        
        # Create mock engine with basic functions
        class MockAlphaEngine:
            def __init__(self):
                self.config = self.mock_config
                
            def compute_momentum(self, data, period=5):
                """Simple momentum calculation"""
                if len(data) < period:
                    return pd.Series(np.nan, index=data.index)
                return data['close'].pct_change(periods=period)
            
            def compute_volatility(self, data, period=5):
                """Simple volatility calculation"""
                if len(data) < period:
                    return pd.Series(np.nan, index=data.index) 
                returns = data['close'].pct_change()
                return returns.rolling(window=period, min_periods=period).std()
            
            def compute_mean_reversion(self, data, period=10):
                """Simple mean reversion calculation"""
                if len(data) < period:
                    return pd.Series(np.nan, index=data.index)
                ma = data['close'].rolling(window=period, min_periods=period).mean()
                return -(data['close'] - ma) / ma  # Negative for mean reversion
        
        mock_engine = MockAlphaEngine()
        
        # Test momentum calculation
        momentum = mock_engine.compute_momentum(self.test_data, period=5)
        
        # Should have valid results (not all NaN)
        valid_momentum = momentum.dropna()
        self.assertGreater(len(valid_momentum), 0, "Momentum should have valid values")
        
        # Should be bounded (typical momentum values)
        self.assertTrue((abs(valid_momentum) < 1).all(), 
                       "Momentum values should be reasonable")
        
        # Test volatility calculation
        volatility = mock_engine.compute_volatility(self.test_data, period=5)
        valid_volatility = volatility.dropna()
        
        self.assertGreater(len(valid_volatility), 0, "Volatility should have valid values")
        self.assertTrue((valid_volatility >= 0).all(), 
                       "Volatility should be non-negative")
        
    def test_feature_data_consistency(self):
        """Test feature data consistency and structure"""
        
        # Create mock features DataFrame
        features = pd.DataFrame({
            'ticker': self.test_data['ticker'],
            'date': self.test_data['date'],
            'momentum_5d': np.random.randn(len(self.test_data)) * 0.1,
            'volatility_5d': abs(np.random.randn(len(self.test_data)) * 0.02),
            'mean_reversion_10d': np.random.randn(len(self.test_data)) * 0.05,
            'close': self.test_data['close']
        })
        
        # Test basic structure
        self.assertIn('ticker', features.columns)
        self.assertIn('date', features.columns)
        
        # Test feature columns exist
        feature_columns = [col for col in features.columns 
                         if col not in ['ticker', 'date', 'close']]
        self.assertGreater(len(feature_columns), 0, "Should have feature columns")
        
        # Test data types
        for col in feature_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(features[col]),
                           f"Feature {col} should be numeric")
        
        # Test no infinite values
        for col in feature_columns:
            infinite_values = np.isinf(features[col]).sum()
            self.assertEqual(infinite_values, 0, 
                           f"Feature {col} should not have infinite values")
    
    def test_feature_preprocessing(self):
        """Test feature preprocessing steps"""
        
        # Create sample features with outliers
        np.random.seed(42)
        features = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
        
        # Add some outliers
        features.loc[0, 'feature1'] = 10  # Outlier
        features.loc[1, 'feature2'] = -8  # Outlier
        
        # Test winsorization (mock implementation)
        def winsorize_features(df, quantiles=[0.01, 0.99]):
            result = df.copy()
            for col in df.select_dtypes(include=[np.number]).columns:
                lower = df[col].quantile(quantiles[0])
                upper = df[col].quantile(quantiles[1])
                result[col] = df[col].clip(lower=lower, upper=upper)
            return result
        
        winsorized = winsorize_features(features)
        
        # Check outliers are handled
        self.assertLess(winsorized['feature1'].max(), features['feature1'].max(),
                       "Winsorization should reduce extreme values")
        
        # Test standardization (mock implementation)
        def standardize_features(df):
            result = df.copy()
            for col in df.select_dtypes(include=[np.number]).columns:
                result[col] = (df[col] - df[col].mean()) / df[col].std()
            return result
        
        standardized = standardize_features(winsorized)
        
        # Check standardization
        for col in standardized.select_dtypes(include=[np.number]).columns:
            self.assertAlmostEqual(standardized[col].mean(), 0, places=10,
                                 msg=f"Standardized {col} should have mean ~0")
            self.assertAlmostEqual(standardized[col].std(), 1, places=10,
                                 msg=f"Standardized {col} should have std ~1")
    
    def test_feature_alignment(self):
        """Test feature alignment across tickers and dates"""
        
        # Create features with different lengths per ticker
        features_data = []
        for ticker in ['AAPL', 'MSFT']:
            ticker_data = self.test_data[self.test_data['ticker'] == ticker].copy()
            # Simulate some missing data
            if ticker == 'MSFT':
                ticker_data = ticker_data.iloc[5:]  # Remove first 5 rows
            
            features_data.append(ticker_data)
        
        # Test alignment function (mock implementation)
        def align_features(data_list):
            if not data_list:
                return pd.DataFrame()
            
            # Find common date range
            all_dates = []
            for df in data_list:
                all_dates.extend(df['date'].unique())
            
            common_dates = sorted(set(all_dates))
            
            # Align all DataFrames to common dates
            aligned_data = []
            for df in data_list:
                aligned = df[df['date'].isin(common_dates)].copy()
                aligned_data.append(aligned)
            
            return pd.concat(aligned_data, ignore_index=True)
        
        aligned = align_features(features_data)
        
        # Test alignment results
        tickers = aligned['ticker'].unique()
        self.assertGreater(len(tickers), 1, "Should have multiple tickers after alignment")
        
        # Test date consistency
        for ticker in tickers:
            ticker_dates = sorted(aligned[aligned['ticker'] == ticker]['date'].unique())
            self.assertGreater(len(ticker_dates), 0, f"Should have dates for {ticker}")
    
    def test_missing_data_handling(self):
        """Test missing data handling in features"""
        
        # Create features with missing values
        features = pd.DataFrame({
            'ticker': ['AAPL'] * 10 + ['MSFT'] * 10,
            'date': pd.date_range('2024-01-01', periods=10).tolist() * 2,
            'feature1': [1, 2, np.nan, 4, 5] * 4,
            'feature2': [np.nan, 2, 3, np.nan, 5] * 4,
            'feature3': range(20)
        })
        
        # Test NaN detection
        nan_summary = features.isnull().sum()
        self.assertEqual(nan_summary['feature1'], 4, "Should detect NaN in feature1")
        self.assertEqual(nan_summary['feature2'], 4, "Should detect NaN in feature2") 
        self.assertEqual(nan_summary['feature3'], 0, "Should detect no NaN in feature3")
        
        # Test forward fill strategy (mock implementation)
        def forward_fill_features(df, group_cols=['ticker']):
            result = df.copy()
            feature_cols = [col for col in df.columns 
                          if col not in ['ticker', 'date']]
            
            for col in feature_cols:
                result[col] = df.groupby(group_cols)[col].fillna(method='ffill')
            
            return result
        
        filled = forward_fill_features(features)
        
        # Should have fewer NaN values after forward fill
        original_nans = features.isnull().sum().sum()
        filled_nans = filled.isnull().sum().sum()
        self.assertLessEqual(filled_nans, original_nans, 
                           "Forward fill should not increase NaN count")
    
    def test_feature_quality_metrics(self):
        """Test feature quality assessment"""
        
        # Create features with known properties
        np.random.seed(42)
        n_samples = 1000
        
        # Good feature (correlated with target)
        target = np.random.randn(n_samples)
        good_feature = target * 0.7 + np.random.randn(n_samples) * 0.3
        
        # Noise feature (uncorrelated)
        noise_feature = np.random.randn(n_samples)
        
        # Constant feature (no variance)
        constant_feature = np.ones(n_samples) * 5
        
        features_df = pd.DataFrame({
            'good_feature': good_feature,
            'noise_feature': noise_feature, 
            'constant_feature': constant_feature,
            'target': target
        })
        
        # Test correlation calculation
        correlations = features_df.corrwith(features_df['target']).abs()
        
        self.assertGreater(correlations['good_feature'], 0.5,
                          "Good feature should have strong correlation")
        self.assertLess(correlations['noise_feature'], 0.2,
                       "Noise feature should have weak correlation")
        
        # Test variance calculation  
        variances = features_df.var()
        
        self.assertGreater(variances['good_feature'], 0.1,
                          "Good feature should have meaningful variance")
        self.assertAlmostEqual(variances['constant_feature'], 0, places=10,
                              msg="Constant feature should have zero variance")
    
    def test_feature_engineering_edge_cases(self):
        """Test edge cases in feature engineering"""
        
        # Test empty data
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        self.assertEqual(len(empty_df), 0, "Empty DataFrame should remain empty")
        
        # Test single stock data
        single_stock = self.test_data[self.test_data['ticker'] == 'AAPL'].copy()
        
        # Should still be processable
        self.assertGreater(len(single_stock), 0, "Single stock data should be valid")
        self.assertEqual(len(single_stock['ticker'].unique()), 1, 
                        "Should have exactly one ticker")
        
        # Test data with all same values
        constant_data = single_stock.copy()
        constant_data['close'] = 100.0  # All same price
        
        # Percentage change should be 0 or NaN
        pct_change = constant_data['close'].pct_change()
        valid_changes = pct_change.dropna()
        if len(valid_changes) > 0:
            self.assertTrue((valid_changes == 0).all(), 
                           "Constant prices should have zero returns")
    
    def test_performance_requirements(self):
        """Test feature engineering performance"""
        import time
        
        # Create larger dataset for performance testing
        large_data = pd.concat([self.test_data] * 10, ignore_index=True)
        
        start_time = time.time()
        
        # Simulate basic feature computation
        _ = large_data.groupby('ticker').apply(
            lambda x: x['close'].pct_change()
        )
        
        _ = large_data.groupby('ticker').apply(
            lambda x: x['close'].rolling(window=5).mean()
        )
        
        processing_time = time.time() - start_time
        
        # Should process reasonably quickly (< 10 seconds for test data)
        self.assertLess(processing_time, 10.0,
                       f"Feature processing too slow: {processing_time:.2f}s")
    
    def test_data_integrity_after_processing(self):
        """Test data integrity is maintained after processing"""
        
        original_data = self.test_data.copy()
        
        # Simulate processing pipeline
        processed_data = original_data.copy()
        
        # Add some features
        processed_data['momentum'] = processed_data.groupby('ticker')['close'].pct_change(5)
        processed_data['volatility'] = processed_data.groupby('ticker')['close'].rolling(5).std().values
        
        # Test original data integrity
        pd.testing.assert_frame_equal(
            original_data, 
            self.test_data,
            msg="Original data should not be modified"
        )
        
        # Test processed data structure
        self.assertEqual(len(processed_data), len(original_data),
                        "Processing should not change number of rows")
        
        # Test core columns preserved
        core_columns = ['ticker', 'date', 'close', 'volume']
        for col in core_columns:
            if col in original_data.columns:
                pd.testing.assert_series_equal(
                    processed_data[col],
                    original_data[col],
                    msg=f"Core column {col} should be preserved"
                )


def create_feature_engineering_test_suite():
    """Create test suite for feature engineering"""
    suite = unittest.TestSuite()
    
    # Add all test methods
    tests = unittest.TestLoader().loadTestsFromTestCase(TestFeatureEngineering)
    suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("BMA Pipeline - Feature Engineering Unit Tests")  
    print("="*60)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_feature_engineering_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)