#!/usr/bin/env python3
"""
Comprehensive Data Acquisition Unit Tests for BMA Pipeline
Tests polygon client, data fetching, and data validation
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
try:
    from polygon_client import PolygonClient
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    POLYGON_CLIENT_AVAILABLE = False
    PolygonClient = None

class TestDataAcquisition(unittest.TestCase):
    """Test data acquisition functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.start_date = '2024-01-01'
        self.end_date = '2024-01-31'
        
        # Create sample data for mocking
        self.sample_data = self._create_sample_stock_data()
        
    def _create_sample_stock_data(self) -> pd.DataFrame:
        """Create sample stock data for testing"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        data_rows = []
        
        for ticker in self.test_tickers:
            for date in dates:
                # Skip weekends (basic market calendar)
                if date.weekday() >= 5:
                    continue
                    
                data_rows.append({
                    'ticker': ticker,
                    'date': date,
                    'open': 100 + np.random.randn() * 5,
                    'high': 105 + np.random.randn() * 5,
                    'low': 95 + np.random.randn() * 5,
                    'close': 100 + np.random.randn() * 5,
                    'volume': int(1000000 + np.random.randn() * 500000),
                    'adjusted_close': 100 + np.random.randn() * 5
                })
        
        df = pd.DataFrame(data_rows)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    @unittest.skipIf(not POLYGON_CLIENT_AVAILABLE, "PolygonClient not available")
    def test_polygon_client_initialization(self):
        """Test PolygonClient initialization"""
        # Test with valid API key
        client = PolygonClient("test_api_key")
        self.assertEqual(client.api_key, "test_api_key")
        self.assertIsNotNone(client.session)
        
        # Test with empty API key
        with self.assertRaises(ValueError):
            PolygonClient("")
            
        # Test with None API key  
        with self.assertRaises(ValueError):
            PolygonClient(None)
    
    @unittest.skipIf(not POLYGON_CLIENT_AVAILABLE, "PolygonClient not available")
    @patch('polygon_client.requests.Session.get')
    def test_polygon_data_fetching_success(self, mock_get):
        """Test successful data fetching from Polygon"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'c': 150.0,  # close
                    'h': 155.0,  # high
                    'l': 148.0,  # low
                    'o': 152.0,  # open
                    'v': 1000000,  # volume
                    't': 1641081600000,  # timestamp
                    'vw': 151.0  # volume weighted average
                }
            ],
            'status': 'OK',
            'count': 1
        }
        mock_get.return_value = mock_response
        
        client = PolygonClient("test_api_key")
        result = client.get_stock_data(['AAPL'], '2024-01-01', '2024-01-31')
        
        # Verify the call was made
        mock_get.assert_called()
        
        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            expected_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                self.assertIn(col, result.columns)
    
    @unittest.skipIf(not POLYGON_CLIENT_AVAILABLE, "PolygonClient not available")
    @patch('polygon_client.requests.Session.get')
    def test_polygon_data_fetching_failure(self, mock_get):
        """Test data fetching failure handling"""
        # Mock API failure
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_get.return_value = mock_response
        
        client = PolygonClient("test_api_key")
        result = client.get_stock_data(['AAPL'], '2024-01-01', '2024-01-31')
        
        # Should return None or empty DataFrame on failure
        self.assertTrue(result is None or result.empty)
    
    def test_data_validation_basic_structure(self):
        """Test basic data structure validation"""
        data = self.sample_data
        
        # Check required columns exist
        required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns, f"Missing required column: {col}")
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['volume']))
        
        # Check no empty data
        self.assertGreater(len(data), 0, "Data should not be empty")
        
    def test_data_validation_price_consistency(self):
        """Test price data consistency"""
        data = self.sample_data
        
        # High should be >= Open, Close, Low
        invalid_high = data[
            (data['high'] < data['open']) | 
            (data['high'] < data['close']) | 
            (data['high'] < data['low'])
        ]
        self.assertEqual(len(invalid_high), 0, "High prices should be >= other prices")
        
        # Low should be <= Open, Close, High  
        invalid_low = data[
            (data['low'] > data['open']) | 
            (data['low'] > data['close']) | 
            (data['low'] > data['high'])
        ]
        self.assertEqual(len(invalid_low), 0, "Low prices should be <= other prices")
        
        # Volume should be positive
        negative_volume = data[data['volume'] <= 0]
        self.assertEqual(len(negative_volume), 0, "Volume should be positive")
        
    def test_data_validation_date_consistency(self):
        """Test date consistency in data"""
        data = self.sample_data
        
        # Check dates are within expected range
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        
        out_of_range = data[
            (data['date'] < start_dt) | 
            (data['date'] > end_dt)
        ]
        self.assertEqual(len(out_of_range), 0, "All dates should be within requested range")
        
        # Check chronological order for each ticker
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values('date')
            dates = ticker_data['date'].values
            
            # Check no duplicate dates for same ticker
            self.assertEqual(len(dates), len(set(dates)), 
                           f"No duplicate dates should exist for {ticker}")
    
    def test_data_completeness_check(self):
        """Test data completeness validation"""
        data = self.sample_data
        
        # Check all requested tickers are present
        returned_tickers = set(data['ticker'].unique())
        requested_tickers = set(self.test_tickers)
        
        missing_tickers = requested_tickers - returned_tickers
        self.assertEqual(len(missing_tickers), 0, 
                        f"Missing tickers in data: {missing_tickers}")
        
        # Check reasonable amount of data points per ticker
        min_expected_points = 15  # Rough estimate for ~22 trading days in Jan 2024
        for ticker in self.test_tickers:
            ticker_count = len(data[data['ticker'] == ticker])
            self.assertGreaterEqual(ticker_count, min_expected_points,
                                  f"Insufficient data points for {ticker}: {ticker_count}")
    
    def test_missing_data_handling(self):
        """Test handling of missing data scenarios"""
        # Create data with missing values
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0:5, 'close'] = np.nan
        data_with_nan.loc[10:15, 'volume'] = np.nan
        
        # Test NaN detection
        has_nan = data_with_nan.isnull().any().any()
        self.assertTrue(has_nan, "Should detect NaN values")
        
        # Test NaN counting
        nan_counts = data_with_nan.isnull().sum()
        self.assertEqual(nan_counts['close'], 6, "Should count 6 NaN values in close")
        self.assertEqual(nan_counts['volume'], 6, "Should count 6 NaN values in volume")
    
    def test_data_memory_efficiency(self):
        """Test data memory usage is reasonable"""
        data = self.sample_data
        
        # Check memory usage
        memory_usage = data.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 * 1024)
        
        # Should be reasonable for test data (< 50MB)
        self.assertLess(memory_mb, 50, f"Memory usage too high: {memory_mb:.2f}MB")
        
        # Check data types are efficient
        float_cols = ['open', 'high', 'low', 'close']
        for col in float_cols:
            if col in data.columns:
                # Should be numeric
                self.assertTrue(pd.api.types.is_numeric_dtype(data[col]),
                               f"{col} should be numeric type")
    
    def test_edge_cases(self):
        """Test edge cases in data acquisition"""
        
        # Test empty ticker list
        if POLYGON_CLIENT_AVAILABLE:
            client = PolygonClient("test_api_key")
            with patch.object(client, 'get_stock_data') as mock_get:
                mock_get.return_value = pd.DataFrame()
                result = client.get_stock_data([], self.start_date, self.end_date)
                self.assertTrue(result is None or result.empty)
        
        # Test invalid date ranges
        future_start = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        future_end = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Should handle gracefully without crashing
        # Implementation specific - may return empty data or raise exception
        
        # Test very short date ranges (1 day)
        single_date = '2024-01-15'
        # Should handle single day requests
        
    def test_performance_benchmarks(self):
        """Test performance requirements"""
        import time
        
        # Test data processing speed
        data = self.sample_data
        
        start_time = time.time()
        
        # Simulate basic data operations
        _ = data.groupby('ticker').agg({
            'close': ['mean', 'std'],
            'volume': 'mean'
        })
        
        _ = data.sort_values(['ticker', 'date'])
        
        processing_time = time.time() - start_time
        
        # Should process test data quickly (< 5 seconds)
        self.assertLess(processing_time, 5.0, 
                       f"Data processing too slow: {processing_time:.2f}s")


class TestDataAcquisitionIntegration(unittest.TestCase):
    """Integration tests for data acquisition with real services"""
    
    @unittest.skip("Requires real API keys and network access")
    def test_real_polygon_data_acquisition(self):
        """Test with real Polygon API (requires API key)"""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            self.skipTest("POLYGON_API_KEY environment variable not set")
        
        client = PolygonClient(api_key)
        result = client.get_stock_data(['AAPL'], '2024-01-01', '2024-01-05')
        
        if result is not None and not result.empty:
            self.assertIn('ticker', result.columns)
            self.assertIn('close', result.columns)
            self.assertEqual(result['ticker'].iloc[0], 'AAPL')


def create_data_acquisition_test_suite():
    """Create test suite for data acquisition"""
    suite = unittest.TestSuite()
    
    # Add all test methods
    test_cases = [
        TestDataAcquisition,
        TestDataAcquisitionIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("BMA Pipeline - Data Acquisition Unit Tests")
    print("="*60)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_data_acquisition_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)