#!/usr/bin/env python3
"""
Comprehensive First Layer Model Training Unit Tests for BMA Pipeline
Tests machine learning models, training pipeline, and cross-validation
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging
import warnings

# Suppress sklearn warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bma_models'))

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class TestFirstLayerModels(unittest.TestCase):
    """Test first layer model training functionality"""
    
    def setUp(self):
        """Set up test environment"""
        np.random.seed(42)  # For reproducible tests
        
        self.X_train, self.X_test, self.y_train, self.y_test = self._create_ml_data()
        self.models = self._get_test_models()
        
    def _create_ml_data(self) -> tuple:
        """Create sample ML training data"""
        # Create features
        n_samples = 1000
        n_features = 20
        
        # Create correlated features with target
        X = np.random.randn(n_samples, n_features)
        
        # Create target with some relationship to features
        true_weights = np.random.randn(n_features) * 0.5
        y = X @ true_weights + np.random.randn(n_samples) * 0.3
        
        # Add some non-linear relationships
        y += 0.1 * X[:, 0] * X[:, 1]  # Interaction term
        y += 0.05 * np.sin(X[:, 2])   # Non-linear term
        
        # Create DataFrame format
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.3, random_state=42, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
    
    def _get_test_models(self) -> dict:
        """Get dictionary of models for testing"""
        models = {}
        
        if SKLEARN_AVAILABLE:
            models.update({
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42),
                'random_forest': RandomForestRegressor(
                    n_estimators=10,  # Small for fast testing
                    max_depth=5,
                    random_state=42,
                    n_jobs=1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=10,  # Small for fast testing
                    max_depth=3,
                    random_state=42
                )
            })
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                verbosity=-1
            )
        
        return models
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_model_training_basic(self):
        """Test basic model training functionality"""
        
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                try:
                    # Train model
                    model.fit(self.X_train, self.y_train)
                    
                    # Make predictions
                    y_pred = model.predict(self.X_test)
                    
                    # Basic checks
                    self.assertEqual(len(y_pred), len(self.y_test),
                                   f"{model_name}: Prediction length mismatch")
                    
                    # Check predictions are numeric and finite
                    self.assertTrue(np.all(np.isfinite(y_pred)),
                                   f"{model_name}: Predictions contain infinite values")
                    
                    # Check R² score is reasonable (> -1, since negative R² is possible)
                    r2 = r2_score(self.y_test, y_pred)
                    self.assertGreater(r2, -1.0,
                                     f"{model_name}: R² score too low: {r2:.3f}")
                    
                except Exception as e:
                    self.fail(f"{model_name} training failed: {e}")
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_model_performance_metrics(self):
        """Test model performance evaluation metrics"""
        
        # Test with a simple model that should perform reasonably well
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Test metric properties
        self.assertGreaterEqual(mse, 0, "MSE should be non-negative")
        self.assertGreaterEqual(mae, 0, "MAE should be non-negative")
        
        # R² should be reasonable for our synthetic data
        self.assertGreater(r2, 0.1, f"R² should be reasonable: {r2:.3f}")
    
    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not available")
    def test_cross_validation_functionality(self):
        """Test cross-validation functionality"""
        
        # Create TimeSeriesSplit for financial data
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = LinearRegression()
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(self.X_train):
            X_train_cv = self.X_train.iloc[train_idx]
            X_val_cv = self.X_train.iloc[val_idx]
            y_train_cv = self.y_train.iloc[train_idx]
            y_val_cv = self.y_train.iloc[val_idx]
            
            # Train and validate
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            score = r2_score(y_val_cv, y_pred_cv)
            cv_scores.append(score)
        
        # Check CV results
        self.assertEqual(len(cv_scores), 3, "Should have 3 CV scores")
        
        # Scores should be reasonable
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        self.assertGreater(mean_score, -0.5, f"Mean CV score too low: {mean_score:.3f}")
        self.assertLess(std_score, 1.0, f"CV score variance too high: {std_score:.3f}")


def create_first_layer_models_test_suite():
    """Create test suite for first layer models"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_cases = [TestFirstLayerModels]
    
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
    print("BMA Pipeline - First Layer Models Unit Tests")
    print("="*60)
    
    # Check dependencies
    print("Dependency check:")
    print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    print("")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_first_layer_models_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)