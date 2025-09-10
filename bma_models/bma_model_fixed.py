#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Model Fixed - Production-Ready Simplified Version
Fixes all critical issues identified in the main model
Focus on reliability, simplicity and correctness
"""

import os
import sys
import logging
import warnings
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ConfigManager:
    """Unified configuration management - single source of truth"""
    
    DEFAULT_CONFIG = {
        # T+10 prediction settings
        'feature_lag': 5,
        'prediction_horizon': 10,
        'min_samples': 100,
        
        # Cross-validation settings
        'cv_folds': 5,
        'embargo_days': 1,
        'min_train_days': 252,
        
        # Model settings
        'random_state': 42,
        'ridge_alpha': 1.0,
        'rf_n_estimators': 100,
        'rf_max_depth': 10,
        
        # Data quality thresholds
        'max_missing_pct': 0.3,
        'min_variance': 1e-10,
        'max_correlation': 0.99,
        
        # Memory management
        'max_memory_gb': 2.0,
        'cleanup_threshold_mb': 500
    }
    
    def __init__(self, config_override: Dict = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config['feature_lag'] > 0, "feature_lag must be positive"
        assert self.config['prediction_horizon'] > 0, "prediction_horizon must be positive"
        assert self.config['cv_folds'] >= 2, "cv_folds must be at least 2"
        assert self.config['embargo_days'] >= 0, "embargo_days cannot be negative"
        assert 0 < self.config['max_missing_pct'] <= 1, "max_missing_pct must be between 0 and 1"
        logger.info("Configuration validated successfully")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)


class DataValidator:
    """Comprehensive data validation and quality checks"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.validation_report = {}
    
    def validate_data(self, data: pd.DataFrame, stage: str = "input") -> Tuple[bool, pd.DataFrame, Dict]:
        """
        Validate data quality and structure
        Returns: (is_valid, cleaned_data, report)
        """
        report = {
            'stage': stage,
            'original_shape': data.shape,
            'issues': [],
            'warnings': []
        }
        
        # Check if data is empty
        if data.empty:
            report['issues'].append("Data is empty")
            return False, data, report
        
        # Check minimum samples
        if len(data) < self.config.get('min_samples'):
            report['issues'].append(f"Insufficient samples: {len(data)} < {self.config.get('min_samples')}")
            return False, data, report
        
        # Check for required index structure
        if not isinstance(data.index, pd.MultiIndex):
            if 'date' in data.columns and 'ticker' in data.columns:
                data = data.set_index(['date', 'ticker'])
                report['warnings'].append("Converted to MultiIndex (date, ticker)")
            else:
                report['issues'].append("Missing required date/ticker columns")
                return False, data, report
        
        # Check for time series continuity
        dates = data.index.get_level_values(0).unique()
        if len(dates) < self.config.get('min_train_days'):
            report['warnings'].append(f"Limited time series: {len(dates)} days")
        
        # Remove columns with too many missing values
        missing_pct = data.isnull().mean()
        bad_cols = missing_pct[missing_pct > self.config.get('max_missing_pct')].index.tolist()
        if bad_cols:
            data = data.drop(columns=bad_cols)
            report['warnings'].append(f"Removed {len(bad_cols)} columns with >30% missing")
        
        # Remove constant columns
        if data.shape[1] > 0:
            variances = data.var()
            const_cols = variances[variances < self.config.get('min_variance')].index.tolist()
            if const_cols:
                data = data.drop(columns=const_cols)
                report['warnings'].append(f"Removed {len(const_cols)} constant columns")
        
        # Check for highly correlated features
        if data.shape[1] > 1:
            corr_matrix = data.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_pairs = upper_tri[upper_tri > self.config.get('max_correlation')].stack()
            if len(high_corr_pairs) > 0:
                report['warnings'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Fill remaining missing values
        data = data.fillna(method='ffill', limit=5).fillna(0)
        
        report['final_shape'] = data.shape
        report['is_valid'] = len(report['issues']) == 0
        
        self.validation_report[stage] = report
        return report['is_valid'], data, report


class SimpleCrossValidator:
    """Simple and robust time series cross-validation"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cv_results = []
    
    def get_cv_splits(self, data: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate time series CV splits with proper embargo
        """
        # Get unique dates
        dates = sorted(data.index.get_level_values(0).unique())
        n_dates = len(dates)
        
        # Calculate split parameters
        min_train = self.config.get('min_train_days')
        embargo = self.config.get('embargo_days')
        n_splits = self.config.get('cv_folds')
        
        if n_dates < min_train + embargo + 20:  # Need at least 20 days for test
            logger.warning(f"Insufficient data for CV: {n_dates} days")
            return []
        
        splits = []
        test_size = max(20, (n_dates - min_train - embargo) // (n_splits + 1))
        
        for i in range(n_splits):
            train_end_idx = min_train + i * test_size
            test_start_idx = train_end_idx + embargo
            test_end_idx = min(test_start_idx + test_size, n_dates)
            
            if test_end_idx > n_dates - 1:
                break
            
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            train_idx = data.index[data.index.get_level_values(0).isin(train_dates)]
            test_idx = data.index[data.index.get_level_values(0).isin(test_dates)]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        logger.info(f"Generated {len(splits)} CV splits with {embargo}-day embargo")
        return splits


class MemoryManager:
    """Simple memory management without complex threading"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.last_cleanup = time.time()
        
    def check_and_cleanup(self, force: bool = False):
        """Simple memory cleanup using garbage collection"""
        current_time = time.time()
        
        # Only cleanup every 30 seconds unless forced
        if not force and (current_time - self.last_cleanup) < 30:
            return
        
        # Run garbage collection
        gc.collect()
        self.last_cleanup = current_time
        
        # Log memory usage if psutil available
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Memory usage: {memory_mb:.1f} MB")
            
            if memory_mb > self.config.get('max_memory_gb') * 1024:
                logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                gc.collect()
        except ImportError:
            pass


class ModelTrainer:
    """Simplified model training with proper error handling"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: str = 'ridge') -> Any:
        """Train a single model with error handling"""
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            # Select and train model
            if model_type == 'ridge':
                model = Ridge(
                    alpha=self.config.get('ridge_alpha'),
                    random_state=self.config.get('random_state')
                )
            elif model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=self.config.get('rf_n_estimators'),
                    max_depth=self.config.get('rf_max_depth'),
                    random_state=self.config.get('random_state'),
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Fit model
            model.fit(X_scaled, y_train)
            
            # Store components
            self.models[model_type] = model
            self.scalers[model_type] = scaler
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_type] = pd.Series(
                    model.feature_importances_,
                    index=X_train.columns
                ).sort_values(ascending=False)
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_type] = pd.Series(
                    np.abs(model.coef_),
                    index=X_train.columns
                ).sort_values(ascending=False)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return None
    
    def predict(self, X_test: pd.DataFrame, model_type: str = 'ridge') -> np.ndarray:
        """Generate predictions with error handling"""
        try:
            if model_type not in self.models:
                logger.error(f"Model {model_type} not trained")
                return np.zeros(len(X_test))
            
            # Scale features
            X_scaled = self.scalers[model_type].transform(X_test)
            
            # Generate predictions
            predictions = self.models[model_type].predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return np.zeros(len(X_test))


class BMAModelFixed:
    """Fixed and simplified BMA model for production use"""
    
    def __init__(self, config_override: Dict = None):
        """Initialize with configuration"""
        self.config = ConfigManager(config_override)
        self.validator = DataValidator(self.config)
        self.cv_splitter = SimpleCrossValidator(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.trainer = ModelTrainer(self.config)
        
        self.training_history = []
        self.evaluation_metrics = {}
        
        logger.info("BMA Model Fixed initialized successfully")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with T+10 alignment"""
        try:
            # Ensure we have the required structure
            if not isinstance(data.index, pd.MultiIndex):
                if 'date' in data.columns and 'ticker' in data.columns:
                    data = data.set_index(['date', 'ticker'])
                else:
                    raise ValueError("Data must have date and ticker columns")
            
            # Sort by date and ticker
            data = data.sort_index()
            
            # Create lagged features (T-5 for T+10 prediction)
            lag = self.config.get('feature_lag')
            feature_cols = [col for col in data.columns if col not in ['returns', 'target']]
            
            lagged_features = []
            for col in feature_cols:
                lagged_col = data.groupby(level='ticker')[col].shift(lag)
                lagged_col.name = f'{col}_lag{lag}'
                lagged_features.append(lagged_col)
            
            features = pd.concat(lagged_features, axis=1)
            
            # Create target (T+10 returns)
            if 'returns' in data.columns:
                horizon = self.config.get('prediction_horizon')
                target = data.groupby(level='ticker')['returns'].shift(-horizon)
                target.name = f'target_t{horizon}'
            else:
                target = pd.Series(index=features.index, name='target', dtype=float)
            
            # Combine features and target
            result = pd.concat([features, target], axis=1)
            
            # Remove rows with NaN target
            result = result[result[target.name].notna()]
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train(self, data: pd.DataFrame, model_types: List[str] = None) -> Dict[str, Any]:
        """Main training pipeline with comprehensive error handling"""
        
        start_time = time.time()
        results = {
            'success': False,
            'models': {},
            'metrics': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Validate input data
            logger.info("Step 1: Validating input data...")
            is_valid, clean_data, validation_report = self.validator.validate_data(data, 'input')
            
            if not is_valid:
                results['errors'].extend(validation_report['issues'])
                return results
            
            results['warnings'].extend(validation_report.get('warnings', []))
            
            # Step 2: Prepare features
            logger.info("Step 2: Preparing features with T+10 alignment...")
            feature_data = self.prepare_features(clean_data)
            
            if feature_data.empty:
                results['errors'].append("Feature preparation failed")
                return results
            
            # Validate prepared features
            is_valid, feature_data, validation_report = self.validator.validate_data(
                feature_data, 'features'
            )
            
            if not is_valid:
                results['errors'].extend(validation_report['issues'])
                return results
            
            # Step 3: Split features and target
            target_col = [col for col in feature_data.columns if 'target' in col][0]
            X = feature_data.drop(columns=[target_col])
            y = feature_data[target_col]
            
            logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
            
            # Step 4: Generate CV splits
            logger.info("Step 4: Generating cross-validation splits...")
            cv_splits = self.cv_splitter.get_cv_splits(feature_data)
            
            if not cv_splits:
                results['errors'].append("Could not generate CV splits")
                return results
            
            # Step 5: Train models
            if model_types is None:
                model_types = ['ridge', 'rf']
            
            logger.info(f"Step 5: Training models: {model_types}")
            
            for model_type in model_types:
                logger.info(f"Training {model_type} model...")
                cv_scores = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                    # Get train/test data
                    X_train = X.loc[train_idx]
                    y_train = y.loc[train_idx]
                    X_test = X.loc[test_idx]
                    y_test = y.loc[test_idx]
                    
                    # Train model
                    model = self.trainer.train_model(X_train, y_train, model_type)
                    
                    if model is None:
                        results['warnings'].append(f"Failed to train {model_type} on fold {fold_idx}")
                        continue
                    
                    # Generate predictions
                    predictions = self.trainer.predict(X_test, model_type)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, predictions)
                    
                    # Calculate IC (information coefficient)
                    ic = np.corrcoef(y_test, predictions)[0, 1] if len(y_test) > 1 else 0
                    
                    cv_scores.append({
                        'fold': fold_idx,
                        'mse': mse,
                        'ic': ic,
                        'n_train': len(X_train),
                        'n_test': len(X_test)
                    })
                    
                    # Memory cleanup
                    self.memory_manager.check_and_cleanup()
                
                # Store results
                if cv_scores:
                    results['models'][model_type] = self.trainer.models.get(model_type)
                    results['metrics'][model_type] = {
                        'cv_scores': cv_scores,
                        'mean_ic': np.mean([s['ic'] for s in cv_scores]),
                        'std_ic': np.std([s['ic'] for s in cv_scores]),
                        'mean_mse': np.mean([s['mse'] for s in cv_scores]),
                        'feature_importance': self.trainer.feature_importance.get(model_type, {})
                    }
                    
                    logger.info(f"{model_type} - Mean IC: {results['metrics'][model_type]['mean_ic']:.4f}")
            
            # Step 6: Train final models on all data
            logger.info("Step 6: Training final models on all data...")
            
            for model_type in model_types:
                if model_type in results['models']:
                    final_model = self.trainer.train_model(X, y, model_type)
                    results['models'][model_type] = final_model
            
            # Calculate summary metrics
            results['summary'] = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_models': len(results['models']),
                'training_time': time.time() - start_time,
                'best_model': max(results['metrics'].items(), 
                                 key=lambda x: x[1]['mean_ic'])[0] if results['metrics'] else None
            }
            
            results['success'] = True
            logger.info(f"Training completed successfully in {results['summary']['training_time']:.1f}s")
            
        except Exception as e:
            logger.error(f"Training pipeline error: {e}")
            results['errors'].append(f"Pipeline error: {str(e)}")
            results['traceback'] = traceback.format_exc()
        
        finally:
            # Final cleanup
            self.memory_manager.check_and_cleanup(force=True)
        
        return results
    
    def predict(self, data: pd.DataFrame, model_type: str = None) -> pd.DataFrame:
        """Generate predictions for new data"""
        try:
            # Prepare features
            feature_data = self.prepare_features(data)
            
            if feature_data.empty:
                logger.error("Feature preparation failed for prediction")
                return pd.DataFrame()
            
            # Get features
            target_cols = [col for col in feature_data.columns if 'target' in col]
            X = feature_data.drop(columns=target_cols, errors='ignore')
            
            # Select model
            if model_type is None:
                # Use best model from training
                if hasattr(self, 'evaluation_metrics') and self.evaluation_metrics:
                    model_type = max(self.evaluation_metrics.items(), 
                                   key=lambda x: x[1].get('mean_ic', 0))[0]
                else:
                    model_type = 'ridge'
            
            # Generate predictions
            predictions = self.trainer.predict(X, model_type)
            
            # Create result DataFrame
            result = pd.DataFrame(
                predictions,
                index=X.index,
                columns=[f'prediction_{model_type}']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return pd.DataFrame()
    
    def evaluate(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model predictions"""
        try:
            # Align data
            common_idx = data.index.intersection(predictions.index)
            
            if len(common_idx) == 0:
                logger.error("No common indices for evaluation")
                return {}
            
            # Get actual returns
            if 'returns' in data.columns:
                actual = data.loc[common_idx, 'returns']
            else:
                logger.error("No returns column for evaluation")
                return {}
            
            pred = predictions.loc[common_idx].iloc[:, 0]
            
            # Calculate metrics
            metrics = {
                'ic': np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else 0,
                'mse': mean_squared_error(actual, pred),
                'mae': np.mean(np.abs(actual - pred)),
                'n_samples': len(actual)
            }
            
            # Calculate IC by date
            ic_by_date = []
            for date in data.index.get_level_values(0).unique():
                date_mask = data.index.get_level_values(0) == date
                if date_mask.sum() > 1:
                    date_actual = actual[date_mask]
                    date_pred = pred[date_mask]
                    if len(date_actual) > 1:
                        date_ic = np.corrcoef(date_actual, date_pred)[0, 1]
                        ic_by_date.append(date_ic)
            
            if ic_by_date:
                metrics['mean_ic'] = np.mean(ic_by_date)
                metrics['std_ic'] = np.std(ic_by_date)
                metrics['ir'] = metrics['mean_ic'] / metrics['std_ic'] if metrics['std_ic'] > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {}


def create_sample_data(n_dates: int = 500, n_tickers: int = 100, n_features: int = 20) -> pd.DataFrame:
    """Create sample data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_dates, freq='D')
    tickers = [f'TICKER_{i:03d}' for i in range(n_tickers)]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # Create features
    data = pd.DataFrame(
        np.random.randn(len(index), n_features),
        index=index,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add returns
    data['returns'] = np.random.randn(len(index)) * 0.01
    
    return data


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("BMA Model Fixed - Production Ready Version")
    logger.info("=" * 60)
    
    # Create sample data
    logger.info("Creating sample data...")
    data = create_sample_data(n_dates=500, n_tickers=50, n_features=10)
    logger.info(f"Data shape: {data.shape}")
    
    # Initialize model
    logger.info("\nInitializing BMA model...")
    model = BMAModelFixed()
    
    # Train model
    logger.info("\nTraining model...")
    results = model.train(data, model_types=['ridge', 'rf'])
    
    # Print results
    if results['success']:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS:")
        logger.info("=" * 60)
        
        for model_type, metrics in results['metrics'].items():
            logger.info(f"\n{model_type.upper()} Model:")
            logger.info(f"  Mean IC: {metrics['mean_ic']:.4f}")
            logger.info(f"  Std IC: {metrics['std_ic']:.4f}")
            logger.info(f"  IR: {metrics['mean_ic']/metrics['std_ic']:.4f}" if metrics['std_ic'] > 0 else "  IR: N/A")
            logger.info(f"  Mean MSE: {metrics['mean_mse']:.6f}")
            
            if 'feature_importance' in metrics and not metrics['feature_importance'].empty:
                logger.info(f"  Top 5 Features:")
                for feat, imp in list(metrics['feature_importance'].head().items()):
                    logger.info(f"    - {feat}: {imp:.4f}")
        
        logger.info(f"\nSummary:")
        logger.info(f"  Models trained: {results['summary']['n_models']}")
        logger.info(f"  Best model: {results['summary']['best_model']}")
        logger.info(f"  Training time: {results['summary']['training_time']:.1f}s")
        
        # Generate predictions
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING PREDICTIONS:")
        logger.info("=" * 60)
        
        test_data = data.iloc[-10000:]  # Last portion for testing
        predictions = model.predict(test_data)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Evaluate
        if not predictions.empty:
            metrics = model.evaluate(test_data, predictions)
            logger.info("\nEvaluation Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        logger.error("\n" + "=" * 60)
        logger.error("TRAINING FAILED:")
        logger.error("=" * 60)
        for error in results['errors']:
            logger.error(f"  - {error}")
        for warning in results['warnings']:
            logger.warning(f"  - {warning}")
        
        if 'traceback' in results:
            logger.error("\nTraceback:")
            logger.error(results['traceback'])
    
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()