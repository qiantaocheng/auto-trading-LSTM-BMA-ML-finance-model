#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real OOS (Out-of-Sample) Data Manager for BMA Weight Updates
真实样本外数据管理器 - 替代Mock OOS

This module manages real out-of-sample predictions from previous CV folds
and rolling windows for accurate BMA weight updates.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class OOSPrediction:
    """Single OOS prediction record"""
    timestamp: datetime
    fold_id: str
    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    feature_hash: str
    model_version: str
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate prediction metrics"""
        from scipy.stats import spearmanr
        
        # Remove NaN values
        mask = ~(self.predictions.isna() | self.actuals.isna())
        pred_clean = self.predictions[mask]
        actual_clean = self.actuals[mask]
        
        if len(pred_clean) < 10:  # Need minimum samples
            return {}
        
        # Calculate metrics
        ic, _ = spearmanr(pred_clean, actual_clean)
        mse = np.mean((pred_clean - actual_clean) ** 2)
        mae = np.mean(np.abs(pred_clean - actual_clean))
        
        return {
            'ic': ic if not np.isnan(ic) else 0.0,
            'mse': mse,
            'mae': mae,
            'n_samples': len(pred_clean)
        }


class RealOOSManager:
    """
    Manages real out-of-sample predictions for BMA weight updates
    管理真实样本外预测用于BMA权重更新
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/oos_predictions",
                 max_history: int = 20,
                 min_samples_for_update: int = 100):
        """
        Initialize OOS manager
        
        Args:
            cache_dir: Directory to cache OOS predictions
            max_history: Maximum number of historical predictions to keep
            min_samples_for_update: Minimum samples needed for weight update
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        self.min_samples_for_update = min_samples_for_update
        
        # Storage for OOS predictions
        self.oos_history: deque = deque(maxlen=max_history)
        self.current_fold_predictions: Dict[str, List[OOSPrediction]] = {}
        
        # Aggregated OOS data for BMA updates
        self.aggregated_oos: Optional[pd.DataFrame] = None
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"RealOOSManager initialized with {len(self.oos_history)} historical predictions")
    
    def add_fold_predictions(self, 
                           fold_id: str,
                           model_predictions: Dict[str, pd.Series],
                           actuals: pd.Series,
                           model_versions: Optional[Dict[str, str]] = None) -> bool:
        """
        Add predictions from a CV fold
        
        Args:
            fold_id: Unique identifier for the fold
            model_predictions: Dictionary of model_name -> predictions
            actuals: Actual target values
            model_versions: Optional model version tracking
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now()
            feature_hash = self._compute_feature_hash(model_predictions)
            
            for model_name, predictions in model_predictions.items():
                oos_pred = OOSPrediction(
                    timestamp=timestamp,
                    fold_id=fold_id,
                    model_name=model_name,
                    predictions=predictions,
                    actuals=actuals,
                    feature_hash=feature_hash,
                    model_version=model_versions.get(model_name, 'unknown') if model_versions else 'unknown'
                )
                
                # Add to history
                self.oos_history.append(oos_pred)
                
                # Add to current fold tracking
                if fold_id not in self.current_fold_predictions:
                    self.current_fold_predictions[fold_id] = []
                self.current_fold_predictions[fold_id].append(oos_pred)
            
            # Update aggregated OOS
            self._update_aggregated_oos()
            
            logger.info(f"Added OOS predictions for fold {fold_id} with {len(model_predictions)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add fold predictions: {e}")
            return False
    
    def get_bma_update_data(self, 
                           min_folds: int = 3,
                           lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get aggregated OOS data for BMA weight updates
        
        Args:
            min_folds: Minimum number of folds required
            lookback_days: Days of history to consider
            
        Returns:
            DataFrame with columns: target, model1_pred, model2_pred, ...
        """
        
        # Filter by recency
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_predictions = [
            p for p in self.oos_history 
            if p.timestamp >= cutoff_date
        ]
        
        if len(recent_predictions) < self.min_samples_for_update:
            logger.warning(f"Insufficient OOS samples: {len(recent_predictions)} < {self.min_samples_for_update}")
            return None
        
        # Check fold diversity
        unique_folds = len(set(p.fold_id for p in recent_predictions))
        if unique_folds < min_folds:
            logger.warning(f"Insufficient fold diversity: {unique_folds} < {min_folds}")
            return None
        
        # Aggregate predictions by model
        model_data = {}
        actuals_list = []
        
        # Group by model
        model_groups = {}
        for pred in recent_predictions:
            if pred.model_name not in model_groups:
                model_groups[pred.model_name] = []
            model_groups[pred.model_name].append(pred)
        
        # Combine predictions for each model
        for model_name, preds in model_groups.items():
            combined_preds = pd.concat([p.predictions for p in preds])
            combined_actuals = pd.concat([p.actuals for p in preds])
            
            model_data[f'{model_name}_pred'] = combined_preds
            if 'target' not in model_data:
                model_data['target'] = combined_actuals
        
        # Create DataFrame
        oos_df = pd.DataFrame(model_data)
        
        # Remove rows with any NaN
        oos_df = oos_df.dropna()
        
        if len(oos_df) < self.min_samples_for_update:
            logger.warning(f"Insufficient clean samples after NaN removal: {len(oos_df)}")
            return None
        
        logger.info(f"Prepared BMA update data with {len(oos_df)} samples from {unique_folds} folds")
        
        return oos_df
    
    def get_model_performance_history(self, 
                                     model_name: str,
                                     lookback_days: int = 30) -> pd.DataFrame:
        """
        Get historical performance metrics for a specific model
        
        Args:
            model_name: Name of the model
            lookback_days: Days of history to analyze
            
        Returns:
            DataFrame with performance metrics over time
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        model_predictions = [
            p for p in self.oos_history 
            if p.model_name == model_name and p.timestamp >= cutoff_date
        ]
        
        if not model_predictions:
            return pd.DataFrame()
        
        # Calculate metrics for each prediction
        metrics_list = []
        for pred in model_predictions:
            metrics = pred.calculate_metrics()
            if metrics:
                metrics['timestamp'] = pred.timestamp
                metrics['fold_id'] = pred.fold_id
                metrics['model_version'] = pred.model_version
                metrics_list.append(metrics)
        
        if not metrics_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics_list)
        df = df.sort_values('timestamp')
        
        # Add rolling statistics
        if len(df) > 3:
            df['ic_rolling_mean'] = df['ic'].rolling(window=3, min_periods=1).mean()
            df['ic_rolling_std'] = df['ic'].rolling(window=3, min_periods=1).std()
        
        return df
    
    def get_fold_comparison(self, fold_ids: List[str]) -> pd.DataFrame:
        """
        Compare performance across specific folds
        
        Args:
            fold_ids: List of fold IDs to compare
            
        Returns:
            DataFrame with fold comparison metrics
        """
        comparison_data = []
        
        for fold_id in fold_ids:
            if fold_id not in self.current_fold_predictions:
                continue
                
            fold_preds = self.current_fold_predictions[fold_id]
            
            for pred in fold_preds:
                metrics = pred.calculate_metrics()
                if metrics:
                    metrics['fold_id'] = fold_id
                    metrics['model_name'] = pred.model_name
                    metrics['timestamp'] = pred.timestamp
                    comparison_data.append(metrics)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Pivot for easier comparison
        pivot_df = df.pivot_table(
            index='fold_id',
            columns='model_name',
            values='ic',
            aggfunc='mean'
        )
        
        return pivot_df
    
    def _update_aggregated_oos(self):
        """Update aggregated OOS data"""
        
        # Get recent data
        oos_df = self.get_bma_update_data()
        
        if oos_df is not None:
            self.aggregated_oos = oos_df
            logger.debug(f"Updated aggregated OOS with {len(oos_df)} samples")
    
    def _compute_feature_hash(self, model_predictions: Dict[str, pd.Series]) -> str:
        """Compute hash for feature set identification"""
        import hashlib
        
        # Use first model's index as feature identifier
        if model_predictions:
            first_pred = next(iter(model_predictions.values()))
            index_str = str(sorted(first_pred.index.tolist()))
            return hashlib.md5(index_str.encode()).hexdigest()[:8]
        return "unknown"
    
    def _save_cache(self):
        """Save OOS history to cache"""
        try:
            cache_file = self.cache_dir / "oos_history.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(list(self.oos_history), f)
            
            # Also save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'n_predictions': len(self.oos_history),
                'unique_folds': len(set(p.fold_id for p in self.oos_history)),
                'unique_models': len(set(p.model_name for p in self.oos_history))
            }
            
            metadata_file = self.cache_dir / "oos_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Saved OOS cache with {len(self.oos_history)} predictions")
            
        except Exception as e:
            logger.error(f"Failed to save OOS cache: {e}")
    
    def _load_cache(self):
        """Load OOS history from cache"""
        try:
            cache_file = self.cache_dir / "oos_history.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    history_list = pickle.load(f)
                    self.oos_history = deque(history_list, maxlen=self.max_history)
                    
                # Rebuild current fold predictions
                for pred in self.oos_history:
                    if pred.fold_id not in self.current_fold_predictions:
                        self.current_fold_predictions[pred.fold_id] = []
                    self.current_fold_predictions[pred.fold_id].append(pred)
                    
                logger.info(f"Loaded {len(self.oos_history)} OOS predictions from cache")
                
        except Exception as e:
            logger.warning(f"Failed to load OOS cache: {e}")
            self.oos_history = deque(maxlen=self.max_history)
    
    def clear_old_predictions(self, days_to_keep: int = 60):
        """Clear predictions older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Filter history
        new_history = [
            p for p in self.oos_history 
            if p.timestamp >= cutoff_date
        ]
        
        removed = len(self.oos_history) - len(new_history)
        
        self.oos_history = deque(new_history, maxlen=self.max_history)
        
        # Rebuild current fold predictions
        self.current_fold_predictions.clear()
        for pred in self.oos_history:
            if pred.fold_id not in self.current_fold_predictions:
                self.current_fold_predictions[pred.fold_id] = []
            self.current_fold_predictions[pred.fold_id].append(pred)
        
        # Save updated cache
        self._save_cache()
        
        logger.info(f"Cleared {removed} old OOS predictions")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of OOS data"""
        
        if not self.oos_history:
            return {
                'total_predictions': 0,
                'unique_folds': 0,
                'unique_models': 0,
                'date_range': None
            }
        
        timestamps = [p.timestamp for p in self.oos_history]
        
        return {
            'total_predictions': len(self.oos_history),
            'unique_folds': len(set(p.fold_id for p in self.oos_history)),
            'unique_models': len(set(p.model_name for p in self.oos_history)),
            'date_range': (min(timestamps), max(timestamps)),
            'avg_samples_per_fold': len(self.oos_history) / len(set(p.fold_id for p in self.oos_history))
        }


# Example usage and testing
if __name__ == "__main__":
    # Create OOS manager
    oos_manager = RealOOSManager()
    
    print("\n" + "=" * 60)
    print("REAL OOS MANAGER TEST")
    print("=" * 60)
    
    # Simulate adding fold predictions
    # np.random.seed removed
    
    for fold_id in range(5):
        # Create mock predictions
        n_samples = 100
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
        
        model_predictions = {
            'lightgbm': pd.Series(np.zeros(n_samples), index=dates),
            'xgboost': pd.Series(np.zeros(n_samples), index=dates),
            'catboost': pd.Series(np.zeros(n_samples), index=dates)
        }
        
        actuals = pd.Series(np.zeros(n_samples), index=dates)
        
        # Add to manager
        success = oos_manager.add_fold_predictions(
            fold_id=f"fold_{fold_id}",
            model_predictions=model_predictions,
            actuals=actuals
        )
        
        print(f"Added fold {fold_id}: {'Success' if success else 'Failed'}")
    
    # Get BMA update data
    print("\nGetting BMA update data...")
    bma_data = oos_manager.get_bma_update_data()
    
    if bma_data is not None:
        print(f"BMA update data shape: {bma_data.shape}")
        print(f"Columns: {list(bma_data.columns)}")
        print(f"Sample correlations:")
        print(bma_data.corr())
    
    # Get performance history
    print("\nModel performance history for 'lightgbm':")
    perf_history = oos_manager.get_model_performance_history('lightgbm')
    if not perf_history.empty:
        print(perf_history[['timestamp', 'fold_id', 'ic', 'n_samples']].head())
    
    # Get summary statistics
    print("\nSummary statistics:")
    summary = oos_manager.get_summary_statistics()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)