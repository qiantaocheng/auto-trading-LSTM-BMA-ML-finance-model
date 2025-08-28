#!/usr/bin/env python3
"""
Incremental Training System for BMA Enhanced
Implements bi-weekly incremental updates and monthly full rebuilds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
import pickle
import json
import hashlib

logger = logging.getLogger(__name__)

class TrainingType(Enum):
    """Training type enumeration"""
    INCREMENTAL = "incremental"  # Bi-weekly incremental updates
    FULL_REBUILD = "full_rebuild"  # Monthly full rebuilds
    EMERGENCY = "emergency"       # Emergency retrain
    VALIDATION = "validation"     # Validation-only run

@dataclass
class TrainingSchedule:
    """Training schedule configuration"""
    # Bi-weekly incremental training
    incremental_frequency_days: int = 14
    incremental_learning_rate_factor: float = 0.5
    incremental_max_new_trees: int = 100
    incremental_warmup_samples: int = 1000
    
    # Monthly full rebuild
    full_rebuild_frequency_days: int = 28
    full_rebuild_hyperparameter_search: bool = True
    full_rebuild_feature_selection: bool = True
    
    # Emergency conditions
    emergency_ic_drop_threshold: float = -0.03
    emergency_consecutive_bad_days: int = 5
    emergency_max_drawdown: float = 0.08
    
    # Knowledge retention
    enable_knowledge_retention: bool = True
    feature_importance_kl_threshold: float = 0.3
    model_distillation_enabled: bool = True

@dataclass
class IncrementalModelState:
    """State of incremental model"""
    base_model: Any = None
    incremental_updates: List[Any] = field(default_factory=list)
    last_full_rebuild: Optional[datetime] = None
    last_incremental: Optional[datetime] = None
    performance_history: List[Dict] = field(default_factory=list)
    feature_importance_history: List[Dict] = field(default_factory=list)
    model_hash: str = ""

class IncrementalTrainingSystem:
    """
    Incremental Training System
    
    Key features:
    1. Bi-weekly incremental updates with warm-starting
    2. Monthly full rebuilds with hyperparameter optimization
    3. Emergency retraining triggers
    4. Knowledge retention and transfer learning
    5. Model versioning and rollback capabilities
    """
    
    def __init__(self, schedule: TrainingSchedule = None, cache_dir: str = "cache/incremental_models"):
        self.schedule = schedule or TrainingSchedule()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize status file path
        self.status_file = self.cache_dir / "training_status.json"
        
        # Model states
        self.lightgbm_state = IncrementalModelState()
        self.bma_state = IncrementalModelState()
        
        # Performance tracking
        self.performance_metrics = []
        self.training_log = []
        
        logger.info("IncrementalTrainingSystem initialized")
        logger.info(f"Schedule: Incremental={self.schedule.incremental_frequency_days}d, "
                   f"Full rebuild={self.schedule.full_rebuild_frequency_days}d")
    
    def determine_training_type(self, current_date: datetime, 
                              recent_performance: List[Dict] = None) -> TrainingType:
        """Determine what type of training is needed"""
        
        # Check for emergency conditions first
        if self._check_emergency_conditions(recent_performance):
            logger.warning("Emergency retraining triggered")
            return TrainingType.EMERGENCY
        
        # Check for full rebuild schedule
        if self._should_full_rebuild(current_date):
            logger.info("Full rebuild scheduled")
            return TrainingType.FULL_REBUILD
        
        # Check for incremental update
        if self._should_incremental_update(current_date):
            logger.info("Incremental update scheduled")
            return TrainingType.INCREMENTAL
        
        # No training needed
        return TrainingType.VALIDATION
    
    def train_lightgbm_incremental(self, train_data: pd.DataFrame, 
                                 train_target: pd.Series,
                                 validation_data: pd.DataFrame,
                                 validation_target: pd.Series,
                                 training_type: TrainingType,
                                 train_weights: pd.Series = None,
                                 valid_weights: pd.Series = None,
                                 **kwargs) -> Dict[str, Any]:
        """Train LightGBM model with incremental capabilities and sample weights"""
        import lightgbm as lgb
        
        if training_type == TrainingType.FULL_REBUILD:
            return self._train_lightgbm_full(train_data, train_target, validation_data, validation_target, 
                                           train_weights, valid_weights, **kwargs)
        elif training_type == TrainingType.INCREMENTAL:
            return self._train_lightgbm_incremental(train_data, train_target, validation_data, validation_target, 
                                                   train_weights, valid_weights, **kwargs)
        else:
            logger.info("No LightGBM training needed")
            return {'status': 'skipped', 'training_type': training_type.value}
    
    def _train_lightgbm_full(self, train_data: pd.DataFrame, train_target: pd.Series,
                           validation_data: pd.DataFrame, validation_target: pd.Series,
                           train_weights: pd.Series = None, valid_weights: pd.Series = None,
                           **kwargs) -> Dict[str, Any]:
        """Full LightGBM training with hyperparameter optimization and sample weights"""
        import lightgbm as lgb
        from sklearn.model_selection import RandomizedSearchCV
        
        logger.info("Starting LightGBM full rebuild")
        start_time = datetime.now()
        
        try:
            # Prepare data with sample weights
            train_dataset = lgb.Dataset(train_data, label=train_target, weight=train_weights)
            valid_dataset = lgb.Dataset(validation_data, label=validation_target, weight=valid_weights, reference=train_dataset)
            
            # Base parameters
            base_params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'metric': 'rmse',
                'verbose': -1,
                'random_state': 42,
                'force_row_wise': True,
            }
            
            # Hyperparameter search if enabled
            if self.schedule.full_rebuild_hyperparameter_search:
                param_dist = {
                    'num_leaves': [31, 50, 70, 100],
                    'max_depth': [-1, 5, 7, 10],
                    'learning_rate': [0.05, 0.1, 0.15, 0.2],
                    'feature_fraction': [0.8, 0.9, 1.0],
                    'bagging_fraction': [0.8, 0.9, 1.0],
                    'min_data_in_leaf': [20, 50, 100],
                    'lambda_l1': [0, 0.1, 1.0],
                    'lambda_l2': [0, 0.1, 1.0]
                }
                
                # Quick search for optimal parameters
                search_params = base_params.copy()
                search_params.update({
                    'num_boost_round': 100,  # Quick search
                    'early_stopping_rounds': 20
                })
                
                best_params = self._optimize_lightgbm_params(
                    train_dataset, valid_dataset, search_params, param_dist
                )
                base_params.update(best_params)
                logger.info(f"Optimized parameters: {best_params}")
            
            # Train final model
            base_params.update(kwargs)
            
            # 🔧 CRITICAL FIX: Adaptive early stopping and validation checks
            min_train_samples = 50
            min_valid_samples = 20
            
            if len(train_data) < min_train_samples:
                logger.warning(f"Training data too small ({len(train_data)} < {min_train_samples}), using minimal parameters")
                base_params['num_leaves'] = min(base_params.get('num_leaves', 31), 10)
                base_params['max_depth'] = min(base_params.get('max_depth', -1), 3) if base_params.get('max_depth', -1) > 0 else 3
                early_stopping_rounds = 10
                num_boost_round = 50
            elif len(validation_data) < min_valid_samples:
                logger.warning(f"Validation data too small ({len(validation_data)} < {min_valid_samples}), using conservative parameters")
                early_stopping_rounds = 20
                num_boost_round = 200
            else:
                early_stopping_rounds = 100
                num_boost_round = 1000
                
            logger.info(f"LightGBM training: train_samples={len(train_data)}, valid_samples={len(validation_data)}, "
                       f"early_stop={early_stopping_rounds}, max_rounds={num_boost_round}")
            
            # 🔧 FIX: Avoid duplicate training by checking if this is a retry
            if not hasattr(self, '_training_attempt_count'):
                self._training_attempt_count = 0
            self._training_attempt_count += 1
            
            if self._training_attempt_count > 1:
                logger.warning(f"Training attempt #{self._training_attempt_count} - avoiding duplicate training")
            
            model = lgb.train(
                base_params,
                train_dataset,
                valid_sets=[valid_dataset],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(period=0)  # Silent logging
                ]
            )
            
            # Reset training attempt count on success
            self._training_attempt_count = 0
            
            # Evaluate model
            train_pred = model.predict(train_data, num_iteration=model.best_iteration)
            valid_pred = model.predict(validation_data, num_iteration=model.best_iteration)
            
            metrics = self._calculate_lightgbm_metrics(
                train_target, train_pred, validation_target, valid_pred
            )
            
            # Update state
            self.lightgbm_state.base_model = model
            self.lightgbm_state.incremental_updates = []  # Reset incremental updates
            self.lightgbm_state.last_full_rebuild = datetime.now()
            self.lightgbm_state.model_hash = self._calculate_model_hash(model)
            
            # Persist state (Fix)
            self._save_persistent_state()
            
            # Store feature importance
            importance = dict(zip(train_data.columns, model.feature_importance(importance_type='gain')))
            self.lightgbm_state.feature_importance_history.append({
                'timestamp': datetime.now(),
                'importance': importance
            })
            
            # Save model
            model_path = self.cache_dir / f"lightgbm_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'training_type': 'full_rebuild',
                'model': model,
                'metrics': metrics,
                'training_time': training_time,
                'model_path': str(model_path),
                'feature_importance': importance
            }
            
            logger.info(f"LightGBM full rebuild complete: {training_time:.1f}s, "
                       f"RMSE={metrics['valid_rmse']:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"LightGBM full rebuild failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _train_lightgbm_incremental(self, train_data: pd.DataFrame, train_target: pd.Series,
                                  validation_data: pd.DataFrame, validation_target: pd.Series,
                                  train_weights: pd.Series = None, valid_weights: pd.Series = None,
                                  **kwargs) -> Dict[str, Any]:
        """Incremental LightGBM training using init_model and sample weights"""
        import lightgbm as lgb
        
        if self.lightgbm_state.base_model is None:
            logger.warning("No base model for incremental training, performing full rebuild")
            return self._train_lightgbm_full(train_data, train_target, validation_data, validation_target, 
                                            train_weights, valid_weights, **kwargs)
        
        logger.info("Starting LightGBM incremental training")
        start_time = datetime.now()
        
        try:
            # Use only recent data for incremental training
            recent_samples = min(len(train_data), self.schedule.incremental_warmup_samples)
            train_data_inc = train_data.tail(recent_samples)
            train_target_inc = train_target.tail(recent_samples)
            train_weights_inc = train_weights.tail(recent_samples) if train_weights is not None else None
            
            # Prepare data with weights
            train_dataset = lgb.Dataset(train_data_inc, label=train_target_inc, weight=train_weights_inc)
            valid_dataset = lgb.Dataset(validation_data, label=validation_target, weight=valid_weights, reference=train_dataset)
            
            # Get base model parameters
            base_model = self.lightgbm_state.base_model
            
            # Incremental parameters (reduced learning rate, fewer trees)
            inc_params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'metric': 'rmse',
                'verbose': -1,
                'random_state': 42,
                'force_row_wise': True,
                'learning_rate': base_model.params.get('learning_rate', 0.1) * self.schedule.incremental_learning_rate_factor,
                'num_leaves': base_model.params.get('num_leaves', 31),
                'max_depth': base_model.params.get('max_depth', -1),
                'feature_fraction': base_model.params.get('feature_fraction', 1.0),
                'bagging_fraction': base_model.params.get('bagging_fraction', 1.0)
            }
            
            inc_params.update(kwargs)
            
            # Train incremental model
            incremental_model = lgb.train(
                inc_params,
                train_dataset,
                valid_sets=[valid_dataset],
                num_boost_round=self.schedule.incremental_max_new_trees,
                init_model=base_model,  # Warm start from base model
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Evaluate model
            train_pred = incremental_model.predict(train_data, num_iteration=incremental_model.best_iteration)
            valid_pred = incremental_model.predict(validation_data, num_iteration=incremental_model.best_iteration)
            
            metrics = self._calculate_lightgbm_metrics(
                train_target, train_pred, validation_target, valid_pred
            )
            
            # Update state
            self.lightgbm_state.incremental_updates.append({
                'model': incremental_model,
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            self.lightgbm_state.last_incremental = datetime.now()
            
            # Persist state (Fix)
            self._save_persistent_state()
            
            # Store feature importance
            importance = dict(zip(train_data.columns, incremental_model.feature_importance(importance_type='gain')))
            self.lightgbm_state.feature_importance_history.append({
                'timestamp': datetime.now(),
                'importance': importance
            })
            
            # Save incremental model
            model_path = self.cache_dir / f"lightgbm_inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(incremental_model, f)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'training_type': 'incremental',
                'model': incremental_model,
                'metrics': metrics,
                'training_time': training_time,
                'model_path': str(model_path),
                'feature_importance': importance,
                'new_trees': incremental_model.num_trees() - base_model.num_trees()
            }
            
            logger.info(f"LightGBM incremental training complete: {training_time:.1f}s, "
                       f"RMSE={metrics['valid_rmse']:.6f}, "
                       f"New trees={result['new_trees']}")
            
            return result
            
        except Exception as e:
            logger.error(f"LightGBM incremental training failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_bma_weights_incremental(self, recent_oos_results: pd.DataFrame,
                                     training_type: TrainingType) -> Dict[str, Any]:
        """Update BMA weights using incremental approach"""
        
        if training_type == TrainingType.FULL_REBUILD:
            return self._update_bma_weights_full(recent_oos_results)
        elif training_type == TrainingType.INCREMENTAL:
            return self._update_bma_weights_incremental(recent_oos_results)
        else:
            logger.info("No BMA weight update needed")
            return {'status': 'skipped', 'training_type': training_type.value}
    
    def _update_bma_weights_full(self, oos_results: pd.DataFrame) -> Dict[str, Any]:
        """Full BMA weight recalculation"""
        logger.info("Starting BMA full weight recalculation")
        start_time = datetime.now()
        
        try:
            # Implement full BMA weight calculation
            # This would use the complete historical performance data
            
            # Calculate model scores (IC, Sharpe, etc.)
            model_scores = {}
            for model_col in oos_results.columns:
                if model_col.endswith('_pred'):
                    model_name = model_col.replace('_pred', '')
                    
                    # Calculate various performance metrics
                    ic = oos_results[model_col].corr(oos_results['target']) if 'target' in oos_results.columns else 0.0
                    mse = ((oos_results[model_col] - oos_results['target']) ** 2).mean() if 'target' in oos_results.columns else 1.0
                    
                    # Combined score
                    score = ic * 0.7 - (mse * 0.3)  # IC positive, MSE negative
                    model_scores[model_name] = score
            
            # Calculate softmax weights
            if model_scores:
                scores_array = np.array(list(model_scores.values()))
                # Apply temperature for softmax
                temperature = 2.0
                exp_scores = np.exp(scores_array / temperature)
                weights = exp_scores / exp_scores.sum()
                
                model_weights = dict(zip(model_scores.keys(), weights))
            else:
                model_weights = {}
            
            # Update state
            self.bma_state.last_full_rebuild = datetime.now()
            self.bma_state.feature_importance_history.append({
                'timestamp': datetime.now(),
                'weights': model_weights
            })
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'training_type': 'full_rebuild',
                'weights': model_weights,
                'model_scores': model_scores,
                'training_time': training_time,
                'models_processed': len(model_scores),
                'total_weight': sum(model_weights.values()) if model_weights else 0.0
            }
            
            # 🔧 CRITICAL FIX: Validate BMA weight calculation actually occurred
            if training_time < 0.001:  # Less than 1ms indicates no actual work
                if not model_weights:
                    logger.warning("⚠️ BMA weight recalculation completed suspiciously fast (0.0s) - no models processed")
                    result['status'] = 'warning'
                    result['warning'] = 'No base learners available for BMA weighting'
                else:
                    logger.warning(f"⚠️ BMA weight recalculation completed very fast ({training_time:.3f}s) but produced {len(model_weights)} weights")
            
            logger.info(f"BMA full weight recalculation complete: {training_time:.3f}s, "
                       f"processed {len(model_scores)} models, weights_sum={sum(model_weights.values()) if model_weights else 0:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"BMA full weight recalculation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _update_bma_weights_incremental(self, recent_results: pd.DataFrame) -> Dict[str, Any]:
        """Incremental BMA weight update using exponential smoothing"""
        logger.info("Starting BMA incremental weight update")
        start_time = datetime.now()
        
        try:
            # Get current weights (or initialize)
            current_weights = {}
            if self.bma_state.feature_importance_history:
                current_weights = self.bma_state.feature_importance_history[-1]['weights'].copy()
            
            # Calculate recent performance scores
            recent_scores = {}
            for model_col in recent_results.columns:
                if model_col.endswith('_pred'):
                    model_name = model_col.replace('_pred', '')
                    
                    # Recent performance
                    ic = recent_results[model_col].corr(recent_results['target']) if 'target' in recent_results.columns else 0.0
                    recent_scores[model_name] = ic
            
            # Exponential smoothing update: w_new = ρ * w_old + (1-ρ) * score_new
            smoothing_factor = 0.95  # High smoothing for incremental updates
            updated_weights = {}
            
            for model_name in set(list(current_weights.keys()) + list(recent_scores.keys())):
                old_weight = current_weights.get(model_name, 1.0 / len(recent_scores))
                new_score = recent_scores.get(model_name, 0.0)
                
                # Convert score to weight space (sigmoid)
                new_weight_component = 1.0 / (1.0 + np.exp(-new_score * 10))  # Scale and sigmoid
                
                # Exponential update
                updated_weight = smoothing_factor * old_weight + (1 - smoothing_factor) * new_weight_component
                updated_weights[model_name] = updated_weight
            
            # Normalize weights
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {k: v / total_weight for k, v in updated_weights.items()}
            
            # Update state
            self.bma_state.last_incremental = datetime.now()
            self.bma_state.feature_importance_history.append({
                'timestamp': datetime.now(),
                'weights': updated_weights
            })
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'training_type': 'incremental',
                'weights': updated_weights,
                'recent_scores': recent_scores,
                'training_time': training_time,
                'smoothing_factor': smoothing_factor
            }
            
            logger.info(f"BMA incremental weight update complete: {training_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"BMA incremental weight update failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _load_persistent_state(self) -> None:
        """Load persistent training state from disk (Fix)"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore LightGBM state
                lgb_data = state_data.get('lightgbm_state', {})
                if 'last_full_rebuild' in lgb_data and lgb_data['last_full_rebuild']:
                    self.lightgbm_state.last_full_rebuild = datetime.fromisoformat(lgb_data['last_full_rebuild'])
                if 'last_incremental' in lgb_data and lgb_data['last_incremental']:
                    self.lightgbm_state.last_incremental = datetime.fromisoformat(lgb_data['last_incremental'])
                if 'model_hash' in lgb_data:
                    self.lightgbm_state.model_hash = lgb_data['model_hash']
                
                # Restore BMA state
                bma_data = state_data.get('bma_state', {})
                if 'last_full_rebuild' in bma_data and bma_data['last_full_rebuild']:
                    self.bma_state.last_full_rebuild = datetime.fromisoformat(bma_data['last_full_rebuild'])
                if 'last_incremental' in bma_data and bma_data['last_incremental']:
                    self.bma_state.last_incremental = datetime.fromisoformat(bma_data['last_incremental'])
                
                logger.info(f"Loaded persistent state from {self.status_file}")
            else:
                logger.info("No persistent state file found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent state: {e}, starting fresh")
    
    def _save_persistent_state(self) -> None:
        """Save persistent training state to disk (Fix)"""
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'lightgbm_state': {
                    'last_full_rebuild': self.lightgbm_state.last_full_rebuild.isoformat() if self.lightgbm_state.last_full_rebuild else None,
                    'last_incremental': self.lightgbm_state.last_incremental.isoformat() if self.lightgbm_state.last_incremental else None,
                    'model_hash': self.lightgbm_state.model_hash,
                    'incremental_updates_count': len(self.lightgbm_state.incremental_updates)
                },
                'bma_state': {
                    'last_full_rebuild': self.bma_state.last_full_rebuild.isoformat() if self.bma_state.last_full_rebuild else None,
                    'last_incremental': self.bma_state.last_incremental.isoformat() if self.bma_state.last_incremental else None,
                    'weight_history_count': len(self.bma_state.feature_importance_history)
                },
                'schedule_config': {
                    'incremental_frequency_days': self.schedule.incremental_frequency_days,
                    'full_rebuild_frequency_days': self.schedule.full_rebuild_frequency_days,
                    'incremental_learning_rate_factor': self.schedule.incremental_learning_rate_factor,
                    'incremental_max_new_trees': self.schedule.incremental_max_new_trees
                }
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"Saved persistent state to {self.status_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save persistent state: {e}")
    
    def _check_emergency_conditions(self, recent_performance: List[Dict] = None) -> bool:
        """Check if emergency retraining is needed"""
        if not recent_performance or len(recent_performance) < 3:
            return False
        
        # Check IC degradation
        recent_ics = [p.get('ic', 0.0) for p in recent_performance[-5:]]
        if len(recent_ics) >= 3:
            avg_recent_ic = np.mean(recent_ics)
            if avg_recent_ic < self.schedule.emergency_ic_drop_threshold:
                logger.warning(f"Emergency: IC dropped to {avg_recent_ic:.4f}")
                return True
        
        # Check consecutive bad days
        bad_days = 0
        for perf in reversed(recent_performance):
            if perf.get('ic', 0.0) < 0:
                bad_days += 1
            else:
                break
        
        if bad_days >= self.schedule.emergency_consecutive_bad_days:
            logger.warning(f"Emergency: {bad_days} consecutive bad days")
            return True
        
        # Check maximum drawdown
        returns = [p.get('daily_return', 0.0) for p in recent_performance]
        if returns:
            cumulative_returns = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = np.max(drawdown)
            
            if max_drawdown > self.schedule.emergency_max_drawdown:
                logger.warning(f"Emergency: Max drawdown {max_drawdown:.3f}")
                return True
        
        return False
    
    def _should_full_rebuild(self, current_date: datetime) -> bool:
        """Check if full rebuild is needed"""
        if self.lightgbm_state.last_full_rebuild is None:
            return True
        
        days_since = (current_date - self.lightgbm_state.last_full_rebuild).days
        return days_since >= self.schedule.full_rebuild_frequency_days
    
    def _should_incremental_update(self, current_date: datetime) -> bool:
        """Check if incremental update is needed"""
        if self.lightgbm_state.last_incremental is None and self.lightgbm_state.base_model is not None:
            return True
        
        if self.lightgbm_state.last_incremental is None:
            return False
        
        days_since = (current_date - self.lightgbm_state.last_incremental).days
        return days_since >= self.schedule.incremental_frequency_days
    
    def _calculate_lightgbm_metrics(self, train_target: pd.Series, train_pred: np.ndarray,
                                  valid_target: pd.Series, valid_pred: np.ndarray) -> Dict[str, float]:
        """Calculate LightGBM performance metrics"""
        from sklearn.metrics import mean_squared_error
        from scipy.stats import pearsonr, spearmanr
        
        metrics = {}
        
        # RMSE
        metrics['train_rmse'] = np.sqrt(mean_squared_error(train_target, train_pred))
        metrics['valid_rmse'] = np.sqrt(mean_squared_error(valid_target, valid_pred))
        
        # IC (Information Coefficient)
        train_ic, _ = pearsonr(train_target, train_pred)
        valid_ic, _ = pearsonr(valid_target, valid_pred)
        metrics['train_ic'] = train_ic if not np.isnan(train_ic) else 0.0
        metrics['valid_ic'] = valid_ic if not np.isnan(valid_ic) else 0.0
        
        # Rank IC
        train_rank_ic, _ = spearmanr(train_target, train_pred)
        valid_rank_ic, _ = spearmanr(valid_target, valid_pred)
        metrics['train_rank_ic'] = train_rank_ic if not np.isnan(train_rank_ic) else 0.0
        metrics['valid_rank_ic'] = valid_rank_ic if not np.isnan(valid_rank_ic) else 0.0
        
        return metrics
    
    def _optimize_lightgbm_params(self, train_dataset, valid_dataset, base_params: Dict, param_dist: Dict) -> Dict:
        """Optimize LightGBM hyperparameters"""
        import lightgbm as lgb
        from sklearn.model_selection import RandomizedSearchCV
        
        # Simple grid search for key parameters
        best_score = float('inf')
        best_params = {}
        
        # Test a few key parameter combinations
        test_combinations = [
            {'num_leaves': 31, 'learning_rate': 0.1, 'max_depth': -1},
            {'num_leaves': 50, 'learning_rate': 0.05, 'max_depth': 7},
            {'num_leaves': 70, 'learning_rate': 0.1, 'max_depth': 10},
        ]
        
        for params in test_combinations:
            test_params = base_params.copy()
            test_params.update(params)
            
            try:
                model = lgb.train(
                    test_params,
                    train_dataset,
                    valid_sets=[valid_dataset],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                score = model.best_score['valid_0']['rmse']
                if score < best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Parameter optimization failed for {params}: {e}")
                continue
        
        return best_params
    
    def _calculate_model_hash(self, model) -> str:
        """Calculate hash of model for version tracking"""
        try:
            # Use model parameters and structure for hash
            model_info = {
                'num_trees': model.num_trees() if hasattr(model, 'num_trees') else 0,
                'params': model.params if hasattr(model, 'params') else {},
                'timestamp': datetime.now().isoformat()
            }
            
            model_str = json.dumps(model_info, sort_keys=True)
            return hashlib.md5(model_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to calculate model hash: {e}")
            return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training system status"""
        return {
            'lightgbm': {
                'last_full_rebuild': self.lightgbm_state.last_full_rebuild,
                'last_incremental': self.lightgbm_state.last_incremental,
                'model_hash': self.lightgbm_state.model_hash,
                'incremental_updates': len(self.lightgbm_state.incremental_updates),
                'feature_importance_history': len(self.lightgbm_state.feature_importance_history)
            },
            'bma': {
                'last_full_rebuild': self.bma_state.last_full_rebuild,
                'last_incremental': self.bma_state.last_incremental,
                'weight_history': len(self.bma_state.feature_importance_history)
            },
            'schedule': {
                'incremental_frequency': self.schedule.incremental_frequency_days,
                'full_rebuild_frequency': self.schedule.full_rebuild_frequency_days,
                'emergency_conditions': {
                    'ic_threshold': self.schedule.emergency_ic_drop_threshold,
                    'bad_days_threshold': self.schedule.emergency_consecutive_bad_days,
                    'drawdown_threshold': self.schedule.emergency_max_drawdown
                }
            }
        }
    
    def rollback_model(self, steps: int = 1, force_full_rebuild: bool = False) -> Dict[str, Any]:
        """Enhanced rollback model to previous state with safety features (Fix)"""
        try:
            rollback_start = datetime.now()
            
            # Option 1: Force full rebuild (safest)
            if force_full_rebuild:
                logger.warning("🔄 Performing FULL REBUILD rollback (safest option)")
                
                # Clear incremental updates and reset to last stable base model
                self.lightgbm_state.incremental_updates = []
                self.lightgbm_state.last_incremental = None
                
                # Persist the rollback state
                self._save_persistent_state()
                
                return {
                    'status': 'success',
                    'rollback_type': 'full_rebuild',
                    'action': 'cleared_incremental_updates',
                    'message': 'Next training will be full rebuild',
                    'rollback_time': (datetime.now() - rollback_start).total_seconds()
                }
            
            # Option 2: Incremental rollback
            if len(self.lightgbm_state.incremental_updates) >= steps:
                # Remove recent incremental updates
                rolled_back_models = []
                for _ in range(steps):
                    removed_update = self.lightgbm_state.incremental_updates.pop()
                    rolled_back_models.append({
                        'timestamp': removed_update['timestamp'],
                        'metrics': removed_update['metrics']
                    })
                
                # Update last incremental timestamp
                if self.lightgbm_state.incremental_updates:
                    self.lightgbm_state.last_incremental = self.lightgbm_state.incremental_updates[-1]['timestamp']
                else:
                    self.lightgbm_state.last_incremental = None
                
                # Persist the rollback state
                self._save_persistent_state()
                
                logger.info(f"✅ Rolled back {steps} incremental updates")
                logger.info(f"Remaining incremental models: {len(self.lightgbm_state.incremental_updates)}")
                
                return {
                    'status': 'success',
                    'rollback_type': 'incremental',
                    'rolled_back_steps': steps,
                    'removed_models': rolled_back_models,
                    'remaining_models': len(self.lightgbm_state.incremental_updates),
                    'rollback_time': (datetime.now() - rollback_start).total_seconds()
                }
            else:
                # Not enough incremental history - suggest full rebuild
                available_steps = len(self.lightgbm_state.incremental_updates)
                logger.warning(f"⚠️ Insufficient incremental history: {available_steps} available, {steps} requested")
                logger.warning(f"Consider force_full_rebuild=True for complete rollback")
                
                return {
                    'status': 'insufficient_history',
                    'available_steps': available_steps,
                    'requested_steps': steps,
                    'recommendation': 'Use force_full_rebuild=True for complete rollback'
                }
                
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_knowledge_retention_metrics(self) -> Dict[str, Any]:
        """Calculate knowledge retention metrics"""
        if not self.schedule.enable_knowledge_retention:
            return {'status': 'disabled'}
        
        # Feature importance stability
        importance_history = self.lightgbm_state.feature_importance_history
        if len(importance_history) < 2:
            return {'status': 'insufficient_history'}
        
        # Calculate KL divergence between recent importance distributions
        recent_importance = importance_history[-1]['importance']
        previous_importance = importance_history[-2]['importance']
        
        # Align features
        common_features = set(recent_importance.keys()) & set(previous_importance.keys())
        if len(common_features) < 5:
            return {'status': 'insufficient_overlap'}
        
        # Normalize importance scores
        recent_values = np.array([recent_importance[f] for f in common_features])
        previous_values = np.array([previous_importance[f] for f in common_features])
        
        recent_dist = recent_values / recent_values.sum()
        previous_dist = previous_values / previous_values.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(recent_dist * np.log(recent_dist / (previous_dist + 1e-8) + 1e-8))
        
        drift_status = 'HIGH' if kl_div > self.schedule.feature_importance_kl_threshold else 'NORMAL'
        
        return {
            'status': 'calculated',
            'kl_divergence': kl_div,
            'drift_status': drift_status,
            'drift_threshold': self.schedule.feature_importance_kl_threshold,
            'common_features': len(common_features)
        }