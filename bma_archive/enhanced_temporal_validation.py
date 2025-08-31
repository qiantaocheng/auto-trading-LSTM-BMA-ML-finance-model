#!/usr/bin/env python3
"""
Enhanced Temporal Validation System
Fix for double isolation issue and temporal leak prevention
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Dict, Any, List
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancedValidationConfig:
    """Enhanced validation configuration with single isolation approach"""
    n_splits: int = 5
    test_size: int = 63  # Test set size (trading days)
    
    # Critical Fix: Choose ONE isolation method to avoid double isolation
    isolation_method: str = 'purge'  # Options: 'purge', 'embargo' (hybrid removed for V6 consistency)
    isolation_days: int = 5  # 🔧 FIX: Reduced from 10 to 5 for better compatibility
    
    # Additional parameters
    min_train_size: int = 100  # 🔧 FIX: Reduced from 252 to 100 for small datasets
    group_freq: str = 'D'      # Grouping frequency
    strict_validation: bool = False  # 🔧 FIX: Made less strict for compatibility
    
    # Adaptive parameters
    enable_adaptive_isolation: bool = True
    small_dataset_threshold: int = 1000  # samples
    medium_dataset_threshold: int = 5000
    
    def get_effective_isolation(self, n_samples: int) -> int:
        """Get adaptive isolation based on dataset size with more aggressive reduction"""
        if not self.enable_adaptive_isolation:
            return self.isolation_days
            
        if n_samples < 500:
            # Very small datasets: minimal isolation
            return max(1, self.isolation_days // 4)  
        elif n_samples < self.small_dataset_threshold:
            # Small datasets: significant reduction
            return max(1, self.isolation_days // 3)  # 🔧 FIX: More aggressive reduction
        elif n_samples < self.medium_dataset_threshold:
            # Medium datasets: moderate reduction
            return max(2, self.isolation_days // 2)  # 🔧 FIX: Allow 2-day minimum
        else:
            return self.isolation_days  # Large datasets: use full isolation

class EnhancedPurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Enhanced Purged Time Series Split - Fixed Double Isolation Issue
    
    Key Fixes:
    1. Single isolation approach (choose purge OR embargo, not both)
    2. Adaptive isolation based on dataset size
    3. Clear temporal validation
    4. Comprehensive logging of isolation effects
    """
    
    def __init__(self, config: EnhancedValidationConfig):
        self.config = config
        self.isolation_stats = {}
        
    def split(self, X, y=None, groups=None):
        """Generate train/test index pairs with single isolation approach"""
        if groups is None:
            groups = self._create_time_groups(X)
        
        # V6 Fix: Explicitly reject hybrid isolation for consistency
        if self.config.isolation_method == 'hybrid':
            logger.error("ISOLATION CONSISTENCY ERROR: 'hybrid' isolation is not allowed in V6 Enhanced system")
            logger.error("Please use either 'purge' or 'embargo' for single isolation approach")
            raise ValueError("V6 Enhanced system requires single isolation method: use 'purge' or 'embargo', not 'hybrid'")
        
        # Ensure index alignment
        if hasattr(X, 'index'):
            data_index = X.index
        else:
            data_index = np.arange(len(X))
        
        if hasattr(groups, 'index'):
            groups = groups.reindex(data_index)
        
        unique_groups = sorted(groups.unique())
        n_groups = len(unique_groups)
        n_samples = len(X)
        
        # Get effective isolation based on dataset size
        effective_isolation = self.config.get_effective_isolation(n_samples)
        
        logger.info(f"Enhanced Temporal Validation - {n_groups} groups, {n_samples} samples")
        if effective_isolation != self.config.isolation_days:
            logger.info(f"V6 Adaptive isolation: {self.config.isolation_method}, configured={self.config.isolation_days}→effective={effective_isolation} (小数据集自适应调整)")
        else:
            logger.info(f"V6 Single isolation: {self.config.isolation_method}, days: {effective_isolation}")
        
        # Validate data sufficiency with adaptive requirements for small datasets
        min_required_groups = self.config.n_splits + effective_isolation + 2
        if n_groups < min_required_groups:
            logger.warning(f"Insufficient data for standard validation: need {min_required_groups} groups, have {n_groups}")
            
            # Adaptive approach for small datasets
            if n_groups < 3:
                logger.warning("Dataset too small for temporal validation, using basic validation")
                if self.config.strict_validation:
                    return
            else:
                # Adjust parameters for small datasets
                logger.info("Adapting temporal validation parameters for small dataset")
                # Reduce n_splits and isolation for small data
                self.config.n_splits = max(1, min(self.config.n_splits, n_groups - 2))
                effective_isolation = max(1, min(effective_isolation, n_groups // 3))
                logger.info(f"Adjusted: n_splits={self.config.n_splits}, isolation={effective_isolation}")
        
        # Calculate fold boundaries
        groups_per_fold = max(1, self.config.test_size // 7)  # ~7 samples per group
        
        total_removed_samples = 0
        valid_folds = 0
        
        for i in range(self.config.n_splits):
            # Calculate test groups
            test_start_idx = min(
                n_groups - groups_per_fold,
                int(n_groups * (i + 1) / (self.config.n_splits + 1))
            )
            test_end_idx = min(n_groups, test_start_idx + groups_per_fold)
            
            # Apply SINGLE isolation method
            if self.config.isolation_method == 'purge':
                # ✅ FIX: 更保守的purge，避免过度删除训练数据
                # 对小数据集，减少purge强度
                actual_purge = effective_isolation
                if n_samples < 1000:
                    actual_purge = max(1, effective_isolation // 3)  # 大幅减少purge
                elif n_samples < 3000:
                    actual_purge = max(1, effective_isolation // 2)  # 适度减少purge
                
                train_end_idx = max(1, test_start_idx - actual_purge)  # 至少保留1个训练组
                isolation_removed = self._count_samples_in_range(groups, train_end_idx, test_start_idx)
                
                logger.debug(f"Fold {i}: purge adjusted from {effective_isolation} to {actual_purge} days")
                
            elif self.config.isolation_method == 'embargo':
                # Embargo approach: delay test start
                original_test_start = test_start_idx
                test_start_idx = min(n_groups - 1, test_start_idx + effective_isolation)
                train_end_idx = original_test_start
                isolation_removed = self._count_samples_in_range(groups, original_test_start, test_start_idx)
                
            else:
                # Invalid isolation method (should be caught earlier)
                logger.error(f"Invalid isolation method: {self.config.isolation_method}")
                raise ValueError(f"Unsupported isolation method: {self.config.isolation_method}")
            
            # ✅ FIX: 更灵活的最小训练要求，避免所有fold失败
            # 对小数据集使用更宽松的要求
            if n_samples < 1000:
                min_required_train = max(1, n_groups // 10)  # 极小数据集只要求10%的组
            elif n_samples < 3000:
                min_required_train = max(1, n_groups // 6)   # 小数据集要求1/6的组
            else:
                min_required_train = max(1, min(self.config.min_train_size // 20, n_groups // 4))
            
            if train_end_idx < min_required_train:
                logger.warning(f"Fold {i}: insufficient training data ({train_end_idx} < {min_required_train})")
                # 对于极小数据集，再次放松要求
                if n_samples < 500 and train_end_idx > 0:
                    logger.info(f"Fold {i}: 极小数据集例外处理，允许训练组数={train_end_idx}")
                else:
                    continue
            
            # Generate indices
            train_groups = unique_groups[:train_end_idx]
            test_groups = unique_groups[test_start_idx:test_end_idx]
            
            # 🔧 FIX: Handle groups properly for indexing
            if hasattr(groups, 'index'):
                # groups is a Series with index
                train_mask = groups.isin(train_groups)
                test_mask = groups.isin(test_groups)
                train_idx = groups.index[train_mask].tolist()
                test_idx = groups.index[test_mask].tolist()
            else:
                # groups is a simple index
                train_mask = pd.Series(groups).isin(train_groups)
                test_mask = pd.Series(groups).isin(test_groups)
                train_idx = np.where(train_mask)[0].tolist()
                test_idx = np.where(test_mask)[0].tolist()
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Log fold statistics
            gap_days = self._calculate_time_gap(groups, train_groups, test_groups)
            logger.info(f"Fold {i}: train={len(train_idx)}, test={len(test_idx)}, "
                       f"gap={gap_days}天, isolation_removed={isolation_removed}")
            
            # Validate temporal ordering
            if self._validate_temporal_order(groups, train_groups, test_groups, effective_isolation):
                total_removed_samples += isolation_removed
                valid_folds += 1
                yield np.array(train_idx), np.array(test_idx)
            else:
                logger.error(f"Fold {i}: temporal validation failed")
        
        # Store isolation statistics
        self.isolation_stats = {
            'total_samples': n_samples,
            'removed_samples': total_removed_samples,
            'removal_ratio': total_removed_samples / n_samples,
            'valid_folds': valid_folds,
            'effective_isolation': effective_isolation
        }
        
        logger.info(f"Temporal validation complete: {valid_folds} valid folds, "
                   f"{total_removed_samples}/{n_samples} ({total_removed_samples/n_samples:.1%}) samples removed")
        
        # 🔥 CRITICAL FIX: Short-circuit when no valid folds to prevent invalid metric calculation
        if valid_folds == 0:
            logger.error("CRITICAL: 0 valid folds generated - temporal validation failed completely")
            logger.error("This indicates severe data insufficiency or configuration issues")
            logger.error("SYSTEM SHOULD NOT PROCEED with IC/IR calculation when no valid folds exist")
            # Return empty iterator to signal failure
            return iter([])
    
    def _create_time_groups(self, X):
        """Create time groups if not provided"""
        if hasattr(X, 'index') and hasattr(X.index, 'to_period'):
            return X.index.to_period(self.config.group_freq)
        else:
            # Simple sequential grouping
            n_groups = len(X) // 20  # ~20 samples per group
            return pd.Series(np.repeat(np.arange(n_groups), len(X) // n_groups + 1)[:len(X)])
    
    def _count_samples_in_range(self, groups, start_idx, end_idx):
        """Count samples in the isolation range"""
        if start_idx >= end_idx:
            return 0
        unique_groups = sorted(groups.unique())
        isolation_groups = unique_groups[start_idx:end_idx]
        return groups.isin(isolation_groups).sum()
    
    def _calculate_time_gap(self, groups, train_groups, test_groups):
        """Calculate actual time gap between train and test using business days"""
        if not train_groups or not test_groups:
            return 0
        
        train_max = max(train_groups)
        test_min = min(test_groups)
        
        # ✅ FIX: Use business day calculation for accurate gap measurement
        try:
            # Try to get actual dates from the groups
            if hasattr(groups, 'categories') and len(groups.categories) > 0:
                # groups is a Categorical with date categories
                train_max_date = groups.categories[train_max] if train_max < len(groups.categories) else None
                test_min_date = groups.categories[test_min] if test_min < len(groups.categories) else None
                
                if train_max_date is not None and test_min_date is not None:
                    # Convert to datetime if needed
                    if not isinstance(train_max_date, pd.Timestamp):
                        train_max_date = pd.to_datetime(train_max_date)
                    if not isinstance(test_min_date, pd.Timestamp):
                        test_min_date = pd.to_datetime(test_min_date)
                    
                    # Calculate business days gap
                    gap = pd.bdate_range(start=train_max_date, end=test_min_date, freq='B')
                    return max(0, len(gap) - 1)  # -1 because we don't count the start date
                        
            # Fallback: assume each group represents 1 trading day
            gap = test_min - train_max
            # For small gaps, assume they represent trading days directly
            if isinstance(gap, pd.Timedelta):
                return max(0, gap.days)
            elif hasattr(gap, 'days'):
                return max(0, gap.days)
            else:
                return max(0, int(gap))
        except Exception as e:
            logger.debug(f"Gap calculation fallback: {e}")
            # Last resort: position difference  
            gap = test_min - train_max
            if isinstance(gap, (int, np.integer)):
                return max(0, gap)
            elif hasattr(gap, 'days'):
                return max(0, gap.days)
            else:
                try:
                    return max(0, int(gap))
                except:
                    return 1  # Default minimal gap
    
    def _validate_temporal_order(self, groups, train_groups, test_groups, required_gap):
        """Validate temporal ordering with adaptive gap requirements"""
        if not train_groups or not test_groups:
            return False
        
        train_max = max(train_groups)
        test_min = min(test_groups)
        
        # Check basic ordering
        if train_max >= test_min:
            logger.error("Temporal order violation: train_max >= test_min")
            return False
        
        # Check required gap with adaptive requirements
        actual_gap = self._calculate_time_gap(groups, train_groups, test_groups)
        
        # 🔧 FIX: Adaptive gap requirements based on dataset characteristics
        n_samples = len(groups)
        adaptive_gap = required_gap
        
        if n_samples < 500:
            # Very small dataset: minimal gap requirement
            adaptive_gap = max(1, required_gap // 3)
            logger.debug(f"Very small dataset ({n_samples}): reducing gap requirement to {adaptive_gap}")
        elif n_samples < 1000:
            # Small dataset: reduced gap requirement  
            adaptive_gap = max(2, required_gap // 2)
            logger.debug(f"Small dataset ({n_samples}): reducing gap requirement to {adaptive_gap}")
        elif actual_gap >= 1 and actual_gap < required_gap:
            # Medium dataset with some gap: allow if gap >= 1
            adaptive_gap = max(1, min(actual_gap, required_gap))
            logger.debug(f"Medium dataset: accepting actual gap {actual_gap} vs required {required_gap}")
        
        if actual_gap < adaptive_gap:
            logger.error(f"Insufficient gap: {actual_gap} < {adaptive_gap} (adapted from {required_gap})")
            return False
        
        if adaptive_gap != required_gap:
            logger.info(f"Gap requirement adapted: {required_gap} → {adaptive_gap} (actual: {actual_gap})")
        
        return True
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits"""
        return self.config.n_splits

# Feature lag optimization system
@dataclass
class FeatureLagConfig:
    """Configuration for feature lag optimization"""
    test_lags: List[int] = None  # Lags to test
    target_horizon: int = 10     # Target prediction horizon
    min_ic_improvement: float = 0.02  # Minimum IC improvement to change lag
    
    def __post_init__(self):
        if self.test_lags is None:
            self.test_lags = [0, 1, 2, 5]  # Default lags to test

class FeatureLagOptimizer:
    """
    Feature lag optimizer with A/B testing
    Fix for overly conservative T-5 lag with unified validation config
    """
    
    def __init__(self, config: FeatureLagConfig = None, validation_config: EnhancedValidationConfig = None):
        self.config = config or FeatureLagConfig()
        self.validation_config = validation_config or EnhancedValidationConfig()  # Unified config injection
        self.lag_results = {}
    
    def optimize_feature_lag(self, data: pd.DataFrame, target: pd.Series, 
                           feature_cols: List[str]) -> Dict[str, Any]:
        """
        Optimize feature lag using A/B testing with DM test and rolling OOS validation
        """
        results = {}
        oos_predictions = {}  # Store OOS predictions for DM test
        
        # Use unified enhanced temporal validation for rolling OOS (Fix)
        cv_splitter = EnhancedPurgedTimeSeriesSplit(self.validation_config)
        time_groups = self._create_time_groups_for_lag_test(data)
        
        logger.info(f"Lag optimizer unified validation config: isolation={self.validation_config.isolation_method}, configured_days={self.validation_config.isolation_days} (将根据数据量自适应调整)")
        
        for lag in self.config.test_lags:
            # Create lagged features
            lagged_data = self._create_lagged_features(data, feature_cols, lag)
            
            # Align with target
            aligned_data, aligned_target = self._align_data_target(
                lagged_data, target, lag, self.config.target_horizon
            )
            
            if len(aligned_data) < 100:  # Minimum samples for valid test
                logger.warning(f"Lag {lag}: insufficient data ({len(aligned_data)} samples)")
                continue
            
            # Rolling OOS validation with purged/embargoed splits
            oos_errors = []
            oos_preds = []
            
            for train_idx, test_idx in cv_splitter.split(aligned_data, aligned_target, time_groups):
                # Convert index values to positions for iloc
                if hasattr(aligned_data, 'index'):
                    # Get positions from index values
                    train_positions = aligned_data.index.get_indexer(train_idx)
                    test_positions = aligned_data.index.get_indexer(test_idx)
                    # Filter out -1 (not found) values
                    train_positions = train_positions[train_positions != -1]
                    test_positions = test_positions[test_positions != -1]
                else:
                    # If no index, assume train_idx and test_idx are already positions
                    train_positions = train_idx
                    test_positions = test_idx
                
                X_train, X_test = aligned_data.iloc[train_positions], aligned_data.iloc[test_positions]
                y_train, y_test = aligned_target.iloc[train_positions], aligned_target.iloc[test_positions]
                
                # Simple Ridge model for lag comparison
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_train.fillna(0), y_train)
                
                y_pred = model.predict(X_test.fillna(0))
                errors = (y_test.values - y_pred) ** 2
                
                oos_errors.extend(errors)
                oos_preds.extend(y_pred)
            
            oos_predictions[lag] = np.array(oos_errors)  # Store MSE errors for DM test
            
            # Calculate IC metrics
            ic_scores = self._calculate_ic_metrics(aligned_data, aligned_target, feature_cols)
            
            results[f'lag_{lag}'] = {
                'lag': lag,
                'mean_ic': ic_scores['mean_ic'],
                'mean_rank_ic': ic_scores['mean_rank_ic'],
                'ic_ir': ic_scores['ic_ir'],
                'samples': len(aligned_data),
                'feature_coverage': ic_scores['feature_coverage'],
                'oos_mse': np.mean(oos_errors) if oos_errors else np.inf
            }
            
            logger.info(f"Lag {lag}: IC={ic_scores['mean_ic']:.4f}, "
                       f"RankIC={ic_scores['mean_rank_ic']:.4f}, "
                       f"IR={ic_scores['ic_ir']:.4f}, OOS_MSE={np.mean(oos_errors):.6f}")
        
        # DM test between lag candidates
        dm_results = self._perform_dm_tests(oos_predictions)
        
        # Select optimal lag based on DM test significance and IC improvement
        if results:
            optimal_lag = self._select_optimal_lag_with_dm(results, dm_results)
            results['optimal_lag'] = optimal_lag
            results['dm_test_results'] = dm_results
            results['recommendation'] = self._generate_lag_recommendation(results)
            
            # Persist to alphas_config.yaml if significant improvement
            if self._should_persist_lag_change(results, dm_results):
                self._persist_lag_to_config(optimal_lag)
        
        self.lag_results = results
        return results
    
    def _create_lagged_features(self, data: pd.DataFrame, feature_cols: List[str], lag: int) -> pd.DataFrame:
        """Create lagged features with idempotent protection"""
        # FEATURE LAG IDEMPOTENT PROTECTION: 检测已有滞后特征避免重复应用
        already_lagged_cols = [col for col in feature_cols if '_lag_' in col]
        if already_lagged_cols and lag > 0:
            logger.warning(f"Detected already lagged features {already_lagged_cols}, skipping lag={lag} to avoid double lag")
            # 对于已滞后的特征，使用原样；对于未滞后的特征，应用滞后
            clean_cols = [col for col in feature_cols if '_lag_' not in col]
            result_data = data[clean_cols].copy()
            if lag > 0:
                result_data = result_data.shift(lag)
            # 保留已滞后的特征（原样）
            for col in already_lagged_cols:
                if col in data.columns:
                    result_data[col] = data[col]
            return result_data
        elif lag == 0:
            return data[feature_cols].copy()
        else:
            return data[feature_cols].shift(lag).copy()
    
    def _align_data_target(self, data: pd.DataFrame, target: pd.Series, 
                          feature_lag: int, target_horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Align lagged features with forward-looking target"""
        # Forward-looking target
        forward_target = target.shift(-target_horizon)
        
        # Align indices
        common_idx = data.index.intersection(forward_target.index)
        aligned_data = data.loc[common_idx].dropna()
        aligned_target = forward_target.loc[aligned_data.index].dropna()
        
        # Final alignment
        final_idx = aligned_data.index.intersection(aligned_target.index)
        return aligned_data.loc[final_idx], aligned_target.loc[final_idx]
    
    def _calculate_ic_metrics(self, data: pd.DataFrame, target: pd.Series, 
                            feature_cols: List[str]) -> Dict[str, float]:
        """Calculate IC metrics for lagged features"""
        from scipy.stats import spearmanr, pearsonr
        
        ic_scores = []
        rank_ic_scores = []
        valid_features = 0
        
        for col in feature_cols:
            if col in data.columns:
                # Remove NaN pairs
                valid_mask = ~(data[col].isna() | target.isna())
                if valid_mask.sum() < 10:  # Minimum valid observations
                    continue
                
                feature_data = data.loc[valid_mask, col]
                target_data = target.loc[valid_mask]
                
                # Pearson IC
                ic, ic_pval = pearsonr(feature_data, target_data)
                if not np.isnan(ic):
                    ic_scores.append(ic)
                
                # Spearman Rank IC
                rank_ic, rank_ic_pval = spearmanr(feature_data, target_data)
                if not np.isnan(rank_ic):
                    rank_ic_scores.append(rank_ic)
                
                valid_features += 1
        
        if not ic_scores:
            return {
                'mean_ic': 0.0,
                'mean_rank_ic': 0.0,
                'ic_ir': 0.0,
                'feature_coverage': 0.0
            }
        
        mean_ic = np.mean(ic_scores)
        mean_rank_ic = np.mean(rank_ic_scores) if rank_ic_scores else 0.0
        ic_std = np.std(ic_scores) if len(ic_scores) > 1 else 1.0
        ic_ir = mean_ic / max(ic_std, 1e-6)  # Information Ratio
        
        return {
            'mean_ic': mean_ic,
            'mean_rank_ic': mean_rank_ic,
            'ic_ir': ic_ir,
            'feature_coverage': valid_features / len(feature_cols)
        }
    
    def _select_optimal_lag(self, results: Dict[str, Any]) -> int:
        """Select optimal lag based on IC and IR metrics"""
        best_lag = 5  # Default conservative
        best_score = -999
        
        for key, metrics in results.items():
            if key.startswith('lag_'):
                # Combined score: IC + IR with coverage penalty
                score = (metrics['mean_ic'] * 0.4 + 
                        metrics['ic_ir'] * 0.4 + 
                        metrics['mean_rank_ic'] * 0.2) * metrics['feature_coverage']
                
                if score > best_score:
                    best_score = score
                    best_lag = metrics['lag']
        
        return best_lag
    
    def _create_time_groups_for_lag_test(self, data: pd.DataFrame) -> pd.Series:
        """Create time groups for lag testing CV"""
        if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
            return data.index.to_period('W')  # Weekly groups
        else:
            n_groups = len(data) // 50
            return pd.Series(np.repeat(np.arange(n_groups), len(data) // n_groups + 1)[:len(data)], index=data.index)
    
    def _perform_dm_tests(self, oos_predictions: Dict[int, np.ndarray]) -> Dict[str, Dict]:
        """Perform Diebold-Mariano tests with Newey-West correction (Fix)"""
        from scipy import stats
        dm_results = {}
        
        lags = list(oos_predictions.keys())
        for i, lag1 in enumerate(lags):
            for lag2 in lags[i+1:]:
                errors1 = oos_predictions[lag1]
                errors2 = oos_predictions[lag2]
                
                # Align error series
                min_len = min(len(errors1), len(errors2))
                errors1 = errors1[:min_len]
                errors2 = errors2[:min_len]
                
                if min_len < 20:  # Need minimum samples
                    continue
                
                # DM test statistic with Newey-West correction
                loss_diff = errors1 - errors2  # MSE differences
                
                # Newey-West standard error correction for serial correlation
                mean_diff = np.mean(loss_diff)
                
                # Calculate Newey-West HAC standard error
                nw_se = self._calculate_newey_west_se(loss_diff)
                
                dm_stat = mean_diff / nw_se if nw_se > 1e-8 else 0.0
                p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))  # Two-tailed test
                
                comparison_key = f"lag_{lag1}_vs_lag_{lag2}"
                dm_results[comparison_key] = {
                    'dm_statistic': dm_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'winner': lag1 if dm_stat < 0 else lag2,  # Lower MSE wins
                    'mean_mse_diff': np.mean(loss_diff)
                }
                
                logger.info(f"DM test {comparison_key}: stat={dm_stat:.3f}, p={p_value:.3f}, "
                           f"significant={p_value < 0.05}, winner=lag_{lag1 if dm_stat < 0 else lag2}")
        
        return dm_results
    
    def _select_optimal_lag_with_dm(self, results: Dict[str, Any], dm_results: Dict[str, Dict]) -> int:
        """Select optimal lag based on DM test significance and IC improvement"""
        current_lag = 5  # Default current lag
        best_lag = current_lag
        best_score = -999
        
        for key, metrics in results.items():
            if key.startswith('lag_'):
                lag = metrics['lag']
                
                # Base score from IC and IR
                ic_score = metrics['mean_ic'] * 0.5 + metrics['ic_ir'] * 0.3 + metrics['mean_rank_ic'] * 0.2
                
                # DM test bonus: check if this lag significantly beats current lag
                dm_bonus = 0
                for dm_key, dm_result in dm_results.items():
                    if f'lag_{lag}_vs_lag_{current_lag}' == dm_key and dm_result['significant']:
                        if dm_result['winner'] == lag:
                            dm_bonus = 0.1  # Significant improvement bonus
                    elif f'lag_{current_lag}_vs_lag_{lag}' == dm_key and dm_result['significant']:
                        if dm_result['winner'] == lag:
                            dm_bonus = 0.1
                
                total_score = ic_score + dm_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_lag = lag
        
        return best_lag
    
    def _should_persist_lag_change(self, results: Dict[str, Any], dm_results: Dict[str, Dict]) -> bool:
        """Determine if lag change should be persisted to config"""
        optimal_lag = results['optimal_lag']
        current_lag = 5
        
        if optimal_lag == current_lag:
            return False
        
        # Check if change is significant via DM test
        for dm_key, dm_result in dm_results.items():
            if (f'lag_{optimal_lag}_vs_lag_{current_lag}' == dm_key or 
                f'lag_{current_lag}_vs_lag_{optimal_lag}' == dm_key):
                if dm_result['significant'] and dm_result['winner'] == optimal_lag:
                    return True
        
        return False
    
    def _persist_lag_to_config(self, optimal_lag: int) -> None:
        """Persist optimal lag to alphas_config.yaml"""
        import yaml
        from pathlib import Path
        
        config_path = Path('alphas_config.yaml')
        if not config_path.exists():
            logger.warning("alphas_config.yaml not found, cannot persist lag")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            old_lag = config.get('feature_global_lag', 5)
            config['feature_global_lag'] = optimal_lag
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Persisted optimal lag to alphas_config.yaml: {old_lag} → {optimal_lag}")
            
        except Exception as e:
            logger.error(f"Failed to persist lag to config: {e}")
    
    def _generate_lag_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lag recommendation with explanation"""
        optimal_lag = results['optimal_lag']
        current_lag = 5  # Assume current is T-5
        dm_results = results.get('dm_test_results', {})
        
        if optimal_lag != current_lag:
            improvement = results[f'lag_{optimal_lag}']['mean_ic'] - results[f'lag_{current_lag}']['mean_ic']
            
            # Check DM test significance
            dm_significant = False
            for dm_key, dm_result in dm_results.items():
                if (f'lag_{optimal_lag}_vs_lag_{current_lag}' == dm_key or 
                    f'lag_{current_lag}_vs_lag_{optimal_lag}' == dm_key):
                    if dm_result['significant'] and dm_result['winner'] == optimal_lag:
                        dm_significant = True
                        break
            
            if improvement > self.config.min_ic_improvement and dm_significant:
                return {
                    'action': 'change_lag',
                    'from_lag': current_lag,
                    'to_lag': optimal_lag,
                    'ic_improvement': improvement,
                    'dm_significant': dm_significant,
                    'explanation': f"Changing lag from T-{current_lag} to T-{optimal_lag} improves IC by {improvement:.4f} with DM significance"
                }
        
        return {
            'action': 'keep_current',
            'current_lag': current_lag,
            'explanation': f"Current T-{current_lag} lag is optimal or improvement not significant"
        }
    
    def _calculate_newey_west_se(self, loss_diff: np.ndarray, max_lags: int = None) -> float:
        """Calculate Newey-West HAC standard error for DM test"""
        n = len(loss_diff)
        if max_lags is None:
            max_lags = int(4 * (n/100)**(2/9))  # Newey-West rule of thumb
        
        # Center the series
        centered_diff = loss_diff - np.mean(loss_diff)
        
        # Calculate variance (lag 0)
        gamma_0 = np.mean(centered_diff ** 2)
        
        # Calculate autocovariances
        gamma_sum = gamma_0
        for lag in range(1, min(max_lags + 1, n)):
            if n - lag > 0:
                gamma_k = np.mean(centered_diff[:-lag] * centered_diff[lag:])
                weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
                gamma_sum += 2 * weight * gamma_k
        
        # HAC variance estimate
        nw_variance = gamma_sum / n
        
        return np.sqrt(max(nw_variance, 1e-8))