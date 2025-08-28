#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Timing Configuration for BMA Enhanced Model
统一时序配置 - 解决多处真值源问题

This module provides a single source of truth for all timing-related parameters
to prevent information leakage and ensure consistency across the system.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTimingConfig:
    """
    Unified timing configuration - single source of truth
    统一时序配置 - 单一真相源
    """
    
    # Core prediction parameters (核心预测参数)
    prediction_horizon: int = 10  # T+10 prediction
    feature_lag: int = 5  # T-5 feature lag (optimized from T-0 to T-5)
    
    # Isolation parameters (隔离参数)
    # CRITICAL: effective_isolation = prediction_horizon + feature_lag
    effective_isolation: int = field(init=False)  # Will be computed
    
    # Cross-validation parameters (交叉验证参数)
    cv_gap_days: int = field(init=False)  # Will match effective_isolation
    cv_embargo_days: int = field(init=False)  # Will match effective_isolation
    cv_n_splits: int = 5
    
    # Walk-forward parameters (滚动窗口参数)
    train_window_months: int = 24  # 2 years
    test_window_months: int = 6
    step_months: int = 3
    
    # Purge parameters (清洗参数)
    purge_days: int = field(init=False)  # Will match effective_isolation
    embargo_days: int = field(init=False)  # Will match effective_isolation
    
    # Time decay parameters (时间衰减参数)
    sample_weight_halflife: int = 75  # Optimized from 120 to 75
    
    # Feature-specific decay (特征族衰减)
    factor_decay_mapping: Dict[str, int] = field(default_factory=lambda: {
        'momentum': 20,
        'reversal': 5,
        'value': 60,
        'quality': 90,
        'volatility': 10,
        'liquidity': 15,
        'sentiment': 7,
        'technical': 15,
        'fundamental': 45
    })
    
    # Isolation method selection (隔离方法选择)
    isolation_method: str = 'purge'  # 'purge' or 'embargo', not both
    
    # Safety parameters (安全参数)
    safety_gap: int = 2  # Additional safety buffer
    min_train_samples: int = 252  # Minimum 1 year of data
    
    # Logging and validation (日志和验证)
    verbose: bool = True
    strict_validation: bool = True
    
    def __post_init__(self):
        """Calculate derived parameters to ensure consistency"""
        
        # Calculate effective isolation
        self.effective_isolation = self.prediction_horizon + self.feature_lag
        
        # Synchronize all isolation parameters
        self.cv_gap_days = self.effective_isolation
        self.cv_embargo_days = self.effective_isolation
        self.purge_days = self.effective_isolation
        self.embargo_days = self.effective_isolation
        
        # Log configuration
        if self.verbose:
            logger.info("=" * 60)
            logger.info("UNIFIED TIMING CONFIGURATION INITIALIZED")
            logger.info("=" * 60)
            logger.info(f"Prediction Horizon: T+{self.prediction_horizon}")
            logger.info(f"Feature Lag: T-{self.feature_lag}")
            logger.info(f"Effective Isolation: {self.effective_isolation} days")
            logger.info(f"Isolation Method: {self.isolation_method}")
            logger.info(f"CV Gap/Embargo: {self.cv_gap_days} days")
            logger.info(f"Sample Weight Halflife: {self.sample_weight_halflife} days")
            logger.info("=" * 60)
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate timing configuration for consistency"""
        
        errors = []
        warnings = []
        
        # Check effective isolation
        if self.effective_isolation < self.prediction_horizon:
            errors.append(
                f"Effective isolation ({self.effective_isolation}) < prediction horizon ({self.prediction_horizon}). "
                "This may cause information leakage!"
            )
        
        # Check minimum isolation
        min_required_isolation = self.prediction_horizon + self.safety_gap
        if self.effective_isolation < min_required_isolation:
            warnings.append(
                f"Effective isolation ({self.effective_isolation}) < recommended minimum ({min_required_isolation}). "
                "Consider increasing feature lag or safety gap."
            )
        
        # Check CV parameters
        if self.cv_gap_days != self.effective_isolation:
            errors.append(
                f"CV gap ({self.cv_gap_days}) != effective isolation ({self.effective_isolation}). "
                "This breaks the single source of truth!"
            )
        
        # Check isolation method
        if self.isolation_method not in ['purge', 'embargo']:
            errors.append(
                f"Invalid isolation method: {self.isolation_method}. "
                "Must be 'purge' or 'embargo'."
            )
        
        # Check train window
        min_train_days = self.train_window_months * 30
        if min_train_days < self.min_train_samples:
            warnings.append(
                f"Train window ({min_train_days} days) may be too short. "
                f"Recommended minimum: {self.min_train_samples} days."
            )
        
        # Handle validation results
        if errors:
            error_msg = "\n".join(errors)
            if self.strict_validation:
                raise ValueError(f"Timing configuration validation failed:\n{error_msg}")
            else:
                logger.error(f"Timing configuration errors:\n{error_msg}")
        
        if warnings:
            warning_msg = "\n".join(warnings)
            logger.warning(f"Timing configuration warnings:\n{warning_msg}")
    
    def update_from_lag_optimization(self, optimal_lag: int):
        """
        Update configuration after lag optimization
        
        Args:
            optimal_lag: Optimized feature lag from A/B testing
        """
        old_lag = self.feature_lag
        self.feature_lag = optimal_lag
        
        # Recalculate effective isolation
        old_isolation = self.effective_isolation
        self.__post_init__()  # Recalculate all derived parameters
        
        logger.info(f"Configuration updated from lag optimization:")
        logger.info(f"  Feature lag: {old_lag} -> {optimal_lag}")
        logger.info(f"  Effective isolation: {old_isolation} -> {self.effective_isolation}")
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration"""
        return {
            'n_splits': self.cv_n_splits,
            'gap': self.cv_gap_days,
            'embargo': self.cv_embargo_days if self.isolation_method == 'embargo' else 0,
            'purge': self.purge_days if self.isolation_method == 'purge' else 0,
            'test_size': self.test_window_months * 30,  # Convert to days
        }
    
    def get_walk_forward_config(self) -> Dict[str, Any]:
        """Get walk-forward configuration"""
        return {
            'train_window_days': self.train_window_months * 30,
            'test_window_days': self.test_window_months * 30,
            'step_days': self.step_months * 30,
            'gap': self.effective_isolation,
        }
    
    def get_factor_decay(self, factor_family: str) -> int:
        """
        Get decay halflife for specific factor family
        
        Args:
            factor_family: Name of factor family
            
        Returns:
            Decay halflife in days
        """
        return self.factor_decay_mapping.get(
            factor_family, 
            self.sample_weight_halflife  # Default to global halflife
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'prediction_horizon': self.prediction_horizon,
            'feature_lag': self.feature_lag,
            'effective_isolation': self.effective_isolation,
            'cv_gap_days': self.cv_gap_days,
            'cv_embargo_days': self.cv_embargo_days,
            'cv_n_splits': self.cv_n_splits,
            'train_window_months': self.train_window_months,
            'test_window_months': self.test_window_months,
            'step_months': self.step_months,
            'purge_days': self.purge_days,
            'embargo_days': self.embargo_days,
            'sample_weight_halflife': self.sample_weight_halflife,
            'factor_decay_mapping': self.factor_decay_mapping,
            'isolation_method': self.isolation_method,
            'safety_gap': self.safety_gap,
            'min_train_samples': self.min_train_samples,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedTimingConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def validate_data_alignment(self, 
                               train_end: Any, 
                               test_start: Any,
                               date_column: str = 'date') -> bool:
        """
        Validate that train/test split respects timing isolation
        
        Args:
            train_end: End date/index of training data
            test_start: Start date/index of test data
            date_column: Name of date column
            
        Returns:
            True if validation passes
        """
        import pandas as pd
        
        # Convert to datetime if needed
        if not isinstance(train_end, pd.Timestamp):
            train_end = pd.to_datetime(train_end)
        if not isinstance(test_start, pd.Timestamp):
            test_start = pd.to_datetime(test_start)
        
        # Calculate actual gap
        actual_gap = (test_start - train_end).days
        
        # Check if gap is sufficient
        is_valid = actual_gap >= self.effective_isolation
        
        if self.verbose:
            status = "✓ VALID" if is_valid else "✗ INVALID"
            logger.info(f"Data alignment validation: {status}")
            logger.info(f"  Train end: {train_end}")
            logger.info(f"  Test start: {test_start}")
            logger.info(f"  Actual gap: {actual_gap} days")
            logger.info(f"  Required gap: {self.effective_isolation} days")
        
        if not is_valid and self.strict_validation:
            raise ValueError(
                f"Insufficient isolation: {actual_gap} days < {self.effective_isolation} days required. "
                f"Train ends at {train_end}, test starts at {test_start}."
            )
        
        return is_valid


# Singleton instance for global access
_global_timing_config: Optional[UnifiedTimingConfig] = None


def get_unified_timing_config() -> UnifiedTimingConfig:
    """Get the global unified timing configuration"""
    global _global_timing_config
    if _global_timing_config is None:
        _global_timing_config = UnifiedTimingConfig()
    return _global_timing_config


def set_unified_timing_config(config: UnifiedTimingConfig):
    """Set the global unified timing configuration"""
    global _global_timing_config
    _global_timing_config = config
    logger.info("Global timing configuration updated")


def reset_timing_config():
    """Reset to default timing configuration"""
    global _global_timing_config
    _global_timing_config = UnifiedTimingConfig()
    logger.info("Timing configuration reset to defaults")


# Example usage and testing
if __name__ == "__main__":
    # Create unified config
    config = UnifiedTimingConfig()
    
    print("\n" + "=" * 60)
    print("UNIFIED TIMING CONFIGURATION TEST")
    print("=" * 60)
    
    # Display configuration
    print(f"\nCore Parameters:")
    print(f"  Prediction Horizon: T+{config.prediction_horizon}")
    print(f"  Feature Lag: T-{config.feature_lag}")
    print(f"  Effective Isolation: {config.effective_isolation} days")
    
    print(f"\nCV Configuration:")
    cv_config = config.get_cv_config()
    for key, value in cv_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nWalk-Forward Configuration:")
    wf_config = config.get_walk_forward_config()
    for key, value in wf_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nFactor Decay Halflives:")
    for family in ['momentum', 'reversal', 'value', 'volatility']:
        print(f"  {family}: {config.get_factor_decay(family)} days")
    
    # Test lag optimization update
    print(f"\nTesting lag optimization update...")
    config.update_from_lag_optimization(optimal_lag=3)
    print(f"  New effective isolation: {config.effective_isolation} days")
    
    # Test data alignment validation
    print(f"\nTesting data alignment validation...")
    import pandas as pd
    train_end = pd.Timestamp('2024-01-01')
    test_start_valid = pd.Timestamp('2024-01-16')  # 15 days gap
    test_start_invalid = pd.Timestamp('2024-01-10')  # 9 days gap
    
    print(f"  Valid gap (15 days): ", end="")
    try:
        config.validate_data_alignment(train_end, test_start_valid)
        print("PASS")
    except ValueError as e:
        print(f"FAIL: {e}")
    
    print(f"  Invalid gap (9 days): ", end="")
    try:
        # Temporarily disable strict validation for testing
        config.strict_validation = False
        is_valid = config.validate_data_alignment(train_end, test_start_invalid)
        print("PASS (warning mode)" if not is_valid else "Unexpected pass")
    except ValueError as e:
        print(f"Expected failure: {e}")
    
    print("\n" + "=" * 60)
    print("CONFIGURATION TEST COMPLETE")
    print("=" * 60)