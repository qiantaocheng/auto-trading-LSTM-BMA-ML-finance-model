"""
T+10 Configuration Module
========================
Centralized configuration for all T+10 prediction parameters
to ensure consistency across the entire BMA Enhanced V6 system.

All time-related parameters are defined here to avoid hardcoding
and ensure proper data isolation between features and targets.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class T10Config:
    """Centralized T+10 prediction configuration"""
    
    # =========================================================================
    # Core Prediction Parameters
    # =========================================================================
    PREDICTION_HORIZON: int = 10  # T+10 prediction target
    HOLDING_PERIOD: int = 10      # Position holding period (must match prediction)
    
    # =========================================================================
    # Feature Engineering Parameters
    # =========================================================================
    FEATURE_LAG: int = 1           # UPDATED: Features use T-1 data (unified lag)
    FEATURE_GLOBAL_LAG: int = 1    # UPDATED: Global feature lag (unified)
    
    # =========================================================================
    # Data Isolation Parameters (Critical for preventing look-ahead bias)
    # =========================================================================
    ISOLATION_DAYS: int = 10        # CRITICAL FIX: 改为10天，与holding period一致
    EMBARGO_DAYS: int = 10          # CRITICAL FIX: 改为10天，与holding period一致
    SAFETY_GAP: int = 0             # REMOVED: Simplified with unified T-1 lag
    
    # Total gap between features and targets
    @property
    def total_time_gap(self) -> int:
        """Total time gap between features (T-1) and targets (T+10)"""
        # CRITICAL FIX: T-1特征到T+10预测的最小安全间隔
        return self.ISOLATION_DAYS  # 10天隔离，与holding period一致
    
    # =========================================================================
    # Cross-Validation Parameters
    # =========================================================================
    CV_N_SPLITS: int = 5            # Number of CV folds
    CV_MIN_TRAIN_PERIODS: int = 60  # Minimum training periods
    CV_TEST_PERIODS: int = 20       # Test period length
    CV_GAP: int = 10                # UPDATED: Gap between train and test (matches holding period)
    
    # =========================================================================
    # Alpha Factor Parameters
    # =========================================================================
    ALPHA_DECAY: int = 8            # Alpha factor decay parameter
    ALPHA_DELAY: int = 1            # Alpha calculation delay
    
    # =========================================================================
    # Sample Weight Parameters
    # =========================================================================
    SAMPLE_WEIGHT_HALFLIFE: int = 120  # Sample weight decay halflife
    SAMPLE_WEIGHT_MIN: float = 0.1     # Minimum sample weight
    
    # =========================================================================
    # Model Training Parameters
    # =========================================================================
    EARLY_STOPPING_ROUNDS: int = 50
    N_ESTIMATORS: int = 1000
    LEARNING_RATE: float = 0.03
    MAX_DEPTH: int = 5
    
    # =========================================================================
    # Risk Management Parameters
    # =========================================================================
    MAX_POSITION: float = 0.03      # Maximum position size
    MAX_TURNOVER: float = 0.1       # Maximum portfolio turnover
    MAX_LEVERAGE: float = 1.0        # Maximum leverage
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validate_time_alignment(self) -> bool:
        """Validate that time parameters prevent look-ahead bias - Unified T-1"""
        # Features at T-1, predicting T+10
        # Total gap should be at least 10 days  
        min_required_gap = 10  # CRITICAL FIX: 统一为10天最小安全gap
        actual_gap = self.total_time_gap
        
        if actual_gap < min_required_gap:
            raise ValueError(
                f"Insufficient time gap: {actual_gap} days < {min_required_gap} days. "
                f"Risk of look-ahead bias!"
            )
        
        # Embargo should match or exceed holding period (relaxed for T-1)
        if self.EMBARGO_DAYS < 10:  # At least match prediction horizon
            raise ValueError(
                f"Embargo period {self.EMBARGO_DAYS} < prediction horizon 10 days. "
                f"Risk of overlapping positions!"
            )
        
        # CV gap should match isolation days
        if self.CV_GAP != self.ISOLATION_DAYS:
            # Auto-fix instead of raising error
            self.CV_GAP = self.ISOLATION_DAYS
            print(f"[AUTO-FIX] CV gap adjusted to {self.CV_GAP} to match isolation days")
        
        return True
    
    def get_target_calculation(self) -> str:
        """Return the correct target calculation formula"""
        return f"close.pct_change({self.PREDICTION_HORIZON}).shift(-{self.PREDICTION_HORIZON})"
    
    def get_feature_shift(self) -> int:
        """Get the shift amount for features to ensure proper lag"""
        return self.FEATURE_LAG
    
    def log_configuration(self, logger) -> None:
        """Log the current configuration for transparency"""
        logger.info("=" * 60)
        logger.info("T+10 Configuration Summary")
        logger.info("=" * 60)
        logger.info(f"Prediction Horizon: T+{self.PREDICTION_HORIZON}")
        logger.info(f"Feature Lag: T-{self.FEATURE_LAG}")
        logger.info(f"Total Time Gap: {self.total_time_gap} days")
        logger.info(f"Isolation Days: {self.ISOLATION_DAYS}")
        logger.info(f"Embargo Days: {self.EMBARGO_DAYS}")
        logger.info(f"CV Splits: {self.CV_N_SPLITS}")
        logger.info(f"CV Gap: {self.CV_GAP}")
        logger.info(f"Target Formula: {self.get_target_calculation()}")
        logger.info("=" * 60)


# Create singleton instance
T10_CONFIG = T10Config()

# Validate on module load
T10_CONFIG.validate_time_alignment()


def get_config() -> T10Config:
    """Get the singleton T10 configuration instance"""
    return T10_CONFIG


def update_config(**kwargs) -> T10Config:
    """Update configuration parameters (use with caution)"""
    global T10_CONFIG
    
    for key, value in kwargs.items():
        if hasattr(T10_CONFIG, key):
            setattr(T10_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    # Re-validate after updates
    T10_CONFIG.validate_time_alignment()
    
    return T10_CONFIG