"""
Enhanced Prediction Configuration
================================

Configuration management for the new Triple-Barrier + Meta-Label + OOF Isotonic pipeline
and PIT factor processing
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default configuration for enhanced prediction modules
DEFAULT_ENHANCED_CONFIG = {
    # Labeling pipeline configuration
    "labeling": {
        # Triple Barrier settings
        "tp_sigma": 2.0,                    # Take profit threshold (σ multiplier)
        "sl_sigma": 2.0,                    # Stop loss threshold (σ multiplier)
        "max_holding_days": 5,              # Maximum holding period (trading days)
        "min_ret_threshold": 0.0005,        # Minimum return threshold for labeling
        "min_ret_for_exec": 0.001,          # Minimum return for execution meta-label
        "vol_lookback": 20,                 # Volatility calculation lookback period
        
        # Meta-labeling settings
        "strategy_type": "directional",     # directional, mean_reverting, momentum
        
        # OOF calibration settings
        "oof_n_splits": 5,                  # Number of cross-validation folds
        "require_oof_validation": True,     # Require OOF validation to pass quality checks
        "min_oof_samples": 100,             # Minimum OOF samples for calibration
        "min_auc_threshold": 0.52,          # Minimum AUC for meta-label calibration
        
        # Purged CV settings
        "purged_embargo_days": 5,           # Embargo period in days for purged CV
        
        # Validation settings
        "validate_calibration": True,       # Whether to validate calibration quality
        "fallback_on_failure": True        # Use fallback if calibration fails
    },
    
    # Factor pipeline configuration
    "factors": {
        # PIT alignment settings
        "announcement_lag_days": 90,        # Financial announcement lag
        "min_stocks_per_date": 30,          # Minimum stocks per cross-section
        
        # Factor processing settings
        "winsorize_limits": [0.01, 0.99],   # Winsorization percentiles
        "standardize_method": "zscore",     # zscore or rank
        "enable_neutralization": True,      # Enable industry/size/beta neutralization
        "growth_lookback_periods": [1, 2, 3], # Years for growth factors
        
        # Neutralization settings
        "neutralize_industry": True,        # Neutralize industry effects
        "neutralize_size": True,           # Neutralize size effects  
        "neutralize_beta": True,           # Neutralize beta effects (if available)
        
        # Integration settings
        "integration_method": "concat",     # concat or weighted
        "pit_factor_weight": 0.5,          # Weight for PIT factors in weighted integration
        "existing_factor_weight": 0.5      # Weight for existing factors
    },
    
    # Unified trading core settings
    "unified_trading_core": {
        # Enhanced modules
        "enable_enhanced_prediction": True,  # Enable enhanced prediction modules
        "enable_enhanced_factors": True,     # Enable enhanced factor processing
        "enable_calibrated_signals": True,   # Enable signal calibration
        
        # Performance settings
        "batch_signal_processing": True,     # Process signals in batches
        "signal_caching": False,            # Cache calibrated signals (memory intensive)
        "max_batch_size": 100,              # Maximum signals per batch
        
        # Fallback settings
        "fallback_on_error": True,          # Use legacy methods on error
        "log_calibration_metrics": True,    # Log detailed calibration metrics
        "validate_signal_quality": True     # Validate signal quality before execution
    },
    
    # Model training settings (for offline training)
    "training": {
        # Base models
        "base_regressor": "LGBMRegressor",   # Base regression model
        "base_classifier": "LGBMClassifier", # Base classification model
        
        # Regressor parameters
        "regressor_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        
        # Classifier parameters  
        "classifier_params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        
        # Training settings
        "train_test_split": 0.8,            # Train/test split ratio
        "min_training_samples": 1000,       # Minimum samples for training
        "retrain_frequency_days": 30,       # Model retraining frequency
        "save_models": True,                # Save trained models
        "model_save_path": "models/"        # Model save directory
    }
}

class EnhancedPredictionConfig:
    """Enhanced prediction configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "enhanced_prediction_config.json"
        )
        self._config = DEFAULT_ENHANCED_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Deep merge with default config
                self._deep_merge(self._config, file_config)
                logger.info(f"Enhanced prediction config loaded from {self.config_path}")
            else:
                # Create default config file
                self.save_config()
                logger.info(f"Created default enhanced prediction config at {self.config_path}")
                
        except Exception as e:
            logger.warning(f"Failed to load enhanced prediction config: {e}, using defaults")
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_labeling_config(self) -> Dict[str, Any]:
        """Get labeling pipeline configuration"""
        return self._config.get('labeling', {})
    
    def get_factors_config(self) -> Dict[str, Any]:
        """Get factor pipeline configuration"""
        return self._config.get('factors', {})
    
    def get_unified_core_config(self) -> Dict[str, Any]:
        """Get unified trading core configuration"""
        return self._config.get('unified_trading_core', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get model training configuration"""
        return self._config.get('training', {})
    
    def is_enhanced_prediction_enabled(self) -> bool:
        """Check if enhanced prediction is enabled"""
        return self.get('unified_trading_core.enable_enhanced_prediction', True)
    
    def is_enhanced_factors_enabled(self) -> bool:
        """Check if enhanced factors are enabled"""
        return self.get('unified_trading_core.enable_enhanced_factors', True)
    
    def is_signal_calibration_enabled(self) -> bool:
        """Check if signal calibration is enabled"""
        return self.get('unified_trading_core.enable_calibrated_signals', True)
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Add metadata
            config_with_meta = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "description": "Enhanced prediction pipeline configuration"
                },
                **self._config
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Enhanced prediction config saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced prediction config: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        issues = []
        
        # Validate labeling config
        labeling = self.get_labeling_config()
        if labeling.get('tp_sigma', 0) <= 0:
            issues.append("labeling.tp_sigma must be positive")
        if labeling.get('sl_sigma', 0) <= 0:
            issues.append("labeling.sl_sigma must be positive")
        if labeling.get('max_holding_days', 0) <= 0:
            issues.append("labeling.max_holding_days must be positive")
        
        # Validate factors config
        factors = self.get_factors_config()
        limits = factors.get('winsorize_limits', [])
        if not isinstance(limits, list) or len(limits) != 2 or limits[0] >= limits[1]:
            issues.append("factors.winsorize_limits must be [lower, upper] with lower < upper")
        
        # Validate training config
        training = self.get_training_config()
        split = training.get('train_test_split', 0)
        if not 0 < split < 1:
            issues.append("training.train_test_split must be between 0 and 1")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self._config.copy()

# Global instance
_enhanced_config = None

def get_enhanced_prediction_config(config_path: Optional[str] = None) -> EnhancedPredictionConfig:
    """Get global enhanced prediction configuration instance"""
    global _enhanced_config
    if _enhanced_config is None:
        _enhanced_config = EnhancedPredictionConfig(config_path)
    return _enhanced_config

def create_enhanced_prediction_config(config_path: Optional[str] = None) -> EnhancedPredictionConfig:
    """Create new enhanced prediction configuration instance"""
    return EnhancedPredictionConfig(config_path)

# Configuration validation utility
def validate_enhanced_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate enhanced prediction configuration"""
    config = get_enhanced_prediction_config(config_path)
    return config.validate_config()

# Configuration export utility
def export_config_template(output_path: str):
    """Export default configuration template"""
    try:
        template = {
            "metadata": {
                "template_version": "1.0.0",
                "description": "Enhanced prediction configuration template",
                "created_at": datetime.now().isoformat()
            },
            **DEFAULT_ENHANCED_CONFIG
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration template exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export configuration template: {e}")
        raise

# Example usage and integration guide
def example_integration():
    """Example of how to integrate enhanced prediction config"""
    
    # Get configuration
    config = get_enhanced_prediction_config()
    
    # Check if enhanced features are enabled
    if config.is_enhanced_prediction_enabled():
        print("Enhanced prediction is enabled")
        
        # Get specific configuration sections
        labeling_config = config.get_labeling_config()
        factors_config = config.get_factors_config()
        
        # Use in pipeline initialization
        # pipeline = create_enhanced_labeling_pipeline(labeling_config)
        # factor_pipeline = create_enhanced_factor_pipeline(factors_config)
        
    # Validate configuration
    validation = config.validate_config()
    if not validation['valid']:
        print(f"Configuration issues: {validation['issues']}")
    
    # Update configuration programmatically
    config.set('labeling.tp_sigma', 2.5)
    config.save_config()
    
    return config

if __name__ == "__main__":
    # Run example
    example_integration()