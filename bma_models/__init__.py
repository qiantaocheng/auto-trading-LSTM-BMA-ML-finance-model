"""
BMA Models Package
==================

This package contains all BMA (Bayesian Model Averaging) related models and algorithms.

Modules:
- bma_walkforward_enhanced: Enhanced walkforward BMA implementation
- learning_to_rank_bma: Learning to rank with BMA
- robust_bma_weighting: Robust BMA weighting strategies
- enhanced_alpha_strategies: Alpha generation strategies
- model_monitoring: Model performance monitoring
- model_version_control: Model versioning and management
"""

# Version information
__version__ = "1.0.0"
__author__ = "Trading System"

# Import main components
try:
    from .bma_walkforward_enhanced import *
except ImportError:
    pass

try:
    from .learning_to_rank_bma import *
except ImportError:
    pass

try:
    from .robust_bma_weighting import *
except ImportError:
    pass

try:
    from .enhanced_alpha_strategies import *
except ImportError:
    pass

# Utility imports
try:
    from .model_monitoring import ModelMonitor
except ImportError:
    pass

try:
    from .model_version_control import ModelVersionControl
except ImportError:
    pass