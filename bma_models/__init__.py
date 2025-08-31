"""
BMA Models Package
==================

This package contains all BMA (Bayesian Model Averaging) related models and algorithms.

Modules:
- bma_enhanced_integrated_system: Main BMA enhanced system with integrated ML components
- enhanced_alpha_strategies: Alpha generation strategies
- unified_feature_pipeline: Unified feature engineering pipeline
- model_monitoring: Model performance monitoring
- model_version_control: Model versioning and management
"""

# Version information
__version__ = "1.0.0"
__author__ = "Trading System"

# Import main components
try:
    from .bma_enhanced_integrated_system import *
except ImportError:
    pass

try:
    from .unified_feature_pipeline import *
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