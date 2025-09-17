"""
BMA Models Package
==================

This package contains all BMA (Bayesian Model Averaging) related models and algorithms.

Core modules:
- 量化模型_bma_ultra_enhanced: Main BMA ultra enhanced system
- enhanced_alpha_strategies: Alpha generation strategies
- unified_feature_pipeline: Unified feature engineering pipeline
- unified_market_data_manager: Market data management
- polygon_client: Polygon API client
"""

# Version information
__version__ = "1.0.0"
__author__ = "Trading System"

# Import only verified core components
from .unified_feature_pipeline import UnifiedFeaturePipeline, FeaturePipelineConfig
from .enhanced_alpha_strategies import AlphaStrategiesEngine
from .unified_market_data_manager import UnifiedMarketDataManager
from .polygon_client import PolygonClient