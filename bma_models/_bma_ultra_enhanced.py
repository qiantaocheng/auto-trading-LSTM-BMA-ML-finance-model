"""ASCII shim for legacy module name."""
import importlib

_module = importlib.import_module("bma_models.\u91cf\u5316\u6a21\u578b_bma_ultra_enhanced")
UltraEnhancedQuantitativeModel = getattr(_module, "UltraEnhancedQuantitativeModel")
__all__ = ["UltraEnhancedQuantitativeModel"]
