#!/usr/bin/env python3
"""Trace model initialization to find infinite loop source"""

import logging
import sys

# Custom logger to capture initialization calls
class InitTracker:
    def __init__(self):
        self.call_count = {}
        
    def track_call(self, function_name):
        if function_name not in self.call_count:
            self.call_count[function_name] = 0
        self.call_count[function_name] += 1
        print(f"TRACK: {function_name} called {self.call_count[function_name]} times")
        
        if self.call_count[function_name] > 5:  # Stop after 5 calls to prevent infinite output
            print(f"STOPPING: {function_name} called too many times!")
            sys.exit(1)

# Global tracker
tracker = InitTracker()

def test_model_init_trace():
    """Test model initialization with call tracking"""
    try:
        print("=== Tracing Model Initialization ===")
        
        # Monkey patch the initialization method to track calls
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        # Get original methods
        original_init = UltraEnhancedQuantitativeModel.__init__
        original_alpha_init = UltraEnhancedQuantitativeModel._init_alpha_summary_processor
        
        def traced_init(self, *args, **kwargs):
            tracker.track_call("__init__")
            try:
                result = original_init(self, *args, **kwargs)
                print("TRACE: __init__ completed successfully")
                return result
            except Exception as e:
                print(f"TRACE: __init__ failed with exception: {e}")
                import traceback
                traceback.print_exc()
                raise
            
        def traced_alpha_init(self):
            tracker.track_call("_init_alpha_summary_processor")
            return original_alpha_init(self)
        
        # Apply patches
        UltraEnhancedQuantitativeModel.__init__ = traced_init
        UltraEnhancedQuantitativeModel._init_alpha_summary_processor = traced_alpha_init
        
        print("Creating model with tracing...")
        model = UltraEnhancedQuantitativeModel()
        
        print(f"Model created successfully!")
        print(f"Call counts: {tracker.call_count}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Final call counts: {tracker.call_count}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_init_trace()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")