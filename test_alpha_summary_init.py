#!/usr/bin/env python3
"""Test Alpha Summary Processor initialization"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bma_models'))

def test_alpha_summary_init():
    try:
        print("=== Testing Alpha Summary Processor ===")
        
        # Test import
        print("1. Testing import...")
        from bma_models.alpha_summary_features import create_alpha_summary_processor, AlphaSummaryConfig
        print("SUCCESS: Import successful")
        
        # Test config creation
        print("2. Testing config creation...")
        alpha_config = AlphaSummaryConfig(
            max_alpha_features=18,
            include_alpha_strategy_signal=True,
            pca_variance_explained=0.85,
            pls_n_components=8
        )
        print(f"SUCCESS: Config created - max_features={alpha_config.max_alpha_features}")
        
        # Test processor creation
        print("3. Testing processor creation...")
        processor = create_alpha_summary_processor(alpha_config.__dict__)
        print("SUCCESS: Processor created")
        
        # Test processor attributes
        print("4. Testing processor attributes...")
        print(f"Processor config: {processor.config}")
        print(f"Has process_alpha_to_summary: {hasattr(processor, 'process_alpha_to_summary')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpha_summary_init()
    if success:
        print("\nSUCCESS: Alpha Summary Processor test passed")
    else:
        print("\nFAILED: Alpha Summary Processor test failed")