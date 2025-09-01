#!/usr/bin/env python3
"""Test Alpha Summary Processor in main model"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_alpha_summary_in_model():
    try:
        print("=== Testing Alpha Summary in Main Model ===")
        
        # Import main model
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        print("Main model imported successfully")
        
        # Create model instance
        print("Creating model instance...")
        model = UltraEnhancedQuantitativeModel()
        print("Model created successfully")
        
        # Check Alpha summary processor
        has_alpha_processor = hasattr(model, 'alpha_summary_processor')
        alpha_processor_not_none = has_alpha_processor and model.alpha_summary_processor is not None
        
        print(f"Has alpha_summary_processor attribute: {has_alpha_processor}")
        print(f"Alpha summary processor not None: {alpha_processor_not_none}")
        
        if alpha_processor_not_none:
            print("SUCCESS: Alpha summary processor initialized correctly")
            print(f"Config: {model.alpha_summary_processor.config.max_alpha_features} max features")
            return True
        else:
            print("FAILED: Alpha summary processor not initialized")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpha_summary_in_model()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")