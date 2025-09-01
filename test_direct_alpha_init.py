#!/usr/bin/env python3
"""Direct test of Alpha summary processor initialization"""

import logging
logging.basicConfig(level=logging.INFO)

def test_direct_alpha_init():
    try:
        print("=== Direct Alpha Init Test ===")
        
        # Import and create minimal instance
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        # Create instance 
        model = UltraEnhancedQuantitativeModel()
        
        print(f"Model created. Has alpha_summary_processor: {hasattr(model, 'alpha_summary_processor')}")
        
        # Try direct call to init method
        print("Calling _init_alpha_summary_processor directly...")
        model._init_alpha_summary_processor()
        
        # Check result
        has_processor = hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor is not None
        print(f"After direct call - Has processor: {has_processor}")
        
        if has_processor:
            print(f"Processor config: {model.alpha_summary_processor.config.max_alpha_features}")
            return True
        else:
            print("Direct call failed")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_alpha_init()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")