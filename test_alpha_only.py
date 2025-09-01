#!/usr/bin/env python3
"""Test Alpha engine initialization only"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bma_models'))

def test_alpha_engine_only():
    try:
        print("=== Testing Alpha Engine Only ===")
        
        # Test direct import
        print("1. Testing direct import...")
        from bma_models.enhanced_alpha_strategies import AlphaStrategiesEngine
        print("SUCCESS: AlphaStrategiesEngine imported successfully")
        
        # Test initialization with correct config
        print("2. Testing initialization...")
        config_path = "alphas_config.yaml"
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found: {config_path}")
            return False
            
        engine = AlphaStrategiesEngine(config_path)
        print(f"SUCCESS: AlphaStrategiesEngine initialized with {len(engine.alpha_functions)} functions")
        
        # Test required methods
        print("3. Testing required methods...")
        has_compute_all_alphas = hasattr(engine, 'compute_all_alphas')
        has_alpha_functions = hasattr(engine, 'alpha_functions')
        
        print(f"compute_all_alphas: {has_compute_all_alphas}")
        print(f"alpha_functions: {has_alpha_functions}")
        
        if has_compute_all_alphas and has_alpha_functions:
            print("SUCCESS: Alpha Engine: ALL METHODS AVAILABLE")
            return True
        else:
            print("ERROR: Alpha Engine: Missing methods")
            return False
            
    except Exception as e:
        print(f"ERROR: Alpha Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpha_engine_only()
    if success:
        print("\nSUCCESS: ALPHA ENGINE TEST PASSED")
    else:
        print("\nFAILED: ALPHA ENGINE TEST FAILED")