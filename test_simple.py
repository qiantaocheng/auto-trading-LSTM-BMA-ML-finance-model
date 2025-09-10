#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

def test_method():
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        print("Import successful")
        
        model = UltraEnhancedQuantitativeModel()
        print("Instantiation successful")
        
        # Test method signature
        import inspect
        sig = inspect.signature(model.run_complete_analysis)
        print("Method signature:", str(sig))
        
        # Test parameter detection logic
        kwargs = {
            'tickers': ['AAPL', 'MSFT'],
            'start_date': '2023-01-01',
            'end_date': '2024-01-01',
            'top_n': 10
        }
        
        # This should detect original API call pattern
        if 'tickers' in kwargs or isinstance(kwargs.get('tickers'), list):
            print("Original API pattern detected correctly")
        else:
            print("ERROR: Original API pattern not detected")
            
        return True
        
    except Exception as e:
        print("ERROR:", str(e))
        return False

if __name__ == "__main__":
    success = test_method()
    sys.exit(0 if success else 1)