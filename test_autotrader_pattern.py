#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

def test_autotrader_calling_pattern():
    """Test the exact calling pattern used by autotrader"""
    
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        print("Import successful")
        
        model = UltraEnhancedQuantitativeModel()
        print("Model instantiation successful")
        
        # This is the exact pattern from autotrader error log:
        # model.run_complete_analysis(tickers=tickers, start_date=start_date, end_date=end_date, top_n=10)
        
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        test_start_date = '2022-09-08'
        test_end_date = '2025-09-07'
        test_top_n = 10
        
        print("Testing autotrader calling pattern...")
        print(f"Calling: run_complete_analysis(tickers={test_tickers}, start_date='{test_start_date}', end_date='{test_end_date}', top_n={test_top_n})")
        
        # Test the method call (will likely fail due to data issues, but should not fail with parameter error)
        try:
            result = model.run_complete_analysis(
                tickers=test_tickers,
                start_date=test_start_date, 
                end_date=test_end_date,
                top_n=test_top_n
            )
            print("SUCCESS: Method call completed without parameter errors")
            print("Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
            return True
            
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                print("ERROR: Still has parameter issues:", str(e))
                return False
            elif "got multiple values for argument" in str(e):
                print("ERROR: Multiple values parameter issue:", str(e))
                return False
            else:
                print("SUCCESS: No parameter errors, other error:", str(e))
                return True
                
        except Exception as e:
            print("SUCCESS: No parameter errors, execution error (expected):", str(e))
            return True
            
    except Exception as e:
        print("ERROR:", str(e))
        return False

if __name__ == "__main__":
    success = test_autotrader_calling_pattern()
    print("Test result:", "PASS" if success else "FAIL")
    sys.exit(0 if success else 1)