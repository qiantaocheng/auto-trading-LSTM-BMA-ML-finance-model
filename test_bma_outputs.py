#!/usr/bin/env python3
"""
Simple test script to run BMA with 10 stocks and generate complete Excel/JSON outputs
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import the BMA model
sys.path.append('.')
from 量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

def main():
    """Run BMA with limited stocks and generate outputs"""
    print("=== BMA Output Generation Test ===")
    
    # Initialize model
    model = UltraEnhancedQuantitativeModel()
    
    # Use a small set of popular stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
    
    print(f"Testing with {len(test_tickers)} stocks: {test_tickers}")
    
    try:
        # Run complete analysis
        results = model.run_complete_analysis(
            tickers=test_tickers,
            start_date='2024-06-01',  # Shorter timeframe for faster processing
            end_date='2025-08-16',
            top_n=10
        )
        
        if results.get('success', False):
            print("\n[SUCCESS] BMA Analysis Completed Successfully!")
            print(f"Result file: {results.get('result_file', 'Not generated')}")
            print(f"Total processing time: {results.get('total_time', 0):.1f} seconds")
            
            # Check if files were generated
            result_dir = Path("result")
            timestamp = datetime.now().strftime("%Y%m%d")
            
            recent_files = []
            if result_dir.exists():
                for file in result_dir.glob(f"*{timestamp}*"):
                    recent_files.append(file.name)
            
            if recent_files:
                print(f"\n[FILES] Generated files ({len(recent_files)}):")
                for file in sorted(recent_files):
                    print(f"  - {file}")
            else:
                print("\n[WARNING] No files found for today")
                
        else:
            print(f"\n[ERROR] Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n[EXCEPTION] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()