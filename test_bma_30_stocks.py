#!/usr/bin/env python3
"""
Test BMA with 30 stocks to generate proper Excel/JSON outputs
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
    """Run BMA with 30 stocks and generate complete outputs"""
    print("=== BMA 30-Stock Test ===")
    
    # Initialize model
    model = UltraEnhancedQuantitativeModel()
    
    # Use the first 30 popular stocks
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM',
        'ORCL', 'ADBE', 'PYPL', 'INTC', 'IBM', 'UBER', 'SPOT', 'ZM', 'TWTR', 'SNAP',
        'SQ', 'ROKU', 'PINS', 'DOCU', 'ZOOM', 'SHOP', 'PLTR', 'RBLX', 'COIN', 'HOOD'
    ]
    
    print(f"Testing with {len(test_tickers)} stocks")
    print(f"Stocks: {test_tickers[:10]}...")
    
    try:
        # Run complete analysis with shorter timeframe for faster processing
        results = model.run_complete_analysis(
            tickers=test_tickers,
            start_date='2024-01-01',
            end_date='2025-08-16',
            top_n=30  # Get all 30 recommendations
        )
        
        if results.get('success', False):
            print(f"\n[SUCCESS] BMA Analysis Completed!")
            print(f"Processing time: {results.get('total_time', 0):.1f} seconds")
            
            # Check recommendations
            recommendations = results.get('recommendations', [])
            print(f"Recommendations generated: {len(recommendations)}")
            
            if recommendations:
                print("\nTop 5 Recommendations:")
                for i, rec in enumerate(recommendations[:5]):
                    ticker = rec.get('ticker', 'N/A')
                    rating = rec.get('rating', 'N/A')
                    ret = rec.get('expected_return', 0)
                    conf = rec.get('confidence_score', 0)
                    print(f"  {i+1}. {ticker} - {rating} (Return: {ret:.1%}, Confidence: {conf:.1%})")
            
            # Check result files generated today
            result_dir = Path("result")
            timestamp = datetime.now().strftime("%Y%m%d")
            
            recent_files = []
            if result_dir.exists():
                for file in result_dir.glob(f"*{timestamp}*"):
                    recent_files.append(file.name)
            
            if recent_files:
                print(f"\n[FILES] Generated files ({len(recent_files)}):")
                for file in sorted(recent_files)[-5:]:  # Show last 5 files
                    print(f"  - {file}")
            
            # Check specific file contents
            latest_excel = None
            latest_json = None
            for file in recent_files:
                if 'ultra_enhanced_recommendations' in file and file.endswith('.xlsx'):
                    latest_excel = result_dir / file
                elif 'top10_tickers' in file and file.endswith('.json'):
                    latest_json = result_dir / file
            
            if latest_excel and latest_excel.exists():
                try:
                    df = pd.read_excel(latest_excel)
                    print(f"\n[EXCEL] {latest_excel.name}: {len(df)} rows, {len(df.columns)} columns")
                    if len(df) > 0:
                        print(f"Columns: {list(df.columns)}")
                        print(f"Sample: {df.iloc[0].to_dict()}")
                except Exception as e:
                    print(f"[EXCEL ERROR] {e}")
                    
        else:
            print(f"\n[ERROR] Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()