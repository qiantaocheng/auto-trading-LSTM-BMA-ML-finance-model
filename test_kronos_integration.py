#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kronos Integration Test Script
测试Kronos K线预测模型集成
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kronos_service():
    """Test Kronos service without UI"""
    print("=" * 60)
    print("Testing Kronos Service (Non-UI)")
    print("=" * 60)

    from kronos import KronosService

    # Initialize service
    service = KronosService()

    # Test prediction for AAPL
    print("\nGenerating predictions for AAPL...")
    result = service.predict_stock(
        symbol="AAPL",
        period="1mo",
        interval="1d",
        pred_len=10,
        model_size="base",  # default to base; may fallback to statistical predictor
        temperature=0.7
    )

    if result["status"] == "success":
        print(f"✅ Successfully generated predictions for {result['symbol']}")
        predictions = result["predictions"]
        try:
            # DataFrame path
            print(f"   - Prediction count: {len(predictions)}")
            print(f"   - First predicted close: ${predictions['close'].iloc[0]:.2f}")
            print(f"   - Last predicted close: ${predictions['close'].iloc[-1]:.2f}")
        except Exception:
            # Numpy array path (fallback predictor)
            import numpy as _np
            if isinstance(predictions, _np.ndarray) and predictions.ndim == 2 and predictions.shape[1] >= 4:
                print(f"   - Prediction count: {len(predictions)}")
                print(f"   - First predicted close: ${predictions[0, 3]:.2f}")
                print(f"   - Last predicted close: ${predictions[-1, 3]:.2f}")
            else:
                print("   - Predictions format unexpected")

        # Calculate statistics
        stats = service.get_statistics(predictions)
        if stats:
            print("\n📊 Prediction Statistics:")
            print(f"   - Mean close: ${stats['price_stats']['mean_close']:.2f}")
            print(f"   - Price range: ${stats['price_stats']['price_range']:.2f}")
            print(f"   - Total change: {stats['change_stats']['total_change']:.2f}%")
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

def test_kronos_ui():
    """Test Kronos UI in tkinter app"""
    print("\n" + "=" * 60)
    print("Testing Kronos UI Integration")
    print("=" * 60)

    try:
        from autotrader.app import AutoTraderGUI
        import tkinter as tk

        print("\nLaunching AutoTrader GUI with Kronos tab...")
        print("Please check the 'Kronos预测' tab in the application")
        print("\nInstructions:")
        print("1. Click on the 'Kronos预测' tab")
        print("2. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)")
        print("3. Select model size and prediction length")
        print("4. Click '🚀 生成预测' button")
        print("5. View results in the tabs below")

        # Create and run the app
        app = AutoTraderGUI()

        # Check if Kronos tab loaded
        if hasattr(app, 'kronos_predictor'):
            print("\n✅ Kronos tab loaded successfully!")
        else:
            print("\n⚠️ Kronos tab not found, but may still be functional")

        # Start the GUI
        app.mainloop()

    except Exception as e:
        print(f"\n❌ Error loading UI: {str(e)}")
        print("This might be due to missing tkinter or other dependencies")

def main():
    """Main test function"""
    print("\n🔮 KRONOS K-LINE PREDICTION MODEL INTEGRATION TEST")
    print("=" * 60)

    # Test options
    print("\nSelect test option:")
    print("1. Test Kronos Service (Non-UI)")
    print("2. Test Kronos UI Integration")
    print("3. Run both tests")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        test_kronos_service()
    elif choice == "2":
        test_kronos_ui()
    elif choice == "3":
        test_kronos_service()
        input("\nPress Enter to continue to UI test...")
        test_kronos_ui()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()