#!/usr/bin/env python3
"""
Test script to verify Alpha features with dimensionality reduction are properly integrated
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpha_integration():
    """Test the full Alpha integration pipeline"""
    try:
        # Import the main model
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        logger.info("=== Alpha Integration Test ===")
        
        # 1. Create model instance
        model = UltraEnhancedQuantitativeModel()
        logger.info("✅ Model created successfully")
        
        # 2. Check if Alpha engine is available
        has_alpha_engine = hasattr(model, 'alpha_engine') and model.alpha_engine is not None
        logger.info(f"Alpha engine available: {has_alpha_engine}")
        if has_alpha_engine:
            logger.info(f"Alpha functions: {len(model.alpha_engine.alpha_functions)}")
        
        # 3. Check if Alpha summary processor is available  
        has_alpha_processor = hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor is not None
        logger.info(f"Alpha summary processor available: {has_alpha_processor}")
        
        # 4. Create test data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        tickers = [f'STOCK_{i:03d}' for i in range(10)]
        
        test_data = []
        for date in dates[::7]:  # Weekly data
            for ticker in tickers:
                test_data.append({
                    'date': date,
                    'ticker': ticker,
                    'open': 100 + np.random.randn() * 5,
                    'high': 105 + np.random.randn() * 5,
                    'low': 95 + np.random.randn() * 5,
                    'close': 100 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                    'target': np.random.randn() * 0.02,  # 2% volatility target
                    'COUNTRY': 'US'  # Add country data to avoid error
                })
        
        feature_data = pd.DataFrame(test_data)
        logger.info(f"✅ Test data created: {feature_data.shape}")
        
        # 5. Test feature creation with Alpha integration
        stock_data = {}
        for ticker in tickers:
            ticker_data = feature_data[feature_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            stock_data[ticker] = ticker_data[['date', 'open', 'high', 'low', 'close', 'volume', 'COUNTRY']]
        
        # 6. Test traditional feature creation
        logger.info("Testing traditional feature creation...")
        traditional_features = model.create_traditional_features(stock_data)
        if traditional_features is not None and not traditional_features.empty:
            logger.info(f"✅ Traditional features: {traditional_features.shape}")
            traditional_cols = len([col for col in traditional_features.columns 
                                  if col not in ['ticker', 'date', 'target']])
            logger.info(f"Traditional feature columns: {traditional_cols}")
        else:
            logger.error("❌ Traditional feature creation failed")
            return False
        
        # 7. Test Alpha integration
        logger.info("Testing Alpha feature integration...")
        try:
            alpha_result = model._integrate_alpha_summary_features(traditional_features, stock_data)
            if alpha_result is not None and not alpha_result.empty:
                alpha_cols = len([col for col in alpha_result.columns 
                                if col not in ['ticker', 'date', 'target']])
                added_alpha_features = alpha_cols - traditional_cols
                logger.info(f"✅ Alpha integration successful: {alpha_result.shape}")
                logger.info(f"Total feature columns: {alpha_cols}")
                logger.info(f"Added Alpha features: {added_alpha_features}")
                
                # Check for specific Alpha feature names
                alpha_feature_names = [col for col in alpha_result.columns 
                                     if any(x in col.lower() for x in ['alpha_pc', 'alpha_composite', 'alpha_summary'])]
                logger.info(f"Alpha feature names: {alpha_feature_names}")
                
                if added_alpha_features > 0:
                    logger.info("🎉 ALPHA INTEGRATION WITH DIMENSIONALITY REDUCTION: SUCCESS")
                    return True
                else:
                    logger.warning("⚠️ Alpha integration did not add new features")
                    return False
            else:
                logger.warning("⚠️ Alpha integration returned None or empty")
                return False
        except Exception as e:
            logger.error(f"❌ Alpha integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpha_integration()
    if success:
        print("\n🎉 ALPHA INTEGRATION TEST: PASSED")
        print("✅ Alpha features with dimensionality reduction are properly integrated into ML pipeline")
    else:
        print("\n❌ ALPHA INTEGRATION TEST: FAILED") 
        print("⚠️ Alpha features may not be properly integrated")