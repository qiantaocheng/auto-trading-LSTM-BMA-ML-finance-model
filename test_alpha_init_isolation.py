#!/usr/bin/env python3
"""Isolate Alpha Summary Processor initialization to find infinite loop source"""

import logging
logging.basicConfig(level=logging.INFO)

def test_alpha_init_isolation():
    """Test Alpha initialization components in isolation"""
    try:
        print("=== Testing Alpha initialization in isolation ===")
        
        # Step 1: Test config creation
        print("1. Testing AlphaSummaryConfig...")
        from bma_models.alpha_summary_features import AlphaSummaryConfig
        alpha_config = AlphaSummaryConfig(
            max_alpha_features=18,
            include_alpha_strategy_signal=True,
            pca_variance_explained=0.85,
            pls_n_components=8
        )
        print(f"Config created: {alpha_config.max_alpha_features} features")
        
        # Step 2: Test processor creation
        print("2. Testing AlphaSummaryProcessor...")
        from bma_models.alpha_summary_features import AlphaSummaryProcessor
        processor = AlphaSummaryProcessor(alpha_config)
        print(f"Processor created: {processor}")
        
        # Step 3: Test factory function
        print("3. Testing create_alpha_summary_processor...")
        from bma_models.alpha_summary_features import create_alpha_summary_processor
        processor2 = create_alpha_summary_processor(alpha_config.__dict__)
        print(f"Factory processor created: {processor2}")
        
        # Step 4: Test assignment
        print("4. Testing assignment...")
        test_model = type('TestModel', (), {})()
        test_model.alpha_summary_processor = processor2
        print(f"Assignment successful: {test_model.alpha_summary_processor}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpha_init_isolation()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")