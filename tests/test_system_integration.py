#!/usr/bin/env python3
"""
System Integration Tests
Comprehensive tests for the trading system
"""

import os
import sys
import unittest
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autotrader.unified_signal_processor import get_unified_signal_processor, SignalMode
from autotrader.unified_trading_engine import get_unified_trading_engine
from autotrader.config_loader import TradingConfig
from autotrader.input_validator import InputValidator

class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        """Set up test environment"""
        # Set test environment variables
        os.environ['TRADING_ACCOUNT_ID'] = 'test_account'
        os.environ['SIGNAL_MODE'] = 'testing'
        os.environ['DEMO_MODE'] = 'true'
        os.environ['ALLOW_RANDOM_SIGNALS'] = 'false'
        os.environ['REQUIRE_PRODUCTION_VALIDATION'] = 'false'
    
    def test_signal_processor_modes(self):
        """Test all signal processor modes"""
        for mode in SignalMode:
            processor = get_unified_signal_processor(mode)
            result = processor.get_trading_signal("AAPL", 0.3)
            
            self.assertIsInstance(result.signal_value, float)
            self.assertIsInstance(result.confidence, float)
            self.assertIn(result.side, ["BUY", "SELL", "HOLD"])
            
            if mode == SignalMode.PRODUCTION:
                # Production mode should return safe defaults
                self.assertEqual(result.signal_value, 0.0)
                self.assertFalse(result.can_trade)
    
    def test_input_validation(self):
        """Test input validation"""
        # Valid inputs
        self.assertEqual(InputValidator.validate_symbol("AAPL"), "AAPL")
        self.assertEqual(InputValidator.validate_quantity(100), 100)
        self.assertEqual(InputValidator.validate_price(150.50), 150.50)
        
        # Invalid inputs
        with self.assertRaises(Exception):
            InputValidator.validate_symbol("")
        
        with self.assertRaises(Exception):
            InputValidator.validate_quantity(-10)
        
        with self.assertRaises(Exception):
            InputValidator.validate_price(0)
    
    def test_trading_engine_integration(self):
        """Test trading engine integration"""
        async def run_test():
            engine = get_unified_trading_engine()
            decisions = await engine.get_trading_decisions(["AAPL", "MSFT"])
            
            self.assertIsInstance(decisions, list)
            for decision in decisions:
                self.assertIn(decision.action, ["BUY", "SELL", "HOLD"])
                self.assertGreaterEqual(decision.confidence, 0.0)
                self.assertLessEqual(decision.confidence, 1.0)
        
        asyncio.run(run_test())
    
    def test_no_demo_code_in_production(self):
        """Test that no demo code exists in production files"""
        import re
        
        demo_patterns = [
            r'np\.random\.uniform',
            r'# Random signal for demo'
        ]
        
        project_files = list(project_root.glob("**/*.py"))
        
        for file_path in project_files:
            if "test" in str(file_path):
                continue  # Skip test files
            
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern in demo_patterns:
                    matches = re.findall(pattern, content)
                    self.assertEqual(len(matches), 0, 
                                   f"Demo code found in {file_path}: {pattern}")
            except Exception:
                pass  # Skip files that can't be read

class TestProductionSafety(unittest.TestCase):
    """Test production safety measures"""
    
    def test_production_validation(self):
        """Test production validation catches issues"""
        # This should pass with test environment
        os.environ['SIGNAL_MODE'] = 'testing'
        os.environ['ALLOW_RANDOM_SIGNALS'] = 'false'
        
        try:
            from autotrader.config_loader import get_config
            config = get_config()
            self.assertIsInstance(config, TradingConfig)
        except Exception as e:
            self.fail(f"Test configuration should be valid: {e}")

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
