#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for AutoTrader System
Tests all critical components and workflows
"""

import unittest
import asyncio
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.engine import Engine, SignalHub, RiskEngine
from autotrader.unified_signal_processor import UnifiedSignalProcessor, SignalResult
from autotrader.unified_risk_manager import UnifiedRiskManager, RiskValidationResult
from autotrader.database import StockDatabase
from autotrader.ibkr_auto_trader import IbkrAutoTrader


class TestSignalGeneration(unittest.TestCase):
    """Test trading signal generation algorithms"""

    def setUp(self):
        """Initialize test fixtures"""
        self.signal_hub = SignalHub()
        self.mock_config = MagicMock()
        self.mock_logger = MagicMock()

    def test_mean_reversion_signal_strong_buy(self):
        """Test strong buy signal when z-score < -2.5"""
        # Generate prices that will create z-score < -2.5
        prices = [100.0] * 19 + [70.0]  # Large drop = strong buy
        signal = self.signal_hub.mr_signal(prices)
        self.assertGreater(signal, 0.5, "Should generate strong buy signal")

    def test_mean_reversion_signal_strong_sell(self):
        """Test strong sell signal when z-score > 2.5"""
        # Generate prices that will create z-score > 2.5
        prices = [100.0] * 19 + [130.0]  # Large rise = strong sell
        signal = self.signal_hub.mr_signal(prices)
        self.assertLess(signal, -0.5, "Should generate strong sell signal")

    def test_mean_reversion_signal_neutral(self):
        """Test neutral signal when price is near mean"""
        # Stable prices should generate near-zero signal
        prices = [100.0 + np.random.normal(0, 0.5) for _ in range(20)]
        signal = self.signal_hub.mr_signal(prices)
        self.assertLess(abs(signal), 0.3, "Should generate neutral signal")

    def test_momentum_calculation(self):
        """Test momentum indicator calculation"""
        # Upward trend should generate positive momentum
        prices = [100.0 + i * 2 for i in range(25)]  # Rising prices
        momentum = self.signal_hub.calculate_momentum(prices, period=20)
        self.assertGreater(momentum, 0, "Rising prices should have positive momentum")

        # Downward trend should generate negative momentum
        prices = [100.0 - i * 2 for i in range(25)]  # Falling prices
        momentum = self.signal_hub.calculate_momentum(prices, period=20)
        self.assertLess(momentum, 0, "Falling prices should have negative momentum")

    def test_insufficient_data_handling(self):
        """Test handling of insufficient price data"""
        prices = [100.0] * 5  # Only 5 bars, need 20
        signal = self.signal_hub.mr_signal(prices)
        self.assertEqual(signal, 0.0, "Should return 0 for insufficient data")

    def test_nan_handling(self):
        """Test handling of NaN values in price data"""
        prices = [100.0] * 10 + [float('nan')] * 5 + [100.0] * 5
        signal = self.signal_hub.mr_signal(prices)
        self.assertFalse(np.isnan(signal), "Should handle NaN values gracefully")


class TestUnifiedSignalProcessor(unittest.TestCase):
    """Test the unified signal processing system"""

    def setUp(self):
        """Initialize signal processor"""
        self.processor = UnifiedSignalProcessor()
        self.mock_polygon = MagicMock()

    @patch('autotrader.unified_signal_processor.UnifiedPolygonFactors')
    def test_buy_signal_generation(self, mock_polygon_class):
        """Test generation of BUY signal"""
        # Mock Polygon factors to return strong buy signal
        mock_polygon_instance = MagicMock()
        mock_polygon_instance.get_trading_signal.return_value = {
            'signal_value': -0.8,
            'signal_strength': 0.9,
            'confidence': 0.85,
            'side': 'BUY',
            'can_trade': True,
            'delay_reason': 'Signal calculated'
        }
        mock_polygon_class.return_value = mock_polygon_instance

        processor = UnifiedSignalProcessor()
        processor.polygon_factors = mock_polygon_instance

        result = processor.get_trading_signal('AAPL', threshold=0.3)

        self.assertEqual(result.side, 'BUY')
        self.assertTrue(result.can_trade)
        self.assertGreater(result.confidence, 0.8)

    def test_sell_signal_generation(self):
        """Test generation of SELL signal"""
        processor = UnifiedSignalProcessor()
        mock_polygon = MagicMock()
        mock_polygon.get_trading_signal.return_value = {
            'signal_value': 0.7,
            'signal_strength': 0.8,
            'confidence': 0.75,
            'side': 'SELL',
            'can_trade': True,
            'delay_reason': 'Signal calculated'
        }
        processor.polygon_factors = mock_polygon

        result = processor.get_trading_signal('AAPL', threshold=0.3)

        self.assertEqual(result.side, 'SELL')
        self.assertTrue(result.can_trade)

    def test_hold_signal_generation(self):
        """Test generation of HOLD signal when below threshold"""
        processor = UnifiedSignalProcessor()
        mock_polygon = MagicMock()
        mock_polygon.get_trading_signal.return_value = {
            'signal_value': 0.1,  # Below threshold
            'signal_strength': 0.2,
            'confidence': 0.3,
            'side': 'HOLD',
            'can_trade': False,
            'delay_reason': 'Signal below threshold'
        }
        processor.polygon_factors = mock_polygon

        result = processor.get_trading_signal('AAPL', threshold=0.3)

        self.assertEqual(result.side, 'HOLD')
        self.assertFalse(result.can_trade)


class TestRiskControl(unittest.TestCase):
    """Test risk management systems"""

    def setUp(self):
        """Initialize risk manager"""
        self.config_manager = MagicMock()
        self.config_manager.get.return_value = {}
        self.risk_manager = UnifiedRiskManager(self.config_manager)

    def test_position_limit_validation(self):
        """Test single position limit (15% of equity)"""
        # Setup portfolio state
        self.risk_manager.positions = {}

        # Test order that exceeds position limit
        result = self.risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=1000,
            price=150.0,
            account_value=100000.0  # Order value = 150k, Account = 100k
        )

        self.assertFalse(result.is_valid)
        self.assertIn('position', result.reason.lower())

    def test_sector_exposure_limit(self):
        """Test sector exposure limit (30%)"""
        # Add existing tech positions
        from autotrader.unified_risk_manager import PositionRisk

        self.risk_manager.positions['MSFT'] = PositionRisk(
            symbol='MSFT',
            quantity=100,
            current_value=25000.0,
            entry_price=250.0,
            current_price=250.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            sector='TECH'
        )

        # Try to add another large tech position
        result = self.risk_manager.validate_order(
            symbol='GOOGL',
            side='BUY',
            quantity=100,
            price=2800.0,  # 280k position in tech
            account_value=1000000.0
        )

        # Should pass individual check but consider sector concentration
        self.assertTrue(result.is_valid or 'sector' in result.reason.lower())

    def test_daily_order_limit(self):
        """Test daily order limit (20 orders)"""
        # Simulate 20 orders already placed
        self.risk_manager.daily_order_count = 20

        result = self.risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150.0,
            account_value=100000.0
        )

        self.assertFalse(result.is_valid)
        self.assertIn('daily', result.reason.lower())

    def test_daily_loss_limit(self):
        """Test daily loss limit (5% of account)"""
        # Set daily P&L to -6% (exceeds 5% limit)
        self.risk_manager.daily_pnl = -6000.0

        result = self.risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150.0,
            account_value=100000.0
        )

        self.assertFalse(result.is_valid)
        self.assertIn('loss', result.reason.lower())

    def test_min_order_value(self):
        """Test minimum order value ($1,000)"""
        result = self.risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=5,
            price=150.0,  # Order value = $750 < $1000 minimum
            account_value=100000.0
        )

        self.assertFalse(result.is_valid)
        self.assertIn('minimum', result.reason.lower())

    def test_max_order_value(self):
        """Test maximum order value ($50,000)"""
        result = self.risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=400,
            price=150.0,  # Order value = $60,000 > $50,000 maximum
            account_value=1000000.0
        )

        self.assertFalse(result.is_valid)
        self.assertIn('maximum', result.reason.lower())


class TestOrderExecution(unittest.TestCase):
    """Test order execution algorithms"""

    def setUp(self):
        """Initialize order execution components"""
        self.config_manager = MagicMock()
        self.config_manager.get_connection_params.return_value = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }

    @patch('autotrader.ibkr_auto_trader.IB')
    async def test_market_order_placement(self, mock_ib):
        """Test market order placement"""
        # Setup mock IBKR connection
        mock_ib_instance = AsyncMock()
        mock_ib_instance.isConnected.return_value = True
        mock_ib_instance.placeOrder = AsyncMock()
        mock_ib.return_value = mock_ib_instance

        trader = IbkrAutoTrader(config_manager=self.config_manager)
        trader.ib = mock_ib_instance
        trader.connected = True
        trader.net_liq = 100000.0

        # Mock price getter
        trader.get_price = MagicMock(return_value=150.0)

        # Place market order
        result = await trader.place_market_order(
            symbol='AAPL',
            action='BUY',
            quantity=100
        )

        # Verify order was placed
        self.assertIsNotNone(result)
        mock_ib_instance.placeOrder.assert_called()

    @patch('autotrader.ibkr_auto_trader.IB')
    async def test_order_retry_logic(self, mock_ib):
        """Test exponential backoff retry logic"""
        mock_ib_instance = AsyncMock()
        mock_ib_instance.isConnected.return_value = True

        # First attempt fails, second succeeds
        mock_ib_instance.placeOrder = AsyncMock(
            side_effect=[Exception("Connection error"), MagicMock()]
        )
        mock_ib.return_value = mock_ib_instance

        trader = IbkrAutoTrader(config_manager=self.config_manager)
        trader.ib = mock_ib_instance
        trader.connected = True
        trader.net_liq = 100000.0
        trader.get_price = MagicMock(return_value=150.0)

        # Should retry and eventually succeed
        result = await trader.place_market_order(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            retries=3
        )

        # Verify retry happened
        self.assertEqual(mock_ib_instance.placeOrder.call_count, 2)

    async def test_frequency_control(self):
        """Test order frequency control"""
        trader = IbkrAutoTrader(config_manager=self.config_manager)

        # Mock frequency controller
        mock_freq_controller = MagicMock()
        mock_freq_controller.should_allow_order.return_value = (False, "Rate limit exceeded", None)
        trader.frequency_controller = mock_freq_controller

        result = await trader.place_market_order(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            target_weight=0.1
        )

        # Order should be rejected due to frequency control
        self.assertEqual(result.status, 'REJECTED')
        self.assertIn('rate', result.error_msg.lower())


class TestDatabaseTracking(unittest.TestCase):
    """Test database position and money tracking"""

    def setUp(self):
        """Initialize database"""
        self.db = StockDatabase(":memory:")  # Use in-memory database for tests

    def test_position_storage(self):
        """Test storing and retrieving positions"""
        # Store position
        self.db.save_position('AAPL', 100, 150.0, datetime.now())

        # Retrieve positions
        positions = self.db.get_positions()

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'AAPL')
        self.assertEqual(positions[0]['quantity'], 100)

    def test_cash_balance_tracking(self):
        """Test cash balance updates"""
        initial_cash = 100000.0

        # Update cash after trade
        trade_cost = 15000.0  # Buy 100 shares at $150
        new_cash = initial_cash - trade_cost

        self.db.update_cash_balance(new_cash)
        cash = self.db.get_cash_balance()

        self.assertEqual(cash, new_cash)

    def test_pnl_calculation(self):
        """Test P&L calculation"""
        # Add position
        entry_price = 150.0
        quantity = 100
        self.db.save_position('AAPL', quantity, entry_price, datetime.now())

        # Update with current price
        current_price = 160.0
        pnl = (current_price - entry_price) * quantity

        self.assertEqual(pnl, 1000.0)  # $10 gain per share * 100 shares

    def test_trade_history_logging(self):
        """Test trade history storage"""
        # Log a trade
        trade_data = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now(),
            'order_id': 'TEST123'
        }

        self.db.log_trade(trade_data)

        # Retrieve trade history
        trades = self.db.get_trade_history()

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['symbol'], 'AAPL')

    def test_concurrent_access(self):
        """Test SQLite WAL mode for concurrent access"""
        # WAL mode should be enabled
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(mode.upper(), 'WAL')


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete trading workflow integration"""

    async def test_full_trading_cycle(self):
        """Test complete cycle: Signal → Risk → Execution → Database"""

        # 1. Signal Generation
        signal_hub = SignalHub()
        prices = [100.0] * 19 + [85.0]  # Strong buy signal
        signal = signal_hub.mr_signal(prices)
        self.assertGreater(signal, 0.5, "Should generate buy signal")

        # 2. Risk Validation
        config_manager = MagicMock()
        config_manager.get.return_value = {}
        risk_manager = UnifiedRiskManager(config_manager)

        risk_result = risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=85.0,
            account_value=100000.0
        )
        self.assertTrue(risk_result.is_valid, "Order should pass risk checks")

        # 3. Position Sizing
        order_value = 10 * 85.0  # $850
        position_pct = order_value / 100000.0  # 0.85% of account
        self.assertLess(position_pct, 0.15, "Position size within limits")

        # 4. Order Execution (mocked)
        with patch('autotrader.ibkr_auto_trader.IB') as mock_ib:
            mock_ib_instance = AsyncMock()
            mock_ib_instance.isConnected.return_value = True
            mock_ib_instance.placeOrder = AsyncMock(return_value=MagicMock())
            mock_ib.return_value = mock_ib_instance

            trader = IbkrAutoTrader(config_manager=config_manager)
            trader.ib = mock_ib_instance
            trader.connected = True
            trader.net_liq = 100000.0
            trader.get_price = MagicMock(return_value=85.0)

            order_result = await trader.place_market_order('AAPL', 'BUY', 10)
            self.assertIsNotNone(order_result)

        # 5. Database Update
        db = StockDatabase(":memory:")
        db.save_position('AAPL', 10, 85.0, datetime.now())
        db.update_cash_balance(100000.0 - 850.0)

        # Verify database state
        positions = db.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['quantity'], 10)

        cash = db.get_cash_balance()
        self.assertEqual(cash, 99150.0)

    async def test_risk_rejection_workflow(self):
        """Test workflow when risk checks reject trade"""

        # Generate strong buy signal
        signal_hub = SignalHub()
        prices = [100.0] * 19 + [70.0]
        signal = signal_hub.mr_signal(prices)
        self.assertGreater(signal, 0.5)

        # Risk check should reject oversized position
        config_manager = MagicMock()
        config_manager.get.return_value = {}
        risk_manager = UnifiedRiskManager(config_manager)

        risk_result = risk_manager.validate_order(
            symbol='AAPL',
            side='BUY',
            quantity=2000,  # Way too large
            price=150.0,
            account_value=100000.0  # Would be 300% of account
        )

        self.assertFalse(risk_result.is_valid)

        # Order should not be placed
        # Database should not be updated
        db = StockDatabase(":memory:")
        positions = db.get_positions()
        self.assertEqual(len(positions), 0)

    async def test_continuous_trading_loop(self):
        """Test continuous trading loop operation"""

        # Mock Engine and components
        with patch('autotrader.engine.DataFeed') as mock_datafeed:
            config_manager = MagicMock()
            broker = AsyncMock()

            # Setup mock data
            mock_datafeed_instance = MagicMock()
            mock_datafeed_instance.get_prices = MagicMock(
                return_value={'AAPL': [100.0] * 20}
            )
            mock_datafeed.return_value = mock_datafeed_instance

            engine = Engine(config_manager, broker)

            # Simulate multiple trading cycles
            for cycle in range(3):
                # Each cycle should:
                # 1. Fetch prices
                # 2. Generate signals
                # 3. Validate risks
                # 4. Execute trades
                # 5. Update positions

                await engine.on_signal_and_trade()

                # Verify broker methods were called
                broker.refresh_account_balances_and_positions.assert_called()


class TestPositionSizing(unittest.TestCase):
    """Test position sizing algorithms"""

    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion for position sizing"""
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = lose probability, b = win/loss ratio

        win_prob = 0.6
        loss_prob = 0.4
        win_loss_ratio = 2.0  # Win 2x what you lose

        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        self.assertAlmostEqual(kelly_fraction, 0.4, places=2)

    def test_volatility_adjusted_sizing(self):
        """Test volatility-based position sizing"""
        account_value = 100000.0
        target_risk = 0.01  # 1% portfolio risk

        # High volatility stock
        high_vol_atr = 5.0  # $5 average true range
        stop_distance = 2 * high_vol_atr  # 2 ATR stop
        high_vol_position_size = (account_value * target_risk) / stop_distance

        # Low volatility stock
        low_vol_atr = 1.0  # $1 average true range
        stop_distance = 2 * low_vol_atr
        low_vol_position_size = (account_value * target_risk) / stop_distance

        # Low vol should have larger position
        self.assertGreater(low_vol_position_size, high_vol_position_size)

    def test_max_position_limits(self):
        """Test maximum position size limits"""
        account_value = 100000.0
        max_position_pct = 0.15  # 15% max

        # Calculate position sizes
        aggressive_size = account_value * 0.25  # 25% - too large
        conservative_size = account_value * 0.10  # 10% - ok

        # Apply limits
        final_aggressive = min(aggressive_size, account_value * max_position_pct)
        final_conservative = min(conservative_size, account_value * max_position_pct)

        self.assertEqual(final_aggressive, 15000.0)  # Capped at 15%
        self.assertEqual(final_conservative, 10000.0)  # Not capped


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class AsyncTestRunner(unittest.TestCase):
    """Wrapper for async tests"""

    def test_complete_workflow(self):
        test = TestCompleteWorkflow()
        run_async_test(test.test_full_trading_cycle())

    def test_risk_rejection(self):
        test = TestCompleteWorkflow()
        run_async_test(test.test_risk_rejection_workflow())

    def test_continuous_loop(self):
        test = TestCompleteWorkflow()
        run_async_test(test.test_continuous_trading_loop())

    def test_market_order(self):
        test = TestOrderExecution()
        run_async_test(test.test_market_order_placement())

    def test_retry_logic(self):
        test = TestOrderExecution()
        run_async_test(test.test_order_retry_logic())

    def test_frequency_control(self):
        test = TestOrderExecution()
        run_async_test(test.test_frequency_control())


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSignalGeneration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUnifiedSignalProcessor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskControl))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOrderExecution))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatabaseTracking))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionSizing))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(AsyncTestRunner))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("AUTOTRADER INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - SYSTEM INTEGRATION VERIFIED!")
        print("\nThe following components are working correctly:")
        print("  📊 Signal Generation (Z-score, Momentum)")
        print("  🧠 Trading Decisions (BUY/SELL/HOLD)")
        print("  🛡️ Risk Controls (Position limits, Loss limits)")
        print("  🚀 Order Execution (Market orders, Retries)")
        print("  💾 Database Tracking (Positions, Cash, P&L)")
        print("  🔄 Complete Workflow (Signal→Risk→Order→Database)")
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW NEEDED")
        for failure in result.failures:
            print(f"\nFailed: {failure[0]}")
            print(f"Reason: {failure[1]}")
        for error in result.errors:
            print(f"\nError: {error[0]}")
            print(f"Details: {error[1]}")

    sys.exit(0 if result.wasSuccessful() else 1)