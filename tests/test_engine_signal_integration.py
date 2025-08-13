#!/usr/bin/env python3
import unittest
import asyncio
from autotrader.engine import Engine, SignalHub
from autotrader.unified_config import get_unified_config
from types import SimpleNamespace


class DummyBroker:
    def __init__(self):
        self.ib = SimpleNamespace()
        self.tickers = {}
        self.positions = {}
        self.net_liq = 100000.0
        self.cash_balance = 90000.0

    async def connect(self):
        return None

    async def refresh_account_balances_and_positions(self):
        return None

    async def subscribe(self, s):
        return None


class TestEngineSignalIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_multi_factor_signal(self):
        broker = DummyBroker()
        config_manager = get_unified_config()
        engine = Engine(config_manager, broker)
        sh = engine.signal

        # fabricate series
        closes = [100 + i * 0.2 for i in range(60)]
        highs = [c * 1.01 for c in closes]
        lows = [c * 0.99 for c in closes]
        vols = [1_000_000 + i * 1000 for i in range(60)]

        score = sh.multi_factor_signal(closes[-50:], highs[-50:], lows[-50:], vols[-50:])
        self.assertTrue(-1.0 <= score <= 1.0)


if __name__ == '__main__':
    unittest.main()