#!/usr/bin/env python3
import unittest
from autotrader.config import HotConfig
from autotrader.ibkr_auto_trader import IbkrAutoTrader


class TestRiskConfigSync(unittest.TestCase):
    def test_sync_limits(self):
        cfg = HotConfig().get()
        # tweak values for test
        cfg["CONFIG"]["capital"]["cash_reserve_pct"] = 0.12
        cfg["CONFIG"]["capital"]["max_single_position_pct"] = 0.11

        trader = IbkrAutoTrader("127.0.0.1", 7497, 1)
        # simulate sync
        trader._sync_risk_limits_from_config(cfg)

        self.assertAlmostEqual(trader.order_verify_cfg["cash_reserve_pct"], 0.12, places=6)
        self.assertAlmostEqual(trader.order_verify_cfg["max_single_position_pct"], 0.11, places=6)
        self.assertAlmostEqual(trader.risk_manager.max_single_position, 0.11, places=6)


if __name__ == '__main__':
    unittest.main()