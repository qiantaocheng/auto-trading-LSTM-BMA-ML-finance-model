#!/usr/bin/env python3
import unittest
from autotrader.unified_config import get_unified_config
from autotrader.ibkr_auto_trader import IbkrAutoTrader


class TestRiskConfigSync(unittest.TestCase):
    def test_sync_limits(self):
        config_manager = get_unified_config()
        cfg = config_manager._get_merged_config()
        # tweak values for test
        cfg["capital"]["cash_reserve_pct"] = 0.12
        cfg["capital"]["max_single_position_pct"] = 0.11

        # 创建trader实例
        trader = IbkrAutoTrader(config_manager=config_manager)
        # simulate sync
        trader._sync_risk_limits_from_config({"CONFIG": cfg})

        # 验证配置同步
        self.assertAlmostEqual(trader.order_verify_cfg["cash_reserve_pct"], 0.12, places=6)
        self.assertAlmostEqual(trader.order_verify_cfg["max_single_position_pct"], 0.11, places=6)
        # 风险管理器已统一，不再测试单独属性


if __name__ == '__main__':
    unittest.main()