#!/usr/bin/env python3
import unittest
import asyncio
from types import SimpleNamespace
from datetime import datetime
import contextlib

from autotrader.ibkr_auto_trader import IbkrAutoTrader


class StubOrder:
    def __init__(self, orderId: int):
        self.orderId = orderId


class StubOrderStatus:
    def __init__(self, status: str = 'Filled', filled: int = 0, avgFillPrice: float = 0.0):
        self.status = status
        self.filled = filled
        self.avgFillPrice = avgFillPrice


class StubTrade:
    def __init__(self, orderId: int, qty: int = 0, avg_price: float = 0.0):
        self.order = StubOrder(orderId)
        self.orderStatus = StubOrderStatus('Filled', qty, avg_price)
        self.doneEvent = asyncio.Event()
        self.doneEvent.set()


class StubIB:
    def __init__(self):
        self.order_counter = 1000
        self.cancel_called = False
        self.placed_orders = []

    def placeOrder(self, contract, order):
        self.order_counter += 1
        self.placed_orders.append((contract, order))
        qty = int(getattr(order, 'totalQuantity', 0) or getattr(order, 'totalQuantity', 0))
        avg = float(getattr(order, 'lmtPrice', 100.0) or 100.0)
        return StubTrade(self.order_counter, qty, avg)

    def reqGlobalCancel(self):
        self.cancel_called = True

    async def qualifyContractsAsync(self, contract):
        return [contract]

    def reqMktData(self, contract, snapshot=False):
        return None

    def ticker(self, contract):
        return None

    def isConnected(self):
        return True


class TestOrderFlowWithStubs(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stub_ib = StubIB()
        # 修复：使用正确的初始化参数
        from autotrader.unified_config import get_unified_config
        config_manager = get_unified_config()
        self.trader = IbkrAutoTrader(config_manager=config_manager, ib_client=self.stub_ib)
        self.trader.account_ready = True
        self.trader.net_liq = 100000.0
        self.trader.cash_balance = 50000.0
        # 提供价格
        self.trader.tickers['AAPL'] = SimpleNamespace(last=100.0, bid=99.0, ask=101.0, close=None)

    async def test_market_order_submit(self):
        ref = await self.trader.place_market_order('AAPL', 'BUY', 10)
        self.assertIsNotNone(ref)
        self.assertGreater(ref.order_id, 0)
        self.assertEqual(self.trader._daily_order_count, 1)

    async def test_cancel_all_open_orders_calls_ib(self):
        self.trader.cancel_all_open_orders()
        self.assertTrue(self.stub_ib.cancel_called)

    async def test_dynamic_stop_manager_places_stop_when_enabled(self):
        # 启用本地动态止损
        self.trader.risk_config['risk_management']['enable_local_dynamic_stop_for_bracket'] = True
        # 设置入场状态
        symbol = 'AAPL'
        self.trader._stop_state[symbol] = {
            'entry_price': 100.0,
            'entry_time': datetime.now(),
            'qty': 10,
            'stop_trade': None,
            'current_stop': 95.0,
        }
        # 生成假bars用于ATR
        async def fake_bars(sym, lookback_days):
            class Bar:
                def __init__(self, h, l, c):
                    self.high = h
                    self.low = l
                    self.close = c
            return [Bar(101 + i*0.01, 99 - i*0.01, 100 + i*0.01) for i in range(30)]
        self.trader._fetch_daily_bars = fake_bars  # type: ignore
        # 缩短间隔
        self.trader.dynamic_stop_cfg['update_interval_sec'] = 0.05
        # 启动一次并很快停止
        task = asyncio.create_task(self.trader._dynamic_stop_manager(symbol))
        await asyncio.sleep(0.12)
        # 触发停止
        self.trader._stop_event = self.trader._stop_event or asyncio.Event()
        self.trader._stop_event.set()
        await asyncio.sleep(0.05)
        # 至少应有一次下发
        self.assertGreaterEqual(len(self.stub_ib.placed_orders), 1)
        # 清理任务
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


if __name__ == '__main__':
    unittest.main()