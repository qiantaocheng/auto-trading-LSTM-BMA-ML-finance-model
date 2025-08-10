#!/usr/bin/env python3
import unittest
import asyncio
from autotrader.order_state_machine import OrderManager, OrderType, OrderState


class TestOrderStateMachine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.om = OrderManager()

    async def test_create_and_transition(self):
        order = await self.om.create_order(1001, "AAPL", "BUY", 50, OrderType.MARKET)
        self.assertEqual(order.order_id, 1001)
        self.assertEqual(order.state, OrderState.PENDING)

        ok = await self.om.update_order_state(1001, OrderState.SUBMITTED, {"liquidity": 0.9})
        self.assertTrue(ok)
        self.assertEqual(order.state, OrderState.SUBMITTED)

        ok = await self.om.update_order_state(1001, OrderState.FILLED, {"filled_quantity": 50, "avg_fill_price": 190.5})
        self.assertTrue(ok)
        self.assertEqual(order.state, OrderState.FILLED)
        self.assertTrue(order.is_terminal())

        stats = await self.om.get_statistics()
        self.assertEqual(stats["filled_orders"], 1)


if __name__ == '__main__':
    unittest.main()