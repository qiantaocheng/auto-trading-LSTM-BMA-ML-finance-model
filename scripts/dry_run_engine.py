#!/usr/bin/env python3
import asyncio
from types import SimpleNamespace
from autotrader.engine import Engine
from autotrader.unified_config import get_unified_config


class DummyTicker:
    def __init__(self, last=100.0, bid=99.5, ask=100.5, close=None):
        self.last = last
        self.bid = bid
        self.ask = ask
        self.close = close
        self.bidSize = 100
        self.askSize = 100


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
        self.tickers.setdefault(s, DummyTicker())
        return None

    async def qualify_stock(self, s):
        return SimpleNamespace(symbol=s)


async def main():
    config_manager = get_unified_config()
    # 精简 universe 以便干跑
    broker = DummyBroker()
    engine = Engine(config_manager, broker)

    # 预置tickers
    universe = config_manager.get_universe()
    for s in universe[:3]:  # 只取前3个用于测试
        broker.tickers[s] = DummyTicker()

    # 运行一次信号与交易流程（不会真正下单）
    await engine.on_signal_and_trade()
    print("Dry run completed.")


if __name__ == "__main__":
    asyncio.run(main())