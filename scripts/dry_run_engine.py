#!/usr/bin/env python3
import asyncio
from types import SimpleNamespace
from autotrader.engine import Engine
from autotrader.config import HotConfig


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
    cfg = HotConfig()
    # 精简 universe 以便干跑
    cfg.get()["CONFIG"]["scanner"]["universe"] = ["AAPL", "MSFT", "NVDA"]
    broker = DummyBroker()
    engine = Engine(cfg, broker)

    # 预置tickers
    for s in cfg.get()["CONFIG"]["scanner"]["universe"]:
        broker.tickers[s] = DummyTicker()

    # 运行一次信号与交易流程（不会真正下单）
    await engine.on_signal_and_trade()
    print("Dry run completed.")


if __name__ == "__main__":
    asyncio.run(main())