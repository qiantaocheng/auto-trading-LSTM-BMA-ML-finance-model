#!/usr/bin/env python
"""IBKR trading bridge CLI using ib_async synchronous API.

Usage:
    python trade_bridge.py --host 127.0.0.1 --port 4002 --client-id 123 buy --symbol AAPL --quantity 10
    python trade_bridge.py --port 4002 sell --symbol AAPL --quantity 10
    python trade_bridge.py --port 4002 price --symbol AAPL
    python trade_bridge.py --port 4002 cash
    python trade_bridge.py market-status

Each command prints a single JSON object to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from ib_async import IB, Stock, MarketOrder


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def cmd_buy(args):
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    try:
        contract = Stock(args.symbol.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("BUY", args.quantity)
        trade = ib.placeOrder(contract, order)

        # ib.sleep() pumps the event loop so fill messages are received
        for _ in range(60):
            ib.sleep(0.5)
            if trade.isDone():
                break

        status = trade.orderStatus.status
        avg_price = 0.0
        if trade.orderStatus.avgFillPrice:
            avg_price = _safe_float(trade.orderStatus.avgFillPrice)
        elif trade.fills:
            avg_price = _safe_float(trade.fills[-1].execution.avgPrice)

        return {
            "success": status in ("Filled", "Submitted", "PreSubmitted"),
            "order_id": trade.order.orderId,
            "symbol": args.symbol.upper(),
            "action": "BUY",
            "quantity": args.quantity,
            "avg_price": avg_price,
            "status": status,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "symbol": args.symbol.upper(), "action": "BUY",
                "order_id": 0, "quantity": args.quantity, "avg_price": 0.0}
    finally:
        ib.disconnect()


def cmd_sell(args):
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    try:
        contract = Stock(args.symbol.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("SELL", args.quantity)
        trade = ib.placeOrder(contract, order)

        for _ in range(60):
            ib.sleep(0.5)
            if trade.isDone():
                break

        status = trade.orderStatus.status
        avg_price = 0.0
        if trade.orderStatus.avgFillPrice:
            avg_price = _safe_float(trade.orderStatus.avgFillPrice)
        elif trade.fills:
            avg_price = _safe_float(trade.fills[-1].execution.avgPrice)

        return {
            "success": status in ("Filled", "Submitted", "PreSubmitted"),
            "order_id": trade.order.orderId,
            "symbol": args.symbol.upper(),
            "action": "SELL",
            "quantity": args.quantity,
            "avg_price": avg_price,
            "status": status,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "symbol": args.symbol.upper(), "action": "SELL",
                "order_id": 0, "quantity": args.quantity, "avg_price": 0.0}
    finally:
        ib.disconnect()


def cmd_price(args):
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    try:
        ib.reqMarketDataType(3)  # Enable delayed data as fallback
        contract = Stock(args.symbol.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqTickers(contract)
        ib.sleep(2)  # Give time for data to arrive
        [ticker] = ib.reqTickers(contract)
        price = ticker.marketPrice()
        if price != price:  # NaN check
            price = ticker.close
        if price != price:
            return {"symbol": args.symbol.upper(), "price": None, "error": "Price unavailable"}
        return {"symbol": args.symbol.upper(), "price": _safe_float(price)}
    except Exception as e:
        return {"symbol": args.symbol.upper(), "price": None, "error": str(e)}
    finally:
        ib.disconnect()


def cmd_cash(args):
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    try:
        ib.reqAccountSummary()
        ib.sleep(2)  # Give time for account data
        net_liq = 0.0
        cash = 0.0
        for av in ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                net_liq = _safe_float(av.value)
            elif av.tag == "TotalCashValue" and av.currency == "USD":
                cash = _safe_float(av.value)
        return {"cash": cash, "net_liq": net_liq}
    except Exception as e:
        return {"cash": 0.0, "net_liq": 0.0, "error": str(e)}
    finally:
        ib.disconnect()


def cmd_market_status(args):
    try:
        import pytz
        from datetime import datetime

        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)

        is_weekday = now_et.weekday() < 5
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_open = is_weekday and market_open <= now_et <= market_close
        is_friday = now_et.weekday() == 4

        return {
            "is_open": is_open,
            "is_friday": is_friday,
            "et_time": now_et.strftime("%Y-%m-%dT%H:%M:%S"),
            "et_hour": now_et.hour,
            "et_minute": now_et.minute,
            "weekday": now_et.weekday(),
        }
    except Exception as e:
        return {"is_open": False, "is_friday": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="IBKR Trading Bridge")

    # IBKR connection parameters (must come BEFORE subcommand)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=123)

    subparsers = parser.add_subparsers(dest="command", required=True)

    buy_parser = subparsers.add_parser("buy")
    buy_parser.add_argument("--symbol", required=True)
    buy_parser.add_argument("--quantity", type=int, required=True)

    sell_parser = subparsers.add_parser("sell")
    sell_parser.add_argument("--symbol", required=True)
    sell_parser.add_argument("--quantity", type=int, required=True)

    price_parser = subparsers.add_parser("price")
    price_parser.add_argument("--symbol", required=True)

    subparsers.add_parser("cash")
    subparsers.add_parser("market-status")

    args = parser.parse_args()

    handlers = {
        "buy": cmd_buy,
        "sell": cmd_sell,
        "price": cmd_price,
        "cash": cmd_cash,
        "market-status": cmd_market_status,
    }

    handler = handlers[args.command]
    result = handler(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
