#!/usr/bin/env python
"""Emit current IBKR portfolio snapshot with Polygon API delayed prices."""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import urllib.error


def _safe_float(v, default=0.0):
    """Convert to float, replacing NaN/Inf with default."""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def fetch_polygon_prices(symbols: list[str], api_key: str) -> dict[str, float]:
    """Fetch delayed prices from Polygon API for a list of symbols."""
    prices = {}
    if not api_key or not symbols:
        return prices

    # Try batch snapshot first (most current delayed data)
    tickers_param = ",".join(symbols)
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?tickers={tickers_param}&apiKey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "OK" and "tickers" in data:
                for t in data["tickers"]:
                    sym = t.get("ticker", "")
                    # Use lastTrade price, fall back to day close, then prevDay close
                    price = None
                    if t.get("lastTrade", {}).get("p"):
                        price = _safe_float(t["lastTrade"]["p"])
                    if not price and t.get("day", {}).get("c"):
                        price = _safe_float(t["day"]["c"])
                    if not price and t.get("prevDay", {}).get("c"):
                        price = _safe_float(t["prevDay"]["c"])
                    if price and price > 0:
                        prices[sym] = price
                if len(prices) == len(symbols):
                    return prices
    except (urllib.error.HTTPError, urllib.error.URLError, Exception):
        pass  # Fall back to individual prev close calls

    # Fallback: individual previous close for missing symbols
    for sym in symbols:
        if sym in prices:
            continue
        url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?adjusted=true&apiKey={api_key}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data.get("resultsCount", 0) > 0 and data.get("results"):
                    close = _safe_float(data["results"][0].get("c", 0))
                    if close > 0:
                        prices[sym] = close
        except (urllib.error.HTTPError, urllib.error.URLError, Exception):
            pass

    return prices


def capture_snapshot(host: str, port: int, client_id: int, polygon_key: str) -> dict:
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
        ib.reqAccountSummary()
        ib.sleep(2)

        # Get account values
        cash = 0.0
        for av in ib.accountValues():
            if av.tag == "TotalCashValue" and av.currency == "USD":
                cash = _safe_float(av.value)

        # Get positions (just symbols + quantities)
        positions = []
        for pos in ib.positions():
            symbol = pos.contract.symbol
            quantity = int(pos.position)
            avg_cost = _safe_float(pos.avgCost)
            if quantity != 0:
                positions.append({"symbol": symbol, "quantity": quantity, "avg_cost": avg_cost})

    finally:
        ib.disconnect()

    # Fetch prices from Polygon API
    symbols = [p["symbol"] for p in positions]
    polygon_prices = fetch_polygon_prices(symbols, polygon_key) if polygon_key else {}

    # Build holdings with Polygon prices (fall back to IBKR avgCost)
    holdings = []
    stock_value = 0.0
    for p in positions:
        sym = p["symbol"]
        qty = p["quantity"]
        price = polygon_prices.get(sym, p["avg_cost"])
        market_value = qty * price
        stock_value += market_value
        holdings.append({
            "symbol": sym,
            "quantity": qty,
            "market_price": _safe_float(price),
        })

    # Calculate net_liq = cash + stock value
    net_liq = cash + stock_value

    return {"net_liq": net_liq, "cash": cash, "holdings": holdings}


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR Portfolio Snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=123)
    parser.add_argument("--polygon-key", default="")
    args = parser.parse_args()

    try:
        snapshot = capture_snapshot(args.host, args.port, args.client_id, args.polygon_key)
        print(json.dumps(snapshot, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
