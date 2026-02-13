"""Seed the positions table from broker_positions (IBKR truth).

ETF rotation tickers are identified by matching known Portfolio B tickers.
All others get strategy='Manual'.
"""
import sqlite3

db_path = "D:/trade/TraderApp/publish_v4/TraderApp.db"

# Portfolio B ETF tickers
ETF_TICKERS = {"QQQ", "USMV", "QUAL", "PDBC", "COPX", "URA", "DBA"}

# Portfolio B target weights
ETF_WEIGHTS = {
    "QQQ": 0.25,
    "USMV": 0.25,
    "QUAL": 0.20,
    "PDBC": 0.15,
    "COPX": 0.05,
    "URA": 0.05,
    "DBA": 0.05,
}

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get broker positions
c.execute("SELECT symbol, quantity, avg_cost FROM broker_positions WHERE quantity > 0")
broker_rows = c.fetchall()

# Check what's already in positions
c.execute("SELECT symbol FROM positions")
existing = {r[0] for r in c.fetchall()}

now = "2026-02-10T15:00:00Z"
inserted = 0

for symbol, qty, avg_cost in broker_rows:
    if symbol in existing:
        print(f"  SKIP {symbol} (already in positions)")
        continue

    if symbol.upper() in ETF_TICKERS:
        strategy = "ETF"
        weight = ETF_WEIGHTS.get(symbol.upper())
        scheduled_exit = None
    else:
        strategy = "Manual"
        weight = None
        # Manual: far-future exit
        scheduled_exit = "2099-12-31T00:00:00Z"

    c.execute(
        """INSERT INTO positions (symbol, strategy, shares, entry_price, entered_at, scheduled_exit, target_weight, last_rebalanced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol.upper(), strategy, qty, avg_cost, now, scheduled_exit, weight, now if strategy == "ETF" else None)
    )
    inserted += 1
    print(f"  INSERT {symbol}: {strategy}, {qty} shares @ {avg_cost:.2f}" +
          (f", weight={weight}" if weight else ""))

conn.commit()
print(f"\nDone: {inserted} positions inserted")

# Verify
c.execute("SELECT symbol, strategy, shares, entry_price, target_weight FROM positions ORDER BY strategy, symbol")
print("\n=== positions table ===")
for r in c.fetchall():
    print(f"  {r}")

conn.close()
