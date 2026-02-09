#!/usr/bin/env python
"""Initialize the new TraderApp SQLite database."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "src" / "Trader.App" / "TraderApp.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    symbol TEXT PRIMARY KEY,
    tag INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS auto_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    scheduled_exit TEXT NOT NULL,
    shares INTEGER NOT NULL,
    entry_price REAL,
    UNIQUE(symbol)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_fill_price REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
    print(f"[OK] Initialized database at {DB_PATH}")

if __name__ == "__main__":
    main()
