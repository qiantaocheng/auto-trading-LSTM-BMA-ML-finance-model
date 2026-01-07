#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NASDAQ-listed symbols from nasdaqtrader and write a clean ticker file.

Data source (public):
  https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt

Output format:
  One ticker per line, uppercase, no comments.
"""

from __future__ import annotations

import argparse
import os
import urllib.request


URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="data/universe/nasdaq_listed_stocks.txt")
    ap.add_argument("--include-etf", action="store_true", help="Include ETFs (default excludes ETFs)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Empty download")

    # Header columns: Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
    header = lines[0].split("|")
    try:
        i_sym = header.index("Symbol")
        i_test = header.index("Test Issue")
        i_etf = header.index("ETF")
    except Exception:
        # fallback by known positions
        i_sym, i_test, i_etf = 0, 3, 6

    out = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            break
        parts = ln.split("|")
        if len(parts) <= max(i_sym, i_test, i_etf):
            continue
        sym = parts[i_sym].strip().upper()
        test = parts[i_test].strip().upper()
        etf = parts[i_etf].strip().upper()
        if not sym or sym == "SYMBOL":
            continue
        if test == "Y":
            continue
        if (not args.include_etf) and etf == "Y":
            continue
        # remove weird symbols used by nasdaqtrader for test issues
        if "^" in sym or "/" in sym:
            continue
        out.append(sym)

    # de-dupe keep order
    out = list(dict.fromkeys(out))
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"WROTE {args.output} ({len(out)} tickers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


