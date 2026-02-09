#!/usr/bin/env python
"""
Apply filters to the MultiIndex parquet file:
1. Remove warrants (tickers ending with W patterns like DJTWW)
2. Min price: exclude stocks with median close below threshold (default $2)
3. Min market cap: exclude stocks with market cap < 100M (loaded from JSON or Polygon API)
4. Liquidity: exclude low volume stocks (if Volume column exists)

Creates a new filtered file: polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_FILTERED.parquet
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Configuration
TRADE_DIR = Path(r"D:\trade")
DATA_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet"
OUTPUT_FILE = DATA_FILE.parent / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO_FILTERED.parquet"
# Market cap source: JSON with ticker -> { "market_cap": number }. Tickers with cap >= min_market_cap are kept.
MARKET_CAP_JSON = TRADE_DIR / "filtered_tickers_over_100m_detail.json"
MIN_MARKET_CAP = 100_000_000  # 100M USD

# Known warrant tickers (explicitly identified)
# These are warrants that caused significant drag in Top 10
WARRANT_BLACKLIST = {
    "DJTWW",   # Trump Media warrants - biggest drag (-492% cumulative)
    "DWACW",   # DWAC warrants
}

# Known legitimate tickers that end with W (NOT warrants)
LEGITIMATE_W_TICKERS = {
    "PANW",   # Palo Alto Networks
    "SNOW",   # Snowflake
    "SCHW",   # Charles Schwab
    "TROW",   # T Rowe Price
    "CHRW",   # CH Robinson Worldwide
    "ACIW",   # ACI Worldwide
    "DNOW",   # NOW Inc
    "AROW",   # Arrow Financial
    "EZPW",   # EZCORP
    "HROW",   # Harrow Health
    "INSW",   # International Seaways
    "MATW",   # Matthews International
    "NWS",    # News Corp
    "PLOW",   # Douglas Dynamics
    "SKYW",   # SkyWest
    "HAYW",   # Hayward Holdings
    "FLYW",   # Flywire
    "STGW",   # Stagwell
    "ZWS",    # Zurn Elkay Water
    "STEW",   # SRH Total Return
    "WS",     # Worthington Steel
}


def is_warrant(ticker: str) -> bool:
    """Check if ticker is a warrant - conservative approach"""
    # Explicit blacklist
    if ticker in WARRANT_BLACKLIST:
        return True
    # Explicit whitelist of legitimate tickers
    if ticker in LEGITIMATE_W_TICKERS:
        return False
    # Pattern: ends with WW (common warrant pattern like DJTWW, DWACW)
    if ticker.endswith('WW') and len(ticker) >= 4:
        return True
    # Pattern: .W or .WS suffix (explicit warrant notation)
    if '.W' in ticker or ticker.endswith('.WS'):
        return True
    return False


def load_market_caps_from_json(path: Path) -> Dict[str, float]:
    """Load ticker -> market_cap from JSON (e.g. filtered_tickers_over_100m_detail.json)."""
    caps = {}
    if not path.exists():
        return caps
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ticker, obj in data.items():
            if not isinstance(obj, dict):
                continue
            mc = obj.get("market_cap")
            if mc is not None and isinstance(mc, (int, float)) and not pd.isna(mc):
                caps[str(ticker).strip().upper()] = float(mc)
        print(f"  Loaded market cap for {len(caps)} tickers from {path.name}")
    except Exception as e:
        print(f"  Warning: could not load market cap JSON: {e}")
    return caps


def fetch_market_caps_polygon(tickers: list, min_cap: Optional[float] = None) -> Dict[str, float]:
    """Fetch market cap from Polygon API for given tickers. If min_cap is None, return all caps; else return only ticker -> cap for cap >= min_cap."""
    caps = {}
    if not tickers:
        return caps
    try:
        import sys
        sys.path.insert(0, str(TRADE_DIR))
        from polygon_client import polygon_client
        # Pass min_market_cap=None to get all caps, then we filter locally; or pass min_cap to get only passers
        df = polygon_client.get_batch_market_caps(tickers, min_market_cap=min_cap)
        if not df.empty and "ticker" in df.columns and "market_cap" in df.columns:
            caps = dict(zip(df["ticker"].astype(str).str.upper(), df["market_cap"]))
        label = f">= ${min_cap/1e6:.0f}M" if min_cap else "all"
        print(f"  Fetched market cap from Polygon for {len(caps)} tickers ({label})")
    except Exception as e:
        print(f"  Warning: Polygon market cap fetch failed: {e}")
    return caps


def filter_multiindex_data(
    min_price: float = 2.0,
    min_avg_volume: float = 100000,
    min_market_cap: float = MIN_MARKET_CAP,
    market_cap_json: Optional[Path] = None,
    use_polygon_for_missing_cap: bool = False,
):
    """Apply filters to MultiIndex data and save new file."""

    if market_cap_json is None:
        market_cap_json = MARKET_CAP_JSON

    print("="*70)
    print("FILTERING MULTIINDEX DATA")
    print("="*70)
    print(f"Input: {DATA_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Filters:")
    print(f"  1. Remove warrants (tickers ending with W patterns)")
    print(f"  2. Min price: ${min_price}")
    print(f"  3. Min market cap: ${min_market_cap/1e6:.0f}M")
    print(f"  4. Min avg volume: {min_avg_volume:,.0f}")
    print("="*70)

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(DATA_FILE)

    original_rows = len(df)
    if isinstance(df.index, pd.MultiIndex):
        all_tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        all_tickers = df['ticker'].unique().tolist()

    print(f"Original: {len(all_tickers)} tickers, {original_rows:,} rows")
    print(f"\nColumns available: {list(df.columns)[:15]}...")

    # ===== Filter 1: Remove warrants =====
    print("\n--- Filter 1: Warrants ---")
    warrants = [t for t in all_tickers if is_warrant(t)]
    print(f"Warrants found ({len(warrants)}): {warrants}")

    if isinstance(df.index, pd.MultiIndex):
        ticker_level = df.index.get_level_values('ticker')
        warrant_mask = ticker_level.isin(warrants)
        df = df[~warrant_mask]
    else:
        df = df[~df['ticker'].isin(warrants)]

    print(f"Rows after warrant filter: {len(df):,}")

    # ===== Filter 2: Low price stocks =====
    print(f"\n--- Filter 2: Price < ${min_price} ---")
    # Check for both 'close' and 'Close' column names
    close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
    if close_col:
        if isinstance(df.index, pd.MultiIndex):
            # Get median close price per ticker
            ticker_median_price = df.groupby(level='ticker')[close_col].median()
            low_price_tickers = ticker_median_price[ticker_median_price < min_price].index.tolist()
            df = df[~df.index.get_level_values('ticker').isin(low_price_tickers)]
        else:
            ticker_median_price = df.groupby('ticker')[close_col].median()
            low_price_tickers = ticker_median_price[ticker_median_price < min_price].index.tolist()
            df = df[~df['ticker'].isin(low_price_tickers)]

        print(f"Low price tickers ({len(low_price_tickers)}): {low_price_tickers[:30]}{'...' if len(low_price_tickers) > 30 else ''}")
        print(f"Rows after price filter: {len(df):,}")
    else:
        print("'close'/'Close' column not found - skipping price filter")

    # ===== Filter 3: Low volume stocks =====
    print(f"\n--- Filter 3: Avg Volume < {min_avg_volume:,.0f} ---")
    # Check for both 'volume' and 'Volume' column names
    vol_col = 'Volume' if 'Volume' in df.columns else ('volume' if 'volume' in df.columns else None)
    if vol_col:
        if isinstance(df.index, pd.MultiIndex):
            ticker_avg_vol = df.groupby(level='ticker')[vol_col].mean()
            low_vol_tickers = ticker_avg_vol[ticker_avg_vol < min_avg_volume].index.tolist()
            df = df[~df.index.get_level_values('ticker').isin(low_vol_tickers)]
        else:
            ticker_avg_vol = df.groupby('ticker')[vol_col].mean()
            low_vol_tickers = ticker_avg_vol[ticker_avg_vol < min_avg_volume].index.tolist()
            df = df[~df['ticker'].isin(low_vol_tickers)]

        print(f"Low volume tickers ({len(low_vol_tickers)}): {low_vol_tickers[:30]}{'...' if len(low_vol_tickers) > 30 else ''}")
        print(f"Rows after volume filter: {len(df):,}")
    else:
        print("'volume'/'Volume' column not found - skipping volume filter")

    # ===== Filter 4: Market cap < 100M =====
    print(f"\n--- Filter 4: Market cap < ${min_market_cap/1e6:.0f}M ---")
    if isinstance(df.index, pd.MultiIndex):
        current_tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        current_tickers = df['ticker'].unique().tolist()

    caps = load_market_caps_from_json(market_cap_json)
    missing_or_low = [t for t in current_tickers if t not in caps or caps[t] < min_market_cap]
    if missing_or_low:
        print(f"  Fetching market cap from Polygon for {len(missing_or_low)} tickers (missing or < ${min_market_cap/1e6:.0f}M in JSON)...")
        polygon_caps = fetch_market_caps_polygon(missing_or_low, min_cap=None)
        for t, cap in polygon_caps.items():
            caps[t] = cap
        # Save merged caps to JSON for next time (merge with existing so we don't lose other tickers)
        try:
            existing = {}
            if market_cap_json.exists():
                with open(market_cap_json, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    if isinstance(v, dict) and v.get("market_cap") is not None:
                        existing[k] = {"market_cap": v["market_cap"]}
            for t, c in caps.items():
                existing[t] = {"market_cap": c}
            with open(market_cap_json, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=0)
            print(f"  Updated filter file: {market_cap_json.name}")
        except Exception as e:
            print(f"  Could not save filter JSON: {e}")

    keep_cap_tickers = [t for t in current_tickers if caps.get(t) is not None and caps[t] >= min_market_cap]
    low_cap_tickers = [t for t in current_tickers if t not in keep_cap_tickers]
    print(f"Low cap / no cap tickers ({len(low_cap_tickers)}): {low_cap_tickers[:30]}{'...' if len(low_cap_tickers) > 30 else ''}")

    if isinstance(df.index, pd.MultiIndex):
        df = df[df.index.get_level_values('ticker').isin(keep_cap_tickers)]
    else:
        df = df[df['ticker'].isin(keep_cap_tickers)]
    print(f"Rows after market cap filter: {len(df):,}")

    # ===== Summary =====
    final_rows = len(df)
    if isinstance(df.index, pd.MultiIndex):
        final_tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        final_tickers = df['ticker'].unique().tolist()

    print("\n" + "="*70)
    print("FILTER SUMMARY")
    print("="*70)
    print(f"Tickers: {len(all_tickers)} -> {len(final_tickers)} ({len(all_tickers) - len(final_tickers)} removed)")
    print(f"Rows: {original_rows:,} -> {final_rows:,} ({original_rows - final_rows:,} removed)")

    removed_tickers = set(all_tickers) - set(final_tickers)
    print(f"\nAll removed tickers ({len(removed_tickers)}):")
    for t in sorted(removed_tickers):
        print(f"  {t}")

    # Save filtered data
    print(f"\nSaving filtered data to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE)
    print("Done!")

    print("\n" + "="*70)
    print("TO USE FILTERED DATA:")
    print(f"Update DATA_FILE in scripts to: {OUTPUT_FILE}")
    print("="*70)

    return df


if __name__ == "__main__":
    filter_multiindex_data(min_price=2.0, min_avg_volume=100000, min_market_cap=100_000_000)
