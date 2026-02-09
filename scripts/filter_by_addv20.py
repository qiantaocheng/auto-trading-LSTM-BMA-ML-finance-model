#!/usr/bin/env python
"""
Filter stocks by ADDV20 (Average Daily Dollar Volume over 20 days).

ADDV20 = mean(Close * Volume) over trailing 20 days
Remove stocks with ADDV20 < 1 million on 2026-02-02

If filtered file exists, skip. Otherwise filter and save.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Paths
TRADE_DIR = Path(r"D:\trade")
SOURCE_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet"
OUTPUT_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO_ADDV_FILTERED.parquet"

# Filter parameters
REFERENCE_DATE = "2026-02-02"
ADDV_LOOKBACK = 20  # 20 trading days
ADDV_THRESHOLD = 1_000_000  # $1 million minimum

os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))


def calculate_addv20_from_parquet(df: pd.DataFrame, ref_date: str, lookback: int = 20) -> pd.DataFrame:
    """
    Calculate ADDV20 for each ticker using data from parquet.

    ADDV20 = mean(Close * Volume) over trailing 20 days ending at ref_date
    """
    ref_date = pd.Timestamp(ref_date)

    # Get date column
    if isinstance(df.index, pd.MultiIndex):
        dates = df.index.get_level_values('date')
        tickers = df.index.get_level_values('ticker')
    else:
        dates = pd.to_datetime(df['date'])
        tickers = df['ticker']

    # Find dates for lookback window
    unique_dates = sorted(dates.unique())

    # Find ref_date or closest date before it
    valid_dates = [d for d in unique_dates if d <= ref_date]
    if not valid_dates:
        raise ValueError(f"No data available before {ref_date}")

    actual_ref_date = valid_dates[-1]
    print(f"Reference date: {ref_date.date()} -> Using: {actual_ref_date.date() if hasattr(actual_ref_date, 'date') else actual_ref_date}")

    # Get lookback window dates
    ref_idx = unique_dates.index(actual_ref_date)
    start_idx = max(0, ref_idx - lookback + 1)
    window_dates = unique_dates[start_idx:ref_idx + 1]

    print(f"ADDV20 window: {len(window_dates)} days from {window_dates[0]} to {window_dates[-1]}")

    # Filter data to window
    if isinstance(df.index, pd.MultiIndex):
        mask = df.index.get_level_values('date').isin(window_dates)
        window_df = df.loc[mask].copy()
    else:
        mask = df['date'].isin(window_dates)
        window_df = df.loc[mask].copy()

    # Check for Close and Volume columns
    close_col = None
    volume_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'close' or col_lower == 'adj_close':
            close_col = col
        elif col_lower == 'volume':
            volume_col = col

    if close_col is None or volume_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing Close or Volume column. Found close={close_col}, volume={volume_col}")

    print(f"Using columns: Close='{close_col}', Volume='{volume_col}'")

    # Calculate dollar volume
    window_df['dollar_volume'] = window_df[close_col] * window_df[volume_col]

    # Calculate ADDV20 per ticker
    if isinstance(window_df.index, pd.MultiIndex):
        addv20 = window_df.groupby('ticker')['dollar_volume'].mean()
    else:
        addv20 = window_df.groupby('ticker')['dollar_volume'].mean()

    addv20_df = addv20.reset_index()
    addv20_df.columns = ['ticker', 'ADDV20']

    return addv20_df


def fetch_from_polygon(tickers: list, ref_date: str, lookback: int = 20) -> pd.DataFrame:
    """
    Fetch Close and Volume from Polygon API for ADDV20 calculation.
    """
    try:
        from polygon_client import PolygonClient
    except ImportError:
        print("Warning: polygon_client not available, trying direct API...")
        return fetch_from_polygon_direct(tickers, ref_date, lookback)

    client = PolygonClient()

    ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
    start_dt = ref_dt - timedelta(days=lookback * 2)  # Extra buffer for trading days

    results = []

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  Fetching {i}/{len(tickers)}...")

        try:
            bars = client.get_daily_bars(
                ticker,
                start_dt.strftime("%Y-%m-%d"),
                ref_date
            )

            if bars is not None and len(bars) > 0:
                # Get last 20 trading days
                bars = bars.tail(lookback)
                dollar_vol = (bars['close'] * bars['volume']).mean()
                results.append({'ticker': ticker, 'ADDV20': dollar_vol})
        except Exception as e:
            pass  # Skip failed tickers

    return pd.DataFrame(results)


def fetch_from_polygon_direct(tickers: list, ref_date: str, lookback: int = 20) -> pd.DataFrame:
    """
    Fetch directly from Polygon REST API.
    """
    import requests

    # Try to get API key
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        api_key_file = TRADE_DIR / ".polygon_api_key"
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()

    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in environment or .polygon_api_key file")

    ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
    start_dt = ref_dt - timedelta(days=lookback * 2)

    results = []

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Fetching {i}/{len(tickers)}...")

        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_dt.strftime('%Y-%m-%d')}/{ref_date}"
            resp = requests.get(url, params={'apiKey': api_key, 'limit': 50})

            if resp.status_code == 200:
                data = resp.json()
                if 'results' in data and len(data['results']) > 0:
                    bars = data['results'][-lookback:]  # Last 20 days
                    dollar_vols = [b['c'] * b['v'] for b in bars]
                    addv20 = np.mean(dollar_vols)
                    results.append({'ticker': ticker, 'ADDV20': addv20})
        except Exception as e:
            pass

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("ADDV20 FILTER - Remove stocks with ADDV20 < $1M")
    print("=" * 70)
    print(f"Reference date: {REFERENCE_DATE}")
    print(f"Lookback: {ADDV_LOOKBACK} days")
    print(f"Threshold: ${ADDV_THRESHOLD:,}")
    print(f"Source: {SOURCE_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)

    # Check if output already exists
    if OUTPUT_FILE.exists():
        print(f"\nFiltered file already exists: {OUTPUT_FILE}")
        print("Loading to show stats...")

        df_filtered = pd.read_parquet(OUTPUT_FILE)

        if isinstance(df_filtered.index, pd.MultiIndex):
            n_tickers = df_filtered.index.get_level_values('ticker').nunique()
            n_dates = df_filtered.index.get_level_values('date').nunique()
        else:
            n_tickers = df_filtered['ticker'].nunique()
            n_dates = df_filtered['date'].nunique()

        print(f"  Tickers: {n_tickers}")
        print(f"  Dates: {n_dates}")
        print(f"  Total rows: {len(df_filtered):,}")
        print("\nSkipping - file already exists.")
        return 0

    # Load source data
    print(f"\nLoading source data...")
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    df = pd.read_parquet(SOURCE_FILE)
    print(f"  Loaded: {len(df):,} rows")

    # Get unique tickers
    if isinstance(df.index, pd.MultiIndex):
        all_tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        all_tickers = df['ticker'].unique().tolist()

    print(f"  Total tickers: {len(all_tickers)}")

    # Check if we have Close and Volume in the data
    has_close = any(c.lower() in ['close', 'adj_close'] for c in df.columns)
    has_volume = any(c.lower() == 'volume' for c in df.columns)

    if has_close and has_volume:
        print("\nCalculating ADDV20 from parquet data...")
        addv20_df = calculate_addv20_from_parquet(df, REFERENCE_DATE, ADDV_LOOKBACK)
    else:
        print("\nClose/Volume not in parquet, fetching from Polygon API...")
        addv20_df = fetch_from_polygon(all_tickers, REFERENCE_DATE, ADDV_LOOKBACK)

    print(f"\nADDV20 calculated for {len(addv20_df)} tickers")

    # Filter by threshold
    valid_tickers = addv20_df[addv20_df['ADDV20'] >= ADDV_THRESHOLD]['ticker'].tolist()
    removed_tickers = addv20_df[addv20_df['ADDV20'] < ADDV_THRESHOLD]['ticker'].tolist()

    print(f"\nFiltering results:")
    print(f"  Tickers with ADDV20 >= ${ADDV_THRESHOLD/1e6:.1f}M: {len(valid_tickers)}")
    print(f"  Tickers removed (ADDV20 < ${ADDV_THRESHOLD/1e6:.1f}M): {len(removed_tickers)}")

    # Show ADDV20 distribution
    print(f"\nADDV20 distribution:")
    print(f"  Min:    ${addv20_df['ADDV20'].min():,.0f}")
    print(f"  25%:    ${addv20_df['ADDV20'].quantile(0.25):,.0f}")
    print(f"  Median: ${addv20_df['ADDV20'].quantile(0.50):,.0f}")
    print(f"  75%:    ${addv20_df['ADDV20'].quantile(0.75):,.0f}")
    print(f"  Max:    ${addv20_df['ADDV20'].max():,.0f}")

    # Filter the main dataframe
    print(f"\nFiltering main dataframe...")
    if isinstance(df.index, pd.MultiIndex):
        mask = df.index.get_level_values('ticker').isin(valid_tickers)
        df_filtered = df.loc[mask].copy()
    else:
        mask = df['ticker'].isin(valid_tickers)
        df_filtered = df.loc[mask].copy()

    print(f"  Original rows: {len(df):,}")
    print(f"  Filtered rows: {len(df_filtered):,}")
    print(f"  Removed: {len(df) - len(df_filtered):,} rows ({(1 - len(df_filtered)/len(df))*100:.1f}%)")

    # Save filtered data
    print(f"\nSaving to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(OUTPUT_FILE)

    # Also save ADDV20 values for reference
    addv20_file = OUTPUT_FILE.parent / "addv20_values_20260202.csv"
    addv20_df.to_csv(addv20_file, index=False)
    print(f"ADDV20 values saved to: {addv20_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("FILTER COMPLETE")
    print("=" * 70)
    if isinstance(df_filtered.index, pd.MultiIndex):
        n_tickers = df_filtered.index.get_level_values('ticker').nunique()
        n_dates = df_filtered.index.get_level_values('date').nunique()
    else:
        n_tickers = df_filtered['ticker'].nunique()
        n_dates = df_filtered['date'].nunique()

    print(f"  Final tickers: {n_tickers}")
    print(f"  Final dates: {n_dates}")
    print(f"  Final rows: {len(df_filtered):,}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
