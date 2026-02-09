#!/usr/bin/env python3
"""Filter tickers based on ADDV20 (avg dollar volume over last 20 trading days ending at a target date)."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

TRADE_DIR = Path(r"D:/trade")
MICRO_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet"
RAW_FILE = TRADE_DIR / "data" / "raw_ohlcv" / "polygon_raw_ohlcv_2021_2026.parquet"
OUTPUT_FILE = MICRO_FILE.with_name(MICRO_FILE.stem + "_ADDV_FILTERED.parquet")
TARGET_DATE_DEFAULT = datetime(2026, 2, 2)
DEFAULT_THRESHOLD = 1_000_000
LOOKBACK_DAYS = 90


def _load_raw_cache(start_date: datetime, end_date: datetime) -> Tuple[Optional[pd.core.groupby.DataFrameGroupBy], Optional[pd.Timestamp]]:
    if not RAW_FILE.exists():
        return None, None
    df = pd.read_parquet(RAW_FILE).reset_index()
    df['date'] = pd.to_datetime(df['date'])
    filtered = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), ['date', 'ticker', 'Close', 'Volume']]
    if filtered.empty:
        return None, None
    return filtered.groupby('ticker', sort=False), filtered['date'].max()


def _fetch_polygon_series(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    try:
        import sys
        sys.path.insert(0, str(TRADE_DIR))
        from polygon_client import polygon_client
    except Exception as exc:
        print(f"  Polygon client unavailable ({exc}); cannot fetch {symbol}")
        return None
    try:
        df = polygon_client.get_historical_bars(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            'day',
            1,
        )
        if df.empty:
            return None
        df = df.reset_index()[['Date', 'Close', 'Volume']].rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as exc:
        print(f"  Polygon fetch failed for {symbol}: {exc}")
        return None


def _compute_addv(series: pd.DataFrame, target_date: datetime) -> Optional[float]:
    if series is None or series.empty:
        return None
    series = series[series['date'] <= target_date].sort_values('date')
    tail = series.tail(20)
    if len(tail) < 20:
        return None
    return float((tail['Close'] * tail['Volume']).mean())


def main():
    parser = argparse.ArgumentParser(description='Filter tickers by ADDV20 using Polygon data')
    parser.add_argument('--target-date', type=str, default=TARGET_DATE_DEFAULT.strftime('%Y-%m-%d'))
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--output', type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
    start_buffer = target_date - timedelta(days=LOOKBACK_DAYS)
    output_path = Path(args.output)

    micro_df = pd.read_parquet(MICRO_FILE)
    tickers = micro_df.index.get_level_values('ticker').unique().tolist()
    print(f"Tickers loaded: {len(tickers):,}")

    raw_groups, raw_max_date = _load_raw_cache(start_buffer, target_date)
    if raw_groups is None or raw_max_date is None or raw_max_date.to_pydatetime() < target_date:
        raw_groups = None
        print("Raw cache insufficient; will fetch required tickers from Polygon")

    addv_map: Dict[str, float] = {}
    missing = []

    for idx, ticker in enumerate(tickers, 1):
        if raw_groups is not None and ticker in raw_groups.groups:
            group_df = raw_groups.get_group(ticker)
        else:
            group_df = _fetch_polygon_series(ticker, start_buffer, target_date)
        addv = _compute_addv(group_df, target_date)
        if addv is None:
            missing.append(ticker)
        else:
            addv_map[ticker] = addv
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(tickers)} tickers")

    print(f"ADDV computed for {len(addv_map)} tickers; missing {len(missing)}")

    keep_tickers = [t for t, val in addv_map.items() if val >= args.threshold]
    print(f"Tickers passing threshold ({args.threshold:,.0f}): {len(keep_tickers)}")

    filtered_df = micro_df[micro_df.index.get_level_values('ticker').isin(keep_tickers)]
    filtered_df.to_parquet(output_path)
    print(f"Filtered rows: {len(filtered_df):,}; saved to {output_path}")


if __name__ == '__main__':
    main()
