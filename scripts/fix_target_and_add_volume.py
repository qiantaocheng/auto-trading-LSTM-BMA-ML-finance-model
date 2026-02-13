#!/usr/bin/env python3
"""Fix target to use T+1 execution lag and add Volume for PIT filtering.

Reads the existing full-features parquet (factors are fine, only target changes)
and the raw OHLCV parquet to recompute target = Close[T+1+5] / Close[T+1] - 1.
Also joins Volume from raw data for downstream PIT universe filtering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 5

INPUT_FEATURES = Path("data/factor_exports/polygon_full_features_T5.parquet")
INPUT_OHLCV = Path("data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
OUTPUT = Path("data/factor_exports/polygon_full_features_T5_v2.parquet")


def main():
    print("=" * 70)
    print("Fix target (T+1 execution lag) + add Volume for PIT filtering")
    print("=" * 70)

    # Load existing features (factors are correct, only target needs fixing)
    print(f"\n[1] Loading features: {INPUT_FEATURES}")
    df = pd.read_parquet(INPUT_FEATURES)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['date', 'ticker']).sort_index()
    print(f"    Shape: {df.shape}, tickers: {df.index.get_level_values('ticker').nunique()}")
    print(f"    Old target stats: mean={df['target'].mean():.6f}, median={df['target'].median():.6f}")

    # Load raw OHLCV for Close prices (needed for accurate target recomputation)
    print(f"\n[2] Loading raw OHLCV: {INPUT_OHLCV}")
    raw = pd.read_parquet(INPUT_OHLCV, columns=['date', 'ticker', 'Close', 'Volume'])
    raw['date'] = pd.to_datetime(raw['date']).dt.tz_localize(None)
    raw['ticker'] = raw['ticker'].astype(str).str.upper().str.strip()
    raw = raw.sort_values(['ticker', 'date']).drop_duplicates(subset=['date', 'ticker'], keep='last')
    raw = raw.set_index(['date', 'ticker']).sort_index()
    print(f"    Raw shape: {raw.shape}")

    # Recompute target with T+1 execution lag
    # target = Close[T+1+HORIZON] / Close[T+1] - 1
    print(f"\n[3] Recomputing target with T+1 execution lag (horizon={HORIZON})")
    close = raw['Close']
    next_close = close.groupby(level='ticker').shift(-1)
    future_close = close.groupby(level='ticker').shift(-(1 + HORIZON))
    new_target = (future_close / next_close - 1).replace([np.inf, -np.inf], np.nan)

    # Also recompute excess return vs QQQ with same T+1 lag
    qqq_close = raw.loc[(slice(None), 'QQQ'), 'Close'] if 'QQQ' in raw.index.get_level_values('ticker') else None
    qqq_fwd = None
    if qqq_close is not None and len(qqq_close) > 0:
        qqq_close_flat = qqq_close.droplevel('ticker').sort_index()
        qqq_next = qqq_close_flat.shift(-1)
        qqq_future = qqq_close_flat.shift(-(1 + HORIZON))
        qqq_fwd = (qqq_future / qqq_next - 1).replace([np.inf, -np.inf], np.nan)
        print(f"    QQQ forward return computed: {qqq_fwd.notna().sum()} valid days")

    # Map new target back to the features index
    common_idx = df.index.intersection(new_target.index)
    print(f"    Common index size: {len(common_idx)} / {len(df)} ({100*len(common_idx)/len(df):.1f}%)")

    df['target'] = new_target.reindex(df.index)

    # Recompute excess return
    if qqq_fwd is not None:
        feature_dates = df.index.get_level_values('date')
        qqq_aligned = feature_dates.map(qqq_fwd)
        df['target_excess_qqq'] = df['target'] - qqq_aligned.values

    # Drop rows where target is NaN (training mode)
    before = len(df)
    df = df.dropna(subset=['target'])
    after = len(df)
    print(f"    Dropped {before - after} rows without target ({before} -> {after})")

    print(f"    New target stats: mean={df['target'].mean():.6f}, median={df['target'].median():.6f}")

    # Add Volume from raw data for PIT filtering
    print(f"\n[4] Adding Volume column from raw OHLCV")
    if 'Volume' not in df.columns:
        volume = raw['Volume'].reindex(df.index)
        df['Volume'] = volume
        print(f"    Volume coverage: {df['Volume'].notna().sum()} / {len(df)}")
    else:
        print(f"    Volume already present")

    # Save
    print(f"\n[5] Saving to {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)
    print(f"    Saved: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
