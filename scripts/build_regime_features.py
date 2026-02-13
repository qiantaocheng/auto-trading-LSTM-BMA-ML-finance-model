#!/usr/bin/env python3
"""Download SPY + VIX daily data and build regime features for LambdaRank fusion tests.

Regime features (broadcast to all tickers per date):
  regime_vix             - VIX close level
  regime_vix_20d_chg     - VIX % change over 20 trading days
  regime_spy_ma200_dev   - (SPY close / SPY MA200) - 1
  regime_spy_above_ma    - 1 if SPY > MA200 else 0
  regime_hvr             - SPY 20d realized vol / 60d realized vol
  regime_spy_mom_1m      - SPY return over last 21 trading days
  regime_spy_dd_20d      - SPY max drawdown over last 20 trading days

Interaction features (stock-level × regime):
  interact_mom_vix       - momentum_10d × regime_vix
  interact_mom_bull      - momentum_10d × regime_spy_above_ma

Output: data/factor_exports/polygon_full_features_T5_v2_regime.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_V2 = Path("data/factor_exports/polygon_full_features_T5_v2.parquet")
OUTPUT = Path("data/factor_exports/polygon_full_features_T5_v2_regime.parquet")


def download_spy_vix(start: str = "2020-01-01", end: str = "2026-02-10") -> pd.DataFrame:
    """Download SPY and VIX daily close via yfinance."""
    import yfinance as yf

    print(f"Downloading SPY + ^VIX from yfinance ({start} to {end})...")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    regime = pd.DataFrame(index=spy.index)
    regime['spy_close'] = spy['Close']
    regime['vix_close'] = vix['Close'].reindex(spy.index, method='ffill')
    regime.index = pd.to_datetime(regime.index).tz_localize(None)
    regime = regime.sort_index().dropna()
    print(f"  SPY rows: {len(spy)}, VIX rows: {len(vix)}, merged: {len(regime)}")
    return regime


def compute_regime_features(regime: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 regime features from SPY + VIX daily data."""
    spy = regime['spy_close']
    vix = regime['vix_close']

    rf = pd.DataFrame(index=regime.index)

    # 1. VIX level
    rf['regime_vix'] = vix

    # 2. VIX 20-day % change
    rf['regime_vix_20d_chg'] = vix.pct_change(20)

    # 3. SPY deviation from MA200
    ma200 = spy.rolling(200, min_periods=200).mean()
    rf['regime_spy_ma200_dev'] = (spy / ma200) - 1

    # 4. SPY above MA200 (binary)
    rf['regime_spy_above_ma'] = (spy > ma200).astype(float)

    # 5. Historical volatility ratio (20d / 60d)
    log_ret = np.log(spy / spy.shift(1))
    vol_20 = log_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_60 = log_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
    rf['regime_hvr'] = vol_20 / vol_60

    # 6. SPY 1-month momentum (21 trading days)
    rf['regime_spy_mom_1m'] = spy.pct_change(21)

    # 7. SPY 20-day max drawdown
    roll_max = spy.rolling(20, min_periods=20).max()
    rf['regime_spy_dd_20d'] = (spy - roll_max) / roll_max

    print(f"  Regime features computed: {list(rf.columns)}")
    print(f"  Date range: {rf.index[0]} to {rf.index[-1]}")
    print(f"  NaN rows (any): {rf.isna().any(axis=1).sum()} / {len(rf)}")
    return rf


def main():
    print("=" * 70)
    print("Build regime features for LambdaRank fusion tests")
    print("=" * 70)

    # Step 1: Download SPY + VIX
    print("\n[1] Downloading SPY + VIX data...")
    regime_raw = download_spy_vix()

    # Step 2: Compute regime features
    print("\n[2] Computing regime features...")
    regime_features = compute_regime_features(regime_raw)

    # Step 3: Load v2 parquet
    print(f"\n[3] Loading base data: {INPUT_V2}")
    df = pd.read_parquet(INPUT_V2)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['date', 'ticker']).sort_index()
    print(f"    Shape: {df.shape}")
    print(f"    Date range: {df.index.get_level_values('date').min()} to "
          f"{df.index.get_level_values('date').max()}")

    # Step 4: Merge regime features (broadcast to all tickers per date)
    print("\n[4] Merging regime features...")
    feature_dates = df.index.get_level_values('date')

    # Build date-only lookup (v2 parquet dates have 05:00:00 UTC offset)
    regime_by_date = regime_features.copy()
    regime_by_date.index = regime_by_date.index.normalize()  # strip time
    # Also try with the exact feature dates
    feature_dates_normalized = feature_dates.normalize()

    for col in regime_features.columns:
        lookup = regime_by_date[col].to_dict()
        mapped = feature_dates_normalized.map(lookup)
        df[col] = mapped.values
        coverage = (~pd.isna(df[col])).sum()
        print(f"    {col}: {coverage}/{len(df)} coverage ({100*coverage/len(df):.1f}%)")

    # Step 5: Compute interaction features
    print("\n[5] Computing interaction features...")
    if 'momentum_10d' in df.columns:
        df['interact_mom_vix'] = df['momentum_10d'] * df['regime_vix']
        df['interact_mom_bull'] = df['momentum_10d'] * df['regime_spy_above_ma']
        print(f"    interact_mom_vix: non-null={df['interact_mom_vix'].notna().sum()}")
        print(f"    interact_mom_bull: non-null={df['interact_mom_bull'].notna().sum()}")
    else:
        print("    WARNING: momentum_10d not found, skipping interaction features")

    # Step 6: Save
    print(f"\n[6] Saving to {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)
    print(f"    Shape: {df.shape}")
    print(f"    Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nDone!")


if __name__ == '__main__':
    main()
