#!/usr/bin/env python3
"""Add sector-relative features to the regime parquet.

Computes:
  sector_rel_momentum_10d  - stock raw 10d momentum minus sector median
  sector_rel_reversal_3d   - stock raw 3d reversal minus sector median
  sector_rel_rsi_14        - stock RSI(14) minus sector median RSI

Uses raw Close to recompute momentum/reversal (parquet features are already z-scored).
RSI is recomputed from Close as well.

Output: data/factor_exports/polygon_full_features_T5_v3_bull.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/factor_exports/polygon_full_features_T5_v2_regime.parquet")
SECTOR_CSV = Path("data/sector_mapping.csv")
OUTPUT = Path("data/factor_exports/polygon_full_features_T5_v3_bull.parquet")


def compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a close price series (per ticker)."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main():
    print("=" * 70)
    print("Add sector-relative features")
    print("=" * 70)

    # Load sector mapping
    print(f"\n[1] Loading sector mapping: {SECTOR_CSV}")
    sectors = pd.read_csv(SECTOR_CSV)
    print(f"  Tickers: {len(sectors)}")
    print(f"  Sectors: {sectors['sector'].nunique()}")
    print(f"  Unknown: {(sectors['sector'] == 'Unknown').sum()}")

    # Load data
    print(f"\n[2] Loading data: {INPUT}")
    df = pd.read_parquet(INPUT)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['date', 'ticker']).sort_index()
    print(f"  Shape: {df.shape}")

    # Map sectors
    print("\n[3] Mapping sectors...")
    sector_map = sectors.set_index('ticker')['sector'].to_dict()
    ticker_idx = df.index.get_level_values('ticker')
    df['_sector'] = ticker_idx.map(sector_map).fillna('Unknown')
    print(f"  Sector coverage: {(df['_sector'] != 'Unknown').sum()}/{len(df)} "
          f"({100*(df['_sector'] != 'Unknown').sum()/len(df):.1f}%)")

    # Recompute raw momentum and reversal from Close
    print("\n[4] Computing raw factors from Close...")
    close = df['Close']

    # Raw 10d momentum per ticker
    raw_mom_10d = close.groupby(level='ticker').pct_change(10)
    # Raw 3d reversal per ticker
    raw_rev_3d = close.groupby(level='ticker').pct_change(3) * -1
    # RSI(14) per ticker
    raw_rsi = close.groupby(level='ticker').transform(lambda x: compute_rsi(x, 14))

    print(f"  raw_mom_10d: non-null={raw_mom_10d.notna().sum()}")
    print(f"  raw_rev_3d: non-null={raw_rev_3d.notna().sum()}")
    print(f"  raw_rsi: non-null={raw_rsi.notna().sum()}")

    # Compute sector medians per date
    print("\n[5] Computing sector-relative features...")
    df['_raw_mom_10d'] = raw_mom_10d.values
    df['_raw_rev_3d'] = raw_rev_3d.values
    df['_raw_rsi'] = raw_rsi.values

    # Group by (date, sector) and compute median
    sector_med_mom = df.groupby([df.index.get_level_values('date'), '_sector'])['_raw_mom_10d'].transform('median')
    sector_med_rev = df.groupby([df.index.get_level_values('date'), '_sector'])['_raw_rev_3d'].transform('median')
    sector_med_rsi = df.groupby([df.index.get_level_values('date'), '_sector'])['_raw_rsi'].transform('median')

    df['sector_rel_momentum_10d'] = df['_raw_mom_10d'] - sector_med_mom
    df['sector_rel_reversal_3d'] = df['_raw_rev_3d'] - sector_med_rev
    df['sector_rel_rsi_14'] = df['_raw_rsi'] - sector_med_rsi

    print(f"  sector_rel_momentum_10d: non-null={df['sector_rel_momentum_10d'].notna().sum()}")
    print(f"  sector_rel_reversal_3d: non-null={df['sector_rel_reversal_3d'].notna().sum()}")
    print(f"  sector_rel_rsi_14: non-null={df['sector_rel_rsi_14'].notna().sum()}")

    # Clean up temp columns
    df.drop(columns=['_sector', '_raw_mom_10d', '_raw_rev_3d', '_raw_rsi'], inplace=True)

    # Save
    print(f"\n[6] Saving to {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)
    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nDone!")


if __name__ == '__main__':
    main()
