#!/usr/bin/env python3
"""Fetch sector/industry mapping for all tickers using yfinance.

Checkpoints every 100 tickers to data/sector_mapping_checkpoint.csv.
Final output: data/sector_mapping.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
PARQUET = Path("data/factor_exports/polygon_full_features_T5_v2_regime.parquet")
CHECKPOINT = DATA_DIR / "sector_mapping_checkpoint.csv"
OUTPUT = DATA_DIR / "sector_mapping.csv"
BATCH_SIZE = 100


def main():
    print("Loading unique tickers from parquet...")
    df = pd.read_parquet(PARQUET, columns=[])
    tickers = sorted(df.index.get_level_values('ticker').unique().tolist())
    print(f"  Total tickers: {len(tickers)}")

    # Resume from checkpoint if exists
    done = {}
    if CHECKPOINT.exists():
        cp = pd.read_csv(CHECKPOINT)
        done = {row['ticker']: row for _, row in cp.iterrows()}
        print(f"  Resuming from checkpoint: {len(done)} already fetched")

    remaining = [t for t in tickers if t not in done]
    print(f"  Remaining: {len(remaining)}")

    results = list(done.values())

    for i, ticker in enumerate(remaining):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            if sector is None:
                sector = 'Unknown'
            if industry is None:
                industry = 'Unknown'
        except Exception as e:
            sector = 'Unknown'
            industry = 'Unknown'

        results.append({'ticker': ticker, 'sector': sector, 'industry': industry})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(remaining)}] {ticker} -> {sector}/{industry}")

        # Checkpoint every BATCH_SIZE
        if (i + 1) % BATCH_SIZE == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
            print(f"  Checkpoint saved: {len(results)} tickers")

        # Small delay to avoid rate limiting
        time.sleep(0.15)

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT, index=False)
    print(f"\nDone! Saved {len(result_df)} tickers to {OUTPUT}")

    # Print sector distribution
    print("\nSector distribution:")
    for sector, count in result_df['sector'].value_counts().items():
        print(f"  {sector}: {count}")


if __name__ == '__main__':
    main()
