#!/usr/bin/env python
"""
Run CORE8 with filtered data:
1. Remove warrants (tickers ending with W patterns)
2. Market cap proxy: exclude stocks with close < $5
3. Liquidity: exclude low volume stocks (if volume data available)
"""

import subprocess
import sys
import os
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
DATA_FILE = r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"
TRADE_DIR = Path(r"D:\trade")
OUTPUT_BASE = TRADE_DIR / "results" / "core8_filtered"

# CORE8 factors
CORE8 = [
    "rsi_21",
    "trend_r2_60",
    "vol_ratio_20d",
    "near_52w_high",
    "obv_divergence",
    "liquid_momentum",
    "volume_price_corr_10d",
    "5_days_reversal"
]

# Warrant patterns - tickers ending with W, WS, or containing W at end after letters
WARRANT_PATTERNS = [
    r'^[A-Z]{2,4}W$',      # DJTWW, DWACW (2-4 letters + W)
    r'^[A-Z]{2,4}WS$',     # Warrants with WS suffix
    r'^[A-Z]+\.WS$',       # Warrants like XXX.WS
    r'^[A-Z]+\.W$',        # Warrants like XXX.W
]

# Known problematic tickers from analysis
BLACKLIST = {
    "DJTWW",  # Trump Media warrants - biggest drag
}

os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))


def is_warrant(ticker: str) -> bool:
    """Check if ticker is a warrant"""
    if ticker in BLACKLIST:
        return True
    for pattern in WARRANT_PATTERNS:
        if re.match(pattern, ticker):
            return True
    # Additional heuristic: if ends with 'WW' or 'W' after 3+ chars
    if len(ticker) >= 4 and ticker.endswith('W'):
        return True
    return False


def filter_data(df: pd.DataFrame, min_price: float = 5.0) -> pd.DataFrame:
    """Apply filters to data"""
    print("\n=== APPLYING FILTERS ===")

    original_rows = len(df)
    if isinstance(df.index, pd.MultiIndex):
        original_tickers = df.index.get_level_values('ticker').unique()
    else:
        original_tickers = df['ticker'].unique()
    print(f"Original: {len(original_tickers)} tickers, {original_rows} rows")

    # Filter 1: Remove warrants
    if isinstance(df.index, pd.MultiIndex):
        tickers = df.index.get_level_values('ticker')
        warrant_mask = tickers.map(is_warrant)
        warrants_removed = tickers[warrant_mask].unique().tolist()
        df = df[~warrant_mask]
    else:
        warrant_mask = df['ticker'].apply(is_warrant)
        warrants_removed = df[warrant_mask]['ticker'].unique().tolist()
        df = df[~warrant_mask]

    print(f"\nFilter 1 - Warrants removed ({len(warrants_removed)}): {warrants_removed[:20]}{'...' if len(warrants_removed) > 20 else ''}")

    # Filter 2: Price filter (proxy for market cap)
    if 'close' in df.columns:
        # For each ticker, check if median close is below threshold
        if isinstance(df.index, pd.MultiIndex):
            ticker_median_price = df.groupby(level='ticker')['close'].median()
            low_price_tickers = ticker_median_price[ticker_median_price < min_price].index.tolist()
            df = df[~df.index.get_level_values('ticker').isin(low_price_tickers)]
        else:
            ticker_median_price = df.groupby('ticker')['close'].median()
            low_price_tickers = ticker_median_price[ticker_median_price < min_price].index.tolist()
            df = df[~df['ticker'].isin(low_price_tickers)]
        print(f"\nFilter 2 - Low price (<${min_price}) tickers removed ({len(low_price_tickers)}): {low_price_tickers[:20]}{'...' if len(low_price_tickers) > 20 else ''}")

    # Filter 3: Liquidity (if volume data available)
    if 'volume' in df.columns:
        # Exclude tickers with very low average volume
        min_avg_volume = 100000  # 100K shares
        if isinstance(df.index, pd.MultiIndex):
            ticker_avg_vol = df.groupby(level='ticker')['volume'].mean()
            low_vol_tickers = ticker_avg_vol[ticker_avg_vol < min_avg_volume].index.tolist()
            df = df[~df.index.get_level_values('ticker').isin(low_vol_tickers)]
        else:
            ticker_avg_vol = df.groupby('ticker')['volume'].mean()
            low_vol_tickers = ticker_avg_vol[ticker_avg_vol < min_avg_volume].index.tolist()
            df = df[~df['ticker'].isin(low_vol_tickers)]
        print(f"\nFilter 3 - Low volume (<100K avg) tickers removed ({len(low_vol_tickers)}): {low_vol_tickers[:20]}{'...' if len(low_vol_tickers) > 20 else ''}")

    # Summary
    final_rows = len(df)
    if isinstance(df.index, pd.MultiIndex):
        final_tickers = df.index.get_level_values('ticker').unique()
    else:
        final_tickers = df['ticker'].unique()

    print(f"\n=== FILTER SUMMARY ===")
    print(f"Tickers: {len(original_tickers)} -> {len(final_tickers)} ({len(original_tickers) - len(final_tickers)} removed)")
    print(f"Rows: {original_rows} -> {final_rows} ({original_rows - final_rows} removed)")

    return df


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_BASE / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CORE8 WITH FILTERED DATA")
    print("="*70)
    print(f"Data: {DATA_FILE}")
    print(f"Factors: {CORE8}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Load and filter data
    print(f"\nLoading data: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    print(f"Columns: {list(df.columns)[:20]}...")

    # Apply filters
    df_filtered = filter_data(df, min_price=5.0)

    # Save filtered data
    filtered_file = output_dir / "filtered_data.parquet"
    df_filtered.to_parquet(filtered_file)
    print(f"\nSaved filtered data: {filtered_file}")

    # Run CORE8 evaluation
    run_dir = output_dir / "CORE8_filtered"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save feature list
    (run_dir / "features.json").write_text(json.dumps(CORE8, indent=2))

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--data-file", str(filtered_file),
        "--features"] + CORE8 + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--lambdarank-only",
        "--models", "lambdarank",
        "--top-n", "20",
        "--ema-top-n", "300",
        "--ema-length", "2",
        "--ema-beta", "0.7",
        "--ema-min-days", "2",
        "--output-dir", str(run_dir),
        "--log-level", "WARNING"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)
    env["BMA_TRAIN_ONLY_MODEL"] = "lambdarank"
    env["BMA_DISABLE_COMPULSORY_FEATURES"] = "1"

    print("\n" + "="*70)
    print("Running 80/20 time split (LambdaRank only)...")
    print("="*70)

    result = subprocess.run(cmd, env=env, cwd=str(TRADE_DIR), capture_output=True, text=True)

    # Collect metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for sub_dir in run_dir.glob("run_*"):
        lr_file = sub_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv"
        if lr_file.exists():
            lr_df = pd.read_csv(lr_file)
            print(f"\nTop5-15 Metrics:")
            print(f"  Mean: {lr_df['top_gross_return'].mean()*100:.2f}%")
            print(f"  Median: {lr_df['top_gross_return'].median()*100:.2f}%")
            print(f"  Accumulated: {lr_df.iloc[-1]['acc_return']*100:.1f}%")
            print(f"  MaxDD: {lr_df['drawdown'].min():.1f}%")
            print(f"  Win Rate: {(lr_df['top_gross_return'] > 0).mean()*100:.1f}%")
            print(f"  Periods: {len(lr_df)}")

        bucket_file = sub_dir / "lambdarank_bucket_returns.csv"
        if bucket_file.exists():
            bucket_df = pd.read_csv(bucket_file)
            print(f"\nBucket Returns:")
            print(bucket_df.to_string(index=False))

    if result.returncode != 0:
        print(f"\nWarnings/Errors: {result.stderr[-500:] if result.stderr else 'None'}")

    print(f"\n\nResults saved to: {output_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
