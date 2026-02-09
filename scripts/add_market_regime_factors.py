"""
Add Market Regime Factors to Existing Factor File
Adds qqq_tlt_ratio_z and vol_ratio_5_20 (market-level features from QQQ)
These are universal across all tickers for each day
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from polygon import RESTClient
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'polygon-api-client'])
    from polygon import RESTClient

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf


def fetch_qqq_from_polygon(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch QQQ OHLC data from Polygon API"""
    print("Fetching QQQ data from Polygon API...")
    client = RESTClient(api_key)

    aggs = []
    for a in client.list_aggs("QQQ", 1, "day", start_date, end_date, limit=50000):
        aggs.append({
            'timestamp': a.timestamp,
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume,
        })

    df = pd.DataFrame(aggs)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df = df.set_index('date').sort_index()
    df = df.drop(columns=['timestamp'])
    print(f"  Fetched {len(df)} days of QQQ data")
    return df


def fetch_tlt_from_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch TLT data from yfinance"""
    print("Fetching TLT data from yfinance...")
    tlt = yf.download("TLT", start=start_date, end=end_date, progress=False)

    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)

    tlt = tlt[['Close']].rename(columns={'Close': 'tlt_close'})
    tlt.index = pd.to_datetime(tlt.index).normalize()
    tlt.index.name = 'date'
    print(f"  Fetched {len(tlt)} days of TLT data")
    return tlt


def calculate_parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Parkinson Volatility using High/Low prices (annualized)"""
    const = 1.0 / (4.0 * np.log(2.0))
    log_hl_sq = (np.log(df['high'] / df['low'])) ** 2
    parkinson_vol = np.sqrt(log_hl_sq.rolling(window=window).mean() * const)
    return parkinson_vol * np.sqrt(252)  # Annualize


def compute_market_regime_factors(qqq_df: pd.DataFrame, tlt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market-level regime factors:
    1. vol_ratio_5_20: Short-term vs Long-term Parkinson volatility ratio (regime change signal)
    2. qqq_tlt_ratio_z: Z-scored QQQ/TLT ratio (risk premium indicator)
    """
    print("\nComputing market regime factors...")

    # Merge TLT data
    df = qqq_df.copy()
    df = df.join(tlt_df, how='left')
    df['tlt_close'] = df['tlt_close'].ffill()

    # 1. vol_ratio_5_20: Parkinson volatility ratio (5d / 20d)
    # Higher value = volatility spike = potential regime change
    parkinson_vol_5d = calculate_parkinson_volatility(df, window=5)
    parkinson_vol_20d = calculate_parkinson_volatility(df, window=20)
    df['vol_ratio_5_20'] = parkinson_vol_5d / parkinson_vol_20d

    # 2. qqq_tlt_ratio_z: Z-scored risk premium
    # QQQ/TLT ratio measures risk appetite (higher = risk-on, lower = flight to safety)
    df['qqq_tlt_ratio'] = df['close'] / df['tlt_close']
    roll_mean = df['qqq_tlt_ratio'].rolling(60).mean()
    roll_std = df['qqq_tlt_ratio'].rolling(60).std()
    df['qqq_tlt_ratio_z'] = (df['qqq_tlt_ratio'] - roll_mean) / roll_std

    # Keep only the two factors
    result = df[['vol_ratio_5_20', 'qqq_tlt_ratio_z']].copy()

    print(f"  vol_ratio_5_20 range: {result['vol_ratio_5_20'].min():.3f} to {result['vol_ratio_5_20'].max():.3f}")
    print(f"  qqq_tlt_ratio_z range: {result['qqq_tlt_ratio_z'].min():.3f} to {result['qqq_tlt_ratio_z'].max():.3f}")

    return result


def main():
    # Configuration
    API_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"
    INPUT_PATH = "D:/trade/data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"
    OUTPUT_PATH = "D:/trade/data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"

    print("=" * 70)
    print("ADD MARKET REGIME FACTORS TO FACTOR FILE")
    print("=" * 70)

    # Step 1: Load existing factor file
    print(f"\nLoading existing factor file: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Index: {df.index.names}")
    print(f"  Existing columns: {list(df.columns)}")

    # Get date range from existing data
    dates = df.index.get_level_values('date')
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')
    print(f"  Date range: {start_date} to {end_date}")

    # Check if factors already exist
    if 'vol_ratio_5_20' in df.columns and 'qqq_tlt_ratio_z' in df.columns:
        print("\n  Factors already exist in file. Recalculating...")
        df = df.drop(columns=['vol_ratio_5_20', 'qqq_tlt_ratio_z'])

    # Step 2: Fetch market data (need extra lookback for rolling windows)
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
    qqq_df = fetch_qqq_from_polygon(API_KEY, lookback_start, end_date)
    tlt_df = fetch_tlt_from_yfinance(lookback_start, end_date)

    # Step 3: Compute market regime factors
    market_factors = compute_market_regime_factors(qqq_df, tlt_df)

    # Step 4: Merge with existing data
    print("\nMerging market factors with existing data...")

    # Reset index to access date column
    df_reset = df.reset_index()

    # Normalize dates for proper join
    df_reset['date'] = pd.to_datetime(df_reset['date']).dt.normalize()
    market_factors.index = market_factors.index.normalize()

    # Join on date
    df_reset = df_reset.merge(
        market_factors.reset_index(),
        on='date',
        how='left'
    )

    # Check for missing values
    missing_vol = df_reset['vol_ratio_5_20'].isna().sum()
    missing_qqq = df_reset['qqq_tlt_ratio_z'].isna().sum()
    print(f"  Missing vol_ratio_5_20: {missing_vol} ({missing_vol/len(df_reset)*100:.2f}%)")
    print(f"  Missing qqq_tlt_ratio_z: {missing_qqq} ({missing_qqq/len(df_reset)*100:.2f}%)")

    # Fill missing with 1.0 for vol_ratio (neutral) and 0.0 for z-score
    df_reset['vol_ratio_5_20'] = df_reset['vol_ratio_5_20'].fillna(1.0)
    df_reset['qqq_tlt_ratio_z'] = df_reset['qqq_tlt_ratio_z'].fillna(0.0)

    # Restore MultiIndex
    df_reset = df_reset.set_index(['date', 'ticker'])
    df_reset = df_reset.sort_index()

    # Step 5: Save updated file
    print(f"\nSaving updated factor file: {OUTPUT_PATH}")
    print(f"  New shape: {df_reset.shape}")
    print(f"  New columns: {list(df_reset.columns)}")
    df_reset.to_parquet(OUTPUT_PATH)

    # Step 6: Verify
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    verify_df = pd.read_parquet(OUTPUT_PATH)
    print(f"  Shape: {verify_df.shape}")
    print(f"  Columns: {list(verify_df.columns)}")
    print("\n  New factor statistics:")
    print(verify_df[['vol_ratio_5_20', 'qqq_tlt_ratio_z']].describe())

    # Show sample
    print("\n  Sample (last 5 rows for first ticker):")
    first_ticker = verify_df.index.get_level_values('ticker')[0]
    sample = verify_df.loc[(slice(None), first_ticker), ['vol_ratio_5_20', 'qqq_tlt_ratio_z']].tail(5)
    print(sample)

    print("\n" + "=" * 70)
    print("DONE! Market regime factors added successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
