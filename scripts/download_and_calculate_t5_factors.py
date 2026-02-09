#!/usr/bin/env python
"""
Download raw OHLCV data from Polygon API and calculate all factor variations for T+5 prediction.

Factors to calculate:
- volume_price_corr: 5d, 10d (longer horizons to reduce noise)
- rsi: 7, 14
- reversal: 2d, 3d, 5d
- momentum: 3d, 5d, 10d
- liquid_momentum: 3d, 5d, 10d
- ivol: 5, 10
- vol_ratio: 5d, 10d
- trend_strength: 5d, 10d
- sharpe_momentum: 5d, 10d
- price_ma_deviation: 10, 20, 30

New factors (5):
- avg_trade_size (ATS)
- max_effect (MAX Effect - highest daily return in past N days)
- gk_volatility (Garman-Klass Volatility)
- price_vwap_deviation (Price-to-VWAP Deviation)
- intraday_intensity (Intraday Intensity Index)

Output: T+5 target with all micro factors
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADE_DIR = Path(r"D:\trade")
RAW_DATA_DIR = TRADE_DIR / "data" / "raw_ohlcv"
# Original paths (never overwritten by this script)
RAW_DATA_FILE = RAW_DATA_DIR / "polygon_raw_ohlcv_2021_2026.parquet"
OUTPUT_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet"
# Copy paths: script writes here so originals are preserved
RAW_DATA_FILE_COPY = RAW_DATA_DIR / "polygon_raw_ohlcv_2021_2026_copy.parquet"
OUTPUT_FILE_COPY = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO_copy.parquet"
IC_RESULTS_FILE = TRADE_DIR / "scripts" / "t5_micro_factor_ic_grid_results.csv"

# Ticker list: one ticker per line (used for download)
TICKER_LIST_FILE = TRADE_DIR / "data" / "all_tickers_list.txt"
# Reference multiindex file to get date range (same as main factor dataset)
REFERENCE_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"

# Import polygon client
sys.path.insert(0, str(TRADE_DIR))
try:
    from polygon_client import polygon_client
    HAS_POLYGON_CLIENT = True
except ImportError:
    HAS_POLYGON_CLIENT = False
    logger.warning("polygon_client not available, will use direct API calls")


def get_tickers_from_txt() -> List[str]:
    """Load ticker list from data/all_tickers_list.txt (one ticker per line)."""
    path = TICKER_LIST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Ticker list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    logger.info(f"Loaded {len(tickers)} tickers from {path}")
    return tickers


def get_dates_from_reference() -> tuple:
    """Get date range from reference multiindex file, or use default range if file missing."""
    if REFERENCE_FILE.exists():
        logger.info(f"Loading reference file for date range: {REFERENCE_FILE}")
        df = pd.read_parquet(REFERENCE_FILE)
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values("date").unique()
        else:
            dates = df["date"].unique()
        min_date = pd.to_datetime(dates.min())
        max_date = pd.to_datetime(dates.max())
        logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
        return min_date, max_date
    # Fallback: no reference file ?use fixed range (same structure as typical factor dataset)
    min_date = pd.to_datetime("2021-01-01")
    max_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    logger.info(f"No reference file; using default date range: {min_date.date()} to {max_date.date()}")
    return min_date, max_date


def download_ohlcv_from_polygon(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a single ticker from Polygon API."""
    try:
        if HAS_POLYGON_CLIENT:
            # Use existing polygon_client
            df = polygon_client.get_historical_bars(ticker, start_date, end_date, 'day', 1)
            if df.empty:
                return None

            # Reset index and add ticker
            df = df.reset_index()
            df['ticker'] = ticker
            df = df.rename(columns={
                'Date': 'date',
                'TradeCount': 'Transactions'
            })

            # Ensure required columns exist
            cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in ['VWAP', 'Transactions']:
                if col in df.columns:
                    cols.append(col)
                else:
                    df[col] = np.nan
                    cols.append(col)

            return df[cols]
        else:
            # Fallback to direct API call
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": os.environ.get("POLYGON_API_KEY", "")
            }

            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("resultsCount", 0) == 0:
                return None

            results = data.get("results", [])
            if not results:
                return None

            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df['ticker'] = ticker
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'vw': 'VWAP',
                'n': 'Transactions'
            })

            return df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Transactions']]

    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def _raw_data_path_for_read() -> Path:
    """Prefer copy so we never rely on overwriting original."""
    if RAW_DATA_FILE_COPY.exists():
        return RAW_DATA_FILE_COPY
    return RAW_DATA_FILE


def _factors_output_path_for_read() -> Path:
    """Prefer copy when loading factors."""
    if OUTPUT_FILE_COPY.exists():
        return OUTPUT_FILE_COPY
    return OUTPUT_FILE


def download_all_raw_data(tickers: List[str], start_date: str, end_date: str, force_redownload: bool = False) -> pd.DataFrame:
    """Download raw OHLCV data for all tickers and save to COPY file (original never overwritten)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    read_path = _raw_data_path_for_read()

    # Check if raw data (copy or original) already exists
    if read_path.exists() and not force_redownload:
        logger.info(f"Loading existing raw data: {read_path}")
        df = pd.read_parquet(read_path)
        logger.info(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    logger.info(f"Downloading raw OHLCV data for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")
    all_data = []
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(tickers)} tickers ({len(all_data)} successful)")

        df = download_ohlcv_from_polygon(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)

        # Rate limiting - polygon_client already has built-in delay
        if not HAS_POLYGON_CLIENT:
            time.sleep(0.15)  # 150ms delay for direct API calls

    if not all_data:
        raise ValueError("No data downloaded!")

    result = pd.concat(all_data, ignore_index=True)

    # Save raw data to COPY only (original preserved)
    result.to_parquet(RAW_DATA_FILE_COPY)
    logger.info(f"=" * 60)
    logger.info(f"RAW DATA DOWNLOAD COMPLETE")
    logger.info(f"Saved to: {RAW_DATA_FILE_COPY} (original not overwritten)")
    logger.info(f"Total rows: {len(result):,}")
    logger.info(f"Successful tickers: {result['ticker'].nunique()}")
    logger.info(f"Failed tickers: {len(failed_tickers)}")
    logger.info(f"=" * 60)

    if failed_tickers and len(failed_tickers) < 100:
        logger.info(f"Failed tickers: {failed_tickers[:50]}")

    return result


# =============================================================================
# FACTOR CALCULATION FUNCTIONS
# =============================================================================

def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate RSI with given period."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1)  # Avoid lookahead


def calc_reversal(close: pd.Series, period: int) -> pd.Series:
    """Calculate reversal (negative return over N days)."""
    ret = close.pct_change(period)
    return (-ret).shift(1)  # Negative return, shifted to avoid lookahead


def calc_momentum(close: pd.Series, period: int) -> pd.Series:
    """Calculate momentum (return over N days)."""
    ret = close.pct_change(period)
    return ret.shift(1)


def calc_liquid_momentum(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Calculate liquidity-weighted momentum."""
    ret = close.pct_change()
    dollar_vol = close * volume

    weighted_ret = (ret * dollar_vol).rolling(period, min_periods=period).sum()
    total_vol = dollar_vol.rolling(period, min_periods=period).sum()

    liq_mom = weighted_ret / total_vol.replace(0, np.nan)
    return liq_mom.shift(1)


def calc_ivol(close: pd.Series, period: int) -> pd.Series:
    """Calculate idiosyncratic volatility (rolling std of returns)."""
    ret = close.pct_change()
    ivol = ret.rolling(window=period, min_periods=period).std()
    return ivol.shift(1)


def calc_vol_ratio(volume: pd.Series, period: int) -> pd.Series:
    """Calculate volume ratio (current volume / N-day average)."""
    avg_vol = volume.rolling(window=period, min_periods=period).mean().shift(1)
    ratio = volume / avg_vol.replace(0, np.nan)
    return ratio.shift(1)


def calc_trend_strength(close: pd.Series, period: int) -> pd.Series:
    """Calculate trend strength (consistency of price direction)."""
    ret = close.pct_change()
    pos_days = (ret > 0).astype(float).rolling(period, min_periods=period).sum()
    trend_str = (pos_days / period - 0.5) * 2  # Normalize to [-1, 1]
    return trend_str.shift(1)


def calc_sharpe_momentum(close: pd.Series, period: int) -> pd.Series:
    """Calculate Sharpe-like momentum (mean return / std return)."""
    ret = close.pct_change()
    mean_ret = ret.rolling(period, min_periods=period).mean()
    std_ret = ret.rolling(period, min_periods=period).std()
    sharpe = mean_ret / std_ret.replace(0, np.nan)
    return sharpe.shift(1)


def calc_price_ma_deviation(close: pd.Series, period: int) -> pd.Series:
    """Calculate price deviation from N-day moving average."""
    ma = close.rolling(window=period, min_periods=period).mean()
    deviation = (close - ma) / ma.replace(0, np.nan)
    return deviation.shift(1)


def calc_volume_price_corr(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Calculate rolling correlation between price changes and volume."""
    ret = close.pct_change()
    corr = ret.rolling(window=period, min_periods=period).corr(volume)
    return corr.shift(1)


# =============================================================================
# NEW FACTORS (5)
# =============================================================================

def calc_avg_trade_size(volume: pd.Series, transactions: pd.Series, period: int = 20) -> pd.Series:
    """Average Trade Size (ATS) - Volume / Number of Transactions."""
    ats = volume / transactions.replace(0, np.nan)
    avg_ats = ats.rolling(window=period, min_periods=period).mean()
    return avg_ats.shift(1)


def calc_max_effect(close: pd.Series, period: int = 20) -> pd.Series:
    """MAX Effect - Maximum daily return in past N days."""
    ret = close.pct_change()
    max_ret = ret.rolling(window=period, min_periods=period).max()
    return max_ret.shift(1)


def calc_gk_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Garman-Klass Volatility - More efficient volatility estimator using OHLC."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2

    # GK formula: 0.5 * (H-L)^2 - (2*ln(2)-1) * (C-O)^2
    gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    gk_vol = np.sqrt(gk_daily.rolling(window=period, min_periods=period).mean() * 252)
    return gk_vol.shift(1)


def calc_price_vwap_deviation(close: pd.Series, vwap: pd.Series, period: int = 5) -> pd.Series:
    """Price-to-VWAP Deviation - How far price is from VWAP."""
    deviation = (close - vwap) / vwap.replace(0, np.nan)
    avg_dev = deviation.rolling(window=period, min_periods=period).mean()
    return avg_dev.shift(1)


def calc_intraday_intensity(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Intraday Intensity Index - Measures institutional buying/selling pressure."""
    # II = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    range_ = (high - low).replace(0, np.nan)
    ii = ((close - low) - (high - close)) / range_ * volume

    # Normalize by average volume
    avg_vol = volume.rolling(window=period, min_periods=period).mean()
    ii_norm = ii / avg_vol.replace(0, np.nan)

    return ii_norm.rolling(window=period, min_periods=period).mean().shift(1)


def calc_near_52w_high(close: pd.Series, period: int = 252) -> pd.Series:
    """Distance to 52-week high."""
    high_52w = close.rolling(window=period, min_periods=min(period, 60)).max()
    return (close / high_52w.replace(0, np.nan) - 1).shift(1)


def calc_trend_r2(close: pd.Series, period: int = 60) -> pd.Series:
    """Trend R-squared - How well price follows a linear trend."""
    def _r2(arr):
        if len(arr) < period or not np.all(np.isfinite(arr)):
            return np.nan
        x = np.arange(len(arr))
        try:
            corr = np.corrcoef(x, arr)[0, 1]
            return corr ** 2
        except:
            return np.nan

    r2 = close.rolling(window=period, min_periods=period).apply(_r2, raw=True)
    return r2.shift(1)


def calc_obv_divergence(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV Divergence - Divergence between OBV trend and price trend."""
    # OBV calculation
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()

    # Price and OBV momentum
    price_mom = close.pct_change(period)
    obv_mom = obv.pct_change(period)

    # Divergence = OBV momentum - Price momentum (normalized)
    divergence = obv_mom - price_mom
    return divergence.shift(1)


# =============================================================================
# MAIN CALCULATION
# =============================================================================

def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all factors for all tickers."""
    logger.info("Calculating all factors...")

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date'])

    # Group by ticker
    grouped = df.groupby('ticker')

    all_factors = []

    for ticker, group in grouped:
        if len(group) < 60:  # Need minimum history
            continue

        factors = {'date': group['date'].values, 'ticker': ticker}

        close = group['Close']
        volume = group['Volume']
        open_ = group['Open']
        high = group['High']
        low = group['Low']
        vwap = group.get('VWAP', close)  # Fallback to close if no VWAP
        transactions = group.get('Transactions', volume / 100)  # Estimate if missing

        # Volume-Price Correlation variations
        factors['volume_price_corr_5d'] = calc_volume_price_corr(close, volume, 5).values
        factors['volume_price_corr_10d'] = calc_volume_price_corr(close, volume, 10).values

        # RSI variations
        factors['rsi_14'] = calc_rsi(close, 14).values

        # Reversal variations
        factors['reversal_3d'] = calc_reversal(close, 3).values

        # Momentum variations
        for p in [10]:
            factors[f'momentum_{p}d'] = calc_momentum(close, p).values

        # Liquid momentum variations
        for p in [10]:
            factors[f'liquid_momentum_{p}d'] = calc_liquid_momentum(close, volume, p).values

        # Sharpe momentum variations
        factors['sharpe_momentum_5d'] = calc_sharpe_momentum(close, 5).values

        # Price MA deviation variations
        factors['price_ma20_deviation'] = calc_price_ma_deviation(close, 20).values

        # NEW FACTORS
        factors['avg_trade_size'] = calc_avg_trade_size(volume, transactions, 20).values

        # Standard factors
        factors['trend_r2_20'] = calc_trend_r2(close, 20).values
        factors['obv_divergence'] = calc_obv_divergence(close, volume, 20).values

        # Keep Close for target calculation
        factors['Close'] = close.values

        all_factors.append(pd.DataFrame(factors))

    result = pd.concat(all_factors, ignore_index=True)
    logger.info(f"Calculated factors for {result['ticker'].nunique()} tickers, {len(result)} rows")

    return result


def calculate_target_t5(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate T+5 target (5-day forward return)."""
    logger.info("Calculating T+5 target...")

    df = df.sort_values(['ticker', 'date'])

    # Forward return
    df['ret_5'] = df.groupby('ticker')['Close'].transform(
        lambda x: x.shift(-5) / x - 1
    )

    # TODO: Add market return for excess return calculation
    # For now, just use raw return
    df['target'] = df['ret_5']

    # Winsorize target
    lower = df['target'].quantile(0.01)
    upper = df['target'].quantile(0.99)
    df['target'] = df['target'].clip(lower, upper)

    return df


def calculate_ic_icir(df: pd.DataFrame, factor_cols: List[str], split_date: str = None) -> pd.DataFrame:
    """Calculate IC and ICIR for each factor, optionally split by train/test.

    Args:
        df: DataFrame with factor columns and 'target' column
        factor_cols: List of factor column names
        split_date: If provided, calculate IC separately for train (before) and test (after)
    """
    logger.info("Calculating IC and ICIR for all factors...")

    results = []

    # Convert date column if needed
    if 'date' not in df.columns and isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df['date'] = pd.to_datetime(df['date'])

    # Determine periods
    if split_date:
        split_dt = pd.to_datetime(split_date)
        train_mask = df['date'] < split_dt
        test_mask = df['date'] >= split_dt
        logger.info(f"Split date: {split_date}")
        logger.info(f"Train: {train_mask.sum():,} rows, Test: {test_mask.sum():,} rows")
    else:
        train_mask = pd.Series([True] * len(df))
        test_mask = pd.Series([False] * len(df))

    for factor in factor_cols:
        row = {'factor': factor}

        # Full period IC
        daily_ic = df.groupby('date').apply(
            lambda x: x[factor].corr(x['target']) if x[factor].notna().sum() > 10 else np.nan
        ).dropna()

        row['IC'] = daily_ic.mean()
        row['IC_std'] = daily_ic.std()
        row['ICIR'] = row['IC'] / row['IC_std'] if row['IC_std'] > 0 else 0

        # Rank IC (Spearman)
        daily_rank_ic = df.groupby('date').apply(
            lambda x: x[factor].corr(x['target'], method='spearman') if x[factor].notna().sum() > 10 else np.nan
        ).dropna()
        row['Rank_IC'] = daily_rank_ic.mean()

        # Train period IC
        if train_mask.any():
            train_df = df[train_mask]
            train_ic = train_df.groupby('date').apply(
                lambda x: x[factor].corr(x['target']) if x[factor].notna().sum() > 10 else np.nan
            ).dropna()
            row['Train_IC'] = train_ic.mean()
            row['Train_ICIR'] = train_ic.mean() / train_ic.std() if train_ic.std() > 0 else 0

        # Test period IC
        if test_mask.any() and split_date:
            test_df = df[test_mask]
            test_ic = test_df.groupby('date').apply(
                lambda x: x[factor].corr(x['target']) if x[factor].notna().sum() > 10 else np.nan
            ).dropna()
            row['Test_IC'] = test_ic.mean()
            row['Test_ICIR'] = test_ic.mean() / test_ic.std() if test_ic.std() > 0 else 0

            # IC stability (test vs train correlation)
            row['IC_Stability'] = row['Test_IC'] / row['Train_IC'] if abs(row['Train_IC']) > 0.001 else np.nan

        row['coverage'] = df[factor].notna().mean()
        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('ICIR', ascending=False)

    return result_df


def main(mode: str = "all", force_redownload: bool = False):
    """
    Main function with different modes:
    - "download_only": Just download raw OHLCV data
    - "factors_only": Calculate factors from existing raw data
    - "ic_only": Calculate IC/ICIR from existing factors
    - "all": Do everything
    """
    import argparse
    parser = argparse.ArgumentParser(description="T+5 Micro Factor Calculation")
    parser.add_argument("--mode", choices=["download_only", "factors_only", "ic_only", "all"],
                       default="all", help="Execution mode")
    parser.add_argument("--force-redownload", action="store_true",
                       help="Force re-download raw data even if cache exists")
    parser.add_argument("--split-date", type=str, default="2024-10-01",
                       help="Date to split train/test for IC calculation (YYYY-MM-DD)")
    args = parser.parse_args()

    mode = args.mode
    force_redownload = args.force_redownload
    split_date = args.split_date

    logger.info("=" * 70)
    logger.info("T+5 MICRO FACTOR CALCULATION")
    logger.info(f"Mode: {mode}")
    logger.info("=" * 70)

    # Get tickers from data/all_tickers_list.txt, date range from reference parquet
    tickers = get_tickers_from_txt()
    min_date, max_date = get_dates_from_reference()

    # Add buffer for lookback periods (300 days for 252-day rolling windows)
    start_date = (min_date - timedelta(days=300)).strftime("%Y-%m-%d")
    end_date = max_date.strftime("%Y-%m-%d")

    # ===== STEP 1: Download raw data =====
    if mode in ["download_only", "all"]:
        raw_df = download_all_raw_data(tickers, start_date, end_date, force_redownload)
        logger.info(f"Raw data shape: {raw_df.shape}")

        if mode == "download_only":
            logger.info("Download complete. Exiting.")
            return

    # ===== STEP 2: Load raw data if needed =====
    if mode in ["factors_only", "ic_only"]:
        read_path = _raw_data_path_for_read()
        if not read_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {read_path}. Run with --mode=download_only first.")
        raw_df = pd.read_parquet(read_path)
        logger.info(f"Loaded raw data: {raw_df.shape} from {read_path.name}")

    # ===== STEP 3: Calculate factors =====
    if mode in ["factors_only", "all"]:
        factors_df = calculate_all_factors(raw_df)
        logger.info(f"Factors calculated: {factors_df.shape}")

        # Calculate T+5 target
        factors_df = calculate_target_t5(factors_df)

        # Remove rows with missing target
        factors_df = factors_df.dropna(subset=['target'])
        logger.info(f"After removing missing targets: {factors_df.shape}")

        # Create MultiIndex and save to COPY only (original preserved)
        factors_df_indexed = factors_df.set_index(['date', 'ticker'])
        factors_df_indexed = factors_df_indexed.sort_index()

        factors_df_indexed.to_parquet(OUTPUT_FILE_COPY)
        logger.info(f"Saved factors to: {OUTPUT_FILE_COPY} (original not overwritten)")

        if mode == "factors_only":
            logger.info("Factor calculation complete. Exiting.")
            return

    # ===== STEP 4: Calculate IC/ICIR =====
    if mode in ["ic_only", "all"]:
        # Load factors if ic_only mode (from copy or original)
        if mode == "ic_only":
            factors_path = _factors_output_path_for_read()
            if not factors_path.exists():
                raise FileNotFoundError(f"Factors file not found: {factors_path}. Run with --mode=factors_only first.")
            factors_df = pd.read_parquet(factors_path)
            factors_df = factors_df.reset_index()
            logger.info(f"Loaded factors: {factors_df.shape} from {factors_path.name}")

        # Get factor columns (exclude metadata)
        exclude_cols = ['date', 'ticker', 'Close', 'target', 'ret_5']
        factor_cols = [c for c in factors_df.columns if c not in exclude_cols]
        logger.info(f"Analyzing {len(factor_cols)} factors")

        # Calculate IC/ICIR with train/test split
        ic_results = calculate_ic_icir(factors_df, factor_cols, split_date=split_date)

        print("\n" + "=" * 80)
        print("IC/ICIR RESULTS (sorted by ICIR)")
        print(f"Train/Test split: {split_date}")
        print("=" * 80)

        # Format output
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print(ic_results.to_string(index=False))

        # Save IC results
        ic_results.to_csv(IC_RESULTS_FILE, index=False)
        logger.info(f"Saved IC results to: {IC_RESULTS_FILE}")

        # Print summary
        print("\n" + "=" * 80)
        print("TOP 10 FACTORS BY TEST ICIR")
        print("=" * 80)
        if 'Test_ICIR' in ic_results.columns:
            top10 = ic_results.nlargest(10, 'Test_ICIR')[['factor', 'Train_IC', 'Test_IC', 'Train_ICIR', 'Test_ICIR', 'IC_Stability']]
            print(top10.to_string(index=False))

        print("\n" + "=" * 80)
        print("FACTORS WITH STABLE IC (|IC_Stability - 1| < 0.5 AND Train_IC sign == Test_IC sign)")
        print("=" * 80)
        if 'IC_Stability' in ic_results.columns:
            stable = ic_results[
                (ic_results['IC_Stability'].notna()) &
                (abs(ic_results['IC_Stability'] - 1) < 0.5) &
                (ic_results['Train_IC'] * ic_results['Test_IC'] > 0)  # Same sign
            ].sort_values('Test_ICIR', ascending=False)
            print(stable[['factor', 'Train_IC', 'Test_IC', 'Train_ICIR', 'Test_ICIR', 'IC_Stability']].to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


