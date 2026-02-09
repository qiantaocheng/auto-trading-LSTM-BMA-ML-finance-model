"""
Market Regime Feature Generator
Generates regime detection features using Polygon API (QQQ) and yfinance (VIX)
Date range: 2021-01-28 to 2026-01-09
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Install required packages if needed
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
    for a in client.list_aggs(
        "QQQ",
        1,
        "day",
        start_date,
        end_date,
        limit=50000
    ):
        aggs.append({
            'timestamp': a.timestamp,
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume,
            'vwap': a.vwap if hasattr(a, 'vwap') else None
        })

    df = pd.DataFrame(aggs)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df.drop(columns=['timestamp'])

    print(f"  Fetched {len(df)} days of QQQ data")
    return df


def fetch_vix_from_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch VIX data from yfinance"""
    print("Fetching VIX data from yfinance...")

    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    # Handle multi-level columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix = vix[['Close']].rename(columns={'Close': 'vix_close'})
    vix.index = pd.to_datetime(vix.index.date)
    vix.index.name = 'date'

    print(f"  Fetched {len(vix)} days of VIX data")
    return vix


def fetch_tlt_from_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch TLT (20-year Treasury ETF) data for risk premium calculation"""
    print("Fetching TLT data from yfinance...")

    tlt = yf.download("TLT", start=start_date, end=end_date, progress=False)

    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)

    tlt = tlt[['Close']].rename(columns={'Close': 'tlt_close'})
    tlt.index = pd.to_datetime(tlt.index.date)
    tlt.index.name = 'date'

    print(f"  Fetched {len(tlt)} days of TLT data")
    return tlt


def calculate_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Parkinson Volatility using High/Low prices
    Formula: σ = sqrt[(1/(4*n*ln2)) * Σ(ln(H/L))²]
    Returns annualized volatility
    """
    const = 1.0 / (4.0 * np.log(2.0))
    log_hl_sq = (np.log(df['high'] / df['low'])) ** 2
    parkinson_vol = np.sqrt(log_hl_sq.rolling(window=window).mean() * const)
    # Annualize
    parkinson_vol_annualized = parkinson_vol * np.sqrt(252)
    return parkinson_vol_annualized


def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Garman-Klass Volatility (more efficient estimator using OHLC)
    """
    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2

    gk_vol = np.sqrt(
        0.5 * log_hl.rolling(window=window).mean() -
        (2 * np.log(2) - 1) * log_co.rolling(window=window).mean()
    )
    # Annualize
    gk_vol_annualized = gk_vol * np.sqrt(252)
    return gk_vol_annualized


def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all regime detection features
    """
    print("\nEngineering regime features...")

    # =========================================
    # 1. VOLATILITY FEATURES (Fear Component)
    # =========================================

    # Parkinson Volatility (20-day window, annualized)
    df['parkinson_vol_20d'] = calculate_parkinson_volatility(df, window=20)

    # Garman-Klass Volatility (alternative measure)
    df['gk_vol_20d'] = calculate_garman_klass_volatility(df, window=20)

    # Volatility Z-Score (normalized relative to past year)
    roll_mean = df['parkinson_vol_20d'].rolling(252).mean()
    roll_std = df['parkinson_vol_20d'].rolling(252).std()
    df['vol_z_score'] = (df['parkinson_vol_20d'] - roll_mean) / roll_std

    # Rolling volatility percentile (adaptive threshold)
    df['vol_percentile_252d'] = df['parkinson_vol_20d'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Short-term vs Long-term volatility ratio (regime change signal)
    df['parkinson_vol_5d'] = calculate_parkinson_volatility(df, window=5)
    df['vol_ratio_5_20'] = df['parkinson_vol_5d'] / df['parkinson_vol_20d']

    # =========================================
    # 2. TREND FEATURES (Direction Component)
    # =========================================

    # SMA-based trend scores
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Trend Score: Distance from SMA200
    df['trend_score_200'] = (df['close'] - df['sma_200']) / df['sma_200']

    # Trend Score: Distance from SMA50
    df['trend_score_50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # Golden/Death Cross indicator (SMA50 vs SMA200)
    df['sma_cross_ratio'] = df['sma_50'] / df['sma_200']

    # Price momentum (returns)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)

    # =========================================
    # 3. DISCRETE REGIME ID (Categorical)
    # =========================================

    # Primary regime classification (3 states)
    conditions = [
        (df['trend_score_200'] > 0) & (df['vol_z_score'] < 0),   # Quiet Bull
        (df['trend_score_200'] > 0) & (df['vol_z_score'] >= 0),  # Volatile Bull
        (df['trend_score_200'] <= 0)                              # Bear/Correction
    ]
    df['regime_id'] = np.select(conditions, [0, 1, 2], default=2).astype(int)

    # Fine-grained regime (5 states)
    conditions_fine = [
        (df['trend_score_200'] > 0.05) & (df['vol_z_score'] < -0.5),   # Strong Quiet Bull
        (df['trend_score_200'] > 0) & (df['vol_z_score'] < 0),          # Quiet Bull
        (df['trend_score_200'] > 0) & (df['vol_z_score'] >= 0),         # Volatile Bull
        (df['trend_score_200'] > -0.05) & (df['trend_score_200'] <= 0), # Mild Bear
        (df['trend_score_200'] <= -0.05)                                 # Strong Bear
    ]
    df['regime_id_fine'] = np.select(conditions_fine, [0, 1, 2, 3, 4], default=4).astype(int)

    # =========================================
    # 4. SAMPLE WEIGHT FOR LAMBDARANK (IVW)
    # =========================================

    # Inverse Volatility Weight
    df['ivw_weight'] = 1.0 / df['parkinson_vol_20d']
    # Clip to prevent extreme weights
    lower_clip = df['ivw_weight'].quantile(0.05)
    upper_clip = df['ivw_weight'].quantile(0.95)
    df['ivw_weight'] = df['ivw_weight'].clip(lower=lower_clip, upper=upper_clip)
    # Normalize
    df['ivw_weight'] = df['ivw_weight'] / df['ivw_weight'].mean()

    print("  Volatility features: parkinson_vol_20d, gk_vol_20d, vol_z_score, vol_percentile_252d, vol_ratio_5_20")
    print("  Trend features: trend_score_200, trend_score_50, sma_cross_ratio, return_5d/10d/20d")
    print("  Regime IDs: regime_id (3-state), regime_id_fine (5-state)")
    print("  Sample weight: ivw_weight")

    return df


def add_external_features(df: pd.DataFrame, vix_df: pd.DataFrame, tlt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VIX and TLT-based features
    """
    print("\nAdding external features (VIX, TLT)...")

    # Merge VIX
    df = df.join(vix_df, how='left')
    df['vix_close'] = df['vix_close'].ffill()  # Forward fill for missing days

    # VIX features
    df['vix_sma_20'] = df['vix_close'].rolling(20).mean()
    df['vix_z_score'] = (df['vix_close'] - df['vix_sma_20']) / df['vix_close'].rolling(20).std()
    df['vix_percentile_252d'] = df['vix_close'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Merge TLT
    df = df.join(tlt_df, how='left')
    df['tlt_close'] = df['tlt_close'].ffill()

    # Risk Premium: QQQ/TLT ratio (higher = risk-on, lower = flight to safety)
    df['qqq_tlt_ratio'] = df['close'] / df['tlt_close']
    df['qqq_tlt_ratio_z'] = (
        (df['qqq_tlt_ratio'] - df['qqq_tlt_ratio'].rolling(60).mean()) /
        df['qqq_tlt_ratio'].rolling(60).std()
    )

    print("  VIX features: vix_close, vix_z_score, vix_percentile_252d")
    print("  Risk premium: qqq_tlt_ratio, qqq_tlt_ratio_z")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create explicit cross-product features for LambdaRank
    Factor × Regime interactions
    """
    print("\nCreating interaction features...")

    # Momentum × Volatility interactions
    df['mom_10d_x_vol_z'] = df['return_10d'] * df['vol_z_score']
    df['mom_20d_x_vol_z'] = df['return_20d'] * df['vol_z_score']

    # Trend × Volatility interactions
    df['trend_200_x_vol_z'] = df['trend_score_200'] * df['vol_z_score']
    df['trend_50_x_vol_z'] = df['trend_score_50'] * df['vol_z_score']

    # VIX interactions (if available)
    if 'vix_z_score' in df.columns:
        df['mom_10d_x_vix_z'] = df['return_10d'] * df['vix_z_score']
        df['trend_200_x_vix_z'] = df['trend_score_200'] * df['vix_z_score']

    # Risk premium interactions
    if 'qqq_tlt_ratio_z' in df.columns:
        df['mom_10d_x_risk_prem'] = df['return_10d'] * df['qqq_tlt_ratio_z']
        df['trend_200_x_risk_prem'] = df['trend_score_200'] * df['qqq_tlt_ratio_z']

    print("  Created momentum × volatility interactions")
    print("  Created trend × volatility interactions")
    print("  Created VIX and risk premium interactions")

    return df


def main():
    # Configuration
    API_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"
    START_DATE = "2021-01-28"
    END_DATE = "2026-01-09"
    OUTPUT_PATH = "D:/trade/data/regime_features.parquet"
    OUTPUT_CSV_PATH = "D:/trade/data/regime_features.csv"

    print("=" * 60)
    print("MARKET REGIME FEATURE GENERATOR")
    print("=" * 60)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    # Step 1: Fetch data
    qqq_df = fetch_qqq_from_polygon(API_KEY, START_DATE, END_DATE)
    vix_df = fetch_vix_from_yfinance(START_DATE, END_DATE)
    tlt_df = fetch_tlt_from_yfinance(START_DATE, END_DATE)

    # Step 2: Engineer regime features
    df = engineer_regime_features(qqq_df)

    # Step 3: Add external features
    df = add_external_features(df, vix_df, tlt_df)

    # Step 4: Create interaction features
    df = create_interaction_features(df)

    # Step 5: Clean up
    # Drop intermediate columns, keep only features
    cols_to_keep = [
        # Raw prices (for reference)
        'open', 'high', 'low', 'close', 'volume',
        # Volatility features
        'parkinson_vol_20d', 'gk_vol_20d', 'vol_z_score',
        'vol_percentile_252d', 'vol_ratio_5_20',
        # Trend features
        'trend_score_200', 'trend_score_50', 'sma_cross_ratio',
        'return_5d', 'return_10d', 'return_20d',
        # Regime IDs
        'regime_id', 'regime_id_fine',
        # Sample weight
        'ivw_weight',
        # External features
        'vix_close', 'vix_z_score', 'vix_percentile_252d',
        'qqq_tlt_ratio', 'qqq_tlt_ratio_z',
        # Interaction features
        'mom_10d_x_vol_z', 'mom_20d_x_vol_z',
        'trend_200_x_vol_z', 'trend_50_x_vol_z',
        'mom_10d_x_vix_z', 'trend_200_x_vix_z',
        'mom_10d_x_risk_prem', 'trend_200_x_risk_prem'
    ]

    # Filter to existing columns only
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]

    # Drop rows with NaN (due to rolling windows)
    initial_len = len(df)
    df = df.dropna()
    print(f"\nDropped {initial_len - len(df)} rows with NaN values (from rolling windows)")

    # Step 6: Save results
    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_parquet(OUTPUT_PATH)
    df.to_csv(OUTPUT_CSV_PATH)

    print("\n" + "=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Saved to: {OUTPUT_CSV_PATH}")

    # Display sample
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print("\nFeature columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")

    print("\n" + "-" * 60)
    print("Last 10 rows:")
    print(df.tail(10).to_string())

    print("\n" + "-" * 60)
    print("Regime distribution:")
    print(df['regime_id'].value_counts().sort_index())
    regime_labels = {0: 'Quiet Bull', 1: 'Volatile Bull', 2: 'Bear/Correction'}
    for rid, count in df['regime_id'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  {rid} ({regime_labels[rid]}): {count} days ({pct:.1f}%)")

    print("\n" + "-" * 60)
    print("Feature statistics:")
    print(df.describe().T.to_string())

    return df


if __name__ == "__main__":
    df = main()
