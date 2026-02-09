#!/usr/bin/env python3
"""Generate a full-factor MultiIndex parquet with T+5 excess returns.

This script loads pre-downloaded Polygon OHLCV data, computes every factor
available in ``Simple25FactorEngine`` (excluding removed ebit/roa/sentiment
features), and writes a clean ``(date, ticker)`` MultiIndex parquet that
includes both the raw T+5 forward return target and the excess return vs QQQ.

Key properties:
    * Uses the exact factor formulas from ``bma_models.simple_25_factor_engine``
    * Runs in training mode by default (drops rows without future data)
    * Ensures all point-in-time operations (no data/time leakage)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bma_models.simple_25_factor_engine import Simple25FactorEngine  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw_ohlcv" / "polygon_raw_ohlcv_2021_2026.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "factor_exports" / "polygon_full_features_T5.parquet"


def _read_ticker_file(path: Path) -> List[str]:
    values: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            ticker = line.strip().upper()
            if ticker:
                values.append(ticker)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ticker file not found: {path}") from None
    return values


def _resolve_ticker_filter(args: argparse.Namespace) -> Optional[Set[str]]:
    tickers: Set[str] = set()
    if args.tickers:
        tickers.update(t.strip().upper() for t in args.tickers if t)
    if args.tickers_file:
        tickers.update(_read_ticker_file(Path(args.tickers_file)))
    return tickers or None


def _filter_dates(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        start_ts = pd.to_datetime(start).tz_localize(None)
        df = df[df['date'] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end).tz_localize(None)
        df = df[df['date'] <= end_ts]
    return df


def _limit_tickers(df: pd.DataFrame, allowed: Optional[Sequence[str]], max_tickers: Optional[int]) -> pd.DataFrame:
    if allowed:
        allowed_set = set(t.upper() for t in allowed)
        df = df[df['ticker'].isin(allowed_set)]
    if max_tickers is not None and max_tickers > 0:
        ordered = sorted(df['ticker'].unique())[:max_tickers]
        df = df[df['ticker'].isin(ordered)]
    return df


def load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input OHLCV parquet not found: {input_path}")

    logger = logging.getLogger(__name__)
    logger.info("Loading raw OHLCV data: %s", input_path)
    df = pd.read_parquet(input_path)

    required_cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()

    before_rows = len(df)
    df = _filter_dates(df, args.start_date, args.end_date)
    tickers_filter = _resolve_ticker_filter(args)
    df = _limit_tickers(df, tickers_filter, args.max_tickers)
    after_rows = len(df)

    if after_rows == 0:
        raise ValueError("No rows left after applying filters")

    logger.info("Filtered rows: %s -> %s", before_rows, after_rows)
    logger.info("Unique tickers: %d", df['ticker'].nunique())
    logger.info("Date range: %s to %s", df['date'].min().date(), df['date'].max().date())

    # Keep only columns that the factor engine may consume
    useful_cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    for extra in ['VWAP', 'Transactions']:
        if extra in df.columns:
            useful_cols.append(extra)

    df = df[useful_cols].sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


def build_engine(args: argparse.Namespace) -> Simple25FactorEngine:
    engine = Simple25FactorEngine(
        lookback_days=args.lookback_days,
        enable_sentiment=False,  # Ensure we skip optional sentiment callouts
        mode=args.mode,
        horizon=args.horizon,
        skip_cross_sectional_standardization=args.skip_cs_standardization,
    )
    return engine


def compute_features(engine: Simple25FactorEngine, market_data: pd.DataFrame, mode: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info(
        "Computing %d Simple25 factors + targets in %s mode (horizon=T+%d)",
        len(engine.alpha_factors),
        mode.upper(),
        engine.horizon,
    )
    features = engine.compute_all_17_factors(market_data, mode=mode)
    if features is None or len(features) == 0:
        raise RuntimeError("Factor computation returned empty DataFrame")

    if not isinstance(features.index, pd.MultiIndex):
        if {'date', 'ticker'}.issubset(features.columns):
            features = features.set_index(['date', 'ticker'])
        else:
            raise RuntimeError("Engine output missing MultiIndex and date/ticker columns")

    features = features.sort_index()
    logger.info("Feature grid ready: %s rows, %d columns", len(features), len(features.columns))
    if 'target_excess_qqq' in features.columns:
        valid = features['target_excess_qqq'].notna().sum()
        logger.info("Target_excess_qqq coverage: %d/%d rows", valid, len(features))
    return features


def _load_benchmark_series_from_file(path: Path,
                                     ticker: Optional[str],
                                     date_col: str,
                                     price_col: str) -> pd.Series:
    logger = logging.getLogger(__name__)
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    if path.suffix.lower() == '.parquet':
        bench = pd.read_parquet(path)
    else:
        bench = pd.read_csv(path)

    if 'ticker' in bench.columns and ticker:
        bench = bench[bench['ticker'].astype(str).str.upper() == ticker.upper()]
        if bench.empty:
            raise ValueError(f"Ticker {ticker} not found in benchmark file {path}")
    elif ticker and 'ticker' not in bench.columns:
        logger.warning("Benchmark file has no 'ticker' column; using full file for benchmark series")

    if date_col not in bench.columns or price_col not in bench.columns:
        raise ValueError(f"Benchmark file missing required columns: {date_col!r}, {price_col!r}")

    dates = pd.to_datetime(bench[date_col]).dt.tz_localize(None)
    price = pd.to_numeric(bench[price_col], errors='coerce')
    series = pd.Series(price.values, index=dates).sort_index()
    return series


def _compute_forward_from_series(series: pd.Series, horizon: int) -> pd.Series:
    price = series.sort_index().astype(float)
    return price.pct_change(horizon).shift(-horizon)


def ensure_excess_return(features: pd.DataFrame,
                         engine: Simple25FactorEngine,
                         market_data: pd.DataFrame,
                         args: argparse.Namespace) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    if 'target_excess_qqq' in features.columns:
        return features

    logger.warning("target_excess_qqq missing from engine output; attempting local benchmark computation")
    bench_series: Optional[pd.Series] = None

    if args.benchmark_file:
        bench_series = _load_benchmark_series_from_file(
            Path(args.benchmark_file),
            args.benchmark_ticker,
            args.benchmark_date_column,
            args.benchmark_price_column,
        )
    else:
        ticker = (args.benchmark_ticker or '').upper()
        if ticker and ticker in market_data['ticker'].unique():
            df = market_data[market_data['ticker'] == ticker]
            bench_series = pd.Series(df['Close'].values, index=df['date']).sort_index()

    if bench_series is None or bench_series.empty:
        logger.warning(
            "No benchmark data available; set POLYGON_API_KEY or pass --benchmark-file to compute excess returns."
        )
        return features

    forward = _compute_forward_from_series(bench_series, engine.horizon)
    feature_dates = pd.to_datetime(features.index.get_level_values('date')).tz_localize(None)
    feature_dates_norm = feature_dates.normalize()
    aligned = feature_dates_norm.map(forward)
    valid = pd.Series(aligned).notna().sum()
    if valid == 0:
        logger.warning("Benchmark forward series could not be aligned with feature dates; skipping target_excess_qqq")
        return features

    enriched = features.copy()
    enriched['target_excess_qqq'] = enriched['target'] - aligned
    logger.info(
        "target_excess_qqq created from benchmark data (%d/%d aligned rows)",
        valid,
        len(aligned),
    )
    return enriched


def save_multiindex(df: pd.DataFrame, output_path: Path, dry_run: bool) -> None:
    logger = logging.getLogger(__name__)
    if dry_run:
        logger.info("Dry run enabled; skipping save to %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info("MultiIndex parquet written to %s", output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Simple25 factor multiindex with T+5 excess returns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT), help='Input OHLCV parquet path')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT), help='Destination parquet path')
    parser.add_argument('--start-date', type=str, default=None, help='Optional inclusive start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='Optional inclusive end date (YYYY-MM-DD)')
    parser.add_argument('--tickers', nargs='+', help='Explicit list of tickers to include')
    parser.add_argument('--tickers-file', type=str, help='Path to newline-separated ticker list')
    parser.add_argument('--max-tickers', type=int, default=None, help='Limit to the first N tickers (after filtering)')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help='Factor engine mode')
    parser.add_argument('--lookback-days', type=int, default=252, help='Lookback window passed to the factor engine')
    parser.add_argument('--horizon', type=int, default=5, help='Forward horizon in days for the target label')
    parser.add_argument('--skip-cs-standardization', action='store_true', help='Skip cross-sectional standardization inside the engine')
    parser.add_argument('--benchmark-file', type=str, default=None,
                        help='Optional CSV/Parquet containing benchmark prices (e.g., QQQ) to build excess returns')
    parser.add_argument('--benchmark-ticker', type=str, default='QQQ',
                        help='Benchmark ticker symbol used for excess return calculation')
    parser.add_argument('--benchmark-date-column', type=str, default='date',
                        help='Date column name inside the benchmark file')
    parser.add_argument('--benchmark-price-column', type=str, default='Close',
                        help='Price column name inside the benchmark file')
    parser.add_argument('--dry-run', action='store_true', help='Run everything but skip writing the parquet')
    parser.add_argument('--log-level', default='INFO', help='Logging level (INFO, DEBUG, etc.)')
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting full factor generation: horizon=T+%d", args.horizon)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)

    start_time = time.time()
    market_data = load_market_data(args)
    engine = build_engine(args)
    features = compute_features(engine, market_data, args.mode)
    features = ensure_excess_return(features, engine, market_data, args)
    save_multiindex(features, Path(args.output), args.dry_run)
    elapsed = time.time() - start_time
    logger.info("Done. Total time: %.2f minutes", elapsed / 60.0)


if __name__ == '__main__':
    main()
