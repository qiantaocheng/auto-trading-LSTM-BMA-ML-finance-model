#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute Polygon microstructure factors and merge them into an existing MultiIndex dataset."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

DEFAULT_POLYGON_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"
NEW_FACTOR_COLUMNS = [
    "avg_trade_size",
    "max_effect_21d",
    "gk_vol",
    "price_to_vwap5_dev",
    "intraday_intensity_10d",
]


def _get_polygon_key(explicit: Optional[str]) -> str:
    return explicit or os.environ.get("POLYGON_API_KEY") or DEFAULT_POLYGON_KEY


def _fetch_polygon_aggregates(
    ticker: str,
    start_date: str,
    end_date: str,
    session: requests.Session,
    api_key: str,
    rate_limit_delay: float = 0.25,
) -> pd.DataFrame:
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params: Dict[str, str] = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": api_key,
    }
    next_url: Optional[str] = None
    results: List[Dict] = []

    while True:
        url = next_url or base_url
        resp = session.get(url, params=None if next_url else params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results") or []
        results.extend(batch)
        next_url = data.get("next_url")
        time.sleep(rate_limit_delay)
        if not next_url:
            break

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    df["ticker"] = ticker
    keep_cols = {
        "date": "date",
        "ticker": "ticker",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "transactions",
    }
    missing = [src for src in keep_cols if src not in df.columns]
    if missing:
        raise RuntimeError(f"Polygon aggregates missing columns {missing} for {ticker}")
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)
    df = df.sort_values("date")
    return df


def _compute_microstructure_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    df["dollar_volume"] = df["volume"] * df["vwap"]
    df["avg_trade_size"] = df["dollar_volume"] / df["transactions"].replace({0: np.nan})

    ret = df["close"].pct_change()
    df["max_effect_21d"] = ret.rolling(window=21, min_periods=3).max()

    log_hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df["gk_vol"] = np.sqrt(gk_var.clip(lower=0))

    rolling_amt = df["dollar_volume"].rolling(window=5, min_periods=3).sum()
    rolling_vol = df["volume"].rolling(window=5, min_periods=3).sum()
    vwap5 = rolling_amt / rolling_vol.replace({0: np.nan})
    df["price_to_vwap5_dev"] = (df["close"] - vwap5) / vwap5

    price_range = df["high"] - df["low"]
    clv = (2 * df["close"] - df["high"] - df["low"]) / price_range.replace({0: np.nan})
    raw_intensity = clv * df["volume"]
    df["intraday_intensity_10d"] = raw_intensity.rolling(window=10, min_periods=5).mean()

    out = df[["date", "ticker"] + NEW_FACTOR_COLUMNS].dropna(how="all", subset=NEW_FACTOR_COLUMNS)
    out = out.set_index(["date", "ticker"])
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Polygon microstructure factors and merge with an existing T+5 parquet file.")
    parser.add_argument(
        "--base-data",
        type=str,
        default=r"D:\\trade\\data\\factor_exports\\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5.parquet",
        help="Path to the MultiIndex parquet file that already contains the T+5 factors.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Defaults to <base>_MICROSTRUCTURE.parquet if not provided.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Polygon API key (falls back to POLYGON_API_KEY env var or built-in default).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional subset of tickers to process (useful for testing).",
    )
    parser.add_argument(
        "--extra-lookback-days",
        type=int,
        default=40,
        help="Extra lookback days when fetching Polygon data to cover rolling windows (default: 40).",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Process at most this many tickers (debug helper).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("polygon_microstructure")

    base_path = Path(args.base_data)
    if not base_path.exists():
        raise FileNotFoundError(f"Could not find base data: {base_path}")

    api_key = _get_polygon_key(args.api_key)
    logger.info("Using Polygon API key length=%d", len(api_key))

    logger.info("Loading base dataset: %s", base_path)
    base_df = pd.read_parquet(base_path)
    if not isinstance(base_df.index, pd.MultiIndex) or "date" not in base_df.index.names:
        raise RuntimeError("Base parquet must have a MultiIndex with ('date','ticker').")

    base_index = base_df.index
    unique_dates = base_index.get_level_values("date")
    min_date = unique_dates.min()
    max_date = unique_dates.max()
    tickers = base_index.get_level_values("ticker").unique()

    if args.tickers:
        subset = [t.strip().upper() for t in args.tickers]
        tickers = pd.Index([t for t in tickers if t in subset])
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]

    start_date = (min_date - pd.Timedelta(days=args.extra_lookback_days)).date().isoformat()
    end_date = max_date.date().isoformat()
    logger.info("Date window for Polygon fetch: %s -> %s", start_date, end_date)
    logger.info("Tickers to process: %d", len(tickers))

    session = requests.Session()
    features: List[pd.DataFrame] = []
    for idx, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Fetching aggregates for %s", idx, len(tickers), ticker)
        try:
            agg = _fetch_polygon_aggregates(ticker, start_date, end_date, session, api_key)
        except Exception as exc:
            logger.error("Failed to download data for %s: %s", ticker, exc)
            continue

        if agg.empty:
            logger.warning("No Polygon aggregates for %s", ticker)
            continue

        feats = _compute_microstructure_factors(agg)
        if feats.empty:
            logger.warning("Computed factors are empty for %s", ticker)
            continue
        features.append(feats)

    if not features:
        raise RuntimeError("No factors were computed. Check Polygon API access or ticker list.")

    factor_df = pd.concat(features).sort_index()
    factor_df = factor_df[~factor_df.index.duplicated(keep="last")]

    logger.info("Computed factor shape: %s", factor_df.shape)

    aligned = factor_df.reindex(base_index)
    missing_counts = aligned[NEW_FACTOR_COLUMNS].isna().sum()
    logger.info("Missing rows after reindex (expected near the beginning due to rolling windows):\n%s", missing_counts)

    for col in NEW_FACTOR_COLUMNS:
        base_df[col] = aligned[col]

    output_path = Path(args.output) if args.output else base_path.with_name(base_path.stem + "_MICROSTRUCTURE.parquet")
    logger.info("Writing merged dataset to %s", output_path)
    base_df.to_parquet(output_path)
    logger.info("Done. New columns: %s", NEW_FACTOR_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
