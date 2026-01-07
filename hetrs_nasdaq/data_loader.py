from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Symbols:
    qqq: str = "QQQ"
    us10y: str = "^TNX"
    vix: str = "^VIX"
    dxy: str = "DX-Y.NYB"


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is not None:
            out.index = out.index.tz_convert("UTC").tz_localize(None)
        else:
            out.index = out.index.tz_localize(None)
    return out


def _download_yf(symbol: str, start: str) -> pd.DataFrame:
    # Delayed import to keep module importable even if yfinance is not installed.
    import yfinance as yf  # type: ignore

    df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty data for symbol={symbol!r}")
    # yfinance can return MultiIndex columns even for a single ticker, e.g. (Price, Ticker).
    # Normalize to single-level columns: Open/High/Low/Close/Adj Close/Volume.
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the "Price" level, which holds OHLCV names.
        df.columns = df.columns.get_level_values(0)
    df = _to_utc_naive_index(df)
    df = df.sort_index()
    return df


def garman_klass_volatility(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Garman-Klass volatility estimator (daily).

    Formula:
      sigma^2 = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
      sigma = sqrt(max(sigma^2, 0))
    """
    hl = np.log(high / low)
    co = np.log(close / open_)
    var = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (co ** 2)
    var = var.clip(lower=0.0)
    return np.sqrt(var).rename("vol_gk")


def build_dataset(
    start: str = "2010-01-01",
    symbols: Symbols = Symbols(),
) -> pd.DataFrame:
    """
    Returns a clean DataFrame indexed by date with:
      Open, High, Low, Close, Volume, Returns, vol_gk, US10Y, VIX, DXY

    Notes on missing values:
    - Macro series are forward-filled onto QQQ trading days.
    - Leading NaNs are dropped (no backfill is used to avoid lookahead).
    """
    qqq = _download_yf(symbols.qqq, start=start)
    macro_us10y = _download_yf(symbols.us10y, start=start)[["Close"]].rename(
        columns={"Close": "us10y"}
    )
    macro_vix = _download_yf(symbols.vix, start=start)[["Close"]].rename(
        columns={"Close": "vix"}
    )
    macro_dxy = _download_yf(symbols.dxy, start=start)[["Close"]].rename(
        columns={"Close": "dxy"}
    )

    df = qqq[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["returns"] = df["Close"].pct_change()  # simple daily returns (decimal)
    df["vol_gk"] = garman_klass_volatility(
        high=df["High"], low=df["Low"], open_=df["Open"], close=df["Close"]
    )

    df = df.join(macro_us10y, how="left").join(macro_vix, how="left").join(macro_dxy, how="left")
    df[["us10y", "vix", "dxy"]] = df[["us10y", "vix", "dxy"]].ffill()

    # Remove leading rows that still have NaNs (e.g. first return)
    df = df.dropna(axis=0, how="any")

    # Provide prompt-friendly aliases (keep original cols for downstream code)
    df["Returns"] = df["returns"]
    df["Garman_Klass_Vol"] = df["vol_gk"]
    df["US10Y"] = df["us10y"]
    df["VIX"] = df["vix"]
    df["DXY"] = df["dxy"]

    # Final safety check
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Dataset still contains NaNs in columns: {bad}")

    return df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and clean QQQ + macro data (HETRS-NASDAQ).")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    p.add_argument(
        "--out",
        type=str,
        default="data/hetrs_nasdaq/qqq_macro.parquet",
        help="Output parquet path",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = build_dataset(start=args.start)

    out_path = args.out
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(
        f"[data_loader] saved: {out_path} rows={len(df)} "
        f"start={df.index.min().date()} end={df.index.max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


