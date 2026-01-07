#!/usr/bin/env python3
"""
Build a T+10 MultiIndex factor dataset from an existing MultiIndex factor export (allfac).

Why this exists:
- `data/factor_exports/allfac/polygon_factors_batch_*.parquet` does NOT contain raw OHLCV,
  but it DOES contain the legacy factors + Close. We can still derive the requested T+10
  factors by recombining existing factor columns and Close:
  - liquid_momentum: momentum_60d * turnover_proxy (using vol_ratio_20d)
  - obv_divergence: price-momentum vs OBV-momentum divergence proxy
  - ivol_20: rolling std of (stock_ret - spy_ret), computed from Close (requires SPY)
  - rsi_21: add regime context via 200d MA (computed from Close) by sign-inverting in bear regime
  - bollinger_squeeze: make directional via squeeze-flag * sign(20d return)

Outputs:
- Sharded parquet files named {prefix}_batch_XXXX.parquet under --output-dir
- A manifest.parquet mirroring the original manifest structure

Note:
- If input batches contain raw OHLC columns (High/Low), we will additionally compute:
  - rsrs_beta_18: rolling regression slope (High ~ Low) over 18 days per ticker
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


T10_OUTPUT_COLS = [
    "liquid_momentum",
    "obv_divergence",
    "ivol_20",
    "rsi_21",
    "trend_r2_60",
    "near_52w_high",
    "ret_skew_20d",
    "blowoff_ratio",
    "hist_vol_40d",
    "atr_ratio",
    "bollinger_squeeze",
    "vol_ratio_20d",
    "price_ma60_deviation",
    "Close",
    "target",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", type=str, default="data/factor_exports/allfac/manifest.parquet")
    p.add_argument("--input-dir", type=str, default="data/factor_exports/allfac")
    p.add_argument("--output-dir", type=str, default="data/factor_exports/factors_t10")
    p.add_argument("--prefix", type=str, default="factors")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--dropna-target", action="store_true", help="Drop rows with NaN target (train-ready)")
    p.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker for ivol_20 (default SPY)")
    p.add_argument("--benchmark-source", type=str, default="yfinance", choices=["yfinance", "none"],
                   help="Where to get benchmark Close if not present in batches")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        # standardize names
        names = [(n or "") for n in df.index.names]
        names = ["date" if str(n).lower() == "date" else ("ticker" if str(n).lower() in ("ticker", "symbol") else n) for n in names]
        df.index.names = names
        if df.index.names[:2] != ["date", "ticker"]:
            # attempt to reorder
            if "date" in df.index.names and "ticker" in df.index.names:
                df = df.reorder_levels(["date", "ticker"])
                df = df.sort_index()
        return df
    raise ValueError("expected MultiIndex (date, ticker)")


def _load_spy_returns_from_manifest(manifest: pd.DataFrame) -> Optional[pd.Series]:
    """Return a pd.Series indexed by date (normalized) with SPY daily returns."""
    closes: Dict[pd.Timestamp, float] = {}
    for f in manifest["file"].tolist():
        path = Path(str(f))
        if not path.exists():
            # try relative to repo root
            path = Path.cwd() / str(f)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["Close"])
        df = _ensure_multiindex(df)
        try:
            spy = df.xs("SPY", level="ticker", drop_level=False)
        except Exception:
            continue
        # consolidate
        dts = pd.to_datetime(spy.index.get_level_values("date")).normalize()
        for dt, v in zip(dts, spy["Close"].astype(float).values):
            if np.isfinite(v):
                closes[pd.Timestamp(dt)] = float(v)
    if not closes:
        return None
    s = pd.Series(closes).sort_index()
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def _infer_date_range_from_batches(batch_files: List[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer global min/max date by scanning a couple of batches (cheap)."""
    if not batch_files:
        raise ValueError("no batch_files")
    # Read just index (via Close column) from first and last batches and combine.
    mins: List[pd.Timestamp] = []
    maxs: List[pd.Timestamp] = []
    for p in [batch_files[0], batch_files[-1]]:
        df = pd.read_parquet(p, columns=["Close"])
        df = _ensure_multiindex(df)
        dts = pd.to_datetime(df.index.get_level_values("date"))
        mins.append(pd.Timestamp(dts.min()))
        maxs.append(pd.Timestamp(dts.max()))
    return min(mins), max(maxs)

def _fetch_benchmark_returns_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """Fetch daily benchmark returns via yfinance. Returns Series indexed by normalized date."""
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning(f"yfinance not available ({e}); ivol_20 will be 0")
        return None

    # yfinance end is exclusive-ish; extend buffer
    start_s = pd.Timestamp(start).date().isoformat()
    end_s = (pd.Timestamp(end) + pd.Timedelta(days=5)).date().isoformat()
    logger.info(f"Fetching benchmark {ticker} from yfinance: {start_s} -> {end_s}")

    data = yf.download(ticker, start=start_s, end=end_s, auto_adjust=False, progress=False)
    if data is None or len(data) == 0:
        logger.warning(f"yfinance returned empty for {ticker}")
        return None

    # Handle possible column MultiIndex or missing Close
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            close = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            close = data[("Adj Close", ticker)]
        else:
            close = data.xs("Close", axis=1, level=0, drop_level=False).iloc[:, 0]
    else:
        if "Close" in data.columns:
            close = data["Close"]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"]
        else:
            logger.warning(f"yfinance missing Close/Adj Close for {ticker}")
            return None

    close = pd.Series(close).astype(float)
    close.index = pd.to_datetime(close.index).normalize()
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return ret


def _compute_t10_from_allfac(df: pd.DataFrame, spy_ret_by_date: Optional[pd.Series], horizon: int, benchmark_ticker: str) -> pd.DataFrame:
    df = _ensure_multiindex(df)
    # Required base columns
    required = [
        "Close",
        "momentum_60d",
        "vol_ratio_20d",
        "obv_momentum_60d",
        "rsi_21",
        "bollinger_squeeze",
        "price_ma60_deviation",
        "trend_r2_60",
        "near_52w_high",
        "ret_skew_20d",
        "blowoff_ratio",
        "hist_vol_40d",
        "atr_ratio",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input missing columns: {missing}")

    out = pd.DataFrame(index=df.index)

    # --- liquid_momentum (proxy) ---
    vol_proxy = (1.0 + df["vol_ratio_20d"].astype(float)).clip(lower=0.0, upper=20.0)
    out["liquid_momentum"] = (df["momentum_60d"].astype(float) * vol_proxy).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- obv_divergence (proxy) ---
    # If OBV momentum lags price momentum, this is divergence.
    out["obv_divergence"] = (df["momentum_60d"].astype(float) - df["obv_momentum_60d"].astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- ivol_20 (Close-only, requires SPY) ---
    # ivol_20 = rolling std( stock_ret - spy_ret ) over 20 trading days
    dates = pd.to_datetime(df.index.get_level_values("date")).normalize()
    if spy_ret_by_date is None or spy_ret_by_date.empty:
        out["ivol_20"] = 0.0
    else:
        mkt = dates.map(spy_ret_by_date).astype(float)
        # stock daily ret
        stock_ret = df.groupby(level="ticker")["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        diff = (stock_ret - mkt).replace([np.inf, -np.inf], np.nan)
        out["ivol_20"] = diff.groupby(level="ticker").transform(lambda s: s.rolling(20, min_periods=10).std()).fillna(0.0)

    # --- rsi_21 with regime context ---
    # existing rsi_21 in this pipeline is already standardized ~[-1, 1].
    rsi = df["rsi_21"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    close = df["Close"].astype(float)
    ma200 = close.groupby(level="ticker").transform(lambda s: s.rolling(200, min_periods=60).mean())
    bull = (close > ma200).astype(float).fillna(0.0)
    out["rsi_21"] = (bull * rsi) + ((1.0 - bull) * (-rsi))

    # --- bollinger_squeeze directional ---
    # existing bollinger_squeeze is a bandwidth-like series (std20/ma20). Make it directional:
    bw = df["bollinger_squeeze"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q20 = bw.groupby(level="ticker").transform(lambda s: s.rolling(126, min_periods=30).quantile(0.20))
    squeeze_flag = (bw < q20).astype(float)
    dir20 = close.groupby(level="ticker").pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["bollinger_squeeze"] = squeeze_flag * np.sign(dir20)

    # passthrough
    out["trend_r2_60"] = df["trend_r2_60"].astype(float).fillna(0.0)
    out["near_52w_high"] = df["near_52w_high"].astype(float).fillna(0.0)
    out["ret_skew_20d"] = df["ret_skew_20d"].astype(float).fillna(0.0)
    out["blowoff_ratio"] = df["blowoff_ratio"].astype(float).fillna(0.0)
    out["hist_vol_40d"] = df["hist_vol_40d"].astype(float).fillna(0.0)
    out["atr_ratio"] = df["atr_ratio"].astype(float).fillna(0.0)
    out["vol_ratio_20d"] = df["vol_ratio_20d"].astype(float).fillna(0.0)
    out["price_ma60_deviation"] = df["price_ma60_deviation"].astype(float).fillna(0.0)
    out["Close"] = close

    # target T+{horizon}
    out["target"] = out.groupby(level="ticker")["Close"].pct_change(horizon).shift(-horizon)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    in_manifest_path = Path(args.input_manifest)
    symbols_by_batch: Dict[int, object] = {}
    manifest: Optional[pd.DataFrame] = None
    if in_manifest_path.exists():
        manifest = pd.read_parquet(in_manifest_path)
        if "batch_id" in manifest.columns and "symbols" in manifest.columns:
            for _, r in manifest.iterrows():
                try:
                    symbols_by_batch[int(r["batch_id"])] = r.get("symbols", None)
                except Exception:
                    continue
        else:
            logger.warning(f"manifest missing columns; will proceed without symbols: {in_manifest_path}")
    else:
        logger.warning(f"manifest not found; will proceed without symbols: {in_manifest_path}")

    input_dir = Path(args.input_dir)
    batch_files = sorted(input_dir.glob("polygon_factors_batch_*.parquet"))
    if not batch_files:
        raise RuntimeError(f"no input batches found under {input_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SPY return series from input batches (Close-only)...")
    # Build a pseudo-manifest DataFrame for SPY scan using actual file paths
    spy_scan = pd.DataFrame({"file": [str(p) for p in batch_files]})
    spy_ret = _load_spy_returns_from_manifest(spy_scan)
    if spy_ret is None:
        logger.warning(f"{args.benchmark} not found in batches; ivol_20 will be derived from {args.benchmark_source} if enabled")
        if args.benchmark_source == "yfinance":
            start, end = _infer_date_range_from_batches(batch_files)
            spy_ret = _fetch_benchmark_returns_yfinance(args.benchmark, start, end)
    else:
        logger.info(f"{args.benchmark} ret loaded from batches: {spy_ret.notna().sum()} days")

    out_manifest: List[Dict[str, object]] = []
    total_samples = 0

    for f in batch_files:
        # parse batch id from filename
        name = f.name
        try:
            batch_id = int(name.split("_")[-1].split(".")[0])
        except Exception:
            logger.warning(f"cannot parse batch id from {name}, skipping")
            continue

        logger.info(f"[batch {batch_id:04d}] reading {f}")
        df = pd.read_parquet(f)
        df = _ensure_multiindex(df)

        t10 = _compute_t10_from_allfac(df, spy_ret, horizon=int(args.horizon), benchmark_ticker=args.benchmark)
        t10 = t10[T10_OUTPUT_COLS]

        if args.dropna_target:
            before = len(t10)
            t10 = t10.dropna(subset=["target"])
            logger.info(f"[batch {batch_id:04d}] dropna target: {before} -> {len(t10)}")

        out_path = out_dir / f"{args.prefix}_batch_{batch_id:04d}.parquet"
        t10.to_parquet(out_path)

        batch_symbols = symbols_by_batch.get(batch_id, None)
        out_manifest.append(
            {
                "batch_id": batch_id,
                "file": str(out_path),
                "symbols": batch_symbols,
                "sample_count": int(len(t10)),
            }
        )
        total_samples += int(len(t10))

    manifest_out_path = out_dir / "manifest.parquet"
    pd.DataFrame(out_manifest).to_parquet(manifest_out_path, index=False)

    readme = out_dir / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            f"""T+10 factors built from allfac
===========================
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input manifest: {in_manifest_path.as_posix()}
Output dir: {out_dir.as_posix()}
Prefix: {args.prefix}
Horizon: {args.horizon}
Dropna target: {bool(args.dropna_target)}

Columns:
{', '.join(T10_OUTPUT_COLS)}
"""
        )

    logger.info(f"DONE: wrote {len(out_manifest)} batches, total_samples={total_samples:,}")
    logger.info(f"MANIFEST: {manifest_out_path}")


if __name__ == "__main__":
    main()


