import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Reuse the existing Polygon client used across the project
try:
    from polygon_client import PolygonClient  # project root client
except Exception:  # pragma: no cover
    # Fallback to bma_models package client if available
    try:
        from bma_models.polygon_client import PolygonClient  # type: ignore
    except Exception:
        PolygonClient = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    top_n: int = 20
    horizon_days: int = 5
    benchmark_symbol: str = "SPY"  # SP500 proxy
    output_dir: str = os.path.join("D:", os.sep, "trade", "backtest_results")


def _get_polygon_client() -> Optional[PolygonClient]:
    """
    Create a Polygon client using environment variable POLYGON_API_KEY.
    Uses delayed data mode consistent with the project defaults.
    """
    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_APIKEY") or os.environ.get("POLYGON_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY is not set in environment; cannot run Excel backtest.")
        return None
    try:
        client = PolygonClient(api_key=api_key, delayed_data_mode=True)  # delayed mode is fine for backtests
        return client
    except Exception as e:
        logger.error(f"Failed to initialize PolygonClient: {e}")
        return None


def _parse_date(value) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).tz_localize(None).normalize()
    except Exception:
        return None


def _sanitize_ticker(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        s = str(raw).strip().upper()
        if not s:
            return None
        # Basic sanity: polygon tickers are alnum + .-
        return "".join(ch for ch in s if ch.isalnum() or ch in ".-")
    except Exception:
        return None


def _download_history(client: PolygonClient, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = client.get_historical_bars(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), timespan="day", multiplier=1)
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            df = df.sort_index()
            # Ensure tz-naive normalized index
            idx = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index = idx
        except Exception:
            pass
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _compute_t_horizon_return_from_target(client: PolygonClient, symbol: str, target_date: pd.Timestamp, horizon_days: int) -> Optional[float]:
    """
    Compute realized return using only target_date by walking back 'horizon_days' trading bars.
    Return = Close[target] / Close[target - horizon_days] - 1
    """
    # Fetch a buffer around the target to ensure indices exist
    start = (target_date - pd.Timedelta(days=30))
    end = (target_date + pd.Timedelta(days=2))
    hist = _download_history(client, symbol, start, end)
    if hist.empty:
        return None

    # Find the exact target bar (or the nearest previous trading day if exact not present)
    dates = hist.index
    pos = dates.searchsorted(target_date)
    if pos == len(dates) or dates[pos] != target_date:
        # Step back to the last available day <= target_date
        pos = max(0, dates.searchsorted(target_date, side="right") - 1)
    if pos < 0 or pos >= len(dates):
        return None

    base_pos = pos - horizon_days
    if base_pos < 0:
        return None

    try:
        base_close = float(hist.iloc[base_pos]["Close"])
        target_close = float(hist.iloc[pos]["Close"])
        if base_close <= 0 or not np.isfinite(base_close) or not np.isfinite(target_close):
            return None
        return (target_close / base_close) - 1.0
    except Exception:
        return None


def _select_top_n(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    df2 = df.copy()
    # Prefer explicit rank; else sort by final_score desc; else first N
    if "rank" in cols:
        c = cols["rank"]
        with pd.option_context('mode.use_inf_as_na', True):
            df2 = df2.sort_values(c, ascending=True)
    elif "final_score" in cols:
        c = cols["final_score"]
        with pd.option_context('mode.use_inf_as_na', True):
            df2 = df2.sort_values(c, ascending=False)
    df2 = df2.head(top_n)
    return df2


def backtest_from_excel(input_path: str, cfg: Optional[BacktestConfig] = None) -> Tuple[str, pd.DataFrame]:
    """
    Read an Excel workbook with multiple sheets. For each sheet:
      - take the top N tickers (by rank asc or final_score desc),
      - compute realized T+H average return across those tickers,
      - compute SP500 proxy (SPY) average return for the same windows,
    Then write a summary Excel with one 'summary' sheet and per-sheet details.
    Returns: (output_path, summary_df)
    """
    if cfg is None:
        cfg = BacktestConfig()

    client = _get_polygon_client()
    if client is None:
        raise RuntimeError("Polygon client not available. Set POLYGON_API_KEY and retry.")

    # Load workbook
    book = pd.read_excel(input_path, sheet_name=None)
    if not book:
        raise ValueError("No sheets found in Excel file.")

    per_sheet_details: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict] = []

    for sheet_name, df in book.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols_map = {c.lower(): c for c in df.columns}
        if "ticker" not in cols_map and "symbol" not in cols_map:
            logger.warning(f"{sheet_name}: missing 'ticker' column, skipped.")
            continue
        tick_col = cols_map.get("ticker", cols_map.get("symbol"))
        date_col = cols_map.get("date")  # target date (from exported files it is T+H target)

        # Top N selection
        top_df = _select_top_n(df, cfg.top_n)
        # Drop duplicates, sanitize tickers
        top_df = top_df.copy()
        top_df["__ticker__"] = top_df[tick_col].map(_sanitize_ticker)
        top_df = top_df.dropna(subset=["__ticker__"])
        top_df = top_df.drop_duplicates(subset=["__ticker__"])

        # Decide dates per-row: if date column exists use it; otherwise use the mode of provided column or fail
        if date_col and date_col in top_df.columns:
            top_df["__target_date__"] = top_df[date_col].map(_parse_date)
        else:
            # Try to use the most frequent date in the whole sheet (common case: all rows share T+H date)
            if date_col and date_col in df.columns:
                candidates = df[date_col].dropna().map(_parse_date)
                if isinstance(candidates, pd.Series) and candidates.notna().any():
                    common = candidates.mode()
                    target_date = common.iloc[0] if len(common) > 0 else None
                else:
                    target_date = None
            else:
                target_date = None
            top_df["__target_date__"] = target_date

        # Compute realized returns per row
        realized: List[Optional[float]] = []
        bench: List[Optional[float]] = []
        for _, row in top_df.iterrows():
            symbol = row["__ticker__"]
            tdate = row["__target_date__"]
            if tdate is None:
                realized.append(None)
                bench.append(None)
                continue
            r = _compute_t_horizon_return_from_target(client, symbol, tdate, cfg.horizon_days)
            b = _compute_t_horizon_return_from_target(client, cfg.benchmark_symbol, tdate, cfg.horizon_days)
            realized.append(r)
            bench.append(b)

        top_df["realized_ret"] = realized
        top_df["benchmark_ret"] = bench
        # Convert to pct for readability
        top_df["realized_ret_pct"] = (top_df["realized_ret"] * 100.0).round(3)
        top_df["benchmark_ret_pct"] = (top_df["benchmark_ret"] * 100.0).round(3)
        top_df["alpha_pct"] = ( (top_df["realized_ret"] - top_df["benchmark_ret"]) * 100.0 ).round(3)

        # Aggregate
        valid_mask = top_df["realized_ret"].notna()
        n_ok = int(valid_mask.sum())
        avg_ret = float(top_df.loc[valid_mask, "realized_ret"].mean()) if n_ok > 0 else np.nan
        avg_bmk = float(top_df.loc[valid_mask, "benchmark_ret"].mean()) if n_ok > 0 else np.nan
        alpha = (avg_ret - avg_bmk) if np.isfinite(avg_ret) and np.isfinite(avg_bmk) else np.nan

        summary_rows.append({
            "sheet": sheet_name,
            "top_n": min(cfg.top_n, len(top_df)),
            "n_computed": n_ok,
            "avg_return_pct": None if pd.isna(avg_ret) else round(avg_ret * 100.0, 3),
            "avg_sp500_pct": None if pd.isna(avg_bmk) else round(avg_bmk * 100.0, 3),
            "alpha_pct": None if pd.isna(alpha) else round(alpha * 100.0, 3),
        })

        # Store details
        per_sheet_details[sheet_name] = top_df[[tick_col] +
                                               ([date_col] if date_col in (top_df.columns.tolist()) else []) +
                                               ["realized_ret_pct", "benchmark_ret_pct", "alpha_pct"]].rename(
            columns={tick_col: "ticker", date_col or "__date__": "date"}
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("alpha_pct", ascending=False)

    # Output path
    os.makedirs(cfg.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(cfg.output_dir, f"{base}_avg_return_backtest.xlsx")

    # Write workbook
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        for sheet, det in per_sheet_details.items():
            # Excel sheet name max length 31
            safe_name = sheet[:31] if sheet else "sheet"
            det.to_excel(writer, index=False, sheet_name=safe_name)

    logger.info(f"Excel backtest complete. Output: {out_path}")
    return out_path, summary_df


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python -m autotrader.excel_backtester <input_excel_path> [--top 20] [--h 5] [--bench SPY]")
        return 2
    input_path = argv[0]
    top_n = 20
    horizon = 5
    bench = "SPY"
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("--top", "-n") and i + 1 < len(argv):
            top_n = int(argv[i + 1]); i += 2; continue
        if arg in ("--h", "--horizon") and i + 1 < len(argv):
            horizon = int(argv[i + 1]); i += 2; continue
        if arg in ("--bench", "--benchmark") and i + 1 < len(argv):
            bench = str(argv[i + 1]).upper(); i += 2; continue
        i += 1
    cfg = BacktestConfig(top_n=top_n, horizon_days=horizon, benchmark_symbol=bench)
    try:
        out_path, summary_df = backtest_from_excel(input_path, cfg)
        print(f"Output written: {out_path}")
        print(summary_df.to_string(index=False))
        return 0
    except Exception as e:
        logger.error(f"Excel backtest failed: {e}")
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


