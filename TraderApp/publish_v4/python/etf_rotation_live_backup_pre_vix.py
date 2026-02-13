#!/usr/bin/env python
"""ETF Rotation Live Signal — P2 2-Level Cap Strategy.

Computes current target ETF weights based on:
  1. SPY MA200 for 2-level risk cap (100/60/30)
  2. Weighted portfolio blended vol (7 ETFs, static target weights)
  3. Portfolio B strategic weights

Critical correctness guarantees:
  - asof_date = last COMPLETED trading day (never uses today's unclosed bar)
  - All bars validated: count >= minimum, last_date == asof_date or prior
  - Weights for vol = static PORTFOLIO target weights (matches backtest)
  - Extreme-return circuit breaker: if any ETF has |return| > 25% flag as anomaly
  - Structured decision log emitted to stderr for audit

Output: single JSON object on stdout.
Progress: JSON lines on stderr.

Usage:
    python etf_rotation_live.py --polygon-key YOUR_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # D:\trade
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BMA_DIR = ROOT / "bma_models"
if str(BMA_DIR) not in sys.path:
    sys.path.insert(0, str(BMA_DIR))


# ── Portfolio B strategic weights (static — used for vol calculation) ─────
PORTFOLIO = {
    "QQQ":  0.250,
    "USMV": 0.250,
    "QUAL": 0.200,
    "PDBC": 0.150,
    "DBA":  0.050,
    "COPX": 0.050,
    "URA":  0.050,
}

# ── P2 2-Level Cap parameters (from V4Config — matches backtest exactly) ──
TARGET_VOL = 0.12
MAX_LEVERAGE = 1.0
VOL_BLEND_SHORT = 20
VOL_BLEND_LONG = 60
VOL_BLEND_ALPHA = 0.7
VOL_FLOOR = 0.08
VOL_CAP = 0.40
MA200_SHALLOW_CAP = 0.60
MA200_DEEP_CAP = 0.30
MA200_DEEP_THRESHOLD = -0.05
MIN_CASH_PCT = 0.05
FETCH_DAYS = 420           # enough for MA200 + vol + buffer

# ── Data quality thresholds ──────────────────────────────────────────────
MIN_BARS_REQUIRED = VOL_BLEND_LONG + 10   # 70 bars minimum
MAX_STALE_DAYS = 5                         # bar can be at most 5 calendar days old
EXTREME_RETURN_THRESHOLD = 0.25           # |daily return| > 25% = anomaly flag

ET_ZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16  # 4 PM ET


def emit_progress(step: str, progress: int, detail: str = "") -> None:
    msg = json.dumps({"step": step, "progress": progress, "detail": detail})
    print(msg, file=sys.stderr, flush=True)


def emit_decision_log(data: dict) -> None:
    """Emit a structured audit-trail log entry to stderr."""
    msg = json.dumps({"_decision_log": True, **data})
    print(msg, file=sys.stderr, flush=True)


def _resolve_api_key(cli_key: str) -> str:
    if cli_key:
        return cli_key
    env_key = os.environ.get("POLYGON_API_KEY", "")
    if env_key:
        return env_key
    try:
        from api_config import POLYGON_API_KEY
        return POLYGON_API_KEY
    except Exception:
        return ""


def get_asof_date() -> str:
    """Return the last completed trading day as 'YYYY-MM-DD'.

    Rule: if current ET time is before 4 PM ET, use yesterday;
    otherwise use today. Then walk backwards past weekends.
    (Holiday handling: Polygon simply won't have bars for holidays,
    so the last bar date will be the last actual trading day.)
    """
    now_et = datetime.now(ET_ZONE)
    if now_et.hour < MARKET_CLOSE_HOUR:
        # Before market close — today's bar is not yet complete
        candidate = now_et.date() - timedelta(days=1)
    else:
        candidate = now_et.date()

    # Walk back past weekends (holidays are handled by Polygon not having bars)
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate -= timedelta(days=1)

    return candidate.isoformat()


def fetch_ticker_data(api_key: str, ticker: str, asof_date: str) -> pd.DataFrame:
    """Fetch daily bars for a single ticker up to asof_date (inclusive)."""
    from polygon_client import PolygonClient
    client = PolygonClient(api_key=api_key)

    end_date = asof_date
    start_date = (datetime.strptime(asof_date, "%Y-%m-%d") - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")

    df = client.get_historical_bars(ticker, start_date, end_date, "day", 1)
    return df


def validate_ticker_data(ticker: str, df: pd.DataFrame, asof_date: str) -> dict:
    """Validate a ticker's bar data. Returns a dict with validation results."""
    issues = []
    ok = True

    if df is None or df.empty:
        return {"ticker": ticker, "ok": False, "bars": 0, "last_date": None,
                "issues": ["no_data"]}

    bars = len(df)
    last_date = None

    # Get last bar date
    try:
        if hasattr(df.index, "date"):
            last_date = df.index[-1].strftime("%Y-%m-%d")
        elif "Date" in df.columns:
            last_date = str(df["Date"].iloc[-1])[:10]
        else:
            last_date = str(df.index[-1])[:10]
    except Exception:
        issues.append("cannot_determine_last_date")

    # Check minimum bars
    if bars < MIN_BARS_REQUIRED:
        issues.append(f"insufficient_bars:{bars}<{MIN_BARS_REQUIRED}")
        ok = False

    # Check staleness
    if last_date:
        asof_dt = datetime.strptime(asof_date, "%Y-%m-%d").date()
        last_dt = datetime.strptime(last_date[:10], "%Y-%m-%d").date()
        stale_days = (asof_dt - last_dt).days
        if stale_days > MAX_STALE_DAYS:
            issues.append(f"stale:{stale_days}d_old")
            ok = False

    # Check for extreme returns in last 5 days (anomaly detection)
    anomaly_flag = False
    try:
        closes = df["Close"].astype(float).iloc[-6:]
        daily_rets = closes.pct_change().dropna()
        extreme = daily_rets[daily_rets.abs() > EXTREME_RETURN_THRESHOLD]
        if len(extreme) > 0:
            anomaly_flag = True
            issues.append(f"extreme_return:{extreme.iloc[-1]:.1%}")
    except Exception:
        pass

    return {
        "ticker": ticker,
        "ok": ok,
        "bars": bars,
        "last_date": last_date,
        "anomaly": anomaly_flag,
        "issues": issues,
    }


def fetch_all_data(api_key: str, asof_date: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Fetch SPY + all 7 portfolio ETFs up to asof_date.

    Returns:
        spy_df: SPY daily bars
        etf_dfs: dict of {ticker: DataFrame}
        validation_report: dict with data quality info per ticker
    """
    emit_progress("Fetching SPY data", 10, f"asof={asof_date}")

    spy_df = fetch_ticker_data(api_key, "SPY", asof_date)
    spy_validation = validate_ticker_data("SPY", spy_df, asof_date)

    if not spy_validation["ok"]:
        raise RuntimeError(f"SPY data invalid: {spy_validation['issues']}")

    # Block rebalance entirely if SPY has anomalous data
    if spy_validation.get("anomaly"):
        raise RuntimeError("SPY data anomaly — rebalance skipped, retry next trading day")

    emit_progress("SPY data fetched", 20,
                  f"{spy_validation['bars']} bars, last={spy_validation['last_date']}")

    etf_dfs = {}
    validation_report = {"SPY": spy_validation}
    etf_tickers = list(PORTFOLIO.keys())

    for i, ticker in enumerate(etf_tickers):
        pct = 25 + int(25 * i / len(etf_tickers))
        emit_progress("Fetching ETF data", pct, f"{ticker} ({i+1}/{len(etf_tickers)})")
        try:
            df = fetch_ticker_data(api_key, ticker, asof_date)
            v = validate_ticker_data(ticker, df, asof_date)
            validation_report[ticker] = v

            if v["ok"]:
                if v["anomaly"]:
                    # BLOCK anomaly tickers — exclude from vol calc and set weight to 0
                    emit_progress("Anomaly BLOCKED", pct,
                                  f"{ticker}: extreme return detected — EXCLUDED from vol calc")
                else:
                    etf_dfs[ticker] = df
            else:
                emit_progress("Data warning", pct,
                              f"{ticker}: {v['issues']} — excluded from vol calc")
        except Exception as e:
            emit_progress("ETF error", pct, f"{ticker}: {e}")
            validation_report[ticker] = {"ticker": ticker, "ok": False, "issues": [str(e)]}

    bars_last_date = {t: v.get("last_date") for t, v in validation_report.items()}
    anomaly_tickers = [t for t, v in validation_report.items() if v.get("anomaly")]
    valid_count = sum(1 for t in etf_tickers if validation_report.get(t, {}).get("ok"))

    emit_progress("All data fetched", 50,
                  f"SPY + {valid_count}/{len(etf_tickers)} ETFs valid"
                  + (f" | ANOMALIES: {anomaly_tickers}" if anomaly_tickers else ""))

    return spy_df, etf_dfs, validation_report


def compute_portfolio_blended_vol(
    etf_dfs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    asof_date: str,
) -> tuple[float, dict]:
    """Blended portfolio vol using weighted portfolio returns.

    Uses STATIC TARGET WEIGHTS (not current holdings) — matches backtest.
    Returns (blended_vol, debug_info).
    """
    available = [t for t in weights if t in etf_dfs]
    default_debug = {"method": "default_fallback", "available_tickers": available}

    if not available:
        return 0.15, {**default_debug, "reason": "no_available_tickers"}

    # Build aligned close price DataFrame (use only rows up to asof_date)
    asof_dt = pd.Timestamp(asof_date)
    close_dfs = {}
    for ticker in available:
        df = etf_dfs[ticker]
        close = df["Close"].astype(float)
        # Filter to <= asof_date to prevent lookahead
        if hasattr(close.index, "normalize"):
            close = close[close.index.normalize() <= asof_dt]
        close_dfs[ticker] = close

    close_df = pd.DataFrame(close_dfs)
    close_df = close_df.dropna()

    if len(close_df) < VOL_BLEND_LONG:
        return 0.15, {**default_debug, "reason": f"insufficient_aligned_bars:{len(close_df)}"}

    # Log returns (close-to-close, unadjusted — same as backtest assumption)
    log_ret = np.log(close_df / close_df.shift(1)).dropna()

    if len(log_ret) < VOL_BLEND_LONG:
        return 0.15, {**default_debug, "reason": f"insufficient_log_ret:{len(log_ret)}"}

    # Renormalize STATIC target weights for available tickers
    w_arr = np.array([weights[t] for t in available])
    w_arr = w_arr / w_arr.sum()
    weight_map = dict(zip(available, w_arr.tolist()))

    # Weighted portfolio log returns
    port_lr = (log_ret[available] * w_arr).sum(axis=1)

    # Short and long windows (from end of series)
    short_w = port_lr.iloc[-VOL_BLEND_SHORT:]
    long_w = port_lr.iloc[-VOL_BLEND_LONG:]

    v_short = float(short_w.std() * np.sqrt(252)) if len(short_w) > 5 else 0.15
    v_long = float(long_w.std() * np.sqrt(252)) if len(long_w) > 10 else 0.15

    blended = VOL_BLEND_ALPHA * v_short + (1 - VOL_BLEND_ALPHA) * v_long
    result = max(VOL_FLOOR, min(blended, VOL_CAP))

    debug = {
        "method": "portfolio_weighted",
        "available_tickers": available,
        "weights_used": weight_map,
        "aligned_bars": len(close_df),
        "log_ret_bars": len(log_ret),
        "v_short_20d": round(v_short, 6),
        "v_long_60d": round(v_long, 6),
        "blended_raw": round(blended, 6),
        "blended_clamped": round(result, 6),
        "last_portfolio_daily_return": round(float(port_lr.iloc[-1]), 6),
        "realized_vol_5d": round(float(port_lr.iloc[-5:].std() * np.sqrt(252)), 6),
    }
    return result, debug


def compute_two_level_risk_cap(spy_price: float, ma200: float) -> float:
    """2-level MA200 risk cap: 1.0 / 0.60 / 0.30."""
    if np.isnan(ma200) or ma200 <= 0:
        return 1.0
    deviation = (spy_price - ma200) / ma200
    if deviation >= 0:
        return 1.0
    elif deviation > MA200_DEEP_THRESHOLD:
        return MA200_SHALLOW_CAP
    else:
        return MA200_DEEP_CAP


def compute_target_weights(
    spy_df: pd.DataFrame,
    etf_dfs: dict[str, pd.DataFrame],
    validation_report: dict,
    asof_date: str,
) -> dict:
    """Compute P2 2-Level Cap target weights for Portfolio B.

    Uses SPY for MA200 signal, weighted portfolio returns for vol.
    All calculations use only data <= asof_date (no lookahead).
    """
    asof_dt = pd.Timestamp(asof_date)

    # Filter SPY to <= asof_date
    close = spy_df["Close"].astype(float)
    if hasattr(close.index, "normalize"):
        close = close[close.index.normalize() <= asof_dt]

    if len(close) < 200:
        raise RuntimeError(f"Insufficient SPY bars for MA200: only {len(close)}")

    # MA200 from SPY (using only completed bars up to asof_date)
    ma200_series = close.rolling(200).mean()
    if pd.isna(ma200_series.iloc[-1]):
        raise RuntimeError(f"MA200 is NaN (need 200 bars, have {len(close)})")

    spy_price = float(close.iloc[-1])
    ma200 = float(ma200_series.iloc[-1])

    emit_progress("Computing signal", 55, f"SPY={spy_price:.2f} MA200={ma200:.2f}")

    # Portfolio-weighted blended vol (uses static PORTFOLIO weights)
    bvol, vol_debug = compute_portfolio_blended_vol(etf_dfs, PORTFOLIO, asof_date)
    emit_progress("Portfolio vol computed", 65,
                  f"blended_vol={bvol:.4f} (method={vol_debug['method']})")

    # Vol-target exposure
    vol_exposure = TARGET_VOL / bvol if bvol > 0 else 1.0
    vol_exposure = min(vol_exposure, MAX_LEVERAGE)

    # 2-level risk cap (from SPY vs MA200)
    risk_cap = compute_two_level_risk_cap(spy_price, ma200)
    spy_deviation = (spy_price - ma200) / ma200

    # Final exposure — exact backtest order:
    # step1: min(vol_scalar, max_leverage) → already done above
    # step2: min(exposure, risk_cap)
    # step3: min(exposure, 1 - min_cash_pct)
    # step4: clamp [0, 1]
    exposure = min(vol_exposure, risk_cap)
    exposure = min(exposure, 1.0 - MIN_CASH_PCT)
    exposure = max(0.0, min(1.0, exposure))

    emit_progress("Exposure computed", 70,
                  f"vol_target={vol_exposure:.4f} risk_cap={risk_cap:.2f} final={exposure:.4f}")

    # Anomaly tickers — set weight to 0 (blocked from trading)
    blocked_anomaly = [t for t, v in validation_report.items()
                       if v.get("anomaly") and t != "SPY"]

    # Apply strategic weights using final exposure
    etf_weights = {}
    for ticker, w in PORTFOLIO.items():
        if ticker in blocked_anomaly:
            etf_weights[ticker] = 0.0
        else:
            etf_weights[ticker] = round(w * exposure, 6)
    cash_weight = round(1.0 - sum(etf_weights.values()), 6)

    # Get latest price for each ETF from their last bar
    etf_prices = {}
    for ticker in PORTFOLIO:
        if ticker in etf_dfs and not etf_dfs[ticker].empty:
            etf_prices[ticker] = float(etf_dfs[ticker]["Close"].iloc[-1])

    # Anomaly summary
    anomaly_tickers = [t for t, v in validation_report.items() if v.get("anomaly")]
    bars_last_date = {t: v.get("last_date") for t, v in validation_report.items()}

    return {
        "asof_trading_day": asof_date,
        "exposure": round(exposure, 6),
        "risk_cap": risk_cap,
        "spy_deviation": round(spy_deviation, 6),
        "blended_vol": round(bvol, 6),
        "spy_price": round(spy_price, 2),
        "ma200": round(ma200, 2),
        "vol_target_exposure": round(vol_exposure, 6),
        "etf_weights": etf_weights,
        "cash_weight": cash_weight,
        "etf_prices": etf_prices,
        "vol_debug": vol_debug,
        "bars_last_date": bars_last_date,
        "anomaly_tickers": anomaly_tickers,
        "blocked_anomaly_tickers": blocked_anomaly,
        "has_data_anomaly": len(anomaly_tickers) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="ETF Rotation Live Signal")
    parser.add_argument("--polygon-key", type=str, default="")
    args = parser.parse_args()

    try:
        api_key = _resolve_api_key(args.polygon_key)
        if not api_key:
            print(json.dumps({"error": "No Polygon API key found"}))
            sys.exit(1)

        # Determine asof_date FIRST — all data fetched up to this date only
        asof_date = get_asof_date()
        emit_progress("Starting ETF rotation signal", 5,
                      f"P2 2-Level Cap | asof={asof_date}")

        # 1. Fetch SPY + all ETF data (validated, no lookahead)
        spy_df, etf_dfs, validation_report = fetch_all_data(api_key, asof_date)

        # 2. Compute target weights
        result = compute_target_weights(spy_df, etf_dfs, validation_report, asof_date)

        # 3. Emit decision audit log
        emit_decision_log({
            "asof_trading_day": result["asof_trading_day"],
            "bars_last_date_by_ticker": result["bars_last_date"],
            "portfolio_realized_vol_5d": result["vol_debug"].get("realized_vol_5d"),
            "portfolio_blended_vol": result["blended_vol"],
            "raw_vol_target_exposure": result["vol_target_exposure"],
            "risk_cap": result["risk_cap"],
            "spy_deviation": result["spy_deviation"],
            "final_exposure": result["exposure"],
            "anomaly_tickers": result["anomaly_tickers"],
        })

        emit_progress("Complete", 100,
                      f"exposure={result['exposure']:.2%} risk_cap={result['risk_cap']:.0%}"
                      + (" ⚠ ANOMALY" if result["has_data_anomaly"] else ""))

        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
