#!/usr/bin/env python
"""ETF Rotation Live Signal — T10C-Slim with optional HMM System 3.

Two modes:
  V7 mode (default): Uses C# V7 risk-level for portfolio selection + VIX trigger
  HMM Sys3 mode (--use-hmm): 3-state HMM drives portfolio selection + theme budget

T10C-Slim layers (shared between modes):
  - Vol-target: min(0.12 / blended_vol, 1.0)
  - MA200 2-level cap: 1.0 / 0.60 / 0.30
  - Asymmetric deadband 0.02/0.05 — handled by C# scheduler

HMM System 3 (Calmar 1.718, replaces VIX trigger):
  - 3-state GaussianHMM (SAFE/MID/CRISIS) trained on logret, vol_10d, vix_z, mom_z
  - p_risk = p_crisis + 0.5 * p_mid, EMA smoothed (span=4)
  - p_risk < 0.50 → risk-on portfolio (SMH), p_risk >= 0.50 → risk-off (GDX) + theme budget
  - p_risk >= 0.90 → cap exposure at 0.85

Risk-on portfolio: SMH 25%, USMV 25%, QUAL 20%, PDBC 15%, COPX 5%, URA 5%, DBA 5%
Risk-off portfolio: USMV 25%, QUAL 20%, GDX 20%, PDBC 15%, COPX 5%, URA 5%, DBA 10%

Output: single JSON object on stdout.
Progress: JSON lines on stderr.

Usage:
    python etf_rotation_live.py --polygon-key KEY --state-db PATH --use-hmm
    python etf_rotation_live.py --polygon-key KEY --state-db PATH --risk-level RISK_ON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import sqlite3
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # D:\trade
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BMA_DIR = ROOT / "bma_models"
if str(BMA_DIR) not in sys.path:
    sys.path.insert(0, str(BMA_DIR))


# ── T10C-Slim dual portfolio weights ──────────────────────────────────────
PORTFOLIO_RISK_ON = {
    "SMH":  0.250,
    "USMV": 0.250,
    "QUAL": 0.200,
    "PDBC": 0.150,
    "DBA":  0.050,
    "COPX": 0.050,
    "URA":  0.050,
}
PORTFOLIO_RISK_OFF = {
    "USMV": 0.250,
    "QUAL": 0.200,
    "GDX":  0.200,
    "PDBC": 0.150,
    "DBA":  0.100,
    "COPX": 0.050,
    "URA":  0.050,
}
# All unique tickers across both portfolios (for data fetching)
ALL_ETF_TICKERS = sorted(set(PORTFOLIO_RISK_ON) | set(PORTFOLIO_RISK_OFF))

THEME_TICKERS = ["COPX", "URA"]  # Theme legs subject to VIX budget control
DEFENSIVE_TICKERS = ["USMV", "QUAL"]  # Receive theme reallocation

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
HMM_FETCH_DAYS = 900       # HMM needs 252d rolling + 252d training → ~630 trading bars

# ── VIX Dynamic Trigger parameters ─────────────────────────────────────
VIX_PROXY = "^VIX"                    # VIX index (yfinance)
# Trigger thresholds
VIX_TRIGGER_ENABLE_THRESHOLD = 25.0   # SPY < MA200 AND VIX ≥ 25 → enable VIX
VIX_TRIGGER_DISABLE_THRESHOLD = 20.0  # SPY > MA200 AND VIX ≤ 20 → disable VIX
VIX_ENABLE_CONFIRM_DAYS = 2           # Require 2-day confirmation to enable
VIX_DISABLE_CONFIRM_DAYS = 5          # Require 5-day confirmation to disable

# Theme budget when VIX mode is active
THEME_BUDGET_NORMAL = 0.10            # VIX < 20: 10% combined
THEME_BUDGET_MEDIUM = 0.06            # 20 <= VIX < 25: 6% combined
THEME_BUDGET_HIGH = 0.02              # VIX >= 25: 2% combined

# VIX exposure cap (T10C-Slim L4)
VIX_EXPOSURE_CAP = 0.50               # When VIX mode active, cap total exposure at 50%

# ── HMM System 3 parameters ──────────────────────────────────────────────
HMM_N_STATES = 3
HMM_MIN_TRAIN = 252                       # Minimum bars for HMM training
HMM_EMA_SPAN = 4                          # EMA span for p_risk smoothing
HMM_MID_WEIGHT = 0.5                      # Weight for MID state in p_risk combo
SYS3_SWITCH_THRESHOLD = 0.50              # p_risk >= 0.50 → risk-off + theme budget
SYS3_REDUCE_THRESHOLD = 0.90              # p_risk >= 0.90 → cap exposure at 0.85
SYS3_REDUCE_CAP = 0.85                    # Exposure cap when p_risk >= 0.90

# ── Data quality thresholds ──────────────────────────────────────────────
MIN_BARS_REQUIRED = VOL_BLEND_LONG + 10   # 70 bars minimum
MAX_STALE_DAYS = 5                         # bar can be at most 5 calendar days old
EXTREME_RETURN_THRESHOLD = 0.35           # |daily return| > 35% = anomaly flag (commodity ETFs can move 20%+ in high-VIX)

ET_ZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16      # 4 PM ET
ASOF_CUTOFF_MINUTE = 15     # 4:15 PM ET — 15-min buffer for close data availability


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


# ── Database State Management ──────────────────────────────────────────

def get_db_state(db_path: str, key: str, default: str = "") -> str:
    """Read state value from etf_rotation_state table."""
    if not db_path or not os.path.exists(db_path):
        return default
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM etf_rotation_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default


def set_db_state(db_path: str, key: str, value: str) -> None:
    """Write state value to etf_rotation_state table."""
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO etf_rotation_state (key, value) VALUES (?, ?)",
                      (key, value))
        conn.commit()
        conn.close()
    except Exception as e:
        emit_progress("DB state error", 0, str(e))


def load_trigger_state(db_path: str) -> dict:
    """Load VIX trigger state from database."""
    mode = get_db_state(db_path, "vix_trigger_mode", "baseline")
    enable_count = int(get_db_state(db_path, "vix_trigger_enable_count", "0"))
    disable_count = int(get_db_state(db_path, "vix_trigger_disable_count", "0"))

    return {
        "mode": mode,
        "enable_count": enable_count,
        "disable_count": disable_count,
    }


def save_trigger_state(db_path: str, state: dict) -> None:
    """Save VIX trigger state to database."""
    set_db_state(db_path, "vix_trigger_mode", state["mode"])
    set_db_state(db_path, "vix_trigger_enable_count", str(state["enable_count"]))
    set_db_state(db_path, "vix_trigger_disable_count", str(state["disable_count"]))


# ── HMM p_risk Caching ───────────────────────────────────────────────

def load_cached_hmm_p_risk(db_path: str, asof_date: str) -> tuple[float | None, dict | None]:
    """Load cached HMM p_risk from DB if computed for the same asof_date.

    Returns (p_risk_smooth, debug_dict) or (None, None) if not cached.
    """
    cached_date = get_db_state(db_path, "hmm_cached_asof_date", "")
    if cached_date != asof_date:
        return None, None

    p_risk_str = get_db_state(db_path, "hmm_cached_p_risk_smooth", "")
    debug_str = get_db_state(db_path, "hmm_cached_debug", "")

    if not p_risk_str:
        return None, None

    try:
        p_risk = float(p_risk_str)
        debug = json.loads(debug_str) if debug_str else {}
        return p_risk, debug
    except (ValueError, json.JSONDecodeError):
        return None, None


def save_cached_hmm_p_risk(db_path: str, asof_date: str, p_risk_smooth: float, debug: dict) -> None:
    """Cache HMM p_risk to DB keyed by asof_date."""
    set_db_state(db_path, "hmm_cached_asof_date", asof_date)
    set_db_state(db_path, "hmm_cached_p_risk_smooth", f"{p_risk_smooth:.6f}")
    set_db_state(db_path, "hmm_cached_debug", json.dumps(debug))


# ── VIX Dynamic Trigger Logic ──────────────────────────────────────────

def evaluate_trigger_conditions(
    spy_price: float,
    ma200: float,
    vix_price: float | None,
    current_state: dict,
) -> tuple[dict, str]:
    """
    Evaluate VIX trigger state machine logic.

    Rules:
    - Enable VIX: SPY < MA200 AND VIX ≥ 25 (2-day confirmation)
    - Disable VIX: SPY > MA200 AND VIX ≤ 20 (5-day confirmation)
    - Reset opposite counter when conditions switch

    Args:
        spy_price: Current SPY price
        ma200: SPY 200-day moving average
        vix_price: Current VIX level (or None if no VIX data)
        current_state: Current trigger state dict

    Returns:
        (new_state, decision_reason)
    """
    mode = current_state["mode"]
    enable_count = current_state["enable_count"]
    disable_count = current_state["disable_count"]

    # If no VIX data, stay in current mode
    if vix_price is None:
        return current_state, "no_vix_data"

    # Check conditions
    spy_below_ma200 = spy_price < ma200
    spy_above_ma200 = spy_price > ma200
    vix_high = vix_price >= VIX_TRIGGER_ENABLE_THRESHOLD
    vix_low = vix_price <= VIX_TRIGGER_DISABLE_THRESHOLD

    # State machine logic
    if mode == "baseline":
        # Check enable conditions
        if spy_below_ma200 and vix_high:
            enable_count += 1
            disable_count = 0  # Reset opposite counter
            if enable_count >= VIX_ENABLE_CONFIRM_DAYS:
                mode = "vix_active"
                reason = f"TRIGGER ENABLED: SPY<MA200 + VIX≥{VIX_TRIGGER_ENABLE_THRESHOLD} for {enable_count} days"
            else:
                reason = f"enable_pending ({enable_count}/{VIX_ENABLE_CONFIRM_DAYS} days)"
        else:
            enable_count = 0  # Reset if conditions not met
            reason = f"baseline_mode (SPY vs MA200: {spy_price:.2f} vs {ma200:.2f}, VIX: {vix_price:.1f})"

    elif mode == "vix_active":
        # Check disable conditions
        if spy_above_ma200 and vix_low:
            disable_count += 1
            enable_count = 0  # Reset opposite counter
            if disable_count >= VIX_DISABLE_CONFIRM_DAYS:
                mode = "baseline"
                reason = f"TRIGGER DISABLED: SPY>MA200 + VIX≤{VIX_TRIGGER_DISABLE_THRESHOLD} for {disable_count} days"
            else:
                reason = f"disable_pending ({disable_count}/{VIX_DISABLE_CONFIRM_DAYS} days)"
        else:
            disable_count = 0  # Reset if conditions not met
            reason = f"vix_active_mode (SPY vs MA200: {spy_price:.2f} vs {ma200:.2f}, VIX: {vix_price:.1f})"
    else:
        reason = "unknown_mode"

    new_state = {
        "mode": mode,
        "enable_count": enable_count,
        "disable_count": disable_count,
    }

    return new_state, reason


# ── Data Fetching ──────────────────────────────────────────────────────

def get_asof_date() -> str:
    """Return the last completed trading day as 'YYYY-MM-DD'.

    Rule: if current ET time is before 4:15 PM ET, use yesterday;
    otherwise use today. 15-min buffer ensures market close data is available.
    Then walk backwards past weekends.
    (Holiday handling: Polygon simply won't have bars for holidays,
    so the last bar date will be the last actual trading day.)
    """
    now_et = datetime.now(ET_ZONE)
    cutoff_minutes = MARKET_CLOSE_HOUR * 60 + ASOF_CUTOFF_MINUTE  # 16:15 = 975 minutes
    current_minutes = now_et.hour * 60 + now_et.minute
    if current_minutes < cutoff_minutes:
        # Before 4:15 PM ET — today's bar is not yet complete/available
        candidate = now_et.date() - timedelta(days=1)
    else:
        candidate = now_et.date()

    # Walk back past weekends (holidays are handled by Polygon not having bars)
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate -= timedelta(days=1)

    return candidate.isoformat()


def fetch_ticker_data(api_key: str, ticker: str, asof_date: str) -> pd.DataFrame:
    """Fetch daily bars for a single ticker up to asof_date (inclusive)."""
    # Special handling for VIX index — use yfinance
    if ticker == "^VIX":
        try:
            import yfinance as yf
            vix = yf.Ticker(ticker)
            end_dt = datetime.strptime(asof_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=FETCH_DAYS)
            df = vix.history(start=start_dt.strftime("%Y-%m-%d"),
                           end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty:
                return pd.DataFrame()
            # Remove timezone to match Polygon data
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            emit_progress("VIX fetch error", 0, str(e))
            return pd.DataFrame()

    # All other tickers use Polygon
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


def fetch_all_data(api_key: str, asof_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Fetch SPY + VIX proxy + all ETFs from both portfolios up to asof_date.

    Fetches ALL unique tickers across risk-on and risk-off portfolios
    so data is always ready regardless of which portfolio is selected.

    Returns:
        spy_df: SPY daily bars
        vix_df: VIX proxy daily bars
        etf_dfs: dict of {ticker: DataFrame}
        validation_report: dict with data quality info per ticker
    """
    emit_progress("Fetching SPY data", 5, f"asof={asof_date}")

    spy_df = fetch_ticker_data(api_key, "SPY", asof_date)
    spy_validation = validate_ticker_data("SPY", spy_df, asof_date)

    if not spy_validation["ok"]:
        raise RuntimeError(f"SPY data invalid: {spy_validation['issues']}")

    if spy_validation.get("anomaly"):
        raise RuntimeError("SPY data anomaly — rebalance skipped, retry next trading day")

    emit_progress("SPY data fetched", 10,
                  f"{spy_validation['bars']} bars, last={spy_validation['last_date']}")

    # Fetch VIX proxy
    emit_progress("Fetching VIX proxy", 15, VIX_PROXY)
    vix_df = fetch_ticker_data(api_key, VIX_PROXY, asof_date)
    vix_validation = validate_ticker_data(VIX_PROXY, vix_df, asof_date)

    if not vix_validation["ok"]:
        emit_progress("VIX warning", 20, f"{VIX_PROXY}: {vix_validation['issues']} — VIX trigger DISABLED")
        vix_df = pd.DataFrame()  # Empty — will use baseline mode
    else:
        emit_progress("VIX data fetched", 20,
                      f"{vix_validation['bars']} bars, last={vix_validation['last_date']}")

    etf_dfs = {}
    validation_report = {"SPY": spy_validation, VIX_PROXY: vix_validation}
    etf_tickers = ALL_ETF_TICKERS  # All unique tickers from both portfolios

    for i, ticker in enumerate(etf_tickers):
        pct = 25 + int(30 * i / len(etf_tickers))
        emit_progress("Fetching ETF data", pct, f"{ticker} ({i+1}/{len(etf_tickers)})")
        try:
            df = fetch_ticker_data(api_key, ticker, asof_date)
            v = validate_ticker_data(ticker, df, asof_date)
            validation_report[ticker] = v

            if v["ok"]:
                if v["anomaly"]:
                    # BLOCK anomaly tickers — exclude from vol calc and set weight to 0
                    emit_progress("Anomaly BLOCKED", pct,
                                  f"{ticker}: extreme return detected — EXCLUDED")
                else:
                    etf_dfs[ticker] = df
            else:
                emit_progress("Data warning", pct,
                              f"{ticker}: {v['issues']} — excluded from vol calc")
        except Exception as e:
            emit_progress("ETF error", pct, f"{ticker}: {e}")
            validation_report[ticker] = {"ticker": ticker, "ok": False, "issues": [str(e)]}

    anomaly_tickers = [t for t, v in validation_report.items() if v.get("anomaly")]
    valid_count = sum(1 for t in etf_tickers if validation_report.get(t, {}).get("ok"))

    emit_progress("All data fetched", 55,
                  f"SPY + VIX + {valid_count}/{len(etf_tickers)} ETFs valid"
                  + (f" | ANOMALIES: {anomaly_tickers}" if anomaly_tickers else ""))

    return spy_df, vix_df, etf_dfs, validation_report


# ── Signal Computation ─────────────────────────────────────────────────

def compute_vix_theme_budget(vix_price: float | None) -> float:
    """
    Compute VIX-based theme budget for COPX+URA allocation.

    Only called when VIX mode is active.

    Returns:
        theme_budget: max allowed combined weight for COPX+URA
    """
    if vix_price is None:
        # No VIX data — default to NORMAL budget
        return THEME_BUDGET_NORMAL

    # Theme budget based on current VIX level
    if vix_price < VIX_TRIGGER_DISABLE_THRESHOLD:
        budget = THEME_BUDGET_NORMAL
    elif vix_price < VIX_TRIGGER_ENABLE_THRESHOLD:
        budget = THEME_BUDGET_MEDIUM
    else:
        budget = THEME_BUDGET_HIGH

    return budget


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


def apply_theme_budget(
    base_weights: dict[str, float],
    theme_budget: float,
) -> dict[str, float]:
    """
    Adjust portfolio weights to respect VIX theme budget cap.

    If theme legs exceed budget, scale them down and reallocate excess
    to defensive tickers (USMV/QUAL) proportionally.

    Args:
        base_weights: Initial strategic weights (normalized)
        theme_budget: Max allowed combined weight for theme tickers

    Returns:
        Adjusted weights (normalized to sum to 1.0)
    """
    weights = base_weights.copy()

    # Calculate current theme allocation
    theme_current = sum(weights.get(t, 0.0) for t in THEME_TICKERS)

    if theme_current <= theme_budget:
        # Within budget — no adjustment needed
        return weights

    # Excess to reallocate
    excess = theme_current - theme_budget

    # Scale down theme tickers proportionally to budget
    if theme_current > 0:
        scale = theme_budget / theme_current
        for ticker in THEME_TICKERS:
            if ticker in weights:
                weights[ticker] *= scale

    # Reallocate excess to defensive tickers proportionally
    defensive_current = sum(weights.get(t, 0.0) for t in DEFENSIVE_TICKERS)
    if defensive_current > 0:
        for ticker in DEFENSIVE_TICKERS:
            if ticker in weights:
                weights[ticker] += excess * (weights[ticker] / defensive_current)
    else:
        # Fallback: split equally
        for ticker in DEFENSIVE_TICKERS:
            if ticker in weights:
                weights[ticker] += excess / len(DEFENSIVE_TICKERS)

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


# ── HMM System 3 Functions ────────────────────────────────────────────

def prepare_hmm_features(spy_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    """Prepare 4-feature HMM input: logret, vol_10d, vix_z, mom_z.

    Matches the backtest exactly (t10c_full_comparison.py).
    """
    logret = np.log(spy_close / spy_close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()
    vix_mean = vix_close.rolling(252, min_periods=60).mean()
    vix_std = vix_close.rolling(252, min_periods=60).std()
    vix_z = (vix_close - vix_mean) / vix_std.replace(0, np.nan)
    spy_mom = spy_close.pct_change(20, fill_method=None)
    mom_mean = spy_mom.rolling(252, min_periods=60).mean()
    mom_std = spy_mom.rolling(252, min_periods=60).std()
    mom_z = (spy_mom - mom_mean) / mom_std.replace(0, np.nan)
    return pd.DataFrame({
        'logret': logret, 'vol_10d': vol_10d,
        'vix_z': vix_z, 'mom_z': mom_z,
    }, index=spy_close.index).dropna()


def compute_hmm_p_risk(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    asof_date: str,
) -> tuple[float, dict]:
    """Walkforward HMM: fit on all data up to asof_date, return smoothed p_risk.

    Steps (matches backtest):
    1. Build aligned SPY + VIX close series up to asof_date
    2. Compute 4 HMM features
    3. Fit 3-state GaussianHMM on all features (expanding window)
    4. predict_proba on full feature set → posteriors for each day
    5. p_risk = p_crisis + 0.5 * p_mid for each day
    6. EMA smooth (span=4) → take last value

    Returns: (p_risk_smooth, debug_dict)
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    asof_dt = pd.Timestamp(asof_date)

    # Build aligned close series
    spy_close = spy_df["Close"].astype(float)
    if hasattr(spy_close.index, "normalize"):
        spy_close = spy_close[spy_close.index.normalize() <= asof_dt]

    if vix_df is None or vix_df.empty:
        return 0.0, {"error": "no_vix_data"}

    vix_close = vix_df["Close"].astype(float)
    if hasattr(vix_close.index, "normalize"):
        vix_close = vix_close[vix_close.index.normalize() <= asof_dt]

    # Normalize indices to date-only for alignment (Polygon vs yfinance may differ)
    spy_close = spy_close.copy()
    spy_close.index = pd.to_datetime(spy_close.index).normalize()
    vix_close = vix_close.copy()
    vix_close.index = pd.to_datetime(vix_close.index).normalize()

    # Remove duplicate dates (keep last)
    spy_close = spy_close[~spy_close.index.duplicated(keep='last')]
    vix_close = vix_close[~vix_close.index.duplicated(keep='last')]

    # Align indices
    common_idx = spy_close.index.intersection(vix_close.index)
    spy_aligned = spy_close.reindex(common_idx)
    vix_aligned = vix_close.reindex(common_idx)

    # Compute features
    features = prepare_hmm_features(spy_aligned, vix_aligned)

    if len(features) < HMM_MIN_TRAIN:
        emit_progress("HMM warning", 0,
                      f"Insufficient data for HMM: {len(features)} < {HMM_MIN_TRAIN}")
        return 0.0, {"error": "insufficient_data", "feature_bars": len(features)}

    # Fit 3-state HMM
    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = GaussianHMM(
        n_components=HMM_N_STATES, covariance_type="full",
        n_iter=300, random_state=42, tol=1e-5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled)

    # Identify states by vol (column 1 = vol_10d), with vix_z tiebreaker
    # Composite key: vol is primary; vix_z (col2) breaks ties within ~1.5pp vol band
    # Weight 0.003 ensures vix_z range [-2,+3] contributes at most ±0.9pp to sort key
    means_orig = scaler.inverse_transform(model.means_)
    mean_vols = means_orig[:, 1]
    mean_vix_z = means_orig[:, 2]
    composite_key = mean_vols + mean_vix_z * 0.003
    order = np.argsort(composite_key)
    crisis_idx = int(order[2])
    mid_idx = int(order[1])
    safe_idx = int(order[0])

    # Predict posteriors for all days
    posteriors = model.predict_proba(X_scaled)

    # Compute p_risk = p_crisis + 0.5 * p_mid for each day
    p_risk_raw = posteriors[:, crisis_idx] + HMM_MID_WEIGHT * posteriors[:, mid_idx]

    # EMA smooth (matches backtest: span=4)
    p_risk_series = pd.Series(p_risk_raw, index=features.index)
    p_risk_smooth_series = p_risk_series.ewm(span=HMM_EMA_SPAN, adjust=False).mean()
    p_risk_smooth = float(p_risk_smooth_series.iloc[-1])

    # Current state label
    current_state = int(model.predict(X_scaled[-1:].reshape(1, -1))[0])
    state_labels = {int(order[0]): "SAFE", int(order[1]): "MID", int(order[2]): "CRISIS"}
    current_label = state_labels.get(current_state, "UNKNOWN")

    debug = {
        "feature_bars": len(features),
        "feature_start": str(features.index[0].date()),
        "feature_end": str(features.index[-1].date()),
        "p_crisis": round(float(posteriors[-1, crisis_idx]), 4),
        "p_mid": round(float(posteriors[-1, mid_idx]), 4),
        "p_safe": round(float(posteriors[-1, safe_idx]), 4),
        "p_risk_raw": round(float(p_risk_raw[-1]), 4),
        "p_risk_smooth": round(p_risk_smooth, 4),
        "current_state": current_label,
        "mean_vols": [round(float(v), 6) for v in mean_vols[order]],
    }

    return p_risk_smooth, debug


def compute_target_weights(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    etf_dfs: dict[str, pd.DataFrame],
    validation_report: dict,
    asof_date: str,
    trigger_state: dict,
    db_path: str,
    risk_level: str = "RISK_ON",
    use_hmm: bool = False,
) -> dict:
    """Compute T10C-Slim target weights.

    Two modes:
    - V7 mode (use_hmm=False): risk_level from C# → portfolio selection, VIX trigger state machine
    - Sys3 HMM mode (use_hmm=True): HMM p_risk → portfolio selection + theme budget, no VIX trigger

    Shared layers (both modes): vol-target, MA200 2-level cap, min cash, anomaly blocking.
    All calculations use only data <= asof_date (no lookahead).
    """
    asof_dt = pd.Timestamp(asof_date)

    # ── HMM computation (Sys3 mode) ──
    hmm_p_risk_smooth = None
    hmm_debug = None

    if use_hmm:
        # Try cached result first (same asof_date → deterministic, avoids EM randomness)
        cached_p_risk, cached_debug = load_cached_hmm_p_risk(db_path, asof_date)
        if cached_p_risk is not None:
            hmm_p_risk_smooth = cached_p_risk
            hmm_debug = cached_debug
            emit_progress("HMM cached", 58,
                          f"Using cached p_risk={hmm_p_risk_smooth:.4f} "
                          f"state={hmm_debug.get('current_state', '?')} (asof={asof_date})")
        else:
            emit_progress("HMM computing", 56, "Training 3-state HMM on SPY+VIX...")
            try:
                hmm_p_risk_smooth, hmm_debug = compute_hmm_p_risk(spy_df, vix_df, asof_date)
                emit_progress("HMM computed", 58,
                              f"p_risk_smooth={hmm_p_risk_smooth:.4f} "
                              f"state={hmm_debug.get('current_state', '?')}")
                # Cache to DB for same-day determinism
                save_cached_hmm_p_risk(db_path, asof_date, hmm_p_risk_smooth, hmm_debug)
            except Exception as e:
                emit_progress("HMM error", 58, f"HMM failed: {e} — falling back to V7 mode")
                use_hmm = False  # Fallback to V7

    # ── Portfolio selection ──
    if use_hmm and hmm_p_risk_smooth is not None:
        # Sys3: HMM p_risk drives portfolio selection
        is_risk_on = hmm_p_risk_smooth < SYS3_SWITCH_THRESHOLD
        active_portfolio = PORTFOLIO_RISK_ON if is_risk_on else PORTFOLIO_RISK_OFF
        portfolio_label = "HMM risk-on (SMH)" if is_risk_on else "HMM risk-off (GDX)"
        emit_progress("Portfolio selected", 58,
                      f"HMM p_risk={hmm_p_risk_smooth:.4f} → {portfolio_label}")
    else:
        # V7: risk_level from C# determines portfolio
        is_risk_on = risk_level in ("RISK_ON", "RECOVERY_RAMP")
        active_portfolio = PORTFOLIO_RISK_ON if is_risk_on else PORTFOLIO_RISK_OFF
        portfolio_label = "risk-on (SMH)" if is_risk_on else "risk-off (GDX)"
        emit_progress("Portfolio selected", 58,
                      f"risk_level={risk_level} → {portfolio_label}")

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

    emit_progress("Computing signal", 60, f"SPY={spy_price:.2f} MA200={ma200:.2f}")

    # Get VIX price (or None if no data)
    vix_price = None
    if not vix_df.empty:
        vix_close = vix_df["Close"].astype(float)
        if hasattr(vix_close.index, "normalize"):
            vix_close = vix_close[vix_close.index.normalize() <= asof_dt]
        if len(vix_close) > 0:
            vix_price = float(vix_close.iloc[-1])

    # ── VIX trigger (V7 mode only — skipped in HMM mode) ──
    vix_mode_active = False
    vix_cap_applied = False
    theme_budget = None
    new_trigger_state = trigger_state
    trigger_reason = "hmm_mode"

    if not use_hmm:
        # V7: VIX trigger state machine
        new_trigger_state, trigger_reason = evaluate_trigger_conditions(
            spy_price, ma200, vix_price, trigger_state
        )
        save_trigger_state(db_path, new_trigger_state)

        emit_progress("VIX trigger evaluated", 65,
                      f"mode={new_trigger_state['mode']} reason={trigger_reason}")

        vix_mode_active = (new_trigger_state["mode"] == "vix_active")
        if vix_mode_active:
            theme_budget = compute_vix_theme_budget(vix_price)
            emit_progress("VIX mode active", 67,
                          f"theme_budget={theme_budget:.2f} vix={vix_price} "
                          f"exposure_cap={VIX_EXPOSURE_CAP}")
        else:
            emit_progress("Baseline mode", 67,
                          f"VIX trigger disabled — using full {portfolio_label} weights")
    else:
        # HMM mode: theme budget driven by p_risk >= 0.50
        if hmm_p_risk_smooth >= SYS3_SWITCH_THRESHOLD:
            theme_budget = compute_vix_theme_budget(vix_price)
            emit_progress("HMM risk-off + theme budget", 67,
                          f"p_risk={hmm_p_risk_smooth:.4f} ≥ {SYS3_SWITCH_THRESHOLD} "
                          f"→ theme_budget={theme_budget:.2f}")
        else:
            emit_progress("HMM risk-on", 67,
                          f"p_risk={hmm_p_risk_smooth:.4f} < {SYS3_SWITCH_THRESHOLD}")

    # Portfolio-weighted blended vol (uses ACTIVE portfolio's weights)
    bvol, vol_debug = compute_portfolio_blended_vol(etf_dfs, active_portfolio, asof_date)
    emit_progress("Portfolio vol computed", 70,
                  f"blended_vol={bvol:.4f} (method={vol_debug['method']})")

    # Vol-target exposure
    vol_exposure = TARGET_VOL / bvol if bvol > 0 else 1.0
    vol_exposure = min(vol_exposure, MAX_LEVERAGE)

    # 2-level risk cap (from SPY vs MA200) — kept in BOTH modes
    risk_cap = compute_two_level_risk_cap(spy_price, ma200)
    spy_deviation = (spy_price - ma200) / ma200

    # Final exposure — exact backtest order:
    exposure = min(vol_exposure, risk_cap)

    if use_hmm:
        # Sys3: HMM p_risk >= 0.90 → cap at 0.85 (replaces VIX cap)
        hmm_cap_applied = False
        if hmm_p_risk_smooth >= SYS3_REDUCE_THRESHOLD:
            pre_cap = exposure
            exposure = min(exposure, SYS3_REDUCE_CAP)
            if exposure < pre_cap:
                hmm_cap_applied = True
                emit_progress("HMM cap applied", 73,
                              f"p_risk={hmm_p_risk_smooth:.4f} ≥ {SYS3_REDUCE_THRESHOLD} "
                              f"→ exposure {pre_cap:.4f} → {exposure:.4f}")
    else:
        # V7: VIX exposure cap — when VIX mode active, cap at 0.50
        if vix_mode_active:
            pre_cap = exposure
            exposure = min(exposure, VIX_EXPOSURE_CAP)
            if exposure < pre_cap:
                vix_cap_applied = True
                emit_progress("VIX cap applied", 73,
                              f"exposure {pre_cap:.4f} → {exposure:.4f} (cap={VIX_EXPOSURE_CAP})")

    exposure = min(exposure, 1.0 - MIN_CASH_PCT)
    exposure = max(0.0, min(1.0, exposure))

    emit_progress("Exposure computed", 75,
                  f"vol_target={vol_exposure:.4f} risk_cap={risk_cap:.2f} "
                  f"hmm={'yes' if use_hmm else 'no'} final={exposure:.4f}")

    # Anomaly tickers — set weight to 0 (blocked from trading)
    blocked_anomaly = [t for t, v in validation_report.items()
                       if v.get("anomaly") and t not in ["SPY", VIX_PROXY]]

    # Base strategic weights from active portfolio
    strategic_weights = active_portfolio.copy()

    # Apply theme budget control
    if use_hmm:
        # HMM mode: theme budget when p_risk >= 0.50
        if hmm_p_risk_smooth >= SYS3_SWITCH_THRESHOLD and theme_budget is not None:
            strategic_weights = apply_theme_budget(strategic_weights, theme_budget)
            emit_progress("Theme budget applied", 80,
                          f"budget={theme_budget:.2f} "
                          f"COPX={strategic_weights.get('COPX', 0):.4f} "
                          f"URA={strategic_weights.get('URA', 0):.4f}")
        else:
            emit_progress("Full weights used", 80,
                          f"portfolio={portfolio_label}")
    else:
        # V7 mode: theme budget when VIX trigger active
        if vix_mode_active and theme_budget is not None:
            strategic_weights = apply_theme_budget(strategic_weights, theme_budget)
            emit_progress("Theme budget applied", 80,
                          f"budget={theme_budget:.2f} "
                          f"COPX={strategic_weights.get('COPX', 0):.4f} "
                          f"URA={strategic_weights.get('URA', 0):.4f}")
        else:
            emit_progress("Baseline weights used", 80,
                          f"portfolio={portfolio_label}")

    # Apply exposure and anomaly blocks
    etf_weights = {}
    for ticker, w in strategic_weights.items():
        if ticker in blocked_anomaly:
            etf_weights[ticker] = 0.0
        else:
            etf_weights[ticker] = round(w * exposure, 6)
    cash_weight = round(1.0 - sum(etf_weights.values()), 6)

    # Get latest price for each ETF from their last bar (all tickers from both portfolios)
    etf_prices = {}
    for ticker in ALL_ETF_TICKERS:
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
        "vix_price": round(vix_price, 2) if vix_price else None,
        "vix_trigger_mode": new_trigger_state.get("mode", "hmm_mode") if not use_hmm else "hmm_mode",
        "vix_trigger_reason": trigger_reason,
        "vix_trigger_enable_count": new_trigger_state.get("enable_count", 0) if not use_hmm else 0,
        "vix_trigger_disable_count": new_trigger_state.get("disable_count", 0) if not use_hmm else 0,
        "vix_mode_active": vix_mode_active,
        "vix_cap_applied": vix_cap_applied,
        "theme_budget": round(theme_budget, 6) if theme_budget is not None else None,
        "risk_level": risk_level,
        "portfolio_used": portfolio_label,
        "etf_weights": etf_weights,
        "cash_weight": cash_weight,
        "etf_prices": etf_prices,
        "vol_debug": vol_debug,
        "bars_last_date": bars_last_date,
        "anomaly_tickers": anomaly_tickers,
        "blocked_anomaly_tickers": blocked_anomaly,
        "has_data_anomaly": len(anomaly_tickers) > 0,
        "hmm_p_risk_smooth": round(hmm_p_risk_smooth, 4) if hmm_p_risk_smooth is not None else None,
        "hmm_debug": hmm_debug,
        "use_hmm": use_hmm,
    }


def get_nyse_holidays(years: list[int]) -> list[str]:
    """Return NYSE holiday dates for given years using pandas_market_calendars.

    Returns list of 'YYYY-MM-DD' strings.
    """
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    all_holidays = []
    for year in years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        holidays = nyse.holidays().holidays
        year_holidays = [h for h in holidays
                        if pd.Timestamp(start) <= pd.Timestamp(h) <= pd.Timestamp(end)]
        all_holidays.extend(pd.Timestamp(h).strftime("%Y-%m-%d") for h in year_holidays)
    return sorted(set(all_holidays))


def main():
    parser = argparse.ArgumentParser(description="ETF Rotation Live Signal (T10C-Slim)")
    parser.add_argument("--polygon-key", type=str, default="")
    parser.add_argument("--state-db", type=str, default="",
                       help="Path to TraderApp.db for state persistence")
    parser.add_argument("--risk-level", type=str, default="RISK_ON",
                       choices=["RISK_ON", "RISK_OFF", "CRISIS", "RECOVERY_RAMP"],
                       help="Current V7 risk level from C# scheduler")
    parser.add_argument("--use-hmm", action="store_true", default=False,
                       help="Use HMM System 3 for portfolio selection (replaces V7 risk-level)")
    parser.add_argument("--nyse-holidays", action="store_true", default=False,
                       help="Output NYSE holidays for current + next year as JSON, then exit")
    args = parser.parse_args()

    # ── NYSE holidays mode (fast path, no API key needed) ──
    if args.nyse_holidays:
        try:
            now = datetime.now()
            years = [now.year, now.year + 1]
            holidays = get_nyse_holidays(years)
            print(json.dumps({"holidays": holidays, "years": years}))
        except Exception as e:
            print(json.dumps({"error": str(e), "holidays": []}))
            sys.exit(1)
        return

    try:
        api_key = _resolve_api_key(args.polygon_key)
        if not api_key:
            print(json.dumps({"error": "No Polygon API key found"}))
            sys.exit(1)

        db_path = args.state_db
        if not db_path:
            # Default fallback
            db_path = str(ROOT / "TraderApp" / "TraderApp.db")

        risk_level = args.risk_level
        use_hmm = args.use_hmm

        # Load trigger state from database (used in V7 mode, skipped in HMM mode)
        trigger_state = load_trigger_state(db_path)

        mode_label = "HMM Sys3" if use_hmm else f"V7 risk_level={risk_level}"
        emit_progress("Trigger state loaded", 2,
                      f"mode={trigger_state['mode']} enable_count={trigger_state['enable_count']}")

        # HMM mode needs more historical data for 252d rolling features + 252d training
        if use_hmm:
            global FETCH_DAYS
            FETCH_DAYS = HMM_FETCH_DAYS

        # Determine asof_date FIRST — all data fetched up to this date only
        asof_date = get_asof_date()
        emit_progress("Starting ETF rotation signal", 5,
                      f"T10C-Slim | {mode_label} | asof={asof_date} | fetch_days={FETCH_DAYS}")

        # 1. Fetch SPY + VIX + all ETF data from both portfolios (validated, no lookahead)
        spy_df, vix_df, etf_dfs, validation_report = fetch_all_data(api_key, asof_date)

        # 2. Compute target weights
        result = compute_target_weights(spy_df, vix_df, etf_dfs, validation_report,
                                       asof_date, trigger_state, db_path,
                                       risk_level=risk_level,
                                       use_hmm=use_hmm)

        # 3. Emit decision audit log
        decision_log = {
            "asof_trading_day": result["asof_trading_day"],
            "bars_last_date_by_ticker": result["bars_last_date"],
            "portfolio_realized_vol_5d": result["vol_debug"].get("realized_vol_5d"),
            "portfolio_blended_vol": result["blended_vol"],
            "raw_vol_target_exposure": result["vol_target_exposure"],
            "risk_cap": result["risk_cap"],
            "spy_deviation": result["spy_deviation"],
            "vix_price": result["vix_price"],
            "vix_trigger_mode": result["vix_trigger_mode"],
            "vix_trigger_reason": result["vix_trigger_reason"],
            "vix_mode_active": result["vix_mode_active"],
            "vix_cap_applied": result["vix_cap_applied"],
            "theme_budget": result["theme_budget"],
            "risk_level": result["risk_level"],
            "portfolio_used": result["portfolio_used"],
            "final_exposure": result["exposure"],
            "anomaly_tickers": result["anomaly_tickers"],
            "use_hmm": result["use_hmm"],
        }
        if result.get("hmm_p_risk_smooth") is not None:
            decision_log["hmm_p_risk_smooth"] = result["hmm_p_risk_smooth"]
        emit_decision_log(decision_log)

        hmm_info = ""
        if result.get("use_hmm"):
            p = result.get("hmm_p_risk_smooth", 0)
            hmm_info = f" HMM_p_risk={p:.4f}"

        emit_progress("Complete", 100,
                      f"exposure={result['exposure']:.2%} risk_cap={result['risk_cap']:.0%} "
                      f"portfolio={result['portfolio_used']}"
                      + hmm_info
                      + f" vix={result.get('vix_price', 'N/A')}"
                      + (" ANOMALY" if result["has_data_anomaly"] else ""))

        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
