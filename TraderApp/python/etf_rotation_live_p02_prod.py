#!/usr/bin/env python
"""P02 production candidate: dynamic rotation + emergency hedge + phased re-risk.

This script is intentionally separate from etf_rotation_live.py so current
production logic remains untouched.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # D:\trade
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etf_rotation_live import (  # type: ignore
    _resolve_api_key,
    emit_decision_log,
    emit_progress,
    fetch_ticker_data,
    get_asof_date,
    validate_ticker_data,
)
from regime9_logic import Regime9Params, build_regime9_features, detect_regime9  # type: ignore


# Tradable ETF universe (SPY kept as signal ticker only).
# Locked to C2 candidate from robustness tests.
UNIVERSE_20 = [
    "QQQ", "SMH",
    "VTV",
    "COPX", "XLE",
    "GLD",
]

CASH_TICKER = "BIL"
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
ALL_TICKERS = sorted(set(UNIVERSE_20 + [CASH_TICKER, SPY_TICKER, VIX_TICKER]))

RISK_GROWTH = ["QQQ", "SMH"]
REAL_ASSETS = ["COPX", "XLE", "GLD"]
SECTOR_ROTATION: list[str] = []
SECTOR_DEFENSIVE: list[str] = []
THEME_TICKERS = ["COPX", "XLE", "SMH"]
DEFENSIVE_TICKERS = ["VTV", "GLD"]

# Data quality
MIN_BARS_REQUIRED = 70
FETCH_DAYS = 600

# Per-ETF behavior rules (regime-aware and risk-aware).
# risk_scale:
#   >0  -> reduce weight as risk_signal rises above 0.5
#   <0  -> increase weight as risk_signal rises above 0.5
ETF_RULES: Dict[str, dict] = {
    "QQQ":  {"allowed": {"bull", "volatile"}, "cap": {"bull": 0.24, "volatile": 0.08, "bear": 0.00}, "floor": {}, "risk_scale": 1.6},
    "SMH":  {"allowed": {"bull", "volatile"}, "cap": {"bull": 0.22, "volatile": 0.06, "bear": 0.00}, "floor": {}, "risk_scale": 1.9},
    "VTV":  {"allowed": {"bull", "volatile", "bear"}, "cap": {"bull": 0.12, "volatile": 0.10, "bear": 0.08}, "floor": {}, "risk_scale": 0.3},
    "COPX": {"allowed": {"bull", "volatile"}, "cap": {"bull": 0.10, "volatile": 0.03, "bear": 0.00}, "floor": {}, "risk_scale": 1.8},
    "XLE":  {"allowed": {"bull", "volatile"}, "cap": {"bull": 0.10, "volatile": 0.05, "bear": 0.00}, "floor": {}, "risk_scale": 1.2},
    "GLD":  {"allowed": {"bull", "volatile", "bear"}, "cap": {"bull": 0.10, "volatile": 0.14, "bear": 0.18}, "floor": {"volatile": 0.03, "bear": 0.08}, "risk_scale": -0.6},
}


@dataclass(frozen=True)
class P02Config:
    # Sweet-point tuned around regime signal research
    target_vol: float = 0.12
    min_cash: float = 0.05

    ma200_deep_threshold: float = -0.05
    ma200_shallow_cap: float = 0.60
    ma200_deep_cap: float = 0.30

    cap_bull: float = 0.95
    cap_volatile: float = 0.82
    cap_bear: float = 0.60
    cap_emergency: float = 0.50

    emergency_vix: float = 30.0
    emergency_dd20: float = -0.07
    emergency_confirm: int = 1
    emergency_min_hold: int = 8

    rerisk_stage1_days: int = 2
    rerisk_stage2_days: int = 4
    rerisk_full_days: int = 7

    risk_enter: float = 0.66
    risk_exit: float = 0.48
    risk_jump: float = 0.15
    rebalance_days: int = 21

    # Execution smoothing (from original framework strengths)
    deadband: float = 0.04
    max_step: float = 0.15
    min_hold_days: int = 5

    theme_mult_volatile: float = 1.0
    theme_mult_bear: float = 1.0

    # Optional sector momentum sleeve (XLK/XLRE/XLU). Default off to keep
    # base production behavior unchanged unless explicitly enabled.
    sector_sleeve_enabled: bool = False
    sector_budget_bull: float = 0.05
    sector_budget_volatile: float = 0.00
    sector_budget_bear: float = 0.00
    sector_min_score_bull: float = 0.00
    sector_min_score_volatile: float = -0.01
    sector_min_score_bear: float = 0.00

    # Regime-conditioned sector overlay (report-driven):
    # use XLK/XLU as primary switches; XLRE only under strict whitelist.
    sector_overlay_enabled: bool = False
    sector_overlay_lambda_bull: float = 0.05
    sector_overlay_lambda_volatile: float = 0.03
    sector_overlay_lambda_bear: float = 0.0
    sector_overlay_vix_max_bull: float = 20.0
    sector_overlay_vix_max_volatile: float = 24.0
    sector_overlay_risk_max_bull: float = 0.46
    sector_overlay_risk_max_volatile: float = 0.55
    sector_xlre_vix_max: float = 18.0
    sector_xlre_risk_max: float = 0.42
    sector_xlre_mom_buffer: float = 0.01


def get_db_state(db_path: str, key: str, default: str = "") -> str:
    if not db_path or not Path(db_path).exists():
        return default
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS etf_rotation_state (key TEXT PRIMARY KEY, value TEXT)"
        )
        cur.execute("SELECT value FROM etf_rotation_state WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception:
        return default


def set_db_state(db_path: str, key: str, value: str) -> None:
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS etf_rotation_state (key TEXT PRIMARY KEY, value TEXT)"
        )
        cur.execute("INSERT OR REPLACE INTO etf_rotation_state (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()
    except Exception as e:
        emit_progress("DB state error", 0, str(e))


def load_p02_state(db_path: str) -> dict:
    def _b(k: str, d: bool = False) -> bool:
        return get_db_state(db_path, k, "1" if d else "0") == "1"

    def _i(k: str, d: int = 0) -> int:
        try:
            return int(get_db_state(db_path, k, str(d)))
        except Exception:
            return d

    def _f(k: str, d: float = 0.0) -> float:
        try:
            return float(get_db_state(db_path, k, str(d)))
        except Exception:
            return d

    return {
        "emergency_active": _b("p02_emergency_active", False),
        "emergency_days": _i("p02_emergency_days", 0),
        "emergency_confirm": _i("p02_emergency_confirm", 0),
        "rerisk_mode": _b("p02_rerisk_mode", False),
        "rerisk_days": _i("p02_rerisk_days", 0),
        "bull_streak": _i("p02_bull_streak", 0),
        "prev_risk_signal": _f("p02_prev_risk_signal", 0.5),
        "prev_exposure": _f("p02_prev_exposure", 0.0),
        "last_rebal_date": get_db_state(db_path, "p02_last_rebal_date", ""),
    }


def save_p02_state(db_path: str, st: dict) -> None:
    set_db_state(db_path, "p02_emergency_active", "1" if st.get("emergency_active") else "0")
    set_db_state(db_path, "p02_emergency_days", str(int(st.get("emergency_days", 0))))
    set_db_state(db_path, "p02_emergency_confirm", str(int(st.get("emergency_confirm", 0))))
    set_db_state(db_path, "p02_rerisk_mode", "1" if st.get("rerisk_mode") else "0")
    set_db_state(db_path, "p02_rerisk_days", str(int(st.get("rerisk_days", 0))))
    set_db_state(db_path, "p02_bull_streak", str(int(st.get("bull_streak", 0))))
    set_db_state(db_path, "p02_prev_risk_signal", f"{float(st.get('prev_risk_signal', 0.5)):.6f}")
    set_db_state(db_path, "p02_prev_exposure", f"{float(st.get('prev_exposure', 0.0)):.6f}")
    set_db_state(db_path, "p02_last_rebal_date", str(st.get("last_rebal_date", "")))


def fetch_all_data(api_key: str, asof_date: str) -> tuple[dict[str, pd.DataFrame], dict]:
    etf_dfs: Dict[str, pd.DataFrame] = {}
    validation_report = {}
    for i, t in enumerate(ALL_TICKERS):
        pct = 5 + int(40 * (i + 1) / len(ALL_TICKERS))
        emit_progress("Fetching data", pct, f"{t} ({i+1}/{len(ALL_TICKERS)})")
        try:
            df = fetch_ticker_data(api_key, t, asof_date)
            v = validate_ticker_data(t, df, asof_date)
            # tighten minimum bar check for dynamic model
            if v.get("bars", 0) < MIN_BARS_REQUIRED:
                v["ok"] = False
                v.setdefault("issues", []).append("short_history")
            validation_report[t] = v
            if v.get("ok") and not v.get("anomaly"):
                etf_dfs[t] = df
            elif v.get("anomaly"):
                emit_progress("Anomaly blocked", pct, f"{t}: excluded")
        except Exception as e:
            validation_report[t] = {"ticker": t, "ok": False, "issues": [str(e)]}
    return etf_dfs, validation_report


def _extract_close(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    c = df["Close"].astype(float)
    if hasattr(c.index, "normalize"):
        c = c[c.index.normalize() <= asof]
    return c


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(w.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in w.items()}


def _pick_top(scores: pd.Series, n: int) -> list[str]:
    if scores.empty:
        return []
    return list(scores.sort_values(ascending=False).head(n).index)


def rolling_momentum_score(prices: pd.DataFrame, asof: pd.Timestamp, tickers: list[str]) -> pd.Series:
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    if asof not in prices.index:
        return pd.Series(index=tickers, data=0.0)
    loc = prices.index.get_loc(asof)
    if isinstance(loc, slice):
        loc = int(loc.start)
    if loc < 130:
        return pd.Series(index=tickers, data=0.0)
    p = prices.iloc[: loc + 1]
    valid = [t for t in tickers if t in p.columns and p[t].notna().sum() >= 130 and pd.notna(p[t].iloc[-1])]
    if len(valid) == 0:
        return pd.Series(dtype=float)
    r63 = p[valid].pct_change(63).iloc[-1]
    r126 = p[valid].pct_change(126).iloc[-1]
    vol20 = np.log(p[valid] / p[valid].shift(1)).rolling(20).std().iloc[-1] * np.sqrt(252.0)
    score = 0.6 * r126 + 0.4 * r63 - 0.15 * vol20
    return score.replace([np.inf, -np.inf], np.nan).fillna(-999.0)


def build_sector_sleeve(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    regime: str,
    enabled: bool = True,
    budgets: Dict[str, float] | None = None,
    min_scores: Dict[str, float] | None = None,
) -> Dict[str, float]:
    if not enabled:
        return {}

    scores = rolling_momentum_score(prices, asof, [t for t in SECTOR_ROTATION if t in prices.columns])
    if scores.empty:
        return {}

    st = (regime or "bull").lower()
    budgets = budgets or {"bull": 0.05, "volatile": 0.00, "bear": 0.00}
    min_scores = min_scores or {"bull": 0.00, "volatile": -0.01, "bear": 0.00}
    if st == "bull":
        budget = float(budgets.get("bull", 0.0))
        universe = scores
        min_score = float(min_scores.get("bull", 0.0))
    elif st == "volatile":
        budget = float(budgets.get("volatile", 0.0))
        universe = scores[[t for t in SECTOR_DEFENSIVE if t in scores.index]]
        if universe.empty:
            universe = scores
        min_score = float(min_scores.get("volatile", -0.01))
    else:
        budget = float(budgets.get("bear", 0.0))
        universe = scores[[t for t in SECTOR_DEFENSIVE if t in scores.index]]
        min_score = float(min_scores.get("bear", 0.0))

    if budget <= 0.0:
        return {}

    eligible = universe[universe >= min_score]
    if eligible.empty:
        return {"GLD": budget}

    pick = str(eligible.sort_values(ascending=False).index[0])
    return {pick: budget}


def build_regime_weights(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    regime: str,
    sector_enabled: bool = False,
    sector_budgets: Dict[str, float] | None = None,
    sector_min_scores: Dict[str, float] | None = None,
) -> Dict[str, float]:
    regime = (regime or "bull").lower()
    g_scores = rolling_momentum_score(prices, asof, [t for t in RISK_GROWTH if t in prices.columns])
    r_scores = rolling_momentum_score(prices, asof, [t for t in REAL_ASSETS if t in prices.columns])

    g3 = _pick_top(g_scores, 3)
    g2 = _pick_top(g_scores, 2)
    r2 = _pick_top(r_scores, 2)
    r1 = _pick_top(r_scores, 1)

    w: Dict[str, float] = {}
    if regime == "bull":
        w.update({"USMV": 0.22, "QUAL": 0.18, "VTV": 0.08})
        for t in g3:
            w[t] = w.get(t, 0.0) + 0.37 / max(1, len(g3))
        for t in r2:
            w[t] = w.get(t, 0.0) + 0.15 / max(1, len(r2))
    elif regime == "volatile":
        w.update({"USMV": 0.30, "QUAL": 0.20, "VTV": 0.18, "GLD": 0.12})
        for t in g2:
            w[t] = w.get(t, 0.0) + 0.10 / max(1, len(g2))
        for t in r2:
            w[t] = w.get(t, 0.0) + 0.10 / max(1, len(r2))
    else:
        w.update({"GLD": 0.30, "USMV": 0.30, "QUAL": 0.25, "VTV": 0.10})
        for t in r1:
            w[t] = w.get(t, 0.0) + 0.05 / max(1, len(r1))

    for t, wt in build_sector_sleeve(
        prices,
        asof,
        regime,
        enabled=sector_enabled,
        budgets=sector_budgets,
        min_scores=sector_min_scores,
    ).items():
        w[t] = w.get(t, 0.0) + float(wt)
    return _normalize(w)


def _blend_overlay(core_w: Dict[str, float], ov_w: Dict[str, float], lam: float) -> Dict[str, float]:
    if lam <= 0.0 or not ov_w:
        return dict(core_w)
    out: Dict[str, float] = {}
    keys = set(core_w) | set(ov_w)
    for k in keys:
        out[k] = (1.0 - lam) * float(core_w.get(k, 0.0)) + lam * float(ov_w.get(k, 0.0))
    return _normalize(out)


def apply_regime_sector_overlay(
    base_weights: Dict[str, float],
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    regime: str,
    vix_value: float | None,
    risk_signal: float,
    cfg: P02Config,
) -> tuple[Dict[str, float], dict]:
    if not bool(cfg.sector_overlay_enabled):
        return dict(base_weights), {"enabled": False, "reason": "overlay_disabled"}

    st = (regime or "bull").lower()
    if st == "bear":
        return dict(base_weights), {"enabled": True, "reason": "bear_off"}

    if st == "bull":
        lam = float(cfg.sector_overlay_lambda_bull)
        vix_max = float(cfg.sector_overlay_vix_max_bull)
        risk_max = float(cfg.sector_overlay_risk_max_bull)
        primary_pool = [t for t in ["XLK", "XLU"] if t in prices.columns]
    elif st == "volatile":
        lam = float(cfg.sector_overlay_lambda_volatile)
        vix_max = float(cfg.sector_overlay_vix_max_volatile)
        risk_max = float(cfg.sector_overlay_risk_max_volatile)
        primary_pool = [t for t in ["XLU", "XLK"] if t in prices.columns]
    else:
        lam = float(cfg.sector_overlay_lambda_bear)
        vix_max = 99.0
        risk_max = 1.0
        primary_pool = [t for t in ["XLU", "XLK"] if t in prices.columns]

    if lam <= 0.0:
        return dict(base_weights), {"enabled": True, "reason": "lambda_zero"}
    if (vix_value is None) or np.isnan(float(vix_value)) or float(vix_value) >= vix_max:
        return dict(base_weights), {"enabled": True, "reason": "vix_gate_blocked"}
    if float(risk_signal) >= risk_max:
        return dict(base_weights), {"enabled": True, "reason": "risk_gate_blocked"}
    if not primary_pool:
        return dict(base_weights), {"enabled": True, "reason": "primary_pool_missing"}

    s_primary = rolling_momentum_score(prices, asof, primary_pool)
    if s_primary.empty:
        return dict(base_weights), {"enabled": True, "reason": "primary_score_missing"}
    pick = str(s_primary.sort_values(ascending=False).index[0])
    pick_score = float(s_primary.loc[pick])
    reason = f"pick_{pick}"

    # XLRE strict whitelist: bull only + duration support + momentum edge.
    if st == "bull" and "XLRE" in prices.columns:
        s_re = rolling_momentum_score(prices, asof, ["XLRE"])
        if not s_re.empty:
            re_score = float(s_re.iloc[0])
            loc = prices.index.get_loc(asof)
            if isinstance(loc, slice):
                loc = int(loc.start)
            if (
                loc >= 63
                and float(risk_signal) <= float(cfg.sector_xlre_risk_max)
                and float(vix_value) < float(cfg.sector_xlre_vix_max)
            ):
                i_ok = ("IEF" in prices.columns) and (float(prices["IEF"].iloc[loc] / prices["IEF"].iloc[loc - 63] - 1.0) > 0.0)
                t_ok = ("TLT" in prices.columns) and (float(prices["TLT"].iloc[loc] / prices["TLT"].iloc[loc - 63] - 1.0) > 0.0)
                if i_ok and t_ok and re_score >= (pick_score + float(cfg.sector_xlre_mom_buffer)):
                    pick = "XLRE"
                    reason = "pick_XLRE_strict"

    out = _blend_overlay(base_weights, {pick: 1.0}, lam)
    dbg = {
        "enabled": True,
        "reason": reason,
        "lambda": round(float(lam), 6),
        "pick": pick,
        "regime": st,
        "risk_signal": round(float(risk_signal), 6),
        "vix": None if vix_value is None else round(float(vix_value), 4),
    }
    return out, dbg


def apply_theme_budget(weights: Dict[str, float], theme_budget: float) -> Dict[str, float]:
    out = dict(weights)
    theme_now = sum(out.get(t, 0.0) for t in THEME_TICKERS)
    if theme_now <= theme_budget:
        return _normalize(out)
    excess = theme_now - theme_budget
    if theme_now > 0:
        scale = theme_budget / theme_now
        for t in THEME_TICKERS:
            if t in out:
                out[t] *= scale
    def_now = sum(out.get(t, 0.0) for t in DEFENSIVE_TICKERS)
    if def_now > 0:
        for t in DEFENSIVE_TICKERS:
            if t in out:
                out[t] += excess * (out[t] / def_now)
    return _normalize(out)


def apply_etf_characteristics(
    weights: Dict[str, float],
    regime: str,
    risk_signal: float,
) -> tuple[Dict[str, float], dict]:
    """Apply per-ETF regime whitelist/caps and risk sensitivity.

    Returns:
      (adjusted_weights, debug)
    """
    st = (regime or "bull").lower()
    rs = float(max(0.0, min(1.0, risk_signal)))
    before = {t: float(weights.get(t, 0.0)) for t in UNIVERSE_20}
    out = {t: float(weights.get(t, 0.0)) for t in UNIVERSE_20}

    # 1) regime allow-list and hard cap.
    for t in UNIVERSE_20:
        rule = ETF_RULES.get(t, {})
        allowed = rule.get("allowed", {"bull", "volatile", "bear"})
        if st not in allowed:
            out[t] = 0.0
            continue
        cap = float(rule.get("cap", {}).get(st, 1.0))
        out[t] = min(out[t], cap)

    # 2) risk-signal sensitivity adjustment.
    if rs > 0.5:
        dr = rs - 0.5
        for t in UNIVERSE_20:
            if out[t] <= 0:
                continue
            scale = float(ETF_RULES.get(t, {}).get("risk_scale", 0.0))
            mult = 1.0 - dr * scale
            mult = float(max(0.20, min(1.80, mult)))
            out[t] *= mult

    # 3) defensive floor by regime.
    for t in UNIVERSE_20:
        floor_w = float(ETF_RULES.get(t, {}).get("floor", {}).get(st, 0.0))
        if floor_w > 0.0:
            out[t] = max(out[t], floor_w)

    out = _normalize({k: v for k, v in out.items() if v > 0})
    delta = {
        t: round(float(out.get(t, 0.0) - before.get(t, 0.0)), 6)
        for t in UNIVERSE_20
        if abs(out.get(t, 0.0) - before.get(t, 0.0)) > 1e-6
    }
    debug = {
        "regime": st,
        "risk_signal": round(rs, 6),
        "changed_count": len(delta),
        "top_changes": dict(sorted(delta.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]),
    }
    return out, debug


def compute_vix_theme_budget(vix_value: float | None) -> float:
    if vix_value is None or np.isnan(vix_value):
        return 0.10
    if vix_value < 20.0:
        return 0.20
    if vix_value < 25.0:
        return 0.12
    return 0.06


def compute_two_level_risk_cap(spy_price: float, ma200: float, cfg: P02Config) -> float:
    if np.isnan(ma200) or ma200 <= 0:
        return 1.0
    dev = (spy_price - ma200) / ma200
    if dev >= 0:
        return 1.0
    if dev > cfg.ma200_deep_threshold:
        return cfg.ma200_shallow_cap
    return cfg.ma200_deep_cap


def apply_smoothing(cur: float, target: float, days_since_rebal: int, cfg: P02Config) -> float:
    delta = target - cur
    if days_since_rebal < cfg.min_hold_days and abs(delta) < 0.30:
        return cur
    if abs(delta) < cfg.deadband:
        return cur
    if delta > cfg.max_step:
        return cur + cfg.max_step
    if delta < -cfg.max_step:
        return cur - cfg.max_step
    return target


def compute_portfolio_blended_vol(close_df: pd.DataFrame, weights: Dict[str, float]) -> float:
    tks = [t for t in weights if t in close_df.columns]
    if len(tks) == 0:
        return 0.15
    lr = np.log(close_df[tks] / close_df[tks].shift(1)).dropna()
    if len(lr) < 60:
        return 0.15
    w = np.array([weights[t] for t in tks], dtype=float)
    w = w / w.sum()
    port_lr = (lr * w).sum(axis=1)
    s = port_lr.iloc[-20:]
    l = port_lr.iloc[-60:]
    v_short = float(s.std() * np.sqrt(252)) if len(s) > 5 else 0.15
    v_long = float(l.std() * np.sqrt(252)) if len(l) > 10 else 0.15
    v = 0.7 * v_short + 0.3 * v_long
    return max(0.08, min(v, 0.40))


def compute_risk_signal(feat_row: pd.Series) -> float:
    vix_z = float(feat_row.get("vix_z252", np.nan))
    stress_z = float(feat_row.get("stress_z", np.nan))
    ma_dev = float(feat_row.get("spy_ma200_dev", np.nan))
    dd126 = float(feat_row.get("spy_drawdown_126", np.nan))

    z_vix = 0.0 if np.isnan(vix_z) else np.clip(vix_z, -3.0, 4.0)
    z_str = 0.0 if np.isnan(stress_z) else np.clip(stress_z, -3.0, 4.0)
    trend_penalty = 0.0 if np.isnan(ma_dev) else max(0.0, -ma_dev * 12.0)
    dd_penalty = 0.0 if np.isnan(dd126) else max(0.0, -dd126 * 6.0)
    x = 0.55 * z_vix + 0.35 * z_str + 0.35 * trend_penalty + 0.25 * dd_penalty - 0.35
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_target_weights(
    etf_dfs: Dict[str, pd.DataFrame],
    validation: dict,
    asof_date: str,
    state: dict,
    cfg: P02Config,
    risk_level: str = "RISK_ON",
    use_hmm: bool = False,
) -> tuple[dict, dict]:
    asof_dt = pd.Timestamp(asof_date)

    # Build aligned close panel
    close_map = {}
    for t, df in etf_dfs.items():
        close_map[t] = _extract_close(df, asof_dt)
    close_df = pd.DataFrame(close_map).sort_index()
    close_df = close_df.ffill(limit=5)
    close_df = close_df.dropna(subset=[c for c in [SPY_TICKER, VIX_TICKER, CASH_TICKER] if c in close_df.columns])
    if close_df.empty or len(close_df) < 260:
        raise RuntimeError(f"insufficient_aligned_data:{len(close_df)}")

    asof_trading_day = close_df.index[-1]
    spy = close_df[SPY_TICKER].astype(float)
    vix = close_df[VIX_TICKER].astype(float)

    # Regime9 + continuous risk signal
    px = pd.DataFrame({"spy_close": spy, "vix_close": vix}, index=close_df.index)
    feat = build_regime9_features(px).ffill()
    regime = detect_regime9(px, Regime9Params(ma_buffer=0.005, vol_z_high=0.8, stress_z_high=0.6, hysteresis_days=3)).reindex(feat.index).ffill().fillna("bull")

    last_feat = feat.iloc[-1]
    regime_state = str(regime.iloc[-1]).lower()
    risk_signal = compute_risk_signal(last_feat)

    spy_price = float(spy.iloc[-1])
    ma200 = float(spy.rolling(200).mean().iloc[-1])
    vix_now = float(vix.iloc[-1]) if len(vix) else None
    dd20 = float(spy.iloc[-1] / spy.iloc[-21] - 1.0) if len(spy) > 21 else np.nan

    # Emergency state machine
    cond_emergency = (
        (vix_now is not None and vix_now >= cfg.emergency_vix)
        or (not np.isnan(dd20) and dd20 <= cfg.emergency_dd20)
        or (regime_state == "bear" and vix_now is not None and vix_now >= 24.0)
        or (risk_signal >= cfg.risk_enter)
    )
    state["emergency_confirm"] = int(state.get("emergency_confirm", 0)) + 1 if cond_emergency else 0
    if (not state.get("emergency_active")) and state["emergency_confirm"] >= cfg.emergency_confirm:
        state["emergency_active"] = True
        state["rerisk_mode"] = False
        state["emergency_days"] = 0
        state["bull_streak"] = 0

    if regime_state == "bull" and (vix_now is not None and vix_now < 24.0) and risk_signal <= max(cfg.risk_exit, 0.50):
        state["bull_streak"] = int(state.get("bull_streak", 0)) + 1
    else:
        state["bull_streak"] = 0

    if state.get("emergency_active"):
        state["emergency_days"] = int(state.get("emergency_days", 0)) + 1
        can_release = (
            state["emergency_days"] >= cfg.emergency_min_hold
            and state["bull_streak"] >= cfg.rerisk_stage1_days
            and risk_signal <= cfg.risk_exit
        )
        if can_release:
            state["emergency_active"] = False
            state["rerisk_mode"] = True
            state["rerisk_days"] = 0
            state["bull_streak"] = 0
    elif state.get("rerisk_mode"):
        if regime_state == "bull" and risk_signal <= max(cfg.risk_exit, 0.52):
            state["rerisk_days"] = int(state.get("rerisk_days", 0)) + 1
        else:
            state["rerisk_days"] = max(0, int(state.get("rerisk_days", 0)) - 1)
        if state["rerisk_days"] >= cfg.rerisk_full_days:
            state["rerisk_mode"] = False

    # Dynamic strategic weights
    run_state = "bear" if state.get("emergency_active") else regime_state
    sector_budgets = {
        "bull": float(cfg.sector_budget_bull),
        "volatile": float(cfg.sector_budget_volatile),
        "bear": float(cfg.sector_budget_bear),
    }
    sector_min_scores = {
        "bull": float(cfg.sector_min_score_bull),
        "volatile": float(cfg.sector_min_score_volatile),
        "bear": float(cfg.sector_min_score_bear),
    }
    strategic = build_regime_weights(
        close_df,
        asof_trading_day,
        run_state,
        sector_enabled=bool(cfg.sector_sleeve_enabled),
        sector_budgets=sector_budgets,
        sector_min_scores=sector_min_scores,
    )
    strategic, sector_overlay_debug = apply_regime_sector_overlay(
        strategic,
        close_df,
        asof_trading_day,
        run_state,
        vix_now,
        risk_signal,
        cfg,
    )
    tb = compute_vix_theme_budget(vix_now)
    if run_state == "volatile":
        tb *= cfg.theme_mult_volatile
    elif run_state == "bear":
        tb *= cfg.theme_mult_bear
    tb = max(0.0, min(0.20, tb))
    strategic = apply_theme_budget(strategic, tb)
    strategic, etf_rule_debug = apply_etf_characteristics(strategic, run_state, risk_signal)

    # Exposure
    bvol = compute_portfolio_blended_vol(close_df, strategic)
    vol_target_exposure = min(cfg.target_vol / bvol if bvol > 0 else 1.0, 1.0)
    ma200_cap = compute_two_level_risk_cap(spy_price, ma200, cfg)

    cap_regime = cfg.cap_bull if run_state == "bull" else (cfg.cap_volatile if run_state == "volatile" else cfg.cap_bear)
    state_cap = cap_regime
    if state.get("emergency_active"):
        state_cap = min(state_cap, cfg.cap_emergency)
    elif state.get("rerisk_mode"):
        rd = int(state.get("rerisk_days", 0))
        if rd < cfg.rerisk_stage1_days:
            state_cap = min(state_cap, cfg.cap_emergency)
        elif rd < cfg.rerisk_stage2_days:
            state_cap = min(state_cap, 0.65)
        elif rd < cfg.rerisk_full_days:
            state_cap = min(state_cap, 0.80)

    # Keep old-bridge semantics: in HMM mode, very high risk gets an extra exposure cap.
    if use_hmm and risk_signal >= 0.90:
        state_cap = min(state_cap, 0.85)

    risk_cap = min(ma200_cap, state_cap)
    target_exposure = min(vol_target_exposure, risk_cap)

    target_exposure = max(0.0, min(1.0 - cfg.min_cash, target_exposure))

    # Smoothing using persisted exposure + last rebalance
    prev_exposure = float(state.get("prev_exposure", 0.0))
    last_rebal_date = str(state.get("last_rebal_date", ""))
    days_since_rebal = 999
    if last_rebal_date and last_rebal_date in close_df.index.strftime("%Y-%m-%d").tolist():
        try:
            idx_now = close_df.index.get_loc(asof_trading_day)
            idx_prev = close_df.index.get_loc(pd.Timestamp(last_rebal_date))
            days_since_rebal = int(idx_now - idx_prev)
        except Exception:
            days_since_rebal = 999

    new_exposure = apply_smoothing(prev_exposure, target_exposure, days_since_rebal, cfg)

    periodic_due = days_since_rebal >= int(cfg.rebalance_days)
    jump_due = abs(risk_signal - float(state.get("prev_risk_signal", 0.5))) >= cfg.risk_jump
    rebal_trigger = periodic_due or jump_due or cond_emergency
    if rebal_trigger and abs(new_exposure - prev_exposure) > 0.01:
        state["last_rebal_date"] = asof_trading_day.strftime("%Y-%m-%d")

    # Final weights with cash
    weights = {t: 0.0 for t in UNIVERSE_20}
    for t, w in strategic.items():
        if t in weights:
            weights[t] = float(w) * float(new_exposure)
    cash_w = max(0.0, 1.0 - sum(weights.values()))
    total = sum(weights.values()) + cash_w
    if total > 0:
        for t in weights:
            weights[t] /= total
        cash_w /= total

    # Persist state
    state["prev_exposure"] = float(new_exposure)
    state["prev_risk_signal"] = float(risk_signal)

    anomaly_tickers = [k for k, v in validation.items() if v.get("anomaly")]
    blocked_anomaly_tickers = list(anomaly_tickers)
    bars_last_date = {k: v.get("last_date") for k, v in validation.items()}

    etf_prices = {
        t: float(close_df[t].iloc[-1])
        for t in UNIVERSE_20
        if t in close_df.columns and pd.notna(close_df[t].iloc[-1])
    }

    if use_hmm:
        vix_trigger_mode = "hmm_mode"
        vix_trigger_reason = f"p02_hmm_proxy risk_signal={risk_signal:.4f}"
        vix_mode_active = bool(risk_signal >= 0.50 or run_state != "bull")
        vix_cap_applied = bool(risk_signal >= 0.90)
        portfolio_used = "HMM risk-off (GDX)" if vix_mode_active else "HMM risk-on (SMH)"
        hmm_p_risk_smooth = float(risk_signal)
        vix_trigger_enable_count = 0
        vix_trigger_disable_count = 0
    else:
        vix_mode_active = bool(run_state in ("volatile", "bear") or (vix_now is not None and vix_now >= 25.0))
        vix_trigger_mode = "vix_active_mode" if vix_mode_active else "baseline_mode"
        vix_trigger_reason = f"p02_regime={run_state}"
        vix_cap_applied = bool(vix_mode_active and target_exposure < vol_target_exposure)
        portfolio_used = "risk-on (SMH)" if str(risk_level).upper() in ("RISK_ON", "RECOVERY_RAMP") else "risk-off (GDX)"
        hmm_p_risk_smooth = None
        vix_trigger_enable_count = 1 if vix_mode_active else 0
        vix_trigger_disable_count = 0 if vix_mode_active else 1

    etf_weights = {t: round(float(weights.get(t, 0.0)), 6) for t in UNIVERSE_20}

    result = {
        "strategy": "P02_prod_candidate",
        "asof_trading_day": asof_trading_day.strftime("%Y-%m-%d"),
        "weights": etf_weights,  # local/debug alias
        "etf_weights": etf_weights,
        "cash_weight": float(cash_w),
        "exposure": float(new_exposure),
        "risk_cap": float(risk_cap),
        "vol_target_exposure": float(vol_target_exposure),
        "target_exposure": float(target_exposure),
        "blended_vol": float(bvol),
        "spy_price": float(spy_price),
        "ma200": float(ma200),
        "spy_deviation": float((spy_price - ma200) / ma200) if ma200 > 0 else 0.0,
        "vix_price": float(vix_now) if vix_now is not None else None,
        "vix_trigger_mode": vix_trigger_mode,
        "vix_trigger_reason": vix_trigger_reason,
        "vix_trigger_enable_count": int(vix_trigger_enable_count),
        "vix_trigger_disable_count": int(vix_trigger_disable_count),
        "vix_mode_active": bool(vix_mode_active),
        "vix_cap_applied": bool(vix_cap_applied),
        "dd20": float(dd20) if not np.isnan(dd20) else None,
        "risk_level": str(risk_level),
        "portfolio_used": portfolio_used,
        "etf_prices": etf_prices,
        "regime9_state": regime_state,
        "run_state": run_state,
        "risk_signal": float(risk_signal),
        "emergency_condition": bool(cond_emergency),
        "emergency_active": bool(state.get("emergency_active")),
        "rerisk_mode": bool(state.get("rerisk_mode")),
        "rerisk_days": int(state.get("rerisk_days", 0)),
        "theme_budget": float(tb),
        "sector_sleeve_enabled": bool(cfg.sector_sleeve_enabled),
        "sector_budgets": sector_budgets,
        "sector_overlay_enabled": bool(cfg.sector_overlay_enabled),
        "sector_overlay_debug": sector_overlay_debug,
        "has_data_anomaly": len(anomaly_tickers) > 0,
        "anomaly_tickers": anomaly_tickers,
        "blocked_anomaly_tickers": blocked_anomaly_tickers,
        "bars_last_date": bars_last_date,
        "hmm_p_risk_smooth": hmm_p_risk_smooth,
        "use_hmm": bool(use_hmm),
        "etf_rule_debug": etf_rule_debug,
        "validation_summary": {
            "valid_tickers": [k for k, v in validation.items() if v.get("ok") and not v.get("anomaly")],
            "anomaly_tickers": anomaly_tickers,
            "invalid_tickers": [k for k, v in validation.items() if not v.get("ok")],
        },
    }
    return result, state


def get_nyse_holidays(years: list[int]) -> list[str]:
    """Return NYSE holiday dates for given years as YYYY-MM-DD."""
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    all_holidays: list[str] = []
    holidays = nyse.holidays().holidays
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        year_holidays = [h for h in holidays if start <= pd.Timestamp(h) <= end]
        all_holidays.extend(pd.Timestamp(h).strftime("%Y-%m-%d") for h in year_holidays)
    return sorted(set(all_holidays))


def main() -> None:
    parser = argparse.ArgumentParser(description="P02 production candidate live signal")
    parser.add_argument("--polygon-key", type=str, default="")
    parser.add_argument("--state-db", type=str, default="", help="Path to TraderApp.db for state persistence")
    parser.add_argument(
        "--risk-level",
        type=str,
        default="RISK_ON",
        choices=["RISK_ON", "RISK_OFF", "CRISIS", "RECOVERY_RAMP"],
        help="Current risk level from C# scheduler (kept for bridge compatibility)",
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        default=False,
        help="Compatibility flag; P02 maps this to HMM-style output semantics",
    )
    parser.add_argument(
        "--nyse-holidays",
        action="store_true",
        default=False,
        help="Output NYSE holidays for current + next year as JSON, then exit",
    )
    parser.add_argument(
        "--enable-sector-rotation",
        action="store_true",
        default=False,
        help="Enable optional sector momentum sleeve (XLK/XLRE/XLU)",
    )
    parser.add_argument(
        "--enable-sector-overlay",
        action="store_true",
        default=False,
        help="Enable regime-gated sector overlay (XLK/XLU primary, XLRE strict whitelist)",
    )
    args = parser.parse_args()

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

        db_path = args.state_db or str(ROOT / "TraderApp" / "TraderApp.db")
        asof = get_asof_date()
        mode_label = "HMM-compat mode" if args.use_hmm else f"risk_level={args.risk_level}"
        emit_progress("Start", 2, f"asof={asof} | {mode_label}")

        state = load_p02_state(db_path)
        emit_progress("State loaded", 5, f"emergency={state['emergency_active']} rerisk={state['rerisk_mode']}")

        etf_dfs, validation = fetch_all_data(api_key, asof)
        needed = [SPY_TICKER, VIX_TICKER, CASH_TICKER]
        for t in needed:
            if t not in etf_dfs:
                raise RuntimeError(f"missing_required_ticker:{t}")

        emit_progress("Computing target", 60, "P02 dynamic regime logic")
        cfg = P02Config(
            sector_sleeve_enabled=bool(args.enable_sector_rotation),
            sector_overlay_enabled=bool(args.enable_sector_overlay),
        )
        result, new_state = compute_target_weights(
            etf_dfs,
            validation,
            asof,
            state,
            cfg,
            risk_level=args.risk_level,
            use_hmm=args.use_hmm,
        )
        save_p02_state(db_path, new_state)

        emit_decision_log(
            {
                "asof_trading_day": result["asof_trading_day"],
                "risk_level": result["risk_level"],
                "portfolio_used": result["portfolio_used"],
                "risk_signal": result["risk_signal"],
                "regime9_state": result["regime9_state"],
                "run_state": result["run_state"],
                "emergency_condition": result["emergency_condition"],
                "emergency_active": result["emergency_active"],
                "rerisk_mode": result["rerisk_mode"],
                "exposure": result["exposure"],
                "risk_cap": result["risk_cap"],
                "vol_target_exposure": result["vol_target_exposure"],
                "target_exposure": result["target_exposure"],
                "blended_vol": result["blended_vol"],
                "vix_price": result["vix_price"],
                "vix_trigger_mode": result["vix_trigger_mode"],
                "vix_mode_active": result["vix_mode_active"],
                "theme_budget": result["theme_budget"],
                "sector_sleeve_enabled": result["sector_sleeve_enabled"],
                "sector_overlay_enabled": result["sector_overlay_enabled"],
                "sector_overlay_debug": result["sector_overlay_debug"],
                "anomaly_tickers": result["anomaly_tickers"],
                "use_hmm": result["use_hmm"],
                "hmm_p_risk_smooth": result["hmm_p_risk_smooth"],
            }
        )

        emit_progress(
            "Complete",
            100,
            f"exposure={result['exposure']:.2%} risk_cap={result['risk_cap']:.0%} portfolio={result['portfolio_used']}",
        )
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
