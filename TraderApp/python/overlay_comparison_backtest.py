#!/usr/bin/env python3
"""
Risk Overlay Comparison: A(21d) vs B(5d) vs B+SL(5d + adaptive stop)
=====================================================================

Strategy A — Vol-Target Overlay (21-day, T10C-Slim original):
  L1: Vol-target (0.12 / blended_vol)
  L2: MA200 2-level hard cap (1.0 / 0.60 / 0.30)
  L3: VIX cap 0.50 (VIX≥25 2d confirm, VIX≤20 5d release)
  L4: Min cash 5%
  L5: Asymmetric deadband (up=0.02, down=0.05), max step 15%
  Cost: 10 bps

Strategy B — Single-Stock Prediction Overlay (5-day):
  L1: Fixed sizing (Bull: 25%, Bear: 10%)
  L2: MA200 regime with hysteresis (±2%)
  L3: VIX≥35 freeze, VIX≥45 cut 50%
  L4: Min cash 5%
  L5: 5-day full rebalance
  Cost: 15 bps

Strategy B+SL — Same as B + vol-adaptive stop loss:
  stop = -max(3 × stock_vol_5d, 0.15)
  Only 3-sigma moves trigger. Normal vol ignored.
  Checked daily within 5-day holding period.
"""
from __future__ import annotations

import sys, warnings
from typing import Dict, Tuple
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED
# ═══════════════════════════════════════════════════════════════════════════════

MIN_CASH_PCT = 0.05


def _blended_vol(log_ret: pd.Series, loc: int) -> float:
    SHORT, LONG, ALPHA = 20, 60, 0.7
    FLOOR, CAP = 0.08, 0.40
    if loc < LONG:
        return 0.15
    s = log_ret.iloc[max(0, loc - SHORT):loc]
    l = log_ret.iloc[max(0, loc - LONG):loc]
    vs = float(s.std() * np.sqrt(252)) if len(s) > 5 else 0.15
    vl = float(l.std() * np.sqrt(252)) if len(l) > 10 else 0.15
    return max(FLOOR, min(ALPHA * vs + (1 - ALPHA) * vl, CAP))


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY A: T10C-Slim Original (21-day)
# ═══════════════════════════════════════════════════════════════════════════════

def run_strategy_a(
    stock_close: pd.Series, spy_close: pd.Series, vix_close: pd.Series,
    initial_capital: float = 100_000,
) -> Tuple[pd.Series, int, dict]:

    common = stock_close.index.intersection(spy_close.index).intersection(vix_close.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix = vix_close.loc[common].sort_index()

    log_ret = np.log(stock / stock.shift(1)).dropna()
    ma200 = spy.rolling(200).mean()

    warmup = 210
    if len(stock) < warmup + 50:
        return pd.Series(dtype=float), 0, {}

    dates = stock.index[warmup:]
    capital = initial_capital
    equity = []
    trades = 0
    cur_exp = 0.0
    last_rb = -999

    # VIX state
    vix_state = "NORMAL"
    vix_confirm = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)
        sp = float(spy.iloc[loc])
        m2v = float(ma200.iloc[loc])
        v = float(vix.iloc[loc]) if not pd.isna(vix.iloc[loc]) else 15.0

        # VIX state machine
        if vix_state == "NORMAL":
            if v >= 25.0:
                vix_confirm += 1
                if vix_confirm >= 2:
                    vix_state = "RISK_OFF"; vix_confirm = 0
            else:
                vix_confirm = 0
        else:
            if v <= 20.0:
                vix_confirm += 1
                if vix_confirm >= 5:
                    vix_state = "NORMAL"; vix_confirm = 0
            else:
                vix_confirm = 0

        # Daily P&L
        if idx > 0 and loc > 0:
            capital += capital * cur_exp * float(stock.iloc[loc] / stock.iloc[loc - 1] - 1)

        if not (ds >= 21 or idx == 0):
            equity.append(capital)
            continue

        # L1: Vol-target
        lr_loc = log_ret.index.get_loc(date) if date in log_ret.index else None
        if lr_loc is None:
            equity.append(capital); continue
        bvol = _blended_vol(log_ret, lr_loc)
        te = min(0.12 / bvol if bvol > 0 else 1.0, 1.0)

        # L2: MA200 hard cap
        if not (np.isnan(m2v) or m2v <= 0):
            dev = (sp - m2v) / m2v
            if dev < -0.05:
                te = min(te, 0.30)
            elif dev < 0:
                te = min(te, 0.60)

        # L3: VIX cap
        if vix_state == "RISK_OFF":
            te = min(te, 0.50)

        # L4: Min cash
        te = max(0.0, min(0.95, te))

        # L5: Deadband
        delta = te - cur_exp
        if 0 < delta < 0.02:
            te = cur_exp
        elif delta < 0 and abs(delta) < 0.05:
            te = cur_exp
        elif delta > 0.15:
            te = cur_exp + 0.15
        elif delta < -0.15:
            te = cur_exp - 0.15

        ad = abs(te - cur_exp)
        if ad > 0.02:
            capital -= ad * capital * 10.0 / 10_000
            trades += 1; last_rb = idx

        cur_exp = te
        equity.append(capital)

    if not equity:
        return pd.Series(dtype=float), 0, {}
    return pd.Series(equity, index=dates[:len(equity)]), trades, {}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY B: Prediction Overlay (5-day) — with MA200 hysteresis only
# ═══════════════════════════════════════════════════════════════════════════════

def run_strategy_b(
    stock_close: pd.Series, spy_close: pd.Series, vix_close: pd.Series,
    initial_capital: float = 100_000, cost_bps: float = 15.0,
) -> Tuple[pd.Series, int, dict]:

    common = stock_close.index.intersection(spy_close.index).intersection(vix_close.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix = vix_close.loc[common].sort_index()

    ma200 = spy.rolling(200).mean()

    warmup = 210
    if len(stock) < warmup + 50:
        return pd.Series(dtype=float), 0, {}

    dates = stock.index[warmup:]
    capital = initial_capital
    equity = []
    trades = 0
    cur_exp = 0.0
    last_rb = -999
    regime = "BULL"

    bull_d = bear_d = frozen_d = cut_d = switches = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)
        sp = float(spy.iloc[loc])
        m2v = float(ma200.iloc[loc])
        v = float(vix.iloc[loc]) if not pd.isna(vix.iloc[loc]) else 15.0

        # Daily P&L
        if idx > 0 and loc > 0:
            capital += capital * cur_exp * float(stock.iloc[loc] / stock.iloc[loc - 1] - 1)

        if not (ds >= 5 or idx == 0):
            equity.append(capital)
            continue

        # L3: VIX extreme
        if v >= 45.0:
            if cur_exp > 0:
                new = cur_exp * 0.5
                capital -= abs(new - cur_exp) * capital * cost_bps / 10_000
                trades += 1; cur_exp = new; cut_d += 1
            last_rb = idx; equity.append(capital); continue

        if v >= 35.0:
            frozen_d += 1; last_rb = idx; equity.append(capital); continue

        # L2: MA200 regime with hysteresis ±2%
        if not (np.isnan(m2v) or m2v <= 0):
            if regime == "BULL" and sp < m2v * 0.98:
                regime = "BEAR"; switches += 1
            elif regime == "BEAR" and sp > m2v * 1.02:
                regime = "BULL"; switches += 1

        # L1: Fixed sizing (single-stock)
        if regime == "BULL":
            te = 0.25  # 25%
            bull_d += 1
        else:
            te = 0.10  # 10%
            bear_d += 1

        # L4: Min cash
        te = max(0.0, min(0.95, te))

        # L5: 5-day full rebalance
        ad = abs(te - cur_exp)
        if ad > 0.001:
            capital -= ad * capital * cost_bps / 10_000
            trades += 1; last_rb = idx

        cur_exp = te
        equity.append(capital)

    if not equity:
        return pd.Series(dtype=float), 0, {}

    stats = {"bull": bull_d, "bear": bear_d, "frozen": frozen_d, "cuts": cut_d, "switches": switches}
    return pd.Series(equity, index=dates[:len(equity)]), trades, stats


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY B+SL: Prediction + Vol-Adaptive Stop Loss
# ═══════════════════════════════════════════════════════════════════════════════

def run_strategy_b_sl(
    stock_close: pd.Series, spy_close: pd.Series, vix_close: pd.Series,
    initial_capital: float = 100_000, cost_bps: float = 15.0,
) -> Tuple[pd.Series, int, dict]:

    common = stock_close.index.intersection(spy_close.index).intersection(vix_close.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix = vix_close.loc[common].sort_index()

    log_ret = np.log(stock / stock.shift(1)).dropna()
    ma200 = spy.rolling(200).mean()

    warmup = 210
    if len(stock) < warmup + 50:
        return pd.Series(dtype=float), 0, {}

    dates = stock.index[warmup:]
    capital = initial_capital
    equity = []
    trades = 0
    cur_exp = 0.0
    last_rb = -999
    regime = "BULL"

    # Stop loss state
    entry_price = None
    stopped_out = False
    stop_hits = 0
    stop_details = []

    bull_d = bear_d = frozen_d = cut_d = switches = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)
        sp = float(spy.iloc[loc])
        m2v = float(ma200.iloc[loc])
        v = float(vix.iloc[loc]) if not pd.isna(vix.iloc[loc]) else 15.0
        px = float(stock.iloc[loc])

        # Daily P&L
        if idx > 0 and loc > 0:
            capital += capital * cur_exp * float(stock.iloc[loc] / stock.iloc[loc - 1] - 1)

        # ── Daily stop-loss check (every day, not just rebalance) ──
        if cur_exp > 0 and entry_price is not None and not stopped_out:
            # 5-day vol = daily_vol × sqrt(5)
            lr_loc = log_ret.index.get_loc(date) if date in log_ret.index else None
            if lr_loc is not None and lr_loc >= 20:
                daily_vol = float(log_ret.iloc[lr_loc - 20:lr_loc].std())
                vol_5d = daily_vol * np.sqrt(5)
            else:
                vol_5d = 0.10

            stop_threshold = max(3.0 * vol_5d, 0.15)
            loss_from_entry = (px - entry_price) / entry_price

            if loss_from_entry < -stop_threshold:
                capital -= cur_exp * capital * cost_bps / 10_000
                stop_details.append(
                    f"    {date.strftime('%Y-%m-%d')}: px={px:.1f} entry={entry_price:.1f} "
                    f"loss={loss_from_entry:+.1%} stop={-stop_threshold:.1%} vol5d={vol_5d:.1%}"
                )
                cur_exp = 0.0
                stopped_out = True
                stop_hits += 1
                trades += 1

        # Non-rebalance day → skip
        if not (ds >= 5 or idx == 0):
            equity.append(capital)
            continue

        # ── Rebalance day ──
        stopped_out = False  # allow re-entry

        # L3: VIX extreme
        if v >= 45.0:
            if cur_exp > 0:
                new = cur_exp * 0.5
                capital -= abs(new - cur_exp) * capital * cost_bps / 10_000
                trades += 1; cur_exp = new; cut_d += 1
            entry_price = px
            last_rb = idx; equity.append(capital); continue

        if v >= 35.0:
            frozen_d += 1; entry_price = px
            last_rb = idx; equity.append(capital); continue

        # L2: MA200 regime with hysteresis ±2%
        if not (np.isnan(m2v) or m2v <= 0):
            if regime == "BULL" and sp < m2v * 0.98:
                regime = "BEAR"; switches += 1
            elif regime == "BEAR" and sp > m2v * 1.02:
                regime = "BULL"; switches += 1

        # L1: Fixed sizing (single-stock)
        if regime == "BULL":
            te = 0.25  # 25%
            bull_d += 1
        else:
            te = 0.10  # 10%
            bear_d += 1

        # L4: Min cash
        te = max(0.0, min(0.95, te))

        # L5: 5-day full rebalance
        ad = abs(te - cur_exp)
        if ad > 0.001:
            capital -= ad * capital * cost_bps / 10_000
            trades += 1; last_rb = idx

        cur_exp = te
        entry_price = px  # record entry for stop-loss tracking
        equity.append(capital)

    if not equity:
        return pd.Series(dtype=float), 0, {}

    stats = {"bull": bull_d, "bear": bear_d, "frozen": frozen_d, "cuts": cut_d,
             "switches": switches, "stop_hits": stop_hits, "stop_details": stop_details}
    return pd.Series(equity, index=dates[:len(equity)]), trades, stats


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(eq: pd.Series) -> dict:
    if len(eq) < 20:
        return {"cagr": 0, "maxdd": 0, "sharpe": 0, "calmar": 0, "vol": 0, "total_ret": 0}
    dr = eq.pct_change().dropna()
    tr = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    vol = dr.std() * np.sqrt(252)
    ex = dr - 0.04 / 252
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    maxdd = ((eq - eq.cummax()) / eq.cummax()).min()
    calmar = cagr / abs(maxdd) if abs(maxdd) > 0 else 0
    return {"cagr": float(cagr), "maxdd": float(maxdd), "sharpe": float(sharpe),
            "calmar": float(calmar), "vol": float(vol), "total_ret": float(tr)}


def yearly(eq: pd.Series) -> Dict[int, dict]:
    out = {}
    for yr in sorted(eq.index.year.unique()):
        m = eq.index.year == yr
        if m.sum() < 10: continue
        s = eq[m]
        out[yr] = {"ret": float(s.iloc[-1]/s.iloc[0]-1),
                    "dd": float(((s-s.cummax())/s.cummax()).min())}
    return out


def p(c="=", w=130):
    print(c * w)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import yfinance as yf

    stocks = ["NVDA", "AAPL", "TSLA", "META", "SMCI", "NFLX"]

    p()
    print("  Risk Overlay:  A = T10C-Slim 21d  |  B = Pred 5d  |  B+SL = Pred 5d + vol-adaptive stop")
    print("  Stop loss: -max(3 × stock_vol_5d, 15%)  →  only 3-sigma crashes trigger")
    p()

    print("\nLoading...", flush=True)
    tickers = stocks + ["SPY", "^VIX"]
    raw = yf.download(tickers, start="2020-01-01", end="2026-02-12",
                      progress=False, auto_adjust=True, group_by="ticker")

    data = {}
    for t in tickers:
        try:
            c = raw[t]["Close"].dropna().sort_index()
            c.index = pd.to_datetime(c.index).normalize()
            data[t] = c
            print(f"  {t}: {len(c)} bars")
        except Exception as e:
            print(f"  {t}: FAILED ({e})")

    spy, vix = data["SPY"], data["^VIX"]

    # ── Run ──
    results = {}
    for stock in stocks:
        sc = data[stock]
        print(f"  {stock}...", flush=True)

        w = 210
        bh = sc.iloc[w:]; bh_eq = bh / bh.iloc[0] * 100_000

        eq_a, trd_a, _ = run_strategy_a(sc, spy, vix)
        eq_b, trd_b, st_b = run_strategy_b(sc, spy, vix, cost_bps=15.0)
        eq_bsl, trd_bsl, st_bsl = run_strategy_b_sl(sc, spy, vix, cost_bps=15.0)

        common = bh_eq.index.intersection(eq_a.index).intersection(eq_b.index).intersection(eq_bsl.index)

        results[stock] = {
            "B&H": (bh_eq.loc[common], 0),
            "A: T10C 21d": (eq_a.loc[common], trd_a),
            "B: Pred 5d": (eq_b.loc[common], trd_b),
            "B+SL: 3σ stop": (eq_bsl.loc[common], trd_bsl),
            "stats_b": st_b,
            "stats_bsl": st_bsl,
        }
        st = st_b
        print(f"    bull={st['bull']} bear={st['bear']} frozen={st['frozen']} cuts={st['cuts']} switches={st['switches']}")
        sh = st_bsl.get("stop_hits", 0)
        print(f"    stop_hits={sh}")
        for d in st_bsl.get("stop_details", []):
            print(d)

    # ── Summary Table ──
    p()
    print("  RESULTS:  B vs B+SL (vol-adaptive stop loss)")
    p()

    for stock in stocks:
        r = results[stock]
        print(f"\n  {stock}:")
        print(f"  {'Strategy':<18} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Vol':>8} {'Trades':>7} {'Total':>10}")
        print(f"  {'-' * 85}")
        for label in ["B&H", "A: T10C 21d", "B: Pred 5d", "B+SL: 3σ stop"]:
            eq, trd = r[label]
            m = metrics(eq)
            print(f"  {label:<18} {m['cagr']:>7.2%} {m['maxdd']:>7.1%} {m['sharpe']:>7.3f} "
                  f"{m['calmar']:>7.3f} {m['vol']:>7.2%} {trd:>7d} {m['total_ret']:>9.1%}")
        print(f"  {'-' * 85}")

    # ── Yearly Returns ──
    p()
    print("  YEARLY RETURNS")
    p()
    for stock in stocks:
        r = results[stock]
        yr_bh = yearly(r["B&H"][0])
        yr_b = yearly(r["B: Pred 5d"][0])
        yr_bsl = yearly(r["B+SL: 3σ stop"][0])
        all_yrs = sorted(set().union(yr_bh, yr_b, yr_bsl))
        print(f"\n  {stock}:")
        print(f"  {'Yr':<5} {'B&H':>9} {'B':>9} {'B+SL':>9} {'B&H DD':>9} {'B DD':>9} {'B+SL DD':>9}")
        print(f"  {'-' * 60}")
        for yr in all_yrs:
            bh = yr_bh.get(yr, {"ret":0,"dd":0})
            b = yr_b.get(yr, {"ret":0,"dd":0})
            bsl = yr_bsl.get(yr, {"ret":0,"dd":0})
            print(f"  {yr:<5} {bh['ret']:>+8.1%} {b['ret']:>+8.1%} {bsl['ret']:>+8.1%} "
                  f"{bh['dd']:>8.1%} {b['dd']:>8.1%} {bsl['dd']:>8.1%}")

    # ── Trade Cost ──
    p()
    print("  TRADE FREQUENCY")
    p()
    for stock in stocks:
        r = results[stock]
        _, ta = r["A: T10C 21d"]
        _, tb = r["B: Pred 5d"]
        _, tbsl = r["B+SL: 3σ stop"]
        eq_a = r["A: T10C 21d"][0]
        yrs = (eq_a.index[-1] - eq_a.index[0]).days / 365.25
        sh = r["stats_bsl"].get("stop_hits", 0)
        print(f"  {stock}: A={ta} ({ta/yrs:.1f}/yr) | B={tb} ({tb/yrs:.1f}/yr) | B+SL={tbsl} ({tbsl/yrs:.1f}/yr, {sh} stops)")

    # ── Final verdict ──
    p()
    print("  DESIGN SUMMARY")
    p()
    print("""
  Strategy A (T10C-Slim 21d) — unchanged:
    L1: exposure = 12% / blended_vol(70%×20d + 30%×60d, floor=8%, cap=40%)
    L2: SPY < MA200 → cap 60%,  SPY < MA200×0.95 → cap 30%
    L3: VIX≥25 2d confirm → cap 50%,  VIX≤20 5d confirm → release
    L4: min cash 5%
    L5: deadband up=2% down=5%, max step 15%, rebalance 21d

  Strategy B+SL (Prediction 5d) — single-stock:
    L1: Bull=25%, Bear=10%
    L2: SPY > MA200×1.02 → Bull,  SPY < MA200×0.98 → Bear
    L3: VIX≥35 → freeze, VIX≥45 → cut 50%
    L4: min cash 5%
    L5: 5-day full rebalance

  Strategy B+SL — B + vol-adaptive stop loss:
    stop = -max(3 × stock_vol_5d, 0.15)
    vol_5d = rolling_20d_daily_std × sqrt(5)
    Checked daily within holding period. Exit immediately on trigger.
    Re-entry allowed at next 5-day rebalance.
    Purpose: black swan protection (earnings miss, fraud, FDA reject)
    NOT triggered by normal volatility (3-sigma filter).
""")


if __name__ == "__main__":
    main()
