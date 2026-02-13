#!/usr/bin/env python3
"""
VIX Threshold Tuning: Before vs After based on VIX Probability Model
=====================================================================

10-year VIX probability data (2016-2026, N=2535) shows:
  VIX 25-30: 5d Sharpe +0.134, mean +0.42%, P(up) 60.7%  →  DON'T cap
  VIX 30-35: 5d Sharpe +0.276, mean +0.97%, P(up) 61.5%  →  STILL positive
  VIX 35-40: 3d Sharpe +0.468, 5d mean +1.01%             →  DON'T freeze
  VIX 40-50: 5d Sharpe +0.439, 5d mean +2.33%, P(up) 85%  →  DON'T cut
  VIX 50+:   5d Sharpe +0.133, 3d P(up) 52.6%             →  HERE be cautious

Changes tested:
  Strategy A (21d):
    OLD: VIX≥25 2d → cap 0.50, VIX≤20 5d → release
    NEW: VIX≥30 2d → cap 0.50, VIX≤22 5d → release

  Strategy B+SL (5d):
    OLD: VIX≥35 freeze, VIX≥45 cut 50%
    NEW: VIX≥50 cut 50%  (no freeze — VIX 35-50 is mean-reversion gold)
"""
from __future__ import annotations
import sys, warnings
from typing import Dict, Tuple
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

MIN_CASH_PCT = 0.05

def _blended_vol(log_ret, loc):
    if loc < 60: return 0.15
    s = log_ret.iloc[max(0,loc-20):loc]
    l = log_ret.iloc[max(0,loc-60):loc]
    vs = float(s.std()*np.sqrt(252)) if len(s)>5 else 0.15
    vl = float(l.std()*np.sqrt(252)) if len(l)>10 else 0.15
    return max(0.08, min(0.7*vs+0.3*vl, 0.40))


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY A: 21-day vol-target
# ═══════════════════════════════════════════════════════════════════════════════

def run_a(stock_close, spy_close, vix_close, capital=100_000,
          vix_enter=25.0, vix_exit=20.0, vix_enter_days=2, vix_exit_days=5):
    common = stock_close.index.intersection(spy_close.index).intersection(vix_close.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix = vix_close.loc[common].sort_index()
    log_ret = np.log(stock / stock.shift(1)).dropna()
    ma200 = spy.rolling(200).mean()

    warmup = 210
    if len(stock) < warmup + 50: return pd.Series(dtype=float), 0, {}

    dates = stock.index[warmup:]
    eq = []; trades = 0; cur_exp = 0.0; last_rb = -999
    vix_state = "NORMAL"; vix_confirm = 0
    vix_cap_days = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)
        sp = float(spy.iloc[loc]); m2v = float(ma200.iloc[loc])
        v = float(vix.iloc[loc]) if not pd.isna(vix.iloc[loc]) else 15.0

        # VIX state machine (parameterized)
        if vix_state == "NORMAL":
            if v >= vix_enter:
                vix_confirm += 1
                if vix_confirm >= vix_enter_days: vix_state = "RISK_OFF"; vix_confirm = 0
            else: vix_confirm = 0
        else:
            if v <= vix_exit:
                vix_confirm += 1
                if vix_confirm >= vix_exit_days: vix_state = "NORMAL"; vix_confirm = 0
            else: vix_confirm = 0

        if idx > 0 and loc > 0:
            capital += capital * cur_exp * float(stock.iloc[loc]/stock.iloc[loc-1]-1)

        if not (ds >= 21 or idx == 0): eq.append(capital); continue

        lr_loc = log_ret.index.get_loc(date) if date in log_ret.index else None
        if lr_loc is None: eq.append(capital); continue
        bvol = _blended_vol(log_ret, lr_loc)
        te = min(0.12/bvol if bvol>0 else 1.0, 1.0)

        if not (np.isnan(m2v) or m2v<=0):
            dev = (sp-m2v)/m2v
            if dev < -0.05: te = min(te, 0.30)
            elif dev < 0: te = min(te, 0.60)

        if vix_state == "RISK_OFF":
            te = min(te, 0.50); vix_cap_days += 1

        te = max(0.0, min(0.95, te))
        delta = te - cur_exp
        if 0 < delta < 0.02: te = cur_exp
        elif delta < 0 and abs(delta) < 0.05: te = cur_exp
        elif delta > 0.15: te = cur_exp + 0.15
        elif delta < -0.15: te = cur_exp - 0.15

        ad = abs(te - cur_exp)
        if ad > 0.02:
            capital -= ad * capital * 10.0 / 10_000
            trades += 1; last_rb = idx
        cur_exp = te; eq.append(capital)

    if not eq: return pd.Series(dtype=float), 0, {}
    return pd.Series(eq, index=dates[:len(eq)]), trades, {"vix_cap_days": vix_cap_days}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY B+SL: 5-day fixed + adaptive stop
# ═══════════════════════════════════════════════════════════════════════════════

def run_bsl(stock_close, spy_close, vix_close, capital=100_000, cost_bps=15.0,
            vix_freeze=35.0, vix_cut=45.0, use_freeze=True):
    common = stock_close.index.intersection(spy_close.index).intersection(vix_close.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix = vix_close.loc[common].sort_index()
    log_ret = np.log(stock / stock.shift(1)).dropna()
    ma200 = spy.rolling(200).mean()

    warmup = 210
    if len(stock) < warmup + 50: return pd.Series(dtype=float), 0, {}

    dates = stock.index[warmup:]
    eq = []; trades = 0; cur_exp = 0.0; last_rb = -999; regime = "BULL"
    entry_price = None; stopped_out = False
    stop_hits = 0; frozen_d = 0; cut_d = 0; bull_d = 0; bear_d = 0; switches = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)
        sp = float(spy.iloc[loc]); m2v = float(ma200.iloc[loc])
        v = float(vix.iloc[loc]) if not pd.isna(vix.iloc[loc]) else 15.0
        px = float(stock.iloc[loc])

        if idx > 0 and loc > 0:
            capital += capital * cur_exp * float(stock.iloc[loc]/stock.iloc[loc-1]-1)

        # Daily stop-loss check
        if cur_exp > 0 and entry_price is not None and not stopped_out:
            lr_loc = log_ret.index.get_loc(date) if date in log_ret.index else None
            if lr_loc is not None and lr_loc >= 20:
                daily_vol = float(log_ret.iloc[lr_loc-20:lr_loc].std())
                vol_5d = daily_vol * np.sqrt(5)
            else: vol_5d = 0.10
            stop_threshold = max(3.0*vol_5d, 0.15)
            loss = (px - entry_price) / entry_price
            if loss < -stop_threshold:
                capital -= cur_exp * capital * cost_bps / 10_000
                cur_exp = 0.0; stopped_out = True; stop_hits += 1; trades += 1

        if not (ds >= 5 or idx == 0): eq.append(capital); continue
        stopped_out = False

        # VIX extreme handling (parameterized)
        if v >= vix_cut:
            if cur_exp > 0:
                new = cur_exp * 0.5
                capital -= abs(new-cur_exp)*capital*cost_bps/10_000
                trades += 1; cur_exp = new; cut_d += 1
            entry_price = px; last_rb = idx; eq.append(capital); continue

        if use_freeze and v >= vix_freeze:
            frozen_d += 1; entry_price = px; last_rb = idx; eq.append(capital); continue

        # MA200 hysteresis
        if not (np.isnan(m2v) or m2v<=0):
            if regime=="BULL" and sp < m2v*0.98: regime="BEAR"; switches+=1
            elif regime=="BEAR" and sp > m2v*1.02: regime="BULL"; switches+=1

        if regime=="BULL": te = 0.25; bull_d += 1
        else: te = 0.10; bear_d += 1

        te = max(0.0, min(0.95, te))
        ad = abs(te - cur_exp)
        if ad > 0.001:
            capital -= ad*capital*cost_bps/10_000
            trades += 1; last_rb = idx
        cur_exp = te; entry_price = px; eq.append(capital)

    if not eq: return pd.Series(dtype=float), 0, {}
    return pd.Series(eq, index=dates[:len(eq)]), trades, {
        "bull": bull_d, "bear": bear_d, "frozen": frozen_d, "cuts": cut_d,
        "switches": switches, "stop_hits": stop_hits}


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(eq):
    if len(eq) < 20: return {"cagr":0,"maxdd":0,"sharpe":0,"calmar":0,"vol":0,"total":0}
    dr = eq.pct_change().dropna()
    tr = eq.iloc[-1]/eq.iloc[0]-1
    yrs = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = (1+tr)**(1/yrs)-1 if yrs>0 else 0
    vol = dr.std()*np.sqrt(252)
    ex = dr - 0.04/252
    sharpe = ex.mean()/ex.std()*np.sqrt(252) if ex.std()>0 else 0
    maxdd = ((eq-eq.cummax())/eq.cummax()).min()
    calmar = cagr/abs(maxdd) if abs(maxdd)>0 else 0
    return {"cagr":float(cagr),"maxdd":float(maxdd),"sharpe":float(sharpe),
            "calmar":float(calmar),"vol":float(vol),"total":float(tr)}

def yearly(eq):
    out = {}
    for yr in sorted(eq.index.year.unique()):
        m = eq.index.year==yr
        if m.sum()<10: continue
        s = eq[m]
        out[yr] = {"ret":float(s.iloc[-1]/s.iloc[0]-1),
                    "dd":float(((s-s.cummax())/s.cummax()).min())}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import yfinance as yf
    stocks = ["NVDA", "AAPL", "TSLA", "META", "SMCI", "NFLX"]

    print("=" * 130)
    print("  VIX THRESHOLD TUNING: BEFORE vs AFTER (based on 10yr VIX probability model)")
    print("  A: VIX cap 25→30  |  B+SL: freeze 35/cut 45 → no freeze/cut 50")
    print("=" * 130)

    print("\nLoading...", flush=True)
    tickers = stocks + ["SPY", "^VIX"]
    raw = yf.download(tickers, start="2016-01-01", end="2026-02-12",
                      progress=False, auto_adjust=True, group_by="ticker")
    data = {}
    for t in tickers:
        try:
            c = raw[t]["Close"].dropna().sort_index()
            c.index = pd.to_datetime(c.index).normalize()
            data[t] = c
            print("  %s: %d bars" % (t, len(c)))
        except Exception as e:
            print("  %s: FAILED (%s)" % (t, e))

    spy, vix = data["SPY"], data["^VIX"]

    # ── Run all variants ──
    results = {}
    for stock_name in stocks:
        sc = data[stock_name]
        print("  %s..." % stock_name, flush=True)

        w = 210
        bh = sc.iloc[w:]; bh_eq = bh / bh.iloc[0] * 100_000

        # Strategy A: OLD (VIX 25/20) vs NEW (VIX 30/22)
        eq_a_old, t_a_old, st_a_old = run_a(sc, spy, vix, vix_enter=25.0, vix_exit=20.0)
        eq_a_new, t_a_new, st_a_new = run_a(sc, spy, vix, vix_enter=30.0, vix_exit=22.0)

        # Strategy B+SL: OLD (freeze=35, cut=45) vs NEW (no freeze, cut=50)
        eq_b_old, t_b_old, st_b_old = run_bsl(sc, spy, vix, vix_freeze=35.0, vix_cut=45.0, use_freeze=True)
        eq_b_new, t_b_new, st_b_new = run_bsl(sc, spy, vix, vix_freeze=999, vix_cut=50.0, use_freeze=False)

        common = bh_eq.index
        for eq in [eq_a_old, eq_a_new, eq_b_old, eq_b_new]:
            common = common.intersection(eq.index)

        results[stock_name] = {
            "B&H": (bh_eq.loc[common], 0),
            "A old(25/20)": (eq_a_old.loc[common], t_a_old),
            "A new(30/22)": (eq_a_new.loc[common], t_a_new),
            "B old(35/45)": (eq_b_old.loc[common], t_b_old),
            "B new(~~/50)": (eq_b_new.loc[common], t_b_new),
            "st_a_old": st_a_old, "st_a_new": st_a_new,
            "st_b_old": st_b_old, "st_b_new": st_b_new,
        }
        print("    A_old vix_cap_days=%d  A_new vix_cap_days=%d" %
              (st_a_old.get("vix_cap_days",0), st_a_new.get("vix_cap_days",0)))
        print("    B_old frozen=%d cuts=%d stops=%d  B_new frozen=%d cuts=%d stops=%d" %
              (st_b_old.get("frozen",0), st_b_old.get("cuts",0), st_b_old.get("stop_hits",0),
               st_b_new.get("frozen",0), st_b_new.get("cuts",0), st_b_new.get("stop_hits",0)))

    # ── Summary Tables ──
    labels = ["B&H", "A old(25/20)", "A new(30/22)", "B old(35/45)", "B new(~~/50)"]

    print()
    print("=" * 130)
    print("  RESULTS: Strategy A — VIX cap threshold 25 vs 30")
    print("=" * 130)
    for stock_name in stocks:
        r = results[stock_name]
        print("\n  %s:" % stock_name)
        print("  %-16s %8s %8s %8s %8s %8s %7s" % ("Strategy","CAGR","MaxDD","Sharpe","Calmar","Vol","Trades"))
        print("  " + "-"*70)
        for label in ["B&H", "A old(25/20)", "A new(30/22)"]:
            eq, trd = r[label]; m = metrics(eq)
            print("  %-16s %+7.2f%% %7.1f%% %7.3f %7.3f %7.2f%% %6d" %
                  (label, m['cagr']*100, m['maxdd']*100, m['sharpe'], m['calmar'], m['vol']*100, trd))

    print()
    print("=" * 130)
    print("  RESULTS: Strategy B+SL — VIX freeze/cut 35/45 vs no-freeze/cut-50")
    print("=" * 130)
    for stock_name in stocks:
        r = results[stock_name]
        print("\n  %s:" % stock_name)
        print("  %-16s %8s %8s %8s %8s %8s %7s" % ("Strategy","CAGR","MaxDD","Sharpe","Calmar","Vol","Trades"))
        print("  " + "-"*70)
        for label in ["B&H", "B old(35/45)", "B new(~~/50)"]:
            eq, trd = r[label]; m = metrics(eq)
            print("  %-16s %+7.2f%% %7.1f%% %7.3f %7.3f %7.2f%% %6d" %
                  (label, m['cagr']*100, m['maxdd']*100, m['sharpe'], m['calmar'], m['vol']*100, trd))

    # ── Delta Summary ──
    print()
    print("=" * 130)
    print("  DELTA: NEW minus OLD (positive = improvement)")
    print("=" * 130)
    print("\n  %-8s | %12s %12s %12s %12s | %12s %12s %12s %12s" %
          ("Stock", "A ΔCAGR", "A ΔMaxDD", "A ΔSharpe", "A ΔCalmar",
           "B ΔCAGR", "B ΔMaxDD", "B ΔSharpe", "B ΔCalmar"))
    print("  " + "-" * 115)

    a_wins_sharpe = 0; b_wins_sharpe = 0
    a_wins_dd = 0; b_wins_dd = 0

    for stock_name in stocks:
        r = results[stock_name]
        ma_old = metrics(r["A old(25/20)"][0])
        ma_new = metrics(r["A new(30/22)"][0])
        mb_old = metrics(r["B old(35/45)"][0])
        mb_new = metrics(r["B new(~~/50)"][0])

        da_cagr = ma_new['cagr'] - ma_old['cagr']
        da_dd = ma_new['maxdd'] - ma_old['maxdd']  # less negative = better
        da_sh = ma_new['sharpe'] - ma_old['sharpe']
        da_cal = ma_new['calmar'] - ma_old['calmar']

        db_cagr = mb_new['cagr'] - mb_old['cagr']
        db_dd = mb_new['maxdd'] - mb_old['maxdd']
        db_sh = mb_new['sharpe'] - mb_old['sharpe']
        db_cal = mb_new['calmar'] - mb_old['calmar']

        if da_sh > 0: a_wins_sharpe += 1
        if da_dd > 0: a_wins_dd += 1  # less negative maxdd
        if db_sh > 0: b_wins_sharpe += 1
        if db_dd > 0: b_wins_dd += 1

        print("  %-8s | %+11.2f%% %+11.1f%% %+11.3f %+11.3f | %+11.2f%% %+11.1f%% %+11.3f %+11.3f" %
              (stock_name, da_cagr*100, da_dd*100, da_sh, da_cal,
               db_cagr*100, db_dd*100, db_sh, db_cal))

    print()
    print("  Strategy A (VIX 25→30): Sharpe improved %d/%d stocks, MaxDD improved %d/%d" %
          (a_wins_sharpe, len(stocks), a_wins_dd, len(stocks)))
    print("  Strategy B (freeze→none, cut 45→50): Sharpe improved %d/%d stocks, MaxDD improved %d/%d" %
          (b_wins_sharpe, len(stocks), b_wins_dd, len(stocks)))

    # ── Yearly comparison for B ──
    print()
    print("=" * 130)
    print("  YEARLY RETURNS: B old vs B new")
    print("=" * 130)
    for stock_name in stocks:
        r = results[stock_name]
        yr_old = yearly(r["B old(35/45)"][0])
        yr_new = yearly(r["B new(~~/50)"][0])
        all_yrs = sorted(set().union(yr_old, yr_new))
        print("\n  %s:" % stock_name)
        print("  %-5s %9s %9s %9s %9s %9s" % ("Yr","B_old","B_new","Δ","old_DD","new_DD"))
        print("  " + "-" * 50)
        for yr in all_yrs:
            o = yr_old.get(yr, {"ret":0,"dd":0})
            n = yr_new.get(yr, {"ret":0,"dd":0})
            d = n['ret'] - o['ret']
            print("  %-5d %+8.1f%% %+8.1f%% %+8.2f%% %8.1f%% %8.1f%%" %
                  (yr, o['ret']*100, n['ret']*100, d*100, o['dd']*100, n['dd']*100))


if __name__ == "__main__":
    main()
