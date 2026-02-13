#!/usr/bin/env python
"""Deep Analysis: C0 vs C2 Exit — Paired Comparison, Capital Efficiency, State Machine

A2) 2025 sanity check (trade returns, equity, daily returns)
B1) Paired same-entry comparison (Δ = C2 − C0)
B2) Paired by HMM state bucket
C)  Capital efficiency metrics + cost sensitivity
D)  State-machine exit rule (C0 in risk-on, C2 in risk-off)
"""
from __future__ import annotations
import sys, io, os, warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from scipy import stats as sp_stats

if os.name == "nt" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, "D:/trade/pipelines")
from layer1_version_l_test import (
    load_ohlcv, compute_indicators, apply_layer1, build_lookups,
    compute_metrics, load_hmm, run_bt, START, END, TOP_K,
    R_PCT, DAY3_THR, EXT_THR, EXT_HOLD, TRAIL_PCT, HOLD, CAPITAL
)

OUT_DIR = Path("D:/trade/result/layer1_exit_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ── Paired Trade Simulator ────────────────────────────────────────────
def simulate_paired_trades(pb, hmm_lk, hmm_lo=0.25, hmm_hi=0.50,
                           gap_limit=0.08, cost_bps=20, ret_cap=None,
                           start_dt=None, end_dt=None):
    """
    For each entry signal, simulate BOTH C0 and C2 exits independently.
    Returns DataFrame with columns:
      ticker, sig_date, entry_date, entry_price, atr_pct,
      hmm_state, p_crisis,
      ret_c0, days_c0, reason_c0,
      ret_c2, days_c2, reason_c2, extended_c2,
      delta (ret_c2 - ret_c0)
    """
    pl, il, sbd = pb["pl"], pb["il"], pb["sbd"]
    td, ti, ws = pb["td"], pb["ti"], pb["ws"]
    cost = cost_bps / 10_000.0
    start_s = start_dt or ws
    end_s = end_dt or td[-1]
    r_pct = R_PCT

    results = []

    for ds in td:
        if ds < start_s or ds > end_s:
            continue
        idx = ti.get(ds)
        if idx is None or idx <= 0:
            continue
        prev = td[idx - 1]

        # HMM state
        pc = hmm_lk.get(prev, 0) if hmm_lk else 0
        if np.isnan(pc):
            pc = 1.0  # treat NaN as crisis
        if pc < hmm_lo:
            top_k = 8; hmm_state = "risk_on"
        elif pc < hmm_hi:
            top_k = 5; hmm_state = "transition"
        else:
            top_k = 2; hmm_state = "risk_off"

        sigs = sbd.get(prev, [])[:top_k]
        for t, atr in sigs:
            b = pl.get((ds, t))
            if not b or b["Open"] <= 0:
                continue
            ind = il.get((prev, t), {})
            g = ind.get("gap", 0) or 0
            if abs(g) > gap_limit:
                continue

            ep = b["Open"]  # entry price

            # ── Simulate C0 exit ──
            ret_c0, days_c0, reason_c0 = _sim_exit(
                pl, td, ti, t, ds, idx, ep, atr, "C0", r_pct, cost, ret_cap)

            # ── Simulate C2 exit ──
            ret_c2, days_c2, reason_c2 = _sim_exit(
                pl, td, ti, t, ds, idx, ep, atr, "C2", r_pct, cost, ret_cap)

            results.append({
                "ticker": t, "sig_date": prev, "entry_date": ds,
                "entry_price": ep, "atr_pct": atr,
                "p_crisis": pc, "hmm_state": hmm_state,
                "ret_c0": ret_c0, "days_c0": days_c0, "reason_c0": reason_c0,
                "ret_c2": ret_c2, "days_c2": days_c2, "reason_c2": reason_c2,
                "extended_c2": reason_c2 in ("trailing", "ext_time"),
                "delta": ret_c2 - ret_c0,
            })

    return pd.DataFrame(results)


def _sim_exit(pl, td, ti, ticker, entry_ds, entry_idx, ep, atr_pct,
              exit_mode, r_pct, cost, ret_cap):
    """Simulate a single trade's exit. Returns (return_pct, days_held, reason)."""
    hold = HOLD
    ei = entry_idx + hold
    ex_date = td[ei] if ei < len(td) else td[-1]
    R = ep * r_pct
    mc = ep  # max close
    mh = ep  # max high
    pc_prev = ep  # previous close
    extended = False
    trail_hi = 0

    for day_offset in range(1, 40):  # max 40 days safety
        di = entry_idx + day_offset
        if di >= len(td):
            break
        ds = td[di]
        bar = pl.get((ds, ticker))
        if not bar or bar["Open"] <= 0:
            continue

        O, H, L, C = bar["Open"], bar["High"], bar["Low"], bar["Close"]

        # Return cap
        if ret_cap is not None:
            ret_now = (C - ep) / ep if ep > 0 else 0
            if ret_now >= ret_cap:
                cap_price = ep * (1 + ret_cap)
                ex_p = min(cap_price, H)
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                return ret, day_offset, "ret_cap"

        # Scheduled exit / extension
        if ds >= ex_date:
            if exit_mode == "C2" and not extended:
                ret_check = (pc_prev - ep) / ep if ep > 0 else 0
                if ret_check >= EXT_THR * r_pct:
                    ni = entry_idx + EXT_HOLD
                    ex_date = td[ni] if ni < len(td) else td[-1]
                    extended = True
                    trail_hi = mh
                else:
                    ex_p = O
                    ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                    return ret, day_offset, "time"
            else:
                ex_p = O
                reason = "ext_time" if extended else "time"
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                return ret, day_offset, reason

        # Trailing stop (extended C2)
        if extended and day_offset > hold:
            trail_hi = max(trail_hi, H)
            ts = trail_hi * (1 - TRAIL_PCT)
            if L <= ts:
                ex_p = ts
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                return ret, day_offset, "trailing"

        # Day3 early exit (C2)
        if exit_mode == "C2" and day_offset == 3:
            ret_check = (C - ep) / ep if ep > 0 else 0
            if ret_check < DAY3_THR * r_pct and C <= mc * 0.999:
                ex_p = C
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                return ret, day_offset, "day3"

        mc = max(mc, C)
        mh = max(mh, H)
        pc_prev = C

    # Fallback: exit at last known close
    ret = (pc_prev * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
    return ret, 40, "timeout"


# ══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 100)
    log("Deep Exit Analysis: C0 vs C2")
    log("=" * 100)

    log("\n[1/4] Loading data...")
    df = load_ohlcv()
    df = compute_indicators(df)
    mask = apply_layer1(df)
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="2021-01-01", end_date=END)
    tdays = list(sched.index.normalize())
    warmup = pd.Timestamp(START)
    pb = build_lookups(df, mask, tdays, warmup, TOP_K)
    hmm = load_hmm()

    # ══════════════════════════════════════════════════════════════════
    # A2) 2025 SANITY CHECK
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 100)
    log("A2) 2025 SANITY CHECK")
    log("=" * 100)

    for em, el in [("C0", "C0"), ("C2", "C2")]:
        t, e = run_bt(pb, exit_mode=em, hmm_lk=hmm,
                      start_dt="2025-01-02", end_dt="2025-12-31", ret_cap=2.0)
        m = compute_metrics(e["eq"])
        dr = e["eq"].pct_change().dropna()

        log(f"\n  --- HMM+{el} @200% Cap, 2025 ---")
        log(f"  Trade count:     {len(t)}")
        log(f"  Trade mean ret:  {t['ret'].mean():+.3f}%")
        log(f"  Trade median ret:{t['ret'].median():+.3f}%")
        log(f"  Trade std:       {t['ret'].std():.3f}%")
        log(f"  Trade WR:        {(t['ret']>0).mean()*100:.1f}%")
        log(f"  Portfolio CAGR:  {m['cagr']:+.2f}%")
        log(f"  Portfolio Sharpe:{m['sharpe']:+.3f}")
        log(f"  Portfolio MaxDD: {m['max_dd']:+.2f}%")
        log(f"  Daily ret mean:  {dr.mean()*100:+.4f}%")
        log(f"  Daily ret std:   {dr.std()*100:.4f}%")
        log(f"  Daily ret median:{dr.median()*100:+.4f}%")
        log(f"  Positive days:   {(dr>0).sum()}/{len(dr)} ({(dr>0).mean()*100:.1f}%)")

        # Diagnose: why trade mean>0 but CAGR may differ
        if not t.empty:
            avg_days = t["days"].mean()
            med_days = t["days"].median()
            log(f"\n  Avg hold days:   {avg_days:.1f}")
            log(f"  Med hold days:   {med_days:.1f}")

            # Invested vs idle
            total_trading_days = len(dr)
            avg_positions = e["n"].mean()
            log(f"  Avg open positions: {avg_positions:.1f}")
            log(f"  Trading days:    {total_trading_days}")

            # Monthly breakdown
            log(f"\n  Monthly breakdown:")
            t["month"] = t["entry"].str[:7]
            for mo, grp in t.groupby("month"):
                n = len(grp)
                ar = grp["ret"].mean()
                wr = (grp["ret"] > 0).mean() * 100
                pnl = grp["pnl"].sum()
                log(f"    {mo}: N={n:>3d} AvgRet={ar:>+6.2f}% WR={wr:>5.1f}% PnL=${pnl:>+10,.0f}")

            # Worst 10 trades
            log(f"\n  Worst 10 trades:")
            worst = t.nsmallest(10, "pnl")
            for _, r in worst.iterrows():
                log(f"    {r['ticker']:>6s} {r['entry']} ret={r['ret']:>+8.2f}% "
                    f"pnl=${r['pnl']:>+10,.0f} days={r['days']} reason={r['reason']}")

            # Best 10 trades
            log(f"\n  Best 10 trades:")
            best = t.nlargest(10, "pnl")
            for _, r in best.iterrows():
                log(f"    {r['ticker']:>6s} {r['entry']} ret={r['ret']:>+8.2f}% "
                    f"pnl=${r['pnl']:>+10,.0f} days={r['days']} reason={r['reason']}")

    # ══════════════════════════════════════════════════════════════════
    # B1) PAIRED SAME-ENTRY COMPARISON
    # ══════════════════════════════════════════════════════════════════
    log("\n\n" + "=" * 100)
    log("B1) PAIRED SAME-ENTRY COMPARISON (same signal → C0 vs C2 exit)")
    log("=" * 100)

    log("\n  Simulating paired trades (full period)...")
    paired = simulate_paired_trades(pb, hmm, ret_cap=2.0)
    log(f"  Total paired trades: {len(paired)}")

    periods = [
        ("2022", "2022-03-01", "2022-12-31"),
        ("2023", "2023-01-03", "2023-12-31"),
        ("2024", "2024-01-02", "2024-12-31"),
        ("2025", "2025-01-02", "2025-12-31"),
        ("Full", "2022-03-01", "2025-12-31"),
    ]

    log(f"\n  {'Period':>8s}  {'N':>5s}  {'mean_d':>8s}  {'med_d':>8s}  {'std_d':>8s}  "
        f"{'C2wins':>7s}  {'P1':>7s}  {'P5':>7s}  {'P25':>7s}  {'P75':>7s}  {'P95':>7s}  {'P99':>7s}  "
        f"{'t-stat':>7s}  {'p-val':>8s}  {'Sig':>4s}")
    log(f"  {'-'*120}")

    for pn, ps, pe in periods:
        sub = paired[(paired["entry_date"] >= ps) & (paired["entry_date"] <= pe)]
        if sub.empty:
            continue
        d = sub["delta"]
        n = len(d)
        c2_wins = (d > 0).mean() * 100
        t_stat, p_val = sp_stats.ttest_1samp(d, 0) if n > 1 else (0, 1)
        # Wilcoxon signed-rank (paired non-parametric)
        try:
            w_stat, w_p = sp_stats.wilcoxon(d[d != 0])
            # One-sided: is median > 0?
            if d.median() > 0:
                w_p_one = w_p / 2
            else:
                w_p_one = 1 - w_p / 2
        except Exception:
            w_p_one = 1.0

        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        log(f"  {pn:>8s}  {n:>5d}  {d.mean():>+7.2f}%  {d.median():>+7.2f}%  {d.std():>7.2f}%  "
            f"{c2_wins:>6.1f}%  {d.quantile(0.01):>+6.1f}%  {d.quantile(0.05):>+6.1f}%  "
            f"{d.quantile(0.25):>+6.1f}%  {d.quantile(0.75):>+6.1f}%  "
            f"{d.quantile(0.95):>+6.1f}%  {d.quantile(0.99):>+6.1f}%  "
            f"{t_stat:>7.2f}  {p_val:>8.4f}  {sig:>4s}")

    # Tail analysis
    log(f"\n  Tail Analysis (Full Period):")
    d = paired["delta"]
    log(f"    C2 avoids big losses (delta > 0 when C0 ret < -20%): "
        f"{(paired[paired['ret_c0'] < -20]['delta'] > 0).mean()*100:.1f}%")
    log(f"    C2 sacrifices big wins (delta < 0 when C0 ret > +50%): "
        f"{(paired[paired['ret_c0'] > 50]['delta'] < 0).mean()*100:.1f}%")

    # Show where delta comes from
    log(f"\n  Delta contribution by C0 return bucket:")
    bins = [(-999, -20), (-20, -5), (-5, 0), (0, 5), (5, 20), (20, 50), (50, 200), (200, 9999)]
    labels = ["<-20%", "-20 to -5%", "-5 to 0%", "0 to +5%", "+5 to +20%",
              "+20 to +50%", "+50 to +200%", ">+200%"]
    log(f"    {'Bucket':>15s}  {'N':>5s}  {'mean_d':>8s}  {'med_d':>8s}  {'C2wins':>7s}  "
        f"{'avgC0':>8s}  {'avgC2':>8s}")
    log(f"    {'-'*70}")
    for (lo, hi), label in zip(bins, labels):
        sub = paired[(paired["ret_c0"] >= lo) & (paired["ret_c0"] < hi)]
        if sub.empty:
            continue
        dd = sub["delta"]
        log(f"    {label:>15s}  {len(sub):>5d}  {dd.mean():>+7.2f}%  {dd.median():>+7.2f}%  "
            f"{(dd>0).mean()*100:>6.1f}%  {sub['ret_c0'].mean():>+7.2f}%  {sub['ret_c2'].mean():>+7.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # B2) PAIRED BY HMM STATE BUCKET
    # ══════════════════════════════════════════════════════════════════
    log("\n\n" + "=" * 100)
    log("B2) PAIRED COMPARISON BY HMM STATE")
    log("=" * 100)

    for state_label in ["risk_on", "transition", "risk_off"]:
        sub = paired[paired["hmm_state"] == state_label]
        if sub.empty:
            continue
        d = sub["delta"]
        n = len(d)
        t_stat, p_val = sp_stats.ttest_1samp(d, 0) if n > 1 else (0, 1)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        log(f"\n  {state_label:>12s}: N={n:>5d}  mean_d={d.mean():>+6.2f}%  med_d={d.median():>+6.2f}%  "
            f"std={d.std():>6.2f}%  C2wins={((d>0).mean()*100):>5.1f}%  "
            f"t={t_stat:>+6.2f}  p={p_val:.4f} {sig}")
        log(f"    C0: mean={sub['ret_c0'].mean():>+6.2f}% WR={(sub['ret_c0']>0).mean()*100:.1f}%  "
            f"C2: mean={sub['ret_c2'].mean():>+6.2f}% WR={(sub['ret_c2']>0).mean()*100:.1f}%")
        log(f"    C0 days: {sub['days_c0'].mean():.1f}  C2 days: {sub['days_c2'].mean():.1f}")

    # By realized volatility (atr_pct as proxy)
    log(f"\n  By Volatility (ATR% quartile):")
    paired["vol_q"] = pd.qcut(paired["atr_pct"], 4, labels=["Low", "Med-Low", "Med-High", "High"])
    for vq in ["Low", "Med-Low", "Med-High", "High"]:
        sub = paired[paired["vol_q"] == vq]
        if sub.empty:
            continue
        d = sub["delta"]
        n = len(d)
        t_stat, p_val = sp_stats.ttest_1samp(d, 0) if n > 1 else (0, 1)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        log(f"    {vq:>10s}: N={n:>5d}  mean_d={d.mean():>+6.2f}%  med_d={d.median():>+6.2f}%  "
            f"C2wins={((d>0).mean()*100):>5.1f}%  t={t_stat:>+6.2f} p={p_val:.4f} {sig}")

    # Cross: HMM state × volatility
    log(f"\n  HMM State x Volatility (mean delta):")
    log(f"    {'':>12s}  {'Low':>8s}  {'Med-Low':>8s}  {'Med-High':>8s}  {'High':>8s}")
    log(f"    {'-'*50}")
    for state in ["risk_on", "transition", "risk_off"]:
        vals = []
        for vq in ["Low", "Med-Low", "Med-High", "High"]:
            sub = paired[(paired["hmm_state"] == state) & (paired["vol_q"] == vq)]
            if len(sub) > 0:
                vals.append(f"{sub['delta'].mean():>+7.2f}%")
            else:
                vals.append(f"{'N/A':>8s}")
        log(f"    {state:>12s}  {'  '.join(vals)}")

    # ══════════════════════════════════════════════════════════════════
    # C) CAPITAL EFFICIENCY
    # ══════════════════════════════════════════════════════════════════
    log("\n\n" + "=" * 100)
    log("C) CAPITAL EFFICIENCY METRICS")
    log("=" * 100)

    log(f"\n  {'Period':>8s}  {'Exit':>4s}  {'N':>5s}  {'AvgDays':>8s}  {'MedDays':>8s}  "
        f"{'Ret/Day':>8s}  {'Turnover':>9s}  {'AvgPos':>7s}  {'Utiliz%':>8s}")
    log(f"  {'-'*80}")

    for pn, ps, pe in periods:
        for em, el in [("C0", "C0"), ("C2", "C2")]:
            t, e = run_bt(pb, exit_mode=em, hmm_lk=hmm,
                          start_dt=ps, end_dt=pe, ret_cap=2.0)
            n = len(t)
            if n == 0:
                continue
            avg_d = t["days"].mean()
            med_d = t["days"].median()
            ret_per_day = t["ret"].mean() / avg_d if avg_d > 0 else 0
            total_days = len(e)
            avg_pos = e["n"].mean()
            # Utilization: fraction of capital-days invested
            # Approximate: avg_positions * avg_hold / (total_days * max_positions)
            utiliz = avg_pos / TOP_K * 100  # rough %
            turnover = n  # total trades per year
            log(f"  {pn:>8s}  {el:>4s}  {n:>5d}  {avg_d:>7.1f}d  {med_d:>7.1f}d  "
                f"{ret_per_day:>+7.3f}%  {n:>9d}  {avg_pos:>6.1f}  {utiliz:>7.1f}%")

    # Cost sensitivity
    log(f"\n  Cost Sensitivity (Full Period, @200% Cap):")
    log(f"    {'Cost':>6s}  {'C0 Sharpe':>10s}  {'C0 CAGR':>10s}  {'C2 Sharpe':>10s}  {'C2 CAGR':>10s}  "
        f"{'C0 N':>6s}  {'C2 N':>6s}")
    log(f"    {'-'*65}")
    for bps in [0, 5, 10, 20, 50]:
        t0, e0 = run_bt(pb, exit_mode="C0", hmm_lk=hmm, cost_bps=bps, ret_cap=2.0)
        t2, e2 = run_bt(pb, exit_mode="C2", hmm_lk=hmm, cost_bps=bps, ret_cap=2.0)
        m0 = compute_metrics(e0["eq"])
        m2 = compute_metrics(e2["eq"])
        log(f"    {bps:>4d}bp  {m0['sharpe']:>+10.3f}  {m0['cagr']:>+9.2f}%  "
            f"{m2['sharpe']:>+10.3f}  {m2['cagr']:>+9.2f}%  {len(t0):>6d}  {len(t2):>6d}")

    # Capital recovery speed
    log(f"\n  Capital Recovery (time between exit and next entry, same slot):")
    for em, el in [("C0", "C0"), ("C2", "C2")]:
        t, _ = run_bt(pb, exit_mode=em, hmm_lk=hmm, ret_cap=2.0)
        if t.empty:
            continue
        # For each ticker, compute gap between exit and next entry of ANY ticker
        exits = t.sort_values("exit")
        gaps = []
        exit_dates = exits["exit"].values
        entry_dates = exits["entry"].values
        # Simple: gap = time between consecutive entries (proxy for capital reuse)
        if len(entry_dates) > 1:
            td_list = pb["td"]
            td_idx = pb["ti"]
            for i in range(1, min(len(entry_dates), 5000)):
                ei1 = td_idx.get(entry_dates[i-1], 0)
                ei2 = td_idx.get(entry_dates[i], 0)
                gaps.append(ei2 - ei1)
        if gaps:
            gaps = np.array(gaps)
            log(f"    {el}: avg gap between entries={gaps.mean():.1f}d  "
                f"median={np.median(gaps):.1f}d  "
                f"same-day entries={(gaps==0).mean()*100:.0f}%")

    # ══════════════════════════════════════════════════════════════════
    # D) STATE-MACHINE EXIT RULE
    # ══════════════════════════════════════════════════════════════════
    log("\n\n" + "=" * 100)
    log("D) STATE-MACHINE EXIT: C0 in risk-on, C2 in risk-off/transition")
    log("=" * 100)

    # Implement hybrid: use HMM state at ENTRY to decide exit mode
    # This requires a modified backtest
    def run_bt_hybrid(pb, hmm_lk, hmm_lo=0.25, hmm_hi=0.50,
                      crisis_threshold=0.50, cost_bps=20, ret_cap=None):
        """Hybrid: C0 for risk-on entries, C2 for risk-off/transition entries."""
        pl, il, sbd = pb["pl"], pb["il"], pb["sbd"]
        td, ti, ws = pb["td"], pb["ti"], pb["ws"]
        cost = cost_bps / 10_000.0
        r_pct = R_PCT

        positions = []
        trades = []
        cash = CAPITAL
        equity = []
        open_tk = set()

        for ds in td:
            if ds < ws:
                equity.append({"date": ds, "eq": CAPITAL, "n": 0})
                continue

            keep = []
            for p in positions:
                tk = p["ticker"]
                bar = pl.get((ds, tk))
                if not bar or bar["Open"] <= 0:
                    keep.append(p); continue

                O, H, L, C = bar["Open"], bar["High"], bar["Low"], bar["Close"]
                p["days"] += 1
                ep = p["ep"]
                em = p["exit_mode"]
                exited = False; ex_p = 0; ex_r = ""

                # Return cap
                if ret_cap is not None and not exited:
                    ret_now = (C - ep) / ep if ep > 0 else 0
                    if ret_now >= ret_cap:
                        cap_price = ep * (1 + ret_cap)
                        ex_p = min(cap_price, H)
                        ex_r = "ret_cap"; exited = True

                # Scheduled exit / extension
                if not exited and ds >= p["ex_date"]:
                    if em == "C2" and not p.get("ext", False):
                        ret = (p["pc"] - ep) / ep if ep > 0 else 0
                        if ret >= EXT_THR * r_pct:
                            ei2 = ti.get(p["entry_d"], 0)
                            ni = ei2 + EXT_HOLD
                            p["ex_date"] = td[ni] if ni < len(td) else td[-1]
                            p["ext"] = True
                            p["trail_hi"] = p.get("mh", H)
                        else:
                            ex_p = O; ex_r = "time"; exited = True
                    else:
                        ex_p = O
                        ex_r = "ext_time" if p.get("ext", False) else "time"
                        exited = True

                # Trailing (extended C2)
                if not exited and p.get("ext", False) and p["days"] > HOLD:
                    p["trail_hi"] = max(p.get("trail_hi", H), H)
                    ts = p["trail_hi"] * (1 - TRAIL_PCT)
                    if L <= ts:
                        ex_p = ts; ex_r = "trailing"; exited = True

                # Day3 (C2 only)
                if not exited and em == "C2" and p["days"] == 3:
                    ret = (C - ep) / ep if ep > 0 else 0
                    mc = p.get("mc", ep)
                    if ret < DAY3_THR * r_pct and C <= mc * 0.999:
                        ex_p = C; ex_r = "day3"; exited = True

                p["mc"] = max(p.get("mc", ep), C)
                p["mh"] = max(p.get("mh", ep), H)
                p["pc"] = C

                if exited:
                    proc = p["sh"] * ex_p * (1 - cost)
                    pnl = proc - p["$"]
                    ret = pnl / p["$"] * 100 if p["$"] > 0 else 0
                    trades.append({
                        "ticker": tk, "sig": p["sig"], "entry": p["entry_d"],
                        "exit": ds, "ep": ep, "ex_p": ex_p, "sh": p["sh"],
                        "$": p["$"], "pnl": pnl, "ret": ret, "reason": ex_r,
                        "days": p["days"], "ext": p.get("ext", False),
                        "exit_mode": em,
                    })
                    cash += proc
                    open_tk.discard(tk)
                else:
                    keep.append(p)

            positions = keep

            # Entries
            idx = ti.get(ds)
            if idx is not None and idx > 0:
                prev = td[idx - 1]
                pc = hmm_lk.get(prev, 0) if hmm_lk else 0
                if np.isnan(pc):
                    pc = 1.0
                if pc < hmm_lo:
                    top_k = 8; exposure = 1.0
                elif pc < hmm_hi:
                    top_k = 5; exposure = 0.60
                else:
                    top_k = 2; exposure = 0.30

                # Decide exit mode based on HMM state
                exit_mode = "C0" if pc < crisis_threshold else "C2"

                sigs = sbd.get(prev, [])[:top_k]
                valid = []
                for t, atr in sigs:
                    if t in open_tk: continue
                    b = pl.get((ds, t))
                    if not b or b["Open"] <= 0: continue
                    ind = il.get((prev, t), {})
                    g = ind.get("gap", 0) or 0
                    if abs(g) > 0.08: continue
                    valid.append((t, b["Open"], atr))

                deploy = cash * exposure
                if valid and deploy > 100:
                    per = deploy / len(valid)
                    if per > 50:
                        for tk, op, atr in valid:
                            bc = min(per * (1 + cost), cash)
                            sh = (bc / (1 + cost)) / op
                            ei2 = idx + HOLD
                            ed = td[ei2] if ei2 < len(td) else td[-1]
                            positions.append({
                                "ticker": tk, "entry_d": ds, "ep": op, "sh": sh,
                                "$": bc, "sig": prev, "atr_pct": atr,
                                "ex_date": ed, "R": op * r_pct, "cat_stop": 0,
                                "days": 0, "mc": op, "mh": op, "pc": op,
                                "ext": False, "trail_hi": 0,
                                "exit_mode": exit_mode,
                            })
                            cash -= bc
                            open_tk.add(tk)

            pv = sum(p["sh"] * pl.get((ds, p["ticker"]), {}).get("Close", p["ep"])
                     for p in positions)
            equity.append({"date": ds, "eq": cash + pv, "n": len(positions)})

        tdf = pd.DataFrame(trades)
        edf = pd.DataFrame(equity)
        if not edf.empty:
            edf["date"] = pd.to_datetime(edf["date"])
            edf = edf.set_index("date")
        return tdf, edf

    # Test different crisis thresholds for hybrid
    log(f"\n  Hybrid exit: C0 when p_crisis < threshold, C2 when >= threshold")
    log(f"  @200% cap, full period")
    log(f"\n  {'Threshold':>10s}  {'N':>5s}  {'Sharpe':>7s}  {'CAGR':>9s}  {'MaxDD':>9s}  "
        f"{'N_C0':>5s}  {'N_C2':>5s}  {'WR':>6s}")
    log(f"  {'-'*70}")

    # Baselines
    t0, e0 = run_bt(pb, exit_mode="C0", hmm_lk=hmm, ret_cap=2.0)
    m0 = compute_metrics(e0["eq"])
    t2, e2 = run_bt(pb, exit_mode="C2", hmm_lk=hmm, ret_cap=2.0)
    m2 = compute_metrics(e2["eq"])
    log(f"  {'Pure C0':>10s}  {len(t0):>5d}  {m0['sharpe']:>7.3f}  {m0['cagr']:>+8.2f}%  "
        f"{m0['max_dd']:>+8.2f}%  {len(t0):>5d}  {'0':>5s}  "
        f"{(t0['ret']>0).mean()*100 if not t0.empty else 0:>5.1f}%")
    log(f"  {'Pure C2':>10s}  {len(t2):>5d}  {m2['sharpe']:>7.3f}  {m2['cagr']:>+8.2f}%  "
        f"{m2['max_dd']:>+8.2f}%  {'0':>5s}  {len(t2):>5d}  "
        f"{(t2['ret']>0).mean()*100 if not t2.empty else 0:>5.1f}%")

    for threshold in [0.25, 0.35, 0.50, 0.65, 0.80]:
        th, eh = run_bt_hybrid(pb, hmm, crisis_threshold=threshold, ret_cap=2.0)
        mh = compute_metrics(eh["eq"])
        n_c0 = 0; n_c2 = 0
        if not th.empty and "exit_mode" in th.columns:
            n_c0 = (th["exit_mode"] == "C0").sum()
            n_c2 = (th["exit_mode"] == "C2").sum()
        wr = (th["ret"] > 0).mean() * 100 if not th.empty else 0
        log(f"  {f'p>={threshold:.2f}→C2':>10s}  {len(th):>5d}  {mh['sharpe']:>7.3f}  "
            f"{mh['cagr']:>+8.2f}%  {mh['max_dd']:>+8.2f}%  {n_c0:>5d}  {n_c2:>5d}  {wr:>5.1f}%")

    # Walk-forward hybrid
    log(f"\n  Walk-Forward: Hybrid (p>=0.50 -> C2) vs Pure C0 vs Pure C2")
    log(f"  @200% cap")

    folds = [
        ("Fold 1", "2024-01-22", "2024-12-31"),
        ("Fold 2", "2025-01-22", "2025-12-31"),
    ]
    for fn, vs, ve in folds:
        t0f, e0f = run_bt(pb, exit_mode="C0", hmm_lk=hmm, start_dt=vs, end_dt=ve, ret_cap=2.0)
        t2f, e2f = run_bt(pb, exit_mode="C2", hmm_lk=hmm, start_dt=vs, end_dt=ve, ret_cap=2.0)
        m0f = compute_metrics(e0f["eq"])
        m2f = compute_metrics(e2f["eq"])
        # For hybrid we can't easily pass start/end to run_bt_hybrid, so compute full then slice
        # Actually let's just report full period hybrid since the walk-forward train selected HMM+C0
        log(f"\n  {fn} ({vs} -> {ve}):")
        log(f"    Pure C0: Sharpe={m0f['sharpe']:>+.3f} CAGR={m0f['cagr']:>+.2f}% DD={m0f['max_dd']:>+.2f}%")
        log(f"    Pure C2: Sharpe={m2f['sharpe']:>+.3f} CAGR={m2f['cagr']:>+.2f}% DD={m2f['max_dd']:>+.2f}%")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: 2025 equity C0 vs C2
    _, e0_25 = run_bt(pb, exit_mode="C0", hmm_lk=hmm, start_dt="2025-01-02", end_dt="2025-12-31", ret_cap=2.0)
    _, e2_25 = run_bt(pb, exit_mode="C2", hmm_lk=hmm, start_dt="2025-01-02", end_dt="2025-12-31", ret_cap=2.0)
    ax = axes[0, 0]
    ax.plot(e0_25.index, e0_25["eq"], label="C0", linewidth=1.2)
    ax.plot(e2_25.index, e2_25["eq"], label="C2", linewidth=1.2)
    ax.set_title("2025: C0 vs C2 Equity @200% Cap")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 2: Delta histogram (full period)
    ax = axes[0, 1]
    d = paired["delta"].clip(-50, 50)
    ax.hist(d, bins=100, alpha=0.7, color="#2ca02c", edgecolor="none")
    ax.axvline(d.mean(), color="red", linestyle="--", label=f"mean={d.mean():+.2f}%")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Paired Delta (C2 - C0) Distribution")
    ax.set_xlabel("Delta %"); ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 3: Full period equity comparison
    _, e0_f = run_bt(pb, exit_mode="C0", hmm_lk=hmm, ret_cap=2.0)
    _, e2_f = run_bt(pb, exit_mode="C2", hmm_lk=hmm, ret_cap=2.0)
    _, eh_f = run_bt_hybrid(pb, hmm, crisis_threshold=0.50, ret_cap=2.0)
    ax = axes[1, 0]
    ax.plot(e0_f.index, e0_f["eq"], label="Pure C0", linewidth=1.2)
    ax.plot(e2_f.index, e2_f["eq"], label="Pure C2", linewidth=1.2)
    ax.plot(eh_f.index, eh_f["eq"], label="Hybrid (p>=0.5→C2)", linewidth=1.2, linestyle="--")
    ax.set_title("Full Period: C0 vs C2 vs Hybrid @200% Cap")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 4: Delta by HMM state (box plot)
    ax = axes[1, 1]
    states = ["risk_on", "transition", "risk_off"]
    data_by_state = [paired[paired["hmm_state"] == s]["delta"].clip(-50, 50) for s in states]
    bp = ax.boxplot(data_by_state, labels=states, patch_artist=True,
                    medianprops=dict(color="red"))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Delta (C2-C0) by HMM State")
    ax.set_ylabel("Delta %"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "exit_deep_analysis.png", dpi=150)
    plt.close(fig)
    log(f"\nSaved: {OUT_DIR / 'exit_deep_analysis.png'}")

    log("\nDone!")


if __name__ == "__main__":
    main()
