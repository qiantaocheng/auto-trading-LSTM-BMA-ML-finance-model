#!/usr/bin/env python
"""Version L Exit Strategy Test — Lottery Momentum Optimized

Test 1: Catastrophe Stop Width Scan (C0 exit, no TP)
  - Baseline C0 (no stop), Stop 3/4/5×ATR, Gap-down -15%/-20%

Test 2: HMM(0.25/0.5) + C2 Walk-Forward Validation
  - C2: Day3 weak exit (100%) + winner extension (15d, 10% trail)
  - R = 8% of entry price (not ATR)

Test 3: Full Period Combined (best stop + C2 + HMM)
"""
from __future__ import annotations
import argparse, sys, warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
RAW_PATH = Path("D:/trade/data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
HMM_PATH = Path("D:/trade/result/layer1_hmm/hmm_p_crisis_series.csv")
OUT_DIR  = Path("D:/trade/result/layer1_version_l")

CAPITAL    = 100_000.0
RF         = 0.04
START      = "2022-03-01"
END        = "2025-12-31"
PRICE_MAX  = 100.0
VOL_MIN    = 50_000
RVOL_MIN   = 1.5
RET_MIN    = 0.02
TOP_K      = 8
HOLD       = 7          # base hold days
R_PCT      = 0.08       # R = 8% of entry price
DAY3_THR   = 0.5        # Day3: exit if ret < 0.5R
EXT_THR    = 2.0        # extend if ret >= 2R at Day7
EXT_HOLD   = 15         # extended hold
TRAIL_PCT  = 0.10       # 10% trailing stop

import io, os
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ── Data Loading ────────────────────────────────────────────────────────
def load_ohlcv():
    log("  Reading parquet...")
    df = pd.read_parquet(RAW_PATH, columns=["date","ticker","Open","High","Low","Close","Volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date","ticker"]).sort_index()
    log(f"  {len(df):,} rows, {df.index.get_level_values('ticker').nunique()} tickers")
    return df

def compute_indicators(df):
    g = df.groupby(level="ticker", group_keys=False)
    log("  daily_return, RVOL, ATR14...")
    df["daily_return"] = g["Close"].pct_change()
    df["vol_20d_avg"] = g["Volume"].transform(lambda x: x.rolling(20,min_periods=15).mean().shift(1))
    df["rvol"] = df["Volume"] / df["vol_20d_avg"]
    df["prev_close"] = g["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["prev_close"]).abs()
    tr3 = (df["Low"]  - df["prev_close"]).abs()
    df["tr"] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    df["atr14"] = g["tr"].transform(lambda x: x.rolling(14,min_periods=10).mean())
    df["atr_pct"] = df["atr14"] / df["Close"] * 100
    df["next_open"] = g["Open"].shift(-1)
    df["opening_gap"] = (df["next_open"] - df["Close"]) / df["Close"]
    df["score"] = df["rvol"] * df["daily_return"]
    log(f"  Done: {len(df):,} rows")
    return df

def apply_layer1(df):
    return ((df["Close"]<PRICE_MAX)&(df["Volume"]>VOL_MIN)&
            (df["rvol"]>RVOL_MIN)&(df["daily_return"]>RET_MIN))

def build_lookups(df, mask, trading_days, warmup_end, top_k=8):
    log("  Building lookups (vectorized)...")
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    ds_arr = dates.strftime("%Y-%m-%d")
    keys = list(zip(ds_arr, tickers))
    o,h,l,c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    price_lk = {k:{"Open":o[i],"High":h[i],"Low":l[i],"Close":c[i]} for i,k in enumerate(keys)}
    atr_v = df["atr_pct"].values; gap_v = df["opening_gap"].values
    ind_lk = {k:{"atr_pct":atr_v[i],"gap":gap_v[i]} for i,k in enumerate(keys)}
    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds:i for i,ds in enumerate(td_strs)}
    ws = warmup_end.strftime("%Y-%m-%d")
    passed = df[mask].copy()
    avail = set(df.index.get_level_values("date").unique())
    sbd = {}
    for td in trading_days:
        tn = pd.Timestamp(td).normalize()
        if tn <= warmup_end or tn not in avail: continue
        ds = tn.strftime("%Y-%m-%d")
        try:
            dp = passed.loc[tn]
            if isinstance(dp, pd.Series):
                if isinstance(dp.name, str):
                    a = dp.get("atr_pct",5.0)
                    sbd[ds] = [(dp.name, a if pd.notna(a) else 5.0)]
            else:
                tk = dp.nlargest(top_k,"score")
                sbd[ds] = [(t, tk.loc[t,"atr_pct"] if pd.notna(tk.loc[t,"atr_pct"]) else 5.0)
                           for t in tk.index.tolist()]
        except KeyError: continue
    log(f"  Lookups: {len(price_lk):,} prices, {len(sbd)} signal days")
    return {"pl":price_lk,"il":ind_lk,"sbd":sbd,"td":td_strs,"ti":td_idx,"ws":ws}

def compute_metrics(eq, label=""):
    r = eq.pct_change().dropna(); n = len(r)
    if n < 10:
        return dict(label=label,cagr=0,sharpe=0,sortino=0,max_dd=0,calmar=0,total=0)
    tot = eq.iloc[-1]/eq.iloc[0]-1; yr = n/252
    cagr = (1+tot)**(1/yr)-1 if yr>0 else 0
    ex = r - RF/252
    sh = ex.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    ds = r[r<0]
    so = ex.mean()/ds.std()*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    dd = (eq-eq.cummax())/eq.cummax(); mdd = dd.min()
    ca = cagr/abs(mdd) if mdd!=0 else 0
    return dict(label=label,cagr=cagr*100,sharpe=sh,sortino=so,max_dd=mdd*100,calmar=ca,total=tot*100)

def load_hmm():
    df = pd.read_csv(HMM_PATH, parse_dates=["date"])
    lk = {row["date"].strftime("%Y-%m-%d"): row["p_crisis_smooth"] for _,row in df.iterrows()}
    log(f"  HMM: {len(lk)} days, p_crisis [{min(lk.values()):.4f}, {max(lk.values()):.4f}]")
    return lk

# ── Unified Backtest ────────────────────────────────────────────────────
def run_bt(pb, stop_mode="none", exit_mode="C0",
           hmm_lk=None, hmm_lo=0.25, hmm_hi=0.50,
           r_pct=R_PCT, cost_bps=20, gap_limit=0.08,
           start_dt=None, end_dt=None, ret_cap=None):
    """
    stop_mode: "none" | "atr3" | "atr4" | "atr5" | "gap15" | "gap20"
    exit_mode: "C0" (fixed 7d) | "C2" (Day3 exit + winner extend)
    ret_cap: if set, exit position when Close return >= ret_cap (e.g. 2.0 = 200%)
    """
    pl,il,sbd = pb["pl"],pb["il"],pb["sbd"]
    td,ti,ws  = pb["td"],pb["ti"],pb["ws"]

    # Parse stop config
    stop_atr = int(stop_mode[3:]) if stop_mode.startswith("atr") else 0
    gap_th   = -int(stop_mode[3:])/100.0 if stop_mode.startswith("gap") else 0

    start_s = start_dt or ws
    end_s   = end_dt or td[-1]
    cost    = cost_bps/10_000.0

    positions = []
    trades    = []
    cash      = CAPITAL
    equity    = []
    open_tk   = set()

    for ds in td:
        if ds < start_s:
            equity.append({"date":ds,"eq":CAPITAL,"n":0})
            continue

        # Stop entering after end_dt, but keep monitoring
        past_end = ds > end_s

        # ── Monitor positions ──
        keep = []
        for p in positions:
            tk = p["ticker"]
            bar = pl.get((ds, tk))
            if not bar or bar["Open"]<=0:
                keep.append(p); continue

            O,H,L,C = bar["Open"],bar["High"],bar["Low"],bar["Close"]
            p["days"] += 1
            ep = p["ep"]; R = p["R"]
            exited = False; ex_p = 0; ex_r = ""

            # 1) Catastrophe ATR stop
            if stop_atr > 0 and not exited:
                cs = p.get("cat_stop",0)
                if cs > 0:
                    if O <= cs:
                        ex_p=O; ex_r="cat_gap"; exited=True
                    elif L <= cs:
                        ex_p=cs; ex_r="cat_stop"; exited=True

            # 2) Catastrophe gap stop
            if gap_th < 0 and not exited:
                pc = p.get("pc", ep)
                if pc > 0 and (O-pc)/pc <= gap_th:
                    ex_p=O; ex_r="gap_crash"; exited=True

            # 2b) Return cap — exit if Close return >= cap
            if ret_cap is not None and not exited:
                ret_now = (C - ep) / ep if ep > 0 else 0
                if ret_now >= ret_cap:
                    cap_price = ep * (1 + ret_cap)
                    ex_p = min(cap_price, H)  # can't exceed High
                    ex_r = "ret_cap"; exited = True

            # 3) Scheduled exit / extension check
            if not exited and ds >= p["ex_date"]:
                if exit_mode=="C2" and not p.get("ext",False):
                    # Extension decision: use prev day Close
                    ret = (p["pc"]-ep)/ep if ep>0 else 0
                    if ret >= EXT_THR * r_pct:
                        ei = ti.get(p["entry_d"],0)
                        ni = ei + EXT_HOLD
                        p["ex_date"] = td[ni] if ni<len(td) else td[-1]
                        p["ext"] = True
                        p["trail_hi"] = p.get("mh", H)
                    else:
                        ex_p=O; ex_r="time"; exited=True
                else:
                    ex_p=O
                    ex_r = "ext_time" if p.get("ext",False) else "time"
                    exited=True

            # 4) Trailing stop (extended C2)
            if not exited and p.get("ext",False) and p["days"]>HOLD:
                p["trail_hi"] = max(p.get("trail_hi",H), H)
                ts = p["trail_hi"]*(1-TRAIL_PCT)
                if L <= ts:
                    ex_p=ts; ex_r="trailing"; exited=True

            # 5) Day3 early exit (C2, full exit)
            if not exited and exit_mode=="C2" and p["days"]==3:
                ret = (C-ep)/ep if ep>0 else 0
                mc = p.get("mc", ep)
                if ret < DAY3_THR*r_pct and C <= mc*0.999:
                    ex_p=C; ex_r="day3"; exited=True

            # Update tracking
            p["mc"] = max(p.get("mc",ep), C)
            p["mh"] = max(p.get("mh",ep), H)
            p["pc"] = C

            if exited:
                proc = p["sh"]*ex_p*(1-cost)
                pnl  = proc - p["$"]
                ret  = pnl/p["$"]*100 if p["$"]>0 else 0
                trades.append({
                    "ticker":tk, "sig":p["sig"], "entry":p["entry_d"],
                    "exit":ds, "ep":ep, "ex_p":ex_p, "sh":p["sh"],
                    "$":p["$"], "pnl":pnl, "ret":ret, "reason":ex_r,
                    "days":p["days"], "ext":p.get("ext",False),
                    "atr_pct":p["atr_pct"],
                })
                cash += proc
                open_tk.discard(tk)
            else:
                keep.append(p)

        positions = keep

        # ── Entries (only within period) ──
        if not past_end:
            idx = ti.get(ds)
            if idx is not None and idx > 0:
                prev = td[idx-1]
                top_k = TOP_K; exposure = 1.0
                if hmm_lk:
                    pc = hmm_lk.get(prev, 0)
                    if pc < hmm_lo:   top_k=8; exposure=1.0
                    elif pc < hmm_hi: top_k=5; exposure=0.60
                    else:             top_k=2; exposure=0.30

                sigs = sbd.get(prev,[])[:top_k]
                valid = []
                for t,atr in sigs:
                    if t in open_tk: continue
                    b = pl.get((ds,t))
                    if not b or b["Open"]<=0: continue
                    ind = il.get((prev,t),{})
                    g = ind.get("gap",0) or 0
                    if abs(g)>gap_limit: continue
                    valid.append((t, b["Open"], atr))

                deploy = cash * exposure
                if valid and deploy > 100:
                    per = deploy/len(valid)
                    if per > 50:
                        for tk,op,atr in valid:
                            bc = min(per*(1+cost), cash)
                            sh = (bc/(1+cost))/op
                            ei = idx + HOLD
                            ed = td[ei] if ei<len(td) else td[-1]
                            R = op * r_pct
                            cs = 0
                            if stop_atr > 0:
                                atr_d = op*atr/100.0
                                cs = op - stop_atr*atr_d
                                if cs<=0: cs=0
                            positions.append({
                                "ticker":tk,"entry_d":ds,"ep":op,"sh":sh,
                                "$":bc,"sig":prev,"atr_pct":atr,
                                "ex_date":ed,"R":R,"cat_stop":cs,
                                "days":0,"mc":op,"mh":op,"pc":op,
                                "ext":False,"trail_hi":0,
                            })
                            cash -= bc
                            open_tk.add(tk)

        # MTM
        pv = sum(p["sh"]*pl.get((ds,p["ticker"]),{}).get("Close",p["ep"])
                 for p in positions)
        equity.append({"date":ds,"eq":cash+pv,"n":len(positions)})

        # Early stop if past end and no positions
        if past_end and not positions:
            break

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity)
    if not edf.empty:
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf.set_index("date")
    return tdf, edf

# ── Main ────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []  # report accumulator

    log("="*80)
    log("Version L Exit Strategy Test — Lottery Momentum Optimized")
    log("="*80)

    # ── [1] Load & validate ──
    log("\n[1/6] Loading OHLCV data...")
    df = load_ohlcv()
    log("[2/6] Computing indicators...")
    df = compute_indicators(df)
    mask = apply_layer1(df)

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="2021-01-01", end_date=END)
    tdays = list(sched.index.normalize())
    warmup = pd.Timestamp(START)

    dates = df.index.get_level_values("date")
    log(f"  Data range: {dates.min().date()} → {dates.max().date()}")
    log(f"  Layer 1 passers: {mask.sum():,}")
    td_in_period = sum(1 for t in tdays if t >= warmup)
    log(f"  Trading days in test period: {td_in_period}")

    log("\n[3/6] Building lookups...")
    pb = build_lookups(df, mask, tdays, warmup, TOP_K)

    # ── C0 baseline validation ──
    log("\n  C0 Baseline Validation...")
    t0, e0 = run_bt(pb, stop_mode="none", exit_mode="C0")
    m0 = compute_metrics(e0["eq"], "C0")
    log(f"  C0: {len(t0)} trades, Sharpe={m0['sharpe']:.3f}, "
        f"CAGR={m0['cagr']:+.2f}%, MaxDD={m0['max_dd']:+.2f}%")

    if not t0.empty:
        log("  Sample trades:")
        for _,r in t0.head(5).iterrows():
            log(f"    {r['ticker']:>6s} entry={r['entry']} exit={r['exit']} "
                f"ep={r['ep']:.2f} xp={r['ex_p']:.2f} ret={r['ret']:+.2f}%")

    ok = abs(len(t0)-1015)<10 and abs(m0["sharpe"]-0.845)<0.1
    log(f"  [{'OK' if ok else 'FAIL'}] C0 baseline {'matches' if ok else 'MISMATCH'} expected "
        f"(~1015 trades, ~0.845 Sharpe)")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Catastrophe Stop Width Scan
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 1: Catastrophe Stop Width Scan (C0 exit, no TP)")
    log("  Target: stop trigger rate <10%, preserve right tail (max win ~1630%)")
    log("="*90)

    scenarios_1 = [
        ("none",  "C0 Baseline"),
        ("atr3",  "Stop 3×ATR"),
        ("atr4",  "Stop 4×ATR"),
        ("atr5",  "Stop 5×ATR"),
        ("gap15", "Gap ≤-15%"),
        ("gap20", "Gap ≤-20%"),
    ]

    r1 = []
    for sm, label in scenarios_1:
        t, e = run_bt(pb, stop_mode=sm, exit_mode="C0")
        m = compute_metrics(e["eq"], label)
        n = len(t)
        if not t.empty:
            sc = t["reason"].str.contains("cat_|gap_").sum()
            sr = sc/n*100
            ar = t["ret"].mean(); mr = t["ret"].median()
            wr = (t["ret"]>0).mean()*100
            mx = t["ret"].max(); mn = t["ret"].min()
        else:
            sc=sr=ar=mr=wr=mx=mn=0
        r1.append(dict(label=label,sm=sm,n=n,sc=sc,sr=sr,ar=ar,mr=mr,
                       wr=wr,sh=m["sharpe"],cagr=m["cagr"],mdd=m["max_dd"],
                       mx=mx,mn=mn,so=m["sortino"],ca=m["calmar"],tot=m["total"]))
        log(f"  {label:>20s}: N={n:>5d} StopRate={sr:>5.1f}% Sh={m['sharpe']:>6.3f} "
            f"CAGR={m['cagr']:>+7.2f}% MaxDD={m['max_dd']:>+7.2f}% "
            f"MaxWin={mx:>+8.1f}% MaxLoss={mn:>+7.1f}%")

    lines.append("="*110)
    lines.append("TEST 1: Catastrophe Stop Width Scan (C0 exit, no TP)")
    lines.append("="*110)
    hdr = (f"{'Scenario':>20s}  {'N':>5s}  {'StopRate':>8s}  {'AvgRet':>8s}  {'MedRet':>8s}  "
           f"{'WR':>6s}  {'Sharpe':>7s}  {'CAGR':>9s}  {'MaxDD':>9s}  {'MaxWin':>10s}  {'MaxLoss':>9s}")
    lines.append(hdr); lines.append("-"*len(hdr))
    for r in r1:
        lines.append(
            f"{r['label']:>20s}  {r['n']:>5d}  {r['sr']:>7.1f}%  {r['ar']:>+7.2f}%  "
            f"{r['mr']:>+7.2f}%  {r['wr']:>5.1f}%  {r['sh']:>7.3f}  {r['cagr']:>+8.2f}%  "
            f"{r['mdd']:>+8.2f}%  {r['mx']:>+9.1f}%  {r['mn']:>+8.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: HMM(0.25/0.5) + C2 Walk-Forward
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 2: HMM(0.25/0.5) + C2 Walk-Forward Validation")
    log(f"  C2: Day3 exit (<{DAY3_THR*R_PCT*100:.1f}%, no new high, 100% out) + "
        f"Extension (>={EXT_THR*R_PCT*100:.0f}% → {EXT_HOLD}d, {TRAIL_PCT*100:.0f}% trail)")
    log(f"  R = {R_PCT*100:.0f}% of entry price")
    log("="*90)

    log("  Loading HMM p_crisis series...")
    hmm = load_hmm()

    folds = [
        ("Fold 1","2022-03-01","2023-12-31","2024-01-22","2024-12-31"),
        ("Fold 2","2022-03-01","2024-12-31","2025-01-22","2025-12-31"),
        ("Fold 3","2023-01-01","2024-12-31","2025-01-22","2025-12-31"),
    ]

    r2 = []
    for fn, ts, te, vs, ve in folds:
        log(f"\n  --- {fn}: Train {ts}→{te}, Test {vs}→{ve} ---")

        # HMM + C0 train/test
        _,eq_c0_tr = run_bt(pb,exit_mode="C0",hmm_lk=hmm,start_dt=ts,end_dt=te)
        m_c0_tr = compute_metrics(eq_c0_tr["eq"],"")
        t_c0_te,eq_c0_te = run_bt(pb,exit_mode="C0",hmm_lk=hmm,start_dt=vs,end_dt=ve)
        m_c0_te = compute_metrics(eq_c0_te["eq"],"")

        # HMM + C2 train/test
        _,eq_c2_tr = run_bt(pb,exit_mode="C2",hmm_lk=hmm,start_dt=ts,end_dt=te)
        m_c2_tr = compute_metrics(eq_c2_tr["eq"],"")
        t_c2_te,eq_c2_te = run_bt(pb,exit_mode="C2",hmm_lk=hmm,start_dt=vs,end_dt=ve)
        m_c2_te = compute_metrics(eq_c2_te["eq"],"")

        log(f"    HMM+C0: Train Sh={m_c0_tr['sharpe']:.3f} | "
            f"Test Sh={m_c0_te['sharpe']:.3f} DD={m_c0_te['max_dd']:+.1f}% N={len(t_c0_te)}")
        log(f"    HMM+C2: Train Sh={m_c2_tr['sharpe']:.3f} | "
            f"Test Sh={m_c2_te['sharpe']:.3f} DD={m_c2_te['max_dd']:+.1f}% N={len(t_c2_te)}")

        if not t_c2_te.empty:
            ext_n = t_c2_te["ext"].sum() if "ext" in t_c2_te.columns else 0
            rc = t_c2_te["reason"].value_counts().to_dict()
            log(f"    C2 reasons: {rc}")
            log(f"    Extended: {ext_n} trades")
            if ext_n > 0:
                ext_t = t_c2_te[t_c2_te["ext"]==True]
                log(f"    Extended avg ret: {ext_t['ret'].mean():+.2f}%, avg days: {ext_t['days'].mean():.1f}")

        r2.append(dict(fn=fn,
            c0_tr=m_c0_tr["sharpe"],c0_te=m_c0_te["sharpe"],c0_dd=m_c0_te["max_dd"],
            c0_cagr=m_c0_te["cagr"],c0_n=len(t_c0_te),
            c2_tr=m_c2_tr["sharpe"],c2_te=m_c2_te["sharpe"],c2_dd=m_c2_te["max_dd"],
            c2_cagr=m_c2_te["cagr"],c2_n=len(t_c2_te)))

    lines.append("\n"+"="*110)
    lines.append(f"TEST 2: HMM(0.25/0.5) + C2 Walk-Forward (R={R_PCT*100:.0f}% of entry)")
    lines.append(f"  C2: Day3 (<{DAY3_THR*R_PCT*100:.1f}%, no new high, 100% out) + "
                 f"Extend (>={EXT_THR*R_PCT*100:.0f}% → {EXT_HOLD}d, {TRAIL_PCT*100:.0f}% trail)")
    lines.append("="*110)
    hdr2 = (f"{'Fold':>8s}  {'C0 Train':>9s}  {'C0 Test':>8s}  {'C0 CAGR':>9s}  {'C0 DD':>8s}  {'C0 N':>5s}  "
            f"{'C2 Train':>9s}  {'C2 Test':>8s}  {'C2 CAGR':>9s}  {'C2 DD':>8s}  {'C2 N':>5s}")
    lines.append(hdr2); lines.append("-"*len(hdr2))
    for r in r2:
        lines.append(
            f"{r['fn']:>8s}  {r['c0_tr']:>9.3f}  {r['c0_te']:>8.3f}  {r['c0_cagr']:>+8.2f}%  "
            f"{r['c0_dd']:>+7.2f}%  {r['c0_n']:>5d}  {r['c2_tr']:>9.3f}  {r['c2_te']:>8.3f}  "
            f"{r['c2_cagr']:>+8.2f}%  {r['c2_dd']:>+7.2f}%  {r['c2_n']:>5d}")

    # ══════════════════════════════════════════════════════════════════════
    # OUTLIER ANALYSIS — trade return distribution
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("OUTLIER ANALYSIS: Trade Return Distribution")
    log("="*90)

    # Get trades for key strategies
    outlier_scenarios = [
        ("none","C0",None,   "C0 Baseline"),
        ("none","C0",hmm,    "HMM + C0"),
        ("none","C2",hmm,    "HMM + C2"),
    ]
    for sm,em,h,label in outlier_scenarios:
        t,_ = run_bt(pb, stop_mode=sm, exit_mode=em, hmm_lk=h)
        if t.empty: continue
        rets = t["ret"]
        pnls = t["pnl"]
        total_pnl = pnls.sum()
        log(f"\n  {label} ({len(t)} trades):")
        log(f"    Percentiles: P1={rets.quantile(0.01):+.1f}% P5={rets.quantile(0.05):+.1f}% "
            f"P25={rets.quantile(0.25):+.1f}% P50={rets.quantile(0.50):+.1f}% "
            f"P75={rets.quantile(0.75):+.1f}% P95={rets.quantile(0.95):+.1f}% "
            f"P99={rets.quantile(0.99):+.1f}%")
        # Top 5 trades by PnL
        top5 = t.nlargest(5,"pnl")
        top5_pnl = top5["pnl"].sum()
        top10_pnl = t.nlargest(10,"pnl")["pnl"].sum()
        top20_pnl = t.nlargest(20,"pnl")["pnl"].sum()
        log(f"    Total PnL: ${total_pnl:,.0f}")
        log(f"    Top 5 trades PnL: ${top5_pnl:,.0f} ({top5_pnl/total_pnl*100:.1f}% of total)")
        log(f"    Top 10 trades PnL: ${top10_pnl:,.0f} ({top10_pnl/total_pnl*100:.1f}%)")
        log(f"    Top 20 trades PnL: ${top20_pnl:,.0f} ({top20_pnl/total_pnl*100:.1f}%)")
        log(f"    Trades >200%: {(rets>200).sum()} | >500%: {(rets>500).sum()} | >1000%: {(rets>1000).sum()}")
        log(f"    Top 5 trades:")
        for _,r in top5.iterrows():
            log(f"      {r['ticker']:>6s} {r['entry']} ret={r['ret']:>+8.1f}% "
                f"pnl=${r['pnl']:>+10,.0f} days={r['days']} reason={r['reason']}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Capped Return Comparison (remove outlier effect)
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 3: Return-Capped Portfolio Comparison")
    log("  Re-run with position return caps at 100%, 200%, 500%, uncapped")
    log("="*90)

    caps = [1.0, 2.0, 5.0, None]  # 100%, 200%, 500%, uncapped
    cap_labels = ["Cap 100%", "Cap 200%", "Cap 500%", "Uncapped"]

    strats = [
        ("none","C0",None,   "C0"),
        ("none","C0",hmm,    "HMM+C0"),
        ("none","C2",hmm,    "HMM+C2"),
        ("atr4","C2",hmm,    "HMM+C2+4xATR"),
    ]

    r3 = []
    for sm,em,h,slabel in strats:
        for cap, clabel in zip(caps, cap_labels):
            t, e = run_bt(pb, stop_mode=sm, exit_mode=em, hmm_lk=h, ret_cap=cap)
            m = compute_metrics(e["eq"], "")
            n = len(t)
            mx = t["ret"].max() if not t.empty else 0
            mn = t["ret"].min() if not t.empty else 0
            wr = (t["ret"]>0).mean()*100 if not t.empty else 0
            ext = t["ext"].sum() if not t.empty and "ext" in t.columns else 0
            r3.append(dict(strat=slabel,cap=clabel,n=n,sh=m["sharpe"],cagr=m["cagr"],
                           mdd=m["max_dd"],so=m["sortino"],ca=m["calmar"],mx=mx,wr=wr,ext=ext))
            log(f"  {slabel:>12s} {clabel:>10s}: N={n:>5d} Sh={m['sharpe']:>6.3f} "
                f"CAGR={m['cagr']:>+8.2f}% MaxDD={m['max_dd']:>+7.2f}% MaxWin={mx:>+8.1f}%")
        log("")  # blank line between strategies

    lines.append("\n"+"="*110)
    lines.append("TEST 3: Return-Capped Portfolio Comparison (isolate outlier impact)")
    lines.append("  Caps: exit position when unrealized return reaches cap")
    lines.append("="*110)
    hdr3 = (f"{'Strategy':>12s}  {'Cap':>10s}  {'N':>5s}  {'Sharpe':>7s}  {'Sortino':>8s}  "
            f"{'CAGR':>9s}  {'MaxDD':>9s}  {'Calmar':>7s}  {'WR':>6s}  {'MaxWin':>10s}")
    lines.append(hdr3); lines.append("-"*len(hdr3))
    for r in r3:
        lines.append(
            f"{r['strat']:>12s}  {r['cap']:>10s}  {r['n']:>5d}  {r['sh']:>7.3f}  "
            f"{r['so']:>8.3f}  {r['cagr']:>+8.2f}%  {r['mdd']:>+8.2f}%  {r['ca']:>7.3f}  "
            f"{r['wr']:>5.1f}%  {r['mx']:>+9.1f}%")

    # ── Save report ──
    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "version_l_results.txt").write_text(report)

    # ── Equity plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (sm,em,h,slabel) in zip(axes.flat, strats):
        for cap, clabel, color in zip(caps, cap_labels,
                                       ["#d62728","#ff7f0e","#2ca02c","#1f77b4"]):
            _, e = run_bt(pb, stop_mode=sm, exit_mode=em, hmm_lk=h, ret_cap=cap)
            ax.plot(e.index, e["eq"].values, label=clabel, linewidth=1.2, color=color)
        ax.set_title(slabel, fontsize=11)
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Return Cap Sensitivity — Portfolio Equity", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_capped.png", dpi=150)
    plt.close(fig)

    log(f"\nSaved: {OUT_DIR/'version_l_results.txt'}")
    log("Done!")

if __name__ == "__main__":
    main()
