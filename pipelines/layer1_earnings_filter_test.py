#!/usr/bin/env python
"""Layer 1 Earnings Filter Test — Add fundamental filter to HMM+C2

Strategy: Run HMM+C2 first (no earnings filter) to identify which tickers
actually get traded. Then fetch Polygon financials ONLY for those tickers
(much smaller set). Then re-run with earnings filter applied as the LAST
filter in the entry pipeline.

Usage:
  python D:\trade\pipelines\layer1_earnings_filter_test.py --polygon-key <KEY>
  python D:\trade\pipelines\layer1_earnings_filter_test.py --polygon-key <KEY> --fetch-only
  python D:\trade\pipelines\layer1_earnings_filter_test.py --polygon-key <KEY> --skip-fetch
"""
from __future__ import annotations
import argparse, json, sys, time, warnings, io, os
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
RAW_PATH  = Path("D:/trade/data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
HMM_PATH  = Path("D:/trade/result/layer1_hmm/hmm_p_crisis_series.csv")
OUT_DIR   = Path("D:/trade/result/layer1_earnings_filter")
CACHE_DIR = OUT_DIR / "earnings_cache"

CAPITAL    = 100_000.0
RF         = 0.04
START      = "2022-03-01"
END        = "2025-12-31"
PRICE_MAX  = 100.0
VOL_MIN    = 50_000
RVOL_MIN   = 1.5
RET_MIN    = 0.02
TOP_K      = 8
HOLD       = 7
R_PCT      = 0.08
DAY3_THR   = 0.5
EXT_THR    = 2.0
EXT_HOLD   = 15
TRAIL_PCT  = 0.10

POLYGON_BASE = "https://api.polygon.io"
RATE_LIMIT_S = 1.5    # financials endpoint allows ~30 calls/min on free tier

if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ── Data Loading (reused from version_l_test.py) ─────────────────────
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

# ── Polygon Financials Fetch ──────────────────────────────────────────
def fetch_financials(ticker: str, api_key: str) -> list[dict]:
    """Fetch quarterly financials from Polygon. Returns raw results list."""
    cache_file = CACHE_DIR / f"{ticker}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = (f"{POLYGON_BASE}/vX/reference/financials?"
           f"ticker={ticker}&timeframe=quarterly&order=desc&limit=20&apiKey={api_key}")
    try:
        req = Request(url, headers={"User-Agent": "PythonScript/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("results", [])
        with open(cache_file, "w") as f:
            json.dump(results, f)
        return results
    except HTTPError as e:
        if e.code == 429:
            log(f"    Rate limited on {ticker}, waiting 60s...")
            time.sleep(60)
            return fetch_financials(ticker, api_key)
        log(f"    HTTP {e.code} for {ticker}")
        with open(cache_file, "w") as f:
            json.dump([], f)
        return []
    except (URLError, TimeoutError) as e:
        log(f"    Error fetching {ticker}: {e}")
        with open(cache_file, "w") as f:
            json.dump([], f)
        return []

def extract_eps_revenue(result: dict) -> dict:
    """Extract EPS and revenue from a single Polygon financials result."""
    fin = result.get("financials", {})
    inc = fin.get("income_statement", {})
    eps_data = inc.get("basic_earnings_per_share", {})
    eps = eps_data.get("value") if eps_data else None
    rev_data = inc.get("revenues", {})
    rev = rev_data.get("value") if rev_data else None
    ni_data = inc.get("net_income_loss", {})
    ni = ni_data.get("value") if ni_data else None
    return {
        "ticker": result.get("tickers", [None])[0],
        "end_date": result.get("end_date"),
        "filing_date": result.get("filing_date"),
        "fiscal_period": result.get("fiscal_period"),
        "fiscal_year": result.get("fiscal_year"),
        "eps": eps, "revenue": rev, "net_income": ni,
    }

def build_earnings_lookup(tickers_to_fetch: set, api_key: str) -> dict:
    """Fetch financials for given tickers and build earnings_timeline lookup."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cached = set()
    to_fetch = []
    for tk in sorted(tickers_to_fetch):
        if (CACHE_DIR / f"{tk}.json").exists():
            cached.add(tk)
        else:
            to_fetch.append(tk)

    log(f"  Earnings: {len(cached)} cached, {len(to_fetch)} to fetch")

    if to_fetch and api_key:
        est_min = len(to_fetch) * RATE_LIMIT_S / 60
        log(f"  Fetching {len(to_fetch)} tickers (~{est_min:.1f} min)")
        for i, tk in enumerate(to_fetch):
            if (i+1) % 20 == 0:
                log(f"    Progress: {i+1}/{len(to_fetch)} ({(i+1)/len(to_fetch)*100:.0f}%)")
            fetch_financials(tk, api_key)
            if i < len(to_fetch) - 1:
                time.sleep(RATE_LIMIT_S)
        log(f"  Fetch complete")

    # Parse cached data into timeline
    earnings_timeline = {}
    no_data = 0
    for tk in tickers_to_fetch:
        cache_file = CACHE_DIR / f"{tk}.json"
        if not cache_file.exists():
            no_data += 1; continue
        with open(cache_file) as f:
            results = json.load(f)
        if not results:
            no_data += 1; continue
        quarters = []
        for r in results:
            parsed = extract_eps_revenue(r)
            if parsed["filing_date"] and (parsed["eps"] is not None or parsed["revenue"] is not None):
                quarters.append(parsed)
        if quarters:
            quarters.sort(key=lambda x: x["end_date"] or "")
            earnings_timeline[tk] = quarters

    log(f"  Parsed: {len(earnings_timeline)} with data, {no_data} without")
    return earnings_timeline

def passes_earnings_filter(ticker: str, signal_date_str: str,
                           earnings_timeline: dict) -> bool:
    """
    No lookahead: only use reports filed BEFORE signal_date.
    Pass criteria (any of):
    1. EPS > 0 AND EPS > same-quarter-last-year EPS (YoY growth)
    2. Revenue > same-quarter-last-year revenue * 1.05 (>5% YoY growth)
    3. Fallback: no YoY comparison available but EPS > 0
    """
    timeline = earnings_timeline.get(ticker)
    if not timeline:
        return False

    available = [q for q in timeline if q["filing_date"] and q["filing_date"] < signal_date_str]
    if not available:
        return False

    latest = available[-1]
    latest_eps = latest["eps"]
    latest_rev = latest["revenue"]
    latest_fy = latest["fiscal_year"]
    latest_fp = latest["fiscal_period"]

    # Find YoY comparison
    yoy_quarter = None
    if latest_fy and latest_fp:
        prev_fy = str(int(latest_fy) - 1) if latest_fy else None
        if prev_fy:
            for q in available:
                if q["fiscal_year"] == prev_fy and q["fiscal_period"] == latest_fp:
                    yoy_quarter = q
                    break

    # Criteria 1: EPS > 0 AND growing YoY
    if latest_eps is not None and latest_eps > 0:
        if yoy_quarter and yoy_quarter["eps"] is not None:
            if latest_eps > yoy_quarter["eps"]:
                return True

    # Criteria 2: Revenue growing >5% YoY
    if latest_rev is not None and latest_rev > 0:
        if yoy_quarter and yoy_quarter["revenue"] is not None and yoy_quarter["revenue"] > 0:
            if (latest_rev - yoy_quarter["revenue"]) / abs(yoy_quarter["revenue"]) > 0.05:
                return True

    # Fallback: no YoY available, pass if EPS > 0
    if yoy_quarter is None and latest_eps is not None and latest_eps > 0:
        return True

    return False

# ── Unified Backtest ──────────────────────────────────────────────────
def run_bt(pb, stop_mode="none", exit_mode="C0",
           hmm_lk=None, hmm_lo=0.25, hmm_hi=0.50,
           r_pct=R_PCT, cost_bps=20, gap_limit=0.08,
           start_dt=None, end_dt=None, ret_cap=None,
           earnings_filter=None):
    """
    earnings_filter: if set, dict from build_earnings_lookup.
    Applied as LAST filter before entry — after HMM, gap, top_k.
    """
    pl,il,sbd = pb["pl"],pb["il"],pb["sbd"]
    td,ti,ws  = pb["td"],pb["ti"],pb["ws"]

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
    earn_pass = 0
    earn_fail = 0

    for ds in td:
        if ds < start_s:
            equity.append({"date":ds,"eq":CAPITAL,"n":0})
            continue

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

            if stop_atr > 0 and not exited:
                cs = p.get("cat_stop",0)
                if cs > 0:
                    if O <= cs:
                        ex_p=O; ex_r="cat_gap"; exited=True
                    elif L <= cs:
                        ex_p=cs; ex_r="cat_stop"; exited=True

            if gap_th < 0 and not exited:
                pc = p.get("pc", ep)
                if pc > 0 and (O-pc)/pc <= gap_th:
                    ex_p=O; ex_r="gap_crash"; exited=True

            if ret_cap is not None and not exited:
                ret_now = (C - ep) / ep if ep > 0 else 0
                if ret_now >= ret_cap:
                    cap_price = ep * (1 + ret_cap)
                    ex_p = min(cap_price, H)
                    ex_r = "ret_cap"; exited = True

            if not exited and ds >= p["ex_date"]:
                if exit_mode=="C2" and not p.get("ext",False):
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

            if not exited and p.get("ext",False) and p["days"]>HOLD:
                p["trail_hi"] = max(p.get("trail_hi",H), H)
                ts = p["trail_hi"]*(1-TRAIL_PCT)
                if L <= ts:
                    ex_p=ts; ex_r="trailing"; exited=True

            if not exited and exit_mode=="C2" and p["days"]==3:
                ret = (C-ep)/ep if ep>0 else 0
                mc = p.get("mc", ep)
                if ret < DAY3_THR*r_pct and C <= mc*0.999:
                    ex_p=C; ex_r="day3"; exited=True

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

        # ── Entries ──
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
                    # EARNINGS FILTER — applied LAST, after all other filters
                    if earnings_filter is not None:
                        if passes_earnings_filter(t, prev, earnings_filter):
                            earn_pass += 1
                        else:
                            earn_fail += 1
                            continue
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

        pv = sum(p["sh"]*pl.get((ds,p["ticker"]),{}).get("Close",p["ep"])
                 for p in positions)
        equity.append({"date":ds,"eq":cash+pv,"n":len(positions)})

        if past_end and not positions:
            break

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity)
    if not edf.empty:
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf.set_index("date")

    stats = {"earn_pass": earn_pass, "earn_fail": earn_fail}
    return tdf, edf, stats

# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polygon-key", default="")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch earnings data, don't run backtest")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching, use cached data only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    log("="*80)
    log("Layer 1 Earnings Filter Test")
    log(f"Period: {START} -> {END}")
    log("="*80)

    # ── [1] Load data ──
    log("\n[1/6] Loading OHLCV data...")
    df = load_ohlcv()
    log("[2/6] Computing indicators...")
    df = compute_indicators(df)
    mask = apply_layer1(df)

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="2021-01-01", end_date=END)
    tdays = list(sched.index.normalize())
    warmup = pd.Timestamp(START)
    log(f"  Layer 1 passers: {mask.sum():,}")

    log("\n[3/6] Building lookups...")
    pb = build_lookups(df, mask, tdays, warmup, TOP_K)

    # ── [4] First pass: run HMM+C2 WITHOUT earnings filter to find traded tickers ──
    log("\n[4/6] First pass — identify traded tickers (no earnings filter)...")
    hmm = load_hmm()

    t_c2, e_c2, _ = run_bt(pb, exit_mode="C2", hmm_lk=hmm, ret_cap=2.0)
    t_c0, e_c0, _ = run_bt(pb, exit_mode="C0", hmm_lk=hmm, ret_cap=2.0)
    m_c2 = compute_metrics(e_c2["eq"])
    m_c0 = compute_metrics(e_c0["eq"])
    log(f"  HMM+C0 baseline: {len(t_c0)} trades, Sharpe={m_c0['sharpe']:.3f}")
    log(f"  HMM+C2 baseline: {len(t_c2)} trades, Sharpe={m_c2['sharpe']:.3f}")

    # Collect unique tickers that actually got traded
    traded_tickers = set()
    if not t_c2.empty:
        traded_tickers.update(t_c2["ticker"].unique())
    if not t_c0.empty:
        traded_tickers.update(t_c0["ticker"].unique())
    log(f"  Unique traded tickers: {len(traded_tickers)}")
    log(f"  (vs all signal tickers: {len(set(t for sigs in pb['sbd'].values() for t,_ in sigs))})")

    # ── [5] Fetch earnings ONLY for traded tickers ──
    log("\n[5/6] Fetching earnings data (traded tickers only)...")
    api_key = args.polygon_key if not args.skip_fetch else ""
    earnings_timeline = build_earnings_lookup(traded_tickers, api_key)

    if args.fetch_only:
        log(f"\n--fetch-only: {len(list(CACHE_DIR.glob('*.json')))} cached files. Stopping.")
        return

    # Quick coverage stats
    has_data = sum(1 for tk in traded_tickers if tk in earnings_timeline)
    log(f"  Coverage: {has_data}/{len(traded_tickers)} traded tickers have earnings data "
        f"({has_data/len(traded_tickers)*100:.0f}%)")

    # ── [6] Backtests with earnings filter ──
    log("\n[6/6] Running comparison backtests...")

    caps = [2.0, 5.0, None]
    cap_labels = ["Cap 200%", "Cap 500%", "Uncapped"]

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: HMM+C0 with and without earnings filter
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 1: HMM+C0 — With vs Without Earnings Filter")
    log("="*90)

    r1 = []
    for cap, clabel in zip(caps, cap_labels):
        t_nf, e_nf, _ = run_bt(pb, exit_mode="C0", hmm_lk=hmm, ret_cap=cap)
        m_nf = compute_metrics(e_nf["eq"])
        t_ef, e_ef, s_ef = run_bt(pb, exit_mode="C0", hmm_lk=hmm, ret_cap=cap,
                                    earnings_filter=earnings_timeline)
        m_ef = compute_metrics(e_ef["eq"])
        wr_nf = (t_nf["ret"]>0).mean()*100 if not t_nf.empty else 0
        wr_ef = (t_ef["ret"]>0).mean()*100 if not t_ef.empty else 0
        r1.append(dict(cap=clabel,
                       n_nf=len(t_nf), sh_nf=m_nf["sharpe"], cagr_nf=m_nf["cagr"],
                       mdd_nf=m_nf["max_dd"], wr_nf=wr_nf,
                       n_ef=len(t_ef), sh_ef=m_ef["sharpe"], cagr_ef=m_ef["cagr"],
                       mdd_ef=m_ef["max_dd"], wr_ef=wr_ef,
                       pass_n=s_ef["earn_pass"], fail_n=s_ef["earn_fail"]))
        log(f"  {clabel:>10s} NoFilter: N={len(t_nf):>5d} Sh={m_nf['sharpe']:>6.3f} "
            f"CAGR={m_nf['cagr']:>+8.2f}% DD={m_nf['max_dd']:>+7.2f}% WR={wr_nf:>5.1f}%")
        log(f"  {clabel:>10s}  +Earns:  N={len(t_ef):>5d} Sh={m_ef['sharpe']:>6.3f} "
            f"CAGR={m_ef['cagr']:>+8.2f}% DD={m_ef['max_dd']:>+7.2f}% WR={wr_ef:>5.1f}% "
            f"(pass={s_ef['earn_pass']}, fail={s_ef['earn_fail']})")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: HMM+C2 with and without earnings filter
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 2: HMM+C2 — With vs Without Earnings Filter")
    log("="*90)

    r2 = []
    for cap, clabel in zip(caps, cap_labels):
        t_nf, e_nf, _ = run_bt(pb, exit_mode="C2", hmm_lk=hmm, ret_cap=cap)
        m_nf = compute_metrics(e_nf["eq"])
        t_ef, e_ef, s_ef = run_bt(pb, exit_mode="C2", hmm_lk=hmm, ret_cap=cap,
                                    earnings_filter=earnings_timeline)
        m_ef = compute_metrics(e_ef["eq"])
        wr_nf = (t_nf["ret"]>0).mean()*100 if not t_nf.empty else 0
        wr_ef = (t_ef["ret"]>0).mean()*100 if not t_ef.empty else 0
        ar_nf = t_nf["ret"].mean() if not t_nf.empty else 0
        ar_ef = t_ef["ret"].mean() if not t_ef.empty else 0
        r2.append(dict(cap=clabel,
                       n_nf=len(t_nf), sh_nf=m_nf["sharpe"], cagr_nf=m_nf["cagr"],
                       mdd_nf=m_nf["max_dd"], wr_nf=wr_nf, ar_nf=ar_nf,
                       n_ef=len(t_ef), sh_ef=m_ef["sharpe"], cagr_ef=m_ef["cagr"],
                       mdd_ef=m_ef["max_dd"], wr_ef=wr_ef, ar_ef=ar_ef,
                       so_nf=m_nf["sortino"], so_ef=m_ef["sortino"],
                       pass_n=s_ef["earn_pass"], fail_n=s_ef["earn_fail"]))
        log(f"  {clabel:>10s} NoFilter: N={len(t_nf):>5d} Sh={m_nf['sharpe']:>6.3f} "
            f"CAGR={m_nf['cagr']:>+8.2f}% DD={m_nf['max_dd']:>+7.2f}% WR={wr_nf:>5.1f}%")
        log(f"  {clabel:>10s}  +Earns:  N={len(t_ef):>5d} Sh={m_ef['sharpe']:>6.3f} "
            f"CAGR={m_ef['cagr']:>+8.2f}% DD={m_ef['max_dd']:>+7.2f}% WR={wr_ef:>5.1f}% "
            f"(pass={s_ef['earn_pass']}, fail={s_ef['earn_fail']})")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: Walk-Forward with Earnings Filter
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 3: HMM+C2 Walk-Forward @200% Cap — With vs Without Earnings")
    log("="*90)

    folds = [
        ("Fold 1","2022-03-01","2023-12-31","2024-01-22","2024-12-31"),
        ("Fold 2","2022-03-01","2024-12-31","2025-01-22","2025-12-31"),
        ("Fold 3","2023-01-01","2024-12-31","2025-01-22","2025-12-31"),
    ]

    r3 = []
    for fn, ts, te, vs, ve in folds:
        log(f"\n  --- {fn}: Test {vs} -> {ve} ---")
        t_nf, e_nf, _ = run_bt(pb, exit_mode="C2", hmm_lk=hmm,
                                start_dt=vs, end_dt=ve, ret_cap=2.0)
        m_nf = compute_metrics(e_nf["eq"])
        t_ef, e_ef, _ = run_bt(pb, exit_mode="C2", hmm_lk=hmm,
                                start_dt=vs, end_dt=ve, ret_cap=2.0,
                                earnings_filter=earnings_timeline)
        m_ef = compute_metrics(e_ef["eq"])
        wr_nf = (t_nf["ret"]>0).mean()*100 if not t_nf.empty else 0
        wr_ef = (t_ef["ret"]>0).mean()*100 if not t_ef.empty else 0
        r3.append(dict(fn=fn,
                       n_nf=len(t_nf), sh_nf=m_nf["sharpe"], cagr_nf=m_nf["cagr"],
                       mdd_nf=m_nf["max_dd"], wr_nf=wr_nf,
                       n_ef=len(t_ef), sh_ef=m_ef["sharpe"], cagr_ef=m_ef["cagr"],
                       mdd_ef=m_ef["max_dd"], wr_ef=wr_ef))
        log(f"    NoFilter: N={len(t_nf):>5d} Sh={m_nf['sharpe']:>6.3f} "
            f"CAGR={m_nf['cagr']:>+8.2f}% DD={m_nf['max_dd']:>+7.2f}%")
        log(f"    +Earns:   N={len(t_ef):>5d} Sh={m_ef['sharpe']:>6.3f} "
            f"CAGR={m_ef['cagr']:>+8.2f}% DD={m_ef['max_dd']:>+7.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: Trade-Level Analysis — Earnings Filter on Winners vs Losers
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "="*90)
    log("TEST 4: Trade-Level Analysis — Earnings Filter Impact")
    log("="*90)

    t_all = t_c2  # reuse from first pass (HMM+C2 @200% cap)
    if not t_all.empty:
        passed_set = set()
        failed_set = set()
        for _, row in t_all.iterrows():
            key = (row["ticker"], row["sig"])
            if passes_earnings_filter(row["ticker"], row["sig"], earnings_timeline):
                passed_set.add(key)
            else:
                failed_set.add(key)

        pass_mask = t_all.apply(lambda r: (r["ticker"], r["sig"]) in passed_set, axis=1)
        t_passed = t_all[pass_mask]
        t_failed = t_all[~pass_mask]

        log(f"\n  All trades (HMM+C2 @200% cap): {len(t_all)}")
        log(f"  Passes earnings: {len(t_passed)} ({len(t_passed)/len(t_all)*100:.1f}%)")
        log(f"  Fails earnings:  {len(t_failed)} ({len(t_failed)/len(t_all)*100:.1f}%)")

        for label, grp in [("PASSED", t_passed), ("FAILED", t_failed)]:
            if grp.empty: continue
            log(f"\n  {label} trades ({len(grp)}):")
            log(f"    Avg Return:    {grp['ret'].mean():+.2f}%")
            log(f"    Median Return: {grp['ret'].median():+.2f}%")
            log(f"    Win Rate:      {(grp['ret']>0).mean()*100:.1f}%")
            log(f"    Total PnL:     ${grp['pnl'].sum():,.0f}")
            log(f"    Max Win:       {grp['ret'].max():+.1f}%")
            log(f"    Max Loss:      {grp['ret'].min():+.1f}%")

        log(f"\n  Year-by-year (Passed vs Failed):")
        for yr in sorted(t_all["entry"].str[:4].unique()):
            tp = t_passed[t_passed["entry"].str[:4]==yr]
            tf = t_failed[t_failed["entry"].str[:4]==yr]
            p_wr = (tp["ret"]>0).mean()*100 if len(tp)>0 else 0
            p_ar = tp["ret"].mean() if len(tp)>0 else 0
            f_wr = (tf["ret"]>0).mean()*100 if len(tf)>0 else 0
            f_ar = tf["ret"].mean() if len(tf)>0 else 0
            log(f"    {yr}: Pass N={len(tp):>4d} WR={p_wr:>5.1f}% Avg={p_ar:>+6.2f}% | "
                f"Fail N={len(tf):>4d} WR={f_wr:>5.1f}% Avg={f_ar:>+6.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Save report
    # ═══════════════════════════════════════════════════════════════════
    lines.append("="*110)
    lines.append("EARNINGS FILTER TEST RESULTS")
    lines.append(f"Filter: EPS>0 + YoY EPS growth, OR Revenue YoY growth >5%")
    lines.append(f"Applied as LAST filter (after Layer1 + HMM + gap + topK)")
    lines.append(f"Period: {START} -> {END}")
    lines.append("="*110)

    lines.append("\n" + "="*110)
    lines.append("TEST 1: HMM+C0 — Earnings Filter Impact")
    lines.append("="*110)
    hdr = (f"{'Cap':>10s}  {'N(no)':>6s}  {'Sh(no)':>7s}  {'CAGR(no)':>9s}  {'DD(no)':>8s}  {'WR(no)':>7s}  "
           f"{'N(+E)':>6s}  {'Sh(+E)':>7s}  {'CAGR(+E)':>9s}  {'DD(+E)':>8s}  {'WR(+E)':>7s}")
    lines.append(hdr); lines.append("-"*len(hdr))
    for r in r1:
        lines.append(
            f"{r['cap']:>10s}  {r['n_nf']:>6d}  {r['sh_nf']:>7.3f}  {r['cagr_nf']:>+8.2f}%  "
            f"{r['mdd_nf']:>+7.2f}%  {r['wr_nf']:>6.1f}%  "
            f"{r['n_ef']:>6d}  {r['sh_ef']:>7.3f}  {r['cagr_ef']:>+8.2f}%  "
            f"{r['mdd_ef']:>+7.2f}%  {r['wr_ef']:>6.1f}%")

    lines.append("\n" + "="*110)
    lines.append("TEST 2: HMM+C2 — Earnings Filter Impact")
    lines.append("="*110)
    lines.append(hdr); lines.append("-"*len(hdr))
    for r in r2:
        lines.append(
            f"{r['cap']:>10s}  {r['n_nf']:>6d}  {r['sh_nf']:>7.3f}  {r['cagr_nf']:>+8.2f}%  "
            f"{r['mdd_nf']:>+7.2f}%  {r['wr_nf']:>6.1f}%  "
            f"{r['n_ef']:>6d}  {r['sh_ef']:>7.3f}  {r['cagr_ef']:>+8.2f}%  "
            f"{r['mdd_ef']:>+7.2f}%  {r['wr_ef']:>6.1f}%")

    lines.append("\n" + "="*110)
    lines.append("TEST 3: HMM+C2 Walk-Forward @200% Cap — Earnings Filter")
    lines.append("="*110)
    hdr3 = (f"{'Fold':>8s}  {'N(no)':>6s}  {'Sh(no)':>7s}  {'CAGR(no)':>9s}  {'DD(no)':>8s}  "
            f"{'N(+E)':>6s}  {'Sh(+E)':>7s}  {'CAGR(+E)':>9s}  {'DD(+E)':>8s}")
    lines.append(hdr3); lines.append("-"*len(hdr3))
    for r in r3:
        lines.append(
            f"{r['fn']:>8s}  {r['n_nf']:>6d}  {r['sh_nf']:>7.3f}  {r['cagr_nf']:>+8.2f}%  "
            f"{r['mdd_nf']:>+7.2f}%  "
            f"{r['n_ef']:>6d}  {r['sh_ef']:>7.3f}  {r['cagr_ef']:>+8.2f}%  "
            f"{r['mdd_ef']:>+7.2f}%")

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "earnings_filter_results.txt").write_text(report, encoding="utf-8")

    # ── Equity plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_configs = [
        ("C0", "HMM+C0 @200% Cap", 2.0),
        ("C2", "HMM+C2 @200% Cap", 2.0),
        ("C2", "HMM+C2 Uncapped", None),
        ("C2", "HMM+C2 @500% Cap", 5.0),
    ]
    for ax, (em, title, cap) in zip(axes.flat, plot_configs):
        _, e_nf, _ = run_bt(pb, exit_mode=em, hmm_lk=hmm, ret_cap=cap)
        _, e_ef, _ = run_bt(pb, exit_mode=em, hmm_lk=hmm, ret_cap=cap,
                            earnings_filter=earnings_timeline)
        ax.plot(e_nf.index, e_nf["eq"], label="No Filter", linewidth=1.2, color="#1f77b4")
        ax.plot(e_ef.index, e_ef["eq"], label="+Earnings", linewidth=1.2, color="#d62728")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylabel("Portfolio Value ($)")

    fig.suptitle("Earnings Filter Impact — Portfolio Equity", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "earnings_filter_equity.png", dpi=150)
    plt.close(fig)

    log(f"\nSaved: {OUT_DIR / 'earnings_filter_results.txt'}")
    log(f"Saved: {OUT_DIR / 'earnings_filter_equity.png'}")
    log("Done!")

if __name__ == "__main__":
    main()
