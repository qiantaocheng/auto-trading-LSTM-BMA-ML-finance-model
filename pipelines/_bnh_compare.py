"""Compare HMM+C2 exit vs Buy-and-Hold same stocks for N days."""
from __future__ import annotations
import sys, io, os, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import pandas_market_calendars as mcal

if os.name == "nt" and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, "D:/trade/pipelines")
from layer1_version_l_test import (
    load_ohlcv, compute_indicators, apply_layer1, build_lookups,
    compute_metrics, load_hmm, run_bt, START, END, TOP_K,
    R_PCT, DAY3_THR, EXT_THR, EXT_HOLD, TRAIL_PCT, HOLD, CAPITAL
)

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ── Simple hold-N-days exit simulator ─────────────────────────────────
def sim_hold_n(ep, prices_after, n_days, cost=0.002):
    """Buy at ep, hold for n_days trading days, exit at Open of day n+1.
    prices_after: list of (O, H, L, C) for days after entry.
    Returns (ret_pct, actual_days_held).
    """
    if len(prices_after) < n_days:
        # Not enough data, exit at last available close
        last_c = prices_after[-1][3] if prices_after else ep
        ret = (last_c * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
        return ret, len(prices_after)

    # Exit at Open of day n_days (0-indexed: day 0 is first day after entry)
    # If n_days=10, we hold through day 9, exit at Open of day 10
    if len(prices_after) > n_days:
        exit_price = prices_after[n_days][0]  # Open of exit day
    else:
        exit_price = prices_after[n_days - 1][3]  # Close of last day

    ret = (exit_price * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
    return ret, n_days


def sim_c2(ep, prices_after, cost=0.002, r_pct=0.08, day3_thr=0.5,
           ext_thr=2.0, ext_hold=15, trail_pct=0.10, hold=7, ret_cap=200):
    """C2 exit: Day3 early + winner extension with trailing stop."""
    R = ep * r_pct
    mc = ep   # max close
    mh = ep   # max high
    pc_prev = ep
    extended = False
    trail_stop = 0.0

    for i, (O, H, L, C) in enumerate(prices_after):
        day = i + 1  # day 1, 2, 3, ...

        # Day3 early exit check
        if day == 3 and not extended:
            ret_so_far = (C - ep) / ep
            if ret_so_far < day3_thr * r_pct and C < mc:
                ex_p = C
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                if ret_cap and ret > ret_cap: ret = ret_cap
                return ret, day, "day3"

        # Normal exit at day hold+1 Open
        if day == hold and not extended:
            ret_at_close = (C - ep) / ep
            if ret_at_close >= ext_thr * r_pct:
                extended = True
                trail_stop = C * (1 - trail_pct)
            else:
                ex_p = prices_after[hold][0] if len(prices_after) > hold else C
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                if ret_cap and ret > ret_cap: ret = ret_cap
                return ret, hold, "time"

        if day > hold and not extended:
            ex_p = O
            ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
            if ret_cap and ret > ret_cap: ret = ret_cap
            return ret, day, "time"

        # Extended: trailing stop
        if extended:
            trail_stop = max(trail_stop, C * (1 - trail_pct))
            if L <= trail_stop:
                ex_p = trail_stop
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                if ret_cap and ret > ret_cap: ret = ret_cap
                return ret, day, "trailing"
            if day >= ext_hold:
                ex_p = prices_after[ext_hold][0] if len(prices_after) > ext_hold else C
                ret = (ex_p * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
                if ret_cap and ret > ret_cap: ret = ret_cap
                return ret, day, "ext_time"

        mc = max(mc, C)
        mh = max(mh, H)
        pc_prev = C

    # Fallback
    ret = (pc_prev * (1 - cost) - ep * (1 + cost)) / (ep * (1 + cost)) * 100
    if ret_cap and ret > ret_cap: ret = ret_cap
    return ret, len(prices_after), "timeout"


def main():
    log("=" * 80)
    log("Buy-and-Hold vs C2 Exit — Same Stocks Comparison")
    log("=" * 80)

    log("\n[1] Loading data...")
    df = load_ohlcv()
    df = compute_indicators(df)
    mask = apply_layer1(df)
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date="2021-01-01", end_date=END)
    tdays = list(sched.index.normalize())
    warmup = pd.Timestamp(START)
    pb = build_lookups(df, mask, tdays, warmup, TOP_K)
    hmm = load_hmm()

    price_lk = pb["pl"]
    sig_days = pb["sbd"]
    td_strs  = pb["td"]
    td_idx   = pb["ti"]

    # Build list of all HMM+C2 entry signals
    log("\n[2] Generating entry signals (HMM budgeted)...")
    hmm_lo, hmm_hi = 0.25, 0.50
    gap_limit = 0.08

    entries = []  # (ticker, sig_date, entry_date_idx, entry_price)

    start_idx = td_idx.get(pd.Timestamp(START), 0)

    for i in range(start_idx, len(tdays)):
        td = tdays[i]
        td_s = td_strs[i]

        if td_s not in sig_days:
            continue

        # HMM budget
        hmm_s = td.strftime("%Y-%m-%d")
        p_crisis = hmm.get(hmm_s, 0.0)
        if p_crisis < hmm_lo:
            k, alloc = 8, 1.0
        elif p_crisis < hmm_hi:
            k, alloc = 5, 0.6
        else:
            k, alloc = 2, 0.3

        candidates = sig_days[td_s][:k]  # list of (ticker, atr_pct)

        # Entry is next trading day
        if i + 1 >= len(tdays):
            continue
        entry_idx = i + 1
        entry_td_s = td_strs[entry_idx]

        for ticker, atr_pct in candidates:
            key = (entry_td_s, ticker)
            if key not in price_lk:
                continue
            pr = price_lk[key]
            O = pr["Open"]
            prev_key = (td_s, ticker)
            if prev_key not in price_lk:
                continue
            prev_C = price_lk[prev_key]["Close"]
            gap = (O - prev_C) / prev_C if prev_C > 0 else 0
            if gap > gap_limit:
                continue

            entries.append((ticker, td_s, entry_idx, O, alloc))

    log(f"  Total entry signals: {len(entries)}")

    # For each entry, get price series for up to 25 days after
    log("\n[3] Simulating exits...")
    HOLD_PERIODS = [5, 7, 10, 15, 20]
    MAX_LOOKAHEAD = 25

    results = []
    for ticker, sig_date, entry_idx, entry_price, alloc in entries:
        # Collect price bars for days after entry
        prices_after = []
        for j in range(1, MAX_LOOKAHEAD + 1):
            idx = entry_idx + j
            if idx >= len(tdays):
                break
            key = (td_strs[idx], ticker)
            if key not in price_lk:
                break
            pr = price_lk[key]
            prices_after.append((pr["Open"], pr["High"], pr["Low"], pr["Close"]))

        if len(prices_after) < 3:
            continue

        # C2 exit
        c2_ret, c2_days, c2_reason = sim_c2(entry_price, prices_after)

        # Buy and hold for N days
        row = {
            "ticker": ticker, "sig_date": sig_date,
            "entry_price": entry_price,
            "c2_ret": c2_ret, "c2_days": c2_days, "c2_reason": c2_reason,
        }
        for n in HOLD_PERIODS:
            ret, days = sim_hold_n(entry_price, prices_after, n)
            # Apply same 200% cap
            if ret > 200: ret = 200.0
            row[f"bnh_{n}d_ret"] = ret
            row[f"bnh_{n}d_days"] = days

        results.append(row)

    rdf = pd.DataFrame(results)
    log(f"  Total paired trades: {len(rdf)}")

    # ── Summary ──────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("RESULTS: C2 Exit vs Buy-and-Hold (same entry signals, paired)")
    log("=" * 80)

    # Overall
    log(f"\n  Total trades: {len(rdf)}")
    log(f"  C2 exit: mean={rdf.c2_ret.mean():+.2f}%  median={rdf.c2_ret.median():+.2f}%  "
        f"std={rdf.c2_ret.std():.2f}%  WR={100*(rdf.c2_ret>0).mean():.1f}%  "
        f"avg_days={rdf.c2_days.mean():.1f}")

    for n in HOLD_PERIODS:
        col = f"bnh_{n}d_ret"
        log(f"  BnH {n:2d}d:  mean={rdf[col].mean():+.2f}%  median={rdf[col].median():+.2f}%  "
            f"std={rdf[col].std():.2f}%  WR={100*(rdf[col]>0).mean():.1f}%  "
            f"avg_days={n:.1f}")

    # Paired delta for each hold period
    log(f"\n  {'Hold':>6s}   {'mean_Δ':>8s}  {'med_Δ':>8s}  {'C2wins':>7s}  {'t-stat':>7s}  {'p-val':>8s}  Sig")
    log(f"  {'-'*65}")

    from scipy import stats as sp_stats
    for n in HOLD_PERIODS:
        col = f"bnh_{n}d_ret"
        delta = rdf.c2_ret - rdf[col]
        c2w = (delta > 0).mean() * 100
        t, p = sp_stats.ttest_1samp(delta, 0)
        sig = "***" if p < 0.001 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        log(f"  C2-BnH{n:2d}d  {delta.mean():+8.3f}%  {delta.median():+8.3f}%  {c2w:6.1f}%  {t:+7.2f}  {p:8.4f}  {sig}")

    # By year
    rdf["year"] = pd.to_datetime(rdf.sig_date).dt.year
    log(f"\n  Year-by-year: C2 vs BnH 10d (paired delta)")
    log(f"  {'Year':>6s}  {'N':>6s}  {'C2 mean':>8s}  {'BnH10 mean':>10s}  {'mean_Δ':>8s}  {'C2wins':>7s}  {'p-val':>8s}")
    log(f"  {'-'*65}")
    for yr in sorted(rdf.year.unique()):
        sub = rdf[rdf.year == yr]
        delta = sub.c2_ret - sub.bnh_10d_ret
        c2w = (delta > 0).mean() * 100
        t, p = sp_stats.ttest_1samp(delta, 0)
        log(f"  {yr:>6d}  {len(sub):>6d}  {sub.c2_ret.mean():+8.2f}%  {sub.bnh_10d_ret.mean():+10.2f}%  "
            f"{delta.mean():+8.3f}%  {c2w:6.1f}%  {p:8.4f}")

    # By C2 exit reason
    log(f"\n  By C2 exit reason: C2 vs BnH 10d")
    log(f"  {'Reason':>10s}  {'N':>6s}  {'C2 mean':>8s}  {'BnH10 mean':>10s}  {'mean_Δ':>8s}  {'C2wins':>7s}")
    log(f"  {'-'*60}")
    for reason in ["day3", "time", "trailing", "ext_time", "ret_cap"]:
        sub = rdf[rdf.c2_reason == reason]
        if len(sub) == 0:
            continue
        delta = sub.c2_ret - sub.bnh_10d_ret
        c2w = (delta > 0).mean() * 100
        log(f"  {reason:>10s}  {len(sub):>6d}  {sub.c2_ret.mean():+8.2f}%  {sub.bnh_10d_ret.mean():+10.2f}%  "
            f"{delta.mean():+8.3f}%  {c2w:6.1f}%")

    # Distribution of C2 exit reasons
    log(f"\n  C2 Exit Reason Distribution:")
    for reason, cnt in rdf.c2_reason.value_counts().items():
        log(f"    {reason:>10s}: {cnt:5d} ({100*cnt/len(rdf):.1f}%)")

    # Tail analysis
    log(f"\n  Tail Analysis (C2 vs BnH 10d):")
    big_loss_bnh = rdf[rdf.bnh_10d_ret < -20]
    if len(big_loss_bnh) > 0:
        delta_bl = big_loss_bnh.c2_ret - big_loss_bnh.bnh_10d_ret
        log(f"    When BnH10 loses >20% (N={len(big_loss_bnh)}): C2 better {(delta_bl>0).mean()*100:.1f}%, avg delta={delta_bl.mean():+.2f}%")

    big_win_bnh = rdf[rdf.bnh_10d_ret > 50]
    if len(big_win_bnh) > 0:
        delta_bw = big_win_bnh.c2_ret - big_win_bnh.bnh_10d_ret
        log(f"    When BnH10 wins >50% (N={len(big_win_bnh)}): C2 better {(delta_bw>0).mean()*100:.1f}%, avg delta={delta_bw.mean():+.2f}%")

    # Risk-adjusted: Sharpe approximation from trade returns
    log(f"\n  Risk-Adjusted (trade-level Sharpe proxy, annualized):")
    ann = np.sqrt(252)
    for label, col in [("C2", "c2_ret")] + [(f"BnH{n}d", f"bnh_{n}d_ret") for n in HOLD_PERIODS]:
        m = rdf[col].mean()
        s = rdf[col].std()
        sharpe = (m / s) * ann if s > 0 else 0
        log(f"    {label:>8s}: mean={m:+.3f}%  std={s:.3f}%  Sharpe_proxy={sharpe:.3f}")

    log("\nDone!")


if __name__ == "__main__":
    main()
