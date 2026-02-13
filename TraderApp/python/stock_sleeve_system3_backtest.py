#!/usr/bin/env python3
"""
System 3: Stock Prediction Sleeve — Complete Backtest
=====================================================

Normal state:
  5-day rebalance, 10 stocks x 10% = 100% of sleeve capital

VIX Mean-Reversion Boost (only when DD ladder NOT active):
  VIX > 30 and falling  -> 12% per stock (120% total)
  VIX > 40 and falling  -> 15% per stock (150% total)
  VIX > 50 and rising   -> freeze (no rebalance, keep current)
  Boost lasts 1 rebalance cycle (5 days), then back to 10%

DD Ladder (portfolio drawdown -> reduce exposure):
  DD >  -5%: 10 stocks (normal)
  DD <= -5%:  8 stocks
  DD <= -8%:  6 stocks
  DD <= -12%: 4 stocks
  DD <= -18%: 2 stocks
  Priority: DD ladder ALWAYS overrides VIX boost

Per-stock stop loss: 2-sigma x 5d vol, floor -10%

Note: DD ladder implemented as proportional weight reduction (conservative
test without stock-selection alpha). In production, prediction model would
select which stocks to keep/drop.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────
STOCKS = ["NVDA", "AAPL", "TSLA", "META", "SMCI", "NFLX",
          "MSFT", "GOOGL", "AMZN", "AMD"]
SPY = "SPY"
VIX_T = "^VIX"
START = "2016-01-01"
END   = "2026-02-12"
COST  = 10          # round-trip bps

REBAL      = 5      # trading days
W_BASE     = 0.10   # 10% per stock
W_VIX30    = 0.12   # VIX>30 falling boost
W_VIX40    = 0.15   # VIX>40 falling boost
VIX_FREEZE = 50     # VIX>50 rising -> freeze
SL_SIGMA   = 2.0    # stop loss sigma
SL_FLOOR   = 0.10   # stop floor 10%

DD_LADDER = [        # (threshold, fraction_of_stocks)
    (-0.05, 1.0),    # DD > -5%  -> 100% = 10 stocks
    (-0.08, 0.8),    # DD > -8%  ->  80% =  8 stocks
    (-0.12, 0.6),    # DD > -12% ->  60% =  6 stocks
    (-0.18, 0.4),    # DD > -18% ->  40% =  4 stocks
    (-999., 0.2),    # worse     ->  20% =  2 stocks
]


def load():
    tickers = STOCKS + [SPY, VIX_T]
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    px = raw["Close"].ffill()
    # align: need all stocks present
    first_valid = max(px[s].first_valid_index() for s in STOCKS)
    px = px.loc[first_valid:]
    px = px.dropna(subset=STOCKS + [SPY, VIX_T])
    return px


def dd_scale(dd):
    """Return exposure scale factor from DD ladder (1.0 = normal, 0.2 = min)."""
    for thr, frac in DD_LADDER:
        if dd > thr:
            return frac
    return 0.2


def run(px, vix, use_vix=True, use_dd=True, use_sl=True):
    """Portfolio simulation. Returns equity curve + stats dict."""
    dates = px.index
    N = len(STOCKS)
    log_r = np.log(px[STOCKS] / px[STOCKS].shift(1))

    cap = 1.0
    hwm = 1.0
    w   = {s: 0.0  for s in STOCKS}
    ep  = {s: None for s in STOCKS}
    stopped = set()

    eq = [cap]
    rc = REBAL          # trigger first rebalance immediately
    trades = 0; stops = 0; vix_boosts = 0; dd_cuts = 0; freezes = 0

    for i in range(1, len(dates)):
        dt   = dates[i]
        prev = dates[i - 1]

        # ── 1. Daily P&L ──
        pnl = 0.0
        for s in STOCKS:
            if w[s] > 0:
                pnl += w[s] * (px.loc[dt, s] / px.loc[prev, s] - 1)
        cap *= (1 + pnl)
        hwm = max(hwm, cap)
        dd  = cap / hwm - 1

        # ── 2. Daily stop-loss check ──
        if use_sl:
            for s in STOCKS:
                if w[s] > 0 and ep[s] is not None and s not in stopped:
                    loss = px.loc[dt, s] / ep[s] - 1
                    idx  = log_r.index.get_loc(dt)
                    if idx >= 20:
                        dv  = float(log_r[s].iloc[idx - 20:idx].std())
                        v5  = dv * np.sqrt(5)
                    else:
                        v5 = 0.10
                    thr = max(SL_SIGMA * v5, SL_FLOOR)
                    if loss < -thr:
                        cap -= w[s] * cap * COST / 10_000
                        w[s] = 0.0
                        stopped.add(s)
                        stops += 1; trades += 1

        # ── 3. Rebalance every REBAL days ──
        rc += 1
        if rc >= REBAL:
            rc = 0

            # DD ladder -> exposure scale
            if use_dd:
                scale = dd_scale(dd)
                dd_active = (scale < 1.0)
            else:
                scale = 1.0
                dd_active = False

            if dd_active:
                dd_cuts += 1

            # VIX logic (disabled if DD active)
            vt = float(vix.iloc[i])
            vy = float(vix.iloc[i - 1])
            falling = vt < vy

            if use_vix and not dd_active:
                if vt > VIX_FREEZE and not falling:
                    freezes += 1
                    eq.append(cap)
                    continue                      # freeze: skip rebalance
                elif vt > 40 and falling:
                    pw = W_VIX40
                    vix_boosts += 1
                elif vt > 30 and falling:
                    pw = W_VIX30
                    vix_boosts += 1
                else:
                    pw = W_BASE
            else:
                pw = W_BASE

            # Apply DD scale to per-stock weight
            pw_eff = pw * scale

            # Set weights
            for s in STOCKS:
                old = w[s]
                new = pw_eff
                if abs(new - old) > 0.001:
                    cap -= abs(new - old) * cap * COST / 10_000
                    trades += 1
                w[s] = new
                ep[s] = px.loc[dt, s]
                stopped.discard(s)

        eq.append(cap)

    eq = np.array(eq)
    return dict(eq=eq, dates=dates, trades=trades, stops=stops,
                vix_boosts=vix_boosts, dd_cuts=dd_cuts, freezes=freezes)


def met(eq, dates):
    yrs  = (dates[-1] - dates[0]).days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1
    r    = np.diff(eq) / eq[:-1]
    vol  = np.std(r) * np.sqrt(252)
    sh   = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
    rm   = np.maximum.accumulate(eq)
    mdd  = np.min(eq / rm - 1)
    cal  = cagr / abs(mdd) if mdd != 0 else 0
    return dict(CAGR=cagr, MaxDD=mdd, Sharpe=sh, Calmar=cal, Vol=vol)


def yearly(eq, dates):
    rows = []
    for yr in range(2016, 2027):
        mask = np.array([d.year == yr for d in dates])
        if mask.sum() < 10: continue
        ye = eq[mask]
        r  = ye[-1] / ye[0] - 1
        hm = np.maximum.accumulate(ye)
        dd = np.min(ye / hm - 1)
        rows.append((yr, r, dd))
    return rows


# ═══════════════════════════════════════════════════════════════════
def main():
    W = 110
    sep = "=" * W

    print(sep)
    print("  SYSTEM 3: Stock Prediction Sleeve — Complete Backtest")
    print("  10 stocks x 10%, 5d rebal | VIX mean-reversion boost | DD ladder | 2-sigma SL")
    print(sep)

    print("\nLoading...")
    px  = load()
    vix = px[VIX_T]
    print(f"  Period: {px.index[0].strftime('%Y-%m-%d')} — {px.index[-1].strftime('%Y-%m-%d')}  ({len(px)} bars)")
    for s in STOCKS:
        print(f"    {s:>5}: ${px[s].iloc[0]:>8.2f} -> ${px[s].iloc[-1]:>8.2f}  ({px[s].iloc[-1]/px[s].iloc[0]-1:>+.0%})")

    # ── Run all variants ──
    cfgs = [
        ("EW Baseline",    dict(use_vix=False, use_dd=False, use_sl=False)),
        ("+ SL only",      dict(use_vix=False, use_dd=False, use_sl=True)),
        ("+ DD only",      dict(use_vix=False, use_dd=True,  use_sl=False)),
        ("+ VIX only",     dict(use_vix=True,  use_dd=False, use_sl=False)),
        ("DD + SL",        dict(use_vix=False, use_dd=True,  use_sl=True)),
        ("VIX + SL",       dict(use_vix=True,  use_dd=False, use_sl=True)),
        ("VIX + DD",       dict(use_vix=True,  use_dd=True,  use_sl=False)),
        ("FULL Sys3",      dict(use_vix=True,  use_dd=True,  use_sl=True)),
    ]

    res = {}
    for name, kw in cfgs:
        print(f"  Running {name}...")
        r = run(px, vix, **kw)
        m = met(r["eq"], r["dates"])
        res[name] = {**m, **r}

    # SPY B&H
    spy_eq = px[SPY].values / px[SPY].values[0]
    spy_m  = met(spy_eq, px.index)

    # ── Performance Table ──
    print(f"\n{sep}")
    print("  PERFORMANCE COMPARISON")
    print(sep)
    hdr = f"  {'Strategy':<16} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Vol':>8} {'Trd':>6} {'SL':>4} {'VIX+':>5} {'DD-':>4} {'Frz':>4}"
    print(hdr)
    print("  " + "-" * (W - 2))

    print(f"  {'SPY B&H':<16} {spy_m['CAGR']:>+7.2%} {spy_m['MaxDD']:>+7.1%} "
          f"{spy_m['Sharpe']:>8.3f} {spy_m['Calmar']:>8.3f} {spy_m['Vol']:>7.2%} "
          f"{'—':>6} {'—':>4} {'—':>5} {'—':>4} {'—':>4}")

    for name, _ in cfgs:
        r = res[name]
        print(f"  {name:<16} {r['CAGR']:>+7.2%} {r['MaxDD']:>+7.1%} "
              f"{r['Sharpe']:>8.3f} {r['Calmar']:>8.3f} {r['Vol']:>7.2%} "
              f"{r['trades']:>6} {r['stops']:>4} {r['vix_boosts']:>5} "
              f"{r['dd_cuts']:>4} {r['freezes']:>4}")

    # ── Full vs Baseline delta ──
    base = res["EW Baseline"]
    full = res["FULL Sys3"]
    print(f"\n  FULL System 3 vs EW Baseline:")
    print(f"    CAGR:   {base['CAGR']:>+.2%} -> {full['CAGR']:>+.2%}  (delta {full['CAGR']-base['CAGR']:>+.2%})")
    print(f"    MaxDD:  {base['MaxDD']:>+.1%} -> {full['MaxDD']:>+.1%}  (delta {full['MaxDD']-base['MaxDD']:>+.1%})")
    print(f"    Sharpe: {base['Sharpe']:>.3f} -> {full['Sharpe']:>.3f}  (delta {full['Sharpe']-base['Sharpe']:>+.3f})")
    print(f"    Calmar: {base['Calmar']:>.3f} -> {full['Calmar']:>.3f}  (delta {full['Calmar']-base['Calmar']:>+.3f})")

    # ── Layer contribution ──
    print(f"\n{sep}")
    print("  LAYER CONTRIBUTION ANALYSIS")
    print("  (what happens when you REMOVE each layer from the full system)")
    print(sep)

    ablations = [
        ("Stop Loss",  "FULL Sys3", "VIX + DD"),
        ("DD Ladder",  "FULL Sys3", "VIX + SL"),
        ("VIX Boost",  "FULL Sys3", "DD + SL"),
    ]

    print(f"  {'Layer removed':<16} {'ΔCAGR':>8} {'ΔMaxDD':>8} {'ΔSharpe':>9} {'ΔCalmar':>9}  Assessment")
    print("  " + "-" * 75)

    for layer, full_name, without_name in ablations:
        f = res[full_name]
        w = res[without_name]
        dc = f["CAGR"] - w["CAGR"]
        dm = f["MaxDD"] - w["MaxDD"]
        ds = f["Sharpe"] - w["Sharpe"]
        dcal = f["Calmar"] - w["Calmar"]
        # Assessment: if removing hurts (full > without), layer is valuable
        verdict = "KEEPS" if dcal > 0.01 else ("neutral" if abs(dcal) < 0.01 else "HURTS")
        print(f"  w/o {layer:<11} {dc:>+7.2%} {dm:>+7.1%} {ds:>+9.3f} {dcal:>+9.3f}  {layer} {verdict}")

    # ── Yearly breakdown ──
    print(f"\n{sep}")
    print("  YEARLY RETURNS: EW Baseline vs FULL System 3")
    print(sep)

    yb = yearly(base["eq"], base["dates"])
    yf_ = yearly(full["eq"], full["dates"])

    print(f"  {'Yr':>6}  {'Base':>8}  {'Full':>8}  {'Delta':>8}  {'Base DD':>8}  {'Full DD':>8}")
    print(f"  {'----':>6}  {'------':>8}  {'------':>8}  {'-----':>8}  {'-------':>8}  {'-------':>8}")

    for (yr, br, bdd), (_, fr, fdd) in zip(yb, yf_):
        d = fr - br
        print(f"  {yr:>6}  {br:>+7.1%}  {fr:>+7.1%}  {d:>+7.1%}  {bdd:>+7.1%}  {fdd:>+7.1%}")

    tb = base["eq"][-1] / base["eq"][0]
    tf = full["eq"][-1] / full["eq"][0]
    print(f"\n  Total return: Base {tb:.2f}x  |  Full {tf:.2f}x")

    # ── VIX event analysis ──
    print(f"\n{sep}")
    print("  VIX EVENT LOG (Full System 3)")
    print(sep)

    # Reconstruct events by re-running with logging
    dates = px.index
    vix_events = []
    dd_events = []
    sl_events = []
    cap = 1.0; hwm = 1.0
    w_dict = {s: 0.0 for s in STOCKS}
    ep_dict = {s: None for s in STOCKS}
    stopped_set = set()
    rc2 = REBAL
    log_r2 = np.log(px[STOCKS] / px[STOCKS].shift(1))

    for i in range(1, len(dates)):
        dt = dates[i]; prev = dates[i-1]
        pnl = sum(w_dict[s] * (px.loc[dt, s]/px.loc[prev, s]-1)
                   for s in STOCKS if w_dict[s] > 0)
        cap *= (1 + pnl)
        hwm = max(hwm, cap)
        dd = cap / hwm - 1

        # Stop loss
        for s in STOCKS:
            if w_dict[s] > 0 and ep_dict[s] and s not in stopped_set:
                loss = px.loc[dt, s] / ep_dict[s] - 1
                idx = log_r2.index.get_loc(dt)
                dv = float(log_r2[s].iloc[max(0,idx-20):idx].std()) if idx >= 20 else 0.10
                v5 = dv * np.sqrt(5)
                thr = max(SL_SIGMA * v5, SL_FLOOR)
                if loss < -thr:
                    sl_events.append((dt.strftime('%Y-%m-%d'), s,
                                      f"{loss:+.1%}", f"-{thr:.1%}",
                                      f"VIX={vix.iloc[i]:.0f}"))
                    cap -= w_dict[s] * cap * COST / 10_000
                    w_dict[s] = 0.0
                    stopped_set.add(s)

        rc2 += 1
        if rc2 >= REBAL:
            rc2 = 0
            scale = dd_scale(dd)
            dd_active = scale < 1.0
            vt = float(vix.iloc[i]); vy = float(vix.iloc[i-1])
            falling = vt < vy

            if dd_active:
                dd_events.append((dt.strftime('%Y-%m-%d'), f"DD={dd:+.1%}",
                                  f"scale={scale:.0%}", f"VIX={vt:.0f}"))

            if not dd_active:
                if vt > VIX_FREEZE and not falling:
                    vix_events.append((dt.strftime('%Y-%m-%d'), f"VIX={vt:.0f}",
                                       "FREEZE", f"DD={dd:+.1%}"))
                    continue
                elif vt > 40 and falling:
                    vix_events.append((dt.strftime('%Y-%m-%d'), f"VIX={vt:.0f}",
                                       "BOOST 15%", f"DD={dd:+.1%}"))
                elif vt > 30 and falling:
                    vix_events.append((dt.strftime('%Y-%m-%d'), f"VIX={vt:.0f}",
                                       "BOOST 12%", f"DD={dd:+.1%}"))

            pw_eff = (W_VIX40 if vt > 40 and falling and not dd_active
                      else W_VIX30 if vt > 30 and falling and not dd_active
                      else W_BASE) * scale
            for s in STOCKS:
                w_dict[s] = pw_eff
                ep_dict[s] = px.loc[dt, s]
                stopped_set.discard(s)

    print(f"\n  VIX Boost/Freeze events ({len(vix_events)} total):")
    for ev in vix_events[:30]:
        print(f"    {ev[0]}  {ev[1]:<10}  {ev[2]:<12}  {ev[3]}")
    if len(vix_events) > 30:
        print(f"    ... and {len(vix_events)-30} more")

    print(f"\n  DD Ladder activations ({len(dd_events)} total, showing first 20):")
    for ev in dd_events[:20]:
        print(f"    {ev[0]}  {ev[1]:<12}  {ev[2]:<12}  {ev[3]}")
    if len(dd_events) > 20:
        print(f"    ... and {len(dd_events)-20} more")

    print(f"\n  Stop-loss hits ({len(sl_events)} total):")
    for ev in sl_events:
        print(f"    {ev[0]}  {ev[1]:<6}  loss={ev[2]:>7}  thr={ev[3]:>7}  {ev[4]}")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == "__main__":
    main()
