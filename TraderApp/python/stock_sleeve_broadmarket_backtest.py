#!/usr/bin/env python3
"""
System 3 Broad Market Test
==========================
Tests System 3 (VIX boost + DD ladder + 2σ SL) across market cap segments:
  - Large cap: 150 from S&P 500
  - Mid cap:   150 from S&P 400
  - Small cap:  150 from S&P 600

Also tests ETF portfolio (System 2) with current T10C-Slim ETFs.

Stock selection: Top 10 by 20-day momentum (proxy for prediction model).
"""

import yfinance as yf
import numpy as np
import pandas as pd
import warnings, time, sys
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────
SPY = "SPY"; VIX_T = "^VIX"
START = "2016-01-01"; END = "2026-02-12"
COST = 10

# System 3 params
REBAL      = 5
TOP_N      = 10     # hold top 10 stocks
W_BASE     = 0.10
W_VIX30    = 0.12
W_VIX40    = 0.15
VIX_FREEZE = 50
SL_SIGMA   = 2.0
SL_FLOOR   = 0.10

DD_LADDER = [(-0.05, 10), (-0.08, 8), (-0.12, 6), (-0.18, 4), (-999., 2)]

# T10C-Slim ETF portfolio
ETF_RISK_ON  = {"SMH": 0.25, "USMV": 0.25, "QUAL": 0.20, "PDBC": 0.15,
                "COPX": 0.05, "URA": 0.05, "DBA": 0.05}
ETF_RISK_OFF = {"USMV": 0.25, "QUAL": 0.20, "GDX": 0.20, "PDBC": 0.15,
                "COPX": 0.05, "URA": 0.05, "DBA": 0.10}
ETF_TICKERS  = list(set(list(ETF_RISK_ON) + list(ETF_RISK_OFF)))


# ═══ Data Loading ═══════════════════════════════════════════
def get_index_tickers(url, col="Symbol"):
    """Fetch tickers from Wikipedia S&P index page."""
    import requests, io
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        sym_col = None
        for c in df.columns:
            cl = str(c).lower()
            if "symbol" in cl or "ticker" in cl:
                sym_col = c; break
        if sym_col is None:
            sym_col = df.columns[0]
        tickers = df[sym_col].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers = [t for t in tickers if t and t != "nan" and len(t) <= 5 and t[0].isalpha()]
        return tickers
    except Exception as e:
        print(f"    WARNING: Could not fetch from {url}: {e}")
        return get_fallback_tickers(url)


def get_fallback_tickers(url):
    """Hardcoded fallback lists if Wikipedia is unavailable."""
    if "500" in url:
        return [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
            "JPM","V","PG","XOM","HD","MA","AVGO","CVX","MRK","ABBV",
            "LLY","PEP","KO","COST","TMO","ADBE","MCD","WMT","CRM","CSCO",
            "ACN","ABT","LIN","DHR","NKE","TXN","PM","NEE","BMY","UPS",
            "RTX","AMGN","HON","QCOM","LOW","UNP","IBM","INTC","INTU","SPGI",
            "ELV","GE","CAT","BA","AXP","SBUX","ISRG","DE","GILD","BLK",
            "AMD","ADI","SYK","MDLZ","BKNG","LRCX","VRTX","PLD","ADP","REGN",
            "AMAT","MMC","CI","CB","DUK","SO","CME","PYPL","SCHW","BSX",
            "ZTS","PGR","AON","TMUS","ICE","MO","CL","FI","MCK","WM",
            "SHW","KLAC","SNPS","CDNS","APH","CMG","ABNB","PANW","MSI","ETN",
            "GD","PH","HUM","PSA","NXPI","MCHP","MSCI","ORLY","HCA","MNST",
            "ECL","ROP","CTAS","DXCM","AIG","STZ","KMB","SLB","TT","WELL",
            "HLT","CARR","OTIS","KDP","RSG","IQV","PCAR","TEL","CPRT","IDXX",
            "GWW","YUM","MTD","EW","BKR","DOW","CEG","ON","FAST","FANG",
            "VRSK","PPG","AMP","ALL","ODFL","CTSH","AWK","WBD","DLR","SPG",
        ]
    elif "400" in url:
        return [
            "DECK","WSM","RBC","BURL","FNF","POOL","MANH","ELS","RNR","UFPI",
            "TECH","LSTR","WBS","CBSH","FHI","SNX","IBOC","EXPO","BOH","CADE",
            "POWI","HUBG","ABM","ASGN","CWEN","KNF","NMIH","REXR","SKY","WDFC",
            "OZK","PBH","CWST","GGG","FSS","SBRA","SWX","LANC","FIZZ","MDU",
            "NDSN","TREX","TTEK","WTS","CW","CHE","LPX","AYI","SCI","ENSG",
            "MGEE","JBT","KWR","NEU","IPAR","NJR","ESE","EPAM","MKSI","NVST",
            "AZTA","OLED","SLAB","POWL","AVNT","BMI","ALIT","ALKS","VIAV","SEM",
            "OMCL","PRGS","SIGI","LNTH","FELE","MATX","SAIC","KRYS","PCVX","ENOV",
            "DORM","AIN","AVT","BCO","ICFI","ITRI","MTN","NGVT","RRX","SXT",
            "TGNA","TTMI","VCEL","VCYT","WK","XRAY","ZWS","ALLE","AMG","AOS",
            "BJ","CGNX","CLH","CRVL","CUZ","DCI","FBIN","GBCI","GH","HELE",
            "HMST","HQY","IAC","JBGS","KBR","KVUE","LITE","MASI","MEDP","MOD",
            "MTSI","NOVT","NVT","NXST","OGE","OLN","PCTY","PI","PNR","PSN",
            "RHP","RPM","SITE","SPSC","SSNC","STRA","TKR","TPX","TRMB","TWST",
            "VOYA","VSH","WAL","WEX","WMS","WYNN","X","ZI","COOP","CALM",
        ]
    elif "600" in url:
        return [
            "AAON","ABCB","ABG","AEIS","AMWD","APAM","ARCB","ASTE","AVAV","AXNX",
            "AX","BANF","BBSI","BCPC","BDC","BFAM","BGS","BKE","BLKB","BMI",
            "BOOT","BRC","CARG","CASH","CBT","CENTA","CEVA","CMCO","CMP","CNK",
            "CNMD","CNXN","COHU","CORT","CPK","CRC","CSWI","CVCO","DAN","DCOM",
            "DIN","DNOW","DOCN","DXC","EAT","EBC","EIG","ELAN","EPC","ETD",
            "EVTC","EXLS","FARO","FBK","FCFS","FIGS","FL","FLO","FLWS","FN",
            "FOXF","FRME","FSP","GEF","GKOS","GMED","GTY","GVA","HAFC","HALO",
            "HBI","HEES","HLI","HRMY","HTLF","HUBG","IBKR","ICHR","IIVI","INDB",
            "INTA","JACK","JJSF","KALU","KFRC","KMPR","KNSA","KTOS","LANC","LBRT",
            "LGIH","LMB","LMND","LNN","LPRO","LXP","MBUU","MCW","MGRC","MMSI",
            "MQ","MRCY","MUR","MYRG","NAVI","NBR","NEOG","NHC","NNI","NSIT",
            "NWE","OFG","OI","OMCL","ONB","ORA","OSIS","OTTR","OUT","PAYO",
            "PBH","PCVX","PEBO","PEGA","PGNY","PLMR","PLUS","POWL","PRFT","PRIM",
            "PRO","PTCT","PUMP","QLYS","RAMP","RDN","REPX","RGP","RHP","RMBS",
            "RRR","RXRX","SABR","SAH","SBCF","SIG","SKT","SLG","SM","SMPL",
        ]
    return []


def download_batch(tickers, label="", min_bars=1500):
    """Download price data for tickers, return DataFrame of Close prices."""
    print(f"  Downloading {label} ({len(tickers)} tickers)...")
    # Download in chunks to avoid timeout
    chunk_size = 100
    all_px = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            raw = yf.download(chunk, start=START, end=END,
                              auto_adjust=True, progress=False, threads=True)
            if "Close" in raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else "Close" in raw.columns:
                if isinstance(raw.columns, pd.MultiIndex):
                    px = raw["Close"]
                else:
                    px = raw[["Close"]].rename(columns={"Close": chunk[0]})
                all_px.append(px)
        except Exception as e:
            print(f"    Chunk {i//chunk_size + 1} failed: {e}")
        time.sleep(0.5)

    if not all_px:
        return pd.DataFrame()

    px = pd.concat(all_px, axis=1)
    px = px.ffill()

    # Filter: need min_bars of data
    valid = []
    for col in px.columns:
        if px[col].notna().sum() >= min_bars:
            valid.append(col)
    px = px[valid]

    print(f"    Got {len(valid)} stocks with >= {min_bars} bars")
    return px


# ═══ System 3 Portfolio Engine ══════════════════════════════
def dd_max_stocks(dd):
    for thr, n in DD_LADDER:
        if dd > thr:
            return n
    return 2


def run_s3(stock_px, vix, top_n=TOP_N, use_vix=True, use_dd=True, use_sl=True):
    """
    Run System 3 on a universe of stocks.
    Every 5 days: rank by 20d momentum, hold top_n at W_BASE weight.
    Apply VIX boost, DD ladder, per-stock stop loss.
    """
    dates  = stock_px.index
    stocks = stock_px.columns.tolist()
    N_univ = len(stocks)
    log_r  = np.log(stock_px / stock_px.shift(1))

    cap = 1.0; hwm = 1.0
    holdings = {}   # {stock: weight}
    entry_px = {}   # {stock: entry_price}
    stopped  = set()

    eq = [cap]
    rc = REBAL   # trigger first rebalance immediately
    trades = 0; stops = 0; vboost = 0; ddcuts = 0; freezes = 0
    turnover_shares = 0

    for i in range(1, len(dates)):
        dt   = dates[i]
        prev = dates[i - 1]

        # 1. Daily P&L
        pnl = 0.0
        for s, w in holdings.items():
            if w > 0 and s in stock_px.columns:
                p1 = stock_px.loc[dt, s]
                p0 = stock_px.loc[prev, s]
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    pnl += w * (p1 / p0 - 1)
        cap *= (1 + pnl)
        hwm = max(hwm, cap)
        dd = cap / hwm - 1

        # 2. Daily stop loss
        if use_sl:
            to_stop = []
            for s, w in holdings.items():
                if w > 0 and s in entry_px and s not in stopped:
                    ep = entry_px[s]
                    cp = stock_px.loc[dt, s]
                    if pd.isna(cp) or pd.isna(ep) or ep <= 0:
                        continue
                    loss = cp / ep - 1
                    idx = log_r.index.get_loc(dt)
                    if idx >= 20 and s in log_r.columns:
                        dv = float(log_r[s].iloc[idx-20:idx].std())
                        if np.isnan(dv): dv = 0.10
                        v5 = dv * np.sqrt(5)
                    else:
                        v5 = 0.10
                    thr = max(SL_SIGMA * v5, SL_FLOOR)
                    if loss < -thr:
                        to_stop.append(s)
            for s in to_stop:
                cap -= holdings[s] * cap * COST / 10_000
                holdings[s] = 0.0
                stopped.add(s)
                stops += 1; trades += 1

        # 3. Rebalance
        rc += 1
        if rc >= REBAL:
            rc = 0

            # DD ladder
            if use_dd:
                n_hold = dd_max_stocks(dd)
                n_hold = min(n_hold, top_n)  # can't hold more than top_n
                dd_active = (n_hold < top_n)
            else:
                n_hold = top_n
                dd_active = False

            if dd_active:
                ddcuts += 1

            # VIX
            vt = float(vix.iloc[i]) if i < len(vix) else 20.0
            vy = float(vix.iloc[i-1]) if i > 0 else vt
            falling = vt < vy

            if use_vix and not dd_active:
                if vt > VIX_FREEZE and not falling:
                    freezes += 1
                    eq.append(cap)
                    continue
                elif vt > 40 and falling:
                    pw = W_VIX40; vboost += 1
                elif vt > 30 and falling:
                    pw = W_VIX30; vboost += 1
                else:
                    pw = W_BASE
            else:
                pw = W_BASE

            # Rank by 20d momentum -> select top n_hold
            if i >= 20:
                mom = {}
                for s in stocks:
                    p_now  = stock_px.loc[dt, s]
                    p_prev = stock_px.iloc[max(0, i-20)][s]
                    if pd.notna(p_now) and pd.notna(p_prev) and p_prev > 0:
                        mom[s] = p_now / p_prev - 1
                ranked = sorted(mom.keys(), key=lambda x: mom[x], reverse=True)
                selected = set(ranked[:n_hold])
            else:
                selected = set(stocks[:n_hold])

            # Update holdings
            new_holdings = {}
            for s in stocks:
                old_w = holdings.get(s, 0.0)
                new_w = pw if s in selected else 0.0
                if abs(new_w - old_w) > 0.001:
                    cap -= abs(new_w - old_w) * cap * COST / 10_000
                    trades += 1
                    turnover_shares += 1
                new_holdings[s] = new_w
                if new_w > 0:
                    entry_px[s] = stock_px.loc[dt, s]
                    stopped.discard(s)
                else:
                    entry_px.pop(s, None)
                    stopped.discard(s)
            holdings = {s: w for s, w in new_holdings.items() if w > 0}

        eq.append(cap)

    eq = np.array(eq)
    return dict(eq=eq, dates=dates, trades=trades, stops=stops,
                vboost=vboost, ddcuts=ddcuts, freezes=freezes)


# ═══ ETF Portfolio Engine ═══════════════════════════════════
def run_etf_portfolio(etf_px, vix, spy_px, weights=None, rebal_days=21):
    """Simple ETF portfolio with fixed weights, rebalanced every N days."""
    if weights is None:
        weights = ETF_RISK_ON
    dates = etf_px.index
    tickers = [t for t in weights if t in etf_px.columns]
    if not tickers:
        return None

    # Normalize weights to available tickers
    total_w = sum(weights[t] for t in tickers)
    w = {t: weights[t] / total_w for t in tickers}

    cap = 1.0; hwm = 1.0
    held_w = {t: 0.0 for t in tickers}
    eq = [cap]
    rc = rebal_days
    trades = 0

    for i in range(1, len(dates)):
        dt = dates[i]; prev = dates[i-1]
        pnl = 0.0
        for t in tickers:
            if held_w[t] > 0:
                p1 = etf_px.loc[dt, t]
                p0 = etf_px.loc[prev, t]
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    pnl += held_w[t] * (p1/p0 - 1)
        cap *= (1 + pnl)
        hwm = max(hwm, cap)

        rc += 1
        if rc >= rebal_days:
            rc = 0
            for t in tickers:
                old = held_w[t]; new = w[t]
                if abs(new - old) > 0.001:
                    cap -= abs(new - old) * cap * COST / 10_000
                    trades += 1
                held_w[t] = new
        eq.append(cap)

    eq = np.array(eq)
    return dict(eq=eq, dates=dates, trades=trades,
                stops=0, vboost=0, ddcuts=0, freezes=0)


# ═══ Metrics ════════════════════════════════════════════════
def met(eq, dates):
    yrs  = (dates[-1] - dates[0]).days / 365.25
    cagr = (eq[-1] / eq[0]) ** (1/yrs) - 1
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
        ret = ye[-1] / ye[0] - 1
        hm = np.maximum.accumulate(ye)
        dd = np.min(ye / hm - 1)
        rows.append((yr, ret, dd))
    return rows


# ═══ Main ═══════════════════════════════════════════════════
def main():
    W = 115
    sep = "=" * W

    print(sep)
    print("  SYSTEM 3 BROAD MARKET TEST + ETF PORTFOLIO")
    print("  Stock Sleeve: Top-10 by momentum from 150 per cap segment")
    print("  ETF Portfolio: T10C-Slim Risk-ON weights, 21d rebalance")
    print(sep)

    # ── 1. Get stock lists ──
    print("\n[1/4] Fetching stock lists...")
    sp500 = get_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    print(f"  S&P 500: {len(sp500)} tickers")

    sp400 = get_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    print(f"  S&P 400: {len(sp400)} tickers")

    sp600 = get_index_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
    print(f"  S&P 600: {len(sp600)} tickers")

    # Take first 150 from each (after download filtering)
    segments = {
        "Large Cap (SP500)": sp500[:200],   # request 200, filter to 150
        "Mid Cap (SP400)":   sp400[:200],
        "Small Cap (SP600)": sp600[:200],
    }

    # ── 2. Download data ──
    print(f"\n[2/4] Downloading price data...")

    # Common data
    common = yf.download([SPY, VIX_T], start=START, end=END,
                         auto_adjust=True, progress=False)
    spy_px = common["Close"][SPY].ffill()
    vix_px = common["Close"][VIX_T].ffill()

    # ETF data
    etf_raw = yf.download(ETF_TICKERS + [SPY], start=START, end=END,
                          auto_adjust=True, progress=False)
    etf_px  = etf_raw["Close"].ffill().dropna(how="all")

    # Stock data per segment
    seg_data = {}
    for seg_name, tickers in segments.items():
        # Remove any overlap with ETFs or SPY/VIX
        tickers = [t for t in tickers
                   if t not in ETF_TICKERS and t != SPY and t != VIX_T]
        px = download_batch(tickers, label=seg_name, min_bars=1500)
        if len(px.columns) > 150:
            px = px[px.columns[:150]]
        seg_data[seg_name] = px
        print(f"    {seg_name}: {len(px.columns)} stocks ready")

    # ── 3. Run backtests ──
    print(f"\n[3/4] Running backtests...")

    all_results = {}

    # SPY B&H
    spy_eq = spy_px.values / spy_px.values[0]
    spy_m  = met(spy_eq, spy_px.index)
    all_results["SPY B&H"] = {**spy_m, "trades": 0, "stops": 0,
                               "vboost": 0, "ddcuts": 0, "freezes": 0,
                               "eq": spy_eq, "dates": spy_px.index}

    # ETF Portfolio (T10C-Slim Risk-ON, 21d rebalance)
    print("  ETF Portfolio (T10C-Slim Risk-ON 21d)...")
    etf_common = etf_px.dropna(subset=[t for t in ETF_RISK_ON if t in etf_px.columns])
    if len(etf_common) > 0:
        etf_vix = vix_px.reindex(etf_common.index).ffill()
        etf_spy = spy_px.reindex(etf_common.index).ffill()
        etf_r = run_etf_portfolio(etf_common, etf_vix, etf_spy,
                                  weights=ETF_RISK_ON, rebal_days=21)
        if etf_r:
            etf_m = met(etf_r["eq"], etf_r["dates"])
            all_results["ETF T10C-Slim"] = {**etf_m, **etf_r}

    # Per-segment System 3
    for seg_name, px in seg_data.items():
        if px.empty or len(px.columns) < 10:
            print(f"  SKIP {seg_name}: not enough stocks")
            continue

        # Align VIX
        common_idx = px.index
        vix_aligned = vix_px.reindex(common_idx).ffill()

        # Baseline: top-10 momentum, no risk layers
        print(f"  {seg_name} — Baseline...")
        r_base = run_s3(px, vix_aligned, use_vix=False, use_dd=False, use_sl=False)
        m_base = met(r_base["eq"], r_base["dates"])
        all_results[f"{seg_name} Base"] = {**m_base, **r_base}

        # Full System 3
        print(f"  {seg_name} — Full System 3...")
        r_full = run_s3(px, vix_aligned, use_vix=True, use_dd=True, use_sl=True)
        m_full = met(r_full["eq"], r_full["dates"])
        all_results[f"{seg_name} Sys3"] = {**m_full, **r_full}

        # SL only
        print(f"  {seg_name} — SL only...")
        r_sl = run_s3(px, vix_aligned, use_vix=False, use_dd=False, use_sl=True)
        m_sl = met(r_sl["eq"], r_sl["dates"])
        all_results[f"{seg_name} +SL"] = {**m_sl, **r_sl}

        # DD only
        print(f"  {seg_name} — DD only...")
        r_dd = run_s3(px, vix_aligned, use_vix=False, use_dd=True, use_sl=False)
        m_dd = met(r_dd["eq"], r_dd["dates"])
        all_results[f"{seg_name} +DD"] = {**m_dd, **r_dd}

    # ── 4. Results ──
    print(f"\n[4/4] Results")
    print(f"\n{sep}")
    print("  OVERALL PERFORMANCE")
    print(sep)

    hdr = (f"  {'Strategy':<28} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} "
           f"{'Calmar':>8} {'Vol':>8} {'Trd':>6} {'SL':>4} {'VIX+':>4} "
           f"{'DD-':>4} {'Frz':>4}")
    print(hdr)
    print("  " + "-" * (W - 2))

    display_order = [
        "SPY B&H", "ETF T10C-Slim",
    ]
    # Add segment results in order
    for seg_name in segments:
        for suffix in ["Base", "+SL", "+DD", "Sys3"]:
            key = f"{seg_name} {suffix}"
            if key in all_results:
                display_order.append(key)

    for name in display_order:
        if name not in all_results:
            continue
        r = all_results[name]
        trd = r.get("trades", 0)
        sl  = r.get("stops", 0)
        vb  = r.get("vboost", 0)
        dd  = r.get("ddcuts", 0)
        frz = r.get("freezes", 0)
        print(f"  {name:<28} {r['CAGR']:>+7.2%} {r['MaxDD']:>+7.1%} "
              f"{r['Sharpe']:>8.3f} {r['Calmar']:>8.3f} {r['Vol']:>7.2%} "
              f"{trd:>6} {sl:>4} {vb:>4} {dd:>4} {frz:>4}")

    # ── Delta tables per segment ──
    print(f"\n{sep}")
    print("  SYSTEM 3 IMPROVEMENT BY SEGMENT (Full Sys3 vs Baseline)")
    print(sep)

    print(f"  {'Segment':<28} {'dCAGR':>8} {'dMaxDD':>8} {'dSharpe':>9} {'dCalmar':>9}")
    print("  " + "-" * 65)

    for seg_name in segments:
        bk = f"{seg_name} Base"
        fk = f"{seg_name} Sys3"
        if bk in all_results and fk in all_results:
            b = all_results[bk]; f = all_results[fk]
            print(f"  {seg_name:<28} "
                  f"{f['CAGR']-b['CAGR']:>+7.2%} "
                  f"{f['MaxDD']-b['MaxDD']:>+7.1%} "
                  f"{f['Sharpe']-b['Sharpe']:>+9.3f} "
                  f"{f['Calmar']-b['Calmar']:>+9.3f}")

    # ── Layer contribution per segment ──
    print(f"\n{sep}")
    print("  LAYER CONTRIBUTION BY SEGMENT")
    print(sep)

    for seg_name in segments:
        bk  = f"{seg_name} Base"
        slk = f"{seg_name} +SL"
        ddk = f"{seg_name} +DD"
        fk  = f"{seg_name} Sys3"

        if not all(k in all_results for k in [bk, slk, ddk, fk]):
            continue

        b  = all_results[bk]
        sl = all_results[slk]
        dd = all_results[ddk]
        f  = all_results[fk]

        print(f"\n  {seg_name}:")
        print(f"    {'Layer':<16} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
        print(f"    {'-----':<16} {'----':>8} {'-----':>8} {'------':>8} {'------':>8}")
        print(f"    {'Baseline':<16} {b['CAGR']:>+7.2%} {b['MaxDD']:>+7.1%} "
              f"{b['Sharpe']:>8.3f} {b['Calmar']:>8.3f}")
        print(f"    {'+ SL':<16} {sl['CAGR']:>+7.2%} {sl['MaxDD']:>+7.1%} "
              f"{sl['Sharpe']:>8.3f} {sl['Calmar']:>8.3f}")
        print(f"    {'+ DD':<16} {dd['CAGR']:>+7.2%} {dd['MaxDD']:>+7.1%} "
              f"{dd['Sharpe']:>8.3f} {dd['Calmar']:>8.3f}")
        print(f"    {'FULL Sys3':<16} {f['CAGR']:>+7.2%} {f['MaxDD']:>+7.1%} "
              f"{f['Sharpe']:>8.3f} {f['Calmar']:>8.3f}")

    # ── Yearly per segment ──
    print(f"\n{sep}")
    print("  YEARLY: Baseline vs System 3 per segment")
    print(sep)

    for seg_name in segments:
        bk = f"{seg_name} Base"
        fk = f"{seg_name} Sys3"
        if bk not in all_results or fk not in all_results:
            continue

        b = all_results[bk]; f = all_results[fk]
        yb = yearly(b["eq"], b["dates"])
        yf_ = yearly(f["eq"], f["dates"])

        print(f"\n  {seg_name}:")
        print(f"  {'Yr':>6}  {'Base':>8}  {'Sys3':>8}  {'Delta':>8}  {'B DD':>8}  {'S3 DD':>8}")
        print(f"  {'----':>6}  {'----':>8}  {'----':>8}  {'-----':>8}  {'----':>8}  {'-----':>8}")

        for (yr, br, bdd), (_, fr, fdd) in zip(yb, yf_):
            print(f"  {yr:>6}  {br:>+7.1%}  {fr:>+7.1%}  {fr-br:>+7.1%}  "
                  f"{bdd:>+7.1%}  {fdd:>+7.1%}")

        tb = b["eq"][-1] / b["eq"][0]
        tf = f["eq"][-1] / f["eq"][0]
        print(f"    Total: Base {tb:.2f}x  Sys3 {tf:.2f}x")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == "__main__":
    main()
