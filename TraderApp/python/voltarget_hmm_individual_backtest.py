#!/usr/bin/env python3
"""
Vol-Target + HMM Individual Stock Backtest
==========================================
Each stock runs INDEPENDENTLY through the same algorithm:

  1. blended_vol = 0.7 * stock_20d_vol + 0.3 * stock_60d_vol  (floor 8%, cap 40%)
  2. base_exposure = 0.12 / blended_vol  (cap 1.0)
  3. HMM risk adjustment (from SPY):
       p_risk < 0.5  → ×1.00
       p_risk 0.5-0.9 → ×0.75
       p_risk ≥ 0.9  → ×0.50
  4. MA200 cap (from SPY):
       SPY > MA200       → 1.00
       SPY < MA200       → 0.60
       SPY < MA200×0.95  → 0.30
  5. final_exposure = min(hmm_adjusted, ma200_cap, 0.95)
  6. Deadband: up 2%, down 5%; max step 15%
  7. Rebalance every 21 trading days

Compare: B&H vs MA200 only vs MA200 + HMM (no vol-target, default 100% exposure)
Tests on 150 stocks per cap segment (S&P 500/400/600).
"""

import yfinance as yf
import numpy as np
import pandas as pd
import warnings, time, sys, io, requests
warnings.filterwarnings("ignore")

SPY_T = "SPY"; VIX_T = "^VIX"
START = "2014-01-01"  # extra buffer for MA200 + vol warmup
END   = "2026-02-12"
BT_START = "2016-01-01"  # actual backtest start
COST  = 10  # round-trip bps

# Vol-target params
TARGET_VOL = 0.12
VOL_SHORT  = 20
VOL_LONG   = 60
VOL_ALPHA  = 0.7
VOL_FLOOR  = 0.08
VOL_CAP    = 0.40
MAX_LEV    = 1.0

# MA200 caps
MA200_SHALLOW = 0.60
MA200_DEEP    = 0.30
MA200_DEEP_THR = 0.95  # SPY < MA200 * 0.95

# Execution
REBAL       = 21
DB_UP       = 0.02
DB_DOWN     = 0.05
MAX_STEP    = 0.15
MAX_EXP     = 0.95

# HMM proxy params (vol-ratio sigmoid)
HMM_CENTER  = 1.3   # vol_ratio center for sigmoid
HMM_STEEPNESS = 4.0
HMM_EMA_SPAN = 5


# ═══ Data ═══════════════════════════════════════════════════════
def get_index_tickers(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        sym_col = None
        for c in df.columns:
            if "symbol" in str(c).lower() or "ticker" in str(c).lower():
                sym_col = c; break
        if sym_col is None: sym_col = df.columns[0]
        tickers = df[sym_col].astype(str).str.replace(".", "-", regex=False).tolist()
        return [t for t in tickers if t and t != "nan" and len(t) <= 5 and t[0].isalpha()]
    except:
        return get_fallback(url)

def get_fallback(url):
    if "500" in url:
        return ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
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
                "VRSK","PPG","AMP","ALL","ODFL","CTSH","AWK","WBD","DLR","SPG"]
    elif "400" in url:
        return ["DECK","WSM","RBC","BURL","FNF","POOL","MANH","ELS","RNR","UFPI",
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
                "VOYA","VSH","WAL","WEX","WMS","WYNN","X","ZI","COOP","CALM"]
    elif "600" in url:
        return ["AAON","ABCB","ABG","AEIS","AMWD","APAM","ARCB","ASTE","AVAV","AXNX",
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
                "RRR","RXRX","SABR","SAH","SBCF","SIG","SKT","SLG","SM","SMPL"]
    return []

def download_batch(tickers, label="", min_bars=1500):
    print(f"  Downloading {label} ({len(tickers)} tickers)...")
    chunk = 100
    all_px = []
    for i in range(0, len(tickers), chunk):
        c = tickers[i:i+chunk]
        try:
            raw = yf.download(c, start=START, end=END, auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                all_px.append(raw["Close"])
            elif len(c) == 1:
                all_px.append(raw[["Close"]].rename(columns={"Close": c[0]}))
        except Exception as e:
            print(f"    Chunk failed: {e}")
        time.sleep(0.3)
    if not all_px: return pd.DataFrame()
    px = pd.concat(all_px, axis=1).ffill()
    valid = [c for c in px.columns if px[c].notna().sum() >= min_bars]
    px = px[valid]
    print(f"    {len(valid)} stocks with >= {min_bars} bars")
    return px


# ═══ HMM Proxy (vol-ratio sigmoid) ═════════════════════════════
def compute_p_risk(spy_log_ret):
    """Compute p_risk proxy from SPY realized vol ratio."""
    v20 = spy_log_ret.rolling(VOL_SHORT).std() * np.sqrt(252)
    v60 = spy_log_ret.rolling(VOL_LONG).std() * np.sqrt(252)
    ratio = v20 / v60
    p = 1.0 / (1.0 + np.exp(-(ratio - HMM_CENTER) * HMM_STEEPNESS))
    p = p.ewm(span=HMM_EMA_SPAN).mean().fillna(0.0)
    return p

def try_hmmlearn_p_risk(spy_log_ret, window=500, retrain_freq=21):
    """Try proper HMM via hmmlearn. Returns None if unavailable."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None

    dates = spy_log_ret.index
    p_risk = pd.Series(0.0, index=dates)
    X_full = spy_log_ret.values.reshape(-1, 1)
    model = None
    crisis_idx = 0

    for i in range(window, len(dates)):
        if model is None or i % retrain_freq == 0:
            X_train = X_full[max(0, i-window):i]
            mask = ~np.isnan(X_train.flatten())
            X_clean = X_train[mask].reshape(-1, 1)
            if len(X_clean) < 200:
                continue
            try:
                m = GaussianHMM(n_components=2, covariance_type="full",
                                n_iter=50, random_state=42, tol=0.01)
                m.fit(X_clean)
                crisis_idx = int(np.argmax(m.covars_.flatten()))
                model = m
            except:
                continue

        if model is not None and not np.isnan(X_full[i, 0]):
            try:
                probs = model.predict_proba(X_full[i:i+1])
                p_risk.iloc[i] = probs[0, crisis_idx]
            except:
                p_risk.iloc[i] = p_risk.iloc[i-1] if i > 0 else 0.0

    p_risk = p_risk.ewm(span=HMM_EMA_SPAN).mean()
    return p_risk


# ═══ Per-Stock Backtest Engine ══════════════════════════════════
def run_single_stock(stock_px, spy_px, spy_ma200, p_risk, use_hmm=True, use_ma200=True):
    """
    Run regime-based system on a single stock (NO vol-target).
    Default exposure = 100%.
    MA200 cap + HMM risk gate only.
    """
    dates = stock_px.index

    cap = 1.0
    cur_exp = 0.0
    eq = [cap]
    trades = 0
    rc = REBAL  # trigger first rebalance immediately

    for i in range(1, len(dates)):
        dt = dates[i]

        # Daily P&L
        if cur_exp > 0:
            r = stock_px.iloc[i] / stock_px.iloc[i-1] - 1
            if not np.isnan(r):
                cap *= (1 + cur_exp * r)

        # Rebalance every 21 days
        rc += 1
        if rc >= REBAL:
            rc = 0

            # 1. Start at 100%
            base_exp = 1.0

            # 2. HMM adjustment
            if use_hmm and dt in p_risk.index:
                pr = float(p_risk.loc[dt].iloc[0]) if hasattr(p_risk.loc[dt], 'iloc') else float(p_risk.loc[dt])
                if pr >= 0.9:
                    base_exp *= 0.50
                elif pr >= 0.5:
                    base_exp *= 0.75

            # 3. MA200 cap
            if use_ma200 and dt in spy_px.index and dt in spy_ma200.index:
                spy_p = float(spy_px.loc[dt].iloc[0]) if hasattr(spy_px.loc[dt], 'iloc') else float(spy_px.loc[dt])
                ma = float(spy_ma200.loc[dt].iloc[0]) if hasattr(spy_ma200.loc[dt], 'iloc') else float(spy_ma200.loc[dt])
                if not np.isnan(spy_p) and not np.isnan(ma) and ma > 0:
                    if spy_p < ma * MA200_DEEP_THR:
                        base_exp = min(base_exp, MA200_DEEP)
                    elif spy_p < ma:
                        base_exp = min(base_exp, MA200_SHALLOW)

            target_exp = base_exp

            # Apply change + transaction cost
            if abs(target_exp - cur_exp) > 0.001:
                cap -= abs(target_exp - cur_exp) * cap * COST / 10_000
                cur_exp = target_exp
                trades += 1

        eq.append(cap)

    eq = np.array(eq)
    return eq, trades


def met(eq, dates):
    n = min(len(eq), len(dates))
    eq = eq[:n]; dates = dates[:n]
    yrs = (dates[-1] - dates[0]).days / 365.25
    if yrs <= 0 or eq[0] <= 0: return dict(CAGR=0, MaxDD=0, Sharpe=0, Calmar=0, Vol=0)
    cagr = (eq[-1] / eq[0]) ** (1/yrs) - 1
    r = np.diff(eq) / eq[:-1]
    r = r[~np.isnan(r)]
    vol = np.std(r) * np.sqrt(252) if len(r) > 0 else 0
    sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
    rm = np.maximum.accumulate(eq)
    mdd = np.min(eq / rm - 1)
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(CAGR=cagr, MaxDD=mdd, Sharpe=sh, Calmar=cal, Vol=vol)


# ═══ Main ═══════════════════════════════════════════════════════
def main():
    W = 115
    sep = "=" * W
    print(sep)
    print("  MA200 + HMM: Individual Stock Backtest (per stock, NOT portfolio)")
    print("  21d rebalance | default 100% | HMM risk gate | MA200 cap | NO vol-target")
    print(sep)

    # 1. Get stock lists
    print("\n[1/4] Fetching stock lists...")
    sp500 = get_index_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp400 = get_index_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    sp600 = get_index_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
    print(f"  SP500: {len(sp500)}, SP400: {len(sp400)}, SP600: {len(sp600)}")

    segments = {
        "Large (SP500)": sp500[:200],
        "Mid (SP400)":   sp400[:200],
        "Small (SP600)": sp600[:200],
    }

    # 2. Download
    print(f"\n[2/4] Downloading price data...")
    common = yf.download([SPY_T], start=START, end=END, auto_adjust=True, progress=False)
    spy_px = common["Close"].ffill() if "Close" in common.columns else common[("Close", SPY_T)].ffill()

    spy_log_ret = np.log(spy_px / spy_px.shift(1))
    spy_ma200 = spy_px.rolling(200).mean()

    # Compute p_risk
    print("  Computing HMM p_risk...")
    p_risk_hmm = try_hmmlearn_p_risk(spy_log_ret)
    if p_risk_hmm is not None:
        p_risk = p_risk_hmm
        print("    Using hmmlearn GaussianHMM (2-state)")
    else:
        p_risk = compute_p_risk(spy_log_ret)
        print("    Using vol-ratio proxy (hmmlearn not installed)")

    # Quick stats on p_risk
    bt_mask = p_risk.index >= BT_START
    pr_bt = p_risk[bt_mask]
    print(f"    p_risk stats (backtest period): mean={pr_bt.mean():.3f} "
          f"median={pr_bt.median():.3f} >0.5={(pr_bt>0.5).mean():.1%} >0.9={(pr_bt>0.9).mean():.1%}")

    seg_data = {}
    for seg_name, tickers in segments.items():
        tickers = [t for t in tickers if t != SPY_T and t != VIX_T]
        px = download_batch(tickers, label=seg_name, min_bars=1500)
        if len(px.columns) > 150:
            px = px[px.columns[:150]]
        seg_data[seg_name] = px
        print(f"    {seg_name}: {len(px.columns)} stocks")

    # 3. Run backtests
    print(f"\n[3/4] Running per-stock backtests...")

    # Trim to backtest period
    bt_start_dt = pd.Timestamp(BT_START)

    all_seg_results = {}

    for seg_name, px in seg_data.items():
        if px.empty or len(px.columns) < 10:
            print(f"  SKIP {seg_name}")
            continue

        stocks = px.columns.tolist()
        n = len(stocks)
        print(f"\n  {seg_name} ({n} stocks)...")

        # Run 3 variants per stock: B&H, MA200 only, MA200+HMM
        bh_metrics = []
        orig_metrics = []
        new_metrics = []

        for j, s in enumerate(stocks):
            if (j+1) % 50 == 0:
                print(f"    {j+1}/{n}...")

            stock_px = px[s].dropna()
            # Align with backtest period
            stock_bt = stock_px[stock_px.index >= bt_start_dt]
            if len(stock_bt) < 500:
                continue

            dates_bt = stock_bt.index

            # B&H
            bh_eq = stock_bt.values / stock_bt.values[0]
            bh_m = met(bh_eq, dates_bt)

            # Align spy/p_risk to stock dates
            spy_aligned = spy_px.reindex(stock_px.index).ffill()
            ma_aligned = spy_ma200.reindex(stock_px.index).ffill()
            pr_aligned = p_risk.reindex(stock_px.index).fillna(0)

            # Original: MA200 only (no HMM)
            eq_orig, tr_orig = run_single_stock(
                stock_px, spy_aligned, ma_aligned, pr_aligned,
                use_hmm=False, use_ma200=True)
            # Trim to backtest period
            bt_idx = stock_px.index.get_indexer([bt_start_dt], method="nearest")[0]
            eq_orig_bt = eq_orig[bt_idx:] / eq_orig[bt_idx] if bt_idx < len(eq_orig) else eq_orig
            dates_orig = stock_px.index[bt_idx:bt_idx+len(eq_orig_bt)]
            if len(eq_orig_bt) < 500:
                continue
            orig_m = met(eq_orig_bt, dates_orig)

            # New: MA200 + HMM
            eq_new, tr_new = run_single_stock(
                stock_px, spy_aligned, ma_aligned, pr_aligned,
                use_hmm=True, use_ma200=True)
            eq_new_bt = eq_new[bt_idx:] / eq_new[bt_idx] if bt_idx < len(eq_new) else eq_new
            new_m = met(eq_new_bt, dates_orig[:len(eq_new_bt)])

            bh_metrics.append(bh_m)
            orig_metrics.append({**orig_m, "trades": tr_orig})
            new_metrics.append({**new_m, "trades": tr_new})

        if not bh_metrics:
            continue

        all_seg_results[seg_name] = {
            "n": len(bh_metrics),
            "bh": bh_metrics,
            "orig": orig_metrics,
            "new": new_metrics,
        }

    # 4. Results
    print(f"\n[4/4] Results")
    print(f"\n{sep}")
    print("  AGGREGATE RESULTS BY SEGMENT (median across all stocks)")
    print(sep)

    def agg(ms, key):
        vals = [m[key] for m in ms if not np.isnan(m[key])]
        return np.median(vals) if vals else 0

    def agg_mean(ms, key):
        vals = [m[key] for m in ms if not np.isnan(m[key])]
        return np.mean(vals) if vals else 0

    hdr = (f"  {'Segment + Strategy':<32} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} "
           f"{'Calmar':>8} {'Vol':>8}  {'n':>4}")
    print(hdr)
    print("  " + "-" * (W - 2))

    for seg_name, sr in all_seg_results.items():
        n = sr["n"]
        for label, data in [("B&H", sr["bh"]),
                            ("MA200 only", sr["orig"]),
                            ("MA200 + HMM", sr["new"])]:
            name = f"{seg_name} {label}"
            print(f"  {name:<32} "
                  f"{agg(data,'CAGR'):>+7.2%} {agg(data,'MaxDD'):>+7.1%} "
                  f"{agg(data,'Sharpe'):>8.3f} {agg(data,'Calmar'):>8.3f} "
                  f"{agg(data,'Vol'):>7.2%}  {n:>4}")
        print()

    # Delta table: HMM improvement
    print(f"\n{sep}")
    print("  HMM IMPROVEMENT (New vs Original, median delta per stock)")
    print(sep)

    print(f"  {'Segment':<20} {'dCAGR':>8} {'dMaxDD':>8} {'dSharpe':>9} {'dCalmar':>9}  "
          f"{'%Calmar+':>9} {'%Sharpe+':>9}")
    print("  " + "-" * 80)

    for seg_name, sr in all_seg_results.items():
        n = sr["n"]
        d_cagr = [sr["new"][i]["CAGR"] - sr["orig"][i]["CAGR"] for i in range(n)]
        d_mdd  = [sr["new"][i]["MaxDD"] - sr["orig"][i]["MaxDD"] for i in range(n)]
        d_sh   = [sr["new"][i]["Sharpe"] - sr["orig"][i]["Sharpe"] for i in range(n)]
        d_cal  = [sr["new"][i]["Calmar"] - sr["orig"][i]["Calmar"] for i in range(n)]

        pct_cal_better = sum(1 for x in d_cal if x > 0.001) / n
        pct_sh_better  = sum(1 for x in d_sh if x > 0.001) / n

        print(f"  {seg_name:<20} "
              f"{np.median(d_cagr):>+7.2%} {np.median(d_mdd):>+7.1%} "
              f"{np.median(d_sh):>+9.3f} {np.median(d_cal):>+9.3f}  "
              f"{pct_cal_better:>8.0%} {pct_sh_better:>8.0%}")

    # Distribution analysis
    print(f"\n{sep}")
    print("  DISTRIBUTION: Calmar ratio (Original vs New)")
    print(sep)

    for seg_name, sr in all_seg_results.items():
        orig_cals = sorted([m["Calmar"] for m in sr["orig"]])
        new_cals  = sorted([m["Calmar"] for m in sr["new"]])
        n = len(orig_cals)

        pcts = [10, 25, 50, 75, 90]
        print(f"\n  {seg_name} ({n} stocks):")
        print(f"    {'Percentile':<12} {'Original':>10} {'+ HMM':>10} {'Delta':>10}")
        print(f"    {'----------':<12} {'--------':>10} {'-----':>10} {'-----':>10}")
        for p in pcts:
            idx = min(int(n * p / 100), n - 1)
            o = orig_cals[idx]
            nw = new_cals[idx]
            print(f"    {'P' + str(p):<12} {o:>10.3f} {nw:>10.3f} {nw-o:>+10.3f}")

    # p_risk regime stats
    print(f"\n{sep}")
    print("  p_risk REGIME DISTRIBUTION (backtest period)")
    print(sep)

    pr_bt = p_risk[p_risk.index >= BT_START]
    total = len(pr_bt)
    low = (pr_bt < 0.5).sum()
    mid = ((pr_bt >= 0.5) & (pr_bt < 0.9)).sum()
    high = (pr_bt >= 0.9).sum()
    print(f"  p_risk < 0.5  (×1.00): {low:>5} days ({low/total:>5.1%})")
    print(f"  p_risk 0.5-0.9 (×0.75): {mid:>5} days ({mid/total:>5.1%})")
    print(f"  p_risk ≥ 0.9  (×0.50): {high:>5} days ({high/total:>5.1%})")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == "__main__":
    main()
