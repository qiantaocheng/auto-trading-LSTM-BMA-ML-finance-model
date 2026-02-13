#!/usr/bin/env python3
"""
HMM Risk-Gate Comparison on Walk-Forward Predictions
=====================================================
Replicates the exact TraderApp HMM logic (3-state GaussianHMM on SPY,
EMA(4) p_crisis, hysteresis) and compares NON-OVERLAP T+10 returns:
  A) No HMM  — always hold top-10 for 10 trading days
  B) With HMM — skip trade when crisis_mode=True

Uses raw OHLCV for:
  1. SPY data → HMM features (log_ret + 10d vol)
  2. Per-ticker T+10 forward returns (Close[t+10] / Close[t] - 1)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

# ── HMM params (mirror hmm_bridge.py exactly) ───────────────────
N_STATES = 3
HMM_WINDOW = 1000
EMA_SPAN = 4
GAMMA = 2
RISK_GATE_MIN = 0.05

CRISIS_ENTER_THRESH = 0.70
CRISIS_EXIT_THRESH = 0.40
CRISIS_CONFIRM_DAYS = 2
SAFE_CONFIRM_DAYS = 2
COOLDOWN_DAYS = 3


# ── HMM engine ──────────────────────────────────────────────────
def compute_spy_features(spy_close: pd.Series) -> pd.DataFrame:
    """log return + 10d rolling vol (backward-looking only)."""
    logret = np.log(spy_close / spy_close.shift(1))
    vol10 = logret.rolling(10, min_periods=10).std()
    feats = pd.DataFrame({"mkt_logret_1d": logret, "mkt_vol_10d": vol10},
                         index=spy_close.index).dropna()
    return feats


def train_hmm_at_date(spy_features: pd.DataFrame, as_of: pd.Timestamp):
    """Train 3-state HMM on SPY features up to as_of (inclusive).
    Returns (model, crisis_state_idx, scaler) or None if insufficient data.
    """
    feats = spy_features.loc[:as_of]
    if len(feats) < 200:
        return None
    if len(feats) > HMM_WINDOW:
        feats = feats.iloc[-HMM_WINDOW:]

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    model = GaussianHMM(n_components=N_STATES, covariance_type="full",
                        n_iter=200, random_state=42, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    # label states by vol (ascending: SAFE, MID, CRISIS)
    means_orig = scaler.inverse_transform(model.means_)
    order = np.argsort(means_orig[:, 1])
    crisis_idx = int(order[2])
    return model, crisis_idx, scaler


def get_p_crisis(model, crisis_idx, scaler, spy_features, as_of):
    """Posterior p(crisis) for the latest day."""
    feats = spy_features.loc[:as_of]
    if len(feats) > HMM_WINDOW:
        feats = feats.iloc[-HMM_WINDOW:]
    X = scaler.transform(feats.values)
    post = model.predict_proba(X)
    return float(post[-1, crisis_idx])


class HysteresisState:
    """Replicates hmm_bridge.py hysteresis exactly."""
    def __init__(self):
        self.crisis_mode = False
        self.crisis_confirm = 0
        self.safe_confirm = 0
        self.cooldown = 0
        self.p_history: list = []

    def update(self, p_crisis_raw: float) -> float:
        """Feed raw p_crisis, return smoothed p_crisis and update crisis_mode."""
        self.p_history.append(p_crisis_raw)
        if len(self.p_history) > 30:
            self.p_history = self.p_history[-30:]

        # EMA smoothing (span=4)
        s = pd.Series(self.p_history)
        p_smooth = float(s.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])

        # hysteresis
        if self.cooldown > 0:
            self.cooldown -= 1

        if not self.crisis_mode:
            if p_smooth >= CRISIS_ENTER_THRESH and self.cooldown == 0:
                self.crisis_confirm += 1
                if self.crisis_confirm >= CRISIS_CONFIRM_DAYS:
                    self.crisis_mode = True
                    self.crisis_confirm = 0
                    self.safe_confirm = 0
            else:
                self.crisis_confirm = 0
        else:
            if p_smooth <= CRISIS_EXIT_THRESH:
                self.safe_confirm += 1
                if self.safe_confirm >= SAFE_CONFIRM_DAYS:
                    self.crisis_mode = False
                    self.safe_confirm = 0
                    self.crisis_confirm = 0
                    self.cooldown = COOLDOWN_DAYS
            else:
                self.safe_confirm = 0

        return p_smooth

    @property
    def risk_gate(self) -> float:
        if not self.p_history:
            return 1.0
        s = pd.Series(self.p_history)
        p_smooth = float(s.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])
        rg = (1.0 - p_smooth) ** GAMMA
        return 0.0 if rg < RISK_GATE_MIN else rg


# ── T+10 return computation ─────────────────────────────────────
def build_t10_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """For each (date, ticker), compute Close[t+10] / Close[t] - 1."""
    ohlcv = ohlcv.sort_values(["ticker", "date"])
    # per-ticker: shift close back by 10 trading days
    ohlcv["close_t10"] = ohlcv.groupby("ticker")["Close"].shift(-10)
    ohlcv["ret_t10"] = ohlcv["close_t10"] / ohlcv["Close"] - 1
    out = ohlcv.dropna(subset=["ret_t10"])[["date", "ticker", "ret_t10"]]
    return out.set_index(["date", "ticker"]).sort_index()


# ── analysis ─────────────────────────────────────────────────────
def max_drawdown(rets):
    cum = np.cumprod(1 + rets)
    return float((cum / np.maximum.accumulate(cum) - 1).min()) if len(rets) else 0.0


def stats_block(rets, freq_days, label):
    if len(rets) == 0:
        return {}
    m = float(np.mean(rets))
    med = float(np.median(rets))
    wr = float(np.mean(rets > 0))
    s = float(np.std(rets, ddof=1)) if len(rets) > 1 else np.nan
    sh = m / s * np.sqrt(252.0 / freq_days) if s > 0 else np.nan
    dd = max_drawdown(rets)
    return {
        f"{label}_mean": m, f"{label}_median": med, f"{label}_winrate": wr,
        f"{label}_sharpe": sh, f"{label}_maxdd": dd, f"{label}_n": len(rets),
    }


def main():
    out_dir = Path("results/walkforward_lambdarank")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────
    print("Loading walk-forward predictions …")
    pred_df = pd.read_parquet(out_dir / "walk_forward_predictions.parquet")

    print("Loading raw OHLCV …")
    ohlcv = pd.read_parquet("data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")

    # ── SPY features for HMM (download via yfinance — not in stock OHLCV) ─
    print("Downloading SPY data via yfinance …")
    import yfinance as yf
    spy_raw = yf.download("SPY", start="2019-01-01", end="2026-02-01",
                          progress=False, auto_adjust=True)
    spy_raw.index = spy_raw.index.tz_localize(None)
    spy_close = spy_raw["Close"].squeeze()
    spy_features = compute_spy_features(spy_close)
    print(f"  SPY features: {len(spy_features)} days "
          f"({spy_features.index[0].date()} → {spy_features.index[-1].date()})")

    # ── T+10 returns from OHLCV ──────────────────────────
    print("Computing T+10 forward returns from OHLCV …")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    t10_df = build_t10_returns(ohlcv)
    print(f"  T+10 returns: {len(t10_df)} rows")

    # ── normalize all date indices to midnight (pred has 05:00 offset) ──
    pred_df.index = pred_df.index.set_levels(
        pred_df.index.levels[0].normalize(), level="date")
    t10_df.index = t10_df.index.set_levels(
        t10_df.index.levels[0].normalize(), level="date")

    # ── non-overlap dates (every 10 trading days) ────────
    pred_dates = pred_df.index.get_level_values("date").unique().sort_values()
    non_overlap_dates = pred_dates[::10]  # every 10 trading days for T+10
    print(f"  Non-overlap dates (every 10d): {len(non_overlap_dates)}")

    # ── HMM at each rebalance date ───────────────────────
    # Retrain HMM every 63 trading days to save time, re-use between retrains
    print("\nRunning HMM on each rebalance date …")
    hmm_retrain_freq = 63
    hysteresis = HysteresisState()

    # We need daily HMM updates for proper hysteresis state tracking.
    # Run HMM on ALL trading days, but only record results at non-overlap dates.
    all_trading_days = spy_features.index.sort_values()
    # Only process days within our prediction date range
    start_pred = pred_dates[0]
    # Start HMM warmup 30 days before first prediction
    warmup_start_idx = max(0, all_trading_days.get_indexer([start_pred], method="ffill")[0] - 30)
    hmm_days = all_trading_days[warmup_start_idx:]

    current_model = None
    last_retrain_idx = -999
    crisis_log = []

    for i, day in enumerate(hmm_days):
        # Retrain HMM periodically
        if current_model is None or (i - last_retrain_idx) >= hmm_retrain_freq:
            result = train_hmm_at_date(spy_features, day)
            if result is not None:
                current_model, crisis_idx, scaler = result
                last_retrain_idx = i

        if current_model is None:
            continue

        p_raw = get_p_crisis(current_model, crisis_idx, scaler, spy_features, day)
        p_smooth = hysteresis.update(p_raw)

        crisis_log.append({
            "date": day,
            "p_crisis": p_raw,
            "p_smooth": p_smooth,
            "crisis_mode": hysteresis.crisis_mode,
            "risk_gate": hysteresis.risk_gate,
        })

    crisis_df = pd.DataFrame(crisis_log).set_index("date")
    n_crisis = crisis_df["crisis_mode"].sum()
    print(f"  HMM processed {len(crisis_df)} days, crisis_mode days: {n_crisis} "
          f"({n_crisis/len(crisis_df)*100:.1f}%)")

    # ── evaluate non-overlap T+10 ────────────────────────
    print("\nEvaluating non-overlap T+10 returns …\n")

    no_hmm_rets = []
    with_hmm_rets = []
    skipped_dates = []
    detail_rows = []

    for d in non_overlap_dates:
        # get model predictions for this date
        try:
            day_preds = pred_df.loc[d]
        except KeyError:
            continue
        if len(day_preds) < 20:
            continue

        # pick top 10 by prediction score
        order = np.argsort(-day_preds["pred"].values)
        top10_tickers = day_preds.index.get_level_values("ticker").values[order[:10]]

        # get T+10 returns for these tickers
        t10_rets = []
        for tk in top10_tickers:
            try:
                ret = t10_df.loc[(d, tk), "ret_t10"]
                if np.isfinite(ret):
                    t10_rets.append(float(ret))
            except KeyError:
                pass

        if len(t10_rets) < 5:  # need at least 5 stocks
            continue

        portfolio_ret = float(np.mean(t10_rets))

        # NO HMM: always take the trade
        no_hmm_rets.append(portfolio_ret)

        # WITH HMM: check crisis_mode
        try:
            crisis_row = crisis_df.loc[d]
            in_crisis = bool(crisis_row["crisis_mode"])
            rg = float(crisis_row["risk_gate"])
            p_s = float(crisis_row["p_smooth"])
        except KeyError:
            in_crisis = False
            rg = 1.0
            p_s = 0.0

        if not in_crisis:
            with_hmm_rets.append(portfolio_ret)
        else:
            with_hmm_rets.append(0.0)  # in cash during crisis
            skipped_dates.append(d)

        detail_rows.append({
            "date": str(d.date()),
            "top10_ret_t10": portfolio_ret,
            "crisis_mode": in_crisis,
            "risk_gate": rg,
            "p_smooth": p_s,
            "action": "SKIP (cash)" if in_crisis else "TRADE",
        })

    no_hmm_rets = np.array(no_hmm_rets)
    with_hmm_rets = np.array(with_hmm_rets)

    # ── results ──────────────────────────────────────────
    print("=" * 70)
    print("NON-OVERLAP T+10 COMPARISON: NO HMM vs WITH HMM")
    print("=" * 70)

    for label, rets in [("NO_HMM", no_hmm_rets), ("WITH_HMM", with_hmm_rets)]:
        s = stats_block(rets, 10, label)
        print(f"\n  {label} (n={s.get(f'{label}_n', 0)})")
        print(f"    Mean:     {s.get(f'{label}_mean', 0):.6f}  ({s.get(f'{label}_mean', 0)*100:.3f}%)")
        print(f"    Median:   {s.get(f'{label}_median', 0):.6f}  ({s.get(f'{label}_median', 0)*100:.3f}%)")
        print(f"    WinRate:  {s.get(f'{label}_winrate', 0):.4f}  ({s.get(f'{label}_winrate', 0)*100:.1f}%)")
        print(f"    Sharpe:   {s.get(f'{label}_sharpe', 0):.4f}")
        print(f"    MaxDD:    {s.get(f'{label}_maxdd', 0):.4f}  ({s.get(f'{label}_maxdd', 0)*100:.2f}%)")

    # improvement
    s_no = stats_block(no_hmm_rets, 10, "no")
    s_hm = stats_block(with_hmm_rets, 10, "hm")
    print(f"\n  IMPROVEMENT (HMM vs No HMM)")
    print(f"    Sharpe: {s_no.get('no_sharpe',0):.4f} → {s_hm.get('hm_sharpe',0):.4f}  "
          f"(Δ = {s_hm.get('hm_sharpe',0) - s_no.get('no_sharpe',0):+.4f})")
    print(f"    MaxDD:  {s_no.get('no_maxdd',0):.4f} → {s_hm.get('hm_maxdd',0):.4f}  "
          f"(Δ = {s_hm.get('hm_maxdd',0) - s_no.get('no_maxdd',0):+.4f})")
    print(f"    Mean:   {s_no.get('no_mean',0)*100:.3f}% → {s_hm.get('hm_mean',0)*100:.3f}%  "
          f"(Δ = {(s_hm.get('hm_mean',0) - s_no.get('no_mean',0))*100:+.3f}%)")
    print(f"    Trades skipped by HMM: {len(skipped_dates)} / {len(no_hmm_rets)}")

    # ── crisis period analysis ───────────────────────────
    if skipped_dates:
        print(f"\n  CRISIS PERIODS (trades skipped by HMM):")
        for d in skipped_dates:
            row = [r for r in detail_rows if r["date"] == str(d.date())]
            if row:
                r = row[0]
                ret = r["top10_ret_t10"]
                print(f"    {r['date']}  ret={ret*100:+.2f}%  p_smooth={r['p_smooth']:.3f}")
        # average return during crisis: what we AVOIDED
        avoided = [r["top10_ret_t10"] for r in detail_rows if r["action"] == "SKIP (cash)"]
        if avoided:
            avg_avoided = np.mean(avoided)
            print(f"    Avg return avoided: {avg_avoided*100:+.3f}%")
            print(f"    → {'GOOD: avoided losses' if avg_avoided < 0 else 'BAD: missed gains'}")

    # ── statistical test: paired difference ──────────────
    print(f"\n  STATISTICAL SIGNIFICANCE")
    diff = with_hmm_rets - no_hmm_rets
    if np.any(diff != 0):
        t, p = sp_stats.ttest_1samp(diff, 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    Paired t-test (HMM - NoHMM): t={t:.3f}  p={p:.4f}  {sig}")
    else:
        print(f"    No difference (HMM never triggered crisis)")

    # ── cumulative return comparison ─────────────────────
    cum_no = np.cumprod(1 + no_hmm_rets)
    cum_hm = np.cumprod(1 + with_hmm_rets)
    print(f"\n  CUMULATIVE RETURN")
    print(f"    No HMM:   {(cum_no[-1] - 1)*100:.2f}%")
    print(f"    With HMM: {(cum_hm[-1] - 1)*100:.2f}%")

    # ── save detail ──────────────────────────────────────
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(out_dir / "hmm_comparison_detail.csv", index=False)
    crisis_df.to_csv(out_dir / "hmm_crisis_log.csv")

    summary = {
        "no_hmm": stats_block(no_hmm_rets, 10, "no_hmm"),
        "with_hmm": stats_block(with_hmm_rets, 10, "with_hmm"),
        "trades_skipped": len(skipped_dates),
        "total_trades": len(no_hmm_rets),
        "cum_return_no_hmm": float(cum_no[-1] - 1),
        "cum_return_with_hmm": float(cum_hm[-1] - 1),
    }
    with open(out_dir / "hmm_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
