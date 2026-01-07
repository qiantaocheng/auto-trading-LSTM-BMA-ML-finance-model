from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.backtest import CPCVConfig, cpcv_splits
from hetrs_nasdaq.meta_model import (
    MetaLabelModel,
    MetaModelConfig,
    build_meta_features,
    build_meta_labels,
    primary_signal_from_p50,
)
from hetrs_nasdaq.repro import set_global_seed


def future_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    return close.astype(float).pct_change(horizon).shift(-horizon).rename(f"future_ret_{horizon}d")


def dynamic_cost_rate(vol_gk: pd.Series) -> pd.Series:
    """
    Dynamic transaction cost in decimal return terms per unit turnover:
      basis_points = 5 + 0.1 * vol_gk
      cost_rate = basis_points / 10000
    """
    bp = 5.0 + 0.1 * vol_gk.astype(float)
    return (bp / 10000.0).rename("cost_rate")


def simulate_portfolio_dynamic_cost(
    df: pd.DataFrame,
    position: pd.Series,
) -> pd.DataFrame:
    """
    Next-day PnL with dynamic costs:
      r_port[t] = pos[t-1] * r_mkt[t] - cost_rate[t] * |pos[t] - pos[t-1]|
    """
    out = pd.DataFrame(index=df.index)
    pos = position.astype(float).clip(-1.0, 1.0).reindex(df.index).fillna(0.0)
    prev = pos.shift(1).fillna(0.0)

    r_mkt = df["returns"].astype(float)
    c = dynamic_cost_rate(df["vol_gk"]).reindex(df.index).fillna(0.0005)
    turnover = (pos - prev).abs()

    out["position"] = pos
    out["r_mkt"] = r_mkt
    out["cost_rate"] = c
    out["turnover"] = turnover
    out["trade_cost"] = c * turnover
    out["r_port"] = prev * r_mkt - out["trade_cost"]
    out["equity"] = (1.0 + out["r_port"].fillna(0.0)).cumprod()
    out["equity_benchmark"] = (1.0 + out["r_mkt"].fillna(0.0)).cumprod()
    return out


def sharpe(r: pd.Series, periods: int = 252) -> float:
    x = r.dropna().astype(float).values
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    x = equity.astype(float)
    peak = x.cummax()
    dd = (x / peak) - 1.0
    return float(dd.min())


def probabilistic_sharpe_ratio(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado).
    PSR = Phi( ((SR - SR*) * sqrt(N-1)) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR^2) )
    """
    x = r.dropna().astype(float).values
    n = x.size
    if n < 30:
        return float("nan")
    sr = sharpe(pd.Series(x), periods=252)
    # sample skewness / kurtosis (Pearson)
    m = x.mean()
    s = x.std(ddof=1)
    if s <= 1e-12:
        return float("nan")
    z = (x - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    if denom <= 1e-12:
        return float("nan")
    stat = ((sr - sr_benchmark) * np.sqrt(n - 1.0)) / np.sqrt(denom)
    try:
        from scipy.stats import norm

        return float(norm.cdf(stat))
    except Exception:
        # fallback approximation
        return float(0.5 * (1.0 + np.math.erf(stat / np.sqrt(2.0))))


@dataclass(frozen=True)
class V31Config:
    # Updated to best cumulative-return config from the latest sweep:
    # primary_threshold=0.001, meta_prob_threshold=0.55
    primary_threshold: float = 0.001
    meta_prob_threshold: float = 0.55
    seed: int = 42


def run_cpcv_meta_backtest(
    df: pd.DataFrame,
    outdir: str,
    cfg: CPCVConfig = CPCVConfig(),
    v31: V31Config = V31Config(),
) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    set_global_seed(v31.seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Need TFT preds and features
    required = {"tft_p10", "tft_p50", "tft_p90", "vol_gk", "returns", "Close", "macd"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns for v3.1 backtest: {missing}")

    df = df.sort_index().copy()
    df["future_ret_5d"] = future_return(df["Close"], horizon=5)
    df["primary_signal"] = primary_signal_from_p50(df["tft_p50"], threshold=v31.primary_threshold)

    n = len(df)

    # Store OOS positions for naive and meta strategies
    pos_naive_sum = pd.Series(0.0, index=df.index)
    pos_naive_cnt = pd.Series(0.0, index=df.index)
    pos_meta_sum = pd.Series(0.0, index=df.index)
    pos_meta_cnt = pd.Series(0.0, index=df.index)

    learned = []

    split_id = 0
    for train_idx, test_idx in cpcv_splits(n, cfg):
        split_id += 1
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        # Naive positions: directly trade primary signal
        pos_naive = train["primary_signal"].astype(float)  # train-side for fit completeness
        pos_naive_test = test["primary_signal"].astype(float)

        # Meta labeling training data: only where primary signal != 0
        sig_train = train["primary_signal"]
        mask_trade = sig_train != 0
        X_train = build_meta_features(train.loc[mask_trade])
        y_train = build_meta_labels(train.loc[mask_trade], sig_train.loc[mask_trade], train["future_ret_5d"].loc[mask_trade])

        # Drop any NaNs in meta features/labels
        m = (~X_train.isna().any(axis=1)) & (~y_train.isna())
        X_train = X_train.loc[m]
        y_train = y_train.loc[m]

        model = None
        if len(X_train) >= 100 and y_train.nunique() > 1:
            mm = MetaLabelModel(MetaModelConfig(random_state=v31.seed))
            mm.fit(X_train, y_train)
            model = mm

        # Inference: prob_success on test, only for days where primary signal != 0
        pos_meta_test = pd.Series(0.0, index=test.index, name="pos_meta")
        if model is not None:
            sig_test = test["primary_signal"]
            mask_trade_t = sig_test != 0
            X_test = build_meta_features(test.loc[mask_trade_t])
            m2 = ~X_test.isna().any(axis=1)
            X_test = X_test.loc[m2]
            if len(X_test) > 0:
                prob = model.predict_proba(X_test)
                keep = prob > float(v31.meta_prob_threshold)
                chosen_idx = X_test.index[keep]
                pos_meta_test.loc[chosen_idx] = sig_test.loc[chosen_idx].astype(float)

                learned.append(
                    {
                        "split": int(split_id),
                        "train_rows": int(len(train)),
                        "train_trade_rows": int(mask_trade.sum()),
                        "meta_used": True,
                        "meta_prob_threshold": float(v31.meta_prob_threshold),
                        "test_trade_rows": int(mask_trade_t.sum()),
                        "test_kept_trades": int(len(chosen_idx)),
                        "keep_rate": float(len(chosen_idx) / max(1, mask_trade_t.sum())),
                    }
                )
            else:
                learned.append({"split": int(split_id), "meta_used": True, "test_kept_trades": 0})
        else:
            learned.append({"split": int(split_id), "meta_used": False})

        pos_naive_sum.loc[test.index] += pos_naive_test
        pos_naive_cnt.loc[test.index] += 1.0
        pos_meta_sum.loc[test.index] += pos_meta_test
        pos_meta_cnt.loc[test.index] += 1.0

    pos_naive_oos = (pos_naive_sum / pos_naive_cnt.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
    pos_meta_oos = (pos_meta_sum / pos_meta_cnt.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)

    sim_naive = simulate_portfolio_dynamic_cost(df, pos_naive_oos)
    sim_meta = simulate_portfolio_dynamic_cost(df, pos_meta_oos)
    sim_bh = pd.DataFrame(index=df.index)
    sim_bh["equity"] = (1.0 + df["returns"].fillna(0.0)).cumprod()

    metrics = {
        "splits_used": int(split_id),
        "primary_threshold": float(v31.primary_threshold),
        "meta_prob_threshold": float(v31.meta_prob_threshold),
        "buyhold_cum_return": float(sim_bh["equity"].iloc[-1] - 1.0),
        "buyhold_sharpe": float(sharpe(df["returns"])),
        "buyhold_max_dd": float(max_drawdown(sim_bh["equity"])),
        "naive_cum_return": float(sim_naive["equity"].iloc[-1] - 1.0),
        "naive_sharpe": float(sharpe(sim_naive["r_port"])),
        "naive_psr": float(probabilistic_sharpe_ratio(sim_naive["r_port"])),
        "naive_max_dd": float(max_drawdown(sim_naive["equity"])),
        "meta_cum_return": float(sim_meta["equity"].iloc[-1] - 1.0),
        "meta_sharpe": float(sharpe(sim_meta["r_port"])),
        "meta_psr": float(probabilistic_sharpe_ratio(sim_meta["r_port"])),
        "meta_max_dd": float(max_drawdown(sim_meta["equity"])),
    }

    # Save artifacts
    sim_naive.assign(position=pos_naive_oos).to_csv(Path(outdir) / "naive_timeseries.csv")
    sim_meta.assign(position=pos_meta_oos).to_csv(Path(outdir) / "meta_timeseries.csv")
    pd.DataFrame(learned).to_csv(Path(outdir) / "meta_folds.csv", index=False)
    (Path(outdir) / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Plot tear sheet
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sim_bh.index, sim_bh["equity"], label="Buy&Hold QQQ", alpha=0.85)
    ax.plot(sim_naive.index, sim_naive["equity"], label="Naive TFT (p50 threshold)", alpha=0.9)
    ax.plot(sim_meta.index, sim_meta["equity"], label="Meta-TFT (RF filter)", alpha=0.95)
    ax.set_title("Tear-sheet: Buy&Hold vs Naive TFT vs Meta-TFT (CPCV OOS blended)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "tearsheet.png", dpi=150)
    plt.close(fig)

    return metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-labeling + explainable backtest (HETRS-NASDAQ v3.1).")
    p.add_argument("--in", dest="inp", required=True, help="Input parquet (features + tft preds)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--k-test-groups", type=int, default=2)
    p.add_argument("--purge", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--primary-threshold", type=float, default=0.002)
    p.add_argument("--meta-prob-threshold", type=float, default=0.6)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp)
    cfg = CPCVConfig(
        n_groups=int(args.n_groups),
        k_test_groups=int(args.k_test_groups),
        purge=int(args.purge),
        embargo=float(args.embargo),
    )
    v31 = V31Config(
        primary_threshold=float(args.primary_threshold),
        meta_prob_threshold=float(args.meta_prob_threshold),
        seed=int(args.seed),
    )
    metrics = run_cpcv_meta_backtest(df, outdir=args.outdir, cfg=cfg, v31=v31)
    print(
        f"[backtest_v2] done. meta_sharpe={metrics['meta_sharpe']:.3f} "
        f"naive_sharpe={metrics['naive_sharpe']:.3f} buyhold_sharpe={metrics['buyhold_sharpe']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


