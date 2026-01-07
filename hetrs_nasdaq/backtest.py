from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import json

from hetrs_nasdaq.repro import set_global_seed

def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().astype(float).values
    if r.size < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    x = equity.astype(float)
    peak = x.cummax()
    dd = (x / peak) - 1.0
    return float(dd.min())


def _ensure_action_baseline(df: pd.DataFrame) -> pd.Series:
    """
    If no RL predictions are available, produce a deterministic baseline action.
    Preference order:
      - sign(tft_p50)
      - sign(rolling mean of returns)
    """
    if "tft_p50" in df.columns:
        return np.sign(df["tft_p50"].astype(float)).rename("action")
    return np.sign(df["returns"].rolling(5).mean().fillna(0.0)).rename("action")


def threshold_hysteresis_action(
    signal: pd.Series,
    buy_th: float,
    sell_th: float,
    allow_short: bool = False,
    short_buy_th: Optional[float] = None,
    short_sell_th: Optional[float] = None,
) -> pd.Series:
    """
    Convert a predictive signal into a position series using hysteresis thresholds.

    Long/flat (default):
    - enter long when signal > buy_th
    - exit to flat when signal < sell_th
    - otherwise keep previous position

    Optional symmetric shorting:
    - enter short when signal < -short_buy_th
    - exit short to flat when signal > -short_sell_th
    """
    sig = signal.astype(float)
    idx = sig.index
    pos = np.zeros(len(sig), dtype=float)

    buy_th = float(buy_th)
    sell_th = float(sell_th)
    if sell_th >= buy_th:
        raise ValueError("sell_th must be < buy_th for hysteresis.")

    if allow_short:
        sb = float(short_buy_th) if short_buy_th is not None else buy_th
        ss = float(short_sell_th) if short_sell_th is not None else sell_th
        if ss >= sb:
            raise ValueError("short_sell_th must be < short_buy_th for hysteresis.")

    cur = 0.0
    for i, x in enumerate(sig.values):
        if not np.isfinite(x):
            pos[i] = cur
            continue

        # Exit rules first
        if cur > 0 and x < sell_th:
            cur = 0.0
        if allow_short and cur < 0:
            sb = float(short_buy_th) if short_buy_th is not None else buy_th
            ss = float(short_sell_th) if short_sell_th is not None else sell_th
            if x > -ss:
                cur = 0.0

        # Entry rules
        if cur == 0.0:
            if x > buy_th:
                cur = 1.0
            elif allow_short:
                sb = float(short_buy_th) if short_buy_th is not None else buy_th
                if x < -sb:
                    cur = -1.0

        pos[i] = cur

    return pd.Series(pos, index=idx, name="action")


def _sharpe_from_sim(sim: pd.DataFrame) -> float:
    return _annualized_sharpe(sim["r_port"])


def _cum_return_from_sim(sim: pd.DataFrame) -> float:
    eq = sim["equity"].astype(float)
    if eq.empty:
        return 0.0
    return float(eq.iloc[-1] - 1.0)


def _max_dd_from_sim(sim: pd.DataFrame) -> float:
    return float(_max_drawdown(sim["equity"]))


def score_simulation(sim: pd.DataFrame, objective: str) -> float:
    """
    Higher-is-better score.
    - sharpe: maximize annualized Sharpe
    - return: maximize cumulative return
    - drawdown: maximize max_drawdown (less negative is better)
    """
    objective = str(objective).lower()
    if objective == "sharpe":
        return _sharpe_from_sim(sim)
    if objective == "return":
        return _cum_return_from_sim(sim)
    if objective == "drawdown":
        return _max_dd_from_sim(sim)
    raise ValueError(f"Unknown objective: {objective!r}")


def learn_thresholds_grid(
    df_train: pd.DataFrame,
    signal_col: str = "tft_p50",
    cost_rate: float = 0.0005,
    allow_short: bool = False,
    objective: str = "sharpe",
    buy_quantiles: tuple[float, ...] = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    sell_quantiles: tuple[float, ...] = (0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1),
) -> dict:
    """
    Deterministically learn thresholds by grid search over signal quantiles.
    Optimizes training Sharpe (includes transaction costs).
    """
    if signal_col not in df_train.columns:
        raise ValueError(f"signal_col {signal_col!r} not found in training data.")

    sig = df_train[signal_col].astype(float).dropna()
    if sig.empty:
        raise ValueError("Training signal is empty after dropna.")

    candidates = []
    for bq in buy_quantiles:
        for sq in sell_quantiles:
            if sq >= bq:
                continue
            buy_th = float(sig.quantile(bq))
            sell_th = float(sig.quantile(sq))
            if not np.isfinite(buy_th) or not np.isfinite(sell_th):
                continue
            if sell_th >= buy_th:
                continue
            candidates.append((buy_th, sell_th))

    if not candidates:
        raise ValueError("No valid threshold candidates generated.")

    best = {"buy_th": None, "sell_th": None, "train_score": -np.inf}
    for buy_th, sell_th in candidates:
        action = threshold_hysteresis_action(
            df_train[signal_col],
            buy_th=buy_th,
            sell_th=sell_th,
            allow_short=allow_short,
        )
        sim = simulate_portfolio(df_train, action, cost_rate=cost_rate)
        s = score_simulation(sim, objective=objective)
        if s > best["train_score"]:
            best = {"buy_th": float(buy_th), "sell_th": float(sell_th), "train_score": float(s)}

    return best

def simulate_portfolio(
    df: pd.DataFrame,
    action: pd.Series,
    cost_rate: float = 0.0005,
) -> pd.DataFrame:
    """
    Simulate next-day PnL: r_port[t+1] = pos[t] * r_mkt[t+1] - cost*|pos[t]-pos[t-1]|
    Assumes action is desired position at time t (close).
    """
    out = pd.DataFrame(index=df.index)
    pos = action.astype(float).clip(-1.0, 1.0).reindex(df.index).fillna(0.0)
    pos_prev = pos.shift(1).fillna(0.0)

    r_mkt = df["returns"].astype(float)
    trade_cost = cost_rate * (pos - pos_prev).abs()
    r_port = pos_prev * r_mkt - trade_cost

    out["position"] = pos
    out["trade_cost"] = trade_cost
    out["r_mkt"] = r_mkt
    out["r_port"] = r_port
    out["equity"] = (1.0 + out["r_port"].fillna(0.0)).cumprod()
    out["equity_benchmark"] = (1.0 + out["r_mkt"].fillna(0.0)).cumprod()
    return out


def triple_barrier_stats(
    df: pd.DataFrame,
    horizon: int = 5,
    barrier_mult: float = 1.5,
) -> dict:
    """
    Triple barrier evaluation on the underlying market path (not strategy-specific).
    Uses vol_gk at entry to set +/- barriers:
      pt = barrier_mult * vol_gk * sqrt(horizon)
      sl = barrier_mult * vol_gk * sqrt(horizon)
    """
    close = df["Close"].astype(float)
    vol = df["vol_gk"].astype(float)

    wins = 0
    losses = 0
    timeouts = 0

    idx = close.index
    for i in range(0, len(df) - horizon - 1):
        c0 = float(close.iloc[i])
        if not np.isfinite(c0):
            continue
        b = float(barrier_mult * vol.iloc[i] * np.sqrt(horizon))
        if not np.isfinite(b) or b <= 0:
            continue

        pt = b
        sl = -b

        touched = None
        for j in range(1, horizon + 1):
            cj = float(close.iloc[i + j])
            ret = (cj / c0) - 1.0
            if ret >= pt:
                touched = "pt"
                break
            if ret <= sl:
                touched = "sl"
                break

        if touched == "pt":
            wins += 1
        elif touched == "sl":
            losses += 1
        else:
            timeouts += 1

    total = wins + losses + timeouts
    return {
        "tb_total": int(total),
        "tb_win": int(wins),
        "tb_loss": int(losses),
        "tb_timeout": int(timeouts),
        "tb_win_rate": float(wins / total) if total else 0.0,
    }


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    k_test_groups: int = 2
    purge: int = 5
    embargo: float = 5  # if < 1.0 treated as fraction of dataset length


def _group_slices(n: int, n_groups: int) -> list[slice]:
    """
    Split [0..n) into contiguous slices.
    """
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def cpcv_splits(n: int, cfg: CPCVConfig) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Generate CPCV train/test indices.
    - Data is divided into cfg.n_groups contiguous groups
    - Choose cfg.k_test_groups as test; remaining as train
    - Apply purge around each test block and embargo after test block
    """
    slices = _group_slices(n, cfg.n_groups)
    groups = list(range(cfg.n_groups))

    for test_groups in itertools.combinations(groups, cfg.k_test_groups):
        test_mask = np.zeros(n, dtype=bool)
        for g in test_groups:
            test_mask[slices[g]] = True

        # Apply purge/embargo by expanding the excluded region around test blocks
        excluded = test_mask.copy()
        test_idx = np.where(test_mask)[0]
        if test_idx.size:
            # Purge before and after
            for t in test_idx:
                lo = max(0, t - cfg.purge)
                hi = min(n, t + cfg.purge + 1)
                excluded[lo:hi] = True

            # Embargo after the end of each contiguous test segment
            embargo_n = int(cfg.embargo * n) if float(cfg.embargo) < 1.0 else int(cfg.embargo)
            diffs = np.diff(test_idx)
            segment_ends = [test_idx[i] for i in range(len(diffs)) if diffs[i] > 1] + [test_idx[-1]]
            for end in segment_ends:
                lo = end + 1
                hi = min(n, end + 1 + embargo_n)
                excluded[lo:hi] = True

        train_idx = np.where(~excluded)[0]
        test_idx = np.where(test_mask)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        yield train_idx, test_idx


def backtest_cpcv(
    df: pd.DataFrame,
    rl_dir: Optional[str],
    outdir: str,
    cfg: CPCVConfig = CPCVConfig(),
    cost_rate: float = 0.0005,
    seed: int = 42,
    policy: str = "baseline",
    signal_col: str = "tft_p50",
    allow_short: bool = False,
    threshold_objective: str = "sharpe",
) -> dict:
    from matplotlib import pyplot as plt

    set_global_seed(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Optional RL ensemble
    ensemble = None
    if rl_dir is not None:
        try:
            from hetrs_nasdaq.rl_agent import load_ensemble

            ensemble = load_ensemble(df, rl_dir=rl_dir, cost_rate=cost_rate, seed=seed)
        except Exception:
            ensemble = None

    n = len(df)
    oos_action_sum = pd.Series(0.0, index=df.index)
    oos_action_cnt = pd.Series(0.0, index=df.index)
    learned_thresholds = []

    split_count = 0
    for train_idx, test_idx in cpcv_splits(n, cfg):
        split_count += 1
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        if policy == "rl" and ensemble is not None:
            # Roll through test df to generate actions from ensemble
            obs_cols = ["tft_p10", "tft_p50", "tft_p90", "regime_prob_0", "regime_prob_1", "regime_prob_2"]
            for c in obs_cols:
                if c not in test_df.columns:
                    if c.startswith("tft_"):
                        raise ValueError(
                            f"Missing {c}. Please generate TFT predictions (tft_p10/p50/p90) instead of using proxies."
                        )
                    else:
                        test_df[c] = 1.0 / 3.0

            acts = []
            for _, row in test_df.iterrows():
                obs = row[obs_cols].astype(float).values.astype(np.float32)
                acts.append(ensemble.predict_action(obs))
            action = pd.Series(acts, index=test_df.index, name="action")
        elif policy == "threshold":
            if signal_col not in train_df.columns or signal_col not in test_df.columns:
                raise ValueError(
                    f"policy=threshold requires signal_col={signal_col!r} in both train and test data."
                )
            th = learn_thresholds_grid(
                df_train=train_df,
                signal_col=signal_col,
                cost_rate=cost_rate,
                allow_short=allow_short,
                objective=threshold_objective,
            )
            learned_thresholds.append(
                {
                    "split": int(split_count),
                    "buy_th": th["buy_th"],
                    "sell_th": th["sell_th"],
                    "train_score": th["train_score"],
                    "objective": str(threshold_objective),
                }
            )
            action = threshold_hysteresis_action(
                test_df[signal_col],
                buy_th=float(th["buy_th"]),
                sell_th=float(th["sell_th"]),
                allow_short=allow_short,
            )
        else:
            # Deterministic baseline (no learned threshold, no RL)
            action = _ensure_action_baseline(test_df)

        oos_action_sum.loc[test_df.index] += action
        oos_action_cnt.loc[test_df.index] += 1.0

    oos_action = (oos_action_sum / oos_action_cnt.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
    sim = simulate_portfolio(df, oos_action, cost_rate=cost_rate)

    sharpe = _annualized_sharpe(sim["r_port"])
    mdd = _max_drawdown(sim["equity"])
    win_rate = float((sim["r_port"] > 0).mean())
    tb = triple_barrier_stats(df)

    metrics = {
        "splits_used": int(split_count),
        "policy": str(policy),
        "signal_col": str(signal_col),
        "allow_short": bool(allow_short),
        "threshold_objective": str(threshold_objective) if policy == "threshold" else None,
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "win_rate_daily": float(win_rate),
        "cum_return": float(sim["equity"].iloc[-1] - 1.0),
        "benchmark_sharpe": float(_annualized_sharpe(sim["r_mkt"])),
        "benchmark_max_drawdown": float(_max_drawdown(sim["equity_benchmark"])),
        "benchmark_cum_return": float(sim["equity_benchmark"].iloc[-1] - 1.0),
        **tb,
    }

    # Save timeseries
    sim_out = sim.copy()
    sim_out["action"] = oos_action
    sim_out.to_csv(Path(outdir) / "cpcv_timeseries.csv")

    if learned_thresholds:
        pd.DataFrame(learned_thresholds).to_csv(Path(outdir) / "learned_thresholds.csv", index=False)

    # Plot equity with risk regime shading
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sim_out.index, sim_out["equity"], label="Strategy equity")
    ax.plot(sim_out.index, sim_out["equity_benchmark"], label="QQQ buy&hold", alpha=0.8)

    if {"regime_prob_0", "regime_prob_1", "regime_prob_2"}.issubset(df.columns):
        # Mark "high risk" as the most probable regime having prob > 0.6
        probs = df[["regime_prob_0", "regime_prob_1", "regime_prob_2"]].astype(float)
        maxp = probs.max(axis=1)
        high_risk = maxp > 0.6
        # Shade contiguous regions
        in_block = False
        start = None
        for t, flag in high_risk.items():
            if flag and not in_block:
                in_block = True
                start = t
            elif (not flag) and in_block:
                ax.axvspan(start, t, color="gray", alpha=0.15)
                in_block = False
        if in_block and start is not None:
            ax.axvspan(start, high_risk.index[-1], color="gray", alpha=0.15)

    ax.set_title("CPCV OOS equity (gray = high-risk regime)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "cpcv_equity.png", dpi=150)
    plt.close(fig)

    (Path(outdir) / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPCV backtest (HETRS-NASDAQ).")
    p.add_argument("--in", dest="inp", required=True, help="Input features parquet")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--rl_dir", default=None, help="Directory with trained RL agents (optional)")
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--k-test-groups", type=int, default=2)
    p.add_argument("--purge", type=int, default=5)
    p.add_argument("--embargo", type=float, default=5, help="Days (int) or fraction (<1.0) of dataset length")
    p.add_argument("--cost-rate", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--policy",
        type=str,
        default="baseline",
        choices=["baseline", "threshold", "rl"],
        help="Trading policy. baseline=sign(signal), threshold=learn buy/sell thresholds on train folds, rl=SB3 ensemble",
    )
    p.add_argument("--signal-col", type=str, default="tft_p50", help="Signal column used by baseline/threshold policy")
    p.add_argument("--allow-short", action="store_true", help="Allow shorting (threshold policy uses symmetric rules)")
    p.add_argument(
        "--threshold-objective",
        type=str,
        default="sharpe",
        choices=["sharpe", "return", "drawdown"],
        help="Objective used to learn buy/sell thresholds on train folds.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp).sort_index()
    cfg = CPCVConfig(
        n_groups=int(args.n_groups),
        k_test_groups=int(args.k_test_groups),
        purge=int(args.purge),
        embargo=int(args.embargo),
    )
    metrics = backtest_cpcv(
        df,
        rl_dir=args.rl_dir,
        outdir=args.outdir,
        cfg=cfg,
        cost_rate=args.cost_rate,
        seed=args.seed,
        policy=args.policy,
        signal_col=args.signal_col,
        allow_short=bool(args.allow_short),
        threshold_objective=args.threshold_objective,
    )
    print(f"[backtest] done. sharpe={metrics['sharpe']:.3f} max_dd={metrics['max_drawdown']:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


