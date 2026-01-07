from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _set_mpl_agg() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def _annualized_sharpe(period_returns: pd.Series, periods_per_year: float) -> float:
    r = period_returns.dropna().astype(float).values
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


@dataclass(frozen=True)
class WalkforwardConfig:
    horizon_days: int = 10
    purge_days: int = 5
    train_window_periods: int = 252  # trading days in training window
    annual_retrain: bool = False
    min_train_years: int = 4
    test_years: int = 1
    ridge_alpha: float = 1.0
    top_ns: tuple[int, ...] = (10, 20, 30)
    bottom_ns: tuple[int, ...] = (10, 20)
    max_rebalances: Optional[int] = None
    seed: int = 42
    checkpoint_every: int = 25  # write partial results every N rebalances
    progress_every: int = 5  # log progress every N rebalances


def load_factor_data(path: str) -> pd.DataFrame:
    """
    Load factor parquet. Supports:
    - MultiIndex(date,ticker) parquet
    - Flat parquet with columns date,ticker
    """
    df = pd.read_parquet(path)
    # Normalize to a consistent MultiIndex(date,ticker) with date at midnight (no time component).
    if isinstance(df.index, pd.MultiIndex) and {"date", "ticker"}.issubset(df.index.names):
        out = df.reset_index()
    else:
        if not {"date", "ticker"}.issubset(df.columns):
            raise ValueError("Factor file must have MultiIndex(date,ticker) or columns date,ticker")
        out = df.copy()

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["ticker"] = out["ticker"].astype(str)
    out = out.set_index(["date", "ticker"]).sort_index()
    return out


def resolve_target_col(df: pd.DataFrame) -> str:
    # Prefer explicit ret_fwd_10d-like naming, else 'target'
    for c in ("ret_fwd_10d", "target", "ret_fwd", "y"):
        if c in df.columns:
            return c
    raise ValueError("Could not find target column (expected 'target' or 'ret_fwd_10d').")


def get_rebalance_dates(unique_dates: pd.DatetimeIndex, horizon_days: int) -> pd.DatetimeIndex:
    """
    Rebalance every `horizon_days` trading days.
    We must also ensure target exists (forward return), so avoid last horizon_days.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(unique_dates).unique()))
    if len(dates) <= horizon_days:
        return pd.DatetimeIndex([])
    return dates[: -horizon_days : horizon_days]


def get_train_end_idx(reb_i: int, horizon_days: int, purge_days: int) -> int:
    # Train must end at least horizon+purge trading days before the rebalance date.
    return reb_i - int(horizon_days) - int(purge_days)


def _audit_row(
    *,
    reb_date: pd.Timestamp,
    reb_i: int,
    train_end_i: int,
    train_max_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
    cfg: WalkforwardConfig,
    mode: str,
) -> dict:
    allowed_end_i = get_train_end_idx(reb_i, cfg.horizon_days, cfg.purge_days)
    allowed_end_i = max(int(allowed_end_i), 0)
    allowed_max_date = pd.Timestamp(dates[allowed_end_i]) if allowed_end_i < len(dates) else pd.Timestamp(dates[-1])
    ok = bool(train_end_i <= allowed_end_i)
    return {
        "mode": mode,
        "rebalance_date": reb_date,
        "rebalance_i": int(reb_i),
        "train_end_i": int(train_end_i),
        "train_max_date": train_max_date,
        "allowed_end_i": int(allowed_end_i),
        "allowed_max_date": allowed_max_date,
        "horizon_days": int(cfg.horizon_days),
        "purge_days": int(cfg.purge_days),
        "ok": ok,
    }


def get_training_slice(
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reb_date: pd.Timestamp,
    cfg: WalkforwardConfig,
) -> pd.DataFrame:
    reb_i = int(np.where(dates == reb_date)[0][0])
    train_end_i = get_train_end_idx(reb_i, cfg.horizon_days, cfg.purge_days)
    if train_end_i <= 0:
        return pd.DataFrame()

    start_i = max(0, train_end_i - cfg.train_window_periods + 1)
    train_dates = dates[start_i : train_end_i + 1]
    train = df.loc[(train_dates, slice(None)), :].copy()
    return train


def get_test_slice(df: pd.DataFrame, reb_date: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(reb_date, slice(None)), :].copy()


def fit_ridge(train: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float, seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = train[feature_cols].astype("float32")
    y = train[target_col].astype("float32")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=float(alpha), random_state=int(seed))),
        ]
    )
    model.fit(X.values, y.values)
    return model


def predict(model, test: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    X = test[feature_cols].astype("float32").values
    p = model.predict(X)
    return pd.Series(p, index=test.index, name="prediction")


def top_n_return(preds: pd.DataFrame, n: int, target_col: str) -> float:
    d = preds.sort_values("prediction", ascending=False).head(int(n))
    return float(d[target_col].mean()) if not d.empty else float("nan")


def bottom_n_return(preds: pd.DataFrame, n: int, target_col: str) -> float:
    d = preds.sort_values("prediction", ascending=True).head(int(n))
    return float(d[target_col].mean()) if not d.empty else float("nan")


def download_qqq(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf  # type: ignore

    q = yf.download(
        "QQQ",
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=7)).date()),
        progress=False,
        auto_adjust=False,
    )
    if q is None or q.empty:
        raise RuntimeError("Failed to download QQQ from yfinance.")
    if isinstance(q.columns, pd.MultiIndex):
        q.columns = q.columns.get_level_values(0)
    q.index = pd.to_datetime(q.index)
    q = q.sort_index()
    return q[["Close"]].copy()


def qqq_horizon_returns(qqq_close: pd.Series, horizon_days: int) -> pd.Series:
    # trading-days shift
    ret = qqq_close.shift(-horizon_days) / qqq_close - 1.0
    return ret.rename("qqq_ret")


def run_walkforward(
    df: pd.DataFrame,
    cfg: WalkforwardConfig,
    outdir: Optional[Path] = None,
) -> dict:
    target_col = resolve_target_col(df)
    df = df.copy()
    # Drop rows without target (cannot score)
    df = df.dropna(subset=[target_col])

    # Features = everything except target and Close (keep Close if needed externally)
    feature_cols = [c for c in df.columns if c not in {target_col, "Close"}]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    dates = pd.DatetimeIndex(df.index.get_level_values("date").unique()).sort_values()
    all_rebalances = get_rebalance_dates(dates, cfg.horizon_days)
    if cfg.max_rebalances is not None:
        all_rebalances = all_rebalances[: int(cfg.max_rebalances)]

    # Benchmark series
    qqq = download_qqq(all_rebalances.min(), all_rebalances.max())
    qqq_ret = qqq_horizon_returns(qqq["Close"], cfg.horizon_days)

    equity = {
        "top10": 1.0,
        "top20": 1.0,
        "top30": 1.0,
        "bottom10": 1.0,
        "bottom20": 1.0,
        "long_short10": 1.0,
        "qqq": 1.0,
    }
    rows = []
    picks = []
    trades = []
    audit = []

    prev_holdings_top10: dict[str, float] = {}  # ticker -> weight

    # Build rebalance blocks.
    # If annual_retrain:
    # - Train on first `min_train_years` calendar years
    # - Test on next `test_years` calendar years
    # - Expand train window by `test_years` and repeat (expanding window)
    if cfg.annual_retrain:
        years = sorted(pd.DatetimeIndex(dates).year.unique())
        if len(years) < (cfg.min_train_years + cfg.test_years):
            raise ValueError("Not enough years in data for annual retrain.")

        blocks: list[tuple[list[int], list[int], pd.DatetimeIndex]] = []
        train_years_count = int(cfg.min_train_years)
        while True:
            test_start_yi = train_years_count
            test_end_yi = test_start_yi + int(cfg.test_years) - 1
            if test_end_yi >= len(years):
                break
            train_yrs = years[0:train_years_count]
            test_yrs = years[test_start_yi : test_end_yi + 1]
            test_reb = all_rebalances[all_rebalances.year.isin(test_yrs)]
            blocks.append((train_yrs, test_yrs, test_reb))
            train_years_count += int(cfg.test_years)
        block_iter = blocks
    else:
        block_iter = [(None, None, all_rebalances)]

    total_rebalances = int(sum(len(x[2]) for x in block_iter))
    done = 0
    feature_cols = [c for c in df.columns if c not in {target_col, "Close"}]

    for train_yrs, test_yrs, rebalances in block_iter:
        model_for_block = None
        train_end_date: Optional[pd.Timestamp] = None
        train_end_i_for_block: Optional[int] = None

        if cfg.annual_retrain:
            if rebalances.empty:
                continue
            test_start = pd.Timestamp(rebalances.min())
            test_start_i = int(np.where(dates == test_start)[0][0])
            train_end_i = get_train_end_idx(test_start_i, cfg.horizon_days, cfg.purge_days)
            if train_end_i <= 0:
                continue
            train_end_date = pd.Timestamp(dates[train_end_i])
            train_end_i_for_block = int(train_end_i)
            train_slice = df.loc[(dates[0 : train_end_i + 1], slice(None)), :].copy()
            if train_slice.empty:
                continue
            model_for_block = fit_ridge(train_slice, feature_cols, target_col, alpha=cfg.ridge_alpha, seed=cfg.seed)

        for reb_date in rebalances:
            done += 1
            reb_date = pd.Timestamp(reb_date)
            reb_i = int(np.where(dates == reb_date)[0][0])

            test = get_test_slice(df, reb_date)
            if test.empty:
                continue

            if cfg.progress_every and (done % int(cfg.progress_every) == 0):
                if cfg.annual_retrain and train_end_date is not None:
                    tr_info = f"train_end={train_end_date.date()}"
                else:
                    tr_info = f"train_window={cfg.train_window_periods}"
                print(
                    f"[walkforward_top10] {done}/{total_rebalances} rebalance={reb_date.date()} {tr_info} "
                    f"test_rows={len(test):,}"
                )

            if cfg.annual_retrain:
                if model_for_block is None:
                    continue
                # Leakage audit: block training ends BEFORE the entire test year begins (strict).
                if train_end_i_for_block is None or train_end_date is None:
                    continue
                audit.append(
                    _audit_row(
                        reb_date=reb_date,
                        reb_i=reb_i,
                        train_end_i=train_end_i_for_block,
                        train_max_date=train_end_date,
                        dates=dates,
                        cfg=cfg,
                        mode="annual_retrain",
                    )
                )
                if not audit[-1]["ok"]:
                    raise AssertionError(f"Leakage detected (annual): {audit[-1]}")
                pred = predict(model_for_block, test, feature_cols)
            else:
                train = get_training_slice(df, dates, reb_date, cfg)
                if train.empty:
                    continue
                # Leakage safety: ensure max(train_date) < reb_date
                train_max = pd.Timestamp(train.index.get_level_values("date").max())
                if train_max >= reb_date:
                    continue
                # Leakage audit: ensure training ends at least (horizon+purge) trading days before rebalance.
                train_end_i = get_train_end_idx(reb_i, cfg.horizon_days, cfg.purge_days)
                if train_end_i <= 0:
                    continue
                audit.append(
                    _audit_row(
                        reb_date=reb_date,
                        reb_i=reb_i,
                        train_end_i=train_end_i,
                        train_max_date=train_max,
                        dates=dates,
                        cfg=cfg,
                        mode="rolling_window",
                    )
                )
                if not audit[-1]["ok"]:
                    raise AssertionError(f"Leakage detected (rolling): {audit[-1]}")
                model = fit_ridge(train, feature_cols, target_col, alpha=cfg.ridge_alpha, seed=cfg.seed)
                pred = predict(model, test, feature_cols)

            preds_df = test[[target_col]].copy()
            preds_df["prediction"] = pred
            preds_df = preds_df.reset_index()

            # Bucket returns at this rebalance date
            top10_ret = top_n_return(preds_df, 10, target_col)
            top20_ret = top_n_return(preds_df, 20, target_col)
            top30_ret = top_n_return(preds_df, 30, target_col)
            bottom10_ret = bottom_n_return(preds_df, 10, target_col)
            bottom20_ret = bottom_n_return(preds_df, 20, target_col)

            # Convert percent targets if needed
            # Heuristic: if abs(target) > 1.0 (e.g. 2.5), treat as percent.
            def _norm(x: float) -> float:
                if not np.isfinite(x):
                    return float("nan")
                return x / 100.0 if abs(x) > 1.0 else x

            top10_ret = _norm(top10_ret)
            top20_ret = _norm(top20_ret)
            top30_ret = _norm(top30_ret)
            bottom10_ret = _norm(bottom10_ret)
            bottom20_ret = _norm(bottom20_ret)

            qqq_r = qqq_ret.reindex([reb_date]).iloc[0] if reb_date in qqq_ret.index else float("nan")

            # Update equities
            for k, r in [
                ("top10", top10_ret),
                ("top20", top20_ret),
                ("top30", top30_ret),
                ("bottom10", bottom10_ret),
                ("bottom20", bottom20_ret),
            ]:
                if np.isfinite(r):
                    equity[k] *= (1.0 + r)
            if np.isfinite(qqq_r):
                equity["qqq"] *= (1.0 + float(qqq_r))

            # Long-short 10: +0.5 Top10, -0.5 Bottom10
            if np.isfinite(top10_ret) and np.isfinite(bottom10_ret):
                ls_ret = 0.5 * float(top10_ret) - 0.5 * float(bottom10_ret)
                equity["long_short10"] *= (1.0 + ls_ret)
            else:
                ls_ret = float("nan")

            rows.append(
                {
                    "date": reb_date,
                    "top10_ret": top10_ret,
                    "top20_ret": top20_ret,
                    "top30_ret": top30_ret,
                    "bottom10_ret": bottom10_ret,
                    "bottom20_ret": bottom20_ret,
                    "long_short10_ret": ls_ret,
                    "qqq_ret": float(qqq_r) if np.isfinite(qqq_r) else np.nan,
                    "top10_equity": equity["top10"],
                    "top20_equity": equity["top20"],
                    "top30_equity": equity["top30"],
                    "bottom10_equity": equity["bottom10"],
                    "bottom20_equity": equity["bottom20"],
                    "long_short10_equity": equity["long_short10"],
                    "qqq_equity": equity["qqq"],
                }
            )

            # Record top10 picks + BUY/HOLD/SELL (full rebalance: exit old, enter new)
            top10 = preds_df.sort_values("prediction", ascending=False).head(10).copy()
            top10["weight"] = 1.0 / 10.0
            top10["rebalance_date"] = reb_date
            picks.append(top10[["rebalance_date", "ticker", "prediction", target_col, "weight"]])

            new_holdings = {
                str(t): float(w) for t, w in zip(top10["ticker"].astype(str), top10["weight"].astype(float))
            }
            prev_set = set(prev_holdings_top10.keys())
            new_set = set(new_holdings.keys())
            buy = sorted(new_set - prev_set)
            sell = sorted(prev_set - new_set)
            hold = sorted(new_set & prev_set)

            for t in buy:
                trades.append({"date": reb_date, "ticker": t, "action": "BUY", "weight": new_holdings[t]})
            for t in hold:
                trades.append({"date": reb_date, "ticker": t, "action": "HOLD", "weight": new_holdings[t]})
            for t in sell:
                trades.append({"date": reb_date, "ticker": t, "action": "SELL", "weight": 0.0})

            # Turnover proxy: sum |w_new - w_prev| over union tickers
            prev_w = pd.Series(prev_holdings_top10, dtype=float)
            new_w = pd.Series(new_holdings, dtype=float)
            union = prev_w.index.union(new_w.index)
            turnover = float((new_w.reindex(union).fillna(0.0) - prev_w.reindex(union).fillna(0.0)).abs().sum())
            rows[-1]["top10_turnover"] = turnover
            prev_holdings_top10 = new_holdings

            # Periodic checkpoints
            if outdir is not None and cfg.checkpoint_every and (done % int(cfg.checkpoint_every) == 0):
                outdir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).sort_values("date").to_csv(outdir / "equity_curve.partial.csv", index=False)
                if picks:
                    pd.concat(picks, ignore_index=True).to_csv(outdir / "top_picks.partial.csv", index=False)
                if trades:
                    pd.DataFrame(trades).to_csv(outdir / "trades.partial.csv", index=False)
                (outdir / "progress.json").write_text(
                    json.dumps(
                        {"rebalance_done": done, "rebalance_total": total_rebalances, "last_date": str(reb_date)},
                        ensure_ascii=False,
                        indent=2,
                    )
                )

    equity_df = pd.DataFrame(rows).sort_values("date")
    picks_df = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    trades_df = pd.DataFrame(trades).sort_values(["date", "action", "ticker"]) if trades else pd.DataFrame()

    # Metrics
    periods_per_year = 252.0 / float(cfg.horizon_days)
    strat_ret = equity_df["top10_ret"]
    metrics = {
        "periods": int(len(equity_df)),
        "horizon_days": int(cfg.horizon_days),
        "purge_days": int(cfg.purge_days),
        "train_window_periods": int(cfg.train_window_periods),
        "top10_total_return": float(equity_df["top10_equity"].iloc[-1] - 1.0) if not equity_df.empty else 0.0,
        "qqq_total_return": float(equity_df["qqq_equity"].iloc[-1] - 1.0) if not equity_df.empty else 0.0,
        "top10_sharpe": float(_annualized_sharpe(strat_ret, periods_per_year=periods_per_year)),
        "qqq_sharpe": float(_annualized_sharpe(equity_df["qqq_ret"], periods_per_year=periods_per_year))
        if not equity_df.empty
        else 0.0,
        "top10_max_dd": float(_max_drawdown(equity_df["top10_equity"])) if not equity_df.empty else 0.0,
        "qqq_max_dd": float(_max_drawdown(equity_df["qqq_equity"])) if not equity_df.empty else 0.0,
        "avg_top10_turnover": float(equity_df["top10_turnover"].dropna().mean())
        if ("top10_turnover" in equity_df.columns and not equity_df.empty)
        else 0.0,
    }

    # Save leakage audit if requested
    audit_df = pd.DataFrame(audit)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        if not audit_df.empty:
            audit_df.sort_values(["rebalance_date"]).to_csv(outdir / "leakage_audit.csv", index=False)

    return {"equity": equity_df, "picks": picks_df, "trades": trades_df, "metrics": metrics, "audit": audit_df}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward Top-N backtest (Ridge) vs QQQ (T+10).")
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--horizon-days", type=int, default=10)
    p.add_argument("--purge-days", type=int, default=5)
    p.add_argument("--train-window-periods", type=int, default=252, help="Training window in trading days.")
    p.add_argument("--annual-retrain", action="store_true", help="Use 4y train -> 1y test walk-forward with annual retrain (expanding).")
    p.add_argument("--min-train-years", type=int, default=4)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--max-rebalances", type=int, default=30, help="Limit rebalances for quick runs; set 0 for no limit.")
    p.add_argument("--checkpoint-every", type=int, default=25, help="Write partial outputs every N rebalances.")
    p.add_argument("--progress-every", type=int, default=5, help="Print progress every N rebalances.")
    p.add_argument("--outdir", type=str, default="result/walkforward_top10")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = WalkforwardConfig(
        horizon_days=int(args.horizon_days),
        purge_days=int(args.purge_days),
        train_window_periods=int(args.train_window_periods),
        annual_retrain=bool(args.annual_retrain),
        min_train_years=int(args.min_train_years),
        test_years=int(args.test_years),
        ridge_alpha=float(args.ridge_alpha),
        max_rebalances=None if int(args.max_rebalances) == 0 else int(args.max_rebalances),
        seed=int(args.seed),
        checkpoint_every=int(args.checkpoint_every),
        progress_every=int(args.progress_every),
    )

    df = load_factor_data(args.data_file)
    outdir = Path(args.outdir)
    res = run_walkforward(df, cfg, outdir=outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    equity_path = outdir / "equity_curve.csv"
    picks_path = outdir / "top_picks.csv"
    summary_path = outdir / "summary.json"
    plot_path = outdir / "plot.png"

    res["equity"].to_csv(equity_path, index=False)
    res["picks"].to_csv(picks_path, index=False)
    (outdir / "trades.csv").write_text("")  # placeholder to ensure file exists even if empty
    if isinstance(res.get("trades"), pd.DataFrame) and not res["trades"].empty:
        res["trades"].to_csv(outdir / "trades.csv", index=False)
    summary_path.write_text(json.dumps(res["metrics"], ensure_ascii=False, indent=2))

    _set_mpl_agg()
    from matplotlib import pyplot as plt

    eq = res["equity"].copy()
    if not eq.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eq["date"], eq["top10_equity"], label="Top10")
        ax.plot(eq["date"], eq["top20_equity"], label="Top20", alpha=0.9)
        ax.plot(eq["date"], eq["top30_equity"], label="Top30", alpha=0.9)
        ax.plot(eq["date"], eq["bottom10_equity"], label="Bottom10", alpha=0.7)
        ax.plot(eq["date"], eq["bottom20_equity"], label="Bottom20", alpha=0.7)
        ax.plot(eq["date"], eq["long_short10_equity"], label="Long-Short10", alpha=0.85)
        ax.plot(eq["date"], eq["qqq_equity"], label="QQQ", alpha=0.85)
        ax.set_title("Walk-forward buckets vs QQQ (T+10)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        # Per-period returns comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(eq["date"], eq["top10_ret"], label="Top10 ret")
        ax.plot(eq["date"], eq["bottom10_ret"], label="Bottom10 ret", alpha=0.8)
        ax.plot(eq["date"], eq["long_short10_ret"], label="Long-Short10 ret", alpha=0.9)
        ax.plot(eq["date"], eq["qqq_ret"], label="QQQ ret", alpha=0.85)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
        ax.set_title("Per-period returns (each rebalance)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "returns_per_period.png", dpi=150)
        plt.close(fig)

    print(f"[walkforward_top10] saved: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


