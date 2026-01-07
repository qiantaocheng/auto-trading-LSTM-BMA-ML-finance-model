from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 10
    rebalance_step: int = 1  # use every prediction date; set >1 to downsample
    horizon_days: int = 10  # forward return horizon behind the `actual` column (trading days)
    max_weight_per_name: float = 0.15
    min_weight_per_name: float = 0.0
    allow_short: bool = False

    # Market gating thresholds (from HETRS signals)
    market_good_prob: float = 0.70
    market_bad_prob: float = 0.55

    # Position sizing profile
    top3_boost: float = 1.25  # +25% weight multiplier for ranks 1-3
    bottom_bucket_cut: float = 0.60  # down-weight ranks 7-10 by 40%


def _annualized_sharpe(r: pd.Series, periods_per_year: int = 252) -> float:
    x = r.dropna().astype(float).values
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    x = equity.astype(float)
    peak = x.cummax()
    dd = (x / peak) - 1.0
    return float(dd.min())


def load_ridge_predictions(path: str) -> pd.DataFrame:
    """
    Expect parquet/csv with at least: date, ticker, prediction.
    Optionally: actual (forward return).
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if not {"date", "ticker", "prediction"}.issubset(df.columns):
        raise ValueError("Ridge predictions must contain columns: date,ticker,prediction")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["prediction"] = df["prediction"].astype(float)
    if "actual" in df.columns:
        df["actual"] = df["actual"].astype(float)
    return df


def load_market_signals(path: str) -> pd.DataFrame:
    """
    Expect parquet with index=date and columns:
      - tft_p50
      - meta_prob_success
      - market_regime
      - exposure_scalar
    """
    ms = pd.read_parquet(path).copy()
    if not isinstance(ms.index, pd.DatetimeIndex):
        raise ValueError("Market signals must be indexed by date.")
    return ms.sort_index()


def _rebalance_dates(dates: pd.DatetimeIndex, step: int) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(dates) == 0:
        return dates
    return dates[:: max(1, int(step))]


def market_state_for_date(ms: pd.DataFrame, d: pd.Timestamp, cfg: StrategyConfig) -> tuple[str, float]:
    """
    Returns (state, exposure_scalar).
    If no market signal available, treat as neutral.
    """
    if d not in ms.index:
        return "neutral", 0.6
    row = ms.loc[d]
    exp = float(row.get("exposure_scalar", 0.6))
    state = str(row.get("market_regime", "neutral"))
    # enforce thresholds if raw fields present
    if "meta_prob_success" in row and "tft_p50" in row:
        p = float(row["meta_prob_success"])
        t = float(row["tft_p50"])
        if (t > 0) and (p >= cfg.market_good_prob):
            return "favorable", 1.0
        if (t < 0) or (p <= cfg.market_bad_prob):
            return "unfavorable", 0.2
        return "neutral", 0.6
    return state, exp


def compute_target_weights(top_df: pd.DataFrame, cfg: StrategyConfig, exposure: float) -> pd.Series:
    """
    Build weights for the Top-N names for the rebalance date.
    Uses rank-based weights + boosts/cuts, then scales by exposure.
    """
    df = top_df.sort_values("prediction", ascending=False).head(cfg.top_n).copy()
    df["rank"] = np.arange(1, len(df) + 1)

    # Base weights: linear decay by rank
    w = (cfg.top_n + 1 - df["rank"]).astype(float)
    w = w / w.sum()

    # Top 1-3 boost
    w = w.where(df["rank"] > 3, w * cfg.top3_boost)
    # Ranks 7-10 cut (if exists)
    w = w.where(df["rank"] < 7, w * cfg.bottom_bucket_cut)

    # Normalize again
    w = w / w.sum()

    # Apply exposure scalar (cash remainder)
    w = w * float(np.clip(exposure, 0.0, 1.0))

    # Cap per-name
    w = w.clip(lower=cfg.min_weight_per_name, upper=cfg.max_weight_per_name)
    # Renormalize to the exposure (after clipping)
    total = w.sum()
    if total > 0:
        w = w * (float(np.clip(exposure, 0.0, 1.0)) / total)

    return pd.Series(w.values, index=df["ticker"].values, name="target_weight")


def backtest_portfolio(
    preds: pd.DataFrame,
    ms: pd.DataFrame,
    cfg: StrategyConfig,
) -> dict:
    """
    Backtest using "actual" column as the forward return for the holding period.
    If "actual" is missing, we can only generate weights/trades (no performance).
    """
    dates = pd.to_datetime(preds["date"].unique())
    # IMPORTANT: ridge_stacking_predictions files are typically produced at the rebalance cadence already.
    # Therefore, default behavior is to rebalance on *each prediction date*.
    reb_dates = _rebalance_dates(pd.DatetimeIndex(dates), cfg.rebalance_step)

    # Track weights on rebalance dates
    weights_records = []
    trades_records = []
    equity = 1.0
    equity_curve = []
    prev_w = pd.Series(dtype=float)

    for d in reb_dates:
        day_df = preds[preds["date"] == d].copy()
        if day_df.empty:
            continue

        state, exposure = market_state_for_date(ms, d, cfg)
        target_w = compute_target_weights(day_df, cfg, exposure=exposure)

        # Trades = delta weights
        combined = pd.DataFrame({"prev": prev_w, "target": target_w}).fillna(0.0)
        combined["delta"] = combined["target"] - combined["prev"]
        combined["date"] = d
        combined["market_state"] = state
        trades = combined[combined["delta"].abs() > 1e-12].copy()
        trades["action"] = np.where(trades["delta"] > 0, "BUY/ADD", "SELL/REDUCE")
        trades_records.append(trades.reset_index().rename(columns={"index": "ticker"}))

        wrec = target_w.reset_index()
        wrec.columns = ["ticker", "weight"]
        wrec["date"] = d
        wrec["market_state"] = state
        weights_records.append(wrec)

        # Performance update using actual forward returns (assumed % or decimal? treat as percent if abs>2)
        if "actual" in day_df.columns:
            actual = day_df.set_index("ticker")["actual"].reindex(target_w.index)
            r = actual.astype(float)
            # Heuristic: many pipelines store forward returns in *percent* units.
            # If any magnitude exceeds 1.0, we treat values as percent and convert to decimals.
            if r.abs().max() > 1.0:
                r = r / 100.0
            port_ret = float((target_w * r).sum())
            equity *= (1.0 + port_ret)
            equity_curve.append({"date": d, "equity": equity, "portfolio_return": port_ret, "market_state": state})

        prev_w = target_w

    weights_df = pd.concat(weights_records, ignore_index=True) if weights_records else pd.DataFrame()
    trades_df = pd.concat(trades_records, ignore_index=True) if trades_records else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve).set_index("date") if equity_curve else pd.DataFrame()

    metrics = {}
    if not equity_df.empty:
        r = equity_df["portfolio_return"]
        # Annualize using horizon_days (returns are per holding period, e.g., 10 trading days).
        periods_per_year = max(1, int(round(252 / max(1, cfg.horizon_days))))
        metrics = {
            "cagr": float((equity_df["equity"].iloc[-1] ** (periods_per_year / len(equity_df))) - 1.0),
            "sharpe": float(_annualized_sharpe(r, periods_per_year=periods_per_year)),
            "max_drawdown": float(_max_drawdown(equity_df["equity"])),
            "total_return": float(equity_df["equity"].iloc[-1] - 1.0),
            "periods": int(len(equity_df)),
            "periods_per_year": int(periods_per_year),
        }

    # Turnover (sum abs delta)
    if not trades_df.empty:
        metrics["avg_turnover_per_rebalance"] = float(trades_df.groupby("date")["delta"].apply(lambda x: x.abs().sum()).mean())
        metrics["rebalance_count"] = int(trades_df["date"].nunique())

    return {"metrics": metrics, "weights": weights_df, "trades": trades_df, "equity": equity_df}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ridge-stacking portfolio manager (Top-N bi-weekly).")
    p.add_argument("--ridge-preds", required=True, help="Path to ridge_stacking_predictions_*.parquet (date,ticker,prediction,actual)")
    p.add_argument("--market-signals", required=True, help="Parquet with market signals (from hetrs_nasdaq.market_signals)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--rebalance-step",
        type=int,
        default=1,
        help="Use every Nth prediction date. Default 1 = rebalance on each prediction date.",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=10,
        help="Forward return horizon behind the 'actual' column, used for annualization.",
    )
    p.add_argument("--max-weight", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds = load_ridge_predictions(args.ridge_preds)
    ms = load_market_signals(args.market_signals)

    cfg = StrategyConfig(
        top_n=int(args.top_n),
        rebalance_step=int(args.rebalance_step),
        horizon_days=int(args.horizon_days),
        max_weight_per_name=float(args.max_weight),
    )

    res = backtest_portfolio(preds, ms, cfg)
    res["weights"].to_csv(outdir / "weights.csv", index=False)
    res["trades"].to_csv(outdir / "trades.csv", index=False)
    if isinstance(res["equity"], pd.DataFrame) and not res["equity"].empty:
        res["equity"].to_csv(outdir / "equity.csv")

    (outdir / "metrics.json").write_text(json.dumps(res["metrics"], ensure_ascii=False, indent=2))
    print(f"[ridge_portfolio] saved: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


