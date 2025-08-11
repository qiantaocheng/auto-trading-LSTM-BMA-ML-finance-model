#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weekly BMA Backtest Framework

Goal
- Weekly rebalance (W-FRI). Signals from weekly close-only features.
- No transaction costs. Long-only, equal-weight Top-N.
- Metrics: Annualized return (CAGR), Max Drawdown, Sharpe (risk-free 0).

Data Inputs
- Close prices pivot DataFrame (index: daily DateTimeIndex, columns: tickers, values: close).
  The script resamples to weekly (W-FRI) closes.

Model
- Uses QuantitativeModel (BMA ensemble) from `量化模型_bma_enhanced.py`.
- Training windows are rolling over the weekly panel.
- Preprocessing: relies on QuantitativeModel's saved preprocessing_function
  (lag to t-1, winsorization, schmidt orthogonalization) using 'date'/'ticker'.

Usage (example)
  python scripts/weekly_bma_backtest.py --closes_csv path/to/daily_closes.csv \
         --start 2018-01-05 --end 2020-12-25 --top_n 10 --train_weeks 52

Notes
- For a production setup, wire your own data loader that supplies `daily_closes` DataFrame.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _add_project_root_to_path():
    """Ensure project root is importable so we can import the unicode-named module."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_add_project_root_to_path()

try:
    # Direct import of unicode-named module (Python 3 supports it)
    from 量化模型_bma_enhanced import (
        QuantitativeModel,
    )
except Exception as import_error:
    raise RuntimeError(
        f"Unable to import QuantitativeModel from 量化模型_bma_enhanced.py: {import_error}"
    )


def compute_weekly_closes_from_daily(daily_closes: pd.DataFrame) -> pd.DataFrame:
    """Resample daily close prices to weekly Friday close. Drops all-NaN rows.

    Args:
        daily_closes: DataFrame with daily DateTimeIndex (tz-naive), columns=tickers
    Returns:
        weekly_closes: DataFrame indexed by Fridays (W-FRI), last available close of week
    """
    if not isinstance(daily_closes.index, pd.DatetimeIndex):
        raise ValueError("daily_closes index must be a DatetimeIndex")
    weekly = daily_closes.resample("W-FRI").last()
    weekly = weekly.dropna(how="all")
    return weekly


def build_weekly_close_features(weekly_close: pd.Series) -> pd.DataFrame:
    """Build close-only weekly features for a single ticker.

    Features include momentum, mean-reversion, and normalized signals:
    - ret_1w, ret_4w, ret_12w
    - mom_4_12 = ret_4w - ret_12w
    - sma_4w, sma_12w, sma_ratio_4_12 = sma_4w / sma_12w - 1
    - zscore_4w = (close - sma_4w) / std_4w
    - vol_4w = std of weekly returns over 4w

    Args:
        weekly_close: Series indexed by weekly dates (W-FRI)
    Returns:
        DataFrame with same index, feature columns
    """
    close = weekly_close.astype(float)
    ret_1w = close.pct_change(1)
    ret_4w = close.pct_change(4)
    ret_12w = close.pct_change(12)

    sma_4w = close.rolling(4).mean()
    sma_12w = close.rolling(12).mean()
    std_4w = close.rolling(4).std()
    vol_4w = ret_1w.rolling(4).std()

    mom_4_12 = ret_4w - ret_12w
    sma_ratio_4_12 = (sma_4w / (sma_12w.replace(0, np.nan))) - 1.0
    zscore_4w = (close - sma_4w) / (std_4w.replace(0, np.nan))

    df = pd.DataFrame(
        {
            "ret_1w": ret_1w,
            "ret_4w": ret_4w,
            "ret_12w": ret_12w,
            "mom_4_12": mom_4_12,
            "sma_4w": sma_4w,
            "sma_12w": sma_12w,
            "sma_ratio_4_12": sma_ratio_4_12,
            "zscore_4w": zscore_4w,
            "vol_4w": vol_4w,
        }
    )
    return df


def build_panel_features_from_weekly_closes(weekly_closes: pd.DataFrame) -> pd.DataFrame:
    """Build close-only weekly features for all tickers and return a long-form panel.

    Output columns: ['date','ticker', <features...>, 'close']
    """
    frames: List[pd.DataFrame] = []
    for ticker in weekly_closes.columns:
        series = weekly_closes[ticker].dropna()
        if series.empty:
            continue
        feats = build_weekly_close_features(series)
        feats = feats.assign(ticker=ticker, date=feats.index, close=series)
        frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"])  # empty
    panel = pd.concat(frames, axis=0, ignore_index=False)
    # ensure consistent column order
    cols = ["date", "ticker"] + [c for c in panel.columns if c not in ("date", "ticker")]
    panel = panel.reset_index(drop=True)[cols]
    return panel


def compute_next_week_return(weekly_close: pd.Series) -> pd.Series:
    """Compute next-week forward return (t -> t+1 close).

    Returns a Series aligned on t (current week), using shift(-1).
    """
    return weekly_close.shift(-1) / weekly_close - 1.0


def attach_targets(panel_features: pd.DataFrame, weekly_closes: pd.DataFrame) -> pd.DataFrame:
    """Attach next-week target to long-form features panel.

    Expects panel with 'date' and 'ticker'. Looks up close from `weekly_closes`.
    """
    if panel_features.empty:
        return panel_features
    # Map next-week return per ticker
    targets: List[pd.Series] = []
    for ticker, group in panel_features.groupby("ticker"):
        if ticker not in weekly_closes.columns:
            continue
        series = weekly_closes[ticker]
        fwd_ret = compute_next_week_return(series)
        # align to group dates
        aligned = fwd_ret.reindex(pd.to_datetime(group["date"]))
        targets.append(pd.Series(aligned.values, index=group.index))
    if targets:
        target_series = pd.concat(targets).sort_index()
        panel_features = panel_features.copy()
        panel_features["target"] = target_series
    return panel_features


@dataclass
class BacktestConfig:
    top_n: int = 10
    train_weeks: int = 52
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None


class WeeklyBMAPortfolioBacktester:
    """Weekly long-only Top-N backtester using QuantitativeModel (BMA)."""

    def __init__(self, weekly_closes: pd.DataFrame, config: BacktestConfig):
        if not isinstance(weekly_closes.index, pd.DatetimeIndex):
            raise ValueError("weekly_closes must be indexed by DatetimeIndex")
        self.weekly_closes = weekly_closes.sort_index()
        self.config = config
        self.model: Optional[QuantitativeModel] = None
        self.returns: List[float] = []
        self.return_dates: List[pd.Timestamp] = []

    def _build_training_data(
        self, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> pd.DataFrame:
        closes_window = self.weekly_closes.loc[window_start:window_end]
        panel = build_panel_features_from_weekly_closes(closes_window)
        panel = attach_targets(panel, closes_window)
        # Keep rows with finite target and finite features
        if "target" in panel.columns:
            panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"])  # keep target notna
        return panel

    def _train_model(self, train_df: pd.DataFrame) -> QuantitativeModel:
        model = QuantitativeModel()

        # Prepare X, y with meta columns for preprocessing inside model
        feature_cols = [c for c in train_df.columns if c not in ["date", "ticker", "target", "close"]]
        X = train_df[[*feature_cols]].copy()
        X["date"] = pd.to_datetime(train_df["date"]).values
        X["ticker"] = train_df["ticker"].values
        y = train_df["target"].astype(float)

        # Dates for time-decay weights
        dates = pd.to_datetime(train_df["date"]) if "date" in train_df.columns else None
        tickers = train_df["ticker"] if "ticker" in train_df.columns else None

        # Train via QuantitativeModel's BMA pipeline
        # This method will run internal preprocessing including lagging and orthogonalization.
        model.train_models_with_bma(
            X=X,
            y=y,
            enable_hyperopt=False,
            apply_preprocessing=True,
            dates=dates,
            tickers=tickers,
        )
        return model

    def _predict_current_week(self, model: QuantitativeModel, week_date: pd.Timestamp) -> pd.Series:
        # Build features for the specific week across all tickers
        closes_up_to_week = self.weekly_closes.loc[:week_date]
        current_panel = build_panel_features_from_weekly_closes(closes_up_to_week.tail(60))  # small lookback for features
        current_week_rows = current_panel[current_panel["date"] == week_date]
        if current_week_rows.empty:
            return pd.Series(dtype=float)

        # Apply the same preprocessing as training (saved on the model)
        if hasattr(model, "preprocessing_function") and model.preprocessing_function is not None:
            X_with_meta = current_week_rows.copy()
            X_processed = model.preprocessing_function(X_with_meta)
            # Keep rows with no NaNs in features
            valid_idx = ~X_processed.drop(["date", "ticker"], axis=1).isnull().any(axis=1)
            X_processed = X_processed[valid_idx]
            # Save original ticker alignment after processing
            tickers = X_processed["ticker"].values
            X_feat = X_processed.drop(["date", "ticker"], axis=1)
        else:
            # Fallback: drop meta columns and hope columns align
            tickers = current_week_rows["ticker"].values
            X_feat = current_week_rows.drop(columns=["date", "ticker", "close"], errors="ignore")

        # Predict with BMA
        y_pred = model.predict_with_bma(X_feat)
        # Map back to tickers
        return pd.Series(y_pred, index=tickers)

    def run(self) -> Dict[str, float]:
        cfg = self.config
        start = cfg.start_date or self.weekly_closes.index.min()
        end = cfg.end_date or self.weekly_closes.index.max()

        # Ensure weekly alignment to existing index dates
        all_weeks = self.weekly_closes.index[(self.weekly_closes.index >= start) & (self.weekly_closes.index <= end)]
        all_weeks = pd.DatetimeIndex(all_weeks)

        for idx in range(cfg.train_weeks, len(all_weeks) - 1):
            signal_week = all_weeks[idx]
            next_week = all_weeks[idx + 1]
            window_start = all_weeks[idx - cfg.train_weeks]

            # 1) Build train set and train model
            train_df = self._build_training_data(window_start, signal_week)
            if train_df.empty or train_df["target"].notna().sum() < 50:
                # Not enough data to train
                continue
            model = self._train_model(train_df)

            # 2) Predict for signal week
            pred_scores = self._predict_current_week(model, signal_week)
            if pred_scores.empty:
                continue

            # 3) Select Top-N tickers
            top = pred_scores.sort_values(ascending=False).head(cfg.top_n)
            selected = list(top.index)
            if len(selected) == 0:
                continue

            # 4) Compute realized portfolio return (equal-weight) from signal_week -> next_week
            current_prices = self.weekly_closes.loc[signal_week, selected]
            next_prices = self.weekly_closes.loc[next_week, selected]
            # Drop any tickers with missing prices on either week
            valid = (~current_prices.isna()) & (~next_prices.isna())
            if valid.sum() == 0:
                continue
            rets = next_prices[valid] / current_prices[valid] - 1.0
            portfolio_ret = rets.mean()

            self.returns.append(float(portfolio_ret))
            self.return_dates.append(next_week)

        return self._evaluate()

    def _evaluate(self) -> Dict[str, float]:
        if not self.returns:
            return {"annual_return": np.nan, "max_drawdown": np.nan, "sharpe": np.nan}

        ret_series = pd.Series(self.returns, index=pd.DatetimeIndex(self.return_dates)).sort_index()
        equity = (1.0 + ret_series).cumprod()

        T = len(ret_series)
        annual_factor = 52.0
        final_nav = float(equity.iloc[-1])
        annual_return = final_nav ** (annual_factor / max(T, 1)) - 1.0

        rolling_max = equity.cummax()
        drawdowns = equity / rolling_max - 1.0
        max_drawdown = float(drawdowns.min())  # negative
        max_drawdown_pct = -max_drawdown

        mean_w = float(ret_series.mean())
        std_w = float(ret_series.std())
        sharpe = (mean_w / std_w) * np.sqrt(annual_factor) if std_w > 0 else np.nan

        return {
            "annual_return": float(annual_return),
            "max_drawdown": float(max_drawdown_pct),
            "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        }


def _load_daily_closes_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect first column to be date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()
    return df


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Weekly BMA Backtest")
    parser.add_argument("--closes_csv", type=str, required=True, help="CSV of daily closes with Date column + ticker columns")
    parser.add_argument("--start", type=str, default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--top_n", type=int, default=10, help="Top-N stocks per week")
    parser.add_argument("--train_weeks", type=int, default=52, help="Training window length in weeks")

    args = parser.parse_args(argv)

    daily_closes = _load_daily_closes_from_csv(args.closes_csv)
    weekly_closes = compute_weekly_closes_from_daily(daily_closes)

    cfg = BacktestConfig(
        top_n=args.top_n,
        train_weeks=args.train_weeks,
        start_date=pd.to_datetime(args.start) if args.start else None,
        end_date=pd.to_datetime(args.end) if args.end else None,
    )

    backtester = WeeklyBMAPortfolioBacktester(weekly_closes=weekly_closes, config=cfg)
    results = backtester.run()

    print("Weekly BMA Backtest Results")
    print(f"Annual Return (CAGR): {results['annual_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe (rf=0): {results['sharpe']:.3f}")


if __name__ == "__main__":
    main()


