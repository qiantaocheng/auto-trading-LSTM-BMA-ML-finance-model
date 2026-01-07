from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.meta_model import MetaLabelModel, MetaModelConfig, build_meta_features
from hetrs_nasdaq.repro import set_global_seed


@dataclass(frozen=True)
class MarketSignalConfig:
    seed: int = 42
    # Causal fit: expanding window for meta model
    min_train_obs: int = 252
    refit_interval: int = 21
    primary_threshold: float = 0.001  # for display only; primary signal is not used as a filter here


def _future_return_5d(close: pd.Series) -> pd.Series:
    return close.astype(float).pct_change(5).shift(-5)


def compute_meta_prob_success_expanding(
    df: pd.DataFrame,
    cfg: MarketSignalConfig = MarketSignalConfig(),
) -> pd.Series:
    """
    Compute a **causal** meta_prob_success time series for QQQ:
    - Label: future 5d return > 0 (market up) => 1 else 0
    - Features: same X_meta used in v3.1
    - Fit: expanding window, refit every `refit_interval` rows

    This produces a probability you can use as a "market confidence" signal.
    """
    set_global_seed(cfg.seed)
    df = df.sort_index().copy()
    y = (_future_return_5d(df["Close"]) > 0).astype(int)
    X_all = build_meta_features(df)

    prob = pd.Series(index=df.index, dtype=float, name="meta_prob_success")
    last_fit_i: Optional[int] = None
    model: Optional[MetaLabelModel] = None

    for i in range(len(df)):
        if i < cfg.min_train_obs:
            continue
        if last_fit_i is None or (i - last_fit_i) >= cfg.refit_interval:
            train = slice(0, i)  # strictly past
            X_train = X_all.iloc[train]
            y_train = y.iloc[train]
            m = (~X_train.isna().any(axis=1)) & (~y_train.isna())
            X_train = X_train.loc[m]
            y_train = y_train.loc[m]
            if len(X_train) >= cfg.min_train_obs and y_train.nunique() > 1:
                mm = MetaLabelModel(MetaModelConfig(random_state=cfg.seed))
                mm.fit(X_train, y_train)
                model = mm
            else:
                model = None
            last_fit_i = i

        if model is None:
            continue
        row = X_all.iloc[i : i + 1]
        if row.isna().any(axis=1).iloc[0]:
            continue
        prob.iloc[i] = float(model.predict_proba(row)[0])

    return prob


def build_market_signals(
    features_with_tft: pd.DataFrame,
    cfg: MarketSignalConfig = MarketSignalConfig(),
) -> pd.DataFrame:
    """
    Output columns:
    - tft_p50: market trend
    - meta_prob_success: causal confidence estimate
    - market_regime: {favorable, neutral, unfavorable}
    - exposure_scalar: [0..1] suggested gross exposure scaling
    """
    df = features_with_tft.sort_index().copy()
    if "tft_p50" not in df.columns:
        raise ValueError("features_with_tft must contain tft_p50 for market signals.")

    ms = pd.DataFrame(index=df.index)
    ms["tft_p50"] = df["tft_p50"].astype(float)
    ms["meta_prob_success"] = compute_meta_prob_success_expanding(df, cfg=cfg)

    # Regime mapping (configurable later)
    cond_fav = (ms["tft_p50"] > 0) & (ms["meta_prob_success"] >= 0.70)
    cond_unfav = (ms["tft_p50"] < 0) | (ms["meta_prob_success"] <= 0.55)

    ms["market_regime"] = "neutral"
    ms.loc[cond_fav, "market_regime"] = "favorable"
    ms.loc[cond_unfav, "market_regime"] = "unfavorable"

    # Exposure scalar: unfavorable -> low exposure, neutral -> medium, favorable -> high
    ms["exposure_scalar"] = 0.6
    ms.loc[ms["market_regime"] == "favorable", "exposure_scalar"] = 1.0
    ms.loc[ms["market_regime"] == "unfavorable", "exposure_scalar"] = 0.2

    return ms


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily market signals from HETRS-NASDAQ features+TFT.")
    p.add_argument("--in", dest="inp", required=True, help="Input parquet with QQQ features + tft_p10/p50/p90")
    p.add_argument("--out", required=True, help="Output parquet path for market signals")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-obs", type=int, default=252)
    p.add_argument("--refit-interval", type=int, default=21)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp)
    cfg = MarketSignalConfig(
        seed=int(args.seed),
        min_train_obs=int(args.min_train_obs),
        refit_interval=int(args.refit_interval),
    )
    ms = build_market_signals(df, cfg=cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ms.to_parquet(args.out)
    meta = {
        "rows": int(len(ms)),
        "start": str(ms.index.min()),
        "end": str(ms.index.max()),
        "seed": cfg.seed,
        "min_train_obs": cfg.min_train_obs,
        "refit_interval": cfg.refit_interval,
    }
    (Path(args.out).with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[market_signals] saved: {args.out} rows={len(ms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


