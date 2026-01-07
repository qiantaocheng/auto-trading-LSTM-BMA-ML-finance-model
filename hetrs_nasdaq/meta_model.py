from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.repro import set_global_seed


def primary_signal_from_p50(p50: pd.Series, threshold: float = 0.002) -> pd.Series:
    """
    Naive primary signal based on TFT median prediction.
      +1 if p50 > threshold
      -1 if p50 < -threshold
       0 otherwise
    """
    x = p50.astype(float)
    sig = pd.Series(0, index=x.index, dtype=int)
    sig[x > threshold] = 1
    sig[x < -threshold] = -1
    return sig.rename("primary_signal")


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meta-features X_meta:
      - tft_uncertainty = tft_p90 - tft_p10
      - vol_gk
      - regime probabilities (regime_p0/p1/p2 preferred; fallback to regime_prob_0/1/2)
      - rsi, macd
    """
    out = pd.DataFrame(index=df.index)
    out["tft_uncertainty"] = df["tft_p90"].astype(float) - df["tft_p10"].astype(float)
    out["vol_gk"] = df["vol_gk"].astype(float)

    if {"regime_p0", "regime_p1", "regime_p2"}.issubset(df.columns):
        out["regime_p0"] = df["regime_p0"].astype(float)
        out["regime_p1"] = df["regime_p1"].astype(float)
        out["regime_p2"] = df["regime_p2"].astype(float)
    elif {"regime_prob_0", "regime_prob_1", "regime_prob_2"}.issubset(df.columns):
        out["regime_p0"] = df["regime_prob_0"].astype(float)
        out["regime_p1"] = df["regime_prob_1"].astype(float)
        out["regime_p2"] = df["regime_prob_2"].astype(float)
    else:
        out["regime_p0"] = 1.0 / 3.0
        out["regime_p1"] = 1.0 / 3.0
        out["regime_p2"] = 1.0 / 3.0

    out["rsi"] = df["rsi"].astype(float) if "rsi" in df.columns else df["rsi_14"].astype(float)
    out["macd"] = df["macd"].astype(float)
    return out


def build_meta_labels(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    future_return_5d: pd.Series,
) -> pd.Series:
    """
    Meta-labels y_meta:
      - If primary_signal == +1 and future 5d return > 0 => y=1
      - If primary_signal == -1 and future 5d return < 0 => y=1
      - Else y=0
    Only defined when primary_signal != 0.
    """
    sig = primary_signal.reindex(df.index).astype(int)
    y = pd.Series(0, index=df.index, dtype=int)
    fr = future_return_5d.reindex(df.index).astype(float)

    y[(sig == 1) & (fr > 0)] = 1
    y[(sig == -1) & (fr < 0)] = 1

    # Only meaningful on trade attempts; keep 0 elsewhere but caller usually filters sig!=0.
    return y.rename("y_meta")


@dataclass(frozen=True)
class MetaModelConfig:
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42


class MetaLabelModel:
    def __init__(self, cfg: MetaModelConfig = MetaModelConfig()):
        self.cfg = cfg
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLabelModel":
        from sklearn.ensemble import RandomForestClassifier

        set_global_seed(self.cfg.random_state)
        clf = RandomForestClassifier(
            n_estimators=int(self.cfg.n_estimators),
            max_depth=int(self.cfg.max_depth),
            random_state=int(self.cfg.random_state),
            n_jobs=-1,
        )
        clf.fit(X.values, y.values)
        self.model = clf
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MetaLabelModel is not fitted.")
        p = self.model.predict_proba(X.values)
        # probability of class 1
        return p[:, 1]


