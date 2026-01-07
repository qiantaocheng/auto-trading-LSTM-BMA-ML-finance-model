from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _require_statsmodels():
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        return adfuller
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for ADF test. Install: pip install statsmodels"
        ) from e


def _try_ta_import():
    try:
        import ta  # type: ignore

        return ta
    except Exception:
        return None


def _get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """
    Fixed-width window fractional differencing weights (Lopez de Prado FFD).
    Stops when abs(weight) < threshold.
    """
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Apply fixed-width window fractional differencing to a series.
    Returns a series aligned with input index; leading values are NaN until the window is full.
    """
    series = series.astype(float)
    w = _get_weights_ffd(d=d, threshold=threshold)
    width = len(w)
    out = pd.Series(index=series.index, dtype=float, name=f"{series.name}_ffd")

    x = series.values
    for i in range(width - 1, len(series)):
        window = x[i - width + 1 : i + 1]
        if np.any(np.isnan(window)):
            out.iloc[i] = np.nan
        else:
            out.iloc[i] = float(np.dot(w, window))
    return out


def find_min_d_for_adf(
    series: pd.Series,
    d_min: float = 0.1,
    d_max: float = 1.0,
    step: float = 0.05,
    threshold: float = 1e-5,
    pvalue_target: float = 0.05,
    min_obs: int = 200,
) -> float:
    """
    Search for the smallest d in [d_min, d_max] such that ADF test p-value < pvalue_target.
    """
    adfuller = _require_statsmodels()

    s = series.dropna().astype(float)
    if len(s) < min_obs:
        raise ValueError(f"Not enough observations for ADF search: n={len(s)} < {min_obs}")

    best_d = d_max
    for d in np.arange(d_min, d_max + 1e-12, step):
        fd = frac_diff_ffd(s, d=float(d), threshold=threshold).dropna()
        if len(fd) < min_obs:
            continue
        p = float(adfuller(fd.values, autolag="AIC")[1])
        if p < pvalue_target:
            best_d = float(d)
            break
    return best_d


@dataclass(frozen=True)
class RegimeModel:
    gmm: object
    scaler: object
    feature_cols: tuple[str, ...]


def fit_regime_gmm(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("returns", "vol_gk"),
    n_components: int = 3,
    fit_until: Optional[pd.Timestamp] = None,
    random_state: int = 42,
) -> RegimeModel:
    """
    Fit a 3-state GMM using (returns, vol_gk).
    If fit_until is provided, fit only on df.loc[:fit_until] to avoid lookahead.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    data = df.copy()
    if fit_until is not None:
        data = data.loc[:fit_until]

    X = data.loc[:, list(feature_cols)].dropna()
    if X.empty:
        raise ValueError("No data available to fit regime GMM.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(Xs)
    return RegimeModel(gmm=gmm, scaler=scaler, feature_cols=feature_cols)


def add_regime_probabilities(df: pd.DataFrame, model: RegimeModel) -> pd.DataFrame:
    out = df.copy()
    X = out.loc[:, list(model.feature_cols)]
    mask = ~X.isna().any(axis=1)
    probs = np.full((len(out), 3), np.nan, dtype=float)
    if mask.any():
        Xs = model.scaler.transform(X.loc[mask].values)
        p = model.gmm.predict_proba(Xs)
        probs[mask.values, :] = p
    for i in range(3):
        out[f"regime_prob_{i}"] = probs[:, i]
    return out


def add_regime_probabilities_expanding(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("returns", "vol_gk"),
    n_components: int = 3,
    min_obs: int = 252,
    refit_interval: int = 21,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Causal / no-lookahead regime probabilities:
    - For each date t, fit GMM on data up to t-1 (expanding window)
    - Refit only every `refit_interval` rows for speed
    """
    out = df.copy()
    probs = np.full((len(out), 3), np.nan, dtype=float)

    last_fit_i = None
    model: Optional[RegimeModel] = None

    for i in range(len(out)):
        if i < min_obs:
            continue

        if last_fit_i is None or (i - last_fit_i) >= refit_interval:
            train = out.iloc[:i].copy()  # strictly past
            model = fit_regime_gmm(
                train,
                feature_cols=feature_cols,
                n_components=n_components,
                fit_until=None,
                random_state=random_state,
            )
            last_fit_i = i

        assert model is not None
        row = out.iloc[i : i + 1].copy()
        X = row.loc[:, list(feature_cols)]
        if X.isna().any(axis=1).iloc[0]:
            continue
        Xs = model.scaler.transform(X.values)
        p = model.gmm.predict_proba(Xs)[0]
        probs[i, :] = p

    for k in range(3):
        out[f"regime_prob_{k}"] = probs[:, k]
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - RSI(14)
    - MACD (12,26,9) -> macd, macd_signal, macd_diff
    - Bollinger Bands width (20,2) -> bb_width
    """
    out = df.copy()
    ta = _try_ta_import()
    close = out["Close"].astype(float)

    if ta is not None:
        out["rsi_14"] = ta.momentum.rsi(close, window=14, fillna=np.nan)
        out["macd"] = ta.trend.macd(close, window_slow=26, window_fast=12, fillna=np.nan)
        out["macd_signal"] = ta.trend.macd_signal(
            close, window_slow=26, window_fast=12, window_sign=9, fillna=np.nan
        )
        out["macd_diff"] = ta.trend.macd_diff(
            close, window_slow=26, window_fast=12, window_sign=9, fillna=np.nan
        )
        bb_h = ta.volatility.bollinger_hband(close, window=20, window_dev=2, fillna=np.nan)
        bb_l = ta.volatility.bollinger_lband(close, window=20, window_dev=2, fillna=np.nan)
        bb_m = ta.volatility.bollinger_mavg(close, window=20, fillna=np.nan)
        out["bb_width"] = ((bb_h - bb_l) / bb_m).replace([np.inf, -np.inf], np.nan)
        # Prompt-compatible aliases
        out["rsi"] = out["rsi_14"]
        return out

    # Fallback (no ta installed): minimal implementations
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi"] = out["rsi_14"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_diff"] = out["macd"] - out["macd_signal"]

    mavg = close.rolling(20).mean()
    mstd = close.rolling(20).std(ddof=0)
    bb_h = mavg + 2.0 * mstd
    bb_l = mavg - 2.0 * mstd
    out["bb_width"] = ((bb_h - bb_l) / mavg).replace([np.inf, -np.inf], np.nan)
    return out


def build_features(
    df: pd.DataFrame,
    ffd_threshold: float = 1e-5,
    d_step: float = 0.05,
    d_search_ratio: float = 0.6,
    regime_min_obs: int = 252,
    regime_refit_interval: int = 21,
    random_state: int = 42,
) -> pd.DataFrame:
    out = df.copy()

    # Fractional differencing on Close
    n = len(out)
    cut = max(200, int(n * float(d_search_ratio)))
    d = find_min_d_for_adf(out["Close"].iloc[:cut], step=d_step, threshold=ffd_threshold)
    out["close_ffd"] = frac_diff_ffd(out["Close"], d=d, threshold=ffd_threshold)

    # Regime probabilities (causal expanding fit to avoid lookahead)
    out = add_regime_probabilities_expanding(
        out,
        feature_cols=("returns", "vol_gk"),
        n_components=3,
        min_obs=regime_min_obs,
        refit_interval=regime_refit_interval,
        random_state=random_state,
    )

    # Technical indicators
    out = add_technical_indicators(out)

    # Prompt-compatible regime aliases
    if {"regime_prob_0", "regime_prob_1", "regime_prob_2"}.issubset(out.columns):
        out["regime_p0"] = out["regime_prob_0"]
        out["regime_p1"] = out["regime_prob_1"]
        out["regime_p2"] = out["regime_prob_2"]

    # Prompt-compatible macro alias
    if "us10y" in out.columns and "tnx" not in out.columns:
        out["tnx"] = out["us10y"]

    # Clean up for downstream models: keep NaNs only where unavoidable then drop
    out = out.dropna(axis=0, how="any")
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature engineering (HETRS-NASDAQ).")
    p.add_argument("--in", dest="inp", required=True, help="Input parquet path from data_loader")
    p.add_argument("--out", dest="out", required=True, help="Output parquet path")
    p.add_argument("--ffd-threshold", type=float, default=1e-5)
    p.add_argument("--d-step", type=float, default=0.05)
    p.add_argument("--d-search-ratio", type=float, default=0.6, help="Use first N% of data to choose fracdiff d")
    p.add_argument("--regime-min-obs", type=int, default=252)
    p.add_argument("--regime-refit-interval", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input must be indexed by date (DatetimeIndex).")

    feat = build_features(
        df,
        ffd_threshold=args.ffd_threshold,
        d_step=args.d_step,
        d_search_ratio=args.d_search_ratio,
        regime_min_obs=args.regime_min_obs,
        regime_refit_interval=args.regime_refit_interval,
        random_state=int(args.seed),
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.out)
    print(
        f"[features] saved: {args.out} rows={len(feat)} "
        f"start={feat.index.min().date()} end={feat.index.max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


