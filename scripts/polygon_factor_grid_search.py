#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Polygon aggregates, compute factor variations, and run IC/ICIR grid search."""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

DEFAULT_POLYGON_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"
FACTOR_WINDOWS = {
    'volume_price_corr': [3, 5, 10],
    'rsi': [7, 14],
    'reversal': [2, 3, 5],
    'momentum': [3, 5, 10],
    'liquid_momentum': [3, 5, 10],
    'ivol': [5, 10],
    'vol_ratio': [5, 10],
    'trend_strength': [5, 10],
    'sharpe_momentum': [5, 10],
    'price_ma_dev': [10, 20, 30],
    'trend_r2': [10, 15, 20, 30],
    'alpha_linreg_corr': [5, 10],
    'obv_divergence': [5, 10],
    'near_high': [20, 63],
}
MICROSTRUCTURE_FACTORS = [
    'avg_trade_size',
    'max_effect_21d',
    'gk_vol',
    'price_to_vwap5_dev',
    'intraday_intensity_10d',
]


@dataclass
class PolygonConfig:
    api_key: str
    adjusted: str = 'true'
    sort: str = 'asc'
    limit: int = 50000
    rate_limit_delay: float = 0.25


def _resolve_api_key(explicit: Optional[str]) -> str:
    return explicit or os.environ.get('POLYGON_API_KEY') or DEFAULT_POLYGON_KEY


def _fetch_polygon_aggregates(
    ticker: str,
    start_date: str,
    end_date: str,
    session: requests.Session,
    cfg: PolygonConfig,
) -> pd.DataFrame:
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': cfg.adjusted,
        'sort': cfg.sort,
        'limit': cfg.limit,
        'apiKey': cfg.api_key,
    }
    results: List[Dict] = []
    next_url: Optional[str] = None

    while True:
        url = next_url or base_url
        resp = session.get(url, params=None if next_url else params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get('results') or []
        results.extend(batch)
        next_url = payload.get('next_url')
        time.sleep(cfg.rate_limit_delay)
        if not next_url:
            break

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.normalize()
    df['ticker'] = ticker
    rename_map = {
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume',
        'vw': 'VWAP',
        'n': 'TradeCount',
    }
    missing = [col for col in rename_map if col not in df.columns]
    if missing:
        raise RuntimeError(f"Polygon aggregates missing columns {missing} for {ticker}")
    df = df[list(rename_map.keys()) + ['date', 'ticker']]
    df = df.rename(columns=rename_map)
    return df


def _stack_polygon_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'ticker']).sort_values(['ticker', 'date'])
    combined['date'] = pd.to_datetime(combined['date']).dt.normalize()
    combined['ticker'] = combined['ticker'].astype(str).str.upper()
    combined = combined.set_index(['date', 'ticker']).sort_index()
    return combined


def _ensure_spy_in_universe(tickers: Sequence[str]) -> List[str]:
    ordered = [str(t).upper() for t in tickers]
    if 'SPY' not in ordered:
        ordered.append('SPY')
    return ordered


def _group_transform(series: pd.Series, func) -> pd.Series:
    return series.groupby(level='ticker').transform(func)


def _pct_change(series: pd.Series, window: int, shift: int = 1) -> pd.Series:
    return series.groupby(level='ticker').transform(lambda s: s.pct_change(window).shift(shift))


def _rolling_corr(price_ret: pd.Series, vol_ret: pd.Series, window: int) -> pd.Series:
    tmp = pd.DataFrame({'price_ret': price_ret, 'vol_ret': vol_ret})
    def _corr(group: pd.DataFrame) -> pd.Series:
        return group['price_ret'].rolling(window, min_periods=window).corr(group['vol_ret'])
    return tmp.groupby(level='ticker', group_keys=False).apply(_corr)


def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    def _rsi(s: pd.Series) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
        loss = (-delta).clip(lower=0).rolling(window, min_periods=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return ((rsi - 50) / 50).shift(1)
    return series.groupby(level='ticker').transform(_rsi).fillna(0.0)


def _compute_price_ma_deviation(close: pd.Series, window: int) -> pd.Series:
    def _ma_dev(s: pd.Series) -> pd.Series:
        ma = s.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)
        prev = s.shift(1)
        return (prev / (ma + 1e-10) - 1)
    return close.groupby(level='ticker').transform(_ma_dev).fillna(0.0)


def _compute_trend_r2(close: pd.Series, window: int) -> pd.Series:
    x_base = np.arange(window, dtype=float)
    X = np.column_stack([np.ones(window, dtype=float), x_base])

    def _rolling_r2(arr: np.ndarray) -> float:
        if arr is None or len(arr) != window:
            return 0.0
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            return 0.0
        y = np.log(arr.astype(float))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2_val = 1.0 - ss_res / ss_tot
        return float(max(0.0, min(1.0, r2_val)))

    def _apply(series: pd.Series) -> pd.Series:
        return series.rolling(window, min_periods=window).apply(_rolling_r2, raw=True).fillna(0.0)

    return close.groupby(level='ticker').transform(_apply)


def _compute_alpha_linreg_corr(close: pd.Series, window: int) -> pd.Series:
    timeline = np.arange(window, dtype=float)
    t_mean = timeline.mean()
    t_std = timeline.std(ddof=0)
    if t_std == 0:
        t_std = 1.0

    def _corr(arr: np.ndarray) -> float:
        if arr is None or len(arr) != window:
            return 0.0
        if not np.all(np.isfinite(arr)):
            return 0.0
        price = arr.astype(float)
        p_mean = price.mean()
        p_std = price.std(ddof=0)
        if p_std == 0:
            return 0.0
        cov = np.sum((timeline - t_mean) * (price - p_mean)) / window
        corr = cov / (t_std * p_std)
        return float(max(-1.0, min(1.0, corr)))

    def _apply(series: pd.Series) -> pd.Series:
        return series.rolling(window, min_periods=window).apply(_corr, raw=True).shift(1)

    return close.groupby(level='ticker').transform(_apply).fillna(0.0)


def _compute_obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.groupby(level='ticker').transform(lambda s: np.sign(s.pct_change().fillna(0)))
    obv = (direction * volume).groupby(level='ticker').cumsum()
    return obv.fillna(0.0)


def _compute_near_high(high: pd.Series, close: pd.Series, window: int) -> pd.Series:
    rolling_high = high.groupby(level='ticker').transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 2)).max().shift(1)
    )
    prev_close = close.groupby(level='ticker').transform(lambda s: s.shift(1))
    return (prev_close / (rolling_high + 1e-10) - 1).fillna(0.0)


def _compute_ivol(close: pd.Series, spy_returns: pd.Series, window: int) -> pd.Series:
    stock_ret = close.groupby(level='ticker').transform(lambda s: s.pct_change()).fillna(0.0)
    dates = close.index.get_level_values('date')
    spy_series = pd.Series(spy_returns)
    spy_aligned = spy_series.reindex(dates).fillna(method='ffill').values
    diff = stock_ret.values - spy_aligned

    def _rolling_std(arr: pd.Series) -> pd.Series:
        return arr.rolling(window, min_periods=window).std().shift(1)

    diff_series = pd.Series(diff, index=close.index)
    ivol = diff_series.groupby(level='ticker').transform(_rolling_std)
    return ivol.fillna(0.0)


def _compute_sharpe(close: pd.Series, window: int) -> pd.Series:
    log_close = close.groupby(level='ticker').transform(lambda s: np.log(s))
    log_ret = log_close.groupby(level='ticker').transform(lambda s: s.diff())

    def _apply(series: pd.Series) -> pd.Series:
        mean = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std()
        sharpe = mean / (std.replace(0.0, np.nan))
        return sharpe.shift(1)

    return log_ret.groupby(level='ticker').transform(_apply).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_trend_strength(close: pd.Series, window: int) -> pd.Series:
    def _calc(series: pd.Series) -> pd.Series:
        return series.rolling(window, min_periods=window).apply(
            lambda arr: 0 if np.std(arr) == 0 else (arr[-1] - arr[0]) / (np.std(arr) + 1e-12), raw=False
        ).shift(1)
    return close.groupby(level='ticker').transform(_calc).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_microstructure(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.reset_index()
    df = df.sort_values(['ticker', 'date'])
    frames = []
    for ticker, group in df.groupby('ticker', sort=False):
        g = group.copy()
        g['dollar_volume'] = g['Volume'] * g['VWAP']
        g['avg_trade_size'] = g['dollar_volume'] / g['TradeCount'].replace({0: np.nan})
        ret = g['Close'].pct_change()
        g['max_effect_21d'] = ret.rolling(21, min_periods=3).max()
        log_hl = np.log(g['High'] / g['Low']).replace([np.inf, -np.inf], np.nan)
        log_co = np.log(g['Close'] / g['Open']).replace([np.inf, -np.inf], np.nan)
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        g['gk_vol'] = np.sqrt(gk_var.clip(lower=0))
        rolling_amt = g['dollar_volume'].rolling(5, min_periods=3).sum()
        rolling_vol = g['Volume'].rolling(5, min_periods=3).sum()
        vwap5 = rolling_amt / rolling_vol.replace({0: np.nan})
        g['price_to_vwap5_dev'] = (g['Close'] - vwap5) / vwap5
        price_range = g['High'] - g['Low']
        clv = (2 * g['Close'] - g['High'] - g['Low']) / price_range.replace({0: np.nan})
        raw_intensity = clv * g['Volume']
        g['intraday_intensity_10d'] = raw_intensity.rolling(10, min_periods=5).mean()
        subset = g[['date', 'ticker'] + MICROSTRUCTURE_FACTORS].dropna(how='all', subset=MICROSTRUCTURE_FACTORS)
        frames.append(subset.set_index(['date', 'ticker']))
    if not frames:
        return pd.DataFrame(columns=MICROSTRUCTURE_FACTORS)
    return pd.concat(frames).sort_index()


def _prepare_spy_returns(panel: pd.DataFrame) -> pd.Series:
    if ('SPY' not in panel.index.get_level_values('ticker')):
        raise RuntimeError('SPY data missing; cannot compute ivol factors')
    spy_close = panel.xs('SPY', level='ticker')['Close']
    spy_returns = spy_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return spy_returns


def _compute_all_factors(panel: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    close = panel['Close']
    volume = panel['Volume']
    high = panel['High']
    low = panel['Low']
    open_ = panel['Open']
    vwap = panel.get('VWAP')

    features: Dict[str, pd.Series] = {}

    price_ret = close.groupby(level='ticker').transform(lambda s: s.pct_change())
    vol_ret = volume.groupby(level='ticker').transform(lambda s: s.pct_change())

    for window in FACTOR_WINDOWS['volume_price_corr']:
        corr = _rolling_corr(price_ret, vol_ret, window).replace([np.inf, -np.inf], np.nan)
        features[f'volume_price_corr_{window}d'] = corr

    for window in FACTOR_WINDOWS['rsi']:
        features[f'rsi_{window}'] = _compute_rsi(close, window)

    for window in FACTOR_WINDOWS['reversal']:
        features[f'reversal_{window}d'] = _pct_change(close, window, shift=1).fillna(0.0)

    for window in FACTOR_WINDOWS['momentum']:
        features[f'momentum_{window}d'] = _pct_change(close, window, shift=1).fillna(0.0)

    for window in FACTOR_WINDOWS['liquid_momentum']:
        price_mom = _pct_change(close, window, shift=1).fillna(0.0)
        vol_ma = volume.groupby(level='ticker').transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
        )
        vol_ratio = (volume.groupby(level='ticker').shift(1) / (vol_ma + 1e-10)).replace([np.inf, -np.inf], 1.0)
        features[f'liquid_momentum_{window}d'] = (price_mom * vol_ratio).fillna(0.0)

    spy_returns = _prepare_spy_returns(panel)
    for window in FACTOR_WINDOWS['ivol']:
        features[f'ivol_{window}'] = _compute_ivol(close, spy_returns, window)

    for window in FACTOR_WINDOWS['vol_ratio']:
        vol_ma = volume.groupby(level='ticker').transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)
        )
        vol_prev = volume.groupby(level='ticker').transform(lambda s: s.shift(1))
        features[f'vol_ratio_{window}d'] = (vol_prev / (vol_ma + 1e-10) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for window in FACTOR_WINDOWS['trend_strength']:
        features[f'trend_strength_{window}d'] = _compute_trend_strength(close, window)

    for window in FACTOR_WINDOWS['sharpe_momentum']:
        features[f'sharpe_momentum_{window}d'] = _compute_sharpe(close, window)

    for window in FACTOR_WINDOWS['price_ma_dev']:
        features[f'price_ma{window}_deviation'] = _compute_price_ma_deviation(close, window)

    for window in FACTOR_WINDOWS['trend_r2']:
        features[f'trend_r2_{window}'] = _compute_trend_r2(close, window)

    for window in FACTOR_WINDOWS['alpha_linreg_corr']:
        features[f'alpha_linreg_corr_{window}d'] = _compute_alpha_linreg_corr(close, window)

    obv = _compute_obv_series(close, volume)
    for window in FACTOR_WINDOWS['obv_divergence']:
        obv_mom = obv.groupby(level='ticker').transform(lambda s: s.pct_change(window).shift(1)).fillna(0.0)
        price_mom = _pct_change(close, window, shift=1).fillna(0.0)
        features[f'obv_divergence_{window}d'] = (obv_mom - price_mom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for window in FACTOR_WINDOWS['near_high']:
        features[f'near_{window}d_high'] = _compute_near_high(high, close, window)

    micro_df = _compute_microstructure(panel)
    for col in MICROSTRUCTURE_FACTORS:
        features[col] = micro_df[col] if col in micro_df.columns else pd.Series(dtype=float)

    feature_df = pd.DataFrame(features, index=panel.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    logger.info("Computed feature matrix with shape %s", feature_df.shape)
    return feature_df


def _compute_ic_statistics(factor_series: pd.Series, target_series: pd.Series) -> Tuple[float, float, float, int]:
    merged = pd.DataFrame({'factor': factor_series, 'target': target_series}).dropna()
    if merged.empty:
        return (np.nan, np.nan, np.nan, 0)
    daily_groups = merged.groupby(level='date')
    daily_ics: List[float] = []
    for _, group in daily_groups:
        if len(group) < 5:
            continue
        corr, _ = spearmanr(group['factor'], group['target'])
        if not math.isnan(corr):
            daily_ics.append(float(corr))
    if not daily_ics:
        return (np.nan, np.nan, np.nan, 0)
    arr = np.array(daily_ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else float('nan')
    icir = float(mean_ic / std_ic) if std_ic and not math.isnan(std_ic) and std_ic != 0 else float('nan')
    return (mean_ic, std_ic, icir, len(arr))


def _run_grid_search(
    features: pd.DataFrame,
    target: pd.Series,
    output_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    records = []
    for col in features.columns:
        mean_ic, std_ic, icir, samples = _compute_ic_statistics(features[col], target)
        records.append({
            'factor': col,
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'icir': icir,
            'n_days': samples,
        })
        logger.info("Factor %s -> mean_ic=%.6f icir=%s samples=%d", col, mean_ic, icir, samples)
    results = pd.DataFrame(records)
    results = results.sort_values(by='icir', key=lambda s: s.abs().fillna(0), ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polygon factor IC/ICIR grid search")
    parser.add_argument(
        '--base-data',
        type=str,
        default=r"D:\\trade\\data\\factor_exports\\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5.parquet",
        help='Path to MultiIndex parquet with target columns.',
    )
    parser.add_argument('--output', type=str, default='results/polygon_factor_grid_search_icir.csv')
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--max-tickers', type=int, default=None, help='Limit number of tickers (debug only).')
    parser.add_argument('--extra-lookback', type=int, default=400, help='Extra lookback days for Polygon fetch.')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('polygon_grid_search')

    base_path = Path(args.base_data)
    if not base_path.exists():
        raise FileNotFoundError(f'Base data not found: {base_path}')

    logger.info('Loading base dataset: %s', base_path)
    base_df = pd.read_parquet(base_path)
    if not isinstance(base_df.index, pd.MultiIndex) or 'date' not in base_df.index.names:
        raise RuntimeError('Base parquet must have MultiIndex(date,ticker).')
    base_df = base_df.sort_index()
    target = base_df['target']
    base_index = base_df.index

    unique_dates = base_index.get_level_values('date')
    min_date = unique_dates.min() - pd.Timedelta(days=args.extra_lookback)
    max_date = unique_dates.max() + pd.Timedelta(days=5)
    tickers = base_index.get_level_values('ticker').unique().tolist()
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    tickers = _ensure_spy_in_universe(tickers)
    logger.info('Universe: %d tickers, %d dates', len(tickers), unique_dates.nunique())
    logger.info('Fetching Polygon aggregates from %s to %s', min_date.date(), max_date.date())

    cfg = PolygonConfig(api_key=_resolve_api_key(args.api_key))
    session = requests.Session()
    frames = []
    for idx, ticker in enumerate(tickers, start=1):
        try:
            data = _fetch_polygon_aggregates(ticker, min_date.date().isoformat(), max_date.date().isoformat(), session, cfg)
        except Exception as exc:
            logger.error('Failed to download %s: %s', ticker, exc)
            continue
        if data.empty:
            logger.warning('No data returned for %s', ticker)
            continue
        frames.append(data)
        if idx % 50 == 0:
            logger.info('Fetched %d/%d tickers', idx, len(tickers))

    panel = _stack_polygon_frames(frames)
    if panel.empty:
        raise RuntimeError('No Polygon data fetched.')
    logger.info('Polygon panel shape: %s', panel.shape)

    missing_cols = {'Open', 'High', 'Low', 'Close', 'Volume'} - set(panel.columns)
    if missing_cols:
        raise RuntimeError(f'Panel missing columns: {missing_cols}')

    features = _compute_all_factors(panel, logger)
    features = features.reindex(base_index)

    results = _run_grid_search(features, target, Path(args.output), logger)
    logger.info('Saved IC/ICIR results to %s', args.output)

    micro_results = results[results['factor'].isin(MICROSTRUCTURE_FACTORS)]
    logger.info('Microstructure factor IC/ICIR:\n%s', micro_results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
