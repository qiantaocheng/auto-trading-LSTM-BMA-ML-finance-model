#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import sys
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class BacktestWindowResult:
    as_of_date: pd.Timestamp
    target_date: pd.Timestamp
    actual_base_date: pd.Timestamp
    actual_target_date: pd.Timestamp
    hit_rate: float
    top_decile_hit_rate: float
    top_k_return: float
    top_k_hit_rate: float
    mean_return: float
    sample_size: int
    kronos_pass_count: int | None = None
    kronos_pass_hit_rate: float | None = None
    kronos_pass_avg_return: float | None = None


def configure_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / 'autotrader'))
    sys.path.insert(0, str(repo_root / 'bma_models'))


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def pick_stock_pool(min_size: int, substring: str | None, limit: int | None) -> list[str]:
    from autotrader.stock_pool_manager import StockPoolManager

    manager = StockPoolManager()
    pools = manager.get_all_pools()
    candidates: list[tuple[int, str, list[str]]] = []
    for pool in pools:
        tickers = json.loads(pool['tickers']) if isinstance(pool['tickers'], str) else pool['tickers']
        if not isinstance(tickers, list):
            continue
        if len(tickers) < min_size:
            continue
        name = pool.get('pool_name', '') or ''
        if substring and substring not in name:
            continue
        candidates.append((len(tickers), name, tickers))
    if not candidates:
        raise RuntimeError('No stock pool satisfies selection criteria')
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0]
    tickers = selected[2]
    if limit:
        tickers = tickers[:limit]
    logging.info(
        'Selected stock pool %s with %d tickers (using %d)',
        selected[1],
        selected[0],
        len(tickers)
    )
    return tickers


def build_eval_schedule(start: str, end: str, frequency: str) -> list[pd.Timestamp]:
    date_range = pd.date_range(start=start, end=end, freq=frequency)
    return [pd.Timestamp(d).normalize() for d in date_range]


def fetch_close_panel(engine, tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    raw_df = engine.fetch_market_data(
        symbols=tickers,
        use_optimized_downloader=False,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    if raw_df.empty:
        raise RuntimeError('Price download returned empty frame')
    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    df['ticker'] = df['ticker'].astype(str).str.upper()
    close_panel = df.pivot_table(index='date', columns='ticker', values='Close', aggfunc='last').sort_index()
    return close_panel


def compute_actual_returns(
    close_panel: pd.DataFrame,
    base_date: pd.Timestamp,
    horizon_days: int,
    calendar_target_date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    if close_panel.empty:
        raise RuntimeError('Close panel is empty')
    sorted_dates = close_panel.index.sort_values()
    base_candidates = sorted_dates[sorted_dates <= base_date]
    if len(base_candidates) == 0:
        raise RuntimeError('No price history on or before base date')
    actual_base_date = pd.Timestamp(base_candidates[-1])
    base_pos_candidates = (sorted_dates == actual_base_date).nonzero()[0]
    if len(base_pos_candidates) == 0:
        base_pos = sorted_dates.get_indexer([actual_base_date])[0]
        if base_pos < 0:
            base_pos = max(0, sorted_dates.searchsorted(actual_base_date))
    else:
        base_pos = int(base_pos_candidates[0])
    target_pos = base_pos + horizon_days
    if target_pos >= len(sorted_dates):
        target_pos = len(sorted_dates) - 1
        logging.warning(
            'Insufficient forward data to reach T+%d from %s; using last available %s',
            horizon_days,
            actual_base_date.date(),
            sorted_dates[target_pos].date(),
        )
    actual_target_date = pd.Timestamp(sorted_dates[target_pos])
    if calendar_target_date is not None and actual_target_date != calendar_target_date:
        logging.debug(
            'Adjusted prediction target from %s to trading day %s for exact T+%d alignment',
            calendar_target_date.date(),
            actual_target_date.date(),
            horizon_days,
        )
    base_prices = close_panel.loc[actual_base_date]
    target_prices = close_panel.loc[actual_target_date]
    returns = (target_prices - base_prices) / base_prices
    return returns, actual_base_date, actual_target_date


def run_backtest(args: argparse.Namespace) -> None:
    UltraEnhancedModule = importlib.import_module('bma_models.量化模型_bma_ultra_enhanced')
    UltraEnhancedQuantitativeModel = UltraEnhancedModule.UltraEnhancedQuantitativeModel
    from bma_models.simple_25_factor_engine import Simple17FactorEngine
    tickers = pick_stock_pool(
        min_size=args.min_pool_size,
        substring=args.pool_substring,
        limit=args.max_tickers
    )
    eval_dates = build_eval_schedule(args.backtest_start, args.backtest_end, args.frequency)
    if args.limit_windows:
        eval_dates = eval_dates[:args.limit_windows]
    if not eval_dates:
        raise RuntimeError('Evaluation schedule is empty')

    output_root = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backtest_dir = output_root / f'bma_backtest_{timestamp}'
    backtest_dir.mkdir(parents=True, exist_ok=True)

    engine = Simple17FactorEngine(lookback_days=args.price_lookback, horizon=args.horizon_days)

    training_runs = 1
    data_fetch_count = 0
    inference_count = 0
    price_fetch_count = 0

    feature_cache_df: pd.DataFrame | None = None
    feature_cache_path = backtest_dir / 'factor_cache.parquet'

    model = UltraEnhancedQuantitativeModel()
    if hasattr(model, 'use_kronos_validation'):
        model.use_kronos_validation = True
    else:
        setattr(model, 'use_kronos_validation', True)
    if hasattr(model, 'kronos_model'):
        model.kronos_model = None

    first_as_of = eval_dates[0]
    initial_train_start = (first_as_of - timedelta(days=args.train_window_days)).strftime('%Y-%m-%d')
    initial_train_end = first_as_of.strftime('%Y-%m-%d')

    logging.info('Training base model once using window %s -> %s', initial_train_start, initial_train_end)
    try:
        training_results = model.run_complete_analysis(
            tickers=tickers,
            start_date=initial_train_start,
            end_date=initial_train_end,
            top_n=len(tickers)
        )
    except Exception:
        logging.exception('run_complete_analysis failed during initial training for %s', first_as_of.date())
        return

    if not training_results.get('success', False):
        logging.error('Initial training unsuccessful (%s); aborting backtest', training_results.get('error'))
        return

    snapshot_id = getattr(model, 'active_snapshot_id', None)
    if snapshot_id:
        logging.info('Snapshot %s will be reused for inference windows', snapshot_id)
    else:
        logging.error('No model snapshot generated during training; cannot proceed with inference windows')
        return

    window_rows: list[BacktestWindowResult] = []
    group_rows: list[dict] = []
    prediction_rows: list[dict] = []

    window_specs: list[tuple[pd.Timestamp, str, str, dict | None]] = []
    window_specs.append((first_as_of, initial_train_start, initial_train_end, training_results))
    for as_of in eval_dates[1:]:
        window_train_start = (as_of - timedelta(days=args.train_window_days)).strftime('%Y-%m-%d')
        window_train_end = as_of.strftime('%Y-%m-%d')
        window_specs.append((as_of, window_train_start, window_train_end, None))

    total_windows = len(window_specs)

    global_train_start = min(spec[1] for spec in window_specs)
    global_train_end = max(spec[2] for spec in window_specs)

    def _ensure_feature_cache() -> bool:
        nonlocal feature_cache_df, data_fetch_count
        if feature_cache_df is not None:
            return True
        logging.info('Building factor cache covering %s -> %s', global_train_start, global_train_end)
        try:
            feature_cache_df = model.get_data_and_features(tickers, global_train_start, global_train_end)
            data_fetch_count += 1
        except Exception:
            logging.exception('Failed to build factor cache')
            feature_cache_df = None
            return False
        if feature_cache_df is None or len(feature_cache_df) == 0:
            logging.error('Factor cache is empty after build attempt')
            feature_cache_df = None
            return False
        try:
            if not isinstance(feature_cache_df.index, pd.MultiIndex):
                if 'date' in feature_cache_df.columns and 'ticker' in feature_cache_df.columns:
                    feature_cache_df['date'] = pd.to_datetime(feature_cache_df['date']).dt.tz_localize(None).dt.normalize()
                    feature_cache_df['ticker'] = feature_cache_df['ticker'].astype(str).str.upper()
                    feature_cache_df.set_index(['date', 'ticker'], inplace=True)
        except Exception:
            logging.exception('Failed to standardize factor cache index')
        try:
            feature_cache_df.to_parquet(feature_cache_path)
            logging.info('Factor cache saved to %s', feature_cache_path)
        except Exception:
            logging.warning('Unable to persist factor cache to %s', feature_cache_path)
        return True

    def _load_feature_slice(start_date: str, end_date: str) -> pd.DataFrame | None:
        if not _ensure_feature_cache():
            return None
        if feature_cache_df is None or feature_cache_df.empty:
            return None
        try:
            if isinstance(feature_cache_df.index, pd.MultiIndex):
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                slice_df = feature_cache_df.loc[(slice(start_ts, end_ts), slice(None)), :]
            else:
                df = feature_cache_df
                if 'date' not in df.columns:
                    logging.error('Factor cache missing date column; cannot slice')
                    return None
                mask = (pd.to_datetime(df['date']) >= pd.Timestamp(start_date)) & (pd.to_datetime(df['date']) <= pd.Timestamp(end_date))
                slice_df = df.loc[mask]
            if isinstance(slice_df, pd.DataFrame) and not slice_df.empty:
                return slice_df.copy()
            logging.warning('Factor slice %s -> %s is empty', start_date, end_date)
            return None
        except Exception:
            logging.exception('Failed to extract factor slice %s -> %s', start_date, end_date)
            return None

    for idx, (as_of, window_train_start, window_train_end, cached_results) in enumerate(window_specs, 1):
        logging.info('=== Window %d/%d | as_of=%s ===', idx, total_windows, as_of.date())
        if cached_results is not None:
            results = cached_results
        else:
            feature_data = _load_feature_slice(window_train_start, window_train_end)
            if feature_data is None or len(feature_data) == 0:
                logging.warning('No cached feature data available for %s', as_of.date())
                continue
            feature_for_inference = feature_data.reset_index() if isinstance(feature_data.index, pd.MultiIndex) else feature_data
            try:
                # Pass as_of date to prevent Kronos data leakage
                results = model.predict_with_snapshot(
                    feature_for_inference,
                    snapshot_id=snapshot_id,
                    universe_tickers=tickers,
                    as_of_date=as_of  # Explicitly pass backtest date for time alignment
                )
                inference_count += 1
            except Exception:
                logging.exception('Snapshot inference failed for %s', as_of.date())
                continue
            if not results.get('success', False):
                logging.warning('Snapshot inference unsuccessful for %s (%s)', as_of.date(), results.get('error'))
                continue

        pred_series = results.get('predictions')
        if pred_series is None or len(pred_series) == 0:
            logging.warning('No predictions returned for %s', as_of.date())
            continue
        pred_series = pred_series.copy()

        base_dates = pred_series.index.get_level_values('date')
        unique_base_dates = base_dates.unique()
        if len(unique_base_dates) != 1:
            logging.warning('Unexpected multiple base dates: %s', list(unique_base_dates))
        base_date = pd.Timestamp(unique_base_dates[-1])

        final_df = getattr(model, '_last_final_predictions_df', pd.DataFrame())
        if final_df.empty or 'date' not in final_df.columns:
            target_date = base_date + timedelta(days=args.horizon_days)
        else:
            target_date = pd.to_datetime(final_df.iloc[0]['date']).tz_localize(None).normalize()

        pred_df = pred_series.to_frame('pred_score').reset_index()
        pred_df['pred_rank_pct'] = pred_df['pred_score'].rank(pct=True, method='average')

        try:
            panel = fetch_close_panel(
                engine,
                pred_df['ticker'].tolist(),
                start_date=base_date - timedelta(days=args.price_lead_days),
                end_date=target_date + timedelta(days=args.price_lag_days)
            )
            price_fetch_count += 1
        except Exception:
            logging.exception('Price download failed for window %s', as_of.date())
            continue

        try:
            actual_returns, actual_base_date, actual_target_date = compute_actual_returns(
                panel,
                base_date,
                args.horizon_days,
                target_date
            )
        except Exception:
            logging.exception('Failed to compute actual returns for %s', as_of.date())
            continue

        merged = pred_df.merge(actual_returns.rename('actual_return'), on='ticker', how='left')
        merged['actual_return'].fillna(0.0, inplace=True)
        merged['is_up'] = merged['actual_return'] > 0
        merged['pred_up'] = merged['pred_score'] > 0
        merged['correct'] = merged['is_up'] == merged['pred_up']
        merged['as_of_date'] = base_date
        merged['target_date'] = actual_target_date

        top_decile_cut = merged['pred_rank_pct'].quantile(0.9)
        top_decile = merged[merged['pred_rank_pct'] >= top_decile_cut]
        hit_rate = merged['correct'].mean()
        top_decile_hit_rate = top_decile['correct'].mean() if not top_decile.empty else np.nan
        top_k = merged.nlargest(args.top_k, 'pred_score')
        top_k_return = top_k['actual_return'].mean() if not top_k.empty else np.nan
        mean_return = merged['actual_return'].mean()

        ordered = merged.sort_values('pred_score', ascending=False).reset_index(drop=True)

        def _record_group(name: str, df: pd.DataFrame) -> None:
            if df is None or df.empty:
                return
            group_rows.append({
                'as_of_date': base_date,
                'group': name,
                'size': int(len(df)),
                'hit_rate': float(df['correct'].mean()) if len(df) > 0 else float('nan'),
                'avg_return': float(df['actual_return'].mean()) if len(df) > 0 else float('nan')
            })

        _record_group('top_0_10', ordered.head(10))
        _record_group('top_10_20', ordered.iloc[10:20])
        _record_group('top_20_30', ordered.iloc[20:30])
        _record_group('bottom_10', ordered.tail(10))
        _record_group('bottom_20', ordered.tail(20))

        kronos_pass_metrics = (None, None, None)
        kronos_pass_records = None
        kronos_source = results.get('kronos_top35') if isinstance(results, dict) else None
        kronos_merge = None
        if kronos_source is not None:
            kronos_df = pd.DataFrame(kronos_source) if not isinstance(kronos_source, pd.DataFrame) else kronos_source.copy()
            if not kronos_df.empty and 'ticker' in kronos_df.columns:
                kronos_df['ticker'] = kronos_df['ticker'].astype(str).str.upper()
                kronos_cols = [col for col in ['kronos_pass', 'kronos_t3_return', 'bma_rank', 'rank'] if col in kronos_df.columns]
                kronos_merge = kronos_df[['ticker'] + kronos_cols].merge(merged, on='ticker', how='inner')
        if kronos_merge is None or kronos_merge.empty:
            kronos_merge = ordered.head(min(35, len(ordered))).copy()
            if not kronos_merge.empty:
                kronos_merge['kronos_pass'] = kronos_merge['pred_score'] > 0
                kronos_merge['kronos_t3_return'] = np.nan
        if kronos_merge is not None and not kronos_merge.empty:
            _record_group('kronos_top35', kronos_merge)
            passed = kronos_merge[kronos_merge.get('kronos_pass', False) == True].copy()
            _record_group('kronos_pass', passed)
            kronos_merge.to_csv(backtest_dir / f"kronos_top35_{base_date.strftime('%Y%m%d')}.csv", index=False)
            if not passed.empty:
                kronos_pass_metrics = (len(passed), passed['correct'].mean(), passed['actual_return'].mean())
                passed_sorted = passed.sort_values('pred_score', ascending=False).reset_index(drop=True)
                passed_sorted['kronos_rank'] = passed_sorted.index + 1
                kronos_pass_records = passed_sorted
        top_k_hit_rate = top_k['correct'].mean() if not top_k.empty else np.nan
        window_rows.append(BacktestWindowResult(
            as_of_date=base_date,
            target_date=target_date,
            actual_base_date=actual_base_date,
            actual_target_date=actual_target_date,
            hit_rate=float(hit_rate) if not np.isnan(hit_rate) else np.nan,
            top_decile_hit_rate=float(top_decile_hit_rate) if not np.isnan(top_decile_hit_rate) else np.nan,
            top_k_return=float(top_k_return) if not np.isnan(top_k_return) else np.nan,
            top_k_hit_rate=float(top_k_hit_rate) if not np.isnan(top_k_hit_rate) else np.nan,
            mean_return=float(mean_return) if not np.isnan(mean_return) else np.nan,
            sample_size=len(merged),
            kronos_pass_count=kronos_pass_metrics[0] if kronos_pass_metrics[0] is not None else None,
            kronos_pass_hit_rate=float(kronos_pass_metrics[1]) if kronos_pass_metrics[1] is not None else None,
            kronos_pass_avg_return=float(kronos_pass_metrics[2]) if kronos_pass_metrics[2] is not None else None
        ))

        prediction_rows.extend(merged.to_dict('records'))

        window_path = backtest_dir / f"predictions_{base_date.strftime('%Y%m%d')}.csv"
        merged.to_csv(window_path, index=False)

        if kronos_pass_records is not None:
            kronos_path = backtest_dir / f"kronos_pass_{base_date.strftime('%Y%m%d')}.csv"
            kronos_pass_records.to_csv(kronos_path, index=False)

    if not window_rows:
        logging.error('No successful backtest windows recorded')
        return

    windows_df = pd.DataFrame([row.__dict__ for row in window_rows])
    predictions_df = pd.DataFrame(prediction_rows)

    windows_path = backtest_dir / 'window_metrics.csv'
    predictions_path = backtest_dir / 'prediction_details.csv'
    windows_df.to_csv(windows_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    plot_accuracy(windows_df, backtest_dir)
    plot_calibration(predictions_df, backtest_dir)
    plot_topk_returns(windows_df, backtest_dir, args.top_k)

    if group_rows:
        group_df = pd.DataFrame(group_rows)
        group_df.to_csv(backtest_dir / 'group_metrics.csv', index=False)
        group_summary = group_df.groupby('group', as_index=False)[['hit_rate', 'avg_return']].mean()
        group_summary.to_csv(backtest_dir / 'group_summary.csv', index=False)

        logging.info('Average metrics by group across windows:')
        for _, row in group_summary.iterrows():
            avg_return = row.get('avg_return')
            hit_rate_val = row.get('hit_rate')
            avg_return_str = f"{avg_return:.4f}" if pd.notna(avg_return) else 'nan'
            hit_rate_str = f"{hit_rate_val * 100:.2f}%" if pd.notna(hit_rate_val) else 'nan'
            logging.info('  %s -> avg_return=%s | hit_rate=%s', row['group'], avg_return_str, hit_rate_str)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(group_summary['group'], group_summary['avg_return'])
        ax.set_title('Average T+10 Return by Group')
        ax.set_ylabel('Average return')
        ax.set_xlabel('Group')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        fig.tight_layout()
        fig.savefig(backtest_dir / 'group_avg_return.png', dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(group_summary['group'], group_summary['hit_rate'])
        ax.set_title('Hit Rate by Group')
        ax.set_ylabel('Hit rate')
        ax.set_xlabel('Group')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        fig.tight_layout()
        fig.savefig(backtest_dir / 'group_hit_rate.png', dpi=200)
        plt.close(fig)
    else:
        group_df = pd.DataFrame()

    logging.info('Backtest stats: training_runs=%d | factor_fetches=%d | snapshot_inferences=%d | price_fetches=%d', training_runs, data_fetch_count, inference_count, price_fetch_count)
    if feature_cache_df is not None and not feature_cache_df.empty:
        logging.info('Factor cache path: %s', feature_cache_path)

    logging.info('Backtest completed. Results stored under %s', backtest_dir)


def plot_accuracy(windows_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(windows_df['as_of_date'], windows_df['hit_rate'], marker='o', label='Overall hit rate')
    ax.plot(windows_df['as_of_date'], windows_df['top_decile_hit_rate'], marker='s', label='Top decile hit rate')
    ax.set_title('T+10 Directional Accuracy by Window')
    ax.set_xlabel('As-of date')
    ax.set_ylabel('Hit rate')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_by_window.png', dpi=200)
    plt.close(fig)


def plot_calibration(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    if predictions_df.empty:
        return
    predictions_df['decile'] = (predictions_df['pred_rank_pct'] * 10).clip(0, 9.999).astype(int)
    hit_rates = predictions_df.groupby('decile')['is_up'].mean().sort_index()
    avg_returns = predictions_df.groupby('decile')['actual_return'].mean().sort_index()
    calibration = pd.DataFrame({'hit_rate': hit_rates, 'avg_return': avg_returns})
    calibration.to_csv(output_dir / 'calibration_deciles.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    hit_rates.plot(kind='bar', ax=ax)
    ax.set_title('Calibration: Actual Up Probability by Prediction Decile')
    ax.set_xlabel('Prediction decile (higher = more bullish)')
    ax.set_ylabel('Observed up probability')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'calibration_deciles.png', dpi=200)
    plt.close(fig)


def plot_topk_returns(windows_df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    if windows_df.empty:
        return
    cumulative = (1 + windows_df['top_k_return'].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(windows_df['as_of_date'], cumulative, marker='o')
    ax.set_title(f'Cumulative Return: Equal-Weight Long Top {top_k}')
    ax.set_xlabel('As-of date')
    ax.set_ylabel('Growth of $1')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / 'cumulative_return_topk.png', dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BMA Ultra rolling T+10 backtest')
    parser.add_argument('--backtest-start', default='2023-01-01')
    parser.add_argument('--backtest-end', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--frequency', default='BM', help='Evaluation frequency (pandas offset alias)')
    parser.add_argument('--train-window-days', type=int, default=365 * 3)
    parser.add_argument('--horizon-days', type=int, default=10)
    parser.add_argument('--min-pool-size', type=int, default=2000)
    parser.add_argument('--pool-substring', default=None, help='Substring to match pool name (optional)')
    parser.add_argument('--max-tickers', type=int, default=None, help='Optional cap on number of tickers')
    parser.add_argument('--limit-windows', type=int, default=None, help='Limit number of evaluation windows for pilot runs')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--price-lookback', type=int, default=400, help='Lookback days for price fetch engine')
    parser.add_argument('--price-lead-days', type=int, default=10)
    parser.add_argument('--price-lag-days', type=int, default=10)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    configure_paths()
    args = parse_args()
    configure_logging(args.verbose)
    import json  # Lazy import so StockPoolManager pick-up works without polluting module graph
    globals()['json'] = json
    run_backtest(args)


if __name__ == '__main__':
    main()








