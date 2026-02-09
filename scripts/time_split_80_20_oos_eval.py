#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90/10 time-split train/test on MultiIndex(date,ticker) factors.

Design:
  - Split by unique dates (sorted)
  - Train on first 90% dates, BUT purge a gap = horizon_days to avoid label leakage
    (because target at date t uses forward returns through t+horizon_days)
  - Evaluate on last 10% dates with STANDALONE prediction loop (NO ComprehensiveModelBacktest dependency)
  - Daily rebalancing: one prediction per trading day (overlapping observations)
  - HAC corrections: Newey-West (lag≥10) or Hansen-Hodrick standard errors
  - Explicit disclosure: "基于重叠观测 (overlapping observations)"
  - Report Top-20 expected return for ridge_stacking on the test window
  - Produce per-period and cumulative plots vs NASDAQ proxy (QQQ via yfinance fallback)

Outputs (under output-dir/run_<ts>/):
  - snapshot_id.txt
  - report_df.csv
  - ridge_top20_timeseries.csv
  - top20_vs_qqq.png
  - top20_vs_qqq_cumulative.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# HAC correction functions (standalone, no ComprehensiveModelBacktest dependency)


# ==================== FEATURE ALIGNMENT FIX ====================
def align_test_features_with_model(X_test: pd.DataFrame, model, model_name: str, logger) -> pd.DataFrame:
    """
    确保测试数据特征与模型训练时使用的特征完全一致

    Args:
        X_test: 测试数据特征
        model: 训练好的模型
        model_name: 模型名称（用于日志）
        logger: 日志记录器

    Returns:
        对齐后的测试数据特征
    """
    # 获取训练时的特征名
    train_features = None

    # 不同模型存储特征名的属性不同
    if hasattr(model, 'feature_names_in_'):
        train_features = list(model.feature_names_in_)
    elif hasattr(model, 'feature_name_'):  # XGBoost
        train_features = list(model.feature_name_)
    elif hasattr(model, '_Booster') and hasattr(model._Booster, 'feature_names'):  # XGBoost另一种方式
        train_features = list(model._Booster.feature_names)
    elif hasattr(model, 'feature_names'):
        train_features = list(model.feature_names)

    if train_features is None:
        logger.warning(f"[{model_name}] 无法获取训练特征名，使用原始特征")
        return X_test

    # 检查缺失和额外的特征
    test_features = set(X_test.columns)
    train_features_set = set(train_features)

    missing_features = train_features_set - test_features
    extra_features = test_features - train_features_set

    if missing_features:
        logger.warning(f"[{model_name}] 测试数据缺失特征（将填充0）: {missing_features}")
        for feat in missing_features:
            X_test[feat] = 0.0

    if extra_features:
        # Don't log - just silently align features to match training
        # Extra features are not passed to model, but original data is not modified
        pass

    # 选择训练特征并保持顺序
    try:
        X_aligned = X_test[train_features].copy()
        logger.info(f"[{model_name}] 特征对齐完成: {len(train_features)}个特征")
        return X_aligned
    except KeyError as e:
        logger.error(f"[{model_name}] 特征对齐失败: {e}")
        return X_test
def _limit_predictions_to_top_k_per_date(predictions_df: pd.DataFrame, top_k: int = 40, score_col: str = 'prediction', date_col: str = 'date') -> pd.DataFrame:
    """Return dataframe restricted to top_k rows per date based on score_col."""
    if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
        return predictions_df
    if score_col not in predictions_df.columns or date_col not in predictions_df.columns:
        return predictions_df
    # Sort by date ascending then score descending
    df_sorted = predictions_df.sort_values([date_col, score_col], ascending=[True, False]).copy()
    limited = df_sorted.groupby(date_col, group_keys=False).head(top_k)
    if limited.empty:
        return predictions_df
    return limited

# ==================== END FEATURE ALIGNMENT FIX ====================

def calculate_newey_west_hac_ic(
    predictions_df: pd.DataFrame,
    lag: int,
    use_rank: bool = False
) -> Dict[str, float]:
    """
    Calculate IC with Newey-West HAC standard errors.
    
    🔥 关键修复：先按日聚合IC，再对日度IC序列做HAC
    
    正确流程：
    1. 按日期分组
    2. 对每一天计算IC（横截面correlation）
    3. 得到日度IC序列（长度≈259天）
    4. 对日度IC序列做Newey-West HAC
    
    References:
    - Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
      heteroskedasticity and autocorrelation consistent covariance matrix.
      Econometrica, 55(3), 703-708.
    
    Args:
        predictions_df: DataFrame with columns [date, ticker, prediction, actual]
        lag: Lag order (should be ≈horizon or 2*horizon for overlapping returns)
        use_rank: If True, use rank correlation (Rank IC), else use Pearson correlation (IC)
    
    Returns:
        Dict with IC, t-stat, SE, p-value, note
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for HAC corrections. Install: pip install statsmodels")
    
    # 🔥 步骤1: 按日期分组，计算每天的IC（横截面correlation）
    # 确保日期已排序（HAC需要时间序列顺序）
    predictions_df_sorted = predictions_df.sort_values('date').copy()
    
    daily_ics = []
    daily_dates = []
    
    for date, date_group in predictions_df_sorted.groupby('date'):
        # 提取当天的预测和实际值
        date_pred = date_group['prediction'].values
        date_actual = date_group['actual'].values
        
        # 移除NaN
        valid_mask = ~(np.isnan(date_pred) | np.isnan(date_actual))
        date_pred_clean = date_pred[valid_mask]
        date_actual_clean = date_actual[valid_mask]
        
        # 需要至少2个样本才能计算correlation
        if len(date_pred_clean) < 2:
            continue
        
        # 计算当天的IC（横截面correlation）
        if use_rank:
            # Rank IC: 使用rank correlation
            from scipy.stats import spearmanr
            ic_day, _ = spearmanr(date_pred_clean, date_actual_clean)
        else:
            # IC: 使用Pearson correlation
            ic_day = float(np.corrcoef(date_pred_clean, date_actual_clean)[0, 1])
        
        if not (np.isnan(ic_day) or np.isinf(ic_day)):
            daily_ics.append(ic_day)
            daily_dates.append(date)
    
    # 🔥 步骤2: 检查是否有足够的日度IC
    if len(daily_ics) < 10:
        return {
            'IC': np.nan,
            'IC_pvalue': np.nan,
            'IC_tstat': np.nan,
            'IC_se_hac': np.nan,
            'note': f'Insufficient daily ICs ({len(daily_ics)} < 10)'
        }
    
    # 🔥 步骤3: 计算平均IC
    daily_ics_array = np.array(daily_ics)
    mean_ic = float(np.mean(daily_ics_array))
    
    # 🔥 步骤4: 对日度IC序列做Newey-West HAC
    # IC序列的均值是我们要估计的参数
    # HAC标准误用于IC均值的统计推断
    # 使用常数项回归估计IC均值（OLS估计均值）
    X = np.ones((len(daily_ics_array), 1))
    y = daily_ics_array
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    
    # 标准误是IC均值的标准误
    ic_se = float(np.sqrt(results.cov_params()[0, 0]))
    
    # t-stat = mean_ic / se
    ic_tstat = mean_ic / ic_se if ic_se > 0 else np.nan
    
    # p-value (two-tailed)
    from scipy.stats import t
    if not np.isnan(ic_tstat):
        df = len(daily_ics_array) - 1  # degrees of freedom
        ic_pvalue = 2 * (1 - t.cdf(abs(ic_tstat), df))
    else:
        ic_pvalue = np.nan
    
    return {
        'IC': mean_ic,
        'IC_pvalue': ic_pvalue,
        'IC_tstat': ic_tstat,
        'IC_se_hac': ic_se,
        'note': f'Newey-West HAC on daily ICs (n_days={len(daily_ics)}, lag={lag})'
    }


def calculate_hansen_hodrick_se_ic(
    predictions_df: pd.DataFrame,
    horizon: int,
    use_rank: bool = False
) -> Dict[str, float]:
    """
    Calculate IC with Hansen-Hodrick standard errors.
    
    🔥 关键修复：先按日聚合IC，再对日度IC序列做HAC
    
    正确流程：
    1. 按日期分组
    2. 对每一天计算IC（横截面correlation）
    3. 得到日度IC序列（长度≈259天）
    4. 对日度IC序列做Hansen-Hodrick HAC (lag = horizon - 1)
    
    References:
    - Hansen, L. P., & Hodrick, R. J. (1980). Forward exchange rates as optimal
      predictors of future spot rates: An econometric analysis.
      Journal of Political Economy, 88(5), 829-853.
    
    For h-period overlapping returns, use lag = h-1.
    
    Args:
        predictions_df: DataFrame with columns [date, ticker, prediction, actual]
        horizon: Prediction horizon (days)
        use_rank: If True, use rank correlation (Rank IC), else use Pearson correlation (IC)
    
    Returns:
        Dict with IC, t-stat, SE, p-value, note
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for HAC corrections. Install: pip install statsmodels")
    
    # 🔥 步骤1: 按日期分组，计算每天的IC（横截面correlation）
    # 确保日期已排序（HAC需要时间序列顺序）
    predictions_df_sorted = predictions_df.sort_values('date').copy()
    
    daily_ics = []
    daily_dates = []
    
    for date, date_group in predictions_df_sorted.groupby('date'):
        # 提取当天的预测和实际值
        date_pred = date_group['prediction'].values
        date_actual = date_group['actual'].values
        
        # 移除NaN
        valid_mask = ~(np.isnan(date_pred) | np.isnan(date_actual))
        date_pred_clean = date_pred[valid_mask]
        date_actual_clean = date_actual[valid_mask]
        
        # 需要至少2个样本才能计算correlation
        if len(date_pred_clean) < 2:
            continue
        
        # 计算当天的IC（横截面correlation）
        if use_rank:
            # Rank IC: 使用rank correlation
            from scipy.stats import spearmanr
            ic_day, _ = spearmanr(date_pred_clean, date_actual_clean)
        else:
            # IC: 使用Pearson correlation
            ic_day = float(np.corrcoef(date_pred_clean, date_actual_clean)[0, 1])
        
        if not (np.isnan(ic_day) or np.isinf(ic_day)):
            daily_ics.append(ic_day)
            daily_dates.append(date)
    
    # 🔥 步骤2: 检查是否有足够的日度IC
    if len(daily_ics) < 10:
        return {
            'IC': np.nan,
            'IC_pvalue': np.nan,
            'IC_tstat': np.nan,
            'IC_se_hac': np.nan,
            'note': f'Insufficient daily ICs ({len(daily_ics)} < 10)'
        }
    
    # 🔥 步骤3: 计算平均IC
    daily_ics_array = np.array(daily_ics)
    mean_ic = float(np.mean(daily_ics_array))
    
    # 🔥 步骤4: 对日度IC序列做Hansen-Hodrick HAC
    # For h-period overlapping returns, use lag = h-1
    lag = max(horizon - 1, 1)
    
    # IC序列的均值是我们要估计的参数
    # HAC标准误用于IC均值的统计推断
    # 使用常数项回归估计IC均值（OLS估计均值）
    X = np.ones((len(daily_ics_array), 1))
    y = daily_ics_array
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    
    # 标准误是IC均值的标准误
    ic_se = float(np.sqrt(results.cov_params()[0, 0]))
    
    # t-stat = mean_ic / se
    ic_tstat = mean_ic / ic_se if ic_se > 0 else np.nan
    
    # p-value (two-tailed)
    from scipy.stats import t
    if not np.isnan(ic_tstat):
        df = len(daily_ics_array) - 1  # degrees of freedom
        ic_pvalue = 2 * (1 - t.cdf(abs(ic_tstat), df))
    else:
        ic_pvalue = np.nan
    
    return {
        'IC': mean_ic,
        'IC_pvalue': ic_pvalue,
        'IC_tstat': ic_tstat,
        'IC_se_hac': ic_se,
        'note': f'Hansen-Hodrick SE on daily ICs (n_days={len(daily_ics)}, lag={lag} for {horizon}-day horizon)'
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet", help="Parquet shards dir or a single parquet file.")
    p.add_argument("--data-dir", type=str, default="data/factor_exports")
    p.add_argument("--data-file", type=str, default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet")
    p.add_argument("--horizon-days", type=int, default=5)
    p.add_argument("--split", type=float, default=0.8, help="Train split fraction by time (default 0.8 for 80/20).")
    p.add_argument("--model", type=str, default="catboost", help="Primary model for legacy single-model TopN plot exports.")
    p.add_argument("--models", nargs="+", default=["catboost", "lambdarank", "ridge_stacking"], help="If provided, export TopN OOS plots/metrics for these models (e.g. elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking). Default: catboost lambdarank ridge_stacking.")
    p.add_argument("--top-n", type=int, default=20)
    # Removed --rebalance-mode: Now always uses non-overlapping for backtest metrics (accumulated return, max drawdown, Sharpe, graphs)
    # Average/median returns are calculated daily for prediction quality assessment
    p.add_argument("--hac-method", type=str, default="newey-west", choices=["newey-west", "hansen-hodrick"], help="HAC method: newey-west (default) or hansen-hodrick")
    p.add_argument("--hac-lag", type=int, default=None, help="HAC lag order (default: max(5, 2*horizon_days) for Newey-West)")
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost (bps) applied each rebalance as: turnover * cost_bps/1e4.")
    p.add_argument("--benchmark", type=str, default="QQQ")
    p.add_argument("--ridge-base-cols", nargs="+", default=None, help="Override RidgeStacker base_cols for this run (e.g. pred_catboost pred_elastic pred_xgb pred_lightgbm_ranker pred_lambdarank).")
    p.add_argument("--snapshot-id", type=str, default=None, help="Use existing snapshot ID instead of training (skip training phase)")
    p.add_argument("--output-dir", type=str, default="results/t10_time_split_80_20_final")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--ema-top-n", type=int, default=-1, 
                   help="Only apply EMA to stocks in top N (default: -1 to disable EMA, set to 0 to apply to all, set to >0 for top N filter)")
    p.add_argument("--ema-min-days", type=int, default=3,
                   help="Minimum consecutive days in top N to apply EMA (default: 3, only used when --ema-top-n > 0)")
    p.add_argument("--exclude-tickers", nargs="+", default=None,
                   help="List of tickers to exclude from training and testing (e.g. --exclude-tickers SBET TICKER2)")
    return p.parse_args()


def _compute_benchmark_tplus_from_yfinance(bench: str, rebalance_dates: pd.Series, horizon_days: int, logger: logging.Logger) -> pd.Series:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.warning("yfinance not available: %s", e)
        return pd.Series(dtype=float)

    dates = pd.to_datetime(rebalance_dates).dropna().sort_values()
    if len(dates) == 0:
        return pd.Series(dtype=float)

    start = (dates.min() - pd.Timedelta(days=30)).date().isoformat()
    end = (dates.max() + pd.Timedelta(days=30)).date().isoformat()
    logger.info("Fetching benchmark %s via yfinance (%s -> %s)...", bench, start, end)
    px = yf.download(
        tickers=bench,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if px is None or px.empty:
        return pd.Series(dtype=float)

    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    trading_days = close.index

    def _ret(d: pd.Timestamp) -> float:
        base_candidates = trading_days[trading_days <= d]
        if len(base_candidates) == 0:
            return float("nan")
        base = pd.Timestamp(base_candidates[-1])
        base_pos = int(trading_days.get_indexer([base])[0])
        tgt_pos = base_pos + int(horizon_days)
        if tgt_pos >= len(trading_days):
            return float("nan")
        tgt = pd.Timestamp(trading_days[tgt_pos])
        b = float(close.loc[base])
        t = float(close.loc[tgt])
        return (t - b) / b if b else float("nan")

    out = pd.Series({_d: _ret(pd.Timestamp(_d)) for _d in dates})
    out.index = pd.to_datetime(out.index)
    out.name = "benchmark_return"
    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _max_drawdown_non_overlap(series_pct: pd.Series, horizon: int) -> float:
    """Compute max drawdown using non-overlapping observations spaced by horizon days."""
    returns = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
    if horizon > 0 and not returns.empty:
        returns = returns.iloc[::horizon].reset_index(drop=True)
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min() * 100.0)


def calc_top10_accumulated_10d_rebalance(
    predictions_df: pd.DataFrame,
    top_n: int = 10,
    step: int = 10,
    out_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    bucket_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    计算Top bucket的10天rebalance累计收益曲线
    
    Args:
        predictions_df: DataFrame with columns [date, ticker, prediction, actual]
            - date: 预测日（T+0）
            - ticker: 股票代码
            - prediction: 预测分数，越大越好
            - actual: T+10 forward return (小数，0.018=1.8%)
        top_n: Top N股票数量，默认10（如果指定bucket_range则忽略）
        step: Rebalance间隔天数，默认10
        out_dir: 输出目录，如果提供则保存CSV和PNG
        model_name: 模型名称
        logger: 日志记录器
        bucket_range: 可选，指定bucket范围 (start, end)，如 (5, 15) 表示 Top 5-15
    
    Returns:
        DataFrame with columns [date, top_gross_return, acc_value, acc_return]
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 数据清理
    df = predictions_df.copy()
    df = df.dropna(subset=['prediction', 'actual'])
    
    # 确保date是datetime类型并排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 确定使用的bucket范围
    if bucket_range is not None:
        bucket_start, bucket_end = bucket_range
        bucket_label = f"Top{bucket_start}-{bucket_end}"
        use_bucket = True
    else:
        bucket_start, bucket_end = 1, top_n
        bucket_label = f"Top{top_n}"
        use_bucket = False
    
    # 获取所有唯一交易日（去重、升序）
    all_dates = sorted(df['date'].unique())
    
    if len(all_dates) == 0:
        logger.warning("No valid dates found in predictions_df")
        return pd.DataFrame(columns=['date', 'top_gross_return', 'acc_value', 'acc_return'])
    
    # 生成rebalance_dates：每step天取一次
    rebalance_dates = all_dates[::step]
    
    if len(rebalance_dates) == 0:
        logger.warning("No rebalance dates found")
        return pd.DataFrame(columns=['date', 'top_gross_return', 'acc_value', 'acc_return'])
    
    logger.info(f"Total trading days: {len(all_dates)}, Rebalance dates: {len(rebalance_dates)} (step={step})")
    
    # 存储每次rebalance的收益
    rows = []
    
    for t in rebalance_dates:
        # 取当天数据
        g = df[df['date'] == t].copy()
        
        if len(g) < bucket_end:
            logger.debug(f"Skipping date {t}: only {len(g)} stocks available (need {bucket_end})")
            continue
        
        # 按prediction降序排序
        g = g.sort_values('prediction', ascending=False)
        
        # 取bucket范围
        if use_bucket:
            # Top 5-15: 取第5到第15名（索引4到14）
            bucket_stocks = g.iloc[bucket_start-1:bucket_end]
        else:
            # Top N: 取前N名
            bucket_stocks = g.head(top_n)
        
        # 计算该次持有期收益（mean of actual）
        r_t = float(bucket_stocks['actual'].mean())
        
        rows.append({
            'date': t,
            'top_gross_return': r_t,
        })
    
    if len(rows) == 0:
        logger.warning("No valid rebalance periods found")
        return pd.DataFrame(columns=['date', 'top_gross_return', 'acc_value', 'acc_return'])
    
    # 构建DataFrame
    ts = pd.DataFrame(rows).sort_values('date')
    
    # 计算累计复利收益
    # acc_value = cumprod(1 + r_t)
    ts['acc_value'] = (1.0 + ts['top_gross_return']).cumprod()
    
    # acc_return = acc_value - 1
    ts['acc_return'] = ts['acc_value'] - 1.0
    
    # 🔧 Calculate max drawdown (non-overlap)
    running_max = ts['acc_value'].expanding().max()
    drawdown = (ts['acc_value'] / running_max - 1.0) * 100.0
    max_drawdown = float(drawdown.min())
    ts['drawdown'] = drawdown
    
    # 打印日志
    avg_return = float(ts['top_gross_return'].mean())
    final_acc_return = float(ts['acc_return'].iloc[-1])
    logger.info(f"{bucket_label} 10-day rebalance accumulated return:")
    logger.info(f"  Rebalance次数: {len(ts)}")
    logger.info(f"  平均{bucket_label.lower()}_gross_return: {avg_return:.6f} ({avg_return*100:.4f}%)")
    logger.info(f"  最终acc_return: {final_acc_return:.6f} ({final_acc_return*100:.4f}%)")
    logger.info(f"  最大回撤: {max_drawdown:.4f}%")
    
    # 保存文件
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据model_name和bucket_range生成文件名
        if use_bucket:
            file_suffix = f"top{bucket_start}_{bucket_end}"
        else:
            file_suffix = f"top{top_n}"
        
        if model_name:
            csv_filename = f"{model_name}_{file_suffix}_rebalance10d_accumulated.csv"
            png_filename = f"{model_name}_{file_suffix}_rebalance10d_accumulated.png"
            plot_title = f"{model_name} - {bucket_label} 10-Day Rebalance Accumulated Return Curve"
            plot_label = f'{model_name} {bucket_label} Accumulated Return'
        else:
            csv_filename = f"{file_suffix}_rebalance10d_accumulated.csv"
            png_filename = f"{file_suffix}_rebalance10d_accumulated.png"
            plot_title = f"{bucket_label} 10-Day Rebalance Accumulated Return Curve"
            plot_label = f'{bucket_label} Accumulated Return'
        
        # 保存CSV（包含drawdown列）
        csv_path = out_dir / csv_filename
        ts.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved CSV: {csv_path}")
        
        # 保存PNG图表
        png_path = out_dir / png_filename
        plt.figure(figsize=(14, 7))
        plt.plot(ts['date'], ts['acc_return'] * 100.0, linewidth=2.0, label=plot_label)
        plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        plt.title(plot_title)
        plt.xlabel("Rebalance Date")
        plt.ylabel("Accumulated Return (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        logger.info(f"Saved PNG: {png_path}")
    
    return ts


def calculate_group_returns_standalone(
    predictions: pd.DataFrame,
    top_n: int = 20,
    bottom_n: int = 20,
    cost_bps: float = 0.0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate Top-N and Bottom-N group returns (standalone, no ComprehensiveModelBacktest dependency)"""
    cost_rate = cost_bps / 10000.0
    rows = []
    prev_top_tickers = set()
    
    for date, date_group in predictions.groupby('date'):
        valid = date_group.dropna(subset=['prediction', 'actual'])
        if len(valid) < top_n + bottom_n:
            continue
        
        sorted_group = valid.sort_values('prediction', ascending=False)
        
        # Top N
        top_n_group = sorted_group.head(top_n)
        top_return_mean = float(top_n_group['actual'].mean())
        top_return_median = float(top_n_group['actual'].median())
        top_tickers = set(top_n_group['ticker'].astype(str).str.upper().str.strip())
        
        # Calculate turnover
        turnover = float(len(top_tickers.symmetric_difference(prev_top_tickers))) / top_n if prev_top_tickers else 1.0
        cost = turnover * cost_rate
        top_return_net_mean = top_return_mean - cost
        top_return_net_median = top_return_median - cost
        
        # Bottom N
        bottom_n_group = sorted_group.tail(bottom_n)
        bottom_return_mean = float(bottom_n_group['actual'].mean())
        bottom_return_median = float(bottom_n_group['actual'].median())
        
        rows.append({
            'date': pd.to_datetime(date),
            'top_return_mean': top_return_mean,
            'top_return_median': top_return_median,
            'top_return_net_mean': top_return_net_mean,
            'top_return_net_median': top_return_net_median,
            'bottom_return_mean': bottom_return_mean,
            'bottom_return_median': bottom_return_median,
            'top_turnover': turnover,
            'top_cost': cost,
        })
        
        prev_top_tickers = top_tickers
    
    if not rows:
        return {}, pd.DataFrame()
    
    ts_df = pd.DataFrame(rows).sort_values('date')
    
    summary = {
        # Mean metrics
        'avg_top_return': float(ts_df['top_return_mean'].mean()),
        'avg_top_return_net': float(ts_df['top_return_net_mean'].mean()),
        'avg_bottom_return': float(ts_df['bottom_return_mean'].mean()),
        # Median metrics
        'median_top_return': float(ts_df['top_return_median'].median()),
        'median_top_return_net': float(ts_df['top_return_net_median'].median()),
        'median_bottom_return': float(ts_df['bottom_return_median'].median()),
        # Other metrics
        'avg_top_turnover': float(ts_df['top_turnover'].mean()),
        'avg_top_cost': float(ts_df['top_cost'].mean()),
    }
    
    # Calculate Sharpe (using mean series for consistency)
    if len(ts_df) > 1:
        top_ret_series_mean = ts_df['top_return_net_mean']
        top_ret_series_median = ts_df['top_return_net_median']
        summary['top_sharpe_net'] = float(top_ret_series_mean.mean() / top_ret_series_mean.std() * np.sqrt(252 / 10)) if top_ret_series_mean.std() > 0 else np.nan
        summary['top_sharpe_net_median'] = float(top_ret_series_median.mean() / top_ret_series_median.std() * np.sqrt(252 / 10)) if top_ret_series_median.std() > 0 else np.nan
        summary['win_rate'] = float((top_ret_series_mean > 0).mean())
        summary['win_rate_median'] = float((top_ret_series_median > 0).mean())
    
    # For backward compatibility, add 'top_return' and 'top_return_net' columns using mean
    ts_df['top_return'] = ts_df['top_return_mean']
    ts_df['top_return_net'] = ts_df['top_return_net_mean']
    ts_df['bottom_return'] = ts_df['bottom_return_mean']
    
    return summary, ts_df


def calculate_group_returns_hold10d_nonoverlap(
    predictions: pd.DataFrame,
    top_n: int = 10,
    horizon_days: int = 10,
    cost_bps: float = 0.0,
    start_offset: int = 0,
) -> pd.DataFrame:
    """
    非重叠持有回测（符合：买TopN等权->持有horizon_days->到期全卖->买下一期）
    输入 predictions 必须包含列：['date','ticker','prediction','actual']
    其中 actual = t -> t+horizon_days 的实际收益（小数，例如0.02=2%）
    输出 out 的 top_return/top_return_net 以"小数"表示（后面脚本会 *100 再 _cum_pct）
    
    这个版本返回的格式直接匹配现有的 _cum_pct() 处理逻辑，确保累计收益计算正确。
    """
    req = {"date", "ticker", "prediction", "actual"}
    miss = req - set(predictions.columns)
    if miss:
        raise ValueError(f"predictions missing columns: {miss}")

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "prediction"], ascending=[True, False])

    unique_dates = df["date"].drop_duplicates().sort_values().to_list()
    if not unique_dates:
        return pd.DataFrame()

    # ✅ 只在每 horizon_days 个交易日 rebalance 一次（非重叠）
    rebalance_dates = unique_dates[start_offset::horizon_days]

    cost_rate = cost_bps / 10000.0  # 10bps -> 0.001
    prev_holdings = set()
    rows = []

    for d in rebalance_dates:
        g = df[df["date"] == d].dropna(subset=['prediction', 'actual'])
        if g.empty:
            continue

        top = g.head(top_n)
        cur_holdings = set(top["ticker"].astype(str).str.upper().str.strip().tolist())

        # 本期收益：等权持有 horizon_days（由 actual 提供）
        gross_mean = float(pd.to_numeric(top["actual"], errors="coerce").mean())
        gross_median = float(pd.to_numeric(top["actual"], errors="coerce").median())

        # turnover：只在换仓日计算
        if prev_holdings:
            sym_diff = cur_holdings.symmetric_difference(prev_holdings)
            turnover = len(sym_diff) / top_n if top_n > 0 else 0.0
        else:
            # 第一期：空仓->满仓，只买入
            turnover = 1.0

        # 成本口径与旧版一致：cost = turnover * 单边cost_rate
        # 若完全换仓：sym_diff=2N => turnover=2 => cost=2*cost_rate（买+卖）
        cost = turnover * cost_rate

        net_mean = gross_mean - cost
        net_median = gross_median - cost

        rows.append({
            "date": d,
            "top_return": gross_mean,            # 小数
            "top_return_net": net_mean,          # 小数
            "top_return_median": gross_median,   # 小数
            "top_return_net_median": net_median, # 小数
            "top_turnover": turnover,
            "top_cost": cost,
        })

        prev_holdings = cur_holdings

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def apply_ema_smoothing(predictions_df: pd.DataFrame, model_name: str, ema_history: dict, 
                        weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)) -> pd.DataFrame:
    """
    应用3天EMA平滑：S_smooth_t = 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}
    
    Args:
        predictions_df: 预测DataFrame，包含date, ticker, prediction列
        model_name: 模型名称
        ema_history: 历史预测分数字典 {model_name: {ticker: [S_t, S_{t-1}, S_{t-2}]}}
        weights: EMA权重 (today, t-1, t-2)，默认(0.6, 0.3, 0.1)
    
    Returns:
        添加了prediction_smooth列的DataFrame
    """
    if model_name not in ema_history:
        ema_history[model_name] = {}
    
    predictions_df = predictions_df.copy()
    predictions_df['prediction_smooth'] = np.nan
    
    # 按日期排序
    predictions_df = predictions_df.sort_values('date')
    
    for date, group in predictions_df.groupby('date'):
        for idx, row in group.iterrows():
            ticker = row['ticker']
            score_today = row['prediction']
            
            # 获取历史分数
            if ticker not in ema_history[model_name]:
                ema_history[model_name][ticker] = []
            
            history = ema_history[model_name][ticker]
            
            # 计算平滑分数
            if pd.isna(score_today):
                # 如果今天的分数是NaN，平滑分数也是NaN
                smooth_score = np.nan
            elif len(history) == 0:
                # 第一天：使用原始分数
                smooth_score = score_today
            elif len(history) == 1:
                # 第二天：0.6*S_t + 0.3*S_{t-1}
                if pd.isna(history[0]):
                    smooth_score = score_today  # 如果历史是NaN，只用今天的
                else:
                    smooth_score = weights[0] * score_today + weights[1] * history[0]
            else:
                # 第三天及以后：0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}
                hist_0 = history[0] if not pd.isna(history[0]) else 0.0
                hist_1 = history[1] if not pd.isna(history[1]) else 0.0
                smooth_score = (weights[0] * score_today + 
                               weights[1] * hist_0 + 
                               weights[2] * hist_1)
            
            predictions_df.loc[idx, 'prediction_smooth'] = smooth_score
            
            # 更新历史（保留最近3天）
            history.insert(0, score_today)
            if len(history) > 2:
                history.pop()
    
    return predictions_df


def backtest_top10_hold_10days(predictions_df: pd.DataFrame, horizon_days: int = 10, 
                                use_smooth: bool = False, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Top10选股，等权买入，持有10天卖出回测
    
    Args:
        predictions_df: 预测DataFrame，包含date, ticker, prediction, prediction_smooth, actual列
        horizon_days: 持有天数（默认10天）
        use_smooth: 是否使用EMA平滑分数
        logger: 日志记录器
    
    Returns:
        回测结果DataFrame，包含buy_date, sell_date, ticker, return等信息
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 选择使用的预测分数
    score_col = 'prediction_smooth' if use_smooth else 'prediction'
    if score_col not in predictions_df.columns:
        logger.warning(f"列{score_col}不存在，使用prediction")
        score_col = 'prediction'
    
    # 按日期排序
    predictions_df = predictions_df.sort_values('date').copy()
    
    # 创建实际收益率的日期映射（用于查找买入日期对应的T+10收益率）
    # actual列已经是T+10的收益率，所以直接使用买入日期的actual值
    actual_map = {}
    for _, row in predictions_df.iterrows():
        key = (row['date'], row['ticker'])
        actual_map[key] = row['actual']
    
    results = []
    holdings = {}  # {ticker: {'buy_date': date, 'buy_score': score}}
    
    unique_dates = sorted(predictions_df['date'].unique())
    
    for i, current_date in enumerate(unique_dates):
        # 卖出到期持仓
        to_sell = []
        for ticker, holding_info in list(holdings.items()):
            buy_date_held = holding_info['buy_date']
            days_held = (current_date - buy_date_held).days
            
            if days_held >= horizon_days:
                # 使用买入日期的actual值（已经是T+10收益率）
                actual_return = actual_map.get((buy_date_held, ticker), np.nan)
                
                if not np.isnan(actual_return):
                    results.append({
                        'buy_date': buy_date_held,
                        'sell_date': current_date,
                        'ticker': ticker,
                        'buy_score': holding_info['buy_score'],
                        'days_held': days_held,
                        'return': actual_return,
                        'use_smooth': use_smooth
                    })
                
                to_sell.append(ticker)
        
        # 移除已卖出的持仓
        for ticker in to_sell:
            del holdings[ticker]
        
        # 买入新的Top10
        day_data = predictions_df[predictions_df['date'] == current_date].copy()
        if len(day_data) < 10:
            continue
        
        # 按分数排序，选择Top10
        day_data = day_data.sort_values(score_col, ascending=False)
        top10 = day_data.head(10)
        
        # 等权买入（每个ticker分配1/10的权重）
        for _, row in top10.iterrows():
            ticker = row['ticker']
            score = row[score_col]
            
            # 如果已有持仓，跳过（避免重复买入）
            if ticker not in holdings:
                holdings[ticker] = {
                    'buy_date': current_date,
                    'buy_score': score
                }
    
    # 处理最后未卖出的持仓（在测试期结束时卖出）
    last_date = unique_dates[-1] if unique_dates else None
    if last_date:
        for ticker, holding_info in holdings.items():
            buy_date_held = holding_info['buy_date']
            days_held = (last_date - buy_date_held).days
            
            # 使用买入日期的actual值（已经是T+10收益率）
            actual_return = actual_map.get((buy_date_held, ticker), np.nan)
            
            if not np.isnan(actual_return):
                results.append({
                    'buy_date': buy_date_held,
                    'sell_date': last_date,
                    'ticker': ticker,
                    'buy_score': holding_info['buy_score'],
                    'days_held': days_held,
                    'return': actual_return,
                    'use_smooth': use_smooth
                })
    
    return pd.DataFrame(results)


def calculate_bucket_returns_standalone(
    predictions: pd.DataFrame,
    top_buckets: List[Tuple[int, int]],
    bottom_buckets: List[Tuple[int, int]],
    cost_bps: float = 0.0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate bucket returns (standalone version, no ComprehensiveModelBacktest dependency)"""
    cost_rate = cost_bps / 10000.0
    rows = []
    
    for date, date_group in predictions.groupby('date'):
        valid = date_group.dropna(subset=['prediction', 'actual'])
        if len(valid) < 30:
            continue
        
        sorted_group = valid.sort_values('prediction', ascending=False).reset_index(drop=True)
        n = len(sorted_group)
        
        row = {'date': pd.to_datetime(date), 'n_stocks': n}
        
        # Top buckets
        for a, b in top_buckets:
            if a <= n and b <= n:
                s = sorted_group.iloc[a-1:b]['actual']
                row[f'top_{a}_{b}_return_mean'] = float(s.mean()) if len(s) else np.nan
                row[f'top_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
                row[f'top_{a}_{b}_return_net_mean'] = row[f'top_{a}_{b}_return_mean']  # Simplified
                row[f'top_{a}_{b}_return_net_median'] = row[f'top_{a}_{b}_return_median']  # Simplified
            elif a <= n and b > n:
                # If b > n, use all available stocks from a to n
                s = sorted_group.iloc[a-1:n]['actual']
                row[f'top_{a}_{b}_return_mean'] = float(s.mean()) if len(s) else np.nan
                row[f'top_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
                row[f'top_{a}_{b}_return_net_mean'] = row[f'top_{a}_{b}_return_mean']  # Simplified
                row[f'top_{a}_{b}_return_net_median'] = row[f'top_{a}_{b}_return_median']  # Simplified
        
        # Bottom buckets
        for a, b in bottom_buckets:
            if a <= n:
                start = max(0, n - b)
                end = n - (a - 1)
                s = sorted_group.iloc[start:end]['actual']
                row[f'bottom_{a}_{b}_return_mean'] = float(s.mean()) if len(s) else np.nan
                row[f'bottom_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
        
        rows.append(row)
    
    if not rows:
        return {}, pd.DataFrame()
    
    df = pd.DataFrame(rows).sort_values('date')
    
    summary = {}
    for col in df.columns:
        if col.endswith('_return_mean') and df[col].notna().any():
            # Mean of daily mean returns
            summary[f'avg_{col.replace("_mean", "")}'] = float(df[col].mean())
            # Median of daily mean returns
            summary[f'median_{col.replace("_mean", "")}'] = float(df[col].median())
        elif col.endswith('_return_median') and df[col].notna().any():
            # Mean of daily median returns
            summary[f'avg_{col.replace("_median", "")}_from_median'] = float(df[col].mean())
            # Median of daily median returns
            summary[f'median_{col.replace("_median", "")}_from_median'] = float(df[col].median())
    
    # For backward compatibility, add columns without _mean/_median suffix using mean values
    for col in df.columns:
        if col.endswith('_return_mean'):
            base_col = col.replace('_mean', '')
            df[base_col] = df[col]
        elif col.endswith('_return_net_mean'):
            base_col = col.replace('_mean', '')
            df[base_col] = df[col]
    
    return summary, df


def calculate_bucket_returns_hold_horizon_nonoverlap(
    predictions: pd.DataFrame,
    top_buckets: List[Tuple[int, int]],
    bottom_buckets: List[Tuple[int, int]],
    horizon_days: int = 10,
    cost_bps: float = 0.0,
    start_offset: int = 0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    非重叠持有回测的bucket returns版本
    
    参数说明：
    - predictions: 必须含 ['date','ticker','prediction','actual']
    - top_buckets, bottom_buckets: 排名区间列表，如[(1,10), (5,15)]
    - horizon_days: 持有期天数（默认10天）
    - cost_bps: 交易成本（bps）
    - start_offset: 错位起点
    
    Returns:
        (summary_dict, timeseries_df)
    """
    req_cols = {"date", "ticker", "prediction", "actual"}
    missing = req_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "prediction"], ascending=[True, False])

    unique_dates = df["date"].drop_duplicates().sort_values().to_list()
    if len(unique_dates) == 0:
        return {}, pd.DataFrame()

    rebalance_dates = unique_dates[start_offset::horizon_days]
    if len(rebalance_dates) == 0:
        return {}, pd.DataFrame()

    cost_rate = cost_bps / 10000.0
    rows = []

    for d in rebalance_dates:
        g = df[df["date"] == d].dropna(subset=['prediction', 'actual'])
        if g.empty:
            continue

        sorted_group = g.sort_values('prediction', ascending=False).reset_index(drop=True)
        n = len(sorted_group)
        
        if n < 30:  # 至少需要30只股票
            continue

        row = {'date': pd.to_datetime(d), 'n_stocks': n}

        # Top buckets
        for a, b in top_buckets:
            if a <= n:
                s = sorted_group.iloc[a-1:b]['actual']
                row[f'top_{a}_{b}_return_mean'] = float(s.mean()) if len(s) else np.nan
                row[f'top_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan
                # 简化：bucket returns不单独计算cost（已在group returns中计算）
                row[f'top_{a}_{b}_return_net_mean'] = row[f'top_{a}_{b}_return_mean']
                row[f'top_{a}_{b}_return_net_median'] = row[f'top_{a}_{b}_return_median']

        # Bottom buckets
        for a, b in bottom_buckets:
            if a <= n:
                start = max(0, n - b)
                end = n - (a - 1)
                s = sorted_group.iloc[start:end]['actual']
                row[f'bottom_{a}_{b}_return_mean'] = float(s.mean()) if len(s) else np.nan
                row[f'bottom_{a}_{b}_return_median'] = float(s.median()) if len(s) else np.nan

        rows.append(row)

    if not rows:
        return {}, pd.DataFrame()

    out_df = pd.DataFrame(rows).sort_values('date')

    summary = {}
    for col in out_df.columns:
        if col.endswith('_return_mean') and out_df[col].notna().any():
            summary[f'avg_{col.replace("_mean", "")}'] = float(out_df[col].mean())
            summary[f'median_{col.replace("_mean", "")}'] = float(out_df[col].median())
        elif col.endswith('_return_median') and out_df[col].notna().any():
            summary[f'avg_{col.replace("_median", "")}_from_median'] = float(out_df[col].mean())
            summary[f'median_{col.replace("_median", "")}_from_median'] = float(out_df[col].median())

    # For backward compatibility
    for col in out_df.columns:
        if col.endswith('_return_mean'):
            base_col = col.replace('_mean', '')
            out_df[base_col] = out_df[col]
        elif col.endswith('_return_net_mean'):
            base_col = col.replace('_mean', '')
            out_df[base_col] = out_df[col]

    return summary, out_df


def _write_model_topn_vs_benchmark(
    *,
    run_dir: Path,
    model_name: str,
    preds: pd.DataFrame,
    top_n: int,
    horizon: int,
    bench: str,
    bench_ret: pd.Series,
    cost_bps: float,
    logger: logging.Logger,
    rebalance_mode: str = "nonoverlap",  # Always use non-overlapping for backtest metrics
) -> dict:
    """Write per-model TopN time series + plots and return a small summary dict (percent units).
    Always uses non-overlapping for backtest metrics (accumulated return, max drawdown, Sharpe, graphs).
    """
    # Always use non-overlapping for backtest metrics
    group_ts = calculate_group_returns_hold10d_nonoverlap(
        preds, top_n=top_n, horizon_days=horizon, cost_bps=cost_bps, start_offset=0
    )
    if group_ts.empty:
        raise RuntimeError(f"Group return time series is empty on test window for model={model_name}")
    
    # Calculate summary for logging (daily calculation for avg/median returns)
    from scripts.time_split_80_20_oos_eval import calculate_group_returns_standalone
    group_summary_daily, _ = calculate_group_returns_standalone(preds, top_n=top_n, bottom_n=top_n, cost_bps=cost_bps)
    
    # Calculate backtest metrics from non-overlapping
    periods_per_year = 252.0 / horizon
    if len(group_ts) > 1:
        net_mean_series = group_ts['top_return_net'].dropna()
        net_median_series = group_ts['top_return_net_median'].dropna()
        if len(net_mean_series) > 1 and net_mean_series.std() > 0:
            backtest_summary = {
                'top_sharpe_net': float((net_mean_series.mean() / net_mean_series.std()) * np.sqrt(periods_per_year)),
                'win_rate': float((net_mean_series > 0).mean()),
            }
        else:
            backtest_summary = {'top_sharpe_net': np.nan, 'win_rate': np.nan}
        if len(net_median_series) > 1 and net_median_series.std() > 0:
            backtest_summary['top_sharpe_net_median'] = float((net_median_series.mean() / net_median_series.std()) * np.sqrt(periods_per_year))
            backtest_summary['win_rate_median'] = float((net_median_series > 0).mean())
        else:
            backtest_summary['top_sharpe_net_median'] = np.nan
            backtest_summary['win_rate_median'] = np.nan
    else:
        backtest_summary = {'top_sharpe_net': np.nan, 'top_sharpe_net_median': np.nan, 'win_rate': np.nan, 'win_rate_median': np.nan}
    
    # Combine summaries
    group_summary = {**group_summary_daily, **backtest_summary}

    cols = ["date", "top_return"]
    if "top_return_net" in group_ts.columns:
        cols.append("top_return_net")
    if "top_turnover" in group_ts.columns:
        cols.append("top_turnover")
    if "top_cost" in group_ts.columns:
        cols.append("top_cost")
    # Non-overlapping mode: no cumulative columns needed (will be calculated by _cum_pct)

    out = group_ts[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date")
    out["benchmark_return"] = out["date"].map(lambda d: float(bench_ret.get(d, float("nan"))) if hasattr(bench_ret, "get") else float("nan"))

    # Convert to percent
    out["top_return"] = pd.to_numeric(out["top_return"], errors="coerce") * 100.0
    if "top_return_net" in out.columns:
        out["top_return_net"] = pd.to_numeric(out["top_return_net"], errors="coerce") * 100.0
    out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce") * 100.0

    # Cumulative (compounded) - non-overlapping mode: out["top_return"] is 10-day period returns (小数), *100后按期复利
    def _cum_pct(s_pct: pd.Series) -> pd.Series:
        r = pd.to_numeric(s_pct, errors="coerce").fillna(0.0) / 100.0
        return (1.0 + r).cumprod() - 1.0

    out["cum_top_return"] = _cum_pct(out["top_return"]) * 100.0
    if "top_return_net" in out.columns:
        out["cum_top_return_net"] = _cum_pct(out["top_return_net"]) * 100.0
    out["cum_benchmark_return"] = _cum_pct(out["benchmark_return"]) * 100.0

    # Save time series
    ts_path = run_dir / f"{model_name}_top{top_n}_timeseries.csv"
    out.to_csv(ts_path, index=False, encoding="utf-8")

    # Plots (period)
    plt.figure(figsize=(14, 7))
    plt.plot(out["date"], out["top_return"], label=f"{model_name} Top{top_n} (gross, period)", linewidth=1.6)
    if "top_return_net" in out.columns and cost_bps > 0:
        plt.plot(out["date"], out["top_return_net"], label=f"{model_name} Top{top_n} (net {cost_bps:g}bp, period)", linewidth=1.9)
    plt.plot(out["date"], out["benchmark_return"], label=f"{bench} (period)", linewidth=2.0, linestyle="--", color="black")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"OOS period returns (Top{top_n}, T+{horizon}) vs {bench} — model={model_name} — cost={cost_bps:g}bp")
    plt.xlabel("Rebalance date (test window)")
    plt.ylabel("Return over next horizon (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / f"{model_name}_top{top_n}_vs_{bench.lower()}.png", dpi=160)
    plt.close()

    # Plots (cumulative)
    plt.figure(figsize=(14, 7))
    plt.plot(out["date"], out["cum_top_return"], label=f"{model_name} Top{top_n} (gross, cum)", linewidth=1.6)
    if "cum_top_return_net" in out.columns and cost_bps > 0:
        plt.plot(out["date"], out["cum_top_return_net"], label=f"{model_name} Top{top_n} (net {cost_bps:g}bp, cum)", linewidth=2.0)
    plt.plot(out["date"], out["cum_benchmark_return"], label=f"{bench} (cum)", linewidth=2.2, linestyle="--", color="black")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"OOS cumulative return (Top{top_n}, T+{horizon}) vs {bench} — model={model_name} — cost={cost_bps:g}bp")
    plt.xlabel("Rebalance date (test window)")
    plt.ylabel("Cumulative return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / f"{model_name}_top{top_n}_vs_{bench.lower()}_cumulative.png", dpi=160)
    plt.close()

    logger.info("[%s] OOS Top%d avg return gross (mean, %%): %.6f", model_name, top_n, float(group_summary.get("avg_top_return", float("nan"))) * 100.0)
    logger.info("[%s] OOS Top%d median return gross (median, %%): %.6f", model_name, top_n, float(group_summary.get("median_top_return", float("nan"))) * 100.0)
    if cost_bps > 0 and "avg_top_return_net" in group_summary:
        logger.info("[%s] OOS Top%d avg return net (mean, %%): %.6f", model_name, top_n, float(group_summary.get("avg_top_return_net", float("nan"))) * 100.0)
        logger.info("[%s] OOS Top%d median return net (median, %%): %.6f", model_name, top_n, float(group_summary.get("median_top_return_net", float("nan"))) * 100.0)

    # Calculate max drawdown from equity curve (non-overlapping mode)
    def max_drawdown_from_equity(equity_pct: pd.Series) -> float:
        """Calculate max drawdown from cumulative return series (in percent)"""
        equity_decimal = pd.to_numeric(equity_pct, errors="coerce").fillna(0.0) / 100.0 + 1.0  # Convert to equity curve
        running_max = equity_decimal.cummax()
        dd = (equity_decimal - running_max) / running_max
        return float(dd.min() * 100.0)  # Convert back to percent
    
    # Use net cumulative return for drawdown
    if "cum_top_return_net" in out.columns:
        top_max_dd = max_drawdown_from_equity(out["cum_top_return_net"])
    else:
        top_max_dd = max_drawdown_from_equity(out["cum_top_return"])
    bench_max_dd = max_drawdown_from_equity(out["cum_benchmark_return"])

    return {
        "model": model_name,
        "top_n": top_n,
        # Mean metrics
        "avg_top_return_pct": _safe_float(pd.to_numeric(out["top_return"], errors="coerce").mean()),
        "avg_top_return_net_pct": _safe_float(pd.to_numeric(out["top_return_net"], errors="coerce").mean()) if "top_return_net" in out.columns else float("nan"),
        "avg_benchmark_return_pct": _safe_float(pd.to_numeric(out["benchmark_return"], errors="coerce").mean()),
        # Median metrics
        "median_top_return_pct": _safe_float(pd.to_numeric(out["top_return"], errors="coerce").median()),
        "median_top_return_net_pct": _safe_float(pd.to_numeric(out["top_return_net"], errors="coerce").median()) if "top_return_net" in out.columns else float("nan"),
        "median_benchmark_return_pct": _safe_float(pd.to_numeric(out["benchmark_return"], errors="coerce").median()),
        # Cumulative returns
        "end_cum_top_return_pct": _safe_float(out["cum_top_return"].iloc[-1]) if len(out) else float("nan"),
        "end_cum_top_return_net_pct": _safe_float(out["cum_top_return_net"].iloc[-1]) if "cum_top_return_net" in out.columns and len(out) else float("nan"),
        "end_cum_benchmark_return_pct": _safe_float(out["cum_benchmark_return"].iloc[-1]) if len(out) else float("nan"),
        "timeseries_csv": str(ts_path).replace("\\", "/"),
        "top_max_drawdown_pct_non_overlap": _safe_float(top_max_dd),
        "benchmark_max_drawdown_pct_non_overlap": _safe_float(bench_max_dd),
        "max_drawdown_horizon": horizon,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("time_split_90_10")

    # Make imports work on Windows
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(args.horizon_days)
    split = float(args.split)
    if not (0.5 < split < 0.95):
        raise ValueError("--split must be in (0.5, 0.95) for a meaningful train/test split")

    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    from bma_models.model_registry import load_models_from_snapshot
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Load data directly (NO ComprehensiveModelBacktest dependency)
    logger.info("Loading data to compute 90/10 time split...")
    data_path = Path(args.data_file)
    train_data_path = Path(args.train_data)
    
    # Determine which path to use: prefer data_file if it exists, otherwise use train_data
    use_path = None
    if data_path.exists():
        use_path = data_path
        logger.info(f"Using data_file path: {use_path}")
    elif train_data_path.exists():
        use_path = train_data_path
        logger.info(f"data_file not found, using train_data path: {use_path}")
    else:
        # Provide helpful error message with suggestions
        error_msg = f"Neither data_file nor train_data found:\n"
        error_msg += f"  - data_file: {data_path.absolute()}\n"
        error_msg += f"  - train_data: {train_data_path.absolute()}\n\n"
        
        # Check for common alternative paths
        common_paths = [
            Path("data/factor_exports/polygon_factors_all_filtered.parquet"),
            Path("data/factor_exports/factors/factors_all.parquet"),
            Path("data/factor_exports/factors"),
        ]
        existing_paths = [p for p in common_paths if p.exists()]
        
        if existing_paths:
            error_msg += "Found these alternative data files:\n"
            for p in existing_paths:
                error_msg += f"  - {p.absolute()}\n"
            error_msg += f"\nTry using: --data-file \"{existing_paths[0]}\""
        else:
            error_msg += "No data files found in common locations. Please check your data file path."
        
        raise FileNotFoundError(error_msg)
    
    # Handle both directory (multiple parquet files) and single file cases
    if use_path.is_dir():
        logger.info(f"Loading parquet files from directory: {use_path}")
        parquet_files = sorted(use_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {use_path}")
        logger.info(f"Found {len(parquet_files)} parquet files, loading and concatenating...")
        dfs = []
        for pf in parquet_files:
            try:
                df_part = pd.read_parquet(pf)
                dfs.append(df_part)
            except Exception as e:
                logger.warning(f"Failed to load {pf}: {e}, skipping...")
        if not dfs:
            raise RuntimeError(f"Failed to load any parquet files from {use_path}")
        df = pd.concat(dfs, ignore_index=False)
        logger.info(f"Loaded and concatenated {len(dfs)} parquet files, total shape: {df.shape}")
    elif use_path.is_file():
        logger.info(f"Loading single parquet file: {use_path}")
        # Use memory-efficient loading with pyarrow
        import pyarrow.parquet as pq
        try:
            # Try memory-mapped reading for large files
            logger.info("Attempting memory-efficient parquet loading...")
            table = pq.read_table(str(use_path), memory_map=True)
            df = table.to_pandas(split_blocks=True, self_destruct=True)
            del table
            import gc
            gc.collect()
            logger.info(f"Successfully loaded with memory mapping, shape: {df.shape}")
        except Exception as mem_e:
            logger.warning(f"Memory-mapped loading failed: {mem_e}, trying standard loading...")
            df = pd.read_parquet(str(use_path))
    else:
        raise FileNotFoundError(f"Path exists but is neither file nor directory: {use_path}")
    
    target_col = f'ret_fwd_{horizon}d'
    if target_col in df.columns:
        df['target'] = df[target_col].copy()
        logger.info(f"Aligned 'target' column to {target_col}.")
    elif 'target' in df.columns:
        logger.warning(f"Existing 'target' column found but {target_col} missing; keeping original target.")
    else:
        logger.warning(f"Target column {target_col} missing; future metrics may lack actual returns.")

    # Ensure MultiIndex format - handle both MultiIndex and column-based formats
    if isinstance(df.index, pd.MultiIndex):
        # Already MultiIndex - check and standardize names
        index_names = [str(n).lower() if n else '' for n in df.index.names]
        if 'date' in index_names and ('ticker' in index_names or 'symbol' in index_names):
            # Reorder to ensure date, ticker order
            if 'symbol' in index_names:
                df = df.reset_index().rename(columns={'symbol': 'ticker'})
                df = df.set_index(['date', 'ticker'])
            else:
                df = df.reorder_levels(['date', 'ticker'])
                df.index.names = ['date', 'ticker']
        else:
            logger.warning(f"MultiIndex has unexpected names: {df.index.names}, attempting to fix...")
            # Try to reset and rebuild
            df = df.reset_index()
            if 'date' not in df.columns or 'ticker' not in df.columns:
                raise RuntimeError(f"Data has MultiIndex but cannot find date/ticker: {df.index.names}")
    elif 'date' in df.columns and 'ticker' in df.columns:
        # Has date/ticker columns - convert to MultiIndex
        logger.info("Converting date/ticker columns to MultiIndex...")
        dates = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        tickers = df['ticker'].astype(str).str.strip().str.upper()
        df = df.drop(['date', 'ticker'], axis=1)
        df.index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
    elif 'date' in df.columns and 'symbol' in df.columns:
        # Has date/symbol columns - convert to MultiIndex
        logger.info("Converting date/symbol columns to MultiIndex...")
        dates = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        tickers = df['symbol'].astype(str).str.strip().str.upper()
        df = df.drop(['date', 'symbol'], axis=1)
        df.index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
    else:
        raise RuntimeError(f"Data must have MultiIndex(date, ticker) or columns [date, ticker]. "
                          f"Found index: {type(df.index)}, columns: {list(df.columns)[:10]}")
    
    # Ensure date is datetime and ticker is string
    df.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(df.index.get_level_values('date')).tz_localize(None),
        df.index.get_level_values('ticker').astype(str)
    ], names=['date', 'ticker'])
    
    # Sort by index for efficient lookups
    df = df.sort_index()
    
    # Exclude specific tickers if provided
    if args.exclude_tickers:
        exclude_set = {str(t).upper().strip() for t in args.exclude_tickers}
        before_exclude = df.index.get_level_values('ticker').nunique()
        ticker_level = df.index.get_level_values('ticker').astype(str).str.upper().str.strip()
        mask = ~ticker_level.isin(exclude_set)
        df = df.loc[mask].copy()
        after_exclude = df.index.get_level_values('ticker').nunique() if len(df) > 0 else 0
        logger.info(f"🚫 [EXCLUDE] 排除股票: {sorted(exclude_set)}")
        logger.info(f"🚫 [EXCLUDE] 股票数量: {before_exclude} → {after_exclude} (排除 {before_exclude - after_exclude} 个股票)")
    
    # 🔥 Add Sato factors if missing (for MultiIndex data)
    if 'feat_sato_momentum_10d' not in df.columns or 'feat_sato_divergence_10d' not in df.columns:
        try:
            logger.info("🔥 Computing Sato Square Root Factors (momentum + divergence)...")
            from scripts.sato_factor_calculation import calculate_sato_factors
            
            # Prepare data for Sato calculation
            sato_data = df.copy()
            
            # Ensure we have adj_close (use Close if not available)
            if 'adj_close' not in sato_data.columns:
                if 'Close' in sato_data.columns:
                    sato_data['adj_close'] = sato_data['Close']
                else:
                    logger.warning("⚠️ No Close or adj_close found, skipping Sato factors")
                    sato_data = None
            
            if sato_data is not None:
                # Check if vol_ratio_20d exists
                has_vol_ratio = 'vol_ratio_20d' in sato_data.columns
                
                # If Volume doesn't exist, estimate from vol_ratio_20d
                if 'Volume' not in sato_data.columns:
                    if has_vol_ratio:
                        base_volume = 1_000_000
                        sato_data['Volume'] = base_volume * sato_data['vol_ratio_20d'].fillna(1.0).clip(lower=0.1, upper=10.0)
                        use_vol_ratio = True
                    else:
                        logger.warning("⚠️ No Volume or vol_ratio_20d found, skipping Sato factors")
                        sato_data = None
                else:
                    use_vol_ratio = has_vol_ratio
                
                if sato_data is not None:
                    # Calculate Sato factors
                    sato_factors_df = calculate_sato_factors(
                        df=sato_data,
                        price_col='adj_close',
                        volume_col='Volume',
                        vol_ratio_col='vol_ratio_20d',
                        lookback_days=10,
                        vol_window=20,
                        use_vol_ratio_directly=use_vol_ratio
                    )
                    
                    # Add Sato factors to df
                    df['feat_sato_momentum_10d'] = sato_factors_df['feat_sato_momentum_10d'].reindex(df.index).fillna(0.0)
                    df['feat_sato_divergence_10d'] = sato_factors_df['feat_sato_divergence_10d'].reindex(df.index).fillna(0.0)
                    logger.info("✅ Sato factors added to dataset")
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute Sato factors: {e}, continuing without them...")
            # Add zero-filled columns if missing
            if 'feat_sato_momentum_10d' not in df.columns:
                df['feat_sato_momentum_10d'] = 0.0
            if 'feat_sato_divergence_10d' not in df.columns:
                df['feat_sato_divergence_10d'] = 0.0
    
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        raise RuntimeError("Expected MultiIndex with 'date' level in factors dataset.")

    dates = pd.Index(pd.to_datetime(df.index.get_level_values("date")).tz_localize(None).unique()).sort_values()
    n_dates = len(dates)
    if n_dates < 200:
        logger.warning("Only %d unique dates detected; 90/10 split may be noisy.", n_dates)

    split_idx = int(n_dates * split)
    # Purge leakage gap = horizon days (labels use forward returns)
    train_end_idx = max(0, split_idx - 1 - horizon)
    if train_end_idx <= 0:
        raise RuntimeError("Not enough dates to apply purge gap; reduce horizon or use more history.")

    train_start = dates[0]
    train_end = dates[train_end_idx]
    test_start = dates[split_idx]
    test_end = dates[-1]

    logger.info("Time split (purged): train=%s..%s, test=%s..%s (dates=%d, split=%.2f, gap=%d)",
                train_start.date(), train_end.date(), test_start.date(), test_end.date(), n_dates, split, horizon)
    logger.info("Costs: cost_bps=%.4f (applied on test backtest only)", float(args.cost_bps or 0.0))

    # Use existing snapshot_id if provided, otherwise train
    snapshot_id = args.snapshot_id
    if snapshot_id:
        logger.info("=" * 80)
        logger.info(f"📦 Using existing snapshot_id: {snapshot_id} (skipping training)")
        logger.info("=" * 80)
    else:
        # Train snapshot on train window only
        # Optional: override RidgeStacker base_cols for this run by writing a temp unified_config
        # and setting BMA_TEMP_CONFIG_PATH (respected by UnifiedTrainingConfig).
        import os
        temp_cfg_path = None
        try:
            if args.ridge_base_cols:
                try:
                    import yaml  # type: ignore

                    base_cfg = Path("bma_models/unified_config.yaml")
                    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
                    training_cfg = cfg.setdefault("training", {})
                    meta_ranker_cfg = training_cfg.setdefault("meta_ranker", {})
                    meta_ranker_cfg["base_cols"] = [str(x) for x in args.ridge_base_cols]

                    temp_cfg_path = run_dir / "unified_config_override.yaml"
                    temp_cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
                    os.environ["BMA_TEMP_CONFIG_PATH"] = str(temp_cfg_path)
                    logger.info("🔧 Overriding MetaRankerStacker base_cols: %s", meta_ranker_cfg["base_cols"])
                    logger.info("🔧 Using temp config: %s", temp_cfg_path)
                except Exception as e:
                    logger.warning("Failed to apply --ridge-base-cols override; proceeding with default config. err=%s", e)

            model = UltraEnhancedQuantitativeModel()
            # Use --data-file if provided (non-default), otherwise fall back to --train-data
            # Check if user explicitly provided --data-file (not using default)
            if args.data_file and args.data_file != "data/factor_exports/factors/factors_all.parquet":
                training_data_path = args.data_file
            else:
                training_data_path = args.train_data
            train_res = model.train_from_document(
                training_data_path=str(Path(training_data_path)),
                top_n=50,
                start_date=str(train_start.date()),
                end_date=str(train_end.date()),
            )
        finally:
            os.environ.pop("BMA_TEMP_CONFIG_PATH", None)
        if not train_res.get("success", False):
            raise RuntimeError(f"Training failed: {train_res.get('error')}")

        snapshot_id = getattr(model, "active_snapshot_id", None) or train_res.get("snapshot_id")
        if not snapshot_id:
            raise RuntimeError("Training succeeded but did not yield snapshot_id.")

        # If --ridge-base-cols provided, refit RidgeStacker on the training OOF stacker_data with the requested base_cols,
        # then save a NEW snapshot (reusing the already-trained base models) so backtest truly reflects the new Ridge design.
        if args.ridge_base_cols:
            try:
                stacker_data = getattr(model, "_last_stacker_data", None)
                if stacker_data is None or not isinstance(stacker_data, pd.DataFrame) or stacker_data.empty:
                    raise RuntimeError("Missing model._last_stacker_data; cannot refit RidgeStacker with custom base_cols.")

                want_cols = [str(c) for c in args.ridge_base_cols]
                missing = [c for c in want_cols if c not in stacker_data.columns]
                if missing:
                    raise RuntimeError(f"stacker_data missing required columns for ridge_base_cols: {missing}. Available={list(stacker_data.columns)[:30]}")

                from bma_models.meta_ranker_stacker import MetaRankerStacker
                from bma_models.model_registry import load_models_from_snapshot, save_model_snapshot
                from bma_models.unified_config_loader import CONFIG

                loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
                base_models = loaded.get("models") or {}
                lambda_model = loaded.get("lambda_rank_stacker") or getattr(model, "lambda_rank_stacker", None)
                lambda_pct = loaded.get("lambda_percentile_transformer") or getattr(model, "lambda_percentile_transformer", None)

                # Prepare snapshot payload in the format expected by save_model_snapshot
                formatted_models = {
                    "elastic_net": {"model": base_models.get("elastic_net")},
                    "xgboost": {"model": base_models.get("xgboost")},
                    "catboost": {"model": base_models.get("catboost")},
                    # REMOVED: "lightgbm_ranker": {"model": base_models.get("lightgbm_ranker")},  # LightGBM Ranker disabled
                }

                # Load MetaRankerStacker config from unified_config.yaml
                yaml_config = CONFIG._load_yaml_config()
                meta_ranker_cfg = yaml_config.get('training', {}).get('meta_ranker', {})
                
                logger.info("🔧 Re-fitting MetaRankerStacker on OOF stacker_data with base_cols=%s", want_cols)
                meta_ranker = MetaRankerStacker(
                    base_cols=tuple(want_cols),
                    n_quantiles=meta_ranker_cfg.get('n_quantiles', 64),
                    label_gain_power=meta_ranker_cfg.get('label_gain_power', 1.7),  # Updated: 1.7
                    num_boost_round=meta_ranker_cfg.get('num_boost_round', 140),  # Updated: 140
                    early_stopping_rounds=meta_ranker_cfg.get('fit_params', {}).get('early_stopping_rounds', 40),  # Updated: 40
                    lgb_params={
                        'objective': meta_ranker_cfg.get('objective', 'lambdarank'),
                        'metric': meta_ranker_cfg.get('metric', 'ndcg'),
                        'ndcg_eval_at': meta_ranker_cfg.get('ndcg_eval_at', [10, 30]),
                        'num_leaves': meta_ranker_cfg.get('num_leaves', 31),  # Updated: 31
                        'max_depth': meta_ranker_cfg.get('max_depth', 4),
                        'learning_rate': meta_ranker_cfg.get('learning_rate', 0.03),  # Updated: 0.03
                        'min_data_in_leaf': meta_ranker_cfg.get('min_data_in_leaf', 200),  # Updated: 200
                        'lambda_l1': meta_ranker_cfg.get('lambda_l1', 0.0),  # Updated: 0.0
                        'lambda_l2': meta_ranker_cfg.get('lambda_l2', 15.0),  # Updated: 15.0
                        'feature_fraction': meta_ranker_cfg.get('feature_fraction', 1.0),
                        'bagging_fraction': meta_ranker_cfg.get('bagging_fraction', 0.8),  # Updated: 0.8
                        'bagging_freq': meta_ranker_cfg.get('bagging_freq', 1),
                        'lambdarank_truncation_level': meta_ranker_cfg.get('lambdarank_truncation_level', 1200),  # Updated: 1200
                        'sigmoid': meta_ranker_cfg.get('sigmoid', 1.2),  # Updated: 1.2
                        'verbose': meta_ranker_cfg.get('verbose', -1),
                        'random_state': meta_ranker_cfg.get('random_state', 42),
                    },
                    use_purged_cv=True,
                    use_internal_cv=True,
                    random_state=42
                )
                meta_ranker.fit(stacker_data, max_train_to_today=True)

                tag = f"time_split_meta_ranker_basecols_{'-'.join(want_cols)}"
                snapshot_id = save_model_snapshot(
                    training_results={"models": formatted_models},
                    meta_ranker_stacker=meta_ranker,  # 🔧 FIX: 使用meta_ranker_stacker参数
                    lambda_rank_stacker=lambda_model,
                    rank_aware_blender=None,
                    lambda_percentile_transformer=lambda_pct,
                    tag=tag,
                )
                logger.info("✅ New snapshot with updated MetaRankerStacker saved: %s", snapshot_id)
            except Exception as e:
                logger.warning("Failed to refit/save MetaRankerStacker with --ridge-base-cols; falling back to training snapshot. err=%s", e)

    (run_dir / "snapshot_id.txt").write_text(str(snapshot_id), encoding="utf-8")
    logger.info("Snapshot: %s", snapshot_id)

    # Decide whether to load CatBoost for backtest based on ridge base columns.
    # If ridge uses 'pred_catboost', backtest must load catboost model to compute ridge_stacking.
    load_catboost_for_ridge = False
    try:
        from bma_models.model_registry import load_manifest
        import json as _json

        manifest = load_manifest(str(snapshot_id))
        ridge_meta_path = (manifest.get("paths") or {}).get("ridge_meta_json")
        if ridge_meta_path and Path(ridge_meta_path).exists():
            meta = _json.loads(Path(ridge_meta_path).read_text(encoding="utf-8"))
            base_cols = list(meta.get("base_cols") or [])
            load_catboost_for_ridge = "pred_catboost" in set(map(str, base_cols))
    except Exception as e:
        logger.warning("Could not inspect ridge_meta.json to decide CatBoost loading: %s", e)

    # ========== STANDALONE PREDICTION LOOP (NO ComprehensiveModelBacktest) ==========
    logger.info("=" * 80)
    logger.info("🔮 Loading models and running standalone predictions (DAILY rebalancing)")
    logger.info("=" * 80)

    # Load models directly
    loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=bool(load_catboost_for_ridge))
    models_dict = loaded.get("models", {})
    ridge_stacker = loaded.get("ridge_stacker")
    lambda_rank_stacker = loaded.get("lambda_rank_stacker")
    
    # Log stacker status
    if ridge_stacker is not None:
        stacker_type = type(ridge_stacker).__name__
        is_fitted = getattr(ridge_stacker, 'fitted_', False)
        has_lightgbm = hasattr(ridge_stacker, 'lightgbm_model') and ridge_stacker.lightgbm_model is not None
        has_ridge = hasattr(ridge_stacker, 'ridge_model') and ridge_stacker.ridge_model is not None
        base_cols = getattr(ridge_stacker, 'base_cols', getattr(ridge_stacker, 'actual_feature_cols_', []))
        logger.info(f"✅ Loaded {stacker_type}: fitted={is_fitted}, has_lightgbm={has_lightgbm}, has_ridge={has_ridge}")
        logger.info(f"   Base columns: {base_cols}")
    else:
        logger.warning("⚠️  No ridge_stacker loaded from snapshot!")

    # Filter data to test window
    test_data = df.loc[
        (df.index.get_level_values('date') >= test_start) & 
        (df.index.get_level_values('date') <= test_end)
    ].copy()

    # Get rebalance dates (DAILY - all trading days, ONE PREDICTION PER DAY)
    rebalance_dates = test_data.index.get_level_values("date").unique().sort_values()
    rebalance_dates = pd.to_datetime(rebalance_dates).tz_localize(None)
    if args.max_weeks:
        max_dates = int(args.max_weeks * 5)  # Approximate trading days per week
        if len(rebalance_dates) > max_dates:
            rebalance_dates = rebalance_dates[-max_dates:]

    logger.info(f"📅 Daily rebalancing: {len(rebalance_dates)} trading days (OVERLAPPING observations)")
    logger.info(f"⚠️  Statistical inference will use HAC corrections (lag≥10)")

    # Determine HAC lag
    hac_method = getattr(args, 'hac_method', 'newey-west')
    if hac_method == "newey-west":
        hac_lag = getattr(args, 'hac_lag', None) or max(10, 2 * horizon)
    else:  # hansen-hodrick
        hac_lag = getattr(args, 'hac_lag', None) or max(horizon - 1, 1)

    logger.info(f"📊 HAC Method: {hac_method}, Lag: {hac_lag}")
    logger.info(f"📝 Disclosure: 基于重叠观测 (overlapping observations), 使用{'Newey-West HAC' if hac_method == 'newey-west' else 'Hansen-Hodrick'}标准误 (lag={hac_lag})")

    # Initialize results storage
    all_results = {
        'elastic_net': [],
        'xgboost': [],
        'catboost': [],
        # REMOVED: 'lightgbm_ranker': [],  # LightGBM Ranker disabled
        'lambdarank': [],
        'ridge_stacking': []
    }
    
    # EMA smoothing is applied in the prediction loop (see apply_ema_smoothing call below)
    # EMA smoothing uses 3-day history with weights (0.6, 0.3, 0.1) for 90/10 split test
    
    # 🔧 Top10持有10天回测：存储持仓信息
    # 格式: {buy_date: {ticker: {'buy_score': score, 'sell_date': sell_date}}}
    holdings = {}  # 当前持仓
    holding_period_days = horizon  # 持有10天

    # Get feature columns - ONLY use specified factors as input
    # 🔥 UPDATED: Matching backup file features (2026-01-23)
    # 🔥 CRITICAL FIX: Exclude target_new and Close_new to prevent data leakage!
    allowed_feature_cols = [
        'volume_price_corr_3d',
        'rsi_14',
        'reversal_3d',
        'momentum_10d',
        'liquid_momentum_10d',
        'sharpe_momentum_5d',
        'price_ma20_deviation',
        'avg_trade_size',
        'trend_r2_20',
        'dollar_vol_20',
        'ret_skew_20d',
        'reversal_5d',
        'near_52w_high',
        'atr_pct_14',
        'amihud_20',
    ]
    exclude_cols = {'target', 'Close', 'sector', 'target_new', 'Close_new', f'ret_fwd_{horizon}d', 'ret_fwd_5d', 'ret_fwd_10d'}
    # Only use allowed features that exist in test_data
    all_feature_cols = [col for col in allowed_feature_cols if col in test_data.columns and col not in exclude_cols]
    logger.info(f"Using {len(all_feature_cols)} specified features: {all_feature_cols}")
    logger.info(f"Test data has {len(test_data.columns)} total columns, using only {len(all_feature_cols)} features")

    # Rolling prediction loop (ONE PREDICTION PER DAY)
    seen_dates = set()
    for pred_date in rebalance_dates:
        if pred_date in seen_dates:
            logger.warning(f"⚠️ 跳过重复日期: {pred_date}")
            continue
        seen_dates.add(pred_date)
        
        try:
            date_data = test_data.xs(pred_date, level='date', drop_level=True)
        except KeyError:
            continue
        
        if len(date_data) == 0:
            continue
        
        # Prepare features
        X = date_data[all_feature_cols].fillna(0)
        if X.empty:
            continue
        
        # Get actual target and apply hard clipping (same as training)
        # 🔧 硬截断target（clip，不是删除）- 防止极端值（如9000%收益）炸穿评估
        # 训练与评估同时：clip(y, -0.55, +0.55) （10天收益±55%已经很宽了）
        actual_target = date_data['target'] if 'target' in date_data.columns else pd.Series(np.nan, index=X.index)
        if actual_target.notna().any():
            # Load clip bounds from config (same as training)
            try:
                from bma_models.unified_config_loader import CONFIG
                yaml_config = CONFIG._load_yaml_config()
                extreme_filter_config = yaml_config.get('training', {}).get('extreme_target_filter', {})
                if extreme_filter_config.get('enabled', True) and extreme_filter_config.get('method') == 'hard_clip':
                    clip_lower = extreme_filter_config.get('clip_lower', -0.55)
                    clip_upper = extreme_filter_config.get('clip_upper', 0.55)
                    actual_target = actual_target.clip(lower=clip_lower, upper=clip_upper)
            except Exception as e:
                # Fallback to default clip if config loading fails
                actual_target = actual_target.clip(lower=-0.55, upper=0.55)
        tickers = X.index.tolist()
        
        # Predict with each model
        preds_dict = {}
        
        # ElasticNet
        if 'elastic_net' in models_dict and models_dict['elastic_net'] is not None:
            try:
                X_aligned = align_test_features_with_model(X, models_dict['elastic_net'], 'ElasticNet', logger)
                pred = models_dict['elastic_net'].predict(X_aligned)
                preds_dict['elastic_net'] = pred if isinstance(pred, np.ndarray) else pred.values
            except Exception as e:
                logger.warning(f"ElasticNet prediction failed for {pred_date}: {e}", exc_info=True)
        
        # XGBoost
        if 'xgboost' in models_dict and models_dict['xgboost'] is not None:
            try:
                X_aligned = align_test_features_with_model(X, models_dict['xgboost'], 'XGBoost', logger)
                pred = models_dict['xgboost'].predict(X_aligned)
                preds_dict['xgboost'] = pred if isinstance(pred, np.ndarray) else pred.values
            except Exception as e:
                logger.warning(f"XGBoost prediction failed for {pred_date}: {e}", exc_info=True)
        
        # CatBoost
        if 'catboost' in models_dict and models_dict['catboost'] is not None:
            try:
                X_aligned = align_test_features_with_model(X, models_dict['catboost'], 'CatBoost', logger)
                pred = models_dict['catboost'].predict(X_aligned)
                preds_dict['catboost'] = pred if isinstance(pred, np.ndarray) else pred.values
            except Exception as e:
                logger.warning(f"CatBoost prediction failed for {pred_date}: {e}", exc_info=True)
        
        # LightGBM ranker (DISABLED - removed from first layer)
        # REMOVED: LightGBM Ranker has been completely disabled from first layer
        # if 'lightgbm_ranker' in models_dict and models_dict['lightgbm_ranker'] is not None:
        #     try:
        #         X_aligned = align_test_features_with_model(X, models_dict['lightgbm_ranker'], 'LightGBM Ranker', logger)
        #         pred = models_dict['lightgbm_ranker'].predict(X_aligned)
        #         preds_dict['lightgbm_ranker'] = pred if isinstance(pred, np.ndarray) else (pred.values if hasattr(pred, 'values') else np.array(pred))
        #     except Exception as e:
        #         logger.warning(f"LightGBM ranker prediction failed for {pred_date}: {e}", exc_info=True)

        # LambdaRank
        if lambda_rank_stacker is not None:
            try:
                X_lambda = X.copy()
                X_lambda.index = pd.MultiIndex.from_arrays(
                    [[pred_date] * len(X_lambda), X_lambda.index],
                    names=['date', 'ticker']
                )
                pred_result = lambda_rank_stacker.predict(X_lambda)
                if isinstance(pred_result, pd.DataFrame) and 'lambda_score' in pred_result.columns:
                    # CRITICAL FIX: Ensure predictions are aligned with original ticker order
                    # pred_result may have different index order, so we need to align it
                    pred_series = pred_result['lambda_score']
                    # Reindex to match X_lambda index to ensure correct ticker-prediction mapping
                    pred_series_aligned = pred_series.reindex(X_lambda.index)
                    preds_dict['lambdarank'] = pred_series_aligned.values
                elif isinstance(pred_result, pd.Series):
                    # Reindex to match X_lambda index
                    pred_result_aligned = pred_result.reindex(X_lambda.index)
                    preds_dict['lambdarank'] = pred_result_aligned.values
                else:
                    preds_dict['lambdarank'] = np.array(pred_result).ravel()
            except Exception as e:
                logger.warning(f"LambdaRank prediction failed for {pred_date}: {e}", exc_info=True)
        
        # Ridge Stacking / MetaRankerStacker (LightGBM Ranker as second layer)
        if ridge_stacker is not None:
            try:
                # Check if stacker is fitted
                is_fitted = getattr(ridge_stacker, 'fitted_', False) or (
                    hasattr(ridge_stacker, 'lightgbm_model') and ridge_stacker.lightgbm_model is not None
                ) or (
                    hasattr(ridge_stacker, 'ridge_model') and ridge_stacker.ridge_model is not None
                )
                
                if not is_fitted:
                    logger.warning(f"Ridge/MetaRanker stacker not fitted for {pred_date}, skipping prediction")
                    preds_dict['ridge_stacking'] = np.full(len(X), np.nan)
                else:
                    # Create stacking features with MultiIndex (date, ticker)
                    stacking_features = pd.DataFrame(index=X.index)
                    # Add MultiIndex if not already present
                    if not isinstance(stacking_features.index, pd.MultiIndex):
                        stacking_features.index = pd.MultiIndex.from_arrays(
                            [[pred_date] * len(stacking_features), stacking_features.index],
                            names=['date', 'ticker']
                        )

                    # Add first-layer predictions
                    if 'elastic_net' in preds_dict:
                        stacking_features['pred_elastic'] = preds_dict['elastic_net'][:len(X)]
                    if 'xgboost' in preds_dict:
                        stacking_features['pred_xgb'] = preds_dict['xgboost'][:len(X)]
                    if 'catboost' in preds_dict:
                        stacking_features['pred_catboost'] = preds_dict['catboost'][:len(X)]
                    # REMOVED: LightGBM Ranker disabled from first layer
                    # if 'lightgbm_ranker' in preds_dict:
                    #     stacking_features['pred_lightgbm_ranker'] = preds_dict['lightgbm_ranker'][:len(X)]
                    if 'lambdarank' in preds_dict:
                        stacking_features['pred_lambdarank'] = preds_dict['lambdarank'][:len(X)]

                    # Ensure all required columns exist
                    expected_cols = getattr(ridge_stacker, 'actual_feature_cols_', None) or getattr(ridge_stacker, 'base_cols', [])
                    if not expected_cols:
                        # Fallback to default base columns
                        expected_cols = ['pred_catboost', 'pred_elastic', 'pred_xgb', 'pred_lambdarank']  # Removed 'pred_lightgbm_ranker'
                    
                    for col in expected_cols:
                        if col not in stacking_features.columns:
                            stacking_features[col] = 0.0
                    
                    # Select only expected columns in correct order
                    stacking_features = stacking_features[list(expected_cols)].copy()

                    # Predict using MetaRankerStacker or RidgeStacker
                    ridge_pred = ridge_stacker.predict(stacking_features)
                    
                    # Handle different return types
                    if isinstance(ridge_pred, pd.DataFrame):
                        # MetaRankerStacker returns DataFrame with 'score' column
                        if 'score' in ridge_pred.columns:
                            pred_values = ridge_pred['score'].values
                        else:
                            pred_values = ridge_pred.iloc[:, 0].values
                    elif isinstance(ridge_pred, pd.Series):
                        pred_values = ridge_pred.values
                    else:
                        pred_values = np.array(ridge_pred).ravel()
                    
                    # Ensure correct length
                    if len(pred_values) != len(X):
                        logger.warning(f"Ridge/MetaRanker prediction length ({len(pred_values)}) != X length ({len(X)}), adjusting")
                        if len(pred_values) < len(X):
                            pred_values = np.pad(pred_values, (0, len(X) - len(pred_values)), constant_values=np.nan)
                        else:
                            pred_values = pred_values[:len(X)]
                    
                    preds_dict['ridge_stacking'] = pred_values
                    logger.debug(f"✅ Ridge/MetaRanker prediction successful for {pred_date}: {len(pred_values)} predictions")
            except Exception as e:
                logger.error(f"❌ Ridge/MetaRanker Stacking prediction failed for {pred_date}: {e}", exc_info=True)
                # Always add ridge_stacking to preds_dict, even if prediction failed
                preds_dict['ridge_stacking'] = np.full(len(X), np.nan)
        
        # Save predictions (ensure one prediction per day)
        for model_name, pred_values in preds_dict.items():
            if model_name in all_results:
                n_preds = min(len(pred_values), len(tickers))
                # CRITICAL: Ensure pred_values is a numpy array and has correct length
                if isinstance(pred_values, (pd.Series, pd.DataFrame)):
                    pred_values = pred_values.values
                pred_values = np.asarray(pred_values)
                if len(pred_values) != len(tickers):
                    logger.warning(f"⚠️  {model_name}预测值长度({len(pred_values)})与ticker数量({len(tickers)})不匹配，截断或填充")
                    if len(pred_values) < len(tickers):
                        # Pad with NaN if predictions are shorter
                        pred_values = np.pad(pred_values, (0, len(tickers) - len(pred_values)), constant_values=np.nan)
                    else:
                        # Truncate if predictions are longer
                        pred_values = pred_values[:len(tickers)]
                
                pred_df = pd.DataFrame({
                    'date': pred_date,
                    'ticker': tickers[:n_preds],
                    'prediction': pred_values[:n_preds],
                    'actual': actual_target.values[:n_preds] if len(actual_target) >= n_preds else np.full(n_preds, np.nan)
                })
                all_results[model_name].append(pred_df)
                
                # EMA history tracking is done in apply_ema_smoothing function (see below)

    # Concatenate all predictions and apply EMA smoothing
    ema_history = {}  # Initialize EMA history dictionary
    for model_name, pred_list in all_results.items():
        if len(pred_list) > 0:
            all_results[model_name] = pd.concat(pred_list, axis=0, ignore_index=True)
            unique_dates = all_results[model_name]['date'].nunique()
            total_rows = len(all_results[model_name])
            logger.info(f"✅ {model_name}: {total_rows} 条预测, {unique_dates} 个唯一日期 (one prediction per day ✓)")
            
            # 🔧 Apply EMA smoothing (3-day EMA: 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2})
            # Check if EMA is disabled (--ema-top-n -1 means disable EMA completely)
            if hasattr(args, 'ema_top_n') and args.ema_top_n == -1:
                # EMA完全禁用：使用原始分数
                logger.info(f"📊 EMA smoothing DISABLED for {model_name} (using raw predictions)...")
                all_results[model_name]['prediction_smooth'] = all_results[model_name]['prediction']
            elif args.ema_top_n is not None and args.ema_top_n > 0:
                # Use Top300 filter version: only apply EMA to stocks in top N for consecutive days
                logger.info(f"📊 Applying EMA smoothing to {model_name} predictions (Top{args.ema_top_n} filter, min {args.ema_min_days} days)...")
                from scripts.apply_ema_smoothing_top300 import apply_ema_smoothing_top300_filter
                all_results[model_name] = apply_ema_smoothing_top300_filter(
                    all_results[model_name], 
                    model_name=model_name,
                    ema_history=ema_history,
                    weights=(0.6, 0.3, 0.1),
                    top_n=args.ema_top_n,
                    min_days_in_top=args.ema_min_days
                )
                # Log EMA coverage statistics
                if 'in_top300_3days' in all_results[model_name].columns:
                    ema_coverage = all_results[model_name]['in_top300_3days'].mean() * 100
                    logger.info(f"   EMA coverage: {ema_coverage:.2f}% of predictions applied EMA")
            else:
                # Use original version: apply EMA to all stocks
                logger.info(f"📊 Applying EMA smoothing to {model_name} predictions (all stocks)...")
                all_results[model_name] = apply_ema_smoothing(
                    all_results[model_name], 
                    model_name=model_name,
                    ema_history=ema_history,
                    weights=(0.6, 0.3, 0.1)
                )
            # Use smoothed predictions for ranking (or raw if EMA disabled)
            all_results[model_name]['prediction'] = all_results[model_name]['prediction_smooth'].fillna(all_results[model_name]['prediction'])
            if hasattr(args, 'ema_top_n') and args.ema_top_n == -1:
                logger.info(f"✅ Using raw predictions for {model_name} (EMA disabled)")
            else:
                logger.info(f"✅ EMA smoothing applied to {model_name}")
        else:
            all_results[model_name] = pd.DataFrame()
            logger.warning(f"⚠️  {model_name}: No predictions collected")
    
    # Additional check for ridge_stacking
    if 'ridge_stacking' in all_results:
        if all_results['ridge_stacking'].empty or len(all_results['ridge_stacking']) == 0:
            logger.error(f"❌ CRITICAL: ridge_stacking is empty after prediction loop!")
            logger.error(f"   Available models: {list(all_results.keys())}")
            logger.error(f"   Ridge stacker status: fitted={getattr(ridge_stacker, 'fitted_', False) if ridge_stacker else 'None'}")
            if ridge_stacker:
                logger.error(f"   Stacker type: {type(ridge_stacker).__name__}")
                logger.error(f"   Has lightgbm_model: {hasattr(ridge_stacker, 'lightgbm_model')}")
                logger.error(f"   Has ridge_model: {hasattr(ridge_stacker, 'ridge_model')}")
                if hasattr(ridge_stacker, 'lightgbm_model'):
                    logger.error(f"   lightgbm_model is None: {ridge_stacker.lightgbm_model is None}")
                if hasattr(ridge_stacker, 'ridge_model'):
                    logger.error(f"   ridge_model is None: {ridge_stacker.ridge_model is None}")

    # Calculate metrics with HAC corrections
    logger.info("=" * 80)
    logger.info("📊 Calculating metrics with HAC corrections")
    logger.info("=" * 80)

    report_rows = []
    for model_name, predictions in all_results.items():
        # Check if predictions is a DataFrame (not a list) and not empty
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            logger.warning(f"Skipping {model_name}: predictions not available or empty")
            continue
        
        # Save predictions DataFrame for diagnosis
        predictions_file = run_dir / f"{model_name}_predictions_diagnosis.csv"
        predictions.to_csv(predictions_file, index=False)
        logger.info(f"Saved predictions for diagnosis: {predictions_file}")
        
        logger.info(f"\nAnalyzing {model_name}...")
        # DEBUG: Check prediction value ranges to ensure they're different
        pred_stats = predictions['prediction'].describe()
        logger.info(f"  Prediction stats: min={pred_stats['min']:.6f}, max={pred_stats['max']:.6f}, mean={pred_stats['mean']:.6f}, std={pred_stats['std']:.6f}")

        # Limit IC computation to Top-40 rows per date
        filtered_for_ic = _limit_predictions_to_top_k_per_date(predictions, top_k=40)
        if filtered_for_ic is predictions:
            logger.info('  Top-40 filter left dataset unchanged for IC calculation')
        else:
            logger.info('  Top-40 filter applied for IC calculation: %d -> %d rows', len(predictions), len(filtered_for_ic))

        
        # 🔥 Calculate metrics with HAC (FIXED: 先按日聚合IC，再对日度IC序列做HAC)
        # 确保predictions DataFrame有date列
        if 'date' not in filtered_for_ic.columns:
            logger.error(f"❌ {model_name}: predictions DataFrame缺少'date'列，无法计算HAC")
            ic_result = {'IC': np.nan, 'IC_pvalue': np.nan, 'IC_tstat': np.nan, 'IC_se_hac': np.nan, 'note': 'Missing date column'}
            rank_ic_result = {'IC': np.nan, 'IC_pvalue': np.nan, 'IC_tstat': np.nan, 'IC_se_hac': np.nan, 'note': 'Missing date column'}
        else:
            if hac_method == "newey-west":
                # IC: 使用Pearson correlation
                ic_result = calculate_newey_west_hac_ic(
                    filtered_for_ic,
                    lag=hac_lag,
                    use_rank=False
                )
                # Rank IC: 使用rank correlation
                rank_ic_result = calculate_newey_west_hac_ic(
                    filtered_for_ic,
                    lag=hac_lag,
                    use_rank=True
                )
            else:  # hansen-hodrick
                # IC: 使用Pearson correlation
                ic_result = calculate_hansen_hodrick_se_ic(
                    filtered_for_ic,
                    horizon=horizon,
                    use_rank=False
                )
                # Rank IC: 使用rank correlation
                rank_ic_result = calculate_hansen_hodrick_se_ic(
                    filtered_for_ic,
                    horizon=horizon,
                    use_rank=True
                )
        
        # Calculate metrics: 
        # - Average/Median returns: daily calculation (for prediction quality assessment)
        # - Accumulated return, max drawdown, Sharpe, graphs: non-overlapping (for backtest)
        horizon = int(args.horizon_days)
        
        logger.info(f"  Calculating daily average/median returns (for prediction quality)")
        # Daily calculation for average/median returns (prediction quality metrics)
        group_summary_daily, _ = calculate_group_returns_standalone(
            predictions, top_n=30, bottom_n=30, cost_bps=float(args.cost_bps or 0.0)
        )
        
        # Daily bucket returns for average/median (prediction quality metrics)
        top_buckets = [(1, 10), (5, 15), (11, 20), (21, 30)]  # 🔧 Added Top5-Top15 bucket
        bottom_buckets = [(1, 10), (11, 20), (21, 30)]
        bucket_summary_daily, _ = calculate_bucket_returns_standalone(
            predictions, top_buckets=top_buckets, bottom_buckets=bottom_buckets, cost_bps=float(args.cost_bps or 0.0)
        )
        
        logger.info(f"  Calculating non-overlapping backtest metrics (hold {horizon} days)")
        # Non-overlapping calculation for backtest metrics (accumulated return, max drawdown, Sharpe, graphs)
        group_ts = calculate_group_returns_hold10d_nonoverlap(
            predictions, top_n=30, horizon_days=horizon, cost_bps=float(args.cost_bps or 0.0), start_offset=0
        )
        
        # Save non-overlapping timeseries for this model (for accumulated return calculation)
        if not group_ts.empty:
            ts_file = run_dir / f"{model_name}_top30_nonoverlap_timeseries.csv"
            group_ts.to_csv(ts_file, index=False)
            logger.info(f"  Saved non-overlapping timeseries: {ts_file.name} ({len(group_ts)} periods)")
        
        # Calculate Sharpe and win_rate from non-overlapping backtest
        if group_ts.empty:
            backtest_summary = {}
        else:
            periods_per_year = 252.0 / horizon
            net_mean_series = group_ts['top_return_net'].dropna()
            net_median_series = group_ts['top_return_net_median'].dropna()
            if len(net_mean_series) > 1 and net_mean_series.std() > 0:
                backtest_summary = {
                    'top_sharpe_net': float((net_mean_series.mean() / net_mean_series.std()) * np.sqrt(periods_per_year)),
                    'win_rate': float((net_mean_series > 0).mean()),
                }
            else:
                backtest_summary = {'top_sharpe_net': np.nan, 'win_rate': np.nan}
            if len(net_median_series) > 1 and net_median_series.std() > 0:
                backtest_summary['top_sharpe_net_median'] = float((net_median_series.mean() / net_median_series.std()) * np.sqrt(periods_per_year))
                backtest_summary['win_rate_median'] = float((net_median_series > 0).mean())
            else:
                backtest_summary['top_sharpe_net_median'] = np.nan
                backtest_summary['win_rate_median'] = np.nan
        
        # Combine: daily averages/medians + non-overlapping backtest metrics
        group_summary = {
            **group_summary_daily,  # avg_top_return, avg_top_return_net, median_top_return, median_top_return_net, avg_top_turnover, avg_top_cost
            **backtest_summary,  # top_sharpe_net, top_sharpe_net_median, win_rate, win_rate_median
        }
        bucket_summary = bucket_summary_daily  # Use daily bucket returns for averages/medians
        
        # Build report row
        hac_disclosure_note = (
            f"非重叠持有期回测 (non-overlapping hold periods): "
            f"每{horizon}个交易日再平衡一次，持有期收益按期复利。"
            f"累计收益、回撤、Sharpe和图表基于非重叠回测。"
            f"平均收益和中位数收益基于每日预测质量评估。"
            f"Statistical inference uses HAC corrections for IC/Rank IC (overlapping observations)."
        )
        
        row = {
            'Model': model_name,
            'N_Predictions': len(predictions),
            'IC': ic_result['IC'],
            'IC_pvalue': ic_result['IC_pvalue'],
            'IC_tstat': ic_result['IC_tstat'],
            'IC_se_hac': ic_result['IC_se_hac'],
            'Rank_IC': rank_ic_result['IC'],
            'Rank_IC_pvalue': rank_ic_result['IC_pvalue'],
            'Rank_IC_tstat': rank_ic_result['IC_tstat'],
            'Rank_IC_se_hac': rank_ic_result['IC_se_hac'],
            'MSE': mean_squared_error(predictions['actual'], predictions['prediction']),
            'MAE': mean_absolute_error(predictions['actual'], predictions['prediction']),
            'R2': r2_score(predictions['actual'], predictions['prediction']),
            **group_summary,  # 包含 top_sharpe_net, top_sharpe_net_median, win_rate, win_rate_median
            **bucket_summary,
            'hac_method': hac_method,
            'hac_lag': hac_lag,
            'note': hac_disclosure_note
        }
        
        report_rows.append(row)
        logger.info(f"  IC: {ic_result['IC']:.4f} (t-stat={ic_result['IC_tstat']:.2f}, SE={ic_result['IC_se_hac']:.6f})")
        logger.info(f"  Rank IC: {rank_ic_result['IC']:.4f} (t-stat={rank_ic_result['IC_tstat']:.2f}, SE={rank_ic_result['IC_se_hac']:.6f})")
        
        # 🔥 验证修复效果：检查SE是否合理
        if ic_result['IC_se_hac'] > 100 or rank_ic_result['IC_se_hac'] > 100:
            logger.warning(f"  ⚠️  HAC SE异常大 (IC SE={ic_result['IC_se_hac']:.2f}, Rank IC SE={rank_ic_result['IC_se_hac']:.2f})")
            logger.warning(f"  ⚠️  如果SE>100，说明HAC计算可能还有问题")
        else:
            logger.info(f"  ✅ HAC SE合理 (IC SE={ic_result['IC_se_hac']:.6f}, Rank IC SE={rank_ic_result['IC_se_hac']:.6f})")
        
        # 🔥 验证统计口径一致性：IC和Rank IC的t-stat应该接近
        ic_tstat_diff = abs(ic_result['IC_tstat'] - rank_ic_result['IC_tstat'])
        if ic_tstat_diff > 5:
            logger.warning(f"  ⚠️  IC和Rank IC的t-stat差异较大 ({ic_tstat_diff:.2f})，可能统计口径不一致")
        else:
            logger.debug(f"  ✅ IC和Rank IC的t-stat接近 (差异={ic_tstat_diff:.2f})")
        
        logger.info(f"  Note: {hac_disclosure_note}")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(run_dir / "report_df.csv", index=False, encoding="utf-8")
    logger.info(f"\n✅ Generated report_df with {len(report_df)} models (HAC-corrected)")
    
    # REMOVED: EMA smoothing comparison backtest - EMA smoothing has been moved to live prediction
    # ========== END Top10持有10天回测 ==========
    # ========== END STANDALONE PREDICTION LOOP ==========

    # Benchmark returns (yfinance fallback) computed once on test-window rebalance dates of primary model.
    primary_model = str(args.model).strip()
    
    # If primary_model is empty, try to find a valid model
    if primary_model not in all_results or not isinstance(all_results[primary_model], pd.DataFrame) or all_results[primary_model].empty:
        logger.warning(f"⚠️  Primary model '{primary_model}' missing/empty, checking available models...")
        # Try to find a valid model
        valid_models = [m for m in all_results.keys() 
                       if isinstance(all_results[m], pd.DataFrame) and not all_results[m].empty]
        if valid_models:
            logger.warning(f"   Available valid models: {valid_models}")
            # Prefer ridge_stacking if available, else use first valid model
            if 'ridge_stacking' in valid_models:
                primary_model = 'ridge_stacking'
                logger.info(f"   Using 'ridge_stacking' as primary model")
            else:
                primary_model = valid_models[0]
                logger.info(f"   Using '{primary_model}' as primary model (fallback)")
        else:
            raise RuntimeError(f"Model '{primary_model}' missing/empty in backtest results. Available keys: {list(all_results.keys())}, but all are empty!")
    
    if primary_model not in all_results or all_results[primary_model].empty:
        raise RuntimeError(f"Model '{primary_model}' is still empty after fallback check. Available: {list(all_results.keys())}")

    top_n = int(args.top_n)
    bench = str(args.benchmark).upper().strip()
    # Use primary model's dates to fetch benchmark (non-overlapping mode)
    _preds_for_bench = all_results[primary_model]
    
    # Get rebalance dates from non-overlapping function
    _tmp_ts = calculate_group_returns_hold10d_nonoverlap(
        _preds_for_bench, top_n=top_n, horizon_days=horizon, cost_bps=float(args.cost_bps or 0.0), start_offset=0
    )
    if _tmp_ts.empty:
        raise RuntimeError("Group return time series is empty on test window for benchmark date extraction.")
    # Benchmark should also be 10-day returns (non-overlapping)
    bench_ret = _compute_benchmark_tplus_from_yfinance(bench, _tmp_ts["date"], horizon, logger)

    # Export TopN vs benchmark for multiple models (if provided), else only for --model
    models_to_export = [primary_model] if not args.models else [str(m).strip() for m in args.models if str(m).strip()]
    summaries = []
    for m in models_to_export:
        if m not in all_results or not isinstance(all_results[m], pd.DataFrame) or all_results[m].empty:
            logger.warning("Skipping model=%s (missing/empty in all_results). Available=%s", m, list(all_results.keys()))
            continue
        summaries.append(
            _write_model_topn_vs_benchmark(
                run_dir=run_dir,
                model_name=m,
                preds=all_results[m],
                top_n=top_n,
                horizon=horizon,
                bench=bench,
                bench_ret=bench_ret,
                cost_bps=float(args.cost_bps or 0.0),
                logger=logger,
                rebalance_mode="nonoverlap",  # Always use non-overlapping for backtest metrics
            )
        )

    # Keep legacy filename for ridge_stacking for backward compatibility (if present)
    for s in summaries:
        if s.get("model") == "ridge_stacking":
            try:
                src = run_dir / f"ridge_stacking_top{top_n}_timeseries.csv"
                if src.exists():
                    src.replace(run_dir / "ridge_top20_timeseries.csv")
            except Exception:
                pass

    # Write explicit OOS metrics (so downstream summaries don't need to recompute)
    # NOTE: All metrics here are computed only from saved test-window series in this script.
    metrics = {
        "snapshot_id": str(snapshot_id),
        "model": primary_model,
        "top_n": top_n,
        "horizon_days": horizon,
        "split": split,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "rebalance_mode": "nonoverlap",  # Always use non-overlapping for backtest metrics
        "max_weeks": int(args.max_weeks),
        "benchmark": bench,
        "cost_bps": _safe_float(args.cost_bps or 0.0),
        "n_test_rebalances": int(_tmp_ts.shape[0]),
    }

    # Add predictive + backtest summary metrics from standalone report_df (test window).
    # This is the authoritative source for IC/RankIC/MSE/MAE/R2 and (gross/net) avg_top_return at Top-30.
    # IMPORTANT: Daily rebalancing uses HAC corrections (Newey-West or Hansen-Hodrick) for overlapping observations
    try:
        if isinstance(report_df, pd.DataFrame) and "Model" in report_df.columns:
            rr = report_df.loc[report_df["Model"].astype(str) == str(primary_model)].head(1)
            if not rr.empty:
                row = rr.iloc[0].to_dict()
                metrics.update(
                    {
                        # predictive metrics (unitless / model scale)
                        "IC": _safe_float(row.get("IC")),
                        "IC_pvalue": _safe_float(row.get("IC_pvalue")),
                        "Rank_IC": _safe_float(row.get("Rank_IC")),
                        "Rank_IC_pvalue": _safe_float(row.get("Rank_IC_pvalue")),
                        "MSE": _safe_float(row.get("MSE")),
                        "MAE": _safe_float(row.get("MAE")),
                        "R2": _safe_float(row.get("R2")),
                        # HAC-corrected statistics (when rebalance_mode='daily')
                        "IC_tstat": _safe_float(row.get("IC_tstat")),
                        "IC_se_hac": _safe_float(row.get("IC_se_hac")),
                        "Rank_IC_tstat": _safe_float(row.get("Rank_IC_tstat")),
                        "Rank_IC_se_hac": _safe_float(row.get("Rank_IC_se_hac")),
                        "hac_note": str(row.get("note", "")) if row.get("note") else "",
                        # backtest summary (Top-30 portfolio, per rebalance period; return units)
                        # Mean metrics
                        "avg_top_return": _safe_float(row.get("avg_top_return")),
                        "avg_top_return_net": _safe_float(row.get("avg_top_return_net")),
                        "avg_bottom_return": _safe_float(row.get("avg_bottom_return", float("nan"))),
                        # Median metrics
                        "median_top_return": _safe_float(row.get("median_top_return", float("nan"))),
                        "median_top_return_net": _safe_float(row.get("median_top_return_net", float("nan"))),
                        "median_bottom_return": _safe_float(row.get("median_bottom_return", float("nan"))),
                        # Other metrics
                        "avg_top_turnover": _safe_float(row.get("avg_top_turnover")),
                        "avg_top_cost": _safe_float(row.get("avg_top_cost")),
                        "win_rate": _safe_float(row.get("win_rate")),
                        "win_rate_median": _safe_float(row.get("win_rate_median", float("nan"))),
                        "top_sharpe_net": _safe_float(row.get("top_sharpe_net", float("nan"))),
                        "top_sharpe_net_median": _safe_float(row.get("top_sharpe_net_median", float("nan"))),
                        # Mean and median of top bucket (Top 1-10) T+10 return
                        "avg_top_bucket_t5_return": _safe_float(row.get("avg_top_1_10_return", float("nan"))),
                        "avg_top_bucket_t5_return_net": _safe_float(row.get("avg_top_1_10_return_net", float("nan"))),
                        "median_top_bucket_t5_return": _safe_float(row.get("median_top_1_10_return_from_median", float("nan"))),
                        "median_top_bucket_t5_return_net": _safe_float(row.get("median_top_1_10_return_net_from_median", float("nan"))),
                        "long_short_sharpe": _safe_float(row.get("long_short_sharpe")),
                    }
                )
    except Exception as e:
        logger.warning("Could not merge report_df metrics into oos_metrics: %s", e)
    # Add per-model TopN summaries to oos_metrics_all_models.csv (percent units) and keep oos_metrics.csv for primary model.
    if summaries:
        all_oos = pd.DataFrame(summaries)
        all_oos.to_csv(run_dir / "oos_topn_vs_benchmark_all_models.csv", index=False, encoding="utf-8")

    (run_dir / "oos_metrics.json").write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(run_dir / "oos_metrics.csv", index=False, encoding="utf-8")

    # Calculate bucket returns for all models (top 0-10, 10-20, 20-30; bottom 0-10, 10-20, 20-30)
    logger.info("=" * 80)
    logger.info("Calculating bucket returns for all models...")
    logger.info("=" * 80)
    
    # Define buckets: top 1-10, 5-15, 11-20, 21-30; bottom 1-10, 11-20, 21-30
    top_buckets = [(1, 10), (5, 15), (11, 20), (21, 30)]  # 🔧 Added Top5-Top15 bucket
    bottom_buckets = [(1, 10), (11, 20), (21, 30)]  # Bottom buckets are counted from the end
    
    # Calculate bucket returns and generate plots for each model
    for model_name in models_to_export:
        if model_name not in all_results or not isinstance(all_results[model_name], pd.DataFrame) or all_results[model_name].empty:
            logger.warning("Skipping bucket analysis for model=%s (missing/empty)", model_name)
            continue
        
        logger.info("Processing bucket returns for model: %s", model_name)
        preds = all_results[model_name]
        
        # Calculate bucket returns
        bucket_summary, bucket_ts = calculate_bucket_returns_standalone(
            predictions=preds,
            top_buckets=top_buckets,
            bottom_buckets=bottom_buckets,
            cost_bps=float(args.cost_bps or 0.0)
        )
        
        if bucket_ts.empty:
            logger.warning("Bucket time series is empty for model=%s", model_name)
            continue
        
        # Add benchmark returns
        bucket_ts["date"] = pd.to_datetime(bucket_ts["date"])
        bucket_ts = bucket_ts.sort_values("date")
        bucket_ts["benchmark_return"] = bucket_ts["date"].map(
            lambda d: float(bench_ret.get(d, float("nan"))) if hasattr(bench_ret, "get") else float("nan")
        )
        
        # Convert to percentage
        # Ensure all bucket columns exist (including top_5_15_return)
        bucket_cols = [f"top_{a}_{b}_return" for (a, b) in top_buckets]
        bucket_cols += [f"top_{a}_{b}_return_net" for (a, b) in top_buckets]
        bucket_cols += [f"bottom_{a}_{b}_return" for (a, b) in bottom_buckets]
        bucket_cols += ["benchmark_return"]
        
        # Also add simplified columns (without _mean suffix) for backward compatibility
        # This ensures top_5_15_return is available even if only top_5_15_return_mean exists
        for (a, b) in top_buckets:
            mean_col = f"top_{a}_{b}_return_mean"
            if mean_col in bucket_ts.columns:
                if f"top_{a}_{b}_return" not in bucket_ts.columns:
                    bucket_ts[f"top_{a}_{b}_return"] = bucket_ts[mean_col]
            net_mean_col = f"top_{a}_{b}_return_net_mean"
            if net_mean_col in bucket_ts.columns:
                if f"top_{a}_{b}_return_net" not in bucket_ts.columns:
                    bucket_ts[f"top_{a}_{b}_return_net"] = bucket_ts[net_mean_col]
        
        for col in bucket_cols:
            if col in bucket_ts.columns:
                bucket_ts[col] = pd.to_numeric(bucket_ts[col], errors="coerce") * 100.0
        
        # Calculate cumulative returns
        def _cum_pct(series_pct: pd.Series) -> pd.Series:
            r = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
            return (1.0 + r).cumprod() - 1.0
        
        for (a, b) in top_buckets:
            bucket_ts[f"cum_top_{a}_{b}_return"] = _cum_pct(bucket_ts[f"top_{a}_{b}_return"]) * 100.0
            bucket_ts[f"cum_top_{a}_{b}_return_net"] = _cum_pct(bucket_ts[f"top_{a}_{b}_return_net"]) * 100.0
        
        for (a, b) in bottom_buckets:
            bucket_ts[f"cum_bottom_{a}_{b}_return"] = _cum_pct(bucket_ts[f"bottom_{a}_{b}_return"]) * 100.0
        
        bucket_ts["cum_benchmark_return"] = _cum_pct(bucket_ts["benchmark_return"]) * 100.0
        
        # Save bucket CSV
        bucket_csv = run_dir / f"{model_name}_bucket_returns.csv"
        bucket_ts.to_csv(bucket_csv, index=False, encoding="utf-8")
        logger.info("Saved bucket CSV: %s", bucket_csv)
        
        # Save bucket summary
        bucket_summary_df = pd.DataFrame([bucket_summary])
        bucket_summary_csv = run_dir / f"{model_name}_bucket_summary.csv"
        bucket_summary_df.to_csv(bucket_summary_csv, index=False, encoding="utf-8")
        logger.info("Saved bucket summary: %s", bucket_summary_csv)
        
        # Generate individual plot for this model
        _plot_bucket_returns(
            bucket_ts=bucket_ts,
            model_name=model_name,
            top_buckets=top_buckets,
            bottom_buckets=bottom_buckets,
            bench=bench,
            horizon=horizon,
            cost_bps=float(args.cost_bps or 0.0),
            run_dir=run_dir,
            logger=logger,
        )
        
        # Log bucket summary - Mean and Median
        logger.info("[%s] Bucket Returns - Mean (%%):", model_name)
        # Log mean of top bucket (Top 1-10) T+10 return
        avg_top_bucket_mean = bucket_summary.get("avg_top_1_10_return", float("nan"))
        avg_top_bucket_net_mean = bucket_summary.get("avg_top_1_10_return_net", float("nan"))
        logger.info("  Top 1-10 (mean): gross=%.4f%%, net=%.4f%%", 
                   float(avg_top_bucket_mean) * 100.0 if pd.notna(avg_top_bucket_mean) else float("nan"),
                   float(avg_top_bucket_net_mean) * 100.0 if pd.notna(avg_top_bucket_net_mean) else float("nan"))
        for (a, b) in top_buckets:
            avg_gross_mean = bucket_summary.get(f"avg_top_{a}_{b}_return", float("nan"))
            avg_net_mean = bucket_summary.get(f"avg_top_{a}_{b}_return_net", float("nan"))
            logger.info("  Top %d-%d (mean): gross=%.4f%%, net=%.4f%%", a, b, 
                       float(avg_gross_mean) * 100.0 if pd.notna(avg_gross_mean) else float("nan"),
                       float(avg_net_mean) * 100.0 if pd.notna(avg_net_mean) else float("nan"))
        for (a, b) in bottom_buckets:
            avg_bottom_mean = bucket_summary.get(f"avg_bottom_{a}_{b}_return", float("nan"))
            logger.info("  Bottom %d-%d (mean): %.4f%%", a, b,
                       float(avg_bottom_mean) * 100.0 if pd.notna(avg_bottom_mean) else float("nan"))
        
        logger.info("[%s] Bucket Returns - Median (%%):", model_name)
        # Log median of top bucket (Top 1-10) T+10 return
        median_top_bucket = bucket_summary.get("median_top_1_10_return_from_median", float("nan"))
        median_top_bucket_net = bucket_summary.get("median_top_1_10_return_net_from_median", float("nan"))
        logger.info("  Top 1-10 (median): gross=%.4f%%, net=%.4f%%", 
                   float(median_top_bucket) * 100.0 if pd.notna(median_top_bucket) else float("nan"),
                   float(median_top_bucket_net) * 100.0 if pd.notna(median_top_bucket_net) else float("nan"))
        for (a, b) in top_buckets:
            median_gross = bucket_summary.get(f"median_top_{a}_{b}_return_from_median", float("nan"))
            median_net = bucket_summary.get(f"median_top_{a}_{b}_return_net_from_median", float("nan"))
            logger.info("  Top %d-%d (median): gross=%.4f%%, net=%.4f%%", a, b, 
                       float(median_gross) * 100.0 if pd.notna(median_gross) else float("nan"),
                       float(median_net) * 100.0 if pd.notna(median_net) else float("nan"))
        for (a, b) in bottom_buckets:
            median_bottom = bucket_summary.get(f"median_bottom_{a}_{b}_return_from_median", float("nan"))
            logger.info("  Bottom %d-%d (median): %.4f%%", a, b,
                       float(median_bottom) * 100.0 if pd.notna(median_bottom) else float("nan"))

    # NOTE: plotting is handled per-model in _write_model_topn_vs_benchmark().
    # Avoid duplicate single-model plotting here (which previously referenced an out-of-scope `out`).

    # NOTE: per-model OOS return logging is handled in _write_model_topn_vs_benchmark().
    
    # Calculate Top 5-15 10-day rebalance accumulated return curve for all models
    logger.info("=" * 80)
    logger.info("Calculating Top 5-15 10-day rebalance accumulated return curve for all models...")
    logger.info("=" * 80)
    
    # Use models_to_export to calculate for all models
    for model_name in models_to_export:
        if model_name not in all_results or not isinstance(all_results[model_name], pd.DataFrame) or all_results[model_name].empty:
            logger.warning(f"⚠️  Skipping Top 5-15 accumulated return calculation for '{model_name}' (missing/empty)")
            continue
        
        try:
            # Calculate Top 5-15 accumulated return (using bucket 5-15)
            top5_15_accumulated_ts = calc_top10_accumulated_10d_rebalance(
                predictions_df=all_results[model_name],
                top_n=15,  # Use 15 to get top 15, then filter to 5-15
                step=10,
                out_dir=run_dir,
                model_name=model_name,
                logger=logger,
                bucket_range=(5, 15)  # Specify bucket range 5-15
            )
            if not top5_15_accumulated_ts.empty:
                logger.info(f"✅ Successfully calculated Top 5-15 10-day rebalance accumulated return for {model_name}")
            else:
                logger.warning(f"⚠️  Top 5-15 accumulated return calculation returned empty DataFrame for {model_name}")
                # Fallback: try to use bucket returns if available
                bucket_file = run_dir / f"{model_name}_bucket_returns.csv"
                if bucket_file.exists():
                    bucket_df = pd.read_csv(bucket_file)
                    # Check for top_5_15_return or top_5_15_return_mean
                    col_name = None
                    for col in ['top_5_15_return', 'top_5_15_return_mean']:
                        if col in bucket_df.columns:
                            col_name = col
                            break
                    if col_name:
                        logger.info(f"Using bucket returns for Top 5-15 accumulated return for {model_name}")
                        # Calculate cumulative from bucket returns
                        data = bucket_df[col_name].dropna() / 100.0
                        if len(data) > 0:
                            cum_data = (1 + data).cumprod() - 1
                            result_df = pd.DataFrame({
                                'date': bucket_df['date'],
                                'top_gross_return': data.values,
                                'acc_value': (1 + data).cumprod().values,
                                'acc_return': cum_data.values
                            })
                            result_df.to_csv(run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv", index=False)
                            logger.info(f"✅ Saved Top 5-15 accumulated return from bucket data for {model_name}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to calculate Top 5-15 accumulated return for {model_name}: {e}", exc_info=True)
    
    # Generate results_summary_for_word_doc.json with HAC-corrected statistics for all models
    logger.info("=" * 80)
    logger.info("Generating results_summary_for_word_doc.json with HAC corrections...")
    logger.info("=" * 80)
    
    results_summary = {}
    # Always use HAC for IC/Rank IC (overlapping observations for prediction quality)
    # But backtest metrics (accumulated return, max drawdown, Sharpe, graphs) use non-overlapping
    use_hac = True  # IC/Rank IC always need HAC corrections (overlapping observations)
    
    # Determine HAC method and lag (must be defined before use)
    hac_method = getattr(args, 'hac_method', 'newey-west')
    if hac_method == "newey-west":
        hac_lag = getattr(args, 'hac_lag', None) or max(10, 2 * horizon)
        hac_disclosure = (
            f"IC/Rank IC基于重叠观测 (overlapping observations), "
            f"使用Newey-West HAC校正 (lag={hac_lag}, lag≥10). "
            f"回测指标（累计收益、回撤、Sharpe、图表）基于非重叠持有期回测。"
            f"Statistical inference uses Newey-West (1987) heteroskedasticity and "
            f"autocorrelation consistent standard errors with {hac_lag} lags for IC/Rank IC."
        )
    else:  # hansen-hodrick
        hac_lag = getattr(args, 'hac_lag', None) or max(horizon - 1, 1)
        hac_disclosure = (
            f"IC/Rank IC基于重叠观测 (overlapping observations), "
            f"使用Hansen-Hodrick标准误 (lag={hac_lag} for {horizon}-day horizon). "
            f"回测指标（累计收益、回撤、Sharpe、图表）基于非重叠持有期回测。"
            f"Statistical inference uses Hansen-Hodrick (1980) standard errors "
            f"for {horizon}-period overlapping returns for IC/Rank IC."
        )
    logger.info(f"HAC Correction: {hac_disclosure}")
    
    # Extract metrics for all models from report_df
    if isinstance(report_df, pd.DataFrame) and "Model" in report_df.columns:
        for model_name in models_to_export:
            if model_name not in all_results or not isinstance(all_results[model_name], pd.DataFrame) or all_results[model_name].empty:
                continue
            
            model_row = report_df.loc[report_df["Model"].astype(str) == str(model_name)].head(1)
            if model_row.empty:
                logger.warning(f"No metrics found in report_df for model: {model_name}")
                continue
            
            row_dict = model_row.iloc[0].to_dict()
            
            # Extract bucket returns for this model
            model_bucket_summary = {}
            if model_name in [m for m in models_to_export if m in all_results]:
                try:
                    preds = all_results[model_name]
                    bucket_summary, _ = calculate_bucket_returns_standalone(
                        predictions=preds,
                        top_buckets=[(1, 10), (11, 20), (21, 30)],
                        bottom_buckets=[(1, 10), (11, 20), (21, 30)],
                        cost_bps=float(args.cost_bps or 0.0),
                    )
                    model_bucket_summary = bucket_summary
                except Exception as e:
                    logger.warning(f"Failed to calculate bucket returns for {model_name}: {e}")
            
            # Build model summary with HAC statistics
            model_summary = {
                "metrics": {
                    "IC": _safe_float(row_dict.get("IC")),
                    "IC_pvalue": _safe_float(row_dict.get("IC_pvalue")),
                    "Rank_IC": _safe_float(row_dict.get("Rank_IC")),
                    "Rank_IC_pvalue": _safe_float(row_dict.get("Rank_IC_pvalue")),
                    "MSE": _safe_float(row_dict.get("MSE")),
                    "MAE": _safe_float(row_dict.get("MAE")),
                    "R2": _safe_float(row_dict.get("R2")),
                },
                "returns": {
                    # Mean metrics
                    "avg_top_return": _safe_float(row_dict.get("avg_top_return")),
                    "avg_top_return_net": _safe_float(row_dict.get("avg_top_return_net")),
                    "avg_bottom_return": _safe_float(row_dict.get("avg_bottom_return", float("nan"))),
                    # Median metrics
                    "median_top_return": _safe_float(row_dict.get("median_top_return", float("nan"))),
                    "median_top_return_net": _safe_float(row_dict.get("median_top_return_net", float("nan"))),
                    "median_bottom_return": _safe_float(row_dict.get("median_bottom_return", float("nan"))),
                    # Other metrics
                    "long_short_sharpe": _safe_float(row_dict.get("long_short_sharpe")),
                    "win_rate": _safe_float(row_dict.get("win_rate")),
                    "win_rate_median": _safe_float(row_dict.get("win_rate_median", float("nan"))),
                    "top_sharpe_net": _safe_float(row_dict.get("top_sharpe_net", float("nan"))),
                    "top_sharpe_net_median": _safe_float(row_dict.get("top_sharpe_net_median", float("nan"))),
                    # Mean and median of top bucket (Top 1-10) T+10 return
                    "avg_top_bucket_t5_return": _safe_float(model_bucket_summary.get("avg_top_1_10_return", float("nan"))),
                    "avg_top_bucket_t5_return_net": _safe_float(model_bucket_summary.get("avg_top_1_10_return_net", float("nan"))),
                    "median_top_bucket_t5_return": _safe_float(model_bucket_summary.get("median_top_1_10_return_from_median", float("nan"))),
                    "median_top_bucket_t5_return_net": _safe_float(model_bucket_summary.get("median_top_1_10_return_net_from_median", float("nan"))),
                },
                "bucket_summary": model_bucket_summary,
            }
            
            # Add HAC statistics if available (when rebalance_mode='daily')
            if use_hac:
                model_summary["metrics"].update({
                    "IC_tstat": _safe_float(row_dict.get("IC_tstat")),
                    "IC_se_hac": _safe_float(row_dict.get("IC_se_hac")),
                    "Rank_IC_tstat": _safe_float(row_dict.get("Rank_IC_tstat")),
                    "Rank_IC_se_hac": _safe_float(row_dict.get("Rank_IC_se_hac")),
                    "note": str(row_dict.get("note", hac_disclosure)) if row_dict.get("note") else hac_disclosure,
                })
                logger.info(f"[{model_name}] HAC-corrected IC: {model_summary['metrics'].get('IC', np.nan):.4f} "
                           f"(t-stat={model_summary['metrics'].get('IC_tstat', np.nan):.2f}, "
                           f"SE={model_summary['metrics'].get('IC_se_hac', np.nan):.4f})")
            
            results_summary[model_name] = model_summary
    
    # Add metadata with explicit HAC disclosure
    # Note: hac_method and hac_lag are already defined above (line 1026-1044)
    results_summary["metadata"] = {
        "snapshot_id": str(snapshot_id),
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "horizon_days": horizon,
        "rebalance_mode": "nonoverlap",  # Backtest metrics use non-overlapping
        "n_test_rebalances": int(_tmp_ts.shape[0]) if not _tmp_ts.empty else 0,
        "hac_correction_applied": use_hac,
        "hac_method": hac_method if use_hac else None,
        "hac_lag": hac_lag if use_hac else None,
        "overlapping_observations": use_hac,
        "disclosure_note": (
            f"IC/Rank IC基于重叠观测 (overlapping observations), "
            f"使用{'Newey-West HAC' if hac_method == 'newey-west' else 'Hansen-Hodrick'}标准误 (lag={hac_lag}). "
            f"回测指标（累计收益、回撤、Sharpe、图表）基于非重叠持有期回测（每{horizon}个交易日再平衡一次）。"
            f"平均收益和中位数收益基于每日预测质量评估。"
            f"Statistical inference uses HAC corrections for IC/Rank IC (overlapping observations). "
            f"Backtest metrics use non-overlapping hold periods ({horizon} days)."
        ),
    }
    
    # Save results summary
    summary_file = run_dir / "results_summary_for_word_doc.json"
    summary_file.write_text(json.dumps(results_summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"✅ Saved results summary: {summary_file}")
    
    # Generate complete metrics report
    logger.info("=" * 80)
    logger.info("Generating complete metrics report...")
    logger.info("=" * 80)
    _generate_complete_metrics_report(
        run_dir, 
        models_to_export, 
        logger,
        ema_top_n=getattr(args, 'ema_top_n', None),
        ema_min_days=getattr(args, 'ema_min_days', 3)
    )
    
    logger.info("Saved outputs: %s", run_dir)
    return 0


def _generate_complete_metrics_report(
    run_dir: Path,
    models: List[str],
    logger: logging.Logger,
    ema_top_n: Optional[int] = None,
    ema_min_days: int = 3,
) -> None:
    """
    Generate complete metrics report for all models (Overlap and Non-Overlap metrics).
    
    This function reads the generated CSV files and calculates all metrics as specified
    in EWMA_IMPLEMENTATION_SUMMARY.md:
    - Overlap metrics (daily observations): avg, median, std, win rate, Sharpe
    - Non-Overlap metrics (10-day periods): avg, median, std, win rate, cumulative return,
      max drawdown, annualized return, Sharpe
    """
    logger.info("=" * 80)
    logger.info("完整指标报告（EWMA平滑后）")
    logger.info("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("完整指标报告（EWMA平滑后）")
    report_lines.append("=" * 80)
    
    for model_name in models:
        logger.info(f"\n【{model_name.upper()}】")
        logger.info("=" * 80)
        report_lines.append(f"\n【{model_name.upper()}】")
        report_lines.append("=" * 80)
        
        # Overlap metrics (daily)
        try:
            bucket_df = pd.read_csv(run_dir / f"{model_name}_bucket_returns.csv")
            
            # Top 5-15
            if 'top_5_15_return' in bucket_df.columns:
                top5_15 = bucket_df['top_5_15_return'].dropna() / 100.0
                
                logger.info(f"\n【Overlap 指标 - Top 5-15（{len(top5_15)}个交易日）】")
                logger.info("-" * 80)
                report_lines.append(f"\n【Overlap 指标 - Top 5-15（{len(top5_15)}个交易日）】")
                report_lines.append("-" * 80)
                
                avg_return = top5_15.mean() * 100
                median_return = top5_15.median() * 100
                std_return = top5_15.std() * 100
                win_rate = (top5_15 > 0).mean() * 100
                sharpe = (top5_15.mean() / top5_15.std()) * np.sqrt(252) if top5_15.std() > 0 else 0
                
                logger.info(f"平均收益: {avg_return:.4f}%")
                logger.info(f"中位数收益: {median_return:.4f}%")
                logger.info(f"标准差: {std_return:.4f}%")
                logger.info(f"Overlap 胜率: {win_rate:.2f}%")
                logger.info(f"Sharpe Ratio (年化): {sharpe:.4f}")
                
                report_lines.append(f"平均收益: {avg_return:.4f}%")
                report_lines.append(f"中位数收益: {median_return:.4f}%")
                report_lines.append(f"标准差: {std_return:.4f}%")
                report_lines.append(f"Overlap 胜率: {win_rate:.2f}%")
                report_lines.append(f"Sharpe Ratio (年化): {sharpe:.4f}")
        except FileNotFoundError:
            logger.warning(f"Bucket returns文件未找到: {model_name}_bucket_returns.csv")
            report_lines.append(f"⚠️ Bucket returns文件未找到: {model_name}_bucket_returns.csv")
        except Exception as e:
            logger.warning(f"处理Overlap指标时出错: {e}")
            report_lines.append(f"⚠️ 处理Overlap指标时出错: {e}")
        
        # Non-Overlap metrics
        try:
            nonoverlap_df = pd.read_csv(run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv")
            
            logger.info(f"\n【Non-Overlap 指标 - Top 5-15（{len(nonoverlap_df)}期，每10天）】")
            logger.info("-" * 80)
            report_lines.append(f"\n【Non-Overlap 指标 - Top 5-15（{len(nonoverlap_df)}期，每10天）】")
            report_lines.append("-" * 80)
            
            if 'top_gross_return' in nonoverlap_df.columns:
                period_returns = nonoverlap_df['top_gross_return']
                
                avg_period_return = period_returns.mean() * 100
                median_period_return = period_returns.median() * 100
                std_period_return = period_returns.std() * 100
                win_rate_nonoverlap = (period_returns > 0).mean() * 100
                
                logger.info(f"平均期间收益: {avg_period_return:.4f}%")
                logger.info(f"中位数期间收益: {median_period_return:.4f}%")
                logger.info(f"标准差: {std_period_return:.4f}%")
                logger.info(f"Non-Overlap 胜率: {win_rate_nonoverlap:.2f}%")
                
                report_lines.append(f"平均期间收益: {avg_period_return:.4f}%")
                report_lines.append(f"中位数期间收益: {median_period_return:.4f}%")
                report_lines.append(f"标准差: {std_period_return:.4f}%")
                report_lines.append(f"Non-Overlap 胜率: {win_rate_nonoverlap:.2f}%")
                
                # Sharpe (based on periods)
                if period_returns.std() > 0:
                    period_sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(25)
                    logger.info(f"Sharpe Ratio (基于期间): {period_sharpe:.4f}")
                    report_lines.append(f"Sharpe Ratio (基于期间): {period_sharpe:.4f}")
            
            if 'acc_return' in nonoverlap_df.columns:
                final_acc = nonoverlap_df['acc_return'].iloc[-1] * 100
                logger.info(f"\n累积收益: {final_acc:.4f}%")
                report_lines.append(f"\n累积收益: {final_acc:.4f}%")
                
                # Max drawdown
                if 'drawdown' in nonoverlap_df.columns:
                    max_dd = nonoverlap_df['drawdown'].min()
                    logger.info(f"最大回撤: {max_dd:.4f}%")
                    report_lines.append(f"最大回撤: {max_dd:.4f}%")
                else:
                    # Calculate if not present
                    cum_returns = (1 + period_returns).cumprod()
                    running_max = cum_returns.expanding().max()
                    drawdown = (cum_returns / running_max - 1) * 100
                    max_dd = drawdown.min()
                    logger.info(f"最大回撤: {max_dd:.4f}%")
                    report_lines.append(f"最大回撤: {max_dd:.4f}%")
                
                # Annualized return
                total_days = len(nonoverlap_df) * 10
                annualized_return = ((1 + final_acc/100) ** (252 / total_days) - 1) * 100
                logger.info(f"年化收益: {annualized_return:.4f}%")
                report_lines.append(f"年化收益: {annualized_return:.4f}%")
                
        except FileNotFoundError:
            logger.warning(f"Non-Overlap文件未找到: {model_name}_top5_15_rebalance10d_accumulated.csv")
            report_lines.append(f"⚠️ Non-Overlap文件未找到: {model_name}_top5_15_rebalance10d_accumulated.csv")
        except Exception as e:
            logger.warning(f"处理Non-Overlap指标时出错: {e}")
            report_lines.append(f"⚠️ 处理Non-Overlap指标时出错: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("【说明】")
    logger.info("-" * 80)
    if ema_top_n is not None and ema_top_n == -1:
        ema_note = "所有预测使用原始分数（EMA已禁用）"
    else:
        ema_note = "所有预测已应用EWMA平滑（3天EMA: 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}）"
        if ema_top_n is not None and ema_top_n > 0:
            ema_note += f"，仅对连续{ema_min_days}天都在Top{ema_top_n}的股票应用EMA"
    logger.info(ema_note)
    logger.info("=" * 80)
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("【说明】")
    report_lines.append("-" * 80)
    report_lines.append(ema_note)
    report_lines.append("=" * 80)
    
    # Save report to file
    report_file = run_dir / "complete_metrics_report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"✅ Saved complete metrics report: {report_file}")


def _plot_bucket_returns(
    bucket_ts: pd.DataFrame,
    model_name: str,
    top_buckets: list,
    bottom_buckets: list,
    bench: str,
    horizon: int,
    cost_bps: float,
    run_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate bucket returns plot for a single model"""
    try:
        # Plot 1: Per-period returns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top subplot: Top buckets
        for (a, b) in top_buckets:
            col = f"top_{a}_{b}_return"
            if col in bucket_ts.columns:
                ax1.plot(bucket_ts["date"], bucket_ts[col], linewidth=1.6, label=f"Top {a}-{b} (gross)")
            col_net = f"top_{a}_{b}_return_net"
            if col_net in bucket_ts.columns:
                ax1.plot(bucket_ts["date"], bucket_ts[col_net], linewidth=1.2, linestyle=":", 
                        label=f"Top {a}-{b} (net {cost_bps:g}bp)")
        
        if "benchmark_return" in bucket_ts.columns:
            ax1.plot(bucket_ts["date"], bucket_ts["benchmark_return"], linewidth=2.0, 
                    linestyle="--", color="black", label=f"{bench} (T+{horizon})")
        
        ax1.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax1.set_title(f"{model_name}: T+{horizon} Top Bucket Returns vs {bench} (Per-Period)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Rebalance Date")
        ax1.set_ylabel("Median Return (%)")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Bottom buckets
        for (a, b) in bottom_buckets:
            col = f"bottom_{a}_{b}_return"
            if col in bucket_ts.columns:
                ax2.plot(bucket_ts["date"], bucket_ts[col], linewidth=1.6, label=f"Bottom {a}-{b}")
        
        if "benchmark_return" in bucket_ts.columns:
            ax2.plot(bucket_ts["date"], bucket_ts["benchmark_return"], linewidth=2.0, 
                    linestyle="--", color="black", label=f"{bench} (T+{horizon})")
        
        ax2.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax2.set_title(f"{model_name}: T+{horizon} Bottom Bucket Returns vs {bench} (Per-Period)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Rebalance Date")
        ax2.set_ylabel("Median Return (%)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        png_path = run_dir / f"{model_name}_bucket_returns_period.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved bucket plot (per-period): %s", png_path)
        
        # Plot 2: Cumulative returns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top subplot: Cumulative top buckets
        for (a, b) in top_buckets:
            col = f"cum_top_{a}_{b}_return"
            if col in bucket_ts.columns:
                ax1.plot(bucket_ts["date"], bucket_ts[col], linewidth=1.8, label=f"Top {a}-{b} (cum gross)")
            col_net = f"cum_top_{a}_{b}_return_net"
            if col_net in bucket_ts.columns:
                ax1.plot(bucket_ts["date"], bucket_ts[col_net], linewidth=1.2, linestyle=":", 
                        label=f"Top {a}-{b} (cum net {cost_bps:g}bp)")
        
        if "cum_benchmark_return" in bucket_ts.columns:
            ax1.plot(bucket_ts["date"], bucket_ts["cum_benchmark_return"], linewidth=2.0, 
                    linestyle="--", color="black", label=f"{bench} (cum)")
        
        ax1.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax1.set_title(f"{model_name}: T+{horizon} Top Bucket Cumulative Returns vs {bench}", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Rebalance Date")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Cumulative bottom buckets
        for (a, b) in bottom_buckets:
            col = f"cum_bottom_{a}_{b}_return"
            if col in bucket_ts.columns:
                ax2.plot(bucket_ts["date"], bucket_ts[col], linewidth=1.8, label=f"Bottom {a}-{b} (cum)")
        
        if "cum_benchmark_return" in bucket_ts.columns:
            ax2.plot(bucket_ts["date"], bucket_ts["cum_benchmark_return"], linewidth=2.0, 
                    linestyle="--", color="black", label=f"{bench} (cum)")
        
        ax2.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
        ax2.set_title(f"{model_name}: T+{horizon} Bottom Bucket Cumulative Returns vs {bench}", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Rebalance Date")
        ax2.set_ylabel("Cumulative Return (%)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        png_path = run_dir / f"{model_name}_bucket_returns_cumulative.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved bucket plot (cumulative): %s", png_path)
        
    except Exception as e:
        logger.error("Failed to generate bucket plots for model=%s: %s", model_name, e)
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())


