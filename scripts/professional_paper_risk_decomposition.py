"""
Risk Decomposition Analysis for Academic Paper
==============================================
Computes comprehensive risk metrics including drawdown analysis, Calmar ratio,
downside capture, and tail risk measures.

For academic rigor, this provides:
1. Maximum Drawdown and drawdown distribution
2. Calmar Ratio (annualized return / max drawdown)
3. Sortino Ratio (downside-risk-adjusted return)
4. Upside/Downside Capture ratios vs benchmark
5. VaR and CVaR (tail risk)
6. Worst/Best periods analysis
7. Rolling risk metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns.

    Args:
        cumulative_returns: Cumulative return series

    Returns:
        Drawdown series (as negative percentages)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown


def compute_max_drawdown(returns: pd.Series) -> Dict:
    """
    Compute maximum drawdown and related statistics.

    Args:
        returns: Period returns series

    Returns:
        Dict with max_drawdown, start_date, trough_date, recovery_date, duration
    """
    cumulative = (1 + returns).cumprod()
    drawdown = compute_drawdown_series(cumulative)

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find start of drawdown (last peak before trough)
    start_idx = cumulative[:max_dd_idx].idxmax()

    # Find recovery (first date after trough where we reach the pre-drawdown peak)
    peak_value = cumulative[start_idx]
    recovery_mask = cumulative[max_dd_idx:] >= peak_value
    if recovery_mask.any():
        recovery_idx = cumulative[max_dd_idx:][recovery_mask].index[0]
        duration_days = (recovery_idx - start_idx).days if hasattr(start_idx, 'days') else len(cumulative[start_idx:recovery_idx])
    else:
        recovery_idx = None
        duration_days = (cumulative.index[-1] - start_idx).days if hasattr(start_idx, 'days') else len(cumulative[start_idx:])

    return {
        "max_drawdown": max_dd,
        "start_date": str(start_idx) if start_idx is not None else None,
        "trough_date": str(max_dd_idx) if max_dd_idx is not None else None,
        "recovery_date": str(recovery_idx) if recovery_idx is not None else "Not Recovered",
        "duration_periods": duration_days
    }


def compute_downside_metrics(returns: pd.Series, mar: float = 0.0, periods_per_year: float = 25.2) -> Dict:
    """
    Compute downside risk metrics.

    Args:
        returns: Period returns series
        mar: Minimum Acceptable Return (default 0)
        periods_per_year: Number of periods per year for annualization

    Returns:
        Dict with downside_deviation, sortino_ratio
    """
    # Downside returns (below MAR)
    downside_returns = returns[returns < mar]

    if len(downside_returns) == 0:
        downside_dev = 0.0
        sortino = np.inf
    else:
        # Downside deviation
        downside_dev = np.sqrt(((downside_returns - mar) ** 2).mean())

        # Sortino ratio
        mean_return = returns.mean()
        sortino = (mean_return - mar) / downside_dev if downside_dev > 0 else np.inf

    # Annualize
    annualized_downside_dev = downside_dev * np.sqrt(periods_per_year)
    annualized_sortino = sortino * np.sqrt(periods_per_year)

    return {
        "downside_deviation": downside_dev,
        "annualized_downside_deviation": annualized_downside_dev,
        "sortino_ratio": sortino,
        "annualized_sortino_ratio": annualized_sortino
    }


def compute_capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict:
    """
    Compute upside and downside capture ratios.

    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series

    Returns:
        Dict with upside_capture, downside_capture, capture_ratio
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Upside: periods where benchmark > 0
    upside_mask = aligned['benchmark'] > 0
    if upside_mask.sum() > 0:
        upside_strategy = aligned.loc[upside_mask, 'strategy'].mean()
        upside_benchmark = aligned.loc[upside_mask, 'benchmark'].mean()
        upside_capture = upside_strategy / upside_benchmark if upside_benchmark != 0 else np.nan
    else:
        upside_capture = np.nan

    # Downside: periods where benchmark < 0
    downside_mask = aligned['benchmark'] < 0
    if downside_mask.sum() > 0:
        downside_strategy = aligned.loc[downside_mask, 'strategy'].mean()
        downside_benchmark = aligned.loc[downside_mask, 'benchmark'].mean()
        downside_capture = downside_strategy / downside_benchmark if downside_benchmark != 0 else np.nan
    else:
        downside_capture = np.nan

    # Capture ratio (upside / downside)
    capture_ratio = upside_capture / downside_capture if (not np.isnan(upside_capture) and not np.isnan(downside_capture) and downside_capture != 0) else np.nan

    return {
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "capture_ratio": capture_ratio,
        "upside_periods": upside_mask.sum(),
        "downside_periods": downside_mask.sum()
    }


def compute_tail_risk(returns: pd.Series, confidence_levels: list = [0.95, 0.99]) -> Dict:
    """
    Compute VaR and CVaR (Expected Shortfall).

    Args:
        returns: Returns series
        confidence_levels: List of confidence levels

    Returns:
        Dict with VaR and CVaR at different confidence levels
    """
    tail_risk = {}

    for conf in confidence_levels:
        alpha = 1 - conf
        var = returns.quantile(alpha)
        cvar = returns[returns <= var].mean()

        tail_risk[f"VaR_{int(conf*100)}"] = var
        tail_risk[f"CVaR_{int(conf*100)}"] = cvar

    return tail_risk


def analyze_risk_decomposition(
    backtest_results_csv: str,
    output_dir: str,
    model_name: str = "lambdarank",
    benchmark_col: str = "benchmark_return",
    return_col_suffix: str = "_top_return_net",
    periods_per_year: float = 25.2,  # ~252 trading days / 10-day periods
):
    """
    Main risk decomposition analysis function.

    Args:
        backtest_results_csv: Path to weekly returns CSV from backtest
        output_dir: Output directory
        model_name: Model to analyze
        benchmark_col: Benchmark column name
        return_col_suffix: Suffix for return columns
        periods_per_year: Periods per year for annualization (default 25.2 for T+10)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading backtest results from {backtest_results_csv}...")
    data = pd.read_csv(backtest_results_csv)

    # Determine return column
    return_col = f"{model_name}{return_col_suffix}"

    if return_col not in data.columns:
        print(f"Error: Column {return_col} not found in data")
        print(f"Available columns: {data.columns.tolist()}")
        return

    # Get returns series (convert from percentage to decimal if needed)
    strategy_returns = data[return_col].dropna()
    # Check if returns are in percentage format (abs values > 1)
    if strategy_returns.abs().max() > 1:
        print("Converting returns from percentage to decimal format...")
        strategy_returns = strategy_returns / 100

    benchmark_returns = None
    if benchmark_col in data.columns:
        benchmark_returns = data[benchmark_col].dropna()
        if benchmark_returns.abs().max() > 1:
            benchmark_returns = benchmark_returns / 100

    print(f"Analyzing {len(strategy_returns)} periods for {model_name}...")

    # 1. Basic statistics
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe = mean_return / std_return if std_return > 0 else np.nan
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)

    # 2. Drawdown analysis
    dd_stats = compute_max_drawdown(strategy_returns)

    # 3. Calmar Ratio (Annualized Return / Max Drawdown)
    calmar_ratio = annualized_return / abs(dd_stats['max_drawdown']) if dd_stats['max_drawdown'] != 0 else np.inf

    # 4. Downside metrics
    downside_stats = compute_downside_metrics(strategy_returns, mar=0.0, periods_per_year=periods_per_year)

    # 5. Tail risk
    tail_risk = compute_tail_risk(strategy_returns)

    # 6. Best/Worst periods
    best_period = strategy_returns.max()
    worst_period = strategy_returns.min()
    best_period_idx = strategy_returns.idxmax()
    worst_period_idx = strategy_returns.idxmin()

    # 7. Win rate
    win_rate = (strategy_returns > 0).mean()

    # 8. Capture ratios (if benchmark available)
    if benchmark_returns is not None:
        capture_stats = compute_capture_ratios(strategy_returns, benchmark_returns)

        # Benchmark stats for comparison
        bm_mean = benchmark_returns.mean()
        bm_std = benchmark_returns.std()
        bm_sharpe = bm_mean / bm_std if bm_std > 0 else np.nan
        bm_dd_stats = compute_max_drawdown(benchmark_returns)
    else:
        capture_stats = {}
        bm_mean = np.nan
        bm_std = np.nan
        bm_sharpe = np.nan
        bm_dd_stats = {}

    # Compile results (convert numpy types to Python types for JSON serialization)
    def to_python_type(val):
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val

    risk_metrics = {
        "model": model_name,
        "n_periods": int(len(strategy_returns)),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "sharpe_ratio": float(sharpe),
        "annualized_return": float(annualized_return),
        "annualized_std": float(annualized_std),
        "annualized_sharpe": float(annualized_sharpe),
        "win_rate": float(win_rate),
        **{k: to_python_type(v) for k, v in dd_stats.items()},
        "calmar_ratio": float(calmar_ratio),
        **{k: to_python_type(v) for k, v in downside_stats.items()},
        **{k: to_python_type(v) for k, v in tail_risk.items()},
        "best_period_return": float(best_period),
        "worst_period_return": float(worst_period),
        "best_period_idx": str(best_period_idx),
        "worst_period_idx": str(worst_period_idx),
        **{k: to_python_type(v) for k, v in capture_stats.items()},
        "benchmark_mean_return": float(bm_mean) if not np.isnan(bm_mean) else None,
        "benchmark_std_return": float(bm_std) if not np.isnan(bm_std) else None,
        "benchmark_sharpe": float(bm_sharpe) if not np.isnan(bm_sharpe) else None,
        "benchmark_max_drawdown": float(bm_dd_stats.get('max_drawdown', np.nan)) if bm_dd_stats and not np.isnan(bm_dd_stats.get('max_drawdown', np.nan)) else None
    }

    # Save metrics
    metrics_df = pd.DataFrame([risk_metrics])
    metrics_df.to_csv(output_path / f"{model_name}_risk_metrics.csv", index=False)
    print(f"Saved risk metrics to {output_path / f'{model_name}_risk_metrics.csv'}")

    with open(output_path / f"{model_name}_risk_metrics.json", 'w') as f:
        json.dump(risk_metrics, f, indent=2)

    # Visualizations
    print("Generating risk visualizations...")

    # Plot 1: Drawdown over time
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    cumulative = (1 + strategy_returns).cumprod()
    drawdown = compute_drawdown_series(cumulative)

    axes[0].plot(cumulative.index, cumulative.values, label='Cumulative Return', color='blue', linewidth=2)
    if benchmark_returns is not None:
        bm_cumulative = (1 + benchmark_returns).cumprod()
        axes[0].plot(bm_cumulative.index, bm_cumulative.values, label='Benchmark', color='gray', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].set_title(f'{model_name.upper()} - Cumulative Returns and Drawdown', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3, label='Drawdown')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Period', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_drawdown_analysis.png", dpi=300)
    plt.close()

    # Plot 2: Return distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(strategy_returns * 100, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(strategy_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {strategy_returns.mean()*100:.2f}%')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel('Period Return (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(strategy_returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_return_distribution.png", dpi=300)
    plt.close()

    # Plot 3: Rolling metrics
    window = 10  # 10-period rolling window
    if len(strategy_returns) >= window:
        rolling_mean = strategy_returns.rolling(window).mean()
        rolling_std = strategy_returns.rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(rolling_mean.index, rolling_mean.values * 100, label=f'{window}-Period Rolling Mean', color='blue', linewidth=2)
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0].set_ylabel('Mean Return (%)', fontsize=12)
        axes[0].set_title(f'{model_name.upper()} - Rolling Risk Metrics (Window={window})', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(rolling_std.index, rolling_std.values * 100, label=f'{window}-Period Rolling Std', color='orange', linewidth=2)
        axes[1].set_ylabel('Std Deviation (%)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, label=f'{window}-Period Rolling Sharpe', color='green', linewidth=2)
        axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[2].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[2].set_xlabel('Period', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_rolling_metrics.png", dpi=300)
        plt.close()

    print(f"\n=== RISK DECOMPOSITION COMPLETE ===")
    print(f"Model: {model_name}")
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Annualized Sharpe: {annualized_sharpe:.2f}")
    print(f"Max Drawdown: {dd_stats['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Sortino Ratio (Ann.): {downside_stats['annualized_sortino_ratio']:.2f}")
    if capture_stats:
        print(f"Upside Capture: {capture_stats.get('upside_capture', np.nan)*100:.2f}%")
        print(f"Downside Capture: {capture_stats.get('downside_capture', np.nan)*100:.2f}%")
    print(f"Win Rate: {win_rate*100:.2f}%")

    return risk_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk Decomposition Analysis")
    parser.add_argument(
        "--backtest-results",
        type=str,
        required=True,
        help="Path to backtest weekly returns CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/risk_decomposition",
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name to analyze"
    )
    parser.add_argument(
        "--benchmark-col",
        type=str,
        default="benchmark_return",
        help="Benchmark column name"
    )

    args = parser.parse_args()

    analyze_risk_decomposition(
        backtest_results_csv=args.backtest_results,
        output_dir=args.output_dir,
        model_name=args.model_name,
        benchmark_col=args.benchmark_col
    )
