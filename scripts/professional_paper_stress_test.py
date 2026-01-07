"""
Stress Test Analysis for Academic Paper
========================================
Analyzes strategy performance during different market regimes and
stress periods to demonstrate robustness.

For academic rigor, this provides:
1. Performance by market regime (bull, bear, volatile, calm)
2. Analysis of drawdown periods
3. Performance in different volatility quintiles
4. Regime-specific risk metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def classify_market_regime(
    benchmark_returns: pd.Series,
    volatility_window: int = 20
) -> pd.DataFrame:
    """
    Classify each period into market regimes.

    Args:
        benchmark_returns: Benchmark return series
        volatility_window: Window for volatility calculation

    Returns:
        DataFrame with regime classifications
    """
    regimes = pd.DataFrame(index=benchmark_returns.index)
    regimes['return'] = benchmark_returns

    # Calculate rolling volatility
    regimes['volatility'] = benchmark_returns.rolling(volatility_window).std()

    # Calculate rolling mean return
    regimes['trend'] = benchmark_returns.rolling(volatility_window).mean()

    # Classify regime
    vol_median = regimes['volatility'].median()
    trend_threshold = 0

    def classify(row):
        if pd.isna(row['volatility']) or pd.isna(row['trend']):
            return 'Unknown'

        # High volatility
        if row['volatility'] > vol_median:
            if row['trend'] > trend_threshold:
                return 'Volatile Bull'
            else:
                return 'Volatile Bear'
        # Low volatility
        else:
            if row['trend'] > trend_threshold:
                return 'Calm Bull'
            else:
                return 'Calm Bear'

    regimes['regime'] = regimes.apply(classify, axis=1)

    return regimes


def analyze_by_regime(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    model_name: str = "lambdarank"
) -> Dict:
    """
    Analyze performance by market regime.
    """
    # Classify regimes
    regimes_df = classify_market_regime(benchmark_returns)

    # Merge with strategy returns
    analysis_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'regime': regimes_df['regime']
    }).dropna()

    # Compute metrics by regime
    regime_stats = []

    for regime in analysis_df['regime'].unique():
        if regime == 'Unknown':
            continue

        regime_data = analysis_df[analysis_df['regime'] == regime]

        if len(regime_data) < 3:
            continue

        stats = {
            'regime': regime,
            'n_periods': len(regime_data),
            'strategy_mean_return': regime_data['strategy_return'].mean(),
            'strategy_std_return': regime_data['strategy_return'].std(),
            'benchmark_mean_return': regime_data['benchmark_return'].mean(),
            'benchmark_std_return': regime_data['benchmark_return'].std(),
            'strategy_sharpe': regime_data['strategy_return'].mean() / regime_data['strategy_return'].std() if regime_data['strategy_return'].std() > 0 else np.nan,
            'benchmark_sharpe': regime_data['benchmark_return'].mean() / regime_data['benchmark_return'].std() if regime_data['benchmark_return'].std() > 0 else np.nan,
            'win_rate': (regime_data['strategy_return'] > 0).mean(),
            'outperformance_rate': (regime_data['strategy_return'] > regime_data['benchmark_return']).mean()
        }

        regime_stats.append(stats)

    return pd.DataFrame(regime_stats)


def analyze_volatility_quintiles(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Analyze performance by volatility quintiles.
    """
    # Calculate rolling volatility
    volatility = benchmark_returns.rolling(vol_window).std()

    # Create quintiles
    volatility_quintiles = pd.qcut(volatility, q=5, labels=['Q1-Low', 'Q2', 'Q3', 'Q4', 'Q5-High'], duplicates='drop')

    # Merge
    analysis_df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'benchmark_return': benchmark_returns,
        'volatility': volatility,
        'vol_quintile': volatility_quintiles
    }).dropna()

    # Stats by quintile
    quintile_stats = []

    for quintile in analysis_df['vol_quintile'].unique():
        quintile_data = analysis_df[analysis_df['vol_quintile'] == quintile]

        stats = {
            'volatility_quintile': quintile,
            'n_periods': len(quintile_data),
            'avg_volatility': quintile_data['volatility'].mean(),
            'strategy_mean_return': quintile_data['strategy_return'].mean(),
            'strategy_std_return': quintile_data['strategy_return'].std(),
            'benchmark_mean_return': quintile_data['benchmark_return'].mean(),
            'strategy_sharpe': quintile_data['strategy_return'].mean() / quintile_data['strategy_return'].std() if quintile_data['strategy_return'].std() > 0 else np.nan,
            'win_rate': (quintile_data['strategy_return'] > 0).mean()
        }

        quintile_stats.append(stats)

    return pd.DataFrame(quintile_stats)


def analyze_drawdown_periods(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Dict:
    """
    Analyze performance during drawdown periods.
    """
    # Calculate cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # Calculate running max
    strategy_running_max = strategy_cum.cummax()
    benchmark_running_max = benchmark_cum.cummax()

    # Calculate drawdown
    strategy_dd = (strategy_cum - strategy_running_max) / strategy_running_max
    benchmark_dd = (benchmark_cum - benchmark_running_max) / benchmark_running_max

    # Find periods in drawdown (DD > -5%)
    strategy_in_dd = strategy_dd < -0.05
    benchmark_in_dd = benchmark_dd < -0.05

    # Performance during benchmark drawdowns
    during_benchmark_dd = strategy_returns[benchmark_in_dd]

    if len(during_benchmark_dd) > 0:
        performance_during_dd = {
            'n_periods_in_benchmark_drawdown': int(benchmark_in_dd.sum()),
            'strategy_avg_return_during_benchmark_dd': float(during_benchmark_dd.mean()),
            'strategy_win_rate_during_benchmark_dd': float((during_benchmark_dd > 0).mean()),
            'benchmark_avg_return_during_dd': float(benchmark_returns[benchmark_in_dd].mean())
        }
    else:
        performance_during_dd = {
            'n_periods_in_benchmark_drawdown': 0
        }

    return performance_during_dd


def analyze_stress_test(
    backtest_results: str,
    output_dir: str,
    model_name: str = "lambdarank",
    return_col: str = "lambdarank_top_return_net",
    benchmark_col: str = "benchmark_return"
):
    """
    Main stress test analysis function.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading backtest results from {backtest_results}...")
    data = pd.read_csv(backtest_results)

    # Convert from percentage if needed
    strategy_returns = data[return_col].dropna()
    benchmark_returns = data[benchmark_col].dropna()

    if strategy_returns.abs().max() > 1:
        print("Converting returns from percentage to decimal...")
        strategy_returns = strategy_returns / 100
        benchmark_returns = benchmark_returns / 100

    print(f"Analyzing {len(strategy_returns)} periods...")

    # 1. Regime analysis
    print("Analyzing performance by market regime...")
    regime_stats = analyze_by_regime(strategy_returns, benchmark_returns, model_name)
    regime_stats.to_csv(output_path / f"{model_name}_regime_analysis.csv", index=False)

    # 2. Volatility quintile analysis
    print("Analyzing performance by volatility quintile...")
    volatility_stats = analyze_volatility_quintiles(strategy_returns, benchmark_returns)
    volatility_stats.to_csv(output_path / f"{model_name}_volatility_quintile_analysis.csv", index=False)

    # 3. Drawdown period analysis
    print("Analyzing performance during drawdown periods...")
    drawdown_stats = analyze_drawdown_periods(strategy_returns, benchmark_returns)

    # Visualizations
    print("Generating stress test visualizations...")

    # Plot 1: Performance by regime
    if len(regime_stats) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Mean returns by regime
        regimes = regime_stats['regime']
        strategy_returns_by_regime = regime_stats['strategy_mean_return'] * 100
        benchmark_returns_by_regime = regime_stats['benchmark_mean_return'] * 100

        x = np.arange(len(regimes))
        width = 0.35

        axes[0].bar(x - width/2, strategy_returns_by_regime, width, label='Strategy', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, benchmark_returns_by_regime, width, label='Benchmark', color='gray', alpha=0.7)
        axes[0].set_xlabel('Market Regime', fontsize=12)
        axes[0].set_ylabel('Mean Return (%)', fontsize=12)
        axes[0].set_title('Performance by Market Regime', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(regimes, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Sharpe ratio by regime
        strategy_sharpe_by_regime = regime_stats['strategy_sharpe']
        benchmark_sharpe_by_regime = regime_stats['benchmark_sharpe']

        axes[1].bar(x - width/2, strategy_sharpe_by_regime, width, label='Strategy', color='green', alpha=0.7)
        axes[1].bar(x + width/2, benchmark_sharpe_by_regime, width, label='Benchmark', color='gray', alpha=0.7)
        axes[1].set_xlabel('Market Regime', fontsize=12)
        axes[1].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[1].set_title('Sharpe Ratio by Market Regime', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regimes, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_regime_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 2: Performance by volatility quintile
    if len(volatility_stats) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        quintiles = volatility_stats['volatility_quintile']
        strategy_returns_by_vol = volatility_stats['strategy_mean_return'] * 100
        benchmark_returns_by_vol = volatility_stats['benchmark_mean_return'] * 100

        x = np.arange(len(quintiles))
        width = 0.35

        axes[0].bar(x - width/2, strategy_returns_by_vol, width, label='Strategy', color='blue', alpha=0.7)
        axes[0].bar(x + width/2, benchmark_returns_by_vol, width, label='Benchmark', color='gray', alpha=0.7)
        axes[0].set_xlabel('Volatility Quintile', fontsize=12)
        axes[0].set_ylabel('Mean Return (%)', fontsize=12)
        axes[0].set_title('Performance by Volatility Quintile', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(quintiles, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Sharpe by quintile
        strategy_sharpe_by_vol = volatility_stats['strategy_sharpe']

        axes[1].bar(x, strategy_sharpe_by_vol, color='green', alpha=0.7)
        axes[1].set_xlabel('Volatility Quintile', fontsize=12)
        axes[1].set_ylabel('Strategy Sharpe Ratio', fontsize=12)
        axes[1].set_title('Sharpe Ratio by Volatility Quintile', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(quintiles, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_volatility_quintile_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Summary report
    report = {
        "model": model_name,
        "analysis_date": datetime.now().isoformat(),
        "regime_analysis": regime_stats.to_dict('records') if len(regime_stats) > 0 else [],
        "volatility_quintile_analysis": volatility_stats.to_dict('records') if len(volatility_stats) > 0 else [],
        "drawdown_period_analysis": drawdown_stats
    }

    with open(output_path / f"{model_name}_stress_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== STRESS TEST ANALYSIS COMPLETE ===")
    print(f"Model: {model_name}")

    if len(regime_stats) > 0:
        print("\nPerformance by Market Regime:")
        for _, row in regime_stats.iterrows():
            print(f"  {row['regime']}: {row['strategy_mean_return']*100:.2f}% (n={row['n_periods']:.0f})")

    if len(volatility_stats) > 0:
        print("\nPerformance by Volatility Quintile:")
        for _, row in volatility_stats.iterrows():
            print(f"  {row['volatility_quintile']}: {row['strategy_mean_return']*100:.2f}% (n={row['n_periods']:.0f})")

    if drawdown_stats.get('n_periods_in_benchmark_drawdown', 0) > 0:
        print(f"\nPerformance During Benchmark Drawdowns:")
        print(f"  Periods in DD: {drawdown_stats['n_periods_in_benchmark_drawdown']}")
        print(f"  Strategy Return: {drawdown_stats['strategy_avg_return_during_benchmark_dd']*100:.2f}%")
        print(f"  Win Rate: {drawdown_stats['strategy_win_rate_during_benchmark_dd']*100:.1f}%")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stress Test Analysis")
    parser.add_argument(
        "--backtest-results",
        type=str,
        required=True,
        help="Path to backtest results CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/stress_test",
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name"
    )

    args = parser.parse_args()

    analyze_stress_test(
        backtest_results=args.backtest_results,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
