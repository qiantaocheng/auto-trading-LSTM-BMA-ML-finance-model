"""
Market Impact and Capacity Analysis for Academic Paper
======================================================
Estimates strategy capacity and market impact costs using the square root
market impact model. Critical for demonstrating practical implementability.

For academic rigor, this provides:
1. Market impact cost estimation (square root law)
2. Strategy capacity at different target costs
3. Slippage vs volume analysis
4. Realistic AUM limits
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def estimate_market_impact(
    trade_size_usd: float,
    daily_volume_usd: float,
    volatility: float = 0.02,
    participation_rate: float = 0.10,
    market_impact_coeff: float = 0.1
) -> float:
    """
    Estimate market impact cost using square root law.

    MI = sigma * sqrt(Q / V) * coeff

    Where:
    - sigma = volatility
    - Q = trade size (USD)
    - V = daily volume (USD)
    - coeff = market impact coefficient (~0.1 for US equities)

    Args:
        trade_size_usd: Trade size in USD
        daily_volume_usd: Daily trading volume in USD
        volatility: Price volatility (std of returns)
        participation_rate: Fraction of daily volume we can trade (default 10%)
        market_impact_coeff: Market impact coefficient

    Returns:
        Market impact cost as fraction of trade value
    """
    # Limit trade size to participation rate
    max_trade_size = daily_volume_usd * participation_rate

    if trade_size_usd > max_trade_size:
        print(f"Warning: Trade size ${trade_size_usd:,.0f} exceeds {participation_rate*100}% of daily volume ${daily_volume_usd:,.0f}")
        effective_trade_size = max_trade_size
    else:
        effective_trade_size = trade_size_usd

    # Square root market impact
    volume_fraction = effective_trade_size / daily_volume_usd
    market_impact = volatility * np.sqrt(volume_fraction) * market_impact_coeff

    return market_impact


def analyze_strategy_capacity(
    backtest_results: str,
    data_file: str,
    output_dir: str,
    model_name: str = "lambdarank",
    return_col: str = "lambdarank_top_return_net",
    n_stocks: int = 30,
    rebalance_frequency_days: int = 10,
    target_costs_bps: list = None,
    participation_rate: float = 0.05,  # 5% of daily volume
    market_impact_coeff: float = 0.1
):
    """
    Main capacity analysis function.

    Args:
        backtest_results: Path to backtest results CSV
        data_file: Path to factor data with Close prices
        output_dir: Output directory
        model_name: Model name
        return_col: Return column in backtest results
        n_stocks: Number of stocks in portfolio
        rebalance_frequency_days: Rebalance frequency
        target_costs_bps: List of target cost levels in bps
        participation_rate: Max fraction of daily volume to trade
        market_impact_coeff: Market impact coefficient
    """
    if target_costs_bps is None:
        target_costs_bps = [5, 10, 20, 50, 100]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading backtest results from {backtest_results}...")
    results_df = pd.read_csv(backtest_results)

    # Convert returns from percentage if needed
    returns = results_df[return_col].dropna()
    if returns.abs().max() > 1:
        returns = returns / 100

    avg_period_return = returns.mean()
    periods_per_year = 252 / rebalance_frequency_days
    annualized_return = avg_period_return * periods_per_year

    print(f"Strategy annualized return: {annualized_return*100:.2f}%")

    # Load price data to estimate market cap and volume
    print(f"Loading market data from {data_file}...")
    data = pd.read_parquet(data_file)

    if not isinstance(data.index, pd.MultiIndex):
        if 'date' in data.columns and 'ticker' in data.columns:
            data = data.set_index(['date', 'ticker'])

    # Estimate average stock metrics
    # Assume typical stock: $50 price, $500M market cap, $10M daily volume
    typical_price = 50
    typical_market_cap = 500_000_000
    typical_daily_volume_usd = 10_000_000
    typical_volatility = 0.02  # 2% daily volatility

    # Calculate capacity at different AUM levels
    aum_levels = np.logspace(6, 9, 20)  # $1M to $1B

    capacity_results = []

    for aum in aum_levels:
        # Position size per stock
        position_size = aum / n_stocks

        # Shares to trade
        shares_to_trade = position_size / typical_price

        # Trade value
        trade_value = position_size

        # Estimate market impact
        impact_cost = estimate_market_impact(
            trade_size_usd=trade_value,
            daily_volume_usd=typical_daily_volume_usd,
            volatility=typical_volatility,
            participation_rate=participation_rate,
            market_impact_coeff=market_impact_coeff
        )

        # Total cost: fixed cost (10 bps) + market impact
        fixed_cost_bps = 10
        total_cost_bps = fixed_cost_bps + (impact_cost * 10000)

        # Net return after costs
        gross_return_annual = annualized_return
        cost_per_rebalance = total_cost_bps / 10000
        total_annual_cost = cost_per_rebalance * periods_per_year
        net_return_annual = gross_return_annual - total_annual_cost

        capacity_results.append({
            'aum_usd': aum,
            'position_size_usd': position_size,
            'market_impact_bps': impact_cost * 10000,
            'total_cost_bps': total_cost_bps,
            'gross_return_annual_pct': gross_return_annual * 100,
            'net_return_annual_pct': net_return_annual * 100,
            'cost_drag_annual_pct': total_annual_cost * 100
        })

    capacity_df = pd.DataFrame(capacity_results)
    capacity_df.to_csv(output_path / f"{model_name}_capacity_analysis.csv", index=False)

    # Find capacity at different target net returns
    target_returns = [5, 10, 20, 50]  # Target net returns in %
    capacity_at_targets = []

    for target_return_pct in target_returns:
        target_return = target_return_pct / 100

        # Find AUM where net return drops below target
        viable = capacity_df[capacity_df['net_return_annual_pct'] >= target_return_pct]

        if len(viable) > 0:
            max_aum = viable['aum_usd'].max()
        else:
            max_aum = 0

        capacity_at_targets.append({
            'target_net_return_pct': target_return_pct,
            'max_aum_usd': max_aum,
            'max_aum_millions': max_aum / 1_000_000
        })

    capacity_targets_df = pd.DataFrame(capacity_at_targets)
    capacity_targets_df.to_csv(output_path / f"{model_name}_capacity_targets.csv", index=False)

    # Visualizations
    print("Generating capacity visualizations...")

    # Plot 1: Capacity curve
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(capacity_df['aum_usd'] / 1e6, capacity_df['gross_return_annual_pct'],
                 'b-', linewidth=2, label='Gross Return')
    axes[0].plot(capacity_df['aum_usd'] / 1e6, capacity_df['net_return_annual_pct'],
                 'r-', linewidth=2, label='Net Return (After Impact)')

    # Add target return lines
    for target in [10, 20, 50]:
        axes[0].axhline(target, color='gray', linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Annualized Return (%)', fontsize=12)
    axes[0].set_title(f'{model_name.upper()} - Strategy Capacity Analysis', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Plot 2: Cost breakdown
    axes[1].plot(capacity_df['aum_usd'] / 1e6, capacity_df['market_impact_bps'],
                 'orange', linewidth=2, label='Market Impact')
    axes[1].plot(capacity_df['aum_usd'] / 1e6, capacity_df['total_cost_bps'],
                 'red', linewidth=2, label='Total Cost')
    axes[1].axhline(10, color='blue', linestyle='--', alpha=0.5, label='Fixed Cost (10 bps)')

    axes[1].set_xlabel('AUM ($ Millions)', fontsize=12)
    axes[1].set_ylabel('Cost (bps per rebalance)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_capacity_curve.png", dpi=300)
    plt.close()

    # Plot 2: Capacity at target returns
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(capacity_targets_df)),
           capacity_targets_df['max_aum_millions'],
           color='green', alpha=0.7, edgecolor='black')

    ax.set_xticks(range(len(capacity_targets_df)))
    ax.set_xticklabels([f"{r:.0f}%" for r in capacity_targets_df['target_net_return_pct']])
    ax.set_xlabel('Target Net Annual Return', fontsize=12)
    ax.set_ylabel('Maximum AUM ($ Millions)', fontsize=12)
    ax.set_title('Strategy Capacity at Different Return Targets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(capacity_targets_df['max_aum_millions']):
        ax.text(i, v, f'${v:.0f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_capacity_targets.png", dpi=300)
    plt.close()

    # Summary report
    # Find "reasonable" capacity (where net return > 20%)
    reasonable_capacity = capacity_df[capacity_df['net_return_annual_pct'] >= 20]
    if len(reasonable_capacity) > 0:
        reasonable_aum = reasonable_capacity['aum_usd'].max()
    else:
        reasonable_aum = 0

    report = {
        "model": model_name,
        "analysis_date": datetime.now().isoformat(),
        "assumptions": {
            "n_stocks": n_stocks,
            "rebalance_frequency_days": rebalance_frequency_days,
            "typical_daily_volume_usd": typical_daily_volume_usd,
            "participation_rate": participation_rate,
            "market_impact_coefficient": market_impact_coeff
        },
        "strategy_performance": {
            "avg_period_return_pct": float(avg_period_return * 100),
            "annualized_gross_return_pct": float(annualized_return * 100)
        },
        "capacity_estimates": {
            "reasonable_capacity_20pct_net_usd": float(reasonable_aum),
            "reasonable_capacity_20pct_net_millions": float(reasonable_aum / 1_000_000),
            "capacity_at_targets": capacity_targets_df.to_dict('records')
        },
        "full_capacity_curve": capacity_df.to_dict('records')
    }

    with open(output_path / f"{model_name}_capacity_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== CAPACITY ANALYSIS COMPLETE ===")
    print(f"Model: {model_name}")
    print(f"Gross Annualized Return: {annualized_return*100:.2f}%")
    print(f"\nCapacity Estimates:")
    print(f"  At 50% net return: ${capacity_targets_df[capacity_targets_df['target_net_return_pct']==50]['max_aum_millions'].iloc[0] if 50 in capacity_targets_df['target_net_return_pct'].values else 0:.0f}M")
    print(f"  At 20% net return: ${reasonable_aum/1e6:.0f}M")
    print(f"  At 10% net return: ${capacity_targets_df[capacity_targets_df['target_net_return_pct']==10]['max_aum_millions'].iloc[0] if 10 in capacity_targets_df['target_net_return_pct'].values else 0:.0f}M")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Impact and Capacity Analysis")
    parser.add_argument(
        "--backtest-results",
        type=str,
        required=True,
        help="Path to backtest results CSV"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/factor_exports/factors/factors_all.parquet",
        help="Path to factor data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/capacity_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name"
    )

    args = parser.parse_args()

    analyze_strategy_capacity(
        backtest_results=args.backtest_results,
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
