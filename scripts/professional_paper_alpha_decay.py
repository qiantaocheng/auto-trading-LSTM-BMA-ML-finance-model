"""
Alpha Decay Analysis for Academic Paper
========================================
Measures how quickly the predictive signal degrades if execution is delayed.
Critical for demonstrating signal "shelf-life" and practical implementability.

For academic rigor, this provides:
1. Forward return decay curves (T+1, T+2, ..., T+20)
2. IC decay over different holding periods
3. Optimal holding period identification
4. Signal half-life estimation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def compute_forward_returns(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    horizons: List[int] = [1, 2, 3, 5, 7, 10, 15, 20],
    pred_col: str = "prediction",
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Compute forward returns at multiple horizons for alpha decay analysis.

    Args:
        predictions: DataFrame with predictions (MultiIndex: date, ticker)
        data: DataFrame with price/return data (MultiIndex: date, ticker)
        horizons: List of forward horizons in trading days
        pred_col: Prediction column name
        price_col: Price column name for computing returns

    Returns:
        DataFrame with predictions and forward returns at each horizon
    """
    result_df = predictions[[pred_col]].copy()

    for horizon in horizons:
        print(f"  Computing T+{horizon} forward returns...")

        # For each date-ticker, get the return from T to T+horizon
        forward_returns = []

        for (date, ticker), row in result_df.iterrows():
            try:
                # Get current and future price
                current_price = data.loc[(date, ticker), price_col]

                # Find the date that is 'horizon' days forward
                ticker_data = data.xs(ticker, level='ticker')
                current_idx = ticker_data.index.get_loc(date)

                if current_idx + horizon < len(ticker_data):
                    future_date = ticker_data.index[current_idx + horizon]
                    future_price = ticker_data.loc[future_date, price_col]

                    fwd_return = (future_price - current_price) / current_price
                else:
                    fwd_return = np.nan

            except (KeyError, IndexError):
                fwd_return = np.nan

            forward_returns.append(fwd_return)

        result_df[f"fwd_return_T{horizon}"] = forward_returns

    return result_df


def compute_ic_at_horizon(
    df: pd.DataFrame,
    pred_col: str,
    return_col: str
) -> Dict:
    """
    Compute IC and Rank IC for a specific horizon.
    """
    valid_data = df[[pred_col, return_col]].dropna()

    if len(valid_data) < 30:
        return {
            "IC": np.nan,
            "Rank_IC": np.nan,
            "IC_pvalue": np.nan,
            "n_samples": len(valid_data)
        }

    ic, ic_pval = stats.pearsonr(valid_data[pred_col], valid_data[return_col])
    rank_ic, rank_ic_pval = stats.spearmanr(valid_data[pred_col], valid_data[return_col])

    return {
        "IC": ic,
        "Rank_IC": rank_ic,
        "IC_pvalue": ic_pval,
        "Rank_IC_pvalue": rank_ic_pval,
        "n_samples": len(valid_data)
    }


def estimate_half_life(decay_curve: pd.DataFrame, metric_col: str = "IC") -> float:
    """
    Estimate the half-life of the signal (when IC drops to 50% of initial value).

    Args:
        decay_curve: DataFrame with horizon and IC values
        metric_col: Metric column to analyze

    Returns:
        Estimated half-life in days
    """
    initial_ic = decay_curve[metric_col].iloc[0]
    half_ic = initial_ic / 2

    # Find the first horizon where IC drops below half
    below_half = decay_curve[decay_curve[metric_col] < half_ic]

    if len(below_half) > 0:
        return below_half['horizon'].iloc[0]
    else:
        return np.nan


def analyze_alpha_decay(
    predictions_file: str,
    data_file: str,
    output_dir: str,
    model_name: str = "lambdarank",
    horizons: List[int] = None,
    pred_col: str = "prediction",
    price_col: str = "Close",
    target_horizon: int = 10
):
    """
    Main alpha decay analysis function.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 7, 10, 15, 20]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_file}...")
    predictions = pd.read_parquet(predictions_file)

    # Ensure MultiIndex
    if not isinstance(predictions.index, pd.MultiIndex):
        if 'date' in predictions.columns and 'ticker' in predictions.columns:
            predictions = predictions.set_index(['date', 'ticker'])

    print(f"Loading data from {data_file}...")
    data = pd.read_parquet(data_file)

    if not isinstance(data.index, pd.MultiIndex):
        if 'date' in data.columns and 'ticker' in data.columns:
            data = data.set_index(['date', 'ticker'])

    # Compute forward returns at all horizons
    print("Computing forward returns at multiple horizons...")
    decay_df = compute_forward_returns(
        predictions, data, horizons, pred_col, price_col
    )

    # Compute IC at each horizon
    print("Computing IC decay curve...")
    decay_curve = []

    for horizon in horizons:
        return_col = f"fwd_return_T{horizon}"
        ic_stats = compute_ic_at_horizon(decay_df, pred_col, return_col)

        decay_curve.append({
            "horizon": horizon,
            **ic_stats
        })

    decay_df_summary = pd.DataFrame(decay_curve)

    # Save results
    decay_df_summary.to_csv(output_path / f"{model_name}_alpha_decay_curve.csv", index=False)
    print(f"Saved alpha decay curve to {output_path / f'{model_name}_alpha_decay_curve.csv'}")

    # Estimate half-life
    half_life_ic = estimate_half_life(decay_df_summary, "IC")
    half_life_rank_ic = estimate_half_life(decay_df_summary, "Rank_IC")

    # Visualizations
    print("Generating alpha decay visualizations...")

    # Plot 1: IC Decay Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IC decay
    axes[0].plot(decay_df_summary['horizon'], decay_df_summary['IC'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].axvline(target_horizon, color='red', linestyle='--', alpha=0.5, label=f'Target Horizon (T+{target_horizon})')
    axes[0].set_xlabel('Forward Horizon (Days)', fontsize=12)
    axes[0].set_ylabel('Information Coefficient', fontsize=12)
    axes[0].set_title('IC Decay Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rank IC decay
    axes[1].plot(decay_df_summary['horizon'], decay_df_summary['Rank_IC'], 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].axvline(target_horizon, color='red', linestyle='--', alpha=0.5, label=f'Target Horizon (T+{target_horizon})')
    axes[1].set_xlabel('Forward Horizon (Days)', fontsize=12)
    axes[1].set_ylabel('Rank IC (Spearman)', fontsize=12)
    axes[1].set_title('Rank IC Decay Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{model_name.upper()} - Alpha Decay Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_alpha_decay_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Statistical Significance Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create significance matrix
    sig_matrix = []
    for idx, row in decay_df_summary.iterrows():
        sig_ic = "***" if row.get('IC_pvalue', 1) < 0.01 else ("**" if row.get('IC_pvalue', 1) < 0.05 else ("*" if row.get('IC_pvalue', 1) < 0.1 else ""))
        sig_rank_ic = "***" if row.get('Rank_IC_pvalue', 1) < 0.01 else ("**" if row.get('Rank_IC_pvalue', 1) < 0.05 else ("*" if row.get('Rank_IC_pvalue', 1) < 0.1 else ""))

        sig_matrix.append({
            "Horizon": f"T+{row['horizon']}",
            "IC": f"{row['IC']:.4f} {sig_ic}",
            "Rank_IC": f"{row['Rank_IC']:.4f} {sig_rank_ic}",
            "N": int(row['n_samples'])
        })

    sig_df = pd.DataFrame(sig_matrix)

    # Display as table
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=sig_df.values, colLabels=sig_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code by significance
    for i in range(1, len(sig_df) + 1):
        if "***" in sig_df.iloc[i-1]['IC']:
            table[(i, 1)].set_facecolor('#90EE90')
        elif "**" in sig_df.iloc[i-1]['IC']:
            table[(i, 1)].set_facecolor('#FFFFE0')

    plt.title(f'{model_name.upper()} - IC Significance by Horizon\n(*** p<0.01, ** p<0.05, * p<0.1)', fontsize=14, fontweight='bold')
    plt.savefig(output_path / f"{model_name}_ic_significance_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Summary report
    valid_ics = decay_df_summary['IC'].dropna()

    if len(valid_ics) > 0:
        max_ic = float(valid_ics.max())
        max_ic_idx = valid_ics.idxmax()
        max_ic_horizon = int(decay_df_summary.loc[max_ic_idx, 'horizon'])
    else:
        max_ic = None
        max_ic_horizon = None

    report = {
        "model": model_name,
        "analysis_date": datetime.now().isoformat(),
        "target_horizon": target_horizon,
        "ic_at_target_horizon": float(decay_df_summary[decay_df_summary['horizon'] == target_horizon]['IC'].iloc[0]) if target_horizon in decay_df_summary['horizon'].values and not np.isnan(decay_df_summary[decay_df_summary['horizon'] == target_horizon]['IC'].iloc[0]) else None,
        "rank_ic_at_target_horizon": float(decay_df_summary[decay_df_summary['horizon'] == target_horizon]['Rank_IC'].iloc[0]) if target_horizon in decay_df_summary['horizon'].values and not np.isnan(decay_df_summary[decay_df_summary['horizon'] == target_horizon]['Rank_IC'].iloc[0]) else None,
        "ic_half_life_days": float(half_life_ic) if not np.isnan(half_life_ic) else None,
        "rank_ic_half_life_days": float(half_life_rank_ic) if not np.isnan(half_life_rank_ic) else None,
        "max_ic": max_ic,
        "max_ic_horizon": max_ic_horizon,
        "decay_curve": decay_df_summary.to_dict('records')
    }

    with open(output_path / f"{model_name}_alpha_decay_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== ALPHA DECAY ANALYSIS COMPLETE ===")
    print(f"Model: {model_name}")
    if report['ic_at_target_horizon'] is not None:
        print(f"IC at Target Horizon (T+{target_horizon}): {report['ic_at_target_horizon']:.4f}")
    else:
        print(f"IC at Target Horizon (T+{target_horizon}): N/A")

    if report['max_ic'] is not None:
        print(f"Max IC: {report['max_ic']:.4f} at T+{report['max_ic_horizon']}")
    else:
        print("Max IC: N/A (insufficient data)")

    if report['ic_half_life_days'] is not None:
        print(f"IC Half-Life: {report['ic_half_life_days']:.1f} days")
    else:
        print("IC Half-Life: Signal persists beyond tested horizons")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpha Decay Analysis")
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to predictions parquet file"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/factor_exports/factors/factors_all.parquet",
        help="Path to factor data with prices"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/alpha_decay",
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name"
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=10,
        help="Target horizon in days"
    )

    args = parser.parse_args()

    analyze_alpha_decay(
        predictions_file=args.predictions_file,
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        target_horizon=args.target_horizon
    )
