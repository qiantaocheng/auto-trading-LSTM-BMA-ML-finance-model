"""
Prediction Stability Analysis for Academic Paper
=================================================
Measures the stability of model predictions over time using rank correlation
between consecutive periods. High stability indicates lower turnover and
more reliable signals.

For academic rigor, this provides:
1. Day-over-day rank correlation of predictions
2. Turnover analysis
3. Prediction volatility metrics
4. Stability vs performance trade-off analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_rank_correlation_stability(
    predictions: pd.DataFrame,
    pred_col: str = "prediction"
) -> pd.DataFrame:
    """
    Compute day-over-day rank correlation of predictions.

    Args:
        predictions: DataFrame with predictions (MultiIndex: date, ticker)
        pred_col: Prediction column name

    Returns:
        DataFrame with date and rank correlation to previous period
    """
    dates = predictions.index.get_level_values('date').unique().sort_values()

    stability_results = []

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        # Get predictions for both dates
        prev_preds = predictions.xs(prev_date, level='date')[[pred_col]]
        curr_preds = predictions.xs(curr_date, level='date')[[pred_col]]

        # Merge on ticker (inner join - only tickers present in both periods)
        merged = prev_preds.merge(
            curr_preds,
            left_index=True,
            right_index=True,
            suffixes=('_prev', '_curr')
        )

        if len(merged) < 10:
            continue

        # Compute rank correlation
        rank_corr, rank_pval = stats.spearmanr(
            merged[f'{pred_col}_prev'],
            merged[f'{pred_col}_curr']
        )

        # Also compute Pearson correlation
        pearson_corr, pearson_pval = stats.pearsonr(
            merged[f'{pred_col}_prev'],
            merged[f'{pred_col}_curr']
        )

        stability_results.append({
            'date': curr_date,
            'prev_date': prev_date,
            'rank_correlation': rank_corr,
            'rank_pvalue': rank_pval,
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_pval,
            'n_common_tickers': len(merged)
        })

    return pd.DataFrame(stability_results)


def compute_prediction_volatility(
    predictions: pd.DataFrame,
    pred_col: str = "prediction",
    window: int = 5
) -> pd.DataFrame:
    """
    Compute rolling volatility of predictions for each ticker.

    Args:
        predictions: DataFrame with predictions (MultiIndex: date, ticker)
        pred_col: Prediction column name
        window: Rolling window size

    Returns:
        DataFrame with volatility metrics by ticker
    """
    # Unstack to get tickers as columns
    pred_wide = predictions[pred_col].unstack(level='ticker')

    # Compute rolling std for each ticker
    rolling_std = pred_wide.rolling(window).std()

    # Aggregate statistics
    volatility_stats = {
        'mean_volatility': rolling_std.mean().mean(),
        'median_volatility': rolling_std.median().median(),
        'max_volatility': rolling_std.max().max(),
        'min_volatility': rolling_std.min().min(),
        'std_volatility': rolling_std.std().mean()
    }

    return volatility_stats, rolling_std


def compute_top_k_overlap(
    predictions: pd.DataFrame,
    pred_col: str = "prediction",
    k: int = 30
) -> pd.DataFrame:
    """
    Compute overlap in top-K stocks between consecutive periods.

    Args:
        predictions: DataFrame with predictions
        pred_col: Prediction column name
        k: Number of top stocks to consider

    Returns:
        DataFrame with overlap metrics
    """
    dates = predictions.index.get_level_values('date').unique().sort_values()

    overlap_results = []

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        # Get top K for both periods
        prev_preds = predictions.xs(prev_date, level='date')[[pred_col]].sort_values(
            pred_col, ascending=False
        ).head(k)
        curr_preds = predictions.xs(curr_date, level='date')[[pred_col]].sort_values(
            pred_col, ascending=False
        ).head(k)

        # Compute overlap
        prev_tickers = set(prev_preds.index)
        curr_tickers = set(curr_preds.index)

        overlap_count = len(prev_tickers.intersection(curr_tickers))
        overlap_pct = overlap_count / k

        # Compute turnover (1 - overlap)
        turnover = 1 - overlap_pct

        overlap_results.append({
            'date': curr_date,
            'prev_date': prev_date,
            'overlap_count': overlap_count,
            'overlap_pct': overlap_pct,
            'turnover': turnover,
            'k': k
        })

    return pd.DataFrame(overlap_results)


def analyze_stability(
    predictions_file: str,
    output_dir: str,
    model_name: str = "lambdarank",
    pred_col: str = "prediction",
    top_k: int = 30,
    volatility_window: int = 5
):
    """
    Main stability analysis function.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_file}...")
    predictions = pd.read_parquet(predictions_file)

    # Ensure MultiIndex
    if not isinstance(predictions.index, pd.MultiIndex):
        if 'date' in predictions.columns and 'ticker' in predictions.columns:
            predictions = predictions.set_index(['date', 'ticker'])

    if pred_col not in predictions.columns:
        print(f"Error: Column {pred_col} not found")
        return

    # 1. Rank correlation stability
    print("Computing rank correlation stability...")
    stability_df = compute_rank_correlation_stability(predictions, pred_col)

    if len(stability_df) == 0:
        print("Warning: Insufficient data for stability analysis")
        return

    stability_df.to_csv(output_path / f"{model_name}_rank_correlation_stability.csv", index=False)

    # 2. Top-K overlap analysis
    print(f"Computing Top-{top_k} overlap...")
    overlap_df = compute_top_k_overlap(predictions, pred_col, top_k)
    overlap_df.to_csv(output_path / f"{model_name}_top{top_k}_overlap.csv", index=False)

    # 3. Prediction volatility
    print("Computing prediction volatility...")
    volatility_stats, rolling_volatility = compute_prediction_volatility(
        predictions, pred_col, volatility_window
    )

    # Summary statistics
    mean_rank_corr = stability_df['rank_correlation'].mean()
    median_rank_corr = stability_df['rank_correlation'].median()
    std_rank_corr = stability_df['rank_correlation'].std()

    mean_overlap = overlap_df['overlap_pct'].mean()
    median_overlap = overlap_df['overlap_pct'].median()
    mean_turnover = overlap_df['turnover'].mean()

    # Visualizations
    print("Generating stability visualizations...")

    # Plot 1: Rank Correlation Over Time
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(stability_df['date'], stability_df['rank_correlation'], 'o-',
                 linewidth=2, markersize=6, color='blue', alpha=0.7)
    axes[0].axhline(mean_rank_corr, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {mean_rank_corr:.3f}')
    axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[0].set_ylabel('Rank Correlation', fontsize=12)
    axes[0].set_title(f'{model_name.upper()} - Prediction Stability Over Time',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Top-K Overlap/Turnover
    axes[1].plot(overlap_df['date'], overlap_df['overlap_pct'] * 100, 'o-',
                 linewidth=2, markersize=6, color='green', alpha=0.7, label='Overlap %')
    axes[1].axhline(mean_overlap * 100, color='green', linestyle='--',
                    linewidth=2, label=f'Mean Overlap: {mean_overlap*100:.1f}%')

    ax2 = axes[1].twinx()
    ax2.plot(overlap_df['date'], overlap_df['turnover'] * 100, 's-',
             linewidth=2, markersize=6, color='orange', alpha=0.7, label='Turnover %')
    ax2.axhline(mean_turnover * 100, color='orange', linestyle='--',
                linewidth=2, label=f'Mean Turnover: {mean_turnover*100:.1f}%')

    axes[1].set_ylabel(f'Top-{top_k} Overlap (%)', fontsize=12, color='green')
    ax2.set_ylabel('Turnover (%)', fontsize=12, color='orange')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_stability_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Rank correlation distribution
    axes[0].hist(stability_df['rank_correlation'], bins=20, alpha=0.7,
                 color='blue', edgecolor='black')
    axes[0].axvline(mean_rank_corr, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {mean_rank_corr:.3f}')
    axes[0].set_xlabel('Rank Correlation', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Day-over-Day Rank Correlation',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Turnover distribution
    axes[1].hist(overlap_df['turnover'] * 100, bins=20, alpha=0.7,
                 color='orange', edgecolor='black')
    axes[1].axvline(mean_turnover * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {mean_turnover*100:.1f}%')
    axes[1].set_xlabel('Turnover (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Distribution of Top-{top_k} Turnover',
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_stability_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Scatter - Stability vs Turnover
    fig, ax = plt.subplots(figsize=(10, 8))

    # Merge data for scatter
    merged_scatter = stability_df.merge(
        overlap_df,
        left_on='date',
        right_on='date',
        how='inner'
    )

    ax.scatter(merged_scatter['rank_correlation'],
               merged_scatter['turnover'] * 100,
               alpha=0.6, s=100, color='purple', edgecolor='black')

    # Add trend line
    if len(merged_scatter) > 2:
        z = np.polyfit(merged_scatter['rank_correlation'],
                      merged_scatter['turnover'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged_scatter['rank_correlation'].min(),
                            merged_scatter['rank_correlation'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8,
                label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')

    ax.set_xlabel('Rank Correlation (Stability)', fontsize=12)
    ax.set_ylabel('Turnover (%)', fontsize=12)
    ax.set_title('Stability vs Turnover Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_stability_vs_turnover.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Summary report
    report = {
        "model": model_name,
        "analysis_date": datetime.now().isoformat(),
        "n_periods": int(len(stability_df)),
        "rank_correlation_stats": {
            "mean": float(mean_rank_corr),
            "median": float(median_rank_corr),
            "std": float(std_rank_corr),
            "min": float(stability_df['rank_correlation'].min()),
            "max": float(stability_df['rank_correlation'].max())
        },
        "top_k_overlap_stats": {
            "k": top_k,
            "mean_overlap_pct": float(mean_overlap * 100),
            "median_overlap_pct": float(median_overlap * 100),
            "mean_turnover_pct": float(mean_turnover * 100)
        },
        "prediction_volatility_stats": {
            k: float(v) if not isinstance(v, (int, float)) else v
            for k, v in volatility_stats.items()
        }
    }

    with open(output_path / f"{model_name}_stability_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== STABILITY ANALYSIS COMPLETE ===")
    print(f"Model: {model_name}")
    print(f"Mean Rank Correlation: {mean_rank_corr:.3f} (higher = more stable)")
    print(f"Median Rank Correlation: {median_rank_corr:.3f}")
    print(f"Mean Top-{top_k} Overlap: {mean_overlap*100:.1f}%")
    print(f"Mean Turnover: {mean_turnover*100:.1f}%")
    print(f"Prediction Volatility: {volatility_stats['mean_volatility']:.4f}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Stability Analysis")
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to predictions parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/stability_test",
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top K stocks for overlap analysis"
    )

    args = parser.parse_args()

    analyze_stability(
        predictions_file=args.predictions_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        top_k=args.top_k
    )
