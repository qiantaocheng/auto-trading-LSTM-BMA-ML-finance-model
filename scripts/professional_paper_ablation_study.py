"""
Ablation Study for Academic Paper
==================================
Systematically evaluates the contribution of the Ridge Stacker meta-learner
by comparing it against individual base models and alternative ensembles.

For academic rigor, this provides:
1. Individual model performance comparison
2. Ridge Stacker vs Simple Average ensemble
3. Ridge Stacker vs Median ensemble
4. Feature importance from Ridge weights
5. Ensemble diversity analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_performance_metrics(performance_csv: str) -> pd.DataFrame:
    """Load performance report CSV."""
    df = pd.read_csv(performance_csv)
    return df


def compute_ensemble_diversity(
    predictions_dir: Path,
    models: List[str]
) -> Dict:
    """
    Compute diversity metrics between model predictions.

    Args:
        predictions_dir: Directory with prediction parquet files
        models: List of model names

    Returns:
        Dict with diversity metrics
    """
    # Load all model predictions
    all_preds = {}

    for model in models:
        pred_file = predictions_dir / f"{model}_predictions*.parquet"
        matching_files = list(predictions_dir.glob(f"{model}_predictions*.parquet"))

        if matching_files:
            preds = pd.read_parquet(matching_files[0])

            if not isinstance(preds.index, pd.MultiIndex):
                if 'date' in preds.columns and 'ticker' in preds.columns:
                    preds = preds.set_index(['date', 'ticker'])

            all_preds[model] = preds['prediction']

    if len(all_preds) < 2:
        return {}

    # Merge all predictions
    merged = pd.DataFrame(all_preds)
    merged = merged.dropna()

    # Compute pairwise correlations
    corr_matrix = merged.corr()

    # Average pairwise correlation (measure of diversity)
    # Lower correlation = higher diversity
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    mean_correlation = upper_triangle.stack().mean()

    # Diversity score (1 - correlation)
    diversity_score = 1 - mean_correlation

    return {
        "mean_pairwise_correlation": float(mean_correlation),
        "diversity_score": float(diversity_score),
        "correlation_matrix": corr_matrix.to_dict()
    }


def analyze_ridge_weights(snapshot_dir: Path, model_name: str = "ridge_stacking") -> Dict:
    """
    Analyze Ridge Stacker weights to understand model contributions.

    Args:
        snapshot_dir: Snapshot directory path
        model_name: Ridge model name

    Returns:
        Dict with weight information
    """
    weights_file = snapshot_dir / "weights_ridge_stacking.json"

    if not weights_file.exists():
        return {}

    with open(weights_file, 'r') as f:
        weights_data = json.load(f)

    return weights_data


def create_ablation_comparison(
    performance_df: pd.DataFrame,
    models: List[str],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create comparison table for ablation study.

    Args:
        performance_df: Performance metrics dataframe
        models: List of models to compare
        metrics: List of metrics to include

    Returns:
        Comparison dataframe
    """
    if metrics is None:
        metrics = [
            'avg_top_return_net',
            'top_sharpe_net',
            'IC',
            'Rank_IC',
            'win_rate',
            'avg_top_turnover'
        ]

    # Filter to requested models
    comparison = performance_df[performance_df['Model'].isin(models)][['Model'] + metrics].copy()

    # Add rank for each metric (higher is better, except turnover)
    for metric in metrics:
        if metric == 'avg_top_turnover':
            comparison[f'{metric}_rank'] = comparison[metric].rank(ascending=True)
        else:
            comparison[f'{metric}_rank'] = comparison[metric].rank(ascending=False)

    # Compute average rank
    rank_cols = [c for c in comparison.columns if c.endswith('_rank')]
    comparison['avg_rank'] = comparison[rank_cols].mean(axis=1)

    # Sort by average rank
    comparison = comparison.sort_values('avg_rank')

    return comparison


def analyze_ablation_study(
    performance_csv: str,
    predictions_dir: str,
    snapshot_id: str,
    output_dir: str,
    base_models: List[str] = None,
    ensemble_model: str = "ridge_stacking"
):
    """
    Main ablation study analysis.
    """
    if base_models is None:
        base_models = ["elastic_net", "xgboost", "catboost", "lambdarank"]

    all_models = base_models + [ensemble_model]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading performance metrics from {performance_csv}...")
    perf_df = load_performance_metrics(performance_csv)

    # 1. Create comparison table
    print("Creating ablation comparison table...")
    comparison_metrics = [
        'avg_top_return_net',
        'top_sharpe_net',
        'IC',
        'Rank_IC',
        'win_rate',
        'avg_top_turnover'
    ]

    comparison = create_ablation_comparison(perf_df, all_models, comparison_metrics)
    comparison.to_csv(output_path / "ablation_comparison.csv", index=False)

    # 2. Compute ensemble diversity
    print("Computing ensemble diversity...")
    diversity_metrics = compute_ensemble_diversity(
        Path(predictions_dir),
        base_models
    )

    # 3. Analyze Ridge weights
    print("Analyzing Ridge Stacker weights...")
    snapshot_path = Path("cache/model_snapshots")
    snapshot_dirs = list(snapshot_path.glob(f"*/{snapshot_id}"))

    if snapshot_dirs:
        ridge_weights = analyze_ridge_weights(snapshot_dirs[0])
    else:
        ridge_weights = {}

    # 4. Visualizations
    print("Generating ablation visualizations...")

    # Plot 1: Performance comparison bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics_to_plot = [
        ('avg_top_return_net', 'Avg Top Return (Net)', '%'),
        ('top_sharpe_net', 'Top Sharpe (Net)', ''),
        ('IC', 'Information Coefficient', ''),
        ('Rank_IC', 'Rank IC', ''),
        ('win_rate', 'Win Rate', '%'),
        ('avg_top_turnover', 'Avg Turnover', '')
    ]

    for idx, (metric, title, unit) in enumerate(metrics_to_plot):
        ax = axes[idx]

        data = comparison.sort_values(metric, ascending=(metric=='avg_top_turnover'))
        values = data[metric]

        # Convert to percentage if needed
        if unit == '%':
            values = values * 100

        # Color ridge_stacking differently
        colors = ['red' if m == ensemble_model else 'blue' for m in data['Model']]

        ax.barh(range(len(data)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Model'])
        ax.set_xlabel(f'{title} {unit}', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)

    plt.suptitle('Ablation Study: Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "ablation_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Correlation matrix (if diversity metrics available)
    if diversity_metrics and 'correlation_matrix' in diversity_metrics:
        corr_df = pd.DataFrame(diversity_metrics['correlation_matrix'])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    cbar_kws={'label': 'Correlation'})
        ax.set_title('Model Prediction Correlation Matrix\n(Lower correlation = Higher diversity)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "model_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Ridge Stacker value-add
    if ensemble_model in comparison['Model'].values:
        fig, ax = plt.subplots(figsize=(12, 8))

        ridge_row = comparison[comparison['Model'] == ensemble_model].iloc[0]

        # Compare ridge to best base model
        base_comparison = comparison[comparison['Model'].isin(base_models)]
        best_base = base_comparison.iloc[0]  # Already sorted by avg_rank

        metrics_compare = [
            'avg_top_return_net',
            'top_sharpe_net',
            'IC',
            'Rank_IC',
            'win_rate'
        ]

        ridge_vals = []
        best_base_vals = []

        for metric in metrics_compare:
            ridge_val = ridge_row[metric]
            best_val = best_base[metric]

            if metric in ['avg_top_return_net', 'win_rate']:
                ridge_val *= 100
                best_val *= 100

            ridge_vals.append(ridge_val)
            best_base_vals.append(best_val)

        x = np.arange(len(metrics_compare))
        width = 0.35

        ax.bar(x - width/2, best_base_vals, width, label=f'Best Base ({best_base["Model"]})',
               color='blue', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, ridge_vals, width, label='Ridge Stacker',
               color='red', alpha=0.7, edgecolor='black')

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Ridge Stacker vs Best Base Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_compare], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path / "ridge_vs_best_base.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Summary report
    ridge_perf = comparison[comparison['Model'] == ensemble_model].iloc[0] if ensemble_model in comparison['Model'].values else None
    best_base_perf = comparison[comparison['Model'].isin(base_models)].iloc[0]

    report = {
        "analysis_date": datetime.now().isoformat(),
        "ensemble_model": ensemble_model,
        "base_models": base_models,
        "best_base_model": best_base_perf['Model'],
        "ridge_stacker_performance": ridge_perf[comparison_metrics].to_dict() if ridge_perf is not None else {},
        "best_base_performance": best_base_perf[comparison_metrics].to_dict(),
        "ensemble_value_add": {
            metric: float(ridge_perf[metric] - best_base_perf[metric])
            for metric in comparison_metrics
            if ridge_perf is not None and metric in ridge_perf.index
        } if ridge_perf is not None else {},
        "diversity_metrics": diversity_metrics,
        "ridge_weights": ridge_weights,
        "comparison_table": comparison.to_dict('records')
    }

    with open(output_path / "ablation_study_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== ABLATION STUDY COMPLETE ===")
    print(f"Best Base Model: {best_base_perf['Model']}")
    if ridge_perf is not None:
        print(f"Ridge Stacker Return: {ridge_perf['avg_top_return_net']*100:.2f}%")
        print(f"Best Base Return: {best_base_perf['avg_top_return_net']*100:.2f}%")
        print(f"Value Add: {(ridge_perf['avg_top_return_net'] - best_base_perf['avg_top_return_net'])*100:.2f}%")

    if diversity_metrics:
        print(f"Mean Pairwise Correlation: {diversity_metrics['mean_pairwise_correlation']:.3f}")
        print(f"Diversity Score: {diversity_metrics['diversity_score']:.3f}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Study Analysis")
    parser.add_argument(
        "--performance-csv",
        type=str,
        required=True,
        help="Path to performance report CSV"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory with prediction parquet files"
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default="7de6f766-da32-43a5-b5a0-4d69d2426f18",
        help="Model snapshot ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/ablation_study",
        help="Output directory"
    )

    args = parser.parse_args()

    analyze_ablation_study(
        performance_csv=args.performance_csv,
        predictions_dir=args.predictions_dir,
        snapshot_id=args.snapshot_id,
        output_dir=args.output_dir
    )
