"""
Feature Taxonomy and IC Analysis for Academic Paper
====================================================
Categorizes features into economic groups and computes Information Coefficients
by category to explain what drives model performance.

For academic rigor, this provides:
1. Feature taxonomy (Momentum, Mean-Reversion, Volatility, Quality, Liquidity)
2. IC and Rank IC by feature category
3. Statistical significance tests
4. Feature correlation heatmap within categories
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Feature taxonomy based on financial theory
FEATURE_TAXONOMY = {
    "Momentum": [
        "liquid_momentum",
        "near_52w_high",
        "price_ma60_deviation",
        "rsi_21",
    ],
    "Mean_Reversion": [
        "obv_divergence",
        "bollinger_squeeze",
    ],
    "Volatility": [
        "ivol_20",
        "hist_vol_40d",
        "vol_ratio_20d",
        "atr_ratio",
        "ret_skew_20d",
    ],
    "Quality_Trend": [
        "trend_r2_60",
        "blowoff_ratio",
    ],
}


def compute_ic_metrics(
    data: pd.DataFrame,
    feature_col: str,
    target_col: str = "target"
) -> Dict:
    """
    Compute IC, Rank IC, and statistical tests for a single feature.

    Returns:
        Dict with IC, Rank_IC, t_stat, p_value, n_samples
    """
    # Remove NaNs
    valid_mask = data[[feature_col, target_col]].notna().all(axis=1)
    valid_data = data[valid_mask]

    if len(valid_data) < 30:
        return {
            "IC": np.nan,
            "Rank_IC": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_samples": len(valid_data)
        }

    # Compute IC (Pearson correlation)
    ic, ic_pvalue = stats.pearsonr(valid_data[feature_col], valid_data[target_col])

    # Compute Rank IC (Spearman correlation)
    rank_ic, rank_ic_pvalue = stats.spearmanr(valid_data[feature_col], valid_data[target_col])

    # T-statistic for IC
    n = len(valid_data)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2) if abs(ic) < 0.9999 else np.nan

    return {
        "IC": ic,
        "Rank_IC": rank_ic,
        "IC_pvalue": ic_pvalue,
        "Rank_IC_pvalue": rank_ic_pvalue,
        "t_stat": t_stat,
        "n_samples": n
    }


def compute_time_series_ic(
    data: pd.DataFrame,
    feature_col: str,
    target_col: str = "target",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Compute cross-sectional IC for each date.

    Returns:
        DataFrame with date and IC values
    """
    ic_series = []

    for date in data.index.get_level_values(date_col).unique():
        date_slice = data.xs(date, level=date_col)

        valid_mask = date_slice[[feature_col, target_col]].notna().all(axis=1)
        valid_data = date_slice[valid_mask]

        if len(valid_data) >= 10:  # Minimum 10 stocks per date
            ic, _ = stats.pearsonr(valid_data[feature_col], valid_data[target_col])
            rank_ic, _ = stats.spearmanr(valid_data[feature_col], valid_data[target_col])
        else:
            ic = np.nan
            rank_ic = np.nan

        ic_series.append({
            'date': date,
            'IC': ic,
            'Rank_IC': rank_ic,
            'n_stocks': len(valid_data)
        })

    return pd.DataFrame(ic_series)


def analyze_feature_taxonomy(
    data_file: str,
    output_dir: str,
    target_col: str = "target"
):
    """
    Main analysis function for feature taxonomy.
    """
    print(f"Loading data from {data_file}...")
    data = pd.read_parquet(data_file)

    # Ensure MultiIndex format
    if not isinstance(data.index, pd.MultiIndex):
        if 'date' in data.columns and 'ticker' in data.columns:
            data = data.set_index(['date', 'ticker'])
        else:
            raise ValueError("Data must have MultiIndex(date, ticker) or columns [date, ticker]")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Computing IC metrics by feature...")

    # 1. Compute IC for each individual feature
    individual_ic_results = []

    all_features = [f for category in FEATURE_TAXONOMY.values() for f in category]

    for feature in all_features:
        if feature not in data.columns:
            print(f"  Warning: {feature} not found in data")
            continue

        print(f"  Analyzing {feature}...")
        metrics = compute_ic_metrics(data, feature, target_col)

        # Find category
        category = next(
            (cat for cat, feats in FEATURE_TAXONOMY.items() if feature in feats),
            "Unknown"
        )

        individual_ic_results.append({
            "feature": feature,
            "category": category,
            **metrics
        })

    # Create DataFrame
    ic_df = pd.DataFrame(individual_ic_results)
    ic_df['IC_abs_sort'] = ic_df['IC'].abs()
    ic_df = ic_df.sort_values("IC_abs_sort", ascending=False)

    # Save individual feature IC
    ic_df.to_csv(output_path / "feature_ic_analysis.csv", index=False)
    print(f"\nSaved individual feature IC to {output_path / 'feature_ic_analysis.csv'}")

    # 2. Aggregate by category
    category_summary = []

    for category, features in FEATURE_TAXONOMY.items():
        category_features = [f for f in features if f in data.columns]

        if not category_features:
            continue

        # Get ICs for this category
        category_ics = ic_df[ic_df['category'] == category]

        category_summary.append({
            "category": category,
            "n_features": len(category_features),
            "mean_IC": category_ics['IC'].mean(),
            "median_IC": category_ics['IC'].median(),
            "mean_abs_IC": category_ics['IC'].abs().mean(),
            "mean_Rank_IC": category_ics['Rank_IC'].mean(),
            "mean_abs_Rank_IC": category_ics['Rank_IC'].abs().mean(),
            "mean_t_stat": category_ics['t_stat'].mean(),
            "significant_features": (category_ics['IC_pvalue'] < 0.05).sum(),
            "top_features": ', '.join(
                category_ics.nlargest(3, 'IC', keep='all')['feature'].tolist()
            )
        })

    category_df = pd.DataFrame(category_summary)
    category_df = category_df.sort_values("mean_abs_IC", ascending=False)

    # Save category summary
    category_df.to_csv(output_path / "feature_category_summary.csv", index=False)
    print(f"Saved category summary to {output_path / 'feature_category_summary.csv'}")

    # 3. Create visualizations
    print("\nGenerating visualizations...")

    # Plot 1: IC by category (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    ic_df_plot = ic_df.dropna(subset=['IC'])

    categories_order = category_df.sort_values('mean_abs_IC', ascending=False)['category'].tolist()

    sns.boxplot(
        data=ic_df_plot,
        x='category',
        y='IC',
        order=categories_order,
        ax=ax
    )
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Information Coefficient Distribution by Feature Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Category', fontsize=12)
    ax.set_ylabel('IC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "ic_by_category_boxplot.png", dpi=300)
    plt.close()

    # Plot 2: Top 10 features by absolute IC
    fig, ax = plt.subplots(figsize=(10, 8))
    ic_df['IC_abs'] = ic_df['IC'].abs()
    top_features = ic_df.nlargest(10, 'IC_abs')

    colors = top_features['IC'].apply(lambda x: 'green' if x > 0 else 'red')

    ax.barh(range(len(top_features)), top_features['IC'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Information Coefficient', fontsize=12)
    ax.set_title('Top 10 Features by Absolute IC', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path / "top10_features_ic.png", dpi=300)
    plt.close()

    # Plot 3: IC vs Rank IC scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for category in FEATURE_TAXONOMY.keys():
        category_data = ic_df[ic_df['category'] == category]
        ax.scatter(
            category_data['IC'],
            category_data['Rank_IC'],
            label=category,
            alpha=0.7,
            s=100
        )

    ax.plot([-0.05, 0.05], [-0.05, 0.05], 'k--', alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Pearson IC', fontsize=12)
    ax.set_ylabel('Spearman Rank IC', fontsize=12)
    ax.set_title('IC vs Rank IC by Feature Category', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path / "ic_vs_rankic_scatter.png", dpi=300)
    plt.close()

    # 4. Generate summary report
    report = {
        "analysis_date": datetime.now().isoformat(),
        "data_file": data_file,
        "n_samples": len(data),
        "n_features_analyzed": len(individual_ic_results),
        "n_categories": len(FEATURE_TAXONOMY),
        "best_category": category_df.iloc[0]['category'],
        "best_category_mean_abs_ic": float(category_df.iloc[0]['mean_abs_IC']),
        "top_3_features": ic_df.nlargest(3, 'IC_abs')[['feature', 'IC', 'category']].to_dict('records'),
        "category_rankings": category_df.to_dict('records')
    }

    with open(output_path / "feature_taxonomy_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_path}")
    print("\n=== SUMMARY ===")
    print(f"Best performing category: {report['best_category']} (mean |IC| = {report['best_category_mean_abs_ic']:.4f})")
    print("\nTop 3 features:")
    for i, feat in enumerate(report['top_3_features'], 1):
        print(f"  {i}. {feat['feature']} (IC={feat['IC']:.4f}, {feat['category']})")

    return ic_df, category_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Taxonomy and IC Analysis")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/factor_exports/factors/factors_all.parquet",
        help="Path to factor data parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/feature_taxonomy",
        help="Output directory for results"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column name"
    )

    args = parser.parse_args()

    analyze_feature_taxonomy(
        data_file=args.data_file,
        output_dir=args.output_dir,
        target_col=args.target_col
    )
