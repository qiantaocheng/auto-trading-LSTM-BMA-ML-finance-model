"""
SHAP Feature Importance Analysis for Academic Paper
===================================================
Generates SHAP (SHapley Additive exPlanations) values for the LambdaRank model
to explain feature importance and interactions.

For academic rigor, this provides:
1. Global feature importance via SHAP values
2. Feature interaction analysis
3. Directional impact (positive/negative)
4. SHAP summary and dependence plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def load_model_from_snapshot(
    snapshot_id: str,
    model_name: str = "lambdarank",
    cache_dir: str = "cache/model_snapshots"
) -> tuple:
    """
    Load a trained LightGBM model from snapshot cache.

    Args:
        snapshot_id: Snapshot UUID
        model_name: Model name (e.g., 'lambdarank')
        cache_dir: Base directory for model snapshots

    Returns:
        Tuple of (model, feature_cols) or (None, None) if not found
    """
    cache_path = Path(cache_dir)

    # Find snapshot directory (organized by date)
    snapshot_dirs = list(cache_path.glob(f"*/{snapshot_id}"))

    if not snapshot_dirs:
        print(f"Error: Snapshot {snapshot_id} not found in {cache_dir}")
        return None, None

    snapshot_dir = snapshot_dirs[0]

    # Try different model file naming conventions
    possible_files = [
        snapshot_dir / f"{model_name}_lgb.txt",
        snapshot_dir / f"{model_name}_model.txt",
        snapshot_dir / f"{model_name}.txt",
    ]

    model_file = None
    for possible_file in possible_files:
        if possible_file.exists():
            model_file = possible_file
            break

    if model_file is None:
        print(f"Error: Model file not found. Tried: {[str(f) for f in possible_files]}")
        return None, None

    print(f"Loading model from {model_file}...")
    model = lgb.Booster(model_file=str(model_file))

    # Load metadata to get actual feature names
    meta_file = snapshot_dir / f"{model_name}_meta.json"
    feature_cols = None

    if meta_file.exists():
        print(f"Loading metadata from {meta_file}...")
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            feature_cols = meta.get('base_cols', None)
            print(f"Found {len(feature_cols)} features in metadata: {feature_cols}")

    return model, feature_cols


def compute_shap_values(
    model: lgb.Booster,
    data: pd.DataFrame,
    feature_cols: list,
    sample_size: int = 5000,
    random_state: int = 42
) -> tuple:
    """
    Compute SHAP values for a sample of data.

    Args:
        model: Trained LightGBM model
        data: DataFrame with features
        feature_cols: List of feature column names
        sample_size: Number of samples to use (for speed)
        random_state: Random seed

    Returns:
        (shap_values, X_sample, explainer)
    """
    # Sample data for SHAP (can be slow on full dataset)
    if len(data) > sample_size:
        print(f"Sampling {sample_size} rows from {len(data)} total...")
        X_sample = data[feature_cols].sample(n=sample_size, random_state=random_state)
    else:
        X_sample = data[feature_cols]

    # Remove any NaN values
    X_sample = X_sample.dropna()

    print(f"Computing SHAP values for {len(X_sample)} samples...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample, explainer


def analyze_shap_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20
):
    """
    Analyze and visualize SHAP feature importance.

    Args:
        shap_values: SHAP values array
        X_sample: Feature dataframe
        output_dir: Output directory
        top_n: Number of top features to display
    """
    # Compute mean absolute SHAP values
    shap_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })

    shap_importance = shap_importance.sort_values('mean_abs_shap', ascending=False)

    # Save importance table
    shap_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"Saved SHAP importance to {output_dir / 'shap_feature_importance.csv'}")

    # Plot 1: SHAP Summary Plot (beeswarm)
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False
    )
    plt.title("SHAP Feature Importance (LambdaRank)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: SHAP Bar Plot (mean absolute)
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=top_n,
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Top 6 SHAP Dependence Plots
    print("Generating SHAP dependence plots...")
    top_6_features = shap_importance.head(6)['feature'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(top_6_features):
        shap.dependence_plot(
            feature,
            shap_values,
            X_sample,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f"SHAP Dependence: {feature}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence_top6.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: SHAP Directional Impact
    fig, ax = plt.subplots(figsize=(10, 8))
    top_15 = shap_importance.head(15)

    colors = top_15['mean_shap'].apply(lambda x: 'green' if x > 0 else 'red')

    ax.barh(range(len(top_15)), top_15['mean_shap'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'])
    ax.set_xlabel('Mean SHAP Value (Directional)', fontsize=12)
    ax.set_title('SHAP Directional Impact (Top 15 Features)', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_directional_impact.png", dpi=300, bbox_inches='tight')
    plt.close()

    return shap_importance


def main(
    snapshot_id: str,
    data_file: str,
    output_dir: str,
    model_name: str = "lambdarank",
    sample_size: int = 5000,
    feature_cols: Optional[list] = None
):
    """
    Main SHAP analysis function.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and get feature names from metadata
    model, meta_feature_cols = load_model_from_snapshot(snapshot_id, model_name)
    if model is None:
        return

    # Use metadata feature names if available, otherwise use provided feature_cols
    if meta_feature_cols is not None:
        feature_cols = meta_feature_cols
        print(f"Using {len(feature_cols)} features from metadata")
    elif feature_cols is None:
        feature_cols = model.feature_name()
        print(f"Using {len(feature_cols)} features from model")

    # Load data
    print(f"Loading data from {data_file}...")
    data = pd.read_parquet(data_file)

    # Ensure MultiIndex format
    if not isinstance(data.index, pd.MultiIndex):
        if 'date' in data.columns and 'ticker' in data.columns:
            data = data.set_index(['date', 'ticker'])
        else:
            raise ValueError("Data must have MultiIndex(date, ticker)")

    # Verify all features exist in data
    missing_features = [f for f in feature_cols if f not in data.columns]
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
        feature_cols = [f for f in feature_cols if f in data.columns]

    # Filter to valid data (no NaN targets)
    if 'target' in data.columns:
        data = data[data['target'].notna()]

    # Compute SHAP values
    shap_values, X_sample, explainer = compute_shap_values(
        model, data, feature_cols, sample_size
    )

    # Analyze and visualize
    shap_importance = analyze_shap_importance(
        shap_values, X_sample, output_path
    )

    # Save summary report
    report = {
        "analysis_date": datetime.now().isoformat(),
        "snapshot_id": snapshot_id,
        "model_name": model_name,
        "data_file": data_file,
        "n_features": len(feature_cols),
        "n_samples_analyzed": len(X_sample),
        "top_5_features": shap_importance.head(5)[['feature', 'mean_abs_shap', 'mean_shap']].to_dict('records')
    }

    with open(output_path / "shap_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n=== SHAP ANALYSIS COMPLETE ===")
    print(f"Results saved to {output_path}")
    print("\nTop 5 Most Important Features:")
    for i, row in shap_importance.head(5).iterrows():
        direction = "↑" if row['mean_shap'] > 0 else "↓"
        print(f"  {row['feature']}: {row['mean_abs_shap']:.6f} {direction}")

    return shap_importance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis")
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default="7de6f766-da32-43a5-b5a0-4d69d2426f18",
        help="Model snapshot UUID"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambdarank",
        help="Model name to analyze"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/factor_exports/factors/factors_all.parquet",
        help="Path to factor data parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/shap_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of samples for SHAP computation"
    )

    args = parser.parse_args()

    main(
        snapshot_id=args.snapshot_id,
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_size=args.sample_size
    )
