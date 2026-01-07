"""
Generate Architecture Diagram for Academic Paper
================================================
Creates a professional architecture diagram showing the Ridge Stacking
pipeline and model ensemble structure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def create_architecture_diagram(output_path: str):
    """
    Create Ridge Stacking architecture diagram.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Ridge Stacking Architecture', fontsize=18, fontweight='bold',
            ha='center', va='top')

    # Color scheme
    color_data = '#E3F2FD'
    color_model = '#BBDEFB'
    color_meta = '#90CAF9'
    color_output = '#64B5F6'

    # Layer 1: Input Data
    data_box = FancyBboxPatch((0.5, 7.5), 13, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_data, linewidth=2)
    ax.add_patch(data_box)
    ax.text(7, 8, 'Feature Matrix X\n(date, ticker) × 13 features', fontsize=11,
            ha='center', va='center', fontweight='bold')

    # Arrow down
    arrow = FancyArrowPatch((7, 7.5), (7, 7), arrowstyle='->', mutation_scale=30,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

    # Layer 2: Purged K-Fold CV
    cv_box = FancyBboxPatch((0.5, 6), 13, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='darkblue', facecolor='#FFF9C4', linewidth=2,
                            linestyle='--')
    ax.add_patch(cv_box)
    ax.text(7, 6.4, 'Purged & Embargoed 6-Fold CV (Gap=10d, Embargo=10d)', fontsize=10,
            ha='center', va='center', style='italic')

    # Arrow down
    arrow = FancyArrowPatch((7, 6), (7, 5.3), arrowstyle='->', mutation_scale=30,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

    # Layer 3: Base Models
    base_y = 4.2
    base_models = [
        ('Elastic Net\n(Linear)', 1.5),
        ('XGBoost\n(GBDT)', 4),
        ('CatBoost\n(GBDT)', 6.5),
        ('LambdaRank\n(Ranking)', 9),
        ('(+ Others)', 11.5)
    ]

    for model_name, x_pos in base_models:
        if model_name == '(+ Others)':
            box = FancyBboxPatch((x_pos-0.7, base_y), 1.4, 1, boxstyle="round,pad=0.05",
                                edgecolor='gray', facecolor='white', linewidth=1.5,
                                linestyle=':')
        else:
            box = FancyBboxPatch((x_pos-0.7, base_y), 1.4, 1, boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=color_model, linewidth=2)
        ax.add_patch(box)
        ax.text(x_pos, base_y+0.5, model_name, fontsize=9, ha='center', va='center',
                fontweight='bold' if model_name != '(+ Others)' else 'normal')

        # Arrow from CV to base model
        arrow = FancyArrowPatch((7, 5.3), (x_pos, base_y+1), arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='gray', alpha=0.5)
        ax.add_patch(arrow)

    # Text: "Train on different folds"
    ax.text(7, 3.8, 'Stage 1: Train base learners via purged CV', fontsize=10,
            ha='center', va='top', style='italic', color='darkblue')

    # Arrow down from base models to predictions
    for model_name, x_pos in base_models[:-1]:  # Exclude "(+ Others)"
        arrow = FancyArrowPatch((x_pos, base_y), (x_pos, 2.8), arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)

    # Layer 4: Out-of-fold predictions
    pred_y = 2
    pred_box = FancyBboxPatch((0.5, pred_y), 13, 0.6, boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(pred_box)
    ax.text(7, pred_y+0.3, 'Out-of-Fold Predictions: [pred_EN, pred_XGB, pred_CAT, pred_LR, ...]',
            fontsize=10, ha='center', va='center', fontweight='bold')

    # Arrow down to meta-learner
    arrow = FancyArrowPatch((7, pred_y), (7, 1.2), arrowstyle='->', mutation_scale=30,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

    # Layer 5: Ridge Meta-Learner
    meta_box = FancyBboxPatch((2, 0.3), 10, 0.7, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_meta, linewidth=2.5)
    ax.add_patch(meta_box)
    ax.text(7, 0.65, 'Stage 2: Ridge Meta-Learner', fontsize=12, ha='center', va='center',
            fontweight='bold')

    # Formula
    ax.text(7, 0.35, r'$\hat{y}_{meta} = \alpha_1 \cdot pred_{EN} + \alpha_2 \cdot pred_{XGB} + \alpha_3 \cdot pred_{CAT} + \alpha_4 \cdot pred_{LR}$',
            fontsize=9, ha='center', va='center', style='italic')

    # Legend box
    legend_elements = [
        mpatches.Patch(facecolor=color_data, edgecolor='black', label='Input Data'),
        mpatches.Patch(facecolor=color_model, edgecolor='black', label='Base Learners'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='black', label='OOF Predictions'),
        mpatches.Patch(facecolor=color_meta, edgecolor='black', label='Meta-Learner')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, fancybox=True)

    # Add annotations
    ax.annotate('Prevents\nOverfitting', xy=(12.5, 6.4), xytext=(13.5, 7),
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', lw=1))

    ax.annotate('Ensemble\nDiversity', xy=(11.5, 4.7), xytext=(12.5, 5.5),
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                arrowprops=dict(arrowstyle='->', lw=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
    plt.close()


def create_feature_heatmap(feature_importance_csv: str, output_path: str):
    """
    Create a heatmap of feature importance across models.
    """
    import pandas as pd
    import seaborn as sns

    # Load SHAP importance
    shap_df = pd.read_csv(feature_importance_csv)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot top 10 features
    top_features = shap_df.nlargest(10, 'mean_abs_shap')

    # Create bar plot with directional coloring
    colors = ['green' if x > 0 else 'red' for x in top_features['mean_shap']]

    ax.barh(range(len(top_features)), top_features['mean_abs_shap'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax.set_title('LambdaRank Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['mean_abs_shap'], i, f" {row['mean_abs_shap']:.4f}",
                va='center', fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.7, label='Positive Impact'),
        mpatches.Patch(facecolor='red', alpha=0.7, label='Negative Impact')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance heatmap saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Professional Visualizations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/professional_paper_analyses/visualizations",
        help="Output directory"
    )
    parser.add_argument(
        "--shap-importance",
        type=str,
        default="results/professional_paper_analyses/shap_analysis/shap_feature_importance.csv",
        help="Path to SHAP importance CSV"
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate architecture diagram
    print("Generating architecture diagram...")
    create_architecture_diagram(output_path / "ridge_stacking_architecture.png")

    # Generate feature importance heatmap
    if Path(args.shap_importance).exists():
        print("Generating feature importance visualization...")
        create_feature_heatmap(args.shap_importance, output_path / "feature_importance_chart.png")

    print("\n=== VISUALIZATIONS COMPLETE ===")
