#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick script to check grid search progress"""

import pandas as pd
from pathlib import Path

summary_file = Path("results/t10_feature_combo_grid_parallel_run500_no_overlap/combo_summary.csv")

if not summary_file.exists():
    print("❌ Summary file not found!")
    exit(1)

df = pd.read_csv(summary_file)

print(f"📊 Grid Search Progress")
print(f"=" * 60)
print(f"Total combos completed: {len(df)} / 500 ({len(df)/500*100:.1f}%)")
print(f"Last combo: {df.iloc[-1]['combo']}")
print(f"Last combo k (features): {df.iloc[-1]['k']}")

print(f"\n🏆 Best Results So Far:")
print(f"  XGBoost:     {df['xgboost_avg_top_return'].max():.4f} ({df.loc[df['xgboost_avg_top_return'].idxmax(), 'combo']})")
print(f"  CatBoost:    {df['catboost_avg_top_return'].max():.4f} ({df.loc[df['catboost_avg_top_return'].idxmax(), 'combo']})")
print(f"  LambdaRank:  {df['lambdarank_avg_top_return'].max():.4f} ({df.loc[df['lambdarank_avg_top_return'].idxmax(), 'combo']})")
print(f"  ElasticNet:  {df['elastic_net_avg_top_return'].max():.4f} ({df.loc[df['elastic_net_avg_top_return'].idxmax(), 'combo']})")

print(f"\n📈 Average Performance Across All Combos:")
print(f"  XGBoost:     {df['xgboost_avg_top_return'].mean():.4f}")
print(f"  CatBoost:    {df['catboost_avg_top_return'].mean():.4f}")
print(f"  LambdaRank:  {df['lambdarank_avg_top_return'].mean():.4f}")
print(f"  ElasticNet:  {df['elastic_net_avg_top_return'].mean():.4f}")


