#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 CatBoost 和 LambdaRank 的累计收益"""

import pandas as pd
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_032758")

cb = pd.read_csv(run_dir / "catboost_top20_timeseries.csv")
lr = pd.read_csv(run_dir / "lambdarank_top20_timeseries.csv")

print("="*80)
print("CatBoost 和 LambdaRank 累计收益（基于非重叠回测）")
print("="*80)

print("\n=== CatBoost ===")
print(f"最终累计收益 (Gross): {cb['cum_top_return'].iloc[-1]:.2f}%")
print(f"最终累计收益 (Net):   {cb['cum_top_return_net'].iloc[-1]:.2f}%")
print(f"时间序列行数: {len(cb)} (每10天一期)")

print("\n=== LambdaRank ===")
print(f"最终累计收益 (Gross): {lr['cum_top_return'].iloc[-1]:.2f}%")
print(f"最终累计收益 (Net):   {lr['cum_top_return_net'].iloc[-1]:.2f}%")
print(f"时间序列行数: {len(lr)} (每10天一期)")

print("\n=== 生成的图片文件 ===")
png_files = list(run_dir.glob("*catboost*.png")) + list(run_dir.glob("*lambdarank*.png"))
for f in sorted(png_files):
    print(f"  - {f.name}")
