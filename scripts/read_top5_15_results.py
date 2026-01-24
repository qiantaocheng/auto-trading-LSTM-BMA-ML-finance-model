#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 Top 5-15 累计收益结果"""

import pandas as pd
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_033750")

print("="*80)
print("Top 5-15 累计收益结果")
print("="*80)

for model_name in ['catboost', 'lambdarank']:
    csv_file = run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv"
    png_file = run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.png"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"\n=== {model_name.upper()} Top 5-15 ===")
        print(f"最终累计收益: {df['acc_return'].iloc[-1]*100:.2f}%")
        print(f"平均每期收益: {df['top_gross_return'].mean()*100:.4f}%")
        print(f"期数: {len(df)}")
        print(f"CSV文件: {csv_file.name}")
        if png_file.exists():
            print(f"PNG图片: {png_file.name}")
    else:
        print(f"\n{model_name}: 文件不存在")

print("\n" + "="*80)
