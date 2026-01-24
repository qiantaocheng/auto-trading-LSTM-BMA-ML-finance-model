#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 LambdaRank 各分桶的表现"""

import pandas as pd
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_032758")

# Read bucket summary
summary = pd.read_csv(run_dir / "lambdarank_bucket_summary.csv")
returns = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

print("="*80)
print("LambdaRank 各分桶收益表现")
print("="*80)

print("\n=== 分桶汇总统计 ===")
print(summary.to_string(index=False))

print("\n=== 详细时间序列（前10行）===")
print(returns.head(10).to_string(index=False))

# Calculate statistics for each bucket
print("\n=== 各分桶统计摘要 ===")
bucket_cols = [col for col in returns.columns if 'return' in col.lower() and 'bucket' not in col.lower()]
for col in bucket_cols:
    if col in returns.columns:
        data = returns[col].dropna()
        if len(data) > 0:
            print(f"\n{col}:")
            print(f"  平均收益: {data.mean()*100:.4f}%")
            print(f"  中位数收益: {data.median()*100:.4f}%")
            print(f"  标准差: {data.std()*100:.4f}%")
            print(f"  胜率: {(data > 0).mean()*100:.2f}%")
            print(f"  最大收益: {data.max()*100:.4f}%")
            print(f"  最小收益: {data.min()*100:.4f}%")
