#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show LambdaRank Top 5-15 results
"""

import pandas as pd
from pathlib import Path

result_dir = Path("results/t10_time_split_80_20_sato/run_20260121_113311")

print("=" * 80)
print("LambdaRank Top 5-15 Results")
print("=" * 80)

# Read report_df.csv
report_file = result_dir / "report_df.csv"
df = pd.read_csv(report_file)
lambdarank = df[df['Model'] == 'lambdarank'].iloc[0]

print("\n=== Daily Metrics (Overlap) ===")
print(f"Avg Top 5-15 Return: {lambdarank['avg_top_5_15_return']:.4f} ({lambdarank['avg_top_5_15_return']*100:.2f}%)")
print(f"Median Top 5-15 Return: {lambdarank['median_top_5_15_return']:.4f} ({lambdarank['median_top_5_15_return']*100:.2f}%)")
print(f"Avg Top 5-15 Return (from median): {lambdarank['avg_top_5_15_return_from_median']:.4f} ({lambdarank['avg_top_5_15_return_from_median']*100:.2f}%)")
print(f"Median Top 5-15 Return (from median): {lambdarank['median_top_5_15_return_from_median']:.4f} ({lambdarank['median_top_5_15_return_from_median']*100:.2f}%)")

# Read accumulated returns
accum_file = result_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv"
if accum_file.exists():
    accum_df = pd.read_csv(accum_file)
    print("\n=== Non-Overlap Accumulated Returns (25 periods, 10-day rebalance) ===")
    print(f"Total periods: {len(accum_df)}")
    print(f"Final accumulated return: {accum_df['acc_return'].iloc[-1]:.2%}")
    print(f"Final accumulated value: {accum_df['acc_value'].iloc[-1]:.4f}")
    print(f"Max drawdown: {accum_df['drawdown'].min():.2f}%")
    
    # Calculate statistics
    period_returns = accum_df['top_gross_return']
    print(f"\nPeriod Return Statistics:")
    print(f"  Average period return: {period_returns.mean():.4f} ({period_returns.mean()*100:.2f}%)")
    print(f"  Median period return: {period_returns.median():.4f} ({period_returns.median()*100:.2f}%)")
    print(f"  Std dev: {period_returns.std():.4f} ({period_returns.std()*100:.2f}%)")
    print(f"  Win rate: {(period_returns > 0).sum() / len(period_returns):.2%}")
    
    print("\n=== Last 10 Periods ===")
    print(accum_df[['date', 'top_gross_return', 'acc_return', 'drawdown']].tail(10).to_string(index=False))
else:
    print("\n[INFO] lambdarank_top5_15_rebalance10d_accumulated.csv not found")

# Read bucket summary
bucket_file = result_dir / "lambdarank_bucket_summary.csv"
if bucket_file.exists():
    bucket_df = pd.read_csv(bucket_file)
    print("\n=== Bucket Summary ===")
    print(bucket_df.to_string(index=False))

print("\n" + "=" * 80)
print("Summary from complete_metrics_report.txt:")
print("=" * 80)
print("Overlap Metrics (249 trading days):")
print("  Average return: 0.8134%")
print("  Median return: 1.4704%")
print("  Std dev: 11.5102%")
print("  Win rate: 53.41%")
print("  Sharpe Ratio (annualized): 1.1218")
print("\nNon-Overlap Metrics (25 periods, 10-day rebalance):")
print("  Average period return: -1.7071%")
print("  Median period return: -0.3160%")
print("  Std dev: 11.5007%")
print("  Win rate: 48.00%")
print("  Sharpe Ratio (period-based): -0.7422")
print("\nCumulative Metrics:")
print("  Cumulative return: -45.4625%")
print("  Max drawdown: -47.8256%")
print("  Annualized return: -45.7264%")
print("=" * 80)
