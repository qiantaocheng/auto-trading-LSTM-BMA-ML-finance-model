#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析dropna对样本数量的影响"""

import pandas as pd
import numpy as np

# 模拟场景
years = 3
stocks = 30
trading_days_per_year = 250
total_days = years * trading_days_per_year  # 750天

theoretical_samples = total_days * stocks
print(f"=== 样本数量分析 ===")
print(f"数据时间跨度: {years}年")
print(f"股票数量: {stocks}只")
print(f"交易日数: {total_days}天")
print(f"理论样本数: {theoretical_samples:,}个 ({total_days} × {stocks})")
print()

# 因子所需lookback期
factors_lookback = {
    'momentum_10d_ex1': 10,
    'rsi_7': 7,
    'bollinger_squeeze': 20,
    'obv_momentum': 20,
    'atr_ratio': 20,
    'ivol_60d': 60,
    'liquidity_factor': 20,
    'near_52w_high': 252,  # 52周
    'reversal_1d': 1,
    'rel_volume_spike': 20,
    'mom_accel_5_2': 5,
    'overnight_intraday_gap': 180,  # 需要180天
    'max_lottery_factor': 365,  # 需要365天！
    'streak_reversal': 30,
    'price_efficiency_5d': 5,
    'nr7_breakout_bias': 7
}

print("=== 因子Lookback要求 ===")
for factor, days in sorted(factors_lookback.items(), key=lambda x: x[1], reverse=True):
    print(f"{factor:30s}: {days:3d}天")

max_lookback = max(factors_lookback.values())
print(f"\n最大Lookback: {max_lookback}天")
print()

# 计算有效样本数（假设dropna()）
# dropna会删除前max_lookback天的所有数据
valid_days = total_days - max_lookback
if valid_days < 0:
    valid_days = 0

valid_samples = valid_days * stocks

print(f"=== Dropna影响分析 ===")
print(f"前{max_lookback}天数据会因为长期因子产生NaN")
print(f"这相当于: {max_lookback/trading_days_per_year:.2f}年的数据")
print(f"dropna后有效交易日: {valid_days}天")
print(f"dropna后有效样本数: {valid_samples:,}个")
print(f"样本损失率: {(1 - valid_samples/theoretical_samples)*100:.1f}%")
print()

# T+5 horizon shift的影响
horizon = 5
final_days = valid_days - horizon
final_samples = final_days * stocks

print(f"=== T+5 Horizon Shift影响 ===")
print(f"Horizon shift会再移除最后{horizon}天数据")
print(f"最终有效交易日: {final_days}天")
print(f"最终可用样本数: {final_samples:,}个")
print(f"相比理论值损失: {(1 - final_samples/theoretical_samples)*100:.1f}%")
print()

print(f"=== 结论 ===")
print(f"理论样本: {theoretical_samples:,}个")
print(f"实际样本: {final_samples:,}个")
print(f"损失原因:")
print(f"  1. max_lottery_factor需要365天数据 → 损失{max_lookback}天 × {stocks}股 = {max_lookback * stocks:,}样本")
print(f"  2. T+5预测horizon → 损失{horizon}天 × {stocks}股 = {horizon * stocks:,}样本")
print(f"  3. 总损失: {theoretical_samples - final_samples:,}样本 ({(1 - final_samples/theoretical_samples)*100:.1f}%)")
