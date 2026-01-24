#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List all current active factors"""

import sys
sys.path.insert(0, 'D:/trade')

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS, T5_ALPHA_FACTORS

print("=" * 80)
print("当前因子配置")
print("=" * 80)

print(f"\n【T+10 因子配置】({len(T10_ALPHA_FACTORS)} 个因子)")
print("-" * 80)
for i, factor in enumerate(T10_ALPHA_FACTORS, 1):
    print(f"{i:2d}. {factor}")

print(f"\n【T+5 因子配置】({len(T5_ALPHA_FACTORS)} 个因子)")
print("-" * 80)
for i, factor in enumerate(T5_ALPHA_FACTORS, 1):
    print(f"{i:2d}. {factor}")

print("\n" + "=" * 80)
print("已移除的因子")
print("=" * 80)
removed_factors = [
    'making_new_low_5d',
    'bollinger_squeeze',
    'blowoff_ratio',
    'downside_beta_252',
    'downside_beta_ewm_21',
    'roa',
    'ebit',
    'ret_skew_20d',
]
for i, factor in enumerate(removed_factors, 1):
    print(f"{i}. {factor}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"当前活跃因子数量: {len(T10_ALPHA_FACTORS)} 个 (T+10配置)")
print(f"已移除因子数量: {len(removed_factors)} 个")
print("=" * 80)
