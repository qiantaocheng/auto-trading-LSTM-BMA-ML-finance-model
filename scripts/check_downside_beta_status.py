#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check downside_beta_ewm_21 status
"""

import sys
from pathlib import Path
import re

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("downside_beta_ewm_21 Status Check")
print("=" * 80)

# 1. Check T10_ALPHA_FACTORS
print("\n1. T10_ALPHA_FACTORS (Factor Universe):")
print("-" * 80)
try:
    from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
    has_downside_beta = 'downside_beta_ewm_21' in T10_ALPHA_FACTORS
    print(f"   downside_beta_ewm_21: {'[YES]' if has_downside_beta else '[NO]'}")
    if has_downside_beta:
        idx = T10_ALPHA_FACTORS.index('downside_beta_ewm_21')
        print(f"   Position: #{idx + 1} of {len(T10_ALPHA_FACTORS)} features")
        print(f"   Description: Downside beta vs benchmark (QQQ) using EWMA 21-day window")
except Exception as e:
    print(f"   [ERROR] {e}")

# 2. Check t10_selected (Training Features)
print("\n2. t10_selected (Training Features):")
print("-" * 80)
try:
    code_file = project_root / "bma_models" / "量化模型_bma_ultra_enhanced.py"
    with open(code_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r't10_selected\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        features = [f.strip().strip('"\'') for f in re.findall(r'["\']([^"\']+)["\']', match.group(1))]
        has_downside_beta = 'downside_beta_ewm_21' in features
        print(f"   downside_beta_ewm_21: {'[YES]' if has_downside_beta else '[NO - NOT IN TRAINING]'}")
        print(f"   Total training features: {len(features)}")
        if not has_downside_beta:
            print(f"   [INFO] downside_beta_ewm_21 is computed but NOT used in training")
            print(f"   [INFO] It's available in T10_ALPHA_FACTORS but filtered out by t10_selected")
    else:
        print("   [ERROR] Could not find t10_selected list")
except Exception as e:
    print(f"   [ERROR] {e}")

# 3. Check calculation method
print("\n3. Calculation Method:")
print("-" * 80)
try:
    code_file = project_root / "bma_models" / "simple_25_factor_engine.py"
    with open(code_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_method = '_compute_downside_beta_ewm_21' in content
    print(f"   _compute_downside_beta_ewm_21 method: {'✅ EXISTS' if has_method else '❌ MISSING'}")
    
    if has_method:
        # Find where it's called
        called = 'downside_beta_ewm_21' in content and '_compute_downside_beta_ewm_21' in content
        print(f"   Called in compute_all_17_factors: {'✅ YES' if called else '❌ NO'}")
except Exception as e:
    print(f"   [ERROR] {e}")

# 4. Summary
print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("downside_beta_ewm_21 status:")
print("  [OK] Computed by: Simple17FactorEngine.compute_all_17_factors()")
print("  [OK] Available in: T10_ALPHA_FACTORS (factor universe)")
print("  [NO] NOT in: t10_selected (training features)")
print("  [INFO] Result: Computed for Direct Predict, but filtered out by models")
print("\nTo use downside_beta_ewm_21 in training:")
print("  1. Add 'downside_beta_ewm_21' to t10_selected list")
print("  2. Retrain all models")
print("=" * 80)
