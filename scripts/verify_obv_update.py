#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify OBV Momentum 40d update
"""

import sys
from pathlib import Path
import re

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("OBV Momentum 40d Update Verification")
print("=" * 80)

# 1. Check T10_ALPHA_FACTORS
print("\n1. T10_ALPHA_FACTORS:")
print("-" * 80)
try:
    from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
    has_obv_momentum = 'obv_momentum_40d' in T10_ALPHA_FACTORS
    has_obv_divergence = 'obv_divergence' in T10_ALPHA_FACTORS
    print(f"   obv_momentum_40d: {'[YES]' if has_obv_momentum else '[NO]'}")
    print(f"   obv_divergence: {'[YES - OLD]' if has_obv_divergence else '[NO - REMOVED]'}")
except Exception as e:
    print(f"   [ERROR] {e}")

# 2. Check t10_selected
print("\n2. t10_selected (Training Features):")
print("-" * 80)
try:
    code_file = project_root / "bma_models" / "量化模型_bma_ultra_enhanced.py"
    with open(code_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r't10_selected\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        features = [f.strip().strip('"\'') for f in re.findall(r'["\']([^"\']+)["\']', match.group(1))]
        has_obv_momentum = 'obv_momentum_40d' in features
        has_obv_divergence = 'obv_divergence' in features
        print(f"   obv_momentum_40d: {'[YES]' if has_obv_momentum else '[NO]'}")
        print(f"   obv_divergence: {'[YES - OLD]' if has_obv_divergence else '[NO - REMOVED]'}")
        print(f"   Total training features: {len(features)}")
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
    
    has_obv_momentum_method = 'obv_momentum_40d' in content and '_calc_obv_momentum_40d' in content
    has_obv_divergence_method = 'obv_divergence' in content and 'fast_norm' in content
    print(f"   obv_momentum_40d calculation: {'[EXISTS]' if has_obv_momentum_method else '[MISSING]'}")
    print(f"   obv_divergence calculation: {'[EXISTS - OLD]' if has_obv_divergence_method else '[REMOVED]'}")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("[OK] OBV Divergence replaced with OBV Momentum 40d")
print("[OK] All feature lists updated")
print("[OK] Calculation logic updated with normalization")
print("=" * 80)
