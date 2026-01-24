#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final verification of 15 factors integration"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

print("=" * 80)
print("Final Verification: 15 Factors Integration")
print("=" * 80)

# Check T10_ALPHA_FACTORS
print(f"\nT10_ALPHA_FACTORS: {len(T10_ALPHA_FACTORS)} factors")
print(f"Factors: {T10_ALPHA_FACTORS}")

# Check t10_selected
model = UltraEnhancedQuantitativeModel(preserve_state=False)
model.horizon = 10
t10_selected = model._base_feature_overrides.get('elastic_net', [])

print(f"\nt10_selected: {len(t10_selected)} factors")
print(f"Factors: {t10_selected}")

# Verify match
match = set(T10_ALPHA_FACTORS) == set(t10_selected)
all_in = all(f in t10_selected for f in T10_ALPHA_FACTORS)

print(f"\nMatch: {match}")
print(f"All T10_ALPHA_FACTORS in t10_selected: {all_in}")

if match and all_in:
    print("\n[OK] All 15 factors correctly integrated!")
    sys.exit(0)
else:
    print("\n[ERROR] Mismatch detected!")
    missing = [f for f in T10_ALPHA_FACTORS if f not in t10_selected]
    extra = [f for f in t10_selected if f not in T10_ALPHA_FACTORS]
    if missing:
        print(f"Missing in t10_selected: {missing}")
    if extra:
        print(f"Extra in t10_selected: {extra}")
    sys.exit(1)
