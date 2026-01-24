#!/usr/bin/env python3
"""Verify that direct prediction uses the same factors as training"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 1. Get factors from training script (80/20 eval)
print("=" * 80)
print("1. FACTORS USED IN 80/20 TRAINING SCRIPT")
print("=" * 80)
from scripts.time_split_80_20_oos_eval import _parse_args
import inspect

# Read the training script to get allowed_feature_cols
training_script_path = Path("scripts/time_split_80_20_oos_eval.py")
with open(training_script_path, 'r', encoding='utf-8') as f:
    training_code = f.read()

# Extract allowed_feature_cols from the script
import re
pattern = r"allowed_feature_cols\s*=\s*\[(.*?)\]"
match = re.search(pattern, training_code, re.DOTALL)
if match:
    factors_str = match.group(1)
    # Parse the factors
    training_factors = []
    for line in factors_str.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract factor name (remove quotes and comments)
            factor = line.split(',')[0].strip().strip("'\"")
            if factor:
                training_factors.append(factor)
    
    print(f"Training script factors ({len(training_factors)}):")
    for i, f in enumerate(training_factors, 1):
        print(f"  {i:2d}. {f}")
else:
    print("ERROR: Could not find allowed_feature_cols in training script")
    training_factors = []

# 2. Get factors from Simple17FactorEngine (used by direct prediction)
print("\n" + "=" * 80)
print("2. FACTORS USED IN DIRECT PREDICTION (Simple17FactorEngine)")
print("=" * 80)
from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS, T5_ALPHA_FACTORS

print(f"\nT10_ALPHA_FACTORS ({len(T10_ALPHA_FACTORS)} factors) - Used for horizon=10:")
for i, f in enumerate(T10_ALPHA_FACTORS, 1):
    print(f"  {i:2d}. {f}")

print(f"\nT5_ALPHA_FACTORS ({len(T5_ALPHA_FACTORS)} factors) - Used for horizon=5:")
for i, f in enumerate(T5_ALPHA_FACTORS, 1):
    print(f"  {i:2d}. {f}")

# 3. Compare
print("\n" + "=" * 80)
print("3. COMPARISON")
print("=" * 80)

training_set = set(training_factors)
t10_set = set(T10_ALPHA_FACTORS)

print(f"\nTraining script factors: {len(training_set)}")
print(f"T10_ALPHA_FACTORS: {len(t10_set)}")

missing_in_t10 = training_set - t10_set
extra_in_t10 = t10_set - training_set

if not missing_in_t10 and not extra_in_t10:
    print("\n[PASS] PERFECT MATCH: Training and Direct Prediction use identical factors!")
else:
    if missing_in_t10:
        print(f"\n[FAIL] Factors in training but NOT in T10_ALPHA_FACTORS:")
        for f in sorted(missing_in_t10):
            print(f"  - {f}")
    
    if extra_in_t10:
        print(f"\n[WARN] Factors in T10_ALPHA_FACTORS but NOT in training:")
        for f in sorted(extra_in_t10):
            print(f"  - {f}")

# 4. Check if ret_skew_20d is present (should NOT be)
print("\n" + "=" * 80)
print("4. CHECK FOR REMOVED FACTORS")
print("=" * 80)

removed_factors = ['ret_skew_20d', 'making_new_low_5d', 'bollinger_squeeze', 'blowoff_ratio', 
                   'downside_beta_252', 'downside_beta_ewm_21', 'roa', 'ebit']

issues = []
for factor in removed_factors:
    in_training = factor in training_set
    in_t10 = factor in t10_set
    in_t5 = factor in set(T5_ALPHA_FACTORS)
    
    if in_training or in_t10 or in_t5:
        issues.append(factor)
        print(f"[FAIL] {factor}:")
        if in_training:
            print(f"   - Still in training script!")
        if in_t10:
            print(f"   - Still in T10_ALPHA_FACTORS!")
        if in_t5:
            print(f"   - Still in T5_ALPHA_FACTORS!")

if not issues:
    print("[PASS] All removed factors are correctly absent from both training and direct prediction")

# 5. Summary
print("\n" + "=" * 80)
print("5. SUMMARY")
print("=" * 80)

if not missing_in_t10 and not extra_in_t10 and not issues:
    print("[PASS] VERIFICATION PASSED: Direct prediction and training use identical factors!")
    print(f"   Both use {len(training_set)} factors (14 factors after removing ret_skew_20d)")
else:
    print("[FAIL] VERIFICATION FAILED: Differences found!")
    print("   Action required: Fix inconsistencies")
