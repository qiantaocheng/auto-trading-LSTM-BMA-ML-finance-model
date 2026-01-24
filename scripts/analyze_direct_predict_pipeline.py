#!/usr/bin/env python3
"""Analyze direct prediction pipeline for potential issues"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("DIRECT PREDICTION PIPELINE ANALYSIS")
print("=" * 80)

issues = []
warnings = []

# 1. Check factor consistency
print("\n[1] FACTOR CONSISTENCY CHECK")
print("-" * 80)
from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
from scripts.time_split_80_20_oos_eval import _parse_args

# Read training script
training_script = Path("scripts/time_split_80_20_oos_eval.py").read_text(encoding='utf-8')
import re
pattern = r"allowed_feature_cols\s*=\s*\[(.*?)\]"
match = re.search(pattern, training_script, re.DOTALL)
if match:
    factors_str = match.group(1)
    training_factors = []
    for line in factors_str.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            factor = line.split(',')[0].strip().strip("'\"")
            if factor:
                training_factors.append(factor)
    
    training_set = set(training_factors)
    t10_set = set(T10_ALPHA_FACTORS)
    
    if training_set == t10_set:
        print("[PASS] Factors match between training and direct prediction")
    else:
        issues.append("Factor mismatch between training and direct prediction")
        print("[FAIL] Factor mismatch!")
        print(f"  Training: {sorted(training_set)}")
        print(f"  Direct Predict: {sorted(t10_set)}")

# 2. Check Meta Stacker base_cols
print("\n[2] META STACKER BASE_COLS CHECK")
print("-" * 80)
import yaml
config_path = Path("bma_models/unified_config.yaml")
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    meta_ranker_base_cols = config.get('training', {}).get('meta_ranker', {}).get('base_cols', [])
    print(f"Meta Stacker base_cols from config: {meta_ranker_base_cols}")
    
    required_cols = ['pred_catboost', 'pred_elastic', 'pred_xgb', 'pred_lambdarank']
    missing_cols = [c for c in required_cols if c not in meta_ranker_base_cols]
    
    if missing_cols:
        issues.append(f"Meta Stacker base_cols missing: {missing_cols}")
        print(f"[FAIL] Missing columns in base_cols: {missing_cols}")
    else:
        print("[PASS] All required columns present in base_cols")
    
    if 'pred_catboost' not in meta_ranker_base_cols:
        issues.append("CRITICAL: pred_catboost not in Meta Stacker base_cols!")
        print("[FAIL] CRITICAL: pred_catboost missing from base_cols!")
    else:
        print("[PASS] pred_catboost is in base_cols")

# 3. Check predict_with_snapshot LambdaRank handling
print("\n[3] PREDICT_WITH_SNAPSHOT LAMBDARANK HANDLING")
print("-" * 80)
predict_code = Path("bma_models/量化模型_bma_ultra_enhanced.py").read_text(encoding='utf-8')

# Check if pred_lambdarank is added to ridge_input
# OPTIMIZED: pred_lambdarank is added to first_layer_preds, then copied to ridge_input
if "pred_lambdarank" in predict_code:
    if "first_layer_preds['pred_lambdarank']" in predict_code and "ridge_input = first_layer_preds.copy()" in predict_code:
        # Check order: pred_lambdarank should be added to first_layer_preds BEFORE ridge_input copy
        first_layer_add_pos = predict_code.find("first_layer_preds['pred_lambdarank']")
        ridge_input_copy_pos = predict_code.find("ridge_input = first_layer_preds.copy()")
        if first_layer_add_pos > 0 and ridge_input_copy_pos > 0:
            if first_layer_add_pos < ridge_input_copy_pos:
                print("[PASS] pred_lambdarank is added to first_layer_preds BEFORE ridge_input copy (OPTIMIZED)")
            else:
                print("[WARN] pred_lambdarank added to first_layer_preds AFTER ridge_input copy - may be missing")
                issues.append("pred_lambdarank may not be in ridge_input")
        else:
            print("[PASS] pred_lambdarank is added to first_layer_preds and copied to ridge_input")
    elif "ridge_input['pred_lambdarank']" in predict_code:
        print("[PASS] pred_lambdarank is directly added to ridge_input")
    else:
        issues.append("pred_lambdarank may not be added to ridge_input")
        print("[WARN] pred_lambdarank addition to ridge_input not clearly found")
else:
    issues.append("pred_lambdarank not found in prediction code")
    print("[FAIL] pred_lambdarank not found")

# Check order: when is pred_lambdarank added relative to ridge_input column ordering
ridge_input_order_pattern = r"ridge_input\s*=\s*ridge_input\[list\(ridge_base_cols\)\]"
if re.search(ridge_input_order_pattern, predict_code):
    # Check if pred_lambdarank is added after this ordering
    add_lambda_pattern = r"ridge_input\['pred_lambdarank'\]\s*="
    reorder_pattern = r"Re-order columns to match base_cols|Re-ordered ridge_input columns"
    if re.search(add_lambda_pattern, predict_code):
        # Check if it's after the ordering
        order_match = list(re.finditer(ridge_input_order_pattern, predict_code))[0]
        add_match = list(re.finditer(add_lambda_pattern, predict_code))[0]
        if add_match.start() > order_match.start():
            # Check if there's a re-order after adding pred_lambdarank
            if re.search(reorder_pattern, predict_code):
                # Check if reorder is after adding pred_lambdarank
                reorder_match = list(re.finditer(reorder_pattern, predict_code))[0]
                if reorder_match.start() > add_match.start():
                    print("[PASS] pred_lambdarank added after ordering, but re-ordered afterwards (FIXED)")
                else:
                    warnings.append("pred_lambdarank is added AFTER ridge_input column ordering - may cause column order mismatch")
                    print("[WARN] pred_lambdarank added after column ordering - potential order mismatch")
            else:
                warnings.append("pred_lambdarank is added AFTER ridge_input column ordering - may cause column order mismatch")
                print("[WARN] pred_lambdarank added after column ordering - potential order mismatch")

# 4. Check CatBoost model loading
print("\n[4] CATBOOST MODEL LOADING CHECK")
print("-" * 80)
catboost_load_pattern = r"CatBoostRegressor.*load_model|catboost_cbm"
if re.search(catboost_load_pattern, predict_code):
    print("[PASS] CatBoost model loading code found")
    
    # Check if it handles missing CatBoost gracefully
    if "CatBoostRegressor is not None" in predict_code:
        print("[PASS] CatBoost loading checks for None")
    else:
        warnings.append("CatBoost loading may not check for None properly")
    
    # Check if pred_catboost is added to first_layer_preds
    if "first_layer_preds['pred_catboost']" in predict_code:
        print("[PASS] pred_catboost is added to first_layer_preds")
    else:
        issues.append("pred_catboost may not be added to first_layer_preds")
        print("[FAIL] pred_catboost addition to first_layer_preds not found")
else:
    issues.append("CatBoost model loading code not found")
    print("[FAIL] CatBoost loading code not found")

# 5. Check feature alignment
print("\n[5] FEATURE ALIGNMENT CHECK")
print("-" * 80)
if "fill_missing_features_with_median" in predict_code:
    print("[PASS] Missing feature filling function exists")
else:
    warnings.append("Missing feature filling may use 0.0 instead of median")
    print("[WARN] Missing feature filling function not found")

# Check if feature_names_by_model is used
if "feature_names_by_model" in predict_code:
    print("[PASS] feature_names_by_model is used for per-model feature selection")
else:
    warnings.append("feature_names_by_model may not be used")
    print("[WARN] feature_names_by_model usage not clearly found")

# 6. Check date filtering logic
print("\n[6] DATE FILTERING LOGIC CHECK")
print("-" * 80)
direct_predict_code = Path("scripts/direct_predict_ewma_excel.py").read_text(encoding='utf-8')

# Check if prediction_days filtering happens before or after factor calculation
if "prediction_days > 0" in direct_predict_code and "Filtered to last" in direct_predict_code:
    # Check if filtering happens after compute_all_17_factors
    if "compute_all_17_factors" in direct_predict_code and "Filtered to last" in direct_predict_code:
        compute_pos = direct_predict_code.find("compute_all_17_factors")
        filter_pos = direct_predict_code.find("Filtered to last")
        if filter_pos > compute_pos:
            print("[PASS] Date filtering happens AFTER factor calculation (correct)")
        else:
            warnings.append("Date filtering may happen before factor calculation")
            print("[WARN] Date filtering may happen before factor calculation")
    else:
        print("[INFO] Cannot determine filtering order")

# 7. Check Meta Stacker input validation
print("\n[7] META STACKER INPUT VALIDATION")
print("-" * 80)
if "ridge_input shape" in predict_code and "ridge_input columns" in predict_code:
    print("[PASS] ridge_input shape and columns are logged")
else:
    warnings.append("ridge_input validation logging may be missing")
    print("[WARN] ridge_input validation logging not found")

# Check if all base_cols are verified
if "ridge_base_cols" in predict_code and "for col in ridge_base_cols" in predict_code:
    print("[PASS] ridge_base_cols are checked/validated")
else:
    warnings.append("ridge_base_cols validation may be incomplete")
    print("[WARN] ridge_base_cols validation not clearly found")

# 8. Check prediction result handling
print("\n[8] PREDICTION RESULT HANDLING")
print("-" * 80)
if "score_raw" in direct_predict_code and "score_smoothed" in direct_predict_code:
    print("[PASS] Raw and smoothed scores are handled")
else:
    warnings.append("Score handling may be incomplete")
    print("[WARN] Score handling not clearly found")

# Check if base model scores are extracted
base_model_scores = ['score_lambdarank', 'score_catboost', 'score_elastic', 'score_xgb']
missing_scores = [s for s in base_model_scores if s not in direct_predict_code]
if missing_scores:
    warnings.append(f"Base model scores may not be extracted: {missing_scores}")
    print(f"[WARN] Base model scores not found: {missing_scores}")
else:
    print("[PASS] All base model scores are handled")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if issues:
    print(f"\n[CRITICAL ISSUES] ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n[PASS] No critical issues found")

if warnings:
    print(f"\n[WARNINGS] ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
else:
    print("\n[PASS] No warnings")

print("\n" + "=" * 80)
if not issues and not warnings:
    print("[OVERALL] Pipeline looks good!")
else:
    print("[OVERALL] Issues/warnings found - review recommended")
