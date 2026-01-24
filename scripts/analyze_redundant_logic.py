#!/usr/bin/env python3
"""Analyze redundant logic in direct prediction pipeline"""
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("REDUNDANT LOGIC ANALYSIS - Direct Prediction Pipeline")
print("=" * 80)

predict_code = Path("bma_models/量化模型_bma_ultra_enhanced.py").read_text(encoding='utf-8')

issues = []
warnings = []
optimizations = []

# 1. Multiple ridge_input copies and reordering
print("\n[1] RIDGE_INPUT COPY AND REORDERING REDUNDANCY")
print("-" * 80)

# Find all ridge_input assignments
ridge_input_patterns = [
    (r"ridge_input\s*=\s*first_layer_preds\.copy\(\)", "Initial copy from first_layer_preds"),
    (r"ridge_input\s*=\s*ridge_input\[list\(ridge_base_cols\)\]\.copy\(\)", "First reorder by base_cols"),
    (r"ridge_input\s*=\s*ridge_input\[available_base_cols\]\.copy\(\)", "Second reorder after adding pred_lambdarank"),
    (r"ridge_input\s*=\s*ridge_input\.drop\(columns=", "Drop columns"),
    (r"ridge_input\[col\]\s*=", "Assign column"),
]

copy_count = 0
reorder_count = 0
for pattern, desc in ridge_input_patterns:
    matches = list(re.finditer(pattern, predict_code))
    if matches:
        for match in matches:
            line_num = predict_code[:match.start()].count('\n') + 1
            if 'copy()' in match.group():
                copy_count += 1
            if '[' in match.group() and ']' in match.group() and 'base_cols' in match.group():
                reorder_count += 1
            print(f"  Line {line_num}: {desc}")

if copy_count > 2:
    issues.append(f"Too many ridge_input copies ({copy_count}): Performance impact")
    print(f"[ISSUE] {copy_count} copies detected - should optimize to 1-2 copies max")

if reorder_count > 1:
    issues.append(f"Multiple reorderings ({reorder_count}): pred_lambdarank should be added BEFORE first reorder")
    print(f"[ISSUE] {reorder_count} reorderings detected - should consolidate to single reorder")

# 2. LambdaRank prediction addition redundancy
print("\n[2] LAMBDARANK PREDICTION ADDITION REDUNDANCY")
print("-" * 80)

# Check if pred_lambdarank is added multiple times
lambda_add_patterns = [
    (r"first_layer_preds\['pred_lambdarank'\]\s*=", "Add to first_layer_preds"),
    (r"ridge_input\['pred_lambdarank'\]\s*=", "Add to ridge_input"),
]

lambda_additions = []
for pattern, desc in lambda_add_patterns:
    matches = list(re.finditer(pattern, predict_code))
    for match in matches:
        line_num = predict_code[:match.start()].count('\n') + 1
        lambda_additions.append((line_num, desc))
        print(f"  Line {line_num}: {desc}")

if len(lambda_additions) > 1:
    # Check if they're redundant
    first_layer_add = [x for x in lambda_additions if 'first_layer_preds' in x[1]]
    ridge_input_add = [x for x in lambda_additions if 'ridge_input' in x[1]]
    
    if first_layer_add and ridge_input_add:
        # Check order
        first_line = first_layer_add[0][0]
        ridge_line = ridge_input_add[0][0]
        
        if ridge_line > first_line:
            # Check if ridge_input already has pred_lambdarank from first_layer_preds.copy()
            # If ridge_input = first_layer_preds.copy() happens before adding to ridge_input,
            # then adding to first_layer_preds should be enough
            copy_line = predict_code.find("ridge_input = first_layer_preds.copy()")
            copy_line_num = predict_code[:copy_line].count('\n') + 1 if copy_line >= 0 else 0
            
            if copy_line_num < first_line < ridge_line:
                warnings.append("Redundant: pred_lambdarank added to first_layer_preds, then copied to ridge_input, then added again to ridge_input")
                print(f"[WARN] Redundant addition: pred_lambdarank added to first_layer_preds (line {first_line}), then to ridge_input (line {ridge_line})")
                print(f"       If ridge_input = first_layer_preds.copy() happens at line {copy_line_num}, the second addition may be redundant")

# 3. lambda_percentile handling redundancy
print("\n[3] LAMBDA_PERCENTILE HANDLING REDUNDANCY")
print("-" * 80)

lambda_percentile_patterns = [
    (r"ridge_input\['lambda_percentile'\]\s*=", "Add lambda_percentile to ridge_input"),
]

lambda_percentile_additions = []
for pattern, desc in lambda_percentile_patterns:
    matches = list(re.finditer(pattern, predict_code))
    for match in matches:
        line_num = predict_code[:match.start()].count('\n') + 1
        lambda_percentile_additions.append((line_num, desc))
        print(f"  Line {line_num}: {desc}")

if len(lambda_percentile_additions) > 1:
    issues.append(f"lambda_percentile added {len(lambda_percentile_additions)} times - should consolidate")
    print(f"[ISSUE] lambda_percentile added {len(lambda_percentile_additions)} times - redundant logic")

# 4. Column ordering logic redundancy
print("\n[4] COLUMN ORDERING LOGIC REDUNDANCY")
print("-" * 80)

# Check if pred_lambdarank is in ridge_base_cols by default
if "ridge_base_cols_raw = ridge_meta.get('base_cols') or ('pred_catboost', 'pred_elastic', 'pred_xgb', 'pred_lambdarank')" in predict_code:
    print("[INFO] pred_lambdarank is in default ridge_base_cols")
    
    # Check if pred_lambdarank is added after first reorder
    reorder1_pos = predict_code.find("ridge_input = ridge_input[list(ridge_base_cols)].copy()")
    add_lambda_pos = predict_code.find("ridge_input['pred_lambdarank'] =")
    
    if reorder1_pos >= 0 and add_lambda_pos >= 0 and add_lambda_pos > reorder1_pos:
        issues.append("pred_lambdarank should be added BEFORE first reorder, not after")
        print("[ISSUE] pred_lambdarank added AFTER first reorder - causes need for second reorder")
        optimizations.append("Move pred_lambdarank addition to first_layer_preds BEFORE creating ridge_input")

# 5. MultiIndex check redundancy
print("\n[5] MULTIINDEX CHECK REDUNDANCY")
print("-" * 80)

multiindex_check = r"if not isinstance\(ridge_input\.index, pd\.MultiIndex\)"
if re.search(multiindex_check, predict_code):
    # Check if first_layer_preds already has MultiIndex
    if "first_layer_preds = pd.DataFrame(index=X_df.index)" in predict_code:
        # X_df should already have MultiIndex from _prepare_standard_data_format
        print("[INFO] first_layer_preds created from X_df.index (should already be MultiIndex)")
        
        # Check if the MultiIndex check is necessary
        if "isinstance(ridge_input.index, pd.MultiIndex)" in predict_code:
            # If ridge_input = first_layer_preds.copy(), it should already have MultiIndex
            warnings.append("MultiIndex check may be redundant if first_layer_preds already has MultiIndex")
            print("[WARN] MultiIndex check may be redundant - first_layer_preds should already have MultiIndex")

# 6. Exception handling redundancy
print("\n[6] EXCEPTION HANDLING REDUNDANCY")
print("-" * 80)

# Check for multiple try-except blocks doing similar things
try_except_pattern = r"try:.*?except.*?:"
try_except_blocks = list(re.finditer(try_except_pattern, predict_code, re.DOTALL))
print(f"[INFO] Found {len(try_except_blocks)} try-except blocks")

# Check for lambda_percentile exception handling
lambda_percentile_except = [
    m for m in try_except_blocks 
    if 'lambda_percentile' in predict_code[m.start():m.end()]
]
if len(lambda_percentile_except) > 1:
    warnings.append(f"Multiple exception handlers for lambda_percentile ({len(lambda_percentile_except)})")
    print(f"[WARN] {len(lambda_percentile_except)} exception handlers for lambda_percentile - may be redundant")

# 7. Feature filling redundancy
print("\n[7] FEATURE FILLING LOGIC REDUNDANCY")
print("-" * 80)

# Check if fill_missing_features_with_median is called multiple times
fill_median_pattern = r"fill_missing_features_with_median"
fill_median_calls = list(re.finditer(fill_median_pattern, predict_code))
print(f"[INFO] fill_missing_features_with_median called {len(fill_median_calls)} times")

# Check if there's inline median filling logic that duplicates the function
inline_median_pattern = r"ref_median\s*=\s*.*\.median\(\)\.median\(\)"
inline_median_calls = list(re.finditer(inline_median_pattern, predict_code))
if len(inline_median_calls) > len(fill_median_calls):
    warnings.append(f"Inline median filling ({len(inline_median_calls)}) duplicates fill_missing_features_with_median function ({len(fill_median_calls)})")
    print(f"[WARN] Inline median filling logic duplicates fill_missing_features_with_median function")

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

if optimizations:
    print(f"\n[OPTIMIZATION OPPORTUNITIES] ({len(optimizations)}):")
    for i, opt in enumerate(optimizations, 1):
        print(f"  {i}. {opt}")

print("\n" + "=" * 80)
