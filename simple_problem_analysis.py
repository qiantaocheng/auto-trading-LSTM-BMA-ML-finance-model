#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Problem Analysis for Original BMA Ultra Enhanced Model
"""

import os
import sys

def analyze_bma_problems():
    """Analyze problems in the original BMA model"""
    print("=" * 80)
    print("BMA ULTRA ENHANCED - PROBLEM ANALYSIS")
    print("=" * 80)
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Cannot read file: {e}")
        return
    
    critical_issues = []
    medium_issues = []
    minor_issues = []
    
    print("\n1. CRITICAL ISSUES (MUST FIX):")
    print("-" * 50)
    
    # Data type issues - main cause of training failures
    if "Symbol" in content and "XGBRegressor" in content:
        issue = "String columns (Symbol, COUNTRY, SECTOR) passed to XGBoost/LightGBM causing training failures"
        critical_issues.append(issue)
        print(f"   CRITICAL: {issue}")
    
    # Missing module
    if "cross_sectional_standardization" in content:
        issue = "Missing 'cross_sectional_standardization' module causes alpha computation failures"
        critical_issues.append(issue)
        print(f"   CRITICAL: {issue}")
    
    # Regime detection missing method
    if "detect_regimes" in content:
        issue = "Regime detection method 'detect_regimes' not implemented in LeakFreeRegimeDetector"
        critical_issues.append(issue)
        print(f"   CRITICAL: {issue}")
    
    # Sample weight errors
    if "sample_weight_half_life" in content and "NoneType" in content:
        issue = "Sample weight unification fails with 'NoneType' errors"
        critical_issues.append(issue)
        print(f"   CRITICAL: {issue}")
    
    print("\n2. MEDIUM ISSUES (SHOULD FIX):")
    print("-" * 50)
    
    # Data type handling
    if "select_dtypes" not in content:
        issue = "No explicit data type filtering before model training"
        medium_issues.append(issue)
        print(f"   MEDIUM: {issue}")
    
    # Error handling
    broad_except_count = content.count("except Exception as e:")
    if broad_except_count > 20:
        issue = f"Excessive broad exception handling ({broad_except_count} occurrences) masks specific errors"
        medium_issues.append(issue)
        print(f"   MEDIUM: {issue}")
    
    # Memory management
    if "pd.concat" in content and "memory_usage" not in content:
        issue = "Large DataFrame operations without memory monitoring"
        medium_issues.append(issue)
        print(f"   MEDIUM: {issue}")
    
    # Configuration validation
    if "config" in content and "validate_config" not in content:
        issue = "No configuration validation"
        medium_issues.append(issue)
        print(f"   MEDIUM: {issue}")
    
    # API error handling
    if "polygon_client" in content and "requests.exceptions" not in content:
        issue = "Missing HTTP exception handling for API calls"
        medium_issues.append(issue)
        print(f"   MEDIUM: {issue}")
    
    print("\n3. MINOR ISSUES (NICE TO FIX):")
    print("-" * 50)
    
    # Hardcoded parameters
    hardcoded_params = ["alpha=0.01", "n_estimators=100", "max_depth=6"]
    for param in hardcoded_params:
        if param in content:
            issue = f"Hardcoded hyperparameter: {param}"
            minor_issues.append(issue)
            print(f"   MINOR: {issue}")
            break  # Just show one example
    
    # File size
    lines = len(content.split('\n'))
    if lines > 2000:
        issue = f"Very large file ({lines} lines) - consider splitting into modules"
        minor_issues.append(issue)
        print(f"   MINOR: {issue}")
    
    # Documentation
    method_count = content.count("def ")
    docstring_count = content.count('"""')
    if docstring_count < method_count // 2:
        issue = "Insufficient documentation - missing docstrings"
        minor_issues.append(issue)
        print(f"   MINOR: {issue}")
    
    # Performance
    if "iterrows()" in content:
        issue = "Using inefficient iterrows() - consider vectorized operations"
        minor_issues.append(issue)
        print(f"   MINOR: {issue}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("PROBLEM SUMMARY")
    print("=" * 80)
    
    total_critical = len(critical_issues)
    total_medium = len(medium_issues)
    total_minor = len(minor_issues)
    total_issues = total_critical + total_medium + total_minor
    
    print(f"\nTotal Issues Found: {total_issues}")
    print(f"  Critical Issues: {total_critical} (MUST FIX)")
    print(f"  Medium Issues:   {total_medium} (SHOULD FIX)")
    print(f"  Minor Issues:    {total_minor} (NICE TO FIX)")
    
    print(f"\nPRIORITY FIXES:")
    print("1. Fix data type handling for XGBoost/LightGBM (Filter out string columns)")
    print("2. Create cross_sectional_standardization module")
    print("3. Implement detect_regimes method in LeakFreeRegimeDetector")
    print("4. Fix sample weight unification NoneType errors")
    print("5. Add proper categorical variable encoding")
    
    # Detailed explanations
    print(f"\nDETAILED ISSUE EXPLANATIONS:")
    print("-" * 50)
    
    print("\nISSUE 1 - Data Type Problems:")
    print("  Problem: String columns like 'Symbol', 'COUNTRY', 'SECTOR' are passed to")
    print("           XGBoost and LightGBM models which only accept numeric data")
    print("  Error:   'could not convert string to float' and 'pandas dtypes must be int, float or bool'")
    print("  Fix:     Add data type filtering: X_numeric = X.select_dtypes(include=[np.number])")
    
    print("\nISSUE 2 - Missing Module:")
    print("  Problem: Alpha strategies try to import 'cross_sectional_standardization' but it doesn't exist")
    print("  Error:   ImportError causes fallback to basic zscore")
    print("  Fix:     Create the missing module or remove the import")
    
    print("\nISSUE 3 - Missing Method:")
    print("  Problem: Code calls regime_detector.detect_regimes() but method doesn't exist")
    print("  Error:   AttributeError: 'LeakFreeRegimeDetector' object has no attribute 'detect_regimes'")
    print("  Fix:     Implement the detect_regimes method or use alternative approach")
    
    print("\nISSUE 4 - Sample Weight Errors:")
    print("  Problem: Sample weight unification tries to access attributes on None objects")
    print("  Error:   'NoneType' object has no attribute 'sample_weight_half_life'")
    print("  Fix:     Add null checks before accessing sample weight attributes")
    
    # Save to file
    with open('bma_problems_summary.txt', 'w', encoding='utf-8') as f:
        f.write("BMA ULTRA ENHANCED - PROBLEM SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Issues: {total_issues}\n")
        f.write(f"Critical: {total_critical}, Medium: {total_medium}, Minor: {total_minor}\n\n")
        
        f.write("CRITICAL ISSUES:\n")
        for i, issue in enumerate(critical_issues, 1):
            f.write(f"{i}. {issue}\n")
        
        f.write("\nMEDIUM ISSUES:\n")
        for i, issue in enumerate(medium_issues, 1):
            f.write(f"{i}. {issue}\n")
        
        f.write("\nMINOR ISSUES:\n")
        for i, issue in enumerate(minor_issues, 1):
            f.write(f"{i}. {issue}\n")
    
    print(f"\nReport saved to 'bma_problems_summary.txt'")

if __name__ == "__main__":
    analyze_bma_problems()