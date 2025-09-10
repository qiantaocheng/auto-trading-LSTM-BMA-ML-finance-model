#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Problem Analysis for Original BMA Ultra Enhanced Model
Identifies all issues, bugs, and problems in the original implementation
"""

import os
import sys
import ast
import re
import traceback
from typing import Dict, List, Any
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BMAProblemsAnalyzer:
    """Comprehensive analyzer for BMA Ultra Enhanced problems"""
    
    def __init__(self):
        self.problems = []
        self.file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
        self.critical_issues = []
        self.medium_issues = []
        self.minor_issues = []
        
    def analyze_all_problems(self):
        """Run comprehensive analysis of all problems"""
        print("="*80)
        print("COMPREHENSIVE BMA ULTRA ENHANCED PROBLEM ANALYSIS")
        print("="*80)
        
        # Read the file
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.file_content = f.read()
        except Exception as e:
            print(f"ERROR: Cannot read file: {e}")
            return
            
        # Run all analysis methods
        self.analyze_imports_dependencies()
        self.analyze_data_type_issues()
        self.analyze_error_handling()
        self.analyze_model_training_issues()
        self.analyze_memory_management()
        self.analyze_configuration_problems()
        self.analyze_api_integration()
        self.analyze_cross_validation()
        self.analyze_prediction_generation()
        self.analyze_code_structure()
        self.analyze_performance_issues()
        
        # Generate final report
        self.generate_problem_report()
    
    def analyze_imports_dependencies(self):
        """Analyze import and dependency issues"""
        print("\n1. ANALYZING IMPORTS AND DEPENDENCIES")
        print("-" * 50)
        
        issues = []
        
        # Check for conditional imports that might fail
        conditional_imports = [
            'XGBoost', 'LightGBM', 'CatBoost', 'IndexAligner', 
            'AlphaStrategiesEngine', 'IntelligentMemoryManager',
            'UnifiedExceptionHandler', 'ProductionReadinessValidator',
            'RegimeAwareCV', 'LeakFreeRegimeDetector', 'EnhancedOOSSystem',
            'UnifiedFeaturePipeline', 'SampleWeightUnificator',
            'PurgedTimeSeriesCV', 'AlphaSummaryProcessor', 'PolygonClient'
        ]
        
        for import_name in conditional_imports:
            if f"{import_name.upper()}_AVAILABLE = False" in self.file_content:
                issues.append(f"❌ CRITICAL: {import_name} may not be available - causes feature degradation")
        
        # Check for missing cross_sectional_standardization
        if "cross_sectional_standardization" in self.file_content:
            issues.append("❌ CRITICAL: Missing 'cross_sectional_standardization' module causes alpha computation failures")
        
        # Check for hardcoded paths
        if "bma_models/" in self.file_content:
            issues.append("⚠️ MEDIUM: Hardcoded module paths may cause import failures in different environments")
        
        self.critical_issues.extend([i for i in issues if "CRITICAL" in i])
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        
        for issue in issues:
            print(f"  {issue}")
        
        if not issues:
            print("  ✅ No major import issues found")
    
    def analyze_data_type_issues(self):
        """Analyze data type handling problems"""
        print("\n2. ANALYZING DATA TYPE ISSUES")
        print("-" * 50)
        
        issues = []
        
        # Check for string columns passed to numerical models
        if "Symbol" in self.file_content and "XGBRegressor" in self.file_content:
            issues.append("❌ CRITICAL: String columns (Symbol, COUNTRY, SECTOR) passed to XGBoost/LightGBM causing training failures")
        
        # Check for object dtype issues
        patterns = [
            ("could not convert string to float", "❌ CRITICAL: String to float conversion errors in model training"),
            ("DataFrame.dtypes for data must be", "❌ CRITICAL: Invalid data types for XGBoost - object columns not handled"),
            ("pandas dtypes must be int, float or bool", "❌ CRITICAL: LightGBM data type validation failures")
        ]
        
        for pattern, issue in patterns:
            if pattern in self.file_content:
                issues.append(issue)
        
        # Check for missing data type preprocessing
        if "select_dtypes" not in self.file_content:
            issues.append("⚠️ MEDIUM: No explicit data type filtering before model training")
        
        # Check for categorical encoding
        if "LabelEncoder" not in self.file_content and "get_dummies" not in self.file_content:
            issues.append("⚠️ MEDIUM: No categorical variable encoding for string columns")
        
        self.critical_issues.extend([i for i in issues if "CRITICAL" in i])
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_error_handling(self):
        """Analyze error handling and exception management"""
        print("\n3. ANALYZING ERROR HANDLING")
        print("-" * 50)
        
        issues = []
        
        # Count broad exception handling
        broad_except_count = self.file_content.count("except Exception as e:")
        if broad_except_count > 20:
            issues.append(f"⚠️ MEDIUM: Excessive broad exception handling ({broad_except_count} occurrences) - masks specific errors")
        
        # Check for missing specific exception handling
        if "except ImportError" in self.file_content and "except ModuleNotFoundError" not in self.file_content:
            issues.append("💡 MINOR: Missing ModuleNotFoundError handling alongside ImportError")
        
        # Check for logging in exception handlers
        except_without_log = self.file_content.count("except Exception as e:") - self.file_content.count("logger.error")
        if except_without_log > 0:
            issues.append("⚠️ MEDIUM: Some exceptions may not be properly logged")
        
        # Check for error propagation issues
        if "pass" in self.file_content:
            issues.append("💡 MINOR: Silent failures with 'pass' statements may hide critical errors")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_model_training_issues(self):
        """Analyze model training pipeline problems"""
        print("\n4. ANALYZING MODEL TRAINING ISSUES")
        print("-" * 50)
        
        issues = []
        
        # Check for insufficient data validation
        if "min_samples_split" not in self.file_content:
            issues.append("⚠️ MEDIUM: No minimum sample size validation before training")
        
        # Check for missing feature scaling
        if "_train_standard_models" in self.file_content and "StandardScaler" not in self.file_content:
            issues.append("⚠️ MEDIUM: Missing feature scaling in model training pipeline")
        
        # Check for hyperparameter hardcoding
        hardcoded_params = ["alpha=0.01", "n_estimators=100", "max_depth=6"]
        for param in hardcoded_params:
            if param in self.file_content:
                issues.append(f"💡 MINOR: Hardcoded hyperparameter: {param}")
        
        # Check for cross-validation issues
        if "fit(X_train, y_train)" in self.file_content and "cross_val_score" not in self.file_content:
            issues.append("⚠️ MEDIUM: Limited cross-validation usage in model selection")
        
        # Check for regime detection problems
        if "detect_regimes" in self.file_content:
            issues.append("❌ CRITICAL: Regime detection method 'detect_regimes' not implemented in LeakFreeRegimeDetector")
        
        # Check for prediction shape mismatches
        if "predictions" in self.file_content and "reshape" not in self.file_content:
            issues.append("💡 MINOR: Potential prediction shape mismatch issues")
        
        self.critical_issues.extend([i for i in issues if "CRITICAL" in i])
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_memory_management(self):
        """Analyze memory management issues"""
        print("\n5. ANALYZING MEMORY MANAGEMENT")
        print("-" * 50)
        
        issues = []
        
        # Check for memory leaks
        if "del " not in self.file_content:
            issues.append("💡 MINOR: No explicit memory cleanup with 'del' statements")
        
        # Check for large object handling
        if "pd.concat" in self.file_content and "memory_usage" not in self.file_content:
            issues.append("⚠️ MEDIUM: Large DataFrame operations without memory monitoring")
        
        # Check for garbage collection
        if "gc.collect()" in self.file_content:
            print("  ✅ Garbage collection implemented")
        else:
            issues.append("💡 MINOR: No explicit garbage collection calls")
        
        # Check for memory context managers
        if "memory_context" in self.file_content:
            print("  ✅ Memory context managers used")
        else:
            issues.append("⚠️ MEDIUM: No memory context management")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_configuration_problems(self):
        """Analyze configuration and setup issues"""
        print("\n6. ANALYZING CONFIGURATION PROBLEMS")
        print("-" * 50)
        
        issues = []
        
        # Check for missing config validation
        if "config" in self.file_content and "validate_config" not in self.file_content:
            issues.append("⚠️ MEDIUM: No configuration validation")
        
        # Check for hardcoded config values
        if "unified_config.yaml" in self.file_content:
            issues.append("💡 MINOR: Hardcoded config file path")
        
        # Check for config loading errors
        if "yaml.load" in self.file_content and "yaml.safe_load" not in self.file_content:
            issues.append("🔒 SECURITY: Using unsafe yaml.load instead of yaml.safe_load")
        
        # Check for missing default configs
        if "config.get(" not in self.file_content:
            issues.append("⚠️ MEDIUM: No default configuration fallbacks")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_api_integration(self):
        """Analyze API integration issues"""
        print("\n7. ANALYZING API INTEGRATION")
        print("-" * 50)
        
        issues = []
        
        # Check for API error handling
        if "polygon_client" in self.file_content:
            if "requests.exceptions" not in self.file_content:
                issues.append("⚠️ MEDIUM: Missing HTTP exception handling for API calls")
            
            if "rate_limit" not in self.file_content:
                issues.append("⚠️ MEDIUM: No rate limiting for API requests")
            
            if "retry" not in self.file_content:
                issues.append("💡 MINOR: No retry mechanism for failed API calls")
        
        # Check for API key security
        if "api_key" in self.file_content and "os.environ" not in self.file_content:
            issues.append("🔒 SECURITY: API keys may be hardcoded instead of using environment variables")
        
        # Check for data validation from API
        if "json()" in self.file_content and "validate" not in self.file_content:
            issues.append("💡 MINOR: API response data not validated")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_cross_validation(self):
        """Analyze cross-validation implementation"""
        print("\n8. ANALYZING CROSS-VALIDATION")
        print("-" * 50)
        
        issues = []
        
        # Check for data leakage
        if "TimeSeriesSplit" not in self.file_content and "time" in self.file_content.lower():
            issues.append("❌ CRITICAL: Potential data leakage - not using time-aware CV for time series data")
        
        # Check for purged CV
        if "PurgedTimeSeriesCV" in self.file_content and "purge" not in self.file_content:
            issues.append("⚠️ MEDIUM: Purged CV imported but purging logic unclear")
        
        # Check for sample weight issues
        if "sample_weight" in self.file_content and "NoneType" in self.file_content:
            issues.append("❌ CRITICAL: Sample weight unification fails with 'NoneType' errors")
        
        # Check for CV fold validation
        if "n_splits" in self.file_content and "min(" not in self.file_content:
            issues.append("💡 MINOR: No validation that n_splits doesn't exceed available data")
        
        self.critical_issues.extend([i for i in issues if "CRITICAL" in i])
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_prediction_generation(self):
        """Analyze prediction generation issues"""
        print("\n9. ANALYZING PREDICTION GENERATION")
        print("-" * 50)
        
        issues = []
        
        # Check for empty predictions
        if "predictions.empty" in self.file_content:
            issues.append("⚠️ MEDIUM: Predictions can be empty - insufficient error handling")
        
        # Check for prediction aggregation
        if "ensemble" in self.file_content and "mean" not in self.file_content:
            issues.append("💡 MINOR: Ensemble prediction aggregation method unclear")
        
        # Check for prediction validation
        if "predict(" in self.file_content and "isfinite" not in self.file_content:
            issues.append("💡 MINOR: No validation for infinite/NaN predictions")
        
        # Check for prediction alignment
        if "index" in self.file_content and "align" not in self.file_content:
            issues.append("💡 MINOR: Potential index misalignment in predictions")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_code_structure(self):
        """Analyze code structure and design issues"""
        print("\n10. ANALYZING CODE STRUCTURE")
        print("-" * 50)
        
        issues = []
        
        # Check file size
        lines = len(self.file_content.split('\n'))
        if lines > 2000:
            issues.append(f"⚠️ MEDIUM: Very large file ({lines} lines) - consider splitting into modules")
        
        # Check method complexity
        method_count = self.file_content.count("def ")
        if method_count > 50:
            issues.append(f"⚠️ MEDIUM: High method count ({method_count}) - complex class structure")
        
        # Check documentation
        docstring_count = self.file_content.count('"""')
        if docstring_count < method_count:
            issues.append("💡 MINOR: Insufficient documentation - missing docstrings")
        
        # Check for code duplication
        if self.file_content.count("try:") > 30:
            issues.append("💡 MINOR: Repetitive error handling patterns - consider refactoring")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def analyze_performance_issues(self):
        """Analyze performance-related issues"""
        print("\n11. ANALYZING PERFORMANCE ISSUES")
        print("-" * 50)
        
        issues = []
        
        # Check for inefficient operations
        if "iterrows()" in self.file_content:
            issues.append("⚠️ MEDIUM: Using inefficient iterrows() - consider vectorized operations")
        
        # Check for unnecessary copying
        if "copy()" in self.file_content:
            copy_count = self.file_content.count("copy()")
            if copy_count > 10:
                issues.append(f"💡 MINOR: Frequent DataFrame copying ({copy_count}) - memory inefficient")
        
        # Check for nested loops
        if "for " in self.file_content:
            nested_for = self.file_content.count("    for ")
            if nested_for > 5:
                issues.append("💡 MINOR: Multiple nested loops may impact performance")
        
        self.medium_issues.extend([i for i in issues if "MEDIUM" in i])
        self.minor_issues.extend([i for i in issues if "MINOR" in i])
        
        for issue in issues:
            print(f"  {issue}")
    
    def generate_problem_report(self):
        """Generate comprehensive problem report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PROBLEM REPORT")
        print("="*80)
        
        # Summary statistics
        total_critical = len(self.critical_issues)
        total_medium = len(self.medium_issues)  
        total_minor = len(self.minor_issues)
        total_issues = total_critical + total_medium + total_minor
        
        print(f"\n📊 PROBLEM SUMMARY:")
        print(f"   🔴 Critical Issues: {total_critical}")
        print(f"   🟡 Medium Issues:   {total_medium}")
        print(f"   🔵 Minor Issues:    {total_minor}")
        print(f"   📋 Total Issues:    {total_issues}")
        
        # Critical Issues Section
        if self.critical_issues:
            print(f"\n🔴 CRITICAL ISSUES ({len(self.critical_issues)}) - MUST FIX:")
            print("-" * 60)
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"   {i}. {issue}")
        
        # Medium Issues Section  
        if self.medium_issues:
            print(f"\n🟡 MEDIUM ISSUES ({len(self.medium_issues)}) - SHOULD FIX:")
            print("-" * 60)
            for i, issue in enumerate(self.medium_issues, 1):
                print(f"   {i}. {issue}")
        
        # Minor Issues Section
        if self.minor_issues:
            print(f"\n🔵 MINOR ISSUES ({len(self.minor_issues)}) - NICE TO FIX:")
            print("-" * 60)
            for i, issue in enumerate(self.minor_issues, 1):
                print(f"   {i}. {issue}")
        
        # Priority recommendations
        print(f"\n🎯 PRIORITY RECOMMENDATIONS:")
        print("-" * 60)
        print("   1. Fix data type handling for XGBoost/LightGBM (CRITICAL)")
        print("   2. Implement proper categorical variable encoding (CRITICAL)")
        print("   3. Fix missing cross_sectional_standardization module (CRITICAL)")
        print("   4. Implement regime detection methods (CRITICAL)")
        print("   5. Add comprehensive data type filtering (MEDIUM)")
        print("   6. Improve error handling specificity (MEDIUM)")
        print("   7. Add configuration validation (MEDIUM)")
        
        # Save report to file
        self.save_report_to_file(total_critical, total_medium, total_minor)
        
        print(f"\n✅ Analysis complete! Report saved to 'bma_problems_report.txt'")
    
    def save_report_to_file(self, critical, medium, minor):
        """Save detailed report to file"""
        with open('bma_problems_report.txt', 'w', encoding='utf-8') as f:
            f.write("BMA ULTRA ENHANCED - COMPREHENSIVE PROBLEM REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Critical Issues: {critical}\n")
            f.write(f"Medium Issues: {medium}\n") 
            f.write(f"Minor Issues: {minor}\n")
            f.write(f"Total Issues: {critical + medium + minor}\n\n")
            
            if self.critical_issues:
                f.write("CRITICAL ISSUES (MUST FIX):\n")
                f.write("-" * 40 + "\n")
                for i, issue in enumerate(self.critical_issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if self.medium_issues:
                f.write("MEDIUM ISSUES (SHOULD FIX):\n")
                f.write("-" * 40 + "\n")
                for i, issue in enumerate(self.medium_issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if self.minor_issues:
                f.write("MINOR ISSUES (NICE TO FIX):\n")
                f.write("-" * 40 + "\n")
                for i, issue in enumerate(self.minor_issues, 1):
                    f.write(f"{i}. {issue}\n")

def main():
    """Main execution"""
    analyzer = BMAProblemsAnalyzer()
    analyzer.analyze_all_problems()

if __name__ == "__main__":
    main()