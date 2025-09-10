#!/usr/bin/env python3
"""
简化的数据结构分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from collections import Counter

def analyze_data_structure_issues():
    print("=== BMA Data Structure Analysis ===")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    issues = {
        'index_operations': [],
        'memory_issues': [],
        'data_type_issues': [],
        'alignment_issues': [],
        'potential_leakage': []
    }
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        if line_clean.startswith('#'):
            continue
        
        # Index operations
        index_ops = ['.reset_index()', '.set_index(', '.reindex(', 'MultiIndex', '.loc[', '.iloc[']
        if any(op in line for op in index_ops):
            issues['index_operations'].append((i, line_clean[:80]))
        
        # Memory inefficiencies  
        memory_ops = ['.copy()', 'iterrows()', '.append(', 'pd.concat', 'lambda']
        if any(op in line for op in memory_ops):
            issues['memory_issues'].append((i, line_clean[:80]))
        
        # Data type operations
        dtype_ops = ['.astype(', 'pd.to_datetime', '.fillna(', '.dropna(', 'dtype=']
        if any(op in line for op in dtype_ops):
            issues['data_type_issues'].append((i, line_clean[:80]))
        
        # Alignment operations
        align_ops = ['.merge(', '.join(', '.align(']
        if any(op in line for op in align_ops):
            issues['alignment_issues'].append((i, line_clean[:80]))
        
        # Potential data leakage
        leakage_ops = ['shift(-', '.expanding()', 'forward', 'center=True']
        if any(op in line for op in leakage_ops):
            issues['potential_leakage'].append((i, line_clean[:80]))
    
    # Report findings
    print("\nFINDINGS:")
    for issue_type, issue_list in issues.items():
        print(f"{issue_type.replace('_', ' ').title()}: {len(issue_list)} instances")
    
    # Show critical examples
    print("\nCRITICAL ISSUES:")
    
    if issues['potential_leakage']:
        print(f"\nPotential Data Leakage ({len(issues['potential_leakage'])} instances):")
        for line_num, context in issues['potential_leakage'][:5]:
            print(f"  Line {line_num}: {context}")
    
    if len(issues['index_operations']) > 50:
        print(f"\nExcessive Index Operations ({len(issues['index_operations'])} instances):")
        print("  This suggests inconsistent indexing strategy")
    
    if len(issues['memory_issues']) > 20:
        print(f"\nMemory Inefficiencies ({len(issues['memory_issues'])} instances):")
        memory_critical = [item for item in issues['memory_issues'] if 'iterrows' in item[1] or 'append' in item[1]]
        if memory_critical:
            print("  Critical memory issues:")
            for line_num, context in memory_critical[:3]:
                print(f"    Line {line_num}: {context}")
    
    # Risk assessment
    total_issues = sum(len(v) for v in issues.values())
    leakage_risk = len(issues['potential_leakage'])
    memory_risk = len(issues['memory_issues'])
    
    print(f"\nRISK ASSESSMENT:")
    print(f"Total Issues: {total_issues}")
    print(f"Data Leakage Risk: {'HIGH' if leakage_risk > 10 else 'MEDIUM' if leakage_risk > 0 else 'LOW'}")
    print(f"Memory Risk: {'HIGH' if memory_risk > 30 else 'MEDIUM' if memory_risk > 10 else 'LOW'}")
    
    if total_issues > 100:
        risk_level = "CRITICAL"
    elif total_issues > 50:
        risk_level = "HIGH"
    elif total_issues > 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    print(f"Overall Risk Level: {risk_level}")
    
    return issues, risk_level, total_issues

def analyze_specific_patterns():
    """分析特定的问题模式"""
    
    print("\n=== SPECIFIC PATTERN ANALYSIS ===")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patterns = {
        'data_leakage': [
            (r'shift\(-\d+\)', 'Future data access'),
            (r'\.expanding\(\)', 'Expanding window (potential lookahead)'),
            (r'fillna.*forward', 'Forward fill (uses future data)'),
            (r'rolling.*center=True', 'Centered rolling window')
        ],
        'performance': [
            (r'for.*iterrows', 'Slow row iteration'),
            (r'\.append.*in.*for', 'Inefficient append in loop'),
            (r'pd\.concat.*for', 'Concat in loop'),
            (r'\.copy\(\).*\.copy\(\)', 'Multiple copies')
        ],
        'consistency': [
            (r'\.reset_index.*\.set_index', 'Index reset/set pattern'),
            (r'MultiIndex.*reset_index', 'MultiIndex inconsistency'),
            (r'\.merge.*\.merge', 'Multiple merges'),
            (r'astype.*astype', 'Multiple type conversions')
        ]
    }
    
    found_patterns = {}
    
    for category, pattern_list in patterns.items():
        found_patterns[category] = []
        for pattern, description in pattern_list:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                found_patterns[category].append((description, len(matches)))
    
    # Report specific patterns
    for category, pattern_results in found_patterns.items():
        if pattern_results:
            print(f"\n{category.upper()} ISSUES:")
            for description, count in pattern_results:
                print(f"  {description}: {count} occurrences")
    
    return found_patterns

def generate_recommendations(issues, risk_level, total_issues):
    """生成修复建议"""
    
    print(f"\n=== RECOMMENDATIONS ===")
    
    if risk_level in ["CRITICAL", "HIGH"]:
        print("URGENT ACTIONS REQUIRED:")
        
        if len(issues['potential_leakage']) > 0:
            print("1. CRITICAL: Review all data leakage risks")
            print("   - Check shift(-X) operations")
            print("   - Validate forward fill strategies")
            print("   - Ensure no lookahead bias")
        
        if len(issues['index_operations']) > 50:
            print("2. HIGH: Consolidate indexing strategy")
            print("   - Standardize on MultiIndex(date, ticker)")
            print("   - Minimize reset_index/set_index cycles")
        
        if len(issues['memory_issues']) > 20:
            print("3. HIGH: Optimize memory usage")
            print("   - Replace iterrows() with vectorized operations")
            print("   - Avoid append() in loops")
            print("   - Minimize unnecessary copy() operations")
    
    elif risk_level == "MEDIUM":
        print("RECOMMENDED IMPROVEMENTS:")
        print("1. Review data type consistency")
        print("2. Optimize merge/join operations")
        print("3. Standardize column naming patterns")
    
    else:
        print("MINOR OPTIMIZATIONS:")
        print("1. Code cleanup for better maintainability")
        print("2. Consider performance profiling")
    
    # Specific priority recommendations
    print(f"\nPRIORITY ORDER:")
    
    priorities = []
    if len(issues['potential_leakage']) > 0:
        priorities.append("1. Fix data leakage risks (CRITICAL)")
    if len(issues['memory_issues']) > 30:
        priorities.append("2. Optimize memory usage (HIGH)")
    if len(issues['index_operations']) > 50:
        priorities.append("3. Standardize indexing (HIGH)")
    if len(issues['alignment_issues']) > 10:
        priorities.append("4. Review merge/join operations (MEDIUM)")
    
    for priority in priorities:
        print(f"   {priority}")
    
    if not priorities:
        print("   No critical issues detected - system is relatively healthy")

if __name__ == "__main__":
    try:
        # Main analysis
        issues, risk_level, total_issues = analyze_data_structure_issues()
        
        # Specific patterns
        patterns = analyze_specific_patterns()
        
        # Generate recommendations
        generate_recommendations(issues, risk_level, total_issues)
        
        print(f"\nANALYSIS COMPLETE")
        print(f"Total issues found: {total_issues}")
        print(f"Risk level: {risk_level}")
        
        if risk_level in ["CRITICAL", "HIGH"]:
            print("Immediate attention required!")
        else:
            print("System is in acceptable condition")
            
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()