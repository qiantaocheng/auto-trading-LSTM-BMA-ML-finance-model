#!/usr/bin/env python3
"""
全面分析数据结构问题的诊断脚本
1. 语法错误检测
2. 数据结构模式分析
3. 内存使用模式分析
4. 索引策略问题
5. 性能瓶颈识别
"""

import os
import re
import ast
import sys
from typing import List, Dict, Any
import subprocess

def analyze_syntax_errors(file_path: str) -> List[str]:
    """分析语法错误"""
    print("1. 分析语法错误...")
    errors = []
    
    try:
        # 使用py_compile检查语法
        result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            errors.append(f"语法错误: {result.stderr}")
    except Exception as e:
        errors.append(f"语法检查失败: {e}")
    
    return errors

def analyze_data_structure_patterns(content: str) -> Dict[str, Any]:
    """分析数据结构使用模式"""
    print("2. 分析数据结构使用模式...")
    
    patterns = {
        'copy_operations': len(re.findall(r'\.copy\(\)', content)),
        'reset_index_operations': len(re.findall(r'\.reset_index\(\)', content)),
        'set_index_operations': len(re.findall(r'\.set_index\(', content)),
        'concat_operations': len(re.findall(r'pd\.concat\(', content)),
        'merge_operations': len(re.findall(r'\.merge\(', content)),
        'fillna_operations': len(re.findall(r'\.fillna\(', content)),
        'shift_operations': len(re.findall(r'\.shift\(', content)),
        'rolling_operations': len(re.findall(r'\.rolling\(', content)),
        'multiindex_checks': len(re.findall(r'isinstance.*MultiIndex', content)),
        'dataframe_creations': len(re.findall(r'pd\.DataFrame\(', content)),
    }
    
    return patterns

def analyze_memory_issues(content: str) -> List[str]:
    """分析内存使用问题"""
    print("3. 分析内存使用问题...")
    issues = []
    
    # 检查大型数据复制
    copy_in_loops = re.findall(r'for.*:\s*\n.*\.copy\(\)', content, re.MULTILINE)
    if copy_in_loops:
        issues.append(f"发现循环中的copy操作: {len(copy_in_loops)}个")
    
    # 检查连续的DataFrame操作
    chained_ops = re.findall(r'\.copy\(\)\..*\..*\.', content)
    if chained_ops:
        issues.append(f"发现链式DataFrame操作: {len(chained_ops)}个")
    
    # 检查不必要的中间变量
    temp_vars = re.findall(r'temp_\w+\s*=.*\.copy\(\)', content)
    if temp_vars:
        issues.append(f"发现临时变量copy: {len(temp_vars)}个")
    
    return issues

def analyze_index_strategy_issues(content: str) -> List[str]:
    """分析索引策略问题"""
    print("4. 分析索引策略问题...")
    issues = []
    
    # 检查索引重置模式
    reset_set_pattern = re.findall(r'\.reset_index\(\).*\.set_index\(', content)
    if reset_set_pattern:
        issues.append(f"发现reset_index -> set_index模式: {len(reset_set_pattern)}个")
    
    # 检查不一致的索引命名
    index_names = re.findall(r'\.set_index\(\[(.*?)\]\)', content)
    unique_patterns = set(index_names)
    if len(unique_patterns) > 1:
        issues.append(f"索引命名不一致: {unique_patterns}")
    
    # 检查MultiIndex频繁检查
    multiindex_checks = len(re.findall(r'isinstance.*MultiIndex', content))
    if multiindex_checks > 20:
        issues.append(f"过多MultiIndex检查: {multiindex_checks}次")
    
    return issues

def analyze_temporal_safety_issues(content: str) -> List[str]:
    """分析时间安全问题"""
    print("5. 分析时间安全问题...")
    issues = []
    
    # 检查负数shift
    negative_shifts = re.findall(r'\.shift\(-\d+\)', content)
    if negative_shifts:
        issues.append(f"发现负数shift操作: {len(negative_shifts)}个 - 可能导致数据泄漏")
    
    # 检查后向填充
    backward_fill = re.findall(r"fillna\(method=['\"]backward['\"]", content)
    if backward_fill:
        issues.append(f"发现后向填充: {len(backward_fill)}个 - 会导致数据泄漏")
    
    # 检查center=True的rolling
    center_rolling = re.findall(r'\.rolling\([^)]*center=True', content)
    if center_rolling:
        issues.append(f"发现center=True的rolling: {len(center_rolling)}个 - 使用未来数据")
    
    return issues

def find_problematic_code_sections(content: str) -> Dict[str, List[str]]:
    """找出有问题的代码段"""
    print("6. 识别有问题的代码段...")
    
    problems = {
        'malformed_lines': [],
        'incomplete_blocks': [],
        'syntax_issues': []
    }
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # 检查格式错误的行
        if '# OPTIMIZED:' in line and not line.strip().endswith((':', ')', ']', '}')):
            if "'" in line and line.count("'") % 2 == 1:
                problems['malformed_lines'].append(f"Line {i}: {line.strip()}")
        
        # 检查未完成的try块
        if line.strip().startswith('try:'):
            # 查找对应的except或finally
            found_except = False
            for j in range(i, min(i+50, len(lines))):
                if lines[j].strip().startswith(('except', 'finally')):
                    found_except = True
                    break
            if not found_except:
                problems['incomplete_blocks'].append(f"Line {i}: 未完成的try块")
    
    return problems

def generate_fix_recommendations(analysis_results: Dict[str, Any]) -> List[str]:
    """生成修复建议"""
    print("7. 生成修复建议...")
    
    recommendations = []
    
    # 语法错误修复
    if analysis_results['syntax_errors']:
        recommendations.append("🔴 紧急: 修复语法错误")
        recommendations.extend([f"  - {error}" for error in analysis_results['syntax_errors']])
    
    # 性能优化建议
    patterns = analysis_results['data_patterns']
    if patterns['copy_operations'] > 20:
        recommendations.append(f"⚡ 优化: 减少copy操作 (当前: {patterns['copy_operations']}次)")
    
    if patterns['reset_index_operations'] > 15:
        recommendations.append(f"⚡ 优化: 统一索引策略 (当前reset_index: {patterns['reset_index_operations']}次)")
    
    # 内存优化建议
    if analysis_results['memory_issues']:
        recommendations.append("💾 内存优化:")
        recommendations.extend([f"  - {issue}" for issue in analysis_results['memory_issues']])
    
    # 时间安全建议
    if analysis_results['temporal_issues']:
        recommendations.append("⚠️  时间安全修复:")
        recommendations.extend([f"  - {issue}" for issue in analysis_results['temporal_issues']])
    
    return recommendations

def run_comprehensive_analysis():
    """运行全面分析"""
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    print("=== 开始全面数据结构问题分析 ===\n")
    
    # 读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"文件大小: {len(content)} 字符\n")
    except Exception as e:
        print(f"文件读取失败: {e}")
        return None
    
    # 执行各项分析
    analysis_results = {
        'syntax_errors': analyze_syntax_errors(file_path),
        'data_patterns': analyze_data_structure_patterns(content),
        'memory_issues': analyze_memory_issues(content),
        'index_issues': analyze_index_strategy_issues(content),
        'temporal_issues': analyze_temporal_safety_issues(content),
        'code_problems': find_problematic_code_sections(content)
    }
    
    # 生成报告
    print("\n=== 分析结果报告 ===")
    
    print("\n📊 数据结构操作统计:")
    for op, count in analysis_results['data_patterns'].items():
        if count > 0:
            status = "🔴" if count > 30 else "🟡" if count > 15 else "🟢"
            print(f"  {status} {op}: {count}")
    
    print("\n🔍 发现的问题:")
    total_issues = 0
    
    for category, issues in analysis_results.items():
        if isinstance(issues, list) and issues:
            total_issues += len(issues)
            print(f"\n{category.upper()}:")
            for issue in issues:
                print(f"  - {issue}")
        elif isinstance(issues, dict):
            for subcategory, subitems in issues.items():
                if subitems:
                    total_issues += len(subitems)
                    print(f"\n{subcategory.upper()}:")
                    for item in subitems:
                        print(f"  - {item}")
    
    print(f"\n📈 总计发现问题: {total_issues} 个")
    
    # 生成修复建议
    recommendations = generate_fix_recommendations(analysis_results)
    
    print("\n🔧 修复建议:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n=== 分析完成 ===")
    
    return analysis_results

if __name__ == "__main__":
    run_comprehensive_analysis()