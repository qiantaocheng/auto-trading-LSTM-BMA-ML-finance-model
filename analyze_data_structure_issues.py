#!/usr/bin/env python3
"""
深度分析BMA模型中数据结构可能存在的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

def analyze_dataframe_usage_patterns():
    """分析DataFrame使用模式中的潜在问题"""
    
    print("=== 分析DataFrame使用模式 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    issues = {
        'index_inconsistencies': [],      # 索引不一致问题
        'data_type_conflicts': [],        # 数据类型冲突  
        'memory_inefficiencies': [],      # 内存效率问题
        'alignment_issues': [],           # 数据对齐问题
        'column_naming_conflicts': [],    # 列名冲突
        'shape_assumptions': [],          # 形状假设问题
        'copy_inefficiencies': [],        # 复制效率问题
        'nan_handling_inconsistencies': [] # NaN处理不一致
    }
    
    # 跟踪DataFrame变量
    dataframe_vars = set()
    index_operations = []
    data_type_operations = []
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        if line_clean.startswith('#'):
            continue
        
        # 1. 检测索引操作问题
        index_patterns = [
            (r'\.reset_index\(\)', 'reset_index'),
            (r'\.set_index\([^)]+\)', 'set_index'), 
            (r'\.reindex\([^)]+\)', 'reindex'),
            (r'\.loc\[[^]]+\]', 'loc_access'),
            (r'\.iloc\[[^]]+\]', 'iloc_access'),
            (r'\.index\s*=', 'index_assignment'),
            (r'MultiIndex', 'multiindex_usage')
        ]
        
        for pattern, operation_type in index_patterns:
            if re.search(pattern, line):
                issues['index_inconsistencies'].append({
                    'line': i,
                    'operation': operation_type,
                    'context': line_clean[:100],
                    'potential_issue': get_index_issue_description(operation_type)
                })
        
        # 2. 检测数据类型问题
        dtype_patterns = [
            (r'\.astype\([^)]+\)', 'explicit_conversion'),
            (r'pd\.to_datetime\([^)]+\)', 'datetime_conversion'),
            (r'pd\.to_numeric\([^)]+\)', 'numeric_conversion'),
            (r'\.fillna\([^)]+\)', 'fillna_operation'),
            (r'\.dropna\([^)]+\)', 'dropna_operation'),
            (r'dtype\s*=', 'dtype_specification')
        ]
        
        for pattern, operation_type in dtype_patterns:
            if re.search(pattern, line):
                issues['data_type_conflicts'].append({
                    'line': i,
                    'operation': operation_type,
                    'context': line_clean[:100],
                    'potential_issue': get_dtype_issue_description(operation_type)
                })
        
        # 3. 检测内存效率问题
        memory_patterns = [
            (r'\.copy\(\)', 'unnecessary_copy'),
            (r'pd\.concat\([^)]+\)', 'concat_operation'),
            (r'\.append\([^)]+\)', 'append_operation'),
            (r'for.*in.*\.iterrows\(\)', 'inefficient_iteration'),
            (r'\.apply\(lambda.*\)', 'lambda_apply'),
            (r'\+.*pd\.DataFrame', 'dataframe_addition')
        ]
        
        for pattern, operation_type in memory_patterns:
            if re.search(pattern, line):
                issues['memory_inefficiencies'].append({
                    'line': i,
                    'operation': operation_type, 
                    'context': line_clean[:100],
                    'potential_issue': get_memory_issue_description(operation_type)
                })
        
        # 4. 检测数据对齐问题
        alignment_patterns = [
            (r'\.join\([^)]+\)', 'join_operation'),
            (r'\.merge\([^)]+\)', 'merge_operation'),
            (r'\.align\([^)]+\)', 'align_operation'),
            (r'\[.*\]\s*=.*\[.*\]', 'index_assignment_mismatch'),
            (r'\.reindex_like\([^)]+\)', 'reindex_like')
        ]
        
        for pattern, operation_type in alignment_patterns:
            if re.search(pattern, line):
                issues['alignment_issues'].append({
                    'line': i,
                    'operation': operation_type,
                    'context': line_clean[:100],
                    'potential_issue': get_alignment_issue_description(operation_type)
                })
        
        # 5. 检测列名冲突
        if 'columns' in line and ('=' in line or 'rename' in line):
            issues['column_naming_conflicts'].append({
                'line': i,
                'context': line_clean[:100],
                'potential_issue': '列名操作可能导致命名冲突'
            })
        
        # 6. 检测形状假设
        shape_patterns = [
            (r'\.shape\[0\]', 'row_count_assumption'),
            (r'\.shape\[1\]', 'column_count_assumption'),
            (r'len\([^)]+\)', 'length_assumption'),
            (r'\.empty', 'emptiness_check')
        ]
        
        for pattern, assumption_type in shape_patterns:
            if re.search(pattern, line):
                issues['shape_assumptions'].append({
                    'line': i,
                    'assumption': assumption_type,
                    'context': line_clean[:100],
                    'potential_issue': get_shape_issue_description(assumption_type)
                })
    
    # 分析结果
    print("1. [CRITICAL] 索引操作问题:")
    if issues['index_inconsistencies']:
        index_operations_count = Counter(item['operation'] for item in issues['index_inconsistencies'])
        print(f"   发现 {len(issues['index_inconsistencies'])} 个索引操作，类型分布:")
        for op_type, count in index_operations_count.most_common(5):
            print(f"     {op_type}: {count} 次")
        
        print(f"   前5个潜在问题:")
        for item in issues['index_inconsistencies'][:5]:
            print(f"     Line {item['line']}: {item['operation']} - {item['potential_issue']}")
    else:
        print("   [OK] 未发现明显索引问题")
    
    print(f"\n2. [HIGH] 数据类型冲突:")
    if issues['data_type_conflicts']:
        dtype_operations_count = Counter(item['operation'] for item in issues['data_type_conflicts'])
        print(f"   发现 {len(issues['data_type_conflicts'])} 个数据类型操作:")
        for op_type, count in dtype_operations_count.most_common(3):
            print(f"     {op_type}: {count} 次")
        
        print(f"   前3个潜在问题:")
        for item in issues['data_type_conflicts'][:3]:
            print(f"     Line {item['line']}: {item['potential_issue']}")
    else:
        print("   [OK] 数据类型操作相对安全")
    
    print(f"\n3. [MEDIUM] 内存效率问题:")
    if issues['memory_inefficiencies']:
        memory_operations_count = Counter(item['operation'] for item in issues['memory_inefficiencies'])
        print(f"   发现 {len(issues['memory_inefficiencies'])} 个内存效率问题:")
        for op_type, count in memory_operations_count.most_common(3):
            print(f"     {op_type}: {count} 次")
        
        high_impact_memory_issues = [
            item for item in issues['memory_inefficiencies'] 
            if item['operation'] in ['inefficient_iteration', 'unnecessary_copy', 'append_operation']
        ]
        if high_impact_memory_issues:
            print(f"   高影响内存问题 ({len(high_impact_memory_issues)} 个):")
            for item in high_impact_memory_issues[:3]:
                print(f"     Line {item['line']}: {item['potential_issue']}")
    else:
        print("   [OK] 内存使用相对高效")
    
    print(f"\n4. [MEDIUM] 数据对齐问题:")
    if issues['alignment_issues']:
        print(f"   发现 {len(issues['alignment_issues'])} 个数据对齐操作")
        merge_operations = [item for item in issues['alignment_issues'] if 'merge' in item['operation']]
        join_operations = [item for item in issues['alignment_issues'] if 'join' in item['operation']]
        
        print(f"     Merge操作: {len(merge_operations)} 个")
        print(f"     Join操作: {len(join_operations)} 个")
        
        if merge_operations:
            print(f"   Merge操作示例:")
            for item in merge_operations[:2]:
                print(f"     Line {item['line']}: {item['context'][:60]}...")
    else:
        print("   [OK] 数据对齐操作较少")
    
    print(f"\n5. [LOW] 其他结构问题:")
    print(f"   列名操作: {len(issues['column_naming_conflicts'])} 个")
    print(f"   形状假设: {len(issues['shape_assumptions'])} 个")
    
    return issues

def get_index_issue_description(operation_type):
    """获取索引操作问题描述"""
    descriptions = {
        'reset_index': '频繁reset_index可能导致索引丢失和性能下降',
        'set_index': '重复set_index操作可能导致索引不一致',
        'reindex': 'reindex操作可能引入NaN和数据不对齐',
        'loc_access': 'loc访问可能因索引不匹配而失败',
        'iloc_access': 'iloc访问假设特定的行序，可能脆弱',
        'index_assignment': '直接索引赋值可能破坏索引完整性',
        'multiindex_usage': 'MultiIndex使用不当可能导致复杂性'
    }
    return descriptions.get(operation_type, '未知索引问题')

def get_dtype_issue_description(operation_type):
    """获取数据类型问题描述"""
    descriptions = {
        'explicit_conversion': '显式类型转换可能丢失数据或引发错误',
        'datetime_conversion': '日期时间转换可能因格式不一致而失败',
        'numeric_conversion': '数值转换可能因非数值数据而失败',
        'fillna_operation': 'fillna策略不一致可能导致数据质量问题',
        'dropna_operation': 'dropna可能意外删除重要数据',
        'dtype_specification': 'dtype规范不当可能导致内存浪费'
    }
    return descriptions.get(operation_type, '未知数据类型问题')

def get_memory_issue_description(operation_type):
    """获取内存问题描述"""
    descriptions = {
        'unnecessary_copy': '不必要的copy()操作浪费内存',
        'concat_operation': 'concat操作可能导致内存碎片',
        'append_operation': 'append操作在循环中非常低效',
        'inefficient_iteration': 'iterrows()比向量化操作慢数倍',
        'lambda_apply': 'lambda apply比内置函数慢',
        'dataframe_addition': 'DataFrame直接相加可能导致意外行为'
    }
    return descriptions.get(operation_type, '未知内存问题')

def get_alignment_issue_description(operation_type):
    """获取对齐问题描述"""
    descriptions = {
        'join_operation': 'join操作可能导致数据错位或丢失',
        'merge_operation': 'merge参数不当可能导致意外结果',
        'align_operation': 'align操作可能引入不必要的NaN',
        'index_assignment_mismatch': '索引赋值不匹配可能导致数据错位',
        'reindex_like': 'reindex_like可能改变数据结构'
    }
    return descriptions.get(operation_type, '未知对齐问题')

def get_shape_issue_description(assumption_type):
    """获取形状假设问题描述"""
    descriptions = {
        'row_count_assumption': '假设特定行数可能在数据变化时失败',
        'column_count_assumption': '假设特定列数可能导致索引错误',
        'length_assumption': 'len()假设可能在空数据时失败',
        'emptiness_check': 'empty检查可能不足以处理所有边界情况'
    }
    return descriptions.get(assumption_type, '未知形状假设问题')

def analyze_data_flow_patterns():
    """分析数据流模式中的问题"""
    
    print(f"\n=== 数据流模式分析 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找数据流模式
    data_flow_issues = {
        'circular_dependencies': [],
        'data_leakage_risks': [], 
        'inconsistent_transformations': [],
        'pipeline_bottlenecks': []
    }
    
    lines = content.split('\n')
    
    # 跟踪数据变换序列
    transformations = []
    current_dataframe = None
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        # 检测数据流模式
        if '=' in line and any(df_op in line for df_op in ['.drop', '.fillna', '.transform', '.apply']):
            transformations.append({
                'line': i,
                'transformation': line_clean,
                'type': 'modification'
            })
        
        # 检测潜在的数据泄漏
        leakage_patterns = [
            r'shift\(-\d+\)',           # 向前shift（使用未来数据）
            r'\.expanding\(\)',         # expanding window（可能包含未来信息）
            r'fillna.*method.*forward', # 前向填充
            r'\.rolling\(.*center=True' # 中心化滚动窗口
        ]
        
        for pattern in leakage_patterns:
            if re.search(pattern, line):
                data_flow_issues['data_leakage_risks'].append({
                    'line': i,
                    'pattern': pattern,
                    'context': line_clean,
                    'risk_level': 'HIGH' if 'shift(-' in line else 'MEDIUM'
                })
    
    # 分析结果
    print("数据流风险评估:")
    print(f"潜在数据泄漏风险: {len(data_flow_issues['data_leakage_risks'])} 个")
    
    if data_flow_issues['data_leakage_risks']:
        high_risk = [item for item in data_flow_issues['data_leakage_risks'] if item['risk_level'] == 'HIGH']
        if high_risk:
            print(f"高风险数据泄漏 ({len(high_risk)} 个):")
            for item in high_risk[:3]:
                print(f"  Line {item['line']}: {item['context'][:60]}...")
    
    return data_flow_issues

def detect_data_structure_antipatterns():
    """检测数据结构反模式"""
    
    print(f"\n=== 数据结构反模式检测 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    antipatterns = {
        'god_dataframe': [],        # 过大的DataFrame
        'magic_columns': [],        # 硬编码列名
        'inconsistent_dtypes': [],  # 数据类型不一致
        'nested_loops_on_df': [],   # DataFrame上的嵌套循环
        'string_operations_on_categorical': []  # 分类数据上的字符串操作
    }
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        # 检测硬编码列名（魔法字符串）
        column_patterns = [
            r"'[A-Za-z_][A-Za-z0-9_]*'",  # 引号中的列名
            r'"[A-Za-z_][A-Za-z0-9_]*"'   # 双引号中的列名
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, line)
            if matches and any(keyword in line.lower() for keyword in ['column', 'col', '[', 'drop']):
                antipatterns['magic_columns'].append({
                    'line': i,
                    'columns': matches,
                    'context': line_clean[:80]
                })
        
        # 检测DataFrame上的嵌套循环
        if 'for' in line and any(df_indicator in line for df_indicator in ['.iterrows', '.itertuples', '.items']):
            # 查看接下来几行是否还有for循环
            context_lines = lines[i:i+5] if i < len(lines)-5 else lines[i:]
            if any('for' in context_line for context_line in context_lines[1:]):
                antipatterns['nested_loops_on_df'].append({
                    'line': i,
                    'context': line_clean,
                    'issue': '在DataFrame上使用嵌套循环，性能极差'
                })
    
    # 报告反模式
    print("检测到的反模式:")
    
    if antipatterns['magic_columns']:
        print(f"硬编码列名: {len(antipatterns['magic_columns'])} 处")
        print("  示例:")
        for item in antipatterns['magic_columns'][:3]:
            print(f"    Line {item['line']}: {item['columns']}")
    
    if antipatterns['nested_loops_on_df']:
        print(f"DataFrame嵌套循环: {len(antipatterns['nested_loops_on_df'])} 处")
        for item in antipatterns['nested_loops_on_df']:
            print(f"  Line {item['line']}: {item['issue']}")
    
    return antipatterns

def run_comprehensive_data_structure_analysis():
    """运行全面的数据结构分析"""
    
    print("🔍 开始BMA系统数据结构深度分析")
    print("=" * 60)
    
    # 1. DataFrame使用模式分析
    print("第一阶段：DataFrame使用模式分析")
    df_issues = analyze_dataframe_usage_patterns()
    
    # 2. 数据流模式分析  
    print("第二阶段：数据流模式分析")
    flow_issues = analyze_data_flow_patterns()
    
    # 3. 反模式检测
    print("第三阶段：数据结构反模式检测") 
    antipatterns = detect_data_structure_antipatterns()
    
    # 综合评估
    print("\n" + "=" * 60)
    print("📊 数据结构问题综合评估")
    
    critical_issues = len(df_issues['index_inconsistencies'])
    high_issues = len(df_issues['data_type_conflicts']) + len(flow_issues['data_leakage_risks'])
    medium_issues = len(df_issues['memory_inefficiencies']) + len(df_issues['alignment_issues'])
    
    total_issues = critical_issues + high_issues + medium_issues
    
    print(f"CRITICAL级别: {critical_issues} (索引问题)")
    print(f"HIGH级别: {high_issues} (数据类型 + 数据泄漏)")  
    print(f"MEDIUM级别: {medium_issues} (内存 + 对齐)")
    print(f"总问题数: {total_issues}")
    
    # 风险评级
    if critical_issues > 20:
        risk_level = "CRITICAL"
        print(f"\n🚨 [CRITICAL] 数据结构存在严重问题！")
    elif high_issues > 15:
        risk_level = "HIGH" 
        print(f"\n⚠️ [HIGH] 数据结构问题较多，需要优化")
    elif total_issues > 50:
        risk_level = "MEDIUM"
        print(f"\n📝 [MEDIUM] 数据结构有改进空间")
    else:
        risk_level = "LOW"
        print(f"\n✅ [GOOD] 数据结构相对健康")
    
    # 优先修复建议
    print(f"\n🔧 优先修复建议:")
    
    if critical_issues > 0:
        print("1. 立即修复索引操作不一致问题")
    if len(flow_issues['data_leakage_risks']) > 0:
        print("2. 检查并修复数据泄漏风险")
    if len(df_issues['memory_inefficiencies']) > 5:
        print("3. 优化内存使用效率")
    if len(antipatterns['magic_columns']) > 10:
        print("4. 重构硬编码列名，使用配置管理")
    
    return {
        'risk_level': risk_level,
        'total_issues': total_issues,
        'critical': critical_issues,
        'high': high_issues,
        'medium': medium_issues,
        'detailed_issues': {
            'dataframe_issues': df_issues,
            'flow_issues': flow_issues,
            'antipatterns': antipatterns
        }
    }

if __name__ == "__main__":
    try:
        result = run_comprehensive_data_structure_analysis()
        print(f"\n📋 分析完成！数据结构风险等级: {result['risk_level']}")
        
        if result['total_issues'] > 30:
            print(f"\n💡 建议：数据结构问题较多，考虑进行系统性重构")
        else:
            print(f"\n✨ 数据结构整体质量可接受，进行针对性优化即可")
            
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()