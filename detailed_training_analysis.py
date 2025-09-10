#!/usr/bin/env python3
"""深入分析训练流程的具体问题和修复方案"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_specific_training_issues():
    """深入分析训练流程的具体问题"""
    
    print("=== 深入分析训练流程具体问题 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    specific_issues = {
        'bare_except_blocks': [],      # 裸except块
        'training_method_issues': [],  # 训练方法问题  
        'data_validation_missing': [], # 数据验证缺失
        'production_safety': [],      # 生产安全问题
        'memory_leaks': [],           # 内存泄漏
        'error_recovery': []          # 错误恢复机制
    }
    
    # 详细分析每个问题
    current_method = None
    method_start_line = 0
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        # 跳过注释
        if line_clean.startswith('#'):
            continue
        
        # 跟踪当前方法
        if line_clean.startswith('def '):
            current_method = line_clean.split('(')[0].replace('def ', '').strip()
            method_start_line = i
        
        # 1. 分析裸except块
        if line_clean == 'except:':
            # 查看后续处理
            next_lines = lines[i:i+5] if i < len(lines)-5 else lines[i:]
            has_logging = any('logger.' in nl or 'print(' in nl for nl in next_lines)
            has_reraise = any('raise' in nl for nl in next_lines)
            
            issue_details = {
                'line': i,
                'method': current_method,
                'has_logging': has_logging,
                'has_reraise': has_reraise,
                'severity': 'HIGH' if not has_logging else 'MEDIUM'
            }
            specific_issues['bare_except_blocks'].append(issue_details)
        
        # 2. 分析训练方法问题
        if current_method and any(keyword in current_method.lower() for keyword in 
                                 ['train', 'fit', 'learn', 'predict', 'optimize']):
            # 检查方法是否有输入验证
            method_lines = lines[method_start_line-1:i+20] if i < method_start_line + 50 else []
            
            has_input_validation = any(
                'assert' in ml or 'raise ValueError' in ml or 'raise TypeError' in ml 
                for ml in method_lines
            )
            
            has_shape_check = any(
                '.shape' in ml or 'len(' in ml or 'empty' in ml
                for ml in method_lines
            )
            
            has_null_check = any(
                'isna()' in ml or 'isnull()' in ml or 'dropna' in ml
                for ml in method_lines
            )
            
            if not has_input_validation:
                issue_details = {
                    'line': method_start_line,
                    'method': current_method,
                    'missing_validation': True,
                    'has_shape_check': has_shape_check,
                    'has_null_check': has_null_check,
                    'severity': 'MEDIUM'
                }
                specific_issues['training_method_issues'].append(issue_details)
        
        # 3. 生产安全问题
        if 'print(' in line and current_method:
            # 在生产代码中使用print
            if any(keyword in current_method.lower() for keyword in 
                   ['train', 'predict', 'analyze', 'process']):
                issue_details = {
                    'line': i,
                    'method': current_method,
                    'issue_type': 'print_in_production',
                    'content': line_clean[:80],
                    'severity': 'MEDIUM'
                }
                specific_issues['production_safety'].append(issue_details)
        
        # 4. 内存泄漏检测
        if 'del ' in line:
            # 手动内存管理
            next_line = lines[i] if i < len(lines) else ""
            prev_line = lines[i-2] if i > 1 else ""
            
            issue_details = {
                'line': i,
                'method': current_method,
                'content': line_clean,
                'context': f"Prev: {prev_line.strip()[:40]}, Next: {next_line.strip()[:40]}",
                'severity': 'LOW'
            }
            specific_issues['memory_leaks'].append(issue_details)
    
    # 输出详细分析结果
    print("=== 关键问题详细分析 ===\n")
    
    print("1. [HIGH PRIORITY] 裸except块分析:")
    if specific_issues['bare_except_blocks']:
        print(f"   发现 {len(specific_issues['bare_except_blocks'])} 个裸except块")
        
        critical_blocks = [b for b in specific_issues['bare_except_blocks'] if b['severity'] == 'HIGH']
        medium_blocks = [b for b in specific_issues['bare_except_blocks'] if b['severity'] == 'MEDIUM']
        
        if critical_blocks:
            print(f"   [CRITICAL] {len(critical_blocks)} 个无日志记录的裸except:")
            for block in critical_blocks[:5]:
                print(f"     Line {block['line']} in {block['method']}: 无异常处理")
        
        if medium_blocks:
            print(f"   [MEDIUM] {len(medium_blocks)} 个有日志的裸except:")
            for block in medium_blocks[:3]:
                print(f"     Line {block['line']} in {block['method']}: 有日志但应指定异常类型")
    else:
        print("   [OK] 未发现裸except块")
    
    print(f"\n2. [MEDIUM PRIORITY] 训练方法验证缺失:")
    if specific_issues['training_method_issues']:
        print(f"   发现 {len(specific_issues['training_method_issues'])} 个训练方法缺少验证")
        
        for issue in specific_issues['training_method_issues'][:5]:
            validation_status = []
            if not issue['has_shape_check']:
                validation_status.append("无形状检查")
            if not issue['has_null_check']:
                validation_status.append("无空值检查")
            
            print(f"     Line {issue['line']}: {issue['method']} - {', '.join(validation_status)}")
    else:
        print("   [OK] 训练方法验证相对完善")
    
    print(f"\n3. [MEDIUM PRIORITY] 生产安全问题:")
    if specific_issues['production_safety']:
        print(f"   发现 {len(specific_issues['production_safety'])} 个生产安全问题")
        
        print_issues = [p for p in specific_issues['production_safety'] if p['issue_type'] == 'print_in_production']
        if print_issues:
            print(f"   [WARNING] {len(print_issues)} 个生产代码中的print语句:")
            for issue in print_issues[:5]:
                print(f"     Line {issue['line']} in {issue['method']}: {issue['content'][:50]}...")
    else:
        print("   [OK] 生产安全相对良好")
    
    print(f"\n4. [LOW PRIORITY] 内存管理分析:")
    if specific_issues['memory_leaks']:
        print(f"   发现 {len(specific_issues['memory_leaks'])} 个手动内存管理")
        for issue in specific_issues['memory_leaks'][:3]:
            print(f"     Line {issue['line']}: {issue['content']}")
    else:
        print("   [OK] 未发现明显内存管理问题")
    
    # 提供具体的修复建议
    print(f"\n=== 具体修复建议 ===\n")
    
    if specific_issues['bare_except_blocks']:
        print("1. 修复裸except块:")
        print("   替换 'except:' 为 'except Exception as e:'")
        print("   添加异常日志: logger.error(f'异常: {e}')")
        print("   考虑具体异常类型而非通用Exception")
    
    if specific_issues['training_method_issues']:
        print("\n2. 增强训练方法验证:")
        print("   添加输入形状检查: assert X.shape[0] > 0")
        print("   添加空值检查: assert not X.isnull().any().any()")
        print("   添加数据类型验证: assert isinstance(X, pd.DataFrame)")
    
    if specific_issues['production_safety']:
        print("\n3. 提高生产安全:")
        print("   替换print为logger.info/debug")
        print("   添加环境检查控制调试输出")
        print("   使用配置控制详细程度")
    
    # 生成修复优先级
    total_critical = len([b for b in specific_issues['bare_except_blocks'] if b['severity'] == 'HIGH'])
    total_high = len(specific_issues['production_safety'])
    total_medium = len(specific_issues['training_method_issues'])
    
    print(f"\n=== 修复优先级 ===")
    print(f"CRITICAL (立即修复): {total_critical} 个问题")
    print(f"HIGH (本周修复): {total_high} 个问题") 
    print(f"MEDIUM (本月修复): {total_medium} 个问题")
    
    if total_critical > 0:
        priority = "URGENT"
    elif total_high > 5:
        priority = "HIGH"  
    elif total_medium > 10:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    print(f"\n整体修复优先级: {priority}")
    
    return {
        'critical_issues': total_critical,
        'high_issues': total_high,
        'medium_issues': total_medium,
        'priority': priority,
        'bare_except_count': len(specific_issues['bare_except_blocks']),
        'training_method_issues': len(specific_issues['training_method_issues'])
    }

if __name__ == "__main__":
    result = analyze_specific_training_issues()
    
    if result['critical_issues'] > 0:
        print(f"\n[URGENT] 需要立即修复 {result['critical_issues']} 个严重问题!")
    elif result['priority'] in ['HIGH', 'MEDIUM']:
        print(f"\n[ACTION NEEDED] 训练流程需要改进，优先级: {result['priority']}")
    else:
        print(f"\n[GOOD] 训练流程相对健康，少量优化即可")