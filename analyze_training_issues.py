#!/usr/bin/env python3
"""分析整个训练流程可能存在的问题"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from collections import defaultdict

def analyze_training_pipeline_issues():
    """分析训练流程的潜在问题"""
    
    print("=== 分析整个训练流程可能存在的问题 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    issues = {
        'data_leakage_risks': [],        # 数据泄漏风险
        'error_handling_gaps': [],       # 错误处理缺失
        'performance_bottlenecks': [],   # 性能瓶颈
        'memory_issues': [],            # 内存问题
        'missing_validations': [],      # 缺失的验证
        'deprecated_patterns': [],      # 过时的模式
        'testing_gaps': [],            # 测试缺失
        'production_risks': []          # 生产风险
    }
    
    # 关键训练相关方法
    training_methods = []
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        # 跳过注释
        if line_clean.startswith('#'):
            continue
            
        # 1. 数据泄漏风险检测
        leakage_patterns = [
            r'shift\(-\d+\)',           # 未来数据shift
            r'\.loc\[.*:\]',            # 可能的前瞻访问
            r'fillna\(.*method.*forward\)', # 前向填充
            r'\.expanding\(\)',         # 扩展窗口（可能泄漏）
            r'\.rolling\(\).*center=True', # 居中滚动窗口
        ]
        
        for pattern in leakage_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues['data_leakage_risks'].append((i, line_clean, pattern))
        
        # 2. 错误处理缺失
        if ('try:' in line or 'except:' in line or 
            '.get(' in line or 'KeyError' in line or 'AttributeError' in line):
            # 检查是否有裸except
            if re.match(r'\s*except:\s*$', line):
                issues['error_handling_gaps'].append((i, line_clean, "裸except子句"))
            # 检查是否缺少具体异常处理
            elif 'except Exception as e:' in line:
                next_lines = lines[i:i+3] if i < len(lines)-3 else lines[i:]
                if not any('logger.' in nl or 'print(' in nl for nl in next_lines):
                    issues['error_handling_gaps'].append((i, line_clean, "异常未记录"))
        
        # 3. 性能瓶颈检测
        bottleneck_patterns = [
            r'for.*in.*\.iterrows\(\)',  # 低效的iterrows
            r'\.apply\(lambda',          # lambda函数在apply中
            r'pd\.concat.*for.*in',      # 循环中的concat
            r'\.append\(.*for.*in',      # 循环中的append
            r'nested.*for.*for',         # 嵌套循环
        ]
        
        for pattern in bottleneck_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues['performance_bottlenecks'].append((i, line_clean, pattern))
        
        # 4. 内存问题
        memory_patterns = [
            r'pd\.read_csv.*chunksize',  # 大文件但无分块
            r'\.copy\(\).*\.copy\(\)',   # 重复复制
            r'del\s+\w+',               # 手动删除（可能内存管理问题）
        ]
        
        for pattern in memory_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues['memory_issues'].append((i, line_clean, pattern))
        
        # 5. 缺失验证
        if any(keyword in line.lower() for keyword in ['train', 'fit', 'predict']):
            # 检查输入验证
            if 'def ' in line and ('train' in line or 'fit' in line):
                method_lines = lines[i:i+10] if i < len(lines)-10 else lines[i:]
                has_validation = any(
                    'assert' in ml or 'raise' in ml or 'ValueError' in ml 
                    for ml in method_lines
                )
                if not has_validation:
                    issues['missing_validations'].append((i, line_clean, "缺少输入验证"))
        
        # 6. 过时模式
        deprecated_patterns = [
            r'\.ix\[',                   # 过时的ix索引
            r'np\.int\b',               # 过时的numpy类型
            r'pd\.np\.',                # 过时的pandas.np
            r'sklearn\.externals',       # 过时的sklearn externals
        ]
        
        for pattern in deprecated_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues['deprecated_patterns'].append((i, line_clean, pattern))
        
        # 7. 测试缺失检测
        if 'def test_' in line or 'class Test' in line:
            # 有测试方法，记录
            pass
        elif 'def ' in line and any(kw in line for kw in ['train', 'predict', 'evaluate']):
            # 关键方法但可能缺少测试
            issues['testing_gaps'].append((i, line_clean, "关键方法可能缺少测试"))
        
        # 8. 生产风险
        production_risk_patterns = [
            r'print\(',                  # 生产环境中的print
            r'debug.*=.*True',          # 调试模式未关闭
            r'import\s+pdb',            # 调试器导入
            r'\.sample\(\)',            # 随机采样（不可重现）
            r'random\.',               # 随机操作未设置种子
        ]
        
        for pattern in production_risk_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues['production_risks'].append((i, line_clean, pattern))
        
        # 识别训练方法
        if 'def ' in line and any(kw in line.lower() for kw in 
                                 ['train', 'fit', 'learn', 'optimize']):
            training_methods.append((i, line_clean))
    
    # 输出分析结果
    print("1. [CRITICAL] 数据泄漏风险:")
    if issues['data_leakage_risks']:
        for line_num, line_content, pattern in issues['data_leakage_risks'][:10]:
            print(f"   Line {line_num}: {line_content[:60]}... [{pattern}]")
        if len(issues['data_leakage_risks']) > 10:
            print(f"   ... 还有 {len(issues['data_leakage_risks']) - 10} 个风险")
    else:
        print("   [OK] 未发现明显的数据泄漏风险")
    
    print(f"\n2. [HIGH] 错误处理缺失:")
    if issues['error_handling_gaps']:
        for line_num, line_content, issue_type in issues['error_handling_gaps'][:10]:
            print(f"   Line {line_num}: {line_content[:60]}... [{issue_type}]")
        if len(issues['error_handling_gaps']) > 10:
            print(f"   ... 还有 {len(issues['error_handling_gaps']) - 10} 个问题")
    else:
        print("   [OK] 错误处理相对完善")
    
    print(f"\n3. [MEDIUM] 性能瓶颈:")
    if issues['performance_bottlenecks']:
        for line_num, line_content, pattern in issues['performance_bottlenecks'][:10]:
            print(f"   Line {line_num}: {line_content[:60]}... [{pattern}]")
        if len(issues['performance_bottlenecks']) > 10:
            print(f"   ... 还有 {len(issues['performance_bottlenecks']) - 10} 个瓶颈")
    else:
        print("   [OK] 未发现明显性能瓶颈")
    
    print(f"\n4. [MEDIUM] 内存问题:")
    if issues['memory_issues']:
        for line_num, line_content, pattern in issues['memory_issues'][:5]:
            print(f"   Line {line_num}: {line_content[:60]}... [{pattern}]")
    else:
        print("   [OK] 内存管理相对合理")
    
    print(f"\n5. [MEDIUM] 缺失验证:")
    if issues['missing_validations']:
        for line_num, line_content, issue_type in issues['missing_validations'][:5]:
            print(f"   Line {line_num}: {line_content[:60]}... [{issue_type}]")
    else:
        print("   [OK] 验证相对充分")
    
    print(f"\n6. [LOW] 过时模式:")
    if issues['deprecated_patterns']:
        for line_num, line_content, pattern in issues['deprecated_patterns'][:5]:
            print(f"   Line {line_num}: {line_content[:60]}... [{pattern}]")
    else:
        print("   [OK] 未发现过时模式")
    
    print(f"\n7. [INFO] 测试覆盖:")
    print(f"   发现 {len([m for m in training_methods])} 个训练相关方法")
    if issues['testing_gaps']:
        print(f"   可能缺少测试的方法: {len(issues['testing_gaps'])} 个")
    else:
        print("   [OK] 测试覆盖相对完整")
    
    print(f"\n8. [HIGH] 生产风险:")
    if issues['production_risks']:
        for line_num, line_content, pattern in issues['production_risks'][:10]:
            print(f"   Line {line_num}: {line_content[:60]}... [{pattern}]")
        if len(issues['production_risks']) > 10:
            print(f"   ... 还有 {len(issues['production_risks']) - 10} 个风险")
    else:
        print("   [OK] 生产就绪度较高")
    
    # 训练方法概览
    print(f"\n=== 训练方法概览 ===")
    if training_methods:
        print(f"发现 {len(training_methods)} 个训练相关方法:")
        for line_num, method in training_methods[:10]:
            method_name = method.split('def ')[1].split('(')[0] if 'def ' in method else method
            print(f"   Line {line_num}: {method_name}")
        if len(training_methods) > 10:
            print(f"   ... 还有 {len(training_methods) - 10} 个方法")
    
    # 风险评级
    print(f"\n=== 整体风险评估 ===")
    
    critical_count = len(issues['data_leakage_risks'])
    high_count = len(issues['error_handling_gaps']) + len(issues['production_risks'])
    medium_count = (len(issues['performance_bottlenecks']) + 
                   len(issues['memory_issues']) + 
                   len(issues['missing_validations']))
    
    total_issues = critical_count + high_count + medium_count
    
    print(f"CRITICAL级别问题: {critical_count}")
    print(f"HIGH级别问题: {high_count}")
    print(f"MEDIUM级别问题: {medium_count}")
    print(f"总问题数: {total_issues}")
    
    if critical_count > 0:
        print(f"\n[CRITICAL] 发现 {critical_count} 个严重问题，需要立即修复!")
        risk_level = "HIGH"
    elif high_count > 5:
        print(f"\n[WARNING] 发现 {high_count} 个高优先级问题")
        risk_level = "MEDIUM"
    elif total_issues > 20:
        print(f"\n[INFO] 发现 {total_issues} 个问题，建议逐步改进")
        risk_level = "LOW"
    else:
        print(f"\n[SUCCESS] 整体训练流程相对健康")
        risk_level = "LOW"
    
    return {
        'risk_level': risk_level,
        'total_issues': total_issues,
        'critical': critical_count,
        'high': high_count,
        'medium': medium_count,
        'training_methods': len(training_methods)
    }

if __name__ == "__main__":
    result = analyze_training_pipeline_issues()
    
    print(f"\n=== 修复建议 ===")
    if result['critical'] > 0:
        print("1. 立即修复数据泄漏风险")
    if result['high'] > 0:
        print("2. 改进错误处理和生产安全")
    if result['medium'] > 0:
        print("3. 优化性能和内存使用")
    
    print(f"\n整体风险等级: {result['risk_level']}")