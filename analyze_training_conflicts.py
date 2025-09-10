#!/usr/bin/env python3
"""分析训练流程中的功能冲突和配置冲突"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from collections import defaultdict

def analyze_training_conflicts():
    """分析训练流程中的功能冲突和配置冲突"""
    
    print("=== 分析训练流程功能冲突和配置冲突 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    conflicts = {
        'duplicate_methods': defaultdict(list),        # 重复方法
        'conflicting_configs': defaultdict(list),     # 冲突配置
        'inconsistent_parameters': defaultdict(list), # 不一致参数
        'competing_features': defaultdict(list),      # 竞争特征
        'overlapping_functionality': defaultdict(list), # 重叠功能
        'version_conflicts': defaultdict(list),       # 版本冲突
        'initialization_conflicts': [],               # 初始化冲突
        'data_flow_conflicts': []                     # 数据流冲突
    }
    
    # 跟踪方法定义
    method_definitions = {}
    current_class = None
    
    # 跟踪配置使用
    config_usage = defaultdict(list)
    
    # 跟踪特征计算
    feature_calculations = defaultdict(list)
    
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        
        if line_clean.startswith('#'):
            continue
        
        # 1. 检测重复方法定义
        if line_clean.startswith('def '):
            method_name = line_clean.split('(')[0].replace('def ', '').strip()
            
            if method_name in method_definitions:
                conflicts['duplicate_methods'][method_name].extend([
                    method_definitions[method_name], 
                    {'line': i, 'definition': line_clean}
                ])
            else:
                method_definitions[method_name] = {'line': i, 'definition': line_clean}
        
        # 2. 检测配置冲突
        config_patterns = [
            (r'gap.*=.*(\d+)', 'gap_days'),
            (r'embargo.*=.*(\d+)', 'embargo_days'),
            (r'lag.*=.*(\d+)', 'lag_days'),
            (r'horizon.*=.*(\d+)', 'horizon_days'),
            (r'window.*=.*(\d+)', 'window_size'),
            (r'n_estimators.*=.*(\d+)', 'n_estimators'),
            (r'learning_rate.*=.*([0-9.]+)', 'learning_rate'),
            (r'max_depth.*=.*(\d+)', 'max_depth')
        ]
        
        for pattern, param_type in config_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                for match in matches:
                    config_usage[param_type].append({
                        'line': i,
                        'value': match,
                        'context': line_clean[:80]
                    })
        
        # 3. 检测特征计算冲突
        feature_patterns = [
            (r'\.rolling\(\d+\)', 'rolling_window'),
            (r'\.ewm\(.*\)', 'ewm_calculation'),
            (r'\.pct_change\(\)', 'returns_calculation'),
            (r'\.shift\(-?\d+\)', 'data_shift'),
            (r'beta.*calculation', 'beta_calculation'),
            (r'alpha.*calculation', 'alpha_calculation')
        ]
        
        for pattern, feature_type in feature_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                feature_calculations[feature_type].append({
                    'line': i,
                    'context': line_clean[:100]
                })
        
        # 4. 检测初始化冲突
        if 'self.' in line and '=' in line and '__init__' in content[max(0, i-50):i]:
            # 检查是否有重复的属性初始化
            attr_match = re.search(r'self\.(\w+)\s*=', line)
            if attr_match:
                attr_name = attr_match.group(1)
                if any('self.' + attr_name + ' =' in l for l in lines[:i-1]):
                    conflicts['initialization_conflicts'].append({
                        'line': i,
                        'attribute': attr_name,
                        'context': line_clean
                    })
        
        # 5. 检测版本冲突标记
        version_indicators = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'legacy', 'deprecated', 'old']
        if any(indicator in line.upper() for indicator in version_indicators):
            conflicts['version_conflicts']['mixed_versions'].append({
                'line': i,
                'context': line_clean[:80]
            })
    
    # 分析冲突
    print("=== 功能冲突分析 ===\n")
    
    # 1. 重复方法分析
    print("1. [CRITICAL] 重复方法定义:")
    if conflicts['duplicate_methods']:
        for method_name, definitions in conflicts['duplicate_methods'].items():
            print(f"   方法 '{method_name}' 定义了 {len(definitions)} 次:")
            for defn in definitions[:3]:  # 只显示前3个
                print(f"     Line {defn['line']}: {defn['definition'][:60]}...")
    else:
        print("   [OK] 未发现重复方法定义")
    
    # 2. 配置参数冲突分析
    print(f"\n2. [HIGH] 配置参数不一致:")
    config_conflicts = {}
    for param_type, usages in config_usage.items():
        if len(usages) > 1:
            values = [u['value'] for u in usages]
            unique_values = set(values)
            if len(unique_values) > 1:
                config_conflicts[param_type] = {
                    'values': list(unique_values),
                    'count': len(usages),
                    'usages': usages[:5]  # 只显示前5个
                }
    
    if config_conflicts:
        for param_type, conflict_info in config_conflicts.items():
            print(f"   参数 '{param_type}' 有不同值: {conflict_info['values']}")
            for usage in conflict_info['usages'][:3]:
                print(f"     Line {usage['line']}: {usage['context'][:50]}...")
    else:
        print("   [OK] 未发现明显的参数不一致")
    
    # 3. 特征计算冲突
    print(f"\n3. [MEDIUM] 特征计算重复/冲突:")
    feature_conflicts = {}
    for feature_type, calculations in feature_calculations.items():
        if len(calculations) > 3:  # 如果同一类型计算太多次
            feature_conflicts[feature_type] = calculations
    
    if feature_conflicts:
        for feature_type, calculations in feature_conflicts.items():
            print(f"   特征类型 '{feature_type}' 计算了 {len(calculations)} 次:")
            for calc in calculations[:3]:
                print(f"     Line {calc['line']}: {calc['context'][:50]}...")
    else:
        print("   [OK] 特征计算相对合理")
    
    # 4. 初始化冲突
    print(f"\n4. [MEDIUM] 初始化冲突:")
    if conflicts['initialization_conflicts']:
        print(f"   发现 {len(conflicts['initialization_conflicts'])} 个潜在初始化冲突:")
        for conflict in conflicts['initialization_conflicts'][:5]:
            print(f"     Line {conflict['line']}: {conflict['attribute']} - {conflict['context'][:50]}...")
    else:
        print("   [OK] 未发现明显初始化冲突")
    
    # 5. 版本混合问题
    print(f"\n5. [LOW] 版本混合问题:")
    if conflicts['version_conflicts']['mixed_versions']:
        version_count = len(conflicts['version_conflicts']['mixed_versions'])
        print(f"   发现 {version_count} 处版本标记，可能存在新旧代码混合:")
        for version_ref in conflicts['version_conflicts']['mixed_versions'][:5]:
            print(f"     Line {version_ref['line']}: {version_ref['context'][:50]}...")
    else:
        print("   [OK] 版本相对统一")
    
    # 深度功能冲突分析
    print(f"\n=== 深度功能冲突分析 ===\n")
    
    # 检查训练方法冲突
    training_methods = [name for name in method_definitions.keys() 
                       if any(keyword in name.lower() for keyword in 
                             ['train', 'fit', 'learn', 'optimize'])]
    
    print("6. 训练方法分析:")
    if len(training_methods) > 10:
        print(f"   发现 {len(training_methods)} 个训练相关方法，可能存在功能重复:")
        method_groups = defaultdict(list)
        for method in training_methods:
            if 'traditional' in method.lower():
                method_groups['traditional'].append(method)
            elif 'meta' in method.lower() or 'stack' in method.lower():
                method_groups['meta'].append(method)
            elif 'regime' in method.lower():
                method_groups['regime'].append(method)
            else:
                method_groups['other'].append(method)
        
        for group, methods in method_groups.items():
            if len(methods) > 1:
                print(f"     {group}组: {len(methods)} 个方法 - {', '.join(methods[:3])}...")
    else:
        print(f"   训练方法数量合理: {len(training_methods)} 个")
    
    # 检查预测方法冲突
    prediction_methods = [name for name in method_definitions.keys() 
                         if any(keyword in name.lower() for keyword in 
                               ['predict', 'forecast', 'generate'])]
    
    print(f"\n7. 预测方法分析:")
    if len(prediction_methods) > 5:
        print(f"   发现 {len(prediction_methods)} 个预测相关方法，检查功能重叠:")
        for method in prediction_methods[:5]:
            print(f"     {method}")
    else:
        print(f"   预测方法数量合理: {len(prediction_methods)} 个")
    
    # 配置系统冲突检查
    print(f"\n8. 配置系统冲突:")
    config_access_patterns = [
        'self.config.get',
        'self.unified_config.get', 
        'timing_registry.',
        'GLOBAL_',
        'hardcoded'
    ]
    
    config_access_counts = {}
    for pattern in config_access_patterns:
        count = content.count(pattern)
        if count > 0:
            config_access_counts[pattern] = count
    
    if len(config_access_counts) > 2:
        print("   发现多种配置访问方式，可能存在冲突:")
        for pattern, count in config_access_counts.items():
            print(f"     {pattern}: {count} 次使用")
    else:
        print("   [OK] 配置访问方式相对统一")
    
    # 总体冲突评估
    print(f"\n=== 整体冲突评估 ===")
    
    critical_conflicts = len(conflicts['duplicate_methods'])
    high_conflicts = len(config_conflicts)
    medium_conflicts = (len(feature_conflicts) + 
                       len(conflicts['initialization_conflicts']))
    version_issues = len(conflicts['version_conflicts']['mixed_versions'])
    
    total_conflicts = critical_conflicts + high_conflicts + medium_conflicts
    
    print(f"CRITICAL冲突 (重复方法): {critical_conflicts}")
    print(f"HIGH冲突 (配置不一致): {high_conflicts}")
    print(f"MEDIUM冲突 (功能重复): {medium_conflicts}")
    print(f"版本混合问题: {version_issues}")
    print(f"总冲突数: {total_conflicts}")
    
    if critical_conflicts > 0:
        risk_level = "CRITICAL"
        print(f"\n[CRITICAL] 存在严重功能冲突，需要立即解决!")
    elif high_conflicts > 3:
        risk_level = "HIGH"
        print(f"\n[HIGH] 存在配置冲突，需要优先解决")
    elif total_conflicts > 10:
        risk_level = "MEDIUM"
        print(f"\n[MEDIUM] 存在一些功能重复，建议优化")
    else:
        risk_level = "LOW"
        print(f"\n[GOOD] 功能相对统一，少量优化即可")
    
    # 修复建议
    print(f"\n=== 修复建议 ===")
    
    if critical_conflicts > 0:
        print("1. 立即合并或删除重复方法定义")
    
    if high_conflicts > 0:
        print("2. 统一配置参数值，消除不一致")
        print("3. 建立配置中心化管理")
    
    if medium_conflicts > 0:
        print("4. 重构重复功能，提取公共逻辑")
        print("5. 清理初始化冲突")
    
    if version_issues > 10:
        print("6. 清理旧版本代码，统一代码版本")
    
    return {
        'risk_level': risk_level,
        'critical_conflicts': critical_conflicts,
        'high_conflicts': high_conflicts,
        'medium_conflicts': medium_conflicts,
        'version_issues': version_issues,
        'total_conflicts': total_conflicts,
        'training_methods': len(training_methods),
        'prediction_methods': len(prediction_methods)
    }

if __name__ == "__main__":
    result = analyze_training_conflicts()
    
    if result['critical_conflicts'] > 0:
        print(f"\n[URGENT] 发现 {result['critical_conflicts']} 个严重冲突需要立即修复!")
    elif result['risk_level'] in ['HIGH', 'MEDIUM']:
        print(f"\n[ACTION NEEDED] 训练流程存在冲突，风险级别: {result['risk_level']}")
    else:
        print(f"\n[HEALTHY] 训练流程相对健康，冲突风险: {result['risk_level']}")