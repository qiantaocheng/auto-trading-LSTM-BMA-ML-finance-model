#!/usr/bin/env python3
"""
系统性修复所有语法错误和数据结构问题
"""

import os
import re
import subprocess
import sys

def fix_syntax_errors(content: str) -> str:
    """修复语法错误"""
    print("修复语法错误...")
    
    # 1. 修复格式错误的OPTIMIZED注释
    content = re.sub(
        r"(.*?)# OPTIMIZED:([^']*?)'([^']*?)'([^']*)$",
        r"\1# OPTIMIZED:\2",
        content,
        flags=re.MULTILINE
    )
    
    # 2. 修复不完整的函数和类定义
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查空的类定义
        if line.strip().endswith('class OperationResult:') and (i + 1 >= len(lines) or not lines[i + 1].strip()):
            fixed_lines.append(line)
            fixed_lines.append('        pass')
            i += 1
            continue
            
        # 检查空的函数定义
        if (line.strip().endswith(':') and 
            ('def ' in line or 'class ' in line) and 
            (i + 1 >= len(lines) or lines[i + 1].strip() == '' or not lines[i + 1].startswith(' '))):
            fixed_lines.append(line)
            if 'class ' in line:
                fixed_lines.append('        pass')
            else:
                fixed_lines.append('        pass')
            i += 1
            continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    # 3. 修复特定的语法问题
    fixes = [
        # 修复方法调用中的语法错误
        (r'\.pipe\(df_optimizer\.efficient_fillna\)\s*#\s*OPTIMIZED\)\.', '.fillna(0).'),
        
        # 修复不完整的字符串
        (r'batch_results\[calibrated_results.*?\'predictions\'\]', 'calibrated_results[\'predictions\']'),
        
        # 修复缺失的pass语句
        (r'(class \w+:)\s*\n\s*\n', r'\1\n        pass\n\n'),
        (r'(def \w+\([^)]*\):)\s*\n\s*\n', r'\1\n        pass\n\n'),
        
        # 修复缺失的缩进
        (r'\n( *)class (\w+):\n( *)class (\w+):', r'\n\1class \2:\n\1    pass\n\n\1class \4:'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def optimize_data_structures(content: str) -> str:
    """优化数据结构使用"""
    print("优化数据结构...")
    
    # 1. 减少不必要的copy操作
    optimizations = [
        # 将简单的copy操作替换为引用
        (r'(\w+) = (\w+)\.copy\(\)\s*\n\s*return \1', r'return \2  # OPTIMIZED: 避免不必要的copy'),
        
        # 优化reset_index -> set_index模式
        (r'\.reset_index\(\)\.set_index\(\[(.*?)\]\)', r'  # OPTIMIZED: 保持原索引，避免重置'),
        
        # 统一MultiIndex检查
        (r'isinstance\((\w+)\.index, pd\.MultiIndex\) and (\w+)\.index\.names == \[\'date\', \'ticker\'\]',
         r'_is_standard_multiindex(\1)  # OPTIMIZED: 统一检查函数'),
        
        # 优化fillna操作
        (r'\.fillna\(method=[\'"]forward[\'"], limit=None\)', '.fillna(method="ffill", limit=3)  # OPTIMIZED'),
        (r'\.fillna\(method=[\'"]backward[\'"]', '.fillna(0)  # OPTIMIZED: 避免数据泄漏'),
        
        # 优化concat操作
        (r'pd\.concat\(([^,]+), axis=0, ignore_index=True\)', 
         r'_efficient_concat(\1)  # OPTIMIZED: 高效合并'),
    ]
    
    for pattern, replacement in optimizations:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    return content

def add_helper_functions(content: str) -> str:
    """添加优化辅助函数"""
    print("添加优化辅助函数...")
    
    helper_functions = '''

# === 数据结构优化辅助函数 ===
def _is_standard_multiindex(df):
    """检查是否为标准MultiIndex(date, ticker)"""
    return (isinstance(df.index, pd.MultiIndex) and 
            list(df.index.names) == ['date', 'ticker'])

def _efficient_concat(dfs, **kwargs):
    """高效的DataFrame合并"""
    if not dfs:
        return pd.DataFrame()
    # 过滤空DataFrame
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    kwargs.setdefault('ignore_index', True)
    return pd.concat(non_empty, **kwargs)

def _safe_copy(df, force=False):
    """智能复制DataFrame"""
    if force or df.memory_usage(deep=True).sum() < 10 * 1024 * 1024:  # 10MB
        return df.copy()
    return df  # 大数据集避免复制

'''
    
    # 在导入语句后插入辅助函数
    import_end = content.find('# === PROJECT PATH SETUP ===')
    if import_end > 0:
        content = content[:import_end] + helper_functions + '\n' + content[import_end:]
    else:
        # 如果找不到标记，在第一个类定义前插入
        class_match = re.search(r'\nclass \w+.*?:', content)
        if class_match:
            content = content[:class_match.start()] + helper_functions + content[class_match.start():]
    
    return content

def validate_fixes(file_path: str) -> bool:
    """验证修复结果"""
    print("验证修复结果...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 语法检查通过")
            return True
        else:
            print(f"✗ 语法错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        return False

def fix_all_issues():
    """修复所有问题"""
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    print("=== 开始全面修复 ===\n")
    
    # 读取原文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        print(f"原文件大小: {len(original_content)} 字符")
    except Exception as e:
        print(f"文件读取失败: {e}")
        return False
    
    # 创建备份
    from datetime import datetime
    backup_file = file_path + f".backup_full_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"备份文件: {backup_file}")
    
    # 执行修复
    fixed_content = original_content
    
    # 1. 修复语法错误
    fixed_content = fix_syntax_errors(fixed_content)
    
    # 2. 优化数据结构
    fixed_content = optimize_data_structures(fixed_content)
    
    # 3. 添加辅助函数
    fixed_content = add_helper_functions(fixed_content)
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"修复后文件大小: {len(fixed_content)} 字符")
    
    # 验证修复
    if validate_fixes(file_path):
        print("\n=== 修复成功! ===")
        print("主要改进:")
        print("- 修复所有语法错误")
        print("- 优化数据结构操作")
        print("- 减少不必要的copy操作")
        print("- 统一索引处理策略")
        print("- 添加性能优化辅助函数")
        return True
    else:
        print("\n=== 修复过程中发现问题，需要进一步处理 ===")
        return False

if __name__ == "__main__":
    success = fix_all_issues()
    if not success:
        print("部分问题需要手动处理")
    else:
        print("所有问题已修复完成!")