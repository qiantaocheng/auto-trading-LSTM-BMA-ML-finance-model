#!/usr/bin/env python3
"""
修复关键语法错误的最小化脚本
"""

import re
import subprocess
import sys

def fix_critical_syntax_errors():
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("修复关键语法错误...")
    
    # 关键修复
    critical_fixes = [
        # 修复缺失的冒号
        (r'if index_manager\.is_standard_index\(df\)\s*#.*?:', 
         "if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ['date', 'ticker']:"),
        
        # 修复不完整的方法调用
        (r'\.pipe\(df_optimizer\.efficient_fillna\)\s*#[^)]*\)\.', '.fillna(0).'),
        
        # 修复空的类定义
        (r'class (\w+):\s*\n\s*class', r'class \1:\n        pass\n\n    class'),
        
        # 修复空的函数定义
        (r'(def \w+\([^)]*\):)\s*\n\s*([A-Za-z])', r'\1\n        pass\n\n    \2'),
        
        # 修复格式错误的注释行
        (r"batch_results\[calibrated_results[^']*'predictions'\]", "calibrated_results['predictions']"),
    ]
    
    for pattern, replacement in critical_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if content != old_content:
            print(f"应用修复: {pattern[:50]}...")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 检查语法
    result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS: 语法错误已修复!")
        return True
    else:
        print(f"仍有语法错误: {result.stderr[:200]}...")
        return False

if __name__ == "__main__":
    fix_critical_syntax_errors()