#!/usr/bin/env python3
"""
Final comprehensive syntax fix for BMA Ultra Enhanced
"""

import re

def final_syntax_fix():
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying final syntax fixes...")
    
    # Fix all broken method calls patterns
    fixes = [
        (r'\.select_dtypesinclude\s*=\s*\[([^\]]+)\]', r'.select_dtypes(include=[\1])'),
        (r'\.select_dtypesexclude\s*=\s*\[([^\]]+)\]', r'.select_dtypes(exclude=[\1])'),
        (r'\.fillnamethod\s*=\s*["\']([^"\']+)["\']', r'.fillna(method="\1")'),
        (r'\.fillnavalue\s*=\s*([^,\s]+)', r'.fillna(value=\1)'),
        (r'\.dropnaaxis\s*=\s*([01])', r'.dropna(axis=\1)'),
        (r'\.drop_duplicatessubset\s*=\s*\[([^\]]+)\]', r'.drop_duplicates(subset=[\1])'),
        (r'\.groupbyby\s*=\s*["\']([^"\']+)["\']', r'.groupby(by="\1")'),
        (r'\.sorton\s*=\s*["\']([^"\']+)["\']', r'.sort_values(by="\1")'),
        (r'\.reindexcolumns\s*=\s*([^,\s]+)', r'.reindex(columns=\1)'),
        (r'\.concataxis\s*=\s*([01])', r'.concat(axis=\1)'),
        (r'\.mergeon\s*=\s*["\']([^"\']+)["\']', r'.merge(on="\1")'),
        (r'\.pivottable\(([^)]+)\)values\s*=\s*["\']([^"\']+)["\']', r'.pivot_table(\1, values="\2")'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Final syntax fixes applied!")
    
    try:
        compile(content, file_path, 'exec')
        print("SUCCESS: File compiles correctly!")
        return True
    except SyntaxError as e:
        print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
        print(f"Context: {e.text.strip() if e.text else 'N/A'}")
        return False

if __name__ == "__main__":
    final_syntax_fix()