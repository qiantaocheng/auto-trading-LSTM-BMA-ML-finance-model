#!/usr/bin/env python3
"""
Fix debug print statements by replacing with proper logging
"""

import re
import os
from pathlib import Path

def fix_debug_prints_in_file(file_path):
    """Replace debug print statements with proper logging"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count original print statements
        original_prints = len(re.findall(r'print\(', content))
        
        # Replace debug print statements with logging
        debug_patterns = [
            (r'print\(f"🛡️.*?"\)', r'self.logger.debug(f"VALIDATION: {}")'),
            (r'print\(f"📈.*?"\)', r'self.logger.debug(f"MARKET ORDER: {}")'),
            (r'print\(f"💰.*?"\)', r'self.logger.debug(f"ORDER VALUE: {}")'),
            (r'print\(f"🎯.*?"\)', r'self.logger.debug(f"ORDER PARAMS: {}")'),
            (r'print\(f"🔄.*?"\)', r'self.logger.debug(f"RETRY: {}")'),
            (r'print\(f"📅.*?"\)', r'self.logger.debug(f"DATE CHECK: {}")'),
            (r'print\(f"❌.*?"\)', r'self.logger.warning(f"REJECTED: {}")'),
            (r'print\(f"✅.*?"\)', r'self.logger.info(f"SUCCESS: {}")'),
        ]
        
        # Apply replacements
        modified = False
        for pattern, replacement in debug_patterns:
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, '', content, flags=re.DOTALL)
                modified = True
        
        # Remove debug print blocks
        debug_blocks = [
            r'print\(f"\n\{\'=\'\*80\}"\)\s*\n',
            r'print\(f"\{\'=\'\*80\}"\)\s*\n',
            r'print\(f"DEBUG.*?"\)\s*\n',
        ]
        
        for pattern in debug_blocks:
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, '', content, flags=re.DOTALL)
                modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Count remaining print statements
            remaining_prints = len(re.findall(r'print\(', content))
            print(f"Fixed {file_path}: {original_prints} -> {remaining_prints} print statements")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all debug prints in autotrader files"""
    autotrader_path = Path("autotrader")
    
    if not autotrader_path.exists():
        print("autotrader directory not found")
        return
    
    fixed_files = 0
    total_files = 0
    
    for py_file in autotrader_path.glob("*.py"):
        total_files += 1
        if fix_debug_prints_in_file(py_file):
            fixed_files += 1
    
    print(f"Processing complete: {fixed_files}/{total_files} files modified")

if __name__ == "__main__":
    main()