#!/usr/bin/env python3
"""
System Validation - Check system before trading
"""

import os
import sys
import re
from pathlib import Path

def validate_no_demo_code():
    """Check for demo code in critical files"""
    issues = []
    
    # Check key files only
    key_files = [
        'autotrader/app.py',
        'autotrader/engine.py', 
        'autotrader/ibkr_auto_trader.py'
    ]
    
    demo_patterns = [
        r'np\.random\.uniform',
        r'# Random signal for demo',
        r'signal_strength = np\.random'
    ]
    
    for file_path in key_files:
        file_obj = Path(file_path)
        if file_obj.exists():
            try:
                content = file_obj.read_text(encoding='utf-8')
                for pattern in demo_patterns:
                    if re.search(pattern, content):
                        issues.append(f"Demo code found in {file_path}")
            except Exception:
                pass
    
    return issues

def validate_environment():
    """Validate environment variables"""
    issues = []
    
    required_vars = ['TRADING_ACCOUNT_ID', 'SIGNAL_MODE']
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            issues.append(f"Missing environment variable: {var}")
        elif value in ['your_real_account_id', 'your_account_id_here']:
            issues.append(f"Environment variable {var} not set to real value")
    
    return issues

def main():
    """Run system validation"""
    print("=== Trading System Validation ===")
    
    all_issues = []
    
    # Check demo code
    demo_issues = validate_no_demo_code()
    all_issues.extend(demo_issues)
    
    # Check environment
    env_issues = validate_environment()
    all_issues.extend(env_issues)
    
    if all_issues:
        print("\nVALIDATION FAILED:")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\nSystem is NOT ready for production trading.")
        return 1
    else:
        print("\nVALIDATION PASSED: System appears ready.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
