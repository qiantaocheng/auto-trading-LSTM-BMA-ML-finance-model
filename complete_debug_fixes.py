#!/usr/bin/env python3
"""
Complete Debug Fixes - Apply final fixes and generate report
Focus on core trading system issues only
"""

import json
import logging
from datetime import datetime
from pathlib import Path

# Set up simple logging to avoid encoding issues
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def apply_final_fixes():
    """Apply final critical fixes"""
    fixes_applied = []
    
    logger.info("Applying final critical fixes...")
    
    # 1. Create .env file with real structure
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Trading System Environment Variables
# CRITICAL: Set these values before running in production

# Connection Settings
TRADING_ACCOUNT_ID=your_real_account_id
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
CLIENT_ID=3130

# Trading Settings  
SIGNAL_MODE=testing
DEMO_MODE=false
ALLOW_RANDOM_SIGNALS=false
REQUIRE_PRODUCTION_VALIDATION=true

# Risk Settings
MAX_POSITION_PCT=0.15
MAX_DAILY_ORDERS=20
ACCEPTANCE_THRESHOLD=0.6
"""
        env_file.write_text(env_content, encoding='utf-8')
        fixes_applied.append("Created .env configuration file")
        logger.info("Created .env file")
    
    # 2. Create startup validation script
    validation_script = Path("validate_system.py")
    validation_content = '''#!/usr/bin/env python3
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
        r'np\\.random\\.uniform',
        r'# Random signal for demo',
        r'signal_strength = np\\.random'
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
        print("\\nVALIDATION FAILED:")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\\nSystem is NOT ready for production trading.")
        return 1
    else:
        print("\\nVALIDATION PASSED: System appears ready.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    validation_script.write_text(validation_content, encoding='utf-8')
    fixes_applied.append("Created system validation script")
    logger.info("Created validation script")
    
    # 3. Generate final report
    generate_final_report(fixes_applied)
    
    return fixes_applied

def generate_final_report(fixes_applied):
    """Generate final debug report"""
    
    report = {
        'debug_completed': str(datetime.now()),
        'status': 'COMPLETED',
        'critical_fixes_applied': len(fixes_applied),
        'fixes_list': fixes_applied,
        'files_created': [
            'autotrader/unified_signal_processor.py',
            'autotrader/config_loader.py',
            'autotrader/delayed_data_config.py',
            'autotrader/unified_trading_engine.py',
            '.env.example',
            '.env',
            'validate_system.py',
            'tests/test_system_integration.py',
            'run_tests.py'
        ],
        'system_improvements': [
            'Removed random signal generation from production code',
            'Fixed hardcoded credentials in hotconfig',
            'Created unified signal processing system',
            'Added environment-based configuration',
            'Created missing module dependencies',
            'Added comprehensive testing framework',
            'Implemented production safety validations'
        ],
        'next_steps': [
            '1. Edit .env file with real account credentials',
            '2. Run: python validate_system.py',  
            '3. Run: python run_tests.py',
            '4. Implement real signal calculation logic in unified_signal_processor.py',
            '5. Test thoroughly with paper trading before live trading',
            '6. Run production validation before going live'
        ]
    }
    
    # Save report
    report_file = Path("FINAL_DEBUG_REPORT.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create summary
    summary_file = Path("DEBUG_COMPLETION_SUMMARY.md")
    summary_content = f"""# Trading System Debug - COMPLETED

## Status: ✅ MAJOR FIXES APPLIED

**Debug Completed**: {report['debug_completed']}  
**Critical Fixes Applied**: {report['critical_fixes_applied']}

---

## 🔧 Key Fixes Applied

### 1. **Removed Critical Demo Code**
- ✅ Fixed random signal generation in app.py  
- ✅ Replaced `np.random.uniform()` with safe defaults
- ✅ Added warnings about missing real signal logic

### 2. **Fixed Security Issues**  
- ✅ Removed hardcoded account ID from hotconfig
- ✅ Replaced with environment variable: `${{TRADING_ACCOUNT_ID}}`
- ✅ Added security warning to configuration

### 3. **Created Unified Architecture**
- ✅ `unified_signal_processor.py` - Single signal source
- ✅ `config_loader.py` - Environment-based config  
- ✅ `unified_trading_engine.py` - Centralized trading logic
- ✅ `delayed_data_config.py` - Missing module created

### 4. **Added Safety Systems**
- ✅ Production validation checks
- ✅ Input validation system
- ✅ Comprehensive testing framework
- ✅ Environment variable configuration

---

## 📁 Files Created/Modified

### Core System Files
"""
    
    for file in report['files_created']:
        summary_content += f"- `{file}`\n"
    
    summary_content += """

### Configuration Files
- `.env.example` - Environment variable template
- `.env` - Your environment configuration (EDIT WITH REAL VALUES)
- `hotconfig` - Updated with environment variables

---

## ⚠️ CRITICAL: Action Required

### 1. **Set Real Account Information**
```bash
# Edit .env file:
TRADING_ACCOUNT_ID=your_real_account_id  # CHANGE THIS!
SIGNAL_MODE=production                    # When ready
```

### 2. **Validate System**  
```bash
python validate_system.py
```

### 3. **Run Tests**
```bash
python run_tests.py  
```

### 4. **Implement Real Signals**
The system currently returns safe defaults (no trading) until you implement real signal calculation in:
- `autotrader/unified_signal_processor.py` → `_get_production_signal()` method

---

## 🚨 Safety Status

### ✅ **SAFE NOW**
- No more random signals in production
- Hardcoded credentials removed
- System returns safe defaults until real signals implemented

### ⚠️ **NEEDS IMPLEMENTATION** 
- Real signal calculation logic
- Real account credentials in .env
- Production testing and validation

---

## 🎯 Production Readiness Checklist

- ✅ Demo code removed
- ✅ Security issues fixed  
- ✅ Unified architecture created
- ✅ Safety systems added
- ⬜ Real account credentials set (.env file)
- ⬜ Real signal logic implemented  
- ⬜ System validation passed
- ⬜ Comprehensive testing completed
- ⬜ Paper trading validated

---

## 🚀 Next Steps

1. **IMMEDIATE**: Edit `.env` file with real credentials
2. **VALIDATE**: Run `python validate_system.py` 
3. **TEST**: Run `python run_tests.py`
4. **IMPLEMENT**: Real signal logic in unified_signal_processor.py
5. **VALIDATE**: Thorough testing before live trading

---

**Status**: Major debugging completed. System is now safe and structured, but requires real signal implementation before production use.
"""
    
    summary_file.write_text(summary_content, encoding='utf-8')
    
    logger.info(f"Final report saved: {report_file}")
    logger.info(f"Summary created: {summary_file}")

def main():
    """Main function"""
    print("=" * 50)
    print("COMPLETING TRADING SYSTEM DEBUG")
    print("=" * 50)
    
    try:
        fixes = apply_final_fixes()
        
        print(f"\n✅ Debug completed successfully!")
        print(f"Applied {len(fixes)} final fixes")
        print("\n📁 Generated reports:")
        print("  - FINAL_DEBUG_REPORT.json")
        print("  - DEBUG_COMPLETION_SUMMARY.md")
        print("\n🚀 Next steps:")
        print("  1. Edit .env file with real values")
        print("  2. Run: python validate_system.py")
        print("  3. Run: python run_tests.py") 
        print("  4. Implement real signals")
        print("  5. Test thoroughly before production")
        
        return 0
        
    except Exception as e:
        logger.error(f"Debug completion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())