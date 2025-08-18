#!/usr/bin/env python3
"""
紧急修复脚本 - 修复交易系统中的关键逻辑错误
这些修复必须在生产交易前完成，以避免财务损失
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmergencyFixer:
    """紧急修复交易系统关键问题"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        self.issues_found = []
    
    def scan_for_demo_code(self) -> list:
        """扫描演示代码"""
        demo_patterns = [
            r'np\.random\.uniform',
            r'random\.uniform',
            r'# Random signal for demo',
            r'# For demo purposes',
            r'signal_strength = np\.random',
            r'confidence = np\.random'
        ]
        
        issues = []
        for py_file in self.project_root.glob("**/*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in demo_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            'file': str(py_file),
                            'pattern': pattern,
                            'matches': len(matches),
                            'severity': 'CRITICAL'
                        })
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        return issues
    
    def scan_for_hardcoded_values(self) -> list:
        """扫描硬编码值"""
        hardcode_patterns = [
            r'"c2dvdongg"',
            r"'c2dvdongg'",
            r'"127\.0\.0\.1"',
            r"'127\.0\.0\.1'",
            r'client_id.*=.*3130',
            r'port.*=.*7497',
            r'account_id.*=.*"[^"]*"'
        ]
        
        issues = []
        for file in self.project_root.glob("**/*"):
            if file.suffix in ['.py', '.json', ''] and file.is_file():
                try:
                    content = file.read_text(encoding='utf-8')
                    for pattern in hardcode_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            issues.append({
                                'file': str(file),
                                'pattern': pattern,
                                'matches': len(matches),
                                'severity': 'HIGH'
                            })
                except Exception as e:
                    logger.warning(f"Could not scan {file}: {e}")
        
        return issues
    
    def fix_demo_code_in_app(self):
        """修复app.py中的演示代码"""
        app_file = self.project_root / "autotrader" / "app.py"
        if not app_file.exists():
            logger.warning("app.py not found")
            return
        
        try:
            content = app_file.read_text(encoding='utf-8')
            
            # 替换随机信号生成
            old_demo_code = '''signal_strength = np.random.uniform(-0.1, 0.1)  # Random signal for demo
                confidence = np.random.uniform(0.5, 0.9)'''
            
            new_production_code = '''# 使用真实信号计算，移除演示代码
                try:
                    # 从统一因子管理器获取真实信号
                    from .unified_polygon_factors import get_unified_polygon_factors
                    factors = get_unified_polygon_factors()
                    signal_result = factors.get_trading_signal(symbol)
                    signal_strength = signal_result.get('signal_strength', 0.0)
                    confidence = signal_result.get('confidence', 0.0)
                except Exception as e:
                    logger.error(f"Failed to get real signal for {symbol}: {e}")
                    continue  # 跳过无法获取信号的股票'''
            
            if old_demo_code.replace(' ', '').replace('\n', '') in content.replace(' ', '').replace('\n', ''):
                content = content.replace(
                    'signal_strength = np.random.uniform(-0.1, 0.1)  # Random signal for demo',
                    '# 移除演示代码 - 使用真实信号\n                signal_strength = 0.0  # 默认值，需要真实计算'
                )
                content = content.replace(
                    'confidence = np.random.uniform(0.5, 0.9)',
                    'confidence = 0.0  # 默认值，需要真实计算'
                )
                
                # 添加警告注释
                warning_comment = '''
# ⚠️  CRITICAL FIX APPLIED: 
# 演示代码已移除。必须实现真实的信号计算逻辑。
# 当前使用默认值0.0，这将阻止所有交易，直到实现真实信号。
'''
                content = warning_comment + content
                
                app_file.write_text(content, encoding='utf-8')
                self.fixes_applied.append("移除app.py中的随机信号生成代码")
                logger.info("✅ 修复app.py中的演示代码")
            
        except Exception as e:
            logger.error(f"修复app.py失败: {e}")
    
    def create_environment_config(self):
        """创建环境变量配置示例"""
        env_example = self.project_root / ".env.example"
        env_content = """# 交易系统环境变量配置
# 复制此文件为 .env 并填入真实值

# IBKR连接配置
TRADING_ACCOUNT_ID=your_account_id_here
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
CLIENT_ID=3130

# 交易参数
MAX_ORDER_SIZE=1000000
DEFAULT_STOP_LOSS=0.02
DEFAULT_TAKE_PROFIT=0.05
ACCEPTANCE_THRESHOLD=0.6

# 系统配置
DEMO_MODE=false
LOG_LEVEL=INFO
DATA_SOURCE=polygon

# 安全设置
REQUIRE_PRODUCTION_VALIDATION=true
ALLOW_RANDOM_SIGNALS=false
"""
        
        env_example.write_text(env_content, encoding='utf-8')
        self.fixes_applied.append("创建环境变量配置示例")
        logger.info("✅ 创建环境变量配置示例")
    
    def create_production_validator(self):
        """创建生产环境验证器"""
        validator_file = self.project_root / "autotrader" / "production_validator.py"
        validator_content = '''#!/usr/bin/env python3
"""
生产环境验证器 - 确保系统可以安全用于实际交易
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

class ProductionValidationError(Exception):
    """生产环境验证失败异常"""
    pass

class ProductionValidator:
    """生产环境验证器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []
    
    def validate_all(self) -> bool:
        """执行所有验证检查"""
        self.issues = []
        
        self.check_no_demo_code()
        self.check_no_hardcoded_credentials()
        self.check_required_env_vars()
        self.check_no_random_signals()
        
        if self.issues:
            error_msg = "\\n".join([f"❌ {issue}" for issue in self.issues])
            raise ProductionValidationError(f"生产环境验证失败:\\n{error_msg}")
        
        logger.info("✅ 生产环境验证通过")
        return True
    
    def check_no_demo_code(self):
        """检查没有演示代码"""
        demo_patterns = [
            r'# Random signal for demo',
            r'# For demo purposes',
            r'np\\.random\\.',
            r'random\\.uniform'
        ]
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in demo_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.issues.append(f"发现演示代码: {py_file} ({pattern})")
            except Exception:
                pass
    
    def check_no_hardcoded_credentials(self):
        """检查没有硬编码的凭据"""
        hardcode_patterns = [
            r'"c2dvdongg"',
            r"'c2dvdongg'",
        ]
        
        for file in self.project_root.glob("**/*"):
            if file.suffix in ['.py', '.json', ''] and file.is_file():
                try:
                    content = file.read_text(encoding='utf-8')
                    for pattern in hardcode_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self.issues.append(f"发现硬编码凭据: {file}")
                except Exception:
                    pass
    
    def check_required_env_vars(self):
        """检查必需的环境变量"""
        required_vars = [
            'TRADING_ACCOUNT_ID',
            'IBKR_HOST',
            'IBKR_PORT'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                self.issues.append(f"缺少环境变量: {var}")
    
    def check_no_random_signals(self):
        """检查没有随机信号生成"""
        if os.getenv('ALLOW_RANDOM_SIGNALS', 'false').lower() == 'true':
            return  # 明确允许随机信号（测试环境）
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if 'np.random' in content and 'signal' in content.lower():
                    self.issues.append(f"发现随机信号生成: {py_file}")
            except Exception:
                pass

def validate_production_environment():
    """验证生产环境的便捷函数"""
    validator = ProductionValidator()
    return validator.validate_all()

if __name__ == "__main__":
    try:
        validate_production_environment()
        print("✅ 生产环境验证通过")
    except ProductionValidationError as e:
        print(f"❌ 生产环境验证失败:\\n{e}")
        exit(1)
'''
        
        validator_file.write_text(validator_content, encoding='utf-8')
        self.fixes_applied.append("创建生产环境验证器")
        logger.info("✅ 创建生产环境验证器")
    
    def update_hotconfig_security(self):
        """更新hotconfig安全性"""
        hotconfig_file = self.project_root / "hotconfig"
        if not hotconfig_file.exists():
            logger.warning("hotconfig文件不存在")
            return
        
        try:
            # 读取配置
            with open(hotconfig_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 检查是否有硬编码账户ID
            if config.get('connection', {}).get('account_id') == 'c2dvdongg':
                # 替换为环境变量引用
                config['connection']['account_id'] = '${TRADING_ACCOUNT_ID}'
                
                # 添加警告注释
                config['_WARNING'] = 'SECURITY: 账户ID已移至环境变量。请设置TRADING_ACCOUNT_ID环境变量。'
                
                # 保存更新的配置
                with open(hotconfig_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.fixes_applied.append("移除hotconfig中的硬编码账户ID")
                logger.info("✅ 更新hotconfig安全性")
        
        except Exception as e:
            logger.error(f"更新hotconfig失败: {e}")
    
    def run_all_fixes(self):
        """运行所有紧急修复"""
        logger.info("🚨 开始紧急修复...")
        
        # 1. 扫描问题
        demo_issues = self.scan_for_demo_code()
        hardcode_issues = self.scan_for_hardcoded_values()
        
        logger.info(f"发现 {len(demo_issues)} 个演示代码问题")
        logger.info(f"发现 {len(hardcode_issues)} 个硬编码问题")
        
        # 2. 应用修复
        self.fix_demo_code_in_app()
        self.create_environment_config()
        self.create_production_validator()
        self.update_hotconfig_security()
        
        # 3. 生成报告
        report = {
            'demo_issues_found': len(demo_issues),
            'hardcode_issues_found': len(hardcode_issues),
            'fixes_applied': self.fixes_applied,
            'demo_issues': demo_issues,
            'hardcode_issues': hardcode_issues
        }
        
        report_file = self.project_root / "emergency_fix_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 紧急修复完成。应用了 {len(self.fixes_applied)} 个修复。")
        logger.info(f"📋 详细报告: {report_file}")
        
        return report

def main():
    """主函数"""
    fixer = EmergencyFixer()
    
    print("🚨 交易系统紧急修复工具")
    print("=" * 50)
    print("⚠️  警告: 在运行此修复前，请备份您的代码！")
    print("=" * 50)
    
    user_input = input("确认运行紧急修复? (y/N): ")
    if user_input.lower() != 'y':
        print("❌ 修复已取消")
        return
    
    try:
        report = fixer.run_all_fixes()
        
        print("\\n🎯 修复摘要:")
        print(f"  - 发现演示代码问题: {report['demo_issues_found']}")
        print(f"  - 发现硬编码问题: {report['hardcode_issues_found']}")
        print(f"  - 应用修复: {len(report['fixes_applied'])}")
        
        print("\\n✅ 应用的修复:")
        for fix in report['fixes_applied']:
            print(f"  ✓ {fix}")
        
        print("\\n⚠️  下一步:")
        print("  1. 设置环境变量 (.env 文件)")
        print("  2. 运行生产环境验证: python autotrader/production_validator.py")
        print("  3. 实现真实的信号计算逻辑")
        print("  4. 进行充分测试后再投入生产")
        
    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())