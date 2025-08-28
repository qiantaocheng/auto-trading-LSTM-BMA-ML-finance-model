#!/usr/bin/env python3
"""
修复BMA Enhanced系统中的health_metrics属性访问问题
"""

def fix_health_metrics_access():
    """修复health_metrics属性访问问题"""
    
    bma_file = 'bma_models/量化模型_bma_ultra_enhanced.py'
    
    # 读取原始文件
    with open(bma_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复：在所有访问health_metrics之前添加安全检查
    old_health_access = '''        total_operations = sum(self.health_metrics.values())
        failure_rate = (self.health_metrics['total_exceptions'] / max(total_operations, 1)) * 100
        
        report = {
            'health_metrics': self.health_metrics.copy(),'''
    
    new_health_access = '''        # 确保health_metrics已初始化
        if not hasattr(self, 'health_metrics'):
            self.health_metrics = {
                'universe_load_fallbacks': 0,
                'risk_model_failures': 0,
                'optimization_fallbacks': 0,
                'alpha_computation_failures': 0,
                'neutralization_failures': 0,
                'prediction_failures': 0,
                'total_exceptions': 0,
                'successful_predictions': 0
            }
        
        total_operations = sum(self.health_metrics.values())
        failure_rate = (self.health_metrics['total_exceptions'] / max(total_operations, 1)) * 100
        
        report = {
            'health_metrics': self.health_metrics.copy(),'''
    
    # 应用修复
    if old_health_access in content:
        content = content.replace(old_health_access, new_health_access)
        print("[OK] 修复health_metrics访问安全检查")
    else:
        print("[SKIP] 未找到health_metrics访问代码")
    
    # 额外保护：在所有health_metrics访问处添加安全检查
    health_patterns = [
        "self.health_metrics['",
        "self.health_metrics.values()",
        "self.health_metrics.copy()"
    ]
    
    for pattern in health_patterns:
        if pattern in content and "hasattr(self, 'health_metrics')" not in content.split(pattern)[0].split('\n')[-10:]:
            # 这个模式存在但前10行没有安全检查，添加保护
            parts = content.split(pattern)
            if len(parts) > 1:
                # 在第一次出现前添加检查
                before_first = parts[0]
                after_first = pattern.join(parts[1:])
                
                # 找到包含这个访问的函数或方法
                lines_before = before_first.split('\n')
                # 添加安全初始化到最后一个函数/方法的开始
                for i in range(len(lines_before) - 1, -1, -1):
                    line = lines_before[i].strip()
                    if line.startswith('def ') and 'health_metrics' not in line:
                        # 在这个函数开始后添加安全检查
                        insertion_point = len('\n'.join(lines_before[:i+1])) + 1
                        safe_check = '''
        # 确保health_metrics已初始化
        if not hasattr(self, 'health_metrics'):
            self.health_metrics = {
                'universe_load_fallbacks': 0,
                'risk_model_failures': 0, 
                'optimization_fallbacks': 0,
                'alpha_computation_failures': 0,
                'neutralization_failures': 0,
                'prediction_failures': 0,
                'total_exceptions': 0,
                'successful_predictions': 0
            }'''
                        content = content[:insertion_point] + safe_check + content[insertion_point:]
                        print(f"[OK] 在函数中添加health_metrics安全检查: {line[:30]}...")
                        break
    
    # 保存修复后的文件
    with open(bma_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"health_metrics访问修复完成: {bma_file}")

if __name__ == "__main__":
    print("=== 开始修复health_metrics访问问题 ===")
    fix_health_metrics_access() 
    print("=== health_metrics修复完成 ===")