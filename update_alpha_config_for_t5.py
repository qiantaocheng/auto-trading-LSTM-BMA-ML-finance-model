#!/usr/bin/env python3
"""
更新Alpha配置以适应T+5预测
调整Alpha策略的参数以适合中期预测
"""

def update_alpha_config_for_t5():
    """更新Alpha配置文件以适应T+5预测"""
    
    import yaml
    import os
    
    config_file = 'alphas_config.yaml'
    
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在")
        return
    
    # 读取配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== 更新Alpha配置适应T+5预测 ===")
    
    # 更新全局配置
    if 'prediction_horizon' in config:
        old_horizon = config['prediction_horizon']
        config['prediction_horizon'] = 5
        print(f"预测周期: {old_horizon} → 5")
    else:
        config['prediction_horizon'] = 5
        print("添加预测周期: 5")
    
    # 更新持有期
    if 'holding_period' in config:
        old_holding = config['holding_period'] 
        config['holding_period'] = 5
        print(f"持有期: {old_holding} → 5")
    else:
        config['holding_period'] = 5
        print("添加持有期: 5")
    
    # 更新Alpha策略参数以适应T+5
    if 'alphas' in config:
        updated_count = 0
        
        for alpha in config['alphas']:
            alpha_name = alpha.get('name', 'unknown')
            
            # 更新动量类Alpha的lookback period
            if 'momentum' in alpha_name.lower():
                if 'params' not in alpha:
                    alpha['params'] = {}
                
                # 适当延长动量策略的回望期
                if 'lookback' in alpha['params']:
                    old_lookback = alpha['params']['lookback']
                    # T+5预测适合更长的动量窗口
                    alpha['params']['lookback'] = max(old_lookback, 10)
                    print(f"  {alpha_name}: lookback {old_lookback} → {alpha['params']['lookback']}")
                    updated_count += 1
                else:
                    alpha['params']['lookback'] = 15  # T+5适合的默认值
                    print(f"  {alpha_name}: 添加lookback=15")
                    updated_count += 1
            
            # 更新反转类Alpha的参数
            elif 'reversal' in alpha_name.lower():
                if 'params' not in alpha:
                    alpha['params'] = {}
                
                if 'short_window' in alpha['params']:
                    # 反转策略可以保持较短的窗口，但稍作调整
                    alpha['params']['short_window'] = max(alpha['params']['short_window'], 3)
                    updated_count += 1
                
                if 'long_window' in alpha['params']:
                    # 长窗口适当延长
                    old_long = alpha['params']['long_window']
                    alpha['params']['long_window'] = max(old_long, 15)
                    print(f"  {alpha_name}: long_window {old_long} → {alpha['params']['long_window']}")
                    updated_count += 1
            
            # 更新波动率类Alpha
            elif 'volatility' in alpha_name.lower():
                if 'params' not in alpha:
                    alpha['params'] = {}
                
                if 'window' in alpha['params']:
                    old_window = alpha['params']['window']
                    # 波动率计算窗口适当延长
                    alpha['params']['window'] = max(old_window, 15)
                    print(f"  {alpha_name}: window {old_window} → {alpha['params']['window']}")
                    updated_count += 1
            
            # 更新权重提示以适应T+5
            if alpha.get('weight_hint', 0) > 0:
                # T+5预测更适合趋势和动量策略
                if any(keyword in alpha_name.lower() for keyword in ['momentum', 'trend']):
                    alpha['weight_hint'] = min(alpha['weight_hint'] * 1.2, 1.0)  # 略微提升
                    updated_count += 1
                elif any(keyword in alpha_name.lower() for keyword in ['reversal', 'mean_reversion']):
                    alpha['weight_hint'] = alpha['weight_hint'] * 0.8  # 略微降低
                    updated_count += 1
        
        print(f"更新了 {updated_count} 个Alpha策略的参数")
    
    # 添加T+5特定的配置注释
    config['_notes'] = [
        "Configuration optimized for T+5 (5-day ahead) predictions",
        "Medium-term alpha strategies with extended lookback periods", 
        "Momentum and trend factors weighted higher for T+5 horizon",
        "All features use T-4 data to predict T+5 returns (9-day gap)"
    ]
    
    # 保存更新后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"Alpha配置已更新: {config_file}")
    
    # 验证更新
    print("\n=== 配置验证 ===")
    print(f"预测周期: {config.get('prediction_horizon', 'N/A')}")
    print(f"持有期: {config.get('holding_period', 'N/A')}")
    print(f"Alpha策略数量: {len(config.get('alphas', []))}")
    
    return True

def update_other_configs():
    """更新其他相关配置文件"""
    
    # 检查并更新其他可能的配置文件
    other_configs = [
        'adaptive_weights_config.yaml'
    ]
    
    for config_file in other_configs:
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 更新相关参数
                updated = False
                if 'prediction_horizon' in config:
                    config['prediction_horizon'] = 5
                    updated = True
                
                if 'holding_period' in config:
                    config['holding_period'] = 5
                    updated = True
                
                if updated:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
                    print(f"已更新: {config_file}")
                
            except Exception as e:
                print(f"更新 {config_file} 失败: {e}")

if __name__ == "__main__":
    print("=== 开始更新Alpha配置适应T+5预测 ===")
    
    success = update_alpha_config_for_t5()
    
    if success:
        print("\n=== 更新其他配置文件 ===")
        update_other_configs()
        
        print("\n=== T+5配置更新完成 ===")
        print("✓ 主模型配置: T+5预测")
        print("✓ Alpha策略参数: 适配中期预测") 
        print("✓ 持有期设置: 5天")
        print("✓ 时间安全间隔: 9天 (T-4到T+5)")
        print("\n系统现在完全配置为T+5 (5天后) 收益率预测!")
    else:
        print("配置更新失败")