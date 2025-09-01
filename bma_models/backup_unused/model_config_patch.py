#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主模型配置补丁
确保使用统一配置
"""

def patch_model_init(model_instance):
    """为模型实例打补丁以使用统一配置"""
    from bma_models.config_loader import load_unified_config
    
    # 加载统一配置
    config = load_unified_config()
    
    # 更新模型配置
    if hasattr(model_instance, 'config'):
        # 合并配置
        for key, value in config.items():
            if isinstance(value, dict):
                if key not in model_instance.config:
                    model_instance.config[key] = {}
                model_instance.config[key].update(value)
            else:
                model_instance.config[key] = value
    
    # 确保所有模块都启用
    if hasattr(model_instance, 'module_manager'):
        # 强制启用所有高级特性
        modules_to_enable = ['ltr_ranking', 'stacking', 'regime_aware']
        for module in modules_to_enable:
            if hasattr(model_instance.module_manager, 'force_enable'):
                model_instance.module_manager.force_enable(module)
    
    return model_instance
