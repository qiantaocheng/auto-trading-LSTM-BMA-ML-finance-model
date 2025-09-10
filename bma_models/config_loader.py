#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置加载器
"""

import yaml
import os
from pathlib import Path

def load_unified_config(config_path=None):
    """加载统一配置"""
    if config_path is None:
        config_path = Path(__file__).parent / "unified_config.yaml"
    
    if not os.path.exists(config_path):
        # 返回默认配置
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_default_config():
    """获取默认配置"""
    return {
        'temporal': {
            'prediction_horizon_days': 10,
            'feature_lag_days': 5,
            'cv_gap_days': 1,
            'cv_embargo_days': 1
        },
        'training': {
            'traditional_models': {'enable': True},
            'ltr': {'enable': True},
            'stacking': {'enable': True},
            'regime_aware': {'enable': True}
        },
        'evaluation': {
            'ic_calculation': {
                'use_rank_ic': True,
                'min_samples': 5
            }
        }
    }

class ConfigLoader:
    """统一配置加载器类"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = None
        
    def load(self):
        """加载配置"""
        self.config = load_unified_config(self.config_path)
        return self.config
        
    def get_config(self):
        """获取配置"""
        if self.config is None:
            self.load()
        return self.config
        
    def get_section(self, section_name):
        """获取配置段"""
        config = self.get_config()
        return config.get(section_name, {})
        
    def reload(self):
        """重新加载配置"""
        self.config = None
        return self.load()
