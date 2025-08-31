#!/usr/bin/env python3
"""
统一配置管理器
整合所有分散的配置文件，提供一致的配置接口
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TemporalConfig:
    """时间配置"""
    prediction_horizon: int = 10
    holding_period: int = 10
    feature_lag: int = 1
    isolation_days: int = 10
    embargo_days: int = 10
    cv_gap: int = 10
    min_absolute_gap: int = 12

@dataclass 
class ModelConfig:
    """模型配置"""
    learning_to_rank: bool = True
    regime_detection: bool = True
    uncertainty_aware: bool = False
    quantile_regression: bool = False
    ranking_objective: str = "rank:pairwise"
    beta_calculation_window: int = 252

@dataclass
class RiskConfig:
    """风险管理配置"""
    max_position: float = 0.03
    max_turnover: float = 0.1
    max_country_exposure: float = 0.2
    max_sector_exposure: float = 0.15
    min_liquidity_rank: float = 0.3
    risk_aversion: float = 3.0
    turnover_penalty: float = 1.0

@dataclass
class ProductionConfig:
    """生产环境配置"""
    min_ic_improvement: float = 0.01
    min_absolute_ic: float = 0.02
    max_qlike_degradation: float = 0.1
    min_stability_score: float = 0.5
    enable_or_logic: bool = True
    max_memory_gb: float = 8.0

@dataclass
class SystemConfig:
    """系统配置"""
    cache_dir: str = "./cache/bma_unified"
    log_level: str = "INFO"
    enable_regime_awareness: bool = True
    enable_production_gates: bool = True
    enable_incremental_training: bool = True

class UnifiedConfigManager:
    """统一配置管理器"""
    
    def __init__(self, 
                 config_dir: str = "D:\\trade",
                 environment: str = "development"):
        
        self.config_dir = Path(config_dir)
        self.environment = environment
        
        # 配置组件
        self.temporal = TemporalConfig()
        self.model = ModelConfig()
        self.risk = RiskConfig()
        self.production = ProductionConfig()
        self.system = SystemConfig()
        
        # 加载所有配置
        self._load_all_configs()
        
        logger.info(f"统一配置管理器初始化完成 - 环境: {environment}")
    
    def _load_all_configs(self) -> None:
        """加载所有配置文件"""
        try:
            # 加载YAML配置
            self._load_alphas_config()
            
            # 加载JSON配置
            self._load_trading_config()
            
            # 加载T10配置
            self._load_t10_config()
            
            # 验证配置一致性
            self._validate_config_consistency()
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._use_default_configs()
    
    def _load_alphas_config(self) -> None:
        """加载alphas配置"""
        alphas_config_path = self.config_dir / "alphas_config.yaml"
        
        if alphas_config_path.exists():
            try:
                with open(alphas_config_path, 'r', encoding='utf-8') as f:
                    alphas_config = yaml.safe_load(f)
                
                # 提取相关配置
                if 'feature_global_lag' in alphas_config:
                    self.temporal.feature_lag = alphas_config['feature_global_lag']
                
                if 'holding_period' in alphas_config:
                    self.temporal.holding_period = alphas_config['holding_period']
                    self.temporal.prediction_horizon = alphas_config['holding_period']
                
                if 'max_position' in alphas_config:
                    self.risk.max_position = alphas_config['max_position']
                
                if 'max_turnover' in alphas_config:
                    self.risk.max_turnover = alphas_config['max_turnover']
                
                if 'model_config' in alphas_config:
                    model_cfg = alphas_config['model_config']
                    self.model.learning_to_rank = model_cfg.get('learning_to_rank', True)
                    self.model.regime_detection = model_cfg.get('regime_detection', True)
                
                if 'risk_config' in alphas_config:
                    risk_cfg = alphas_config['risk_config']
                    self.risk.max_country_exposure = risk_cfg.get('max_country_exposure', 0.2)
                    self.risk.max_sector_exposure = risk_cfg.get('max_sector_exposure', 0.15)
                
                logger.info("✅ alphas_config.yaml 加载成功")
                
            except Exception as e:
                logger.error(f"alphas_config.yaml 加载失败: {e}")
    
    def _load_trading_config(self) -> None:
        """加载交易配置"""
        trading_config_path = self.config_dir / "autotrader_unified_config.json"
        
        if trading_config_path.exists():
            try:
                with open(trading_config_path, 'r', encoding='utf-8') as f:
                    trading_config = json.load(f)
                
                # 提取系统配置
                if 'log_level' in trading_config:
                    self.system.log_level = trading_config['log_level']
                
                logger.info("✅ autotrader_unified_config.json 加载成功")
                
            except Exception as e:
                logger.error(f"autotrader_unified_config.json 加载失败: {e}")
    
    def _load_t10_config(self) -> None:
        """加载T10配置"""
        try:
            # 动态导入T10配置
            import sys
            sys.path.append(str(self.config_dir / "bma_models"))
            
            from t10_config import T10_CONFIG
            
            self.temporal.prediction_horizon = T10_CONFIG.PREDICTION_HORIZON
            self.temporal.holding_period = T10_CONFIG.HOLDING_PERIOD
            self.temporal.feature_lag = T10_CONFIG.FEATURE_LAG
            self.temporal.isolation_days = T10_CONFIG.ISOLATION_DAYS
            self.temporal.embargo_days = T10_CONFIG.EMBARGO_DAYS
            self.temporal.cv_gap = T10_CONFIG.CV_GAP
            
            logger.info("✅ t10_config.py 加载成功")
            
        except Exception as e:
            logger.error(f"t10_config.py 加载失败: {e}")
    
    def _validate_config_consistency(self) -> None:
        """验证配置一致性"""
        issues = []
        
        # 时间配置一致性检查
        if self.temporal.holding_period != self.temporal.prediction_horizon:
            issues.append(f"持仓期({self.temporal.holding_period}) != 预测期({self.temporal.prediction_horizon})")
        
        if self.temporal.isolation_days < self.temporal.prediction_horizon:
            issues.append(f"隔离期({self.temporal.isolation_days}) < 预测期({self.temporal.prediction_horizon})")
        
        if self.temporal.cv_gap != self.temporal.isolation_days:
            logger.warning(f"CV间隔({self.temporal.cv_gap}) != 隔离期({self.temporal.isolation_days})")
            # 自动修复
            self.temporal.cv_gap = self.temporal.isolation_days
        
        # 风险配置检查
        if self.risk.max_position > 0.05:
            issues.append(f"最大仓位({self.risk.max_position}) 可能过高")
        
        if self.risk.max_turnover > 0.2:
            issues.append(f"最大换手率({self.risk.max_turnover}) 可能过高")
        
        if issues:
            logger.warning("配置一致性问题:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ 配置一致性验证通过")
    
    def _use_default_configs(self) -> None:
        """使用默认配置"""
        logger.info("使用默认配置")
        # 所有配置已在dataclass中定义默认值
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return {
            'temporal': asdict(self.temporal),
            'model': asdict(self.model),
            'risk': asdict(self.risk),
            'production': asdict(self.production),
            'system': asdict(self.system),
            'metadata': {
                'environment': self.environment,
                'config_dir': str(self.config_dir),
                'loaded_at': datetime.now().isoformat()
            }
        }
    
    def save_unified_config(self, output_path: Optional[str] = None) -> str:
        """保存统一配置文件"""
        if output_path is None:
            output_path = self.config_dir / "unified_config.yaml"
        
        config_dict = self.get_config_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        logger.info(f"统一配置已保存: {output_path}")
        return str(output_path)
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """更新配置"""
        if hasattr(self, section):
            config_obj = getattr(self, section)
            for key, value in updates.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"配置更新: {section}.{key} = {value}")
                else:
                    logger.warning(f"未知配置项: {section}.{key}")
        else:
            logger.error(f"未知配置段: {section}")
    
    def get_legacy_config_for_compatibility(self) -> Dict[str, Any]:
        """获取兼容旧系统的配置"""
        return {
            # T10Config兼容
            'PREDICTION_HORIZON': self.temporal.prediction_horizon,
            'HOLDING_PERIOD': self.temporal.holding_period,
            'FEATURE_LAG': self.temporal.feature_lag,
            'ISOLATION_DAYS': self.temporal.isolation_days,
            'EMBARGO_DAYS': self.temporal.embargo_days,
            'CV_GAP': self.temporal.cv_gap,
            
            # BMAEnhancedConfig兼容
            'enable_regime_awareness': self.system.enable_regime_awareness,
            'enable_production_gates': self.system.enable_production_gates,
            'enable_incremental_training': self.system.enable_incremental_training,
            'cache_dir': self.system.cache_dir,
            
            # 其他常用配置
            'max_position': self.risk.max_position,
            'max_turnover': self.risk.max_turnover,
            'max_memory_gb': self.production.max_memory_gb,
            'beta_calculation_window': self.model.beta_calculation_window
        }

# 全局配置管理器实例
_global_config_manager: Optional[UnifiedConfigManager] = None

def get_unified_config_manager(config_dir: str = "D:\\trade", 
                             environment: str = "development") -> UnifiedConfigManager:
    """获取统一配置管理器"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager(config_dir, environment)
    
    return _global_config_manager

def get_config(section: str) -> Any:
    """获取特定配置段"""
    manager = get_unified_config_manager()
    return getattr(manager, section, None)