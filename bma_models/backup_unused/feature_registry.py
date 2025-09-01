#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两段式特征选择 - 特征注册系统
实现单口径特征管理，避免重复选择和数据窥视
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureRegistry:
    """
    特征注册系统 - 两段式特征选择的核心组件
    
    职责：
    1. 管理Stage-A(RobustFeatureSelector)的选择结果
    2. 为Stage-B(EnhancedMLTrainer)提供统一接口
    3. 避免重复特征选择和数据窥视
    """
    
    def __init__(self, registry_path: str = "bma_models/feature_registry"):
        """
        初始化特征注册系统
        
        Args:
            registry_path: 注册表存储路径
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # 特征选择配置
        self.config = {
            'mode': 'trainer_shrinkage',  # global_only | trainer_shrinkage
            'stage_a': {
                'selector': 'RobustFeatureSelector',
                'target_features': 16,  # Stage-A目标特征数
                'max_features': 32,     # 允许Stage-B裁剪的阈值
            },
            'stage_b': {
                'enabled': True,
                'method': 'importance_only',  # 只用importance，不混用MI/f_regression
                'feature_fraction_range': [0.6, 1.0],  # LightGBM特征采样范围
                'regularization': {
                    'lambda_l1': [0.0, 0.1, 0.5],
                    'lambda_l2': [0.0, 0.1, 0.5]
                }
            },
            'anti_snooping': {
                'enabled': True,
                'max_selection_rounds': 1,  # Stage-A只能选择1次
                'lockdown_after_selection': True
            }
        }
        
        # 当前选择结果缓存
        self._current_selection = None
        self._selection_metadata = None
        
    def register_stage_a_selection(self, 
                                 selected_features: List[str],
                                 feature_metadata: Dict[str, Any],
                                 selection_stats: Dict[str, Any],
                                 selector_config: Dict[str, Any]) -> str:
        """
        注册Stage-A(全局稳健层)的特征选择结果
        
        Args:
            selected_features: Stage-A选择的特征列表
            feature_metadata: 特征元数据(IC统计、簇信息等)
            selection_stats: 选择过程统计
            selector_config: RobustFeatureSelector配置
            
        Returns:
            注册ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        registry_id = f"stage_a_selection_{timestamp}"
        
        # 创建注册记录
        registry_record = {
            'registry_id': registry_id,
            'timestamp': timestamp,
            'stage': 'A',
            'selector': 'RobustFeatureSelector',
            'selected_features': selected_features,
            'feature_count': len(selected_features),
            'feature_metadata': feature_metadata,
            'selection_stats': selection_stats,
            'selector_config': selector_config,
            'status': 'active',
            'anti_snooping': {
                'selection_count': 1,
                'locked': True  # Stage-A选择后立即锁定
            }
        }
        
        # 保存到文件
        registry_file = self.registry_path / f"{registry_id}.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry_record, f, indent=2, ensure_ascii=False, default=str)
        
        # 更新活跃选择
        self._current_selection = selected_features
        self._selection_metadata = registry_record
        
        # 创建快速访问文件
        active_file = self.registry_path / "active_selection.json"
        with open(active_file, 'w', encoding='utf-8') as f:
            json.dump({
                'registry_id': registry_id,
                'selected_features': selected_features,
                'timestamp': timestamp,
                'feature_count': len(selected_features)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Stage-A特征选择已注册: {registry_id}")
        logger.info(f"   特征数量: {len(selected_features)}")
        logger.info(f"   特征列表: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        
        return registry_id
    
    def get_stage_a_features(self) -> Optional[Tuple[List[str], Dict[str, Any]]]:
        """
        获取Stage-A选择的特征(供Stage-B使用)
        
        Returns:
            (selected_features, metadata) 或 None
        """
        # 优先使用缓存
        if self._current_selection is not None:
            return self._current_selection, self._selection_metadata
        
        # 从文件读取
        active_file = self.registry_path / "active_selection.json"
        if not active_file.exists():
            logger.warning("未找到活跃的Stage-A特征选择")
            return None
        
        try:
            with open(active_file, 'r', encoding='utf-8') as f:
                active_record = json.load(f)
            
            # 读取完整记录
            registry_file = self.registry_path / f"{active_record['registry_id']}.json"
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    full_record = json.load(f)
                
                self._current_selection = active_record['selected_features']
                self._selection_metadata = full_record
                
                return self._current_selection, self._selection_metadata
            
        except Exception as e:
            logger.error(f"读取Stage-A特征选择失败: {e}")
        
        return None
    
    def validate_stage_b_request(self, requested_features: List[str]) -> Dict[str, Any]:
        """
        验证Stage-B的特征使用请求
        
        Args:
            requested_features: Stage-B请求使用的特征
            
        Returns:
            验证结果
        """
        validation_result = {
            'valid': False,
            'approved_features': [],
            'rejected_features': [],
            'warnings': [],
            'recommendations': {}
        }
        
        # 获取Stage-A特征
        stage_a_result = self.get_stage_a_features()
        if stage_a_result is None:
            validation_result['warnings'].append("未找到Stage-A特征选择，允许直接使用")
            validation_result['valid'] = True
            validation_result['approved_features'] = requested_features
            return validation_result
        
        stage_a_features, metadata = stage_a_result
        
        # 检查特征是否在Stage-A选择范围内
        approved = []
        rejected = []
        
        for feature in requested_features:
            if feature in stage_a_features:
                approved.append(feature)
            else:
                rejected.append(feature)
        
        validation_result['approved_features'] = approved
        validation_result['rejected_features'] = rejected
        
        # 检查是否需要Stage-B裁剪
        stage_a_count = len(stage_a_features)
        max_features = self.config['stage_a']['max_features']
        
        if stage_a_count > max_features:
            validation_result['warnings'].append(
                f"Stage-A特征数({stage_a_count})超过阈值({max_features})，"
                f"允许Stage-B用importance方法裁剪"
            )
            validation_result['recommendations']['allow_importance_shrinking'] = True
            validation_result['recommendations']['target_features'] = max_features
        else:
            validation_result['recommendations']['allow_importance_shrinking'] = False
            validation_result['warnings'].append(
                f"Stage-A特征数({stage_a_count})已在合理范围，"
                f"建议Stage-B只做模型内收缩(feature_fraction等)"
            )
        
        # 检查反窥视规则
        if rejected:
            validation_result['warnings'].append(
                f"拒绝{len(rejected)}个未经Stage-A选择的特征，防止数据窥视: {rejected[:3]}"
            )
        
        validation_result['valid'] = len(approved) > 0
        
        return validation_result
    
    def get_stage_b_config(self, model_type: str = 'lightgbm') -> Dict[str, Any]:
        """
        获取Stage-B模型内收缩配置
        
        Args:
            model_type: 模型类型 (lightgbm, sklearn_linear, etc.)
            
        Returns:
            Stage-B配置
        """
        base_config = self.config['stage_b'].copy()
        
        if model_type.lower() == 'lightgbm':
            base_config['model_params'] = {
                'feature_fraction': np.random.uniform(0.6, 1.0),  # 特征采样
                'bagging_fraction': np.random.uniform(0.7, 0.9),  # 样本采样
                'lambda_l1': np.random.choice([0.0, 0.1, 0.5]),   # L1正则
                'lambda_l2': np.random.choice([0.0, 0.1, 0.5]),   # L2正则
                'min_data_in_leaf': np.random.choice([20, 30, 50]), # 叶节点样本数
                'num_leaves': np.random.choice([20, 31, 50])       # 叶节点数量
            }
        elif 'linear' in model_type.lower():
            base_config['model_params'] = {
                'alpha': np.random.uniform(0.01, 1.0),  # L1/L2正则强度
                'l1_ratio': np.random.uniform(0.1, 0.9)  # L1比例
            }
        
        return base_config
    
    def create_feature_report(self) -> pd.DataFrame:
        """
        创建特征选择报告
        
        Returns:
            特征选择报告DataFrame
        """
        stage_a_result = self.get_stage_a_features()
        if stage_a_result is None:
            return pd.DataFrame()
        
        features, metadata = stage_a_result
        feature_metadata = metadata.get('feature_metadata', {})
        
        report_data = []
        for feature in features:
            if feature in feature_metadata:
                stats = feature_metadata[feature]
                report_data.append({
                    'feature': feature,
                    'stage': 'A',
                    'ic_mean': stats.get('ic_mean', 0),
                    'ic_std': stats.get('ic_std', 0),
                    'ic_ir': stats.get('ic_ir', 0),
                    'cluster': stats.get('cluster', None),
                    'status': 'active'
                })
        
        return pd.DataFrame(report_data)
    
    def cleanup_old_selections(self, keep_days: int = 7):
        """
        清理旧的特征选择记录
        
        Args:
            keep_days: 保留天数
        """
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        for registry_file in self.registry_path.glob("stage_a_selection_*.json"):
            if registry_file.stat().st_mtime < cutoff_date:
                registry_file.unlink()
                logger.info(f"清理旧注册记录: {registry_file.name}")


# 全局特征注册实例
_global_registry = None

def get_feature_registry() -> FeatureRegistry:
    """获取全局特征注册实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FeatureRegistry()
    return _global_registry


if __name__ == "__main__":
    # 测试特征注册系统
    registry = FeatureRegistry("test_registry")
    
    # 模拟Stage-A选择结果
    selected_features = [f"alpha_factor_{i}" for i in range(1, 17)]  # 16个特征
    feature_metadata = {
        f"alpha_factor_{i}": {
            'ic_mean': 0.02 + i * 0.001,
            'ic_std': 0.05,
            'ic_ir': 0.4 + i * 0.01,
            'cluster': i % 4
        } for i in range(1, 17)
    }
    
    # 注册Stage-A结果
    registry_id = registry.register_stage_a_selection(
        selected_features=selected_features,
        feature_metadata=feature_metadata,
        selection_stats={'total_input': 100, 'final_output': 16},
        selector_config={'ic_window': 126, 'min_ic_mean': 0.01}
    )
    
    print(f"注册ID: {registry_id}")
    
    # 测试Stage-B验证
    validation = registry.validate_stage_b_request(selected_features[:10])
    print(f"Stage-B验证结果: {validation}")
    
    # 生成报告
    report = registry.create_feature_report()
    print("\n特征选择报告:")
    print(report.head())