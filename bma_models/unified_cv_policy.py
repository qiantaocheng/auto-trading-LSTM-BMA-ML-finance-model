#!/usr/bin/env python3
"""
统一CV策略配置中心 - 解决时间泄漏防线不一致问题
================================================================
单一事实来源，禁用向下自适应，统一隔离策略
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CVPolicy:
    """统一CV策略配置 - 单一事实来源"""
    
    # ==================== 核心隔离参数 ====================
    # T10配置作为唯一事实来源
    holding_period: int = 10                    # 持仓期
    isolation_days: int = 10                    # CRITICAL FIX: 统一10天隔离
    embargo_days: int = 10                      # CRITICAL FIX: 统一10天embargo
    gap_days: int = 10                          # CRITICAL FIX: 统一10天gap
    min_isolation_days: int = 10                # CRITICAL FIX: 统一10天最小隔离
    
    # ==================== CV配置 ====================
    n_splits: int = 5                          # CV折数
    test_size: float = 0.2                     # 测试集比例
    min_samples_per_fold: int = 100            # 每折最少样本数
    max_train_size: Optional[int] = None       # 最大训练样本数
    
    # ==================== 时间安全配置 ====================
    purge_method: str = "purge"                # 固定使用purge方法
    enforce_temporal_order: bool = True        # 强制时间顺序
    allow_future_leakage: bool = False         # 禁止未来信息泄漏
    validate_time_gaps: bool = True            # 验证时间间隔
    
    # ==================== 质量控制 ====================
    min_total_samples: int = 500               # 最少总样本数
    min_validation_coverage: float = 0.15      # 最小验证覆盖率
    max_validation_coverage: float = 0.3       # 最大验证覆盖率
    
    # ==================== 一致性控制 ====================
    single_isolation_source: bool = True      # 单一隔离策略来源
    disable_adaptive_reduction: bool = True   # 禁用自适应缩减
    strict_temporal_validation: bool = True   # 严格时间验证

    def validate_consistency(self) -> Dict[str, Any]:
        """验证配置一致性"""
        issues = []
        
        # 检查隔离天数一致性
        if self.isolation_days != self.holding_period:
            issues.append(f"隔离天数({self.isolation_days})与持仓期({self.holding_period})不一致")
        
        if self.embargo_days != self.holding_period:
            issues.append(f"禁带期({self.embargo_days})与持仓期({self.holding_period})不一致")
            
        if self.gap_days < self.holding_period:
            issues.append(f"CV gap({self.gap_days})小于持仓期({self.holding_period})")
        
        # 检查禁用自适应设置
        if not self.disable_adaptive_reduction:
            issues.append("必须禁用自适应缩减以防止时间泄漏")
        
        if self.min_isolation_days < self.holding_period:
            issues.append(f"最小隔离天数({self.min_isolation_days})不能小于持仓期")
        
        # 检查时间安全配置
        if not self.enforce_temporal_order:
            issues.append("必须强制时间顺序")
            
        if self.allow_future_leakage:
            issues.append("不允许未来信息泄漏")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_isolation_stats_template(self) -> Dict[str, Any]:
        """获取隔离统计模板"""
        return {
            'target_isolation_days': self.isolation_days,
            'actual_isolation_days': None,
            'min_gap_achieved': None,
            'embargo_compliance': None,
            'temporal_order_violations': 0,
            'adaptive_reduction_events': 0,  # 应该始终为0
            'validation_passed': False
        }

@dataclass  
class IsolationPolicy:
    """隔离策略配置"""
    method: str = "purge"                      # 隔离方法
    days: int = 10                             # 隔离天数，来自CVPolicy
    strict_mode: bool = True                   # 严格模式，禁止缩减
    
    def __post_init__(self):
        if self.method not in ["purge", "embargo", "both"]:
            raise ValueError("隔离方法必须是 purge, embargo 或 both")

@dataclass
class CalibrationPolicy:
    """校准策略配置"""
    method: str = "isotonic"                   # 校准方法
    min_folds_required: int = 3                # 最少CV折数
    allow_full_sample_fallback: bool = False  # 禁止全样本回退
    strict_oos_only: bool = True               # 仅允许严格OOS校准
    
    def __post_init__(self):
        if self.method not in ["isotonic", "platt", "none"]:
            raise ValueError("校准方法必须是 isotonic, platt 或 none")

class UnifiedCVPolicyManager:
    """统一CV策略管理器"""
    
    def __init__(self, cv_policy: CVPolicy = None):
        """初始化策略管理器"""
        self.cv_policy = cv_policy or CVPolicy()
        self.isolation_policy = IsolationPolicy(days=self.cv_policy.isolation_days)
        self.calibration_policy = CalibrationPolicy()
        
        # 验证配置一致性
        self.validation_result = self.cv_policy.validate_consistency()
        if not self.validation_result['is_valid']:
            logger.error(f"CV策略配置不一致: {self.validation_result['issues']}")
            raise ValueError("CV策略配置验证失败")
        
        logger.info("统一CV策略管理器初始化成功")
    
    def get_cv_params(self) -> Dict[str, Any]:
        """获取CV参数，供所有CV类使用"""
        return {
            'n_splits': self.cv_policy.n_splits,
            'test_size': self.cv_policy.test_size,
            'gap': self.cv_policy.gap_days,
            'embargo': self.cv_policy.embargo_days,
            'purge': self.cv_policy.isolation_days,
            'min_samples_per_fold': self.cv_policy.min_samples_per_fold,
            'max_train_size': self.cv_policy.max_train_size,
        }
    
    def validate_cv_split(self, train_dates: List[datetime], 
                          test_dates: List[datetime]) -> Dict[str, Any]:
        """验证CV切分的时间安全性 - 修复：验证所有train-test对"""
        if not train_dates or not test_dates:
            return {'valid': False, 'reason': '训练或测试日期为空'}
        
        # 🔧 修复: 验证所有训练-测试日期对，而非仅检查min/max
        required_gap = self.cv_policy.gap_days + self.cv_policy.embargo_days
        violations = []
        min_actual_gap = float('inf')
        
        for test_date in test_dates:
            for train_date in train_dates:
                if train_date >= test_date:
                    violations.append({
                        'type': 'temporal_order_violation',
                        'train_date': train_date,
                        'test_date': test_date,
                        'gap_days': (test_date - train_date).days
                    })
                else:
                    actual_gap = (test_date - train_date).days
                    min_actual_gap = min(min_actual_gap, actual_gap)
                    
                    if actual_gap < required_gap:
                        violations.append({
                            'type': 'insufficient_gap',
                            'train_date': train_date,
                            'test_date': test_date,
                            'gap_days': actual_gap,
                            'required_gap': required_gap
                        })
        
        # 原有逻辑保持兼容
        train_max = max(train_dates)
        test_min = min(test_dates)
        legacy_gap = (test_min - train_max).days
        
        validation_result = {
            'valid': len(violations) == 0,
            'actual_gap_days': legacy_gap,  # 保持向后兼容
            'min_actual_gap_days': min_actual_gap if min_actual_gap != float('inf') else legacy_gap,
            'required_gap_days': required_gap,
            'train_max_date': train_max,
            'test_min_date': test_min,
            'temporal_order_ok': train_max < test_min,
            'violations': violations,
            'violation_count': len(violations)
        }
        
        if not validation_result['valid']:
            logger.warning(f"CV切分时间安全性验证失败: 发现{len(violations)}个违规")
            for violation in violations[:3]:  # 只显示前3个
                logger.warning(f"  {violation['type']}: {violation['train_date']} -> {violation['test_date']} ({violation['gap_days']}天)")
        
        return validation_result
    
    def create_isolation_stats(self, actual_isolation_days: int, 
                              gap_achieved: int, embargo_ok: bool,
                              violations: int = 0) -> Dict[str, Any]:
        """创建隔离统计信息"""
        stats = self.cv_policy.get_isolation_stats_template()
        stats.update({
            'actual_isolation_days': actual_isolation_days,
            'min_gap_achieved': gap_achieved,
            'embargo_compliance': embargo_ok,
            'temporal_order_violations': violations,
            'validation_passed': (
                actual_isolation_days >= self.cv_policy.isolation_days and
                gap_achieved >= self.cv_policy.gap_days and
                embargo_ok and violations == 0
            )
        })
        return stats
    
    def should_reject_fold(self, fold_validation: Dict[str, Any]) -> bool:
        """判断是否应该拒绝某个CV折"""
        if not fold_validation.get('valid', False):
            return True
        
        # 严格检查时间间隔
        if fold_validation.get('actual_gap_days', 0) < self.cv_policy.gap_days:
            return True
        
        # 检查时间顺序
        if not fold_validation.get('temporal_order_ok', False):
            return True
        
        return False
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        return {
            'cv_policy': {
                'holding_period': self.cv_policy.holding_period,
                'isolation_days': self.cv_policy.isolation_days,
                'embargo_days': self.cv_policy.embargo_days,
                'gap_days': self.cv_policy.gap_days,
                'disable_adaptive': self.cv_policy.disable_adaptive_reduction
            },
            'isolation_policy': {
                'method': self.isolation_policy.method,
                'days': self.isolation_policy.days,
                'strict_mode': self.isolation_policy.strict_mode
            },
            'calibration_policy': {
                'method': self.calibration_policy.method,
                'min_folds_required': self.calibration_policy.min_folds_required,
                'allow_fallback': self.calibration_policy.allow_full_sample_fallback,
                'strict_oos_only': self.calibration_policy.strict_oos_only
            },
            'validation_result': self.validation_result
        }

# 全局统一策略实例
GLOBAL_CV_POLICY = UnifiedCVPolicyManager()

def get_global_cv_policy() -> UnifiedCVPolicyManager:
    """获取全局CV策略"""
    return GLOBAL_CV_POLICY

if __name__ == "__main__":
    # 测试配置一致性
    policy_manager = UnifiedCVPolicyManager()
    summary = policy_manager.get_policy_summary()
    print("=== 统一CV策略配置摘要 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")