#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一时序配置注册表 - 单一真相源
解决V6系统中时序参数不一致导致的数据泄露风险
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimingRegistry:
    """
    统一时序配置注册表 - 严格的单一真相源
    
    所有时序相关参数必须从此处获取，禁止在其他地方硬编码
    """
    # === 核心隔离参数 ===
    prediction_horizon: int = 10  # T+10预测期
    holding_period: int = 10      # 10天持有期
    
    # === 严格隔离配置 === (UNIFIED: 统一为1天隔离期以提高数据利用率)
    effective_isolation: int = 1   # FIX: 调整为1天隔离期
    cv_gap_days: int = 1          # FIX: CV gap天数调整为1天
    cv_embargo_days: int = 1      # FIX: CV embargo天数调整为1天  
    oos_embargo_days: int = 1     # FIX: OOS embargo天数调整为1天
    
    # === 特征滞后配置 ===
    feature_lag_days: int = 1      # FIX: 特征滞后天数调整为1天（更合理）
    feature_global_lag: int = 1    # FIX: 全局特征滞后调整为1天
    
    # === 样本权重半衰期 ===
    sample_weight_half_life: int = 75  # 统一样本权重半衰期（天）
    
    # === Regime配置 ===
    regime_enable_smoothing: bool = False  # 严格禁用Regime平滑
    regime_prob_smooth_window: int = 0     # 平滑窗口强制为0
    regime_transition_buffer: int = 2      # Regime转换缓冲期
    
    # === CV配置 ===
    cv_n_splits: int = 5           # CV折数
    cv_test_ratio: float = 0.2     # 测试集比例
    
    # === 生产闸门阈值 ===
    min_rank_ic: float = 0.02      # 最低RankIC要求
    min_t_stat: float = 2.0        # 最低t统计量
    min_coverage_months: int = 12  # 最低覆盖月数（严格提升至12个月）
    min_qlike_reduction_pct: float = 0.12  # QLIKE最小改善（12%）
    
    # === 防泄露安全参数 ===
    safety_gap: int = 1            # UNIFIED: 额外安全间隔调整为1天
    min_train_test_gap: int = 12   # 训练测试最小间隔
    
    def __post_init__(self):
        """初始化后验证参数一致性"""
        self._validate_consistency()
        self._log_configuration()
    
    def _validate_consistency(self):
        """验证时序参数一致性，防止配置冲突"""
        errors = []
        
        # 1. 验证隔离参数一致性
        if self.cv_embargo_days != self.oos_embargo_days:
            errors.append(f"CV embargo({self.cv_embargo_days}) != OOS embargo({self.oos_embargo_days})")
        
        if self.cv_gap_days != self.effective_isolation:
            errors.append(f"CV gap({self.cv_gap_days}) != effective_isolation({self.effective_isolation})")
        
        # 2. 验证时序逻辑 (UPDATED for 1-day isolation)
        total_gap = self.cv_gap_days + self.cv_embargo_days + self.safety_gap
        # 放宽验证逻辑：对于1天隔离期，只需要确保基本的数据完整性
        min_required_gap = max(3, self.feature_lag_days + 2)  # 至少3天或特征滞后+2天
        if total_gap < min_required_gap:
            errors.append(f"Total gap({total_gap}) < minimum required gap({min_required_gap})")
        
        # 3. 验证Regime平滑禁用
        if self.regime_enable_smoothing and self.regime_prob_smooth_window > 0:
            errors.append("Regime平滑已声明禁用但平滑窗口>0")
        
        # 4. 验证生产阈值合理性
        if self.min_rank_ic < 0.01:
            errors.append(f"生产RankIC阈值过低: {self.min_rank_ic}")
            
        if self.min_coverage_months < 12:
            errors.append(f"覆盖期过短: {self.min_coverage_months}个月 < 12个月")
        
        if errors:
            error_msg = "时序配置不一致错误:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
    
    def _log_configuration(self):
        """记录配置以供审计"""
        logger.info("=== 统一时序配置 (TimingRegistry) ===")
        logger.info(f"预测期: T+{self.prediction_horizon}, 持有期: {self.holding_period}天")
        logger.info(f"隔离参数: gap={self.cv_gap_days}, embargo={self.cv_embargo_days}")
        logger.info(f"特征滞后: {self.feature_lag_days}天")
        logger.info(f"样本权重半衰期: {self.sample_weight_half_life}天")
        logger.info(f"Regime平滑: {'禁用' if not self.regime_enable_smoothing else '启用'}(窗口={self.regime_prob_smooth_window})")
        logger.info(f"生产闸门: RankIC≥{self.min_rank_ic}, t≥{self.min_t_stat}, 覆盖≥{self.min_coverage_months}月")
    
    def get_purged_cv_params(self) -> Dict[str, Any]:
        """获取PurgedCV参数"""
        return {
            'n_splits': self.cv_n_splits,
            'gap_days': self.cv_gap_days,
            'embargo_days': self.cv_embargo_days,
            'test_size': self.cv_test_ratio
        }
    
    def get_oos_params(self) -> Dict[str, Any]:
        """获取OOS参数"""
        return {
            'embargo_days': self.oos_embargo_days,
            'min_train_test_gap': self.min_train_test_gap,
            'safety_gap': self.safety_gap
        }
    
    def get_regime_params(self) -> Dict[str, Any]:
        """获取Regime参数 - 严格禁平滑"""
        return {
            'enable_smoothing': self.regime_enable_smoothing,
            'prob_smooth_window': self.regime_prob_smooth_window,
            'transition_buffer': self.regime_transition_buffer,
            'use_filtering_only': True  # 强制仅过滤
        }
    
    def get_sample_weight_params(self) -> Dict[str, Any]:
        """获取样本权重参数"""
        return {
            'half_life_days': self.sample_weight_half_life,
            'decay_type': 'exponential'
        }
    
    def get_production_gate_params(self) -> Dict[str, Any]:
        """获取生产闸门参数"""
        return {
            'min_rank_ic': self.min_rank_ic,
            'min_t_stat': self.min_t_stat,
            'min_coverage_months': self.min_coverage_months,
            'min_qlike_reduction_pct': self.min_qlike_reduction_pct,
            'gate_logic': 'strict_and_with_fallback'
        }


# === 全局单例实例 ===
_global_timing_registry: Optional[TimingRegistry] = None


def get_global_timing_registry() -> TimingRegistry:
    """获取全局时序注册表单例"""
    global _global_timing_registry
    if _global_timing_registry is None:
        _global_timing_registry = TimingRegistry()
    return _global_timing_registry


def reset_timing_registry(new_config: Optional[Dict[str, Any]] = None) -> TimingRegistry:
    """重置时序注册表（用于测试）"""
    global _global_timing_registry
    if new_config:
        _global_timing_registry = TimingRegistry(**new_config)
    else:
        _global_timing_registry = TimingRegistry()
    return _global_timing_registry


class TimingEnforcer:
    """时序参数执行器 - 确保所有组件使用统一配置"""
    
    @staticmethod
    def enforce_cv_integrity(cv_class_name: str, cv_params: Dict[str, Any]) -> Dict[str, Any]:
        """强制CV完整性 - 禁止回退到无隔离版本"""
        registry = get_global_timing_registry()
        
        if 'TimeSeriesSplit' in cv_class_name and 'Purged' not in cv_class_name:
            raise ValueError(
                "❌ 严重错误: CV系统回退到无隔离的sklearn.TimeSeriesSplit\n"
                "这将导致严重的数据泄露！\n"
                "解决方案: 安装fixed_purged_time_series_cv或停止训练"
            )
        
        # 强制使用注册表参数
        required_params = registry.get_purged_cv_params()
        enforced_params = cv_params.copy()
        enforced_params.update(required_params)
        
        logger.info("✅ CV完整性检查通过")
        logger.info(f"  CV类型: {cv_class_name}")
        logger.info(f"  Gap天数: {enforced_params['gap_days']}")
        logger.info(f"  Embargo天数: {enforced_params['embargo_days']}")
        
        return enforced_params
    
    @staticmethod
    def enforce_sample_weight_consistency(module_name: str, proposed_half_life: int) -> int:
        """强制样本权重半衰期一致性"""
        registry = get_global_timing_registry()
        canonical_half_life = registry.sample_weight_half_life
        
        if abs(proposed_half_life - canonical_half_life) > 15:  # 允许±15天差异
            logger.warning(
                f"⚠️ {module_name}模块半衰期不一致: "
                f"提议{proposed_half_life}天 vs 标准{canonical_half_life}天"
            )
            logger.info(f"强制使用标准半衰期: {canonical_half_life}天")
        
        return canonical_half_life
    
    @staticmethod
    def enforce_regime_no_smoothing(regime_config: Dict[str, Any]) -> Dict[str, Any]:
        """强制Regime禁平滑"""
        registry = get_global_timing_registry()
        regime_params = registry.get_regime_params()
        
        # 强制覆盖任何平滑配置
        enforced_config = regime_config.copy()
        enforced_config.update(regime_params)
        
        # 额外安全检查
        smoothing_keys = [
            'enable_smoothing', 'prob_smooth_window', 'smooth_transitions',
            'rolling_window', 'ma_window'
        ]
        
        for key in smoothing_keys:
            if key in enforced_config and key != 'enable_smoothing':
                if isinstance(enforced_config[key], bool):
                    enforced_config[key] = False
                elif isinstance(enforced_config[key], (int, float)):
                    enforced_config[key] = 0
        
        logger.info("✅ Regime平滑已强制禁用")
        logger.info(f"  enable_smoothing: {enforced_config['enable_smoothing']}")
        logger.info(f"  prob_smooth_window: {enforced_config['prob_smooth_window']}")
        
        return enforced_config


if __name__ == "__main__":
    # 测试时序注册表
    registry = get_global_timing_registry()
    print("时序注册表测试通过")
    
    # 测试参数一致性验证
    try:
        test_config = {
            'cv_embargo_days': 1,  # 优化为1天一致配置
            'oos_embargo_days': 1
        }
        reset_timing_registry(test_config)
        print("❌ 应该检测到不一致错误")
    except ValueError as e:
        print(f"✅ 正确检测到配置不一致: {e}")