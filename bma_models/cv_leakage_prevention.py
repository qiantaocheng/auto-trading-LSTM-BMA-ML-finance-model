#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV数据泄露防护模块 - 移除危险的CV回退机制，强制使用PurgedTimeSeriesSplit
防止系统回退到无隔离的sklearn.TimeSeriesSplit导致严重数据泄露
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
import warnings

from unified_timing_registry import get_global_timing_registry, TimingEnforcer

logger = logging.getLogger(__name__)


class CVLeakagePreventionError(Exception):
    """CV数据泄露防护异常"""
    pass


class SafeCVWrapper:
    """
    安全CV包装器 - 强制groups参数，禁止危险退化
    
    ⚠️ 禁止自动回退！CV失败必须硬失败或影子模式
    """
    
    def __init__(self, primary_cv, preventer, params):
        self.primary_cv = primary_cv
        self.preventer = preventer
        self.params = params
        self.fallback_cv = None
        self.using_fallback = False
        self.cv_integrity_verified = False
        
    def split(self, X, y=None, groups=None):
        """严格CV分割，强制groups参数验证"""
        
        # 🔥 CRITICAL: 强制groups参数检查
        if groups is None:
            error_msg = (
                "❌ CV INTEGRITY VIOLATION: groups参数是必须的！\n"
                "  修复方法:\n"
                "  groups = df['date'].values  # 或 y.index.to_period('D').values\n"
                "  for tr, va in cv.split(X, y, groups=groups): ...\n"
                "  \n"
                "  📊 当前数据状态:\n"
                f"  - X shape: {getattr(X, 'shape', 'Unknown')}\n"
                f"  - y length: {len(y) if y is not None else 'None'}\n"
                f"  - groups: {groups}\n"
                "  \n"
                "  🚨 禁止退化为无隔离CV！"
            )
            logger.critical(error_msg)
            raise CVLeakagePreventionError(error_msg)
        
        # 🔥 验证groups长度匹配
        if len(groups) != len(X):
            error_msg = f"groups长度({len(groups)}) != X长度({len(X)}), 必须对齐！"
            logger.critical(error_msg)
            raise CVLeakagePreventionError(error_msg)
        
        logger.info(f"📊 CV Integrity Check: X={getattr(X, 'shape', 'Unknown')}, groups={len(groups)}")
        
        try:
            # 使用严格的Purged CV分割器
            fold_count = 0
            for train_idx, test_idx in self.primary_cv.split(X, y, groups):
                fold_count += 1
                
                # 🔥 验证时间隔离完整性
                if hasattr(groups, '__getitem__'):
                    train_dates = [groups[i] for i in train_idx[-5:]]  # 训练集末尾
                    test_dates = [groups[i] for i in test_idx[:5]]     # 测试集开头
                    logger.info(f"Fold {fold_count} 时间隔离: 训练末尾{train_dates} → 测试开头{test_dates}")
                
                yield train_idx, test_idx
            
            if fold_count == 0:
                error_msg = "❌ CV生成0个fold，数据不足或配置错误"
                logger.critical(error_msg)
                raise CVLeakagePreventionError(error_msg)
            
            self.cv_integrity_verified = True
            logger.info(f"✅ CV Integrity通过: 成功生成{fold_count}个fold，时间隔离验证完成")
            
        except Exception as e:
            error_msg = (
                f"❌ CV HARD FAILURE: {e}\n"
                "🚨 禁止退化！请修复CV配置或进入影子模式\n"
                f"建议检查:\n"
                f"  - gap_days: {self.params.get('gap_days', 'Unknown')}\n" 
                f"  - embargo_days: {self.params.get('embargo_days', 'Unknown')}\n"
                f"  - n_splits: {self.params.get('n_splits', 'Unknown')}\n"
                f"  - 数据时间跨度是否足够"
            )
            logger.critical(error_msg)
            raise CVLeakagePreventionError(error_msg)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """获取分割数量"""
        if self.using_fallback and self.fallback_cv:
            return self.fallback_cv.get_n_splits(X, y, groups)
        else:
            return self.primary_cv.get_n_splits(X, y, groups)


class CVLeakagePreventer:
    """
    CV数据泄露防护器
    
    功能：
    1. 检测并阻止使用无隔离的CV方法
    2. 强制要求使用PurgedTimeSeriesSplit
    3. 验证CV配置的时序安全性
    4. 提供安全的CV实例创建
    """
    
    def __init__(self):
        self.timing_registry = get_global_timing_registry()
        self.blocked_cv_classes = [
            'TimeSeriesSplit',
            'KFold', 
            'StratifiedKFold',
            'GroupKFold',
            'LeaveOneOut',
            'LeavePOut'
        ]
        self.allowed_cv_classes = [
            'PurgedTimeSeriesSplit',
            'PurgedKFold',
            'PurgedGroupTimeSeriesSplit'
        ]
        self.intervention_log = []
        
        logger.info("CV数据泄露防护器已初始化")
        logger.info(f"禁用CV类型: {self.blocked_cv_classes}")
        logger.info(f"允许CV类型: {self.allowed_cv_classes}")
    
    def validate_cv_class(self, cv_class_name: str, cv_instance: Any = None) -> bool:
        """
        验证CV类的安全性
        
        Args:
            cv_class_name: CV类名
            cv_instance: CV实例（可选）
            
        Returns:
            是否安全
        """
        logger.info(f"验证CV类安全性: {cv_class_name}")
        
        # 检查是否为危险的CV类
        if any(blocked in cv_class_name for blocked in self.blocked_cv_classes):
            # 特别检查是否为原生TimeSeriesSplit
            if cv_class_name == 'TimeSeriesSplit' and 'Purged' not in str(type(cv_instance)):
                error_msg = (
                    f"❌ 严重数据泄露风险检测: 使用了无隔离的 {cv_class_name}\n"
                    f"这将导致严重的前瞻性偏差！\n"
                    f"必须使用 PurgedTimeSeriesSplit 替代\n"
                    f"解决方案:\n"
                    f"  1. 安装 fixed_purged_time_series_cv 库\n"
                    f"  2. 使用 PurgedTimeSeriesSplit 替代 TimeSeriesSplit\n"
                    f"  3. 或者停止训练直到修复此问题"
                )
                logger.critical(error_msg)
                
                self.intervention_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'cv_class': cv_class_name,
                    'risk_level': 'CRITICAL',
                    'action': 'BLOCKED',
                    'reason': 'No purging/embargo - severe data leakage risk'
                })
                
                raise CVLeakagePreventionError(error_msg)
            
            # 其他被禁用的CV类
            elif not any(allowed in cv_class_name for allowed in self.allowed_cv_classes):
                warning_msg = (
                    f"⚠️ 使用了非时序安全的CV方法: {cv_class_name}\n"
                    f"建议使用 PurgedTimeSeriesSplit 以避免数据泄露风险"
                )
                logger.warning(warning_msg)
                warnings.warn(warning_msg, UserWarning)
                
                self.intervention_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'cv_class': cv_class_name,
                    'risk_level': 'WARNING',
                    'action': 'WARNED',
                    'reason': 'Non-temporal CV method'
                })
                
                return False
        
        # 检查是否为允许的CV类
        if any(allowed in cv_class_name for allowed in self.allowed_cv_classes):
            logger.info(f"✅ CV类 {cv_class_name} 验证通过")
            return True
        
        logger.warning(f"⚠️ 未知CV类: {cv_class_name}")
        return False
    
    def enforce_purged_cv_params(self, cv_params: Dict[str, Any], cv_class_name: str) -> Dict[str, Any]:
        """
        强制执行Purged CV参数
        
        Args:
            cv_params: 原始CV参数
            cv_class_name: CV类名
            
        Returns:
            强制执行后的CV参数
        """
        return TimingEnforcer.enforce_cv_integrity(cv_class_name, cv_params)
    
    def create_safe_cv_splitter(self, prefer_sklearn_compatible=False, **kwargs) -> Any:
        """
        创建安全的CV分割器
        
        Args:
            prefer_sklearn_compatible: 优先选择sklearn兼容的CV（不需要groups参数）
            **kwargs: CV参数
            
        Returns:
            安全的CV分割器实例
        """
        logger.info("创建安全的CV分割器")
        
        # 获取统一的CV参数
        registry_params = self.timing_registry.get_purged_cv_params()
        
        # 合并参数（registry参数优先）
        final_params = {**kwargs, **registry_params}
        
        try:
            # 如果明确要求sklearn兼容，直接使用SimpleSafeTimeSeriesSplit
            if prefer_sklearn_compatible:
                logger.info("明确要求sklearn兼容，直接使用SimpleSafeTimeSeriesSplit")
                try:
                    from simple_safe_cv import SimpleSafeTimeSeriesSplit
                    cv_class_name = "SimpleSafeTimeSeriesSplit"
                    
                    cv_splitter = SimpleSafeTimeSeriesSplit(
                        n_splits=final_params.get('n_splits', 5),
                        gap_days=final_params.get('gap_days', 10),
                        test_size=final_params.get('test_size', 0.2)
                    )
                    logger.info("✅ 使用SimpleSafeTimeSeriesSplit（sklearn兼容优先）")
                    
                except ImportError:
                    logger.error("SimpleSafeTimeSeriesSplit不可用，尝试其他选项")
                    prefer_sklearn_compatible = False  # 回退到常规逻辑
            
            # 常规逻辑：优先尝试使用FixedPurgedGroupTimeSeriesSplit（需要groups参数）
            if not prefer_sklearn_compatible:
                try:
                    from fixed_purged_time_series_cv import FixedPurgedGroupTimeSeriesSplit, ValidationConfig
                    cv_class_name = "FixedPurgedGroupTimeSeriesSplit"
                    
                    # 创建适配的配置
                    validation_config = ValidationConfig(
                        n_splits=final_params.get('n_splits', 5),
                        gap=final_params.get('gap_days', 10),
                        embargo=final_params.get('embargo_days', 0),
                        test_size=final_params.get('test_size', 63),
                        min_train_size=252
                    )
                    
                    # 创建CV实例使用本地实现
                    cv_splitter = FixedPurgedGroupTimeSeriesSplit(validation_config)
                    logger.info("✅ 使用FixedPurgedGroupTimeSeriesSplit（需要groups参数）")
                
                except ImportError:
                    # 如果FixedPurgedGroupTimeSeriesSplit不可用，使用SimpleSafeTimeSeriesSplit作为sklearn兼容的回退
                    logger.warning("FixedPurgedGroupTimeSeriesSplit不可用，使用SimpleSafeTimeSeriesSplit作为sklearn兼容回退")
                    try:
                        from simple_safe_cv import SimpleSafeTimeSeriesSplit
                        cv_class_name = "SimpleSafeTimeSeriesSplit"
                        
                        # 创建sklearn兼容的安全CV（不需要groups参数）
                        cv_splitter = SimpleSafeTimeSeriesSplit(
                            n_splits=final_params.get('n_splits', 5),
                            gap_days=final_params.get('gap_days', 10),
                            test_size=final_params.get('test_size', 0.2)
                        )
                        logger.info("✅ 使用SimpleSafeTimeSeriesSplit（sklearn兼容，无需groups参数）")
                        
                    except ImportError:
                        # 如果两个实现都没有，抛出错误
                        error_msg = (
                            "❌ 无法导入任何安全的CV实现！\n"
                            "尝试了：FixedPurgedGroupTimeSeriesSplit 和 SimpleSafeTimeSeriesSplit\n"
                            "系统拒绝使用无隔离的CV方法以防止数据泄露"
                        )
                        logger.critical(error_msg)
                        raise CVLeakagePreventionError(error_msg)
            
            logger.info(f"✅ 创建安全CV分割器成功: {cv_class_name}")
            logger.info(f"参数: gap={final_params.get('gap_days')}天, "
                       f"embargo={final_params.get('embargo_days')}天, "
                       f"n_splits={final_params.get('n_splits')}")
            
            # 记录成功创建
            self.intervention_log.append({
                'timestamp': pd.Timestamp.now(),
                'cv_class': cv_class_name,
                'risk_level': 'SAFE',
                'action': 'CREATED',
                'reason': 'Safe purged CV with proper isolation',
                'params': final_params
            })
            
            # 创建CV包装器，自动处理groups参数问题
            return SafeCVWrapper(cv_splitter, self, final_params)
            
        except Exception as e:
            error_msg = f"创建安全CV分割器失败: {e}"
            logger.error(error_msg)
            raise CVLeakagePreventionError(error_msg)
    
    def validate_cv_splits(self, cv_splitter: Any, X: np.ndarray, 
                          dates: pd.DatetimeIndex) -> bool:
        """
        验证CV分割的时序安全性
        
        Args:
            cv_splitter: CV分割器
            X: 特征数据
            dates: 日期索引
            
        Returns:
            是否通过验证
        """
        logger.info("验证CV分割时序安全性")
        
        try:
            splits = list(cv_splitter.split(X))
            
            min_gap = self.timing_registry.cv_gap_days
            min_embargo = self.timing_registry.cv_embargo_days
            
            violations = []
            
            for i, (train_idx, test_idx) in enumerate(splits):
                # 获取训练和测试日期
                train_dates = dates[train_idx]
                test_dates = dates[test_idx]
                
                # 检查训练集最大日期和测试集最小日期的间隔
                train_max_date = train_dates.max()
                test_min_date = test_dates.min()
                gap_days = (test_min_date - train_max_date).days
                
                if gap_days < min_gap + min_embargo:
                    violations.append({
                        'fold': i,
                        'gap_days': gap_days,
                        'required_gap': min_gap + min_embargo,
                        'train_max_date': train_max_date,
                        'test_min_date': test_min_date
                    })
                    
                logger.info(f"Fold {i}: gap={gap_days}天, "
                           f"训练期={train_dates.min()}至{train_max_date}, "
                           f"测试期={test_min_date}至{test_dates.max()}")
            
            if violations:
                logger.error("❌ CV分割时序验证失败:")
                for v in violations:
                    logger.error(f"  Fold {v['fold']}: gap={v['gap_days']}天 < 要求{v['required_gap']}天")
                
                self.intervention_log.append({
                    'timestamp': pd.Timestamp.now(),
                    'cv_class': type(cv_splitter).__name__,
                    'risk_level': 'ERROR',
                    'action': 'VALIDATION_FAILED',
                    'reason': f'{len(violations)} folds with insufficient gap',
                    'violations': violations
                })
                
                return False
            else:
                logger.info(f"✅ CV分割时序验证通过: {len(splits)}个折叠")
                return True
                
        except Exception as e:
            logger.error(f"CV分割验证过程出错: {e}")
            return False
    
    def patch_dangerous_cv_imports(self):
        """
        猴子补丁危险的CV导入，防止意外使用
        """
        import sklearn.model_selection
        
        # 保存原始类的引用
        original_TimeSeriesSplit = sklearn.model_selection.TimeSeriesSplit
        
        class SafeTimeSeriesSplit:
            def __init__(self, *args, **kwargs):
                error_msg = (
                    "❌ 检测到尝试使用sklearn.TimeSeriesSplit！\n"
                    "这会导致严重的数据泄露风险！\n"
                    "请使用 PurgedTimeSeriesSplit 替代。\n"
                    "如果确实需要原始功能，请使用 cv_preventer.get_original_timeseriessplit()"
                )
                logger.critical(error_msg)
                raise CVLeakagePreventionError(error_msg)
        
        # 替换危险的类
        sklearn.model_selection.TimeSeriesSplit = SafeTimeSeriesSplit
        
        # 保存原始类以备特殊情况使用
        self._original_TimeSeriesSplit = original_TimeSeriesSplit
        
        logger.info("✅ 已对sklearn.TimeSeriesSplit应用安全补丁")
    
    def get_original_timeseriessplit(self):
        """获取原始TimeSeriesSplit（仅用于特殊测试情况）"""
        logger.warning("⚠️ 请求原始TimeSeriesSplit - 请确保知道数据泄露风险！")
        return self._original_TimeSeriesSplit
    
    def get_prevention_summary(self) -> Dict[str, Any]:
        """获取防护摘要"""
        total_interventions = len(self.intervention_log)
        blocked_count = sum(1 for log in self.intervention_log if log['action'] == 'BLOCKED')
        warned_count = sum(1 for log in self.intervention_log if log['action'] == 'WARNED')
        safe_count = sum(1 for log in self.intervention_log if log['action'] == 'CREATED')
        
        return {
            'total_interventions': total_interventions,
            'blocked_dangerous_cv': blocked_count,
            'warned_risky_cv': warned_count,
            'created_safe_cv': safe_count,
            'intervention_log': self.intervention_log,
            'timing_registry_cv_params': self.timing_registry.get_purged_cv_params()
        }


def prevent_cv_leakage_globally(cv_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    全局防护CV数据泄露
    
    Args:
        cv_configs: CV配置字典 {module_name: cv_config}
        
    Returns:
        防护后的CV配置和创建的安全CV分割器
    """
    preventer = CVLeakagePreventer()
    safe_cv_splitters = {}
    
    logger.info("开始全局CV数据泄露防护")
    
    # 应用安全补丁
    preventer.patch_dangerous_cv_imports()
    
    for module_name, cv_config in cv_configs.items():
        try:
            # 创建安全的CV分割器
            safe_cv = preventer.create_safe_cv_splitter(**cv_config)
            safe_cv_splitters[module_name] = safe_cv
            logger.info(f"✅ 为 {module_name} 创建安全CV分割器")
        except CVLeakagePreventionError as e:
            logger.error(f"❌ {module_name} CV防护失败: {e}")
            # 不创建CV分割器，让调用者处理
            safe_cv_splitters[module_name] = None
    
    # 记录防护摘要
    summary = preventer.get_prevention_summary()
    logger.info("=== 全局CV数据泄露防护完成 ===")
    logger.info(f"总干预次数: {summary['total_interventions']}")
    logger.info(f"阻止危险CV: {summary['blocked_dangerous_cv']}")
    logger.info(f"创建安全CV: {summary['created_safe_cv']}")
    
    return {
        'safe_cv_splitters': safe_cv_splitters,
        'prevention_summary': summary,
        'preventer_instance': preventer
    }


if __name__ == "__main__":
    # 测试CV数据泄露防护
    preventer = CVLeakagePreventer()
    
    # 测试危险CV类检测
    try:
        preventer.validate_cv_class('TimeSeriesSplit')
        print("ERROR: 应该检测到危险CV类")
    except CVLeakagePreventionError:
        print("✅ 正确检测到危险CV类")
    
    # 测试安全CV创建
    try:
        safe_cv = preventer.create_safe_cv_splitter(n_splits=5)
        print(f"✅ 安全CV创建成功: {type(safe_cv).__name__}")
    except CVLeakagePreventionError as e:
        print(f"⚠️ CV创建失败（可能缺少库）: {e}")
    
    # 获取防护摘要
    summary = preventer.get_prevention_summary()
    print("防护摘要:", summary)