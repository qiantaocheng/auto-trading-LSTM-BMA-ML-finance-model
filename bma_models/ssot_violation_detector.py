#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSOT违规检测器 - 统一检测CV创建和特征选择违规
防止重复修改，提供统一的违规检测和修复指导
"""

import sys
import inspect
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class SSOTViolationDetector:
    """SSOT违规统一检测器"""
    
    @staticmethod
    def detect_cv_creation_violation(func_name: str = None, caller_info: str = None) -> None:
        """
        检测CV创建违规 - 禁止内部TimeSeriesSplit/cross_val_score使用
        
        Args:
            func_name: 函数名
            caller_info: 调用者信息
        """
        # 获取调用栈信息
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                lineno = caller_frame.f_lineno
                func_name = func_name or caller_frame.f_code.co_name
        except:
            filename = "unknown"
            lineno = 0
        finally:
            del frame
        
        error_msg = (
            f"🚫 违反SSOT原则：禁止内部创建CV分割器！\n"
            f"📍 违规位置: {filename}:{lineno} in {func_name}\n"
            f"🔧 修复方案：\n"
            f"1. 删除所有 TimeSeriesSplit() 实例化\n"
            f"2. 删除所有 cross_val_score() / cross_val_predict() 调用\n"
            f"3. 使用外部传入的 cv_factory 参数\n"
            f"4. 调用方式：cv_factory(dates) 获取统一CV分割器\n"
            f"5. 示例：cv_splitter = cv_factory(dates); splits = cv_splitter(X, y)\n"
            f"❌ 被拦截的违规操作: CV创建"
        )
        
        raise NotImplementedError(error_msg)
    
    @staticmethod
    def detect_feature_selection_violation(selector_type: str = None, caller_info: str = None) -> None:
        """
        检测特征选择违规 - 禁止非RobustFeatureSelector的特征选择
        
        Args:
            selector_type: 选择器类型
            caller_info: 调用者信息
        """
        # 获取调用栈信息
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                lineno = caller_frame.f_lineno
                func_name = caller_frame.f_code.co_name
        except:
            filename = "unknown"
            lineno = 0
            func_name = "unknown"
        finally:
            del frame
        
        error_msg = (
            f"🚫 违反SSOT原则：禁止内部特征选择操作！\n"
            f"📍 违规位置: {filename}:{lineno} in {func_name}\n"
            f"🔧 修复方案：\n"
            f"1. 删除所有 SelectKBest / RFE / SelectFromModel 实例\n"
            f"2. 删除所有 feature_selection.* 模块调用\n"
            f"3. 仅使用全局 RobustFeatureSelector(robust_feature_selection.py)\n"
            f"4. 调用方式：从外部传入已选择特征的数据\n"
            f"5. 或使用：get_global_robust_selector().transform(X, y, dates)\n"
            f"❌ 被拦截的违规操作: {selector_type or '特征选择'}"
        )
        
        raise NotImplementedError(error_msg)
    
    @staticmethod
    def check_cv_factory_requirement(cv_factory: Any = None, context: str = "unknown") -> None:
        """
        检查cv_factory参数要求
        
        Args:
            cv_factory: CV工厂参数
            context: 上下文信息
        """
        if cv_factory is None:
            # 获取调用栈信息
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back
                if caller_frame:
                    filename = caller_frame.f_code.co_filename
                    lineno = caller_frame.f_lineno
                    func_name = caller_frame.f_code.co_name
            except:
                filename = "unknown"
                lineno = 0
                func_name = "unknown"
            finally:
                del frame
            
            error_msg = (
                f"🚫 违反SSOT原则：缺少必需的cv_factory参数！\n"
                f"📍 违规位置: {filename}:{lineno} in {func_name}\n"
                f"🔧 修复方案：\n"
                f"1. 添加 cv_factory 参数到函数签名\n"
                f"2. 从调用方传入统一的CV工厂\n"
                f"3. 使用：from .unified_cv_factory import get_unified_cv_factory\n"
                f"4. 或从上级调用传入：cv_factory=get_unified_cv_factory().create_cv_factory()\n"
                f"❌ 上下文: {context}"
            )
            
            raise ValueError(error_msg)


def block_cv_creation():
    """装饰器：阻止CV创建违规"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 检查是否试图创建CV
            frame = inspect.currentframe()
            try:
                # 检查调用栈中是否有违规模式
                current_frame = frame.f_back
                while current_frame:
                    code = current_frame.f_code
                    if any(pattern in code.co_names for pattern in 
                          ['TimeSeriesSplit', 'cross_val_score', 'cross_val_predict']):
                        SSOTViolationDetector.detect_cv_creation_violation(
                            func_name=func.__name__,
                            caller_info=f"{code.co_filename}:{current_frame.f_lineno}"
                        )
                    current_frame = current_frame.f_back
            finally:
                del frame
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def block_feature_selection():
    """装饰器：阻止特征选择违规"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 检查是否试图进行特征选择
            frame = inspect.currentframe()
            try:
                current_frame = frame.f_back
                while current_frame:
                    code = current_frame.f_code
                    if any(pattern in code.co_names for pattern in 
                          ['SelectKBest', 'RFE', 'SelectFromModel', 'feature_selection']):
                        SSOTViolationDetector.detect_feature_selection_violation(
                            selector_type="sklearn.feature_selection",
                            caller_info=f"{code.co_filename}:{current_frame.f_lineno}"
                        )
                    current_frame = current_frame.f_back
            finally:
                del frame
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 全局违规检测函数（便捷调用）
def ensure_cv_factory_provided(cv_factory: Any = None, context: str = "ML训练"):
    """确保cv_factory已提供"""
    SSOTViolationDetector.check_cv_factory_requirement(cv_factory, context)

def block_internal_cv_creation(operation: str = "CV创建"):
    """阻止内部CV创建"""
    SSOTViolationDetector.detect_cv_creation_violation(caller_info=operation)

def block_internal_feature_selection(selector_type: str = "特征选择"):
    """阻止内部特征选择"""
    SSOTViolationDetector.detect_feature_selection_violation(selector_type=selector_type)


if __name__ == "__main__":
    # 测试违规检测
    print("测试SSOT违规检测器")
    
    try:
        block_internal_cv_creation("测试CV违规")
    except NotImplementedError as e:
        print("✅ CV违规检测正常:")
        print(str(e)[:200] + "...")
    
    try:
        block_internal_feature_selection("测试特征选择违规")
    except NotImplementedError as e:
        print("✅ 特征选择违规检测正常:")
        print(str(e)[:200] + "...")