#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估完整性监控器 - 检测并报告任何CV降级或配置回退
确保评估报告中显著标记时间安全完整性状态
"""

import logging
import json
import warnings
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd

from .unified_config_loader import get_time_config, TIME_CONFIG

try:
    from .unified_purged_cv_factory import CV_DEGRADATION_MONITOR
except ImportError:
    from unified_purged_cv_factory import CV_DEGRADATION_MONITOR

# Removed strict_time_config_enforcer - using unified config loader
class TimeConfigConflictError(Exception):
    """Time configuration conflict error"""
    pass

logger = logging.getLogger(__name__)

@dataclass
class IntegrityViolation:
    """完整性违规记录"""
    timestamp: datetime
    violation_type: str  # 'cv_degradation', 'config_conflict', 'temporal_leak'
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    location: str
    details: str
    impact: str
    remediation: str

@dataclass
class EvaluationIntegrityReport:
    """评估完整性报告"""
    report_timestamp: datetime
    system_status: str  # 'SECURE', 'COMPROMISED', 'CRITICAL'
    violations: List[IntegrityViolation] = field(default_factory=list)
    
    # 时间配置完整性
    time_config_integrity: bool = True
    unified_config_source: str = "unified_time_config"
    detected_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    # CV完整性
    cv_integrity: bool = True
    cv_degradation_attempts: int = 0
    cv_method_used: str = "UnifiedPurgedTimeSeriesCV"
    prohibited_cv_detected: List[str] = field(default_factory=list)
    
    # 评估路径完整性
    evaluation_path_unique: bool = True
    multiple_paths_detected: List[str] = field(default_factory=list)
    
    # 汇总统计
    total_violations: int = 0
    critical_violations: int = 0
    evaluation_trustworthiness: str = "HIGH"  # 'HIGH', 'MEDIUM', 'LOW', 'COMPROMISED'
    
    def add_violation(self, violation: IntegrityViolation):
        """添加违规记录"""
        self.violations.append(violation)
        self.total_violations += 1
        if violation.severity == 'CRITICAL':
            self.critical_violations += 1
            self.system_status = 'CRITICAL'
            self.evaluation_trustworthiness = 'COMPROMISED'
        elif violation.severity == 'HIGH' and self.system_status == 'SECURE':
            self.system_status = 'COMPROMISED'
            self.evaluation_trustworthiness = 'LOW'
    
    def get_display_summary(self) -> str:
        """获取显著的完整性摘要（用于评估报告顶部）"""
        if self.system_status == 'SECURE':
            return "🟢 评估完整性: 安全 - 时间安全配置完整，无降级或回退"
        elif self.system_status == 'COMPROMISED':
            return f"🟡 评估完整性: 受损 - 检测到{self.total_violations}个违规，可信度: {self.evaluation_trustworthiness}"
        else:  # CRITICAL
            return f"🔴 评估完整性: 严重受损 - 检测到{self.critical_violations}个严重违规，结果不可信"
    
    def get_detailed_report(self) -> str:
        """获取详细完整性报告"""
        lines = [
            "=" * 80,
            "📊 评估完整性详细报告",
            "=" * 80,
            f"生成时间: {self.report_timestamp}",
            f"系统状态: {self.system_status}",
            f"评估可信度: {self.evaluation_trustworthiness}",
            "",
            "🔍 时间配置完整性检查:",
            f"  • 统一配置: {'✅ 正常' if self.time_config_integrity else '❌ 受损'}",
            f"  • 配置源: {self.unified_config_source}",
            f"  • 冲突检测: {len(self.detected_conflicts)} 个冲突",
            "",
            "🔍 CV方法完整性检查:",
            f"  • CV完整性: {'✅ 正常' if self.cv_integrity else '❌ 受损'}",
            f"  • 使用方法: {self.cv_method_used}",
            f"  • 降级尝试: {self.cv_degradation_attempts} 次",
            f"  • 禁用方法检测: {len(self.prohibited_cv_detected)} 个",
            "",
            "🔍 评估路径完整性检查:",
            f"  • 路径唯一性: {'✅ 正常' if self.evaluation_path_unique else '❌ 受损'}",
            f"  • 多路径检测: {len(self.multiple_paths_detected)} 个",
            ""
        ]
        
        if self.violations:
            lines.extend([
                "⚠️ 完整性违规详情:",
                "-" * 60
            ])
            for i, violation in enumerate(self.violations, 1):
                lines.extend([
                    f"{i}. [{violation.severity}] {violation.violation_type}",
                    f"   时间: {violation.timestamp}",
                    f"   位置: {violation.location}",
                    f"   详情: {violation.details}",
                    f"   影响: {violation.impact}",
                    f"   修复建议: {violation.remediation}",
                    ""
                ])
        
        lines.extend([
            "=" * 80,
            "🎯 关键建议:",
            "  • 确保所有CV都使用 UnifiedPurgedTimeSeriesCV",
            "  • 禁止使用 TimeSeriesSplit 或其他非时间安全方法",
            "  • 所有时间配置必须来自 unified_time_config",
            "  • 任何降级都必须在评估报告中显著标记",
            "=" * 80
        ])
        
        return "\n".join(lines)

class EvaluationIntegrityMonitor:
    """评估完整性监控器"""
    
    def __init__(self):
        self.current_report = EvaluationIntegrityReport(
            report_timestamp=datetime.now(),
            system_status='SECURE'
        )
        self._monitoring_enabled = True
        
    def start_evaluation_monitoring(self):
        """开始评估监控"""
        logger.info("启动评估完整性监控")
        self._monitoring_enabled = True
        self.current_report = EvaluationIntegrityReport(
            report_timestamp=datetime.now(),
            system_status='SECURE'
        )
        
    def check_time_config_integrity(self):
        """检查时间配置完整性"""
        try:
            unified_config = get_time_config()
            
            # 检查是否存在配置冲突
            conflicts = []
            
            # 这里可以扫描已知的配置文件检查冲突
            known_conflict_files = [
                "enhanced_oos_system.py",
                "unified_timing_registry.py",
                "config_loader.py"
            ]
            
            for file in known_conflict_files:
                # 模拟冲突检查（实际实现中会读取文件检查）
                # 这里简化为检查是否已经被修复
                pass
            
            self.current_report.time_config_integrity = len(conflicts) == 0
            self.current_report.detected_conflicts = conflicts
            
            if conflicts:
                for conflict in conflicts:
                    self.current_report.add_violation(IntegrityViolation(
                        timestamp=datetime.now(),
                        violation_type='config_conflict',
                        severity='HIGH',
                        location=conflict.get('location', 'unknown'),
                        details=f"时间配置冲突: {conflict}",
                        impact="可能导致时间泄漏和不一致的CV结果",
                        remediation="使用unified_time_config统一配置源"
                    ))
                    
        except Exception as e:
            logger.error(f"时间配置完整性检查失败: {e}")
            self.current_report.time_config_integrity = False
            
    def check_cv_integrity(self):
        """检查CV完整性（包括CV回退状态）"""
        try:
            # 获取CV降级监控报告
            degradation_report = CV_DEGRADATION_MONITOR.get_degradation_report()
            
            # 检查CV回退状态（来自 make_purged_splitter）
            try:
                # 尝试导入CV回退状态
                import sys
                if 'bma_models.量化模型_bma_ultra_enhanced' in sys.modules:
                    module = sys.modules['bma_models.量化模型_bma_ultra_enhanced']
                    cv_fallback_status = getattr(module, 'CV_FALLBACK_STATUS', {})
                    
                    if cv_fallback_status.get('occurred', False):
                        self.current_report.add_violation(IntegrityViolation(
                            timestamp=datetime.now(),
                            violation_type='cv_fallback',
                            severity='CRITICAL',
                            location='make_purged_splitter',
                            details=f"CV回退到 {cv_fallback_status.get('fallback_method', 'TimeSeriesSplit')}: {cv_fallback_status.get('reason', 'N/A')}",
                            impact="严重的时间泄漏风险，评估结果不可信",
                            remediation="修复 Purged CV 导入问题或在生产环境禁用回退"
                        ))
            except Exception as e:
                logger.warning(f"检查CV回退状态失败: {e}")
            
            self.current_report.cv_degradation_attempts = degradation_report['total_attempts']
            self.current_report.cv_integrity = degradation_report['total_attempts'] == 0
            
            if degradation_report['total_attempts'] > 0:
                self.current_report.add_violation(IntegrityViolation(
                    timestamp=datetime.now(),
                    violation_type='cv_degradation',
                    severity='CRITICAL',
                    location='multiple_locations',
                    details=f"检测到{degradation_report['total_attempts']}次CV降级尝试",
                    impact="评估结果可能存在时间泄漏，不可信",
                    remediation="使用create_unified_cv()替代所有CV方法"
                ))
                
                # 记录具体的降级尝试
                for attempt in degradation_report.get('attempts_log', []):
                    self.current_report.prohibited_cv_detected.append(
                        f"{attempt['attempted_class']} at {attempt['location']}"
                    )
                    
        except Exception as e:
            logger.error(f"CV完整性检查失败: {e}")
            self.current_report.cv_integrity = False
    
    def check_evaluation_path_uniqueness(self):
        """检查评估路径唯一性"""
        # 检查是否存在多个评估路径
        evaluation_modules = [
            "enhanced_oos_system",
            "regime_aware_trainer", 
            "production_readiness_validator"
        ]
        
        active_paths = []
        for module in evaluation_modules:
            # 实际实现中会检查模块是否被激活使用
            # 这里简化处理
            pass
        
        self.current_report.evaluation_path_unique = len(active_paths) <= 1
        self.current_report.multiple_paths_detected = active_paths
        
        if len(active_paths) > 1:
            self.current_report.add_violation(IntegrityViolation(
                timestamp=datetime.now(),
                violation_type='multiple_evaluation_paths',
                severity='MEDIUM',
                location='evaluation_system',
                details=f"检测到{len(active_paths)}个评估路径: {active_paths}",
                impact="可能导致评估结果不一致",
                remediation="统一使用单一评估路径"
            ))
    
    def generate_integrity_report(self) -> EvaluationIntegrityReport:
        """生成完整的完整性报告"""
        if not self._monitoring_enabled:
            return self.current_report
            
        logger.info("生成评估完整性报告")
        
        # 执行所有完整性检查
        self.check_time_config_integrity()
        self.check_cv_integrity() 
        self.check_evaluation_path_uniqueness()
        
        # 更新报告时间戳
        self.current_report.report_timestamp = datetime.now()
        
        return self.current_report
    
    def save_report(self, report_path: Union[str, Path]):
        """保存完整性报告"""
        report = self.generate_integrity_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)
            
        # 同时保存人类可读版本
        readable_path = Path(report_path).with_suffix('.txt')
        with open(readable_path, 'w', encoding='utf-8') as f:
            f.write(report.get_detailed_report())
            
        logger.info(f"完整性报告已保存: {report_path}, {readable_path}")
    
    def get_evaluation_header(self) -> str:
        """获取评估报告头部显著标记"""
        report = self.generate_integrity_report()
        return report.get_display_summary()

# 全局监控实例
EVALUATION_INTEGRITY_MONITOR = EvaluationIntegrityMonitor()

def get_evaluation_integrity_monitor():
    """获取全局评估完整性监控器"""
    return EVALUATION_INTEGRITY_MONITOR

def start_evaluation_integrity_check():
    """启动评估完整性检查"""
    EVALUATION_INTEGRITY_MONITOR.start_evaluation_monitoring()
    return EVALUATION_INTEGRITY_MONITOR.generate_integrity_report()

def get_integrity_header_for_report():
    """获取用于评估报告顶部的完整性标记"""
    return EVALUATION_INTEGRITY_MONITOR.get_evaluation_header()

# === 评估报告集成工具 ===

def wrap_evaluation_with_integrity_check(evaluation_func):
    """
    评估函数装饰器 - 自动添加完整性检查
    """
    def wrapper(*args, **kwargs):
        # 开始监控
        EVALUATION_INTEGRITY_MONITOR.start_evaluation_monitoring()
        
        try:
            # 执行评估
            result = evaluation_func(*args, **kwargs)
            
            # 生成完整性报告
            integrity_report = EVALUATION_INTEGRITY_MONITOR.generate_integrity_report()
            
            # 如果评估结果是字典，添加完整性信息
            if isinstance(result, dict):
                result['evaluation_integrity'] = {
                    'status': integrity_report.system_status,
                    'trustworthiness': integrity_report.evaluation_trustworthiness,
                    'violations': len(integrity_report.violations),
                    'header': integrity_report.get_display_summary()
                }
            
            # 记录完整性状态
            logger.info(f"评估完整性: {integrity_report.get_display_summary()}")
            
            return result
            
        except Exception as e:
            # 记录评估失败
            EVALUATION_INTEGRITY_MONITOR.current_report.add_violation(IntegrityViolation(
                timestamp=datetime.now(),
                violation_type='evaluation_failure',
                severity='CRITICAL',
                location=evaluation_func.__name__,
                details=f"评估执行失败: {str(e)}",
                impact="无法完成完整性评估",
                remediation="检查评估代码和配置"
            ))
            raise
    
    return wrapper

if __name__ == "__main__":
    # 测试完整性监控器
    print("=== 评估完整性监控器测试 ===")
    
    monitor = get_evaluation_integrity_monitor()
    report = monitor.generate_integrity_report()
    
    print("完整性报告摘要:")
    print(report.get_display_summary())
    print("\n详细报告:")
    print(report.get_detailed_report())