#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强生产闸门系统 - 修复OR逻辑漏洞和QLIKE方向混淆
实施严格的多维AND + 兜底OR逻辑
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from unified_timing_registry import get_global_timing_registry
except ImportError:
    get_global_timing_registry = lambda: None

logger = logging.getLogger(__name__)


@dataclass
class ProductionGateResult:
    """生产闸门验证结果"""
    passed: bool
    gate_type: str  # 'strict_and', 'fallback_or', 'failed'
    score: float
    details: Dict[str, Any]
    recommendation: str
    risk_level: str


class EnhancedProductionGate:
    """
    增强生产闸门系统
    
    实施严格的多维验证逻辑：
    1. 主要AND逻辑：所有关键指标必须达标
    2. 兜底OR逻辑：概率度量显著优秀时允许影子模式
    3. 修复QLIKE方向混淆问题
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.timing_registry = get_global_timing_registry()
        self.gate_params = self.timing_registry.get_production_gate_params()
        
        if config:
            self.gate_params.update(config)
        
        self.validation_history = []
        
        logger.info("Enhanced Production Gate initialized")
        logger.info(f"  严格AND阈值: RankIC≥{self.gate_params['min_rank_ic']}, t≥{self.gate_params['min_t_stat']}")
        logger.info(f"  覆盖期要求: ≥{self.gate_params['min_coverage_months']}个月")
        logger.info(f"  QLIKE改善要求: ≥{self.gate_params['min_qlike_reduction_pct']:.1%}")
    
    def validate_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证分析结果"""
        try:
            logger.info("开始验证分析结果...")
            
            # 提取关键指标
            model_metrics = analysis_results.get('model_metrics', {})
            coverage_months = analysis_results.get('coverage_months', 0)
            
            # 使用主要的生产验证方法
            if hasattr(self, 'validate_for_production'):
                result = self.validate_for_production(
                    model_metrics=model_metrics,
                    baseline_metrics=None,
                    coverage_months=coverage_months,
                    model_name=analysis_results.get('model_name', 'analysis_model')
                )
                
                # 转换为字典格式
                return {
                    'passed': result.passed,
                    'gate_type': result.gate_type,
                    'score': result.score,
                    'details': result.details,
                    'recommendation': result.recommendation,
                    'risk_level': result.risk_level
                }
            else:
                # 简化验证
                return {
                    'passed': True,
                    'gate_type': 'simplified',
                    'score': 0.8,
                    'details': {'validation_method': 'simplified'},
                    'recommendation': 'Proceed with monitoring',
                    'risk_level': 'medium'
                }
                
        except Exception as e:
            logger.warning(f"分析结果验证失败: {e}")
            return {
                'passed': False,
                'gate_type': 'failed',
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendation': 'Review analysis results',
                'risk_level': 'high'
            }
    
    def validate_for_production(
        self, 
        model_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        coverage_months: float = 0,
        model_name: str = "unnamed_model"
    ) -> ProductionGateResult:
        """
        生产闸门验证主方法
        
        Args:
            model_metrics: 模型性能指标
            baseline_metrics: 基准模型指标（用于对比）
            coverage_months: 覆盖月数
            model_name: 模型名称
            
        Returns:
            ProductionGateResult: 详细的验证结果
        """
        logger.info(f"开始生产闸门验证: {model_name}")
        logger.info(f"覆盖期: {coverage_months:.1f}个月")
        
        # 1. 严格AND逻辑验证
        strict_result = self._validate_strict_and_criteria(model_metrics, coverage_months)
        
        # 2. 如果严格验证通过，直接放行
        if strict_result['passed']:
            result = ProductionGateResult(
                passed=True,
                gate_type='strict_and',
                score=strict_result['score'],
                details=strict_result,
                recommendation="✅ 推荐立即投产 - 所有严格标准均达标",
                risk_level="LOW"
            )
            self._log_validation_result(result, model_name)
            return result
        
        # 3. 兜底OR逻辑验证（仅在有基准对比时）
        if baseline_metrics:
            fallback_result = self._validate_fallback_or_criteria(
                model_metrics, baseline_metrics, coverage_months
            )
            
            if fallback_result['passed']:
                result = ProductionGateResult(
                    passed=True,
                    gate_type='fallback_or',
                    score=fallback_result['score'],
                    details={**strict_result, **fallback_result},
                    recommendation="⚠️ 建议影子模式试运行 - 概率度量显著优秀但IC略低",
                    risk_level="MEDIUM"
                )
                self._log_validation_result(result, model_name)
                return result
        
        # 4. 全部验证失败
        result = ProductionGateResult(
            passed=False,
            gate_type='failed',
            score=strict_result.get('score', 0),
            details=strict_result,
            recommendation="❌ 不建议投产 - 关键指标未达标",
            risk_level="HIGH"
        )
        
        self._log_validation_result(result, model_name)
        return result
    
    def _validate_strict_and_criteria(
        self, 
        metrics: Dict[str, Any], 
        coverage_months: float
    ) -> Dict[str, Any]:
        """
        严格AND逻辑验证
        
        要求：
        1. RankIC_mean ≥ min_rank_ic
        2. |t_stat| ≥ min_t_stat  
        3. 覆盖月数 ≥ min_coverage_months
        4. 成本后收益为正（如果有成本数据）
        """
        checks = {}
        score_components = []
        
        # 1. RankIC检查
        rank_ic = metrics.get('rank_ic_mean', 0)
        rank_ic_required = self.gate_params['min_rank_ic']
        checks['rank_ic'] = {
            'value': rank_ic,
            'required': rank_ic_required,
            'passed': rank_ic >= rank_ic_required,
            'description': f"RankIC均值 {rank_ic:.4f} {'≥' if rank_ic >= rank_ic_required else '<'} {rank_ic_required:.4f}"
        }
        score_components.append(rank_ic / rank_ic_required if rank_ic_required > 0 else 0)
        
        # 2. t统计量检查
        t_stat = abs(metrics.get('rank_ic_t_stat', 0))
        t_stat_required = self.gate_params['min_t_stat']
        checks['t_stat'] = {
            'value': t_stat,
            'required': t_stat_required,
            'passed': t_stat >= t_stat_required,
            'description': f"t统计量 {t_stat:.2f} {'≥' if t_stat >= t_stat_required else '<'} {t_stat_required:.2f}"
        }
        score_components.append(min(t_stat / t_stat_required, 2.0) if t_stat_required > 0 else 0)
        
        # 3. 覆盖期检查
        coverage_required = self.gate_params['min_coverage_months']
        checks['coverage'] = {
            'value': coverage_months,
            'required': coverage_required,
            'passed': coverage_months >= coverage_required,
            'description': f"覆盖期 {coverage_months:.1f}月 {'≥' if coverage_months >= coverage_required else '<'} {coverage_required}月"
        }
        score_components.append(min(coverage_months / coverage_required, 2.0) if coverage_required > 0 else 0)
        
        # 4. 成本后收益检查（如果有数据）
        net_return = metrics.get('net_return_after_cost')
        if net_return is not None:
            checks['net_return'] = {
                'value': net_return,
                'required': 0.0,
                'passed': net_return > 0,
                'description': f"成本后收益 {net_return:.4f} {'>' if net_return > 0 else '≤'} 0"
            }
            score_components.append(max(net_return * 10, 0))  # 放大收益的权重
        
        # 计算综合得分
        overall_score = np.mean(score_components) if score_components else 0
        
        # 所有关键检查必须通过
        critical_checks = ['rank_ic', 't_stat', 'coverage']
        all_passed = all(checks[check]['passed'] for check in critical_checks if check in checks)
        
        # 如果有成本数据，成本后收益也必须为正
        if 'net_return' in checks:
            all_passed = all_passed and checks['net_return']['passed']
        
        return {
            'passed': all_passed,
            'score': overall_score,
            'checks': checks,
            'gate_type': 'strict_and'
        }
    
    def _validate_fallback_or_criteria(
        self, 
        model_metrics: Dict[str, Any], 
        baseline_metrics: Dict[str, Any],
        coverage_months: float
    ) -> Dict[str, Any]:
        """
        兜底OR逻辑验证
        
        条件：QLIKE/CRPS概率度量显著更优(>12%) 且 RankIC略低但≥0.015
        """
        checks = {}
        
        # 1. QLIKE改善检查 - 修复方向混淆
        model_qlike = model_metrics.get('qlike_error', float('inf'))
        baseline_qlike = baseline_metrics.get('qlike_error', float('inf'))
        
        if baseline_qlike > 0 and model_qlike < float('inf'):
            # QLIKE是误差指标，越低越好
            # reduction = (baseline - model) / baseline
            qlike_reduction = (baseline_qlike - model_qlike) / baseline_qlike
            qlike_required = self.gate_params['min_qlike_reduction_pct']
            
            checks['qlike_reduction'] = {
                'value': qlike_reduction,
                'required': qlike_required,
                'passed': qlike_reduction >= qlike_required,
                'description': f"QLIKE误差减少 {qlike_reduction:.1%} {'≥' if qlike_reduction >= qlike_required else '<'} {qlike_required:.1%}"
            }
        else:
            checks['qlike_reduction'] = {
                'value': 0,
                'required': self.gate_params['min_qlike_reduction_pct'],
                'passed': False,
                'description': "QLIKE数据不可用"
            }
        
        # 2. 宽松RankIC检查
        rank_ic = model_metrics.get('rank_ic_mean', 0)
        relaxed_rank_ic = 0.015  # 比严格标准(0.02)略低
        checks['relaxed_rank_ic'] = {
            'value': rank_ic,
            'required': relaxed_rank_ic,
            'passed': rank_ic >= relaxed_rank_ic,
            'description': f"宽松RankIC {rank_ic:.4f} {'≥' if rank_ic >= relaxed_rank_ic else '<'} {relaxed_rank_ic:.4f}"
        }
        
        # 3. 基本覆盖期要求（可以略低）
        min_coverage = max(6, self.gate_params['min_coverage_months'] * 0.75)  # 至少6个月
        checks['basic_coverage'] = {
            'value': coverage_months,
            'required': min_coverage,
            'passed': coverage_months >= min_coverage,
            'description': f"基本覆盖期 {coverage_months:.1f}月 {'≥' if coverage_months >= min_coverage else '<'} {min_coverage:.1f}月"
        }
        
        # 兜底逻辑：概率度量显著改善 + 基本IC要求 + 基本覆盖期
        fallback_passed = (
            checks['qlike_reduction']['passed'] and 
            checks['relaxed_rank_ic']['passed'] and 
            checks['basic_coverage']['passed']
        )
        
        # 计算兜底得分
        if fallback_passed:
            qlike_score = min(checks['qlike_reduction']['value'] / qlike_required, 2.0)
            ic_score = checks['relaxed_rank_ic']['value'] / relaxed_rank_ic
            coverage_score = min(coverage_months / min_coverage, 1.5)
            fallback_score = np.mean([qlike_score, ic_score, coverage_score])
        else:
            fallback_score = 0
        
        return {
            'passed': fallback_passed,
            'score': fallback_score,
            'checks': checks,
            'gate_type': 'fallback_or'
        }
    
    def _log_validation_result(self, result: ProductionGateResult, model_name: str):
        """记录验证结果"""
        self.validation_history.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'result': result
        })
        
        logger.info(f"=== 生产闸门验证结果: {model_name} ===")
        logger.info(f"通过状态: {'✅ PASS' if result.passed else '❌ FAIL'}")
        logger.info(f"验证类型: {result.gate_type}")
        logger.info(f"综合得分: {result.score:.3f}")
        logger.info(f"风险等级: {result.risk_level}")
        logger.info(f"建议: {result.recommendation}")
        
        # 详细检查结果
        if 'checks' in result.details:
            logger.info("详细检查结果:")
            for check_name, check_result in result.details['checks'].items():
                status = "✅" if check_result['passed'] else "❌"
                logger.info(f"  {status} {check_result['description']}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {'message': '暂无验证历史'}
        
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for v in self.validation_history if v['result'].passed)
        
        gate_type_counts = {}
        risk_level_counts = {}
        
        for v in self.validation_history:
            gate_type = v['result'].gate_type
            risk_level = v['result'].risk_level
            
            gate_type_counts[gate_type] = gate_type_counts.get(gate_type, 0) + 1
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
        
        return {
            'total_validations': total_validations,
            'passed_rate': passed_validations / total_validations,
            'gate_type_distribution': gate_type_counts,
            'risk_level_distribution': risk_level_counts,
            'recent_validations': [
                {
                    'model': v['model_name'],
                    'timestamp': v['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'passed': v['result'].passed,
                    'gate_type': v['result'].gate_type,
                    'score': v['result'].score
                }
                for v in self.validation_history[-5:]  # 最近5次
            ]
        }


def create_enhanced_production_gate(config: Optional[Dict[str, Any]] = None) -> EnhancedProductionGate:
    """创建增强生产闸门实例"""
    return EnhancedProductionGate(config)


if __name__ == "__main__":
    # 测试生产闸门
    gate = create_enhanced_production_gate()
    
    # 测试严格通过的情况
    good_metrics = {
        'rank_ic_mean': 0.025,
        'rank_ic_t_stat': 2.5,
        'net_return_after_cost': 0.08
    }
    
    result1 = gate.validate_for_production(good_metrics, coverage_months=15, model_name="good_model")
    print(f"Good model result: {result1.passed}, {result1.gate_type}")
    
    # 测试兜底逻辑的情况
    mediocre_metrics = {
        'rank_ic_mean': 0.018,  # 略低于严格要求
        'rank_ic_t_stat': 1.8,
        'qlike_error': 0.45
    }
    
    baseline_metrics = {
        'qlike_error': 0.55  # 基准QLIKE更高（更差）
    }
    
    result2 = gate.validate_for_production(
        mediocre_metrics, baseline_metrics, coverage_months=8, model_name="mediocre_model"
    )
    print(f"Mediocre model result: {result2.passed}, {result2.gate_type}")
    
    # 测试失败的情况
    bad_metrics = {
        'rank_ic_mean': 0.005,
        'rank_ic_t_stat': 0.8
    }
    
    result3 = gate.validate_for_production(bad_metrics, coverage_months=3, model_name="bad_model")
    print(f"Bad model result: {result3.passed}, {result3.gate_type}")