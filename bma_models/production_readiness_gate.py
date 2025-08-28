#!/usr/bin/env python3
"""
生产就绪闸门系统 - 严格的上线检查和Go/No-Go决策
====================================================
全指标必须同时通过（AND逻辑），连续NO_GO触发回滚
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class GateStatus(Enum):
    """闸门状态枚举"""
    GO = "GO"                          # 通过，可上线
    NO_GO = "NO_GO"                   # 不通过，禁止上线
    WARNING = "WARNING"               # 警告，需人工审核
    CRITICAL = "CRITICAL"             # 严重问题，强制回滚

@dataclass
class GateThreshold:
    """闸门阈值配置"""
    # IC相关阈值（必须同时满足）
    min_ic_mean: float = 0.02              # 最小IC均值
    min_rank_ic_mean: float = 0.02         # 最小RankIC均值
    min_ic_stability: float = 0.6          # 最小IC稳定性
    max_ic_volatility: float = 0.15        # 最大IC波动率
    
    # 预测质量阈值
    max_qlike_score: float = 0.5           # 最大QLIKE分数
    max_brier_score: float = 0.3           # 最大Brier分数
    min_prediction_coverage: float = 0.8   # 最小预测覆盖率
    min_cross_sectional_coverage: float = 0.7  # 最小横截面覆盖率
    
    # 交易成本阈值
    max_turnover_rate: float = 0.15        # 最大换手率
    max_transaction_cost: float = 0.002    # 最大交易成本
    min_liquidity_score: float = 0.6      # 最小流动性分数
    max_market_impact: float = 0.001       # 最大市场冲击
    
    # 稳定性阈值
    min_sharpe_ratio: float = 1.0          # 最小夏普比率
    max_max_drawdown: float = 0.05         # 最大回撤
    min_hit_rate: float = 0.52             # 最小胜率
    min_consistency_score: float = 0.7     # 最小一致性分数
    
    # 技术指标阈值
    min_feature_count: float = 15          # 最少特征数
    max_feature_correlation: float = 0.8   # 最大特征相关性
    min_data_freshness: float = 0.9        # 最小数据新鲜度
    max_missing_rate: float = 0.1          # 最大缺失率
    
    # 时间安全阈值
    min_temporal_gap_days: int = 10        # 最小时间间隔
    max_leakage_rate: float = 0.01         # 最大泄漏率
    min_oos_samples: int = 100             # 最小OOS样本数
    
    # 回滚触发阈值
    max_consecutive_no_go: int = 3         # 最大连续NO_GO次数
    max_critical_events: int = 1           # 最大严重事件数
    rollback_trigger_window: int = 7       # 回滚触发窗口（天）

@dataclass
class GateCheckResult:
    """单项检查结果"""
    check_name: str
    status: GateStatus
    actual_value: float
    threshold_value: float
    passed: bool
    message: str
    severity: str = "info"  # info/warning/error/critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'actual_value': self.actual_value,
            'threshold_value': self.threshold_value,
            'passed': self.passed,
            'message': self.message,
            'severity': self.severity
        }

@dataclass
class GateDecision:
    """闸门决策结果"""
    overall_status: GateStatus
    decision_timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    critical_checks: int
    
    individual_results: List[GateCheckResult] = field(default_factory=list)
    rollback_recommended: bool = False
    rollback_reason: Optional[str] = None
    next_review_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_status': self.overall_status.value,
            'decision_timestamp': self.decision_timestamp,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'warning_checks': self.warning_checks,
            'critical_checks': self.critical_checks,
            'individual_results': [r.to_dict() for r in self.individual_results],
            'rollback_recommended': self.rollback_recommended,
            'rollback_reason': self.rollback_reason,
            'next_review_time': self.next_review_time
        }

class ProductionReadinessGate:
    """生产就绪闸门系统"""
    
    def __init__(self, threshold_config: GateThreshold = None,
                 history_file: str = "gate_decisions.json"):
        """初始化闸门系统"""
        self.thresholds = threshold_config or GateThreshold()
        self.history_file = Path(history_file)
        
        # 加载历史决策记录
        self.decision_history = self._load_decision_history()
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,
            'go_decisions': 0,
            'no_go_decisions': 0,
            'warning_decisions': 0,
            'critical_decisions': 0,
            'rollback_recommendations': 0
        }
        
        logger.info(f"生产就绪闸门系统初始化完成 - 历史记录: {len(self.decision_history)}")
    
    def evaluate_readiness(self, model_metrics: Dict[str, Any],
                          system_metrics: Dict[str, Any],
                          market_conditions: Dict[str, Any] = None) -> GateDecision:
        """
        评估生产就绪度（主接口）
        
        Args:
            model_metrics: 模型性能指标
            system_metrics: 系统技术指标  
            market_conditions: 市场条件（可选）
            
        Returns:
            闸门决策结果
        """
        logger.info("开始生产就绪度评估")
        
        self.stats['total_evaluations'] += 1
        
        # 执行所有检查项
        check_results = []
        
        # 1. IC相关检查
        ic_results = self._check_ic_metrics(model_metrics)
        check_results.extend(ic_results)
        
        # 2. 预测质量检查
        prediction_results = self._check_prediction_quality(model_metrics)
        check_results.extend(prediction_results)
        
        # 3. 交易成本检查
        cost_results = self._check_trading_costs(model_metrics, system_metrics)
        check_results.extend(cost_results)
        
        # 4. 稳定性检查
        stability_results = self._check_stability_metrics(model_metrics)
        check_results.extend(stability_results)
        
        # 5. 技术指标检查
        technical_results = self._check_technical_metrics(system_metrics)
        check_results.extend(technical_results)
        
        # 6. 时间安全检查
        temporal_results = self._check_temporal_safety(system_metrics)
        check_results.extend(temporal_results)
        
        # 7. 市场条件检查（如果提供）
        if market_conditions:
            market_results = self._check_market_conditions(market_conditions)
            check_results.extend(market_results)
        
        # 分析检查结果
        decision = self._make_gate_decision(check_results)
        
        # 检查是否需要回滚
        decision.rollback_recommended, decision.rollback_reason = self._check_rollback_conditions(decision)
        
        # 保存决策历史
        self._save_decision(decision)
        
        # 更新统计
        self._update_stats(decision)
        
        logger.info(f"生产就绪度评估完成 - 决策: {decision.overall_status.value}")
        
        return decision
    
    def _check_ic_metrics(self, model_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查IC相关指标"""
        results = []
        
        # IC均值检查
        ic_mean = model_metrics.get('ic_mean', 0.0)
        results.append(GateCheckResult(
            check_name="IC均值",
            status=GateStatus.GO if ic_mean >= self.thresholds.min_ic_mean else GateStatus.NO_GO,
            actual_value=ic_mean,
            threshold_value=self.thresholds.min_ic_mean,
            passed=ic_mean >= self.thresholds.min_ic_mean,
            message=f"IC均值 {ic_mean:.4f} {'≥' if ic_mean >= self.thresholds.min_ic_mean else '<'} {self.thresholds.min_ic_mean}",
            severity="critical" if ic_mean < self.thresholds.min_ic_mean * 0.5 else "error" if ic_mean < self.thresholds.min_ic_mean else "info"
        ))
        
        # RankIC均值检查
        rank_ic_mean = model_metrics.get('rank_ic_mean', 0.0)
        results.append(GateCheckResult(
            check_name="RankIC均值",
            status=GateStatus.GO if rank_ic_mean >= self.thresholds.min_rank_ic_mean else GateStatus.NO_GO,
            actual_value=rank_ic_mean,
            threshold_value=self.thresholds.min_rank_ic_mean,
            passed=rank_ic_mean >= self.thresholds.min_rank_ic_mean,
            message=f"RankIC均值 {rank_ic_mean:.4f} {'≥' if rank_ic_mean >= self.thresholds.min_rank_ic_mean else '<'} {self.thresholds.min_rank_ic_mean}",
            severity="critical" if rank_ic_mean < 0 else "error" if rank_ic_mean < self.thresholds.min_rank_ic_mean else "info"
        ))
        
        # IC稳定性检查
        ic_stability = model_metrics.get('ic_stability', 0.0)
        results.append(GateCheckResult(
            check_name="IC稳定性",
            status=GateStatus.GO if ic_stability >= self.thresholds.min_ic_stability else GateStatus.NO_GO,
            actual_value=ic_stability,
            threshold_value=self.thresholds.min_ic_stability,
            passed=ic_stability >= self.thresholds.min_ic_stability,
            message=f"IC稳定性 {ic_stability:.4f} {'≥' if ic_stability >= self.thresholds.min_ic_stability else '<'} {self.thresholds.min_ic_stability}",
            severity="warning" if ic_stability < self.thresholds.min_ic_stability else "info"
        ))
        
        # IC波动率检查
        ic_volatility = model_metrics.get('ic_volatility', float('inf'))
        results.append(GateCheckResult(
            check_name="IC波动率",
            status=GateStatus.GO if ic_volatility <= self.thresholds.max_ic_volatility else GateStatus.WARNING,
            actual_value=ic_volatility,
            threshold_value=self.thresholds.max_ic_volatility,
            passed=ic_volatility <= self.thresholds.max_ic_volatility,
            message=f"IC波动率 {ic_volatility:.4f} {'≤' if ic_volatility <= self.thresholds.max_ic_volatility else '>'} {self.thresholds.max_ic_volatility}",
            severity="warning" if ic_volatility > self.thresholds.max_ic_volatility else "info"
        ))
        
        return results
    
    def _check_prediction_quality(self, model_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查预测质量指标"""
        results = []
        
        # QLIKE分数检查
        qlike_score = model_metrics.get('qlike_score', float('inf'))
        results.append(GateCheckResult(
            check_name="QLIKE分数",
            status=GateStatus.GO if qlike_score <= self.thresholds.max_qlike_score else GateStatus.NO_GO,
            actual_value=qlike_score,
            threshold_value=self.thresholds.max_qlike_score,
            passed=qlike_score <= self.thresholds.max_qlike_score,
            message=f"QLIKE分数 {qlike_score:.4f} {'≤' if qlike_score <= self.thresholds.max_qlike_score else '>'} {self.thresholds.max_qlike_score}",
            severity="error" if qlike_score > self.thresholds.max_qlike_score else "info"
        ))
        
        # Brier分数检查
        brier_score = model_metrics.get('brier_score', float('inf'))
        results.append(GateCheckResult(
            check_name="Brier分数",
            status=GateStatus.GO if brier_score <= self.thresholds.max_brier_score else GateStatus.WARNING,
            actual_value=brier_score,
            threshold_value=self.thresholds.max_brier_score,
            passed=brier_score <= self.thresholds.max_brier_score,
            message=f"Brier分数 {brier_score:.4f} {'≤' if brier_score <= self.thresholds.max_brier_score else '>'} {self.thresholds.max_brier_score}",
            severity="warning" if brier_score > self.thresholds.max_brier_score else "info"
        ))
        
        # 预测覆盖率检查
        coverage_rate = model_metrics.get('prediction_coverage', 0.0)
        results.append(GateCheckResult(
            check_name="预测覆盖率",
            status=GateStatus.GO if coverage_rate >= self.thresholds.min_prediction_coverage else GateStatus.NO_GO,
            actual_value=coverage_rate,
            threshold_value=self.thresholds.min_prediction_coverage,
            passed=coverage_rate >= self.thresholds.min_prediction_coverage,
            message=f"预测覆盖率 {coverage_rate:.2%} {'≥' if coverage_rate >= self.thresholds.min_prediction_coverage else '<'} {self.thresholds.min_prediction_coverage:.2%}",
            severity="critical" if coverage_rate < 0.5 else "error" if coverage_rate < self.thresholds.min_prediction_coverage else "info"
        ))
        
        return results
    
    def _check_trading_costs(self, model_metrics: Dict[str, Any], 
                           system_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查交易成本指标"""
        results = []
        
        # 换手率检查
        turnover_rate = model_metrics.get('turnover_rate', float('inf'))
        results.append(GateCheckResult(
            check_name="换手率",
            status=GateStatus.GO if turnover_rate <= self.thresholds.max_turnover_rate else GateStatus.WARNING,
            actual_value=turnover_rate,
            threshold_value=self.thresholds.max_turnover_rate,
            passed=turnover_rate <= self.thresholds.max_turnover_rate,
            message=f"换手率 {turnover_rate:.2%} {'≤' if turnover_rate <= self.thresholds.max_turnover_rate else '>'} {self.thresholds.max_turnover_rate:.2%}",
            severity="warning" if turnover_rate > self.thresholds.max_turnover_rate else "info"
        ))
        
        # 交易成本检查
        transaction_cost = model_metrics.get('transaction_cost', float('inf'))
        results.append(GateCheckResult(
            check_name="交易成本",
            status=GateStatus.GO if transaction_cost <= self.thresholds.max_transaction_cost else GateStatus.WARNING,
            actual_value=transaction_cost,
            threshold_value=self.thresholds.max_transaction_cost,
            passed=transaction_cost <= self.thresholds.max_transaction_cost,
            message=f"交易成本 {transaction_cost:.4f} {'≤' if transaction_cost <= self.thresholds.max_transaction_cost else '>'} {self.thresholds.max_transaction_cost:.4f}",
            severity="warning" if transaction_cost > self.thresholds.max_transaction_cost else "info"
        ))
        
        # 流动性分数检查
        liquidity_score = system_metrics.get('liquidity_score', 0.0)
        results.append(GateCheckResult(
            check_name="流动性分数",
            status=GateStatus.GO if liquidity_score >= self.thresholds.min_liquidity_score else GateStatus.WARNING,
            actual_value=liquidity_score,
            threshold_value=self.thresholds.min_liquidity_score,
            passed=liquidity_score >= self.thresholds.min_liquidity_score,
            message=f"流动性分数 {liquidity_score:.4f} {'≥' if liquidity_score >= self.thresholds.min_liquidity_score else '<'} {self.thresholds.min_liquidity_score}",
            severity="warning" if liquidity_score < self.thresholds.min_liquidity_score else "info"
        ))
        
        return results
    
    def _check_stability_metrics(self, model_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查稳定性指标"""
        results = []
        
        # 夏普比率检查
        sharpe_ratio = model_metrics.get('sharpe_ratio', 0.0)
        results.append(GateCheckResult(
            check_name="夏普比率",
            status=GateStatus.GO if sharpe_ratio >= self.thresholds.min_sharpe_ratio else GateStatus.NO_GO,
            actual_value=sharpe_ratio,
            threshold_value=self.thresholds.min_sharpe_ratio,
            passed=sharpe_ratio >= self.thresholds.min_sharpe_ratio,
            message=f"夏普比率 {sharpe_ratio:.4f} {'≥' if sharpe_ratio >= self.thresholds.min_sharpe_ratio else '<'} {self.thresholds.min_sharpe_ratio}",
            severity="critical" if sharpe_ratio < 0 else "error" if sharpe_ratio < self.thresholds.min_sharpe_ratio else "info"
        ))
        
        # 最大回撤检查
        max_drawdown = model_metrics.get('max_drawdown', float('inf'))
        results.append(GateCheckResult(
            check_name="最大回撤",
            status=GateStatus.GO if max_drawdown <= self.thresholds.max_max_drawdown else GateStatus.WARNING,
            actual_value=max_drawdown,
            threshold_value=self.thresholds.max_max_drawdown,
            passed=max_drawdown <= self.thresholds.max_max_drawdown,
            message=f"最大回撤 {max_drawdown:.2%} {'≤' if max_drawdown <= self.thresholds.max_max_drawdown else '>'} {self.thresholds.max_max_drawdown:.2%}",
            severity="critical" if max_drawdown > 0.1 else "warning" if max_drawdown > self.thresholds.max_max_drawdown else "info"
        ))
        
        # 胜率检查
        hit_rate = model_metrics.get('hit_rate', 0.0)
        results.append(GateCheckResult(
            check_name="胜率",
            status=GateStatus.GO if hit_rate >= self.thresholds.min_hit_rate else GateStatus.WARNING,
            actual_value=hit_rate,
            threshold_value=self.thresholds.min_hit_rate,
            passed=hit_rate >= self.thresholds.min_hit_rate,
            message=f"胜率 {hit_rate:.2%} {'≥' if hit_rate >= self.thresholds.min_hit_rate else '<'} {self.thresholds.min_hit_rate:.2%}",
            severity="warning" if hit_rate < self.thresholds.min_hit_rate else "info"
        ))
        
        return results
    
    def _check_technical_metrics(self, system_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查技术指标"""
        results = []
        
        # 特征数量检查
        feature_count = system_metrics.get('feature_count', 0)
        results.append(GateCheckResult(
            check_name="特征数量",
            status=GateStatus.GO if feature_count >= self.thresholds.min_feature_count else GateStatus.NO_GO,
            actual_value=feature_count,
            threshold_value=self.thresholds.min_feature_count,
            passed=feature_count >= self.thresholds.min_feature_count,
            message=f"特征数量 {feature_count} {'≥' if feature_count >= self.thresholds.min_feature_count else '<'} {self.thresholds.min_feature_count}",
            severity="critical" if feature_count < 10 else "error" if feature_count < self.thresholds.min_feature_count else "info"
        ))
        
        # 数据新鲜度检查
        data_freshness = system_metrics.get('data_freshness', 0.0)
        results.append(GateCheckResult(
            check_name="数据新鲜度",
            status=GateStatus.GO if data_freshness >= self.thresholds.min_data_freshness else GateStatus.WARNING,
            actual_value=data_freshness,
            threshold_value=self.thresholds.min_data_freshness,
            passed=data_freshness >= self.thresholds.min_data_freshness,
            message=f"数据新鲜度 {data_freshness:.2%} {'≥' if data_freshness >= self.thresholds.min_data_freshness else '<'} {self.thresholds.min_data_freshness:.2%}",
            severity="critical" if data_freshness < 0.7 else "warning" if data_freshness < self.thresholds.min_data_freshness else "info"
        ))
        
        return results
    
    def _check_temporal_safety(self, system_metrics: Dict[str, Any]) -> List[GateCheckResult]:
        """检查时间安全性"""
        results = []
        
        # 时间间隔检查
        temporal_gap = system_metrics.get('temporal_gap_days', 0)
        results.append(GateCheckResult(
            check_name="时间间隔",
            status=GateStatus.GO if temporal_gap >= self.thresholds.min_temporal_gap_days else GateStatus.CRITICAL,
            actual_value=temporal_gap,
            threshold_value=self.thresholds.min_temporal_gap_days,
            passed=temporal_gap >= self.thresholds.min_temporal_gap_days,
            message=f"时间间隔 {temporal_gap}天 {'≥' if temporal_gap >= self.thresholds.min_temporal_gap_days else '<'} {self.thresholds.min_temporal_gap_days}天",
            severity="critical" if temporal_gap < self.thresholds.min_temporal_gap_days else "info"
        ))
        
        # 泄漏率检查
        leakage_rate = system_metrics.get('leakage_rate', float('inf'))
        results.append(GateCheckResult(
            check_name="信息泄漏率",
            status=GateStatus.GO if leakage_rate <= self.thresholds.max_leakage_rate else GateStatus.CRITICAL,
            actual_value=leakage_rate,
            threshold_value=self.thresholds.max_leakage_rate,
            passed=leakage_rate <= self.thresholds.max_leakage_rate,
            message=f"信息泄漏率 {leakage_rate:.2%} {'≤' if leakage_rate <= self.thresholds.max_leakage_rate else '>'} {self.thresholds.max_leakage_rate:.2%}",
            severity="critical" if leakage_rate > self.thresholds.max_leakage_rate else "info"
        ))
        
        return results
    
    def _check_market_conditions(self, market_conditions: Dict[str, Any]) -> List[GateCheckResult]:
        """检查市场条件（可选）"""
        results = []
        
        # 市场波动率检查
        market_volatility = market_conditions.get('market_volatility', 0.0)
        # 高波动期间降低阈值
        volatility_threshold = 0.3 if market_volatility > 0.25 else 0.2
        
        results.append(GateCheckResult(
            check_name="市场波动率适应性",
            status=GateStatus.GO if market_volatility <= volatility_threshold else GateStatus.WARNING,
            actual_value=market_volatility,
            threshold_value=volatility_threshold,
            passed=market_volatility <= volatility_threshold,
            message=f"市场波动率 {market_volatility:.2%} {'≤' if market_volatility <= volatility_threshold else '>'} {volatility_threshold:.2%}",
            severity="warning" if market_volatility > volatility_threshold else "info"
        ))
        
        return results
    
    def _make_gate_decision(self, check_results: List[GateCheckResult]) -> GateDecision:
        """根据检查结果做出闸门决策"""
        # 统计各类结果
        total_checks = len(check_results)
        passed_checks = sum(1 for r in check_results if r.passed)
        failed_checks = total_checks - passed_checks
        warning_checks = sum(1 for r in check_results if r.status == GateStatus.WARNING)
        critical_checks = sum(1 for r in check_results if r.status == GateStatus.CRITICAL)
        
        # 决策逻辑：所有检查必须通过（AND逻辑）
        if critical_checks > 0:
            overall_status = GateStatus.CRITICAL
        elif failed_checks > 0:
            overall_status = GateStatus.NO_GO
        elif warning_checks > 0:
            overall_status = GateStatus.WARNING
        else:
            overall_status = GateStatus.GO
        
        # 计算下次审查时间
        if overall_status == GateStatus.GO:
            next_review = datetime.now() + timedelta(days=1)  # 通过后第二天再审查
        elif overall_status == GateStatus.WARNING:
            next_review = datetime.now() + timedelta(hours=6)  # 警告状态6小时后再审查
        else:
            next_review = datetime.now() + timedelta(hours=1)  # 失败状态1小时后再审查
        
        decision = GateDecision(
            overall_status=overall_status,
            decision_timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            critical_checks=critical_checks,
            individual_results=check_results,
            next_review_time=next_review.isoformat()
        )
        
        return decision
    
    def _check_rollback_conditions(self, current_decision: GateDecision) -> Tuple[bool, Optional[str]]:
        """检查是否需要触发回滚"""
        if len(self.decision_history) < self.thresholds.max_consecutive_no_go:
            return False, None
        
        # 检查最近的决策历史
        recent_decisions = self.decision_history[-self.thresholds.max_consecutive_no_go:]
        
        # 连续NO_GO检查
        consecutive_no_go = all(
            d['overall_status'] in ['NO_GO', 'CRITICAL'] 
            for d in recent_decisions
        )
        
        if consecutive_no_go and current_decision.overall_status in [GateStatus.NO_GO, GateStatus.CRITICAL]:
            return True, f"连续 {self.thresholds.max_consecutive_no_go + 1} 次NO_GO决策"
        
        # 严重事件检查
        recent_critical_count = sum(
            1 for d in recent_decisions 
            if d['overall_status'] == 'CRITICAL'
        )
        
        if recent_critical_count >= self.thresholds.max_critical_events:
            return True, f"在 {self.thresholds.rollback_trigger_window} 天内出现 {recent_critical_count} 次严重事件"
        
        return False, None
    
    def _save_decision(self, decision: GateDecision):
        """保存决策历史"""
        self.decision_history.append(decision.to_dict())
        
        # 保持历史记录大小
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
        
        # 保存到文件
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.decision_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存决策历史失败: {e}")
    
    def _load_decision_history(self) -> List[Dict[str, Any]]:
        """加载决策历史"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载决策历史失败: {e}")
        
        return []
    
    def _update_stats(self, decision: GateDecision):
        """更新统计信息"""
        if decision.overall_status == GateStatus.GO:
            self.stats['go_decisions'] += 1
        elif decision.overall_status == GateStatus.NO_GO:
            self.stats['no_go_decisions'] += 1
        elif decision.overall_status == GateStatus.WARNING:
            self.stats['warning_decisions'] += 1
        elif decision.overall_status == GateStatus.CRITICAL:
            self.stats['critical_decisions'] += 1
        
        if decision.rollback_recommended:
            self.stats['rollback_recommendations'] += 1
    
    def get_gate_stats(self) -> Dict[str, Any]:
        """获取闸门统计信息"""
        total_decisions = self.stats['total_evaluations']
        
        return {
            'gate_stats': self.stats,
            'decision_rates': {
                'go_rate': self.stats['go_decisions'] / max(1, total_decisions),
                'no_go_rate': self.stats['no_go_decisions'] / max(1, total_decisions),
                'warning_rate': self.stats['warning_decisions'] / max(1, total_decisions),
                'critical_rate': self.stats['critical_decisions'] / max(1, total_decisions),
                'rollback_rate': self.stats['rollback_recommendations'] / max(1, total_decisions)
            },
            'thresholds': self.thresholds.__dict__,
            'recent_decisions': self.decision_history[-10:] if self.decision_history else []
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        """更新闸门阈值"""
        for key, value in new_thresholds.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"更新闸门阈值: {key} = {value}")
            else:
                logger.warning(f"未知闸门阈值参数: {key}")

# 全局生产就绪闸门
def create_production_readiness_gate(threshold_config: GateThreshold = None) -> ProductionReadinessGate:
    """创建生产就绪闸门"""
    return ProductionReadinessGate(threshold_config)

if __name__ == "__main__":
    # 测试生产就绪闸门
    gate = create_production_readiness_gate()
    
    # 模拟模型指标
    mock_model_metrics = {
        'ic_mean': 0.025,
        'rank_ic_mean': 0.028,
        'ic_stability': 0.65,
        'ic_volatility': 0.12,
        'qlike_score': 0.45,
        'brier_score': 0.25,
        'prediction_coverage': 0.85,
        'turnover_rate': 0.12,
        'transaction_cost': 0.0015,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.04,
        'hit_rate': 0.54
    }
    
    # 模拟系统指标
    mock_system_metrics = {
        'feature_count': 18,
        'liquidity_score': 0.7,
        'data_freshness': 0.92,
        'temporal_gap_days': 10,
        'leakage_rate': 0.005
    }
    
    # 评估生产就绪度
    decision = gate.evaluate_readiness(mock_model_metrics, mock_system_metrics)
    
    print("=== 生产就绪闸门测试 ===")
    print(f"总体状态: {decision.overall_status.value}")
    print(f"通过检查: {decision.passed_checks}/{decision.total_checks}")
    print(f"失败检查: {decision.failed_checks}")
    print(f"警告检查: {decision.warning_checks}")
    print(f"严重检查: {decision.critical_checks}")
    print(f"回滚建议: {'是' if decision.rollback_recommended else '否'}")
    
    if decision.rollback_recommended:
        print(f"回滚原因: {decision.rollback_reason}")
    
    print(f"\n未通过的检查项:")
    for result in decision.individual_results:
        if not result.passed:
            print(f"  - {result.check_name}: {result.message} ({result.severity})")
    
    print(f"\n闸门统计:")
    stats = gate.get_gate_stats()
    for key, value in stats['decision_rates'].items():
        print(f"  {key}: {value:.2%}")