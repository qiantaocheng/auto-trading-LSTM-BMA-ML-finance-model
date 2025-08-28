"""
实时性能监控系统
================
监控因子表现、模型性能和系统健康状态
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    
    # IC指标
    rank_ic: float = 0.0
    ic_stability: float = 0.0
    ic_decay: float = 0.0  # IC衰减速度
    
    # 收益指标
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    value_at_risk: float = 0.0  # VaR
    conditional_var: float = 0.0  # CVaR
    
    # 换手率指标
    turnover: float = 0.0
    turnover_cost: float = 0.0
    
    # 因子指标
    factor_coverage: float = 0.0  # 因子覆盖率
    factor_stability: float = 0.0  # 因子稳定性
    factor_crowding: float = 0.0  # 因子拥挤度
    
    # 系统指标
    latency_ms: float = 0.0  # 延迟
    memory_usage_mb: float = 0.0  # 内存使用
    cpu_usage_pct: float = 0.0  # CPU使用率


@dataclass
class AlertThresholds:
    """警报阈值配置"""
    
    # IC阈值
    min_rank_ic: float = 0.01
    min_ic_stability: float = 0.5
    max_ic_decay_rate: float = 0.5  # 50%衰减警报
    
    # 风险阈值
    max_drawdown: float = 0.15  # 15%最大回撤
    max_volatility: float = 0.30  # 30%年化波动率
    min_sharpe: float = 0.5
    
    # 换手率阈值
    max_turnover: float = 0.50  # 50%日换手率
    max_turnover_cost: float = 0.005  # 0.5%成本
    
    # 因子阈值
    min_factor_coverage: float = 0.80  # 80%覆盖率
    max_factor_crowding: float = 0.80  # 80%拥挤度
    
    # 系统阈值
    max_latency_ms: float = 100
    max_memory_mb: float = 4096
    max_cpu_pct: float = 80


class RealtimePerformanceMonitor:
    """
    实时性能监控器
    跟踪和分析交易系统的实时表现
    """
    
    def __init__(self, 
                 window_size: int = 252,
                 alert_thresholds: Optional[AlertThresholds] = None):
        """
        Args:
            window_size: 监控窗口大小（天）
            alert_thresholds: 警报阈值配置
        """
        self.window_size = window_size
        self.thresholds = alert_thresholds or AlertThresholds()
        
        # 性能历史记录
        self.metrics_history = deque(maxlen=window_size)
        self.alerts_history = []
        
        # 因子性能跟踪
        self.factor_performance = {}
        
        # 实时统计
        self.current_metrics = None
        self.last_update_time = None
        
        # 监控状态
        self.is_monitoring = False
        self.alert_callbacks = []
        
    def update_metrics(self, 
                      predictions: pd.DataFrame,
                      actual_returns: pd.Series,
                      factor_data: pd.DataFrame,
                      system_stats: Optional[Dict] = None):
        """
        更新性能指标
        
        Args:
            predictions: 模型预测
            actual_returns: 实际收益
            factor_data: 因子数据
            system_stats: 系统统计信息
        """
        timestamp = datetime.now()
        
        # 计算IC指标
        ic_metrics = self._calculate_ic_metrics(predictions, actual_returns)
        
        # 计算收益指标
        return_metrics = self._calculate_return_metrics(actual_returns)
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(actual_returns)
        
        # 计算换手率
        turnover_metrics = self._calculate_turnover_metrics(predictions)
        
        # 计算因子指标
        factor_metrics = self._calculate_factor_metrics(factor_data)
        
        # 系统指标
        if system_stats is None:
            system_stats = self._get_system_stats()
        
        # 创建指标对象
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            **ic_metrics,
            **return_metrics,
            **risk_metrics,
            **turnover_metrics,
            **factor_metrics,
            **system_stats
        )
        
        # 更新历史
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        self.last_update_time = timestamp
        
        # 检查警报
        self._check_alerts(metrics)
        
        # 更新因子性能
        self._update_factor_performance(factor_data, actual_returns)
        
        logger.info(f"性能指标已更新: IC={metrics.rank_ic:.4f}, Sharpe={metrics.sharpe_ratio:.2f}")
    
    def _calculate_ic_metrics(self, predictions: pd.DataFrame, returns: pd.Series) -> Dict:
        """计算IC相关指标"""
        try:
            # Rank IC
            from scipy import stats
            rank_ic, _ = stats.spearmanr(predictions.values.flatten(), returns.values)
            
            # IC稳定性（最近20天的IC标准差的倒数）
            recent_ics = []
            for i in range(min(20, len(self.metrics_history))):
                if i < len(self.metrics_history):
                    recent_ics.append(self.metrics_history[-(i+1)].rank_ic)
            
            if len(recent_ics) > 1:
                ic_std = np.std(recent_ics)
                ic_stability = 1.0 / (ic_std + 0.01)  # 避免除零
            else:
                ic_stability = 1.0
            
            # IC衰减（相对于历史平均）
            if len(recent_ics) > 0:
                historical_avg = np.mean(recent_ics)
                ic_decay = (rank_ic - historical_avg) / (abs(historical_avg) + 0.01)
            else:
                ic_decay = 0.0
            
            return {
                'rank_ic': rank_ic,
                'ic_stability': ic_stability,
                'ic_decay': ic_decay
            }
        except Exception as e:
            logger.warning(f"IC计算失败: {e}")
            return {'rank_ic': 0, 'ic_stability': 0, 'ic_decay': 0}
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """计算收益相关指标"""
        try:
            daily_return = returns.iloc[-1] if len(returns) > 0 else 0
            cumulative_return = (1 + returns).prod() - 1
            
            # Sharpe ratio (年化)
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
            else:
                sharpe_ratio = 0
            
            # Calmar ratio
            max_dd = self._calculate_max_drawdown(returns)
            if max_dd < -0.01:
                annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
                calmar_ratio = annualized_return / abs(max_dd)
            else:
                calmar_ratio = 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino_ratio = np.sqrt(252) * returns.mean() / (downside_std + 1e-8)
            else:
                sortino_ratio = sharpe_ratio
            
            return {
                'daily_return': daily_return,
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio
            }
        except Exception as e:
            logger.warning(f"收益指标计算失败: {e}")
            return {
                'daily_return': 0,
                'cumulative_return': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0
            }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """计算风险指标"""
        try:
            # 波动率（年化）
            volatility = returns.std() * np.sqrt(252)
            
            # 最大回撤
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # VaR (95%置信度)
            value_at_risk = np.percentile(returns, 5)
            
            # CVaR (Expected Shortfall)
            var_threshold = value_at_risk
            conditional_var = returns[returns <= var_threshold].mean() if len(returns[returns <= var_threshold]) > 0 else value_at_risk
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'value_at_risk': value_at_risk,
                'conditional_var': conditional_var
            }
        except Exception as e:
            logger.warning(f"风险指标计算失败: {e}")
            return {
                'volatility': 0,
                'max_drawdown': 0,
                'value_at_risk': 0,
                'conditional_var': 0
            }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_turnover_metrics(self, predictions: pd.DataFrame) -> Dict:
        """计算换手率指标"""
        try:
            if len(self.metrics_history) > 0:
                # 获取前一期预测
                # 这里简化处理，实际应该保存预测历史
                turnover = 0.2  # 示例值
                turnover_cost = turnover * 0.001  # 假设10bps成本
            else:
                turnover = 0.0
                turnover_cost = 0.0
            
            return {
                'turnover': turnover,
                'turnover_cost': turnover_cost
            }
        except Exception as e:
            logger.warning(f"换手率计算失败: {e}")
            return {'turnover': 0, 'turnover_cost': 0}
    
    def _calculate_factor_metrics(self, factor_data: pd.DataFrame) -> Dict:
        """计算因子指标"""
        try:
            # 因子覆盖率（非空值比例）
            factor_coverage = 1 - factor_data.isna().sum().sum() / (factor_data.shape[0] * factor_data.shape[1])
            
            # 因子稳定性（相关性的时间稳定性）
            if len(self.factor_performance) > 0:
                stability_scores = []
                for factor in factor_data.columns:
                    if factor in self.factor_performance:
                        recent_corrs = self.factor_performance[factor][-20:]
                        if len(recent_corrs) > 1:
                            stability = 1 / (np.std(recent_corrs) + 0.01)
                            stability_scores.append(stability)
                
                factor_stability = np.mean(stability_scores) if stability_scores else 1.0
            else:
                factor_stability = 1.0
            
            # 因子拥挤度（因子间平均相关性）
            corr_matrix = factor_data.corr().abs()
            n = len(corr_matrix)
            if n > 1:
                off_diagonal_sum = corr_matrix.sum().sum() - n  # 减去对角线
                factor_crowding = off_diagonal_sum / (n * (n - 1))
            else:
                factor_crowding = 0.0
            
            return {
                'factor_coverage': factor_coverage,
                'factor_stability': factor_stability,
                'factor_crowding': factor_crowding
            }
        except Exception as e:
            logger.warning(f"因子指标计算失败: {e}")
            return {
                'factor_coverage': 0,
                'factor_stability': 0,
                'factor_crowding': 0
            }
    
    def _get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        try:
            import psutil
            import time
            
            # CPU使用率
            cpu_usage_pct = psutil.cpu_percent(interval=0.1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # 延迟（模拟）
            latency_ms = 0.0  # 实际应该测量
            
            return {
                'latency_ms': latency_ms,
                'memory_usage_mb': memory_usage_mb,
                'cpu_usage_pct': cpu_usage_pct
            }
        except ImportError:
            # psutil未安装
            return {
                'latency_ms': 0,
                'memory_usage_mb': 0,
                'cpu_usage_pct': 0
            }
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """检查并触发警报"""
        alerts = []
        
        # IC警报
        if metrics.rank_ic < self.thresholds.min_rank_ic:
            alerts.append(('CRITICAL', f'Rank IC过低: {metrics.rank_ic:.4f} < {self.thresholds.min_rank_ic}'))
        
        if metrics.ic_stability < self.thresholds.min_ic_stability:
            alerts.append(('WARNING', f'IC稳定性不足: {metrics.ic_stability:.2f} < {self.thresholds.min_ic_stability}'))
        
        # 风险警报
        if abs(metrics.max_drawdown) > self.thresholds.max_drawdown:
            alerts.append(('CRITICAL', f'最大回撤超限: {abs(metrics.max_drawdown):.1%} > {self.thresholds.max_drawdown:.1%}'))
        
        if metrics.volatility > self.thresholds.max_volatility:
            alerts.append(('WARNING', f'波动率过高: {metrics.volatility:.1%} > {self.thresholds.max_volatility:.1%}'))
        
        if metrics.sharpe_ratio < self.thresholds.min_sharpe:
            alerts.append(('WARNING', f'夏普比率过低: {metrics.sharpe_ratio:.2f} < {self.thresholds.min_sharpe}'))
        
        # 换手率警报
        if metrics.turnover > self.thresholds.max_turnover:
            alerts.append(('WARNING', f'换手率过高: {metrics.turnover:.1%} > {self.thresholds.max_turnover:.1%}'))
        
        # 因子警报
        if metrics.factor_crowding > self.thresholds.max_factor_crowding:
            alerts.append(('WARNING', f'因子拥挤度过高: {metrics.factor_crowding:.2f} > {self.thresholds.max_factor_crowding}'))
        
        # 系统警报
        if metrics.latency_ms > self.thresholds.max_latency_ms:
            alerts.append(('WARNING', f'系统延迟过高: {metrics.latency_ms:.0f}ms > {self.thresholds.max_latency_ms}ms'))
        
        # 触发警报
        for level, message in alerts:
            self._trigger_alert(level, message)
    
    def _trigger_alert(self, level: str, message: str):
        """触发警报"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        
        self.alerts_history.append(alert)
        
        # 记录日志
        if level == 'CRITICAL':
            logger.critical(f"性能警报: {message}")
        else:
            logger.warning(f"性能警报: {message}")
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警报回调失败: {e}")
    
    def _update_factor_performance(self, factor_data: pd.DataFrame, returns: pd.Series):
        """更新因子性能跟踪"""
        for factor in factor_data.columns:
            if factor not in self.factor_performance:
                self.factor_performance[factor] = deque(maxlen=252)
            
            # 计算因子与收益的相关性
            try:
                corr = factor_data[factor].corr(returns)
                self.factor_performance[factor].append(corr)
            except:
                pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-20:]  # 最近20天
        
        summary = {
            'current': {
                'rank_ic': self.current_metrics.rank_ic if self.current_metrics else 0,
                'sharpe_ratio': self.current_metrics.sharpe_ratio if self.current_metrics else 0,
                'max_drawdown': self.current_metrics.max_drawdown if self.current_metrics else 0,
                'turnover': self.current_metrics.turnover if self.current_metrics else 0,
            },
            'recent_20d': {
                'avg_ic': np.mean([m.rank_ic for m in recent_metrics]),
                'avg_sharpe': np.mean([m.sharpe_ratio for m in recent_metrics]),
                'max_drawdown': min([m.max_drawdown for m in recent_metrics]),
                'avg_turnover': np.mean([m.turnover for m in recent_metrics]),
            },
            'alerts': {
                'total': len(self.alerts_history),
                'critical': len([a for a in self.alerts_history if a['level'] == 'CRITICAL']),
                'warning': len([a for a in self.alerts_history if a['level'] == 'WARNING']),
                'recent': self.alerts_history[-5:] if self.alerts_history else []
            },
            'factor_performance': self._get_factor_performance_summary(),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None
        }
        
        return summary
    
    def _get_factor_performance_summary(self) -> Dict:
        """获取因子性能摘要"""
        if not self.factor_performance:
            return {}
        
        summary = {}
        for factor, performance in self.factor_performance.items():
            if performance:
                recent_perf = list(performance)[-20:]
                summary[factor] = {
                    'avg_corr': np.mean(recent_perf),
                    'stability': 1 / (np.std(recent_perf) + 0.01) if len(recent_perf) > 1 else 1.0,
                    'trend': 'up' if len(recent_perf) > 1 and recent_perf[-1] > recent_perf[0] else 'down'
                }
        
        # 按平均相关性排序
        summary = dict(sorted(summary.items(), key=lambda x: x[1]['avg_corr'], reverse=True))
        
        return summary
    
    def export_metrics(self, filepath: str):
        """导出性能指标"""
        metrics_data = []
        for metric in self.metrics_history:
            metrics_data.append({
                'timestamp': metric.timestamp.isoformat(),
                'rank_ic': metric.rank_ic,
                'sharpe_ratio': metric.sharpe_ratio,
                'max_drawdown': metric.max_drawdown,
                'volatility': metric.volatility,
                'turnover': metric.turnover,
                'factor_crowding': metric.factor_crowding
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(filepath, index=False)
        logger.info(f"性能指标已导出到 {filepath}")
    
    def register_alert_callback(self, callback):
        """注册警报回调函数"""
        self.alert_callbacks.append(callback)
        logger.info("警报回调已注册")