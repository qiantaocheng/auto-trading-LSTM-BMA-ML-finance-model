"""
IC-based Dynamic Factor Weighting System
========================================
基于历史IC表现动态调整因子权重，提升预测性能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, field
import warnings

logger = logging.getLogger(__name__)

@dataclass
class WeightingConfig:
    """动态权重配置"""
    ic_lookback: int = 252  # IC回看期
    weight_decay: float = 0.95  # 权重衰减因子
    min_weight: float = 0.01  # 最小权重
    max_weight: float = 0.5   # 最大权重
    stability_weight: float = 0.3  # 稳定性权重
    performance_weight: float = 0.7  # 性能权重
    rebalance_frequency: int = 21  # 重新平衡频率（天）
    min_ic_observations: int = 50  # 最小IC观测数量

class DynamicFactorWeighter:
    """动态因子权重器"""
    
    def __init__(self, config: Optional[WeightingConfig] = None):
        """
        初始化权重器
        
        Args:
            config: 权重配置
        """
        self.config = config or WeightingConfig()
        self.factor_ic_history = {}
        self.current_weights = {}
        self.weight_history = {}
        self.last_rebalance_date = None
        
    def calculate_dynamic_weights(self, data: pd.DataFrame,
                                 factor_cols: List[str],
                                 target_col: str,
                                 date_col: str = 'date',
                                 force_rebalance: bool = False) -> Dict[str, float]:
        """
        计算动态权重
        
        Args:
            data: 历史数据
            factor_cols: 因子列名列表
            target_col: 目标变量列名
            date_col: 日期列名
            force_rebalance: 强制重新平衡
            
        Returns:
            因子权重字典
        """
        if len(data) == 0 or target_col not in data.columns:
            return self._get_equal_weights(factor_cols)
        
        # 检查是否需要重新平衡
        current_date = data[date_col].max()
        if not force_rebalance and self._should_skip_rebalance(current_date):
            return self.current_weights.copy()
        
        logger.info(f"开始计算{len(factor_cols)}个因子的动态权重")
        
        # 计算各因子的IC历史
        factor_ic_metrics = {}
        
        for factor in factor_cols:
            if factor not in data.columns:
                continue
                
            ic_metrics = self._calculate_factor_ic_metrics(
                data, factor, target_col, date_col
            )
            
            if ic_metrics:
                factor_ic_metrics[factor] = ic_metrics
        
        if not factor_ic_metrics:
            logger.warning("无法计算任何因子的IC指标，使用等权重")
            return self._get_equal_weights(factor_cols)
        
        # 基于IC指标计算权重
        raw_weights = self._calculate_performance_weights(factor_ic_metrics)
        
        # 应用权重约束和标准化
        normalized_weights = self._normalize_weights(raw_weights)
        
        # 更新状态
        self.current_weights = normalized_weights
        self.last_rebalance_date = current_date
        self.weight_history[current_date] = normalized_weights.copy()
        
        logger.info(f"✅ 动态权重计算完成: {len(normalized_weights)}个因子")
        self._log_weight_summary(normalized_weights)
        
        return normalized_weights
    
    def _calculate_factor_ic_metrics(self, data: pd.DataFrame, 
                                   factor_col: str, 
                                   target_col: str,
                                   date_col: str) -> Optional[Dict]:
        """计算单个因子的IC指标"""
        try:
            # 构建IC计算数据
            calc_data = data[[date_col, factor_col, target_col]].dropna()
            
            if len(calc_data) < self.config.min_ic_observations:
                return None
            
            # 按日期计算横截面IC
            daily_ics = []
            dates = []
            
            for date, group in calc_data.groupby(date_col):
                if len(group) >= 5:  # 最少5个观测点
                    try:
                        ic, p_value = stats.pearsonr(group[factor_col], group[target_col])
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                            dates.append(date)
                    except:
                        continue
            
            if len(daily_ics) < 10:  # 最少10个有效IC
                return None
            
            ic_series = pd.Series(daily_ics, index=dates)
            
            # 应用时间衰减权重
            if self.config.weight_decay < 1.0:
                decay_weights = np.power(self.config.weight_decay, 
                                       np.arange(len(ic_series))[::-1])
                ic_series = ic_series * decay_weights
            
            # 计算IC统计指标
            metrics = self._calculate_comprehensive_ic_metrics(ic_series)
            
            # 缓存IC历史
            self.factor_ic_history[factor_col] = ic_series
            
            return metrics
            
        except Exception as e:
            logger.warning(f"计算因子{factor_col}的IC指标失败: {e}")
            return None
    
    def _calculate_comprehensive_ic_metrics(self, ic_series: pd.Series) -> Dict:
        """计算综合IC指标"""
        metrics = {
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ic_sharpe': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'ic_hit_rate': (ic_series > 0).mean(),
            'abs_ic_mean': ic_series.abs().mean(),
            'ic_consistency': self._calculate_ic_consistency(ic_series),
            'ic_stability': self._calculate_ic_stability(ic_series),
            'n_observations': len(ic_series)
        }
        
        # 显著性检验
        if len(ic_series) > 1:
            t_stat, p_value = stats.ttest_1samp(ic_series, 0)
            metrics['significance'] = 1 - p_value  # 转换为显著性得分
        else:
            metrics['significance'] = 0
        
        return metrics
    
    def _calculate_ic_consistency(self, ic_series: pd.Series) -> float:
        """计算IC一致性（符号稳定性）"""
        if len(ic_series) == 0:
            return 0.0
        
        # 计算连续同号的最长长度占比
        signs = np.sign(ic_series)
        sign_changes = (signs.diff() != 0).sum()
        consistency_ratio = 1 - (sign_changes / len(ic_series))
        
        return max(0, consistency_ratio)
    
    def _calculate_ic_stability(self, ic_series: pd.Series) -> float:
        """计算IC稳定性（变异系数的倒数）"""
        if len(ic_series) == 0 or ic_series.std() == 0:
            return 0.0
        
        # 使用变异系数的倒数衡量稳定性
        cv = ic_series.std() / abs(ic_series.mean()) if ic_series.mean() != 0 else np.inf
        stability = 1 / (1 + cv)  # 映射到[0, 1]区间
        
        return stability
    
    def _calculate_performance_weights(self, factor_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """基于性能指标计算权重"""
        weights = {}
        
        # 提取关键指标
        performance_scores = {}
        stability_scores = {}
        
        for factor, metrics in factor_metrics.items():
            # 性能得分：IC均值 + IC夏普比 + 显著性
            performance_score = (
                metrics.get('abs_ic_mean', 0) * 2 +
                metrics.get('ic_sharpe', 0) * 1 +
                metrics.get('significance', 0) * 1 +
                metrics.get('ic_hit_rate', 0.5) * 0.5
            )
            
            # 稳定性得分：一致性 + 稳定性
            stability_score = (
                metrics.get('ic_consistency', 0) * 0.6 +
                metrics.get('ic_stability', 0) * 0.4
            )
            
            performance_scores[factor] = max(0, performance_score)
            stability_scores[factor] = max(0, stability_score)
        
        # 标准化得分
        if performance_scores:
            max_perf = max(performance_scores.values())
            max_stab = max(stability_scores.values())
            
            if max_perf > 0:
                performance_scores = {k: v/max_perf for k, v in performance_scores.items()}
            if max_stab > 0:
                stability_scores = {k: v/max_stab for k, v in stability_scores.items()}
        
        # 综合评分
        for factor in factor_metrics.keys():
            perf_score = performance_scores.get(factor, 0)
            stab_score = stability_scores.get(factor, 0)
            
            combined_score = (
                perf_score * self.config.performance_weight +
                stab_score * self.config.stability_weight
            )
            
            weights[factor] = combined_score
        
        return weights
    
    def _normalize_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """标准化权重"""
        if not raw_weights:
            return {}
        
        # 应用最小最大权重约束
        for factor in raw_weights:
            raw_weights[factor] = np.clip(
                raw_weights[factor], 
                self.config.min_weight, 
                self.config.max_weight
            )
        
        # 标准化到和为1
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            normalized = {k: v/total_weight for k, v in raw_weights.items()}
        else:
            # 如果总权重为0，使用等权重
            n_factors = len(raw_weights)
            normalized = {k: 1/n_factors for k in raw_weights.keys()}
        
        return normalized
    
    def _get_equal_weights(self, factor_cols: List[str]) -> Dict[str, float]:
        """获取等权重"""
        if not factor_cols:
            return {}
        
        weight = 1.0 / len(factor_cols)
        return {factor: weight for factor in factor_cols}
    
    def _should_skip_rebalance(self, current_date) -> bool:
        """判断是否应该跳过重新平衡"""
        if self.last_rebalance_date is None:
            return False
        
        # 简化的日期差计算（假设日期是可比较的）
        try:
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance < self.config.rebalance_frequency
        except:
            return False
    
    def _log_weight_summary(self, weights: Dict[str, float]):
        """记录权重摘要"""
        if not weights:
            return
        
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("动态权重分布:")
        for factor, weight in sorted_weights[:5]:  # 显示前5个
            logger.info(f"  {factor}: {weight:.3f}")
        
        weight_stats = {
            'max_weight': max(weights.values()),
            'min_weight': min(weights.values()),
            'weight_std': np.std(list(weights.values())),
            'n_factors': len(weights)
        }
        
        logger.debug(f"权重统计: {weight_stats}")
    
    def get_weight_history(self) -> Dict:
        """获取权重历史"""
        return self.weight_history.copy()
    
    def get_ic_summary(self) -> Dict:
        """获取IC摘要统计"""
        if not self.factor_ic_history:
            return {}
        
        summary = {}
        for factor, ic_series in self.factor_ic_history.items():
            summary[factor] = {
                'mean_ic': ic_series.mean(),
                'std_ic': ic_series.std(),
                'sharpe_ic': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                'hit_rate': (ic_series > 0).mean(),
                'n_observations': len(ic_series)
            }
        
        return summary


# 全局实例
dynamic_factor_weighter = DynamicFactorWeighter()

def calculate_dynamic_factor_weights_predictive_safe(
    data: pd.DataFrame,
    factor_cols: List[str],
    target_col: str,
    date_col: str = 'date',
    ic_lookback: int = 252,
    rebalance_frequency: int = 21
) -> Dict[str, float]:
    """
    预测性能安全的动态因子权重计算
    
    Args:
        data: 历史数据
        factor_cols: 因子列名列表
        target_col: 目标变量列名
        date_col: 日期列名
        ic_lookback: IC回看期
        rebalance_frequency: 重新平衡频率
        
    Returns:
        因子权重字典
    """
    config = WeightingConfig(
        ic_lookback=ic_lookback,
        rebalance_frequency=rebalance_frequency
    )
    
    weighter = DynamicFactorWeighter(config)
    
    return weighter.calculate_dynamic_weights(
        data=data,
        factor_cols=factor_cols,
        target_col=target_col,
        date_col=date_col
    )


if __name__ == "__main__":
    # 测试动态权重系统
    import pandas as pd
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    n_days = 500
    n_stocks = 30
    
    dates = pd.date_range('2023-01-01', periods=n_days)
    tickers = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    
    test_data = []
    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            # 模拟不同质量的因子
            base_return = np.random.randn() * 0.02
            
            test_data.append({
                'date': date,
                'ticker': ticker,
                'factor1': np.random.randn() + base_return * 2,  # 高质量因子
                'factor2': np.random.randn() + base_return * 0.5, # 中等质量因子
                'factor3': np.random.randn(),  # 低质量因子（噪音）
                'factor4': np.random.randn() - base_return * 1.5, # 反向因子
                'future_return': base_return
            })
    
    test_df = pd.DataFrame(test_data)
    
    print("开始动态权重测试...")
    
    # 计算动态权重
    factor_cols = ['factor1', 'factor2', 'factor3', 'factor4']
    
    weighter = DynamicFactorWeighter()
    weights = weighter.calculate_dynamic_weights(
        data=test_df,
        factor_cols=factor_cols,
        target_col='future_return'
    )
    
    print(f"\n动态权重结果:")
    for factor, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {factor}: {weight:.3f}")
    
    # 获取IC摘要
    ic_summary = weighter.get_ic_summary()
    print(f"\nIC表现摘要:")
    for factor, stats in ic_summary.items():
        print(f"  {factor}: IC均值={stats['mean_ic']:.4f}, "
              f"IC夏普={stats['sharpe_ic']:.2f}, 命中率={stats['hit_rate']:.2f}")