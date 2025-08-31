#!/usr/bin/env python3
"""
统一IC/RankIC计算器 - 横截面-时间两阶段标准实现
===========================================================
修复纵向时间序列相关性被误判为选股力的问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ICCalculationConfig:
    """IC计算配置"""
    # 计算方法配置
    use_rank_ic: bool = True                   # 优先使用RankIC（Spearman）
    use_pearson_ic: bool = False               # 可选Pearson IC
    
    # 时间聚合配置
    temporal_aggregation: str = "mean"         # 时间聚合方法: mean, ewm, median
    decay_halflife: int = 30                   # 指数加权半衰期(天)
    min_cross_sectional_samples: int = 5      # 横截面最少样本数(自适应)
    
    # 滚动窗口配置
    ic_lookback_days: int = 252                # IC计算回望天数
    rolling_window: bool = True                # 是否使用滚动窗口
    min_temporal_samples: int = 30             # 时间维度最少样本数(自适应)
    adaptive_min_samples: bool = True          # 启用自适应样本数
    
    # 质量控制
    outlier_method: str = "winsorize"          # 异常值处理：winsorize/clip/none
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)  # Winsorize限制
    handle_missing: str = "drop"               # 缺失值处理：drop/fill/ignore
    
    # 稳健性配置
    bootstrap_samples: int = 0                 # Bootstrap样本数（0=不使用）
    confidence_level: float = 0.95             # 置信区间
    stability_threshold: float = 0.6           # IC稳定性阈值

class UnifiedICCalculator:
    """统一IC计算器 - 机构级标准实现"""
    
    def __init__(self, config: ICCalculationConfig = None):
        """初始化IC计算器"""
        self.config = config or ICCalculationConfig()
        self.cache = {}  # 缓存历史计算结果
        
        # 统计信息
        self.stats = {
            'cross_sectional_calculations': 0,
            'temporal_aggregations': 0,
            'cache_hits': 0,
            'invalid_dates': 0,
            'insufficient_samples': 0
        }
        
        logger.info(f"统一IC计算器初始化完成 - 横截面->时间两阶段方法")
    
    def calculate_cross_sectional_ic(self, factors: pd.Series, returns: pd.Series,
                                   method: str = "spearman") -> float:
        """
        计算单日横截面IC
        
        Args:
            factors: 因子值序列(同日不同标的)
            returns: 对应的前向收益序列
            method: 计算方法 spearman/pearson
            
        Returns:
            横截面IC值
        """
        if len(factors) != len(returns):
            return np.nan
            
        # 对齐非空值
        valid_mask = ~(factors.isna() | returns.isna() | 
                      np.isinf(factors) | np.isinf(returns))
        
        # 自适应横截面样本数检查
        valid_samples = valid_mask.sum()
        effective_min_cross_samples = self.config.min_cross_sectional_samples
        
        # 自适应降低要求
        if valid_samples >= 5:
            effective_min_cross_samples = 5
        elif valid_samples >= 3:
            effective_min_cross_samples = 3  # 最低要求3个样本
            
        if valid_samples < effective_min_cross_samples:
            self.stats['insufficient_samples'] += 1
            return np.nan
        
        factors_clean = factors[valid_mask]
        returns_clean = returns[valid_mask]
        
        # 异常值处理
        if self.config.outlier_method == "winsorize":
            factors_clean = self._winsorize_series(factors_clean)
            returns_clean = self._winsorize_series(returns_clean)
        elif self.config.outlier_method == "clip":
            factors_clean = factors_clean.clip(
                factors_clean.quantile(0.01), 
                factors_clean.quantile(0.99)
            )
            returns_clean = returns_clean.clip(
                returns_clean.quantile(0.01),
                returns_clean.quantile(0.99)
            )
        
        try:
            if method.lower() == "spearman":
                ic_value, p_value = stats.spearmanr(factors_clean, returns_clean)
            elif method.lower() == "pearson":
                ic_value, p_value = stats.pearsonr(factors_clean, returns_clean)
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            self.stats['cross_sectional_calculations'] += 1
            # 🔧 修复: NaN和0.0含义不同，NaN表示无法计算，0.0表示零相关
            return ic_value if not np.isnan(ic_value) else np.nan
            
        except Exception as e:
            logger.debug(f"横截面IC计算失败: {e}")
            return np.nan
    
    def calculate_temporal_ic_series(self, factor_data: pd.DataFrame, 
                                   return_data: pd.DataFrame,
                                   factor_name: str) -> pd.Series:
        """
        计算时间序列IC（逐日横截面->时间聚合）
        
        Args:
            factor_data: 因子数据 (index=date, columns=tickers)
            return_data: 收益数据 (index=date, columns=tickers)  
            factor_name: 因子名称
            
        Returns:
            时间序列IC
        """
        # 对齐时间索引
        common_dates = factor_data.index.intersection(return_data.index)
        
        # 自适应样本数调整 (🔧 修复：提高最低阈值确保统计显著性)
        effective_min_samples = self.config.min_temporal_samples
        if self.config.adaptive_min_samples:
            if len(common_dates) >= 30:
                effective_min_samples = 30  # 理想30天
            elif len(common_dates) >= 25:
                effective_min_samples = 25  # 降级到25天
            elif len(common_dates) >= 20:
                effective_min_samples = 20  # 最低20天（修复：从10天提高到20天）
                logger.warning(f"使用最低样本数阈值: {len(common_dates)}天，统计可靠性可能降低")
        
        if len(common_dates) < effective_min_samples:
            logger.warning(f"时间样本严重不足: {len(common_dates)} < {effective_min_samples}")
            self.stats['insufficient_samples'] += 1
            return pd.Series(dtype=float)
        
        if len(common_dates) < self.config.min_temporal_samples:
            logger.info(f"使用自适应样本数: {len(common_dates)} (标准:{self.config.min_temporal_samples})")
        
        # 按时间排序
        common_dates = sorted(common_dates)
        
        daily_ics = []
        valid_dates = []
        
        for date in common_dates:
            try:
                # 获取当日横截面数据
                factors_cross = factor_data.loc[date]
                returns_cross = return_data.loc[date]
                
                # 计算横截面IC
                method = "spearman" if self.config.use_rank_ic else "pearson"
                daily_ic = self.calculate_cross_sectional_ic(
                    factors_cross, returns_cross, method
                )
                
                if not np.isnan(daily_ic):
                    daily_ics.append(daily_ic)
                    valid_dates.append(date)
                else:
                    self.stats['invalid_dates'] += 1
                    
            except Exception as e:
                logger.debug(f"日期 {date} IC计算失败: {e}")
                self.stats['invalid_dates'] += 1
                continue
        
        if not daily_ics:
            return pd.Series(dtype=float)
        
        ic_series = pd.Series(daily_ics, index=valid_dates)
        return ic_series
    
    def aggregate_temporal_ic(self, ic_series: pd.Series) -> Dict[str, float]:
        """
        时间维度IC聚合统计
        
        Args:
            ic_series: 日度IC序列
            
        Returns:
            IC统计指标字典
        """
        if ic_series.empty:
            return self._get_empty_ic_stats()
        
        # 基础统计
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        
        # IC信息比率
        ir = mean_ic / std_ic if std_ic > 0 else 0.0
        
        # 正IC占比
        hit_rate = (ic_series > 0).mean()
        
        # IC稳定性（绝对值IC的均值）
        ic_stability = ic_series.abs().mean()
        
        # 时间衰减加权IC
        if self.config.temporal_aggregation == "ewm":
            ewm_ic = ic_series.ewm(halflife=self.config.decay_halflife).mean().iloc[-1]
        else:
            ewm_ic = mean_ic
        
        # IC分布统计
        ic_quantiles = ic_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        
        self.stats['temporal_aggregations'] += 1
        
        return {
            'ic_mean': float(mean_ic),
            'ic_std': float(std_ic),
            'ic_ir': float(ir),
            'ic_hit_rate': float(hit_rate),
            'ic_stability': float(ic_stability),
            'ic_ewm': float(ewm_ic),
            'ic_sharpe': float(ir * np.sqrt(252)),  # 年化IC夏普
            'ic_count': len(ic_series),
            'ic_q10': float(ic_quantiles[0.1]),
            'ic_q25': float(ic_quantiles[0.25]),
            'ic_median': float(ic_quantiles[0.5]),
            'ic_q75': float(ic_quantiles[0.75]),
            'ic_q90': float(ic_quantiles[0.9]),
            'ic_skew': float(ic_series.skew()),
            'ic_kurt': float(ic_series.kurtosis())
        }
    
    def calculate_factor_ic_comprehensive(self, factor_data: pd.DataFrame,
                                        return_data: pd.DataFrame,
                                        factor_name: str) -> Dict[str, Any]:
        """
        全面计算因子IC指标（主接口）
        
        Args:
            factor_data: 因子数据
            return_data: 收益数据  
            factor_name: 因子名称
            
        Returns:
            完整IC分析结果
        """
        cache_key = f"{factor_name}_{hash(str(factor_data.shape))}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # 步骤1: 计算日度横截面IC序列
        ic_series = self.calculate_temporal_ic_series(
            factor_data, return_data, factor_name
        )
        
        if ic_series.empty:
            logger.warning(f"因子 {factor_name} IC计算失败 - 无有效数据")
            return self._get_empty_comprehensive_result(factor_name)
        
        # 步骤2: 时间聚合统计
        ic_stats = self.aggregate_temporal_ic(ic_series)
        
        # 步骤3: 滚动窗口IC（如果启用）
        rolling_ic_stats = {}
        if self.config.rolling_window and len(ic_series) >= self.config.ic_lookback_days:
            rolling_ic = ic_series.rolling(self.config.ic_lookback_days).apply(
                lambda x: x.mean() if len(x) >= 30 else np.nan
            )
            rolling_ic_stats = {
                'rolling_ic_mean': float(rolling_ic.mean()),
                'rolling_ic_std': float(rolling_ic.std()),
                'rolling_ic_last': float(rolling_ic.iloc[-1]) if not rolling_ic.empty else np.nan
            }
        
        # 步骤4: Bootstrap置信区间（如果启用）
        bootstrap_stats = {}
        if self.config.bootstrap_samples > 0:
            bootstrap_ics = self._bootstrap_ic(ic_series, self.config.bootstrap_samples)
            alpha = 1 - self.config.confidence_level
            bootstrap_stats = {
                'ic_bootstrap_mean': float(np.mean(bootstrap_ics)),
                'ic_confidence_lower': float(np.percentile(bootstrap_ics, 100*alpha/2)),
                'ic_confidence_upper': float(np.percentile(bootstrap_ics, 100*(1-alpha/2))),
                'ic_bootstrap_std': float(np.std(bootstrap_ics))
            }
        
        # 综合结果
        comprehensive_result = {
            'factor_name': factor_name,
            'calculation_method': 'cross_sectional_temporal_aggregation',
            'ic_stats': ic_stats,
            'rolling_stats': rolling_ic_stats,
            'bootstrap_stats': bootstrap_stats,
            'ic_series': ic_series,  # 原始日度IC序列
            'data_quality': {
                'total_dates': len(factor_data.index),
                'valid_ic_dates': len(ic_series),
                'coverage_rate': len(ic_series) / len(factor_data.index) if len(factor_data.index) > 0 else 0,
                'avg_cross_sectional_samples': factor_data.count(axis=1).mean()
            },
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        # 缓存结果
        self.cache[cache_key] = comprehensive_result
        
        logger.info(f"因子 {factor_name} IC计算完成: IC={ic_stats['ic_mean']:.4f}, "
                   f"IR={ic_stats['ic_ir']:.4f}, 稳定性={ic_stats['ic_stability']:.4f}")
        
        return comprehensive_result
    
    def calculate_multi_factor_ic_matrix(self, factor_data_dict: Dict[str, pd.DataFrame],
                                       return_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多因子IC矩阵
        
        Args:
            factor_data_dict: 多个因子数据字典
            return_data: 收益数据
            
        Returns:
            因子IC矩阵 (rows=factors, cols=ic_metrics)
        """
        ic_matrix_data = []
        
        for factor_name, factor_data in factor_data_dict.items():
            ic_result = self.calculate_factor_ic_comprehensive(
                factor_data, return_data, factor_name
            )
            
            # 提取关键指标
            ic_stats = ic_result['ic_stats']
            row_data = {
                'factor_name': factor_name,
                'ic_mean': ic_stats['ic_mean'],
                'ic_std': ic_stats['ic_std'],
                'ic_ir': ic_stats['ic_ir'],
                'ic_hit_rate': ic_stats['ic_hit_rate'],
                'ic_stability': ic_stats['ic_stability'],
                'ic_sharpe_annual': ic_stats['ic_sharpe'],
                'valid_dates': ic_result['data_quality']['valid_ic_dates'],
                'coverage_rate': ic_result['data_quality']['coverage_rate']
            }
            ic_matrix_data.append(row_data)
        
        ic_matrix_df = pd.DataFrame(ic_matrix_data)
        ic_matrix_df = ic_matrix_df.set_index('factor_name')
        
        logger.info(f"多因子IC矩阵计算完成: {len(factor_data_dict)} 个因子")
        return ic_matrix_df
    
    def _winsorize_series(self, series: pd.Series) -> pd.Series:
        """Winsorize序列"""
        lower_limit, upper_limit = self.config.winsorize_limits
        lower_val = series.quantile(lower_limit)
        upper_val = series.quantile(upper_limit)
        return series.clip(lower_val, upper_val)
    
    def _bootstrap_ic(self, ic_series: pd.Series, n_bootstrap: int) -> List[float]:
        """Bootstrap重采样IC"""
        bootstrap_ics = []
        for _ in range(n_bootstrap):
            sample_ic = ic_series.sample(len(ic_series), replace=True)
            bootstrap_ics.append(sample_ic.mean())
        return bootstrap_ics
    
    def _get_empty_ic_stats(self) -> Dict[str, float]:
        """获取空IC统计"""
        return {
            'ic_mean': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'ic_hit_rate': 0.0,
            'ic_stability': 0.0, 'ic_ewm': 0.0, 'ic_sharpe': 0.0, 'ic_count': 0,
            'ic_q10': 0.0, 'ic_q25': 0.0, 'ic_median': 0.0, 'ic_q75': 0.0, 'ic_q90': 0.0,
            'ic_skew': 0.0, 'ic_kurt': 0.0
        }
    
    def _get_empty_comprehensive_result(self, factor_name: str) -> Dict[str, Any]:
        """获取空的综合结果"""
        return {
            'factor_name': factor_name,
            'calculation_method': 'cross_sectional_temporal_aggregation',
            'ic_stats': self._get_empty_ic_stats(),
            'rolling_stats': {},
            'bootstrap_stats': {},
            'ic_series': pd.Series(dtype=float),
            'data_quality': {
                'total_dates': 0,
                'valid_ic_dates': 0,
                'coverage_rate': 0.0,
                'avg_cross_sectional_samples': 0.0
            },
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        return {
            'stats': self.stats,
            'cache_size': len(self.cache),
            'config': self.config.__dict__,
            'calculation_summary': {
                'cross_sectional_rate': (
                    self.stats['cross_sectional_calculations'] / 
                    max(1, self.stats['cross_sectional_calculations'] + self.stats['invalid_dates'])
                ),
                'temporal_aggregation_count': self.stats['temporal_aggregations'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / 
                    max(1, self.stats['cache_hits'] + self.stats['temporal_aggregations'])
                )
            }
        }

# 全局IC计算器实例
GLOBAL_IC_CALCULATOR = UnifiedICCalculator()

def get_global_ic_calculator() -> UnifiedICCalculator:
    """获取全局IC计算器"""
    return GLOBAL_IC_CALCULATOR

if __name__ == "__main__":
    # 测试IC计算器
    calculator = UnifiedICCalculator()
    
    # 模拟数据测试
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # 创建模拟因子和收益数据
    # np.random.seed removed
    factor_data = pd.DataFrame(
        np.zeros(100), index=dates, columns=tickers
    )
    return_data = pd.DataFrame(
        np.zeros(100) * 0.02, index=dates, columns=tickers
    )
    
    # 测试单因子IC计算
    result = calculator.calculate_factor_ic_comprehensive(
        factor_data, return_data, 'test_factor'
    )
    
    print("=== 横截面-时间两阶段IC计算测试 ===")
    print(f"IC均值: {result['ic_stats']['ic_mean']:.4f}")
    print(f"IC IR: {result['ic_stats']['ic_ir']:.4f}")
    print(f"IC稳定性: {result['ic_stats']['ic_stability']:.4f}")
    print(f"有效日期数: {result['data_quality']['valid_ic_dates']}")
    print(f"覆盖率: {result['data_quality']['coverage_rate']:.2%}")