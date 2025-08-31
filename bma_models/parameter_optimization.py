"""
技术指标参数优化系统
====================
基于滚动IC优化技术指标窗口参数，提升预测性能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ParameterConfig:
    """参数优化配置"""
    min_window: int = 5
    max_window: int = 50
    step_size: int = 5
    ic_lookback: int = 252  # 滚动IC回看期
    min_observations: int = 100  # 最小观测数量
    significance_threshold: float = 0.05  # IC显著性阈值

class TechnicalIndicatorOptimizer:
    """技术指标参数优化器"""
    
    def __init__(self, config: Optional[ParameterConfig] = None):
        """
        初始化优化器
        
        Args:
            config: 优化配置参数
        """
        self.config = config or ParameterConfig()
        self.optimization_results = {}
        self.best_parameters = {}
        
    def optimize_parameter(self, data: pd.DataFrame, 
                          target_col: str,
                          indicator_func: callable,
                          parameter_name: str = 'window',
                          parameter_range: Optional[List[int]] = None,
                          date_col: str = 'date') -> Dict:
        """
        优化单个技术指标参数
        
        Args:
            data: 历史数据
            target_col: 目标变量列名（如未来收益）
            indicator_func: 技术指标函数
            parameter_name: 参数名称
            parameter_range: 参数取值范围
            date_col: 日期列名
            
        Returns:
            优化结果字典
        """
        if parameter_range is None:
            parameter_range = list(range(
                self.config.min_window,
                self.config.max_window + 1,
                self.config.step_size
            ))
        
        logger.info(f"开始优化{indicator_func.__name__}的{parameter_name}参数，范围: {parameter_range}")
        
        results = {}
        
        for param_value in parameter_range:
            try:
                # 计算指标值
                kwargs = {parameter_name: param_value}
                indicator_values = indicator_func(data, **kwargs)
                
                # 计算IC
                ic_results = self._calculate_rolling_ic(
                    data, indicator_values, target_col, date_col
                )
                
                # 统计IC指标
                ic_stats = self._calculate_ic_statistics(ic_results)
                
                results[param_value] = {
                    'ic_mean': ic_stats['mean'],
                    'ic_std': ic_stats['std'],
                    'ic_sharpe': ic_stats['sharpe'],
                    'ic_hit_rate': ic_stats['hit_rate'],
                    'ic_significance': ic_stats['significance'],
                    'n_observations': ic_stats['n_obs'],
                    'stability_score': ic_stats['stability_score']
                }
                
            except Exception as e:
                logger.warning(f"参数{param_value}优化失败: {e}")
                continue
        
        if not results:
            logger.error(f"所有参数值优化失败")
            return {}
        
        # 选择最佳参数
        best_param = self._select_best_parameter(results)
        
        optimization_result = {
            'indicator_name': indicator_func.__name__,
            'parameter_name': parameter_name,
            'best_parameter': best_param,
            'all_results': results,
            'optimization_summary': self._create_optimization_summary(results, best_param)
        }
        
        # 缓存结果
        key = f"{indicator_func.__name__}_{parameter_name}"
        self.optimization_results[key] = optimization_result
        self.best_parameters[key] = best_param
        
        logger.info(f"✅ 参数优化完成: {indicator_func.__name__}.{parameter_name} = {best_param}")
        
        return optimization_result
    
    def _calculate_rolling_ic(self, data: pd.DataFrame, 
                            indicator_values: pd.Series, 
                            target_col: str,
                            date_col: str) -> pd.Series:
        """计算滚动IC"""
        # 构建计算DataFrame
        calc_df = pd.DataFrame({
            'date': data[date_col],
            'indicator': indicator_values,
            'target': data[target_col]
        }).dropna()
        
        if len(calc_df) < self.config.min_observations:
            logger.warning(f"观测数量不足: {len(calc_df)} < {self.config.min_observations}")
            return pd.Series(dtype=float)
        
        # 按日期排序
        calc_df = calc_df.sort_values('date')
        
        # 计算滚动IC
        ic_series = []
        dates = []
        
        for i in range(self.config.ic_lookback, len(calc_df)):
            window_data = calc_df.iloc[i-self.config.ic_lookback:i]
            
            if len(window_data) >= 20:  # 最少20个观测点
                try:
                    ic, p_value = stats.pearsonr(
                        window_data['indicator'], 
                        window_data['target']
                    )
                    ic_series.append(ic if not np.isnan(ic) else 0)
                    dates.append(calc_df.iloc[i]['date'])
                except:
                    ic_series.append(0)
                    dates.append(calc_df.iloc[i]['date'])
        
        return pd.Series(ic_series, index=dates)
    
    def _calculate_ic_statistics(self, ic_series: pd.Series) -> Dict:
        """计算IC统计指标"""
        if len(ic_series) == 0:
            return {
                'mean': 0, 'std': 0, 'sharpe': 0, 'hit_rate': 0,
                'significance': 1, 'n_obs': 0, 'stability_score': 0
            }
        
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) == 0:
            return {
                'mean': 0, 'std': 0, 'sharpe': 0, 'hit_rate': 0,
                'significance': 1, 'n_obs': 0, 'stability_score': 0
            }
        
        mean_ic = ic_clean.mean()
        std_ic = ic_clean.std()
        
        # IC Sharpe ratio
        sharpe = mean_ic / std_ic if std_ic > 0 else 0
        
        # Hit rate (IC > 0的比例)
        hit_rate = (ic_clean > 0).mean()
        
        # 显著性检验 (t-test)
        if len(ic_clean) > 1:
            t_stat, p_value = stats.ttest_1samp(ic_clean, 0)
            significance = p_value
        else:
            significance = 1.0
        
        # 稳定性得分（基于IC波动和持续性）
        stability_score = self._calculate_stability_score(ic_clean)
        
        return {
            'mean': mean_ic,
            'std': std_ic,
            'sharpe': sharpe,
            'hit_rate': hit_rate,
            'significance': significance,
            'n_obs': len(ic_clean),
            'stability_score': stability_score
        }
    
    def _calculate_stability_score(self, ic_series: pd.Series) -> float:
        """计算稳定性得分"""
        if len(ic_series) < 2:
            return 0.0
        
        # 因子1: IC绝对值均值（效果强度）
        abs_mean = ic_series.abs().mean()
        
        # 因子2: IC符号一致性（方向稳定性）
        sign_consistency = max(
            (ic_series > 0).mean(),
            (ic_series < 0).mean()
        )
        
        # 因子3: IC波动的倒数（数值稳定性）
        volatility_penalty = 1 / (1 + ic_series.std()) if ic_series.std() > 0 else 1
        
        # 综合稳定性得分
        stability_score = (abs_mean * 0.4 + 
                         sign_consistency * 0.4 + 
                         volatility_penalty * 0.2)
        
        return stability_score
    
    def _select_best_parameter(self, results: Dict) -> int:
        """选择最佳参数"""
        if not results:
            return self.config.min_window
        
        # 综合评分：IC均值 + IC Sharpe + 稳定性 - 显著性p值
        best_param = None
        best_score = -np.inf
        
        for param, metrics in results.items():
            # 只考虑有足够观测数量的参数
            if metrics['n_observations'] < 20:
                continue
            
            # 综合评分
            score = (
                metrics['ic_mean'] * 10 +           # IC均值权重最大
                metrics['ic_sharpe'] * 5 +          # IC夏普比权重次之
                metrics['stability_score'] * 3 +   # 稳定性得分
                -np.log10(metrics['ic_significance'] + 1e-6) * 2  # 显著性（p值越小越好）
            )
            
            if score > best_score:
                best_score = score
                best_param = param
        
        return best_param if best_param is not None else self.config.min_window
    
    def _create_optimization_summary(self, results: Dict, best_param: int) -> Dict:
        """创建优化摘要"""
        if not results or best_param is None:
            return {}
        
        best_metrics = results.get(best_param, {})
        
        # 统计所有参数的表现
        all_ic_means = [r['ic_mean'] for r in results.values()]
        all_sharpes = [r['ic_sharpe'] for r in results.values()]
        
        return {
            'best_ic_mean': best_metrics.get('ic_mean', 0),
            'best_ic_sharpe': best_metrics.get('ic_sharpe', 0),
            'best_stability_score': best_metrics.get('stability_score', 0),
            'best_significance': best_metrics.get('ic_significance', 1),
            'improvement_over_median': (
                best_metrics.get('ic_mean', 0) - np.median(all_ic_means)
                if all_ic_means else 0
            ),
            'parameter_sensitivity': np.std(all_ic_means) if all_ic_means else 0,
            'n_parameters_tested': len(results)
        }
    
    def get_optimized_parameters(self) -> Dict[str, int]:
        """获取所有优化后的参数"""
        return self.best_parameters.copy()
    
    def optimize_multiple_indicators(self, data: pd.DataFrame,
                                   target_col: str,
                                   indicator_configs: List[Dict],
                                   date_col: str = 'date') -> Dict:
        """
        批量优化多个技术指标
        
        Args:
            data: 历史数据
            target_col: 目标变量列名
            indicator_configs: 指标配置列表
            date_col: 日期列名
            
        Returns:
            所有优化结果
        """
        all_results = {}
        
        logger.info(f"开始批量优化{len(indicator_configs)}个技术指标")
        
        for config in indicator_configs:
            indicator_name = config['indicator_func'].__name__
            
            try:
                result = self.optimize_parameter(
                    data=data,
                    target_col=target_col,
                    indicator_func=config['indicator_func'],
                    parameter_name=config.get('parameter_name', 'window'),
                    parameter_range=config.get('parameter_range'),
                    date_col=date_col
                )
                
                all_results[indicator_name] = result
                
            except Exception as e:
                logger.error(f"优化{indicator_name}失败: {e}")
                continue
        
        logger.info(f"✅ 批量优化完成，成功优化{len(all_results)}个指标")
        
        return all_results


# 便捷函数
def optimize_technical_parameters_predictive_safe(
    data: pd.DataFrame,
    target_col: str,
    indicator_functions: List[callable],
    date_col: str = 'date',
    ic_lookback: int = 252
) -> Dict[str, int]:
    """
    预测性能安全的技术指标参数优化
    
    Args:
        data: 历史数据
        target_col: 目标变量列名
        indicator_functions: 技术指标函数列表
        date_col: 日期列名
        ic_lookback: IC回看期
        
    Returns:
        优化后的参数字典
    """
    config = ParameterConfig(ic_lookback=ic_lookback)
    optimizer = TechnicalIndicatorOptimizer(config)
    
    # 构建配置列表
    indicator_configs = [
        {'indicator_func': func, 'parameter_name': 'window'}
        for func in indicator_functions
    ]
    
    # 批量优化
    results = optimizer.optimize_multiple_indicators(
        data, target_col, indicator_configs, date_col
    )
    
    # 提取最佳参数
    best_params = {}
    for indicator_name, result in results.items():
        if result and 'best_parameter' in result:
            best_params[indicator_name] = result['best_parameter']
    
    return best_params


if __name__ == "__main__":
    # 测试参数优化系统
    import pandas as pd
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    n_days = 500
    n_stocks = 50
    
    dates = pd.date_range('2023-01-01', periods=n_days)
    tickers = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    
    test_data = []
    for date in dates:
        for ticker in tickers:
            test_data.append({
                'date': date,
                'ticker': ticker,
                'Close': 100 + np.random.randn() * 10,
                'Volume': 1000000 + np.random.randint(-500000, 500000),
                'future_return': np.random.randn() * 0.02  # 模拟未来收益
            })
    
    test_df = pd.DataFrame(test_data)
    
    # 定义测试指标函数
    def test_sma(df, window=20):
        """测试SMA指标"""
        return df.groupby('ticker')['Close'].transform(
            lambda x: x.rolling(window).mean()
        )
    
    def test_rsi(df, window=14):
        """测试RSI指标"""
        def calc_rsi(prices):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('ticker')['Close'].transform(calc_rsi)
    
    # 运行参数优化
    print("开始参数优化测试...")
    
    optimizer = TechnicalIndicatorOptimizer()
    
    # 优化SMA参数
    sma_result = optimizer.optimize_parameter(
        data=test_df,
        target_col='future_return',
        indicator_func=test_sma,
        parameter_name='window',
        parameter_range=[5, 10, 15, 20, 25, 30]
    )
    
    print(f"\nSMA最佳参数: {sma_result.get('best_parameter')}")
    print(f"最佳IC均值: {sma_result['optimization_summary'].get('best_ic_mean', 0):.4f}")
    
    # 获取优化后的参数
    optimized_params = optimizer.get_optimized_parameters()
    print(f"\n所有优化后的参数: {optimized_params}")