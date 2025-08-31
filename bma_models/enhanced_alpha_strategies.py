#!/usr/bin/env python3
"""
Enhanced Alpha Strategy Module
Integrates advanced techniques: delay/decay, hump+rank, neutralization, winsorize
"""

import numpy as np
import pandas as pd
import yaml
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
try:
    # 尝试相对导入（当作为模块运行时）
    from .unified_nan_handler import unified_nan_handler, clean_nan_predictive_safe
    from cross_sectional_standardization import CrossSectionalStandardizer, standardize_cross_sectional_predictive_safe
    from .factor_orthogonalization import orthogonalize_factors_predictive_safe, FactorOrthogonalizer
    from .parameter_optimization import TechnicalIndicatorOptimizer, ParameterConfig
    from .dynamic_factor_weighting import DynamicFactorWeighter, calculate_dynamic_factor_weights_predictive_safe
except ImportError:
    # 回退到绝对导入（当直接运行时）
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from unified_nan_handler import unified_nan_handler, clean_nan_predictive_safe
    from cross_sectional_standardization import CrossSectionalStandardizer, standardize_cross_sectional_predictive_safe
    from factor_orthogonalization import orthogonalize_factors_predictive_safe, FactorOrthogonalizer
    from parameter_optimization import TechnicalIndicatorOptimizer, ParameterConfig
    from dynamic_factor_weighting import DynamicFactorWeighter, calculate_dynamic_factor_weights_predictive_safe

# 为了兼容性，创建别名
cross_sectional_standardize = standardize_cross_sectional_predictive_safe
import logging

# Removed external advanced factor dependencies, all factors integrated into this module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaStrategiesEngine:
    """Alpha Strategy Engine: Unified computation, neutralization, ranking, gating"""
    
    def __init__(self, config_path: str = "alphas_config.yaml"):
        """
        Initialize Alpha Strategy Engine
        
        Args:
            config_path: Configuration file path
        """
        self.config = self._load_config(config_path)
        self.alpha_functions = self._register_alpha_functions()
            
        self.alpha_cache = {}  # Cache computation results
        
        # All factors integrated into this module, no external dependencies needed
        logger.info("All Alpha factors integrated into this module")
        
        # ✅ NEW: 导入因子滞后配置
        try:
            from factor_lag_config import factor_lag_manager
            self.lag_manager = factor_lag_manager
            logger.info(f"因子滞后配置加载成功，最大滞后: T-{self.lag_manager.get_max_lag()}")
        except ImportError:
            logger.warning("因子滞后配置未找到，使用默认全局滞后")
            self.lag_manager = None
        
        # ✅ PERFORMANCE FIX: Initialize parameter optimizer
        self.parameter_optimizer = TechnicalIndicatorOptimizer()
        self.optimized_parameters = {}
        
        # ✅ PERFORMANCE FIX: Initialize dynamic factor weighter
        self.factor_weighter = DynamicFactorWeighter()
        self.dynamic_weights = {}
        
        # Statistics
        self.stats = {
            'computation_times': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'neutralization_stats': {},
            'ic_stats': {}
        }
        
        logger.info(f"Alpha Strategy Engine initialized, loaded {len(self.config['alphas'])} factors")
    
    def decay_linear(self, series: pd.Series, decay: int) -> pd.Series:
        """
        应用线性时间衰减权重
        
        Args:
            series: 需要衰减的序列
            decay: 衰减期数
            
        Returns:
            应用衰减权重后的序列
        """
        if decay <= 1:
            return series
        
        try:
            # 创建线性衰减权重：最近的权重最大，历史权重递减
            weights = np.linspace(1, 1/decay, decay)
            weights = weights / weights.sum()  # 归一化
            
            # 对序列应用衰减权重
            result = series.copy().astype(float)  # 确保数据类型为float
            if len(series) >= decay:
                # 使用滚动窗口应用衰减权重
                for i in range(decay-1, len(series)):
                    window_data = series.iloc[i-decay+1:i+1]
                    if len(window_data) == decay:
                        result.iloc[i] = float((window_data.values * weights).sum())
            
            return result.apply(lambda x: self.safe_fillna(x, df))
            
        except Exception as e:
            logger.warning(f"线性衰减计算失败: {e}")
            return series.apply(lambda x: self.safe_fillna(x, df))
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'universe': 'TOPDIV3000',
            'region': 'GLB',
            'neutralization': ['COUNTRY'],
            'rebalance': 'WEEKLY',
            'hump_levels': [0.003, 0.008],
            'winsorize_std': 2.5,
            'truncation': 0.10,
            'temperature': 1.2,
            'alphas': []
        }
    
    def _register_alpha_functions(self) -> Dict[str, Callable]:
        """Register Alpha computation functions - All factors integrated"""
        return {
            # Technical factors
            'momentum': self._compute_momentum,
            'momentum_6_1': self._compute_momentum_6_1,
            'reversal': self._compute_reversal,
            'reversal_5': self._compute_reversal_5,
            'volatility': self._compute_volatility,
            'volume_turnover': self._compute_volume_turnover,
            'amihud': self._compute_amihud_illiquidity,
            'amihud_illiq': self._compute_amihud_illiquidity_new,
            'bid_ask_spread': self._compute_bid_ask_spread,
            'residual_momentum': self._compute_residual_momentum,
            'pead': self._compute_pead,
            
            # Extended momentum factors
            'new_high_proximity': self._compute_52w_new_high_proximity,
            'low_beta': self._compute_low_beta_anomaly,
            'idiosyncratic_vol': self._compute_idiosyncratic_volatility,
            
            # Fundamental factors
            'earnings_surprise': self._compute_earnings_surprise,
            'analyst_revision': self._compute_analyst_revision,
            'ebit_ev': self._compute_ebit_ev,
            'fcf_ev': self._compute_fcf_ev,
            'earnings_yield': self._compute_earnings_yield,
            'sales_yield': self._compute_sales_yield,
            
            # Profitability factors
            'gross_margin': self._compute_gross_margin,
            'operating_profitability': self._compute_operating_profitability,
            'roe_neutralized': self._compute_roe_neutralized,
            'roic_neutralized': self._compute_roic_neutralized,
            'net_margin': self._compute_net_margin,
            'cash_yield': self._compute_cash_yield,
            'shareholder_yield': self._compute_shareholder_yield,
            
            # Accrual factors
            'total_accruals': self._compute_total_accruals,
            'working_capital_accruals': self._compute_working_capital_accruals,
            'net_operating_assets': self._compute_net_operating_assets,
            'asset_growth': self._compute_asset_growth,
            'net_equity_issuance': self._compute_net_equity_issuance,
            'investment_factor': self._compute_investment_factor,
            
            # Quality score factors
            'piotroski_score': self._compute_piotroski_score,
            'ohlson_score': self._compute_ohlson_score,
            'altman_score': self._compute_altman_score,
            'qmj_score': self._compute_qmj_score,
            'earnings_stability': self._compute_earnings_stability,
            
            # Sentiment factors (独立的机器学习特征，无硬编码权重)
            'news_sentiment': self._compute_news_sentiment,
            'market_sentiment': self._compute_market_sentiment,
            'fear_greed_sentiment': self._compute_fear_greed_sentiment,
            'sentiment_momentum': self._compute_sentiment_momentum,
            
            # 🔥 NEW: Real Polygon Training技术指标集成
            'technical_sma_10': self._compute_sma_10,
            'technical_sma_20': self._compute_sma_20,
            'technical_sma_50': self._compute_sma_50,
            'technical_rsi': self._compute_rsi,
            'technical_bb_position': self._compute_bb_position,
            'technical_macd': self._compute_macd,
            'technical_macd_signal': self._compute_macd_signal,
            'technical_macd_histogram': self._compute_macd_histogram,
            'technical_price_momentum_5d': self._compute_price_momentum_5d,
            'technical_price_momentum_20d': self._compute_price_momentum_20d,
            'technical_volume_ratio': self._compute_volume_ratio,
            
            # 🔥 NEW: Real Polygon Training风险指标集成
            'risk_max_drawdown': self._compute_max_drawdown,
            'risk_sharpe_ratio': self._compute_sharpe_ratio,
            'risk_var_95': self._compute_var_95,
            
            # REMOVED: Low-performance factors
            # 'sentiment_volatility': self._compute_sentiment_volatility,  # 数据质量差
            # 'retail_herding_effect': self._compute_retail_herding_effect,  # 计算成本高
            # 'apm_momentum_reversal': self._compute_apm_momentum_reversal,  # 过度工程化
            
            'hump': None,  # Special handling
        }
    
    # ========== Basic Utility Functions ==========
    
    def winsorize_series(self, s: pd.Series, k: float = 2.5) -> pd.Series:
        """Winsorize series: Remove outliers"""
        if s.isna().all():
            return s
        mu, sd = s.mean(), s.std(ddof=0)
        if sd == 0:
            return s
        lo, hi = mu - k * sd, mu + k * sd
        return s.clip(lo, hi)
    
    def zscore_by_group(self, df: pd.DataFrame, col: str, group_cols: List[str]) -> pd.Series:
        """Group standardization"""
        return df.groupby(group_cols)[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        )
    
    def neutralize_factor(self, df: pd.DataFrame, target_col: str, 
                         group_cols: List[str]) -> pd.Series:
        """Time-safe linear regression neutralization - Prevents use of future data"""
        def _neutralize_cross_section_safe(block):
            if len(block) < 2 or target_col not in block.columns:
                return block[target_col] if target_col in block.columns else pd.Series(index=block.index)
            
            y = block[target_col]  # Keep NaN for now
            if len(y) < 2:
                return block[target_col]
            
            # KEY FIX: Use expanding window to ensure only historical data is used
            # In real-time trading, at time T should not know future performance of other stocks on same day
            result = pd.Series(index=block.index, dtype=float)
            
            # Use time-progressive approach to calculate neutralization parameters
            sorted_indices = block.index.tolist()
            
            for i, idx in enumerate(sorted_indices):
                if idx not in y.index:
                    result.loc[idx] = 0.0
                    continue
                
                # CRITICAL FIX: 严格使用历史数据，排除当前时点
                # 在横截面中性化中，不应使用同日其他股票信息
                if i == 0:
                    # 第一个时点没有历史数据，使用原值或简单去均值
                    result.loc[idx] = y.loc[idx] if not pd.isna(y.loc[idx]) else 0.0
                    continue
                    
                hist_indices = sorted_indices[:i]  # 排除当前时点(i)
                hist_y = y.loc[y.index.intersection(hist_indices)]
                
                if len(hist_y) < 2:
                    # 如果历史数据不足，使用历史均值调整
                    hist_mean = hist_y.mean() if len(hist_y) > 0 else 0.0
                    result.loc[idx] = y.loc[idx] - hist_mean if not pd.isna(y.loc[idx]) else 0.0
                    continue
                
                # Build historical dummy variable matrix
                hist_block = block.loc[hist_indices]
                X_df = pd.get_dummies(hist_block[group_cols], drop_first=False)
                X_df = X_df.loc[hist_y.index]
                
                if X_df.shape[1] == 0 or X_df.var().sum() == 0:
                    # CRITICAL FIX: 使用历史数据计算基准
                    hist_mean = hist_y.mean() if len(hist_y) > 0 else 0.0
                    result.loc[idx] = y.loc[idx] - hist_mean if not pd.isna(y.loc[idx]) else 0.0
                    continue
                
                try:
                    # Use historical data to fit regression model
                    lr = LinearRegression(fit_intercept=True)
                    lr.fit(X_df.values, hist_y.values)
                    
                    # Neutralize current point
                    current_X = pd.get_dummies(block.loc[[idx]][group_cols], drop_first=False)
                    current_X = current_X.reindex(columns=X_df.columns, fill_value=0)
                    
                    predicted = lr.predict(current_X.values)[0]
                    result.loc[idx] = y.loc[idx] - predicted
                    
                except Exception as e:
                    logger.warning(f"Point {idx} neutralization failed: {e}")
                    result.loc[idx] = hist_y.loc[idx] - hist_y.mean()
            
            return result.apply(lambda x: self.safe_fillna(x, df))
        
        return df.groupby('date').apply(_neutralize_cross_section_safe).reset_index(level=0, drop=True)
    
    def hump_transform(self, z: pd.Series, hump: float = 0.003) -> pd.Series:
        """Gating transformation: Set small signals to zero"""
        return z.where(z.abs() >= hump, 0.0)
    
    def rank_transform(self, z: pd.Series) -> pd.Series:
        """Ranking transformation"""
        return z.rank(pct=True) - 0.5
    
    def ema_decay(self, s: pd.Series, span: int) -> pd.Series:
        """Time-safe exponential moving average decay - Only use historical data"""
        # ✅ PERFORMANCE FIX: 移除过度保守的shift(1)
        # 差异化滞后已在因子级别应用，此处不需要额外滞后
        # Use expanding window to ensure each time point only uses historical data
        result = s.ewm(span=span, adjust=False).mean()
        # ❌ REMOVED: 移除额外shift(1)以保持信号及时性和强度
        # return result.shift(1)
        return result
    
    def safe_apply_fillna(self, series: pd.Series, df: pd.DataFrame = None) -> pd.Series:
        """Helper method to safely apply fillna without causing float object errors"""
        try:
            if isinstance(series, pd.Series):
                return series.fillna(0.0)
            else:
                # If it's not a Series, create one
                return pd.Series(series, index=df.index if df is not None else None).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=df.index if df is not None else None)
    
    def safe_fillna(self, data: pd.Series, df: pd.DataFrame = None, 
                   date_col: str = 'date') -> pd.Series:
        """
        CRITICAL FIX: 使用全局统一NaN处理策略
        重定向到global_nan_config.unified_nan_handler
        """
        try:
            from global_nan_config import unified_nan_handler
            return unified_nan_handler(data, df, date_col, 'cross_sectional_median')
        except ImportError:
            # FALLBACK: 如果global_nan_config不可用，使用本地逻辑
            logger.warning("使用本地NaN处理fallback逻辑")
            if df is not None and date_col in df.columns:
                # 使用横截面中位数填充
                temp_df = pd.DataFrame({
                    'data': data,
                    'date': df[date_col],
                    'original_index': data.index
                })
                
                def fill_cross_section(group):
                    daily_median = group['data'].median()
                    if pd.isna(daily_median):
                        daily_median = 0
                    group['data'] = group['data'].fillna(daily_median)
                    return group
            
            filled_df = temp_df.groupby('date').apply(fill_cross_section)
            result = pd.Series(index=data.index, dtype=float)
            for idx, row in filled_df.iterrows():
                original_idx = row['original_index']
                result.loc[original_idx] = row['data']
            return result
        else:
            # 如果没有日期信息，使用全局中位数
            return data.fillna(data.median() if not data.isna().all() else 0)
    
    def optimize_technical_parameters(self, df: pd.DataFrame, 
                                    target_col: str = 'future_return_10d',
                                    force_reoptimize: bool = False) -> Dict[str, int]:
        """
        ✅ PERFORMANCE FIX: 优化技术指标参数
        基于滚动IC选择最优窗口参数，提升预测性能
        
        Args:
            df: 历史数据
            target_col: 目标变量列名
            force_reoptimize: 强制重新优化
            
        Returns:
            优化后的参数字典
        """
        if self.optimized_parameters and not force_reoptimize:
            logger.info("使用缓存的优化参数")
            return self.optimized_parameters
        
        if target_col not in df.columns:
            logger.warning(f"目标列{target_col}不存在，跳过参数优化")
            return {}
        
        logger.info("开始优化技术指标参数...")
        
        # 定义需要优化的技术指标函数
        optimization_targets = [
            {
                'name': 'sma_10',
                'func': self._compute_sma_10,
                'param_range': [5, 8, 10, 12, 15]
            },
            {
                'name': 'sma_20', 
                'func': self._compute_sma_20,
                'param_range': [15, 18, 20, 22, 25]
            },
            {
                'name': 'sma_50',
                'func': self._compute_sma_50,
                'param_range': [30, 40, 50, 60, 70]
            },
            {
                'name': 'rsi',
                'func': self._compute_rsi,
                'param_range': [10, 12, 14, 16, 18]
            }
        ]
        
        optimized_params = {}
        
        for target in optimization_targets:
            try:
                # 创建指标函数包装器
                def indicator_wrapper(data, window):
                    # 临时修改默认参数来测试不同窗口
                    original_func = target['func']
                    return original_func(data, window=window)
                
                result = self.parameter_optimizer.optimize_parameter(
                    data=df,
                    target_col=target_col,
                    indicator_func=indicator_wrapper,
                    parameter_name='window',
                    parameter_range=target['param_range']
                )
                
                if result and 'best_parameter' in result:
                    optimized_params[target['name']] = result['best_parameter']
                    logger.info(f"✅ {target['name']}最优参数: {result['best_parameter']} "
                              f"(IC均值: {result['optimization_summary'].get('best_ic_mean', 0):.4f})")
                
            except Exception as e:
                logger.warning(f"优化{target['name']}失败: {e}")
                continue
        
        # 缓存结果
        self.optimized_parameters = optimized_params
        logger.info(f"✅ 技术指标参数优化完成，优化了{len(optimized_params)}个指标")
        
        return optimized_params
    
    def get_optimized_window(self, indicator_name: str, default: int) -> int:
        """
        获取优化后的窗口参数
        
        Args:
            indicator_name: 指标名称
            default: 默认值
            
        Returns:
            优化后的窗口大小
        """
        return self.optimized_parameters.get(indicator_name, default)
    
    def calculate_dynamic_weights(self, df: pd.DataFrame, 
                                alpha_cols: List[str],
                                target_col: str = 'future_return_10d',
                                force_rebalance: bool = False) -> Dict[str, float]:
        """
        ✅ PERFORMANCE FIX: 计算基于IC的动态因子权重
        根据历史IC表现动态调整权重，提升预测性能
        
        Args:
            df: 历史数据
            alpha_cols: Alpha因子列名
            target_col: 目标变量列名
            force_rebalance: 强制重新平衡
            
        Returns:
            动态权重字典
        """
        if not alpha_cols or target_col not in df.columns:
            logger.warning("无法计算动态权重，使用等权重")
            return {col: 1.0/len(alpha_cols) for col in alpha_cols}
        
        try:
            weights = self.factor_weighter.calculate_dynamic_weights(
                data=df,
                factor_cols=alpha_cols,
                target_col=target_col,
                force_rebalance=force_rebalance
            )
            
            # 缓存权重
            self.dynamic_weights = weights
            
            # 记录权重摘要
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            logger.info("✅ 动态权重计算完成:")
            for factor, weight in sorted_weights:
                logger.info(f"   {factor}: {weight:.3f}")
                
            return weights
            
        except Exception as e:
            logger.warning(f"动态权重计算失败，使用等权重: {e}")
            return {col: 1.0/len(alpha_cols) for col in alpha_cols}
    
    def apply_dynamic_weights(self, df: pd.DataFrame, 
                            alpha_cols: List[str],
                            weights: Dict[str, float]) -> pd.Series:
        """
        应用动态权重合成最终Alpha
        
        Args:
            df: 数据
            alpha_cols: Alpha因子列名
            weights: 权重字典
            
        Returns:
            加权后的综合Alpha因子
        """
        if not alpha_cols or not weights:
            return pd.Series(0, index=df.index)
        
        weighted_alpha = pd.Series(0.0, index=df.index)
        
        for col in alpha_cols:
            if col in df.columns and col in weights:
                weight = weights[col]
                factor_values = df[col].fillna(0)
                weighted_alpha += weight * factor_values
        
        return weighted_alpha
    
    # ========== Alpha Factor Computation Functions ==========
    
    def _compute_momentum(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """Time-safe momentum factor: Multi-window price momentum - 数值稳定性增强"""
        # 🔥 CRITICAL FIX: 导入数值稳定性保护
        from numerical_stability import safe_log, safe_divide
        
        results = []
        
        for window in windows:
            # 🛡️ SAFETY FIX: 使用数值安全的动量计算
            def safe_momentum_calc(x):
                """安全的动量计算函数"""
                if len(x) <= window + 2:
                    return pd.Series(index=x.index, dtype=float)
                
                current_price = x.shift(2)
                past_price = x.shift(window + 2)
                
                # 使用安全除法和对数计算
                price_ratio = safe_divide(current_price, past_price, fill_value=1.0)
                momentum_values = safe_log(price_ratio)
                
                return momentum_values
            
            momentum = df.groupby('ticker')['Close'].transform(safe_momentum_calc)

            # Time-safe exponential decay - Use expanding computation to ensure only historical data
            momentum_decayed = momentum.groupby(df['ticker']).apply(
                lambda s: s.expanding(min_periods=1).apply(
                    lambda x: pd.Series(x).ewm(span=decay, adjust=False).mean().iloc[-1]
                    if len(x) > 0 else np.nan
                )
            ).reset_index(level=0, drop=True)

            results.append(momentum_decayed)
        
        # Multi-window average
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_reversal(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """Reversal factor: Short-term price reversal - 数值稳定性增强"""
        # 🔥 CRITICAL FIX: 导入数值稳定性保护
        from numerical_stability import safe_log, safe_divide
        
        results = []
        
        for window in windows:
            # 🛡️ SAFETY FIX: 使用数值安全的反转计算
            def safe_reversal_calc(x):
                """安全的反转计算函数"""
                if len(x) <= window + 1:
                    return pd.Series(index=x.index, dtype=float)
                
                current_price = x.shift(1)
                past_price = x.shift(window + 1)
                
                # 使用安全除法和对数计算，反转信号取负号
                price_ratio = safe_divide(current_price, past_price, fill_value=1.0)
                reversal_values = -safe_log(price_ratio)  # 反转因子取负号
                
                return reversal_values
            
            reversal = df.groupby('ticker')['Close'].transform(safe_reversal_calc)

            # Exponential decay
            reversal_decayed = reversal.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            results.append(reversal_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_volatility(self, df: pd.DataFrame, windows: List[int], 
                           decay: int = 6) -> pd.Series:
        """Volatility factor: Reciprocal of realized volatility"""
        results = []
        
        for window in windows:
            # 🛡️ SAFETY FIX: Calculate log returns with numerical stability
            from numerical_stability import safe_log, safe_divide
            
            def safe_log_returns_calc(x):
                """安全的对数收益率计算"""
                if len(x) <= 1:
                    return pd.Series(index=x.index, dtype=float)
                
                current_price = x
                past_price = x.shift(1)
                price_ratio = safe_divide(current_price, past_price, fill_value=1.0)
                return safe_log(price_ratio)
            
            returns = df.groupby('ticker')['Close'].transform(safe_log_returns_calc)

            # Rolling volatility (calculated independently for each ticker)
            volatility = returns.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).std()
            ).reset_index(level=0, drop=True)

            # 🛡️ SAFETY FIX: Volatility reciprocal (low volatility anomaly)
            from numerical_stability import safe_divide
            inv_volatility = safe_divide(1.0, volatility, fill_value=0.0)

            # Exponential decay
            inv_vol_decayed = inv_volatility.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            results.append(inv_vol_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_volume_turnover(self, df: pd.DataFrame, windows: List[int], 
                                decay: int = 6) -> pd.Series:
        """Volume turnover factor"""
        results = []
        
        for window in windows:
            # Volume relative strength
            if 'volume' in df.columns:
                volume_ma = df.groupby('ticker')['volume'].transform(
                    lambda x: x.rolling(window=window, min_periods=max(1, window//2)).mean()
                )
                volume_ratio = df['volume'] / (volume_ma + 1e-9)
            else:
                # If no volume data, try amount or create synthetic
                if 'amount' in df.columns:
                    volume_ratio = df.groupby('ticker')['amount'].transform(
                        lambda x: x / (x.rolling(window=window, min_periods=max(1, window//2)).mean() + 1e-9)
                    )
                else:
                    # Create synthetic volume using price * constant
                    synthetic_volume = df['Close'] * 1000000  # Assume 1M shares
                    volume_ma = synthetic_volume.groupby(df['ticker']).transform(
                        lambda x: x.rolling(window=window, min_periods=max(1, window//2)).mean()
                    )
                    volume_ratio = synthetic_volume / (volume_ma + 1e-9)
            
            # Exponential decay
            volume_decayed = volume_ratio.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(volume_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_amihud_illiquidity(self, df: pd.DataFrame, windows: List[int], 
                                   decay: int = 6) -> pd.Series:
        """Amihud liquidity indicator: Reciprocal of price impact"""
        results = []
        
        for window in windows:
            # 🛡️ SAFETY FIX: Calculate daily returns with stability
            from numerical_stability import safe_log, safe_divide
            
            def safe_abs_log_returns(x):
                """安全的绝对对数收益率计算"""
                if len(x) <= 1:
                    return pd.Series(index=x.index, dtype=float)
                
                current_price = x
                past_price = x.shift(1)
                price_ratio = safe_divide(current_price, past_price, fill_value=1.0)
                log_returns = safe_log(price_ratio)
                return np.abs(log_returns)
            
            returns = df.groupby('ticker')['Close'].transform(safe_abs_log_returns)
            
            # 🛡️ SAFETY FIX: Amihud liquidity with safe division
            if 'amount' in df.columns:
                amihud = safe_divide(returns, df['amount'], fill_value=0.0)
            elif 'volume' in df.columns:
                # Alternative: use price * volume
                volume_value = df['Close'] * df['volume']
                amihud = safe_divide(returns, volume_value, fill_value=0.0)
            else:
                # Use synthetic volume
                synthetic_volume = df['Close'] * 1000000
                amihud = safe_divide(returns, synthetic_volume, fill_value=0.0)
            
            # Rolling average
            amihud_ma = amihud.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).mean()
            ).reset_index(level=0, drop=True)
            
            # Liquidity = 1 / Amihud (higher liquidity is better)
            liquidity = 1.0 / (amihud_ma + 1e-9)
            
            # Exponential decay
            liquidity_decayed = liquidity.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(liquidity_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_bid_ask_spread(self, df: pd.DataFrame, windows: List[int], 
                               decay: int = 6) -> pd.Series:
        """Bid-ask spread factor (simulated)"""
        results = []
        
        for window in windows:
            # If high-low price data available, use (high-low)/close  as spread proxy
            if 'High' in df.columns and 'Low' in df.columns:
                spread_proxy = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
            else:
                # Alternative: use price volatility as spread proxy
                price_vol = df.groupby('ticker')['Close'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std() / (x + 1e-9)
                )
                spread_proxy = price_vol
            
            # Rolling average spread
            spread_ma = spread_proxy.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).mean()
            ).reset_index(level=0, drop=True)
            
            # Narrow spread factor (smaller spread is better)
            narrow_spread = 1.0 / (spread_ma + 1e-6)
            
            # Exponential decay
            spread_decayed = narrow_spread.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(spread_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_residual_momentum(self, df: pd.DataFrame, windows: List[int], 
                                  decay: int = 6) -> pd.Series:
        """Residual momentum: Idiosyncratic momentum after removing market beta"""
        results = []
        
        for window in windows:
            # Calculate individual stock returns
            stock_returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(1))
            )
            
            # Calculate market returns (equal weight or market cap weighted)
            market_returns = df.groupby('date')['Close'].transform(
                lambda x: np.log(x.mean() / x.shift(1).mean())
            )
            
            # Rolling regression to calculate beta and residuals
            def calculate_residual_momentum(group):
                # Slice from externally pre-computed Series by index to avoid .name dependency
                group_returns = stock_returns.loc[group.index]
                group_market = market_returns.loc[group.index]
                
                residuals = []
                for i in range(len(group_returns)):
                    if i < window:
                        residuals.append(np.nan)
                        continue
                    
                    y = group_returns.iloc[i-window:i]  # Keep NaN for now
                    x = group_market.iloc[i-window:i]  # Keep NaN for now
                    
                    if len(y) < max(1, window//2) or len(x) != len(y):
                        residuals.append(np.nan)
                        continue
                    
                    try:
                        # Simple linear regression: stock return = alpha + beta * market return + residual
                        slope, intercept, _, _, _ = stats.linregress(x.values, y.values)
                        predicted = intercept + slope * x.iloc[-1]
                        residual = y.iloc[-1] - predicted
                        residuals.append(residual)
                    except:
                        residuals.append(0.0)
                
                return pd.Series(residuals, index=group_returns.index)
            
            residual_momentum = df.groupby('ticker').apply(calculate_residual_momentum)
            residual_momentum = residual_momentum.reset_index(level=0, drop=True)

            # Exponential decay
            residual_decayed = residual_momentum.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(residual_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    # ===== v2 New factors: Unified entry into class methods and registered =====
    def _compute_reversal_5(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Time-safe short-term reversal (1-5 days), with safety margin"""
        try:
            g = df.groupby('ticker')['Close']
            # Using T-2 to T-7 data, with safety margin
            rev = -(g.shift(2) / g.shift(7) - 1.0)
            rev_ema = rev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return rev_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Short-term reversal computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_amihud_illiquidity_new(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Improved Amihud illiquidity: More robust rolling median with EMA decay"""
        try:
            window = windows[0] if windows else 22
            returns_abs = df.groupby('ticker')['Close'].apply(lambda s: (s / s.shift(1) - 1).abs()).reset_index(level=0, drop=True)
            if 'amount' in df.columns:
                volume_dollar = df['amount'].replace(0, np.nan)
            elif 'volume' in df.columns:
                volume_dollar = (df['volume'] * df['Close']).replace(0, np.nan)
            else:
                volume_dollar = (1e6 * df['Close']).replace(0, np.nan)
            illiq = (returns_abs / volume_dollar).replace([np.inf, -np.inf], np.nan)
            illiq_rolling = illiq.groupby(df['ticker']).rolling(window, min_periods=max(1, window//2)).median().reset_index(level=0, drop=True)
            illiq_factor = -illiq_rolling
            illiq_ema = illiq_factor.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return illiq_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Amihud illiquidity computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_pead(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """PEAD（财报后漂移）event-driven proxy"""
        try:
            window = windows[0] if windows else 21
            returns_21d = df.groupby('ticker')['Close'].pct_change(periods=window).reset_index(level=0, drop=True)
            if 'volume' in df.columns:
                vol_ma = df.groupby('ticker')['volume'].rolling(window*2).mean().reset_index(level=0, drop=True)
                # Ensure proper index alignment for division
                vol_ratio = pd.Series(df['volume'].values / vol_ma.values, index=df.index)
                vol_anomaly = vol_ratio.groupby(df['ticker']).transform(lambda x: (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8))
            else:
                vol_anomaly = pd.Series(0.0, index=df.index)
            
            # Ensure index alignment
            returns_aligned = pd.Series(returns_21d.values, index=df.index)
            pead_signal = returns_aligned * (1 + vol_anomaly * 0.3)
            
            # Fix threshold calculation with proper index handling
            threshold = pead_signal.groupby(df['ticker']).rolling(252).quantile(0.8).reset_index(level=0, drop=True)
            threshold_aligned = pd.Series(threshold.values, index=df.index)
            pead_filtered = pead_signal.where(pead_signal.abs() > threshold_aligned.abs(), 0)
            
            result = pead_filtered.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return pd.Series(result.values, index=df.index, name='pead').fillna(0.0)
        except Exception as e:
            logger.warning(f"PEAD computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    # ===== New momentum factors =====
    
    def _compute_momentum_6_1(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """6-1momentum：(t-126 to t-21)的pricemomentum，排除最近1个month"""
        try:
            g = df.groupby('ticker')['Close']
            # 6 months ago to1 months ago returns
            momentum_6_1 = (g.shift(21) / g.shift(126) - 1.0)
            momentum_ema = momentum_6_1.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return momentum_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"6-1momentum computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_52w_new_high_proximity(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """52周proximity to new high: Current price as percentage of52周 high price"""
        try:
            window = 252  # 52周 ≈ 252个交易日
            g = df.groupby('ticker')['Close']
            max_52w = g.rolling(window=window, min_periods=min(window//2, 60)).max().reset_index(level=0, drop=True)
            current_price = df['Close']
            # Ensure index alignment
            max_52w_aligned = pd.Series(max_52w.values, index=df.index)
            proximity = current_price / max_52w_aligned
            # Apply decay with proper index handling
            result = proximity.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return pd.Series(result.values, index=df.index, name='new_high_proximity').fillna(0.0)
        except Exception as e:
            logger.warning(f"52周新高接近度 computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_low_beta_anomaly(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """低β异象：Usingrolling闭式估计或 ewm.cov implementation O(N) approximation, take negative (low beta is better)"""
        try:
            window = windows[0] if windows else 60
            close = df['Close']
            ret = close.groupby(df['ticker']).pct_change()
            mkt = close.groupby(df['date']).transform('mean')
            mkt_ret = mkt.groupby(df['ticker']).pct_change()  # 与个股索引对齐

            # Using ewm.cov 的向量化估计 beta = Cov(r_i, r_m)/Var(r_m)
            cov_im = ret.ewm(span=window, min_periods=max(10, window//3)).cov(mkt_ret)
            var_m = mkt_ret.ewm(span=window, min_periods=max(10, window//3)).var()
            beta = cov_im / (var_m + 1e-12)
            low_beta = (-beta).groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return low_beta.fillna(0.0)
        except Exception as e:
            logger.warning(f"低β异象 computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_idiosyncratic_volatility(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """特异volatility率：Using ewm.cov fast estimation of residual variance, take negative (low volatility is better)"""
        try:
            window = windows[0] if windows else 60
            close = df['Close']
            ret = close.groupby(df['ticker']).pct_change()
            mkt = close.groupby(df['date']).transform('mean')
            mkt_ret = mkt.groupby(df['ticker']).pct_change()

            cov_im = ret.ewm(span=window, min_periods=max(20, window//3)).cov(mkt_ret)
            var_m = mkt_ret.ewm(span=window, min_periods=max(20, window//3)).var()
            beta = cov_im / (var_m + 1e-12)
            alpha = ret.ewm(span=window, min_periods=max(20, window//3)).mean() - beta * mkt_ret.ewm(span=window, min_periods=max(20, window//3)).mean()
            residual = ret - (alpha + beta * mkt_ret)
            idio_vol = -residual.ewm(span=window, min_periods=max(20, window//3)).std()
            idio_vol_ema = idio_vol.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return idio_vol_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"特异volatility率 computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    # ===== Fundamental factors（Using proxydata） =====
    
    def _compute_earnings_surprise(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Earnings surpriseSUE：Standardize盈余惊喜（Usingprice反应作为 proxy）"""
        try:
            window = windows[0] if windows else 63  # Quarter
            # Usingprice在财报期间的异常反应作为SUE proxy
            returns = df.groupby('ticker')['Close'].pct_change()
            # Quarter超额return率作为SUE proxy
            quarterly_returns = df.groupby('ticker')['Close'].pct_change(periods=window)
            market_returns = df.groupby('date')['Close'].transform('mean').pct_change(periods=window)
            excess_returns = quarterly_returns - market_returns
            # Standardize
            sue_proxy = excess_returns.groupby(df['ticker']).transform(
                lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
            )
            sue_ema = sue_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return sue_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Earnings surpriseSUE computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_analyst_revision(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """AnalystEPS上调修正（Usingmomentum变化率作为 proxy）"""
        try:
            # Usingmomentum变化作为Analyst预期修正的 proxy
            short_momentum = df.groupby('ticker')['Close'].pct_change(21)  # 1month
            long_momentum = df.groupby('ticker')['Close'].pct_change(63)   # 3month
            revision_proxy = short_momentum - long_momentum
            revision_ema = revision_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return revision_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Analyst修正 computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_ebit_ev(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """EBIT/EVreturn率（Usingreturn率 proxy）"""
        try:
            # Using基于price的return率 proxyEBIT/EV
            if 'volume' in df.columns:
                enterprise_value = df['Close'] * df['volume']  # SimplifiedEV proxy
                ebit_proxy = df.groupby('ticker')['Close'].pct_change(252).abs()  # Annualized return asEBIT proxy
                ebit_ev = ebit_proxy / (enterprise_value / enterprise_value.rolling(252).mean())
                return ebit_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"EBIT/EV computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_fcf_ev(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Free cash flow yieldFCF/EV（Using现金流 proxy）"""
        try:
            # Using基于volume和price的现金流 proxy
            if 'amount' in df.columns:
                fcf_proxy = df['amount'] / df['Close']  # amount/price作为现金流 proxy
                if 'volume' in df.columns:
                    ev_proxy = df['Close'] * df['volume']
                else:
                    ev_proxy = df['Close'] * 1000000  # synthetic volume
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
            elif 'volume' in df.columns:
                fcf_proxy = df['volume'] * df['Close'] / df['Close']  # volume as proxy
                ev_proxy = df['Close'] * df['volume']
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
                fcf_ev = fcf_ev.groupby(df['ticker']).transform(
                    lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
                )
                return fcf_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"FCF/EV computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Earnings yieldE/P（市盈率倒数的 proxy）"""
        try:
            # Usingreturn率历史data作为E/P proxy
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            earnings_yield = annual_return / df['Close'] * 100  # Standardize
            return earnings_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).apply(lambda x: self.safe_fillna(x, df))
        except Exception as e:
            logger.warning(f"Earnings yieldE/P computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_sales_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Sales yieldS/P（市销率倒数的 proxy）"""
        try:
            # Usingvolume作为销售额 proxy
            if 'volume' in df.columns:
                sales_proxy = df['volume']
                sales_yield = sales_proxy / (df['Close'] + 1e-9)
                sales_yield = sales_yield.groupby(df['ticker']).transform(
                    lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
                )
                return sales_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"Sales yieldS/P computation failed: {e}")
            return pd.Series(0.0, index=df.index)
 
    # ===== 高级Alphafactor（暂时移除复杂implementation，保持基础功能） =====
    
    def _compute_gross_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Gross marginGP/Assets（简化implementation）"""
        try:
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            return annual_return.apply(lambda x: self.safe_fillna(x, df))
        except Exception as e:
            logger.warning(f"Gross margin computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_operating_profitability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Operating profitability（简化implementation）"""
        try:
            if 'volume' in df.columns:
                efficiency = df['volume'] / (df['Close'] + 1e-9)
                return efficiency.apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"Operating profitability computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    # 为所有其他高级factor添加简化implementation
    def _compute_roe_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROEneutralize（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(252)
            return returns.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_roic_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROICneutralize（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(126)
            return returns.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net margin（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(63)
            return returns.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_cash_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Cash yield（简化implementation）"""
        try:
            if 'amount' in df.columns:
                cash_yield = df['amount'] / (df['Close'] + 1e-9)
                return cash_yield.apply(lambda x: self.safe_fillna(x, df))
            elif 'volume' in df.columns:
                cash_yield = (df['volume'] * df['Close']) / (df['Close'] + 1e-9)
                return cash_yield.apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_shareholder_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Shareholder yield（简化implementation）"""
        try:
            if 'volume' in df.columns:
                volume_ma = df.groupby('ticker')['volume'].rolling(22).mean()
                ratio = df['volume'] / (volume_ma + 1e-9)
                return ratio.apply(lambda x: self.safe_fillna(x, df))
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    # Accrual factors
    def _compute_total_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Total accruals（简化implementation）"""
        try:
            price_change = df.groupby('ticker')['Close'].pct_change()
            return -price_change.apply(lambda x: self.safe_fillna(x, df))  # Take negative
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_working_capital_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Working capital accruals（简化implementation）"""
        try:
            if 'volume' in df.columns:
                wc_proxy = df.groupby('ticker')['volume'].pct_change()
                return -wc_proxy.apply(lambda x: self.safe_fillna(x, df))  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_operating_assets(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net operating assets（简化implementation）"""
        try:
            if 'volume' in df.columns:
                noa_proxy = df['volume'] / (df['Close'] + 1e-9)
                return -noa_proxy.pct_change().apply(lambda x: self.safe_fillna(x, df))  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_asset_growth(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Asset growth（简化implementation）"""
        try:
            if 'volume' in df.columns:
                market_value = df['Close'] * df['volume']
                growth = market_value.groupby(df['ticker']).pct_change(252)
                return -growth.apply(lambda x: self.safe_fillna(x, df))  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_equity_issuance(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net equity issuance（简化implementation）"""
        try:
            if 'volume' in df.columns:
                volume_spike = df.groupby('ticker')['volume'].pct_change()
                return -volume_spike.apply(lambda x: self.safe_fillna(x, df))  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_investment_factor(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Investment factor（简化implementation）"""
        try:
            # Fix index alignment issue
            price_vol = df.groupby('ticker')['Close'].rolling(22).std().reset_index(level=0, drop=True)
            result = -price_vol.fillna(0.0)  # Take negative
            # Ensure proper index alignment
            return pd.Series(result.values, index=df.index, name='investment_factor').fillna(0.0)
        except Exception as e:
            logger.warning(f"Investment factor computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    # Quality score factors
    def _compute_piotroski_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """PiotroskiScore（简化implementation）"""
        try:
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            score = (annual_return > 0).astype(float)
            return score.apply(lambda x: self.safe_fillna(x, df) if x.isna().any() else x.fillna(0.5))
        except:
            return pd.Series(0.5, index=df.index)
    
    def _compute_ohlson_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """OhlsonScore（简化implementation）"""
        try:
            # ✅ FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            price_vol = df.groupby('ticker')[close_col].rolling(126).std() / df[close_col]
            return -price_vol.apply(lambda x: self.safe_fillna(x, df))  # Take negative，lower risk is better
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_altman_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """AltmanScore（简化implementation）"""
        try:
            # ✅ FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            stability = -returns.rolling(126).std()  # Stability
            return stability.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_qmj_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """QMJ质量Score（简化implementation）"""
        try:
            # ✅ FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            quality = returns.rolling(252).mean() / (returns.rolling(252).std() + 1e-8)
            return quality.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_stability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """盈利Stability（简化implementation）"""
        try:
            # ✅ FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            stability = -returns.rolling(252).std()  # lower volatility is better
            return stability.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0.0, index=df.index)
 
    # ========== Main Computation Pipeline ==========
    
    def compute_all_alphas(self, df) -> pd.DataFrame:
        """
        Compute all Alpha factors
        
        Args:
            df: DataFrame or dict containing price data, must have columns: ['date', 'ticker', 'Close', 'amount', ...]
            
        Returns:
            DataFrame containing all Alpha factors
        """
        logger.info(f"Starting computation of{len(self.config['alphas'])} Alpha factors")
        
        # 🔧 修复数据格式问题：确保输入是DataFrame
        if isinstance(df, dict):
            # 如果输入是dict，尝试转换为DataFrame
            try:
                if 'data' in df and isinstance(df['data'], pd.DataFrame):
                    df_work = df['data'].copy()
                else:
                    # 尝试直接从dict构建DataFrame
                    df_work = pd.DataFrame(df)
                logger.debug(f"Successfully converted dict input to DataFrame: {df_work.shape}")
            except Exception as e:
                logger.error(f"Failed to convert dict to DataFrame: {e}")
                raise ValueError(f"Cannot convert input dict to DataFrame: {e}")
        elif isinstance(df, pd.DataFrame):
            df_work = df.copy()
        else:
            raise ValueError(f"Input must be DataFrame or dict, got {type(df)}")
        
        # Ensure required columns exist
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df_work.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add metadata columns (if not exist) - Use copy to avoid modifying original data
        for col in ['COUNTRY', 'SECTOR', 'SUBINDUSTRY']:
            if col not in df_work.columns:
                df_work[col] = 'Unknown'
        
        alpha_results = {}
        computation_times = {}
        
        for alpha_config in self.config['alphas']:
            alpha_name = alpha_config['name']
            alpha_kind = alpha_config['kind']
            
            try:
                start_time = pd.Timestamp.now()
                
                # Get parameters
                windows = alpha_config.get('windows', [22])
                decay = alpha_config.get('decay', 6)
                delay = alpha_config.get('delay', 1)  # 配置文件中的delay参数
                
                # ✅ NEW: 获取因子特定的滞后配置
                factor_specific_lag = 0
                if self.lag_manager:
                    factor_specific_lag = self.lag_manager.get_lag(alpha_name)
                    if factor_specific_lag != delay:
                        logger.debug(f"因子{alpha_name}: 使用差异化滞后T-{factor_specific_lag}（原delay={delay}）")
                
                if alpha_kind == 'hump':
                    # Special handlinghump变换
                    base_name = alpha_config['base']
                    if base_name not in alpha_results:
                        logger.warning(f"Hump factor{alpha_name}'s base factor{base_name} not found")
                        continue
                    
                    base_factor = alpha_results[base_name].copy()
                    hump_level = alpha_config['hump']
                    alpha_factor = self.hump_transform(base_factor, hump=hump_level)
                else:
                    # Regular factor computation - All factors integrated into this module
                    if alpha_kind not in self.alpha_functions:
                        logger.warning(f"Unknown Alpha type: {alpha_kind}")
                        continue
                    
                    alpha_func = self.alpha_functions[alpha_kind]
                    alpha_factor = alpha_func(
                        df=df_work,
                        windows=windows,
                        decay=decay
                    )
                
                # Data processing pipeline
                alpha_factor = self._process_alpha_pipeline(
                    df=df_work,
                    alpha_factor=alpha_factor,
                    alpha_config=alpha_config,
                    alpha_name=alpha_name
                )
                
                # ✅ NEW: 应用差异化滞后策略
                if self.lag_manager and factor_specific_lag > 0:
                    # 使用因子特定的滞后
                    alpha_factor = alpha_factor.groupby(df_work['ticker']).shift(factor_specific_lag)
                    logger.debug(f"应用差异化滞后 T-{factor_specific_lag} 于 {alpha_name}")
                elif delay and delay > 0:
                    # 回退到配置文件中的delay
                    alpha_factor = alpha_factor.groupby(df_work['ticker']).shift(delay)
                
                # ✅ REMOVED: 不再使用全局统一的lag，改为差异化滞后
                # 原有的global_lag逻辑已被差异化滞后替代
                
                alpha_results[alpha_name] = alpha_factor
                computation_times[alpha_name] = (pd.Timestamp.now() - start_time).total_seconds()
                
                logger.info(f"SUCCESS {alpha_name}:  computation completed ({computation_times[alpha_name]:.2f}s)")
                
            except Exception as e:
                logger.error(f"FAILED {alpha_name}:  computation failed - {e}")
                continue
        
        # 更新Statistics
        self.stats['computation_times'].update(computation_times)
        
        # Build result DataFrame, preserve original columns
        result_df = df_work.copy()
        for alpha_name, alpha_series in alpha_results.items():
            result_df[alpha_name] = alpha_series
        
        if alpha_results:
            logger.info(f"Alpha computation completed，共{len(alpha_results)} factors")
            
            # ✅ PERFORMANCE FIX: Apply factor orthogonalization to remove redundancy
            try:
                # Get all alpha factor columns
                alpha_cols = [col for col in result_df.columns if col.startswith('alpha_')]
                
                if len(alpha_cols) >= 2:
                    logger.info(f"开始正交化{len(alpha_cols)}个Alpha因子，消除冗余")
                    
                    # Apply orthogonalization with correlation threshold 0.8
                    orthogonalizer = FactorOrthogonalizer(
                        method="correlation_filter",  # 使用相关性过滤，更适合Alpha因子
                        correlation_threshold=0.8     # 移除相关性>0.8的冗余因子
                    )
                    
                    # Create temporary DataFrame for orthogonalization
                    ortho_data = result_df[['date', 'ticker'] + alpha_cols].copy()
                    orthogonalized_data = orthogonalizer.fit_transform(ortho_data)
                    
                    # Update result with orthogonalized factors
                    for col in orthogonalizer.retained_factors or alpha_cols:
                        if col in orthogonalized_data.columns:
                            result_df[col] = orthogonalized_data[col]
                    
                    # Remove redundant factors that were filtered out
                    removed_factors = [col for col in alpha_cols if col not in (orthogonalizer.retained_factors or alpha_cols)]
                    for col in removed_factors:
                        if col in result_df.columns:
                            result_df = result_df.drop(columns=[col])
                    
                    retained_count = len(orthogonalizer.retained_factors or alpha_cols)
                    removed_count = len(alpha_cols) - retained_count
                    logger.info(f"✅ 因子正交化完成: 保留{retained_count}个, 移除{removed_count}个冗余因子")
                    
                    # Get factor importance if available
                    importance = orthogonalizer.get_factor_importance()
                    if importance:
                        logger.debug(f"因子重要性排序: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
                
            except Exception as e:
                logger.warning(f"因子正交化失败，继续使用原始因子: {e}")
            
            # ✅ PERFORMANCE FIX: Apply dynamic factor weighting
            try:
                final_alpha_cols = [col for col in result_df.columns if col.startswith('alpha_')]
                
                if len(final_alpha_cols) >= 2 and 'future_return_10d' in result_df.columns:
                    logger.info(f"开始计算{len(final_alpha_cols)}个Alpha因子的动态权重")
                    
                    # Calculate dynamic weights based on IC performance
                    dynamic_weights = self.calculate_dynamic_weights(
                        df=result_df,
                        alpha_cols=final_alpha_cols,
                        target_col='future_return_10d'
                    )
                    
                    # Apply dynamic weights to create a composite alpha
                    if dynamic_weights:
                        composite_alpha = self.apply_dynamic_weights(
                            df=result_df,
                            alpha_cols=final_alpha_cols,
                            weights=dynamic_weights
                        )
                        
                        # Add composite alpha to result
                        result_df['alpha_composite_dynamic'] = composite_alpha
                        
                        logger.info("✅ 动态权重合成Alpha创建成功")
                
            except Exception as e:
                logger.warning(f"动态权重应用失败: {e}")
                
        else:
            logger.error("所有Alphafactor computation failed")
        
        return result_df
    
    def _process_alpha_pipeline(self, df: pd.DataFrame, alpha_factor: pd.Series, 
                               alpha_config: Dict, alpha_name: str) -> pd.Series:
        """Alpha factor processing pipeline：winsorize -> neutralize -> zscore -> transform"""
        
        # 1. Winsorizeremove outliers
        winsorize_std = self.config.get('winsorize_std', 2.5)
        alpha_factor = self.winsorize_series(alpha_factor, k=winsorize_std)
        
        # 2. 构建临时DataFrame进行neutralize
        temp_df = df[['date', 'ticker'] + self.config['neutralization']].copy()
        temp_df[alpha_name] = alpha_factor
        
        # 3. neutralize（default关闭，避免与全局Pipeline重复；仅研究Using时打开）
        if self.config.get('enable_alpha_level_neutralization', False):
            for neutralize_level in self.config['neutralization']:
                if neutralize_level in temp_df.columns:
                    alpha_factor = self.neutralize_factor(
                        temp_df, alpha_name, [neutralize_level]
                    )
                    temp_df[alpha_name] = alpha_factor
        
        # 4. ✅ PERFORMANCE FIX: 横截面标准化，消除市场风格偏移
        try:
            from cross_sectional_standardization import CrossSectionalStandardizer
            
            standardizer = CrossSectionalStandardizer(method="robust_zscore")
            standardized_df = standardizer.fit_transform(
                temp_df[['date', 'ticker', alpha_name]], 
                feature_cols=[alpha_name]
            )
            alpha_factor = standardized_df[alpha_name]
            
        except Exception as e:
            logger.warning(f"横截面标准化失败，使用传统zscore: {e}")
            # 回退到传统zscore方法
            alpha_factor = self.zscore_by_group(
                temp_df, alpha_name, ['date']
            )
        
        # 5. Transform (rank or keep original)
        xform = alpha_config.get('xform', 'zscore')
        if xform == 'rank':
            alpha_factor = temp_df.groupby('date')[alpha_name].transform(
                lambda x: self.rank_transform(x)
            )
        
        return alpha_factor
    
    def compute_oof_scores(self, alpha_df: pd.DataFrame, target: pd.Series, 
                          dates: pd.Series, metric: str = 'ic') -> pd.Series:
        """
        computationOut-of-FoldScore
        
        Args:
            alpha_df: Alpha factor DataFrame
            target: Target variable
            dates: Date sequence
            metric: Score指标 ('ic', 'sharpe', 'fitness')
            
        Returns:
            每个Alpha的OOFScore
        """
        logger.info(f"Starting computation ofOOFScore，指标: {metric}")

        # Unify indices to avoid boolean index misalignment
        try:
            alpha_index = alpha_df.index
            common_index = alpha_index
            if isinstance(target, pd.Series):
                common_index = common_index.intersection(target.index)
            if isinstance(dates, pd.Series):
                common_index = common_index.intersection(dates.index)

            if len(common_index) == 0:
                logger.warning("OOFScore跳过：alpha/target/dates无共同索引")
                return pd.Series(dtype=float)

            alpha_df = alpha_df.loc[common_index]
            if isinstance(target, pd.Series):
                target = target.loc[common_index]
            else:
                target = pd.Series(target, index=common_index)
            if isinstance(dates, pd.Series):
                dates = dates.loc[common_index]
            else:
                dates = pd.Series(dates, index=common_index)
        except Exception as e:
            logger.warning(f"Index alignment failed, trying to continue：{e}")

        # Only evaluate numerical factor columns, exclude ID/price/metadata columns
        exclude_cols = set(['date','ticker','COUNTRY','SECTOR','SUBINDUSTRY',
                            'Open','High','Low','Close','Adj Close',
                            'open','high','low','close','adj_close','volume','amount'])
        factor_cols = [c for c in alpha_df.columns
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(alpha_df[c])]

        # 🚫 SSOT违规检测：阻止内部CV创建
        from .ssot_violation_detector import block_internal_cv_creation
        block_internal_cv_creation("Alpha策略中的TimeSeriesSplit")
        unique_dates = sorted(dates.unique())
        
        scores = {}
        for col in factor_cols:
            col_scores = []
            
            for train_idx, test_idx in tscv.split(unique_dates):
                # Get test period data
                test_dates = [unique_dates[i] for i in test_idx]
                # Usingnumpy布尔数组，避免索引不一致
                test_mask = dates.isin(test_dates).values
                
                if test_mask.sum() == 0:
                    continue
                
                # Usingiloc配合布尔数组，确保位置索引对齐
                y_test = target.iloc[test_mask]
                x_test = alpha_df[col].iloc[test_mask]
                
                # Reset index to ensure alignment
                y_test = y_test.reset_index(drop=True)
                x_test = x_test.reset_index(drop=True)
                
                # Remove NaN values
                valid_mask = ~(x_test.isna() | y_test.isna())
                if valid_mask.sum() < 10:  # Minimum required10 valid samples
                    continue
                
                # 直接Using布尔索引，因为索引已重置
                x_valid = x_test[valid_mask]
                y_valid = y_test[valid_mask]
                
                # computationScore
                if metric == 'ic':
                    score = np.corrcoef(x_valid.values, y_valid.values)[0, 1]
                elif metric == 'sharpe':
                    returns = x_valid.values * y_valid.values
                    score = returns.mean() / (returns.std(ddof=0) + 1e-12)
                elif metric == 'fitness':
                    # Information Coefficient * sqrt(sample size)
                    ic = np.corrcoef(x_valid.values, y_valid.values)[0, 1]
                    score = ic * np.sqrt(len(x_valid))
                else:
                    score = 0.0
                
                if not np.isnan(score):
                    col_scores.append(score)
            
            scores[col] = np.nanmean(col_scores) if col_scores else 0.0
        
        # 更新Statistics
        self.stats['ic_stats'] = scores
        
        result = pd.Series(scores, name=f'oof_{metric}')
        logger.info(f"OOFScorecompleted，average{metric}: {result.mean():.4f}")
        
        return result
    
    def compute_bma_weights(self, scores: pd.Series, temperature: float = None, use_weight_hints: bool = True) -> pd.Series:
        """
        基于ScorecomputationBMA weights，支持weight_hint prior
        
        Args:
            scores: OOFScore
            temperature: Temperature coefficient, controls weight concentration
            use_weight_hints: 是否Usingweight_hint作为 priorweight
            
        Returns:
            BMA weights
        """
        if temperature is None:
            temperature = self.config.get('temperature', 1.2)
        
        # Getweight_hint priorweight
        weight_hints = {}
        if use_weight_hints:
            for alpha_config in self.config.get('alphas', []):
                alpha_name = alpha_config['name']
                if alpha_name in scores.index:
                    weight_hints[alpha_name] = alpha_config.get('weight_hint', 0.05)
        
        # StandardizeScore
        scores_std = (scores - scores.mean()) / (scores.std(ddof=0) + 1e-12)
        scores_scaled = scores_std / max(temperature, 1e-3)
        
        # Log-sum-exp softmax（numerically stable）
        max_score = scores_scaled.max()
        exp_scores = np.exp(scores_scaled - max_score)
        
        # Combineweight_hint prior
        if weight_hints and use_weight_hints:
            hint_weights = pd.Series(weight_hints).reindex(scores.index, fill_value=0.05)
            hint_weights = hint_weights / hint_weights.sum()  # Standardize
            
            # 贝叶斯更新： prior * likelihood
            posterior_weights = hint_weights * exp_scores
            weights = posterior_weights / posterior_weights.sum()
            
            logger.info("Usingweight_hint prior进行贝叶斯weight更新")
        else:
            # Regular softmax
            eps = 1e-6
            weights = (exp_scores + eps) / (exp_scores.sum() + eps * len(exp_scores))
        
        weights_series = pd.Series(weights, index=scores.index, name='bma_weights')
        
        logger.info(f"BMA weights computation completed，weight分布: max={weights.max():.3f}, min={weights.min():.3f}")
        logger.info(f"Main factor weights: {weights_series.nlargest(5).to_dict()}")
        
        return weights_series
    
    def combine_alphas(self, alpha_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        UsingBMA weightsportfolioAlphafactor
        
        Args:
            alpha_df: Alpha factor DataFrame
            weights: BMA weights
            
        Returns:
            Combined Alpha signal
        """
        # 仅Using数值型factor列，排除元data
        exclude_cols = set(['date','ticker','COUNTRY','SECTOR','SUBINDUSTRY',
                            'Open','High','Low','Close','Adj Close',
                            'open','high','low','close','adj_close','volume','amount'])
        factor_cols = [c for c in alpha_df.columns
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(alpha_df[c])]
        if not factor_cols:
            return pd.Series(index=alpha_df.index, dtype=float)

        # Ensure weight alignment (column direction)
        aligned_weights = weights.reindex(factor_cols, fill_value=0.0)
        total_w = aligned_weights.sum()
        if total_w <= 0:
            aligned_weights[:] = 1.0 / len(aligned_weights)
        else:
            aligned_weights = aligned_weights / total_w

        # Column-wise multiplication to avoid type errors from row index alignment
        combined_signal = alpha_df[factor_cols].mul(aligned_weights, axis=1).sum(axis=1)
        
        logger.info(f"Alpha combination completed, signal range: [{combined_signal.min():.4f}, {combined_signal.max():.4f}]")
        
        return combined_signal
    
    def apply_trading_filters(self, signal: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Apply trading filters：humpgating, truncation, position limits
        
        Args:
            signal: Raw signal
            df: DataFrame containing date information
            
        Returns:
            Filtered trading signal
        """
        logger.info("Apply trading filters")
        
        # 1. 截面Standardize
        temp_df = df[['date', 'ticker']].copy()
        temp_df['signal'] = signal
        
        filtered_signal = temp_df.groupby('date')['signal'].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        )
        
        # 2. Humpgating
        hump_levels = self.config.get('hump_levels', [0.003, 0.008])
        for hump_level in hump_levels:
            filtered_signal = self.hump_transform(filtered_signal, hump=hump_level)
        
        # 3. Truncation controls concentration
        truncation = self.config.get('truncation', 0.10)
        if truncation > 0:
            lower_q = filtered_signal.quantile(truncation)
            upper_q = filtered_signal.quantile(1 - truncation)
            filtered_signal = filtered_signal.clip(lower=lower_q, upper=upper_q)
        
        # 4. Only keep top and bottom signals
        top_frac = self.config.get('top_fraction', 0.10)
        if top_frac > 0:
            def mask_top_bottom(x):
                if len(x) < 10:
                    return x
                lo_threshold = x.quantile(top_frac)
                hi_threshold = x.quantile(1 - top_frac)
                return x.where((x <= lo_threshold) | (x >= hi_threshold), 0.0)
            
            temp_df['signal'] = filtered_signal
            filtered_signal = temp_df.groupby('date')['signal'].transform(mask_top_bottom)
        
        logger.info(f"Trading filter completed, non-zero signal ratio: {(filtered_signal != 0).mean():.2%}")
        
        return filtered_signal
    
    # ========== Sentiment Factor Functions ==========
    # 将情绪数据作为独立的机器学习特征，无硬编码权重
    
    def _compute_news_sentiment(self, df: pd.DataFrame, windows: List[int] = [5, 22], 
                               decay: int = 6) -> pd.Series:
        """计算新闻情绪Alpha因子"""
        try:
            # 查找新闻情绪相关列
            news_cols = [col for col in df.columns if col.startswith('news_')]
            
            if not news_cols:
                logger.debug("未找到新闻情绪数据列")
                return pd.Series(0, index=df.index)
            
            # 使用最重要的新闻情绪指标
            primary_cols = ['news_sentiment_mean', 'news_sentiment_momentum_1d', 'news_news_count']
            available_cols = [col for col in primary_cols if col in df.columns]
            
            if available_cols:
                # 计算复合新闻情绪因子（不使用硬编码权重，让模型学习）
                sentiment_factor = pd.Series(0, index=df.index)
                for col in available_cols:
                    col_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                    # 应用时间衰减
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / len(available_cols)  # 简单平均而非硬编码权重
                
                return sentiment_factor.apply(lambda x: self.safe_fillna(x, df))
            else:
                # 使用第一个可用的新闻情绪列
                col = news_cols[0]
                sentiment_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                return self.decay_linear(sentiment_factor, decay)
                
        except Exception as e:
            logger.warning(f"计算新闻情绪因子失败: {e}")
            return pd.Series(0, index=df.index)
    
    def _compute_market_sentiment(self, df: pd.DataFrame, windows: List[int] = [5, 22], 
                                 decay: int = 6) -> pd.Series:
        """计算市场情绪Alpha因子（基于SP500数据）"""
        try:
            # 查找市场情绪相关列
            market_cols = [col for col in df.columns if col.startswith('market_') or 'sp500' in col]
            
            if not market_cols:
                logger.debug("未找到市场情绪数据列")
                return pd.Series(0, index=df.index)
            
            # 优先使用关键市场情绪指标
            priority_cols = [col for col in market_cols if any(keyword in col for keyword in 
                            ['momentum', 'volatility', 'fear', 'sentiment'])]
            
            if priority_cols:
                # 计算复合市场情绪因子
                sentiment_factor = pd.Series(0, index=df.index)
                for col in priority_cols[:3]:  # 限制最多3个因子避免过度拟合
                    col_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / min(3, len(priority_cols))
                
                return sentiment_factor.apply(lambda x: self.safe_fillna(x, df))
            else:
                # 使用第一个可用的市场情绪列
                col = market_cols[0]
                sentiment_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                return self.decay_linear(sentiment_factor, decay)
                
        except Exception as e:
            logger.warning(f"计算市场情绪因子失败: {e}")
            return pd.Series(0, index=df.index)
    
    def _compute_fear_greed_sentiment(self, df: pd.DataFrame, windows: List[int] = [5, 22], 
                                     decay: int = 6) -> pd.Series:
        """计算恐惧贪婪指数Alpha因子"""
        try:
            # 查找恐惧贪婪相关列
            fg_cols = [col for col in df.columns if 'fear_greed' in col or 'fear' in col or 'greed' in col]
            
            if not fg_cols:
                logger.debug("未找到恐惧贪婪指数数据列")
                return pd.Series(0, index=df.index)
            
            # 优先使用规范化的恐惧贪婪指标
            priority_cols = ['fear_greed_normalized', 'market_fear_level', 'market_greed_level']
            available_cols = [col for col in priority_cols if col in df.columns]
            
            if available_cols:
                # 计算复合恐惧贪婪因子
                sentiment_factor = pd.Series(0, index=df.index)
                for col in available_cols:
                    col_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / len(available_cols)
                
                return sentiment_factor.apply(lambda x: self.safe_fillna(x, df))
            else:
                # 使用第一个可用的恐惧贪婪列
                col = fg_cols[0]
                sentiment_factor = df[col].apply(lambda x: self.safe_fillna(x, df))
                # 如果是原始值，进行归一化
                if 'value' in col.lower():
                    sentiment_factor = (sentiment_factor - 50) / 50  # 归一化到[-1,1]
                return self.decay_linear(sentiment_factor, decay)
                
        except Exception as e:
            logger.warning(f"计算恐惧贪婪情绪因子失败: {e}")
            return pd.Series(0, index=df.index)
    
    def _compute_sentiment_momentum(self, df: pd.DataFrame, windows: List[int] = [5, 22], 
                                   decay: int = 6) -> pd.Series:
        """计算情绪动量因子"""
        try:
            # 查找情绪动量相关列
            momentum_cols = [col for col in df.columns if 'sentiment' in col and 'momentum' in col]
            
            if not momentum_cols:
                # 如果没有现成的情绪动量列，从基础情绪因子计算
                sentiment_cols = [col for col in df.columns if any(prefix in col for prefix in 
                                 ['news_sentiment_mean', 'fear_greed_normalized'])]
                
                if sentiment_cols:
                    # 计算短期情绪动量
                    sentiment_factor = pd.Series(0, index=df.index)
                    for col in sentiment_cols[:2]:  # 最多使用2个基础情绪因子
                        col_data = df[col].apply(lambda x: self.safe_fillna(x, df))
                        # 计算短期动量（3天）
                        momentum = col_data.groupby(df['ticker']).diff(3)
                        sentiment_factor += momentum / len(sentiment_cols[:2])
                    
                    return self.decay_linear(sentiment_factor.apply(lambda x: self.safe_fillna(x, df)), decay)
                else:
                    return pd.Series(0, index=df.index)
            else:
                # 使用现成的情绪动量列
                sentiment_factor = df[momentum_cols[0]].apply(lambda x: self.safe_fillna(x, df))
                return self.decay_linear(sentiment_factor, decay)
                
        except Exception as e:
            logger.warning(f"计算情绪动量因子失败: {e}")
            return pd.Series(0, index=df.index)
    
    # REMOVED: 复杂的情绪波动率因子实现 - 数据质量差，计算开销大
    def _compute_sentiment_volatility(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """DEPRECATED: 情绪波动率因子已删除"""
        return pd.Series(0, index=df.index)
    
    # ========== End Sentiment Factors ==========
    
    # ========== Advanced Behavioral Factors ==========
    
    # REMOVED: 超复杂的散户羊群效应因子实现 - 计算成本最高，效果递减
    def _compute_retail_herding_effect(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """DEPRECATED: 散户羊群效应因子已删除 - 计算成本过高"""
        return pd.Series(0, index=df.index)
    
    # REMOVED: APM动量反转因子 - 过度工程化，缺乏日内数据支持
    def _compute_apm_momentum_reversal(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """DEPRECATED: APM动量反转因子已删除 - 过度工程化，实际效果有限"""
        return pd.Series(0, index=df.index)
    
    # ========== 🔥 NEW: Real Polygon Training技术指标集成 ==========
    
    def _compute_sma_10(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # ✅ PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_10', 10)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            # 转换为相对强度信号：当前价格相对均线的偏离度
            return ((df['Close'] / sma) - 1).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sma_20(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # ✅ PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_20', 20)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            return ((df['Close'] / sma) - 1).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sma_50(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # ✅ PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_50', 50)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            # 如果数据不足，回退到可用数据的均线
            available_days = min(20, len(df))
            if available_days >= 10:
                sma = df['Close'].rolling(available_days).mean()
                return ((df['Close'] / sma) - 1).apply(lambda x: self.safe_fillna(x, df))
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            return ((df['Close'] / sma) - 1).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_rsi(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """相对强弱指数(RSI,可优化参数)"""
        # ✅ PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('rsi', 14)
        
        if 'Close' not in df.columns or len(df) < optimal_window + 1:
            return pd.Series(0, index=df.index)
        
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(optimal_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(optimal_window).mean()
            
            rs = gain / loss.replace(0, np.nan)  # 避免除零
            rsi = 100 - (100 / (1 + rs))
            
            # 转换为标准化信号：-1到1范围
            rsi_normalized = (rsi - 50) / 50
            return rsi_normalized.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_bb_position(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """布林带位置"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # 计算价格在布林带中的相对位置 (0-1)
            bb_position = (df['Close'] - lower_band) / (upper_band - lower_band)
            
            # 转换为标准化信号：-1到1范围 (0.5映射到0)
            return ((bb_position - 0.5) * 2).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_macd(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """MACD指标"""
        if 'Close' not in df.columns or len(df) < 26:
            return pd.Series(0, index=df.index)
        
        try:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            
            # 标准化MACD值
            return (macd / df['Close']).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_macd_signal(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """MACD信号线"""
        if 'Close' not in df.columns or len(df) < 35:
            return pd.Series(0, index=df.index)
        
        try:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            return (signal / df['Close']).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_macd_histogram(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """MACD柱状图 (MACD - Signal)"""
        if 'Close' not in df.columns or len(df) < 35:
            return pd.Series(0, index=df.index)
        
        try:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return (histogram / df['Close']).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_price_momentum_5d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """5日价格动量"""
        if 'Close' not in df.columns or len(df) < 6:
            return pd.Series(0, index=df.index)
        
        try:
            momentum_5d = df['Close'].pct_change(5)
            return momentum_5d.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_price_momentum_20d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """20日价格动量"""
        if 'Close' not in df.columns or len(df) < 21:
            # 如果数据不足，使用5日动量
            return self._compute_price_momentum_5d(df, **kwargs)
        
        try:
            momentum_20d = df['Close'].pct_change(20)
            return momentum_20d.apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_volume_ratio(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """成交量比率"""
        if 'Volume' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            volume_ma = df['Volume'].rolling(20).mean()
            volume_ratio = df['Volume'] / volume_ma.replace(0, np.nan)
            
            # 对数变换以标准化极端值
            return np.log1p(volume_ratio - 1).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    # ========== 🔥 NEW: Real Polygon Training风险指标集成 ==========
    
    def _compute_max_drawdown(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """最大回撤"""
        if 'Close' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change().apply(lambda x: self.safe_fillna(x, df))
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            # 使用滚动窗口计算最大回撤
            max_drawdown = drawdown.rolling(20, min_periods=5).min()
            
            # 返回回撤的绝对值作为风险信号
            return abs(max_drawdown).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sharpe_ratio(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """夏普比率（滚动计算）"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change().apply(lambda x: self.safe_fillna(x, df))
            
            # 滚动计算夏普比率 (假设无风险利率为年化2%)
            risk_free_daily = 0.02 / 252
            excess_returns = returns - risk_free_daily
            
            rolling_mean = excess_returns.rolling(20, min_periods=10).mean()
            rolling_std = returns.rolling(20, min_periods=10).std()
            
            sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
            
            # 标准化夏普比率到合理范围
            return np.tanh(sharpe / 2).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_var_95(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """95%置信度的风险价值(VaR)"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change().apply(lambda x: self.safe_fillna(x, df))
            
            # 滚动计算95% VaR
            var_95 = returns.rolling(20, min_periods=10).quantile(0.05)
            
            # 返回VaR的绝对值作为风险指标
            return abs(var_95).apply(lambda x: self.safe_fillna(x, df))
        except:
            return pd.Series(0, index=df.index)
