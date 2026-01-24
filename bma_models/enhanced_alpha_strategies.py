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
from bma_models.unified_purged_cv_factory import create_unified_cv
from bma_models.enhanced_alpha_quality_monitor import EnhancedAlphaQualityMonitor, AlphaFactorQualityReport

# Configure logging first
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 核心依赖 - 必需
try:
    from bma_models.unified_nan_handler import unified_nan_handler, clean_nan_predictive_safe
except ImportError:
    from unified_nan_handler import unified_nan_handler, clean_nan_predictive_safe

try:
    from cross_sectional_standardizer import CrossSectionalStandardizer, standardize_factors_cross_sectionally
    # Create alias for compatibility
    standardize_cross_sectional_predictive_safe = standardize_factors_cross_sectionally
except ImportError:
    CrossSectionalStandardizer = None
    standardize_cross_sectional_predictive_safe = None

# 可选依赖
try:
    from bma_models.factor_orthogonalization import orthogonalize_factors_predictive_safe, FactorOrthogonalizer
except ImportError:
    try:
        from factor_orthogonalization import orthogonalize_factors_predictive_safe, FactorOrthogonalizer
    except ImportError:
        logger.warning("FactorOrthogonalizer not available, using simplified version")
        orthogonalize_factors_predictive_safe = None
        FactorOrthogonalizer = None

# Parameter optimization module removed - functionality integrated inline
TechnicalIndicatorOptimizer = None
ParameterConfig = None

# Dynamic factor weighting removed - using pure PCA approach

# 为了兼容性，创建别名
cross_sectional_standardize = standardize_cross_sectional_predictive_safe

# Removed external advanced factor dependencies, all factors integrated into this module

class AlphaStrategiesEngine:
    """Alpha Strategy Engine: Unified computation, neutralization, ranking, gating"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Alpha Strategy Engine
        
        Args:
            config_path: Configuration file path (auto-detect if None)
        """
        # Auto-detect data availability and choose appropriate config
        if config_path is None:
            self.data_availability = self._detect_data_availability()
            if self.data_availability['has_fundamental_data']:
                config_path = "alphas_config.yaml"
            else:
                config_path = "alphas_config_delayed_data.yaml"
                logger.info("🟡 检测到无基本面数据访问权限，自动切换到延迟数据配置")
        
        self.config = self._load_config(config_path)
        self.alpha_functions = self._register_alpha_functions()
            
        self.alpha_cache = {}  # Cache computation results
        
        # All factors integrated into this module, no external dependencies needed
        logger.info("All Alpha factors integrated into this module")
        
        # [OK] NEW: 导入因子滞后配置
        try:
            from factor_lag_config import factor_lag_manager
            self.lag_manager = factor_lag_manager
            logger.info(f"因子滞后配置加载成功，最大滞后: T-{self.lag_manager.get_max_lag()}")
        except ImportError:
            logger.warning("因子滞后配置未找到，使用默认全局滞后")
            self.lag_manager = None
        
        # [OK] PERFORMANCE FIX: Initialize parameter optimizer
        if TechnicalIndicatorOptimizer is not None:
            self.parameter_optimizer = TechnicalIndicatorOptimizer()
        else:
            self.parameter_optimizer = None
        self.optimized_parameters = {}
        
        # Dynamic factor weighting removed
        
        # Statistics
        self.stats = {
            'computation_times': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'neutralization_stats': {}
        }
        
        # 初始化数据质量监控器
        self.quality_monitor = EnhancedAlphaQualityMonitor(
            strict_mode=False,  # 默认非严格模式，记录但不中断
            log_dir="logs/alpha_quality"
        )
        self.quality_reports = {}  # 存储每个因子的质量报告
        
        # Initialize data providers for fundamental data
        self._init_data_providers()
        
        logger.info(f"Alpha Strategy Engine initialized, loaded {len(self.config['alphas'])} factors")
    
    def _safe_groupby_apply(self, df: pd.DataFrame, groupby_col: str, apply_func, *args, **kwargs) -> pd.Series:
        """
        🔧 CRITICAL FIX: MultiIndex安全的groupby操作
        统一处理groupby.apply，避免reset_index破坏MultiIndex结构
        """
        if isinstance(df.index, pd.MultiIndex) and groupby_col in df.index.names:
            # MultiIndex情况：按指定level进行groupby
            result = df.groupby(level=groupby_col).apply(apply_func, *args, **kwargs)
            # 清理多余的索引层级
            if hasattr(result, 'index') and result.index.nlevels > df.index.nlevels:
                result = result.droplevel(0)
            return result
        elif groupby_col in df.columns:
            # 普通DataFrame情况：按列进行groupby
            if isinstance(df.index, pd.MultiIndex):
                # 如果原来是MultiIndex，尽量保持结构
                result = df.groupby(groupby_col).apply(apply_func, *args, **kwargs)
                return result  # 保持结果的索引结构
            else:
                # 完全普通的情况
                result = df.groupby(groupby_col).apply(apply_func, *args, **kwargs)
                if hasattr(result, 'reset_index'):
                    return result.reset_index(level=0, drop=True)
                return result
        else:
            # 兼容原有逻辑
            logger.warning(f"⚠️ groupby列 '{groupby_col}' 不在索引或列中，使用原始数据")
            return apply_func(df, *args, **kwargs)
    
    def _detect_data_availability(self) -> Dict:
        """检测可用的数据类型和API访问权限"""
        availability = {
            'has_fundamental_data': False,
            'has_options_data': False,
            'has_news_data': False,
            'has_realtime_data': False
        }
        
        try:
            # 尝试获取一个测试股票的基本面数据
            from bma_models.polygon_client import polygon_client
            test_data = polygon_client.get_financials('AAPL', limit=1)
            
            # 如果没有错误且有数据，则有基本面数据访问权限
            if test_data and 'results' in test_data and test_data['results']:
                availability['has_fundamental_data'] = True
                logger.info("[OK] 检测到基本面数据访问权限")
            else:
                logger.info("[ERROR] 无基本面数据访问权限 - 使用技术因子模式")
            
            # 可以添加其他数据类型的检测
            # TODO: 检测期权数据、新闻数据等
            
        except Exception as e:
            logger.warning(f"数据可用性检测失败: {e}")
        
        return availability
    
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
            
            return self.safe_fillna(result, df)
            
        except Exception as e:
            logger.warning(f"线性衰减计算失败: {e}")
            return self.safe_fillna(series, df)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            
            # Merge user config with defaults
            default_config = self._get_default_config()
            default_config.update(user_config)
            return default_config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration with all 25 required factors enabled"""
        # Define the required factors (removed ivol_60d due to multicollinearity with stability_score)
        required_17_factors = [
            'momentum_10d_ex1',
            'rsi', 'bollinger_squeeze',
            'obv_momentum', 'atr_ratio', 'blowoff_ratio', 'stability_score',
            'liquidity_factor',
            'near_52w_high', 'reversal_1d', 'mom_accel_5_2'
        ]
        
        # Create alpha config for each required factor
        alphas_config = []
        for factor_name in required_17_factors:
            alpha_config = {
                'name': factor_name,
                'kind': factor_name,  # Use factor name as kind
                'enabled': True,
                'windows': [20],  # Default window
                'decay': 6,       # Default decay
                'delay': 1        # Default delay
            }
            alphas_config.append(alpha_config)
        
        return {
            'universe': 'TOPDIV3000',
            'region': 'GLB',
            'neutralization': ['COUNTRY'],
            'rebalance': 'WEEKLY',
            'hump_levels': [0.003, 0.008],
            'winsorize_std': 2.5,
            'truncation': 0.10,
            'temperature': 1.2,
            'alphas': alphas_config  # Now includes all 25 factors
        }
    
    def _register_alpha_functions(self) -> Dict[str, Callable]:
        """Register FOCUSED 25 Alpha computation functions - Only selected high-value factors"""
        return {
            # FOCUSED 25 FACTORS - All others commented out
            
            # Momentum factors (1/23) - REMOVED: momentum_20d, momentum_reversal_short
            'momentum_10d_ex1': self._compute_momentum_10d_ex1,

            # Mean reversion factors (2/17) - REMOVED: price_to_ma20
            'rsi': self._compute_rsi,
            'bollinger_squeeze': self._compute_bollinger_squeeze,

            # Volume factors (1/17)
            'obv_momentum': self._compute_obv_momentum,

            # Volatility factors (3/17)
            'atr_ratio': self._compute_atr_ratio,
            'blowoff_ratio': self._compute_blowoff_ratio,
            'stability_score': self._compute_stability_score,

            # REMOVED: ivol_60d (multicollinearity with stability_score, r=-0.95)

            # Fundamental factors (2/17) - REMOVED: growth_proxy, profitability_momentum, growth_acceleration, value_proxy, profitability_proxy, quality_proxy, mfi
            'liquidity_factor': self._compute_liquidity_factor,

            # High-alpha factors (4/17)
            'near_52w_high': self._compute_52w_new_high_proximity,
            'reversal_1d': self._compute_reversal_1d,
            'mom_accel_5_2': self._compute_mom_accel_5_2,
            
            # ===== ALL OTHER FACTORS COMMENTED OUT =====
            
            # OLD Technical factors - COMMENTED OUT
            # 'momentum': self._compute_momentum,
            # 'momentum_6_1': self._compute_momentum_6_1,
            # 'reversal': self._compute_reversal,
            # 'reversal_10': self._compute_reversal_10,
            # 'mean_reversion': self._compute_mean_reversion,
            # 'volume_ratio': self._compute_volume_ratio,
            # 'price_position': self._compute_price_position,
            # 'volatility': self._compute_volatility,
            # 'residual_momentum': self._compute_residual_momentum,
            # 'pead': self._compute_pead,
            
            # OLD Extended momentum factors - COMMENTED OUT
            # 'new_high_proximity': self._compute_52w_new_high_proximity,
            # 'low_beta': self._compute_low_beta_anomaly,
            # 'idiosyncratic_vol': self._compute_idiosyncratic_volatility,
            
            # OLD Fundamental factors - COMMENTED OUT
            # 'earnings_surprise': self._compute_earnings_surprise,
            # 'analyst_revision': self._compute_analyst_revision,
            # 'ebit_ev': self._compute_ebit_ev,
            # 'fcf_ev': self._compute_fcf_ev,
            # 'earnings_yield': self._compute_earnings_yield,
            # 'sales_yield': self._compute_sales_yield,
            # 'pb_ratio': self._compute_pb_ratio,
            
            # OLD Profitability factors - COMMENTED OUT
            # 'gross_margin': self._compute_gross_margin,
            # 'operating_profitability': self._compute_operating_profitability,
            # 'roe_neutralized': self._compute_roe_neutralized,
            # 'roic_neutralized': self._compute_roic_neutralized,
            # 'net_margin': self._compute_net_margin,
            # 'cash_yield': self._compute_cash_yield,
            # 'shareholder_yield': self._compute_shareholder_yield,
            
            # OLD Accrual factors - COMMENTED OUT
            # 'total_accruals': self._compute_total_accruals,
            # 'working_capital_accruals': self._compute_working_capital_accruals,
            # 'net_operating_assets': self._compute_net_operating_assets,
            # 'asset_growth': self._compute_asset_growth,
            # 'net_equity_issuance': self._compute_net_equity_issuance,
            # 'investment_factor': self._compute_investment_factor,
            
            # OLD Quality score factors - COMMENTED OUT
            # 'piotroski_score': self._compute_piotroski_score,
            # 'ohlson_score': self._compute_ohlson_score,
            # 'altman_score': self._compute_altman_score,
            # 'qmj_score': self._compute_qmj_score,
            # 'earnings_stability': self._compute_earnings_stability,
            
            # OLD Sentiment factors - COMMENTED OUT
            # 'news_sentiment': self._compute_news_sentiment,
            # 'market_sentiment_10d': self._compute_market_sentiment_10d,
            # 'fear_greed_sentiment': self._compute_fear_greed_sentiment,
            # 'sentiment_momentum_10d': self._compute_sentiment_momentum_10d,
            
            # OLD Technical indicators - COMMENTED OUT
            # 'technical_sma_10': self._compute_sma_10,
            # 'technical_sma_20': self._compute_sma_20,
            # 'technical_sma_50': self._compute_sma_50,
            # 'technical_rsi': self._compute_rsi,
            # 'technical_bb_position_10d': self._compute_bb_position_10d,
            # 'technical_macd_10d': self._compute_macd_10d,
            # 'technical_price_momentum_5d': self._compute_price_momentum_5d,
            # 'technical_volume_ratio': self._compute_volume_ratio,
            
            # OLD Missing alpha types - COMMENTED OUT
            # 'volume_trend': self._compute_volume_trend,
            # 'gap_momentum': self._compute_gap_momentum,
            # 'intraday_momentum': self._compute_intraday_momentum,
            
            # OLD Risk indicators - COMMENTED OUT
            # 'risk_max_drawdown': self._compute_max_drawdown,
            # 'risk_sharpe_ratio': self._compute_sharpe_ratio,
            # 'risk_var_95': self._compute_var_95,
            
            'hump': None,  # Special handling
        }
    
    # ========== Fundamental Data Provider ==========
    
    def _init_data_providers(self):
        """Initialize data providers for fundamental data"""
        try:
            # Try to import Polygon client
            from bma_models.polygon_client import polygon_client as pc
            self.polygon_client = pc
            logger.info("[OK] Polygon客户端初始化成功")
        except ImportError:
            logger.warning("[WARN] Polygon客户端不可用，基本面因子将使用模拟数据")
            self.polygon_client = None
            
        # Initialize data provider
        self.fundamental_cache = {}
    
    def get_fundamental_data(self, ticker: str, as_of_date: str = None) -> Dict:
        """
        获取基本面数据 - 统一数据源
        
        Args:
            ticker: 股票代码
            as_of_date: 数据截止日期
            
        Returns:
            Dict: 包含基本面数据的字典
        """
        cache_key = f"{ticker}_{as_of_date}"
        if cache_key in self.fundamental_cache:
            return self.fundamental_cache[cache_key]
            
        fundamental_data = {}
        
        try:
            if self.polygon_client:
                # 使用真实的Polygon API获取数据
                try:
                    # 获取财务数据
                    financials = self.polygon_client.get_financials(ticker)
                    if financials and 'results' in financials:
                        latest_financial = financials['results'][0]
                        
                        # 提取关键财务指标
                        fundamental_data.update({
                            'market_cap': latest_financial.get('market_capitalization'),
                            'enterprise_value': latest_financial.get('enterprise_value'),
                            'pe_ratio': latest_financial.get('price_earnings_ratio'),
                            'pb_ratio': latest_financial.get('price_book_ratio'),
                            'debt_to_equity': latest_financial.get('debt_to_equity_ratio'),
                            'roe': latest_financial.get('return_on_equity'),
                            'roa': latest_financial.get('return_on_assets'),
                            'revenue': latest_financial.get('revenues'),
                            'net_income': latest_financial.get('net_income_loss'),
                            'total_assets': latest_financial.get('assets'),
                            'total_debt': latest_financial.get('liabilities'),
                            'book_value': latest_financial.get('equity'),
                            'free_cash_flow': latest_financial.get('net_cash_flow_operating_activities'),
                            'dividend_yield': latest_financial.get('dividend_yield')
                        })
                        
                except Exception as api_error:
                    logger.warning(f"Polygon API获取{ticker}数据失败: {api_error}")
            
            # 如果没有获取到数据，使用模拟数据
            if not fundamental_data:
                fundamental_data = self._get_simulated_fundamental_data(ticker)
                
        except Exception as e:
            logger.error(f"获取{ticker}基本面数据失败: {e}")
            fundamental_data = self._get_simulated_fundamental_data(ticker)
        
        # 缓存结果
        self.fundamental_cache[cache_key] = fundamental_data
        return fundamental_data

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
            
            return self.safe_fillna(result, df)
        
        # 🔧 CRITICAL FIX: 保持MultiIndex结构，避免索引错位
        if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
            # MultiIndex情况：按date level进行groupby，保持索引结构
            result = df.groupby(level='date').apply(_neutralize_cross_section_safe)
            # 移除groupby产生的额外层级，但保持原有MultiIndex结构
            if result.index.nlevels > df.index.nlevels:
                result = result.droplevel(0)
            return result
        else:
            # 非MultiIndex情况：保持原有逻辑
            return df.groupby('date').apply(_neutralize_cross_section_safe).reset_index(level=0, drop=True)
    
    def hump_transform(self, z: pd.Series, hump: float = 0.003) -> pd.Series:
        """Gating transformation: Set small signals to zero"""
        return z.where(z.abs() >= hump, 0.0)
    
    def rank_transform(self, z: pd.Series) -> pd.Series:
        """Ranking transformation"""
        return z.rank(pct=True) - 0.5
    
    def ema_decay(self, s: pd.Series, span: int) -> pd.Series:
        """Time-safe exponential moving average decay - Only use historical data"""
        # [OK] PERFORMANCE FIX: 移除过度保守的shift(1)
        # 差异化滞后已在因子级别应用，此处不需要额外滞后
        # Use expanding window to ensure each time point only uses historical data
        result = s.ewm(span=span, adjust=False).mean()
        # [ERROR] REMOVED: 移除额外shift(1)以保持信号及时性和强度
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
            # 如果global_nan_config不可用，使用本地逻辑
            logger.warning("global_nan_config不可用，使用本地NaN处理")
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
        [OK] PERFORMANCE FIX: 优化技术指标参数
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
                    logger.info(f"[OK] {target['name']}最优参数: {result['best_parameter']} "
                              f"(IC均值: {result['optimization_summary'].get('best_ic_mean', 0):.4f})")
                
            except Exception as e:
                logger.warning(f"优化{target['name']}失败: {e}")
                continue
        
        # 缓存结果
        self.optimized_parameters = optimized_params
        logger.info(f"[OK] 技术指标参数优化完成，优化了{len(optimized_params)}个指标")
        
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
        # Method removed - using equal weights for PCA preprocessing
        return {col: 1.0/len(alpha_cols) for col in alpha_cols}
    
    def apply_dynamic_weights(self, df: pd.DataFrame, 
                            alpha_cols: List[str],
                            weights: Dict[str, float]) -> pd.Series:
        # Method removed - return simple average
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
        """Lightweight momentum factor - optimized version avoiding duplication with Polygon"""
        # Keep this for Alpha engine, but use simpler calculation to avoid exact duplication
        try:
            results = []
            for window in windows:
                g = df.groupby('ticker')['Close']
                # Simple momentum: current/past - 1, shifted by 1 day to align with unified T-1 lag
                momentum = (g.shift(1) / g.shift(window + 1) - 1.0).fillna(0.0)
                
                # Apply decay
                result = momentum.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
                results.append(result)
            
            if results:
                combined = pd.concat(results, axis=1).mean(axis=1)
                return combined.fillna(0.0)
            else:
                return pd.Series(0.0, index=df.index)
                
        except Exception as e:
            logger.warning(f"Lightweight momentum computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_reversal(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """Reversal factor: Short-term price reversal - 数值稳定性增强"""
        # [HOT] CRITICAL FIX: 导入数值稳定性保护
        try:
            from numerical_stability import safe_log, safe_divide
        except ImportError:
            # 简化实现
            def safe_log(x, epsilon=1e-10):
                return np.log(np.maximum(x, epsilon))
            def safe_divide(a, b, epsilon=1e-10):
                return a / np.maximum(np.abs(b), epsilon)
        
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
        """Lightweight volatility factor - optimized version avoiding duplication with Polygon"""
        try:
            window = windows[0] if windows else 20
            g = df.groupby('ticker')['Close']
            
            # Simple volatility: rolling std of returns, shifted to avoid lookahead
            returns = g.pct_change()  # T-1滞后由统一配置控制
            volatility = returns.rolling(window).std().fillna(0.0)
            
            # Invert volatility (low vol = high score)
            vol_factor = -volatility  
            
            # Apply decay
            result = vol_factor.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return result.fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Lightweight volatility computation failed: {e}")
            return pd.Series(0.0, index=df.index)
        results = []
        
        for window in windows:
            # 🛡️ SAFETY FIX: Calculate log returns with numerical stability
            try:
                from numerical_stability import safe_log, safe_divide
            except ImportError:
                # 简化实现
                def safe_log(x, epsilon=1e-10):
                    return np.log(np.maximum(x, epsilon))
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
            
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
            try:
                from numerical_stability import safe_divide
            except ImportError:
                # 简化实现
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
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
                    # Create synthetic volume using price * dynamic multiplier
                    # Use median volume when available, otherwise use conservative estimate
                    if 'Volume' in df.columns and not df['Volume'].isna().all():
                        median_vol = df['Volume'].median()
                        synthetic_volume = df['Close'] * (median_vol / df['Close'].median())
                    else:
                        synthetic_volume = df['Close'] * 100000  # Conservative estimate
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
            try:
                from numerical_stability import safe_log, safe_divide
            except ImportError:
                # 简化实现
                def safe_log(x, epsilon=1e-10):
                    return np.log(np.maximum(x, epsilon))
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
            
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
                # Use synthetic volume with dynamic calculation
                if 'Volume' in df.columns and not df['Volume'].isna().all():
                    median_vol = df['Volume'].median()
                    synthetic_volume = df['Close'] * (median_vol / df['Close'].median())
                else:
                    synthetic_volume = df['Close'] * 100000  # Conservative estimate
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
            # Using T-1 to T-6 data, aligned with unified lag
            rev = -(g.shift(1) / g.shift(6) - 1.0)
            rev_ema = rev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return rev_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Short-term reversal computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_amihud_illiquidity_new(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Improved Amihud illiquidity: More robust rolling median with EMA decay"""
        try:
            window = windows[0] if windows else 22
            # 🔧 CRITICAL FIX: 使用安全的groupby方法，保持MultiIndex结构
            returns_abs = self._safe_groupby_apply(df, 'ticker', lambda s: s['Close'].pct_change().abs())  # T-1滞后由统一配置控制
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
    
    def _compute_mean_reversion(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Lightweight mean reversion factor - optimized version avoiding duplication with Polygon"""
        try:
            window = windows[0] if windows else 20
            g = df.groupby('ticker')['Close']
            
            # Simple mean reversion: (mean - current) / mean, shifted to avoid lookahead
            close_prices = g.shift(1)  # T-1 to align with unified lag
            rolling_mean = close_prices.rolling(window).mean()
            
            # Mean reversion signal: (mean - current) / mean
            mean_reversion = ((rolling_mean - close_prices) / rolling_mean).fillna(0.0)
            
            # Apply decay
            result = mean_reversion.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return result.fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Lightweight mean reversion computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_volume_ratio(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Volume ratio factor - current volume vs average"""
        try:
            try:
                from numerical_stability import safe_divide
            except ImportError:
                # 简化实现
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
            
            window = windows[0] if windows else 20
            volume = df['Volume']
            
            # Calculate rolling average volume
            avg_volume = volume.rolling(window=window, min_periods=1).mean()
            
            # Volume ratio signal
            volume_ratio = safe_divide(volume, avg_volume, fill_value=1.0)
            
            # Apply log transformation and decay
            log_ratio = np.log(volume_ratio)
            result = log_ratio.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return pd.Series(result.values, index=df.index, name='volume_ratio').fillna(0.0)
        except Exception as e:
            logger.warning(f"Volume ratio computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_rsi(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """RSI (Relative Strength Index) factor"""
        try:
            try:
                from numerical_stability import safe_divide
            except ImportError:
                # 简化实现
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
            
            window = windows[0] if windows else 14
            close_prices = df['Close']
            
            # Calculate price changes
            delta = close_prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = (-delta).where(delta < 0, 0)
            
            # Calculate rolling averages
            avg_gains = gains.rolling(window=window, min_periods=1).mean()
            avg_losses = losses.rolling(window=window, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = safe_divide(avg_gains, avg_losses, fill_value=1.0)
            rsi = 100 - safe_divide(100, 1 + rs, fill_value=50.0)
            
            # Normalize RSI to [-1, 1] range
            normalized_rsi = (rsi - 50) / 50
            
            # Apply decay
            result = normalized_rsi.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return pd.Series(result.values, index=df.index, name='rsi').fillna(0.0)
        except Exception as e:
            logger.warning(f"RSI computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_price_position(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Price position within recent range"""
        try:
            try:
                from numerical_stability import safe_divide
            except ImportError:
                # 简化实现
                def safe_divide(a, b, epsilon=1e-10):
                    return a / np.maximum(np.abs(b), epsilon)
            
            window = windows[0] if windows else 20
            close_prices = df['Close']
            
            # Calculate rolling min/max
            rolling_min = close_prices.rolling(window=window, min_periods=1).min()
            rolling_max = close_prices.rolling(window=window, min_periods=1).max()
            
            # Price position: (current - min) / (max - min)
            price_range = rolling_max - rolling_min
            position = safe_divide(close_prices - rolling_min, price_range, fill_value=0.5)
            
            # Normalize to [-1, 1] range
            normalized_position = (position - 0.5) * 2
            
            # Apply decay
            result = normalized_position.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return pd.Series(result.values, index=df.index, name='price_position').fillna(0.0)
        except Exception as e:
            logger.warning(f"Price position computation failed: {e}")
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
        """
        IVOL 特质波动率 (T+10 适用):
        使用60日滚动回归计算相对于SPY的特异性波动率

        步骤:
        1. 计算对数收益: r_i,t = ln(c_t/c_t-1), r_m,t 使用SPY
        2. 60日滚动回归: r_i,t = α + β*r_m,t + ε_i,t
        3. IVOL_60d = sqrt(1/(N-1) * Σ(ε_i,t-k)^2) for k=1 to 60
        4. 每日横截面 winsorize → z-score
        """
        try:
            window = windows[0] if windows else 60
            min_periods = max(30, window // 2)

            # 计算对数收益
            close = df['Close']
            log_returns = close.groupby(df['ticker']).pct_change().reset_index(level=0, drop=True)  # T-1滞后由统一配置控制

            # 获取市场基准收益 (SPY proxy: 使用市场平均作为基准)
            # 注意: 如果有SPY数据，可以直接使用；这里使用市场平均作为proxy
            market_close = df.groupby('date')['Close'].mean()
            market_returns = pd.Series(index=df.index, dtype=float)

            # 为每个日期分配市场收益
            for date in df['date'].unique():
                mask = df['date'] == date
                if date in market_close.index:
                    market_returns.loc[mask] = market_close[date]

            # 计算市场对数收益
            market_log_returns = pd.Series(index=df.index, dtype=float)
            for ticker in df['ticker'].unique():
                ticker_mask = df['ticker'] == ticker
                ticker_dates = df[ticker_mask]['date'].values
                for i, date in enumerate(ticker_dates[1:], 1):
                    prev_date = ticker_dates[i-1]
                    if prev_date in market_close.index and date in market_close.index:
                        market_log_returns.loc[(df['ticker'] == ticker) & (df['date'] == date)] = \
                            np.log(market_close[date] / market_close[prev_date])

            # 为每个股票计算60日滚动回归残差
            ivol_results = pd.Series(index=df.index, dtype=float)

            for ticker in df['ticker'].unique():
                ticker_mask = df['ticker'] == ticker
                ticker_data = df[ticker_mask].copy()
                ticker_returns = log_returns[ticker_mask]
                ticker_market_returns = market_log_returns[ticker_mask]

                # 去除 NaN 值
                valid_mask = ~(ticker_returns.isna() | ticker_market_returns.isna())
                if valid_mask.sum() < min_periods:
                    continue

                ticker_returns_clean = ticker_returns[valid_mask]
                ticker_market_returns_clean = ticker_market_returns[valid_mask]

                # 滚动回归计算残差
                residuals = pd.Series(index=ticker_returns_clean.index, dtype=float)

                for i in range(window - 1, len(ticker_returns_clean)):
                    start_idx = max(0, i - window + 1)

                    y = ticker_returns_clean.iloc[start_idx:i+1].values
                    x = ticker_market_returns_clean.iloc[start_idx:i+1].values

                    if len(y) >= min_periods and len(x) >= min_periods:
                        # 简化CAPM回归: r_i = α + β*r_m + ε
                        x_with_intercept = np.column_stack([np.ones(len(x)), x])
                        try:
                            # 使用最小二乘法
                            beta_coef = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
                            predicted = x_with_intercept @ beta_coef
                            residual = y[-1] - predicted[-1]  # 当前期残差
                            residuals.iloc[i] = residual
                        except:
                            residuals.iloc[i] = 0.0

                # 计算60日残差标准差作为IVOL
                ivol_values = pd.Series(index=residuals.index, dtype=float)
                for i in range(window - 1, len(residuals)):
                    start_idx = max(0, i - window + 1)
                    window_residuals = residuals.iloc[start_idx:i+1]
                    valid_residuals = window_residuals.dropna()
                    if len(valid_residuals) >= min_periods:
                        ivol_values.iloc[i] = valid_residuals.std()

                # 映射回原始索引
                for idx, value in ivol_values.items():
                    if not pd.isna(value):
                        ivol_results.loc[idx] = value

            # 应用EMA衰减
            ivol_ema = ivol_results.groupby(df['ticker']).transform(
                lambda x: self.ema_decay(x, span=decay) if hasattr(self, 'ema_decay') else x
            )

            # 负号处理：低波动率更好
            result = -ivol_ema.fillna(0.0)

            return result

        except Exception as e:
            logger.warning(f"IVOL特质波动率 computation failed: {e}")
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
                return self.safe_fillna(ebit_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)
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
                    # Create enterprise value proxy with dynamic calculation
                    if 'Volume' in df.columns and not df['Volume'].isna().all():
                        median_vol = df['Volume'].median()
                        ev_proxy = df['Close'] * (median_vol / df['Close'].median())
                    else:
                        ev_proxy = df['Close'] * 100000  # Conservative estimate
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
            elif 'volume' in df.columns:
                fcf_proxy = df['volume'] * df['Close'] / df['Close']  # volume as proxy
                ev_proxy = df['Close'] * df['volume']
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
                fcf_ev = fcf_ev.groupby(df['ticker']).transform(
                    lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
                )
                return self.safe_fillna(fcf_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"FCF/EV computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """真实的Earnings Yield (E/P) - 使用基本面数据"""
        try:
            results = []
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].copy()
                
                # 获取基本面数据
                fundamental_data = self.get_fundamental_data(ticker)
                
                if fundamental_data.get('pe_ratio') and fundamental_data['pe_ratio'] > 0:
                    # E/P = 1/PE
                    earnings_yield = 1.0 / fundamental_data['pe_ratio']
                else:
                    # 回退到价格代理方法
                    close_col = 'Close' if 'Close' in ticker_data.columns else 'close'
                    annual_return = ticker_data[close_col].pct_change(252).iloc[-1]
                    earnings_yield = annual_return / ticker_data[close_col].iloc[-1] * 100 if not pd.isna(annual_return) else 0
                
                # 为该ticker的所有行设置相同的值
                ticker_results = pd.Series(earnings_yield, index=ticker_data.index)
                results.append(ticker_results)
            
            combined_results = pd.concat(results) if results else pd.Series(0, index=df.index)
            return self.safe_fillna(combined_results.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)            
        except Exception as e:
            logger.warning(f"真实Earnings yield计算失败，使用回退方法: {e}")
            try:
                # 回退到原始方法 - 修复类型错误
                close_col = 'Close' if 'Close' in df.columns else 'close'
                annual_return = df.groupby('ticker')[close_col].pct_change(252)
                earnings_yield = (annual_return / df[close_col] * 100).fillna(0)
                return self.safe_fillna(earnings_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)
            except Exception as backup_e:
                logger.warning(f"备用方法也失败: {backup_e}")
                return pd.Series(0.0, index=df.index)
    
    def _compute_pb_ratio(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """真实的Price-to-Book比率 - 使用基本面数据"""
        try:
            results = []
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].copy()
                
                # 获取基本面数据
                fundamental_data = self.get_fundamental_data(ticker)
                
                if fundamental_data.get('pb_ratio') and fundamental_data['pb_ratio'] > 0:
                    pb_ratio = fundamental_data['pb_ratio']
                else:
                    # 回退：使用市值/账面价值估算
                    close_col = 'Close' if 'Close' in ticker_data.columns else 'close'
                    volume_col = 'volume' if 'volume' in ticker_data.columns else 'Volume'
                    
                    if volume_col in ticker_data.columns:
                        market_cap = ticker_data[close_col].iloc[-1] * ticker_data[volume_col].iloc[-1]
                        book_value = fundamental_data.get('book_value', market_cap * 0.5)  # 估算账面价值
                        pb_ratio = market_cap / book_value if book_value > 0 else 1.0
                    else:
                        pb_ratio = 1.0
                
                # 为该ticker的所有行设置相同的值
                ticker_results = pd.Series(pb_ratio, index=ticker_data.index)
                results.append(ticker_results)
            
            combined_results = pd.concat(results) if results else pd.Series(1.0, index=df.index)
            return self.safe_fillna(combined_results.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)            
        except Exception as e:
            logger.warning(f"PB比率计算失败: {e}")
            try:
                # 回退方法 - 使用简单估算
                close_col = 'Close' if 'Close' in df.columns else 'close'  
                pb_estimate = (df[close_col] / df[close_col].rolling(252).mean()).fillna(1.0)
                return self.safe_fillna(pb_estimate.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)
            except Exception as backup_e:
                logger.warning(f"PB比率备用方法失败: {backup_e}")
                return pd.Series(1.0, index=df.index)

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
                return self.safe_fillna(sales_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)), df)
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
            return self.safe_fillna(annual_return, df)
        except Exception as e:
            logger.warning(f"Gross margin computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_operating_profitability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Operating profitability（简化implementation）"""
        try:
            if 'volume' in df.columns:
                efficiency = df['volume'] / (df['Close'] + 1e-9)
                return self.safe_fillna(efficiency, df)
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
            return self.safe_fillna(returns, df)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_roic_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROICneutralize（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(126)
            return self.safe_fillna(returns, df)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net margin（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(63)
            return self.safe_fillna(returns, df)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_cash_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Cash yield（简化implementation）"""
        try:
            if 'amount' in df.columns:
                cash_yield = df['amount'] / (df['Close'] + 1e-9)
                return self.safe_fillna(cash_yield, df)
            elif 'volume' in df.columns:
                cash_yield = (df['volume'] * df['Close']) / (df['Close'] + 1e-9)
                return self.safe_fillna(cash_yield, df)
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
                return self.safe_fillna(ratio, df)
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
            # [OK] FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            price_vol = df.groupby('ticker')[close_col].rolling(126).std() / df[close_col]
            return -price_vol.apply(lambda x: self.safe_fillna(x, df))  # Take negative，lower risk is better
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_altman_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """AltmanScore（简化implementation）"""
        try:
            # [OK] FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            stability = -returns.rolling(126).std()  # Stability
            return self.safe_fillna(stability, df)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_qmj_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """QMJ质量Score（简化implementation）"""
        try:
            # [OK] FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            quality = returns.rolling(252).mean() / (returns.rolling(252).std() + 1e-8)
            return self.safe_fillna(quality, df)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_stability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """盈利Stability（简化implementation）"""
        try:
            # [OK] FIX: 兼容'Close'和'close'列名
            close_col = 'Close' if 'Close' in df.columns else 'close'
            returns = df.groupby('ticker')[close_col].pct_change()
            stability = -returns.rolling(252).std()  # lower volatility is better
            return self.safe_fillna(stability, df)
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
        
        # [TOOL] 修复数据格式问题：确保输入是DataFrame
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
            # Check if alpha is enabled
            if not alpha_config.get('enabled', True):
                continue
                
            alpha_name = alpha_config['name']
            alpha_kind = alpha_config.get('kind', alpha_config.get('function', 'momentum'))
            
            try:
                start_time = pd.Timestamp.now()
                
                # Get parameters
                windows = alpha_config.get('windows', [22])
                decay = alpha_config.get('decay', 6)
                delay = alpha_config.get('delay', 1)  # 配置文件中的delay参数
                
                # [OK] NEW: 获取因子特定的滞后配置
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
                
                # 【新增】数据质量监控 - 对每个计算的Alpha因子进行质量检查
                try:
                    quality_report = self.quality_monitor.monitor_alpha_calculation(
                        factor_name=alpha_name,
                        input_data=df_work,
                        output_data=alpha_factor,
                        calculation_func=alpha_func if alpha_kind != 'hump' else None
                    )
                    
                    # 存储质量报告
                    self.quality_reports[alpha_name] = quality_report
                    
                    # 记录质量问题
                    if quality_report.errors:
                        logger.error(f"[{alpha_name}] 数据质量错误: {', '.join(quality_report.errors)}")
                    if quality_report.warnings:
                        logger.warning(f"[{alpha_name}] 数据质量警告: {', '.join(quality_report.warnings)}")
                    
                    # 输出关键质量指标
                    logger.info(f"[{alpha_name}] 质量指标 - 缺失率:{quality_report.output_quality.missing_ratio:.2%}, "
                               f"覆盖率:{quality_report.output_quality.coverage_ratio:.2%}, "
                               f"异常值率:{quality_report.output_quality.outlier_ratio:.2%}")
                    
                except Exception as monitor_error:
                    logger.warning(f"[{alpha_name}] 质量监控失败: {monitor_error}")
                
                # [OK] NEW: 应用差异化滞后策略
                if self.lag_manager and factor_specific_lag > 0:
                    # 使用因子特定的滞后
                    alpha_factor = alpha_factor.groupby(df_work['ticker']).shift(factor_specific_lag)
                    logger.debug(f"应用差异化滞后 T-{factor_specific_lag} 于 {alpha_name}")
                elif delay and delay > 0:
                    # 回退到配置文件中的delay
                    alpha_factor = alpha_factor.groupby(df_work['ticker']).shift(delay)
                
                # [OK] REMOVED: 不再使用全局统一的lag，改为差异化滞后
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
            
            # [OK] PERFORMANCE FIX: Apply factor orthogonalization to remove redundancy
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
                    logger.info(f"[OK] 因子正交化完成: 保留{retained_count}个, 移除{removed_count}个冗余因子")
                    
                    # Get factor importance if available
                    importance = orthogonalizer.get_factor_importance()
                    if importance:
                        logger.debug(f"因子重要性排序: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
                
            except Exception as e:
                logger.warning(f"因子正交化失败，继续使用原始因子: {e}")
            
            # [OK] PERFORMANCE FIX: Apply dynamic factor weighting
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
                    
                    # Dynamic weighting removed - all alpha factors used equally
                
            except Exception as e:
                logger.warning(f"动态权重应用失败: {e}")
                
        else:
            logger.error("所有Alphafactor computation failed")
        
        # 🔧 CRITICAL FIX: 强化MultiIndex处理逻辑，防止索引错位
        if not isinstance(result_df.index, pd.MultiIndex):
            logger.warning("⚠️ 结果不是MultiIndex格式，尝试重建...")
            
            # 方法1：如果有date和ticker列，尝试重建MultiIndex
            if 'date' in result_df.columns and 'ticker' in result_df.columns:
                try:
                    # 📊 数据验证：确保date和ticker数量匹配
                    if len(result_df) != len(result_df['date']) or len(result_df) != len(result_df['ticker']):
                        raise ValueError("日期或股票代码数据不完整")
                    
                    # 🎯 安全的MultiIndex重建
                    dates = pd.to_datetime(result_df['date'])
                    tickers = result_df['ticker'].astype(str)
                    
                    # 验证数据完整性
                    if dates.isnull().any():
                        raise ValueError(f"发现{dates.isnull().sum()}个空日期")
                    if tickers.isnull().any():
                        raise ValueError(f"发现{tickers.isnull().sum()}个空股票代码")
                    
                    # 创建MultiIndex
                    multi_idx = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
                    
                    # 重建DataFrame，保留所有alpha列
                    result_clean = result_df.copy()
                    result_clean.index = multi_idx
                    
                    # 只保留Alpha因子（移除了ivol_60d），移除原始市场数据和元数据列
                    required_17_factors = [
                        'momentum_10d_ex1',
                        'rsi', 'bollinger_squeeze',
                        'obv_momentum', 'atr_ratio', 'blowoff_ratio', 'stability_score',
                        'liquidity_factor',
                        'near_52w_high', 'reversal_1d', 'mom_accel_5_2'
                    ]

                    # 只保留存在的17个因子列
                    alpha_cols_available = [col for col in required_17_factors if col in result_clean.columns]
                    
                    if alpha_cols_available:
                        final_result = result_clean[alpha_cols_available]
                        logger.info(f"✅ MultiIndex重建成功: {final_result.shape} 包含17个高质量因子: {len(alpha_cols_available)}/17")
                        return final_result
                    else:
                        logger.error("❌ 重建后没有可用的特征列")
                        
                except Exception as rebuild_error:
                    logger.error(f"❌ MultiIndex重建失败: {rebuild_error}")
                    logger.info(f"原始DataFrame信息: shape={result_df.shape}, columns={list(result_df.columns)[:10]}...")
            
            # 方法2：如果原始输入是MultiIndex，尝试恢复
            logger.warning("⚠️ MultiIndex重建失败，返回原格式")
            logger.warning("⚠️ 这可能导致后续特征合并时的索引对齐问题")
        else:
            # 对于已经是MultiIndex的情况，也只返回alpha因子列
            required_alpha_factors = [
                'momentum_10d_ex1',
                'rsi', 'bollinger_position', 'price_to_ma20',
                'obv_momentum', 'ad_line', 'atr_20d', 'atr_ratio', 'blowoff_ratio', 'stability_score',
                'macd_histogram', 'stoch_k', 'cci',
                'market_cap_proxy',
                'liquidity_factor', 'growth_proxy', 'profitability_momentum',
                'growth_acceleration', 'quality_consistency', 'financial_resilience'
            ]

            alpha_cols_available = [col for col in required_alpha_factors if col in result_df.columns]

            if alpha_cols_available:
                final_result = result_df[alpha_cols_available]
                logger.info(f"✅ 结果已是MultiIndex格式: {final_result.shape} 包含Alpha因子: {len(alpha_cols_available)}个")
                return final_result
            else:
                logger.error("❌ 没有找到任何Alpha因子列")
                return pd.DataFrame()  # 返回空DataFrame
        
        # 如果以上都失败，返回空DataFrame
        logger.error("❌ 所有返回路径都失败")
        return pd.DataFrame()
    
    def _process_alpha_pipeline(self, df: pd.DataFrame, alpha_factor: pd.Series, 
                               alpha_config: Dict, alpha_name: str) -> pd.Series:
        """Alpha factor processing pipeline：winsorize -> neutralize -> zscore -> transform"""
        
        # 1. Winsorizeremove outliers
        winsorize_std = self.config.get('winsorize_std', 2.5)
        alpha_factor = self.winsorize_series(alpha_factor, k=winsorize_std)
        
        # 2. 构建临时DataFrame进行neutralize
        base_cols = ['date', 'ticker']
        neutralization_cols = []
        for col in self.config['neutralization']:
            if col in df.columns:
                neutralization_cols.append(col)
        
        temp_df = df[base_cols + neutralization_cols].copy()
        temp_df[alpha_name] = alpha_factor
        
        # 3. neutralize（default关闭，避免与全局Pipeline重复；仅研究Using时打开）
        if self.config.get('enable_alpha_level_neutralization', False):
            for neutralize_level in self.config['neutralization']:
                if neutralize_level in temp_df.columns:
                    alpha_factor = self.neutralize_factor(
                        temp_df, alpha_name, [neutralize_level]
                    )
                    temp_df[alpha_name] = alpha_factor
        
        # 4. [OK] PERFORMANCE FIX: 横截面标准化，消除市场风格偏移
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
        try:
            from ssot_violation_detector import block_internal_cv_creation
            block_internal_cv_creation("Alpha策略中的TimeSeriesSplit")
        except ImportError:
            # 备用处理 - 仅记录警告
            logger.debug("SSOT violation detector not available - skipping check")
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
        
        # IC stats removed - using direct scores
        result = pd.Series(scores, name=f'oof_{metric}')
        logger.info(f"OOFScore completed, average {metric}: {result.mean():.4f}")
        
        return result
    
    def compute_bma_weights(self, scores: pd.Series, temperature: float = None) -> pd.Series:
        """
        Pure ML-based BMA weights computation - NO hardcoded weights
        
        Args:
            scores: OOF scores from cross-validation
            temperature: Temperature coefficient, controls weight concentration
            
        Returns:
            BMA weights based purely on performance scores
        """
        if temperature is None:
            temperature = self.config.get('temperature', 1.2)
        
        # Standardize scores
        scores_std = (scores - scores.mean()) / (scores.std(ddof=0) + 1e-12)
        scores_scaled = scores_std / max(temperature, 1e-3)
        
        # Log-sum-exp softmax (numerically stable)
        max_score = scores_scaled.max()
        exp_scores = np.exp(scores_scaled - max_score)
        
        # Pure softmax - no hardcoded priors
        eps = 1e-6
        weights = (exp_scores + eps) / (exp_scores.sum() + eps * len(exp_scores))
        
        weights_series = pd.Series(weights, index=scores.index, name='bma_weights')
        
        logger.info(f"Pure ML BMA weights computed, distribution: max={weights.max():.3f}, min={weights.min():.3f}")
        logger.info(f"Top factor weights: {weights_series.nlargest(5).to_dict()}")
        
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
                    col_factor = df[col]                    # 应用时间衰减
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / len(available_cols)  # 简单平均而非硬编码权重
                
                return sentiment_factor
            else:
                # 使用第一个可用的新闻情绪列
                col = news_cols[0]
                sentiment_factor = df[col]
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
                    col_factor = df[col]
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / min(3, len(priority_cols))
                
                return sentiment_factor
            else:
                # 使用第一个可用的市场情绪列
                col = market_cols[0]
                sentiment_factor = df[col]
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
                    col_factor = df[col]
                    col_factor = self.decay_linear(col_factor, decay)
                    sentiment_factor += col_factor / len(available_cols)
                
                return sentiment_factor
            else:
                # 使用第一个可用的恐惧贪婪列
                col = fg_cols[0]
                sentiment_factor = df[col]                # 如果是原始值，进行归一化
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
                        col_data = df[col]                        # 计算短期动量（3天）
                        momentum = col_data.groupby(df['ticker']).diff(3)
                        sentiment_factor += momentum / len(sentiment_cols[:2])
                    
                    return self.decay_linear(sentiment_factor.apply(lambda x: self.safe_fillna(x, df)), decay)
                else:
                    return pd.Series(0, index=df.index)
            else:
                # 使用现成的情绪动量列
                sentiment_factor = df[momentum_cols[0]]
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
    
    # ========== [HOT] NEW: Real Polygon Training技术指标集成 ==========
    
    def _compute_sma_10(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # [OK] PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_10', 10)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            # 转换为相对强度信号：当前价格相对均线的偏离度
            return self.safe_fillna(((df['Close'] / sma) - 1), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sma_20(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # [OK] PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_20', 20)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            return self.safe_fillna(((df['Close'] / sma) - 1), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sma_50(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """简单移动平均线(可优化参数)"""
        # [OK] PERFORMANCE FIX: 使用优化后的窗口参数
        optimal_window = window or self.get_optimized_window('sma_50', 50)
        
        if 'Close' not in df.columns or len(df) < optimal_window:
            # 如果数据不足，回退到可用数据的均线
            available_days = min(20, len(df))
            if available_days >= 10:
                sma = df['Close'].rolling(available_days).mean()
                return ((df['Close'] / sma) - 1)
            else:
                return pd.Series(0, index=df.index)
        
        try:
            sma = df['Close'].rolling(optimal_window).mean()
            return self.safe_fillna(((df['Close'] / sma) - 1), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_rsi(self, df: pd.DataFrame, window: int = None, **kwargs) -> pd.Series:
        """相对强弱指数(RSI,可优化参数)"""
        # [OK] PERFORMANCE FIX: 使用优化后的窗口参数
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
            return self.safe_fillna(rsi_normalized, df)
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
            return self.safe_fillna(((bb_position - 0.5) * 2), df)
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
            return self.safe_fillna((macd / df['Close']), df)
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
            
            return self.safe_fillna((signal / df['Close']), df)
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
            
            return self.safe_fillna((histogram / df['Close']), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_price_momentum_5d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """5日价格动量"""
        if 'Close' not in df.columns or len(df) < 6:
            return pd.Series(0, index=df.index)
        
        try:
            momentum_5d = df['Close'].pct_change(5)
            return self.safe_fillna(momentum_5d, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_price_momentum_20d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """20日价格动量"""
        if 'Close' not in df.columns or len(df) < 21:
            # 如果数据不足，使用5日动量
            return self._compute_price_momentum_5d(df, **kwargs)
        
        try:
            momentum_20d = df['Close'].pct_change(20)
            return self.safe_fillna(momentum_20d, df)
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
            return self.safe_fillna(np.log1p(volume_ratio - 1), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_volume_trend(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """成交量趋势因子"""
        if 'Volume' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        
        try:
            # 计算成交量的移动平均和趋势
            period = kwargs.get('period', 10)
            volume_ma_short = df['Volume'].rolling(period//2).mean()
            volume_ma_long = df['Volume'].rolling(period).mean()
            
            # 成交量趋势 = 短期均量/长期均量 - 1
            volume_trend = (volume_ma_short / volume_ma_long.replace(0, np.nan)) - 1
            return self.safe_fillna(volume_trend, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_gap_momentum(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """跳空动量因子"""
        if 'Open' not in df.columns or 'Close' not in df.columns or len(df) < 2:
            return pd.Series(0, index=df.index)
        
        try:
            # 计算跳空：今日开盘价 vs 昨日收盘价
            prev_close = df['Close'].shift(1)
            gap = (df['Open'] - prev_close) / prev_close.replace(0, np.nan)
            
            # 累计跳空动量
            period = kwargs.get('period', 10)
            gap_momentum = gap.rolling(period).sum()
            return self.safe_fillna(gap_momentum, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_intraday_momentum(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """日内动量因子"""
        if 'Open' not in df.columns or 'Close' not in df.columns or len(df) < 1:
            return pd.Series(0, index=df.index)
        
        try:
            # 日内动量 = (收盘价 - 开盘价) / 开盘价
            intraday_return = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
            
            # 移动平均平滑
            period = kwargs.get('period', 5)
            intraday_momentum = intraday_return.rolling(period).mean()
            return self.safe_fillna(intraday_momentum, df)
        except:
            return pd.Series(0, index=df.index)
    
    # ========== [HOT] NEW: Real Polygon Training风险指标集成 ==========
    
    def _compute_max_drawdown(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """最大回撤"""
        if 'Close' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change()
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            # 使用滚动窗口计算最大回撤
            max_drawdown = drawdown.rolling(20, min_periods=5).min()
            
            # 返回回撤的绝对值作为风险信号
            return self.safe_fillna(abs(max_drawdown), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_sharpe_ratio(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """夏普比率（滚动计算）"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change()            
            # 滚动计算夏普比率 (假设无风险利率为年化2%)
            risk_free_daily = 0.02 / 252
            excess_returns = returns - risk_free_daily
            
            rolling_mean = excess_returns.rolling(20, min_periods=10).mean()
            rolling_std = returns.rolling(20, min_periods=10).std()
            
            sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
            
            # 标准化夏普比率到合理范围
            return self.safe_fillna(np.tanh(sharpe / 2), df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_var_95(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """95%置信度的风险价值(VaR)"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        
        try:
            returns = df['Close'].pct_change()            
            # 滚动计算95% VaR
            var_95 = returns.rolling(20, min_periods=10).quantile(0.05)
            
            # 返回VaR的绝对值作为风险指标
            return self.safe_fillna(abs(var_95), df)
        except:
            return pd.Series(0, index=df.index)
    
    # ========== T+10 Adapted Factors ==========
    
    def _compute_reversal_10(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """10-day reversal adapted for T+10 prediction"""
        try:
            g = df.groupby('ticker')['Close']
            # Using T-1 to T-11 data for 10-day reversal, adapted for T+10 prediction
            rev = -(g.shift(1) / g.shift(11) - 1.0)
            rev_ema = rev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay))
            return rev_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"10-day reversal computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_market_sentiment_10d(self, df: pd.DataFrame, windows: List[int] = [10, 22], 
                                     decay: int = 25) -> pd.Series:
        """Market sentiment adapted for T+10 prediction"""
        try:
            # Create synthetic sentiment based on price momentum and volatility
            # Longer window for T+10 prediction
            window = windows[0] if windows else 10
            
            g = df.groupby('ticker')['Close']
            returns = g.pct_change()  # T-1滞后由统一配置控制
            
            # Sentiment = momentum (positive) - volatility (negative)  
            momentum = returns.rolling(window).mean()
            volatility = returns.rolling(window).std()
            
            sentiment = momentum - 0.5 * volatility
            sentiment_ema = sentiment.groupby(df['ticker']).transform(
                lambda x: self.ema_decay(x, span=decay)
            )
            
            return sentiment_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Market sentiment 10d computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_sentiment_momentum_10d(self, df: pd.DataFrame, windows: List[int] = [10, 22],
                                       decay: int = 30) -> pd.Series:
        """Sentiment momentum adapted for T+10 prediction"""
        try:
            # Sentiment momentum = change in sentiment over 10-day period
            window = windows[0] if windows else 10
            
            g = df.groupby('ticker')['Close']
            returns = g.pct_change()  # T-1滞后由统一配置控制
            
            # Calculate base sentiment
            momentum = returns.rolling(window).mean()
            volatility = returns.rolling(window).std()
            sentiment = momentum - 0.5 * volatility
            
            # Sentiment momentum = current sentiment - past sentiment
            sentiment_momentum = sentiment - sentiment.shift(window)
            sentiment_momentum_ema = sentiment_momentum.groupby(df['ticker']).transform(
                lambda x: self.ema_decay(x, span=decay)
            )
            
            return sentiment_momentum_ema.fillna(0.0)
        except Exception as e:
            logger.warning(f"Sentiment momentum 10d computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_macd_10d(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """MACD adapted for T+10 prediction with longer periods"""
        try:
            g = df.groupby('ticker')['Close']
            close_prices = g.shift(1)  # T-1 prices
            
            # Longer periods for T+10 prediction: 10 and 20 days instead of 12,26
            fast_period = 10
            slow_period = 20
            signal_period = 9
            
            # Calculate EMAs
            ema_fast = close_prices.ewm(span=fast_period).mean()
            ema_slow = close_prices.ewm(span=slow_period).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # MACD histogram (final signal)
            macd_histogram = macd_line - signal_line
            
            # Apply decay
            macd_result = macd_histogram.groupby(df['ticker']).transform(
                lambda x: self.ema_decay(x, span=decay)
            )
            
            return macd_result.fillna(0.0)
        except Exception as e:
            logger.warning(f"MACD 10d computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_bb_position_10d(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Bollinger Band position adapted for T+10 prediction"""
        try:
            window = windows[0] if windows else 10
            g = df.groupby('ticker')['Close']
            close_prices = g.shift(1)  # T-1 prices
            
            # Calculate Bollinger Bands with longer period
            sma = close_prices.rolling(window).mean()
            std = close_prices.rolling(window).std()
            
            upper_band = sma + (2.0 * std)
            lower_band = sma - (2.0 * std)
            
            # Position within bands: 0.5 = middle, 1.0 = upper, 0.0 = lower
            bb_position = (close_prices - lower_band) / (upper_band - lower_band)
            bb_position = bb_position.clip(0, 1)  # Clamp to [0,1] range
            
            # Center around 0 for factor signal: -0.5 to +0.5
            bb_signal = bb_position - 0.5
            
            # Apply decay
            bb_result = bb_signal.groupby(df['ticker']).transform(
                lambda x: self.ema_decay(x, span=decay)
            )
            
            return bb_result.fillna(0.0)
        except Exception as e:
            logger.warning(f"Bollinger Band position 10d computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    # ========== FOCUSED 25 FACTORS COMPUTATION METHODS ==========
    
    # Momentum factors (3/25)
    def _compute_momentum_10d_ex1(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """10-day price momentum"""
        if 'Close' not in df.columns or len(df) < 11:
            return pd.Series(0, index=df.index)
        try:
            # 10-day momentum: (price_t - price_t-10) / price_t-10
            momentum_10d = df['Close'].pct_change(10)
            return self.safe_fillna(momentum_10d, df)
        except:
            return pd.Series(0, index=df.index)

    # Keep old function name for backward compatibility
    def _compute_momentum_10d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Deprecated: Use _compute_momentum_10d_ex1 instead"""
        return self._compute_momentum_10d_ex1(df, **kwargs)
    
    def _compute_momentum_20d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """20-day price momentum"""
        if 'Close' not in df.columns or len(df) < 21:
            return pd.Series(0, index=df.index)
        try:
            momentum_20d = df['Close'].pct_change(20)
            return self.safe_fillna(momentum_20d, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_momentum_reversal_short(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Short-term momentum reversal signal"""
        if 'Close' not in df.columns or len(df) < 6:
            return pd.Series(0, index=df.index)
        try:
            momentum_1d = df['Close'].pct_change(1)
            momentum_5d = df['Close'].pct_change(5)
            reversal_signal = -momentum_1d * momentum_5d
            return self.safe_fillna(reversal_signal, df)
        except:
            return pd.Series(0, index=df.index)

    def _compute_reversal_1d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """1-day reversal: negative of 1-day return"""
        if 'Close' not in df.columns or len(df) < 2:
            return pd.Series(0, index=df.index)
        try:
            reversal_1d = -df['Close'].pct_change(1)
            return self.safe_fillna(reversal_1d, df)
        except:
            return pd.Series(0, index=df.index)

    def _compute_mom_accel_5_2(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Momentum acceleration: 5-day momentum - 2-day momentum"""
        if 'Close' not in df.columns or len(df) < 6:
            return pd.Series(0, index=df.index)
        try:
            mom_5d = df['Close'].pct_change(5)
            mom_2d = df['Close'].pct_change(2)
            mom_accel = mom_5d - mom_2d
            return self.safe_fillna(mom_accel, df)
        except:
            return pd.Series(0, index=df.index)

    # Mean reversion factors (4/25)
    def _compute_price_to_ma20(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Price relative to 20-day moving average"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            ma20 = df['Close'].rolling(20).mean()
            price_to_ma = (df['Close'] / ma20) - 1
            return self.safe_fillna(price_to_ma, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_bollinger_squeeze(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Bollinger Band volatility squeeze
        🔥 FIX: Shift for pre-market prediction (use previous day's volatility ratio)
        """
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            # 🔥 FIX: Shift for pre-market prediction
            std_20 = df['Close'].rolling(20).std().shift(1)
            std_5 = df['Close'].rolling(5).std().shift(1)
            squeeze = std_5 / (std_20 + 1e-8)
            return self.safe_fillna(squeeze, df)
        except:
            return pd.Series(0, index=df.index)

    # Volume factors (2/25)
    def _compute_obv_momentum(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """On-Balance Volume momentum"""
        if 'Close' not in df.columns or 'Volume' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        try:
            price_change = df['Close'].diff()
            obv = (price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['Volume']).cumsum()
            obv_momentum = obv.pct_change(10)
            return self.safe_fillna(obv_momentum, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_ad_line(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Accumulation/Distribution Line"""
        if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']) or len(df) < 5:
            return pd.Series(0, index=df.index)
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
            ad_line = (clv * df['Volume']).cumsum()
            ad_momentum = ad_line.pct_change(5)
            return self.safe_fillna(ad_momentum, df)
        except:
            return pd.Series(0, index=df.index)
    
    # Volatility factors (2/25)
    def _compute_atr_20d(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """20-day Average True Range"""
        if not all(col in df.columns for col in ['High', 'Low', 'Close']) or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(20).mean()
            atr_normalized = atr / (df['Close'] + 1e-8)
            return self.safe_fillna(atr_normalized, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_atr_ratio(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """ATR ratio (5d/20d expansion/contraction)"""
        if not all(col in df.columns for col in ['High', 'Low', 'Close']) or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_5 = true_range.rolling(5).mean()
            atr_20 = true_range.rolling(20).mean()
            atr_ratio = atr_5 / (atr_20 + 1e-8)
            return self.safe_fillna(atr_ratio, df)
        except:
            return pd.Series(0, index=df.index)
    
    # Technical factors (4/25) - stoch_k, cci, mfi already exist, just need to add them
    def _compute_stoch_k(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Stochastic %K oscillator"""
        if not all(col in df.columns for col in ['High', 'Low', 'Close']) or len(df) < 14:
            return pd.Series(0, index=df.index)
        try:
            lowest_low = df['Low'].rolling(14).min()
            highest_high = df['High'].rolling(14).max()
            stoch_k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
            stoch_k_normalized = (stoch_k - 50) / 50
            return self.safe_fillna(stoch_k_normalized, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_cci(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Commodity Channel Index"""
        if not all(col in df.columns for col in ['High', 'Low', 'Close']) or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
            cci = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
            cci_normalized = cci / 100
            return self.safe_fillna(cci_normalized, df)
        except:
            return pd.Series(0, index=df.index)
    
    # Fundamental factors (10/25)
    def _compute_market_cap_proxy(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Market capitalization proxy"""
        if 'Close' not in df.columns:
            return pd.Series(0, index=df.index)
        try:
            # Use price as proxy for market cap (relative sizing)
            market_cap_proxy = np.log(df['Close'] + 1)
            return self.safe_fillna(market_cap_proxy, df)
        except:
            return pd.Series(0, index=df.index)
    
    # REMOVED: _compute_ivol_60d (multicollinearity with stability_score, r=-0.95, VIF=10.4)

    def _compute_blowoff_ratio(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Blowoff ratio: max 5D log-return / (σ14 + ε)"""
        if 'Close' not in df.columns or len(df) < 14:
            return pd.Series(0, index=df.index)
        try:
            # Calculate log returns
            log_returns = np.log(df['Close'] / df['Close'].shift(1))
            # Get maximum 5-day log return
            max_5d_log_return = log_returns.rolling(5, min_periods=1).max()
            # Calculate 14-day volatility (standard deviation of returns)
            returns = df['Close'].pct_change()
            vol_14d = returns.rolling(14, min_periods=7).std()
            # Compute blowoff ratio
            epsilon = 1e-8
            blowoff_ratio = max_5d_log_return / (vol_14d + epsilon)
            return self.safe_fillna(blowoff_ratio, df)
        except:
            return pd.Series(0, index=df.index)

    def _compute_stability_score(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Stability score: 1 / (σ20 + ε)"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            # Calculate returns
            returns = df['Close'].pct_change()
            # Calculate 20-day volatility (standard deviation)
            vol_20d = returns.rolling(20, min_periods=10).std()
            # Compute stability score
            epsilon = 1e-8
            stability_score = 1.0 / (vol_20d + epsilon)
            return self.safe_fillna(stability_score, df)
        except:
            return pd.Series(0, index=df.index)

    def _compute_liquidity_factor(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Liquidity from volume patterns"""
        if 'Volume' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        try:
            volume_ma = df['Volume'].rolling(10).mean()
            volume_std = df['Volume'].rolling(10).std()
            liquidity_factor = np.log(volume_ma + 1) / (volume_std + 1e-8)
            return self.safe_fillna(liquidity_factor, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_growth_proxy(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Growth factor from momentum"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            momentum_5d = df['Close'].pct_change(5)
            momentum_20d = df['Close'].pct_change(20)
            growth_proxy = momentum_5d + momentum_20d
            return self.safe_fillna(growth_proxy, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_profitability_momentum(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Profitability momentum"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            returns = df['Close'].pct_change()
            cumulative_returns = (1 + returns).rolling(20).apply(lambda x: x.prod()) - 1
            profitability_momentum = cumulative_returns
            return self.safe_fillna(profitability_momentum, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_growth_acceleration(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Growth acceleration"""
        if 'Close' not in df.columns or len(df) < 10:
            return pd.Series(0, index=df.index)
        try:
            momentum_5d = df['Close'].pct_change(5)
            momentum_acceleration = momentum_5d.diff(5)
            return self.safe_fillna(momentum_acceleration, df)
        except:
            return pd.Series(0, index=df.index)
    
    def _compute_quality_consistency(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Quality consistency measure"""
        if 'Close' not in df.columns or len(df) < 20:
            return pd.Series(0, index=df.index)
        try:
            returns = df['Close'].pct_change()
            rolling_sharpe = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)
            quality_consistency = rolling_sharpe
            return self.safe_fillna(quality_consistency, df)
        except:
            return pd.Series(0, index=df.index)
    
    # ========== 数据质量报告方法 ==========
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        获取所有Alpha因子的质量汇总
        
        Returns:
            质量汇总字典
        """
        if not self.quality_reports:
            return {"message": "没有可用的质量报告"}
        
        summary = {
            "total_factors": len(self.quality_reports),
            "factors_with_errors": 0,
            "factors_with_warnings": 0,
            "average_metrics": {},
            "factor_details": {}
        }
        
        # 收集所有指标
        all_missing_ratios = []
        all_coverage_ratios = []
        all_outlier_ratios = []
        all_distribution_scores = []
        
        for factor_name, report in self.quality_reports.items():
            # 统计错误和警告
            if report.errors:
                summary["factors_with_errors"] += 1
            if report.warnings:
                summary["factors_with_warnings"] += 1
            
            # 收集指标
            all_missing_ratios.append(report.output_quality.missing_ratio)
            all_coverage_ratios.append(report.output_quality.coverage_ratio)
            all_outlier_ratios.append(report.output_quality.outlier_ratio)
            all_distribution_scores.append(report.output_quality.distribution_score)
            
            # 记录每个因子的详细信息
            summary["factor_details"][factor_name] = {
                "missing_ratio": f"{report.output_quality.missing_ratio:.2%}",
                "coverage_ratio": f"{report.output_quality.coverage_ratio:.2%}",
                "outlier_ratio": f"{report.output_quality.outlier_ratio:.2%}",
                "distribution_score": f"{report.output_quality.distribution_score:.2f}",
                "errors": len(report.errors),
                "warnings": len(report.warnings)
            }
        
        # 计算平均指标
        if all_missing_ratios:
            summary["average_metrics"] = {
                "avg_missing_ratio": f"{np.mean(all_missing_ratios):.2%}",
                "avg_coverage_ratio": f"{np.mean(all_coverage_ratios):.2%}",
                "avg_outlier_ratio": f"{np.mean(all_outlier_ratios):.2%}",
                "avg_distribution_score": f"{np.mean(all_distribution_scores):.2f}"
            }
        
        return summary
    
    def export_quality_report(self, output_file: str = None):
        """
        导出质量报告到文件
        
        Args:
            output_file: 输出文件路径，默认为logs/alpha_quality/quality_summary_{timestamp}.csv
        """
        if not self.quality_reports:
            logger.warning("没有质量报告可导出")
            return
        
        # 生成默认文件名
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"logs/alpha_quality/quality_summary_{timestamp}.csv"
        
        # 确保目录存在
        output_path = pd.io.common.get_filepath_or_buffer(output_file, mode='w')[0]
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 构建数据框
        data = []
        for factor_name, report in self.quality_reports.items():
            data.append({
                'factor_name': factor_name,
                'timestamp': report.timestamp,
                'input_missing_ratio': report.input_quality.missing_ratio,
                'output_missing_ratio': report.output_quality.missing_ratio,
                'coverage_ratio': report.output_quality.coverage_ratio,
                'outlier_ratio': report.output_quality.outlier_ratio,
                'zero_ratio': report.output_quality.zero_ratio,
                'distribution_score': report.output_quality.distribution_score,
                'stability_score': report.output_quality.stability_score,
                'time_consistency': report.output_quality.time_consistency,
                'error_count': len(report.errors),
                'warning_count': len(report.warnings),
                'recommendations_count': len(report.recommendations)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"质量报告已导出到: {output_file}")
        
        return df
    
    def get_factor_recommendations(self, factor_name: str = None) -> Dict[str, List[str]]:
        """
        获取因子的优化建议
        
        Args:
            factor_name: 因子名称，None表示所有因子
            
        Returns:
            建议字典
        """
        recommendations = {}
        
        if factor_name:
            if factor_name in self.quality_reports:
                report = self.quality_reports[factor_name]
                recommendations[factor_name] = report.recommendations
        else:
            for fname, report in self.quality_reports.items():
                if report.recommendations:
                    recommendations[fname] = report.recommendations
        
        return recommendations
    
    def print_quality_dashboard(self):
        """
        打印质量仪表板
        """
        summary = self.get_quality_summary()
        
        print("\n" + "="*80)
        print("Alpha因子数据质量仪表板")
        print("="*80)
        
        print(f"\n总因子数: {summary['total_factors']}")
        print(f"存在错误的因子: {summary['factors_with_errors']}")
        print(f"存在警告的因子: {summary['factors_with_warnings']}")
        
        if summary.get('average_metrics'):
            print("\n平均质量指标:")
            for metric, value in summary['average_metrics'].items():
                print(f"  {metric}: {value}")
        
        # 找出问题最严重的因子
        if summary.get('factor_details'):
            print("\n需要关注的因子:")
            problem_factors = []
            for fname, details in summary['factor_details'].items():
                if details['errors'] > 0 or details['warnings'] > 2:
                    problem_factors.append((fname, details['errors'], details['warnings']))
            
            if problem_factors:
                problem_factors.sort(key=lambda x: (x[1], x[2]), reverse=True)
                for fname, errors, warnings in problem_factors[:5]:
                    print(f"  {fname}: {errors}个错误, {warnings}个警告")
            else:
                print("  所有因子质量良好")
        
        print("\n" + "="*80)
