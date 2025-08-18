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
        
        # Statistics
        self.stats = {
            'computation_times': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'neutralization_stats': {},
            'ic_stats': {}
        }
        
        logger.info(f"Alpha Strategy Engine initialized, loaded {len(self.config['alphas'])} factors")
    
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
            
            y = block[target_col].dropna()
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
                
                # Only use historical data up to current time point (expanding window)
                hist_indices = sorted_indices[:i+1]
                hist_y = y.loc[y.index.intersection(hist_indices)]
                
                if len(hist_y) < 2:
                    result.loc[idx] = y.loc[idx] - y.loc[hist_y.index].mean()
                    continue
                
                # Build historical dummy variable matrix
                hist_block = block.loc[hist_indices]
                X_df = pd.get_dummies(hist_block[group_cols], drop_first=False)
                X_df = X_df.loc[hist_y.index]
                
                if X_df.shape[1] == 0 or X_df.var().sum() == 0:
                    result.loc[idx] = hist_y.loc[idx] - hist_y.mean()
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
            
            return result.fillna(0)
        
        return df.groupby('date').apply(_neutralize_cross_section_safe).reset_index(level=0, drop=True)
    
    def hump_transform(self, z: pd.Series, hump: float = 0.003) -> pd.Series:
        """Gating transformation: Set small signals to zero"""
        return z.where(z.abs() >= hump, 0.0)
    
    def rank_transform(self, z: pd.Series) -> pd.Series:
        """Ranking transformation"""
        return z.rank(pct=True) - 0.5
    
    def ema_decay(self, s: pd.Series, span: int) -> pd.Series:
        """Time-safe exponential moving average decay - Only use historical data"""
        # Use expanding window to ensure each time point only uses historical data
        result = s.ewm(span=span, adjust=False).mean()
        # Add one period lag to ensure current period data is not used
        return result.shift(1)
    
    # ========== Alpha Factor Computation Functions ==========
    
    def _compute_momentum(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """Time-safe momentum factor: Multi-window price momentum"""
        results = []
        
        for window in windows:
            # Calculate log returns momentum, with safety margin (T-2 to T-window-2)
            momentum = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x.shift(2) / x.shift(window + 2))
            )

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
        """Reversal factor: Short-term price reversal"""
        results = []
        
        for window in windows:
            # Short-term returns, take negative to indicate reversal
            reversal = df.groupby('ticker')['Close'].transform(
                lambda x: -np.log(x.shift(1) / x.shift(window + 1))
            )

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
            # Calculate log returns
            returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(1))
            )

            # Rolling volatility (calculated independently for each ticker)
            volatility = returns.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).std()
            ).reset_index(level=0, drop=True)

            # Volatility reciprocal (low volatility anomaly)
            inv_volatility = 1.0 / (volatility + 1e-6)

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
                # If no volume data, use amount as substitute
                volume_ratio = df.groupby('ticker')['amount'].transform(
                    lambda x: x / (x.rolling(window=window, min_periods=max(1, window//2)).mean() + 1e-9)
                )
            
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
            # Calculate daily returns
            returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.abs(np.log(x / x.shift(1)))
            )
            
            # Amihud liquidity: |return rate| / amount
            if 'amount' in df.columns:
                amihud = returns / (df['amount'] + 1e-9)
            else:
                # Alternative: use price * volume
                amihud = returns / (df['Close'] * df.get('volume', 1) + 1e-9)
            
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
                    
                    y = group_returns.iloc[i-window:i].dropna()
                    x = group_market.iloc[i-window:i].dropna()
                    
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
            return rev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            else:
                volume_dollar = (df.get('volume', 1e6) * df['Close']).replace(0, np.nan)
            illiq = (returns_abs / volume_dollar).replace([np.inf, -np.inf], np.nan)
            illiq_rolling = illiq.groupby(df['ticker']).rolling(window, min_periods=max(1, window//2)).median().reset_index(level=0, drop=True)
            illiq_factor = -illiq_rolling
            return illiq_factor.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"Amihud illiquidity computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_pead(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """PEAD（财报后漂移）event-driven proxy"""
        try:
            window = windows[0] if windows else 21
            returns_21d = df.groupby('ticker')['Close'].pct_change(periods=window)
            if 'volume' in df.columns:
                vol_ma = df.groupby('ticker')['volume'].rolling(window*2).mean()
                vol_ratio = df['volume'] / vol_ma
                vol_anomaly = vol_ratio.groupby(df['ticker']).transform(lambda x: (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8))
            else:
                vol_anomaly = pd.Series(0.0, index=df.index)
            pead_signal = returns_21d * (1 + vol_anomaly * 0.3)
            threshold = pead_signal.groupby(df['ticker']).rolling(252).quantile(0.8).reset_index(level=0, drop=True)
            pead_filtered = pead_signal.where(pead_signal.abs() > threshold.abs(), 0)
            return pead_filtered.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return momentum_6_1.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"6-1momentum computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_52w_new_high_proximity(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """52周proximity to new high: Current price as percentage of52周 high price"""
        try:
            window = 252  # 52周 ≈ 252个交易日
            g = df.groupby('ticker')['Close']
            max_52w = g.rolling(window=window, min_periods=min(window//2, 60)).max()
            current_price = df['Close']
            proximity = current_price / max_52w
            return proximity.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return low_beta.fillna(0)
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
            return idio_vol.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return sue_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return revision_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
                return ebit_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"EBIT/EV computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_fcf_ev(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Free cash flow yieldFCF/EV（Using现金流 proxy）"""
        try:
            # Using基于volume和price的现金流 proxy
            if 'volume' in df.columns and 'amount' in df.columns:
                fcf_proxy = df['amount'] / df['Close']  # amount/price作为现金流 proxy
                ev_proxy = df['Close'] * df['volume']
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
                fcf_ev = fcf_ev.groupby(df['ticker']).transform(
                    lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
                )
                return fcf_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return earnings_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
                return sales_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
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
            return annual_return.fillna(0)
        except Exception as e:
            logger.warning(f"Gross margin computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_operating_profitability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Operating profitability（简化implementation）"""
        try:
            if 'volume' in df.columns:
                efficiency = df['volume'] / (df['Close'] + 1e-9)
                return efficiency.fillna(0)
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
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_roic_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROICneutralize（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(126)
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net margin（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(63)
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_cash_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Cash yield（简化implementation）"""
        try:
            if 'amount' in df.columns:
                cash_yield = df['amount'] / (df['Close'] + 1e-9)
                return cash_yield.fillna(0)
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
                return ratio.fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    # Accrual factors
    def _compute_total_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Total accruals（简化implementation）"""
        try:
            price_change = df.groupby('ticker')['Close'].pct_change()
            return -price_change.fillna(0)  # Take negative
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_working_capital_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Working capital accruals（简化implementation）"""
        try:
            if 'volume' in df.columns:
                wc_proxy = df.groupby('ticker')['volume'].pct_change()
                return -wc_proxy.fillna(0)  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_operating_assets(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net operating assets（简化implementation）"""
        try:
            if 'volume' in df.columns:
                noa_proxy = df['volume'] / (df['Close'] + 1e-9)
                return -noa_proxy.pct_change().fillna(0)  # Take negative
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
                return -growth.fillna(0)  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_equity_issuance(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Net equity issuance（简化implementation）"""
        try:
            if 'volume' in df.columns:
                volume_spike = df.groupby('ticker')['volume'].pct_change()
                return -volume_spike.fillna(0)  # Take negative
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_investment_factor(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Investment factor（简化implementation）"""
        try:
            price_vol = df.groupby('ticker')['Close'].rolling(22).std()
            return -price_vol.fillna(0)  # Take negative
        except:
            return pd.Series(0.0, index=df.index)
    
    # Quality score factors
    def _compute_piotroski_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """PiotroskiScore（简化implementation）"""
        try:
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            score = (annual_return > 0).astype(float)
            return score.fillna(0.5)
        except:
            return pd.Series(0.5, index=df.index)
    
    def _compute_ohlson_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """OhlsonScore（简化implementation）"""
        try:
            price_vol = df.groupby('ticker')['Close'].rolling(126).std() / df['Close']
            return -price_vol.fillna(0)  # Take negative，lower risk is better
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_altman_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """AltmanScore（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            stability = -returns.rolling(126).std()  # Stability
            return stability.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_qmj_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """QMJ质量Score（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            quality = returns.rolling(252).mean() / (returns.rolling(252).std() + 1e-8)
            return quality.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_stability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """盈利Stability（简化implementation）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            stability = -returns.rolling(252).std()  # lower volatility is better
            return stability.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
 
    # ========== Main Computation Pipeline ==========
    
    def compute_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all Alpha factors
        
        Args:
            df: DataFrame containing price data, must have columns: ['date', 'ticker', 'Close', 'amount', ...]
            
        Returns:
            DataFrame containing all Alpha factors
        """
        logger.info(f"Starting computation of{len(self.config['alphas'])} Alpha factors")
        
        # Ensure required columns exist
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add metadata columns (if not exist)
        for col in ['COUNTRY', 'SECTOR', 'SUBINDUSTRY']:
            if col not in df.columns:
                df[col] = 'Unknown'
        
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
                        df=df,
                        windows=windows,
                        decay=decay
                    )
                
                # Data processing pipeline
                alpha_factor = self._process_alpha_pipeline(
                    df=df,
                    alpha_factor=alpha_factor,
                    alpha_config=alpha_config,
                    alpha_name=alpha_name
                )
                
                # Global feature lag to prevent any potential data leakage
                # Usingconfiguration项 feature_global_lag，default2（T-2），表示prediction时仅Using至少T-2 information
                try:
                    global_lag = int(self.config.get('feature_global_lag', 2))
                except Exception:
                    global_lag = 2
                if global_lag and global_lag > 0:
                    alpha_factor = alpha_factor.groupby(df['ticker']).shift(global_lag)
                
                alpha_results[alpha_name] = alpha_factor
                computation_times[alpha_name] = (pd.Timestamp.now() - start_time).total_seconds()
                
                logger.info(f"SUCCESS {alpha_name}:  computation completed ({computation_times[alpha_name]:.2f}s)")
                
            except Exception as e:
                logger.error(f"FAILED {alpha_name}:  computation failed - {e}")
                continue
        
        # 更新Statistics
        self.stats['computation_times'].update(computation_times)
        
        # Build result DataFrame, preserve original columns
        result_df = df.copy()
        for alpha_name, alpha_series in alpha_results.items():
            result_df[alpha_name] = alpha_series
        
        if alpha_results:
            logger.info(f"Alpha computation completed，共{len(alpha_results)} factors")
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
        
        # 4. 截面Standardize
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

        # UsingTimeSeriesSplit进行timeseriescrossvalidation
        tscv = TimeSeriesSplit(n_splits=5)
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
    def get_stats(self) -> Dict:
        """GetcomputationStatistics"""
        return self.stats.copy()


if __name__ == "__main__":
    pass
