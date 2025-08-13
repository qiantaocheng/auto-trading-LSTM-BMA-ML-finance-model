#!/usr/bin/env python3
"""
增强Alpha策略模块
集成delay/decay、hump+rank、中性化、winsorize等高级技术
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

# 移除外部高级因子模块依赖，所有因子已整合到本模块

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaStrategiesEngine:
    """Alpha策略引擎：统一计算、中性化、排序、门控"""
    
    def __init__(self, config_path: str = "alphas_config.yaml"):
        """
        初始化Alpha策略引擎
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.alpha_functions = self._register_alpha_functions()
            
        self.alpha_cache = {}  # 缓存计算结果
        
        # 所有因子已整合到本模块，无需外部依赖
        logger.info("所有Alpha因子已整合到本模块")
        
        # 统计信息
        self.stats = {
            'computation_times': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'neutralization_stats': {},
            'ic_stats': {}
        }
        
        logger.info(f"Alpha策略引擎初始化完成，加载{len(self.config['alphas'])}个因子")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件{config_path}未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
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
        """注册Alpha计算函数 - 所有因子已整合"""
        return {
            # 技术因子
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
            
            # 动量扩展因子
            'new_high_proximity': self._compute_52w_new_high_proximity,
            'low_beta': self._compute_low_beta_anomaly,
            'idiosyncratic_vol': self._compute_idiosyncratic_volatility,
            
            # 基本面因子
            'earnings_surprise': self._compute_earnings_surprise,
            'analyst_revision': self._compute_analyst_revision,
            'ebit_ev': self._compute_ebit_ev,
            'fcf_ev': self._compute_fcf_ev,
            'earnings_yield': self._compute_earnings_yield,
            'sales_yield': self._compute_sales_yield,
            
            # 盈利能力因子
            'gross_margin': self._compute_gross_margin,
            'operating_profitability': self._compute_operating_profitability,
            'roe_neutralized': self._compute_roe_neutralized,
            'roic_neutralized': self._compute_roic_neutralized,
            'net_margin': self._compute_net_margin,
            'cash_yield': self._compute_cash_yield,
            'shareholder_yield': self._compute_shareholder_yield,
            
            # 应计项目因子
            'total_accruals': self._compute_total_accruals,
            'working_capital_accruals': self._compute_working_capital_accruals,
            'net_operating_assets': self._compute_net_operating_assets,
            'asset_growth': self._compute_asset_growth,
            'net_equity_issuance': self._compute_net_equity_issuance,
            'investment_factor': self._compute_investment_factor,
            
            # 质量评分因子
            'piotroski_score': self._compute_piotroski_score,
            'ohlson_score': self._compute_ohlson_score,
            'altman_score': self._compute_altman_score,
            'qmj_score': self._compute_qmj_score,
            'earnings_stability': self._compute_earnings_stability,
            
            'hump': None,  # 特殊处理
        }
    
    # ========== 基础工具函数 ==========
    
    def winsorize_series(self, s: pd.Series, k: float = 2.5) -> pd.Series:
        """Winsorize序列：去极值"""
        if s.isna().all():
            return s
        mu, sd = s.mean(), s.std(ddof=0)
        if sd == 0:
            return s
        lo, hi = mu - k * sd, mu + k * sd
        return s.clip(lo, hi)
    
    def zscore_by_group(self, df: pd.DataFrame, col: str, group_cols: List[str]) -> pd.Series:
        """按组标准化"""
        return df.groupby(group_cols)[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        )
    
    def neutralize_factor(self, df: pd.DataFrame, target_col: str, 
                         group_cols: List[str]) -> pd.Series:
        """线性回归中性化"""
        def _neutralize_cross_section(block):
            if len(block) < 2 or target_col not in block.columns:
                return block[target_col] if target_col in block.columns else pd.Series(index=block.index)
            
            y = block[target_col].dropna()
            if len(y) < 2:
                return block[target_col]
            
            # 构建虚拟变量矩阵
            X_df = pd.get_dummies(block[group_cols], drop_first=False)
            X_df = X_df.loc[y.index]  # 对齐索引
            
            if X_df.shape[1] == 0 or X_df.var().sum() == 0:
                return y - y.mean()
            
            try:
                lr = LinearRegression(fit_intercept=True)
                lr.fit(X_df.values, y.values)
                residuals = y.values - lr.predict(X_df.values)
                
                result = pd.Series(index=block.index, dtype=float)
                result.loc[y.index] = residuals
                return result.fillna(0)
            except Exception as e:
                logger.warning(f"中性化失败: {e}")
                return y - y.mean()
        
        return df.groupby('date').apply(_neutralize_cross_section).reset_index(level=0, drop=True)
    
    def hump_transform(self, z: pd.Series, hump: float = 0.003) -> pd.Series:
        """门控变换：小信号置零"""
        return z.where(z.abs() >= hump, 0.0)
    
    def rank_transform(self, z: pd.Series) -> pd.Series:
        """排序变换"""
        return z.rank(pct=True) - 0.5
    
    def ema_decay(self, s: pd.Series, span: int) -> pd.Series:
        """指数移动平均衰减"""
        return s.ewm(span=span, adjust=False).mean()
    
    # ========== Alpha因子计算函数 ==========
    
    def _compute_momentum(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """动量因子：多窗口价格动量"""
        results = []
        
        for window in windows:
            # 计算对数收益率动量，添加delay=1（避免未来信息）
            momentum = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x.shift(1) / x.shift(window + 1))
            )

            # 指数衰减（按ticker分组对该Series做EWMA）
            momentum_decayed = momentum.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            results.append(momentum_decayed)
        
        # 多窗口平均
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_reversal(self, df: pd.DataFrame, windows: List[int], 
                         decay: int = 6) -> pd.Series:
        """反转因子：短期价格反转"""
        results = []
        
        for window in windows:
            # 短期收益率，取负值表示反转
            reversal = df.groupby('ticker')['Close'].transform(
                lambda x: -np.log(x.shift(1) / x.shift(window + 1))
            )

            # 指数衰减
            reversal_decayed = reversal.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            results.append(reversal_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_volatility(self, df: pd.DataFrame, windows: List[int], 
                           decay: int = 6) -> pd.Series:
        """波动率因子：已实现波动率的倒数"""
        results = []
        
        for window in windows:
            # 计算对数收益率
            returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(1))
            )

            # 滚动波动率（对每个ticker独立计算）
            volatility = returns.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).std()
            ).reset_index(level=0, drop=True)

            # 波动率倒数（低波动率异常）
            inv_volatility = 1.0 / (volatility + 1e-6)

            # 指数衰减
            inv_vol_decayed = inv_volatility.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)

            results.append(inv_vol_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_volume_turnover(self, df: pd.DataFrame, windows: List[int], 
                                decay: int = 6) -> pd.Series:
        """成交量换手率因子"""
        results = []
        
        for window in windows:
            # 成交量相对强度
            if 'volume' in df.columns:
                volume_ma = df.groupby('ticker')['volume'].transform(
                    lambda x: x.rolling(window=window, min_periods=max(1, window//2)).mean()
                )
                volume_ratio = df['volume'] / (volume_ma + 1e-9)
            else:
                # 如果没有成交量数据，用成交额替代
                volume_ratio = df.groupby('ticker')['amount'].transform(
                    lambda x: x / (x.rolling(window=window, min_periods=max(1, window//2)).mean() + 1e-9)
                )
            
            # 指数衰减
            volume_decayed = volume_ratio.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(volume_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_amihud_illiquidity(self, df: pd.DataFrame, windows: List[int], 
                                   decay: int = 6) -> pd.Series:
        """Amihud流动性指标：价格冲击的倒数"""
        results = []
        
        for window in windows:
            # 计算日收益率
            returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.abs(np.log(x / x.shift(1)))
            )
            
            # Amihud流动性：|收益率| / 成交额
            if 'amount' in df.columns:
                amihud = returns / (df['amount'] + 1e-9)
            else:
                # 替代方案：使用价格*成交量
                amihud = returns / (df['Close'] * df.get('volume', 1) + 1e-9)
            
            # 滚动平均
            amihud_ma = amihud.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).mean()
            ).reset_index(level=0, drop=True)
            
            # 流动性 = 1 / Amihud（高流动性更好）
            liquidity = 1.0 / (amihud_ma + 1e-9)
            
            # 指数衰减
            liquidity_decayed = liquidity.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(liquidity_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_bid_ask_spread(self, df: pd.DataFrame, windows: List[int], 
                               decay: int = 6) -> pd.Series:
        """买卖价差因子（模拟）"""
        results = []
        
        for window in windows:
            # 如果有高低价数据，用 (high-low)/close 作为价差代理
            if 'High' in df.columns and 'Low' in df.columns:
                spread_proxy = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
            else:
                # 替代方案：用价格波动作为价差代理
                price_vol = df.groupby('ticker')['Close'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std() / (x + 1e-9)
                )
                spread_proxy = price_vol
            
            # 滚动平均价差
            spread_ma = spread_proxy.groupby(df['ticker']).apply(
                lambda s: s.rolling(window=window, min_periods=max(1, window//2)).mean()
            ).reset_index(level=0, drop=True)
            
            # 窄价差因子（价差越小越好）
            narrow_spread = 1.0 / (spread_ma + 1e-6)
            
            # 指数衰减
            spread_decayed = narrow_spread.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(spread_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    def _compute_residual_momentum(self, df: pd.DataFrame, windows: List[int], 
                                  decay: int = 6) -> pd.Series:
        """残差动量：去除市场beta后的特异动量"""
        results = []
        
        for window in windows:
            # 计算个股收益率
            stock_returns = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(1))
            )
            
            # 计算市场收益率（等权平均或市值加权）
            market_returns = df.groupby('date')['Close'].transform(
                lambda x: np.log(x.mean() / x.shift(1).mean())
            )
            
            # 滚动回归计算beta和残差
            def calculate_residual_momentum(group):
                # 从外部预计算的Series按索引切片，避免.name依赖
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
                        # 简单线性回归：个股收益 = alpha + beta * 市场收益 + 残差
                        slope, intercept, _, _, _ = stats.linregress(x.values, y.values)
                        predicted = intercept + slope * x.iloc[-1]
                        residual = y.iloc[-1] - predicted
                        residuals.append(residual)
                    except:
                        residuals.append(0.0)
                
                return pd.Series(residuals, index=group_returns.index)
            
            residual_momentum = df.groupby('ticker').apply(calculate_residual_momentum)
            residual_momentum = residual_momentum.reset_index(level=0, drop=True)

            # 指数衰减
            residual_decayed = residual_momentum.groupby(df['ticker']).apply(
                lambda s: s.ewm(span=decay, adjust=False).mean()
            ).reset_index(level=0, drop=True)
            
            results.append(residual_decayed)
        
        return pd.concat(results, axis=1).mean(axis=1)
    
    # ===== v2 新增因子：统一进入类方法并在注册表登记 =====
    def _compute_reversal_5(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """短期反转（1-5日），与中长期动量互补"""
        try:
            g = df.groupby('ticker')['Close']
            rev = -(g.shift(1) / g.shift(6) - 1.0)
            return rev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"短期反转计算失败: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_amihud_illiquidity_new(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """改进Amihud非流动性：更稳健的滚动中位数与EMA衰减"""
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
            logger.warning(f"Amihud非流动性计算失败: {e}")
            return pd.Series(0.0, index=df.index)

    def _compute_pead(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """PEAD（财报后漂移）事件驱动代理"""
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
            logger.warning(f"PEAD计算失败: {e}")
            return pd.Series(0.0, index=df.index)

    # ===== 新增动量类因子 =====
    
    def _compute_momentum_6_1(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """6-1动量：(t-126 to t-21)的价格动量，排除最近1个月"""
        try:
            g = df.groupby('ticker')['Close']
            # 6个月前到1个月前的收益率
            momentum_6_1 = (g.shift(21) / g.shift(126) - 1.0)
            return momentum_6_1.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"6-1动量计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_52w_new_high_proximity(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """52周新高接近度：当前价格占52周最高价的比例"""
        try:
            window = 252  # 52周 ≈ 252个交易日
            g = df.groupby('ticker')['Close']
            max_52w = g.rolling(window=window, min_periods=min(window//2, 60)).max()
            current_price = df['Close']
            proximity = current_price / max_52w
            return proximity.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"52周新高接近度计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_low_beta_anomaly(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """低β异象：计算个股相对市场的β，取负值（低β更优）"""
        try:
            window = windows[0] if windows else 60
            stock_returns = df.groupby('ticker')['Close'].pct_change()
            market_returns = df.groupby('date')['Close'].transform('mean').pct_change()
            
            def calc_rolling_beta(stock_ret, market_ret, window):
                betas = []
                for i in range(len(stock_ret)):
                    start_idx = max(0, i - window + 1)
                    x_window = market_ret.iloc[start_idx:i+1]
                    y_window = stock_ret.iloc[start_idx:i+1]
                    valid = ~(x_window.isna() | y_window.isna())
                    
                    if valid.sum() < 10:
                        betas.append(1.0)
                        continue
                    
                    try:
                        x_valid, y_valid = x_window[valid].values, y_window[valid].values
                        if x_valid.std() < 1e-8:
                            betas.append(1.0)
                        else:
                            beta = np.cov(x_valid, y_valid)[0, 1] / np.var(x_valid)
                            betas.append(beta)
                    except:
                        betas.append(1.0)
                return pd.Series(-np.array(betas), index=stock_ret.index)  # 取负值
            
            low_beta = df.groupby('ticker').apply(
                lambda group: calc_rolling_beta(
                    stock_returns[group.index], 
                    market_returns[group.index], 
                    window
                )
            ).reset_index(level=0, drop=True)
            
            return low_beta.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"低β异象计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_idiosyncratic_volatility(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """特异波动率：剔除市场因子后的残差波动率，取负值（低波动更优）"""
        try:
            window = windows[0] if windows else 60
            stock_returns = df.groupby('ticker')['Close'].pct_change()
            market_returns = df.groupby('date')['Close'].transform('mean').pct_change()
            
            def calc_idio_vol(group):
                stock_ret = stock_returns[group.index]
                market_ret = market_returns[group.index]
                
                idio_vols = []
                for i in range(len(group)):
                    start_idx = max(0, i - window + 1)
                    x_window = market_ret.iloc[start_idx:i+1]
                    y_window = stock_ret.iloc[start_idx:i+1]
                    valid = ~(x_window.isna() | y_window.isna())
                    
                    if valid.sum() < 20:
                        idio_vols.append(0.0)
                        continue
                    
                    try:
                        x_valid, y_valid = x_window[valid].values, y_window[valid].values
                        
                        if x_valid.std() < 1e-8:
                            residuals = y_valid - y_valid.mean()
                        else:
                            beta = np.cov(x_valid, y_valid)[0, 1] / np.var(x_valid)
                            alpha = y_valid.mean() - beta * x_valid.mean()
                            residuals = y_valid - (alpha + beta * x_valid)
                        
                        idio_vol = np.std(residuals)
                        idio_vols.append(-idio_vol)  # 取负值，低波动更优
                    except:
                        idio_vols.append(0.0)
                
                return pd.Series(idio_vols, index=group.index)
            
            idio_vol = df.groupby('ticker').apply(calc_idio_vol).reset_index(level=0, drop=True)
            return idio_vol.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"特异波动率计算失败: {e}")
            return pd.Series(0.0, index=df.index)

    # ===== 基本面因子（使用代理数据） =====
    
    def _compute_earnings_surprise(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """财报意外SUE：标准化盈余惊喜（使用价格反应作为代理）"""
        try:
            window = windows[0] if windows else 63  # 季度
            # 使用价格在财报期间的异常反应作为SUE代理
            returns = df.groupby('ticker')['Close'].pct_change()
            # 季度超额收益率作为SUE代理
            quarterly_returns = df.groupby('ticker')['Close'].pct_change(periods=window)
            market_returns = df.groupby('date')['Close'].transform('mean').pct_change(periods=window)
            excess_returns = quarterly_returns - market_returns
            # 标准化
            sue_proxy = excess_returns.groupby(df['ticker']).transform(
                lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
            )
            return sue_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"财报意外SUE计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_analyst_revision(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """分析师EPS上调修正（使用动量变化率作为代理）"""
        try:
            # 使用动量变化作为分析师预期修正的代理
            short_momentum = df.groupby('ticker')['Close'].pct_change(21)  # 1月
            long_momentum = df.groupby('ticker')['Close'].pct_change(63)   # 3月
            revision_proxy = short_momentum - long_momentum
            return revision_proxy.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"分析师修正计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_ebit_ev(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """EBIT/EV收益率（使用收益率代理）"""
        try:
            # 使用基于价格的收益率代理EBIT/EV
            if 'volume' in df.columns:
                enterprise_value = df['Close'] * df['volume']  # 简化的EV代理
                ebit_proxy = df.groupby('ticker')['Close'].pct_change(252).abs()  # 年化收益率作为EBIT代理
                ebit_ev = ebit_proxy / (enterprise_value / enterprise_value.rolling(252).mean())
                return ebit_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"EBIT/EV计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_fcf_ev(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """自由现金流收益率FCF/EV（使用现金流代理）"""
        try:
            # 使用基于成交量和价格的现金流代理
            if 'volume' in df.columns and 'amount' in df.columns:
                fcf_proxy = df['amount'] / df['Close']  # 成交额/价格作为现金流代理
                ev_proxy = df['Close'] * df['volume']
                fcf_ev = fcf_proxy / (ev_proxy + 1e-9)
                fcf_ev = fcf_ev.groupby(df['ticker']).transform(
                    lambda x: (x - x.rolling(252).mean()) / (x.rolling(252).std() + 1e-8)
                )
                return fcf_ev.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"FCF/EV计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """盈利收益率E/P（市盈率倒数的代理）"""
        try:
            # 使用收益率历史数据作为E/P代理
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            earnings_yield = annual_return / df['Close'] * 100  # 标准化
            return earnings_yield.groupby(df['ticker']).transform(lambda x: self.ema_decay(x, span=decay)).fillna(0)
        except Exception as e:
            logger.warning(f"盈利收益率E/P计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_sales_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """销售收益率S/P（市销率倒数的代理）"""
        try:
            # 使用成交量作为销售额代理
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
            logger.warning(f"销售收益率S/P计算失败: {e}")
            return pd.Series(0.0, index=df.index)
 
    # ===== 高级Alpha因子（暂时移除复杂实现，保持基础功能） =====
    
    def _compute_gross_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """毛利率GP/Assets（简化实现）"""
        try:
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            return annual_return.fillna(0)
        except Exception as e:
            logger.warning(f"毛利率计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _compute_operating_profitability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """经营盈利能力（简化实现）"""
        try:
            if 'volume' in df.columns:
                efficiency = df['volume'] / (df['Close'] + 1e-9)
                return efficiency.fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except Exception as e:
            logger.warning(f"经营盈利能力计算失败: {e}")
            return pd.Series(0.0, index=df.index)
    
    # 为所有其他高级因子添加简化实现
    def _compute_roe_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROE中性化（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(252)
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_roic_neutralized(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """ROIC中性化（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(126)
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_margin(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """净利率（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change(63)
            return returns.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_cash_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """现金收益率（简化实现）"""
        try:
            if 'amount' in df.columns:
                cash_yield = df['amount'] / (df['Close'] + 1e-9)
                return cash_yield.fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_shareholder_yield(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """股东收益率（简化实现）"""
        try:
            if 'volume' in df.columns:
                volume_ma = df.groupby('ticker')['volume'].rolling(22).mean()
                ratio = df['volume'] / (volume_ma + 1e-9)
                return ratio.fillna(0)
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    # 应计项目因子
    def _compute_total_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """总应计（简化实现）"""
        try:
            price_change = df.groupby('ticker')['Close'].pct_change()
            return -price_change.fillna(0)  # 取负值
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_working_capital_accruals(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """营运资本应计（简化实现）"""
        try:
            if 'volume' in df.columns:
                wc_proxy = df.groupby('ticker')['volume'].pct_change()
                return -wc_proxy.fillna(0)  # 取负值
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_operating_assets(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """净经营资产（简化实现）"""
        try:
            if 'volume' in df.columns:
                noa_proxy = df['volume'] / (df['Close'] + 1e-9)
                return -noa_proxy.pct_change().fillna(0)  # 取负值
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_asset_growth(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """资产增长（简化实现）"""
        try:
            if 'volume' in df.columns:
                market_value = df['Close'] * df['volume']
                growth = market_value.groupby(df['ticker']).pct_change(252)
                return -growth.fillna(0)  # 取负值
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_net_equity_issuance(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """净股本发行（简化实现）"""
        try:
            if 'volume' in df.columns:
                volume_spike = df.groupby('ticker')['volume'].pct_change()
                return -volume_spike.fillna(0)  # 取负值
            else:
                return pd.Series(0.0, index=df.index)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_investment_factor(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """投资因子（简化实现）"""
        try:
            price_vol = df.groupby('ticker')['Close'].rolling(22).std()
            return -price_vol.fillna(0)  # 取负值
        except:
            return pd.Series(0.0, index=df.index)
    
    # 质量评分因子
    def _compute_piotroski_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Piotroski评分（简化实现）"""
        try:
            annual_return = df.groupby('ticker')['Close'].pct_change(252)
            score = (annual_return > 0).astype(float)
            return score.fillna(0.5)
        except:
            return pd.Series(0.5, index=df.index)
    
    def _compute_ohlson_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Ohlson评分（简化实现）"""
        try:
            price_vol = df.groupby('ticker')['Close'].rolling(126).std() / df['Close']
            return -price_vol.fillna(0)  # 取负值，低风险更优
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_altman_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """Altman评分（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            stability = -returns.rolling(126).std()  # 稳定性
            return stability.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_qmj_score(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """QMJ质量评分（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            quality = returns.rolling(252).mean() / (returns.rolling(252).std() + 1e-8)
            return quality.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
    
    def _compute_earnings_stability(self, df: pd.DataFrame, windows: List[int], decay: int) -> pd.Series:
        """盈利稳定性（简化实现）"""
        try:
            returns = df.groupby('ticker')['Close'].pct_change()
            stability = -returns.rolling(252).std()  # 低波动更优
            return stability.fillna(0)
        except:
            return pd.Series(0.0, index=df.index)
 
    # ========== 主要计算流程 ==========
    
    def compute_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有Alpha因子
        
        Args:
            df: 包含价格数据的DataFrame，必须有columns: ['date', 'ticker', 'Close', 'amount', ...]
            
        Returns:
            包含所有Alpha因子的DataFrame
        """
        logger.info(f"开始计算{len(self.config['alphas'])}个Alpha因子")
        
        # 确保必需的列存在
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 添加元数据列（如果不存在）
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
                
                # 获取参数
                windows = alpha_config.get('windows', [22])
                decay = alpha_config.get('decay', 6)
                
                if alpha_kind == 'hump':
                    # 特殊处理hump变换
                    base_name = alpha_config['base']
                    if base_name not in alpha_results:
                        logger.warning(f"Hump因子{alpha_name}的基础因子{base_name}未找到")
                        continue
                    
                    base_factor = alpha_results[base_name].copy()
                    hump_level = alpha_config['hump']
                    alpha_factor = self.hump_transform(base_factor, hump=hump_level)
                else:
                    # 常规因子计算 - 所有因子已整合到本模块
                    if alpha_kind not in self.alpha_functions:
                        logger.warning(f"未知的Alpha类型: {alpha_kind}")
                        continue
                    
                    alpha_func = self.alpha_functions[alpha_kind]
                    alpha_factor = alpha_func(
                        df=df,
                        windows=windows,
                        decay=decay
                    )
                
                # 数据处理流水线
                alpha_factor = self._process_alpha_pipeline(
                    df=df,
                    alpha_factor=alpha_factor,
                    alpha_config=alpha_config,
                    alpha_name=alpha_name
                )
                
                # 全局特征滞后以防止任何潜在的数据泄露
                # 使用配置项 feature_global_lag，默认2（T-2），表示预测时仅使用至少T-2的信息
                try:
                    global_lag = int(self.config.get('feature_global_lag', 2))
                except Exception:
                    global_lag = 2
                if global_lag and global_lag > 0:
                    alpha_factor = alpha_factor.groupby(df['ticker']).shift(global_lag)
                
                alpha_results[alpha_name] = alpha_factor
                computation_times[alpha_name] = (pd.Timestamp.now() - start_time).total_seconds()
                
                logger.info(f"✓ {alpha_name}: 计算完成 ({computation_times[alpha_name]:.2f}s)")
                
            except Exception as e:
                logger.error(f"✗ {alpha_name}: 计算失败 - {e}")
                continue
        
        # 更新统计信息
        self.stats['computation_times'].update(computation_times)
        
        # 构建结果DataFrame，保留原始列
        result_df = df.copy()
        for alpha_name, alpha_series in alpha_results.items():
            result_df[alpha_name] = alpha_series
        
        if alpha_results:
            logger.info(f"Alpha计算完成，共{len(alpha_results)}个因子")
        else:
            logger.error("所有Alpha因子计算失败")
        
        return result_df
    
    def _process_alpha_pipeline(self, df: pd.DataFrame, alpha_factor: pd.Series, 
                               alpha_config: Dict, alpha_name: str) -> pd.Series:
        """Alpha因子处理流水线：winsorize -> neutralize -> zscore -> transform"""
        
        # 1. Winsorize去极值
        winsorize_std = self.config.get('winsorize_std', 2.5)
        alpha_factor = self.winsorize_series(alpha_factor, k=winsorize_std)
        
        # 2. 构建临时DataFrame进行中性化
        temp_df = df[['date', 'ticker'] + self.config['neutralization']].copy()
        temp_df[alpha_name] = alpha_factor
        
        # 3. 中性化
        for neutralize_level in self.config['neutralization']:
            if neutralize_level in temp_df.columns:
                alpha_factor = self.neutralize_factor(
                    temp_df, alpha_name, [neutralize_level]
                )
                temp_df[alpha_name] = alpha_factor
        
        # 4. 截面标准化
        alpha_factor = self.zscore_by_group(
            temp_df, alpha_name, ['date']
        )
        
        # 5. 变换（rank或保持原样）
        xform = alpha_config.get('xform', 'zscore')
        if xform == 'rank':
            alpha_factor = temp_df.groupby('date')[alpha_name].transform(
                lambda x: self.rank_transform(x)
            )
        
        return alpha_factor
    
    def compute_oof_scores(self, alpha_df: pd.DataFrame, target: pd.Series, 
                          dates: pd.Series, metric: str = 'ic') -> pd.Series:
        """
        计算Out-of-Fold评分
        
        Args:
            alpha_df: Alpha因子DataFrame
            target: 目标变量
            dates: 日期序列
            metric: 评分指标 ('ic', 'sharpe', 'fitness')
            
        Returns:
            每个Alpha的OOF评分
        """
        logger.info(f"开始计算OOF评分，指标: {metric}")
        
        # 只评估数值型的因子列，排除标识/价格/元数据列
        exclude_cols = set(['date','ticker','COUNTRY','SECTOR','SUBINDUSTRY',
                            'Open','High','Low','Close','Adj Close',
                            'open','high','low','close','adj_close','volume','amount'])
        factor_cols = [c for c in alpha_df.columns
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(alpha_df[c])]

        # 使用TimeSeriesSplit进行时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        unique_dates = sorted(dates.unique())
        
        scores = {}
        for col in factor_cols:
            col_scores = []
            
            for train_idx, test_idx in tscv.split(unique_dates):
                # 获取测试期间的数据
                test_dates = [unique_dates[i] for i in test_idx]
                test_mask = dates.isin(test_dates)
                
                if test_mask.sum() == 0:
                    continue
                
                y_test = target[test_mask]
                x_test = alpha_df[col][test_mask]
                
                # 去除NaN值
                valid_mask = ~(x_test.isna() | y_test.isna())
                if valid_mask.sum() < 10:  # 最少需要10个有效样本
                    continue
                
                x_valid = x_test[valid_mask]
                y_valid = y_test[valid_mask]
                
                # 计算评分
                if metric == 'ic':
                    score = np.corrcoef(x_valid.values, y_valid.values)[0, 1]
                elif metric == 'sharpe':
                    returns = x_valid.values * y_valid.values
                    score = returns.mean() / (returns.std(ddof=0) + 1e-12)
                elif metric == 'fitness':
                    # 信息系数 * sqrt(样本数)
                    ic = np.corrcoef(x_valid.values, y_valid.values)[0, 1]
                    score = ic * np.sqrt(len(x_valid))
                else:
                    score = 0.0
                
                if not np.isnan(score):
                    col_scores.append(score)
            
            scores[col] = np.nanmean(col_scores) if col_scores else 0.0
        
        # 更新统计信息
        self.stats['ic_stats'] = scores
        
        result = pd.Series(scores, name=f'oof_{metric}')
        logger.info(f"OOF评分完成，平均{metric}: {result.mean():.4f}")
        
        return result
    
    def compute_bma_weights(self, scores: pd.Series, temperature: float = None, use_weight_hints: bool = True) -> pd.Series:
        """
        基于评分计算BMA权重，支持weight_hint先验
        
        Args:
            scores: OOF评分
            temperature: 温度系数，控制权重集中度
            use_weight_hints: 是否使用weight_hint作为先验权重
            
        Returns:
            BMA权重
        """
        if temperature is None:
            temperature = self.config.get('temperature', 1.2)
        
        # 获取weight_hint先验权重
        weight_hints = {}
        if use_weight_hints:
            for alpha_config in self.config.get('alphas', []):
                alpha_name = alpha_config['name']
                if alpha_name in scores.index:
                    weight_hints[alpha_name] = alpha_config.get('weight_hint', 0.05)
        
        # 标准化评分
        scores_std = (scores - scores.mean()) / (scores.std(ddof=0) + 1e-12)
        scores_scaled = scores_std / max(temperature, 1e-3)
        
        # Log-sum-exp softmax（数值稳定）
        max_score = scores_scaled.max()
        exp_scores = np.exp(scores_scaled - max_score)
        
        # 结合weight_hint先验
        if weight_hints and use_weight_hints:
            hint_weights = pd.Series(weight_hints).reindex(scores.index, fill_value=0.05)
            hint_weights = hint_weights / hint_weights.sum()  # 标准化
            
            # 贝叶斯更新：先验 * 似然
            posterior_weights = hint_weights * exp_scores
            weights = posterior_weights / posterior_weights.sum()
            
            logger.info("使用weight_hint先验进行贝叶斯权重更新")
        else:
            # 普通softmax
            eps = 1e-6
            weights = (exp_scores + eps) / (exp_scores.sum() + eps * len(exp_scores))
        
        weights_series = pd.Series(weights, index=scores.index, name='bma_weights')
        
        logger.info(f"BMA权重计算完成，权重分布: max={weights.max():.3f}, min={weights.min():.3f}")
        logger.info(f"主要因子权重: {weights_series.nlargest(5).to_dict()}")
        
        return weights_series
    
    def combine_alphas(self, alpha_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        使用BMA权重组合Alpha因子
        
        Args:
            alpha_df: Alpha因子DataFrame
            weights: BMA权重
            
        Returns:
            组合后的Alpha信号
        """
        # 仅使用数值型因子列，排除元数据
        exclude_cols = set(['date','ticker','COUNTRY','SECTOR','SUBINDUSTRY',
                            'Open','High','Low','Close','Adj Close',
                            'open','high','low','close','adj_close','volume','amount'])
        factor_cols = [c for c in alpha_df.columns
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(alpha_df[c])]
        if not factor_cols:
            return pd.Series(index=alpha_df.index, dtype=float)

        # 确保权重对齐（列方向）
        aligned_weights = weights.reindex(factor_cols, fill_value=0.0)
        total_w = aligned_weights.sum()
        if total_w <= 0:
            aligned_weights[:] = 1.0 / len(aligned_weights)
        else:
            aligned_weights = aligned_weights / total_w

        # 列方向相乘，避免与行索引对齐导致的类型错误
        combined_signal = alpha_df[factor_cols].mul(aligned_weights, axis=1).sum(axis=1)
        
        logger.info(f"Alpha组合完成，信号范围: [{combined_signal.min():.4f}, {combined_signal.max():.4f}]")
        
        return combined_signal
    
    def get_stats(self) -> Dict:
        """获取计算统计信息"""
        return self.stats.copy()


if __name__ == "__main__":
    pass