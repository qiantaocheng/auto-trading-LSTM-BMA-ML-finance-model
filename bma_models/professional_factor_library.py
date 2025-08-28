"""
专业因子库 - Fama-French & Barra风险因子
==========================================
基于学术研究和业界最佳实践的因子集合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorDecayConfig:
    """因子衰减配置 - 不同因子不同半衰期"""
    
    # 因子半衰期配置（天数）
    FACTOR_HALFLIFE = {
        # 快速衰减因子（1-3天）
        'news_sentiment': 1,           # 新闻情绪最快衰减
        'intraday_momentum': 1,        # 日内动量
        'order_flow_imbalance': 2,     # 订单流失衡
        'short_interest_change': 3,    # 空头兴趣变化
        
        # 中速衰减因子（5-22天）
        'price_momentum': 5,           # 价格动量
        'earnings_surprise': 10,       # 盈利意外
        'analyst_revision': 15,        # 分析师修正
        'volume_momentum': 22,         # 成交量动量
        
        # 慢速衰减因子（66-132天）
        'value_composite': 66,         # 价值因子
        'quality_score': 88,           # 质量分数
        'profitability': 110,          # 盈利能力
        'investment_quality': 132,     # 投资质量
        
        # 极慢衰减因子（252天+）
        'size_factor': 252,            # 市值因子
        'low_volatility': 252,         # 低波动率
        'dividend_yield': 360,         # 股息率
    }
    
    # 默认半衰期
    DEFAULT_HALFLIFE: int = 22
    
    def get_decay_weight(self, factor_name: str, days_ago: int) -> float:
        """计算衰减权重"""
        halflife = self.FACTOR_HALFLIFE.get(factor_name, self.DEFAULT_HALFLIFE)
        # 指数衰减: weight = exp(-lambda * t), lambda = ln(2) / halflife
        decay_rate = np.log(2) / halflife
        return np.exp(-decay_rate * days_ago)


class FamaFrenchFactors:
    """
    Fama-French因子模型
    包括经典3因子、5因子模型
    """
    
    @staticmethod
    def calculate_market_factor(returns: pd.Series, market_returns: pd.Series) -> pd.Series:
        """
        MKT: 市场因子（超额收益）
        """
        return returns - market_returns
    
    @staticmethod
    def calculate_size_factor(market_cap: pd.Series, returns: pd.Series = None) -> pd.Series:
        """
        SMB: Small Minus Big（小市值减大市值）
        """
        # 按市值分组
        median_cap = market_cap.median()
        small_cap_mask = market_cap <= median_cap
        
        # 如果没有提供returns，返回市值因子
        if returns is None:
            # 返回标准化的市值因子
            size_factor = pd.Series(0, index=market_cap.index)
            size_factor[small_cap_mask] = 1
            size_factor[~small_cap_mask] = -1
            return size_factor
        
        # 计算SMB
        small_cap_return = returns[small_cap_mask].mean()
        large_cap_return = returns[~small_cap_mask].mean()
        
        return pd.Series(small_cap_return - large_cap_return, index=market_cap.index)
    
    @staticmethod
    def calculate_value_factor(book_to_market: pd.Series) -> pd.Series:
        """
        HML: High Minus Low（高账面市值比减低账面市值比）
        """
        # 按B/M分组
        terciles = book_to_market.quantile([0.3, 0.7])
        
        low_bm = book_to_market <= terciles.iloc[0]
        high_bm = book_to_market >= terciles.iloc[1]
        
        # 标准化因子值
        value_factor = pd.Series(0, index=book_to_market.index)
        value_factor[high_bm] = 1
        value_factor[low_bm] = -1
        
        return value_factor
    
    @staticmethod
    def calculate_profitability_factor(roe: pd.Series) -> pd.Series:
        """
        RMW: Robust Minus Weak（高盈利减低盈利）
        Fama-French 5因子模型
        """
        # 按ROE分组
        terciles = roe.quantile([0.3, 0.7])
        
        weak_prof = roe <= terciles.iloc[0]
        robust_prof = roe >= terciles.iloc[1]
        
        prof_factor = pd.Series(0, index=roe.index)
        prof_factor[robust_prof] = 1
        prof_factor[weak_prof] = -1
        
        return prof_factor
    
    @staticmethod
    def calculate_investment_factor(asset_growth: pd.Series) -> pd.Series:
        """
        CMA: Conservative Minus Aggressive（保守投资减激进投资）
        Fama-French 5因子模型
        """
        # 按资产增长率分组
        terciles = asset_growth.quantile([0.3, 0.7])
        
        conservative = asset_growth <= terciles.iloc[0]
        aggressive = asset_growth >= terciles.iloc[1]
        
        inv_factor = pd.Series(0, index=asset_growth.index)
        inv_factor[conservative] = 1
        inv_factor[aggressive] = -1
        
        return inv_factor
    
    @staticmethod
    def calculate_momentum_factor(returns_12m: pd.Series, returns_1m: pd.Series) -> pd.Series:
        """
        MOM: Momentum Factor（动量因子）
        Carhart 4因子模型扩展
        """
        # 12个月动量，跳过最近1个月（避免反转效应）
        momentum = returns_12m - returns_1m
        
        # 标准化
        momentum_normalized = (momentum - momentum.mean()) / momentum.std()
        
        return momentum_normalized


class BarraRiskFactors:
    """
    Barra风险模型因子
    USE4模型的核心因子
    """
    
    @staticmethod
    def calculate_volatility_factor(returns: pd.DataFrame, window: int = 252) -> pd.Series:
        """
        Volatility: 历史波动率因子
        """
        return returns.rolling(window=window).std()
    
    @staticmethod
    def calculate_liquidity_factor(volume: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        Liquidity: 流动性因子（换手率）
        """
        return volume / market_cap
    
    @staticmethod
    def calculate_growth_factor(earnings_growth: pd.Series, sales_growth: pd.Series) -> pd.Series:
        """
        Growth: 成长因子
        """
        return (earnings_growth + sales_growth) / 2
    
    @staticmethod
    def calculate_leverage_factor(debt_to_equity: pd.Series) -> pd.Series:
        """
        Leverage: 杠杆因子
        """
        return debt_to_equity
    
    @staticmethod
    def calculate_earnings_yield_factor(earnings: pd.Series, price: pd.Series) -> pd.Series:
        """
        Earnings Yield: 盈利收益率因子
        """
        return earnings / price
    
    @staticmethod
    def calculate_beta_factor(stock_returns: pd.Series, market_returns: pd.Series, 
                            window: int = 252) -> pd.Series:
        """
        Beta: 市场贝塔因子
        """
        # 滚动计算beta
        covariance = stock_returns.rolling(window).cov(market_returns)
        market_variance = market_returns.rolling(window).var()
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_residual_volatility_factor(returns: pd.Series, predicted_returns: pd.Series,
                                           window: int = 60) -> pd.Series:
        """
        Residual Volatility: 残差波动率（特异性风险）
        """
        residuals = returns - predicted_returns
        return residuals.rolling(window=window).std()
    
    @staticmethod
    def calculate_nonlinear_size_factor(market_cap: pd.Series) -> pd.Series:
        """
        Non-linear Size: 非线性市值因子（市值立方根）
        """
        return np.cbrt(market_cap)


class ProfessionalFactorCalculator:
    """
    专业因子计算器
    整合Fama-French、Barra和自定义因子
    """
    
    def __init__(self):
        self.ff_factors = FamaFrenchFactors()
        self.barra_factors = BarraRiskFactors()
        self.decay_config = FactorDecayConfig()
        self.factor_cache = {}
        
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有专业因子
        """
        factors = pd.DataFrame(index=data.index)
        
        # Fama-French 5因子
        if 'market_cap' in data.columns:
            factors['ff_size'] = self.ff_factors.calculate_size_factor(data['market_cap'])
        
        if 'book_to_market' in data.columns:
            factors['ff_value'] = self.ff_factors.calculate_value_factor(data['book_to_market'])
        
        if 'roe' in data.columns:
            factors['ff_profitability'] = self.ff_factors.calculate_profitability_factor(data['roe'])
        
        if 'asset_growth' in data.columns:
            factors['ff_investment'] = self.ff_factors.calculate_investment_factor(data['asset_growth'])
        
        if 'returns_12m' in data.columns and 'returns_1m' in data.columns:
            factors['ff_momentum'] = self.ff_factors.calculate_momentum_factor(
                data['returns_12m'], data['returns_1m']
            )
        
        # Barra风险因子
        if 'returns' in data.columns:
            factors['barra_volatility'] = self.barra_factors.calculate_volatility_factor(data[['returns']])
        
        if 'volume' in data.columns and 'market_cap' in data.columns:
            factors['barra_liquidity'] = self.barra_factors.calculate_liquidity_factor(
                data['volume'], data['market_cap']
            )
        
        if 'earnings_growth' in data.columns and 'sales_growth' in data.columns:
            factors['barra_growth'] = self.barra_factors.calculate_growth_factor(
                data['earnings_growth'], data['sales_growth']
            )
        
        if 'debt_to_equity' in data.columns:
            factors['barra_leverage'] = self.barra_factors.calculate_leverage_factor(data['debt_to_equity'])
        
        if 'earnings' in data.columns and 'price' in data.columns:
            factors['barra_earnings_yield'] = self.barra_factors.calculate_earnings_yield_factor(
                data['earnings'], data['price']
            )
        
        # 应用因子衰减
        factors = self._apply_factor_decay(factors)
        
        # 标准化因子
        factors = self._standardize_factors(factors)
        
        return factors
    
    def _apply_factor_decay(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        应用因子衰减机制
        """
        decayed_factors = factors.copy()
        
        for col in factors.columns:
            if col in self.decay_config.FACTOR_HALFLIFE:
                # 应用指数衰减
                for i in range(1, len(factors)):
                    decay_weight = self.decay_config.get_decay_weight(col, i)
                    decayed_factors.iloc[i, factors.columns.get_loc(col)] *= decay_weight
        
        return decayed_factors
    
    def _standardize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        标准化因子（z-score）
        """
        return (factors - factors.mean()) / (factors.std() + 1e-8)
    
    def get_factor_correlation_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        """
        return factors.corr()
    
    def get_factor_statistics(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        获取因子统计信息
        """
        stats = pd.DataFrame({
            'mean': factors.mean(),
            'std': factors.std(),
            'skew': factors.skew(),
            'kurt': factors.kurt(),
            'min': factors.min(),
            'max': factors.max(),
            'sharpe': factors.mean() / (factors.std() + 1e-8)
        })
        
        return stats