"""
Alpha预测性提升实现方案
========================
将当前PCA+Composite方案升级为专业IC加权方案
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class EnhancedAlphaProcessor:
    """增强型Alpha处理器 - 提升预测性"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.ic_history = {}
        self.weight_history = []
        self.regime_detector = MarketRegimeDetector()
        
    def _get_default_config(self):
        """默认配置"""
        return {
            'lookback_days': 252,
            'ic_decay_halflife': 63,
            'min_ic_threshold': 0.01,
            'n_top_factors': 15,
            'use_interaction_terms': True,
        }
    
    def process_alpha_factors(self, 
                             alpha_df: pd.DataFrame,
                             returns_df: pd.DataFrame,
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """
        主处理流程 - 替代原有的PCA方案
        """
        # Step 1: 市场状态检测
        market_regime = self.regime_detector.detect_regime(market_data)
        logger.info(f"当前市场状态: {market_regime}")
        
        # Step 2: 计算因子IC并筛选
        top_factors = self._select_factors_by_ic(alpha_df, returns_df)
        
        # Step 3: 构建分层特征
        features = self._build_hierarchical_features(
            alpha_df[top_factors], 
            market_regime
        )
        
        # Step 4: 添加交互项（如果配置）
        if self.config['use_interaction_terms']:
            interaction_features = self._create_interaction_features(features)
            features = pd.concat([features, interaction_features], axis=1)
        
        # Step 5: 动态加权组合
        final_features = self._apply_dynamic_weighting(features, returns_df)
        
        return final_features
    
    def _select_factors_by_ic(self, 
                              alpha_df: pd.DataFrame, 
                              returns_df: pd.DataFrame) -> List[str]:
        """
        基于IC选择顶级因子
        
        关键改进：
        1. 使用Rank IC而非Pearson IC（更稳健）
        2. 考虑IC的稳定性，而非只看绝对值
        3. 剔除高相关因子避免冗余
        """
        ic_scores = {}
        ic_stability = {}
        
        for factor in alpha_df.columns:
            if factor in ['date', 'ticker']:
                continue
                
            # 计算滚动Rank IC
            rolling_ic = []
            for i in range(self.config['lookback_days'], len(alpha_df)):
                window_data = alpha_df[factor].iloc[i-self.config['lookback_days']:i]
                window_returns = returns_df.iloc[i-self.config['lookback_days']:i]
                
                # Rank IC (Spearman correlation)
                if len(window_data) > 10 and window_data.std() > 0:
                    rank_ic = window_data.corr(window_returns, method='spearman')
                    rolling_ic.append(rank_ic)
            
            if rolling_ic:
                # IC均值和稳定性
                ic_scores[factor] = np.mean(rolling_ic)
                ic_stability[factor] = np.mean(rolling_ic) / (np.std(rolling_ic) + 1e-6)
        
        # 综合评分 = 0.6 * IC + 0.4 * IC稳定性
        combined_scores = {}
        for factor in ic_scores:
            combined_scores[factor] = (
                0.6 * abs(ic_scores[factor]) + 
                0.4 * ic_stability[factor]
            )
        
        # 选择top N因子
        sorted_factors = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_factors = [f for f, _ in sorted_factors[:self.config['n_top_factors']]]
        
        # 剔除高相关因子
        top_factors = self._remove_correlated_factors(alpha_df[top_factors])
        
        logger.info(f"选择 {len(top_factors)} 个高IC因子")
        for factor in top_factors[:5]:
            logger.info(f"  {factor}: IC={ic_scores.get(factor, 0):.3f}")
        
        return top_factors
    
    def _remove_correlated_factors(self, 
                                   factor_df: pd.DataFrame, 
                                   threshold: float = 0.85) -> List[str]:
        """剔除高度相关的因子"""
        corr_matrix = factor_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找到相关性超过阈值的因子对
        to_drop = []
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            correlated = list(upper_tri.index[upper_tri[column] > threshold])
            to_drop.extend(correlated)
        
        keep_factors = [f for f in factor_df.columns if f not in to_drop]
        return keep_factors
    
    def _build_hierarchical_features(self, 
                                    factor_df: pd.DataFrame,
                                    market_regime: str) -> pd.DataFrame:
        """
        构建分层特征体系
        
        核心思想：
        1. 不同市场环境激活不同因子组合
        2. 因子按类别聚合（动量、价值、质量等）
        3. 类内使用IC加权，类间使用固定权重
        """
        features = pd.DataFrame(index=factor_df.index)
        
        # 因子分类（根据名称模式）
        factor_categories = self._categorize_factors(factor_df.columns)
        
        # 每个类别内部IC加权
        for category, factors in factor_categories.items():
            if not factors:
                continue
                
            category_data = factor_df[factors]
            
            # 计算类内权重
            weights = self._calculate_category_weights(
                category_data, 
                category, 
                market_regime
            )
            
            # 加权组合
            features[f'{category}_composite'] = (category_data * weights).sum(axis=1)
        
        return features
    
    def _categorize_factors(self, factor_names: List[str]) -> Dict[str, List[str]]:
        """因子分类"""
        categories = {
            'momentum': [],
            'value': [],
            'quality': [],
            'volatility': [],
            'liquidity': [],
            'sentiment': [],
            'technical': []
        }
        
        for factor in factor_names:
            factor_lower = factor.lower()
            
            if any(m in factor_lower for m in ['momentum', 'reversal', 'trend']):
                categories['momentum'].append(factor)
            elif any(v in factor_lower for v in ['value', 'earnings', 'book', 'pe', 'pb']):
                categories['value'].append(factor)
            elif any(q in factor_lower for q in ['quality', 'roe', 'roa', 'margin', 'score']):
                categories['quality'].append(factor)
            elif any(vol in factor_lower for vol in ['volatility', 'vol', 'std', 'variance']):
                categories['volatility'].append(factor)
            elif any(liq in factor_lower for liq in ['liquidity', 'volume', 'spread', 'amihud']):
                categories['liquidity'].append(factor)
            elif any(sent in factor_lower for sent in ['sentiment', 'news', 'fear', 'greed']):
                categories['sentiment'].append(factor)
            else:
                categories['technical'].append(factor)
        
        return categories
    
    def _calculate_category_weights(self, 
                                   category_data: pd.DataFrame,
                                   category: str,
                                   market_regime: str) -> np.ndarray:
        """
        计算类别内因子权重
        
        根据市场状态调整权重策略：
        - 牛市：偏重动量
        - 熊市：偏重质量和低风险
        - 高波动：偏重均值回归
        - 低波动：均衡配置
        """
        n_factors = len(category_data.columns)
        
        # 基础等权重
        base_weights = np.ones(n_factors) / n_factors
        
        # 根据市场状态调整
        regime_adjustments = {
            'bull': {'momentum': 1.5, 'quality': 0.8},
            'bear': {'quality': 1.3, 'value': 1.2, 'momentum': 0.7},
            'high_vol': {'volatility': 0.6, 'liquidity': 1.3},
            'low_vol': {}  # 不调整
        }
        
        adjustment = regime_adjustments.get(market_regime, {}).get(category, 1.0)
        weights = base_weights * adjustment
        
        # 归一化
        weights = weights / weights.sum()
        
        return weights
    
    def _create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        创建非线性交互特征
        
        重要交互：
        1. 动量 × 质量 = 高质量动量
        2. 价值 × 质量 = 高质量价值
        3. 流动性 × 波动率 = 流动性风险
        """
        interactions = pd.DataFrame(index=features.index)
        
        # 预定义的重要交互
        important_interactions = [
            ('momentum_composite', 'quality_composite', 'quality_momentum'),
            ('value_composite', 'quality_composite', 'quality_value'),
            ('liquidity_composite', 'volatility_composite', 'liquidity_risk'),
        ]
        
        for feat1, feat2, name in important_interactions:
            if feat1 in features.columns and feat2 in features.columns:
                # 标准化后相乘
                f1_std = (features[feat1] - features[feat1].mean()) / (features[feat1].std() + 1e-6)
                f2_std = (features[feat2] - features[feat2].mean()) / (features[feat2].std() + 1e-6)
                interactions[name] = f1_std * f2_std
        
        return interactions
    
    def _apply_dynamic_weighting(self, 
                                features: pd.DataFrame,
                                returns_df: pd.DataFrame) -> pd.DataFrame:
        """应用动态权重"""
        # 计算每个特征的近期IC
        recent_ic = {}
        for col in features.columns:
            ic = features[col].rolling(20).corr(returns_df)
            recent_ic[col] = ic.iloc[-1] if not ic.empty else 0
        
        # 转换为权重
        weights = pd.Series(recent_ic)
        weights = weights.clip(lower=0)  # 只保留正IC
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1/len(features.columns), index=features.columns)
        
        # 应用权重
        weighted_features = features * weights
        
        return weighted_features


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """
        检测当前市场状态
        
        Returns:
            'bull', 'bear', 'high_vol', 'low_vol'
        """
        if market_data.empty:
            return 'low_vol'
        
        # 简单实现：基于收益率和波动率
        returns = market_data['returns'] if 'returns' in market_data else market_data.pct_change()
        
        # 20日收益率
        recent_return = returns.tail(20).mean()
        
        # 20日实现波动率
        recent_vol = returns.tail(20).std()
        
        # 历史分位数
        historical_vol_percentile = recent_vol > returns.std()
        
        # 分类逻辑
        if recent_return > 0.001:  # 日均收益>0.1%
            return 'bull' if not historical_vol_percentile else 'high_vol'
        else:
            return 'bear' if not historical_vol_percentile else 'high_vol'


# ===============================================================================
# 与现有系统的集成接口
# ===============================================================================

def upgrade_alpha_processing():
    """
    升级现有的alpha_summary_features.py
    用IC加权替代PCA方案
    """
    code = '''
    # 在 alpha_summary_features.py 中替换 _compress_alpha_dimensions 方法
    
    def _compress_alpha_dimensions(self, alpha_values: pd.DataFrame, 
                                   returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """使用IC加权替代PCA"""
        
        # 使用新的增强处理器
        processor = EnhancedAlphaProcessor()
        
        # 如果没有returns，使用模拟数据（实际应该传入真实returns）
        if returns is None:
            returns = pd.Series(np.zeros(len(alpha_values)), index=alpha_values.index)
        
        # 获取市场数据（需要从其他地方获取）
        market_data = pd.DataFrame()  # 实际应该传入市场数据
        
        # 处理alpha因子
        enhanced_features = processor.process_alpha_factors(
            alpha_values, 
            returns,
            market_data
        )
        
        return enhanced_features
    '''
    
    return code