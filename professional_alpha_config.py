"""
Professional Alpha Configuration Based on Institutional Best Practices
======================================================================
基于顶级量化机构（Two Sigma, Renaissance, Citadel）的配置方案
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class ProfessionalAlphaConfig:
    """专业机构级别的Alpha配置"""
    
    # =========================================================================
    # 1. 分层因子架构 (Hierarchical Factor Structure)
    # =========================================================================
    
    # Tier 1: 核心预测因子 (Core Predictive Factors) - 始终保留
    TIER1_CORE_FACTORS = {
        # 动量类 (Momentum) - 30% weight hint
        'momentum': [
            'residual_momentum_factor',      # 残差动量（剔除市场beta）
            'momentum_quality_jitter',        # 质量调整动量
            'intraday_momentum',             # 日内动量
        ],
        
        # 价值类 (Value) - 25% weight hint  
        'value': [
            'earnings_yield_sector_neutral', # 行业中性化EP
            'book_to_market_robust',         # 稳健BM
            'fcf_yield',                     # 自由现金流收益率
        ],
        
        # 质量类 (Quality) - 20% weight hint
        'quality': [
            'qmj_quality_score',             # Quality Minus Junk
            'earnings_quality',              # 盈利质量（应计项）
            'balance_sheet_health',          # 资产负债表健康度
        ],
        
        # 低风险异象 (Low Risk Anomaly) - 15% weight hint
        'low_risk': [
            'idiosyncratic_volatility',      # 特异性波动率（负向）
            'betting_against_beta',          # BAB因子
        ],
        
        # 流动性 (Liquidity) - 10% weight hint
        'liquidity': [
            'amihud_illiquidity',            # Amihud非流动性
            'bid_ask_spread_adjusted',       # 调整买卖价差
        ]
    }
    
    # Tier 2: 条件因子 (Conditional Factors) - 根据市场状态激活
    TIER2_CONDITIONAL_FACTORS = {
        'bull_market': ['momentum_acceleration', 'earnings_revision'],
        'bear_market': ['quality_defensive', 'low_volatility'],
        'high_volatility': ['mean_reversion_sharp', 'liquidity_provision'],
        'low_volatility': ['carry_trade', 'momentum_smooth'],
    }
    
    # =========================================================================
    # 2. 智能特征工程 (Smart Feature Engineering)
    # =========================================================================
    
    # 使用IC加权而非PCA
    USE_IC_WEIGHTING: bool = True
    IC_LOOKBACK_DAYS: int = 252  # 1年回望计算IC
    IC_DECAY_HALFLIFE: int = 63  # IC权重半衰期（3个月）
    
    # 因子正交化设置
    ORTHOGONALIZATION_METHOD: str = 'sequential'  # 'sequential' or 'symmetric'
    ORTHOGONALIZATION_ORDER: List[str] = [
        'market_beta',      # 首先剔除市场风险
        'size',             # 其次剔除规模效应
        'industry',         # 最后剔除行业偏差
    ]
    
    # =========================================================================
    # 3. 动态权重系统 (Dynamic Weighting System)
    # =========================================================================
    
    # 权重边界
    MIN_FACTOR_WEIGHT: float = 0.02  # 最小2%权重
    MAX_FACTOR_WEIGHT: float = 0.15  # 最大15%权重
    
    # 权重更新频率
    WEIGHT_UPDATE_FREQUENCY: str = 'weekly'  # 每周更新
    WEIGHT_SMOOTHING_WINDOW: int = 4  # 4周平滑
    
    # 基于多个指标的权重调整
    WEIGHT_CRITERIA = {
        'ic': 0.40,           # Information Coefficient
        'ir': 0.30,           # Information Ratio  
        'stability': 0.20,    # 因子稳定性
        'orthogonality': 0.10 # 与其他因子的正交性
    }
    
    # =========================================================================
    # 4. 风险控制 (Risk Controls)
    # =========================================================================
    
    # 因子暴露限制
    MAX_FACTOR_EXPOSURE: float = 3.0  # 3倍标准差
    
    # 相关性控制
    MAX_FACTOR_CORRELATION: float = 0.7  # 因子间最大相关性
    
    # 换手率控制
    TARGET_TURNOVER: float = 2.0  # 日换手率目标2%
    TURNOVER_PENALTY: float = 0.001  # 换手惩罚系数
    
    # =========================================================================
    # 5. 特征数量优化 (Feature Count Optimization)
    # =========================================================================
    
    # 分层特征数量
    N_TIER1_FEATURES: int = 15  # 核心层保留15个
    N_TIER2_FEATURES: int = 5   # 条件层最多5个
    N_INTERACTION_FEATURES: int = 3  # 交互特征3个
    TOTAL_FEATURES: int = 20  # 总特征数控制在20个
    
    # =========================================================================
    # 6. 时间序列增强 (Time Series Enhancement)  
    # =========================================================================
    
    # 多时间尺度特征
    TIME_SCALES = {
        'micro': 1,     # 1天（捕捉微观结构）
        'short': 5,     # 1周（短期动量）
        'medium': 22,   # 1月（中期趋势）
        'long': 66,     # 3月（长期均值回归）
    }
    
    # 因子衰减设置
    FACTOR_DECAY_HALFLIFE = {
        'news_sentiment': 2,      # 新闻情绪快速衰减
        'momentum': 22,           # 动量中速衰减
        'value': 66,              # 价值慢速衰减
        'quality': 132,           # 质量因子最稳定
    }
    
    # =========================================================================
    # 7. 机器学习集成 (ML Integration)
    # =========================================================================
    
    # 特征重要性阈值
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.01  # 重要性<1%的特征剔除
    
    # 非线性交互
    INCLUDE_INTERACTIONS: bool = True
    MAX_INTERACTION_DEPTH: int = 2  # 最多2阶交互
    TOP_INTERACTIONS: int = 5  # 保留top 5交互项
    
    # 集成学习设置
    ENSEMBLE_METHODS = {
        'linear': 0.3,      # 线性模型（LASSO）
        'tree': 0.4,        # 树模型（LightGBM）  
        'neural': 0.3,      # 神经网络（浅层）
    }
    
    # =========================================================================
    # 8. 实时监控指标 (Real-time Monitoring)
    # =========================================================================
    
    MONITORING_METRICS = {
        'realized_ic': {'threshold': 0.02, 'window': 20},
        'factor_decay': {'threshold': 0.5, 'window': 60},
        'regime_shift': {'threshold': 2.0, 'window': 20},
        'factor_crowding': {'threshold': 0.8, 'window': 10},
    }
    
    def get_active_features(self, market_regime: str) -> List[str]:
        """根据市场状态返回活跃特征集"""
        # 始终包含Tier 1
        features = []
        for category_factors in self.TIER1_CORE_FACTORS.values():
            features.extend(category_factors)
        
        # 条件性添加Tier 2
        if market_regime in self.TIER2_CONDITIONAL_FACTORS:
            features.extend(self.TIER2_CONDITIONAL_FACTORS[market_regime])
        
        # 限制总数
        return features[:self.TOTAL_FEATURES]
    
    def calculate_ic_weights(self, 
                            historical_returns: np.ndarray,
                            factor_values: np.ndarray,
                            lookback: int = 252) -> np.ndarray:
        """计算基于IC的动态权重"""
        # 计算滚动IC
        ic_scores = []
        for i in range(factor_values.shape[1]):
            # 计算因子i与未来收益的相关性
            ic = np.corrcoef(factor_values[-lookback:, i], 
                            historical_returns[-lookback:])[0, 1]
            ic_scores.append(ic)
        
        # 转换为权重（处理负IC）
        ic_scores = np.array(ic_scores)
        ic_scores = np.maximum(ic_scores, 0)  # 只保留正IC
        
        # 指数衰减加权
        decay_weights = np.exp(-np.arange(len(ic_scores)) / self.IC_DECAY_HALFLIFE)
        ic_scores *= decay_weights[:len(ic_scores)]
        
        # 归一化为权重
        if ic_scores.sum() > 0:
            weights = ic_scores / ic_scores.sum()
        else:
            weights = np.ones(len(ic_scores)) / len(ic_scores)
        
        # 应用权重限制
        weights = np.clip(weights, self.MIN_FACTOR_WEIGHT, self.MAX_FACTOR_WEIGHT)
        weights /= weights.sum()
        
        return weights
    
    def detect_factor_crowding(self, factor_exposures: np.ndarray) -> float:
        """检测因子拥挤度"""
        # 计算因子暴露的横截面离散度
        cross_sectional_std = np.std(factor_exposures, axis=0)
        
        # 计算历史分位数
        historical_percentile = np.percentile(cross_sectional_std, [25, 75])
        
        # 当前离散度相对历史的位置
        current_dispersion = np.mean(cross_sectional_std)
        crowding_score = (current_dispersion - historical_percentile[0]) / \
                        (historical_percentile[1] - historical_percentile[0])
        
        return np.clip(crowding_score, 0, 1)


# =========================================================================
# 专业实施建议
# =========================================================================

IMPLEMENTATION_GUIDELINES = """
1. 数据质量控制
   - 使用Point-in-Time数据避免前视偏差
   - 实施Survivorship Bias Free的universe
   - 每日进行数据完整性检查

2. 因子构建流程
   - Raw Signal → Winsorization → Standardization → Neutralization → Combination
   - 使用MAD (Median Absolute Deviation)进行稳健标准化
   - 行业中性化使用GICS 3级分类

3. 回测要求
   - 最少5年样本外测试
   - 考虑交易成本（买卖价差+市场冲击+手续费）
   - 实施portfolio rebalancing constraints

4. 生产部署
   - 实时因子计算延迟 < 100ms
   - 因子值缓存和增量更新
   - 异常检测和自动降级机制

5. 风险管理
   - 实时监控因子暴露
   - 设置止损和风险限额
   - 定期进行因子有效性评估
"""