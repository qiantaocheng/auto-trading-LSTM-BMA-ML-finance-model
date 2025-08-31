"""
因子滞后配置 - 统一T-1简化配置
统一规则：
- 所有因子类型：T-1（日终决策、次日执行）
- 简化时序管理，避免复杂性
- 确保数据获取现实性和模型一致性

统一滞后 k=1 → 隔离设置 gap/embargo=11天（T+10最小安全间隔）
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class FactorCategory(Enum):
    """因子类别枚举 - 统一T-1"""
    PRICE_TECHNICAL = "price_technical"  # T-1
    EVENT_SENTIMENT = "event_sentiment"  # T-1 (简化)
    FUNDAMENTAL = "fundamental"          # T-1 (简化)
    
@dataclass
class FactorLagConfig:
    """因子滞后配置"""
    name: str
    category: FactorCategory
    lag_days: int
    description: str

# 完整因子滞后映射表 - 统一T-1配置
FACTOR_LAG_MAPPING: Dict[str, int] = {
    # ========== 动量类（6个）T-1 ==========
    "momentum_d22_d66": 1,
    "momentum_hump_0003": 1,
    "momentum_hump_0008": 1,
    "momentum_6_1_enhanced": 1,
    "residual_momentum_factor": 1,
    "sentiment_momentum_factor": 1,  # 统一为T-1
    
    # ========== 反转类（2个）T-1 ==========
    "reversal_d5_d22": 1,
    "reversal_5d_enhanced": 1,
    
    # ========== 波动率类（2个）T-1 ==========
    "volatility_d22_d66": 1,
    "idiosyncratic_volatility_factor": 1,
    
    # ========== 流动性类（4个）T-1 ==========
    "volume_turnover_d22": 1,
    "amihud_illiquidity": 1,  # 统一为T-1
    "bid_ask_spread": 1,
    "amihud_illiq_enhanced": 1,
    
    # ========== 风险类（2个）T-1 ==========
    "low_beta_anomaly_factor": 1,
    "new_high_proximity_factor": 1,
    
    # ========== 价值类（8个）T-1 ==========
    "ebit_ev_yield": 1,  # 统一为T-1
    "fcf_ev_yield": 1,
    "earnings_yield_ep": 1,
    "sales_yield_sp": 1,
    "cash_yield_factor": 1,
    "shareholder_yield_factor": 1,
    "earnings_surprise_sue": 1,  # 统一为T-1
    "pead_earnings_drift": 1,
    
    # ========== 盈利能力类（6个）T-1 ==========
    "gross_margin_factor": 1,  # 统一为T-1
    "operating_profitability_factor": 1,
    "roe_neutralized_factor": 1,
    "roic_neutralized_factor": 1,
    "net_margin_factor": 1,
    "earnings_stability_factor": 1,
    
    # ========== 质量类（10个）T-1 ==========
    "total_accruals_negative": 1,  # 统一为T-1
    "working_capital_accruals_negative": 1,
    "net_operating_assets_negative": 1,
    "asset_growth_negative": 1,
    "net_equity_issuance_factor": 1,
    "investment_factor_capex": 1,
    "piotroski_f_score": 1,
    "ohlson_o_score": 1,
    "altman_z_score": 1,
    "qmj_quality_score": 1,
    
    # ========== 分析师类（1个）T-1 ==========
    "analyst_revision_factor": 1,  # 统一为T-1
    
    # ========== 情绪类（4个）T-1 ==========
    "news_sentiment_factor": 1,
    "market_sentiment_factor": 1,
    "fear_greed_sentiment_factor": 1,
    
    # ========== 技术指标类（11个）T-1 ==========
    "sma_10_technical_indicator": 1,
    "sma_20_technical_indicator": 1,
    "sma_50_technical_indicator": 1,
    "rsi_technical_indicator": 1,
    "bollinger_bands_position": 1,
    "macd_line_indicator": 1,
    "macd_signal_line": 1,
    "macd_histogram_indicator": 1,
    "price_momentum_5d_technical": 1,
    "price_momentum_20d_technical": 1,
    "volume_ratio_technical": 1,
    
    # ========== 风险指标类（3个）T-1 ==========
    "maximum_drawdown_risk": 1,
    "sharpe_ratio_risk_metric": 1,
    "value_at_risk_95_percentile": 1,
    
    # ========== 基础特征（4个）T-1 ==========
    "returns": 1,
    "volatility": 1,
    "rsi": 1,
    "sma_ratio": 1,
}

# 因子类别分组（用于批量调整）
FACTOR_CATEGORIES = {
    "price_technical": [
        "momentum_d22_d66", "momentum_hump_0003", "momentum_hump_0008",
        "momentum_6_1_enhanced", "residual_momentum_factor",
        "reversal_d5_d22", "reversal_5d_enhanced",
        "volatility_d22_d66", "idiosyncratic_volatility_factor",
        "volume_turnover_d22", "low_beta_anomaly_factor", "new_high_proximity_factor",
        "sma_10_technical_indicator", "sma_20_technical_indicator", "sma_50_technical_indicator",
        "rsi_technical_indicator", "bollinger_bands_position",
        "macd_line_indicator", "macd_signal_line", "macd_histogram_indicator",
        "price_momentum_5d_technical", "price_momentum_20d_technical",
        "volume_ratio_technical", "maximum_drawdown_risk",
        "sharpe_ratio_risk_metric", "value_at_risk_95_percentile",
        "returns", "volatility", "rsi", "sma_ratio",
        "market_sentiment_factor", "fear_greed_sentiment_factor"
    ],
    "event_sentiment": [
        "sentiment_momentum_factor", "amihud_illiquidity", "bid_ask_spread",
        "amihud_illiq_enhanced", "earnings_surprise_sue", "pead_earnings_drift",
        "analyst_revision_factor", "news_sentiment_factor"
    ],
    "fundamental": [
        "ebit_ev_yield", "fcf_ev_yield", "earnings_yield_ep", "sales_yield_sp",
        "cash_yield_factor", "shareholder_yield_factor",
        "gross_margin_factor", "operating_profitability_factor",
        "roe_neutralized_factor", "roic_neutralized_factor",
        "net_margin_factor", "earnings_stability_factor",
        "total_accruals_negative", "working_capital_accruals_negative",
        "net_operating_assets_negative", "asset_growth_negative",
        "net_equity_issuance_factor", "investment_factor_capex",
        "piotroski_f_score", "ohlson_o_score", "altman_z_score", "qmj_quality_score"
    ]
}

class FactorLagManager:
    """因子滞后管理器"""
    
    def __init__(self):
        self.lag_mapping = FACTOR_LAG_MAPPING
        self.categories = FACTOR_CATEGORIES
        self.max_lag = max(FACTOR_LAG_MAPPING.values())
        
    def get_lag(self, factor_name: str) -> int:
        """获取因子滞后天数"""
        return self.lag_mapping.get(factor_name, 1)  # 默认T-1
    
    def get_category_factors(self, category: str) -> List[str]:
        """获取某类别的所有因子"""
        return self.categories.get(category, [])
    
    def get_max_lag(self) -> int:
        """获取最大滞后天数"""
        return self.max_lag
    
    def get_required_gap(self, prediction_horizon: int = 10) -> int:
        """计算所需的CV gap/embargo - 与unified_timing_registry保持一致"""
        # CRITICAL FIX: 统一为11天，T+10最小安全间隔
        # 计算逻辑：T-1特征 -> T+10预测，最小安全gap = 10 + 1 = 11天
        return 11  # 统一11天gap，确保与其他系统组件完全一致
    
    def validate_time_alignment(self, feature_time: int, target_time: int) -> bool:
        """验证时间对齐是否安全"""
        # feature_time: T-k (负数)
        # target_time: T+h (正数)
        total_gap = target_time - feature_time
        required_gap = self.get_required_gap()
        return total_gap >= required_gap
    
    def get_factor_info(self, factor_name: str) -> Dict:
        """获取因子详细信息"""
        lag = self.get_lag(factor_name)
        
        # 判断因子类别
        category = None
        for cat, factors in self.categories.items():
            if factor_name in factors:
                category = cat
                break
                
        return {
            "name": factor_name,
            "lag_days": lag,
            "category": category,
            "description": self._get_lag_description(lag)
        }
    
    def _get_lag_description(self, lag: int) -> str:
        """获取滞后描述 - 统一T-1"""
        descriptions = {
            1: "T-1: 统一滞后，日终决策次日执行"
        }
        return descriptions.get(lag, f"T-{lag}: 非标准滞后")
    
    def print_summary(self):
        """打印配置摘要"""
        print("="*60)
        print("因子滞后配置摘要")
        print("="*60)
        
        for category, factors in self.categories.items():
            print(f"\n{category.upper()}类因子:")
            lag_counts = {}
            for factor in factors:
                lag = self.get_lag(factor)
                lag_counts[lag] = lag_counts.get(lag, 0) + 1
            
            for lag, count in sorted(lag_counts.items()):
                print(f"  T-{lag}: {count}个因子")
        
        print(f"\n最大滞后: T-{self.max_lag}")
        print(f"建议CV隔离: gap/embargo = {self.get_required_gap()}天")
        print("="*60)

# 全局实例
factor_lag_manager = FactorLagManager()

if __name__ == "__main__":
    # 测试和验证
    manager = FactorLagManager()
    manager.print_summary()
    
    # 测试几个因子
    test_factors = [
        "momentum_d22_d66",
        "earnings_surprise_sue",
        "piotroski_f_score"
    ]
    
    print("\n示例因子信息:")
    for factor in test_factors:
        info = manager.get_factor_info(factor)
        print(f"  {info['name']}: {info['description']}")