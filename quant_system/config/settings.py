"""
Quantitative Signal System Configuration
========================================

All thresholds are parameterized for systematic optimization.
Default values derived from academic literature and practitioner guidelines.

References:
- Minervini, M. (2013). Trade Like a Stock Market Wizard
- Weinstein, S. (1988). Secrets for Profiting in Bull and Bear Markets
- O'Neil, W. (2009). How to Make Money in Stocks
- Jegadeesh & Titman (1993). Returns to Buying Winners and Selling Losers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification for adaptive signal generation."""
    BULL_STRONG = "bull_strong"      # >80% stocks above 200MA
    BULL_WEAK = "bull_weak"          # 50-80% stocks above 200MA
    NEUTRAL = "neutral"              # 30-50% stocks above 200MA
    BEAR_WEAK = "bear_weak"          # 20-30% stocks above 200MA
    BEAR_STRONG = "bear_strong"      # <20% stocks above 200MA


@dataclass
class TrendTemplateConfig:
    """
    Minervini Trend Template Parameters
    
    Justification:
    - MA periods: Industry standard institutional monitoring windows
    - 52-week thresholds: Empirically validated in "Trade Like a Stock Market Wizard"
    - Slope lookback: ~1 month of trading days for statistical significance
    """
    ma_short: int = 50           # Short-term MA (institutional trading window)
    ma_medium: int = 150         # Medium-term MA (~6 months)
    ma_long: int = 200           # Long-term MA (~10 months, global institutional benchmark)
    
    # 52-week price position thresholds
    min_above_52w_low: float = 0.30    # Must be 30% above 52-week low
    min_near_52w_high: float = 0.75    # Must be within 25% of 52-week high
    
    # Slope calculation
    slope_lookback: int = 20           # Days for slope calculation (~1 month)
    min_slope_months: int = 1          # Minimum months of positive slope
    
    # Scoring weights (sum to 1.0)
    weight_ma_alignment: float = 0.25
    weight_price_position: float = 0.20
    weight_slope: float = 0.20
    weight_acceleration: float = 0.15
    weight_ma_distance: float = 0.20


@dataclass
class StageAnalysisConfig:
    """
    Weinstein Stage Analysis Parameters (Weekly timeframe)
    
    Justification:
    - 30-week MA: Standard in Weinstein methodology (~150 daily)
    - Volume multiplier: 2x ensures institutional participation
    - RS lookback: 52 weeks for annual comparison
    """
    ma_period_weeks: int = 30          # Weekly MA for stage determination
    volume_breakout_multiplier: float = 2.0  # Min volume increase on breakout
    volume_lookback_weeks: int = 4     # Weeks for average volume calculation
    
    # Mansfield RS calculation
    rs_ma_period: int = 52             # Weeks for RS smoothing
    rs_positive_threshold: float = 0.0  # RS must be above zero
    
    # Base detection
    min_base_weeks: int = 7            # Minimum consolidation period
    max_base_weeks: int = 65           # Maximum healthy base duration
    price_contraction_threshold: float = 0.15  # Max price range during base


@dataclass
class VCPConfig:
    """
    Volatility Contraction Pattern Detection Parameters
    
    Justification:
    - Contraction ratio: 0.5-0.8 range from Minervini's documented patterns
    - Min contractions: VCP requires at least 2 visible contractions
    - Volume dry-up: Supply exhaustion signal
    """
    min_contractions: int = 2          # Minimum number of contractions
    max_contractions: int = 6          # Maximum healthy contractions
    
    # Contraction depth rules
    max_first_contraction: float = 0.35   # Max depth of first pullback (35%)
    contraction_decay_ratio: float = 0.65  # Each contraction ≤ 65% of previous
    final_contraction_max: float = 0.10   # Final squeeze typically <10%
    
    # Volume characteristics
    volume_dry_up_threshold: float = 0.70  # Volume < 70% of 50-day avg
    volume_dry_up_days: int = 3           # Consecutive days of low volume
    
    # Pivot point detection
    pivot_breakout_volume_multiplier: float = 1.5
    pivot_lookback_days: int = 10
    
    # Bollinger Band squeeze detection
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_percentile: float = 0.10   # Bandwidth in bottom 10%


@dataclass
class RelativeStrengthConfig:
    """
    Relative Strength Calculation Parameters
    
    Justification:
    - IBD methodology weights recent performance more heavily
    - 12-month total window captures full market cycle
    - RS >= 80 threshold from O'Neil's research on superperformers
    """
    total_lookback_days: int = 252     # 12 months of trading days
    
    # Time-weighted momentum (IBD-style)
    period_weights: Dict[str, float] = field(default_factory=lambda: {
        'quarter_1': 0.40,   # Most recent quarter (double weighted)
        'quarter_2': 0.20,   # Second quarter
        'quarter_3': 0.20,   # Third quarter
        'quarter_4': 0.20,   # Fourth quarter (oldest)
    })
    
    # Percentile thresholds
    min_rs_rating: int = 80            # Minimum RS rating (0-99 scale)
    elite_rs_rating: int = 90          # Elite performers threshold
    
    # Divergence detection
    divergence_lookback_days: int = 20  # Days to check for divergence
    divergence_index_threshold: float = -0.02  # Index down 2%


@dataclass
class ADXConfig:
    """
    Average Directional Index Parameters
    
    Justification:
    - ADX > 25: Academic consensus for trend existence
    - ADX > 50: Overheated trend warning
    - DMI confirmation prevents false signals
    """
    period: int = 14                   # Standard ADX period
    
    trend_threshold: int = 25          # Minimum for trending market
    strong_trend_threshold: int = 40   # Strong trend
    extreme_trend_threshold: int = 50  # Overheated, high reversion risk
    
    # DMI requirements
    require_di_positive: bool = True   # +DI must be above -DI for longs


@dataclass
class FundamentalConfig:
    """
    CAN SLIM Fundamental Parameters
    
    Justification:
    - 25% EPS growth: O'Neil's threshold for growth stocks
    - 17% ROE: Distinguishes quality compounders
    - Institutional increase: Smart money validation
    """
    # Earnings (C & A)
    min_quarterly_eps_growth: float = 0.25   # 25% YoY growth
    min_annual_eps_cagr: float = 0.25        # 3-year CAGR
    min_roe: float = 0.17                     # 17% return on equity
    
    # Earnings acceleration bonus
    acceleration_bonus: float = 0.10         # Extra score for accelerating growth
    
    # Institutional sponsorship (I)
    min_institutional_increase: int = 1      # Quarter-over-quarter increase
    elite_fund_bonus: float = 0.05           # Bonus for top-tier fund ownership
    
    # PEG ratio
    peg_undervalued: float = 1.0
    peg_fair_value: float = 2.0


@dataclass
class RiskManagementConfig:
    """
    Risk Management and Position Sizing Parameters
    
    Justification:
    - ATR multiplier: 2.5-3.0x allows normal volatility while limiting drawdown
    - Kelly fraction: Conservative 1/4 Kelly for robustness
    - Max position: Concentration limits for diversification
    """
    # ATR-based stops
    atr_period: int = 14
    chandelier_multiplier: float = 3.0    # ATR multiplier for trailing stop
    tight_stop_multiplier: float = 2.0    # Tighter stop for new entries
    
    # Position sizing
    max_position_pct: float = 0.10        # Max 10% per position
    target_portfolio_volatility: float = 0.15  # 15% annual vol target
    kelly_fraction: float = 0.25          # Quarter-Kelly for conservatism
    
    # Drawdown limits
    max_stock_drawdown: float = 0.20      # Max 20% loss per position
    max_portfolio_drawdown: float = 0.15  # Max 15% portfolio drawdown
    
    # Pullback vs reversal classification
    pullback_max_volume_ratio: float = 0.80  # Volume < 80% of up-day avg
    pullback_max_days: int = 5              # Max days below support


@dataclass
class UniverseConfig:
    """
    Stock Universe Definition
    
    Justification:
    - Price filter: Eliminates penny stock noise
    - Liquidity filter: Ensures executable ideas
    - Market cap: Focus on investable names
    """
    min_price: float = 10.0              # Minimum stock price
    min_avg_volume: int = 500_000        # Minimum daily shares traded
    min_avg_dollar_volume: float = 5_000_000  # Minimum daily $ volume
    
    min_market_cap: float = 500_000_000  # $500M minimum market cap
    max_market_cap: Optional[float] = None  # No upper limit by default
    
    # Data requirements
    min_trading_days: int = 252          # At least 1 year of data
    exclude_otc: bool = True             # Exclude OTC stocks
    exclude_adrs: bool = False           # Include ADRs by default


@dataclass
class SignalAggregationConfig:
    """
    Signal Combination and Scoring Parameters
    
    Note: Weights should sum to 1.0 for each category.
    """
    # Category weights
    technical_weight: float = 0.45       # Trend, ADX, RS
    pattern_weight: float = 0.25         # VCP, Stage
    fundamental_weight: float = 0.30     # CAN SLIM factors
    
    # Sub-component weights within technical
    trend_template_weight: float = 0.40
    adx_weight: float = 0.25
    relative_strength_weight: float = 0.35
    
    # Sub-component weights within pattern
    vcp_weight: float = 0.50
    stage_weight: float = 0.50
    
    # Minimum scores for inclusion
    min_composite_score: float = 0.60    # Minimum to be considered
    min_technical_score: float = 0.50    # Must pass technical bar
    
    # Regime adjustments (multiply base weights)
    regime_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULL_STRONG.value: {'technical': 1.0, 'pattern': 1.0, 'fundamental': 1.0},
        MarketRegime.BULL_WEAK.value: {'technical': 1.1, 'pattern': 1.0, 'fundamental': 0.9},
        MarketRegime.NEUTRAL.value: {'technical': 1.2, 'pattern': 1.1, 'fundamental': 0.7},
        MarketRegime.BEAR_WEAK.value: {'technical': 1.3, 'pattern': 0.8, 'fundamental': 0.5},
        MarketRegime.BEAR_STRONG.value: {'technical': 0.5, 'pattern': 0.3, 'fundamental': 0.2},  # Cash preferred
    })


@dataclass
class BacktestConfig:
    """Backtesting Engine Configuration"""
    initial_capital: float = 1_000_000
    commission_per_share: float = 0.005
    slippage_pct: float = 0.001          # 0.1% slippage assumption
    
    # Rebalancing
    rebalance_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    max_positions: int = 20              # Maximum concurrent positions
    
    # Performance metrics
    risk_free_rate: float = 0.05         # 5% annual risk-free rate
    benchmark_ticker: str = 'SPY'        # Benchmark for alpha calculation


@dataclass
class SystemConfig:
    """Master Configuration Container"""
    trend_template: TrendTemplateConfig = field(default_factory=TrendTemplateConfig)
    stage_analysis: StageAnalysisConfig = field(default_factory=StageAnalysisConfig)
    vcp: VCPConfig = field(default_factory=VCPConfig)
    relative_strength: RelativeStrengthConfig = field(default_factory=RelativeStrengthConfig)
    adx: ADXConfig = field(default_factory=ADXConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    risk: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    aggregation: SignalAggregationConfig = field(default_factory=SignalAggregationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
