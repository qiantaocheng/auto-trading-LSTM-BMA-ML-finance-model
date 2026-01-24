"""
Risk Management Module
=====================

Implements:
1. ATR-based dynamic stops (Chandelier Exit)
2. Position sizing (volatility-adjusted, Kelly criterion)
3. Pullback vs reversal classification
4. Portfolio-level risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/home/claude/quant_system')

from core.data_types import StockData
from config.settings import RiskManagementConfig


class DrawdownState(Enum):
    """Current drawdown severity."""
    NORMAL = "normal"           # < 5% drawdown
    ELEVATED = "elevated"       # 5-10% drawdown
    HIGH = "high"              # 10-15% drawdown
    CRITICAL = "critical"       # > 15% drawdown


@dataclass
class StopLossResult:
    """Stop loss calculation result."""
    stop_price: float
    stop_type: str  # 'atr', 'percentage', 'support'
    risk_per_share: float
    risk_percentage: float
    atr_value: float
    atr_multiplier: float


@dataclass
class PositionSizeResult:
    """Position sizing calculation result."""
    shares: int
    position_value: float
    position_pct: float  # % of portfolio
    risk_amount: float  # $ at risk
    risk_pct: float  # % of portfolio at risk
    sizing_method: str


@dataclass
class PullbackClassification:
    """Classification of price decline."""
    is_pullback: bool  # True = healthy pullback, False = potential reversal
    confidence: float
    volume_pattern: str  # 'contracting', 'expanding', 'neutral'
    support_holding: bool
    days_below_ma: int
    max_depth_pct: float
    recommendation: str  # 'hold', 'add', 'reduce', 'exit'


class ATRCalculator:
    """Calculate Average True Range with various smoothing methods."""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14,
                     method: str = 'ema') -> pd.Series:
        """
        Calculate ATR.
        
        Args:
            df: OHLC DataFrame
            period: ATR period
            method: 'ema' (Wilder's), 'sma', or 'rma'
            
        Returns:
            ATR series
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothing
        if method == 'ema':
            # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
        elif method == 'sma':
            atr = tr.rolling(window=period).mean()
        else:  # rma
            atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr


class StopLossManager:
    """
    Manage stop loss calculations.
    
    Supports:
    - ATR-based trailing stops (Chandelier Exit)
    - Percentage stops
    - Support-based stops
    """
    
    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig()
        self.atr_calc = ATRCalculator()
    
    def calculate_chandelier_stop(self, data: StockData,
                                 from_entry: bool = False,
                                 entry_price: Optional[float] = None
                                 ) -> StopLossResult:
        """
        Calculate Chandelier Exit stop.
        
        For long positions: Highest High - (ATR × multiplier)
        
        Args:
            data: StockData object
            from_entry: If True, use entry price instead of highest high
            entry_price: Entry price for position (if from_entry=True)
            
        Returns:
            StopLossResult
        """
        df = data.df
        cfg = self.config
        
        # Calculate ATR
        atr = self.atr_calc.calculate_atr(df, cfg.atr_period)
        current_atr = atr.iloc[-1]
        
        # Reference price (highest high since entry or recent period)
        if from_entry and entry_price is not None:
            # For new entry, use tighter stop
            multiplier = cfg.tight_stop_multiplier
            reference_price = entry_price
        else:
            # For trailing stop, use highest high
            multiplier = cfg.chandelier_multiplier
            # Look back 20 days for highest high
            reference_price = df['High'].iloc[-20:].max()
        
        # Calculate stop
        stop_price = reference_price - (current_atr * multiplier)
        
        current_price = df['Close'].iloc[-1]
        risk_per_share = current_price - stop_price
        risk_percentage = risk_per_share / current_price
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type='atr_chandelier',
            risk_per_share=risk_per_share,
            risk_percentage=risk_percentage,
            atr_value=current_atr,
            atr_multiplier=multiplier
        )
    
    def calculate_percentage_stop(self, data: StockData,
                                 max_loss_pct: Optional[float] = None
                                 ) -> StopLossResult:
        """
        Calculate simple percentage stop.
        
        Args:
            data: StockData object
            max_loss_pct: Maximum loss percentage (default from config)
            
        Returns:
            StopLossResult
        """
        if max_loss_pct is None:
            max_loss_pct = self.config.max_stock_drawdown
        
        current_price = data.df['Close'].iloc[-1]
        stop_price = current_price * (1 - max_loss_pct)
        
        # Still calculate ATR for reference
        atr = self.atr_calc.calculate_atr(data.df, self.config.atr_period)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type='percentage',
            risk_per_share=current_price * max_loss_pct,
            risk_percentage=max_loss_pct,
            atr_value=atr.iloc[-1],
            atr_multiplier=0  # Not ATR-based
        )
    
    def find_support_stop(self, data: StockData,
                         ma_periods: list = [50, 150, 200]
                         ) -> StopLossResult:
        """
        Find stop based on nearest support level (moving averages).
        
        Args:
            data: StockData object
            ma_periods: List of MA periods to check
            
        Returns:
            StopLossResult with support-based stop
        """
        df = data.df
        close = df['Close']
        current_price = close.iloc[-1]
        
        # Calculate MAs
        support_levels = []
        for period in ma_periods:
            if len(close) >= period:
                ma = close.rolling(period).mean().iloc[-1]
                if ma < current_price:  # Only consider MAs below current price
                    support_levels.append((period, ma))
        
        # Find nearest support
        if support_levels:
            # Use highest support (nearest below price)
            support_levels.sort(key=lambda x: x[1], reverse=True)
            nearest_period, support_price = support_levels[0]
            
            # Place stop just below support (with buffer)
            stop_price = support_price * 0.98  # 2% below support
        else:
            # Fallback to ATR stop
            return self.calculate_chandelier_stop(data)
        
        atr = self.atr_calc.calculate_atr(df, self.config.atr_period)
        
        risk_per_share = current_price - stop_price
        risk_percentage = risk_per_share / current_price
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=f'support_ma{nearest_period}',
            risk_per_share=risk_per_share,
            risk_percentage=risk_percentage,
            atr_value=atr.iloc[-1],
            atr_multiplier=0
        )


class PositionSizer:
    """
    Calculate appropriate position sizes.
    
    Methods:
    - Fixed percentage risk
    - Volatility-adjusted (inverse volatility)
    - Kelly criterion (simplified)
    """
    
    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig()
    
    def calculate_fixed_risk_size(self,
                                  portfolio_value: float,
                                  entry_price: float,
                                  stop_price: float,
                                  risk_per_trade: float = 0.01
                                  ) -> PositionSizeResult:
        """
        Calculate position size based on fixed risk per trade.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            stop_price: Stop loss price
            risk_per_trade: Max risk per trade as % of portfolio (default 1%)
            
        Returns:
            PositionSizeResult
        """
        # Dollar risk per trade
        risk_amount = portfolio_value * risk_per_trade
        
        # Risk per share
        risk_per_share = entry_price - stop_price
        
        if risk_per_share <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0,
                position_pct=0,
                risk_amount=0,
                risk_pct=0,
                sizing_method='fixed_risk'
            )
        
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        
        # Check max position constraint
        position_value = shares * entry_price
        max_position = portfolio_value * self.config.max_position_pct
        
        if position_value > max_position:
            shares = int(max_position / entry_price)
            position_value = shares * entry_price
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            position_pct=position_value / portfolio_value,
            risk_amount=actual_risk,
            risk_pct=actual_risk / portfolio_value,
            sizing_method='fixed_risk'
        )
    
    def calculate_volatility_adjusted_size(self,
                                          data: StockData,
                                          portfolio_value: float,
                                          target_volatility: Optional[float] = None
                                          ) -> PositionSizeResult:
        """
        Calculate position size using inverse volatility weighting.
        
        Higher volatility stocks get smaller positions.
        
        Args:
            data: StockData object
            portfolio_value: Total portfolio value
            target_volatility: Target portfolio volatility (annualized)
            
        Returns:
            PositionSizeResult
        """
        if target_volatility is None:
            target_volatility = self.config.target_portfolio_volatility
        
        # Calculate stock volatility (annualized)
        returns = data.returns.dropna()
        if len(returns) < 20:
            # Insufficient data, use conservative sizing
            return PositionSizeResult(
                shares=0,
                position_value=0,
                position_pct=0,
                risk_amount=0,
                risk_pct=0,
                sizing_method='volatility_adjusted'
            )
        
        stock_vol = returns.std() * np.sqrt(252)
        
        if stock_vol <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0,
                position_pct=0,
                risk_amount=0,
                risk_pct=0,
                sizing_method='volatility_adjusted'
            )
        
        # Position size inversely proportional to volatility
        # target_vol / stock_vol gives the leverage factor
        # For single position, this gives the fraction of portfolio
        position_pct = min(target_volatility / stock_vol, self.config.max_position_pct)
        
        current_price = data.df['Close'].iloc[-1]
        position_value = portfolio_value * position_pct
        shares = int(position_value / current_price)
        
        # Estimate risk (using VaR at 95% - approximately 1.65 * daily vol)
        daily_vol = stock_vol / np.sqrt(252)
        estimated_daily_loss = position_value * daily_vol * 1.65
        
        return PositionSizeResult(
            shares=shares,
            position_value=shares * current_price,
            position_pct=position_pct,
            risk_amount=estimated_daily_loss,
            risk_pct=estimated_daily_loss / portfolio_value,
            sizing_method='volatility_adjusted'
        )
    
    def calculate_kelly_size(self,
                            win_rate: float,
                            avg_win: float,
                            avg_loss: float,
                            portfolio_value: float,
                            entry_price: float,
                            kelly_fraction: Optional[float] = None
                            ) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - [(1-W) / R]
        Where W = win rate, R = win/loss ratio
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            kelly_fraction: Fraction of Kelly to use (default from config)
            
        Returns:
            PositionSizeResult
        """
        if kelly_fraction is None:
            kelly_fraction = self.config.kelly_fraction
        
        if avg_loss <= 0:
            return PositionSizeResult(
                shares=0,
                position_value=0,
                position_pct=0,
                risk_amount=0,
                risk_pct=0,
                sizing_method='kelly'
            )
        
        # Calculate Kelly percentage
        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fraction (conservative Kelly)
        adjusted_kelly = max(0, kelly_pct * kelly_fraction)
        
        # Cap at max position
        position_pct = min(adjusted_kelly, self.config.max_position_pct)
        
        position_value = portfolio_value * position_pct
        shares = int(position_value / entry_price)
        
        # Estimate risk using average loss
        risk_amount = position_value * avg_loss
        
        return PositionSizeResult(
            shares=shares,
            position_value=shares * entry_price,
            position_pct=position_pct,
            risk_amount=risk_amount,
            risk_pct=risk_amount / portfolio_value,
            sizing_method='kelly'
        )


class PullbackAnalyzer:
    """
    Analyze price declines to classify as pullback vs reversal.
    
    Helps traders distinguish between:
    - Healthy pullback (buying opportunity)
    - Trend reversal (exit signal)
    """
    
    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig()
    
    def classify_pullback(self, data: StockData,
                         entry_price: Optional[float] = None
                         ) -> PullbackClassification:
        """
        Classify current price decline.
        
        Factors analyzed:
        1. Volume pattern (contracting = healthy)
        2. Support levels (holding = healthy)
        3. Decline depth (shallow = healthy)
        4. Duration below MA
        
        Args:
            data: StockData object
            entry_price: Original entry price (optional)
            
        Returns:
            PullbackClassification
        """
        df = data.df
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        
        current_price = close.iloc[-1]
        
        # Calculate key MAs
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        # 1. Find recent high and calculate drawdown
        recent_high = high.iloc[-20:].max()
        max_depth = (recent_high - close.iloc[-20:].min()) / recent_high
        current_depth = (recent_high - current_price) / recent_high
        
        # 2. Volume analysis
        volume_pattern = self._analyze_pullback_volume(df)
        
        # 3. Support analysis
        support_holding = self._check_support_holding(df, ma50, ma200)
        
        # 4. Days below 50MA
        below_ma50 = close < ma50
        if below_ma50.iloc[-1]:
            # Count consecutive days below
            days_below = 0
            for i in range(len(below_ma50) - 1, -1, -1):
                if below_ma50.iloc[i]:
                    days_below += 1
                else:
                    break
        else:
            days_below = 0
        
        # 5. Classification logic
        is_pullback, confidence = self._make_classification(
            volume_pattern=volume_pattern,
            support_holding=support_holding,
            max_depth=max_depth,
            days_below=days_below,
            ma50_value=ma50.iloc[-1],
            ma200_value=ma200.iloc[-1],
            current_price=current_price
        )
        
        # 6. Generate recommendation
        recommendation = self._get_recommendation(
            is_pullback=is_pullback,
            confidence=confidence,
            support_holding=support_holding,
            max_depth=max_depth
        )
        
        return PullbackClassification(
            is_pullback=is_pullback,
            confidence=confidence,
            volume_pattern=volume_pattern,
            support_holding=support_holding,
            days_below_ma=days_below,
            max_depth_pct=max_depth,
            recommendation=recommendation
        )
    
    def _analyze_pullback_volume(self, df: pd.DataFrame) -> str:
        """Analyze volume pattern during pullback."""
        close = df['Close']
        volume = df['Volume']
        
        returns = close.pct_change()
        
        # Compare up-day volume vs down-day volume in last 10 days
        recent = pd.DataFrame({
            'return': returns.iloc[-10:],
            'volume': volume.iloc[-10:]
        })
        
        up_vol = recent[recent['return'] > 0]['volume'].mean()
        down_vol = recent[recent['return'] < 0]['volume'].mean()
        
        if pd.isna(up_vol) or pd.isna(down_vol):
            return 'neutral'
        
        ratio = down_vol / up_vol if up_vol > 0 else 1
        
        if ratio < self.config.pullback_max_volume_ratio:
            return 'contracting'  # Healthy - low volume on down days
        elif ratio > 1.2:
            return 'expanding'  # Unhealthy - high volume on down days
        else:
            return 'neutral'
    
    def _check_support_holding(self, df: pd.DataFrame,
                               ma50: pd.Series,
                               ma200: pd.Series) -> bool:
        """Check if key support levels are holding."""
        close = df['Close']
        low = df['Low']
        
        current_price = close.iloc[-1]
        recent_low = low.iloc[-10:].min()
        
        # Check support at 50MA
        ma50_support = recent_low >= ma50.iloc[-10:].min() * 0.98
        
        # Check support at 200MA
        ma200_support = recent_low >= ma200.iloc[-10:].min() * 0.97
        
        # At least one major support should hold
        return ma50_support or ma200_support
    
    def _make_classification(self,
                            volume_pattern: str,
                            support_holding: bool,
                            max_depth: float,
                            days_below: int,
                            ma50_value: float,
                            ma200_value: float,
                            current_price: float) -> Tuple[bool, float]:
        """Make final pullback classification."""
        
        # Scoring system
        pullback_score = 0
        
        # Volume (most important)
        if volume_pattern == 'contracting':
            pullback_score += 0.35
        elif volume_pattern == 'neutral':
            pullback_score += 0.15
        # Expanding volume is bad
        
        # Support holding
        if support_holding:
            pullback_score += 0.25
        
        # Depth (shallow is better)
        if max_depth < 0.10:
            pullback_score += 0.20
        elif max_depth < 0.15:
            pullback_score += 0.10
        # Deep pullbacks are more concerning
        
        # Days below MA
        if days_below <= self.config.pullback_max_days:
            pullback_score += 0.10
        
        # Price still above 200MA
        if current_price > ma200_value:
            pullback_score += 0.10
        
        # Classification threshold
        is_pullback = pullback_score >= 0.45
        confidence = pullback_score
        
        return is_pullback, confidence
    
    def _get_recommendation(self,
                           is_pullback: bool,
                           confidence: float,
                           support_holding: bool,
                           max_depth: float) -> str:
        """Generate action recommendation."""
        
        if is_pullback:
            if confidence >= 0.7 and support_holding:
                return 'add'  # Consider adding to position
            elif confidence >= 0.5:
                return 'hold'  # Hold position
            else:
                return 'hold'  # Hold but watch closely
        else:
            if max_depth >= 0.20:
                return 'exit'  # Likely reversal, exit
            elif confidence < 0.3:
                return 'reduce'  # Reduce position
            else:
                return 'hold'  # Hold but tight stops


class PortfolioRiskManager:
    """
    Portfolio-level risk management.
    
    Monitors:
    - Total portfolio drawdown
    - Correlation risk
    - Sector concentration
    """
    
    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig()
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.high_water_mark: float = 0
    
    def update_equity(self, portfolio_value: float, date: pd.Timestamp):
        """Update equity curve and high water mark."""
        self.equity_curve[date] = portfolio_value
        
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value
    
    def get_current_drawdown(self) -> Tuple[float, DrawdownState]:
        """Calculate current drawdown from high water mark."""
        if self.high_water_mark <= 0 or len(self.equity_curve) == 0:
            return 0.0, DrawdownState.NORMAL
        
        current_value = self.equity_curve.iloc[-1]
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        
        # Classify drawdown state
        if drawdown < 0.05:
            state = DrawdownState.NORMAL
        elif drawdown < 0.10:
            state = DrawdownState.ELEVATED
        elif drawdown < self.config.max_portfolio_drawdown:
            state = DrawdownState.HIGH
        else:
            state = DrawdownState.CRITICAL
        
        return drawdown, state
    
    def should_reduce_exposure(self) -> Tuple[bool, float]:
        """
        Determine if exposure should be reduced based on drawdown.
        
        Returns:
            (should_reduce, target_exposure_multiplier)
        """
        drawdown, state = self.get_current_drawdown()
        
        if state == DrawdownState.CRITICAL:
            return True, 0.25  # Reduce to 25% exposure
        elif state == DrawdownState.HIGH:
            return True, 0.50  # Reduce to 50% exposure
        elif state == DrawdownState.ELEVATED:
            return True, 0.75  # Reduce to 75% exposure
        else:
            return False, 1.0  # Full exposure OK
