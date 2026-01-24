"""
Core Data Structures and Base Classes
=====================================

Provides foundational types for the signal generation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class SignalStrength(Enum):
    """Discrete signal strength levels for interpretability."""
    STRONG_SELL = -2
    WEAK_SELL = -1
    NEUTRAL = 0
    WEAK_BUY = 1
    STRONG_BUY = 2


@dataclass
class SignalResult:
    """
    Standardized output from any signal generator.
    
    Attributes:
        name: Signal identifier (e.g., 'trend_template', 'vcp')
        score: Normalized score in [0, 1] range
        strength: Discrete strength classification
        confidence: Estimation of signal reliability [0, 1]
        components: Breakdown of sub-scores
        metadata: Additional diagnostic information
        timestamp: When signal was generated
    """
    name: str
    score: float                          # Normalized [0, 1]
    strength: SignalStrength
    confidence: float                     # [0, 1]
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Validate score range
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    @classmethod
    def from_score(cls, name: str, score: float, confidence: float = 0.5,
                   components: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Factory method to create SignalResult with auto-calculated strength."""
        if score >= 0.8:
            strength = SignalStrength.STRONG_BUY
        elif score >= 0.6:
            strength = SignalStrength.WEAK_BUY
        elif score >= 0.4:
            strength = SignalStrength.NEUTRAL
        elif score >= 0.2:
            strength = SignalStrength.WEAK_SELL
        else:
            strength = SignalStrength.STRONG_SELL
            
        return cls(
            name=name,
            score=score,
            strength=strength,
            confidence=confidence,
            components=components or {},
            metadata=metadata or {}
        )


@dataclass
class StockData:
    """
    Container for stock price and volume data with derived fields.
    
    All calculations are vectorized for performance.
    """
    ticker: str
    df: pd.DataFrame  # OHLCV data with DatetimeIndex
    
    # Lazy-computed derived fields
    _returns: Optional[pd.Series] = field(default=None, repr=False)
    _log_returns: Optional[pd.Series] = field(default=None, repr=False)
    
    def __post_init__(self):
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
    
    @property
    def returns(self) -> pd.Series:
        """Simple returns."""
        if self._returns is None:
            self._returns = self.df['Close'].pct_change()
        return self._returns
    
    @property
    def log_returns(self) -> pd.Series:
        """Log returns for statistical analysis."""
        if self._log_returns is None:
            self._log_returns = np.log(self.df['Close'] / self.df['Close'].shift(1))
        return self._log_returns
    
    @property
    def close(self) -> pd.Series:
        return self.df['Close']
    
    @property
    def volume(self) -> pd.Series:
        return self.df['Volume']
    
    @property
    def high(self) -> pd.Series:
        return self.df['High']
    
    @property
    def low(self) -> pd.Series:
        return self.df['Low']
    
    def get_latest(self, n_days: int = 1) -> pd.DataFrame:
        """Get most recent n days of data."""
        return self.df.iloc[-n_days:]


@dataclass
class CompositeSignal:
    """
    Aggregated signal combining multiple signal sources.
    
    Attributes:
        ticker: Stock identifier
        composite_score: Overall weighted score [0, 1]
        technical_score: Technical analysis subscore
        pattern_score: Pattern recognition subscore
        fundamental_score: Fundamental analysis subscore
        signals: Individual signal results
        rank: Percentile rank among universe (if computed)
        regime: Current market regime when signal was generated
    """
    ticker: str
    composite_score: float
    technical_score: float
    pattern_score: float
    fundamental_score: float
    signals: Dict[str, SignalResult]
    rank: Optional[int] = None
    regime: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def passes_threshold(self, min_score: float = 0.6) -> bool:
        """Check if signal passes minimum threshold."""
        return self.composite_score >= min_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'composite_score': self.composite_score,
            'technical_score': self.technical_score,
            'pattern_score': self.pattern_score,
            'fundamental_score': self.fundamental_score,
            'rank': self.rank,
            'regime': self.regime,
            'timestamp': self.timestamp.isoformat(),
            'signals': {k: {'score': v.score, 'strength': v.strength.name}
                       for k, v in self.signals.items()}
        }


class BaseSignalGenerator(ABC):
    """
    Abstract base class for all signal generators.
    
    Ensures consistent interface across different signal types.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self._cache: Dict[str, SignalResult] = {}
    
    @abstractmethod
    def generate(self, data: StockData) -> SignalResult:
        """
        Generate signal for given stock data.
        
        Args:
            data: StockData object with OHLCV data
            
        Returns:
            SignalResult with score, strength, and metadata
        """
        pass
    
    @abstractmethod
    def get_required_lookback(self) -> int:
        """Return minimum data points required for signal calculation."""
        pass
    
    def validate_data(self, data: StockData) -> bool:
        """Check if data has sufficient history."""
        return len(data.df) >= self.get_required_lookback()
    
    def clear_cache(self):
        """Clear cached results."""
        self._cache = {}


@dataclass
class BacktestResult:
    """Results from backtesting a signal strategy."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Trade log
    trades: pd.DataFrame
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    
    # Optional fields with defaults (must be at the end)
    average_return_per_trade: float = 0.0  # Average return percentage per trade (from entry to exit)
    exit_reasons: Dict[str, int] = field(default_factory=dict)  # Exit reason statistics
    
    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULTS                         ║
╠══════════════════════════════════════════════════════════════╣
║  Total Return:        {self.total_return:>10.2%}                        ║
║  Annualized Return:   {self.annualized_return:>10.2%}                        ║
║  Sharpe Ratio:        {self.sharpe_ratio:>10.2f}                        ║
║  Sortino Ratio:       {self.sortino_ratio:>10.2f}                        ║
║  Max Drawdown:        {self.max_drawdown:>10.2%}                        ║
║  Win Rate:            {self.win_rate:>10.2%}                        ║
║  Profit Factor:       {self.profit_factor:>10.2f}                        ║
║  Total Trades:        {self.total_trades:>10d}                        ║
╠══════════════════════════════════════════════════════════════╣
║  vs Benchmark                                                ║
║  Benchmark Return:    {self.benchmark_return:>10.2%}                        ║
║  Alpha:               {self.alpha:>10.2%}                        ║
║  Beta:                {self.beta:>10.2f}                        ║
║  Information Ratio:   {self.information_ratio:>10.2f}                        ║
╚══════════════════════════════════════════════════════════════╝
"""


# Type aliases for clarity
PriceData = pd.DataFrame
SignalDict = Dict[str, SignalResult]
TickerList = List[str]
