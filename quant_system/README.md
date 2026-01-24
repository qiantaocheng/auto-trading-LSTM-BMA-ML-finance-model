# Quantitative Growth Stock Signal System

A comprehensive Python-based signal generation system for identifying long-term growth stocks, implementing methodologies from Mark Minervini, Stan Weinstein, and William O'Neil.

## Overview

This system provides a complete framework for:
- **Signal Generation**: Technical analysis signals based on proven methodologies
- **Pattern Recognition**: VCP (Volatility Contraction Pattern) detection
- **Relative Strength**: IBD-style RS rating calculation
- **Risk Management**: ATR-based stops, position sizing, drawdown monitoring
- **Backtesting**: Full backtesting engine with realistic execution modeling
- **Market Regime Detection**: Adaptive parameter adjustment based on market conditions

## System Architecture

```
quant_system/
├── config/
│   └── settings.py          # All configuration dataclasses
├── core/
│   ├── data_types.py        # Base types and interfaces
│   └── signal_aggregator.py # Signal combination and regime detection
├── signals/
│   ├── trend_signals.py     # Trend Template, ADX, Linear Regression
│   ├── stage_analysis.py    # Weinstein Stage Analysis
│   ├── vcp_detector.py      # VCP pattern recognition
│   └── relative_strength.py # IBD RS Rating
├── utils/
│   └── risk_management.py   # Stops, position sizing, pullback analysis
├── backtest/
│   └── engine.py            # Backtesting engine
├── data/
│   └── data_loader.py       # Data loading utilities
└── main.py                  # Example usage
```

## Installation

```bash
# Clone or extract the system
cd quant_system

# Install dependencies
pip install pandas numpy scipy

# Optional: For data loading from Yahoo Finance
pip install yfinance
```

## Quick Start

```python
import pandas as pd
from quant_system.core import SignalAggregator, StockData
from quant_system.data import DataLoader, generate_sample_data

# Load data
loader = DataLoader()
universe_data = loader.load_universe_yahoo(['AAPL', 'MSFT', 'NVDA'], period='2y')
benchmark_data = loader.load_yahoo('SPY', period='2y')

# Create aggregator
aggregator = SignalAggregator(benchmark_data=benchmark_data)

# Scan universe for signals
signals = aggregator.scan_universe(universe_data, min_score=0.60)

# Print top signals
for signal in signals[:10]:
    print(f"{signal.metadata['symbol']}: Score={signal.composite_score:.2f}")
    print(aggregator.generate_signal_report(signal))
```

## Signal Components

### 1. Trend Template (Minervini)
- MA Alignment: Price > SMA50 > SMA150 > SMA200
- Price Position: 30%+ above 52-week low, within 25% of high
- Slope Analysis: 200-day MA trending up with R² consistency

### 2. Stage Analysis (Weinstein)
- Weekly 30-week moving average analysis
- Stage classification (1-4)
- Volume accumulation/distribution patterns
- Mansfield Relative Strength

### 3. VCP Pattern Detection
- Automatic peak/trough identification
- Contraction depth decay validation
- Volume dry-up detection
- Bollinger Band squeeze recognition
- Pivot point and pocket pivot detection

### 4. Relative Strength (O'Neil/IBD)
- Time-weighted 12-month performance
- Percentile ranking against universe
- Divergence detection vs benchmark
- Momentum consistency analysis

### 5. ADX Trend Strength
- Directional movement analysis
- Trend strength classification
- ADX momentum tracking

## Risk Management

### Stop Loss Methods
- **Chandelier Exit**: ATR-based trailing stops
- **Percentage Stop**: Fixed percentage from entry
- **Support-Based**: Stops below key moving averages

### Position Sizing
- **Fixed Risk**: Risk 1% of portfolio per trade
- **Volatility-Adjusted**: Inverse volatility weighting
- **Kelly Criterion**: With fractional Kelly (0.25x)

### Portfolio Management
- Drawdown monitoring with state classification
- Dynamic exposure reduction
- Pullback analysis for add/reduce decisions

## Market Regime Detection

The system adapts to market conditions:

| Regime | Breadth | Signal Adjustment |
|--------|---------|-------------------|
| BULL_STRONG | ≥80% above 200MA | Full signals |
| BULL_WEAK | 50-80% | Favor technicals |
| NEUTRAL | 30-50% | Strict technical filter |
| BEAR_WEAK | 20-30% | Very selective |
| BEAR_STRONG | <20% | Cash preferred |

## Backtesting

```python
from quant_system.backtest import BacktestEngine, BacktestConfig, generate_backtest_report

config = BacktestConfig(
    initial_capital=100000,
    max_positions=10,
    min_signal_score=0.65
)

engine = BacktestEngine(signal_aggregator=aggregator, config=config)

result = engine.run(
    universe_data=universe_data,
    benchmark_data=benchmark_data,
    start_date='2022-01-01',
    end_date='2024-01-01'
)

print(generate_backtest_report(result, "Growth Strategy"))
```

## Configuration

All parameters are externalized for optimization:

```python
from quant_system.config import SystemConfig

# Get default configuration
config = SystemConfig()

# Modify parameters
config.trend_template.ma_periods = (50, 100, 200)
config.vcp.min_contractions = 3
config.risk_management.kelly_fraction = 0.20

# Apply to aggregator
aggregator = SignalAggregator(config=config.signal_aggregation)
```

## Key Parameters

### Trend Template
| Parameter | Default | Description |
|-----------|---------|-------------|
| MA Periods | 50/150/200 | Moving average windows |
| 52-Week Above Low | 30% | Minimum % above 52-week low |
| 52-Week Near High | 75% | Maximum % from 52-week high |

### VCP Pattern
| Parameter | Default | Description |
|-----------|---------|-------------|
| Min Contractions | 2 | Minimum contraction count |
| Decay Ratio | 0.65 | Each contraction ≤65% of previous |
| Final Squeeze | 10% | Maximum final contraction depth |
| Volume Dry-Up | 70% | Volume threshold for dry-up |

### Relative Strength
| Parameter | Default | Description |
|-----------|---------|-------------|
| Lookback | 252 days | RS calculation period |
| Q1 Weight | 40% | Most recent quarter weight |
| Min RS Rating | 80 | Minimum acceptable RS |

### Risk Management
| Parameter | Default | Description |
|-----------|---------|-------------|
| Chandelier Multiplier | 3.0x ATR | Trailing stop distance |
| Max Position | 10% | Maximum single position |
| Kelly Fraction | 0.25 | Conservative Kelly |
| Max Drawdown | 15% | Portfolio stop level |

## Signal Scoring

Each signal generates a score from 0-1:
- **0.0-0.3**: Strong Sell / Avoid
- **0.3-0.5**: Weak / No Position
- **0.5-0.6**: Neutral / Hold
- **0.6-0.8**: Buy / Accumulate
- **0.8-1.0**: Strong Buy / Full Position

Composite scoring weights:
- Technical: 45%
- Pattern: 25%
- Fundamental (via RS): 30%

## Extending the System

### Adding New Signals

```python
from quant_system.core import BaseSignalGenerator, SignalResult, StockData

class MyCustomSignal(BaseSignalGenerator):
    def __init__(self, config=None):
        self.config = config
    
    def generate(self, stock_data: StockData, **kwargs) -> SignalResult:
        # Your signal logic here
        score = self._calculate_score(stock_data)
        return SignalResult(
            score=score,
            strength=self._score_to_strength(score),
            confidence=0.8,
            components={'my_metric': score},
            metadata={'generator': 'MyCustomSignal'}
        )
    
    def get_required_lookback(self) -> int:
        return 50
```

### Custom Regime Detection

```python
from quant_system.core import MarketRegimeDetector
from quant_system.config import MarketRegime

class MyRegimeDetector(MarketRegimeDetector):
    def detect_regime(self, benchmark_data, universe_data=None):
        # Custom regime logic
        vix = self._get_vix()  # Example
        if vix > 30:
            return MarketRegime.BEAR_STRONG
        # ... more logic
```

## Performance Notes

- Signal generation is vectorized using NumPy/Pandas
- Caching is built into base signal generators
- For large universes (1000+ stocks), use batch processing
- Consider using Parquet format for faster data loading

## Limitations & Future Work

1. **Fundamental Data**: Currently uses RS as fundamental proxy; integrate earnings, ROE, institutional ownership
2. **Transaction Costs**: Basic model; add market impact for large positions
3. **Sector Exposure**: Not yet implemented; add sector concentration limits
4. **Options Integration**: Add options-based signals (put/call ratio, unusual activity)
5. **Machine Learning**: Ready for ML integration via feature extraction

## References

- Minervini, M. (2013). *Trade Like a Stock Market Wizard*
- Weinstein, S. (1988). *Secrets for Profiting in Bull and Bear Markets*
- O'Neil, W. (2009). *How to Make Money in Stocks*

## License

MIT License - Free for personal and commercial use.

---

**Disclaimer**: This software is for educational purposes. Trading involves risk. Past performance does not guarantee future results. Always do your own research before making investment decisions.
