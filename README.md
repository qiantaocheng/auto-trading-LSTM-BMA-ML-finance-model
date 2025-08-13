# AutoTrader - Professional IBKR Automated Trading System

A comprehensive automated trading system built for Interactive Brokers (IBKR) that provides real-time trading, backtesting, risk management, and portfolio optimization capabilities.

## Project Overview

AutoTrader is a professional-grade quantitative trading platform that integrates multiple components to create a complete trading ecosystem. The system is designed with modular architecture, ensuring scalability, maintainability, and robust error handling.

## System Architecture

The project follows a layered architecture pattern with clear separation of concerns:

### Core Infrastructure Layer
- **Event Loop Management**: Thread-safe asyncio event loop handling
- **Configuration Management**: Unified configuration system across all components
- **Resource Monitoring**: System resource tracking and leak prevention
- **Event System**: Decoupled communication between GUI and trading engine

### Trading Engine Layer
- **Signal Processing**: Real-time market data analysis and signal generation
- **Risk Management**: Portfolio-level and position-level risk controls
- **Order Management**: Professional order state machine and execution
- **Market Data**: Real-time and historical data handling

### Data Management Layer
- **Database Operations**: SQLite-based data persistence
- **Data Source Management**: Unified stock universe management
- **Technical Indicators**: Cached indicator calculations for performance
- **Backtesting Engine**: Historical strategy validation

### User Interface Layer
- **GUI Application**: Tkinter-based trading interface
- **System Launcher**: Application startup and mode selection
- **Logging System**: Comprehensive audit and debugging capabilities

## Core Components

### 1. Unified Configuration Manager (`unified_config.py`)
**Purpose**: Centralized configuration management system that resolves conflicts between multiple configuration sources.

**Key Features**:
- Loads configurations from multiple sources (files, database, runtime)
- Implements priority-based configuration merging
- Provides thread-safe configuration access
- Supports runtime configuration updates
- Validates configuration consistency

**Configuration Sources** (in priority order):
1. Runtime configurations (highest priority)
2. HotConfig files
3. Database configurations
4. File-based configurations
5. Default configurations (lowest priority)

### 2. Event Loop Manager (`event_loop_manager.py`)
**Purpose**: Thread-safe asyncio event loop management for GUI applications.

**Key Features**:
- Manages dedicated event loop in separate thread
- Provides thread-safe coroutine submission
- Handles task lifecycle management
- Implements proper cleanup and shutdown procedures
- Prevents event loop conflicts in multi-threaded environments

**Technical Implementation**:
- Uses queue-based command system for thread communication
- Implements timeout mechanisms for coroutine execution
- Provides comprehensive error handling and recovery
- Supports task cancellation and resource cleanup

### 3. Event System (`event_system.py`)
**Purpose**: Decoupled communication system between GUI and trading engine components.

**Key Features**:
- Implements publish-subscribe pattern
- Supports both synchronous and asynchronous event handling
- Provides priority-based event processing
- Implements weak reference management for subscribers
- Supports event filtering and routing

**Event Types**:
- Engine status updates
- Trading signals and orders
- Risk alerts and notifications
- System health monitoring
- User interface updates

### 4. Resource Monitor (`resource_monitor.py`)
**Purpose**: System-wide resource monitoring and leak prevention.

**Key Features**:
- Tracks memory usage and growth patterns
- Monitors active tasks and connections
- Detects file handle leaks
- Implements automatic cleanup triggers
- Provides resource usage statistics

**Monitoring Capabilities**:
- Memory consumption tracking with trend analysis
- Thread count monitoring
- File descriptor tracking
- Task lifecycle monitoring
- Automatic garbage collection triggers

### 5. Trading Engine (`engine.py`)
**Purpose**: Core trading logic and decision-making engine.

**Key Components**:
- **RiskEngine**: Portfolio and position-level risk management
- **SignalHub**: Market signal processing and aggregation
- **OrderRouter**: Order routing and execution management
- **DataFeed**: Real-time market data handling

**Risk Management Features**:
- Position sizing based on volatility
- Portfolio exposure limits
- Correlation-based risk controls
- Dynamic stop-loss management
- Sector concentration limits

### 6. IBKR Auto Trader (`ibkr_auto_trader.py`)
**Purpose**: Interactive Brokers API integration and trading interface.

**Key Features**:
- Real-time market data subscription
- Order placement and management
- Account and position monitoring
- Portfolio performance tracking
- Connection management and recovery

**Trading Capabilities**:
- Market, limit, and bracket orders
- Real-time portfolio updates
- Dynamic stop-loss management
- Multi-symbol trading support
- Commission and slippage handling

### 7. Order State Machine (`order_state_machine.py`)
**Purpose**: Professional order lifecycle management.

**Key Features**:
- Complete order state tracking
- Transition validation and enforcement
- Fill management and partial execution handling
- Order modification and cancellation
- Audit trail maintenance

**Order States**:
- Pending submission
- Submitted
- Partially filled
- Filled
- Cancelled
- Rejected
- Error states

### 8. Enhanced Order Execution (`enhanced_order_execution.py`)
**Purpose**: Advanced order execution algorithms and optimization.

**Key Features**:
- Smart order routing
- Liquidity estimation
- Market impact minimization
- Execution timing optimization
- Cost analysis and reporting

**Execution Algorithms**:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg orders
- Dynamic order splitting
- Market timing optimization

### 9. Data Source Manager (`data_source_manager.py`)
**Purpose**: Unified management of stock universes and data sources.

**Key Features**:
- Multiple data source integration
- Priority-based source selection
- Data consistency validation
- Automatic source synchronization
- Cache management for performance

**Data Sources**:
- Database tickers
- File-based stock lists
- Runtime configurations
- Manual input sources
- External data feeds

### 10. Database Operations (`database.py`)
**Purpose**: SQLite-based data persistence and management.

**Key Features**:
- Stock list management
- Trade history recording
- Risk configuration storage
- Performance data persistence
- Data integrity maintenance

**Database Schema**:
- Tickers table for stock universe
- Trade history for audit trails
- Risk configurations for strategy settings
- Performance metrics for analysis
- User preferences and settings

### 11. Technical Indicator Cache (`indicator_cache.py`)
**Purpose**: Performance optimization for technical indicator calculations.

**Key Features**:
- Cached indicator results
- Time-based cache invalidation
- Memory-efficient storage
- Computation time tracking
- Cache hit/miss statistics

**Supported Indicators**:
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- MACD and other momentum indicators

### 12. Factor Calculations (`factors.py`)
**Purpose**: Technical analysis and factor computation.

**Key Features**:
- Pure computational functions
- No side effects or state management
- Optimized mathematical operations
- Support for vectorized calculations
- Extensible factor framework

**Available Factors**:
- Price-based indicators
- Volume-based indicators
- Volatility measures
- Momentum indicators
- Mean reversion signals

### 13. GUI Application (`app.py`)
**Purpose**: Main user interface for trading operations.

**Key Features**:
- Real-time market data display
- Order entry and management interface
- Portfolio monitoring dashboard
- Risk management controls
- System status and health monitoring

**Interface Components**:
- Market data panels
- Order entry forms
- Portfolio overview
- Risk metrics display
- System configuration panels
- Log and audit viewers

### 14. System Launcher (`launcher.py`)
**Purpose**: Application startup and mode selection.

**Key Features**:
- Multiple launch modes (GUI, strategy, direct trading)
- System health checks
- Dependency validation
- Configuration verification
- Error handling and recovery

**Launch Modes**:
- GUI Mode: Full trading interface
- Strategy Mode: Automated trading with engine
- Direct Mode: Manual trading interface
- Backtest Mode: Historical strategy testing

### 15. Backtest Engine (`backtest_engine.py`)
**Purpose**: Historical strategy validation and performance analysis.

**Key Features**:
- BMA (Bayesian Model Averaging) integration
- Realistic transaction cost modeling
- Risk-adjusted performance metrics
- Multiple rebalancing frequencies
- Comprehensive reporting capabilities

**Backtesting Capabilities**:
- Historical data simulation
- Transaction cost analysis
- Risk metric calculation
- Performance attribution
- Strategy comparison tools

### 16. Backtest Analyzer (`backtest_analyzer.py`)
**Purpose**: Professional backtest result analysis and visualization.

**Key Features**:
- Performance metric calculation
- Risk-adjusted return analysis
- Drawdown analysis
- Sharpe ratio and other risk metrics
- Visualization and reporting

**Analysis Metrics**:
- Total and annualized returns
- Volatility and Sharpe ratios
- Maximum drawdown analysis
- Win rate and profit factor
- Risk-adjusted performance measures

### 17. Engine Logger (`engine_logger.py`)
**Purpose**: Specialized logging system for trading engine components.

**Key Features**:
- Event-driven logging
- Thread-safe log handling
- Integration with GUI display
- Log level management
- Performance monitoring

**Logging Capabilities**:
- Real-time log streaming
- Log level filtering
- Performance metrics logging
- Error tracking and reporting
- Audit trail maintenance

### 18. Trading Auditor (`trading_auditor_v2.py`)
**Purpose**: Compliance and audit trail management.

**Key Features**:
- Trade compliance checking
- Regulatory reporting
- Audit trail maintenance
- Risk event logging
- Performance monitoring

**Audit Capabilities**:
- Trade execution verification
- Compliance rule checking
- Regulatory reporting
- Risk limit monitoring
- Performance attribution

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Interactive Brokers TWS or IB Gateway
- Required Python packages (see requirements.txt)

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r autotrader/requirements.txt`
3. Configure IBKR connection settings
4. Set up database and configuration files
5. Launch the application: `python autotrader/launcher.py`

### Configuration
The system uses a unified configuration approach:
- Main configuration: `config.json`
- Risk settings: `data/risk_config.json`
- Database: `data/autotrader_stocks.db`

## Usage

### Starting the System
```bash
python autotrader/launcher.py
```

### Running Backtests
```bash
python autotrader/backtest_engine.py --start-date 2023-01-01 --end-date 2023-12-31
```

### Direct Trading Mode
Launch the GUI and use the direct trading interface for manual trading operations.

## System Requirements

### Hardware
- Minimum 4GB RAM
- 2GB free disk space
- Stable internet connection for market data

### Software
- Windows 10/11, macOS, or Linux
- Python 3.8+
- Interactive Brokers TWS or IB Gateway
- Required Python packages (numpy, pandas, ib_insync, etc.)

## Performance Characteristics

### Optimization Features
- Cached technical indicator calculations
- Efficient database operations
- Memory leak prevention
- Resource usage monitoring
- Asynchronous processing

### Scalability
- Modular architecture supports component scaling
- Configurable resource limits
- Multi-threaded processing capabilities
- Event-driven communication system

## Security and Compliance

### Data Security
- Local data storage with encryption options
- Secure API communication
- Audit trail maintenance
- Access control mechanisms

### Trading Compliance
- Risk limit enforcement
- Position monitoring
- Regulatory reporting capabilities
- Compliance rule checking

## Development and Maintenance

### Code Organization
- Modular architecture with clear separation of concerns
- Comprehensive error handling
- Extensive logging and debugging capabilities
- Unit test framework support

### Maintenance Features
- Resource monitoring and cleanup
- Automatic error recovery
- Performance optimization
- Configuration management

## Support and Documentation

### Documentation
- Comprehensive code documentation
- Configuration guides
- API reference documentation
- Troubleshooting guides

### Error Handling
- Comprehensive exception handling
- Detailed error logging
- Recovery mechanisms
- User-friendly error messages

## License and Legal

This software is provided for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.

## Contributing

The project follows standard software development practices:
- Code review process
- Documentation requirements
- Testing standards
- Performance benchmarks

## Future Enhancements

Planned improvements include:
- Additional broker integrations
- Advanced machine learning models
- Enhanced visualization capabilities
- Mobile application support
- Cloud deployment options

This AutoTrader system represents a comprehensive solution for automated trading, combining professional-grade components with user-friendly interfaces to create a powerful trading platform suitable for both research and live trading applications.
