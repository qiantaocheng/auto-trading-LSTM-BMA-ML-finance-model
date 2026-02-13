# TraderApp Architecture

## Overview

TraderApp is a hybrid C#/.NET 10 WPF + Python quantitative trading system that combines ML-based stock prediction, HMM regime detection, automated trade execution via IBKR, earnings surprise scanning (info only), and P2 2-Level Cap ETF rotation. The desktop app provides real-time monitoring, manual trading, and capital tracking with a SQLite backend.

```
+-------------------+     JSON/IPC      +-------------------+
|   WPF Desktop     | <===============> |  Python Scripts   |
|   (C# .NET 10)    |   (subprocess)    |  (ML, IBKR, HMM) |
+-------------------+                   +-------------------+
        |                                       |
        v                                       v
+-------------------+               +-------------------+
|   SQLite DB       |               |  Polygon API      |
|  (TraderApp.db)   |               |  (prices, news)   |
|  WAL mode         |               +-------------------+
+-------------------+                       |
                                            v
                                    +-------------------+
                                    |   IBKR Gateway    |
                                    |  (TWS / Paper)    |
                                    +-------------------+
```

---

## Capital Allocation (75 / 15 / 10)

| Strategy | Service | Capital | Frequency | Risk Control |
|----------|---------|---------|-----------|--------------|
| **ETF Rotation (P2 2-Level Cap)** | `EtfRotationSchedulerService` | **75%** | Every 21 trading days | MA200 cap + vol-target; no HMM; no stop loss |
| **Direct Prediction (Top 10)** | `TradingSchedulerService` | **15%** | After prediction run | HMM risk_gate; 4.5% stop loss; T+5 exit |
| **Earnings Scanner** | `DirectPredictionViewModel` | **10% (info only)** | Manual scan | Display only; no auto-trade |

---

## Project Structure

```
TraderApp/
├── src/
│   ├── Trader.App/                        # WPF UI + DI host
│   │   ├── App.xaml.cs                    # DI, single-instance guard (Mutex), startup
│   │   ├── MainWindow.xaml(.cs)           # Shell: TabControl + Exit button; hide-to-tray on close
│   │   ├── Commands/
│   │   │   └── AsyncRelayCommand.cs
│   │   ├── Converters/
│   │   │   ├── CapitalHistoryGeometryConverter.cs
│   │   │   ├── InverseBooleanConverter.cs
│   │   │   └── ReturnToColorConverter.cs
│   │   ├── ViewModels/
│   │   │   ├── ViewModelBase.cs
│   │   │   ├── ShellViewModel.cs
│   │   │   └── Pages/
│   │   │       ├── MonitorViewModel.cs          # Portfolio monitoring + chart + ETF status
│   │   │       ├── DatabaseViewModel.cs         # Unified position view
│   │   │       └── DirectPredictionViewModel.cs # ML prediction + earnings scan (info only)
│   │   ├── Views/
│   │   │   ├── MonitorView.xaml(.cs)
│   │   │   ├── DatabaseView.xaml(.cs)
│   │   │   └── DirectPredictionView.xaml(.cs)
│   │   ├── Services/
│   │   │   ├── TradingSchedulerService.cs       # Prediction auto-trading + stop loss
│   │   │   └── EtfRotationSchedulerService.cs   # ETF rotation (21-day, P2 2-Level Cap)
│   │   └── Themes/
│   │       └── Colors.xaml
│   │
│   ├── Trader.Core/
│   │   ├── Options/ApplicationOptions.cs        # Config POCOs
│   │   ├── Repositories/TraderDatabase.cs       # SQLite CRUD (all tables, WAL mode)
│   │   └── Services/
│   │       ├── PortfolioService.cs
│   │       ├── PolygonPriceService.cs           # Snapshot + prev-close endpoints
│   │       ├── PolygonCalendarService.cs
│   │       └── WritableOptionsService.cs
│   │
│   └── Trader.PythonBridge/
│       └── Services/
│           ├── PythonPredictionBridge.cs
│           ├── PythonTradingBridge.cs           # SemaphoreSlim serialization + 60s timeout
│           ├── PythonHmmBridge.cs
│           ├── PythonEarningsBridge.cs
│           └── PythonEtfRotationBridge.cs       # etf_rotation_live.py executor
│
├── python/
│   ├── predict_bridge.py
│   ├── trade_bridge.py                          # IBKR CLI with retry + top-level error handling
│   ├── hmm_bridge.py
│   ├── ibkr_portfolio_snapshot.py
│   ├── earnings_scanner.py                      # T+0 to T-2 scanner (info only)
│   └── etf_rotation_live.py                     # P2 2-Level Cap signal generator
│
├── publish_v4/                                  # Published exe + runtime
├── TraderApp.bat                                # Launcher -> publish_v4\Trader.App.exe
└── TraderApp.sln
```

---

## Single-Instance & Lifecycle

### Single-Instance Guard (`App.xaml.cs`)
- Named `Mutex` (`Global\TraderAutoPilot_Instance`) created on startup
- If mutex already owned -> second instance signals `EventWaitHandle` (`Global\TraderAutoPilot_ShowWindow`) then exits immediately
- First instance runs a background daemon thread (`SingleInstanceListener`) that waits on the event and calls `ShowFromTray()` when signaled

### Window Close Behavior
- Pressing the **X button** hides the window to the system tray -- background services **keep running**
- A tray balloon tip reminds the user the app is still active
- Double-clicking the tray icon or choosing "Show Trader" from the tray context menu restores the window
- Auto-show: window reappears automatically Mon-Fri 9:25-16:05 ET (trading hours)

### Termination
- **"Exit App" button** (red, top-right of main window): calls `ExitApplication()` -> stops all `BackgroundService` instances -> terminates process
- **Tray context menu -> "Exit"**: same as above
- `OnExit`: releases Mutex and EventWaitHandle, then calls `_host.StopAsync()`

---

## Robustness Features

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **IBKR serialization** | `SemaphoreSlim(1,1)` in `PythonTradingBridge` | Prevents client ID collision between TradingScheduler and EtfRotationScheduler |
| **Python connection retry** | `_connect_with_retry()` in `trade_bridge.py` | 3 retries with incrementing client IDs on "already in use" error |
| **Process timeout** | `CancellationTokenSource` with 60s/10s in `PythonTradingBridge` | Kills hung Python processes; prevents semaphore deadlock |
| **SQLite WAL** | `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000` | Concurrent read/write safety; auto-retry on SQLITE_BUSY |
| **Sell-in-flight guard** | `ConcurrentDictionary<string,byte>` in `TradingSchedulerService` | Prevents duplicate sells for same symbol during fill wait |
| **Stop loss interval** | 10s | Frequent checks with sell-in-flight guard preventing duplicates |
| **Capital history gating** | Market hours only (Mon-Fri 9:30-16:00 ET), 5-min interval | Prevents disk growth from 24/7 recording |
| **Capital history cleanup** | `TrimCapitalHistory(365)` on app startup | Deletes records older than 1 year |
| **ETF rebalance safety** | DB updated AFTER successful trade fills | Prevents DB/broker desync on trade failure |
| **Position price persistence** | `UpdatePositionPrices()` on each Polygon refresh | Keeps `positions.current_price` up to date |
| **Trade idempotency** | `trade_intents` table with SHA256 intent IDs | Prevents duplicate orders on restart |
| **Partial fill handling** | `FilledQty`/`RemainingQty` in all buy/sell results | Correct DB state even on partial fills |
| **Top-level error handling** | `trade_bridge.py main()` try/except | Ensures valid JSON output on any unexpected error |

---

## File Descriptions

### C# Layer

| File | Purpose |
|------|---------|
| **App.xaml.cs** | DI container via `Host.CreateDefaultBuilder()`. Reads `appsettings.json`. Enforces single-instance with named Mutex + EventWaitHandle. Starts background daemon thread to listen for show-window signals from second-instance attempts. Registers auto-start in Windows startup registry. |
| **MainWindow.xaml.cs** | Shell window. Exit button (terminates everything). `OnClosing` hides to tray instead of exiting. `ShowFromTray()` restores window. `DispatcherTimer` auto-shows during US trading hours. |
| **ShellViewModel.cs** | Composition root holding the three page ViewModels. |
| **MonitorViewModel.cs** | Connects to IBKR, polls portfolio every 5s when connected. Step 1: IBKR portfolio snapshot (Python subprocess, cached until invalidated). Step 2: Polygon batch price fetch for all IBKR symbols. Step 3: Calculate net_liq, update Holdings grid. Step 4: Persist Polygon prices to `positions.current_price`. Step 5: `SyncHoldingsToDatabase` -- writes to `broker_positions` (IBKR mirror), syncs entry prices, auto-seeds positions if empty. Step 6: Capital history to SQLite every 5 min (market hours only, 90-day cleanup on startup). Reports ETF rotation status. |
| **DatabaseViewModel.cs** | Unified position view from `positions` table. Refreshes every 5s with fresh Polygon prices. Manual buy: Polygon price -> IBKR market order -> DB record. Manual sell: IBKR market order -> DB delete. |
| **DirectPredictionViewModel.cs** | Runs ML prediction via `PythonPredictionBridge`, shows top 20 results, stores top 10 as pending buys. Earnings scanner: bulk Polygon news T+0 to T-2 -> filter beats -> display (info only, no auto-trade). |
| **TradingSchedulerService.cs** | `BackgroundService` with two loops: (1) stop loss every 10s -- Polygon prices, 4.5% threshold, `ConcurrentDictionary` sell-in-flight guard; (2) main loop every 60s -- pending buys execution, 5-day rebalance, HMM crisis liquidation. Applies to AutoBuy and Manual positions (NOT ETF). |
| **EtfRotationSchedulerService.cs** | `BackgroundService` for P2 2-Level Cap ETF rotation (75% capital, no HMM, no stop loss). Main loop every 10s: counts NYSE trading days; triggers rebalance every 21 trading days or on first run with no positions. DB updated after successful trade fills (not before). Full decision audit log. State persisted in `etf_rotation_state` DB table. |
| **TraderDatabase.cs** | SQLite CRUD. WAL mode + busy_timeout=5000. All tables. NYSE holiday-aware static methods. `TrimCapitalHistory()` for 90-day cleanup. `UpdatePositionPrices()` for Polygon price persistence. |
| **PythonTradingBridge.cs** | Spawns `trade_bridge.py` subprocess. `SemaphoreSlim(1,1)` serializes all IBKR calls (skips lock for `market-status`). 60s timeout on IBKR commands, 10s on market-status. Kills hung processes on timeout. |
| **PolygonPriceService.cs** | Batch snapshot endpoint -> individual `/v2/aggs/ticker/{sym}/prev` fallback. Free tier: 15-min delayed. |
| **PythonEtfRotationBridge.cs** | Spawns `etf_rotation_live.py`. 60s timeout. Reads stderr for progress, stdout for result JSON. |

### Python Layer

| File | Purpose |
|------|---------|
| **etf_rotation_live.py** | P2 2-Level Cap signal generator. Portfolio-weighted blended vol (not SPY proxy). 2-level MA200 risk cap. Validates per ticker: min 70 bars, max 5 days stale, flags extreme returns >25%. |
| **predict_bridge.py** | ML prediction pipeline: 3200+ tickers, 300+ days OHLCV, 17 alpha factors, ensemble (ElasticNet, XGBoost, CatBoost, LambdaRank, MetaRankerStacker), EMA(4) smoothing, Excel report. |
| **trade_bridge.py** | IBKR trading CLI (ib_async sync API). Commands: `buy`, `sell`, `price`, `cash`, `market-status`. Connection retry (3x with incrementing client IDs). Top-level error handling ensures valid JSON output. |
| **hmm_bridge.py** | 3-state Gaussian HMM on SPY log returns + 10d rolling vol. `risk_gate = (1 - p_crisis_smooth)^2`. Crisis hysteresis. 5-day rebalance counter in `hmm_state.json`. |
| **ibkr_portfolio_snapshot.py** | IBKR connect -> positions + cash -> Polygon delayed prices -> net_liq. |
| **earnings_scanner.py** | Bulk Polygon news (T+0 to T-2), filter: title event word + earnings keywords + beat/miss classification. Info only, no auto-trade. |

---

## Database Schema

```sql
-- Unified positions (AutoBuy, ETF, Manual)
positions (
  id PK AUTOINCREMENT, symbol TEXT UNIQUE,
  strategy TEXT NOT NULL,    -- 'AutoBuy', 'ETF', 'Manual'
  shares INT, entry_price REAL, current_price REAL,
  entered_at TEXT, scheduled_exit TEXT,
  target_weight REAL,        -- ETF only
  last_rebalanced TEXT,      -- ETF only
  note TEXT
)
INDEX idx_positions_strategy ON positions(strategy)

-- Ticker universe
tickers (symbol PK, tag INT, source TEXT, added_at TEXT)

-- Trade log (all strategies)
trades (id PK, symbol, side, quantity, avg_fill_price, note, created_at)

-- ML prediction scores
direct_predictions (id PK, ts INT, snapshot_id, as_of, ticker, score, ema4, created_at)

-- Capital balance history (5-min samples during market hours, 1-year retention)
capital_history (id PK, timestamp, net_liq, cash, stock_value, created_at)
  INDEX idx_capital_history_timestamp ON capital_history(timestamp DESC)

-- ETF rotation persistent state (key-value)
etf_rotation_state (key TEXT PK, value TEXT)

-- IBKR position mirror (refreshed every 5s)
broker_positions (symbol PK, quantity INT, avg_cost REAL, market_value REAL, last_synced TEXT)

-- Trade idempotency (prevents duplicate orders)
trade_intents (
  intent_id TEXT PK, strategy TEXT, symbol TEXT, side TEXT,
  quantity INT, status TEXT DEFAULT 'pending',
  order_id INT, error TEXT, created_at TEXT, executed_at TEXT
)
INDEX idx_trade_intents_lookup ON trade_intents(strategy, symbol, side)

-- Pending buys from prediction (executed by scheduler)
pending_buys (id PK AUTOINCREMENT, symbol TEXT, rank INT, score REAL, created_at TEXT)
```

**SQLite Configuration**: WAL mode (`journal_mode=WAL`) + `busy_timeout=5000` for concurrent read/write safety.

---

## Data Flow

### 1. ETF Rotation Flow (every 21 trading days)
```
EtfRotationSchedulerService main loop (every 10s)
  -> count NYSE trading day (once per unique date, market open)
  -> if days >= 21 OR no positions:
    -> idempotency check: last_rebalance_run_date == today? -> skip
    -> PythonEtfRotationBridge.GetTargetWeightsAsync()
      -> etf_rotation_live.py --polygon-key KEY
        -> Polygon API: 400 days bars for SPY + 7 ETFs
        -> compute portfolio-weighted blended vol
        -> compute vol_target + 2-level MA200 risk_cap
      <- JSON {exposure, etf_weights, asof_trading_day, ...}
    -> apply deadband/step/min-hold; log decision audit
    -> PythonTradingBridge.GetCashAsync() -> net_liq
    -> PolygonPriceService.GetPricesAsync(etf_tickers)
    -> compute target_shares per ETF
    -> execute sells first -> execute buys (IBKR market orders)
    -> DB: update positions AFTER successful fills
    -> reset counter, record last_rebalance_run_date
```

### 2. Portfolio Monitoring (every 5s when connected)
```
DispatcherTimer -> MonitorViewModel.RefreshAsync()
  -> (first call / invalidated) PortfolioService.GetSnapshotAsync()
    -> ibkr_portfolio_snapshot.py -> IBKR + Polygon
  <- {cash, positions[symbol, qty]}
  -> PolygonPriceService.GetPricesAsync(ibkr_symbols)
  <- {symbol: price}
  -> net_liq = cash + sum(qty * price) -> update Holdings grid
  -> UpdatePositionPrices -> persist Polygon prices to positions.current_price
  -> CaptureCapitalPoint -> capital_history (5-min, market hours only)
  -> SyncHoldingsToDatabase -> broker_positions (IBKR mirror only)
  -> UpdateReturns -> 1D/1W/1M/6M/1Y from capital_history
```

### 3. Stop Loss (every 30s)
```
TradingSchedulerService.CheckStopLossAsync()
  -> DB: GetPositions() where strategy != 'ETF'
  -> PolygonPriceService.GetPricesAsync(symbols)
  -> for each position:
    -> if symbol in _sellingSymbols -> skip (sell already in flight)
    -> if currentPrice <= entryPrice * 0.955:
      -> _sellingSymbols.TryAdd(symbol)
      -> PythonTradingBridge.SellAsync(symbol, shares) [60s timeout]
      -> DB: InsertTrade(SELL, "StopLoss"), DeletePosition
      -> _sellingSymbols.TryRemove(symbol)
```

### 4. Prediction -> Pending Buys -> Auto-Buy (15% capital)
```
User clicks "Run Prediction"
  -> PythonPredictionBridge.RunAsync() -> predict_bridge.py
  <- {top20, top10, excel_path}
  -> DB: direct_predictions, pending_buys (top 10)

TradingSchedulerService main loop (every 60s):
  -> if no AutoBuy positions + pending_buys exist:
    -> HMM risk_gate check (block if crisis or risk_gate < 0.05)
    -> GetCashAsync -> budget = net_liq * 0.15 / 10
    -> for each pending buy:
      -> shares = floor(budget * risk_gate / price)
      -> PythonTradingBridge.BuyAsync(symbol, shares) [60s timeout]
      -> DB: InsertPosition("AutoBuy", T+5), InsertTrade
    -> ClearPendingBuys
```

---

## ETF Rotation -- P2 2-Level Cap + Portfolio B

### Portfolio B (static strategic weights)
```
QQQ  25%   USMV 25%   QUAL 20%   PDBC 15%
DBA   5%   COPX  5%   URA   5%
Cash (BIL) = 1.0 - sum(etf_weights)
```

### Signal Calculation (etf_rotation_live.py)
1. Determine `asof_date` = last COMPLETED trading day (ET timezone, 16:00 cutoff)
2. Fetch 400 days of bars for SPY + all 7 ETFs; filter to <= asof_date
3. Compute portfolio-weighted blended vol:
   - `portfolio_return_t = sum(strategic_weight_i * log_return_i_t)` for all 7 ETFs
   - `blended_vol = 0.7 * vol20 + 0.3 * vol60`, clamped [0.08, 0.40]
4. `vol_target_exposure = min(0.12 / blended_vol, 1.0)`
5. 2-level MA200 risk cap (SPY): >= 0 -> 1.0; (-5%, 0) -> 0.60; < -5% -> 0.30
6. `raw_exposure = min(vol_target_exposure, risk_cap, 0.95)`

### Deadband + Step + Min-Hold (matches backtest)
1. **Min-hold**: if `days_since_rebalance < 5` AND `|delta| < 30%` -> skip
2. **Deadband**: if `|delta| < 5%` -> skip
3. **Max step**: if `|delta| > 15%` -> clamp to +/-15%

### Backtest Result
- **Sharpe 0.85, CAGR 12.64%, MaxDD -12.6%**
- Period: 2022-02-24 to 2026-02-09, 10 bps cost

---

## Configuration (appsettings.json)

```json
{
  "Python": {
    "Executable": "D:/trade/venv/Scripts/python.exe",
    "PredictScript": "D:/trade/TraderApp/python/predict_bridge.py",
    "TradingScript": "D:/trade/TraderApp/python/trade_bridge.py",
    "HmmScript": "D:/trade/TraderApp/python/hmm_bridge.py",
    "EarningsScript": "D:/trade/TraderApp/python/earnings_scanner.py",
    "EtfRotationScript": "D:/trade/TraderApp/python/etf_rotation_live.py",
    "PortfolioScript": "D:/trade/TraderApp/python/ibkr_portfolio_snapshot.py",
    "ParquetPath": "D:/trade/data/factor_exports/polygon_full_features_T5.parquet",
    "DefaultSnapshot": "<snapshot-uuid>"
  },
  "Polygon": { "ApiKey": "<key>" },
  "Database": { "FileName": "TraderApp.db" },
  "IBKR": {
    "Mode": "Paper",
    "Host": "127.0.0.1",
    "PaperPort": 4002,
    "LivePort": 4001,
    "ClientId": 3136
  }
}
```

**IBKR Client IDs**: All trading uses ClientId=3136 (serialized by SemaphoreSlim). Portfolio snapshot uses ClientId+1 (3137).

---

## Key Design Decisions

1. **Python subprocess IPC**: Heavy ML and IBKR work done in Python via `ProcessStartInfo` + JSON stdout/stderr. `ib_async` scripts use synchronous API (`ib.connect`, `ib.sleep`) -- NOT async/await.

2. **Polygon API for prices**: All price data from Polygon (15-min delayed free tier). Not IBKR real-time. Avoids IBKR rate limits and client ID conflicts.

3. **Unified positions table**: Single `positions` table with `strategy` column (AutoBuy, ETF, Manual). Replaced separate `auto_positions` + `etf_positions` tables. Migration runs automatically.

4. **SQLite as source of truth**: `positions` is the ledger for all holdings. `broker_positions` is a read-only IBKR mirror (never modified by trading logic). WAL mode + busy_timeout for concurrent safety.

5. **ETF rotation uses NO HMM**: P2 2-Level Cap has its own risk management (MA200 2-level cap + vol-target). HMM only gates direct prediction strategy.

6. **No stop loss for ETF rotation**: Matches backtest. Stop loss only for AutoBuy and Manual positions.

7. **Serialized IBKR access**: Single `SemaphoreSlim(1,1)` in `PythonTradingBridge` ensures only one Python process connects to IBKR at a time. `market-status` command skips the lock (no IBKR connection needed).

8. **Earnings scanner is info-only**: Backtest showed no edge for earnings auto-trading. Scanner displays T+0 to T-2 earnings beat stocks for manual review.

9. **DB updated after trade fills**: ETF rebalance updates DB positions only after successful IBKR fills, preventing DB/broker desync on trade failure.

10. **10-second stop loss interval**: Frequent checks with `ConcurrentDictionary` sell-in-flight guard preventing duplicate sells during fill wait.

---

## Launch & Build

```cmd
# Launch published app (background services included)
D:\trade\TraderApp\TraderApp.bat          -> publish_v4\Trader.App.exe

# Build
dotnet build src\Trader.App\Trader.App.csproj -c Release

# Publish (close app first -- DLLs will be locked otherwise)
dotnet publish src\Trader.App\Trader.App.csproj -c Release -r win-x64 ^
  --self-contained false -o publish_v4 --nologo

# If app is running and locks DLLs, publish to alternate folder:
dotnet publish src\Trader.App\Trader.App.csproj -c Release -r win-x64 ^
  --self-contained false -o publish_v5 --nologo
```

**Note**: The app registers itself in `HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run` as `TraderAutoPilot` for auto-start at Windows boot.
