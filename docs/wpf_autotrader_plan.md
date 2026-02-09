# WPF AutoTrader Redesign Plan

## Objectives
- Rebuild the desktop client as a modern C#/.NET 8 WPF application with three focussed pages.
- Preserve the proven LambdaRank snapshot workflow and IBKR account configuration, but isolate it behind services that the WPF app controls.
- Keep the existing SQLite schema (`stock_lists`, `stocks`, `tickers`, `trading_configs`, `trade_history`, `risk_configs`) so downstream scripts and tooling remain compatible.
- Automate the daily Direct Predict -> Buy -> Hold 5 trading days -> Sell cycle using U.S. market hours, while allowing manual overrides that do not get force-liquidated.

## Technology Stack
- **UI**: WPF (.NET 8, MVVM pattern with CommunityToolkit.Mvvm to simplify bindings).
- **Interop**: Python integration via `ProcessStartInfo` or gRPC wrapper that calls the current `UltraEnhancedQuantitativeModel` snapshot (no code rewrite inside `bma_models`).
- **Data**: System.Data.SQLite (or Microsoft.Data.Sqlite) pointing to the existing database file under `data/`.
- **Trading API**: Official IBKR `IBApi` C# client (same “grammar” you use with Trader Workstation today).
- **Scheduling**: `System.Threading.Channels` + hosted background services (Worker pattern) for timers, plus a U.S. market calendar helper (NodaTime or self-managed list of holidays).
- **Logging**: Serilog with rolling files + real-time feed into the Monitor page.

## Solution Layout (new files)
```
/TraderApp.sln
  /src/Trader.App            # WPF UI (views, viewmodels, resources)
  /src/Trader.Core           # Services, schedulers, IBKR wrappers, repositories
  /src/Trader.PythonBridge   # Thin wrapper for calling Python scripts / parsing output
  /python/predict_bridge.py  # Reusable python entry that emits JSON + Excel (calls existing modules)
```

## Core Services
1. **PythonPredictionService**
   - Launches `python predict_bridge.py --snapshot <id> --ema-days 4 --top-n 20 --output <dir>`.
   - Bridge script returns a JSON payload `{ date, excel_path, top20:[{ticker,score,ema}], top10 }` via stdout.
   - Service parses JSON, stores the Excel file path, and raises an event consumed by the Direct Prediction page and TradingOrchestrator.

2. **TradingOrchestrator**
   - Tracks account cash + equity through `IbkrPortfolioService`.
   - Every trading day: 
     1. **07:00 ET** – call `PythonPredictionService`. Persist Top-20 to Excel (matching current formatting) and store Top-10 (LambdaRank EMA) into SQLite `tickers` table with `tag = 0`.
     2. **09:30 ET (market open)** – calculate capital bucket: `floor(0.5 * available_cash / 10 / last_price)` per stock. Submit market buy orders for each of the 10 tickers using IBKR API and immediately submit a bracket stop at -4.5% from fill price.
     3. **T+5 trading days, 15:30 ET** – for every auto-generated ticker still held (tag 0, not manually protected) submit market sell for entire remaining quantity, cancel residual stops, and clear from DB.
     4. Loop: once sells finish, run prediction for the next session.
   - Manual symbols (tag=1) are ignored by the forced sell cycle until the user explicitly deletes them.

3. **MarketCalendarService**
   - Encapsulates U.S. equity trading hours, holidays, early closes. Provides helpers (`NextTradingOpen`, `NextTradingCloseMinus(TimeSpan.FromMinutes(30))`, `AddTradingDays(date, 5)`).
   - Drives TradingOrchestrator scheduler to align timers with real exchange time.

4. **IbkrPortfolioService**
   - Wraps IBKR client connection, order placement, and streaming account updates.
   - Provides events for fills, portfolio updates, cash balance, and health state for the Monitor page.

5. **DatabaseRepository**
   - Thin C# ORM-like layer mapping existing tables. Adds a `tickers.tag` column (if not already there, run migration once) to flag manual = 1.
   - Exposes high-level methods: `SaveTopTenAuto(List<TickerRank>)`, `AddManualTicker(symbol, price, quantity)`, `DeleteTicker(symbol, reason)`, `RecordTrade(...)`.

6. **ManualTradeService**
   - Invoked from the Database page when the user enters a ticker. Queries current market price via IBKR or Polygon, shows quote, lets user enter quantity, then sends a market order tagged as manual (tag=1) so it will not auto-liquidate.

7. **LogAggregator**
   - Central `Channel<LogEntry>` that receives logs from all services; Monitor page binds to a `ReadOnlyObservableCollection` updated on the UI thread every 5 seconds.

## Visual Design Alignment
- Reuse the layout/spacing guidelines from `D:/trade/design` so the WPF surfaces mirror the reference web design (color palette, typography, card spacing).
- Shared resources (brushes, fonts) should be extracted into WPF resource dictionaries that derive their values from the design tokens maintained under `design/src`.
- Charts, tabs, and cards must avoid emoji and stick to the clean enterprise aesthetic defined in the design package.

## UI Pages
### Page 1 – Direct Prediction
- **Run Prediction** button (and schedule status).
- Fields: Snapshot ID (pre-filled), EMA window (fixed to 4 but editable), Top-N (20).
- Shows latest run timestamp, Excel file path (open folder), Top-20 grid (ticker, LambdaRank score, EMA score).
- Contains `“Push Top 10 to Database”` confirmation state + preview of resulting buy basket.
- Displays next scheduled buy window and T+5 sell window computed by `MarketCalendarService`.

### Page 2 – Trading Monitor
- Real-time account card: net liq, cash, buying power, daily P&L, connection status (IBKR / Polygon / Python service).
- Position table (updated every 5 s): ticker, qty, avg price, last price, unrealized P&L, tag.
- Active orders grid + stop levels.
- Live log console (tail of `LogAggregator`).
- **Capital Growth Chart**: a line chart styled per the `design` system, plotting net-liq history sampled every 5 minutes (see “Capital Growth Visualization” section). Users can toggle between 1D/1W/1M windows; tooltips show timestamp, account value, and delta vs. the prior point.

### Page 3 – Database / Manual Control
- Grid bound to `tickers` table, showing `symbol`, `added_at`, `tag` (Auto / Manual), “protected until” date.
- Buttons:
  - **Manual Add**: fetch price, let user enter qty, sets tag=1, places immediate market order.
  - **Delete Selected**: removes from DB and triggers immediate sell (unless user confirms skip because they handle manually).
  - **Import/Export**: simple CSV of ticker list.
- Warning banner when automated cleanup removes a ticker (due to T+5 sell) so operators know what changed.

### Capital Growth Visualization
- Persist the 5-minute snapshots returned by `IbkrPortfolioService` (net-liq, cash) into a lightweight table (for example, `capital_history`) so the chart can draw from rolling history even after restarts.
- Sampling frequency: every 5 minutes while markets are open; pause once the market closes and resume at pre-market.
- Rendering: simple polyline with a filled area under the curve, axis/legend styled per `design` tokens; no emoji, keep the enterprise tone.
- Provide export to CSV for compliance (selection range only) via the Monitor tab.


## Python Bridge Contract
- Script location: `python/predict_bridge.py` (new file) imports current `bma_models` modules, exposes CLI args for snapshot ID, EMA window, etc., and prints JSON to stdout.
- WPF launches with environment variables ensuring it runs inside the existing virtual environment.
- Output JSON example:
```json
{
  "run_id": "2026-02-09T07:05:00Z",
  "as_of": "2026-02-08",
  "excel_path": "D:/trade/results/direct_predict_top30_20260209_070500.xlsx",
  "top20": [ {"ticker": "AAPL", "score": 0.9123, "ema4": 0.8877}, ... ],
  "top10": ["AAPL","MSFT",...]
}
```
- Service writes `top20` into Excel using the existing Python helper so visual parity is preserved.

## Trading Rules Implementation
1. **Capital Slice**: `allocation_per_stock = floor((cash * 0.50) / 10 / price)` (skip ticker if result < 1 share). Keep a remainder ledger so unused cash carries forward.
2. **Stops**: submit bracket or stop order at `price * 0.955`. On partial fills, adjust stop accordingly.
3. **Sell cycle**: maintain metadata table `auto_positions` with `entered_at_trading_day`, `scheduled_exit` = `entered_at + 5 trading days`. Scheduler wakes up daily at 15:00 ET, finds rows with exit = today, and sends market sells at 15:30 ET.
4. **Manual items**: `tag = 1`. They can coexist with auto ones, show up in Monitor page, and must be removed manually. Deleting them via UI triggers a sell right away (unless user unchecks “Sell now”).
5. **DB delete hook**: Repository raises event when symbols are removed so `TradingOrchestrator` can enqueue liquidation orders even if deletion is triggered outside the UI (e.g., external script).

## Error Handling & Health
- Retry Python service failures with exponential backoff (notify operator on page 1).
- IBKR connection watcher: auto-reconnect on disconnect events; show banner on Monitor page.
- Database locking: wrap in async mutex to avoid UI thread blocking.

## Implementation Roadmap
1. **Scaffold solution**: create WPF app + core libraries, configure DI (Microsoft.Extensions.Hosting), connect logging, load config (JSON + env overrides).
2. **DatabaseRepository**: port schema, add optional `tag` column migration, provide CRUD.
3. **Python Bridge**: implement CLI wrapper, test JSON output; add C# service to invoke and parse.
4. **IBKR services**: wrap connection/login flow, expose order/portfolio APIs matching existing “grammar”.
5. **TradingOrchestrator & Calendar**: implement scheduling, capital calculations, stop management, DB syncing.
6. **UI Pages**: build Direct Prediction page first, then Monitor, then Database UI, wiring them to ViewModels.
7. **Manual trade & delete flows**: ensure tag=1 logic works and deletions trigger sells.
8. **End-to-end dry run**: simulate a trading day with mock IBKR + Python responses before hitting production.

## Outstanding Clarifications (if needed later)
- Confirm final location of the Python virtual environment to activate before running `predict_bridge.py`.
- Provide holiday list updates when exchanges announce new closures.
- Decide whether the Excel export should be stored in the same `results/` directory or a new WPF-managed folder.
