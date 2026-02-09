# TraderApp Usage Guide

## Entry Point & Launching
- The WPF client lives in `D:\trade\TraderApp\src\Trader.App`.
- Once the .NET 8 SDK is installed, you can run it directly from the command line:
  ```powershell
  cd D:\trade\TraderApp
  dotnet restore
  dotnet run --project src\Trader.App\Trader.App.csproj
  ```
- `App.xaml` / `App.xaml.cs` hosts the entry point. It bootstraps `Microsoft.Extensions.Hosting`, reads `appsettings.json`, ensures `TraderApp.db` exists beside the executable, and shows `MainWindow`.
- When you publish the project (`dotnet publish -c Release`), the output folder will contain `TraderApp.exe`. Launch that EXE to start the software; the bundled `TraderApp.db` travels with it.

## Configuration
- `src/Trader.App/appsettings.json` holds runtime settings:
  - `Python.Executable` – path to the venv Python (`D:/trade/venv/Scripts/python.exe`).
  - `Python.PredictScript` – the bridge script (`D:/trade/TraderApp/python/predict_bridge.py`).
  - `Python.DefaultSnapshot` – LambdaRank snapshot ID used by default.
  - `Polygon.ApiKey` – API key for `GET /v1/marketstatus/*` (currently `isFExbaO1xdmrV6f6p3zHCxk8IArjeowQ1`).
  - `Database.FileName` – SQLite file copied next to the exe (`TraderApp.db`).

## Major Components
- **Direct Prediction Page** (`DirectPredictionViewModel`)
  - Runs the Python snapshot bridge through `PythonPredictionBridge`.
  - Displays last run time, Excel output path, Top‑20 tickers, and status.
  - Persists LambdaRank Top‑10 into SQLite via `TraderDatabase.ReplaceAutoTickers`.
- **Monitor Page** (`MonitorViewModel`)
  - Polls `IPortfolioService` every 5 seconds.
  - Updates net liquidation, cash, and holdings grid in real time (currently mocked via `MockPortfolioService`; swap in your IBKR implementation).
- **Database Page** (`DatabaseViewModel`)
  - Lists all tickers from `TraderApp.db`.
  - Supports manual add (tagged = 1), delete (trigger sell logic later), and refresh operations.

## Supporting Services
- `TraderDatabase` (SQLite repo): ensures schema (`tickers`, `auto_positions`, `trades`), exposes helper methods for auto/manual tickers.
- `PolygonCalendarService`: wraps `GET /v1/marketstatus/now` and `/upcoming` to feed trading calendars.
- `PythonPredictionBridge`: launches the bridge script inside the configured venv; expects JSON payload `{ run_id, as_of, excel_path, top20, top10 }`.
- `MockPortfolioService`: placeholder returning synthetic holdings; implement `IbkrPortfolioService` and register it to use live IBKR data.

## Python Assets
- `python/predict_bridge.py`: stub wrapper; replace body with the actual LambdaRank/T+10 pipeline. C# already expects its JSON output.
- `python/polygon_calendar.py`: helper to query Polygon’s Massive domain for market status/holidays.
- `scripts/create_trader_db.py`: reinitializes the SQLite DB schema. Executed once (via `D:\trade\venv\Scripts\python.exe scripts\create_trader_db.py`).

## Next Steps
1. Replace the Python stub with the real Direct Predict workflow + Excel export.
2. Implement `IPortfolioService` on top of IBKR Trader Workstation so the Monitor tab shows live holdings.
3. Layer in the trading scheduler (50% capital allocation, -4.5% stops, T+5 sell) using `PolygonCalendarService` for market hours.

Once the .NET SDK is available, open `TraderApp.sln` in Visual Studio or run the commands above to build/run the app.
