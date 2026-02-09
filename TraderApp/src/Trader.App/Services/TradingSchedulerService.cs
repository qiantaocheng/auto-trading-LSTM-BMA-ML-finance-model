using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Trader.App.ViewModels.Pages;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Trader.PythonBridge.Services;

namespace Trader.App.Services;

public sealed class TradingSchedulerService : BackgroundService
{
    private readonly PythonTradingBridge _tradingBridge;
    private readonly PythonHmmBridge _hmmBridge;
    private readonly PolygonPriceService _polygonPrices;
    private readonly TraderDatabase _database;
    private readonly MonitorViewModel _monitor;
    private readonly ILogger<TradingSchedulerService> _logger;

    private const double CapitalAllocationRatio = 0.5;
    private const int MaxPositions = 10;
    private const double StopLossPct = 0.045;
    private const int RebalanceIntervalDays = 5;

    // HMM state
    private HmmResult? _latestHmm;
    private DateTime _lastHmmDate;

    public TradingSchedulerService(
        PythonTradingBridge tradingBridge,
        PythonHmmBridge hmmBridge,
        PolygonPriceService polygonPrices,
        TraderDatabase database,
        MonitorViewModel monitor,
        ILogger<TradingSchedulerService> logger)
    {
        _tradingBridge = tradingBridge;
        _hmmBridge = hmmBridge;
        _polygonPrices = polygonPrices;
        _database = database;
        _monitor = monitor;
        _logger = logger;
    }

    public bool IsAutoTradingEnabled
    {
        get => _monitor.IsAutoTradingEnabled;
        set
        {
            Application.Current.Dispatcher.Invoke(() => _monitor.IsAutoTradingEnabled = value);
        }
    }

    /// <summary>
    /// Called by DirectPredictionViewModel after a successful prediction to auto-buy top 10.
    /// Applies HMM risk_gate to position sizing.
    /// </summary>
    public async Task ExecuteAutoBuyAsync(IReadOnlyList<string> top10Tickers)
    {
        if (!IsAutoTradingEnabled)
        {
            _logger.LogInformation("Auto-trading disabled, skipping auto-buy");
            return;
        }

        try
        {
            // Run HMM assessment if not done today
            await RunDailyHmmAsync().ConfigureAwait(false);

            // Block buying in crisis mode
            if (_latestHmm is { CrisisMode: true })
            {
                ReportEvent("Auto-Buy", "BLOCKED: Crisis mode active — no buying allowed");
                return;
            }

            var riskGate = _latestHmm?.RiskGate ?? 1.0;
            if (riskGate < 0.05)
            {
                ReportEvent("Auto-Buy", $"BLOCKED: risk_gate={riskGate:F4} too low — no buying");
                return;
            }

            ReportEvent("Auto-Buy", $"Starting auto-buy (risk_gate={riskGate:F4})...");

            var status = await _tradingBridge.GetMarketStatusAsync().ConfigureAwait(false);
            if (!status.IsOpen)
            {
                ReportEvent("Auto-Buy", "Market closed, skipping auto-buy");
                return;
            }

            var cashInfo = await _tradingBridge.GetCashAsync().ConfigureAwait(false);
            if (cashInfo.Error is not null)
            {
                ReportEvent("Auto-Buy", $"Failed to get cash: {cashInfo.Error}");
                return;
            }

            var netLiq = (decimal)cashInfo.NetLiq;
            var budgetPerPosition = netLiq * (decimal)CapitalAllocationRatio / MaxPositions;
            ReportEvent("Auto-Buy", $"Net Liq: {netLiq:C}, Budget/pos: {budgetPerPosition:C}, risk_gate: {riskGate:F4}");

            var existingPositions = _database.GetAutoPositions();
            var existingSymbols = new HashSet<string>(existingPositions.Select(p => p.Symbol), StringComparer.OrdinalIgnoreCase);

            var bought = 0;
            foreach (var ticker in top10Tickers)
            {
                if (string.IsNullOrWhiteSpace(ticker)) continue;
                var symbol = ticker.Trim().ToUpperInvariant();

                if (existingSymbols.Contains(symbol))
                {
                    ReportEvent("Auto-Buy", $"Skipping {symbol} (already held)");
                    continue;
                }

                try
                {
                    var polygonPrice = await _polygonPrices.GetPriceAsync(symbol).ConfigureAwait(false);
                    if (polygonPrice is null or <= 0)
                    {
                        ReportEvent("Auto-Buy", $"Skipping {symbol} (no Polygon price)");
                        continue;
                    }

                    var price = polygonPrice.Value;
                    var shares = (int)Math.Floor(budgetPerPosition * (decimal)riskGate / price);
                    if (shares < 1)
                    {
                        ReportEvent("Auto-Buy", $"Skipping {symbol} (shares=0 after risk_gate)");
                        continue;
                    }

                    bought++;
                    ReportEvent("Auto-Buy", $"Buying {symbol}: {shares} shares @ ~{price:C} ({bought}/{top10Tickers.Count})");

                    var buyResult = await _tradingBridge.BuyAsync(symbol, shares).ConfigureAwait(false);
                    if (!buyResult.Success)
                    {
                        ReportEvent("Auto-Buy", $"FAILED {symbol}: {buyResult.Error}");
                        continue;
                    }

                    var fillPrice = buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : price;
                    _database.InsertAutoPosition(symbol, shares, fillPrice);
                    _database.InsertTrade(symbol, "BUY", shares, fillPrice, "AutoBuy");
                    existingSymbols.Add(symbol);

                    ReportEvent("Auto-Buy", $"Bought {symbol}: {shares} shares @ {fillPrice:C}");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to buy {Symbol}", symbol);
                    ReportEvent("Auto-Buy", $"Error buying {symbol}: {ex.Message}");
                }
            }

            UpdatePositionCount();
            ReportEvent("Auto-Buy", $"Complete: bought {bought} positions (risk_gate={riskGate:F4})");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Auto-buy failed");
            ReportEvent("Auto-Buy", $"FAILED: {ex.Message}");
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("TradingSchedulerService started (HMM + 5-day rebalance)");

        // Run HMM assessment immediately on startup
        try
        {
            _logger.LogInformation("Running initial HMM assessment on startup...");
            await RunDailyHmmAsync().ConfigureAwait(false);
            _logger.LogInformation("Initial HMM assessment complete");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to run initial HMM assessment");
        }

        // Run two loops: stop loss every 5s, main logic every 60s
        var stopLossTask = RunStopLossLoopAsync(stoppingToken);
        var mainTask = RunMainLoopAsync(stoppingToken);
        await Task.WhenAll(stopLossTask, mainTask).ConfigureAwait(false);

        _logger.LogInformation("TradingSchedulerService stopped");
    }

    private async Task RunStopLossLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(5_000, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            // Stop loss and scheduled exit ALWAYS run regardless of auto-trading toggle
            if (!_monitor.IsConnected) continue;

            try
            {
                await CheckStopLossAsync(ct).ConfigureAwait(false);
                await CheckScheduledExitsAsync(ct).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Stop loss check error");
            }
        }
    }

    private async Task RunMainLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(60_000, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            if (!IsAutoTradingEnabled) continue;

            try
            {
                var status = await _tradingBridge.GetMarketStatusAsync(ct).ConfigureAwait(false);
                if (!status.IsOpen) continue;

                // Run HMM once per market day
                await RunDailyHmmAsync().ConfigureAwait(false);

                // Crisis early exit (Plan B): sell everything immediately
                if (_latestHmm is { CrisisMode: true })
                {
                    var positions = _database.GetAutoPositions();
                    if (positions.Count > 0)
                    {
                        await ExecuteCrisisExitAsync(ct).ConfigureAwait(false);
                    }
                }

                // 5-day rebalance check
                if (_latestHmm is not null && _latestHmm.RebalanceDayCounter >= RebalanceIntervalDays)
                {
                    await ExecuteRebalanceAsync(ct).ConfigureAwait(false);
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Trading scheduler main loop error");
            }
        }
    }

    private async Task RunDailyHmmAsync()
    {
        var today = DateTime.UtcNow.Date;
        if (_lastHmmDate == today && _latestHmm is not null) return;

        try
        {
            ReportEvent("HMM", "Running daily risk assessment...");

            _latestHmm = await _hmmBridge.GetRiskAssessmentAsync(
                onProgress: p => ReportEvent("HMM", p.Detail)).ConfigureAwait(false);

            _lastHmmDate = today;

            if (_latestHmm.Error is not null)
            {
                ReportEvent("HMM", $"WARNING: {_latestHmm.Error}");
                return;
            }

            var rebalDaysRemaining = RebalanceIntervalDays - _latestHmm.RebalanceDayCounter;
            if (rebalDaysRemaining < 0) rebalDaysRemaining = 0;

            ReportEvent("HMM",
                $"State={_latestHmm.HmmState} RiskGate={_latestHmm.RiskGate:F4} " +
                $"P(Crisis)={_latestHmm.PCrisisSmooth:F4} CrisisMode={_latestHmm.CrisisMode} " +
                $"Rebalance in {rebalDaysRemaining}d");

            try
            {
                Application.Current.Dispatcher.Invoke(() =>
                    _monitor.ReportHmmStatus(
                        _latestHmm.HmmState,
                        _latestHmm.RiskGate,
                        _latestHmm.PCrisisSmooth,
                        _latestHmm.CrisisMode,
                        rebalDaysRemaining));
            }
            catch
            {
                // Dispatcher may not be available
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "HMM assessment failed");
            ReportEvent("HMM", $"FAILED: {ex.Message}");
        }
    }

    /// <summary>
    /// Every 5 trading days: sell all positions, buy new Top10 with risk_gate sizing.
    /// </summary>
    private async Task ExecuteRebalanceAsync(CancellationToken ct)
    {
        ReportEvent("Rebalance", "Starting 5-day rebalance...");

        // 1. Sell all existing positions
        var positions = _database.GetAutoPositions();
        if (positions.Count > 0)
        {
            ReportEvent("Rebalance", $"Selling {positions.Count} existing positions...");
            foreach (var pos in positions)
            {
                try
                {
                    var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                    var fillPrice = result.Success && result.AvgPrice > 0
                        ? (decimal)result.AvgPrice
                        : pos.EntryPrice;

                    _database.InsertTrade(pos.Symbol, "SELL", pos.Shares, fillPrice, "Rebalance");
                    _database.DeleteAutoPosition(pos.Symbol);
                    ReportEvent("Rebalance", $"Sold {pos.Symbol}: {pos.Shares} shares @ {fillPrice:C}");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Rebalance sell failed for {Symbol}", pos.Symbol);
                    ReportEvent("Rebalance", $"Error selling {pos.Symbol}: {ex.Message}");
                }
            }
        }

        // 2. If in crisis mode, don't buy
        var riskGate = _latestHmm?.RiskGate ?? 1.0;
        if (_latestHmm is { CrisisMode: true } || riskGate < 0.05)
        {
            ReportEvent("Rebalance", $"Skipping buy phase: crisis_mode={_latestHmm?.CrisisMode} risk_gate={riskGate:F4}");
            ResetRebalanceCounter();
            UpdatePositionCount();
            return;
        }

        // 3. Get latest Top10 tickers from database
        var tickers = _database.GetTickerRecords()
            .Where(t => t.Tag == 0)
            .Select(t => t.Symbol)
            .Take(MaxPositions)
            .ToList();

        if (tickers.Count == 0)
        {
            ReportEvent("Rebalance", "No tickers available — run prediction first");
            ResetRebalanceCounter();
            UpdatePositionCount();
            return;
        }

        // 4. Buy with risk_gate
        var cashInfo = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
        if (cashInfo.Error is not null)
        {
            ReportEvent("Rebalance", $"Failed to get cash: {cashInfo.Error}");
            ResetRebalanceCounter();
            return;
        }

        var netLiq = (decimal)cashInfo.NetLiq;
        var budgetPerPosition = netLiq * (decimal)CapitalAllocationRatio / MaxPositions;
        ReportEvent("Rebalance", $"Buying Top{tickers.Count} with risk_gate={riskGate:F4}, budget/pos={budgetPerPosition:C}");

        var bought = 0;
        foreach (var symbol in tickers)
        {
            try
            {
                var polygonPrice = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                if (polygonPrice is null or <= 0) continue;

                var price = polygonPrice.Value;
                var shares = (int)Math.Floor(budgetPerPosition * (decimal)riskGate / price);
                if (shares < 1) continue;

                var buyResult = await _tradingBridge.BuyAsync(symbol, shares, ct).ConfigureAwait(false);
                if (!buyResult.Success)
                {
                    ReportEvent("Rebalance", $"FAILED {symbol}: {buyResult.Error}");
                    continue;
                }

                var fillPrice = buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : price;
                _database.InsertAutoPosition(symbol, shares, fillPrice);
                _database.InsertTrade(symbol, "BUY", shares, fillPrice, "Rebalance");
                bought++;
                ReportEvent("Rebalance", $"Bought {symbol}: {shares} shares @ {fillPrice:C}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Rebalance buy failed for {Symbol}", symbol);
                ReportEvent("Rebalance", $"Error buying {symbol}: {ex.Message}");
            }
        }

        ResetRebalanceCounter();
        UpdatePositionCount();
        ReportEvent("Rebalance", $"Rebalance complete: {bought} positions bought (risk_gate={riskGate:F4})");
    }

    /// <summary>
    /// Crisis early exit (Plan B): sell all positions urgently.
    /// </summary>
    private async Task ExecuteCrisisExitAsync(CancellationToken ct)
    {
        var positions = _database.GetAutoPositions();
        if (positions.Count == 0) return;

        ReportEvent("CRISIS EXIT", $"Selling {positions.Count} positions urgently!");

        foreach (var pos in positions)
        {
            try
            {
                var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                var fillPrice = result.Success && result.AvgPrice > 0
                    ? (decimal)result.AvgPrice
                    : pos.EntryPrice;

                _database.InsertTrade(pos.Symbol, "SELL", pos.Shares, fillPrice, "CrisisExit");
                _database.DeleteAutoPosition(pos.Symbol);

                ReportEvent("CRISIS EXIT", result.Success
                    ? $"Sold {pos.Symbol} @ {fillPrice:C}"
                    : $"FAILED {pos.Symbol}: {result.Error}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Crisis exit sell failed for {Symbol}", pos.Symbol);
                ReportEvent("CRISIS EXIT", $"Error selling {pos.Symbol}: {ex.Message}");
            }
        }

        UpdatePositionCount();
        ReportEvent("CRISIS EXIT", "Crisis liquidation complete");
    }

    private async Task CheckStopLossAsync(CancellationToken ct)
    {
        var positions = _database.GetAutoPositions();
        if (positions.Count == 0) return;

        // Batch fetch all prices from Polygon API (single HTTP call)
        var symbols = positions.Select(p => p.Symbol).ToList();
        Dictionary<string, decimal> prices;
        try
        {
            prices = await _polygonPrices.GetPricesAsync(symbols, ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Polygon price fetch failed for stop loss check");
            return;
        }

        foreach (var pos in positions)
        {
            try
            {
                if (!prices.TryGetValue(pos.Symbol, out var currentPrice) || currentPrice <= 0)
                    continue;

                var stopPrice = pos.EntryPrice * (1 - (decimal)StopLossPct);

                if (currentPrice <= stopPrice)
                {
                    ReportEvent("Stop Loss", $"{pos.Symbol}: {currentPrice:C} <= stop {stopPrice:C} (entry {pos.EntryPrice:C})");

                    var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                    var fillPrice = result.Success && result.AvgPrice > 0
                        ? (decimal)result.AvgPrice
                        : currentPrice;

                    _database.InsertTrade(pos.Symbol, "SELL", pos.Shares, fillPrice, "StopLoss");
                    _database.DeleteAutoPosition(pos.Symbol);

                    ReportEvent("Stop Loss", $"Sold {pos.Symbol} @ {fillPrice:C} (loss: {(fillPrice - pos.EntryPrice) / pos.EntryPrice:P1})");
                    UpdatePositionCount();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Stop loss sell failed for {Symbol}", pos.Symbol);
            }
        }
    }

    /// <summary>
    /// Auto-sell positions that have reached their scheduled_exit date (T+5 or T+10).
    /// </summary>
    private async Task CheckScheduledExitsAsync(CancellationToken ct)
    {
        var positions = _database.GetAutoPositions();
        if (positions.Count == 0) return;

        var now = DateTime.UtcNow;
        foreach (var pos in positions)
        {
            if (pos.ScheduledExit > now) continue;

            try
            {
                ReportEvent("Scheduled Exit", $"Selling {pos.Symbol}: {pos.Shares} shares (hold period expired)");

                var prices = await _polygonPrices.GetPricesAsync(new[] { pos.Symbol }, ct).ConfigureAwait(false);
                prices.TryGetValue(pos.Symbol, out var currentPrice);

                var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                var fillPrice = result.Success && result.AvgPrice > 0
                    ? (decimal)result.AvgPrice
                    : currentPrice > 0 ? currentPrice : pos.EntryPrice;

                _database.InsertTrade(pos.Symbol, "SELL", pos.Shares, fillPrice, "ScheduledExit");
                _database.DeleteAutoPosition(pos.Symbol);

                var pnl = (fillPrice - pos.EntryPrice) / pos.EntryPrice;
                ReportEvent("Scheduled Exit", $"Sold {pos.Symbol} @ {fillPrice:C} (P&L: {pnl:P1})");
                UpdatePositionCount();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Scheduled exit sell failed for {Symbol}", pos.Symbol);
            }
        }
    }

    /// <summary>
    /// Buy earnings beat tickers with 30% of available funds, equal weight, T+10 hold.
    /// Called from DirectPredictionViewModel after earnings scan completes.
    /// </summary>
    public async Task ExecuteEarningsBuyAsync(IReadOnlyList<string> tickers)
    {
        if (tickers.Count == 0) return;

        try
        {
            ReportEvent("Earnings Buy", $"Buying {tickers.Count} earnings beat tickers...");

            var cashInfo = await _tradingBridge.GetCashAsync().ConfigureAwait(false);
            if (cashInfo.Error is not null)
            {
                ReportEvent("Earnings Buy", $"Failed to get cash: {cashInfo.Error}");
                return;
            }

            var netLiq = (decimal)cashInfo.NetLiq;
            var totalBudget = netLiq * 0.30m; // 30% of available funds
            var budgetPerPosition = totalBudget / tickers.Count; // Equal weight

            ReportEvent("Earnings Buy", $"Net Liq: {netLiq:C}, 30% budget: {totalBudget:C}, per position: {budgetPerPosition:C}");

            var existingPositions = _database.GetAutoPositions();
            var existingSymbols = new HashSet<string>(existingPositions.Select(p => p.Symbol), StringComparer.OrdinalIgnoreCase);

            var bought = 0;
            foreach (var symbol in tickers)
            {
                if (existingSymbols.Contains(symbol))
                {
                    ReportEvent("Earnings Buy", $"Skipping {symbol} (already held)");
                    continue;
                }

                try
                {
                    var polygonPrice = await _polygonPrices.GetPriceAsync(symbol).ConfigureAwait(false);
                    if (polygonPrice is null or <= 0)
                    {
                        ReportEvent("Earnings Buy", $"Skipping {symbol} (no price)");
                        continue;
                    }

                    var shares = (int)Math.Floor(budgetPerPosition / polygonPrice.Value);
                    if (shares < 1)
                    {
                        ReportEvent("Earnings Buy", $"Skipping {symbol} (shares=0)");
                        continue;
                    }

                    ReportEvent("Earnings Buy", $"Buying {shares} x {symbol} @ ~{polygonPrice:C}");

                    var buyResult = await _tradingBridge.BuyAsync(symbol, shares).ConfigureAwait(false);
                    if (!buyResult.Success)
                    {
                        ReportEvent("Earnings Buy", $"FAILED {symbol}: {buyResult.Error}");
                        continue;
                    }

                    var fillPrice = buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : polygonPrice.Value;

                    // T+10 hold for earnings positions
                    _database.InsertAutoPosition(symbol, shares, fillPrice, holdDays: 10, note: "EarningsBeat");
                    _database.InsertTrade(symbol, "BUY", shares, fillPrice, "EarningsBeat");
                    existingSymbols.Add(symbol);
                    bought++;

                    ReportEvent("Earnings Buy", $"Bought {symbol}: {shares} shares @ {fillPrice:C} (T+10)");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Earnings buy failed for {Symbol}", symbol);
                    ReportEvent("Earnings Buy", $"Error buying {symbol}: {ex.Message}");
                }
            }

            UpdatePositionCount();
            _monitor.InvalidatePositionCache();
            ReportEvent("Earnings Buy", $"Complete: bought {bought} earnings positions (T+10)");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Earnings buy failed");
            ReportEvent("Earnings Buy", $"FAILED: {ex.Message}");
        }
    }

    private void ResetRebalanceCounter()
    {
        // Reset counter in hmm_state.json via Python bridge
        try
        {
            _hmmBridge.ResetRebalanceCounterAsync().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to reset rebalance counter");
        }
        // Force re-run of HMM next iteration
        _lastHmmDate = default;
    }

    private void ReportEvent(string action, string detail)
    {
        _logger.LogInformation("[{Action}] {Detail}", action, detail);
        try
        {
            Application.Current.Dispatcher.Invoke(() =>
                _monitor.ReportTradingEvent(action, detail));
        }
        catch
        {
            // Dispatcher may not be available during shutdown
        }
    }

    private void UpdatePositionCount()
    {
        try
        {
            var count = _database.GetAutoPositions().Count;
            Application.Current.Dispatcher.Invoke(() =>
                _monitor.AutoPositionCount = count);
        }
        catch
        {
            // Ignore
        }
    }
}
