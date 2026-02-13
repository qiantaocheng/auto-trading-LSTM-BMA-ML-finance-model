using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
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
    private readonly PolygonPriceService _polygonPrices;
    private readonly TraderDatabase _database;
    private readonly MonitorViewModel _monitor;
    private readonly ILogger<TradingSchedulerService> _logger;

    private const double CapitalAllocationRatio = 0.15;  // 15% for direct prediction (75% goes to ETF rotation)
    private const int MaxPositions = 10;
    private const double StopLossPct = 0.034; // backtest optimal SL=1% daily → loosened to 3.4% for 10min intraday checks
    private const int RebalanceIntervalDays = 5;

    // ETF portfolio symbols — AutoBuy must not buy these to prevent capital overlap
    // T10C-Slim: risk-on (SMH) + risk-off (GDX) + shared ETFs + common hedges
    private static readonly HashSet<string> EtfPortfolioSymbols = new(StringComparer.OrdinalIgnoreCase)
        { "SMH", "GDX", "USMV", "QUAL", "PDBC", "DBA", "COPX", "URA", "BIL", "SPY", "QQQ" };

    // Trade mutex — prevents stop-loss loop and main loop from racing on the same position
    private readonly SemaphoreSlim _tradeLock = new(1, 1);

    // Per-symbol sell guard — prevents duplicate sells when a sell takes longer than the check interval
    private readonly ConcurrentDictionary<string, byte> _sellingSymbols = new(StringComparer.OrdinalIgnoreCase);

    // MA200 daily multiplier — tracks current exposure level
    // When SPY < MA200, we sell down to reduce exposure; at rebalance we buy at full size
    private double _currentMa200Exposure = 1.0;
    private DateTime _lastMa200AdjustDate;

    // Prediction rebalance — calendar-anchored (no incremental counter)
    private DateTime? _lastPredictionRebalanceDate;
    private const string StateKeyLastPredRebalDate = "last_prediction_rebalance_date";

    // Daily trade count limit (safety guardrail)
    private const int MaxDailyTrades = 50;
    private int _dailyTradeCount;
    private DateTime _dailyTradeCountDate;

    // Daily reconciliation + backup + broker sync
    private bool _reconciliationDoneToday;
    private bool _backupDoneToday;
    private bool _brokerSyncDoneToday;
    private DateTime _lastReconciliationDate;

    public TradingSchedulerService(
        PythonTradingBridge tradingBridge,
        PolygonPriceService polygonPrices,
        TraderDatabase database,
        MonitorViewModel monitor,
        ILogger<TradingSchedulerService> logger)
    {
        _tradingBridge = tradingBridge;
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

        if (!_monitor.IsConnected)
        {
            _logger.LogWarning("Broker disconnected — auto-buy blocked");
            ReportEvent("Auto-Buy", "BLOCKED: Broker disconnected");
            return;
        }

        try
        {
            ReportEvent("Auto-Buy", "Starting auto-buy...");

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

            if (!IsMarginSafe(cashInfo, out var marginReason))
            {
                ReportEvent("Auto-Buy", $"BLOCKED: {marginReason}");
                return;
            }

            var budgetPerPosition = GetSafeBudgetPerPosition(cashInfo);
            ReportEvent("Auto-Buy", $"Net Liq: {(decimal)cashInfo.NetLiq:C}, Budget/pos: {budgetPerPosition:C}");

            // Acquire trade lock to prevent racing with stop-loss / main loop
            await _tradeLock.WaitAsync().ConfigureAwait(false);
            try
            {
                var existingPositions = _database.GetPositions();
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
                    if (EtfPortfolioSymbols.Contains(symbol))
                    {
                        ReportEvent("Auto-Buy", $"Skipping {symbol} (ETF portfolio symbol — capital overlap)");
                        continue;
                    }

                    try
                    {
                        // Skip symbols that were stop-loss'd today — don't buy back immediately
                        var stopLossIntentId = TraderDatabase.GenerateIntentId("StopLoss", symbol, "SELL");
                        if (_database.IsIntentExecuted(stopLossIntentId))
                        {
                            ReportEvent("Auto-Buy", $"Skipping {symbol} — stop-loss sold today");
                            continue;
                        }

                        var intentId = TraderDatabase.GenerateIntentId("AutoBuy", symbol, "BUY");
                        if (_database.IsIntentExecuted(intentId))
                        {
                            ReportEvent("Auto-Buy", $"Skipping {symbol} — already executed today");
                            continue;
                        }

                        var polygonPrice = await _polygonPrices.GetPriceAsync(symbol).ConfigureAwait(false);
                        if (polygonPrice is null or <= 0)
                        {
                            ReportEvent("Auto-Buy", $"Skipping {symbol} (no Polygon price)");
                            continue;
                        }

                        var price = polygonPrice.Value;
                        var shares = (int)Math.Floor(budgetPerPosition / price);
                        if (shares < 1)
                        {
                            ReportEvent("Auto-Buy", $"Skipping {symbol} (shares=0)");
                            continue;
                        }

                        if (!IncrementDailyTradeCount())
                        {
                            ReportEvent("Auto-Buy", $"BLOCKED: Daily trade limit ({MaxDailyTrades}) reached");
                            break;
                        }

                        _database.TryInsertTradeIntent(intentId, "AutoBuy", symbol, "BUY", shares);
                        bought++;
                        ReportEvent("Auto-Buy", $"Buying {symbol}: {shares} shares @ ~{price:C} ({bought}/{top10Tickers.Count})");

                        var buyResult = await _tradingBridge.BuyAsync(symbol, shares).ConfigureAwait(false);
                        if (buyResult.Success)
                        {
                            RecordBuyFill(buyResult, symbol, shares, price, "AutoBuy");
                            _database.MarkIntentExecuted(intentId, buyResult.OrderId);
                            existingSymbols.Add(symbol);
                            var filled = buyResult.FilledQty > 0 ? buyResult.FilledQty : shares;
                            ReportEvent("Auto-Buy", $"Bought {symbol}: {filled} shares @ {(buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : price):C}");
                        }
                        else
                        {
                            _database.MarkIntentFailed(intentId, buyResult.Error ?? "unknown");
                            ReportEvent("Auto-Buy", $"FAILED {symbol}: {buyResult.Error}");
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Failed to buy {Symbol}", symbol);
                        ReportEvent("Auto-Buy", $"Error buying {symbol}: {ex.Message}");
                    }
                }

                UpdatePositionCount();
                ReportEvent("Auto-Buy", $"Complete: bought {bought} positions");
            }
            finally { _tradeLock.Release(); }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Auto-buy failed");
            ReportEvent("Auto-Buy", $"FAILED: {ex.Message}");
        }
    }

    /// <summary>
    /// Executes pending buys from prediction results. Only runs when:
    /// 1. No AutoBuy positions (previous round fully closed)
    /// 2. pending_buys table has entries
    /// 3. Broker connected + market open (already checked by caller)
    /// </summary>
    private async Task ExecutePendingBuysAsync(CancellationToken ct)
    {
        var existingPositions = _database.GetPositions("AutoBuy");
        if (existingPositions.Count > 0)
            return; // previous round not yet fully closed

        var pendingBuys = _database.GetPendingBuys();
        if (pendingBuys.Count == 0)
            return;

        // Skip stale pending buys — if prediction is > 1 trading day old, wait for fresh prediction
        var oldestPending = pendingBuys.Min(p => p.CreatedAt);
        var pendingAge = TraderDatabase.CountTradingDaysBetween(oldestPending, DateTime.UtcNow);
        if (pendingAge > 1)
        {
            _logger.LogInformation("Pending buys are {Age} trading days old — skipping until fresh prediction", pendingAge);
            _database.ClearPendingBuys();
            return;
        }

        ReportEvent("Pending-Buy", $"Executing {pendingBuys.Count} pending buys...");

        var cashInfo = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
        if (cashInfo.Error is not null)
        {
            ReportEvent("Pending-Buy", $"Failed to get cash: {cashInfo.Error}");
            return;
        }

        if (!IsMarginSafe(cashInfo, out var marginReason))
        {
            ReportEvent("Pending-Buy", $"BLOCKED: {marginReason}");
            return;
        }

        var budgetPerPosition = GetSafeBudgetPerPosition(cashInfo);
        ReportEvent("Pending-Buy", $"Net Liq: {(decimal)cashInfo.NetLiq:C}, Budget/pos: {budgetPerPosition:C}");

        var bought = 0;
        foreach (var pending in pendingBuys.Take(MaxPositions))
        {
            var symbol = pending.Symbol.Trim().ToUpperInvariant();
            try
            {
                if (EtfPortfolioSymbols.Contains(symbol))
                {
                    ReportEvent("Pending-Buy", $"Skipping {symbol} (ETF portfolio symbol — capital overlap)");
                    continue;
                }

                // Skip symbols that were stop-loss'd today — don't buy back immediately
                var stopLossIntentId = TraderDatabase.GenerateIntentId("StopLoss", symbol, "SELL");
                if (_database.IsIntentExecuted(stopLossIntentId))
                {
                    ReportEvent("Pending-Buy", $"Skipping {symbol} — stop-loss sold today");
                    continue;
                }

                var intentId = TraderDatabase.GenerateIntentId("AutoBuy", symbol, "BUY");
                if (_database.IsIntentExecuted(intentId))
                {
                    ReportEvent("Pending-Buy", $"Skipping {symbol} — already executed today");
                    continue;
                }

                var polygonPrice = await _polygonPrices.GetPriceAsync(symbol).ConfigureAwait(false);
                if (polygonPrice is null or <= 0)
                {
                    ReportEvent("Pending-Buy", $"Skipping {symbol} (no price)");
                    continue;
                }

                var price = polygonPrice.Value;
                var shares = (int)Math.Floor(budgetPerPosition / price);
                if (shares < 1)
                {
                    ReportEvent("Pending-Buy", $"Skipping {symbol} (shares=0)");
                    continue;
                }

                _database.TryInsertTradeIntent(intentId, "AutoBuy", symbol, "BUY", shares);
                bought++;
                ReportEvent("Pending-Buy", $"Buying {symbol}: {shares} shares @ ~{price:C} ({bought}/{pendingBuys.Count})");

                var buyResult = await _tradingBridge.BuyAsync(symbol, shares, ct).ConfigureAwait(false);
                if (buyResult.Success)
                {
                    RecordBuyFill(buyResult, symbol, shares, price, "AutoBuy");
                    _database.MarkIntentExecuted(intentId, buyResult.OrderId);
                    var filled = buyResult.FilledQty > 0 ? buyResult.FilledQty : shares;
                    ReportEvent("Pending-Buy", $"Bought {symbol}: {filled} shares @ {(buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : price):C}");
                }
                else
                {
                    _database.MarkIntentFailed(intentId, buyResult.Error ?? "unknown");
                    ReportEvent("Pending-Buy", $"FAILED {symbol}: {buyResult.Error}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to buy pending {Symbol}", symbol);
                ReportEvent("Pending-Buy", $"Error buying {symbol}: {ex.Message}");
            }
        }

        // Clear pending buys after execution attempt
        _database.ClearPendingBuys();
        UpdatePositionCount();
        ReportEvent("Pending-Buy", $"Complete: bought {bought}/{pendingBuys.Count} positions");
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("TradingSchedulerService started ({Days}-day rebalance, MA200 daily multiplier via Polygon)", RebalanceIntervalDays);

        // Load calendar-anchored rebalance date
        LoadPredictionRebalanceDate();

        // Run two loops: stop loss every 10min, main logic every 60s
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
                await Task.Delay(600_000, ct).ConfigureAwait(false); // 10 minutes
            }
            catch (OperationCanceledException)
            {
                break;
            }

            // Stop loss and scheduled exit ALWAYS run regardless of auto-trading toggle
            if (!_monitor.IsConnected) continue;

            try
            {
                await _tradeLock.WaitAsync(ct).ConfigureAwait(false);
                try
                {
                    await CheckStopLossAsync(ct).ConfigureAwait(false);
                    await CheckScheduledExitsAsync(ct).ConfigureAwait(false);
                }
                finally { _tradeLock.Release(); }
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

            if (!_monitor.IsConnected) continue;

            try
            {
                var status = await _tradingBridge.GetMarketStatusAsync(ct).ConfigureAwait(false);

                // Reset daily flags at midnight
                var today = DateTime.UtcNow.Date;
                if (_lastReconciliationDate != today)
                {
                    _reconciliationDoneToday = false;
                    _backupDoneToday = false;
                    _brokerSyncDoneToday = false;
                    _lastReconciliationDate = today;
                }

                // Post-market tasks run REGARDLESS of auto-trading toggle (operational health)
                if (!status.IsOpen && status.EtHour >= 16)
                {
                    if (!_reconciliationDoneToday && status.EtMinute >= 5)
                    {
                        await RunDailyReconciliationAsync().ConfigureAwait(false);
                        _reconciliationDoneToday = true;
                    }
                    if (!_backupDoneToday && status.EtMinute >= 15)
                    {
                        RunDailyBackup();
                        _backupDoneToday = true;
                    }
                }

                // Trading actions require auto-trading enabled + market open
                if (!IsAutoTradingEnabled) continue;
                if (!status.IsOpen) continue;

                // Daily MA200 multiplier — adjust exposure based on SPY vs MA200 (via Polygon)
                await AdjustMa200ExposureAsync(ct).ConfigureAwait(false);

                // Acquire trade lock for all buy/sell operations
                await _tradeLock.WaitAsync(ct).ConfigureAwait(false);
                try
                {
                    // 5-day rebalance check (calendar-anchored — survives offline days)
                    var predDaysSince = _lastPredictionRebalanceDate.HasValue
                        ? TraderDatabase.CountTradingDaysBetween(_lastPredictionRebalanceDate.Value, DateTime.UtcNow)
                        : RebalanceIntervalDays;
                    if (predDaysSince >= RebalanceIntervalDays)
                    {
                        await ExecuteRebalanceAsync(ct).ConfigureAwait(false);
                    }

                    // Execute pending buys if no existing auto_positions
                    await ExecutePendingBuysAsync(ct).ConfigureAwait(false);
                }
                finally { _tradeLock.Release(); }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "Trading scheduler main loop error");
            }
        }
    }


    /// <summary>
    /// Every 5 trading days: diff-based rebalance.
    /// Keeps overlapping positions, only sells removed tickers, only buys new ones.
    /// Reduces turnover ~30-50% vs sell-all-buy-all.
    /// </summary>
    private async Task ExecuteRebalanceAsync(CancellationToken ct)
    {
        if (!_monitor.IsConnected)
        {
            ReportEvent("Rebalance", "BLOCKED: Broker disconnected");
            return;
        }

        // Guard: don't start rebalance within 20 minutes of market close (3:40 PM ET)
        // to avoid partial execution with market orders rejected after 4:00 PM
        try
        {
            var marketStatus = await _tradingBridge.GetMarketStatusAsync(ct).ConfigureAwait(false);
            if (marketStatus.EtHour == 15 && marketStatus.EtMinute >= 40)
            {
                ReportEvent("Rebalance", $"Postponing rebalance — only {60 - marketStatus.EtMinute} min to close");
                return;
            }
        }
        catch { /* Non-fatal — proceed with rebalance */ }

        ReportEvent("Rebalance", "Starting 5-day diff-based rebalance...");

        // 1. Get new Top10 tickers from prediction
        var tickerRecords = _database.GetTickerRecords()
            .Where(t => t.Tag == 0)
            .ToList();
        var newTickers = tickerRecords.Take(MaxPositions).Select(t => t.Symbol).ToList();

        if (newTickers.Count == 0)
        {
            ReportEvent("Rebalance", "No tickers available — run prediction first");
            ResetRebalanceCounter();
            UpdatePositionCount();
            return;
        }

        // Warn if prediction data is stale (> 2 trading days old)
        var newestTicker = tickerRecords.OrderByDescending(t => t.AddedAt).FirstOrDefault();
        if (newestTicker != default)
        {
            var predAge = TraderDatabase.CountTradingDaysBetween(newestTicker.AddedAt, DateTime.UtcNow);
            if (predAge > 2)
            {
                _logger.LogWarning("Rebalancing with stale prediction data ({Age} trading days old, from {Date})",
                    predAge, newestTicker.AddedAt.ToString("yyyy-MM-dd"));
                ReportEvent("Rebalance", $"WARNING: Using stale prediction ({predAge} trading days old) — consider re-running prediction");
            }
        }

        var newTickerSet = new HashSet<string>(newTickers, StringComparer.OrdinalIgnoreCase);

        // 2. Compute diff: which to sell, which to keep, which to buy
        var positions = _database.GetPositions("AutoBuy");
        var heldSymbols = new HashSet<string>(positions.Select(p => p.Symbol), StringComparer.OrdinalIgnoreCase);

        var toSell = positions.Where(p => !newTickerSet.Contains(p.Symbol)).ToList();
        var toKeep = positions.Where(p => newTickerSet.Contains(p.Symbol)).ToList();
        var toBuy = newTickers.Where(t => !heldSymbols.Contains(t)).ToList();

        ReportEvent("Rebalance",
            $"Diff: keep {toKeep.Count}, sell {toSell.Count}, buy {toBuy.Count} " +
            $"(turnover {toSell.Count + toBuy.Count}/{newTickers.Count + positions.Count})");

        // 3. Sell removed positions
        foreach (var pos in toSell)
        {
            try
            {
                var intentId = TraderDatabase.GenerateIntentId("Rebalance", pos.Symbol, "SELL");
                if (_database.IsIntentExecuted(intentId))
                {
                    ReportEvent("Rebalance", $"Skipping sell {pos.Symbol} — already executed today");
                    continue;
                }
                _database.TryInsertTradeIntent(intentId, "Rebalance", pos.Symbol, "SELL", pos.Shares);

                var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                if (result.Success)
                {
                    var filled = result.FilledQty > 0 ? result.FilledQty : 0;
                    if (filled == 0)
                    {
                        _logger.LogWarning("Rebalance sell {Symbol}: Success=true but FilledQty=0 — treating as failure", pos.Symbol);
                        _database.MarkIntentFailed(intentId, $"No fills: status={result.Status}");
                        ReportEvent("Rebalance", $"FAILED sell {pos.Symbol}: Success but 0 filled");
                        continue;
                    }
                    _database.MarkIntentExecuted(intentId, result.OrderId);
                    RecordSellFill(result, pos.Symbol, pos.Shares, pos.EntryPrice, "Rebalance");
                    ReportEvent("Rebalance", $"Sold {pos.Symbol}: {filled} shares @ {(result.AvgPrice > 0 ? (decimal)result.AvgPrice : pos.EntryPrice):C}");
                }
                else
                {
                    _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                    ReportEvent("Rebalance", $"FAILED sell {pos.Symbol}: {result.Error}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Rebalance sell failed for {Symbol}", pos.Symbol);
                ReportEvent("Rebalance", $"Error selling {pos.Symbol}: {ex.Message}");
            }
        }

        // 4. Buy new positions (not already held)
        // Re-read position count after sells — cap buys to prevent stuck-position accumulation (S13/S14)
        var currentAfterSells = _database.GetPositions("AutoBuy").Count;
        var maxNewBuys = Math.Max(0, MaxPositions - currentAfterSells);
        if (maxNewBuys < toBuy.Count)
        {
            _logger.LogWarning("Rebalance: capping buys from {Requested} to {Max} — {Stuck} stuck/unsold position(s) detected",
                toBuy.Count, maxNewBuys, currentAfterSells - toKeep.Count);
            ReportEvent("Rebalance", $"WARNING: {currentAfterSells - toKeep.Count} stuck position(s) — limiting buys to {maxNewBuys}");
            toBuy = toBuy.Take(maxNewBuys).ToList();
        }

        if (toBuy.Count > 0)
        {
            var cashInfo = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
            if (cashInfo.Error is not null)
            {
                ReportEvent("Rebalance", $"Failed to get cash: {cashInfo.Error}");
                ResetRebalanceCounter();
                return;
            }

            if (!IsMarginSafe(cashInfo, out var marginReason))
            {
                ReportEvent("Rebalance", $"BLOCKED buy phase: {marginReason}");
                ResetRebalanceCounter();
                return;
            }

            var budgetPerPosition = GetSafeBudgetPerPosition(cashInfo);
            ReportEvent("Rebalance", $"Buying {toBuy.Count} new positions, budget/pos={budgetPerPosition:C}");

            var bought = 0;
            foreach (var symbol in toBuy)
            {
                try
                {
                    if (EtfPortfolioSymbols.Contains(symbol))
                    {
                        ReportEvent("Rebalance", $"Skipping buy {symbol} (ETF portfolio symbol — capital overlap)");
                        continue;
                    }

                    // Skip symbols that were stop-loss'd today — don't buy back immediately
                    var stopLossIntentId = TraderDatabase.GenerateIntentId("StopLoss", symbol, "SELL");
                    if (_database.IsIntentExecuted(stopLossIntentId))
                    {
                        ReportEvent("Rebalance", $"Skipping buy {symbol} — stop-loss sold today");
                        continue;
                    }

                    var intentId = TraderDatabase.GenerateIntentId("Rebalance", symbol, "BUY");
                    if (_database.IsIntentExecuted(intentId))
                    {
                        ReportEvent("Rebalance", $"Skipping buy {symbol} — already executed today");
                        continue;
                    }

                    var polygonPrice = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                    if (polygonPrice is null or <= 0) continue;

                    var price = polygonPrice.Value;
                    var shares = (int)Math.Floor(budgetPerPosition / price);
                    if (shares < 1) continue;

                    _database.TryInsertTradeIntent(intentId, "Rebalance", symbol, "BUY", shares);

                    var buyResult = await _tradingBridge.BuyAsync(symbol, shares, ct).ConfigureAwait(false);
                    if (buyResult.Success)
                    {
                        RecordBuyFill(buyResult, symbol, shares, price, "Rebalance");
                        _database.MarkIntentExecuted(intentId, buyResult.OrderId);
                        bought++;
                        var filled = buyResult.FilledQty > 0 ? buyResult.FilledQty : shares;
                        ReportEvent("Rebalance", $"Bought {symbol}: {filled} shares @ {(buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : price):C}");
                    }
                    else
                    {
                        _database.MarkIntentFailed(intentId, buyResult.Error ?? "unknown");
                        ReportEvent("Rebalance", $"FAILED {symbol}: {buyResult.Error}");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Rebalance buy failed for {Symbol}", symbol);
                    ReportEvent("Rebalance", $"Error buying {symbol}: {ex.Message}");
                }
            }

            ReportEvent("Rebalance", $"Buy phase complete: {bought}/{toBuy.Count} new positions");
        }

        ResetRebalanceCounter();
        _currentMa200Exposure = 1.0; // Reset exposure after rebalance
        UpdatePositionCount();

        // S21: Alert if position count < MaxPositions after rebalance (failed buys/unknown tickers)
        var finalCount = _database.GetPositions("AutoBuy").Count;
        if (finalCount < MaxPositions)
        {
            _logger.LogWarning("Rebalance incomplete: {Count}/{Max} positions filled — {Missing} position(s) could not be bought",
                finalCount, MaxPositions, MaxPositions - finalCount);
            ReportEvent("Rebalance", $"WARNING: Only {finalCount}/{MaxPositions} positions filled — check logs for failed buys");
        }

        ReportEvent("Rebalance", $"Rebalance complete: kept {toKeep.Count}, sold {toSell.Count}, bought {toBuy.Count}");
    }

    /// <summary>
    /// Crisis early exit (Plan B): sell all positions urgently.
    /// </summary>
    private async Task ExecuteCrisisExitAsync(CancellationToken ct)
    {
        if (!_monitor.IsConnected)
        {
            ReportEvent("CRISIS EXIT", "BLOCKED: Broker disconnected — URGENT manual intervention needed!");
            return;
        }

        // Sell AutoBuy (NOT ETF — ETF has its own risk management)
        var positions = _database.GetPositions("AutoBuy");
        if (positions.Count == 0) return;

        ReportEvent("CRISIS EXIT", $"Selling {positions.Count} positions urgently!");

        foreach (var pos in positions)
        {
            try
            {
                var intentId = TraderDatabase.GenerateIntentId("CrisisExit", pos.Symbol, "SELL");
                if (_database.IsIntentExecuted(intentId))
                {
                    ReportEvent("CRISIS EXIT", $"Skipping {pos.Symbol} — already sold today");
                    continue;
                }
                _database.TryInsertTradeIntent(intentId, "CrisisExit", pos.Symbol, "SELL", pos.Shares);

                var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                if (result.Success)
                {
                    _database.MarkIntentExecuted(intentId, result.OrderId);
                    RecordSellFill(result, pos.Symbol, pos.Shares, pos.EntryPrice, "CrisisExit");
                    ReportEvent("CRISIS EXIT", $"Sold {pos.Symbol} @ {(result.AvgPrice > 0 ? (decimal)result.AvgPrice : pos.EntryPrice):C}");
                }
                else
                {
                    _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                    ReportEvent("CRISIS EXIT", $"FAILED {pos.Symbol}: {result.Error}");
                }
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
        // Daily sync: update DB shares+entry_price from broker to handle stock splits
        if (!_brokerSyncDoneToday)
        {
            try
            {
                var synced = _database.SyncPositionsFromBroker();
                _brokerSyncDoneToday = true;
                if (synced > 0)
                {
                    _logger.LogInformation("Broker sync: updated {Count} position(s) (split/correction)", synced);
                    ReportEvent("Broker Sync", $"Updated {synced} position(s) from broker (split/price correction)");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Broker position sync failed — stop-loss may use stale entry prices");
            }
        }

        // Stop loss applies to AutoBuy, Manual — NOT ETF
        var positions = _database.GetPositions()
            .Where(p => p.Strategy != "ETF").ToList();
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

        // If Polygon has been failing consecutively, prices are stale cached values —
        // stop-loss check would silently pass on outdated data, giving false safety
        if (_polygonPrices.IsStale)
        {
            _logger.LogWarning("STOP-LOSS BLIND: Polygon API stale ({Failures} consecutive failures) — prices may be outdated, stop-loss unreliable",
                _polygonPrices.ConsecutiveFailures);
            ReportEvent("Stop Loss", $"WARNING: Polygon stale ({_polygonPrices.ConsecutiveFailures} failures) — stop-loss check using outdated prices!");
        }

        foreach (var pos in positions)
        {
            try
            {
                if (!prices.TryGetValue(pos.Symbol, out var currentPrice) || currentPrice <= 0)
                {
                    // S14: Alert when Polygon returns no price (possible delisting or halt)
                    _logger.LogWarning("STOP-LOSS: No price for {Symbol} (possible delisting/halt) — cannot evaluate stop-loss", pos.Symbol);
                    continue;
                }

                var stopPrice = pos.EntryPrice * (1 - (decimal)StopLossPct);

                if (currentPrice <= stopPrice)
                {
                    // Skip if a sell is already in flight for this symbol
                    if (!_sellingSymbols.TryAdd(pos.Symbol, 0))
                    {
                        _logger.LogDebug("Stop loss: sell already in flight for {Symbol}", pos.Symbol);
                        continue;
                    }

                    try
                    {
                        var intentId = TraderDatabase.GenerateIntentId("StopLoss", pos.Symbol, "SELL");
                        if (_database.IsIntentExecuted(intentId))
                        {
                            ReportEvent("Stop Loss", $"Skipping {pos.Symbol} — already sold today");
                            continue;
                        }
                        _database.TryInsertTradeIntent(intentId, "StopLoss", pos.Symbol, "SELL", pos.Shares);

                        ReportEvent("Stop Loss", $"{pos.Symbol}: {currentPrice:C} <= stop {stopPrice:C} (entry {pos.EntryPrice:C})");

                        var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                        if (result.Success)
                        {
                            _database.MarkIntentExecuted(intentId, result.OrderId);
                            RecordSellFill(result, pos.Symbol, pos.Shares, currentPrice, "StopLoss");
                            var fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : currentPrice;
                            ReportEvent("Stop Loss", $"Sold {pos.Symbol} @ {fillPrice:C} (loss: {(fillPrice - pos.EntryPrice) / pos.EntryPrice:P1})");
                            UpdatePositionCount();
                        }
                        else
                        {
                            _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                            ReportEvent("Stop Loss", $"FAILED {pos.Symbol}: {result.Error}");
                        }
                    }
                    finally
                    {
                        _sellingSymbols.TryRemove(pos.Symbol, out _);
                    }
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
        // Scheduled exits for non-ETF positions with a scheduled_exit date
        var positions = _database.GetPositions()
            .Where(p => p.Strategy != "ETF" && p.ScheduledExit.HasValue).ToList();
        if (positions.Count == 0) return;

        var now = DateTime.UtcNow;
        foreach (var pos in positions)
        {
            if (pos.ScheduledExit is null || pos.ScheduledExit.Value > now) continue;

            try
            {
                var intentId = TraderDatabase.GenerateIntentId("ScheduledExit", pos.Symbol, "SELL");
                if (_database.IsIntentExecuted(intentId))
                {
                    ReportEvent("Scheduled Exit", $"Skipping {pos.Symbol} — already sold today");
                    continue;
                }
                _database.TryInsertTradeIntent(intentId, "ScheduledExit", pos.Symbol, "SELL", pos.Shares);

                ReportEvent("Scheduled Exit", $"Selling {pos.Symbol}: {pos.Shares} shares (hold period expired)");

                var prices = await _polygonPrices.GetPricesAsync(new[] { pos.Symbol }, ct).ConfigureAwait(false);
                prices.TryGetValue(pos.Symbol, out var currentPrice);

                var fallbackPrice = currentPrice > 0 ? currentPrice : pos.EntryPrice;
                var result = await _tradingBridge.SellAsync(pos.Symbol, pos.Shares, ct).ConfigureAwait(false);
                if (result.Success)
                {
                    _database.MarkIntentExecuted(intentId, result.OrderId);
                    RecordSellFill(result, pos.Symbol, pos.Shares, fallbackPrice, "ScheduledExit");
                    var fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : fallbackPrice;
                    var pnl = (fillPrice - pos.EntryPrice) / pos.EntryPrice;
                    ReportEvent("Scheduled Exit", $"Sold {pos.Symbol} @ {fillPrice:C} (P&L: {pnl:P1})");
                    UpdatePositionCount();
                }
                else
                {
                    _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                    ReportEvent("Scheduled Exit", $"FAILED {pos.Symbol}: {result.Error}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Scheduled exit sell failed for {Symbol}", pos.Symbol);
            }
        }
    }

    /// <summary>
    /// MA200 daily multiplier: adjusts position exposure based on SPY vs MA200.
    /// When target exposure drops (SPY below MA200), sells partial shares.
    /// Exposure is restored to 1.0 at next rebalance when positions are re-bought at full size.
    /// Thresholds: SPY >= MA200 → 1.0, SPY < MA200 → 0.60, SPY < 0.95*MA200 → 0.30
    /// </summary>
    private async Task AdjustMa200ExposureAsync(CancellationToken ct)
    {
        var today = DateTime.UtcNow.Date;
        if (_lastMa200AdjustDate == today) return; // once per day

        var (targetExposure, spyPrice, spyMa200) = await _polygonPrices.GetSpyMa200CapAsync(ct).ConfigureAwait(false);
        ReportEvent("MA200", $"SPY={spyPrice:F0} MA200={spyMa200:F0} Cap={targetExposure:F2}");

        if (Math.Abs(targetExposure - _currentMa200Exposure) < 0.01)
        {
            _lastMa200AdjustDate = today;
            ReportEvent("MA200", $"Exposure unchanged: {_currentMa200Exposure:F2} (SPY MA200 cap={targetExposure:F2})");
            return;
        }

        if (targetExposure >= _currentMa200Exposure)
        {
            // Exposure should increase — don't buy mid-cycle, wait for rebalance
            _lastMa200AdjustDate = today;
            ReportEvent("MA200", $"Exposure target {targetExposure:F2} >= current {_currentMa200Exposure:F2} — will restore at next rebalance");
            return;
        }

        // Target exposure is LOWER → sell partial shares to reduce risk
        var positions = _database.GetPositions("AutoBuy");
        if (positions.Count == 0)
        {
            _currentMa200Exposure = targetExposure;
            _lastMa200AdjustDate = today;
            return;
        }

        // Calculate what fraction of each position to sell
        // e.g., current=1.0, target=0.60 → sell 40% of each position
        var sellFraction = 1.0 - (targetExposure / _currentMa200Exposure);
        ReportEvent("MA200", $"Reducing exposure: {_currentMa200Exposure:F2} → {targetExposure:F2} (selling {sellFraction:P0} of each position)");

        await _tradeLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            foreach (var pos in positions)
            {
                var sharesToSell = (int)Math.Floor(pos.Shares * sellFraction);
                if (sharesToSell < 1) continue;

                try
                {
                    if (!_sellingSymbols.TryAdd(pos.Symbol, 0)) continue;
                    try
                    {
                        var intentId = TraderDatabase.GenerateIntentId("MA200Adjust", pos.Symbol, "SELL");
                        if (_database.IsIntentExecuted(intentId)) continue;
                        _database.TryInsertTradeIntent(intentId, "MA200Adjust", pos.Symbol, "SELL", sharesToSell);

                        ReportEvent("MA200", $"Selling {sharesToSell}/{pos.Shares} shares of {pos.Symbol}");

                        var result = await _tradingBridge.SellAsync(pos.Symbol, sharesToSell, ct).ConfigureAwait(false);
                        if (result.Success)
                        {
                            var filled = result.FilledQty > 0 ? result.FilledQty : 0;
                            if (filled > 0)
                            {
                                _database.MarkIntentExecuted(intentId, result.OrderId);
                                var price = result.AvgPrice > 0 ? (decimal)result.AvgPrice : pos.EntryPrice;
                                _database.InsertTrade(pos.Symbol, "SELL", filled, price, "MA200Adjust");
                                _database.UpdatePositionShares(pos.Symbol, pos.Shares - filled);
                                ReportEvent("MA200", $"Sold {filled} shares of {pos.Symbol} @ {price:C}");
                            }
                            else
                            {
                                _database.MarkIntentFailed(intentId, "No fills");
                            }
                        }
                        else
                        {
                            _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                            ReportEvent("MA200", $"FAILED sell {pos.Symbol}: {result.Error}");
                        }
                    }
                    finally
                    {
                        _sellingSymbols.TryRemove(pos.Symbol, out _);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "MA200 exposure adjustment failed for {Symbol}", pos.Symbol);
                }
            }
        }
        finally { _tradeLock.Release(); }

        _currentMa200Exposure = targetExposure;
        _lastMa200AdjustDate = today;
        UpdatePositionCount();
        ReportEvent("MA200", $"Exposure adjusted to {targetExposure:F2}");
    }

    private void ResetRebalanceCounter()
    {
        // Calendar-anchored: record today as the rebalance date
        _lastPredictionRebalanceDate = DateTime.UtcNow;
        SavePredictionRebalanceDate();
    }

    // ── Margin Safety ─────────────────────────────────────────────────

    private const double MaxMarginUtilization = 0.85;
    private const double BuyingPowerSafetyBuffer = 0.92;

    private bool IsMarginSafe(CashResult cashInfo, out string reason)
    {
        reason = "";
        if (cashInfo.MarginUsedPct > MaxMarginUtilization)
        {
            reason = $"Margin utilization {cashInfo.MarginUsedPct:P0} > {MaxMarginUtilization:P0} limit";
            _logger.LogWarning("MARGIN ALERT: {Reason}", reason);
            return false;
        }
        return true;
    }

    private decimal GetSafeBudgetPerPosition(CashResult cashInfo)
    {
        var netLiq = (decimal)cashInfo.NetLiq;
        var strategyBudget = netLiq * (decimal)CapitalAllocationRatio / MaxPositions;
        if (cashInfo.BuyingPower > 0)
        {
            var bpBudget = (decimal)(cashInfo.BuyingPower * BuyingPowerSafetyBuffer) / MaxPositions;
            return Math.Min(strategyBudget, bpBudget);
        }
        return strategyBudget;
    }

    // ── Daily Trade Count ──────────────────────────────────────────────

    private bool IncrementDailyTradeCount()
    {
        var today = DateTime.UtcNow.Date;
        if (_dailyTradeCountDate != today)
        {
            _dailyTradeCount = 0;
            _dailyTradeCountDate = today;
        }
        if (_dailyTradeCount >= MaxDailyTrades)
        {
            _logger.LogWarning("Daily trade limit reached ({Count}/{Max})", _dailyTradeCount, MaxDailyTrades);
            return false;
        }
        _dailyTradeCount++;
        return true;
    }

    // ── Partial Fill Helpers ─────────────────────────────────────────────

    private void RecordBuyFill(BuyResult result, string symbol, int requestedShares,
        decimal fallbackPrice, string strategy, int holdDays = 5, string? note = null)
    {
        // Guard: never use requestedShares as fallback — only record actual fills
        var filled = result.FilledQty > 0 ? result.FilledQty : 0;
        var price = result.AvgPrice > 0 ? (decimal)result.AvgPrice : fallbackPrice;
        if (result.RemainingQty > 0)
            _logger.LogWarning("Partial fill {Strategy} {Symbol}: {Filled}/{Total}",
                strategy, symbol, result.FilledQty, result.TotalQty);
        if (filled > 0)
        {
            _database.InsertPosition(symbol, strategy, filled, price, holdDays, note: note);
            _database.InsertTrade(symbol, "BUY", filled, price, strategy);
        }
        else
        {
            _logger.LogWarning("{Strategy} buy {Symbol}: Success=true but FilledQty=0 — skipping DB write", strategy, symbol);
        }
    }

    private void RecordSellFill(SellResult result, string symbol, int requestedShares,
        decimal fallbackPrice, string strategy)
    {
        // Guard: never use requestedShares as fallback — only record actual fills
        var filled = result.FilledQty > 0 ? result.FilledQty : 0;
        if (filled == 0)
        {
            _logger.LogWarning("{Strategy} sell {Symbol}: Success=true but FilledQty=0 — skipping DB write", strategy, symbol);
            return;
        }
        var price = result.AvgPrice > 0 ? (decimal)result.AvgPrice : fallbackPrice;
        try
        {
            _database.InsertTrade(symbol, "SELL", filled, price, strategy);
            if (result.RemainingQty > 0)
            {
                _database.UpdatePositionShares(symbol, result.RemainingQty);
                _logger.LogWarning("Partial sell {Symbol}: {Remaining} shares remaining", symbol, result.RemainingQty);
            }
            else
                _database.DeletePosition(symbol);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "CRITICAL: DB update failed after sell fill for {Symbol} ({Filled} shares @ {Price:C}). " +
                "Position may be out of sync with broker — reconciliation will detect.", symbol, filled, price);
        }
    }

    // ── Daily Reconciliation (FIX F) ──────────────────────────────────

    private async Task RunDailyReconciliationAsync()
    {
        try
        {
            ReportEvent("Reconciliation", "Running daily position reconciliation...");

            var dbPositions = _database.GetPositions();
            var brokerPositions = _database.GetBrokerPositions();

            var dbBySymbol = dbPositions.ToDictionary(p => p.Symbol, StringComparer.OrdinalIgnoreCase);
            var brokerBySymbol = brokerPositions.ToDictionary(p => p.Symbol, StringComparer.OrdinalIgnoreCase);

            var issues = 0;

            // Check DB positions against broker
            foreach (var (symbol, dbPos) in dbBySymbol)
            {
                if (!brokerBySymbol.TryGetValue(symbol, out var bp))
                {
                    _logger.LogWarning("RECONCILIATION: {Symbol} in DB ({Shares} shares) but NOT in broker", symbol, dbPos.Shares);
                    ReportEvent("Reconciliation", $"WARNING: {symbol} in DB but not at broker ({dbPos.Shares} shares)");
                    issues++;
                }
                else if (bp.Quantity != dbPos.Shares)
                {
                    _logger.LogWarning("RECONCILIATION: {Symbol} qty mismatch — DB={DbQty} Broker={BrokerQty}",
                        symbol, dbPos.Shares, bp.Quantity);
                    ReportEvent("Reconciliation", $"WARNING: {symbol} qty mismatch — DB={dbPos.Shares} Broker={bp.Quantity}");
                    issues++;
                }
            }

            // Check broker positions not in DB
            foreach (var (symbol, bp) in brokerBySymbol)
            {
                if (!dbBySymbol.ContainsKey(symbol))
                {
                    _logger.LogWarning("RECONCILIATION: {Symbol} at broker ({Qty} shares) but NOT in DB", symbol, bp.Quantity);
                    ReportEvent("Reconciliation", $"WARNING: {symbol} at broker but not in DB ({bp.Quantity} shares)");
                    issues++;
                }
            }

            if (issues == 0)
                ReportEvent("Reconciliation", "All positions reconciled — no discrepancies");
            else
                ReportEvent("Reconciliation", $"Found {issues} discrepancies — review manually");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Reconciliation failed");
            ReportEvent("Reconciliation", $"Error: {ex.Message}");
        }
    }

    // ── Daily Backup (FIX G) ────────────────────────────────────────────

    private void RunDailyBackup()
    {
        try
        {
            var integrityOk = _database.RunIntegrityCheck();
            if (!integrityOk)
            {
                _logger.LogCritical("DATABASE INTEGRITY CHECK FAILED — backup skipped");
                ReportEvent("DB Backup", "CRITICAL: Integrity check failed!");
                return;
            }

            var backupDir = Path.Combine(AppContext.BaseDirectory, "backups");
            Directory.CreateDirectory(backupDir);
            var destPath = Path.Combine(backupDir, $"TraderApp_{DateTime.UtcNow:yyyy-MM-dd}.db");
            _database.BackupTo(destPath);
            ReportEvent("DB Backup", $"Daily backup saved: {Path.GetFileName(destPath)}");

            // Cleanup backups older than 7 days
            foreach (var file in Directory.GetFiles(backupDir, "TraderApp_*.db"))
            {
                if (File.GetCreationTimeUtc(file) < DateTime.UtcNow.AddDays(-7))
                {
                    try { File.Delete(file); } catch { }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Daily backup failed");
            ReportEvent("DB Backup", $"Error: {ex.Message}");
        }
    }

    // ── Prediction Rebalance Date Persistence ──────────────────────────

    private void LoadPredictionRebalanceDate()
    {
        try
        {
            var dateStr = _database.GetEtfRotationState(StateKeyLastPredRebalDate);
            if (dateStr is not null && DateTime.TryParse(dateStr, out var dt))
            {
                _lastPredictionRebalanceDate = dt;
                var daysSince = TraderDatabase.CountTradingDaysBetween(dt, DateTime.UtcNow);
                _logger.LogInformation(
                    "Prediction rebalance state: last={Date}, {Days} trading days since, next in {Remaining}",
                    dt.ToString("yyyy-MM-dd"), daysSince, Math.Max(0, RebalanceIntervalDays - daysSince));
            }
            else
            {
                _logger.LogInformation("No prediction rebalance date found — will trigger on first cycle");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load prediction rebalance date");
        }
    }

    private void SavePredictionRebalanceDate()
    {
        try
        {
            if (_lastPredictionRebalanceDate.HasValue)
                _database.SetEtfRotationState(StateKeyLastPredRebalDate,
                    _lastPredictionRebalanceDate.Value.ToString("o"));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save prediction rebalance date");
        }
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
            var count = _database.GetPositions().Count;
            Application.Current.Dispatcher.Invoke(() =>
                _monitor.AutoPositionCount = count);
        }
        catch
        {
            // Ignore
        }
    }
}
