using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Threading;
using Trader.App.Commands;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Trader.App.ViewModels;
using Trader.PythonBridge.Services;

namespace Trader.App.ViewModels.Pages;

public class DatabaseViewModel : ViewModelBase
{
    private readonly TraderDatabase _database;
    private readonly PythonTradingBridge _tradingBridge;
    private readonly PolygonPriceService _polygonPrices;
    private readonly MonitorViewModel _monitor;
    private readonly DispatcherTimer _refreshTimer;
    private string _newTickerSymbol = string.Empty;
    private decimal _investAmount;
    private string _buyStatus = string.Empty;
    private UnifiedPositionRowViewModel? _selectedPosition;

    public DatabaseViewModel(TraderDatabase database, PythonTradingBridge tradingBridge, PolygonPriceService polygonPrices, MonitorViewModel monitor)
    {
        _database = database;
        _tradingBridge = tradingBridge;
        _polygonPrices = polygonPrices;
        _monitor = monitor;
        Positions = new ObservableCollection<UnifiedPositionRowViewModel>();
        BuyCommand = new AsyncRelayCommand(ExecuteBuyAsync);
        DeleteSelectedCommand = new AsyncRelayCommand(DeleteSelectedAsync, () => SelectedPosition is not null);
        RefreshCommand = new AsyncRelayCommand(RefreshWithPricesAsync);
        _ = RefreshWithPricesAsync();

        // Auto-refresh every 5 seconds: fetch Polygon prices + check positions
        _refreshTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(5) };
        _refreshTimer.Tick += async (_, _) => await RefreshWithPricesAsync();
        _refreshTimer.Start();
    }

    public ObservableCollection<UnifiedPositionRowViewModel> Positions { get; }

    // Exposes ETF rotation state from MonitorViewModel for binding in DatabaseView
    public MonitorViewModel Monitor => _monitor;

    public string NewTickerSymbol
    {
        get => _newTickerSymbol;
        set
        {
            _newTickerSymbol = value?.ToUpperInvariant() ?? string.Empty;
            RaisePropertyChanged();
        }
    }

    public decimal InvestAmount
    {
        get => _investAmount;
        set
        {
            _investAmount = value;
            RaisePropertyChanged();
        }
    }

    public string BuyStatus
    {
        get => _buyStatus;
        set
        {
            _buyStatus = value;
            RaisePropertyChanged();
        }
    }

    public UnifiedPositionRowViewModel? SelectedPosition
    {
        get => _selectedPosition;
        set
        {
            _selectedPosition = value;
            RaisePropertyChanged();
            DeleteSelectedCommand.RaiseCanExecuteChanged();
        }
    }

    public AsyncRelayCommand BuyCommand { get; }
    public AsyncRelayCommand DeleteSelectedCommand { get; }
    public AsyncRelayCommand RefreshCommand { get; }

    private async Task RefreshWithPricesAsync()
    {
        try
        {
            // Step 1: Broker live share quantities (from broker_positions table)
            var brokerPositions = _database.GetBrokerPositions();
            var liveQty = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var bp in brokerPositions)
                liveQty[bp.Symbol] = bp.Quantity;

            // Step 2: Get all positions from unified table
            var dbPositions = _database.GetPositions();

            var allSymbols = dbPositions.Select(p => p.Symbol)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Step 3: Fetch fresh Polygon prices
            var prices = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);
            if (allSymbols.Count > 0)
            {
                try
                {
                    prices = await _polygonPrices.GetPricesAsync(allSymbols);
                }
                catch { }
            }

            // Step 4: Batch update current prices in DB
            if (prices.Count > 0)
            {
                try { _database.UpdatePositionPrices(prices); } catch { }
            }

            // Step 5: Build unified positions grid
            var previousSelection = _selectedPosition?.Symbol;
            Positions.Clear();
            foreach (var pos in dbPositions)
            {
                var shares = liveQty.TryGetValue(pos.Symbol, out var ibkrQty) ? ibkrQty : pos.Shares;
                prices.TryGetValue(pos.Symbol, out var curPrice);
                if (curPrice <= 0) curPrice = pos.CurrentPrice;
                var pnl = pos.EntryPrice > 0 && curPrice > 0
                    ? (double)((curPrice - pos.EntryPrice) / pos.EntryPrice * 100)
                    : 0.0;
                Positions.Add(new UnifiedPositionRowViewModel(
                    pos.Symbol, pos.Strategy, shares, pos.EntryPrice, curPrice, pnl,
                    pos.EnteredAt, pos.ScheduledExit, pos.TargetWeight));
            }
            if (previousSelection is not null)
                SelectedPosition = Positions.FirstOrDefault(p => p.Symbol == previousSelection);
        }
        catch
        {
            // Ignore refresh errors
        }
    }

    private static bool IsUsMarketOpen()
    {
        var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
            TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
        if (et.DayOfWeek == DayOfWeek.Saturday || et.DayOfWeek == DayOfWeek.Sunday)
            return false;
        var marketOpen = new TimeSpan(9, 30, 0);
        var marketClose = new TimeSpan(16, 0, 0);
        return et.TimeOfDay >= marketOpen && et.TimeOfDay <= marketClose;
    }

    private async Task ExecuteBuyAsync()
    {
        var symbol = NewTickerSymbol?.Trim().ToUpperInvariant();
        if (string.IsNullOrWhiteSpace(symbol))
        {
            BuyStatus = "Enter a ticker symbol";
            return;
        }

        if (InvestAmount <= 0)
        {
            BuyStatus = "Enter invest amount > 0";
            return;
        }

        if (!_monitor.IsConnected)
        {
            BuyStatus = "Not connected — go to Monitor tab and click Connect first";
            return;
        }

        if (!IsUsMarketOpen())
        {
            BuyStatus = "Market is closed — US market hours are 9:30 AM - 4:00 PM ET, Mon-Fri";
            return;
        }

        // Check if already held
        var existing = _database.GetPosition(symbol);
        if (existing is not null)
        {
            BuyStatus = $"{symbol} is already held in your portfolio";
            return;
        }

        // Check available cash before buying
        BuyStatus = $"Checking available cash...";
        try
        {
            var cashInfo = await _tradingBridge.GetCashAsync();
            if (cashInfo.Error is null && cashInfo.Cash < (double)InvestAmount)
            {
                BuyStatus = $"Insufficient cash: ${cashInfo.Cash:F0} available, need ${InvestAmount:F0}";
                return;
            }
        }
        catch { /* Non-fatal — proceed with buy */ }

        // Warn about recent stop-loss on same symbol (within 24h)
        try
        {
            var recentTrades = _database.GetRecentTrades(symbol, hours: 24);
            var recentStopLoss = recentTrades.FirstOrDefault(t =>
                t.Action == "SELL" && t.Note is not null && t.Note.Contains("StopLoss", StringComparison.OrdinalIgnoreCase));
            if (recentStopLoss is not null)
            {
                BuyStatus = $"WARNING: {symbol} was stop-loss sold in last 24h — proceed with caution";
                // Don't return — just warn; user can retry
            }
        }
        catch { /* Non-fatal */ }

        BuyStatus = $"Getting price for {symbol}...";

        try
        {
            var polygonPrice = await _polygonPrices.GetPriceAsync(symbol);
            if (polygonPrice is null || polygonPrice <= 0)
            {
                BuyStatus = $"Could not get Polygon price for {symbol}";
                return;
            }

            var shares = (int)Math.Floor(InvestAmount / polygonPrice.Value);
            if (shares <= 0)
            {
                BuyStatus = $"{symbol} @ ${polygonPrice:F2} — ${InvestAmount:F0} not enough for 1 share";
                return;
            }

            var intentId = TraderDatabase.GenerateIntentId("Manual", symbol, "BUY");
            if (_database.IsIntentExecuted(intentId))
            {
                BuyStatus = $"{symbol} already bought today";
                return;
            }

            BuyStatus = $"Buying {shares} x {symbol} @ ~${polygonPrice:F2} (${InvestAmount:F0})...";
            _database.TryInsertTradeIntent(intentId, "Manual", symbol, "BUY", shares);

            var buyResult = await _tradingBridge.BuyAsync(symbol, shares);
            if (!buyResult.Success)
            {
                _database.MarkIntentFailed(intentId, buyResult.Error ?? buyResult.Status ?? "unknown error");
                BuyStatus = $"Buy failed: {buyResult.Error ?? buyResult.Status ?? "unknown error"}";
                return;
            }

            _database.MarkIntentExecuted(intentId, buyResult.OrderId);

            var filled = buyResult.FilledQty > 0 ? buyResult.FilledQty : shares;
            var entryPrice = buyResult.AvgPrice > 0 ? (decimal)buyResult.AvgPrice : polygonPrice.Value;

            // Manual buy: strategy='Manual', holdDays=999999 (only manual sell or stop loss)
            _database.InsertPosition(symbol, "Manual", filled, entryPrice, holdDays: 999999, note: "Manual");
            _database.InsertTrade(symbol, "BUY", filled, entryPrice, $"Manual buy ${InvestAmount:F0}");
            _database.AddManualTicker(symbol, "Manual Buy");

            _monitor.InvalidatePositionCache();

            BuyStatus = $"Bought {filled} x {symbol} @ ${entryPrice:F2} (total ${filled * entryPrice:F2})";
            NewTickerSymbol = string.Empty;
            InvestAmount = 0;
            await RefreshWithPricesAsync();
        }
        catch (Exception ex)
        {
            BuyStatus = $"Error: {ex.Message}";
        }
    }

    private async Task DeleteSelectedAsync()
    {
        if (SelectedPosition is null)
        {
            return;
        }

        var symbol = SelectedPosition.Symbol;
        var shares = SelectedPosition.Shares;

        if (shares > 0)
        {
            if (!_monitor.IsConnected)
            {
                BuyStatus = $"Cannot sell {symbol} — not connected. Go to Monitor tab and click Connect first.";
                return;
            }

            var intentId = TraderDatabase.GenerateIntentId("Manual", symbol, "SELL");
            if (_database.IsIntentExecuted(intentId))
            {
                BuyStatus = $"{symbol} already sold today";
                return;
            }

            BuyStatus = $"Selling {shares} x {symbol}...";
            _database.TryInsertTradeIntent(intentId, "Manual", symbol, "SELL", shares);

            try
            {
                var result = await _tradingBridge.SellAsync(symbol, shares);

                if (result.Success)
                {
                    // Guard: treat FilledQty=0 as failure (prevents phantom position deletion)
                    var filled = result.FilledQty > 0 ? result.FilledQty : 0;
                    if (filled == 0)
                    {
                        _database.MarkIntentFailed(intentId, $"No fills: status={result.Status}");
                        BuyStatus = $"FAILED sell {symbol}: Success but 0 shares filled";
                        return;
                    }

                    _database.MarkIntentExecuted(intentId, result.OrderId);

                    decimal fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : 0m;
                    if (fillPrice <= 0)
                    {
                        var polygonPrice = await _polygonPrices.GetPriceAsync(symbol);
                        fillPrice = polygonPrice ?? SelectedPosition.EntryPrice;
                    }

                    _database.InsertTrade(symbol, "SELL", filled, fillPrice, "Manual sell from Database page");
                    if (result.RemainingQty > 0)
                        _database.UpdatePositionShares(symbol, result.RemainingQty);
                    else
                        _database.DeletePosition(symbol);
                    _monitor.InvalidatePositionCache();

                    BuyStatus = $"Sold {filled} x {symbol} @ ${fillPrice:F2} (market)";
                }
                else
                {
                    _database.MarkIntentFailed(intentId, result.Error ?? result.Status ?? "unknown");
                    BuyStatus = $"Sell failed for {symbol}: {result.Error ?? result.Status ?? "unknown"}";
                }
            }
            catch (Exception ex)
            {
                BuyStatus = $"Sell error for {symbol}: {ex.Message}";
            }
        }

        // Also delete from tickers table
        _database.DeleteTicker(symbol);
        await RefreshWithPricesAsync();
    }
}

public sealed class UnifiedPositionRowViewModel
{
    public UnifiedPositionRowViewModel(string symbol, string strategy, int shares, decimal entryPrice, decimal currentPrice,
        double pnlPct, DateTime enteredAt, DateTime? scheduledExit, double? targetWeight)
    {
        Symbol = symbol;
        Strategy = strategy;
        Shares = shares;
        EntryPrice = entryPrice;
        CurrentPrice = currentPrice;
        PnlPct = pnlPct;
        EnteredAt = enteredAt;
        ScheduledExit = scheduledExit;
        TargetWeight = targetWeight.HasValue ? targetWeight.Value * 100.0 : null;
    }

    public string Symbol { get; }
    public string Strategy { get; }
    public int Shares { get; }
    public decimal EntryPrice { get; }
    public decimal CurrentPrice { get; }
    public double PnlPct { get; }
    public DateTime EnteredAt { get; }
    public DateTime? ScheduledExit { get; }
    public double? TargetWeight { get; }
}
