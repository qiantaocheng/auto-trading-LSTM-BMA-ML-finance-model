using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
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
    private string _newTickerSymbol = string.Empty;
    private decimal _investAmount;
    private string _buyStatus = string.Empty;
    private TickerRecordViewModel? _selectedTicker;

    public DatabaseViewModel(TraderDatabase database, PythonTradingBridge tradingBridge, PolygonPriceService polygonPrices, MonitorViewModel monitor)
    {
        _database = database;
        _tradingBridge = tradingBridge;
        _polygonPrices = polygonPrices;
        _monitor = monitor;
        Tickers = new ObservableCollection<TickerRecordViewModel>();
        BuyCommand = new AsyncRelayCommand(ExecuteBuyAsync);
        DeleteSelectedCommand = new AsyncRelayCommand(DeleteSelectedAsync, () => SelectedTicker is not null);
        RefreshCommand = new AsyncRelayCommand(LoadAsync);
        _ = LoadAsync();
    }

    public ObservableCollection<TickerRecordViewModel> Tickers { get; }

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

    public TickerRecordViewModel? SelectedTicker
    {
        get => _selectedTicker;
        set
        {
            _selectedTicker = value;
            RaisePropertyChanged();
            DeleteSelectedCommand.RaiseCanExecuteChanged();
        }
    }

    public AsyncRelayCommand BuyCommand { get; }
    public AsyncRelayCommand DeleteSelectedCommand { get; }
    public AsyncRelayCommand RefreshCommand { get; }

    private Task LoadAsync()
    {
        var records = _database.GetTickerRecords();
        Tickers.Clear();
        foreach (var record in records)
        {
            Tickers.Add(new TickerRecordViewModel(record.Symbol, record.Tag, record.Source, record.AddedAt));
        }
        return Task.CompletedTask;
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

        BuyStatus = $"Getting price for {symbol}...";

        try
        {
            // Get current price from Polygon API
            var polygonPrice = await _polygonPrices.GetPriceAsync(symbol);
            if (polygonPrice is null || polygonPrice <= 0)
            {
                BuyStatus = $"Could not get Polygon price for {symbol}";
                return;
            }

            // Calculate shares: floor(amount / price)
            var shares = (int)Math.Floor(InvestAmount / polygonPrice.Value);
            if (shares <= 0)
            {
                BuyStatus = $"{symbol} @ ${polygonPrice:F2} — ${InvestAmount:F0} not enough for 1 share";
                return;
            }

            BuyStatus = $"Buying {shares} x {symbol} @ ~${polygonPrice:F2} (${InvestAmount:F0})...";

            // Execute buy via IBKR
            var buyResult = await _tradingBridge.BuyAsync(symbol, shares);
            if (!buyResult.Success)
            {
                BuyStatus = $"Buy failed: {buyResult.Error ?? buyResult.Status ?? "unknown error"}";
                return;
            }

            var avgPrice = (decimal)buyResult.AvgPrice;
            if (avgPrice <= 0) avgPrice = polygonPrice.Value;

            // Record in database
            _database.InsertAutoPosition(symbol, shares, avgPrice);
            _database.InsertTrade(symbol, "BUY", shares, avgPrice, $"Manual buy ${InvestAmount:F0}");
            _database.AddManualTicker(symbol, "Manual Buy");

            // Invalidate position cache so Monitor picks up the new holding
            _monitor.InvalidatePositionCache();

            BuyStatus = $"Bought {shares} x {symbol} @ ${avgPrice:F2} (total ${shares * avgPrice:F2})";
            NewTickerSymbol = string.Empty;
            InvestAmount = 0;
            await LoadAsync();
        }
        catch (Exception ex)
        {
            BuyStatus = $"Error: {ex.Message}";
        }
    }

    private async Task DeleteSelectedAsync()
    {
        if (SelectedTicker is null)
        {
            return;
        }

        var symbol = SelectedTicker.Symbol;

        // Check if this ticker has an active position in auto_positions
        var position = _database.GetAutoPosition(symbol);
        if (position is not null && position.Shares > 0)
        {
            // Sell the position first
            if (!_monitor.IsConnected)
            {
                BuyStatus = $"Cannot sell {symbol} — not connected. Go to Monitor tab and click Connect first.";
                return;
            }

            BuyStatus = $"Selling {position.Shares} x {symbol}...";

            try
            {
                var result = await _tradingBridge.SellAsync(symbol, position.Shares);
                var fillPrice = result.Success && result.AvgPrice > 0
                    ? (decimal)result.AvgPrice
                    : position.EntryPrice;

                // Record trade
                _database.InsertTrade(symbol, "SELL", position.Shares, fillPrice, "Manual sell from Database page");

                // Delete from auto_positions
                _database.DeleteAutoPosition(symbol);

                // Invalidate position cache
                _monitor.InvalidatePositionCache();

                BuyStatus = result.Success
                    ? $"Sold {position.Shares} x {symbol} @ ${fillPrice:F2}"
                    : $"Sell sent for {symbol} (status: {result.Status ?? result.Error ?? "unknown"})";
            }
            catch (Exception ex)
            {
                BuyStatus = $"Sell error for {symbol}: {ex.Message}";
                // Still delete from tickers below so user can retry
            }
        }

        // Delete from tickers table
        _database.DeleteTicker(symbol);
        await LoadAsync();
    }
}

public sealed class TickerRecordViewModel
{
    public TickerRecordViewModel(string symbol, int tag, string source, DateTime addedAt)
    {
        Symbol = symbol;
        Tag = tag;
        Source = source;
        AddedAt = addedAt;
    }

    public string Symbol { get; }
    public int Tag { get; }
    public string Source { get; }
    public DateTime AddedAt { get; }

    public string TagLabel => Tag == 1 ? "Manual" : "Auto";
}
