using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Threading;
using Trader.App.ViewModels;
using Trader.Core.Services;
using Trader.Core.Repositories;

namespace Trader.App.ViewModels.Pages;

public class MonitorViewModel : ViewModelBase
{
    private readonly IPortfolioService _portfolioService;
    private readonly PolygonPriceService _polygonPrices;
    private readonly TraderDatabase _database;
    private readonly IWritableOptionsService _writableOptions;
    private readonly DispatcherTimer _timer;
    private bool _isRefreshing;
    private decimal _netLiq;
    private decimal _cash;
    private DateTimeOffset? _lastUpdate;
    private DateTimeOffset? _lastHistorySample;
    private readonly TimeSpan _historySampleInterval = TimeSpan.FromMinutes(1);

    private string _predictionCurrentStep = "Idle";
    private int _predictionProgress;
    private bool _isPredictionRunning;

    private string _tradingStatus = "Idle";
    private bool _isAutoTradingEnabled;
    private int _autoPositionCount;

    private string _hmmState = "N/A";
    private double _riskGate = 1.0;
    private double _pCrisisSmooth;
    private bool _isCrisisMode;
    private int _rebalanceDaysRemaining;
    private string _tradingMode = "Paper";
    private bool _isConnected;
    private int _clientId;

    // Cache IBKR positions (symbol+qty+cash) — updated from portfolio snapshot
    private decimal _ibkrCash;
    private bool _ibkrCacheInitialized;
    private readonly System.Collections.Generic.Dictionary<string, int> _ibkrPositions = new(StringComparer.OrdinalIgnoreCase);

    public MonitorViewModel(IPortfolioService portfolioService, PolygonPriceService polygonPrices, TraderDatabase database, Microsoft.Extensions.Options.IOptions<Trader.Core.Options.IBKROptions> ibkrOptions, IWritableOptionsService writableOptions)
    {
        _tradingMode = ibkrOptions.Value.Mode;
        _clientId = ibkrOptions.Value.ClientId;
        _portfolioService = portfolioService;
        _polygonPrices = polygonPrices;
        _database = database;
        _writableOptions = writableOptions;
        Holdings = new ObservableCollection<HoldingViewModel>();
        CapitalHistory = new ObservableCollection<CapitalPoint>();
        PredictionLog = new ObservableCollection<PredictionLogEntry>();
        TradingLog = new ObservableCollection<TradingLogEntry>();

        // Load capital history from database
        LoadCapitalHistoryFromDatabase();

        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(5)
        };
        _timer.Tick += async (_, _) => await RefreshAsync();
        // Timer is started manually by Connect button, not automatically
    }

    private void LoadCapitalHistoryFromDatabase()
    {
        try
        {
            var history = _database.GetCapitalHistory(limit: 1440); // Last 24 hours at 1-minute intervals
            foreach (var record in history)
            {
                CapitalHistory.Add(new CapitalPoint(new DateTimeOffset(record.Timestamp, TimeSpan.Zero), record.NetLiq));
            }
            if (history.Count > 0)
            {
                _lastHistorySample = new DateTimeOffset(history[^1].Timestamp, TimeSpan.Zero);
            }
        }
        catch
        {
            // Ignore errors loading history - will start fresh
        }
    }

    public ObservableCollection<HoldingViewModel> Holdings { get; }
    public ObservableCollection<CapitalPoint> CapitalHistory { get; }
    public ObservableCollection<PredictionLogEntry> PredictionLog { get; }
    public ObservableCollection<TradingLogEntry> TradingLog { get; }

    public string PredictionCurrentStep
    {
        get => _predictionCurrentStep;
        set
        {
            _predictionCurrentStep = value;
            RaisePropertyChanged();
        }
    }

    public int PredictionProgress
    {
        get => _predictionProgress;
        set
        {
            _predictionProgress = value;
            RaisePropertyChanged();
        }
    }

    public bool IsPredictionRunning
    {
        get => _isPredictionRunning;
        set
        {
            _isPredictionRunning = value;
            RaisePropertyChanged();
        }
    }

    public void ReportPredictionProgress(string step, int progress, string detail)
    {
        PredictionCurrentStep = step;
        PredictionProgress = progress;
        IsPredictionRunning = progress >= 0 && progress < 100;
        PredictionLog.Add(new PredictionLogEntry(DateTimeOffset.Now, step, progress, detail));
        while (PredictionLog.Count > 200)
        {
            PredictionLog.RemoveAt(0);
        }
    }

    public void ResetPredictionProgress()
    {
        PredictionCurrentStep = "Idle";
        PredictionProgress = 0;
        IsPredictionRunning = false;
    }

    // --- Trading Status ---

    public string TradingStatus
    {
        get => _tradingStatus;
        set
        {
            _tradingStatus = value;
            RaisePropertyChanged();
        }
    }

    public bool IsAutoTradingEnabled
    {
        get => _isAutoTradingEnabled;
        set
        {
            _isAutoTradingEnabled = value;
            RaisePropertyChanged();
        }
    }

    public int AutoPositionCount
    {
        get => _autoPositionCount;
        set
        {
            _autoPositionCount = value;
            RaisePropertyChanged();
        }
    }

    public void ReportTradingEvent(string action, string detail)
    {
        TradingStatus = action;
        TradingLog.Add(new TradingLogEntry(DateTimeOffset.Now, action, detail));
        while (TradingLog.Count > 200)
        {
            TradingLog.RemoveAt(0);
        }
    }

    // --- HMM Status ---

    public string HmmState
    {
        get => _hmmState;
        set
        {
            _hmmState = value;
            RaisePropertyChanged();
        }
    }

    public double RiskGate
    {
        get => _riskGate;
        set
        {
            _riskGate = value;
            RaisePropertyChanged();
        }
    }

    public double PCrisisSmooth
    {
        get => _pCrisisSmooth;
        set
        {
            _pCrisisSmooth = value;
            RaisePropertyChanged();
        }
    }

    public bool IsCrisisMode
    {
        get => _isCrisisMode;
        set
        {
            _isCrisisMode = value;
            RaisePropertyChanged();
        }
    }

    public int RebalanceDaysRemaining
    {
        get => _rebalanceDaysRemaining;
        set
        {
            _rebalanceDaysRemaining = value;
            RaisePropertyChanged();
        }
    }

    public string TradingMode
    {
        get => _tradingMode;
        set
        {
            _tradingMode = value;
            RaisePropertyChanged();
        }
    }

    public bool IsConnected
    {
        get => _isConnected;
        set
        {
            _isConnected = value;
            RaisePropertyChanged();
        }
    }

    public int ClientId
    {
        get => _clientId;
        set
        {
            _clientId = value;
            RaisePropertyChanged();
        }
    }

    public void UpdateClientId(int clientId)
    {
        if (clientId < 0)
        {
            return;
        }

        // Disconnect if currently connected
        if (_isConnected)
        {
            Disconnect();
        }

        // Update client ID
        _writableOptions.UpdateClientId(clientId);
        ClientId = clientId;
    }

    public async void Connect()
    {
        if (!_isConnected)
        {
            IsConnected = true;
            _timer.Start();
            await RefreshAsync(); // Immediate first refresh so data shows instantly
        }
    }

    public void Disconnect()
    {
        if (_isConnected)
        {
            IsConnected = false;
            _timer.Stop();
        }
    }

    public void SwitchTradingMode(string newMode)
    {
        if (newMode != "Paper" && newMode != "Live")
        {
            return;
        }

        // Disconnect if currently connected
        if (_isConnected)
        {
            Disconnect();
        }

        // Update mode
        _writableOptions.UpdateTradingMode(newMode);
        TradingMode = newMode;
    }

    public void ReportHmmStatus(string state, double riskGate, double pCrisis, bool crisis, int rebalDays)
    {
        HmmState = state;
        RiskGate = riskGate;
        PCrisisSmooth = pCrisis;
        IsCrisisMode = crisis;
        RebalanceDaysRemaining = rebalDays;
    }

    public decimal NetLiquidation
    {
        get => _netLiq;
        private set
        {
            _netLiq = value;
            RaisePropertyChanged();
        }
    }

    public decimal CashBalance
    {
        get => _cash;
        private set
        {
            _cash = value;
            RaisePropertyChanged();
        }
    }

    public DateTimeOffset? LastUpdate
    {
        get => _lastUpdate;
        private set
        {
            _lastUpdate = value;
            RaisePropertyChanged();
        }
    }

    private async Task RefreshAsync()
    {
        if (_isRefreshing)
        {
            return;
        }

        _isRefreshing = true;
        try
        {
            // Step 1: Get IBKR positions (symbols + quantities + cash) — only on first call or after invalidation
            if (!_ibkrCacheInitialized)
            {
                _ibkrCacheInitialized = true; // Mark immediately so we don't retry the slow Python path every 5s
                try
                {
                    var snapshot = await _portfolioService.GetSnapshotAsync().ConfigureAwait(true);
                    _ibkrCash = snapshot.Cash;
                    _ibkrPositions.Clear();
                    foreach (var h in snapshot.Holdings)
                    {
                        if (h.Quantity != 0)
                            _ibkrPositions[h.Symbol] = h.Quantity;
                    }
                }
                catch
                {
                    // If IBKR fails, use DB positions as fallback
                    _ibkrPositions.Clear();
                    var dbPositions = _database.GetAutoPositions();
                    foreach (var p in dbPositions)
                    {
                        _ibkrPositions[p.Symbol] = p.Shares;
                    }
                }
            }

            // Step 2: Fetch Polygon prices for all positions (fast HTTP, no Python)
            var symbols = _ibkrPositions.Keys.ToList();
            var prices = symbols.Count > 0
                ? await Task.Run(() => _polygonPrices.GetPricesAsync(symbols)).ConfigureAwait(true)
                : new System.Collections.Generic.Dictionary<string, decimal>();

            // Step 3: Build holdings with Polygon prices
            var holdings = new System.Collections.Generic.List<PortfolioHolding>();
            decimal stockValue = 0;
            foreach (var (symbol, qty) in _ibkrPositions)
            {
                var price = prices.TryGetValue(symbol, out var p) ? p : 0m;
                var value = qty * price;
                stockValue += value;
                holdings.Add(new PortfolioHolding(symbol, qty, price));
            }

            // Step 4: Calculate net_liq and update UI
            var netLiq = _ibkrCash + stockValue;
            NetLiquidation = netLiq;
            CashBalance = _ibkrCash;
            LastUpdate = DateTimeOffset.UtcNow;

            Holdings.Clear();
            foreach (var holding in holdings.OrderByDescending(h => h.MarketValue))
            {
                Holdings.Add(new HoldingViewModel(holding.Symbol, holding.Quantity, holding.MarketPrice, holding.MarketValue));
            }

            // Step 5: Build snapshot for capital tracking and DB sync
            var builtSnapshot = new PortfolioSnapshot(netLiq, _ibkrCash, holdings);
            CaptureCapitalPoint(builtSnapshot);
            SyncHoldingsToDatabase(builtSnapshot);
        }
        catch (Exception ex)
        {
            Holdings.Clear();
            Holdings.Add(new HoldingViewModel($"Error: {ex.Message}", 0, 0, 0));
        }
        finally
        {
            _isRefreshing = false;
        }
    }

    /// <summary>
    /// Force re-fetch of IBKR positions on next refresh (e.g. after buy/sell).
    /// </summary>
    public void InvalidatePositionCache()
    {
        _ibkrPositions.Clear();
        _ibkrCacheInitialized = false;
    }

    private void SyncHoldingsToDatabase(PortfolioSnapshot snapshot)
    {
        try
        {
            var dbPositions = _database.GetAutoPositions();
            var dbSymbols = new HashSet<string>(dbPositions.Select(p => p.Symbol), StringComparer.OrdinalIgnoreCase);
            var ibkrSymbols = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // Add IBKR holdings not in DB
            foreach (var holding in snapshot.Holdings)
            {
                if (holding.Quantity <= 0) continue;
                ibkrSymbols.Add(holding.Symbol);
                if (!dbSymbols.Contains(holding.Symbol))
                {
                    _database.InsertAutoPosition(holding.Symbol, holding.Quantity, holding.MarketPrice);
                }
            }

            // Remove DB positions no longer in IBKR
            foreach (var pos in dbPositions)
            {
                if (!ibkrSymbols.Contains(pos.Symbol))
                {
                    _database.DeleteAutoPosition(pos.Symbol);
                }
            }

            AutoPositionCount = _database.GetAutoPositions().Count;
        }
        catch
        {
            // Ignore sync errors — UI still works
        }
    }

    private void CaptureCapitalPoint(PortfolioSnapshot snapshot)
    {
        var now = DateTimeOffset.UtcNow;
        if (CapitalHistory.Count == 0 || _lastHistorySample is null || now - _lastHistorySample >= _historySampleInterval)
        {
            var netLiq = snapshot.NetLiquidation;
            var cash = snapshot.Cash;
            var stockValue = snapshot.Holdings.Sum(h => h.MarketValue);

            // Add to UI collection
            CapitalHistory.Add(new CapitalPoint(now, netLiq));
            while (CapitalHistory.Count > 1440) // Keep 24 hours at 1-minute intervals
            {
                CapitalHistory.RemoveAt(0);
            }

            // Save to database
            try
            {
                _database.InsertCapitalHistory(netLiq, cash, stockValue);
            }
            catch
            {
                // Ignore database errors - UI still works
            }

            _lastHistorySample = now;
        }
    }
}

public sealed class HoldingViewModel
{
    public HoldingViewModel(string symbol, int quantity, decimal price, decimal value)
    {
        Symbol = symbol;
        Quantity = quantity;
        MarketPrice = price;
        MarketValue = value;
    }

    public string Symbol { get; }
    public int Quantity { get; }
    public decimal MarketPrice { get; }
    public decimal MarketValue { get; }
}

public sealed record CapitalPoint(DateTimeOffset Timestamp, decimal NetLiq);

public sealed record PredictionLogEntry(DateTimeOffset Timestamp, string Step, int Progress, string Detail);

public sealed record TradingLogEntry(DateTimeOffset Timestamp, string Action, string Detail);
