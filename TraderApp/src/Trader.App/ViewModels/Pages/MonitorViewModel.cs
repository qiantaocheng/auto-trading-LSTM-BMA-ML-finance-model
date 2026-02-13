using System;
using System.Collections.Generic;
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
    private readonly TimeSpan _historySampleInterval = TimeSpan.FromMinutes(5);

    private string _predictionCurrentStep = "Idle";
    private int _predictionProgress;
    private bool _isPredictionRunning;

    private string _tradingStatus = "Idle";
    private bool _isAutoTradingEnabled;
    private int _autoPositionCount;
    private int _pendingBuyCount;

    private string _hmmState = "N/A";
    private double _riskGate = 1.0;
    private double _pCrisisSmooth;
    private bool _isCrisisMode;
    private int _rebalanceDaysRemaining;
    private string _tradingMode = "Paper";
    private bool _isConnected;
    private int _clientId;

    // ETF Rotation status
    private double _etfExposure;
    private double _etfRiskCap = 1.0;
    private int _etfPositionCount;
    private int _etfTradingDaysRemaining = 21;
    private string _etfStalenessLevel = "None"; // None, Yellow, Red
    private string _vixTriggerMode = "baseline";
    private bool _vixModeActive = false;
    private double? _vixPrice;
    private double? _themeBudget;
    private double? _hmmPRisk;

    // Chart range
    private string _selectedRange = "1D";
    private DateTime _rangeStart;
    private double _chartReturn;
    private double _return1D;
    private double _return1W;
    private double _return1M;
    private double _return6M;
    private double _return1Y;
    private const int MaxChartPoints = 600;

    // Cache IBKR positions (symbol+qty+cash) — updated from portfolio snapshot
    private decimal _ibkrCash;
    private bool _ibkrCacheInitialized;
    private readonly Dictionary<string, int> _ibkrPositions = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, decimal> _ibkrAvgCosts = new(StringComparer.OrdinalIgnoreCase);

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
        ChartCapitalHistory = new ObservableCollection<CapitalPoint>();
        PredictionLog = new ObservableCollection<PredictionLogEntry>();
        TradingLog = new ObservableCollection<TradingLogEntry>();

        // Load chart with default range
        _rangeStart = GetRangeStart("1D");
        LoadChartFromDatabase();

        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(5)
        };
        _timer.Tick += async (_, _) => await RefreshAsync();
        // Timer is started manually by Connect button, not automatically
    }

    private void LoadChartFromDatabase()
    {
        try
        {
            // Trim old records (>365 days) on startup
            _database.TrimCapitalHistory(365);

            var records = _database.GetCapitalHistorySince(_rangeStart);
            ChartCapitalHistory.Clear();

            // Downsample if too many points
            var sampled = Downsample(records, MaxChartPoints);
            foreach (var record in sampled)
            {
                ChartCapitalHistory.Add(new CapitalPoint(new DateTimeOffset(record.Timestamp, TimeSpan.Zero), record.NetLiq));
            }

            // Also set _lastHistorySample from the latest record in DB
            if (records.Count > 0)
            {
                _lastHistorySample = new DateTimeOffset(records[^1].Timestamp, TimeSpan.Zero);
            }

            UpdateReturns();
        }
        catch
        {
            // Ignore errors loading history - will start fresh
        }
    }

    private static IReadOnlyList<CapitalHistoryRecord> Downsample(IReadOnlyList<CapitalHistoryRecord> records, int maxPoints)
    {
        if (records.Count <= maxPoints)
        {
            return records;
        }

        var result = new List<CapitalHistoryRecord>(maxPoints);
        var step = (double)records.Count / maxPoints;
        for (int i = 0; i < maxPoints - 1; i++)
        {
            var idx = (int)(i * step);
            result.Add(records[idx]);
        }
        // Always include the last point
        result.Add(records[^1]);
        return result;
    }

    private void UpdateReturns()
    {
        try
        {
            var currentNetLiq = _netLiq > 0 ? _netLiq : (ChartCapitalHistory.Count > 0 ? ChartCapitalHistory[^1].NetLiq : 0);
            if (currentNetLiq <= 0) return;

            Return1D = CalcReturn(currentNetLiq, TimeSpan.FromDays(1));
            Return1W = CalcReturn(currentNetLiq, TimeSpan.FromDays(7));
            Return1M = CalcReturn(currentNetLiq, TimeSpan.FromDays(30));
            Return6M = CalcReturn(currentNetLiq, TimeSpan.FromDays(182));
            Return1Y = CalcReturn(currentNetLiq, TimeSpan.FromDays(365));

            // Chart return = return for the selected range
            ChartReturn = _selectedRange switch
            {
                "1D" => Return1D,
                "1W" => Return1W,
                "1M" => Return1M,
                "6M" => Return6M,
                "1Y" => Return1Y,
                _ => 0
            };
        }
        catch
        {
            // Ignore calculation errors
        }
    }

    private double CalcReturn(decimal currentNetLiq, TimeSpan lookback)
    {
        var since = DateTime.UtcNow - lookback;
        var first = _database.GetFirstCapitalRecordSince(since);
        if (first is null || first.NetLiq <= 0) return 0;
        return (double)((currentNetLiq - first.NetLiq) / first.NetLiq * 100);
    }

    public ObservableCollection<HoldingViewModel> Holdings { get; }
    public ObservableCollection<CapitalPoint> CapitalHistory { get; }
    public ObservableCollection<CapitalPoint> ChartCapitalHistory { get; }
    public ObservableCollection<PredictionLogEntry> PredictionLog { get; }
    public ObservableCollection<TradingLogEntry> TradingLog { get; }

    // --- Chart Range ---

    public string SelectedRange
    {
        get => _selectedRange;
        set
        {
            if (_selectedRange == value) return;
            _selectedRange = value;
            RaisePropertyChanged();
            _rangeStart = GetRangeStart(value);
            LoadChartFromDatabase();
        }
    }

    public double ChartReturn
    {
        get => _chartReturn;
        private set
        {
            _chartReturn = value;
            RaisePropertyChanged();
        }
    }

    public double Return1D
    {
        get => _return1D;
        private set
        {
            _return1D = value;
            RaisePropertyChanged();
        }
    }

    public double Return1W
    {
        get => _return1W;
        private set
        {
            _return1W = value;
            RaisePropertyChanged();
        }
    }

    public double Return1M
    {
        get => _return1M;
        private set
        {
            _return1M = value;
            RaisePropertyChanged();
        }
    }

    public double Return6M
    {
        get => _return6M;
        private set
        {
            _return6M = value;
            RaisePropertyChanged();
        }
    }

    public double Return1Y
    {
        get => _return1Y;
        private set
        {
            _return1Y = value;
            RaisePropertyChanged();
        }
    }

    public void SelectRange(string range)
    {
        SelectedRange = range;
    }

    private static DateTime GetRangeStart(string range)
    {
        var now = DateTime.UtcNow;
        return range switch
        {
            "1D" => now.AddDays(-1),
            "1W" => now.AddDays(-7),
            "1M" => now.AddMonths(-1),
            "6M" => now.AddMonths(-6),
            "1Y" => now.AddYears(-1),
            _ => now.AddDays(-1),
        };
    }

    // --- Prediction Progress ---

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

    public int PendingBuyCount
    {
        get => _pendingBuyCount;
        set { _pendingBuyCount = value; RaisePropertyChanged(); }
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

        // Switch to mode-specific database
        var newDbPath = System.IO.Path.Combine(AppContext.BaseDirectory, $"TraderApp_{newMode}.db");
        _database.SwitchDatabase(newDbPath);

        // Safety: auto-disable auto-trading when switching to Live
        if (newMode == "Live")
        {
            IsAutoTradingEnabled = false;
        }

        // Reload chart data from new DB
        LoadChartFromDatabase();
    }

    public void ReportHmmStatus(string state, double riskGate, double pCrisis, bool crisis, int rebalDays)
    {
        HmmState = state;
        RiskGate = riskGate;
        PCrisisSmooth = pCrisis;
        IsCrisisMode = crisis;
        RebalanceDaysRemaining = rebalDays;
    }

    // --- ETF Rotation Status ---

    public double EtfExposure
    {
        get => _etfExposure;
        set { _etfExposure = value; RaisePropertyChanged(); }
    }

    public double EtfRiskCap
    {
        get => _etfRiskCap;
        set { _etfRiskCap = value; RaisePropertyChanged(); }
    }

    public int EtfPositionCount
    {
        get => _etfPositionCount;
        set { _etfPositionCount = value; RaisePropertyChanged(); }
    }

    public int EtfTradingDaysRemaining
    {
        get => _etfTradingDaysRemaining;
        set { _etfTradingDaysRemaining = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(EtfCountdownDisplay)); }
    }

    /// <summary>
    /// Shows "Xd" for trading days, or hours countdown when within 1 trading day.
    /// Weekends and holidays do not affect the display.
    /// </summary>
    public string EtfCountdownDisplay
    {
        get
        {
            if (_etfTradingDaysRemaining <= 0)
                return "Today";
            if (_etfTradingDaysRemaining == 1)
            {
                // Calculate hours until next trading day market open (9:30 ET)
                try
                {
                    var nextOpen = Trader.Core.Repositories.TraderDatabase.NextTradingDayOpenUtc(DateTime.UtcNow);
                    var hoursLeft = (nextOpen - DateTime.UtcNow).TotalHours;
                    if (hoursLeft <= 0) return "Today";
                    if (hoursLeft < 1) return $"{(int)(hoursLeft * 60)}m";
                    return $"{(int)hoursLeft}h {(int)(hoursLeft % 1 * 60)}m";
                }
                catch
                {
                    return "1d";
                }
            }
            return $"{_etfTradingDaysRemaining}d";
        }
    }

    public string VixTriggerMode
    {
        get => _vixTriggerMode;
        set { _vixTriggerMode = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(VixTriggerDisplay)); }
    }

    public bool VixModeActive
    {
        get => _vixModeActive;
        set { _vixModeActive = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(VixTriggerDisplay)); }
    }

    public double? VixPrice
    {
        get => _vixPrice;
        set { _vixPrice = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(VixDisplay)); }
    }

    public double? ThemeBudget
    {
        get => _themeBudget;
        set { _themeBudget = value; RaisePropertyChanged(); }
    }

    public string VixTriggerDisplay => _hmmPRisk.HasValue
        ? $"HMM {_hmmPRisk.Value:F2}"
        : _vixModeActive ? "VIX Active" : "Baseline";

    public string VixDisplay => _vixPrice.HasValue ? $"VIX {_vixPrice.Value:F1}" : "VIX N/A";

    public double? HmmPRisk
    {
        get => _hmmPRisk;
        set { _hmmPRisk = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(VixTriggerDisplay)); }
    }

    public string EtfStalenessLevel
    {
        get => _etfStalenessLevel;
        set { _etfStalenessLevel = value; RaisePropertyChanged(); }
    }

    public void ReportEtfRotationStatus(double exposure, double riskCap, int posCount, int tradingDaysRemaining,
        string? vixTriggerMode = null, bool vixModeActive = false, double? vixPrice = null, double? themeBudget = null,
        double? hmmPRisk = null, string? stalenessLevel = null)
    {
        EtfExposure = exposure;
        EtfRiskCap = riskCap;
        EtfPositionCount = posCount;
        EtfTradingDaysRemaining = tradingDaysRemaining;
        VixTriggerMode = vixTriggerMode ?? "baseline";
        VixModeActive = vixModeActive;
        VixPrice = vixPrice;
        ThemeBudget = themeBudget;
        HmmPRisk = hmmPRisk;
        EtfStalenessLevel = stalenessLevel ?? "None";
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
                    _ibkrAvgCosts.Clear();
                    foreach (var h in snapshot.Holdings)
                    {
                        if (h.Quantity != 0)
                        {
                            _ibkrPositions[h.Symbol] = h.Quantity;
                            if (h.AvgCost > 0)
                                _ibkrAvgCosts[h.Symbol] = h.AvgCost;
                        }
                    }
                }
                catch
                {
                    // If IBKR fails, use DB positions as fallback
                    _ibkrPositions.Clear();
                    var dbPositions = _database.GetPositions();
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
                : new Dictionary<string, decimal>();

            // Step 3: Build holdings with Polygon prices
            var holdings = new List<PortfolioHolding>();
            decimal stockValue = 0;
            foreach (var (symbol, qty) in _ibkrPositions)
            {
                var price = prices.TryGetValue(symbol, out var p) ? p : 0m;
                var value = qty * price;
                stockValue += value;
                var avgCost = _ibkrAvgCosts.TryGetValue(symbol, out var ac) ? ac : 0m;
                holdings.Add(new PortfolioHolding(symbol, qty, price, avgCost));
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

            // Step 5: Persist current prices to positions table
            if (prices.Count > 0)
            {
                try { _database.UpdatePositionPrices(prices); } catch { }
            }

            // Step 6: Build snapshot for capital tracking and DB sync
            var builtSnapshot = new PortfolioSnapshot(netLiq, _ibkrCash, holdings);
            CaptureCapitalPoint(builtSnapshot);
            SyncHoldingsToDatabase(builtSnapshot);

            // Step 6: Update return calculations
            UpdateReturns();
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
        _ibkrAvgCosts.Clear();
        _ibkrCacheInitialized = false;
    }

    /// <summary>
    /// Returns the latest live IBKR positions with Polygon prices.
    /// Returns null if not yet connected / no data.
    /// </summary>
    public IReadOnlyList<HoldingViewModel>? GetLiveHoldings()
        => _ibkrCacheInitialized && Holdings.Count > 0 ? Holdings.ToList() : null;

    private void SyncHoldingsToDatabase(PortfolioSnapshot snapshot)
    {
        try
        {
            // Write to broker_positions ONLY — never touch positions table.
            // positions is the strategy truth (entry_price, scheduled_exit, strategy).
            // broker_positions is a pure IBKR mirror (qty, avg_cost, market_value).
            var brokerData = snapshot.Holdings
                .Where(h => h.Quantity != 0)
                .Select(h => (h.Symbol, h.Quantity, h.AvgCost > 0 ? h.AvgCost : h.MarketPrice, h.MarketValue));
            _database.ReplaceBrokerPositions(brokerData);

            // Sync entry prices from broker avg_cost + auto-seed if positions empty
            _database.SyncEntryPricesFromBroker();
            _database.SeedPositionsFromBrokerIfEmpty();

            AutoPositionCount = _database.GetPositions().Count;
            PendingBuyCount = _database.GetPendingBuys().Count;
        }
        catch
        {
            // Ignore sync errors — UI still works
        }
    }

    private void CaptureCapitalPoint(PortfolioSnapshot snapshot)
    {
        var now = DateTimeOffset.UtcNow;
        if (_lastHistorySample is null || now - _lastHistorySample >= _historySampleInterval)
        {
            // Only record during US market hours (Mon-Fri 9:30-16:00 ET)
            try
            {
                var et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                var etNow = TimeZoneInfo.ConvertTimeFromUtc(now.UtcDateTime, et);
                var isWeekday = etNow.DayOfWeek >= DayOfWeek.Monday && etNow.DayOfWeek <= DayOfWeek.Friday;
                var isMarketHours = etNow.TimeOfDay >= new TimeSpan(9, 30, 0) && etNow.TimeOfDay <= new TimeSpan(16, 0, 0);
                if (!isWeekday || !isMarketHours)
                {
                    _lastHistorySample = now; // Reset timer so we don't spam checks
                    return;
                }
            }
            catch { /* If timezone lookup fails, record anyway */ }
            var netLiq = snapshot.NetLiquidation;
            var cash = snapshot.Cash;
            var stockValue = snapshot.Holdings.Sum(h => h.MarketValue);

            var point = new CapitalPoint(now, netLiq);

            // Add to chart if within selected range
            if (now.UtcDateTime >= _rangeStart)
            {
                ChartCapitalHistory.Add(point);
                // Trim points outside the range
                while (ChartCapitalHistory.Count > 0 && ChartCapitalHistory[0].Timestamp.UtcDateTime < _rangeStart)
                {
                    ChartCapitalHistory.RemoveAt(0);
                }
            }

            // Save to database (permanent storage) — only if broker is connected
            if (IsConnected && netLiq > 0)
            {
                try
                {
                    _database.InsertCapitalHistory(netLiq, cash, stockValue);
                }
                catch
                {
                    // Ignore database errors - UI still works
                }
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
