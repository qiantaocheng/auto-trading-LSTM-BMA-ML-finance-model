using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Extensions.Options;
using Trader.App.Commands;
using Trader.App.Services;
using Trader.Core.Options;
using Trader.Core.Services;

namespace Trader.App.ViewModels.Pages;

public class ConnectionViewModel : ViewModelBase
{
    private readonly IbcGatewayService _gatewayService;
    private readonly IWritableOptionsService _writableOptions;
    private readonly IOptionsMonitor<AutoConnectOptions> _autoConnectOptions;
    private readonly IOptionsMonitor<IBKROptions> _ibkrOptions;
    private readonly DispatcherTimer _statusTimer;

    private bool _autoConnectEnabled;
    private string _ibLoginId = string.Empty;
    private string _ibPassword = string.Empty;
    private string _selectedTradingMode = "Paper";
    private bool _isGatewayRunning;
    private string _gatewayStatus = "Unknown";
    private string _lastStartTime = "\u2014";
    private int _preMarketMinutes = 60;
    private bool _isBusy;

    public ConnectionViewModel(
        IbcGatewayService gatewayService,
        IWritableOptionsService writableOptions,
        IOptionsMonitor<AutoConnectOptions> autoConnectOptions,
        IOptionsMonitor<IBKROptions> ibkrOptions)
    {
        _gatewayService = gatewayService;
        _writableOptions = writableOptions;
        _autoConnectOptions = autoConnectOptions;
        _ibkrOptions = ibkrOptions;

        ConnectionLog = new ObservableCollection<string>();

        // Load current settings
        var opts = autoConnectOptions.CurrentValue;
        _autoConnectEnabled = opts.Enabled;
        _ibLoginId = opts.IbLoginId;
        _ibPassword = opts.IbPassword;
        _selectedTradingMode = opts.TradingMode.Equals("live", StringComparison.OrdinalIgnoreCase) ? "Live" : "Paper";
        _preMarketMinutes = opts.PreMarketMinutes;

        // Commands
        StartGatewayCommand = new AsyncRelayCommand(StartGatewayAsync, () => !IsBusy);
        StopGatewayCommand = new RelayCommand(StopGateway, () => !IsBusy);
        SaveSettingsCommand = new RelayCommand(SaveSettings);
        TestConnectionCommand = new AsyncRelayCommand(TestConnectionAsync, () => !IsBusy);

        // Subscribe to gateway events
        _gatewayService.GatewayStatusChanged += OnGatewayStatusChanged;
        _gatewayService.LogMessage += OnLogMessage;

        // Status refresh timer
        _statusTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(10) };
        _statusTimer.Tick += async (_, _) => await RefreshStatusAsync();
        _statusTimer.Start();

        // Initial status
        _ = RefreshStatusAsync();
    }

    // ── Properties ──────────────────────────────────────────────────

    public bool AutoConnectEnabled
    {
        get => _autoConnectEnabled;
        set
        {
            _autoConnectEnabled = value;
            RaisePropertyChanged();
            _writableOptions.UpdateAutoConnectEnabled(value);
            AddLog(value ? "Auto-connect enabled" : "Auto-connect disabled");
        }
    }

    public string IbLoginId
    {
        get => _ibLoginId;
        set { _ibLoginId = value; RaisePropertyChanged(); }
    }

    public string IbPassword
    {
        get => _ibPassword;
        set { _ibPassword = value; RaisePropertyChanged(); }
    }

    public string SelectedTradingMode
    {
        get => _selectedTradingMode;
        set { _selectedTradingMode = value; RaisePropertyChanged(); }
    }

    public bool IsGatewayRunning
    {
        get => _isGatewayRunning;
        private set { _isGatewayRunning = value; RaisePropertyChanged(); RaisePropertyChanged(nameof(GatewayStatusColor)); }
    }

    public string GatewayStatus
    {
        get => _gatewayStatus;
        private set { _gatewayStatus = value; RaisePropertyChanged(); }
    }

    public string GatewayStatusColor => IsGatewayRunning ? "#2ECC71" : "#E74C3C";

    public string LastStartTime
    {
        get => _lastStartTime;
        private set { _lastStartTime = value; RaisePropertyChanged(); }
    }

    public int PreMarketMinutes
    {
        get => _preMarketMinutes;
        set
        {
            _preMarketMinutes = value;
            RaisePropertyChanged();
            RaisePropertyChanged(nameof(ScheduleInfo));
            _writableOptions.UpdateAutoConnectPreMarket(value);
        }
    }

    public string ScheduleInfo
    {
        get
        {
            var openTime = new TimeSpan(9, 30, 0);
            var startTime = openTime.Subtract(TimeSpan.FromMinutes(_preMarketMinutes));
            return $"Gateway auto-starts at {startTime:hh\\:mm} ET on NYSE trading days";
        }
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set { _isBusy = value; RaisePropertyChanged(); }
    }

    public ObservableCollection<string> ConnectionLog { get; }

    // ── Commands ────────────────────────────────────────────────────

    public AsyncRelayCommand StartGatewayCommand { get; }
    public RelayCommand StopGatewayCommand { get; }
    public RelayCommand SaveSettingsCommand { get; }
    public AsyncRelayCommand TestConnectionCommand { get; }

    private async Task StartGatewayAsync()
    {
        IsBusy = true;
        GatewayStatus = "Starting...";
        try
        {
            await _gatewayService.StartGatewayAsync();
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void StopGateway()
    {
        _gatewayService.StopGateway();
        IsGatewayRunning = false;
        GatewayStatus = "Stopped";
    }

    private void SaveSettings()
    {
        var mode = SelectedTradingMode.Equals("Live", StringComparison.OrdinalIgnoreCase) ? "live" : "paper";
        _writableOptions.UpdateAutoConnectCredentials(IbLoginId, IbPassword, mode);

        // Also sync IBKR trading mode
        _writableOptions.UpdateTradingMode(SelectedTradingMode);

        AddLog($"Settings saved (mode={SelectedTradingMode})");
    }

    private async Task TestConnectionAsync()
    {
        IsBusy = true;
        try
        {
            var port = _ibkrOptions.CurrentValue.GetPort();
            AddLog($"Testing connection to port {port}...");
            var result = await _gatewayService.TestConnectionAsync();
            if (result)
            {
                AddLog($"Connection successful on port {port}");
                GatewayStatus = "Running";
            }
            else
            {
                AddLog($"Connection failed \u2014 port {port} not reachable");
                GatewayStatus = "Stopped";
            }
        }
        finally
        {
            IsBusy = false;
        }
    }

    // ── Event handlers ──────────────────────────────────────────────

    private void OnGatewayStatusChanged(object? sender, bool running)
    {
        Application.Current.Dispatcher.Invoke(() =>
        {
            IsGatewayRunning = running;
            GatewayStatus = running ? "Running" : "Stopped";
            if (running && _gatewayService.LastStartTime.HasValue)
                LastStartTime = _gatewayService.LastStartTime.Value.ToString("HH:mm:ss");
        });
    }

    private void OnLogMessage(object? sender, string message)
    {
        Application.Current.Dispatcher.Invoke(() => AddLog(message));
    }

    private async Task RefreshStatusAsync()
    {
        var result = await _gatewayService.TestConnectionAsync();
        IsGatewayRunning = result;
        GatewayStatus = result ? "Running" : "Stopped";
    }

    private void AddLog(string message)
    {
        var entry = message.StartsWith('[') ? message : $"[{DateTime.Now:HH:mm:ss}] {message}";
        ConnectionLog.Insert(0, entry);
        while (ConnectionLog.Count > 50)
            ConnectionLog.RemoveAt(ConnectionLog.Count - 1);
    }
}
