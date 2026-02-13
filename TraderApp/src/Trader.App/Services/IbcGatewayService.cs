using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Windows;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Trader.App.ViewModels.Pages;
using Trader.Core.Options;
using Trader.Core.Repositories;

namespace Trader.App.Services;

public sealed class IbcGatewayService : BackgroundService
{
    private readonly IOptionsMonitor<AutoConnectOptions> _autoConnectOptions;
    private readonly IOptionsMonitor<IBKROptions> _ibkrOptions;
    private readonly MonitorViewModel _monitor;
    private readonly ILogger<IbcGatewayService> _logger;

    private bool _isGatewayRunning;
    private DateTime? _lastStartTime;
    private Process? _gatewayProcess;

    // Auto-connect/disconnect daily flags
    private DateTime _lastAutoActionDate = DateTime.MinValue;
    private bool _autoConnectedToday;
    private bool _autoDisconnectedToday;

    public bool IsGatewayRunning
    {
        get => _isGatewayRunning;
        private set
        {
            if (_isGatewayRunning != value)
            {
                _isGatewayRunning = value;
                GatewayStatusChanged?.Invoke(this, value);
            }
        }
    }

    public DateTime? LastStartTime => _lastStartTime;

    public event EventHandler<bool>? GatewayStatusChanged;
    public event EventHandler<string>? LogMessage;

    public IbcGatewayService(
        IOptionsMonitor<AutoConnectOptions> autoConnectOptions,
        IOptionsMonitor<IBKROptions> ibkrOptions,
        MonitorViewModel monitor,
        ILogger<IbcGatewayService> logger)
    {
        _autoConnectOptions = autoConnectOptions;
        _ibkrOptions = ibkrOptions;
        _monitor = monitor;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("IbcGatewayService started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckAndManageGateway();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in IbcGatewayService loop");
            }

            await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
        }
    }

    private async Task CheckAndManageGateway()
    {
        // Always check gateway status
        var port = _ibkrOptions.CurrentValue.GetPort();
        IsGatewayRunning = await CheckPortAsync(port);

        var opts = _autoConnectOptions.CurrentValue;
        if (!opts.Enabled)
            return;

        // Get current Eastern Time
        var et = GetEasternTime();

        // Reset daily flags on date change
        if (et.Date != _lastAutoActionDate)
        {
            _lastAutoActionDate = et.Date;
            _autoConnectedToday = false;
            _autoDisconnectedToday = false;
        }

        // Skip non-trading days (weekends + NYSE holidays)
        if (!TraderDatabase.IsTradingDay(et.Date))
            return;

        // Auto-start window: (9:30 - PreMarketMinutes) to 16:00 ET
        var marketOpen = new TimeSpan(9, 30, 0);
        var autoStartTime = marketOpen.Subtract(TimeSpan.FromMinutes(opts.PreMarketMinutes));
        var marketCloseTime = new TimeSpan(16, 0, 0);
        var autoDisconnectTime = new TimeSpan(16, 5, 0);

        // ── Auto-disconnect after market close ──
        if (et.TimeOfDay >= autoDisconnectTime && !_autoDisconnectedToday)
        {
            _autoDisconnectedToday = true;

            if (_monitor.IsConnected)
            {
                _logger.LogInformation("Auto-disconnecting — market closed (ET={Time})", et.ToString("HH:mm"));
                EmitLog("Auto-disconnecting — market closed");
                try
                {
                    Application.Current.Dispatcher.Invoke(() => _monitor.Disconnect());
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to auto-disconnect MonitorViewModel");
                }
            }

            if (IsGatewayRunning)
            {
                _logger.LogInformation("Stopping IB Gateway — market closed");
                EmitLog("Stopping gateway — market closed");
                StopGateway();
            }
            return;
        }

        // ── Auto-start gateway during pre-market window ──
        if (et.TimeOfDay < autoStartTime || et.TimeOfDay > marketCloseTime)
            return;

        if (!IsGatewayRunning)
        {
            _logger.LogInformation("Auto-starting IB Gateway (pre-market window)");
            EmitLog("Auto-starting IB Gateway...");
            await StartGatewayAsync();
        }

        // ── Auto-connect MonitorViewModel when gateway comes up ──
        if (IsGatewayRunning && !_autoConnectedToday && !_monitor.IsConnected)
        {
            _autoConnectedToday = true;
            _logger.LogInformation("Auto-connecting to IBKR (gateway port available)");
            EmitLog("Auto-connecting to IBKR...");
            try
            {
                Application.Current.Dispatcher.Invoke(() => _monitor.Connect());
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to auto-connect MonitorViewModel");
            }
        }
    }

    public async Task StartGatewayAsync()
    {
        var opts = _autoConnectOptions.CurrentValue;
        var batPath = opts.StartGatewayBat;

        if (string.IsNullOrEmpty(batPath) || !File.Exists(batPath))
        {
            var msg = $"StartGateway.bat not found: {batPath}";
            _logger.LogWarning(msg);
            EmitLog(msg);
            return;
        }

        // Check if already running
        var port = _ibkrOptions.CurrentValue.GetPort();
        if (await CheckPortAsync(port))
        {
            IsGatewayRunning = true;
            EmitLog("Gateway already running");
            return;
        }

        try
        {
            EmitLog($"Launching: {batPath}");
            _gatewayProcess = Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = $"/c \"{batPath}\" /INLINE",
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(batPath),
            });

            _lastStartTime = DateTime.Now;
            _logger.LogInformation("IB Gateway launch initiated");
            EmitLog("Gateway launch initiated");

            // Wait for port to become available (up to 60s)
            for (int i = 0; i < 12; i++)
            {
                await Task.Delay(5000);
                if (await CheckPortAsync(port))
                {
                    IsGatewayRunning = true;
                    _logger.LogInformation("IB Gateway is now accepting connections on port {Port}", port);
                    EmitLog($"Gateway connected on port {port}");
                    return;
                }
            }

            EmitLog("Gateway started but port not yet available (may need 2FA)");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start IB Gateway");
            EmitLog($"Start failed: {ex.Message}");
        }
    }

    public void StopGateway()
    {
        try
        {
            // Kill IB Gateway processes
            foreach (var proc in Process.GetProcessesByName("ibgateway"))
            {
                proc.Kill();
                _logger.LogInformation("Killed ibgateway process {Pid}", proc.Id);
            }

            _gatewayProcess?.Kill();
            _gatewayProcess = null;
            IsGatewayRunning = false;
            EmitLog("Gateway stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to stop Gateway");
            EmitLog($"Stop failed: {ex.Message}");
        }
    }

    public async Task<bool> TestConnectionAsync()
    {
        var port = _ibkrOptions.CurrentValue.GetPort();
        var result = await CheckPortAsync(port);
        IsGatewayRunning = result;
        return result;
    }

    private static async Task<bool> CheckPortAsync(int port)
    {
        try
        {
            using var client = new TcpClient();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await client.ConnectAsync("127.0.0.1", port, cts.Token);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static DateTime GetEasternTime()
    {
        return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
            TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
    }

    private void EmitLog(string message)
    {
        LogMessage?.Invoke(this, $"[{DateTime.Now:HH:mm:ss}] {message}");
    }
}
