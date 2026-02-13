using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Windows;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Win32;
using Trader.App.Services;
using Trader.App.ViewModels;
using Trader.App.ViewModels.Pages;
using Trader.Core.Options;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Serilog;
using Trader.PythonBridge.Services;

namespace Trader.App;

public partial class App : Application
{
    private IHost? _host;
    private static Mutex? _instanceMutex;
    private static EventWaitHandle? _showWindowEvent;

    // Names for cross-process synchronisation
    private const string MutexName = "Global\\TraderAutoPilot_Instance";
    private const string ShowEventName = "Global\\TraderAutoPilot_ShowWindow";

    public IServiceProvider Services => _host?.Services ?? throw new InvalidOperationException("Application host not initialized.");

    protected override void OnStartup(StartupEventArgs e)
    {
        // ── Single-instance guard ────────────────────────────────────────────
        _instanceMutex = new Mutex(initiallyOwned: true, MutexName, out bool createdNew);
        if (!createdNew)
        {
            // Another instance is already running — signal it to come to front, then quit
            try
            {
                using var ev = EventWaitHandle.OpenExisting(ShowEventName);
                ev.Set();
            }
            catch { }
            _instanceMutex.Dispose();
            Shutdown();
            return;
        }

        // Create the inter-process "show window" event
        _showWindowEvent = new EventWaitHandle(
            initialState: false,
            mode: EventResetMode.AutoReset,
            name: ShowEventName);

        base.OnStartup(e);

        // ── Serilog rolling file logger ──────────────────────────────────────
        var logPath = Path.Combine(AppContext.BaseDirectory, "logs", "trader-.log");
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.File(logPath,
                rollingInterval: RollingInterval.Day,
                retainedFileCountLimit: 30,
                fileSizeLimitBytes: 50 * 1024 * 1024,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] {SourceContext}: {Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        _host = Host.CreateDefaultBuilder()
            .UseSerilog()
            .ConfigureAppConfiguration((context, config) =>
            {
                config.SetBasePath(AppContext.BaseDirectory);
                config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
            })
            .ConfigureServices((context, services) =>
            {
                services.Configure<PythonOptions>(context.Configuration.GetSection("Python"));
                services.Configure<PolygonOptions>(context.Configuration.GetSection("Polygon"));
                services.Configure<DatabaseOptions>(context.Configuration.GetSection("Database"));
                services.Configure<EtfRotationOptions>(context.Configuration.GetSection("EtfRotation"));
                services.Configure<IBKROptions>(context.Configuration.GetSection("IBKR"));
                services.Configure<AutoConnectOptions>(context.Configuration.GetSection("AutoConnect"));
                services.Configure<TelegramOptions>(context.Configuration.GetSection("Telegram"));

                services.AddHttpClient();
                services.AddSingleton(sp =>
                {
                    var options = sp.GetRequiredService<IOptions<DatabaseOptions>>().Value;
                    var ibkr = sp.GetRequiredService<IOptions<IBKROptions>>().Value;
                    var baseName = Path.GetFileNameWithoutExtension(options.FileName ?? "TraderApp.db");
                    var ext = Path.GetExtension(options.FileName ?? "TraderApp.db");
                    var dbPath = Path.Combine(AppContext.BaseDirectory, $"{baseName}_{ibkr.Mode}{ext}");
                    var db = new TraderDatabase(dbPath);
                    db.EnsureSchema();
                    return db;
                });

                services.AddSingleton(sp =>
                {
                    var python = sp.GetRequiredService<IOptions<PythonOptions>>().Value;
                    var exe = python.Executable;
                    var script = python.PredictScript;
                    if (!File.Exists(exe))
                    {
                        throw new FileNotFoundException($"Python executable not found: {exe}");
                    }
                    if (!File.Exists(script))
                    {
                        throw new FileNotFoundException($"Prediction script not found: {script}");
                    }
                    return new PythonPredictionBridge(exe, script);
                });

                services.AddSingleton(sp =>
                {
                    var httpFactory = sp.GetRequiredService<IHttpClientFactory>();
                    var polygon = sp.GetRequiredService<IOptions<PolygonOptions>>().Value;
                    return new PolygonCalendarService(httpFactory.CreateClient(nameof(PolygonCalendarService)), polygon.ApiKey);
                });

                services.AddSingleton(sp =>
                {
                    var httpFactory = sp.GetRequiredService<IHttpClientFactory>();
                    var polygon = sp.GetRequiredService<IOptions<PolygonOptions>>().Value;
                    return new PolygonPriceService(httpFactory.CreateClient(nameof(PolygonPriceService)), polygon.ApiKey);
                });

                services.AddSingleton<IPortfolioService, PythonPortfolioService>();

                services.AddSingleton(sp =>
                {
                    var python = sp.GetRequiredService<IOptions<PythonOptions>>().Value;
                    var ibkrOptions = sp.GetRequiredService<IOptionsMonitor<IBKROptions>>();
                    var exe = python.Executable;
                    var tradingScript = python.TradingScript;
                    if (string.IsNullOrEmpty(tradingScript) || !File.Exists(tradingScript))
                    {
                        throw new FileNotFoundException($"Trading script not found: {tradingScript}");
                    }
                    return new PythonTradingBridge(exe, tradingScript, () =>
                    {
                        var ibkr = ibkrOptions.CurrentValue;
                        return (ibkr.Host, ibkr.GetPort(), ibkr.ClientId);
                    });
                });

                services.AddSingleton(sp =>
                {
                    var python = sp.GetRequiredService<IOptions<PythonOptions>>().Value;
                    var polygon = sp.GetRequiredService<IOptions<PolygonOptions>>().Value;
                    var exe = python.Executable;
                    var hmmScript = python.HmmScript;
                    if (string.IsNullOrEmpty(hmmScript) || !File.Exists(hmmScript))
                    {
                        throw new FileNotFoundException($"HMM script not found: {hmmScript}");
                    }
                    var db = sp.GetRequiredService<TraderDatabase>();
                    return new PythonHmmBridge(exe, hmmScript, polygon.ApiKey, () => db.CurrentDbPath);
                });

                services.AddSingleton(sp =>
                {
                    var python = sp.GetRequiredService<IOptions<PythonOptions>>().Value;
                    var polygon = sp.GetRequiredService<IOptions<PolygonOptions>>().Value;
                    var exe = python.Executable;
                    var earningsScript = python.EarningsScript;
                    var parquetPath = python.ParquetPath;
                    if (string.IsNullOrEmpty(earningsScript) || !File.Exists(earningsScript))
                    {
                        throw new FileNotFoundException($"Earnings script not found: {earningsScript}");
                    }
                    return new PythonEarningsBridge(exe, earningsScript, polygon.ApiKey, parquetPath);
                });

                services.AddSingleton<IWritableOptionsService>(sp =>
                {
                    var ibkrOptions = sp.GetRequiredService<IOptionsMonitor<IBKROptions>>();
                    var autoConnectOptions = sp.GetRequiredService<IOptionsMonitor<AutoConnectOptions>>();
                    var pythonOptions = sp.GetRequiredService<IOptionsMonitor<PythonOptions>>();
                    var appsettingsPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
                    return new WritableOptionsService(ibkrOptions, autoConnectOptions, pythonOptions, appsettingsPath);
                });

                services.AddSingleton(sp =>
                {
                    var python = sp.GetRequiredService<IOptions<PythonOptions>>().Value;
                    var polygon = sp.GetRequiredService<IOptions<PolygonOptions>>().Value;
                    var exe = python.Executable;
                    var etfScript = python.EtfRotationScript;
                    if (string.IsNullOrEmpty(etfScript) || !File.Exists(etfScript))
                    {
                        throw new FileNotFoundException($"ETF rotation script not found: {etfScript}");
                    }
                    var db = sp.GetRequiredService<TraderDatabase>();
                    return new PythonEtfRotationBridge(exe, etfScript, polygon.ApiKey, () => db.CurrentDbPath);
                });

                services.AddSingleton<TradingSchedulerService>();
                services.AddHostedService(sp => sp.GetRequiredService<TradingSchedulerService>());

                services.AddSingleton<EtfRotationSchedulerService>();
                services.AddHostedService(sp => sp.GetRequiredService<EtfRotationSchedulerService>());

                services.AddSingleton(sp => new IbcGatewayService(
                    sp.GetRequiredService<IOptionsMonitor<AutoConnectOptions>>(),
                    sp.GetRequiredService<IOptionsMonitor<IBKROptions>>(),
                    sp.GetRequiredService<MonitorViewModel>(),
                    sp.GetRequiredService<ILogger<IbcGatewayService>>()));
                services.AddHostedService(sp => sp.GetRequiredService<IbcGatewayService>());

                services.AddSingleton(sp => new DirectPredictionViewModel(
                    sp.GetRequiredService<PythonPredictionBridge>(),
                    sp.GetRequiredService<PythonEarningsBridge>(),
                    sp.GetRequiredService<TraderDatabase>(),
                    sp.GetRequiredService<MonitorViewModel>(),
                    sp.GetRequiredService<IOptionsMonitor<PythonOptions>>(),
                    sp.GetRequiredService<IWritableOptionsService>()));
                services.AddSingleton<MonitorViewModel>();
                services.AddSingleton(sp => new DatabaseViewModel(
                    sp.GetRequiredService<TraderDatabase>(),
                    sp.GetRequiredService<PythonTradingBridge>(),
                    sp.GetRequiredService<PolygonPriceService>(),
                    sp.GetRequiredService<MonitorViewModel>()));
                services.AddSingleton<ConnectionViewModel>();
                services.AddSingleton<ShellViewModel>();
            })
            .Build();

        _host.Start();

        // ── Load NYSE holiday calendar from pandas_market_calendars ──────────
        _ = Task.Run(async () =>
        {
            try
            {
                var bridge = _host.Services.GetRequiredService<PythonEtfRotationBridge>();
                var holidays = await bridge.GetNyseHolidaysAsync();
                if (holidays.Count > 0)
                {
                    TraderDatabase.LoadNyseHolidays(holidays);
                    Log.Logger.Information("Loaded {Count} NYSE holidays from pandas_market_calendars", holidays.Count);
                }
            }
            catch (Exception ex)
            {
                Log.Logger.Warning(ex, "Failed to load NYSE holidays — using hardcoded fallback");
            }
        });

        var mainWindow = new MainWindow(_host.Services.GetRequiredService<ShellViewModel>());
        MainWindow = mainWindow;
        mainWindow.Show();

        // ── Listen for "show window" signals from future second-instance attempts ──
        var showEvent = _showWindowEvent;
        var dispatcher = Dispatcher;
        Thread listenerThread = new(() =>
        {
            while (showEvent is not null)
            {
                try { showEvent.WaitOne(); }
                catch { break; }
                dispatcher.BeginInvoke(() => mainWindow.ShowFromTray());
            }
        })
        { IsBackground = true, Name = "SingleInstanceListener" };
        listenerThread.Start();

        // Register auto-start on Windows boot
        EnsureStartupRegistration();
    }

    private static void EnsureStartupRegistration()
    {
        try
        {
            var exePath = Environment.ProcessPath;
            if (string.IsNullOrEmpty(exePath)) return;

            using var key = Registry.CurrentUser.OpenSubKey(
                @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", writable: true);
            if (key is null) return;

            var existing = key.GetValue("TraderAutoPilot") as string;
            if (!string.Equals(existing, exePath, StringComparison.OrdinalIgnoreCase))
            {
                key.SetValue("TraderAutoPilot", exePath);
            }
        }
        catch
        {
            // Non-critical — skip if registry access fails
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        base.OnExit(e);
        // Release single-instance resources
        var ev = Interlocked.Exchange(ref _showWindowEvent, null);
        ev?.Dispose();
        try { _instanceMutex?.ReleaseMutex(); } catch { }
        _instanceMutex?.Dispose();
        // Stop background services
        if (_host is not null)
        {
            _host.StopAsync().GetAwaiter().GetResult();
            _host.Dispose();
        }
        Log.CloseAndFlush();
    }
}
