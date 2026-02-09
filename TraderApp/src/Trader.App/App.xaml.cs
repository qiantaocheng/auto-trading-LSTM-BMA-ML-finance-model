using System;
using System.IO;
using System.Net.Http;
using System.Windows;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Trader.App.Services;
using Trader.App.ViewModels;
using Trader.App.ViewModels.Pages;
using Trader.Core.Options;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Trader.PythonBridge.Services;

namespace Trader.App;

public partial class App : Application
{
    private IHost? _host;

    public IServiceProvider Services => _host?.Services ?? throw new InvalidOperationException("Application host not initialized.");

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        _host = Host.CreateDefaultBuilder()
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
                services.Configure<IBKROptions>(context.Configuration.GetSection("IBKR"));

                services.AddHttpClient();
                services.AddSingleton(sp =>
                {
                    var options = sp.GetRequiredService<IOptions<DatabaseOptions>>().Value;
                    var dbPath = Path.Combine(AppContext.BaseDirectory, options.FileName ?? "TraderApp.db");
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
                    return new PythonHmmBridge(exe, hmmScript, polygon.ApiKey);
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
                    var appsettingsPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
                    return new WritableOptionsService(ibkrOptions, appsettingsPath);
                });

                services.AddSingleton<TradingSchedulerService>();
                services.AddHostedService(sp => sp.GetRequiredService<TradingSchedulerService>());

                services.AddSingleton<DirectPredictionViewModel>();
                services.AddSingleton<MonitorViewModel>();
                services.AddSingleton(sp => new DatabaseViewModel(
                    sp.GetRequiredService<TraderDatabase>(),
                    sp.GetRequiredService<PythonTradingBridge>(),
                    sp.GetRequiredService<PolygonPriceService>(),
                    sp.GetRequiredService<MonitorViewModel>()));
                services.AddSingleton<ShellViewModel>();
            })
            .Build();

        _host.Start();

        var mainWindow = new MainWindow(_host.Services.GetRequiredService<ShellViewModel>());
        MainWindow = mainWindow;
        mainWindow.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        base.OnExit(e);
        if (_host is not null)
        {
            _host.StopAsync().GetAwaiter().GetResult();
            _host.Dispose();
        }
    }
}
