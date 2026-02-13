using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Trader.Core.Options;
using Trader.Core.Repositories;

namespace Trader.Core.Services;

public interface IPortfolioService
{
    Task<PortfolioSnapshot> GetSnapshotAsync(CancellationToken cancellationToken = default);
}

public sealed record PortfolioSnapshot(decimal NetLiquidation, decimal Cash, IReadOnlyList<PortfolioHolding> Holdings);

public sealed record PortfolioHolding(string Symbol, int Quantity, decimal MarketPrice, decimal AvgCost = 0m)
{
    public decimal MarketValue => Quantity * MarketPrice;
}

public sealed class PythonPortfolioService : IPortfolioService
{
    private readonly ILogger<PythonPortfolioService> _logger;
    private readonly string _pythonExe;
    private readonly string _portfolioScript;
    private readonly IOptionsMonitor<IBKROptions> _ibkrOptions;
    private readonly string _polygonApiKey;

    public PythonPortfolioService(IOptions<PythonOptions> pythonOptions, IOptionsMonitor<IBKROptions> ibkrOptions, IOptions<PolygonOptions> polygonOptions, ILogger<PythonPortfolioService> logger)
    {
        _logger = logger;
        var python = pythonOptions.Value;
        _ibkrOptions = ibkrOptions;
        _polygonApiKey = polygonOptions.Value.ApiKey ?? string.Empty;
        _pythonExe = python.Executable ?? throw new ArgumentNullException(nameof(python.Executable));
        _portfolioScript = python.PortfolioScript ?? string.Empty;
        if (!File.Exists(_pythonExe))
        {
            throw new FileNotFoundException($"Python executable not found: {_pythonExe}");
        }
        if (!File.Exists(_portfolioScript))
        {
            throw new FileNotFoundException($"Portfolio snapshot script not found: {_portfolioScript}");
        }
    }

    public async Task<PortfolioSnapshot> GetSnapshotAsync(CancellationToken cancellationToken = default)
    {
        var ibkr = _ibkrOptions.CurrentValue;
        var host = ibkr.Host;
        var port = ibkr.GetPort();
        var clientId = ibkr.ClientId + 1; // Use different client ID to avoid conflict with trading bridge

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{_portfolioScript}\" --host {host} --port {port} --client-id {clientId} --polygon-key {_polygonApiKey}",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_portfolioScript) ?? Environment.CurrentDirectory
        };

        using var process = Process.Start(psi) ?? throw new InvalidOperationException("Failed to launch python portfolio snapshot");
        var stdOutTask = process.StandardOutput.ReadToEndAsync();
        var stdErrTask = process.StandardError.ReadToEndAsync();

        // Add 30-second timeout to prevent hung processes from blocking all future refreshes
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            try { process.Kill(true); } catch { }
            _logger.LogWarning("Portfolio snapshot timed out after 30s. Returning empty snapshot.");
            return new PortfolioSnapshot(0m, 0m, Array.Empty<PortfolioHolding>());
        }

        var output = await stdOutTask.ConfigureAwait(false);
        var error = await stdErrTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            _logger.LogWarning("Python portfolio script failed ({ExitCode}): {Error}. Returning empty snapshot.", process.ExitCode, error);
            return new PortfolioSnapshot(0m, 0m, Array.Empty<PortfolioHolding>());
        }

        try
        {
            var payload = JsonSerializer.Deserialize<PortfolioBridgePayload>(output.Trim(), new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
            if (payload is null)
            {
                _logger.LogWarning("Invalid portfolio payload from python. Returning empty snapshot.");
                return new PortfolioSnapshot(0m, 0m, Array.Empty<PortfolioHolding>());
            }

            var holdings = payload.Holdings?.Select(h => new PortfolioHolding(h.Symbol, h.Quantity, (decimal)h.MarketPrice, (decimal)h.AvgCost)).ToList() ?? new List<PortfolioHolding>();
            return new PortfolioSnapshot((decimal)payload.NetLiq, (decimal)payload.Cash, holdings);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to parse portfolio output. Returning empty snapshot.");
            return new PortfolioSnapshot(0m, 0m, Array.Empty<PortfolioHolding>());
        }
    }

    private sealed record PortfolioBridgePayload(double NetLiq, double Cash, IReadOnlyList<PortfolioBridgeHolding>? Holdings);
    private sealed record PortfolioBridgeHolding(string Symbol, int Quantity, double MarketPrice, double AvgCost = 0);
}

public sealed class MockPortfolioService : IPortfolioService
{
    private readonly TraderDatabase _database;
    private readonly Random _random = new();

    public MockPortfolioService(TraderDatabase database)
    {
        _database = database;
    }

    public Task<PortfolioSnapshot> GetSnapshotAsync(CancellationToken cancellationToken = default)
    {
        var tickers = _database.GetTickerRecords();
        var holdings = tickers
            .Select(t => new PortfolioHolding(t.Symbol, _random.Next(0, 50), (decimal)(_random.NextDouble() * 100 + 50)))
            .ToList();
        var netLiq = holdings.Sum(h => h.MarketValue);
        var cash = Math.Max(0, 1_000_000m - netLiq);
        return Task.FromResult(new PortfolioSnapshot(netLiq, cash, holdings));
    }
}
