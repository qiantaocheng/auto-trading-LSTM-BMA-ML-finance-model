using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.PythonBridge.Services;

public sealed class PythonTradingBridge
{
    private readonly string _pythonExe;
    private readonly string _scriptPath;
    private readonly Func<(string host, int port, int clientId)> _getConnectionParams;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public PythonTradingBridge(string pythonExe, string scriptPath, Func<(string host, int port, int clientId)> getConnectionParams)
    {
        _pythonExe = pythonExe ?? throw new ArgumentNullException(nameof(pythonExe));
        _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
        _getConnectionParams = getConnectionParams ?? throw new ArgumentNullException(nameof(getConnectionParams));
    }

    public Task<BuyResult> BuyAsync(string symbol, int quantity, CancellationToken ct = default)
        => RunCommandAsync<BuyResult>($"buy --symbol {symbol} --quantity {quantity}", ct);

    public Task<SellResult> SellAsync(string symbol, int quantity, CancellationToken ct = default)
        => RunCommandAsync<SellResult>($"sell --symbol {symbol} --quantity {quantity}", ct);

    public Task<PriceResult> GetPriceAsync(string symbol, CancellationToken ct = default)
        => RunCommandAsync<PriceResult>($"price --symbol {symbol}", ct);

    public Task<CashResult> GetCashAsync(CancellationToken ct = default)
        => RunCommandAsync<CashResult>("cash", ct);

    public Task<MarketStatusResult> GetMarketStatusAsync(CancellationToken ct = default)
        => RunCommandAsync<MarketStatusResult>("market-status", ct);

    private async Task<T> RunCommandAsync<T>(string arguments, CancellationToken ct) where T : class
    {
        var (host, port, clientId) = _getConnectionParams();
        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{_scriptPath}\" --host {host} --port {port} --client-id {clientId} {arguments}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath)
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start python trading process");

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync(ct).ConfigureAwait(false);
        var stdout = await stdoutTask.ConfigureAwait(false);
        var stderr = await stderrTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Trading bridge failed (exit {process.ExitCode}): {stderr}");
        }

        var result = JsonSerializer.Deserialize<T>(stdout.Trim(), JsonOptions);
        if (result is null)
        {
            throw new InvalidOperationException("Invalid JSON from trading bridge");
        }

        return result;
    }
}

public sealed record BuyResult(
    [property: JsonPropertyName("success")] bool Success,
    [property: JsonPropertyName("order_id")] int OrderId,
    [property: JsonPropertyName("symbol")] string Symbol,
    [property: JsonPropertyName("action")] string Action,
    [property: JsonPropertyName("quantity")] int Quantity,
    [property: JsonPropertyName("avg_price")] double AvgPrice,
    [property: JsonPropertyName("error")] string? Error,
    [property: JsonPropertyName("status")] string? Status);

public sealed record SellResult(
    [property: JsonPropertyName("success")] bool Success,
    [property: JsonPropertyName("order_id")] int OrderId,
    [property: JsonPropertyName("symbol")] string Symbol,
    [property: JsonPropertyName("action")] string Action,
    [property: JsonPropertyName("quantity")] int Quantity,
    [property: JsonPropertyName("avg_price")] double AvgPrice,
    [property: JsonPropertyName("error")] string? Error,
    [property: JsonPropertyName("status")] string? Status);

public sealed record PriceResult(
    [property: JsonPropertyName("symbol")] string Symbol,
    [property: JsonPropertyName("price")] double? Price,
    [property: JsonPropertyName("error")] string? Error);

public sealed record CashResult(
    [property: JsonPropertyName("cash")] double Cash,
    [property: JsonPropertyName("net_liq")] double NetLiq,
    [property: JsonPropertyName("error")] string? Error);

public sealed record MarketStatusResult(
    [property: JsonPropertyName("is_open")] bool IsOpen,
    [property: JsonPropertyName("is_friday")] bool IsFriday,
    [property: JsonPropertyName("et_time")] string? EtTime,
    [property: JsonPropertyName("et_hour")] int EtHour,
    [property: JsonPropertyName("et_minute")] int EtMinute,
    [property: JsonPropertyName("weekday")] int Weekday,
    [property: JsonPropertyName("error")] string? Error);
