using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.PythonBridge.Services;

public sealed class PythonEarningsBridge
{
    private readonly string _pythonExe;
    private readonly string _scriptPath;
    private readonly string _polygonApiKey;
    private readonly string _parquetPath;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public PythonEarningsBridge(string pythonExe, string scriptPath, string polygonApiKey, string parquetPath)
    {
        _pythonExe = pythonExe;
        _scriptPath = scriptPath;
        _polygonApiKey = polygonApiKey;
        _parquetPath = parquetPath;
    }

    public async Task<IReadOnlyList<EarningsResult>> ScanAsync(
        Action<EarningsProgress>? onProgress = null,
        CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{_scriptPath}\" --polygon-key {_polygonApiKey} --parquet-path \"{_parquetPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath)
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start earnings scanner process");

        var stderrTask = Task.Run(async () =>
        {
            var reader = process.StandardError;
            while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var progress = JsonSerializer.Deserialize<EarningsProgress>(line, JsonOptions);
                    if (progress is not null)
                    {
                        onProgress?.Invoke(progress);
                    }
                }
                catch { }
            }
        }, ct);

        var stdoutTask = process.StandardOutput.ReadToEndAsync();

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromMinutes(15));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { process.Kill(true); } catch { }
            throw new InvalidOperationException("Earnings scanner timed out after 15 minutes");
        }

        await stderrTask.ConfigureAwait(false);
        var stdout = await stdoutTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"Earnings scanner failed (exit {process.ExitCode}): {stdout}");
        }

        var results = JsonSerializer.Deserialize<List<EarningsResult>>(stdout.Trim(), JsonOptions);
        return results ?? new List<EarningsResult>();
    }
}

public sealed record EarningsResult(
    [property: JsonPropertyName("ticker")] string Ticker,
    [property: JsonPropertyName("direction")] string Direction,
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("published")] string Published,
    [property: JsonPropertyName("gap_pct")] double GapPct);

public sealed record EarningsProgress(
    [property: JsonPropertyName("step")] string Step,
    [property: JsonPropertyName("progress")] int Progress,
    [property: JsonPropertyName("detail")] string Detail);
