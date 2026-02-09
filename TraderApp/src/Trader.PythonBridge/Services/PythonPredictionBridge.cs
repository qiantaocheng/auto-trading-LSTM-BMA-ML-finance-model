using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.PythonBridge.Services;

public sealed record PredictionProgress(
    [property: JsonPropertyName("step")] string Step,
    [property: JsonPropertyName("progress")] int Progress,
    [property: JsonPropertyName("detail")] string Detail);

public sealed class PythonPredictionBridge
{
    private readonly string _pythonExe;
    private readonly string _scriptPath;

    public PythonPredictionBridge(string pythonExe, string scriptPath)
    {
        _pythonExe = pythonExe ?? throw new ArgumentNullException(nameof(pythonExe));
        _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
    }

    public async Task<PredictionBridgeResult> RunAsync(
        string snapshotId,
        Action<PredictionProgress>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{_scriptPath}\" --snapshot {snapshotId}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath)
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start python process");

        // Read stderr line-by-line in background for progress updates
        var stderrTask = Task.Run(async () =>
        {
            var reader = process.StandardError;
            while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var progress = JsonSerializer.Deserialize<PredictionProgress>(
                        line, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                    if (progress is not null)
                    {
                        onProgress?.Invoke(progress);
                    }
                }
                catch
                {
                    // Non-JSON stderr line (library warnings etc.) -- ignore
                }
            }
        }, cancellationToken);

        // Read all of stdout (the final JSON result)
        var stdoutTask = process.StandardOutput.ReadToEndAsync();

        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        await stderrTask.ConfigureAwait(false);
        var stdout = await stdoutTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"Python prediction failed (exit {process.ExitCode}): {stdout}");
        }

        var payload = JsonSerializer.Deserialize<PredictionBridgeResult>(
            stdout.Trim(),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        if (payload is null)
        {
            throw new InvalidOperationException("Invalid payload returned from python");
        }
        return payload;
    }
}

public sealed record PredictionBridgeResult(
    [property: JsonPropertyName("run_id")] string RunId,
    [property: JsonPropertyName("as_of")] string AsOf,
    [property: JsonPropertyName("excel_path")] string ExcelPath,
    [property: JsonPropertyName("top20")] IReadOnlyList<PredictionScore> Top20,
    [property: JsonPropertyName("top10")] IReadOnlyList<string> Top10);

public sealed record PredictionScore(
    [property: JsonPropertyName("ticker")] string Ticker,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("ema4")] double Ema4);
