using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.PythonBridge.Services;

public sealed class PythonHmmBridge
{
    private readonly string _pythonExe;
    private readonly string _scriptPath;
    private readonly string _polygonApiKey;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public PythonHmmBridge(string pythonExe, string scriptPath, string polygonApiKey)
    {
        _pythonExe = pythonExe ?? throw new ArgumentNullException(nameof(pythonExe));
        _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
        _polygonApiKey = polygonApiKey ?? "";
    }

    public async Task ResetRebalanceCounterAsync(CancellationToken ct = default)
    {
        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{_scriptPath}\" --reset-counter",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath)
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start HMM bridge process");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { process.Kill(); } catch { }
            throw new InvalidOperationException("HMM reset-counter timed out after 30s");
        }
    }

    public async Task<HmmResult> GetRiskAssessmentAsync(
        Action<HmmProgress>? onProgress = null,
        CancellationToken ct = default)
    {
        var args = $"\"{_scriptPath}\"";
        if (!string.IsNullOrEmpty(_polygonApiKey))
        {
            args += $" --polygon-key {_polygonApiKey}";
        }

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(_scriptPath)
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start HMM bridge process");

        // Read stderr line-by-line for progress updates
        var stderrTask = Task.Run(async () =>
        {
            var reader = process.StandardError;
            while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var progress = JsonSerializer.Deserialize<HmmProgress>(line, JsonOptions);
                    if (progress is not null)
                    {
                        onProgress?.Invoke(progress);
                    }
                }
                catch
                {
                    // Non-JSON stderr line — ignore
                }
            }
        }, ct);

        var stdoutTask = process.StandardOutput.ReadToEndAsync();

        // 60-second timeout to prevent hung processes
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(60));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { process.Kill(); } catch { }
            throw new InvalidOperationException("HMM bridge timed out after 60s");
        }

        await stderrTask.ConfigureAwait(false);
        var stdout = await stdoutTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"HMM bridge failed (exit {process.ExitCode}): {stdout}");
        }

        var result = JsonSerializer.Deserialize<HmmResult>(stdout.Trim(), JsonOptions);
        if (result is null)
        {
            throw new InvalidOperationException("Invalid JSON from HMM bridge");
        }

        return result;
    }
}

public sealed record HmmResult(
    [property: JsonPropertyName("p_crisis")] double PCrisis,
    [property: JsonPropertyName("p_crisis_smooth")] double PCrisisSmooth,
    [property: JsonPropertyName("risk_gate")] double RiskGate,
    [property: JsonPropertyName("hmm_state")] string HmmState,
    [property: JsonPropertyName("crisis_mode")] bool CrisisMode,
    [property: JsonPropertyName("crisis_confirm_days")] int CrisisConfirmDays,
    [property: JsonPropertyName("safe_confirm_days")] int SafeConfirmDays,
    [property: JsonPropertyName("cooldown_remaining")] int CooldownRemaining,
    [property: JsonPropertyName("rebalance_day_counter")] int RebalanceDayCounter,
    [property: JsonPropertyName("training_days")] int TrainingDays,
    [property: JsonPropertyName("features_date")] string? FeaturesDate,
    [property: JsonPropertyName("error")] string? Error);

public sealed record HmmProgress(
    [property: JsonPropertyName("step")] string Step,
    [property: JsonPropertyName("progress")] int Progress,
    [property: JsonPropertyName("detail")] string Detail);
