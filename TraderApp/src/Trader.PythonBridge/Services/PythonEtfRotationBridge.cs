using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Trader.PythonBridge.Services;

public sealed class PythonEtfRotationBridge
{
    private readonly string _pythonExe;
    private readonly string _scriptPath;
    private readonly string _polygonApiKey;
    private readonly Func<string> _stateDbPathFunc;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public PythonEtfRotationBridge(string pythonExe, string scriptPath, string polygonApiKey, Func<string> stateDbPathFunc)
    {
        _pythonExe = pythonExe ?? throw new ArgumentNullException(nameof(pythonExe));
        _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
        _polygonApiKey = polygonApiKey ?? "";
        _stateDbPathFunc = stateDbPathFunc ?? throw new ArgumentNullException(nameof(stateDbPathFunc));
    }

    public async Task<EtfRotationResult> GetTargetWeightsAsync(
        string? riskLevel = null,
        bool useHmm = false,
        Action<EtfRotationProgress>? onProgress = null,
        CancellationToken ct = default)
    {
        var args = $"\"{_scriptPath}\"";
        if (!string.IsNullOrEmpty(_polygonApiKey))
        {
            args += $" --polygon-key {_polygonApiKey}";
        }
        var stateDbPath = _stateDbPathFunc();
        if (!string.IsNullOrEmpty(stateDbPath))
        {
            args += $" --state-db \"{stateDbPath}\"";
        }
        if (useHmm)
        {
            // HMM mode: Python handles portfolio selection via HMM, no --risk-level needed
            args += " --use-hmm";
        }
        else if (!string.IsNullOrEmpty(riskLevel))
        {
            args += $" --risk-level {riskLevel}";
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
            ?? throw new InvalidOperationException("Failed to start ETF rotation bridge process");

        // Read stderr for progress updates
        var stderrTask = Task.Run(async () =>
        {
            var reader = process.StandardError;
            while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    var progress = JsonSerializer.Deserialize<EtfRotationProgress>(line, JsonOptions);
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

        // 120-second timeout (HMM mode needs extra time for training + extended data fetch)
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(120));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { process.Kill(true); } catch { }
            throw new InvalidOperationException("ETF rotation bridge timed out after 60s");
        }

        await stderrTask.ConfigureAwait(false);
        var stdout = await stdoutTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"ETF rotation bridge failed (exit {process.ExitCode}): {stdout}");
        }

        var result = JsonSerializer.Deserialize<EtfRotationResult>(stdout.Trim(), JsonOptions);
        if (result is null)
        {
            throw new InvalidOperationException("Invalid JSON from ETF rotation bridge");
        }

        return result;
    }

    /// <summary>
    /// Call Python to get NYSE holidays for current + next year using pandas_market_calendars.
    /// </summary>
    public async Task<List<DateTime>> GetNyseHolidaysAsync(CancellationToken ct = default)
    {
        var args = $"\"{_scriptPath}\" --nyse-holidays";

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
            ?? throw new InvalidOperationException("Failed to start NYSE holidays process");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { process.Kill(true); } catch { }
            throw new InvalidOperationException("NYSE holidays query timed out");
        }

        var stdout = await process.StandardOutput.ReadToEndAsync().ConfigureAwait(false);
        var result = JsonSerializer.Deserialize<NyseHolidaysResult>(stdout.Trim(), JsonOptions);
        if (result?.Holidays is null)
            return new List<DateTime>();

        return result.Holidays
            .Select(s => DateTime.TryParse(s, out var dt) ? dt : (DateTime?)null)
            .Where(dt => dt.HasValue)
            .Select(dt => dt!.Value.Date)
            .ToList();
    }
}

public sealed record NyseHolidaysResult(
    [property: JsonPropertyName("holidays")] List<string>? Holidays,
    [property: JsonPropertyName("years")] List<int>? Years,
    [property: JsonPropertyName("error")] string? Error);

public sealed record EtfRotationResult(
    [property: JsonPropertyName("exposure")] double Exposure,
    [property: JsonPropertyName("risk_cap")] double RiskCap,
    [property: JsonPropertyName("blended_vol")] double BlendedVol,
    [property: JsonPropertyName("spy_price")] double SpyPrice,
    [property: JsonPropertyName("ma200")] double Ma200,
    [property: JsonPropertyName("spy_deviation")] double SpyDeviation,
    [property: JsonPropertyName("vol_target_exposure")] double VolTargetExposure,
    [property: JsonPropertyName("vix_price")] double? VixPrice,
    [property: JsonPropertyName("vix_trigger_mode")] string? VixTriggerMode,
    [property: JsonPropertyName("vix_trigger_reason")] string? VixTriggerReason,
    [property: JsonPropertyName("vix_trigger_enable_count")] int? VixTriggerEnableCount,
    [property: JsonPropertyName("vix_trigger_disable_count")] int? VixTriggerDisableCount,
    [property: JsonPropertyName("vix_mode_active")] bool VixModeActive,
    [property: JsonPropertyName("vix_cap_applied")] bool VixCapApplied,
    [property: JsonPropertyName("theme_budget")] double? ThemeBudget,
    [property: JsonPropertyName("risk_level")] string? RiskLevel,
    [property: JsonPropertyName("portfolio_used")] string? PortfolioUsed,
    [property: JsonPropertyName("etf_weights")] Dictionary<string, double>? EtfWeights,
    [property: JsonPropertyName("cash_weight")] double CashWeight,
    [property: JsonPropertyName("etf_prices")] Dictionary<string, double>? EtfPrices,
    [property: JsonPropertyName("asof_trading_day")] string? AsofTradingDay,
    [property: JsonPropertyName("has_data_anomaly")] bool HasDataAnomaly,
    [property: JsonPropertyName("anomaly_tickers")] List<string>? AnomalyTickers,
    [property: JsonPropertyName("blocked_anomaly_tickers")] List<string>? BlockedAnomalyTickers,
    [property: JsonPropertyName("bars_last_date")] Dictionary<string, string?>? BarsLastDate,
    [property: JsonPropertyName("hmm_p_risk_smooth")] double? HmmPRiskSmooth,
    [property: JsonPropertyName("use_hmm")] bool UseHmm,
    [property: JsonPropertyName("error")] string? Error);

public sealed record EtfRotationProgress(
    [property: JsonPropertyName("step")] string Step,
    [property: JsonPropertyName("progress")] int Progress,
    [property: JsonPropertyName("detail")] string Detail);
