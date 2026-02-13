using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Trader.App.Commands;
using Trader.App.ViewModels.Pages.Models;
using Trader.Core.Options;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Trader.PythonBridge.Services;
using Microsoft.Extensions.Options;
using Trader.App.ViewModels;

namespace Trader.App.ViewModels.Pages;

public class DirectPredictionViewModel : ViewModelBase
{
    private const int RetrainCycleDays = 90; // trading days (~3 months)

    private readonly PythonPredictionBridge _bridge;
    private readonly PythonEarningsBridge _earningsBridge;
    private readonly TraderDatabase _database;
    private readonly MonitorViewModel _monitor;
    private readonly IWritableOptionsService _writableOptions;
    private readonly IOptionsMonitor<PythonOptions> _pythonOptions;
    private readonly AsyncRelayCommand _runPredictionCommand;
    private readonly AsyncRelayCommand _scanEarningsCommand;
    private readonly AsyncRelayCommand _retrainCommand;
    private DateTimeOffset? _lastRun;
    private string? _excelPath;
    private bool _isBusy;
    private bool _isEarningsBusy;
    private bool _isRetrainBusy;
    private string _snapshotId;
    private string? _statusMessage;
    private string? _earningsStatus;
    private string? _retrainStatus;
    private string _retrainCountdownText = string.Empty;

    public DirectPredictionViewModel(
        PythonPredictionBridge bridge,
        PythonEarningsBridge earningsBridge,
        TraderDatabase database,
        MonitorViewModel monitor,
        IOptionsMonitor<PythonOptions> pythonOptions,
        IWritableOptionsService writableOptions)
    {
        _bridge = bridge;
        _earningsBridge = earningsBridge;
        _database = database;
        _monitor = monitor;
        _pythonOptions = pythonOptions;
        _writableOptions = writableOptions;
        _snapshotId = string.IsNullOrWhiteSpace(pythonOptions.CurrentValue.DefaultSnapshot)
            ? "snapshot"
            : pythonOptions.CurrentValue.DefaultSnapshot;

        TopTwenty = new ObservableCollection<PredictionEntry>();
        EarningsResults = new ObservableCollection<EarningsEntry>();
        _runPredictionCommand = new AsyncRelayCommand(RunPredictionAsync, () => !IsBusy);
        _scanEarningsCommand = new AsyncRelayCommand(ScanEarningsAsync, () => !IsEarningsBusy);
        _retrainCommand = new AsyncRelayCommand(RetrainModelAsync, () => !IsRetrainBusy);

        UpdateRetrainCountdown();
    }

    public ObservableCollection<PredictionEntry> TopTwenty { get; }
    public ObservableCollection<EarningsEntry> EarningsResults { get; }

    public string SnapshotId
    {
        get => _snapshotId;
        set
        {
            if (value == _snapshotId)
            {
                return;
            }
            _snapshotId = value;
            RaisePropertyChanged();
        }
    }

    public DateTimeOffset? LastRun
    {
        get => _lastRun;
        private set
        {
            _lastRun = value;
            RaisePropertyChanged();
        }
    }

    public string? ExcelPath
    {
        get => _excelPath;
        private set
        {
            _excelPath = value;
            RaisePropertyChanged();
        }
    }

    public string? StatusMessage
    {
        get => _statusMessage;
        private set
        {
            _statusMessage = value;
            RaisePropertyChanged();
        }
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            _isBusy = value;
            RaisePropertyChanged();
            _runPredictionCommand.RaiseCanExecuteChanged();
        }
    }

    public bool IsEarningsBusy
    {
        get => _isEarningsBusy;
        private set
        {
            _isEarningsBusy = value;
            RaisePropertyChanged();
            _scanEarningsCommand.RaiseCanExecuteChanged();
        }
    }

    public string? EarningsStatus
    {
        get => _earningsStatus;
        private set
        {
            _earningsStatus = value;
            RaisePropertyChanged();
        }
    }

    public bool IsRetrainBusy
    {
        get => _isRetrainBusy;
        private set
        {
            _isRetrainBusy = value;
            RaisePropertyChanged();
            _retrainCommand.RaiseCanExecuteChanged();
        }
    }

    public string? RetrainStatus
    {
        get => _retrainStatus;
        private set
        {
            _retrainStatus = value;
            RaisePropertyChanged();
        }
    }

    public string RetrainCountdownText
    {
        get => _retrainCountdownText;
        private set
        {
            _retrainCountdownText = value;
            RaisePropertyChanged();
        }
    }

    public AsyncRelayCommand RunPredictionCommand => _runPredictionCommand;
    public AsyncRelayCommand ScanEarningsCommand => _scanEarningsCommand;
    public AsyncRelayCommand RetrainCommand => _retrainCommand;

    private async Task RunPredictionAsync()
    {
        try
        {
            IsBusy = true;
            StatusMessage = "Running prediction...";
            _monitor.ResetPredictionProgress();
            _monitor.IsPredictionRunning = true;

            var result = await _bridge.RunAsync(
                SnapshotId,
                onProgress: progress =>
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        _monitor.ReportPredictionProgress(
                            progress.Step, progress.Progress, progress.Detail);
                        StatusMessage = progress.Step;
                    });
                }).ConfigureAwait(true);

            LastRun = DateTimeOffset.UtcNow;
            ExcelPath = result.ExcelPath;

            TopTwenty.Clear();
            foreach (var entry in result.Top20)
            {
                TopTwenty.Add(new PredictionEntry(entry.Ticker, entry.Score, entry.Ema4));
            }

            // Store top10 in auto-tickers
            await Task.Run(() => _database.ReplaceAutoTickers(result.Top10, "AutoPredict"))
                .ConfigureAwait(true);

            // Store all top20 to direct_predictions table
            await Task.Run(() => _database.InsertDirectPredictions(
                result.Top20.Select(p => (p.Ticker, p.Score, p.Ema4)),
                SnapshotId,
                result.AsOf))
                .ConfigureAwait(true);

            // Store top10 as pending buys — scheduler will execute when broker connected + market open + no existing positions
            var pendingItems = result.Top20.Take(10).Select((p, i) => (p.Ticker, Rank: i + 1, p.Score)).ToList();
            await Task.Run(() => _database.ReplacePendingBuys(pendingItems)).ConfigureAwait(true);

            StatusMessage = $"Stored {result.Top10.Count} tickers as pending buys. Will execute when market opens.";
            _monitor.ReportPredictionProgress("Complete", 100,
                $"Stored {result.Top10.Count} pending buys — awaiting market open");
        }
        catch (Exception ex)
        {
            StatusMessage = $"Prediction failed: {ex.Message}";
            _monitor.ReportPredictionProgress("FAILED", -1, ex.Message);
        }
        finally
        {
            IsBusy = false;
            _monitor.IsPredictionRunning = false;
        }
    }

    private async Task ScanEarningsAsync()
    {
        try
        {
            IsEarningsBusy = true;
            EarningsStatus = "Scanning earnings news (T+0 to T-2)...";
            EarningsResults.Clear();

            var results = await _earningsBridge.ScanAsync(
                onProgress: progress =>
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        EarningsStatus = $"{progress.Step} ({progress.Progress}%) {progress.Detail}";
                    });
                }).ConfigureAwait(true);

            if (results.Count == 0)
            {
                EarningsStatus = "No earnings beat tickers found (T+0 to T-2).";
                return;
            }

            foreach (var r in results)
            {
                EarningsResults.Add(new EarningsEntry(r.Ticker, r.Title, r.Published, r.GapPct));
            }

            EarningsStatus = $"Found {results.Count} earnings beat tickers (info only, no auto-trade).";
        }
        catch (Exception ex)
        {
            EarningsStatus = $"Earnings scan failed: {ex.Message}";
        }
        finally
        {
            IsEarningsBusy = false;
        }
    }

    private async Task RetrainModelAsync()
    {
        var opts = _pythonOptions.CurrentValue;
        if (string.IsNullOrWhiteSpace(opts.RetrainScript) || !File.Exists(opts.RetrainScript))
        {
            RetrainStatus = "RetrainScript not configured or file not found in appsettings.json";
            return;
        }

        try
        {
            IsRetrainBusy = true;
            RetrainStatus = "Training model with full dataset... (this may take 30-60 minutes)";

            var scriptDir = Path.GetDirectoryName(opts.RetrainScript)!;
            var psi = new ProcessStartInfo
            {
                FileName = opts.Executable,
                Arguments = $"\"{opts.RetrainScript}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = scriptDir,
            };

            var exitCode = await Task.Run(async () =>
            {
                using var process = Process.Start(psi)
                    ?? throw new InvalidOperationException("Failed to start retrain process");

                // Drain stderr in background so buffer doesn't block
                var stderrTask = process.StandardError.ReadToEndAsync();
                var stdoutTask = process.StandardOutput.ReadToEndAsync();

                using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromHours(2));
                await process.WaitForExitAsync(cts.Token).ConfigureAwait(false);
                return process.ExitCode;
            }).ConfigureAwait(true);

            if (exitCode != 0)
            {
                RetrainStatus = $"Retrain failed (exit code {exitCode}). Check training logs.";
                return;
            }

            // Read new snapshot ID from latest_snapshot_id.txt
            var projectRoot = Path.GetDirectoryName(Path.GetDirectoryName(opts.RetrainScript))!;
            var snapshotFile = Path.Combine(projectRoot, "latest_snapshot_id.txt");
            if (!File.Exists(snapshotFile))
            {
                RetrainStatus = "Retrain completed but latest_snapshot_id.txt not found.";
                return;
            }

            var newSnapshotId = (await File.ReadAllTextAsync(snapshotFile).ConfigureAwait(true)).Trim();
            if (string.IsNullOrWhiteSpace(newSnapshotId))
            {
                RetrainStatus = "Retrain completed but snapshot ID is empty.";
                return;
            }

            // Persist new snapshot to appsettings.json
            _writableOptions.UpdateDefaultSnapshot(newSnapshotId);

            // Update in-memory snapshot ID
            SnapshotId = newSnapshotId;

            // Record retrain date
            await Task.Run(() => _database.SetEtfRotationState(
                "retrain_last_date", DateTime.UtcNow.ToString("o"))).ConfigureAwait(true);

            UpdateRetrainCountdown();
            RetrainStatus = $"Retrain complete! New snapshot: {newSnapshotId}";
        }
        catch (OperationCanceledException)
        {
            RetrainStatus = "Retrain timed out after 2 hours.";
        }
        catch (Exception ex)
        {
            RetrainStatus = $"Retrain failed: {ex.Message}";
        }
        finally
        {
            IsRetrainBusy = false;
        }
    }

    private void UpdateRetrainCountdown()
    {
        try
        {
            var lastRetrainStr = _database.GetEtfRotationState("retrain_last_date");
            if (string.IsNullOrWhiteSpace(lastRetrainStr) || !DateTime.TryParse(lastRetrainStr, out var lastRetrain))
            {
                RetrainCountdownText = "No retrain recorded";
                return;
            }

            var elapsed = TraderDatabase.CountTradingDaysBetween(lastRetrain, DateTime.UtcNow);
            var remaining = RetrainCycleDays - elapsed;

            RetrainCountdownText = remaining > 0
                ? $"Next retrain in: {remaining} trading days"
                : $"Retrain overdue by {-remaining} days";
        }
        catch
        {
            RetrainCountdownText = "No retrain recorded";
        }
    }
}
