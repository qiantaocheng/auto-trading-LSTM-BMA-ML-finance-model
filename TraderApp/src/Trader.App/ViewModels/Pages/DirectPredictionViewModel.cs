using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Trader.App.Commands;
using Trader.App.ViewModels.Pages.Models;
using Trader.Core.Options;
using Trader.Core.Repositories;
using Trader.PythonBridge.Services;
using Microsoft.Extensions.Options;
using Trader.App.Services;
using Trader.App.ViewModels;

namespace Trader.App.ViewModels.Pages;

public class DirectPredictionViewModel : ViewModelBase
{
    private readonly PythonPredictionBridge _bridge;
    private readonly PythonEarningsBridge _earningsBridge;
    private readonly TraderDatabase _database;
    private readonly MonitorViewModel _monitor;
    private readonly TradingSchedulerService _tradingScheduler;
    private readonly AsyncRelayCommand _runPredictionCommand;
    private readonly AsyncRelayCommand _scanEarningsCommand;
    private DateTimeOffset? _lastRun;
    private string? _excelPath;
    private bool _isBusy;
    private bool _isEarningsBusy;
    private string _snapshotId;
    private string? _statusMessage;
    private string? _earningsStatus;

    public DirectPredictionViewModel(
        PythonPredictionBridge bridge,
        PythonEarningsBridge earningsBridge,
        TraderDatabase database,
        MonitorViewModel monitor,
        TradingSchedulerService tradingScheduler,
        IOptions<PythonOptions> pythonOptions)
    {
        _bridge = bridge;
        _earningsBridge = earningsBridge;
        _database = database;
        _monitor = monitor;
        _tradingScheduler = tradingScheduler;
        _snapshotId = string.IsNullOrWhiteSpace(pythonOptions.Value.DefaultSnapshot)
            ? "snapshot"
            : pythonOptions.Value.DefaultSnapshot;

        TopTwenty = new ObservableCollection<PredictionEntry>();
        EarningsResults = new ObservableCollection<EarningsEntry>();
        _runPredictionCommand = new AsyncRelayCommand(RunPredictionAsync, () => !IsBusy);
        _scanEarningsCommand = new AsyncRelayCommand(ScanEarningsAsync, () => !IsEarningsBusy);
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

    public AsyncRelayCommand RunPredictionCommand => _runPredictionCommand;
    public AsyncRelayCommand ScanEarningsCommand => _scanEarningsCommand;

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

            StatusMessage = $"Stored {result.Top10.Count} tickers and updated Excel report.";
            _monitor.ReportPredictionProgress("Complete", 100,
                $"Stored {result.Top10.Count} tickers");

            // Trigger auto-buy if enabled
            if (_tradingScheduler.IsAutoTradingEnabled)
            {
                StatusMessage = "Starting auto-buy...";
                _ = Task.Run(() => _tradingScheduler.ExecuteAutoBuyAsync(result.Top10));
            }
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
            EarningsStatus = "Scanning earnings news...";
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
                EarningsStatus = "No earnings beat tickers found.";
                return;
            }

            foreach (var r in results)
            {
                EarningsResults.Add(new EarningsEntry(r.Ticker, r.Title, r.Published, r.GapPct));
            }

            EarningsStatus = $"Found {results.Count} earnings beat tickers. Buying...";

            // Auto-buy all beat tickers
            if (!_monitor.IsConnected)
            {
                EarningsStatus = $"Found {results.Count} tickers but IBKR not connected. Connect first.";
                return;
            }

            var tickers = results.Select(r => r.Ticker).ToList();
            await Task.Run(() => _tradingScheduler.ExecuteEarningsBuyAsync(tickers))
                .ConfigureAwait(true);

            _monitor.InvalidatePositionCache();
            EarningsStatus = $"Bought {results.Count} earnings beat tickers (T+10).";
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
}
