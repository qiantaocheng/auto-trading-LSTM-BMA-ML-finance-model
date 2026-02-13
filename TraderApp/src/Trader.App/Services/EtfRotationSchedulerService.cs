using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Trader.App.ViewModels.Pages;
using Trader.Core.Options;
using Trader.Core.Repositories;
using Trader.Core.Services;
using Trader.PythonBridge.Services;

namespace Trader.App.Services;

/// <summary>
/// ETF Rotation scheduler — P2 2-Level Cap strategy with Portfolio B.
/// 75% of net_liq. No stop loss (matches backtest exactly).
/// No HMM — uses its own MA200 + vol-target risk management.
///
/// Flow (matches backtest with 0.85 Sharpe):
///   1. On connect → check etf_positions → if empty → initial full buy
///   2. Every 21 TRADING DAYS → recalculate → apply deadband → rebalance
///   3. Trading day counter only increments on NYSE trading days (not weekends/holidays)
///   4. Deadband: skip if |Δexposure| &lt; 5%
///   5. Max step: clamp position change to ±15%
///   6. Min hold: skip reverse if &lt; 5 trading days since last rebalance &amp; |Δ| &lt; 30%
///
/// State persisted in DB (etf_rotation_state table) so it survives restarts.
/// </summary>
public sealed class EtfRotationSchedulerService : BackgroundService
{
    private readonly PythonEtfRotationBridge _rotationBridge;
    private readonly PythonTradingBridge _tradingBridge;
    private readonly PolygonPriceService _polygonPrices;
    private readonly TraderDatabase _database;
    private readonly MonitorViewModel _monitor;
    private readonly IOptionsMonitor<EtfRotationOptions> _etfOptions;
    private readonly IOptionsMonitor<TelegramOptions> _telegramOptions;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<EtfRotationSchedulerService> _logger;

    // ── Config-backed parameters (read from appsettings.json, hot-reloadable) ──
    private int RebalanceFreqDays => _etfOptions.CurrentValue.RebalanceDays;
    private double ExposureDeadband => _etfOptions.CurrentValue.Deadband;
    private double MaxStep => _etfOptions.CurrentValue.MaxStep;
    private int MinHoldDays => _etfOptions.CurrentValue.MinHoldDays;
    private double EtfCapitalRatio => _etfOptions.CurrentValue.CapitalRatio;
    private bool UseHmm => _etfOptions.CurrentValue.UseHmm;

    // ── DB state keys ────────────────────────────────────────────────────
    private const string StateKeyLastRebalDate = "last_rebalance_date";
    private const string StateKeyLastExposure = "last_exposure";
    private const string StateKeyLastRebalRunDate = "last_rebalance_run_date"; // idempotency key
    private const string StateKeyLastPortfolioUsed = "last_portfolio_used"; // regime switch detection

    // ── Runtime state ────────────────────────────────────────────────────
    private bool _initialCheckDone;
    private double _currentExposure;
    private DateTime? _lastRebalanceDate;
    private string? _lastRebalRunDate;  // "yyyy-MM-dd" — idempotency: prevent double-rebalance same day
    private string? _lastPortfolioUsed;  // "HMM risk-on (SMH)" / "HMM risk-off (GDX)" — regime switch detection

    // VIX Dynamic Trigger state
    private string _vixTriggerMode = "baseline";
    private bool _vixModeActive = false;
    private double? _vixPrice;
    private double? _themeBudget;

    // HMM System 3 state
    private double? _hmmPRiskSmooth;
    // HMM regime switch confirmation: require 2 consecutive days on same side to prevent oscillation
    private string? _hmmPendingPortfolio;  // Portfolio the signal wants to switch TO
    private int _hmmRegimeSwitchConfirmDays = 0;
    private const int HmmRegimeSwitchConfirmRequired = 2;

    // Daily Risk Control state (V7: V3 + Soft Exit + SOFT_PLUS)
    private string _riskLevel = "RISK_ON";          // RISK_ON, RISK_OFF, CRISIS, RECOVERY_RAMP
    private string _crisisSubState = "HARD";        // HARD, SOFT, SOFT_PLUS (within CRISIS)
    private double _portfolioPeakEquity = 0.0;
    private double _cumulativeDepositsWithdrawals = 0.0; // Net D/W adjustment for peak equity
    private double _lastKnownNetLiq = 0.0; // For D/W detection
    private double _lastKnownPortfolioValue = 0.0; // For D/W PnL estimation (ETF positions only)
    private DateTime _lastRiskCheckDate = DateTime.MinValue;
    private DateTime _lastEmergencyAdjustDate = DateTime.MinValue;
    private DateTime _lastCrisisDate = DateTime.MinValue;
    private int _riskOffConfirmDays = 0;
    private int _recoveryConfirmDays = 0;
    private int _spyBelowMa200Days = 0;             // Consecutive days SPY < MA200
    private int _softExitCounter = 0;                // Days VIX < soft exit threshold
    private int _softPlusCounter = 0;                // Days SOFT_PLUS upgrade conditions met
    private int _transitionsThisCycle = 0;           // Cycle limiter (reset on rebalance)
    private double _emergencyExposureCap = 1.0;      // Emergency imposed cap (1.0 = no cap)
    private readonly List<double> _spy10dPrices = new(); // Rolling 10d SPY prices for new-low detection

    // V7 Daily Risk Thresholds (V3 Tail-Only + Soft Exit + SOFT_PLUS)
    private const double VixCrisisThreshold = 40.0;       // VIX ≥40 AND SPY<MA200 → CRISIS
    private const double VixCrisisExtreme = 50.0;         // VIX ≥50 bypasses cooldown
    private const double DdCrisisThreshold = -0.10;       // Portfolio DD ≥10% → CRISIS
    private const double DdRiskOffThreshold = -0.06;      // DD ≥6% for SPY<MA200+DD trigger
    private const double DdRecoveryThreshold = -0.04;     // DD recovered to <4%
    private const int CrisisCooldownDays = 10;

    // CRISIS caps (sub-states: HARD/SOFT/SOFT_PLUS)
    private const double CrisisCap = 0.30;                // HARD: 30%
    private const double SoftCap = 0.55;                  // SOFT: 55%
    private const double SoftPlusCap = 0.80;              // SOFT_PLUS: 80%

    // RISK_OFF caps (dynamic by DD tier)
    private const double RiskOffCapLight = 0.80;          // DD [6-8%) → 80%
    private const double RiskOffCapModerate = 0.70;       // DD [8-10%) → 70%

    // RISK_OFF triggers (V3: market-confirmed, no VIX-only)
    private const int SpyBelowMa200ConfirmDays = 3;       // SPY<MA200 for 3d → RISK_OFF
    private const int RiskOffConfirmDaysRequired = 2;     // SPY<MA200+DD≥6% for 2d

    // Soft CRISIS exit (HARD→SOFT)
    private const double VixSoftExitThreshold = 32.0;     // VIX<32 for N days → relax to 55%
    private const int SoftExitConfirmDaysRequired = 3;

    // SOFT_PLUS ratchet (SOFT→SOFT_PLUS)
    private const double SoftPlusVixMax = 25.0;           // VIX<25 + SPY>MA200 → upgrade to 80%
    private const int SoftPlusConfirmDaysRequired = 3;

    // Revert conditions
    private const double VixRevertThreshold = 40.0;       // VIX≥40 → back to HARD
    private const int NewLowLookback = 10;                // 10d SPY new-low window

    // Recovery
    private const int RecoveryConfirmDaysRequired = 5;
    private const double VixRecoveryThreshold = 20.0;
    private const double RecoveryTurboStep = 0.10;        // +10%/day (turbo: VIX<25+SPY>MA200)
    private const double RecoverySlowStep = 0.05;         // +5%/day (slow)
    private const double RecoveryTurboVixMax = 25.0;

    // Limits
    private const double EmergencyMaxStep = 0.15;
    private const int MaxTransitionsPerCycle = 1;          // Max 1 transition per 21d cycle (CRISIS exempt)

    // Partial fill tracking: ticker → consecutive partial fill count
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _consecutivePartialFills = new(StringComparer.OrdinalIgnoreCase);
    private const int PartialFillLimitThreshold = 2; // Switch to limit order after 2 consecutive partials

    // Staleness tracking
    private const int StalenessWarningDays = 30;
    private const int StalenessAlertDays = 42;
    private DateTime _lastTelegramAlertDate = DateTime.MinValue;

    public EtfRotationSchedulerService(
        PythonEtfRotationBridge rotationBridge,
        PythonTradingBridge tradingBridge,
        PolygonPriceService polygonPrices,
        TraderDatabase database,
        MonitorViewModel monitor,
        IOptionsMonitor<EtfRotationOptions> etfOptions,
        IOptionsMonitor<TelegramOptions> telegramOptions,
        IHttpClientFactory httpClientFactory,
        ILogger<EtfRotationSchedulerService> logger)
    {
        _rotationBridge = rotationBridge;
        _tradingBridge = tradingBridge;
        _polygonPrices = polygonPrices;
        _database = database;
        _monitor = monitor;
        _etfOptions = etfOptions;
        _telegramOptions = telegramOptions;
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    // ── Expose state for GUI ─────────────────────────────────────────────
    public double CurrentExposure => _currentExposure;
    public double CurrentRiskCap { get; private set; } = 1.0;
    public int EtfPositionCount { get; private set; }
    public int TradingDaysSinceRebalance => _lastRebalanceDate.HasValue
        ? TraderDatabase.CountTradingDaysBetween(_lastRebalanceDate.Value, DateTime.UtcNow)
        : RebalanceFreqDays; // No prior rebalance → trigger immediately
    public int TradingDaysRemaining => Math.Max(0, RebalanceFreqDays - TradingDaysSinceRebalance);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation(
            "EtfRotationSchedulerService started (T10C-Slim{HmmMode}): rebalance every {Days} trading days, " +
            "deadband_up={DbUp:P0}, deadband_down={DbDown:P0}, maxStep={MaxStep:P0}, minHold={MinHold}d, capital={Capital:P0}",
            UseHmm ? " + HMM Sys3" : "",
            RebalanceFreqDays, _etfOptions.CurrentValue.DeadbandUp, _etfOptions.CurrentValue.DeadbandDown,
            MaxStep, MinHoldDays, EtfCapitalRatio);

        // Load persisted state from DB
        LoadStateFromDb();
        UpdateEtfPositionCount();
        ReportEtfStatus();

        await RunMainLoopAsync(stoppingToken).ConfigureAwait(false);

        _logger.LogInformation("EtfRotationSchedulerService stopped");
    }

    // ── Main Loop (60s check interval) ──────────────────────────────────

    private async Task RunMainLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(10_000, ct).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }

            if (!_monitor.IsConnected)
            {
                _initialCheckDone = false; // Reset so we re-check on reconnect
                continue;
            }

            try
            {
                // ── DAILY RISK CHECK (runs even when market is closed) ────────
                // Must run BEFORE the market-closed guard since it executes at 4-5 PM ET
                var nowEt = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                    TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
                if (nowEt.Hour >= 15 && nowEt.Hour < 17 && nowEt.DayOfWeek >= DayOfWeek.Monday && nowEt.DayOfWeek <= DayOfWeek.Friday)
                {
                    await RunDailyRiskCheckAsync(ct).ConfigureAwait(false);
                }

                // Check market status
                var status = await _tradingBridge.GetMarketStatusAsync(ct).ConfigureAwait(false);
                if (!status.IsOpen)
                {
                    // Market closed — still update countdown display (no counter change)
                    if (!_initialCheckDone)
                    {
                        _initialCheckDone = true; // Don't retry every 10s
                        ReportEvent("ETF Rotation", "Market closed — waiting for open");
                    }
                    _initialCheckDone = false; // Will re-check when market opens
                    ReportEtfStatus();
                    continue;
                }

                // ── Step 1: Calendar-anchored trading day count ──
                // No incremental counter — computed from last_rebalance_date via CountTradingDaysBetween.
                // This ensures offline/shutdown days are properly counted.
                var tradingDaysSince = TradingDaysSinceRebalance;

                // ── Step 2: Initial check — buy if no positions ──
                if (!_initialCheckDone)
                {
                    _initialCheckDone = true;
                    var positions = _database.GetPositions("ETF");
                    if (positions.Count == 0)
                    {
                        ReportEvent("ETF Rotation", "No ETF positions found — executing initial buy...");
                        await RecalculateAndExecuteAsync(isInitial: true, ct).ConfigureAwait(false);
                        continue;
                    }
                    else
                    {
                        ReportEvent("ETF Rotation",
                            $"Found {positions.Count} ETF positions — {TradingDaysRemaining} trading days until rebalance");
                        ReportEtfStatus();
                        continue;
                    }
                }

                // ── Step 3: Rebalance if 21 trading days reached ──
                if (tradingDaysSince >= RebalanceFreqDays)
                {
                    _logger.LogInformation(
                        "ETF rebalance due: {Days} trading days since last rebalance ({Date})",
                        tradingDaysSince, _lastRebalanceDate?.ToString("yyyy-MM-dd") ?? "never");
                    ReportEvent("ETF Rotation",
                        $"{tradingDaysSince} trading days reached — triggering rebalance");
                    await RecalculateAndExecuteAsync(isInitial: false, ct).ConfigureAwait(false);
                }

                ReportEtfStatus();
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.LogError(ex, "ETF rotation main loop error");
                ReportEvent("ETF Rotation", $"Error: {ex.Message}");
            }
        }
    }

    // ── Core: Recalculate target weights → deadband → update DB → buy/sell ──

    private async Task RecalculateAndExecuteAsync(bool isInitial, CancellationToken ct)
    {
        if (!_monitor.IsConnected)
        {
            ReportEvent("ETF Rebalance", "BLOCKED: Broker disconnected");
            return;
        }

        // ── Idempotency: prevent double-rebalance on same trading day ──
        var todayTradingDate = DateTime.UtcNow.Date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        if (!isInitial && _lastRebalRunDate == todayTradingDate)
        {
            _logger.LogInformation(
                "ETF rebalance already ran today ({Date}) — skipping to prevent double-execution",
                todayTradingDate);
            return;
        }

        ReportEvent("ETF Rebalance", "Fetching Polygon data & recalculating targets...");

        try
        {
            // 1. Get target weights from Python signal
            var signal = await _rotationBridge.GetTargetWeightsAsync(
                riskLevel: UseHmm ? null : _riskLevel,
                useHmm: UseHmm,
                onProgress: p => ReportEvent("ETF Signal", p.Detail), ct).ConfigureAwait(false);

            if (signal.Error is not null)
            {
                ReportEvent("ETF Rebalance", $"Signal error: {signal.Error}");
                return;
            }

            var rawTargetExposure = signal.Exposure;
            CurrentRiskCap = signal.RiskCap;

            // Store VIX trigger state
            _vixTriggerMode = signal.VixTriggerMode ?? "baseline";
            _vixModeActive = signal.VixModeActive;
            _vixPrice = signal.VixPrice;
            _themeBudget = signal.ThemeBudget;
            _hmmPRiskSmooth = signal.HmmPRiskSmooth;

            // Log asof date and data quality
            _logger.LogInformation(
                "ETF signal: asof={AsofDate} exposure={Exposure:P1} risk_cap={RiskCap:P0} " +
                "vol={Vol:P1} spy={Spy:F2} ma200={MA200:F2} spy_dev={Dev:P1} " +
                "vix_mode={VixMode} vix={Vix} vix_cap={VixCap} theme_budget={ThemeBudget} " +
                "portfolio={Portfolio} risk_level={RiskLevel}",
                signal.AsofTradingDay ?? "?", signal.Exposure, signal.RiskCap,
                signal.BlendedVol, signal.SpyPrice, signal.Ma200, signal.SpyDeviation,
                _vixTriggerMode, _vixPrice, signal.VixCapApplied, _themeBudget,
                signal.PortfolioUsed ?? "?", signal.RiskLevel ?? _riskLevel);

            if (signal.HasDataAnomaly)
            {
                var anomalies = signal.AnomalyTickers is not null
                    ? string.Join(", ", signal.AnomalyTickers)
                    : "unknown";
                var blocked = signal.BlockedAnomalyTickers is not null
                    ? string.Join(", ", signal.BlockedAnomalyTickers)
                    : "";
                _logger.LogWarning(
                    "ETF data anomaly: flagged={Tickers} blocked={Blocked}",
                    anomalies, blocked);
                ReportEvent("ETF Rebalance",
                    $"WARNING: anomaly tickers BLOCKED from trading: {(string.IsNullOrEmpty(blocked) ? anomalies : blocked)}");
            }

            if (signal.BarsLastDate is not null)
            {
                foreach (var (ticker, lastDate) in signal.BarsLastDate)
                    _logger.LogDebug("ETF bars last date: {Ticker} = {Date}", ticker, lastDate ?? "N/A");
            }

            ReportEvent("ETF Rebalance",
                $"Exposure={signal.Exposure:P0} RiskCap={signal.RiskCap:P0} " +
                $"Vol={signal.BlendedVol:P1} SPY={signal.SpyPrice:C} MA200={signal.Ma200:C} " +
                $"asof={signal.AsofTradingDay ?? "?"}");

            if (signal.EtfWeights is null || signal.EtfWeights.Count == 0)
            {
                ReportEvent("ETF Rebalance", "No ETF weights returned");
                return;
            }

            // 2. Apply emergency cap + deadband + max step + min hold (matches backtest exactly)
            // V7 emergency cap — skipped in HMM mode (HMM handles risk via p_risk cap in Python)
            var targetExposure = rawTargetExposure;
            if (!UseHmm && _emergencyExposureCap < 1.0)
            {
                targetExposure = Math.Min(rawTargetExposure, _emergencyExposureCap);
                _logger.LogWarning("Rebalance: emergency cap enforced {RiskLevel}/{Sub} — raw {Raw:P1} → capped {Capped:P1}",
                    _riskLevel, _crisisSubState, rawTargetExposure, targetExposure);
            }
            var deadbandApplied = false;
            var stepClamped = false;
            var minHoldBlocked = false;

            // Detect regime switch (portfolio composition change: risk-on ↔ risk-off)
            var currentPortfolio = signal.PortfolioUsed;
            var regimeSwitched = !string.IsNullOrEmpty(_lastPortfolioUsed)
                && !string.IsNullOrEmpty(currentPortfolio)
                && !string.Equals(_lastPortfolioUsed, currentPortfolio, StringComparison.OrdinalIgnoreCase);

            if (regimeSwitched)
            {
                _logger.LogInformation(
                    "ETF regime switch detected: {Old} → {New} — bypassing deadband",
                    _lastPortfolioUsed, currentPortfolio);
            }

            if (!isInitial)
            {
                var delta = targetExposure - _currentExposure;

                // Check min-hold first (matches backtest order)
                // Skip min-hold on regime switch — portfolio composition change takes priority
                var daysSinceRebal = TradingDaysSinceRebalance;
                if (!regimeSwitched && daysSinceRebal < MinHoldDays && Math.Abs(delta) < 0.30)
                    minHoldBlocked = true;

                if (!minHoldBlocked)
                {
                    // T10C-Slim L6: Asymmetric deadband (0.02 up / 0.05 down)
                    // Skip deadband on regime switch — composition must change even if exposure delta is small
                    if (!regimeSwitched)
                    {
                        var dbUp = _etfOptions.CurrentValue.DeadbandUp;
                        var dbDown = _etfOptions.CurrentValue.DeadbandDown;
                        if (delta > 0 && delta < dbUp)
                            deadbandApplied = true;
                        else if (delta < 0 && Math.Abs(delta) < dbDown)
                            deadbandApplied = true;
                    }
                    // Max step still applies (gradual position sizing even on regime switch)
                    if (!deadbandApplied)
                    {
                        if (delta > MaxStep) { targetExposure = _currentExposure + MaxStep; stepClamped = true; }
                        else if (delta < -MaxStep) { targetExposure = _currentExposure - MaxStep; stepClamped = true; }
                    }
                }

                // ── Decision audit log ────────────────────────────────────
                var decisionId = $"{signal.AsofTradingDay ?? todayTradingDate}-{Guid.NewGuid().ToString()[..8]}";
                _logger.LogInformation(
                    "ETF decision [{DecisionId}]: asof={AsofDate} raw_target={RawTarget:P2} " +
                    "current={Current:P2} delta={Delta:P2} min_hold_blocked={MinHold} " +
                    "deadband_applied={Deadband} step_clamped={StepClamped} regime_switch={RegimeSwitch} final={Final:P2}",
                    decisionId, signal.AsofTradingDay ?? "?",
                    rawTargetExposure, _currentExposure, delta,
                    minHoldBlocked, deadbandApplied, stepClamped, regimeSwitched, targetExposure);

                if (minHoldBlocked || deadbandApplied)
                {
                    var reason = minHoldBlocked
                        ? $"Min-hold: only {daysSinceRebal} < {MinHoldDays} trading days since rebalance"
                        : delta > 0
                            ? $"Deadband: Δ = +{delta:P1} < up={_etfOptions.CurrentValue.DeadbandUp:P0}"
                            : $"Deadband: |Δ| = {Math.Abs(delta):P1} < down={_etfOptions.CurrentValue.DeadbandDown:P0}";
                    ReportEvent("ETF Rebalance", $"No change — {reason}");
                    // Anchor next 21-day window from today
                    _lastRebalanceDate = DateTime.UtcNow;
                    _lastRebalRunDate = todayTradingDate;
                    SaveStateToDb();
                    ReportEtfStatus();
                    return;
                }
            }

            // Recalculate weights with the (possibly clamped) exposure
            var etfWeights = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            double totalStrategicWeight = signal.EtfWeights.Values.Sum();
            double rawExposure = signal.Exposure;
            foreach (var (ticker, rawWeight) in signal.EtfWeights)
            {
                // Scale weights proportionally to new exposure
                double adjustedWeight = rawExposure > 0
                    ? rawWeight * (targetExposure / rawExposure)
                    : 0;
                etfWeights[ticker] = adjustedWeight;
            }

            // 3. Get available capital
            var cashInfo = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
            if (cashInfo.Error is not null)
            {
                ReportEvent("ETF Rebalance", $"Failed to get cash: {cashInfo.Error}");
                return;
            }

            // Margin safety check
            if (cashInfo.MarginUsedPct > 0.85)
            {
                _logger.LogWarning("MARGIN ALERT: utilization {Pct:P0} > 85% — ETF rebalance blocked (sells only)",
                    cashInfo.MarginUsedPct);
                ReportEvent("ETF Rebalance", $"WARNING: High margin {cashInfo.MarginUsedPct:P0} — only sells allowed");
            }

            var netLiq = (decimal)cashInfo.NetLiq;
            var budget = netLiq * (decimal)EtfCapitalRatio;
            // Cap budget by buying power if available
            if (cashInfo.BuyingPower > 0)
            {
                var bpBudget = (decimal)(cashInfo.BuyingPower * 0.92);
                if (bpBudget < budget)
                {
                    _logger.LogInformation("ETF budget capped by buying power: {BP:C} < {Budget:C}", bpBudget, budget);
                    budget = bpBudget;
                }
            }
            ReportEvent("ETF Rebalance", $"Net Liq: {netLiq:C}, ETF budget (75%): {budget:C}");

            // 4. Fetch fresh Polygon prices for all ETFs
            var etfTickers = etfWeights.Keys.ToList();
            var freshPrices = await _polygonPrices.GetPricesAsync(etfTickers, ct).ConfigureAwait(false);

            var prices = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);
            foreach (var ticker in etfTickers)
            {
                if (freshPrices.TryGetValue(ticker, out var fp) && fp > 0)
                    prices[ticker] = fp;
                else if (signal.EtfPrices is not null && signal.EtfPrices.TryGetValue(ticker, out var sp) && sp > 0)
                    prices[ticker] = (decimal)sp;
            }

            // 5. Compute target shares
            var targetShares = new Dictionary<string, (int Shares, decimal Price, double Weight)>(StringComparer.OrdinalIgnoreCase);
            foreach (var (ticker, weight) in etfWeights)
            {
                if (!prices.TryGetValue(ticker, out var price) || price <= 0)
                {
                    ReportEvent("ETF Rebalance", $"Skipping {ticker}: no price");
                    continue;
                }
                var tickerBudget = budget * (decimal)weight;
                var shares = (int)Math.Floor(tickerBudget / price);
                targetShares[ticker] = (shares, price, weight);
            }

            // 6. Get current positions from DB
            var currentPositions = _database.GetPositions("ETF")
                .ToDictionary(p => p.Symbol, StringComparer.OrdinalIgnoreCase);

            // 7. Determine changes
            var sellOrders = new List<(string Symbol, int Qty, string Note)>();
            var buyOrders = new List<(string Symbol, int Qty, double Weight)>();
            var partialSellOrders = new List<(string Symbol, int SellQty, int KeepQty, decimal EntryPrice, double Weight)>();

            foreach (var (symbol, pos) in currentPositions)
            {
                if (!targetShares.TryGetValue(symbol, out var target) || target.Shares == 0)
                {
                    sellOrders.Add((symbol, pos.Shares, "Rebalance-Exit"));
                }
                else if (target.Shares < pos.Shares)
                {
                    var sellQty = pos.Shares - target.Shares;
                    partialSellOrders.Add((symbol, sellQty, target.Shares, pos.EntryPrice, target.Weight));
                }
            }

            foreach (var (ticker, target) in targetShares)
            {
                if (target.Shares <= 0) continue;
                var currentShares = currentPositions.TryGetValue(ticker, out var existing) ? existing.Shares : 0;
                if (target.Shares > currentShares)
                {
                    buyOrders.Add((ticker, target.Shares - currentShares, target.Weight));
                }
            }

            var totalChanges = sellOrders.Count + partialSellOrders.Count + buyOrders.Count;
            if (totalChanges == 0)
            {
                foreach (var (ticker, target) in targetShares)
                {
                    if (currentPositions.TryGetValue(ticker, out var pos))
                        _database.InsertOrUpdatePosition(ticker, "ETF", pos.Shares, pos.EntryPrice, target.Weight);
                }
                _currentExposure = targetExposure;
                _lastPortfolioUsed = signal.PortfolioUsed;
                _lastRebalanceDate = DateTime.UtcNow;
                _lastRebalRunDate = todayTradingDate;
                SaveStateToDb();
                ReportEvent("ETF Rebalance", "No changes needed — targets match current positions");
                ReportEtfStatus();
                return;
            }

            ReportEvent("ETF Rebalance",
                $"Changes: {sellOrders.Count} exits, {partialSellOrders.Count} reductions, {buyOrders.Count} buys");

            // 8. Execute SELLS first (DB updated by SellEtfPositionAsync/BuyEtfPositionAsync after successful fills)
            var tradesExecuted = 0;
            foreach (var (symbol, qty, note) in sellOrders)
            {
                if (await SellEtfPositionAsync(symbol, qty, note, ct).ConfigureAwait(false))
                    tradesExecuted++;
            }
            foreach (var (symbol, sellQty, keepQty, entryPrice, weight) in partialSellOrders)
            {
                if (await SellEtfPartialAsync(symbol, sellQty, keepQty, entryPrice, weight, "Rebalance-Reduce", ct).ConfigureAwait(false))
                    tradesExecuted++;
            }

            // 10. Execute BUYS (blocked if margin too high)
            if (cashInfo.MarginUsedPct > 0.85)
            {
                ReportEvent("ETF Rebalance", $"BLOCKED buys: margin {cashInfo.MarginUsedPct:P0} > 85%");
            }
            else
            {
                foreach (var (ticker, qty, weight) in buyOrders)
                {
                    if (await BuyEtfPositionAsync(ticker, qty, weight, ct).ConfigureAwait(false))
                        tradesExecuted++;
                }
            }

            // 11. Update state — only reset rebalance date if at least one trade executed
            // Reset V7 cycle transition counter on rebalance
            _transitionsThisCycle = 0;

            _currentExposure = targetExposure;
            _lastPortfolioUsed = signal.PortfolioUsed;
            if (tradesExecuted > 0)
            {
                _lastRebalanceDate = DateTime.UtcNow;
                _lastRebalRunDate = todayTradingDate;
            }
            else
            {
                _logger.LogWarning("ETF rebalance: no trades executed — will retry next trading day");
            }
            SaveStateToDb();
            UpdateEtfPositionCount();
            ReportEtfStatus();

            ReportEvent("ETF Rebalance",
                $"Complete: {_database.GetPositions("ETF").Count} ETF positions, " +
                $"exposure={targetExposure:P0}, next rebalance in {RebalanceFreqDays} trading days");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ETF rebalance failed");
            ReportEvent("ETF Rebalance", $"FAILED: {ex.Message}");
        }
    }

    // ── Trade Execution Helpers ──────────────────────────────────────────

    private async Task<bool> BuyEtfPositionAsync(string symbol, int quantity, double targetWeight, CancellationToken ct)
    {
        try
        {
            var intentId = TraderDatabase.GenerateIntentId("ETF-Rotation", symbol, "BUY");
            if (_database.IsIntentExecuted(intentId))
            {
                ReportEvent("ETF Buy", $"Skipping {symbol} — already executed today");
                return true;
            }
            _database.TryInsertTradeIntent(intentId, "ETF-Rotation", symbol, "BUY", quantity);

            // Check if we should use limit order (consecutive partial fills)
            var partialCount = _consecutivePartialFills.GetValueOrDefault(symbol, 0);
            BuyResult result;
            if (partialCount >= PartialFillLimitThreshold)
            {
                // Get mid price for limit order
                var midPrice = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                if (midPrice is not null && midPrice > 0)
                {
                    ReportEvent("ETF Buy", $"Buying {quantity} x {symbol} (LIMIT @ {midPrice:C}, 30s window, after {partialCount} partial fills)...");
                    result = await _tradingBridge.BuyLimitAsync(symbol, quantity, (double)midPrice.Value, 30, ct).ConfigureAwait(false);
                }
                else
                {
                    ReportEvent("ETF Buy", $"Buying {quantity} x {symbol} (market — limit price unavailable)...");
                    result = await _tradingBridge.BuyAsync(symbol, quantity, ct).ConfigureAwait(false);
                }
            }
            else
            {
                ReportEvent("ETF Buy", $"Buying {quantity} x {symbol}...");
                result = await _tradingBridge.BuyAsync(symbol, quantity, ct).ConfigureAwait(false);
            }

            if (!result.Success)
            {
                _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                ReportEvent("ETF Buy", $"FAILED {symbol}: {result.Error}");
                return false;
            }

            // Guard: treat FilledQty=0 as failure even if Success=true (prevents phantom positions)
            var filled = result.FilledQty > 0 ? result.FilledQty : 0;
            if (filled == 0)
            {
                _logger.LogWarning("ETF buy {Symbol}: Success=true but FilledQty=0 (status={Status}) — treating as failure", symbol, result.Status);
                _database.MarkIntentFailed(intentId, $"No fills: status={result.Status}");
                return false;
            }

            // Track partial fills — partial fill is final, DB records actual filled shares
            if (result.RemainingQty > 0)
            {
                _consecutivePartialFills.AddOrUpdate(symbol, 1, (_, c) => c + 1);
                _logger.LogWarning("ETF partial fill {Symbol}: {Filled}/{Total} (consecutive={Count})",
                    symbol, result.FilledQty, result.TotalQty, _consecutivePartialFills.GetValueOrDefault(symbol, 0));
            }
            else
            {
                _consecutivePartialFills.TryRemove(symbol, out _); // Reset on full fill
            }

            var fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : 0m;
            if (fillPrice <= 0)
            {
                var pp = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                fillPrice = pp ?? 0;
            }

            var existing = _database.GetPositions("ETF").FirstOrDefault(p =>
                p.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase));
            var totalShares = (existing?.Shares ?? 0) + filled;
            // Weighted average entry_price to preserve cost basis
            var avgEntryPrice = (existing is not null && existing.Shares > 0 && fillPrice > 0)
                ? (existing.EntryPrice * existing.Shares + fillPrice * filled) / totalShares
                : fillPrice;
            // DB writes after confirmed IBKR fill — CRITICAL if these fail
            try
            {
                _database.InsertOrUpdatePosition(symbol, "ETF", totalShares, avgEntryPrice, targetWeight);
                _database.InsertTrade(symbol, "BUY", filled, fillPrice, "ETF-Rotation");
                _database.MarkIntentExecuted(intentId, result.OrderId);
            }
            catch (Exception dbEx)
            {
                _logger.LogCritical(dbEx,
                    "CRITICAL: Buy FILLED at IBKR ({Symbol} {Filled}@{Price}) but DB write FAILED — manual reconciliation required",
                    symbol, filled, fillPrice);
                ReportEvent("ETF Buy", $"CRITICAL: {symbol} bought {filled}@{fillPrice:C} but DB FAILED — check reconciliation!");
                return true; // Buy DID succeed at IBKR — don't retry
            }

            ReportEvent("ETF Buy", $"Bought {symbol}: {filled} shares @ {fillPrice:C}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ETF buy failed for {Symbol}", symbol);
            ReportEvent("ETF Buy", $"Error buying {symbol}: {ex.Message}");
            return false;
        }
    }

    private async Task<bool> SellEtfPositionAsync(string symbol, int quantity, string note, CancellationToken ct)
    {
        try
        {
            var intentId = TraderDatabase.GenerateIntentId("ETF-Rotation", symbol, "SELL");
            if (_database.IsIntentExecuted(intentId))
            {
                ReportEvent("ETF Sell", $"Skipping {symbol} — already sold today");
                return false;
            }
            _database.TryInsertTradeIntent(intentId, "ETF-Rotation", symbol, "SELL", quantity);

            // Check if we should use limit order (consecutive partial fills on this symbol)
            var partialCount = _consecutivePartialFills.GetValueOrDefault(symbol, 0);
            SellResult result;
            if (partialCount >= PartialFillLimitThreshold)
            {
                var midPrice = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                if (midPrice is not null && midPrice > 0)
                {
                    ReportEvent("ETF Sell", $"Selling {quantity} x {symbol} (LIMIT @ {midPrice:C}, 30s, after {partialCount} partials)...");
                    result = await _tradingBridge.SellLimitAsync(symbol, quantity, (double)midPrice.Value, 30, ct).ConfigureAwait(false);
                }
                else
                {
                    ReportEvent("ETF Sell", $"Selling {quantity} x {symbol}...");
                    result = await _tradingBridge.SellAsync(symbol, quantity, ct).ConfigureAwait(false);
                }
            }
            else
            {
                ReportEvent("ETF Sell", $"Selling {quantity} x {symbol}...");
                result = await _tradingBridge.SellAsync(symbol, quantity, ct).ConfigureAwait(false);
            }

            if (result.Success)
            {
                // Guard: treat FilledQty=0 as failure (prevents phantom position deletion)
                var filled = result.FilledQty > 0 ? result.FilledQty : 0;
                if (filled == 0)
                {
                    _logger.LogWarning("ETF sell {Symbol}: Success=true but FilledQty=0 (status={Status}) — treating as failure", symbol, result.Status);
                    _database.MarkIntentFailed(intentId, $"No fills: status={result.Status}");
                    ReportEvent("ETF Sell", $"FAILED {symbol}: Success but 0 filled");
                    return false;
                }

                var fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : 0m;
                if (fillPrice <= 0)
                {
                    var pp = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                    fillPrice = pp ?? 0;
                }

                // DB writes after confirmed IBKR fill — CRITICAL if these fail
                try
                {
                    _database.MarkIntentExecuted(intentId, result.OrderId);
                    _database.InsertTrade(symbol, "SELL", filled, fillPrice, $"ETF-{note}");
                    // Partial fill = final. DB records actual filled. Next rebalance will diff and catch up.
                    if (result.RemainingQty > 0)
                    {
                        _consecutivePartialFills.AddOrUpdate(symbol, 1, (_, c) => c + 1);
                        var actualRemaining = quantity - filled;
                        _logger.LogWarning("ETF partial sell {Symbol}: {Filled}/{Total}, {Remaining} shares remain in position (consecutive={Count})",
                            symbol, filled, quantity, actualRemaining, _consecutivePartialFills.GetValueOrDefault(symbol, 0));
                        var existing = _database.GetPositions("ETF").FirstOrDefault(p =>
                            p.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase));
                        if (existing is not null)
                            _database.InsertOrUpdatePosition(symbol, "ETF", existing.Shares - filled, existing.EntryPrice, 0);
                    }
                    else
                    {
                        _consecutivePartialFills.TryRemove(symbol, out _);
                        _database.DeletePosition(symbol);
                    }
                }
                catch (Exception dbEx)
                {
                    _logger.LogCritical(dbEx,
                        "CRITICAL: Sell FILLED at IBKR ({Symbol} {Filled}@{Price}) but DB write FAILED — manual reconciliation required",
                        symbol, filled, fillPrice);
                    ReportEvent("ETF Sell", $"CRITICAL: {symbol} sold {filled}@{fillPrice:C} but DB FAILED — check reconciliation!");
                    return true; // Sell DID succeed at IBKR — don't retry
                }

                ReportEvent("ETF Sell", $"Sold {symbol}: {filled} shares @ {fillPrice:C}");
                return true;
            }
            else
            {
                _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                ReportEvent("ETF Sell", $"FAILED {symbol}: {result.Error}");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ETF sell failed for {Symbol}", symbol);
            ReportEvent("ETF Sell", $"Error selling {symbol}: {ex.Message}");
            return false;
        }
    }

    private async Task<bool> SellEtfPartialAsync(string symbol, int sellQty, int remainingShares, decimal entryPrice, double targetWeight, string note, CancellationToken ct)
    {
        try
        {
            var intentId = TraderDatabase.GenerateIntentId("ETF-Rotation", symbol, "SELL");
            if (_database.IsIntentExecuted(intentId))
            {
                ReportEvent("ETF Sell", $"Skipping reduce {symbol} — already sold today");
                return false;
            }
            _database.TryInsertTradeIntent(intentId, "ETF-Rotation", symbol, "SELL", sellQty);

            // Check if we should use limit order (consecutive partial fills)
            var partialCount = _consecutivePartialFills.GetValueOrDefault(symbol, 0);
            SellResult result;
            if (partialCount >= PartialFillLimitThreshold)
            {
                var midPrice = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                if (midPrice is not null && midPrice > 0)
                {
                    ReportEvent("ETF Sell", $"Reducing {symbol}: selling {sellQty} (LIMIT @ {midPrice:C}, 30s, after {partialCount} partials)...");
                    result = await _tradingBridge.SellLimitAsync(symbol, sellQty, (double)midPrice.Value, 30, ct).ConfigureAwait(false);
                }
                else
                {
                    ReportEvent("ETF Sell", $"Reducing {symbol}: selling {sellQty}, keeping {remainingShares}...");
                    result = await _tradingBridge.SellAsync(symbol, sellQty, ct).ConfigureAwait(false);
                }
            }
            else
            {
                ReportEvent("ETF Sell", $"Reducing {symbol}: selling {sellQty}, keeping {remainingShares}...");
                result = await _tradingBridge.SellAsync(symbol, sellQty, ct).ConfigureAwait(false);
            }

            if (result.Success)
            {
                // Guard: treat FilledQty=0 as failure (prevents phantom position update)
                var filled = result.FilledQty > 0 ? result.FilledQty : 0;
                if (filled == 0)
                {
                    _logger.LogWarning("ETF partial sell {Symbol}: Success=true but FilledQty=0 (status={Status}) — treating as failure", symbol, result.Status);
                    _database.MarkIntentFailed(intentId, $"No fills: status={result.Status}");
                    ReportEvent("ETF Sell", $"FAILED reduce {symbol}: Success but 0 filled");
                    return false;
                }

                var fillPrice = result.AvgPrice > 0 ? (decimal)result.AvgPrice : 0m;
                if (fillPrice <= 0)
                {
                    var pp = await _polygonPrices.GetPriceAsync(symbol, ct).ConfigureAwait(false);
                    fillPrice = pp ?? 0;
                }

                // DB writes after confirmed IBKR fill — CRITICAL if these fail
                try
                {
                    _database.MarkIntentExecuted(intentId, result.OrderId);
                    _database.InsertTrade(symbol, "SELL", filled, fillPrice, $"ETF-{note}");
                    // Adjust remaining shares for partial fills
                    var actualRemaining = remainingShares + (sellQty - filled);
                    _database.InsertOrUpdatePosition(symbol, "ETF", actualRemaining, entryPrice, targetWeight);
                    if (result.RemainingQty > 0)
                        _logger.LogWarning("ETF partial reduce {Symbol}: sold {Filled}/{Requested}", symbol, filled, sellQty);
                    ReportEvent("ETF Sell", $"Reduced {symbol}: sold {filled} @ {fillPrice:C}, keeping {actualRemaining}");
                }
                catch (Exception dbEx)
                {
                    _logger.LogCritical(dbEx,
                        "CRITICAL: Partial sell FILLED at IBKR ({Symbol} {Filled}@{Price}) but DB write FAILED — manual reconciliation required",
                        symbol, filled, fillPrice);
                    ReportEvent("ETF Sell", $"CRITICAL: {symbol} sold {filled}@{fillPrice:C} but DB FAILED — check reconciliation!");
                    return true; // Sell DID succeed at IBKR — don't retry
                }

                // Track partial fills for sell (outside critical section — non-essential)
                if (result.RemainingQty > 0)
                    _consecutivePartialFills.AddOrUpdate(symbol, 1, (_, c) => c + 1);
                else
                    _consecutivePartialFills.TryRemove(symbol, out _);

                return true;
            }
            else
            {
                _database.MarkIntentFailed(intentId, result.Error ?? "unknown");
                ReportEvent("ETF Sell", $"FAILED reduce {symbol}: {result.Error}");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ETF partial sell failed for {Symbol}", symbol);
            ReportEvent("ETF Sell", $"Error reducing {symbol}: {ex.Message}");
            return false;
        }
    }

    // ── State Persistence ───────────────────────────────────────────────

    private void LoadStateFromDb()
    {
        try
        {
            var rebalDateStr = _database.GetEtfRotationState(StateKeyLastRebalDate);
            if (rebalDateStr is not null && DateTime.TryParse(rebalDateStr, out var dt))
                _lastRebalanceDate = dt;

            var expStr = _database.GetEtfRotationState(StateKeyLastExposure);
            if (expStr is not null && double.TryParse(expStr, CultureInfo.InvariantCulture, out var exp))
                _currentExposure = exp;

            _lastRebalRunDate = _database.GetEtfRotationState(StateKeyLastRebalRunDate);

            _lastPortfolioUsed = _database.GetEtfRotationState(StateKeyLastPortfolioUsed);

            // Load daily risk control state
            var riskLevelStr = _database.GetEtfRotationState("risk_level");
            if (!string.IsNullOrEmpty(riskLevelStr))
                _riskLevel = riskLevelStr;

            var crisisSubStr = _database.GetEtfRotationState("crisis_sub_state");
            if (!string.IsNullOrEmpty(crisisSubStr))
                _crisisSubState = crisisSubStr;

            var capStr = _database.GetEtfRotationState("emergency_exposure_cap");
            if (capStr is not null && double.TryParse(capStr, CultureInfo.InvariantCulture, out var cap))
                _emergencyExposureCap = cap;

            var peakStr = _database.GetEtfRotationState("portfolio_peak_equity");
            if (peakStr is not null && double.TryParse(peakStr, CultureInfo.InvariantCulture, out var peak))
                _portfolioPeakEquity = peak;

            var dwStr = _database.GetEtfRotationState("cumulative_deposits_withdrawals");
            if (dwStr is not null && double.TryParse(dwStr, CultureInfo.InvariantCulture, out var dw))
                _cumulativeDepositsWithdrawals = dw;

            var lastNetLiqStr = _database.GetEtfRotationState("last_known_net_liq");
            if (lastNetLiqStr is not null && double.TryParse(lastNetLiqStr, CultureInfo.InvariantCulture, out var lastNl))
                _lastKnownNetLiq = lastNl;

            var lastPvStr = _database.GetEtfRotationState("last_known_portfolio_value");
            if (lastPvStr is not null && double.TryParse(lastPvStr, CultureInfo.InvariantCulture, out var lastPv))
                _lastKnownPortfolioValue = lastPv;

            var crisisDateStr = _database.GetEtfRotationState("last_crisis_date");
            if (crisisDateStr is not null && DateTime.TryParse(crisisDateStr, out var crisisDt))
                _lastCrisisDate = crisisDt;

            var daysSince = TradingDaysSinceRebalance;
            _logger.LogInformation(
                "ETF rotation state loaded: lastRebal={Date}, {Days} trading days since, exposure={Exposure:P0}, risk={Risk}, peak={Peak:C}, portfolio={Portfolio}",
                _lastRebalanceDate?.ToString("yyyy-MM-dd") ?? "never", daysSince, _currentExposure,
                _riskLevel, _portfolioPeakEquity, _lastPortfolioUsed ?? "unknown");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load ETF rotation state — starting fresh");
        }
    }

    private void SaveStateToDb()
    {
        try
        {
            var pairs = new List<(string Key, string Value)>();
            if (_lastRebalanceDate.HasValue)
                pairs.Add((StateKeyLastRebalDate, _lastRebalanceDate.Value.ToString("o")));
            pairs.Add((StateKeyLastExposure, _currentExposure.ToString("F6", CultureInfo.InvariantCulture)));
            if (_lastRebalRunDate is not null)
                pairs.Add((StateKeyLastRebalRunDate, _lastRebalRunDate));
            if (_lastPortfolioUsed is not null)
                pairs.Add((StateKeyLastPortfolioUsed, _lastPortfolioUsed));
            _database.SetEtfRotationStateBatch(pairs);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save ETF rotation state");
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private void ReportEvent(string action, string detail)
    {
        _logger.LogInformation("[{Action}] {Detail}", action, detail);
        try
        {
            Application.Current.Dispatcher.Invoke(() =>
                _monitor.ReportTradingEvent(action, detail));
        }
        catch { }
    }

    private void UpdateEtfPositionCount()
    {
        try
        {
            EtfPositionCount = _database.GetPositions("ETF").Count;
        }
        catch { }
    }

    private void ReportEtfStatus()
    {
        try
        {
            // Compute staleness level
            var daysSince = TradingDaysSinceRebalance;
            string stalenessLevel;
            if (daysSince >= StalenessAlertDays)
                stalenessLevel = "Red";
            else if (daysSince >= StalenessWarningDays)
                stalenessLevel = "Yellow";
            else
                stalenessLevel = "None";

            Application.Current.Dispatcher.Invoke(() =>
                _monitor.ReportEtfRotationStatus(
                    _currentExposure, CurrentRiskCap,
                    EtfPositionCount, TradingDaysRemaining,
                    _vixTriggerMode, _vixModeActive, _vixPrice, _themeBudget,
                    _hmmPRiskSmooth, stalenessLevel));

            // Log staleness warnings
            if (daysSince >= StalenessWarningDays)
            {
                _logger.LogWarning(
                    "ETF STALENESS: {Days} trading days since last rebalance (warning={WarnDays}, alert={AlertDays})",
                    daysSince, StalenessWarningDays, StalenessAlertDays);
            }

            // Send Telegram alert for red-level staleness (once per day)
            if (daysSince >= StalenessAlertDays)
            {
                var today = DateTime.UtcNow.Date;
                if (_lastTelegramAlertDate != today)
                {
                    _lastTelegramAlertDate = today;
                    _ = SendTelegramAlertAsync(
                        $"ETF Rotation STALE: {daysSince} trading days since last rebalance! " +
                        $"Last rebalance: {_lastRebalanceDate?.ToString("yyyy-MM-dd") ?? "never"}. " +
                        $"Threshold: {StalenessAlertDays}d. Check app immediately.");
                }
            }
        }
        catch { }
    }

    private async Task SendTelegramAlertAsync(string message)
    {
        try
        {
            var opts = _telegramOptions.CurrentValue;
            if (!opts.Enabled) return;

            var url = $"https://api.telegram.org/bot{opts.BotToken}/sendMessage";
            var client = _httpClientFactory.CreateClient("Telegram");
            var content = new FormUrlEncodedContent(new[]
            {
                new KeyValuePair<string, string>("chat_id", opts.ChatId),
                new KeyValuePair<string, string>("text", message),
            });

            var response = await client.PostAsync(url, content).ConfigureAwait(false);
            _logger.LogInformation("Telegram alert sent: {Status}", response.StatusCode);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to send Telegram alert");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DAILY RISK CONTROL SYSTEM (V7: V3 Tail-Only + Soft Exit + SOFT_PLUS)
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Daily risk check - runs Mon-Fri 3-5 PM ET.
    /// Evaluates V7 risk state machine and executes emergency trades.
    /// </summary>
    private async Task RunDailyRiskCheckAsync(CancellationToken ct)
    {
        try
        {
            var today = DateTime.UtcNow.Date;
            if (_lastRiskCheckDate == today)
                return;
            _lastRiskCheckDate = today;

            // HMM mode: daily p_risk check for regime switch detection
            if (UseHmm)
            {
                await RunHmmRegimeSwitchCheckAsync(ct).ConfigureAwait(false);
                return;
            }

            _logger.LogInformation("Daily risk check: Starting V7 evaluation");

            // 1. Get latest signal data (SPY, MA200, VIX)
            var signal = await _rotationBridge.GetTargetWeightsAsync(
                riskLevel: _riskLevel, useHmm: false, ct: ct).ConfigureAwait(false);
            if (signal.Error is not null)
            {
                _logger.LogWarning("Daily risk check: Signal error - {Error}", signal.Error);
                return;
            }

            // 2. Calculate portfolio metrics
            var positions = _database.GetPositions("ETF");
            if (positions.Count == 0)
            {
                _logger.LogInformation("Daily risk check: No ETF positions");
                return;
            }

            var portfolioValue = (double)positions.Sum(p => p.Shares * p.CurrentPrice);

            // ── Deposit/Withdrawal detection ──
            // Get net_liq from broker to detect D/W (portfolio value alone can't distinguish D/W from PnL)
            var cashInfo = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
            var currentNetLiq = cashInfo.Error is null ? cashInfo.NetLiq : portfolioValue;

            if (_lastKnownNetLiq > 0 && currentNetLiq > 0 && _lastKnownPortfolioValue > 0)
            {
                var netLiqChange = currentNetLiq - _lastKnownNetLiq;
                // Estimate daily PnL from ETF position value changes (same base → comparable)
                var dailyPnl = portfolioValue - _lastKnownPortfolioValue;
                // If net_liq change differs significantly from position PnL, the gap is D/W
                // Threshold: difference > $500 AND net_liq change > 2x position PnL
                var dwGap = Math.Abs(netLiqChange - dailyPnl);
                if (dwGap > 500 && (Math.Abs(dailyPnl) < 0.01 || Math.Abs(netLiqChange) > 2 * Math.Abs(dailyPnl)))
                {
                    var depositOrWithdrawal = netLiqChange - dailyPnl;
                    _cumulativeDepositsWithdrawals += depositOrWithdrawal;
                    _logger.LogWarning(
                        "Deposit/Withdrawal detected: net_liq {Old:C} → {New:C}, Δ={Change:C}, PnL≈{PnL:C}, D/W≈{DW:C}, cumulative={Cum:C}",
                        _lastKnownNetLiq, currentNetLiq, netLiqChange, dailyPnl, depositOrWithdrawal, _cumulativeDepositsWithdrawals);
                    _database.SetEtfRotationState("cumulative_deposits_withdrawals",
                        _cumulativeDepositsWithdrawals.ToString("F2", CultureInfo.InvariantCulture));
                }
            }
            _lastKnownNetLiq = currentNetLiq;
            _lastKnownPortfolioValue = portfolioValue;
            _database.SetEtfRotationState("last_known_net_liq",
                currentNetLiq.ToString("F2", CultureInfo.InvariantCulture));
            _database.SetEtfRotationState("last_known_portfolio_value",
                portfolioValue.ToString("F2", CultureInfo.InvariantCulture));

            // Adjusted peak equity: subtract cumulative D/W so deposits don't inflate peak
            var adjustedPeak = _portfolioPeakEquity + _cumulativeDepositsWithdrawals;
            var adjustedPortfolioValue = portfolioValue;

            if (adjustedPortfolioValue > adjustedPeak)
            {
                _portfolioPeakEquity = adjustedPortfolioValue - _cumulativeDepositsWithdrawals;
                _database.SetEtfRotationState("portfolio_peak_equity",
                    _portfolioPeakEquity.ToString("F2", CultureInfo.InvariantCulture));
                adjustedPeak = adjustedPortfolioValue;
            }

            var drawdown = adjustedPeak > 0
                ? (adjustedPortfolioValue - adjustedPeak) / adjustedPeak
                : 0.0;
            var spyDeviation = signal.SpyDeviation;
            var vixPrice = signal.VixPrice ?? 0.0;
            var spyPrice = signal.SpyPrice;

            // Track SPY prices for 10d new-low detection
            _spy10dPrices.Add(spyPrice);
            while (_spy10dPrices.Count > NewLowLookback) _spy10dPrices.RemoveAt(0);

            // Track SPY<MA200 consecutive days
            if (spyDeviation < 0) _spyBelowMa200Days++;
            else _spyBelowMa200Days = 0;

            _logger.LogInformation(
                "Daily risk: DD={DD:P1} SPY_dev={Dev:P1} VIX={Vix:F1} risk={Level}/{Sub} cap={Cap:P1}",
                drawdown, spyDeviation, vixPrice, _riskLevel, _crisisSubState, _emergencyExposureCap);

            // 3. Snapshot old state
            var oldLevel = _riskLevel;
            var oldCap = _emergencyExposureCap;

            // 4. Evaluate V7 risk state (mutates _riskLevel, _crisisSubState, _emergencyExposureCap)
            EvaluateRiskState(drawdown, spyDeviation, vixPrice, spyPrice);

            // 5. Act on changes
            if (_emergencyExposureCap < oldCap - 0.01 && _emergencyExposureCap < _currentExposure - 0.02)
            {
                // Cap decreased → emergency sells
                if (_monitor.IsConnected)
                {
                    var targetExp = Math.Max(_emergencyExposureCap, _currentExposure - EmergencyMaxStep);
                    _logger.LogWarning(
                        "EMERGENCY SELL: {Level}/{Sub} cap {OldCap:P1} → {NewCap:P1}, exposure {Exp:P1} → {Target:P1}",
                        _riskLevel, _crisisSubState, oldCap, _emergencyExposureCap, _currentExposure, targetExp);

                    await ExecuteEmergencySellsAsync(signal, targetExp, ct).ConfigureAwait(false);
                    _currentExposure = targetExp;
                    _lastEmergencyAdjustDate = DateTime.UtcNow;

                    ReportEvent("ETF Emergency",
                        $"Risk {_riskLevel}/{_crisisSubState}: Reduced to {targetExp:P0}");
                }
            }
            else if (_emergencyExposureCap > oldCap + 0.01 && _emergencyExposureCap > _currentExposure + 0.02)
            {
                // Cap increased → buy back (crisis ramp-up or recovery ramp)
                if (_monitor.IsConnected && _riskLevel is "CRISIS" or "RECOVERY_RAMP")
                {
                    var targetExp = Math.Min(_currentExposure + EmergencyMaxStep,
                        Math.Min(signal.Exposure, _emergencyExposureCap));

                    if (targetExp > _currentExposure + 0.02)
                    {
                        _logger.LogInformation(
                            "RAMP-UP BUY: {Level}/{Sub} cap {OldCap:P1} → {NewCap:P1}, exposure {Exp:P1} → {Target:P1}",
                            _riskLevel, _crisisSubState, oldCap, _emergencyExposureCap, _currentExposure, targetExp);

                        await ExecuteRecoveryBuysAsync(signal, targetExp, ct).ConfigureAwait(false);
                        _currentExposure = targetExp;

                        ReportEvent("ETF Recovery",
                            $"Ramp-up ({_riskLevel}/{_crisisSubState}): {_currentExposure:P0}");
                    }
                }
            }

            // 6. Save state if anything changed
            if (_riskLevel != oldLevel || Math.Abs(_emergencyExposureCap - oldCap) > 0.001)
            {
                SaveRiskStateToDb();
                ReportEtfStatus();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Daily risk check failed");
        }
    }

    /// <summary>
    /// HMM daily regime switch check (3-5PM ET).
    /// Gets fresh HMM p_risk — if it crosses 0.50 (regime switch), triggers force rebalance
    /// without waiting 21 days. Skips deadband but keeps max step 15%.
    ///
    /// CONFIRMATION: To prevent oscillation when p_risk hovers around 0.50,
    /// the signal must indicate the SAME new portfolio for 2 consecutive days
    /// before triggering force rebalance. Single-day spikes are ignored.
    /// </summary>
    private async Task RunHmmRegimeSwitchCheckAsync(CancellationToken ct)
    {
        _logger.LogInformation("HMM daily check: Computing p_risk for regime switch detection");

        try
        {
            var signal = await _rotationBridge.GetTargetWeightsAsync(
                riskLevel: null, useHmm: true,
                onProgress: p => _logger.LogDebug("HMM check: {Detail}", p.Detail),
                ct: ct).ConfigureAwait(false);

            if (signal.Error is not null)
            {
                _logger.LogWarning("HMM daily check: signal error — {Error}", signal.Error);
                return;
            }

            _hmmPRiskSmooth = signal.HmmPRiskSmooth;
            var currentPortfolio = signal.PortfolioUsed;

            _logger.LogInformation(
                "HMM daily check: p_risk={PRisk:F4} portfolio={Portfolio} last_portfolio={LastPortfolio} " +
                "pending={Pending} confirm={Confirm}/{Required}",
                _hmmPRiskSmooth, currentPortfolio ?? "?", _lastPortfolioUsed ?? "unknown",
                _hmmPendingPortfolio ?? "none", _hmmRegimeSwitchConfirmDays, HmmRegimeSwitchConfirmRequired);

            // Detect regime switch: portfolio changed since last rebalance
            var wantsSwitch = !string.IsNullOrEmpty(_lastPortfolioUsed)
                && !string.IsNullOrEmpty(currentPortfolio)
                && !string.Equals(_lastPortfolioUsed, currentPortfolio, StringComparison.OrdinalIgnoreCase);

            if (wantsSwitch)
            {
                // Is this the same pending switch direction as yesterday?
                if (string.Equals(_hmmPendingPortfolio, currentPortfolio, StringComparison.OrdinalIgnoreCase))
                {
                    _hmmRegimeSwitchConfirmDays++;
                }
                else
                {
                    // New direction or first detection — start confirmation
                    _hmmPendingPortfolio = currentPortfolio;
                    _hmmRegimeSwitchConfirmDays = 1;
                }

                _logger.LogInformation(
                    "HMM regime switch pending: {Old} → {New}, confirm day {Days}/{Required}",
                    _lastPortfolioUsed, currentPortfolio, _hmmRegimeSwitchConfirmDays, HmmRegimeSwitchConfirmRequired);

                if (_hmmRegimeSwitchConfirmDays >= HmmRegimeSwitchConfirmRequired)
                {
                    _logger.LogWarning(
                        "HMM REGIME SWITCH CONFIRMED: {Old} → {New} (p_risk={PRisk:F4}, {Days}d confirmed) — force rebalance",
                        _lastPortfolioUsed, currentPortfolio, _hmmPRiskSmooth, _hmmRegimeSwitchConfirmDays);
                    ReportEvent("ETF Regime Switch",
                        $"HMM p_risk crossed 0.50 ({_hmmRegimeSwitchConfirmDays}d confirmed): " +
                        $"{_lastPortfolioUsed} → {currentPortfolio} — force rebalance");

                    // Reset confirmation state
                    _hmmPendingPortfolio = null;
                    _hmmRegimeSwitchConfirmDays = 0;

                    // Force rebalance: skip deadband (handled by regime switch detection in RecalculateAndExecuteAsync)
                    // But override idempotency guard — allow re-run even if already ran today
                    _lastRebalRunDate = null;
                    await RecalculateAndExecuteAsync(isInitial: false, ct).ConfigureAwait(false);
                }
            }
            else
            {
                // Signal agrees with current portfolio — reset any pending switch
                if (_hmmRegimeSwitchConfirmDays > 0)
                {
                    _logger.LogInformation(
                        "HMM regime switch cancelled: signal reverted to {Portfolio} after {Days}d pending",
                        currentPortfolio, _hmmRegimeSwitchConfirmDays);
                }
                _hmmPendingPortfolio = null;
                _hmmRegimeSwitchConfirmDays = 0;
            }

            ReportEtfStatus();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "HMM daily regime switch check failed");
        }
    }

    /// <summary>
    /// V7 risk state machine. Mutates _riskLevel, _crisisSubState, _emergencyExposureCap.
    /// States: RISK_ON → RISK_OFF (80%/70%) → CRISIS (HARD 30% → SOFT 55% → SOFT_PLUS 80%)
    ///       → RECOVERY_RAMP → RISK_ON
    /// </summary>
    private void EvaluateRiskState(double drawdown, double spyDeviation, double vixPrice, double spyPrice)
    {
        var daysSinceCrisis = _lastCrisisDate == DateTime.MinValue
            ? int.MaxValue
            : TraderDatabase.CountTradingDaysBetween(_lastCrisisDate, DateTime.UtcNow);
        var cooldown = daysSinceCrisis < CrisisCooldownDays && vixPrice < VixCrisisExtreme;

        var oldLevel = _riskLevel;
        var newLevel = oldLevel;
        var reason = "No change";

        // ── Layer 1: CRISIS triggers ──
        if (!cooldown)
        {
            if (vixPrice >= VixCrisisThreshold && spyDeviation < 0)
            {
                newLevel = "CRISIS"; reason = $"VIX≥{VixCrisisThreshold} ({vixPrice:F1}) + SPY<MA200";
            }
            else if (vixPrice >= VixCrisisExtreme)
            {
                newLevel = "CRISIS"; reason = $"EXTREME VIX ({vixPrice:F1})";
            }
            else if (drawdown <= DdCrisisThreshold)
            {
                newLevel = "CRISIS"; reason = $"DD≥10% ({drawdown:P1})";
            }
        }

        // ── Layer 2: RISK_OFF (V3 market-confirmed, no VIX-only trigger) ──
        if (newLevel != "CRISIS")
        {
            // Trigger 1: SPY<MA200 AND DD≥6% for 2 days
            if (spyDeviation < 0 && drawdown <= DdRiskOffThreshold)
            {
                _riskOffConfirmDays++;
                if (_riskOffConfirmDays >= RiskOffConfirmDaysRequired && oldLevel != "CRISIS")
                {
                    newLevel = "RISK_OFF"; reason = $"SPY<MA200 + DD≥6% ({drawdown:P1}, {_riskOffConfirmDays}d)";
                }
            }
            // Trigger 2: SPY<MA200 for 3 consecutive days
            else if (_spyBelowMa200Days >= SpyBelowMa200ConfirmDays && oldLevel == "RISK_ON")
            {
                newLevel = "RISK_OFF"; reason = $"SPY<MA200 for {_spyBelowMa200Days} consecutive days";
            }
            else if (drawdown > DdRecoveryThreshold)
            {
                _riskOffConfirmDays = 0;
            }
        }

        // ── Soft CRISIS exit: HARD → SOFT ──
        if (oldLevel == "CRISIS" && newLevel == "CRISIS" && _crisisSubState == "HARD")
        {
            if (vixPrice < VixSoftExitThreshold)
            {
                _softExitCounter++;
                if (_softExitCounter >= SoftExitConfirmDaysRequired)
                {
                    _crisisSubState = "SOFT";
                    _emergencyExposureCap = SoftCap;
                    _softExitCounter = 0;
                    _softPlusCounter = 0;
                    _spy10dPrices.Clear();
                    _spy10dPrices.Add(spyPrice);
                    _logger.LogWarning(
                        "CRISIS sub-state: HARD → SOFT (VIX<{Threshold} for {Days}d, cap={Cap:P0})",
                        VixSoftExitThreshold, SoftExitConfirmDaysRequired, SoftCap);
                }
            }
            else
            {
                _softExitCounter = 0;
            }
        }
        // ── SOFT_PLUS ratchet: SOFT → SOFT_PLUS ──
        else if (oldLevel == "CRISIS" && newLevel == "CRISIS" && _crisisSubState == "SOFT")
        {
            // Check revert: VIX spike → back to HARD
            if (vixPrice >= VixRevertThreshold)
            {
                _crisisSubState = "HARD";
                _emergencyExposureCap = CrisisCap;
                _softPlusCounter = 0;
                _lastCrisisDate = DateTime.UtcNow;
                _logger.LogWarning("CRISIS revert: SOFT → HARD (VIX≥{Threshold})", VixRevertThreshold);
            }
            else
            {
                // Check SPY 10d new low → block upgrade
                bool newLowBlocked = false;
                if (_spy10dPrices.Count >= 2)
                {
                    var prevLow = _spy10dPrices.Take(_spy10dPrices.Count - 1).Min();
                    if (spyPrice < prevLow * 0.995) // 0.5% below 10d low
                    {
                        _softPlusCounter = 0;
                        newLowBlocked = true;
                    }
                }

                if (!newLowBlocked)
                {
                    // Upgrade: SPY>MA200 (proxy for MA50) + VIX<25
                    bool upgradeOk = vixPrice < SoftPlusVixMax && spyDeviation > 0;
                    if (upgradeOk)
                    {
                        _softPlusCounter++;
                        if (_softPlusCounter >= SoftPlusConfirmDaysRequired)
                        {
                            _crisisSubState = "SOFT_PLUS";
                            _emergencyExposureCap = SoftPlusCap;
                            _logger.LogWarning("CRISIS upgrade: SOFT → SOFT_PLUS (cap={Cap:P0})", SoftPlusCap);
                        }
                    }
                    else
                    {
                        _softPlusCounter = 0;
                    }
                }
            }
        }
        // ── SOFT_PLUS revert checks ──
        else if (oldLevel == "CRISIS" && newLevel == "CRISIS" && _crisisSubState == "SOFT_PLUS")
        {
            if (vixPrice >= VixRevertThreshold)
            {
                _crisisSubState = "HARD";
                _emergencyExposureCap = CrisisCap;
                _lastCrisisDate = DateTime.UtcNow;
                _logger.LogWarning("CRISIS revert: SOFT_PLUS → HARD (VIX≥{Threshold})", VixRevertThreshold);
            }
            else if (_spy10dPrices.Count >= 2)
            {
                var prevLow = _spy10dPrices.Take(_spy10dPrices.Count - 1).Min();
                if (spyPrice < prevLow * 0.995)
                {
                    _crisisSubState = "SOFT";
                    _emergencyExposureCap = SoftCap;
                    _logger.LogWarning("CRISIS revert: SOFT_PLUS → SOFT (SPY 10d new low)");
                }
            }
        }

        // ── Recovery check ──
        bool recoveryOk = spyDeviation > 0
            && vixPrice <= VixRecoveryThreshold
            && drawdown >= DdRecoveryThreshold;

        if (recoveryOk && oldLevel is "RISK_OFF" or "CRISIS" or "RECOVERY_RAMP")
        {
            _recoveryConfirmDays++;
            if (_recoveryConfirmDays >= RecoveryConfirmDaysRequired)
            {
                if (oldLevel is "RISK_OFF" or "CRISIS")
                {
                    newLevel = "RECOVERY_RAMP";
                    reason = $"Recovery conditions met ({_recoveryConfirmDays}d)";
                }
                else if (oldLevel == "RECOVERY_RAMP" && _emergencyExposureCap >= 0.99)
                {
                    newLevel = "RISK_ON";
                    reason = "Ramp complete";
                }
            }
        }
        else if (oldLevel != "RECOVERY_RAMP")
        {
            _recoveryConfirmDays = 0;
        }

        // ── Apply transition (with cycle limiter) ──
        if (newLevel != oldLevel)
        {
            bool isCrisis = newLevel == "CRISIS";
            bool isRecovery = newLevel is "RECOVERY_RAMP" or "RISK_ON";

            if (isCrisis || isRecovery || _transitionsThisCycle < MaxTransitionsPerCycle)
            {
                bool validTransition =
                    (oldLevel == "RISK_ON" && newLevel is "RISK_OFF" or "CRISIS") ||
                    (oldLevel == "RISK_OFF" && newLevel == "CRISIS") ||
                    newLevel is "RECOVERY_RAMP" or "RISK_ON";

                if (validTransition)
                {
                    _logger.LogWarning(
                        "Risk transition: {Old} → {New} (reason: {Reason})",
                        oldLevel, newLevel, reason);

                    _riskLevel = newLevel;
                    if (!isRecovery) _transitionsThisCycle++;

                    if (newLevel == "CRISIS")
                    {
                        _crisisSubState = "HARD";
                        _emergencyExposureCap = CrisisCap;
                        _lastCrisisDate = DateTime.UtcNow;
                        _softExitCounter = 0;
                        _softPlusCounter = 0;
                    }
                    else if (newLevel == "RISK_OFF")
                    {
                        _emergencyExposureCap = DynamicRiskOffCap(drawdown);
                        _recoveryConfirmDays = 0;
                    }
                    else if (newLevel == "RISK_ON")
                    {
                        _emergencyExposureCap = 1.0;
                        _recoveryConfirmDays = 0;
                        _crisisSubState = "HARD";
                    }
                    // RECOVERY_RAMP: keep current cap, will ramp up daily
                }
            }
        }
        else if (_riskLevel == "RECOVERY_RAMP")
        {
            // Daily recovery ramp: increase cap by step
            bool turbo = vixPrice < RecoveryTurboVixMax && spyDeviation > 0;
            var step = turbo ? RecoveryTurboStep : RecoverySlowStep;
            _emergencyExposureCap = Math.Min(1.0, _emergencyExposureCap + step);
            if (_emergencyExposureCap >= 0.99)
            {
                _riskLevel = "RISK_ON";
                _emergencyExposureCap = 1.0;
                _crisisSubState = "HARD";
                _logger.LogInformation(
                    "Recovery ramp complete ({Phase}) → RISK_ON",
                    turbo ? "turbo" : "slow");
            }
        }
        else if (_riskLevel == "RISK_OFF")
        {
            // Update dynamic cap if DD worsens
            var newCap = DynamicRiskOffCap(drawdown);
            if (newCap < _emergencyExposureCap)
                _emergencyExposureCap = newCap;
        }
    }

    /// <summary>Dynamic RISK_OFF cap based on drawdown tier.</summary>
    private static double DynamicRiskOffCap(double drawdown)
    {
        if (drawdown <= DdCrisisThreshold) return CrisisCap;       // DD≥10% → 30%
        if (drawdown <= -0.08) return RiskOffCapModerate;           // DD [8-10%) → 70%
        return RiskOffCapLight;                                     // DD [6-8%) → 80%
    }

    /// <summary>Save risk control state to DB for restart persistence.</summary>
    private void SaveRiskStateToDb()
    {
        try
        {
            var pairs = new List<(string Key, string Value)>
            {
                ("risk_level", _riskLevel),
                ("crisis_sub_state", _crisisSubState),
                ("emergency_exposure_cap", _emergencyExposureCap.ToString("F4", CultureInfo.InvariantCulture)),
                (StateKeyLastExposure, _currentExposure.ToString("F6", CultureInfo.InvariantCulture)),
            };
            if (_lastCrisisDate != DateTime.MinValue)
                pairs.Add(("last_crisis_date", _lastCrisisDate.ToString("o")));
            _database.SetEtfRotationStateBatch(pairs);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save risk state");
        }
    }

    /// <summary>
    /// Execute emergency sells to reduce exposure (NO BUYS).
    /// </summary>
    private async Task ExecuteEmergencySellsAsync(
        EtfRotationResult signal, double targetExposure, CancellationToken ct)
    {
        var positions = _database.GetPositions("ETF");
        if (positions.Count == 0) return;

        var portfolioValue = (double)positions.Sum(p => p.Shares * p.CurrentPrice);
        var cash = await GetAvailableCashForEtfAsync(ct).ConfigureAwait(false);
        var totalCapital = portfolioValue + cash;

        // Calculate target dollar values — scale signal weights to emergency exposure
        var scaleFactor = signal.Exposure > 0 ? targetExposure / signal.Exposure : 0;
        var targetValues = new Dictionary<string, double>();
        foreach (var (ticker, weight) in signal.EtfWeights ?? new Dictionary<string, double>())
        {
            targetValues[ticker] = weight * scaleFactor * totalCapital;
        }

        // Sell positions that need to be reduced (reuse existing method)
        foreach (var pos in positions.OrderBy(p => p.Symbol))
        {
            var targetValue = targetValues.GetValueOrDefault(pos.Symbol, 0.0);
            var currentValue = (double)(pos.Shares * pos.CurrentPrice);

            if (targetValue < currentValue - 100) // $100 threshold
            {
                var price = (double)pos.CurrentPrice;
                var targetQty = (int)(targetValue / price);
                var sellQty = pos.Shares - targetQty;

                if (sellQty > 0)
                {
                    _logger.LogWarning("Emergency sell: {Symbol} {Qty} @ ${Price:F2} (reduce {Pct:P1})",
                        pos.Symbol, sellQty, price, sellQty / (double)pos.Shares);

                    // Reuse existing partial sell method
                    await SellEtfPartialAsync(pos.Symbol, sellQty, pos.Shares - sellQty,
                        pos.EntryPrice, pos.TargetWeight ?? 0.0, "Emergency", ct).ConfigureAwait(false);

                    await Task.Delay(1000, ct).ConfigureAwait(false); // Rate limit
                }
            }
        }
    }

    // ContinueRecoveryRampAsync removed — V7 handles recovery ramp inside EvaluateRiskState + RunDailyRiskCheckAsync

    /// <summary>
    /// Execute recovery buys to gradually increase exposure.
    /// </summary>
    private async Task ExecuteRecoveryBuysAsync(
        EtfRotationResult signal, double targetExposure, CancellationToken ct)
    {
        var positions = _database.GetPositions("ETF");
        var portfolioValue = (double)positions.Sum(p => p.Shares * p.CurrentPrice);
        var cash = await GetAvailableCashForEtfAsync(ct).ConfigureAwait(false);
        var totalCapital = portfolioValue + cash;

        // Calculate target dollar values — scale signal weights to recovery exposure
        var scaleFactor = signal.Exposure > 0 ? targetExposure / signal.Exposure : 0;
        var targetValues = new Dictionary<string, double>();
        foreach (var (ticker, weight) in signal.EtfWeights ?? new Dictionary<string, double>())
        {
            targetValues[ticker] = weight * scaleFactor * totalCapital;
        }

        // Buy positions that need to be increased
        var prices = signal.EtfPrices ?? new Dictionary<string, double>();
        foreach (var (ticker, targetValue) in targetValues.OrderBy(kv => kv.Key))
        {
            var currentPos = positions.FirstOrDefault(p =>
                p.Symbol.Equals(ticker, StringComparison.OrdinalIgnoreCase));
            var currentValue = currentPos is not null
                ? (double)(currentPos.Shares * currentPos.CurrentPrice)
                : 0.0;

            if (targetValue > currentValue + 100) // $100 threshold
            {
                var price = prices.GetValueOrDefault(ticker, 0.0);
                if (price <= 0) continue;

                var currentQty = currentPos?.Shares ?? 0;
                var targetQty = (int)(targetValue / price);
                var buyQty = targetQty - currentQty;

                if (buyQty > 0)
                {
                    _logger.LogInformation("Recovery buy: {Symbol} {Qty} @ ${Price:F2}",
                        ticker, buyQty, price);

                    // Reuse existing buy method (handles intents, fills, DB updates)
                    var weight = signal.EtfWeights?.GetValueOrDefault(ticker, 0.0) ?? 0.0;
                    await BuyEtfPositionAsync(ticker, buyQty, weight, ct).ConfigureAwait(false);
                    await Task.Delay(1000, ct).ConfigureAwait(false); // Rate limit
                }
            }
        }
    }

    private async Task<double> GetAvailableCashForEtfAsync(CancellationToken ct)
    {
        try
        {
            var cashResult = await _tradingBridge.GetCashAsync(ct).ConfigureAwait(false);
            var netLiq = cashResult.NetLiq;
            var etfRatio = EtfCapitalRatio;  // 75%
            return netLiq * etfRatio;
        }
        catch
        {
            return 0.0;
        }
    }
}
