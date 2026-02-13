# VIX-Enhanced ETF Rotation Robustness Testing Guide

## Overview

The `etf_rotation_vix_robustness.py` script provides a comprehensive suite of tests to validate that the VIX-enhanced strategy is **NOT overfitted** and performs robustly across different conditions.

## Location

```
D:\trade\TraderApp\python\etf_rotation_vix_robustness.py
```

## How to Run

```bash
cd D:\trade\TraderApp\python
python etf_rotation_vix_robustness.py
```

**Estimated Runtime:** 10-15 minutes (runs 30+ backtests)

## Tests Performed

### Test A: Annual Performance Breakdown
**Goal:** Verify consistent performance across different market years

**What it does:**
- Breaks down full backtest (2022-2026) by individual year
- Shows Sharpe/CAGR/MaxDD for each year
- **Key Focus:** 2022 (bear market) and 2025 (recent period)

**Success Criteria:**
- All years should have positive Sharpe
- 2022 should NOT have extreme drawdown
- 2025 should maintain strong performance

---

### Test B: Threshold Perturbation Testing
**Goal:** Verify strategy is NOT overfitted to exact VIX thresholds

**What it does:**
- Tests 9 combinations of VIX thresholds:
  - vix_low: 19, 20, 21
  - vix_high: 24, 25, 26
- Runs full backtest for each combination
- Checks if Sharpe stays above 1.0 for all combinations

**Success Criteria:**
- **ALL** 9 combinations should have Sharpe >= 1.0
- Mean Sharpe should be close to baseline (low std)
- No single threshold set should be dramatically better

**Verdict:** PASS if all >= 1.0, FAIL otherwise

---

### Test C: Cost Pressure Testing
**Goal:** Verify VIX enhancement still adds value at higher costs

**What it does:**
- Tests at 3 cost levels:
  - 10 bps (baseline)
  - 20 bps (2x cost)
  - 30 bps (3x cost)
- Compares P2+VIX Full vs P2 Baseline at each cost level

**Success Criteria:**
- P2+VIX Full should **outperform** P2 Baseline at ALL cost levels
- Sharpe degradation should be similar for both strategies

**Verdict:** PASS if VIX wins at all costs, FAIL otherwise

---

### Test D: Data Source Validation
**Goal:** Verify VIX index data quality

**What it does:**
- Checks VIX data for:
  - Missing days (should be < 5%)
  - Outliers (VIX > 100 or < 5)
  - Reasonable range and statistics

**Success Criteria:**
- < 5% missing data
- No outliers
- VIX range within historical norms (5-80)

**Verdict:** PASS if data quality checks pass

---

### Test E: Out-of-Sample (OOS) Split Testing
**Goal:** Verify strategy works on unseen data (not tuned on it)

**What it does:**
- **In-Sample (IS):** 2022-02-24 to 2024-12-31 (tuning period)
- **Out-of-Sample (OOS):** 2025-01-01 to 2026-02-10 (validation period)
- Runs separate backtests for each period

**Success Criteria:**
- OOS Sharpe should be >= 0.85
- OOS Sharpe degradation should be minimal (< 0.3 from IS)
- OOS should NOT show dramatic performance collapse

**Verdict:** PASS if OOS meets floor (0.85), FAIL otherwise

---

### Test F: VIX Parameter Grid Search (Stage 1)
**Goal:** Find optimal VIX thresholds (coarse grid)

**What it does:**
- Tests 4 combinations:
  - vix_low: 16, 18
  - vix_high: 24, 26
  - Constraint: vix_high >= vix_low + 3
- Ranks by Sharpe
- Selects top 3 for further testing

**Output:**
- Top 3 parameter sets ranked by Sharpe
- Full grid results

---

## Output Files

### 1. JSON Report
**Location:** `D:\trade\TraderApp\result\vix_robustness_report.json`

**Contents:**
- Complete test results in machine-readable format
- All metrics for all test variants
- Detailed breakdowns

**Use Case:** Programmatic analysis, further processing

---

### 2. Markdown Report
**Location:** `D:\trade\TraderApp\result\vix_robustness_report.md`

**Contents:**
- Human-readable summary of all tests
- Tables with key metrics
- Pass/Fail verdicts for each test
- Overall robustness verdict

**Use Case:** Quick review, sharing with others

---

## Interpretation

### Overall Verdict

The script provides an **Overall Verdict** based on Tests B, C, D, E:

- **PASS:** All robustness checks passed
  - Strategy is robust across parameter variations
  - Works at higher costs
  - Validates on OOS data
  - Data quality is good

- **FAIL:** One or more tests failed
  - Review individual test results
  - May indicate overfitting or data issues
  - Consider parameter refinement

---

## What Makes a Strategy Robust?

### Good Signs
1. **Threshold Perturbation:** All 9 combinations have Sharpe > 1.0 (not sensitive to exact thresholds)
2. **Cost Pressure:** VIX still wins at 30 bps cost (value persists at higher friction)
3. **OOS Split:** OOS Sharpe >= 0.85 (generalizes to unseen data)
4. **Annual Consistency:** All years have positive Sharpe (works in different regimes)

### Warning Signs
1. **Threshold Perturbation:** Only 1-2 combinations work well (overfitted to exact params)
2. **Cost Pressure:** VIX loses to baseline at higher costs (turnover eats alpha)
3. **OOS Split:** OOS Sharpe << IS Sharpe (overfitted to training period)
4. **Annual Breakdown:** One great year, rest mediocre (lucky outlier)

---

## Next Steps After Running

### If All Tests Pass
1. Review markdown report for detailed metrics
2. Consider the top 3 parameter sets from grid search
3. Proceed to **Stage 2 grid search** (finer granularity)
4. Consider live trading or paper trading

### If Some Tests Fail
1. Review which specific tests failed
2. **Test B fails:** Widen threshold bands or use adaptive thresholds
3. **Test C fails:** Reduce turnover (wider deadband, longer rebalance freq)
4. **Test E fails:** Re-tune on more recent data, or use walk-forward validation
5. **Test A fails:** Check if strategy breaks in specific regimes

---

## Customization

### Change Test Parameters

Edit `RobustnessConfig` in the script:

```python
@dataclass
class RobustnessConfig:
    # Threshold ranges
    vix_low_range: List[float] = field(default_factory=lambda: [19.0, 20.0, 21.0])
    vix_high_range: List[float] = field(default_factory=lambda: [24.0, 25.0, 26.0])

    # Cost ranges
    cost_range: List[float] = field(default_factory=lambda: [10.0, 20.0, 30.0])

    # Success criteria
    target_sharpe: float = 1.0
    oos_sharpe_floor: float = 0.85
```

### Add More Tests

The script is modular. You can add custom tests by:
1. Creating a new `test_<name>()` function
2. Adding it to `run_robustness_suite()`
3. Including results in JSON/markdown output

---

## Technical Details

### Dependencies
- Uses existing `etf_rotation_v6_vix_enhanced.py` functions
- Reuses backtest engine from `etf_rotation_v4_refined.py`
- Leverages metric computation from `etf_rotation_strategy.py`

### Data Requirements
- Polygon API key (hardcoded in script)
- VIX index data from yfinance (free tier)
- Portfolio tickers + BIL + SPY from Polygon

### Performance
- Each backtest takes ~5-10 seconds
- Total suite: ~30 backtests = 5-15 minutes
- Can parallelize for faster execution (future enhancement)

---

## FAQ

**Q: Why is OOS floor 0.85 instead of 1.0?**
A: OOS periods are shorter and can have higher variance. 0.85 is still excellent and shows the strategy generalizes well.

**Q: What if baseline config (vix_low=20, vix_high=25) is NOT in top 3?**
A: That's fine! Grid search may find better parameters. Use top 3 for Stage 2 refinement.

**Q: Why only 2x2 grid in Test F?**
A: This is a **coarse grid** (Stage 1). After finding promising regions, you can run finer grids in Stage 2.

**Q: Can I test different theme budgets?**
A: Yes! Modify `VixConfig` parameters in the script or add a new grid search for theme budgets.

**Q: How do I know if Sharpe 1.0 is good enough?**
A: For a 75% capital strategy with 10 bps cost and no stop loss:
- Sharpe > 1.0 is excellent
- Sharpe 0.7-1.0 is good
- Sharpe < 0.7 needs improvement

---

## Support

If the script fails or produces unexpected results:
1. Check console output for error messages
2. Verify Polygon API key is valid
3. Ensure yfinance can fetch VIX data
4. Check JSON report for partial results

---

**Last Updated:** 2026-02-11
