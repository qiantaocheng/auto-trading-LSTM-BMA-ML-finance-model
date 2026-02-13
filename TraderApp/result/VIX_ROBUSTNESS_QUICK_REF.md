# VIX Robustness Testing - Quick Reference Card

## Run Tests
```bash
D:\trade\TraderApp\run_vix_robustness.bat
```
**Time:** 10-15 minutes | **Output:** `result/vix_robustness_report.md` + `.json`

---

## Six Tests at a Glance

| Test | What It Checks | Pass Criteria | Fail Means |
|------|----------------|---------------|------------|
| **A: Annual** | Year-by-year consistency | All years positive Sharpe | One-year wonder |
| **B: Threshold** | VIX parameter sensitivity | All 9 combos Sharpe ≥ 1.0 | Overfitted params |
| **C: Cost** | Higher transaction costs | VIX beats baseline at all costs | Turnover too high |
| **D: Data** | VIX data quality | Missing < 5%, no outliers | Unreliable backtest |
| **E: OOS** | Out-of-sample validation | OOS Sharpe ≥ 0.85 | Classic overfit |
| **F: Grid** | Better parameters exist? | Top 3 identified | — |

---

## Pass/Fail Verdicts

**PASS** = All of B, C, D, E pass → Strategy is robust, ready for Stage 2 fine-tuning

**FAIL** = One or more fail → Fix issues before proceeding

---

## Expected Metrics (If PASS)

- **Overall Sharpe:** 1.05-1.15
- **2022 (Bear):** Positive Sharpe, controlled DD
- **2025 (Recent):** Strong performance
- **Threshold Range:** All 9 combos 1.0-1.2 Sharpe
- **30 bps Cost:** VIX still beats baseline
- **OOS Sharpe:** 0.85-1.0

---

## What to Do Next

### All Pass
1. Review `vix_robustness_report.md`
2. Note top 3 parameter sets from Test F
3. Run Stage 2 fine-grid search (next script)
4. Select final parameters
5. Begin paper trading

### Some Fail
1. See "Fail Means" column above
2. Review detailed results in markdown report
3. Implement fixes (see guide)
4. Re-run robustness suite
5. Only proceed after PASS

---

## Files

| File | Purpose |
|------|---------|
| `python/etf_rotation_vix_robustness.py` | Main testing script |
| `run_vix_robustness.bat` | Windows runner |
| `result/vix_robustness_report.json` | Machine-readable output |
| `result/vix_robustness_report.md` | Human-readable report |
| `result/VIX_ROBUSTNESS_TESTING_GUIDE.md` | Detailed documentation |
| `result/VIX_ROBUSTNESS_TESTING_SUMMARY.md` | Testing philosophy |
| `result/VIX_ROBUSTNESS_QUICK_REF.md` | This cheat sheet |

---

## Quick Fixes

| Failed Test | Quick Fix |
|-------------|-----------|
| **B: Threshold** | Widen threshold bands or use adaptive |
| **C: Cost** | Increase deadband, reduce turnover |
| **E: OOS** | Re-tune on recent data, use ensemble |

---

## Key Numbers to Remember

- **Target Sharpe:** 1.0+ (threshold)
- **OOS Floor:** 0.85 (acceptable degradation)
- **Baseline Sharpe:** 0.85 (V5 Portfolio B + P2)
- **VIX Enhancement:** +0.20 to +0.30 Sharpe
- **Cost Tests:** 10, 20, 30 bps
- **Threshold Combos:** 9 (3×3 grid)

---

**Last Updated:** 2026-02-11
