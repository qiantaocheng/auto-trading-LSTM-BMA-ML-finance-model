# VIX-Enhanced ETF Rotation Robustness Testing Report

Generated: 2026-02-11T13:03:15.446232

---

## Test Configuration

**Backtest Period:** 2022-02-24 to 2026-02-10

**VIX Proxy:** ^VIX

**Portfolio:**
- QQQ: 25.0%
- USMV: 25.0%
- QUAL: 20.0%
- PDBC: 15.0%
- DBA: 5.0%
- COPX: 5.0%
- URA: 5.0%

## Test A: Annual Performance Breakdown

**Overall Metrics:**
- Sharpe: 1.131
- CAGR: 16.07%
- MaxDD: -13.0%

**Year-by-Year:**

| Year | Return | Vol | MaxDD | Sharpe |
|------|--------|-----|-------|--------|
| 2023 | 12.5% | 7.5% | -5.6% | 2.01 |
| 2024 | 15.9% | 9.6% | -7.2% | 1.58 |
| 2025 | 13.8% | 12.2% | -13.0% | 1.13 |
| 2026 | 4.0% | 12.4% | -4.1% | 3.03 |

## Test B: Threshold Perturbation Testing

**Verdict:** PASS

- Sharpe Range: 1.078 to 1.144
- Mean Sharpe: 1.114 ¡À 0.025
- All above 1.0: True

**Detailed Results:**

| VIX Low | VIX High | Sharpe | CAGR | MaxDD |
|---------|----------|--------|------|-------|
| 19 | 24 | 1.093 | 15.49% | -13.5% |
| 19 | 25 | 1.078 | 15.38% | -13.8% |
| 19 | 26 | 1.078 | 15.38% | -13.8% |
| 20 | 24 | 1.144 | 15.94% | -12.4% |
| 20 | 25 | 1.131 | 16.07% | -13.0% |
| 20 | 26 | 1.131 | 16.07% | -13.0% |
| 21 | 25 | 1.128 | 16.05% | -13.0% |
| 21 | 26 | 1.128 | 16.05% | -13.0% |

## Test D: Data Source Validation

**Verdict:** PASS

- Total Days: 994
- Missing: 0 (0.0%)
- Range: 11.86 to 52.33
- Mean: 19.02, Median: 17.45

## Test E: Out-of-Sample Split Testing

**Verdict:** FAIL

| Period | Sharpe | CAGR | MaxDD |
|--------|--------|------|-------|
| In-Sample (2022-02-24 to 2024-12-31) | 1.277 | 15.86% | -7.2% |
| Out-of-Sample (2025-01-01 to 2026-02-10) | 0.592 | 5.64% | -0.6% |

**Degradation:** +0.685

## Test F: VIX Parameter Grid Search

**Top 3 Parameter Sets:**

| Rank | VIX Low | VIX High | Sharpe | CAGR | MaxDD |
|------|---------|----------|--------|------|-------|
| 1 | 18 | 26 | 1.030 | 14.38% | -12.6% |
| 2 | 18 | 24 | 1.014 | 13.94% | -11.8% |
| 3 | 16 | 24 | 0.998 | 12.85% | -10.3% |

---

## Overall Verdict

**FAIL**

Some robustness tests failed. Review individual test results for details.
