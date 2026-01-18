#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
80/20 time-split with DAILY ROLLING (Overlapping Observations) + Newey-West HAC Correction

Referee Response:
-----------------
Addresses Major Concern #1: Sample size insufficiency (仅25个独立观测)

改进方案 (Improvement Approach):
- 使用1天滚动观测 (1-day rolling) instead of 10天非重叠 (10-day non-overlapping)
- 观测数量: 25期 → ~1250期 (50x increase)
- 统计推断校正: Newey-West HAC (heteroskedasticity and autocorrelation consistent)
- 滞后阶数: max(10, 2*horizon_days) = 20 lags (robust to overlapping structure)

Justification for Lag Order:
----------------------------
1. Hansen & Hodrick (1980): For h-period overlapping returns, use lag ≥ h-1
2. Newey & West (1987): Automatic lag selection = floor(4*(T/100)^(2/9))
3. Andrews (1991): Data-dependent bandwidth selection
4. Conservative choice: Use 2*horizon_days = 20 lags (超过理论最小值)

Statistical Corrections Applied:
---------------------------------
1. IC t-statistics: Use Newey-West SE with 20 lags
2. Return statistics: Use Hansen-Hodrick SE for h-period returns
3. Sharpe Ratio: Bootstrap with block length = horizon_days
4. Confidence Intervals: Report 95% CI for all metrics

Output clearly states:
----------------------
"基于重叠观测 (overlapping observations)，使用Newey-West HAC校正 (lag=20)"
