# simple_25_factor_engine — 当前保留的计算代码一览

以下因子在 `bma_models/simple_25_factor_engine.py` 中**均有对应计算实现**（按调用顺序与函数整理）。已删除的因子（momentum_60d, ma30_ma60_cross, bollinger_squeeze, rsrs_beta_18, roa, ebit, trend_r2_60, ivol 重复实现等）不再列出。

---

## 1. 动量因子 `_compute_momentum_factors` (约 916–946 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **momentum_10d** | `pct_change(10).shift(1)`，与 T5 一致 |
| **reversal_3d** | `(-pct_change(3)).shift(1)` |
| **reversal_5d** | `(-pct_change(5)).shift(1)` |
| **liquid_momentum_10d** | `(ret*Close*Volume).rolling(10).sum() / (Close*Volume).rolling(10).sum()`，再 `shift(1)`，与 T5 一致 |
| **sharpe_momentum_5d** | 5d `mean(ret)/std(ret)`，再 `shift(1)` |

---

## 2. 均值回归/技术 `_compute_mean_reversion_factors` (约 989–1015 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **rsi_14** | 标准 RSI(14)：`gain/loss` 用 `rolling(14, min_periods=14).mean()`，`rsi = 100 - 100/(1+rs)`，再 `shift(1)` |
| **price_ma20_deviation** | `(Close - ma20)/ma20`，`ma20 = rolling(20, min_periods=20).mean()`，再 `shift(1)` |

---

## 3. 成交量因子 `_compute_volume_factors` (约 1017–1053 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **obv_momentum_60d** | `obv = sign(pct_change(Close))*Volume` 累加；`obv.pct_change(60)` |
| **vol_ratio_20d** | `Volume / rolling(20).mean(Volume).shift(1)`，再 `shift(1)` |
| **volume_price_corr_5d** | `ret.rolling(5).corr(Volume).shift(1)` |
| **volume_price_corr_10d** | `ret.rolling(10).corr(Volume).shift(1)` |
| **obv_divergence_20d** | `obv.pct_change(20) - Close.pct_change(20)`，再 `shift(1)` |
| **avg_trade_size** | 若有 `Transactions`：`(Volume/Transactions).rolling(20).mean().shift(1)`；否则填 0 |

---

## 4. 波动率因子 `_compute_volatility_factors` (约 1055–1070 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **atr_ratio** | `true_range = max(High-Low, |High-prev_Close|, |Low-prev_Close|)`；`atr_5d/atr_20d - 1`，rolling(5) 与 rolling(20) |

---

## 5. 下行 Beta `_compute_downside_beta_ewm_21` (约 1179–1252 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **downside_beta_ewm_21** | 仅当 benchmark(QQQ) 日收益 < 0 时计入；`span=21` EWMA：`Cov_ewm(stock_ret, bench_ret|R_m<0) / Var_ewm(bench_ret|R_m<0)`。仅当 `alpha_factors` 含该名时调用。 |

---

## 6. 高 Alpha `_compute_new_alpha_factors` (约 948–987 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **near_52w_high** | `high_252 = Close.rolling(252, min_periods=60).max()`；`(Close/high_252 - 1).shift(1)` |
| **mom_accel_20_5** | `mom_recent_5d = (close/close_5d)^(1/5)-1`，`mom_prior_15d = (close_5d/close_20d)^(1/15)-1`；`mom_accel_20_5 = mom_recent_5d - mom_prior_15d`。仅当 `alpha_factors` 含该名时写入。 |

---

## 7. 行为因子 `_compute_behavioral_factors` (约 1258–1298 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **streak_reversal** | 日收益阈值 `thr=0.0005` 得符号序列，连续同向计数（run），cap ±5；输出为负的 run。仅当 `alpha_factors` 含该名时调用。 |

---

## 8. 收益偏度 `_compute_ret_skew_20d` (约 849–859 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **ret_skew_20d** | `log(Close/Close.shift(1)).rolling(20, min_periods=20).skew()`，inf/nan 填 0 |

---

## 9. 趋势 R² `_compute_trend_r2_60` (约 878–881 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **trend_r2_20** | 20 日窗口内 `y = log(Close)` 对 `x = [0..19]` 线性回归；`r2 = 1 - ss_res/ss_tot`，clip [0,1]。注：函数名为 _compute_trend_r2_60，但仅输出 trend_r2_20。 |

---

## 10. IVOL `_compute_ivol_20` (约 1072–1105 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **ivol_20** | 若含 SPY：`stock_ret - spy_ret_by_date`，再 `rolling(20, min_periods=10).std()`；无 SPY 时填 0。仅当 `alpha_factors` 含该名时调用。 |

---

## 11. 历史波动率 `_compute_hist_volatility` (约 883–914 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **hist_vol_20** | `log_ret = log(Close/Close.shift(1))`；`sigma20 = log_ret.rolling(20, min_periods=10).std()`；`hist_vol_20 = sigma20.fillna(0)` |

---

## 12. 内联：Falling-knife 风险 (约 461–474 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|

---

## 13. 情感 `_compute_sentiment_for_market_data` (约 799–847 行)

| 因子名 | 计算代码/公式 |
|--------|----------------|
| **sentiment_score**（及 analyzer 其他列） | 调用 `UltraFastSentimentFactor.process_universe_sentiment(...)`，按 (date, ticker) 对齐。仅当 `enable_sentiment=True` 且 API 可用时计算。 |

---

## 未实现（文档曾提及）

- **alpha_linreg_corr_20d**：无 `_compute_alpha_linreg_corr_20d`。
- **vol_ratio_5_20**、**qqq_tlt_ratio_z**：无 `_compute_market_regime_factors`。

---

## 汇总：当前有计算代码的因子名（共 24 个）

1. momentum_10d  
2. reversal_3d  
3. reversal_5d  
4. liquid_momentum_10d  
5. sharpe_momentum_5d  
6. rsi_14  
7. price_ma20_deviation  
9. obv_momentum_60d  
10. vol_ratio_20d  
11. volume_price_corr_10d  
12. obv_divergence_20d  
13. avg_trade_size  
14. atr_ratio  
15. downside_beta_ewm_21  
16. near_52w_high  
17. mom_accel_20_5  
18. streak_reversal  
19. ret_skew_20d  
20. trend_r2_20  
21. ivol_20  
22. hist_vol_20  
24. sentiment_score（及情感模块其它列）

实际参与训练/预测的列由 **`self.alpha_factors`** 决定（默认 T10_ALPHA_FACTORS）；上表中未列入 T10/T5 的因子只有在被加入 `alpha_factors` 时才会出现在最终 DataFrame 中。
