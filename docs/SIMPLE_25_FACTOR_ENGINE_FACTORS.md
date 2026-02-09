# simple_25_factor_engine 因子计算一?

`bma_models/simple_25_factor_engine.py` ?`compute_all_17_factors()` 会按顺序计算以下因子（具体输出列?`self.alpha_factors` 决定，通常?`ALL_17_FACTORS`）?

---

## 1. 动量 (Momentum) ?`_compute_momentum_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **momentum_60d** | 60日价格动?| `Close.pct_change(60).乘 √252 年化`，按 ticker 分组 |
| **momentum_10d** | 10日价格动?| `Close.pct_change(10).乘 √252 年化`（在 alpha_factors ?momentum_10d 时计算） |
| **5_days_reversal** | 5日短期反?| `-Close.pct_change(5).乘 √252 年化`（在 alpha_factors ?5_days_reversal 时计算） |

---

## 2. 均值回?/ 技?(Mean Reversion) ?`_compute_mean_reversion_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **rsi_21** | 21周期 RSI | 标准 RSI(21)，归一化到 [-1,1]；T+10 时在 bear  regime 下取 (100-RSI) |

---

## 3. 成交?(Volume) ?`_compute_volume_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **obv_momentum_60d** | OBV 60日动?| OBV = sign(ret)*Volume 累加，再 pct_change(60).乘 √252 年化 |
| **vol_ratio_20d** | 20日成交量?| 昨日成交?/ 20日均?- 1，乘 √252 年化 |
| **liquid_momentum** | 流动性加权动?| 60日价格动?× 相对 60日均量的成交量比，乘 √252 年化 |
| **obv_divergence** | OBV 与价格背?| OBV 40日动?- 价格 40日动量，乘 √252 年化 |

---

## 4. 波动?(Volatility) ?`_compute_volatility_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **atr_ratio** | ATR ?(5?20? | True Range ?5日均 / 20日均 - 1，乘 √252 年化 |

---

## 6. 基本面代?(Fundamental) ?`_compute_fundamental_factors`

当前**已跳?*（返回空 DataFrame），不再计算 roa/ebit?

---

## 7. ?Alpha (New Alpha) ?`_compute_new_alpha_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **near_52w_high** | ?52 周高?| (昨日 close / 252?High.max).乘 √252 年化 - 1 |
| **mom_accel_20_5** | 动量加速度 (20?vs 5? | ?5日日均收益率 - ?15日日均收益率（仅 T+5 因子集时?|

---

## 8. 行为 (Behavioral) ?`_compute_behavioral_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **streak_reversal** | 连续涨跌反转信号 | 连续同向天数 capped ±5，取负（需 alpha_factors 含此项） |

---

## 9. 低频 / 分布 (Low-Frequency) ?`_compute_trend_r2_60`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **trend_r2_60** | 60日趋?R² | 60?log(Close) 对时?[0..59] 线性回归的 R² |

---

## 10. MA 交叉（可选）?`_compute_ma_cross_30_60`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **ma30_ma60_cross** | MA30/MA60 金死?| sign(MA30 - MA60)，仅 alpha_factors 含此项时计算 |

---

## 11. IVOL（可选）?`_compute_ivol_20`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **ivol_20** | 20日特质波动率 | 股票日收益减 SPY 日收益的 20日滚?std，乘 √252 年化；无 SPY 时用 yfinance 或填 0 |

---

## 12. 历史波动??`_compute_hist_volatility`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **hist_vol_20** | 20日历史波动率 | log 收益 20日滚?std，乘 √252 年化 |

---

## 13. 线性相?Alpha ?`_compute_alpha_linreg_corr_20d`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **alpha_linreg_corr_20d** | 20日价?时间相关 | 20日窗口内 Close 与时?[0..19] ?Pearson 相关，乘 √252 年化；需 alpha_factors 含此?|

---


## 15. 市场状?(Market Regime) ?`_compute_market_regime_factors`

| 因子?| 说明 | 计算方式 |
|--------|------|----------|
| **vol_ratio_5_20** | QQQ 波动率比 5?20?| Parkinson 波动?5?20?比，需 alpha_factors 含此?|
| **qqq_tlt_ratio_z** | QQQ/TLT ?Z 分数 | QQQ/TLT 价格比的滚动 Z 分数，需 alpha_factors 含此?|

---

## 因子集合常量（同文件内）

- **ALL_17_FACTORS**?7 个主因子 + alpha_linreg_corr_20d + vol_ratio_5_20 + qqq_tlt_ratio_z（共 19 个名称，注释仍写 17?
- **DEFAULT_11_FACTORS / TOP_FEATURE_SET**：默?11 因子?0/20 ?direct predict ?override 时使用）
- **TOP_9_FEATURES**? 因子备选集?
- **T5_ALPHA_FACTORS**：T+5 用因子列表（?momentum_60d, mom_accel_20_5, streak_reversal 等）

实际参与计算与输出的列由 `self.alpha_factors`（如 `DEFAULT_FEATURE_SET` ?`ALL_17_FACTORS`）决定；未在 alpha_factors 中的因子不会写入最?DataFrame?

\n---\n\n## ӣ (volume_price_corr)\n\n|  | ˵ | 㷽ʽ |\n|--------|------|----------|\n| **volume_price_corr_5d** | 5 | ret.rolling(5).corr(Volume).shift(1) |\n| **volume_price_corr_10d** | 10 | ret.rolling(10).corr(Volume).shift(1) |\n
