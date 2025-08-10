from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def sma(values: List[float], n: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        out.append(s / n if i >= n - 1 else math.nan)
    return out


def stddev(values: List[float], n: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    s2 = 0.0
    for i, v in enumerate(values):
        s += v
        s2 += v * v
        if i >= n:
            s -= values[i - n]
            s2 -= values[i - n] * values[i - n]
        if i >= n - 1:
            mean = s / n
            var = max(s2 / n - mean * mean, 0.0)
            out.append(math.sqrt(var))
        else:
            out.append(math.nan)
    return out


def rsi(values: List[float], n: int) -> List[float]:
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    avg_gain = sma(gains, n)
    avg_loss = sma(losses, n)
    out: List[float] = []
    for g, l in zip(avg_gain, avg_loss):
        if math.isnan(g) or math.isnan(l) or l == 0:
            out.append(math.nan)
        else:
            rs = g / l
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


def bollinger(values: List[float], n: int, k: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    ma = sma(values, n)
    sd = stddev(values, n)
    upper: List[float] = []
    lower: List[float] = []
    for m, s in zip(ma, sd):
        if math.isnan(m) or math.isnan(s):
            upper.append(math.nan)
            lower.append(math.nan)
        else:
            upper.append(m + k * s)
            lower.append(m - k * s)
    return ma, upper, lower


def zscore(values: List[float], n: int) -> List[float]:
    ma = sma(values, n)
    sd = stddev(values, n)
    out: List[float] = []
    for v, m, s in zip(values, ma, sd):
        if math.isnan(m) or math.isnan(s) or s == 0:
            out.append(math.nan)
        else:
            out.append((v - m) / s)
    return out


def atr(high: List[float], low: List[float], close: List[float], n: int = 14) -> List[float]:
    """Average True Range."""
    tr: List[float] = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            tr.append(
                max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
            )
    return sma(tr, n)

