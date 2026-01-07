#!/usr/bin/env python3
"""Dynamic signal threshold utilities for AutoTrader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DynamicThresholdResult:
    symbol: str
    base_threshold: float
    threshold: float
    volatility_adj: float
    liquidity_adj: float
    cost_adj: float
    metadata: Dict[str, Any]


def _extract_volatility(polygon_signal: Dict[str, Any]) -> float:
    metadata = polygon_signal.get('metadata') or {}
    multi = metadata.get('multi_time_features') or {}
    vol_map = multi.get('volatility') or {}
    for key in ('21d', '20d', '10d'):
        if key in vol_map:
            return max(float(vol_map[key]), 0.0)
    try:
        return float(metadata.get('individual_factors', {}).get('volatility', 0.0))
    except Exception:
        return 0.0


def _compute_liquidity_adj(quote) -> float:
    try:
        if not quote or not quote.bid or not quote.ask or quote.bid <= 0 or quote.ask <= 0:
            return 1.0
        spread = max(quote.ask - quote.bid, 0.0)
        mid = (quote.ask + quote.bid) / 2.0
        if mid <= 0:
            return 1.0
        spread_ratio = spread / mid
        return min(max(1.0 + spread_ratio * 5.0, 0.8), 1.4)
    except Exception:
        return 1.0


def _compute_volatility_adj(volatility: float) -> float:
    if volatility <= 0:
        return 1.0
    return min(max(volatility / 0.02, 0.7), 1.6)


def _compute_cost_adj(alpha_decision) -> float:
    if not alpha_decision:
        return 1.0
    try:
        cost_bps = abs(alpha_decision.total_cost_bps)
    except AttributeError:
        cost_bps = None
    if cost_bps is None or cost_bps <= 0:
        return 1.0
    return min(max(1.0 + cost_bps / 80.0, 0.9), 1.8)


def compute_dynamic_threshold(
    symbol: str,
    base_threshold: float,
    polygon_signal: Dict[str, Any],
    quote,
    config_manager,
    alpha_decision: Optional[Any] = None,
) -> DynamicThresholdResult:
    """Return a threshold adjusted by volatility, liquidity and cost."""
    volatility = _extract_volatility(polygon_signal)
    vol_adj = _compute_volatility_adj(volatility)
    liq_adj = _compute_liquidity_adj(quote)
    cost_adj = _compute_cost_adj(alpha_decision)

    try:
        signal_cfg = (config_manager.get('signals', {}) if config_manager else {})
    except Exception:
        signal_cfg = {}
    vol_weight = signal_cfg.get('dynamic_threshold_vol_weight', 0.5)
    liq_weight = signal_cfg.get('dynamic_threshold_liq_weight', 0.3)
    cost_weight = signal_cfg.get('dynamic_threshold_cost_weight', 0.2)
    total = max(vol_weight + liq_weight + cost_weight, 1e-6)

    blended_adj = (
        vol_adj * vol_weight +
        liq_adj * liq_weight +
        cost_adj * cost_weight
    ) / total

    threshold = max(base_threshold * blended_adj, base_threshold * 0.5)

    return DynamicThresholdResult(
        symbol=symbol,
        base_threshold=base_threshold,
        threshold=threshold,
        volatility_adj=vol_adj,
        liquidity_adj=liq_adj,
        cost_adj=cost_adj,
        metadata={
            'volatility': volatility,
            'blended_adj': blended_adj,
            'vol_weight': vol_weight,
            'liq_weight': liq_weight,
            'cost_weight': cost_weight,
        }
    )
