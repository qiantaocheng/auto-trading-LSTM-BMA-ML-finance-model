#!/usr/bin/env python3
"""
延迟数据配置模块
Delayed Data Configuration for Trading with 15-minute delayed market data

用于管理延迟行情数据（如免费Polygon数据的15分钟延迟）下的交易策略
"""

from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DelayedDataConfig:
    """延迟数据交易配置"""

    # 基础配置
    enabled: bool = True                    # 是否启用延迟数据交易
    data_delay_minutes: int = 15            # 数据延迟时间（分钟）

    # 信号过滤
    min_confidence_threshold: float = 0.8   # 最小置信度阈值（延迟数据需要更高置信度）
    min_signal_strength: float = 0.7        # 最小信号强度

    # 仓位调整
    position_size_reduction: float = 0.4    # 仓位缩减比例（延迟数据风险更高）
    max_single_position_pct: float = 0.08   # 单仓位上限（比实时数据更保守）

    # Alpha调整
    min_alpha_multiplier: float = 1.0       # 最小Alpha倍数（需要更强的Alpha才交易）
    alpha_decay_factor: float = 0.95        # Alpha衰减因子（考虑数据延迟）

    # 风险控制
    max_daily_trades: int = 10              # 日内最大交易次数（延迟数据降低频率）
    min_hold_period_minutes: int = 60       # 最小持仓时间（避免频繁交易）

    # 市场条件
    min_volume_ratio: float = 1.5           # 最小成交量倍数（确保流动性）
    max_spread_bps: float = 50.0            # 最大买卖价差（基点）

    # 执行策略
    use_limit_orders: bool = True           # 优先使用限价单
    limit_order_offset_bps: float = 10.0    # 限价单偏移（基点）

    # 止损止盈
    use_wider_stops: bool = True            # 使用更宽的止损（考虑延迟）
    stop_loss_multiplier: float = 1.5       # 止损倍数（相对于实时数据）
    take_profit_multiplier: float = 1.3     # 止盈倍数


# 默认配置（保守策略）
DEFAULT_DELAYED_CONFIG = DelayedDataConfig()


# 激进配置（更高风险容忍度）
AGGRESSIVE_DELAYED_CONFIG = DelayedDataConfig(
    min_confidence_threshold=0.7,
    position_size_reduction=0.6,
    max_daily_trades=20,
    min_alpha_multiplier=0.8
)


# 保守配置（最低风险）
CONSERVATIVE_DELAYED_CONFIG = DelayedDataConfig(
    min_confidence_threshold=0.9,
    position_size_reduction=0.3,
    max_single_position_pct=0.05,
    max_daily_trades=5,
    min_alpha_multiplier=1.5
)


def should_trade_with_delayed_data(config: DelayedDataConfig) -> Tuple[bool, str]:
    """
    判断是否可以使用延迟数据进行交易

    Args:
        config: 延迟数据配置

    Returns:
        (是否可以交易, 原因)
    """
    # 检查是否启用
    if not config.enabled:
        return False, "Delayed data trading is disabled"

    # 检查配置有效性
    if config.data_delay_minutes < 0:
        return False, "Invalid data delay configuration"

    if config.position_size_reduction <= 0 or config.position_size_reduction > 1:
        return False, "Invalid position size reduction (must be 0-1)"

    if config.min_confidence_threshold <= 0 or config.min_confidence_threshold > 1:
        return False, "Invalid confidence threshold (must be 0-1)"

    # 所有检查通过
    reason = f"Delayed data trading enabled ({config.data_delay_minutes}min delay)"
    return True, reason


def validate_signal_for_delayed_data(
    signal_strength: float,
    confidence: float,
    config: DelayedDataConfig
) -> Tuple[bool, str]:
    """
    验证信号是否满足延迟数据的交易条件

    Args:
        signal_strength: 信号强度
        confidence: 置信度
        config: 延迟数据配置

    Returns:
        (是否通过, 原因)
    """
    # 检查置信度
    if confidence < config.min_confidence_threshold:
        return False, f"Confidence {confidence:.2f} < threshold {config.min_confidence_threshold:.2f}"

    # 检查信号强度
    if abs(signal_strength) < config.min_signal_strength:
        return False, f"Signal strength {abs(signal_strength):.2f} < threshold {config.min_signal_strength:.2f}"

    return True, "Signal meets delayed data criteria"


def adjust_position_size_for_delayed_data(
    original_size: int,
    config: DelayedDataConfig
) -> int:
    """
    根据延迟数据配置调整仓位大小

    Args:
        original_size: 原始仓位大小
        config: 延迟数据配置

    Returns:
        调整后的仓位大小
    """
    adjusted_size = int(original_size * config.position_size_reduction)

    # 确保至少有1股（如果原始大小>0）
    if original_size > 0 and adjusted_size == 0:
        adjusted_size = 1

    logger.debug(
        f"Position size adjusted for delayed data: {original_size} → {adjusted_size} "
        f"(reduction={config.position_size_reduction:.2%})"
    )

    return adjusted_size


def get_delayed_data_config_by_strategy(strategy: str = "default") -> DelayedDataConfig:
    """
    根据策略获取延迟数据配置

    Args:
        strategy: 策略名称 ("default", "aggressive", "conservative")

    Returns:
        对应的延迟数据配置
    """
    strategy = strategy.lower()

    if strategy == "aggressive":
        return AGGRESSIVE_DELAYED_CONFIG
    elif strategy == "conservative":
        return CONSERVATIVE_DELAYED_CONFIG
    else:
        return DEFAULT_DELAYED_CONFIG


# 导出接口
__all__ = [
    'DelayedDataConfig',
    'DEFAULT_DELAYED_CONFIG',
    'AGGRESSIVE_DELAYED_CONFIG',
    'CONSERVATIVE_DELAYED_CONFIG',
    'should_trade_with_delayed_data',
    'validate_signal_for_delayed_data',
    'adjust_position_size_for_delayed_data',
    'get_delayed_data_config_by_strategy',
]
