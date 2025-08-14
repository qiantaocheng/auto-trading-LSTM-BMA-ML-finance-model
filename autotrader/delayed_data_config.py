#!/usr/bin/env python3
"""
延迟数据交易配置
针对Polygon 15分钟延迟数据的特殊配置
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DelayedDataConfig:
    """延迟数据交易配置"""
    
    # 基础配置
    enabled: bool = True
    data_delay_minutes: int = 15  # Polygon数据延迟
    
    # 交易决策调整
    min_alpha_multiplier: float = 1.8  # 延迟环境下需要更高的alpha
    min_confidence_threshold: float = 0.8  # 最低信号置信度
    position_size_reduction: float = 0.4  # 仓位减少到40%
    
    # 风险控制加强
    max_single_position_pct: float = 0.02  # 单票最大仓位2%（vs实时的3%）
    max_portfolio_turnover: float = 0.3  # 最大组合换手30%
    min_holding_minutes: int = 30  # 最短持仓时间30分钟
    
    # 价格验证
    max_price_deviation_pct: float = 0.02  # 价格偏差不超过2%
    require_price_confirmation: bool = True  # 需要价格确认
    
    # 订单执行调整
    limit_order_buffer_bps: int = 10  # 限价单缓冲10bp
    order_timeout_seconds: int = 300  # 订单超时5分钟
    
    # 市场时间检查
    avoid_market_open_minutes: int = 30  # 避开开盘前30分钟
    avoid_market_close_minutes: int = 30  # 避开收盘前30分钟
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enabled': self.enabled,
            'data_delay_minutes': self.data_delay_minutes,
            'min_alpha_multiplier': self.min_alpha_multiplier,
            'min_confidence_threshold': self.min_confidence_threshold,
            'position_size_reduction': self.position_size_reduction,
            'max_single_position_pct': self.max_single_position_pct,
            'max_portfolio_turnover': self.max_portfolio_turnover,
            'min_holding_minutes': self.min_holding_minutes,
            'max_price_deviation_pct': self.max_price_deviation_pct,
            'require_price_confirmation': self.require_price_confirmation,
            'limit_order_buffer_bps': self.limit_order_buffer_bps,
            'order_timeout_seconds': self.order_timeout_seconds,
            'avoid_market_open_minutes': self.avoid_market_open_minutes,
            'avoid_market_close_minutes': self.avoid_market_close_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DelayedDataConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def get_market_session_info():
    """获取市场交易时段信息"""
    from datetime import datetime, time
    import pytz
    
    # 美股交易时间 (EST)
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est).time()
    
    market_open = time(9, 30)  # 9:30 AM EST
    market_close = time(16, 0)  # 4:00 PM EST
    
    # 检查是否在交易时间内
    is_market_hours = market_open <= now_est <= market_close
    
    # 计算距离开盘/收盘的分钟数
    now_minutes = now_est.hour * 60 + now_est.minute
    open_minutes = market_open.hour * 60 + market_open.minute
    close_minutes = market_close.hour * 60 + market_close.minute
    
    minutes_from_open = now_minutes - open_minutes if is_market_hours else 0
    minutes_to_close = close_minutes - now_minutes if is_market_hours else 0
    
    return {
        'is_market_hours': is_market_hours,
        'minutes_from_open': minutes_from_open,
        'minutes_to_close': minutes_to_close,
        'current_time_est': now_est.strftime('%H:%M:%S EST')
    }


def should_trade_with_delayed_data(config: DelayedDataConfig) -> tuple[bool, str]:
    """检查是否适合使用延迟数据进行交易"""
    
    if not config.enabled:
        return False, "延迟数据交易已禁用"
    
    market_info = get_market_session_info()
    
    if not market_info['is_market_hours']:
        return False, "当前非交易时间"
    
    # 避开开盘前后时段
    if market_info['minutes_from_open'] < config.avoid_market_open_minutes:
        return False, f"开盘后{config.avoid_market_open_minutes}分钟内避免交易"
    
    # 避开收盘前时段
    if market_info['minutes_to_close'] < config.avoid_market_close_minutes:
        return False, f"收盘前{config.avoid_market_close_minutes}分钟内避免交易"
    
    return True, f"适合延迟数据交易 ({market_info['current_time_est']})"


# 默认配置实例
DEFAULT_DELAYED_CONFIG = DelayedDataConfig()

if __name__ == "__main__":
    # 测试配置
    config = DelayedDataConfig()
    print("延迟数据交易配置:")
    print(f"  数据延迟: {config.data_delay_minutes}分钟")
    print(f"  仓位缩减: {config.position_size_reduction*100}%")
    print(f"  最小置信度: {config.min_confidence_threshold}")
    
    # 测试市场时间
    can_trade, reason = should_trade_with_delayed_data(config)
    print(f"\n当前交易状态: {'✅ 可以交易' if can_trade else '❌ 不建议交易'}")
    print(f"原因: {reason}")
    
    # 显示市场信息
    market_info = get_market_session_info()
    print(f"\n市场信息:")
    for key, value in market_info.items():
        print(f"  {key}: {value}")