#!/usr/bin/env python3
"""
🔥 P0级别修复：统一时区处理系统
=======================================

确保整个交易系统使用统一的时区处理，避免时间不对齐导致的交易错误。
支持夏令时自动转换、多市场时区协调、时间戳标准化等功能。
"""

import pytz
import logging
from datetime import datetime, timezone, time, timedelta
from typing import Dict, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
import calendar

logger = logging.getLogger(__name__)


class MarketTimezone(Enum):
    """主要市场时区"""
    US_EASTERN = "US/Eastern"      # 美东时间（纽交所、纳斯达克）
    US_CENTRAL = "US/Central"      # 美中时间（芝加哥）
    US_PACIFIC = "US/Pacific"      # 美西时间
    LONDON = "Europe/London"       # 伦敦时间（LSE）
    FRANKFURT = "Europe/Berlin"    # 法兰克福时间
    TOKYO = "Asia/Tokyo"           # 东京时间（TSE）
    HONG_KONG = "Asia/Hong_Kong"   # 香港时间（HKEX）
    SHANGHAI = "Asia/Shanghai"     # 上海时间（SSE/SZSE）
    UTC = "UTC"                    # 协调世界时


@dataclass
class MarketSession:
    """市场交易时段"""
    market_name: str
    timezone_name: str
    open_time: time
    close_time: time
    pre_market_start: Optional[time] = None
    post_market_end: Optional[time] = None


class UnifiedTimezoneHandler:
    """统一时区处理器"""
    
    # 标准市场时段配置
    MARKET_SESSIONS = {
        'NYSE': MarketSession(
            market_name='NYSE',
            timezone_name=MarketTimezone.US_EASTERN.value,
            open_time=time(9, 30),
            close_time=time(16, 0),
            pre_market_start=time(4, 0),
            post_market_end=time(20, 0)
        ),
        'NASDAQ': MarketSession(
            market_name='NASDAQ', 
            timezone_name=MarketTimezone.US_EASTERN.value,
            open_time=time(9, 30),
            close_time=time(16, 0),
            pre_market_start=time(4, 0),
            post_market_end=time(20, 0)
        ),
        'LSE': MarketSession(
            market_name='LSE',
            timezone_name=MarketTimezone.LONDON.value,
            open_time=time(8, 0),
            close_time=time(16, 30)
        ),
        'TSE': MarketSession(
            market_name='TSE',
            timezone_name=MarketTimezone.TOKYO.value,
            open_time=time(9, 0),
            close_time=time(15, 0)
        )
    }
    
    def __init__(self, primary_market: str = 'NYSE'):
        """
        初始化时区处理器
        
        Args:
            primary_market: 主要交易市场，用于确定默认时区
        """
        self.primary_market = primary_market
        self.primary_timezone = self._get_market_timezone(primary_market)
        
        # 时区缓存
        self._timezone_cache: Dict[str, pytz.BaseTzInfo] = {}
        self._preload_timezones()
        
        logger.info(f"Timezone handler initialized - Primary market: {primary_market}")
    
    def _preload_timezones(self):
        """预加载常用时区"""
        common_timezones = [tz.value for tz in MarketTimezone]
        for tz_name in common_timezones:
            try:
                self._timezone_cache[tz_name] = pytz.timezone(tz_name)
            except Exception as e:
                logger.warning(f"Failed to load timezone {tz_name}: {e}")
    
    def _get_market_timezone(self, market: str) -> pytz.BaseTzInfo:
        """获取市场对应的时区"""
        if market in self.MARKET_SESSIONS:
            tz_name = self.MARKET_SESSIONS[market].timezone_name
            return self.get_timezone(tz_name)
        return self.get_timezone(MarketTimezone.UTC.value)
    
    def get_timezone(self, timezone_name: str) -> pytz.BaseTzInfo:
        """获取时区对象（带缓存）"""
        if timezone_name not in self._timezone_cache:
            try:
                self._timezone_cache[timezone_name] = pytz.timezone(timezone_name)
            except Exception as e:
                logger.error(f"Invalid timezone: {timezone_name}, using UTC")
                self._timezone_cache[timezone_name] = pytz.UTC
        
        return self._timezone_cache[timezone_name]
    
    def now_utc(self) -> datetime:
        """获取当前UTC时间（标准化）"""
        return datetime.now(timezone.utc)
    
    def now_market(self, market: str = None) -> datetime:
        """获取指定市场的当前时间"""
        market = market or self.primary_market
        market_tz = self._get_market_timezone(market)
        return datetime.now(market_tz)
    
    def to_utc(self, dt: datetime, from_timezone: str = None) -> datetime:
        """将本地时间转换为UTC时间"""
        if dt.tzinfo is None:
            # 无时区信息的datetime，使用指定时区
            if from_timezone:
                local_tz = self.get_timezone(from_timezone)
            else:
                local_tz = self.primary_timezone
            dt = local_tz.localize(dt)
        
        return dt.astimezone(timezone.utc)
    
    def from_utc(self, utc_dt: datetime, to_timezone: str) -> datetime:
        """将UTC时间转换为指定时区时间"""
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        
        target_tz = self.get_timezone(to_timezone)
        return utc_dt.astimezone(target_tz)
    
    def market_to_market(self, dt: datetime, from_market: str, to_market: str) -> datetime:
        """市场间时间转换"""
        from_tz = self._get_market_timezone(from_market)
        to_tz = self._get_market_timezone(to_market)
        
        # 确保输入时间有时区信息
        if dt.tzinfo is None:
            dt = from_tz.localize(dt)
        
        return dt.astimezone(to_tz)
    
    def standardize_timestamp(self, timestamp: Union[datetime, int, float, str], 
                             source_timezone: str = None) -> datetime:
        """标准化时间戳为UTC datetime"""
        if isinstance(timestamp, datetime):
            return self.to_utc(timestamp, source_timezone)
        
        elif isinstance(timestamp, (int, float)):
            # Unix时间戳
            if timestamp > 1e10:  # 毫秒时间戳
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        elif isinstance(timestamp, str):
            # 字符串时间戳
            try:
                # 尝试ISO格式
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return self.to_utc(dt, source_timezone)
                else:
                    # 尝试其他常见格式
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    return self.to_utc(dt, source_timezone)
            except ValueError as e:
                logger.error(f"Failed to parse timestamp: {timestamp}, error: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    def is_market_open(self, market: str, check_time: datetime = None) -> bool:
        """检查指定市场是否开盘"""
        if market not in self.MARKET_SESSIONS:
            logger.warning(f"Unknown market: {market}")
            return False
        
        session = self.MARKET_SESSIONS[market]
        check_time = check_time or self.now_market(market)
        
        # 转换为市场时区
        if check_time.tzinfo is None:
            market_tz = self.get_timezone(session.timezone_name)
            check_time = market_tz.localize(check_time)
        else:
            market_tz = self.get_timezone(session.timezone_name)
            check_time = check_time.astimezone(market_tz)
        
        # 检查是否为工作日
        if check_time.weekday() >= 5:  # 周六=5, 周日=6
            return False
        
        # 检查时间范围
        current_time = check_time.time()
        return session.open_time <= current_time <= session.close_time
    
    def get_market_close_time(self, market: str, date: datetime = None) -> datetime:
        """获取指定日期的市场收盘时间"""
        if market not in self.MARKET_SESSIONS:
            raise ValueError(f"Unknown market: {market}")
        
        session = self.MARKET_SESSIONS[market]
        date = date or self.now_market(market)
        
        # 构建收盘时间
        market_tz = self.get_timezone(session.timezone_name)
        close_dt = market_tz.localize(
            datetime.combine(date.date(), session.close_time)
        )
        
        return close_dt
    
    def seconds_until_market_close(self, market: str) -> int:
        """距离市场收盘还有多少秒"""
        close_time = self.get_market_close_time(market)
        current_time = self.now_market(market)
        
        if close_time < current_time:
            # 已经收盘，返回到明日收盘的秒数
            next_close = close_time + timedelta(days=1)
            return int((next_close - current_time).total_seconds())
        else:
            return int((close_time - current_time).total_seconds())
    
    def align_trading_timestamps(self, timestamps: List[datetime], 
                                target_timezone: str = "UTC") -> List[datetime]:
        """对齐交易时间戳到统一时区"""
        aligned = []
        target_tz = self.get_timezone(target_timezone)
        
        for ts in timestamps:
            if ts.tzinfo is None:
                # 假设无时区的时间戳是主市场时区
                ts = self.primary_timezone.localize(ts)
            
            aligned_ts = ts.astimezone(target_tz)
            aligned.append(aligned_ts)
        
        return aligned
    
    def get_trading_calendar(self, market: str, year: int) -> List[datetime]:
        """获取指定市场的交易日历（简化版）"""
        if market not in self.MARKET_SESSIONS:
            raise ValueError(f"Unknown market: {market}")
        
        session = self.MARKET_SESSIONS[market]
        market_tz = self.get_timezone(session.timezone_name)
        
        trading_days = []
        
        # 生成一年中的所有工作日
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # 周一到周五
                # 简化处理，不考虑具体假期
                trading_days.append(market_tz.localize(current))
            current += timedelta(days=1)
        
        return trading_days
    
    def validate_timestamp_alignment(self, timestamps: List[datetime], 
                                   tolerance_seconds: int = 60) -> Dict[str, Any]:
        """验证时间戳对齐情况"""
        if not timestamps:
            return {'valid': True, 'issues': []}
        
        issues = []
        utc_timestamps = []
        
        for i, ts in enumerate(timestamps):
            try:
                utc_ts = self.to_utc(ts)
                utc_timestamps.append(utc_ts)
            except Exception as e:
                issues.append(f"Timestamp {i}: conversion error - {e}")
        
        # 检查时间间隔一致性
        if len(utc_timestamps) > 1:
            intervals = []
            for i in range(1, len(utc_timestamps)):
                interval = (utc_timestamps[i] - utc_timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            # 检查间隔变化
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                for i, interval in enumerate(intervals):
                    if abs(interval - avg_interval) > tolerance_seconds:
                        issues.append(f"Irregular interval at position {i+1}: {interval}s vs avg {avg_interval:.1f}s")
        
        return {
            'valid': len(issues) == 0,
            'total_timestamps': len(timestamps),
            'converted_timestamps': len(utc_timestamps),
            'issues': issues,
            'first_timestamp_utc': utc_timestamps[0].isoformat() if utc_timestamps else None,
            'last_timestamp_utc': utc_timestamps[-1].isoformat() if utc_timestamps else None
        }


# 全局实例
_global_timezone_handler: Optional[UnifiedTimezoneHandler] = None


def get_timezone_handler() -> UnifiedTimezoneHandler:
    """获取全局时区处理器"""
    global _global_timezone_handler
    if _global_timezone_handler is None:
        _global_timezone_handler = UnifiedTimezoneHandler()
    return _global_timezone_handler


# 便捷函数
def now_utc() -> datetime:
    """获取当前UTC时间"""
    return get_timezone_handler().now_utc()


def now_market(market: str = 'NYSE') -> datetime:
    """获取市场当前时间"""
    return get_timezone_handler().now_market(market)


def to_utc(dt: datetime, from_timezone: str = None) -> datetime:
    """转换为UTC时间"""
    return get_timezone_handler().to_utc(dt, from_timezone)


def standardize_timestamp(timestamp: Union[datetime, int, float, str], 
                         source_timezone: str = None) -> datetime:
    """标准化时间戳"""
    return get_timezone_handler().standardize_timestamp(timestamp, source_timezone)


if __name__ == "__main__":
    # 测试时区处理
    logging.basicConfig(level=logging.INFO)
    
    handler = UnifiedTimezoneHandler()
    
    # 测试当前时间
    print(f"UTC Now: {handler.now_utc()}")
    print(f"NYSE Now: {handler.now_market('NYSE')}")
    print(f"LSE Now: {handler.now_market('LSE')}")
    
    # 测试市场状态
    print(f"NYSE Open: {handler.is_market_open('NYSE')}")
    print(f"LSE Open: {handler.is_market_open('LSE')}")
    
    # 测试时间转换
    local_time = datetime(2024, 1, 15, 14, 30)  # 下午2:30
    utc_time = handler.to_utc(local_time, 'US/Eastern')
    print(f"EST 2:30 PM -> UTC: {utc_time}")
    
    # 测试时间戳标准化
    timestamps = [
        "2024-01-15T14:30:00",
        1705327800,  # Unix timestamp
        datetime(2024, 1, 15, 14, 30)
    ]
    
    for ts in timestamps:
        standardized = handler.standardize_timestamp(ts, 'US/Eastern')
        print(f"{ts} -> {standardized}")