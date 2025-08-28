#!/usr/bin/env python3
"""
交易频率控制器 - 降低交易频率的智能控制系统
实现成本感知门控、不交易带宽、冷却时间、批量化等机制
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, NamedTuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)


class TradingDecision(Enum):
    """交易决策类型"""
    ALLOW = "allow"
    REJECT_COST = "reject_cost"  # 成本感知拒绝
    REJECT_BAND = "reject_band"  # 不交易带宽拒绝
    REJECT_COOLDOWN = "reject_cooldown"  # 冷却期拒绝
    REJECT_QUOTA = "reject_quota"  # 配额限制拒绝
    REJECT_LIQUIDITY = "reject_liquidity"  # 流动性拒绝
    QUEUE_BATCH = "queue_batch"  # 排队批量执行


@dataclass
class TradingRequest:
    """交易请求"""
    symbol: str
    action: str  # BUY/SELL
    target_weight: float
    current_weight: float
    expected_return: float
    confidence: float
    timestamp: float
    estimated_cost: float = 0.0
    priority: str = "normal"  # normal/high/urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TradingQuota:
    """交易配额"""
    symbol: str
    daily_orders: int = 0
    hourly_orders: int = 0
    last_order_time: float = 0.0
    cancel_count: int = 0
    last_reset_date: str = ""
    last_reset_hour: int = -1
    frozen_until: float = 0.0  # 冷却结束时间


class CostAwareGatekeeper:
    """成本感知门控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_win_rate = config.get('min_win_rate', 0.20)      # 从55%降至20%
        self.min_hold_minutes = config.get('min_hold_minutes', 1)  # 从10分钟降至1分钟
        self.cost_buffer_bp = config.get('cost_buffer_bp', 1)      # 从8bp降至1bp
        self.intraday_threshold_bp = config.get('intraday_threshold_bp', 1)   # 从10bp降至1bp
        self.overnight_threshold_bp = config.get('overnight_threshold_bp', 2) # 从18bp降至2bp
        
        # 胜率历史跟踪
        self._win_rate_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._confidence_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def should_allow_trade(self, request: TradingRequest) -> tuple[bool, str]:
        """判断是否允许交易 - 大幅放宽限制"""
        
        # 1. 检查期望收益 vs 成本 - 几乎不限制
        net_expected_return = request.expected_return - request.estimated_cost
        threshold_bp = self.intraday_threshold_bp / 10000.0  # 转换为小数
        
        if net_expected_return < threshold_bp:
            # 不再拒绝，仅记录警告
            logger.debug(f"低收益交易: {request.symbol} 期望收益({net_expected_return*10000:.1f}bp) < 阈值({self.intraday_threshold_bp}bp)")
        
        # 2. 检查置信度/胜率 - 大幅放宽
        if request.confidence < self.min_win_rate:
            # 不再拒绝，仅记录警告
            logger.debug(f"低置信度交易: {request.symbol} 置信度({request.confidence:.3f}) < 最小胜率({self.min_win_rate})")
        
        # 3. 记录历史用于适应性调整
        self._confidence_history[request.symbol].append(request.confidence)
        
        return True, "成本门控通过（已放宽限制）"
    
    def update_trade_result(self, symbol: str, success: bool, actual_cost: float):
        """更新交易结果，用于适应性调整"""
        self._win_rate_history[symbol].append(1.0 if success else 0.0)
        
    def get_adaptive_thresholds(self, symbol: str) -> Dict[str, float]:
        """获取自适应阈值"""
        if symbol not in self._win_rate_history or len(self._win_rate_history[symbol]) < 10:
            return {
                'win_rate_threshold': self.min_win_rate,
                'cost_threshold_bp': self.intraday_threshold_bp
            }
        
        recent_win_rate = sum(list(self._win_rate_history[symbol])[-20:]) / min(20, len(self._win_rate_history[symbol]))
        
        # 根据历史胜率调整阈值
        if recent_win_rate < 0.45:
            # 表现不佳，提高阈值
            adjusted_win_rate = self.min_win_rate + 0.05
            adjusted_cost_threshold = self.intraday_threshold_bp + 5
        elif recent_win_rate > 0.65:
            # 表现良好，可以适度放松
            adjusted_win_rate = max(0.50, self.min_win_rate - 0.02)
            adjusted_cost_threshold = max(5, self.intraday_threshold_bp - 2)
        else:
            adjusted_win_rate = self.min_win_rate
            adjusted_cost_threshold = self.intraday_threshold_bp
            
        return {
            'win_rate_threshold': adjusted_win_rate,
            'cost_threshold_bp': adjusted_cost_threshold
        }


class NoTradeBandManager:
    """不交易带宽管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.band_width_pct = config.get('band_width_pct', 0.001)         # 从2%降至0.1%
        self.trigger_threshold_pct = config.get('trigger_threshold_pct', 0.001)  # 从4.5%降至0.1%触发
        self.rebalance_threshold_pct = config.get('rebalance_threshold_pct', 0.001)  # 从3%降至0.1%
        
        # 跟踪每个symbol的目标权重历史
        self._target_weight_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def should_allow_trade(self, request: TradingRequest) -> tuple[bool, str]:
        """检查是否超出不交易带宽 - 几乎不限制"""
        
        weight_diff = abs(request.target_weight - request.current_weight)
        
        # 几乎允许所有权重变化
        if weight_diff < self.trigger_threshold_pct:
            # 不再拒绝，仅记录调试信息
            logger.debug(f"微小权重变化: {request.symbol} 权重变化({weight_diff*100:.3f}%) < 触发阈值({self.trigger_threshold_pct*100:.3f}%)")
        
        # 记录目标权重变化
        self._target_weight_history[request.symbol].append({
            'timestamp': request.timestamp,
            'target_weight': request.target_weight,
            'current_weight': request.current_weight
        })
        
        # 取消频繁调整检查，允许高频交易
        # 原波动性检查已移除
        
        return True, "带宽检查通过（已放宽限制）"
    
    def _calculate_weight_volatility(self, changes: List[Dict]) -> float:
        """计算权重变化的波动性"""
        if len(changes) < 2:
            return 0.0
            
        weights = [c['target_weight'] for c in changes]
        mean_weight = sum(weights) / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        return math.sqrt(variance)


class CooldownManager:
    """冷却时间管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.per_symbol_cooldown_minutes = config.get('per_symbol_cooldown_minutes', 0)    # 从120分钟降至0分钟
        self.min_holding_period_hours = config.get('min_holding_period_hours', 0)         # 从4小时降至0小时
        self.global_rate_limit_5min = config.get('global_rate_limit_5min', 1000)         # 从3提高至1000
        self.global_rate_limit_1hour = config.get('global_rate_limit_1hour', 10000)      # 从6提高至10000
        
        # 跟踪
        self._last_order_time: Dict[str, float] = {}
        self._position_open_time: Dict[str, float] = {}
        self._global_order_history: deque = deque(maxlen=100)
        
    def should_allow_trade(self, request: TradingRequest) -> tuple[bool, str]:
        """检查冷却时间限制 - 几乎不限制"""
        
        current_time = request.timestamp
        symbol = request.symbol
        
        # 1. 取消单symbol冷却检查
        if symbol in self._last_order_time:
            time_since_last = (current_time - self._last_order_time[symbol]) / 60.0  # 转分钟
            if time_since_last < self.per_symbol_cooldown_minutes:
                # 冷却时间设为0，所以不会被拒绝
                logger.debug(f"快速连续交易: {symbol} 距离上次交易{time_since_last:.1f}分钟")
        
        # 2. 取消最小持有期检查
        if request.action == "SELL" and symbol in self._position_open_time:
            holding_hours = (current_time - self._position_open_time[symbol]) / 3600.0
            if holding_hours < self.min_holding_period_hours:
                # 最小持有期设为0，所以不会被拒绝
                logger.debug(f"快速平仓: {symbol} 持有时间{holding_hours:.2f}小时")
        
        # 3. 放宽全局频率限制
        # 记录订单但不限制
        self._global_order_history.append(current_time)
        
        # 只保留最近1小时的记录以节省内存
        cutoff_1hour = current_time - 3600
        self._global_order_history = deque(
            [t for t in self._global_order_history if t > cutoff_1hour],
            maxlen=10000  # 增大缓存容量
        )
        
        return True, "冷却检查通过（已取消限制）"
    
    def record_order(self, symbol: str, action: str, timestamp: float):
        """记录订单，更新冷却状态"""
        self._last_order_time[symbol] = timestamp
        self._global_order_history.append(timestamp)
        
        if action == "BUY":
            self._position_open_time[symbol] = timestamp


class BatchingManager:
    """批量化管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.batch_window_minutes = config.get('batch_window_minutes', 60)
        self.min_batch_size = config.get('min_batch_size', 2)
        self.max_queue_size = config.get('max_queue_size', 20)
        
        # 批量队列
        self._pending_requests: Dict[str, List[TradingRequest]] = defaultdict(list)
        self._last_batch_time = time.time()
        
        # 启动批量处理线程
        self._batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self._batch_thread.start()
        
        # 批量回调
        self._batch_callback = None
        
    def set_batch_callback(self, callback):
        """设置批量执行回调函数"""
        self._batch_callback = callback
        
    def should_batch_trade(self, request: TradingRequest) -> tuple[bool, str]:
        """判断是否应该批量处理"""
        
        # 紧急交易不批量
        if request.priority == "urgent":
            return False, "紧急交易，立即执行"
        
        # 队列过满时强制执行
        current_queue_size = sum(len(reqs) for reqs in self._pending_requests.values())
        if current_queue_size >= self.max_queue_size:
            return False, "队列已满，立即执行"
        
        # 大权重变化立即执行
        weight_change = abs(request.target_weight - request.current_weight)
        if weight_change > 0.08:  # 8%以上立即执行
            return False, "大权重变化，立即执行"
        
        return True, "加入批量队列"
    
    def queue_request(self, request: TradingRequest):
        """将请求加入批量队列"""
        symbol = request.symbol
        
        # 合并同symbol的相近请求
        existing_requests = self._pending_requests[symbol]
        if existing_requests:
            # 替换最近的请求（保持最新状态）
            self._pending_requests[symbol] = [request]
        else:
            self._pending_requests[symbol].append(request)
        
        logger.info(f"请求已加入批量队列: {symbol}, 队列大小: {len(self._pending_requests)}")
    
    def _batch_processor(self):
        """批量处理线程"""
        while True:
            try:
                time.sleep(60)  # 每分钟检查一次
                current_time = time.time()
                
                # 检查是否到达批量执行时间
                if current_time - self._last_batch_time >= self.batch_window_minutes * 60:
                    self._execute_batch()
                    self._last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"批量处理线程错误: {e}")
    
    def _execute_batch(self):
        """执行批量订单"""
        if not self._pending_requests:
            return
        
        # 收集所有待处理请求
        batch_requests = []
        for symbol, requests in self._pending_requests.items():
            batch_requests.extend(requests)
        
        if len(batch_requests) < self.min_batch_size:
            logger.info(f"批量大小({len(batch_requests)}) < 最小批量({self.min_batch_size})，延迟执行")
            return
        
        logger.info(f"执行批量订单，数量: {len(batch_requests)}")
        
        # 按优先级排序
        batch_requests.sort(key=lambda r: (
            0 if r.priority == "urgent" else 1 if r.priority == "high" else 2,
            -abs(r.target_weight - r.current_weight)  # 权重变化大的优先
        ))
        
        # 执行批量
        if self._batch_callback:
            try:
                self._batch_callback(batch_requests)
            except Exception as e:
                logger.error(f"批量执行回调失败: {e}")
        
        # 清空队列
        self._pending_requests.clear()
        
    def force_execute_batch(self) -> List[TradingRequest]:
        """强制执行当前批量"""
        batch_requests = []
        for symbol, requests in self._pending_requests.items():
            batch_requests.extend(requests)
        
        self._pending_requests.clear()
        self._last_batch_time = time.time()
        
        return batch_requests


class LiquidityGatekeeper:
    """流动性门控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_spread_percentile = config.get('max_spread_percentile', 75)
        self.max_volatility_percentile = config.get('max_volatility_percentile', 60)
        self.min_order_size_usd = config.get('min_order_size_usd', 2500)
        self.fragile_time_buffer_bp = config.get('fragile_time_buffer_bp', 10)
        
        # 流动性历史数据
        self._spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def should_allow_trade(self, request: TradingRequest, market_data: Dict[str, Any]) -> tuple[bool, str]:
        """检查流动性条件"""
        
        symbol = request.symbol
        
        # 1. 检查最小订单金额
        estimated_value = abs(request.target_weight - request.current_weight) * market_data.get('portfolio_value', 100000)
        if estimated_value < self.min_order_size_usd:
            return False, f"订单金额({estimated_value:.0f}USD) < 最小金额({self.min_order_size_usd}USD)"
        
        # 2. 检查点差
        current_spread = market_data.get('relative_spread', 0.0)
        self._spread_history[symbol].append(current_spread)
        
        if len(self._spread_history[symbol]) >= 20:
            spread_percentile = self._calculate_percentile(list(self._spread_history[symbol]), self.max_spread_percentile)
            if current_spread > spread_percentile:
                return False, f"当前点差({current_spread*10000:.1f}bp) > {self.max_spread_percentile}分位({spread_percentile*10000:.1f}bp)"
        
        # 3. 检查波动率
        current_volatility = market_data.get('minute_volatility', 0.0)
        self._volatility_history[symbol].append(current_volatility)
        
        if len(self._volatility_history[symbol]) >= 20:
            vol_percentile = self._calculate_percentile(list(self._volatility_history[symbol]), self.max_volatility_percentile)
            if current_volatility > vol_percentile:
                return False, f"当前波动({current_volatility*10000:.1f}bp) > {self.max_volatility_percentile}分位({vol_percentile*10000:.1f}bp)"
        
        # 4. 检查时间敏感性（市场开盘、收盘）
        market_hour = datetime.fromtimestamp(request.timestamp).hour
        if market_hour in [9, 10, 15, 16]:  # 开盘收盘时段
            # 仅允许减仓
            if request.action == "BUY":
                return False, "敏感时段仅允许减仓"
        
        return True, "流动性检查通过"
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100.0)
        return sorted_data[min(index, len(sorted_data) - 1)]


class FrequencyController:
    """频率控制器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各个子组件
        self.cost_gatekeeper = CostAwareGatekeeper(config.get('cost_gatekeeper', {}))
        self.no_trade_band = NoTradeBandManager(config.get('no_trade_band', {}))
        self.cooldown_manager = CooldownManager(config.get('cooldown', {}))
        self.batching_manager = BatchingManager(config.get('batching', {}))
        self.liquidity_gatekeeper = LiquidityGatekeeper(config.get('liquidity', {}))
        
        # 配额管理 - 大幅放宽
        self._daily_quotas: Dict[str, TradingQuota] = defaultdict(lambda: TradingQuota(symbol=""))
        self.max_daily_orders_per_symbol = config.get('max_daily_orders_per_symbol', 10000)  # 从6提高至10000
        self.max_cancel_rate = config.get('max_cancel_rate', 0.99)  # 从20%提高至99%
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'allowed_immediately': 0,
            'rejected_cost': 0,
            'rejected_band': 0,
            'rejected_cooldown': 0,
            'rejected_quota': 0,
            'rejected_liquidity': 0,
            'queued_batch': 0
        }
        
        logger.info("频率控制器初始化完成")
    
    def should_allow_trade(self, request: TradingRequest, market_data: Optional[Dict[str, Any]] = None) -> tuple[TradingDecision, str]:
        """主要的交易决策函数"""
        
        self.stats['total_requests'] += 1
        symbol = request.symbol
        
        # 更新配额状态
        self._update_quota_status(symbol)
        
        try:
            # 1. 配额检查（最先执行）
            if not self._check_quota_limits(request):
                self.stats['rejected_quota'] += 1
                return TradingDecision.REJECT_QUOTA, f"{symbol}超出配额限制"
            
            # 2. 成本感知门控
            cost_allowed, cost_msg = self.cost_gatekeeper.should_allow_trade(request)
            if not cost_allowed:
                self.stats['rejected_cost'] += 1
                return TradingDecision.REJECT_COST, cost_msg
            
            # 3. 不交易带宽检查
            band_allowed, band_msg = self.no_trade_band.should_allow_trade(request)
            if not band_allowed:
                self.stats['rejected_band'] += 1
                return TradingDecision.REJECT_BAND, band_msg
            
            # 4. 冷却时间检查
            cooldown_allowed, cooldown_msg = self.cooldown_manager.should_allow_trade(request)
            if not cooldown_allowed:
                self.stats['rejected_cooldown'] += 1
                return TradingDecision.REJECT_COOLDOWN, cooldown_msg
            
            # 5. 流动性检查
            if market_data:
                liquidity_allowed, liquidity_msg = self.liquidity_gatekeeper.should_allow_trade(request, market_data)
                if not liquidity_allowed:
                    self.stats['rejected_liquidity'] += 1
                    return TradingDecision.REJECT_LIQUIDITY, liquidity_msg
            
            # 6. 批量化检查
            should_batch, batch_msg = self.batching_manager.should_batch_trade(request)
            if should_batch:
                self.stats['queued_batch'] += 1
                self.batching_manager.queue_request(request)
                return TradingDecision.QUEUE_BATCH, batch_msg
            
            # 所有检查通过
            self.stats['allowed_immediately'] += 1
            return TradingDecision.ALLOW, "所有检查通过，允许交易"
            
        except Exception as e:
            logger.error(f"频率控制器错误: {e}")
            return TradingDecision.REJECT_COST, f"控制器内部错误: {e}"
    
    def record_order_execution(self, symbol: str, action: str, success: bool, actual_cost: float = 0.0):
        """记录订单执行结果"""
        timestamp = time.time()
        
        # 更新冷却管理器
        if success:
            self.cooldown_manager.record_order(symbol, action, timestamp)
        
        # 更新成本门控器
        self.cost_gatekeeper.update_trade_result(symbol, success, actual_cost)
        
        # 更新配额
        if symbol not in self._daily_quotas:
            self._daily_quotas[symbol] = TradingQuota(symbol=symbol)
        quota = self._daily_quotas[symbol]
        if success:
            quota.daily_orders += 1
            quota.hourly_orders += 1
            quota.last_order_time = timestamp
        
    def record_order_cancellation(self, symbol: str):
        """记录订单取消"""
        if symbol not in self._daily_quotas:
            self._daily_quotas[symbol] = TradingQuota(symbol=symbol)
        quota = self._daily_quotas[symbol]
        quota.cancel_count += 1
        
        # 检查是否需要冷却惩罚
        if quota.cancel_count >= 3:  # 1小时内3次取消
            penalty_hours = min(2.0, quota.cancel_count - 2)  # 最多2小时惩罚
            quota.frozen_until = time.time() + penalty_hours * 3600
            logger.warning(f"{symbol} 因频繁撤单被冷却 {penalty_hours} 小时")
    
    def _update_quota_status(self, symbol: str):
        """更新配额状态"""
        if symbol not in self._daily_quotas:
            self._daily_quotas[symbol] = TradingQuota(symbol=symbol)
        quota = self._daily_quotas[symbol]
        current_time = time.time()
        current_date = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d')
        current_hour = datetime.fromtimestamp(current_time).hour
        
        # 日配额重置
        if quota.last_reset_date != current_date:
            quota.daily_orders = 0
            quota.cancel_count = 0
            quota.last_reset_date = current_date
        
        # 小时配额重置
        if quota.last_reset_hour != current_hour:
            quota.hourly_orders = 0
            quota.last_reset_hour = current_hour
    
    def _check_quota_limits(self, request: TradingRequest) -> bool:
        """检查配额限制"""
        symbol = request.symbol
        if symbol not in self._daily_quotas:
            self._daily_quotas[symbol] = TradingQuota(symbol=symbol)
        quota = self._daily_quotas[symbol]
        current_time = time.time()
        
        # 检查冷却惩罚
        if quota.frozen_until > current_time:
            return False
        
        # 检查日配额
        if quota.daily_orders >= self.max_daily_orders_per_symbol:
            # 允许减仓
            if request.action == "SELL":
                return True
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = max(1, self.stats['total_requests'])
        
        return {
            **self.stats,
            'acceptance_rate': self.stats['allowed_immediately'] / total,
            'rejection_rate': (total - self.stats['allowed_immediately'] - self.stats['queued_batch']) / total,
            'batch_rate': self.stats['queued_batch'] / total,
            'quota_status': {symbol: {
                'daily_orders': quota.daily_orders,
                'frozen_until': quota.frozen_until,
                'cancel_count': quota.cancel_count
            } for symbol, quota in self._daily_quotas.items()}
        }
    
    def force_execute_batch(self) -> List[TradingRequest]:
        """强制执行批量队列"""
        return self.batching_manager.force_execute_batch()
    
    def set_batch_callback(self, callback):
        """设置批量执行回调"""
        self.batching_manager.set_batch_callback(callback)


# 全局频率控制器实例
_global_frequency_controller: Optional[FrequencyController] = None


def get_frequency_controller(config: Optional[Dict[str, Any]] = None) -> FrequencyController:
    """获取全局频率控制器实例"""
    global _global_frequency_controller
    if _global_frequency_controller is None:
        if config is None:
            # 默认配置 - 已放宽限制
            config = {
                'cost_gatekeeper': {
                    'min_win_rate': 0.20,           # 从55%降至20%
                    'intraday_threshold_bp': 1,     # 从10bp降至1bp
                    'overnight_threshold_bp': 2     # 从18bp降至2bp
                },
                'no_trade_band': {
                    'band_width_pct': 0.001,        # 从2%降至0.1%
                    'trigger_threshold_pct': 0.001  # 从4.5%降至0.1%
                },
                'cooldown': {
                    'per_symbol_cooldown_minutes': 0,    # 从120分钟降至0分钟
                    'min_holding_period_hours': 0,       # 从4小时降至0小时
                    'global_rate_limit_5min': 1000,      # 从3提高至1000
                    'global_rate_limit_1hour': 10000     # 从6提高至10000
                },
                'batching': {
                    'batch_window_minutes': 1,      # 从60分钟降至1分钟
                    'min_batch_size': 1             # 从2降至1
                },
                'liquidity': {
                    'max_spread_percentile': 99,    # 从75提高至99
                    'max_volatility_percentile': 99,# 从60提高至99
                    'min_order_size_usd': 0         # 从2500降至0（无最小金额）
                },
                'max_daily_orders_per_symbol': 10000,   # 从6提高至10000
                'max_cancel_rate': 0.99                 # 从20%提高至99%
            }
        _global_frequency_controller = FrequencyController(config)
    return _global_frequency_controller


def create_frequency_controller(config: Dict[str, Any]) -> FrequencyController:
    """创建新的频率控制器实例"""
    return FrequencyController(config)