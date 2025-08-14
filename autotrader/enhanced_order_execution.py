#!/usr/bin/env python3
"""
RiskRewardBalancer 集成器
innoreal-timemarket data（延迟环境）下，as模型信号提供保守但can落地order placement决策：
- 流动性筛选
- 信号门槛
- 动态limit（基at昨收/ATR/30bps 带）
- 交易决策（APPROVE/DEGRADE/REJECT）
- 组合权重分配（Top-N boost + 单票上限）
- 节流andorder placement尺寸控制

注：market data源使use Polygon（延迟），交易order placement走 IBKR。
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math


@dataclass
class RRConfig:
    enabled: bool = False
    min_price: float = 5.0
    min_adv_usd: float = 500_000
    max_median_spread_bps: float = 50.0
    min_alpha_bps: float = 50.0
    min_alpha_vs_15m_sigma: float = 2.0
    max_weight: float = 0.03
    top_n_boost: int = 10
    top_n_boost_multiplier: float = 1.25
    max_child_adv_pct: float = 0.10
    max_child_book_pct: float = 0.10
    min_child_shares: int = 50


@dataclass
class Signal:
    symbol: str
    side: str  # BUY / SELL
    expected_alpha_bps: float
    model_price: Optional[float] = None
    confidence: float = 1.0


@dataclass
class Metrics:
    prev_close: float
    atr_14: Optional[float]
    adv_usd_20: Optional[float]
    median_spread_bps_20: Optional[float]
    sigma_15m: Optional[float] = None


@dataclass
class Quote:
    last: Optional[float]
    tickSize: float
    source: str  # DELAYED / REALTIME


@dataclass
class Decision:
    action: str  # APPROVE / DEGRADE / REJECT
    reason: str
    shrink_to_pct: Optional[float] = None


class RiskRewardBalancer:
    def __init__(self, cfg: RRConfig):
        self.cfg = cfg

    # ---------- Filters ----------
    def passes_static_liquidity_filters(self, m: Metrics) -> bool:
        if m.prev_close is None or m.prev_close < self.cfg.min_price:
            return False
        if m.adv_usd_20 is None or m.adv_usd_20 < self.cfg.min_adv_usd:
            return False
        if m.median_spread_bps_20 is not None and m.median_spread_bps_20 > self.cfg.max_median_spread_bps:
            return False
        return True

    def passes_signal_thresholds(self, s: Signal, m: Metrics) -> bool:
        if s.expected_alpha_bps is None or s.expected_alpha_bps < self.cfg.min_alpha_bps:
            return False
        if m.sigma_15m is not None and m.sigma_15m > 0:
            if s.expected_alpha_bps < self.cfg.min_alpha_vs_15m_sigma * m.sigma_15m * 1e4:  # sigma->bps 近似
                return False
        return True

    # ---------- Pricing ----------
    def _round_to_tick(self, px: float, tick: float) -> float:
        if tick <= 0:
            return px
        return round(px / tick) * tick

    def build_limit_price(self, s: Signal, q_delayed: Quote, m: Metrics, q_rt: Optional[Quote] = None) -> Optional[float]:
        if q_rt and q_rt.last:
            mid = q_rt.last
        else:
            if q_delayed.last is None or m.prev_close is None:
                return None
            band_bps = max(30.0, (m.atr_14 or 0) / (m.prev_close or 1) * 1e4)
            band = band_bps / 1e4 * m.prev_close
            mid = m.prev_close
            if s.side.upper() == 'BUY':
                px = min(mid + band, (s.model_price or mid + band))
            else:
                px = max(mid - band, (s.model_price or mid - band))
            return self._round_to_tick(px, q_delayed.tickSize)

        # real-time分支（简化asuse last 做近似 mid）
        px = mid
        return self._round_to_tick(px, (q_rt or q_delayed).tickSize)

    # ---------- Decision ----------
    def should_trade(self, s: Signal, q_delayed: Quote, m: Metrics, q_rt: Optional[Quote] = None) -> Decision:
        if not self.passes_static_liquidity_filters(m):
            return Decision('REJECT', 'liquidity_filters')
        if not self.passes_signal_thresholds(s, m):
            return Decision('REJECT', 'signal_thresholds')
        # can扩展real-time spread/depth check；延迟场景保守降级
        if q_rt is None:
            if s.expected_alpha_bps < max(self.cfg.min_alpha_bps * 1.5, 75.0):
                return Decision('DEGRADE', 'delayed_env', shrink_to_pct=0.5)
        return Decision('APPROVE', 'ok')

    # ---------- Portfolio ----------
    def allocate_portfolio(self, signals: List[Signal]) -> Dict[str, float]:
        if not signals:
            return {}
        raw = {s.symbol: max(0.0, s.expected_alpha_bps) * (s.confidence or 1.0) for s in signals}
        if not any(raw.values()):
            return {s.symbol: 0.0 for s in signals}
        # Top-N boost
        ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
        boosted = {}
        for i, (sym, val) in enumerate(ranked):
            boosted[sym] = val * (self.cfg.top_n_boost_multiplier if i < self.cfg.top_n_boost else 1.0)
        total = sum(boosted.values())
        weights = {sym: (val / total) for sym, val in boosted.items()}
        # 单票上限
        for sym in list(weights.keys()):
            weights[sym] = min(weights[sym], self.cfg.max_weight)
        # 归一
        z = sum(weights.values()) or 1.0
        weights = {k: v / z for k, v in weights.items()}
        return weights

    # ---------- Sizing ----------
    def compute_child_qty(self, target_delta: int, adv_shares: Optional[float], top_of_book_shares: Optional[float]) -> int:
        if target_delta == 0:
            return 0
        size = abs(target_delta)
        if adv_shares:
            size = min(size, int(adv_shares * self.cfg.max_child_adv_pct))
        if top_of_book_shares:
            size = min(size, int(top_of_book_shares * self.cfg.max_child_book_pct))
        size = max(size, self.cfg.min_child_shares)
        return int(math.copysign(size, target_delta))


class RiskRewardController:
    """for接 GUI and IBKR 门面。
    - cfg.enabled=False when，所has方法透明直通，not改变现has行as。
    - cfg.enabled=True when，应use上述策略。
    """
    def __init__(self, cfg: RRConfig):
        self.cfg = cfg
        self.core = RiskRewardBalancer(cfg)

    def plan_orders(self,
                    model_signals: List[Signal],
                    metrics: Dict[str, Metrics],
                    quotes_delayed: Dict[str, Quote],
                    portfolio_nav: float,
                    current_positions: Dict[str, int]) -> List[Dict]:
        if not self.cfg.enabled:
            return []
        weights = self.core.allocate_portfolio(model_signals)
        planned: List[Dict] = []
        for s in model_signals:
            m = metrics.get(s.symbol)
            qd = quotes_delayed.get(s.symbol)
            if not m or not qd:
                continue
            decision = self.core.should_trade(s, qd, m)
            if decision.action == 'REJECT':
                continue
            limit_px = self.core.build_limit_price(s, qd, m)
            if not limit_px:
                continue
            target_weight = weights.get(s.symbol, 0.0)
            target_shares = int(target_weight * portfolio_nav / max(m.prev_close, 1e-6))
            delta = target_shares - int(current_positions.get(s.symbol, 0))
            if decision.action == 'DEGRADE' and decision.shrink_to_pct:
                delta = int(delta * decision.shrink_to_pct)
            child_qty = self.core.compute_child_qty(delta, adv_shares=(m.adv_usd_20 or 0)/max(m.prev_close,1e-6), top_of_book_shares=None)
            if child_qty == 0:
                continue
            planned.append({
                'symbol': s.symbol,
                'side': 'BUY' if child_qty>0 else 'SELL',
                'quantity': abs(child_qty),
                'limit': float(limit_px)
            })
        return planned
#!/usr/bin/env python3
"""
增强订单执行模块 - 专业级订单执行算法
"""

import asyncio
import time
import logging
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

from ib_insync import IB, Contract, Trade, Order, MarketOrder, LimitOrder

from .order_state_machine import OrderManager, OrderState, OrderType, OrderStateMachine

# OrderRef in ibkr_auto_trader.py in定义，避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ibkr_auto_trader import OrderRef


class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    MARKET = "MARKET"           # market单
    LIMIT = "LIMIT"             # limit单
    ADAPTIVE = "ADAPTIVE"       # 自适应limit
    TWAP = "TWAP"              # when间加权平均price
    VWAP = "VWAP"              # execution量加权平均price
    ICEBERG = "ICEBERG"        # 冰山订单


@dataclass
class ExecutionConfig:
    """执行配置"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    timeout_base: float = 30.0      # 基础超whenwhen间
    timeout_max: float = 120.0      # 最大超whenwhen间
    liquidity_factor: float = 1.0   # 流动性因子 (0-1)
    urgency_factor: float = 0.5     # 紧急程度 (0-1)
    max_retries: int = 3           # 最大重试次数
    partial_fill_min: float = 0.1  # 最小部分execution比例
    price_tolerance: float = 0.02  # price容忍度 (2%)
    
    # TWAP配置
    twap_duration_minutes: int = 10
    twap_slice_count: int = 5
    
    # 冰山单配置
    iceberg_slice_size: float = 0.2  # 每次显示20%


class LiquidityEstimator:
    """流动性估算器"""
    
    def __init__(self):
        self.logger = logging.getLogger("LiquidityEstimator")
        self._liquidity_cache: Dict[str, float] = {}
        self._cache_ttl = 300  # 5分钟缓存
        self._cache_timestamps: Dict[str, float] = {}
    
    async def estimate_liquidity(self, ib: IB, symbol: str) -> float:
        """估算股票流动性 (0-1, 1as最高流动性)"""
        # check缓存
        current_time = time.time()
        if symbol in self._liquidity_cache:
            if current_time - self._cache_timestamps.get(symbol, 0) < self._cache_ttl:
                return self._liquidity_cache[symbol]
        
        try:
            # retrievalticker信息
            from ib_insync import Stock
            contract = Stock(symbol, exchange='SMART', currency='USD')
            ticker = ib.ticker(contract)
            
            if not ticker:
                # 请求ticker数据
                ib.reqMktData(contract)
                await asyncio.sleep(2)  # 等待数据
                ticker = ib.ticker(contract)
            
            # 基at买卖价差估算流动性
            bid = getattr(ticker, 'bid', 0)
            ask = getattr(ticker, 'ask', 0)
            last = getattr(ticker, 'last', 0)
            
            if bid > 0 and ask > 0 and last > 0:
                spread = ask - bid
                spread_pct = spread / last
                
                # 价差越小，流动性越好
                if spread_pct < 0.001:  # 0.1%
                    liquidity = 1.0
                elif spread_pct < 0.005:  # 0.5%
                    liquidity = 0.8
                elif spread_pct < 0.01:   # 1%
                    liquidity = 0.6
                elif spread_pct < 0.02:   # 2%
                    liquidity = 0.4
                else:
                    liquidity = 0.2
            else:
                # 没has买卖价，流动性较低
                liquidity = 0.3
            
            # 缓存结果
            self._liquidity_cache[symbol] = liquidity
            self._cache_timestamps[symbol] = current_time
            
            self.logger.debug(f"{symbol} 流动性估算: {liquidity:.2f}")
            return liquidity
            
        except Exception as e:
            self.logger.warning(f"流动性估算failed {symbol}: {e}")
            return 0.5  # 默认in等流动性


class EnhancedOrderExecutor:
    """增强订单执行器"""
    
    def __init__(self, ib: IB, order_manager: OrderManager):
        self.ib = ib
        self.order_manager = order_manager
        self.liquidity_estimator = LiquidityEstimator()
        self.logger = logging.getLogger("EnhancedOrderExecutor")
        
        # 执行统计
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'timeout_orders': 0,
            'rejected_orders': 0,
            'avg_execution_time': 0.0
        }
        
        # 使use任务生命周期管理器
        from .task_lifecycle_manager import get_task_manager
        self.task_manager = get_task_manager()
    
    def calculate_dynamic_timeout(self, config: ExecutionConfig, liquidity: float) -> float:
        """计算动态超whenwhen间"""
        # 基础超when + 流动性调整 + 紧急程度调整
        base_timeout = config.timeout_base
        liquidity_adjustment = (1.0 - liquidity) * 30  # 低流动性增加最多30 seconds
        urgency_adjustment = (1.0 - config.urgency_factor) * 20  # 低紧急度增加最多20 seconds
        
        dynamic_timeout = base_timeout + liquidity_adjustment + urgency_adjustment
        return min(dynamic_timeout, config.timeout_max)
    
    async def execute_market_order(self, symbol: str, action: str, quantity: int, 
                                 config: ExecutionConfig) -> OrderStateMachine:
        """执行market单 - 带动态超whenand状态轮询"""
        start_time = time.time()
        
        # 估算流动性
        liquidity = await self.liquidity_estimator.estimate_liquidity(self.ib, symbol)
        
        # 计算动态超when
        timeout = self.calculate_dynamic_timeout(config, liquidity)
        
        # 创建contractand订单
        from ib_insync import Stock
        contract = Stock(symbol, exchange='SMART', currency='USD')
        order = MarketOrder(action, quantity)
        
        # order placement
        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        
        # 创建订单状态机
        order_sm = await self.order_manager.create_order(
            order_id=order_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        self.logger.info(f"提交market单: {symbol} {action} {quantity} (ID: {order_id}, 超when: {timeout:.1f}s)")
        
        # updatesas提交状态
        await self.order_manager.update_order_state(
            order_id, OrderState.SUBMITTED, 
            {'trade': trade, 'liquidity': liquidity}, 
            "订单提交"
        )
        
        # 使use任务生命周期管理器创建任务
        execution_task = self.task_manager.create_task(
            self._monitor_order_execution(order_sm, trade, timeout, config),
            task_id=f"market_order_monitor_{order_id}",
            creator="enhanced_order_execution",
            description=f"监控market单执行: {symbol} {action} {quantity}",
            group="order_execution",
            max_lifetime=timeout + 30  # 超whenwhen间 + 缓冲
        )
        
        return order_sm
    
    async def _monitor_order_execution(self, order_sm: OrderStateMachine, trade: Trade, 
                                     timeout: float, config: ExecutionConfig):
        """监控订单执行过程"""
        start_time = time.time()
        check_interval = 0.5  # 500mscheck间隔
        
        try:
            while time.time() - start_time < timeout:
                # check订单状态
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                filled = getattr(trade.orderStatus, 'filled', 0)
                remaining = getattr(trade.orderStatus, 'remaining', order_sm.quantity)
                avg_fill_price = getattr(trade.orderStatus, 'avgFillPrice', 0.0)
                
                # 状态updates
                if status == 'Submitted' and order_sm.state == OrderState.SUBMITTED:
                    await self.order_manager.update_order_state(
                        order_sm.order_id, OrderState.ACKNOWLEDGED, 
                        {'ib_status': status}, "交易所确认"
                    )
                
                elif filled > 0:
                    # hasexecution
                    fill_data = {
                        'filled_quantity': filled,
                        'avg_fill_price': avg_fill_price,
                        'remaining': remaining,
                        'ib_status': status
                    }
                    
                    if filled >= order_sm.quantity:
                        # 完全execution
                        order_sm.update_fill(filled, avg_fill_price)
                        self.logger.info(f"订单完全execution: {order_sm.order_id} - {filled}股 @ ${avg_fill_price:.2f}")
                        break
                    elif filled > order_sm.filled_quantity:
                        # 部分execution
                        order_sm.update_fill(filled, avg_fill_price)
                        self.logger.info(f"订单部分execution: {order_sm.order_id} - {filled}/{order_sm.quantity}股")
                
                elif status in ['Cancelled', 'Rejected']:
                    # 订单be取消or拒绝
                    if status == 'Cancelled':
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.CANCELLED, 
                            {'ib_status': status}, "订单be取消"
                        )
                    else:
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.REJECTED, 
                            {'ib_status': status}, "订单be拒绝"
                        )
                    break
                
                # 等待下次check
                await asyncio.sleep(check_interval)
            
            # 超when处理
            if not order_sm.is_terminal():
                execution_time = time.time() - start_time
                
                # checkis否has部分execution
                if order_sm.filled_quantity > 0:
                    fill_rate = order_sm.filled_quantity / order_sm.quantity
                    if fill_rate >= config.partial_fill_min:
                        # 部分executioncan接受，取消剩余订单
                        self.ib.cancelOrder(trade.order)
                        self.logger.info(f"订单部分completed，取消剩余: {order_sm.order_id} ({fill_rate*100:.1f}%execution)")
                        
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.CANCELLED, 
                            {'execution_time': execution_time, 'partial_fill': True}, 
                            f"超whenafter部分execution取消 ({fill_rate*100:.1f}%)"
                        )
                    else:
                        # execution太少，标记asfailed
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.FAILED, 
                            {'execution_time': execution_time, 'timeout': True}, 
                            f"执行超when ({execution_time:.1f}s)"
                        )
                else:
                    # 完全没hasexecution
                    self.ib.cancelOrder(trade.order)
                    await self.order_manager.update_order_state(
                        order_sm.order_id, OrderState.FAILED, 
                        {'execution_time': execution_time, 'timeout': True}, 
                        f"执行超when，noexecution ({execution_time:.1f}s)"
                    )
            
            # updates统计
            self._update_execution_stats(order_sm, time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"订单监控异常 {order_sm.order_id}: {e}")
            await self.order_manager.update_order_state(
                order_sm.order_id, OrderState.FAILED, 
                {'error': str(e)}, f"监控异常: {e}"
            )
        finally:
            # 任务生命周期管理器会自动清理
            pass
    
    async def execute_adaptive_limit_order(self, symbol: str, action: str, quantity: int,
                                         reference_price: float, config: ExecutionConfig) -> OrderStateMachine:
        """执行自适应limit单"""
        # 基at市场 records件and紧急度调整limit
        liquidity = await self.liquidity_estimator.estimate_liquidity(self.ib, symbol)
        
        # price调整逻辑
        if action.upper() == 'BUY':
            # 买单：根据紧急度and流动性确定limit
            aggressive_factor = config.urgency_factor * (1.0 - liquidity)
            limit_price = reference_price * (1 + config.price_tolerance * aggressive_factor)
        else:
            # 卖单
            aggressive_factor = config.urgency_factor * (1.0 - liquidity)
            limit_price = reference_price * (1 - config.price_tolerance * aggressive_factor)
        
        # 创建limit单
        from ib_insync import Stock
        contract = Stock(symbol, exchange='SMART', currency='USD')
        order = LimitOrder(action, quantity, limit_price)
        
        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        
        # 创建订单状态机
        order_sm = await self.order_manager.create_order(
            order_id=order_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=limit_price
        )
        
        self.logger.info(f"提交自适应limit单: {symbol} {action} {quantity} @ ${limit_price:.2f}")
        
        # start监控（使use任务生命周期管理器）
        timeout = self.calculate_dynamic_timeout(config, liquidity)
        execution_task = self.task_manager.create_task(
            self._monitor_adaptive_limit_execution(order_sm, trade, timeout, config, reference_price),
            task_id=f"adaptive_limit_monitor_{order_id}",
            creator="enhanced_order_execution",
            description=f"监控自适应limit单: {symbol} {action} {quantity}",
            group="order_execution",
            max_lifetime=timeout + 30
        )
        
        return order_sm
    
    async def _monitor_adaptive_limit_execution(self, order_sm: OrderStateMachine, trade: Trade,
                                              timeout: float, config: ExecutionConfig, reference_price: float):
        """监控自适应limit单执行"""
        start_time = time.time()
        price_update_interval = 10.0  # 10 secondsupdates一次price
        last_price_update = start_time
        
        try:
            while time.time() - start_time < timeout and not order_sm.is_terminal():
                current_time = time.time()
                
                # check订单状态
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                filled = getattr(trade.orderStatus, 'filled', 0)
                
                if filled > 0:
                    # hasexecution，updates状态
                    avg_fill_price = getattr(trade.orderStatus, 'avgFillPrice', 0.0)
                    order_sm.update_fill(filled, avg_fill_price)
                    
                    if filled >= order_sm.quantity:
                        self.logger.info(f"自适应limit单completed: {order_sm.order_id}")
                        break
                
                # 动态price调整
                if current_time - last_price_update > price_update_interval:
                    try:
                        # retrieval当beforemarket
                        ticker = self.ib.ticker(trade.contract)
                        if ticker and hasattr(ticker, 'last') and ticker.last > 0:
                            current_market_price = ticker.last
                            
                            # 判断is否需要调整price
                            if order_sm.side == 'BUY':
                                # 买单：if果market上涨太多，提高limit
                                if current_market_price > reference_price * (1 + config.price_tolerance):
                                    new_limit_price = current_market_price * (1 + config.price_tolerance * 0.5)
                                    # 修改订单price
                                    trade.order.lmtPrice = new_limit_price
                                    self.ib.placeOrder(trade.contract, trade.order)
                                    self.logger.info(f"调整买单limit: {order_sm.order_id} ${order_sm.price:.2f} -> ${new_limit_price:.2f}")
                            else:
                                # 卖单：if果market下跌太多，降低limit
                                if current_market_price < reference_price * (1 - config.price_tolerance):
                                    new_limit_price = current_market_price * (1 - config.price_tolerance * 0.5)
                                    trade.order.lmtPrice = new_limit_price
                                    self.ib.placeOrder(trade.contract, trade.order)
                                    self.logger.info(f"调整卖单limit: {order_sm.order_id} ${order_sm.price:.2f} -> ${new_limit_price:.2f}")
                        
                        last_price_update = current_time
                        
                    except Exception as e:
                        self.logger.warning(f"price调整failed {order_sm.order_id}: {e}")
                
                await asyncio.sleep(1.0)  # 1 secondscheck间隔
            
            # 超when处理
            if not order_sm.is_terminal():
                self.ib.cancelOrder(trade.order)
                await self.order_manager.update_order_state(
                    order_sm.order_id, OrderState.CANCELLED, 
                    {'timeout': True}, "自适应limit单超when取消"
                )
            
        except Exception as e:
            self.logger.error(f"自适应limit单监控异常 {order_sm.order_id}: {e}")
            await self.order_manager.update_order_state(
                order_sm.order_id, OrderState.FAILED, 
                {'error': str(e)}, f"监控异常: {e}"
            )
        finally:
            # 任务生命周期管理器会自动清理
            pass
    
    def _update_execution_stats(self, order_sm: OrderStateMachine, execution_time: float):
        """updates执行统计"""
        self.execution_stats['total_orders'] += 1
        
        if order_sm.state == OrderState.FILLED:
            self.execution_stats['successful_orders'] += 1
        elif order_sm.state == OrderState.REJECTED:
            self.execution_stats['rejected_orders'] += 1
        else:
            self.execution_stats['timeout_orders'] += 1
        
        # updates平均执行when间
        total = self.execution_stats['total_orders']
        old_avg = self.execution_stats['avg_execution_time']
        self.execution_stats['avg_execution_time'] = ((old_avg * (total - 1)) + execution_time) / total
    
    async def cancel_order(self, order_id: int, reason: str = "use户取消") -> bool:
        """取消订单"""
        order_sm = await self.order_manager.get_order(order_id)
        if not order_sm:
            return False
        
        if order_sm.is_terminal():
            self.logger.warning(f"订单处at终态，no法取消: {order_id}")
            return False
        
        try:
            # 取消IB订单
            # 这里需要根据实际情况retrievaltradefor象
            # 暂when使use通use取消方法
            self.ib.cancelOrder(order_id)
            
            # updates状态
            await self.order_manager.update_order_state(order_id, OrderState.CANCELLED, reason=reason)
            
            # 通过任务管理器取消任务
            self.task_manager.cancel_task(f"market_order_monitor_{order_id}", "手动取消")
            self.task_manager.cancel_task(f"adaptive_limit_monitor_{order_id}", "手动取消")
            
            return True
            
        except Exception as e:
            self.logger.error(f"取消订单failed {order_id}: {e}")
            return False
    
    async def get_execution_statistics(self) -> Dict[str, Any]:
        """retrieval执行统计信息"""
        stats = self.execution_stats.copy()
        
        # 计算success率
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
        else:
            stats['success_rate'] = 0.0
        
        # 活跃订单数（from任务管理器retrieval）
        order_tasks = self.task_manager.list_tasks(group="order_execution")
        stats['active_executions'] = len(order_tasks)
        
        return stats
    
    async def cleanup(self):
        """清理资源"""
        # 通过任务管理器取消所has订单执行任务
        cancelled_count = self.task_manager.cancel_group("order_execution", "系统清理")
        self.logger.info(f"订单执行器清理，取消了 {cancelled_count} 个任务")

    # ==================== 高级执行算法 ====================
    
    async def execute_twap_order(self, contract, total_quantity: int, 
                                duration_minutes: int = 30, slice_count: int = 10):
        """TWAP (when间加权平均price) 执行算法"""
        symbol = contract.symbol
        action = "BUY" if total_quantity > 0 else "SELL"
        abs_quantity = abs(total_quantity)
        
        self.logger.info(f"startingTWAP执行: {symbol} {action} {abs_quantity}股, {duration_minutes}分钟, {slice_count}片")
        
        # 计算每片参数
        slice_quantity = abs_quantity // slice_count
        remainder = abs_quantity % slice_count
        slice_interval = (duration_minutes * 60) / slice_count  #  seconds
        
        executed_orders = []
        
        try:
            for i in range(slice_count):
                # 计算本片数量
                current_slice = slice_quantity
                if i < remainder:  # 余数分配给before几片
                    current_slice += 1
                
                if current_slice == 0:
                    continue
                
                self.logger.info(f"TWAP片段 {i+1}/{slice_count}: {current_slice}股")
                
                # 执行本片订单
                try:
                    from ib_insync import MarketOrder
                    order = MarketOrder(action, current_slice)
                    trade = self.ib.placeOrder(contract, order)
                    
                    # 等待execution
                    await asyncio.wait_for(trade.doneEvent.wait(), timeout=30.0)
                    
                    executed_orders.append({
                        'slice': i+1,
                        'quantity': current_slice,
                        'filled': getattr(trade.orderStatus, 'filled', 0),
                        'avg_price': getattr(trade.orderStatus, 'avgFillPrice', 0.0),
                        'status': trade.orderStatus.status
                    })
                    
                    # 等待片段间隔（除了最after一片）
                    if i < slice_count - 1:
                        await asyncio.sleep(slice_interval)
                        
                except Exception as e:
                    self.logger.error(f"TWAP片段 {i+1} 执行failed: {e}")
                    continue
            
            # 统计执行结果
            total_filled = sum(order['filled'] for order in executed_orders)
            total_value = sum(order['filled'] * order['avg_price'] for order in executed_orders)
            avg_price = total_value / total_filled if total_filled > 0 else 0
            
            self.logger.info(f"TWAP执行completed: {symbol} 总execution{total_filled}/{abs_quantity}股, 均价${avg_price:.4f}")
            
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"TWAP执行异常: {symbol}: {e}")
            return executed_orders
    
    async def execute_vwap_order(self, contract, total_quantity: int,
                                participation_rate: float = 0.1):
        """VWAP (execution量加权平均price) 执行算法"""
        symbol = contract.symbol
        action = "BUY" if total_quantity > 0 else "SELL"
        abs_quantity = abs(total_quantity)
        
        self.logger.info(f"startingVWAP执行: {symbol} {action} {abs_quantity}股, 参and率{participation_rate:.1%}")
        
        executed_orders = []
        remaining_qty = abs_quantity
        
        try:
            while remaining_qty > 0:
                # retrieval当beforeexecution量
                ticker = self.ib.ticker(contract)
                await self.ib.sleep(1)  # 等待数据updates
                
                # 计算当beforewhen段目标执行量
                if hasattr(ticker, 'volume') and ticker.volume > 0:
                    # 基at实际execution量计算
                    recent_volume = ticker.volume  # 简化版，实际应该is近期N分钟execution量
                    target_volume = int(recent_volume * participation_rate)
                else:
                    # 降级to固定比例
                    target_volume = max(1, int(remaining_qty * 0.1))
                
                # 限制单次执行量
                current_qty = min(target_volume, remaining_qty, 1000)  # 最大1000股
                
                if current_qty <= 0:
                    self.logger.warning("VWAP计算量as0，等待市场活跃")
                    await asyncio.sleep(30)  # 等待30 seconds
                    continue
                
                self.logger.info(f"VWAP执行: {current_qty}股 (剩余{remaining_qty})")
                
                try:
                    from ib_insync import MarketOrder
                    order = MarketOrder(action, current_qty)
                    trade = self.ib.placeOrder(contract, order)
                    
                    # 等待execution
                    await asyncio.wait_for(trade.doneEvent.wait(), timeout=30.0)
                    
                    filled = getattr(trade.orderStatus, 'filled', 0)
                    executed_orders.append({
                        'quantity': current_qty,
                        'filled': filled,
                        'avg_price': getattr(trade.orderStatus, 'avgFillPrice', 0.0),
                        'status': trade.orderStatus.status
                    })
                    
                    remaining_qty -= filled
                    
                    # 动态调整等待when间
                    if remaining_qty > 0:
                        wait_time = max(5, min(60, 30 * (1 - participation_rate)))
                        await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"VWAP片段执行failed: {e}")
                    await asyncio.sleep(10)
                    continue
            
            # 统计执行结果
            total_filled = sum(order['filled'] for order in executed_orders)
            total_value = sum(order['filled'] * order['avg_price'] for order in executed_orders)
            avg_price = total_value / total_filled if total_filled > 0 else 0
            
            self.logger.info(f"VWAP执行completed: {symbol} 总execution{total_filled}/{abs_quantity}股, 均价${avg_price:.4f}")
            
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"VWAP执行异常: {symbol}: {e}")
            return executed_orders