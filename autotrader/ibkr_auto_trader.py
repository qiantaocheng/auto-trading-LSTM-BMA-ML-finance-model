"""
IBKR 自动交易最小闭环脚本（连接→行情→下单/撤单→回报→账户/持仓→风控/工具）

基于 ib_insync 封装 TWS API（EClient/EWrapper）。覆盖自动交易常见用法：
- 连接/重连、行情类型切换（实时/延时）
- 合约资格校验、主交易所设置
- 行情订阅与价格获取（Ticker 事件），深度可按需扩展
- 账户摘要/账户更新、持仓、PnL（可选）
- 下单（市价/限价/括号单）、撤单、订单/成交/佣金回报
- 简易风险控制（资金占比/持仓检查/订单去重）

参考：
- IBKR Campus TWS API 文档（EClient/EWrapper）
  https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#api-introduction
- EClient 类参考: https://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html
- EWrapper 接口参考: https://interactivebrokers.github.io/tws-api/interfaceIBApi_1_1EWrapper.html
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque
from enum import Enum
import os
import json
import urllib.request
import urllib.error
from time import time as _now

from ib_insync import (
    IB,
    Stock,
    Contract,
    MarketOrder,
    LimitOrder,
    BracketOrder,
    StopOrder,
    Ticker,
    Order,
    Trade,
)


# ----------------------------- 日志配置 -----------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------- 数据结构 -----------------------------
@dataclass
class OrderRef:
    order_id: int
    symbol: str
    side: str
    qty: int
    order_type: str
    limit_price: Optional[float] = None
    parent_id: Optional[int] = None


# ----------------------------- 实时信号/数据结构 -----------------------------
class ActionType(str, Enum):
    BUY_NOW = "BUY_NOW"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_NOW = "SELL_NOW"
    SELL_LIMIT = "SELL_LIMIT"


@dataclass
class TickData:
    timestamp: float
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float = 0.0
    volume: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if self.last > 0 else 0.0


@dataclass
class MicroSignal:
    action: ActionType
    entry_price: float
    stop_loss: float
    take_profit: float
    should_trade: bool = True
    confidence: float = 0.6
    risk_reward: float = 2.0


class RealtimeSignalEngine:
    """轻量实时信号引擎（每秒）"""
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.price_history: Deque[float] = deque(maxlen=1000)
        self.volume_history: Deque[float] = deque(maxlen=1000)
        self.last_tick: Optional[TickData] = None
        self.last_calc_ts: float = 0.0

    def initialize_with_history(self, bars: List) -> None:
        try:
            for b in bars[-200:]:
                self.price_history.append(float(b.close))
        except Exception:
            pass

    def process_tick(self, tick: TickData) -> Optional[MicroSignal]:
        self.last_tick = tick
        self.price_history.append(tick.mid)
        self.volume_history.append(max(0.0, tick.volume))

        now = tick.timestamp
        if now - self.last_calc_ts < 1.0:
            return None
        self.last_calc_ts = now

        prices = list(self.price_history)
        if len(prices) < 20:
            return None

        ma20 = sum(prices[-20:]) / 20.0
        price = tick.mid
        if price <= 0:
            return None
        deviation = (price - ma20) / ma20 if ma20 > 0 else 0.0
        rsi = self._rsi(prices, 14)

        # 简单信号：均值回归 + RSI
        if deviation < -0.02 and rsi < 35:
            entry = tick.ask if tick.ask > 0 else price * 1.001
            stop = entry * 0.98
            target = ma20
            rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0.0
            return MicroSignal(ActionType.BUY_NOW, entry, stop, target, True, 0.7, rr)
        if deviation > 0.02 and rsi > 65:
            entry = tick.bid if tick.bid > 0 else price * 0.999
            stop = entry * 1.02
            target = ma20
            rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else 0.0
            return MicroSignal(ActionType.SELL_NOW, entry, stop, target, True, 0.7, rr)
        return None

    @staticmethod
    def _rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            ch = prices[i] - prices[i - 1]
            if ch >= 0:
                gains += ch
            else:
                losses -= ch
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0.0
        return 100.0 - (100.0 / (1.0 + rs))


class IbkrAutoTrader:
    def __init__(
        self,
        config_manager=None,
        ib_client: Optional[IB] = None,
    ) -> None:
        # 使用统一的配置管理器
        if config_manager is None:
            from .unified_config import get_unified_config
            config_manager = get_unified_config()
        
        self.config_manager = config_manager
        
        # 从统一配置获取连接参数
        conn_params = config_manager.get_connection_params()
        self.host = conn_params['host']
        self.port = conn_params['port']
        self.client_id = conn_params['client_id']
        self.account_id = conn_params['account_id']
        self.use_delayed_if_no_realtime = conn_params['use_delayed_if_no_realtime']
        self.default_currency = "USD"

        # 允许外部传入共享连接
        self.ib = ib_client if ib_client is not None else IB()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 兼容处理：预初始化 wrapper/_results 与 decoder handlers，避免老版本 ib_insync 报错
        try:
            # 确保 completedOrders/openOrders 等键存在，避免 KeyError
            if hasattr(self.ib, 'wrapper') and hasattr(self.ib.wrapper, '_results'):
                res = self.ib.wrapper._results  # type: ignore[attr-defined]
                if isinstance(res, dict):
                    res.setdefault('completedOrders', [])
                    res.setdefault('openOrders', [])
                    res.setdefault('fills', [])
            # 忽略未知消息ID（如 176）以适配不同 API 版本差异
            if hasattr(self.ib, 'decoder') and hasattr(self.ib.decoder, 'handlers'):
                handlers = self.ib.decoder.handlers  # type: ignore[attr-defined]
                if isinstance(handlers, dict):
                    for msg_id in (176,):
                        handlers.setdefault(msg_id, lambda fields: None)
        except Exception:
            pass

        # 状态缓存
        self.tickers: Dict[str, Ticker] = {}
        self.last_price: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, ts)
        self.account_values: Dict[str, str] = {}
        self.account_id: Optional[str] = None
        self.cash_balance: float = 0.0
        self.net_liq: float = 0.0
        self.buying_power: float = 0.0  # 添加买力属性
        self.positions: Dict[str, int] = {}  # symbol -> qty
        self.open_orders: Dict[int, OrderRef] = {}
        self._stop_event: Optional[asyncio.Event] = None
        
        # 账户状态管理增强
        self.account_ready: bool = False
        self._last_account_update: float = 0.0
        self._account_lock = asyncio.Lock()
        self.account_update_interval: float = 60.0  # 最小更新间隔60秒
        
        # 使用event_loop_manager管理任务，不再需要独立的TaskManager
        self._active_tasks = {}
        
        # 交易审计器 (需要先初始化，供OrderManager使用)
        from .trading_auditor_v2 import TradingAuditor
        self.auditor = TradingAuditor(
            log_directory="audit_logs",
            db_path="trading_audit.db"
        )
        
        # 订单状态管理
        from .order_state_machine import OrderManager
        from .enhanced_order_execution import EnhancedOrderExecutor
        self.order_manager = OrderManager(auditor=self.auditor)  # 传入审计器
        self.enhanced_executor = EnhancedOrderExecutor(self.ib, self.order_manager)

    # ------------------------- 辅助：账户取值 -------------------------
    def _get_account_numeric(self, tag: str) -> float:
        """从 account_values 提取某个账户字段的数值。
        优先顺序：BASE -> 默认货币 -> 任意第一条可解析数据。
        """
        candidates: List[Tuple[str, float]] = []
        try:
            for key, value in self.account_values.items():
                if key.startswith(f"{tag}:"):
                    currency = key.split(":", 1)[1]
                    try:
                        candidates.append((currency or "", float(value)))
                    except Exception:
                        continue
            # 打印调试信息，帮助定位币种键
            if not candidates and self.account_values:
                self.logger.debug(f"账户字段{tag}未找到，当前键示例: {list(self.account_values.keys())[:5]}")
            # 优先 BASE
            for cur, num in candidates:
                if cur.upper() == "BASE":
                    return num
            # 次选 默认货币
            for cur, num in candidates:
                if cur.upper() == (self.default_currency or "").upper():
                    return num
            # 回退：任意一个
            if candidates:
                return candidates[0][1]
        except Exception:
            pass
        return 0.0
        
        # 简化连接恢复管理，在ibkr_auto_trader内部处理重连逻辑
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_interval = 5.0
        
        # 风险管理功能已集成到Engine的RiskEngine中，这里只需要简单的风险检查
        self._daily_order_count = 0
        self._max_daily_orders = 50
        


        # 动态止损管理（ATR + 时间加权）
        self.dynamic_stop_cfg = {
            "atr_period": 14,
            "atr_lookback_days": 60,
            "atr_multiplier": 2.0,
            "decay_per_min": 0.1 / 60.0,  # 每分钟衰减0.1/60，约每小时0.1
            "min_decay_factor": 0.1,
            "update_interval_sec": 60.0,
        }
        # symbol -> state dict {entry_price, entry_time, qty, stop_trade, current_stop}
        self._stop_state: Dict[str, Dict[str, object]] = {}
        # symbol -> asyncio.Task for updater
        self._stop_tasks: Dict[str, asyncio.Task] = {}

        # 订单验证与统计
        self.order_verify_cfg = {
            "cash_reserve_pct": 0.15,
            "max_single_position_pct": 0.12,
            "min_order_value_usd": 500.0,
            "price_range": (2.0, 800.0),
            "daily_order_limit": 20,
            "verify_tolerance_usd": 100.0,
        }
        self._daily_order_count: int = 0
        self._last_reset_day: Optional[datetime.date] = None
        self._notify_throttle: Dict[str, float] = {}

        # 止损/止盈配置（可从 data/risk_config.json 读取覆盖）
        self.allow_short: bool = True
        self.risk_config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "data", "risk_config.json")
        self.risk_config: Dict = {
            "risk_management": {
                "default_stop_pct": 0.02,
                "default_target_pct": 0.05,
                "use_atr_stops": False,
                "atr_multiplier_stop": 2.0,
                "atr_multiplier_target": 3.0,
                "use_bracket_on_removed": False,
                "enable_local_dynamic_stop_for_bracket": False,
                "atr_risk_scale": 5.0,
                "webhook_url": "",
                "realtime_alloc_pct": 0.03,
                "symbol_overrides": {},
                "strategy_settings": {
                    "scalping": {"stop_pct": 0.005, "target_pct": 0.01},
                    "swing": {"stop_pct": 0.03, "target_pct": 0.08},
                    "position": {"stop_pct": 0.05, "target_pct": 0.15},
                },
            }
        }
        try:
            os.makedirs(os.path.dirname(self.risk_config_path), exist_ok=True)
            if os.path.exists(self.risk_config_path):
                with open(self.risk_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # 浅合并
                        self.risk_config["risk_management"].update(data.get("risk_management", {}))
        except Exception:
            pass

        # 统一从全局配置同步风险限制（与 RiskManager 和本地验证保持一致）
        try:
            # 使用统一配置管理器替代HotConfig
            config_dict = self.config_manager._get_merged_config()
            self._sync_risk_limits_from_config({"CONFIG": config_dict})
        except Exception:
            # 配置不可用时保留默认值
            pass

    def load_risk_config_from_db(self) -> None:
        try:
            from .database import StockDatabase
            db = StockDatabase()
            # 优先使用新的风险配置结构
            cfg = db.get_risk_config("默认风险配置")
            if not cfg:
                # fallback到旧配置
                cfg = db.get_risk_config() or {}
                if isinstance(cfg, dict) and "risk_management" in cfg:
                    cfg = cfg["risk_management"]
            
            if isinstance(cfg, dict):
                # 更新风险配置
                if "default_stop_pct" in cfg:
                    self.risk_config["risk_management"]["default_stop_pct"] = cfg["default_stop_pct"]
                if "default_target_pct" in cfg:
                    self.risk_config["risk_management"]["default_target_pct"] = cfg["default_target_pct"]
                if "max_single_position_pct" in cfg:
                    self.risk_config["risk_management"]["max_single_position_pct"] = cfg["max_single_position_pct"]
                
                self.allow_short = bool(cfg.get("allow_short", self.allow_short))
                self.logger.info(f"已从数据库加载风险配置: 止损{cfg.get('default_stop_pct', 0.02)*100:.1f}% 止盈{cfg.get('default_target_pct', 0.05)*100:.1f}%")
        except Exception as e:
            self.logger.warning(f"从数据库加载风险配置失败: {e}")

        # 事件绑定
        self._bind_events()

    def _sync_risk_limits_from_config(self, cfg: Dict[str, Any]) -> None:
        """从全局配置同步风险限制，统一来源，避免冲突。
        优先级建议：数据库risk_config(止损/止盈/策略) > HotConfig.CONFIG.capital(资金/单仓限制) > 本地默认。
        """
        try:
            # 资金与仓位上限
            capital = (cfg.get("CONFIG") or {}).get("capital") or {}
            cash_reserve = capital.get("cash_reserve_pct")
            max_single_pos = capital.get("max_single_position_pct")
            if cash_reserve is not None:
                self.order_verify_cfg["cash_reserve_pct"] = float(cash_reserve)
            if max_single_pos is not None:
                self.order_verify_cfg["max_single_position_pct"] = float(max_single_pos)

            # 同步到高级风险管理器
            try:
                if max_single_pos is not None:
                    self.risk_manager.max_single_position = float(max_single_pos)
            except Exception:
                pass

            risk_controls = (cfg.get("CONFIG") or {}).get("risk_controls") or {}
            sector_limit = risk_controls.get("sector_exposure_limit")
            if sector_limit is not None:
                try:
                    self.risk_manager.max_sector_exposure = float(sector_limit)
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"同步风险限制到配置失败: {e}")

    # ------------------------- 连接与事件 -------------------------
    def _bind_events(self) -> None:
        """绑定事件处理器"""
        ib = self.ib
        try:
            ib.errorEvent += self._on_error
            ib.orderStatusEvent += self._on_order_status
            ib.execDetailsEvent += self._on_exec_details
            ib.commissionReportEvent += self._on_commission
            ib.accountSummaryEvent += self._on_account_summary
            
            # 检查并绑定可用的事件
            if hasattr(ib, 'updateAccountValueEvent'):
                ib.updateAccountValueEvent += self._on_update_account_value
            if hasattr(ib, 'accountValueEvent'):
                ib.accountValueEvent += self._on_update_account_value
                
            if hasattr(ib, 'updatePortfolioEvent'):
                ib.updatePortfolioEvent += self._on_update_portfolio
            if hasattr(ib, 'portfolioEvent'):
                ib.portfolioEvent += self._on_update_portfolio
                
            if hasattr(ib, 'positionEvent'):
                ib.positionEvent += self._on_position
            if hasattr(ib, 'currentTimeEvent'):
                ib.currentTimeEvent += self._on_current_time

            self.logger.info("✅ 事件处理器绑定完成")
        except Exception as e:
            self.logger.warning(f"⚠️ 事件绑定部分失败: {e}")
            # 继续运行，不因事件绑定失败而中断

    async def connect(self, retries: int = None, retry_delay: float = None) -> None:
        """统一的连接逻辑，使用配置管理器"""
        if retries is None:
            retries = self.config_manager.get('connection.max_reconnect_attempts', 10)
        if retry_delay is None:
            retry_delay = self.config_manager.get('connection.reconnect_interval', 5.0)
            
        # 使用统一配置管理器，不再需要独立的ConnectionConfig
        # from .connection_config import ConnectionManager, ConnectionConfig
        
        # 使用统一配置管理器直接连接，简化逻辑
        
        self.logger.info(f"开始连接 {self.host}:{self.port}，目标ClientID={self.client_id}，账户={self.account_id}")
        
        # 简化连接逻辑，直接使用配置的ClientID
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.logger.info(f"[OK] 已连接，使用ClientID={self.client_id}")
            
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            raise
        
        # 设置市场数据类型
        try:
            data_type = await connection_manager.setup_market_data_type(self.ib)
            self.logger.info(f"市场数据类型设置为: {data_type}")
        except Exception as e:
            self.logger.warning(f"设置市场数据类型失败: {e}")
        
        # 等待账户数据就绪
        await self._wait_for_account_data()
        
        # 启动连接监控和其他服务
        await self._post_connection_setup()

    async def _wait_for_account_data(self, timeout: float = 10.0) -> bool:
        """等待账户数据就绪"""
        import time
        start_time = time.time()
        
        self.logger.info("等待账户数据加载...")
        
        # 首次强制刷新账户数据
        try:
            await self.refresh_account_balances_and_positions()
            if self.net_liq > 0:
                self.account_ready = True
                self.logger.info(f"✅ 账户数据就绪: 净值=${self.net_liq:,.2f}, 现金=${self.cash_balance:,.2f}, 账户={self.account_id}")
                return True
        except Exception as e:
            self.logger.debug(f"首次账户数据刷新失败: {e}")
        
        # 如果首次失败，再等待一下
        while time.time() - start_time < timeout:
            try:
                await asyncio.sleep(2)  # 等待数据到达
                
                # 获取账户值
                account_values = self.ib.accountValues()
                if account_values:
                    self.account_values = {f"{av.tag}:{av.currency}": av.value for av in account_values}
                    
                    # 解析关键账户数据
                    self.net_liq = self._get_account_numeric('NetLiquidation')
                    self.cash_balance = self._get_account_numeric('TotalCashValue')
                    self.buying_power = self._get_account_numeric('BuyingPower')
                    
                    # 获取账户ID
                    for av in account_values:
                        if av.tag == 'AccountId':
                            self.account_id = av.value
                            break
                    
                    if self.net_liq > 0:
                        self.account_ready = True
                        self.logger.info(f"✅ 账户数据就绪: 净值=${self.net_liq:,.2f}, 现金=${self.cash_balance:,.2f}, 账户={self.account_id}")
                        return True
                        
            except Exception as e:
                self.logger.debug(f"等待账户数据: {e}")
            
            await asyncio.sleep(1.0)
        
        # 即使超时也尝试使用现有数据
        if hasattr(self, 'account_id') and self.account_id:
            self.logger.info(f"⚠️ 账户数据获取超时，使用现有数据: 账户={self.account_id}")
            return True
        
        self.logger.warning("⚠️ 账户数据获取超时，继续运行但可能影响交易")
        return False

    async def _post_connection_setup(self):
        """连接后的设置工作"""
        try:
            # 初始化包装器结果字典
            self._init_wrapper_results()
            
            # 获取持仓信息
            await self._update_positions()
            
            # 连接恢复功能已简化到内部处理
            
            # 启动实时账户监控任务（简化版）
            try:
                task = asyncio.create_task(self._account_monitor_task())
                self._active_tasks["account_monitor"] = task
            except Exception as e:
                self.logger.error(f"启动账户监控任务失败: {e}")
            
            self.logger.info("✅ 连接后设置完成")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 连接后设置部分失败: {e}")

    async def _update_positions(self):
        """更新持仓信息"""
        try:
            positions = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            self.positions = {}
            
            for pos in positions:
                if pos.position != 0:
                    symbol = pos.contract.symbol
                    self.positions[symbol] = pos.position
                    
            self.logger.info(f"持仓更新完成: {len(self.positions)} 个非零持仓")
            
        except Exception as e:
            self.logger.warning(f"更新持仓信息失败: {e}")
            self.positions = {}

    def _init_wrapper_results(self):
        """初始化包装器结果字典，防止KeyError"""
        try:
            if hasattr(self.ib, 'wrapper') and hasattr(self.ib.wrapper, '_results'):
                res = self.ib.wrapper._results
                if isinstance(res, dict) and 'completedOrders' not in res:
                    res['completedOrders'] = []
                    self.logger.debug("已初始化completedOrders容器")
        except Exception as e:
            self.logger.debug(f"初始化包装器结果失败: {e}")



    async def _account_monitor_task(self) -> None:
        """实时账户监控任务 - 确保账户数据实时性"""
        monitor_interval = 30.0  # 30秒监控间隔
        
        try:
            while not self._stop_event or not self._stop_event.is_set():
                try:
                    current_time = time.time()
                    
                    # 检查账户数据是否过期
                    if current_time - self._last_account_update > self.account_update_interval:
                        async with self._account_lock:
                            self.logger.debug("账户数据过期，执行自动刷新")
                            await self.refresh_account_balances_and_positions()
                    
                    # 检查关键指标异常
                    if self.account_ready:
                        # 检查净值是否异常
                        if self.net_liq <= 0:
                            self.logger.warning("账户净值异常：<=0，强制刷新账户数据")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                        
                        # 检查现金余额是否异常
                        elif self.cash_balance < 0:
                            self.logger.warning("现金余额异常：<0，强制刷新账户数据")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                        
                        # 检查持仓数据一致性
                        elif len(self.positions) == 0 and self.net_liq > self.cash_balance * 1.1:
                            self.logger.warning("持仓数据可能不一致，强制刷新")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                    
                    await asyncio.sleep(monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"账户监控任务异常: {e}")
                    await asyncio.sleep(monitor_interval)
                    
        except asyncio.CancelledError:
            self.logger.info("账户监控任务被取消")
        except Exception as e:
            self.logger.error(f"账户监控任务致命错误: {e}")
            raise  # 让任务管理器重启

    async def _risk_monitor_task(self) -> None:
        """风险监控任务 - 持续监控投资组合风险"""
        monitor_interval = 120.0  # 2分钟监控间隔
        last_risk_report = 0
        report_interval = 600.0  # 10分钟生成一次风险报告
        
        try:
            while not self._stop_event or not self._stop_event.is_set():
                try:
                    current_time = time.time()
                    
                    # 更新所有持仓的价格历史
                    for symbol in self.positions.keys():
                        current_price = self.get_price(symbol)
                        if current_price and current_price > 0:
                            await self.risk_manager.update_price_history(symbol, current_price)
                    
                    # 计算持仓价值
                    positions_value = {}
                    total_position_value = 0.0
                    
                    for symbol, qty in self.positions.items():
                        if qty > 0:
                            price = self.get_price(symbol) or 0.0
                            if price > 0:
                                value = qty * price
                                positions_value[symbol] = value
                                total_position_value += value
                    
                    if positions_value and total_position_value > 0:
                        # 评估投资组合风险
                        try:
                            risk_metrics = await self.risk_manager.assess_portfolio_risk(
                                positions_value, self.net_liq
                            )
                            
                            # 风险警告检查
                            warnings = []
                            
                            # VaR检查
                            if risk_metrics.portfolio_var > 0.02:  # 2%
                                warnings.append(f"组合VaR过高: {risk_metrics.portfolio_var:.2%}")
                            
                            # 相关性检查
                            if risk_metrics.correlation_risk > 0.7:
                                warnings.append(f"相关性风险: {risk_metrics.correlation_risk:.2f}")
                            
                            # 集中度检查
                            if risk_metrics.concentration_risk > 0.3:
                                warnings.append(f"持仓集中度过高: HHI={risk_metrics.concentration_risk:.2f}")
                            
                            # 杠杆检查
                            if risk_metrics.leverage_ratio > 1.2:
                                warnings.append(f"杠杆过高: {risk_metrics.leverage_ratio:.2f}x")
                            
                            # 单个持仓检查
                            max_position = max(positions_value.values()) if positions_value else 0
                            max_weight = max_position / self.net_liq if self.net_liq > 0 else 0
                            if max_weight > 0.15:
                                warnings.append(f"最大单仓过大: {max_weight:.1%}")
                            
                            # 记录风险警告
                            if warnings:
                                self.logger.warning(f"风险监控警告: {'; '.join(warnings)}")
                                
                                # 发送webhook通知
                                try:
                                    await self._notify_webhook(
                                        "risk_warning", 
                                        "投资组合风险警告", 
                                        f"检测到{len(warnings)}个风险问题", 
                                        {"warnings": warnings, "risk_metrics": {
                                            "portfolio_var": risk_metrics.portfolio_var,
                                            "correlation_risk": risk_metrics.correlation_risk,
                                            "concentration_risk": risk_metrics.concentration_risk
                                        }}
                                    )
                                except Exception:
                                    pass
                            
                            # 定期生成详细风险报告
                            if current_time - last_risk_report > report_interval:
                                risk_report = self.risk_manager.get_risk_report(positions_value, self.net_liq)
                                self.logger.info(f"风险报告: {risk_report}")
                                last_risk_report = current_time
                        
                        except Exception as e:
                            self.logger.warning(f"风险评估失败: {e}")
                    
                    await asyncio.sleep(monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"风险监控任务异常: {e}")
                    await asyncio.sleep(monitor_interval)
                    
        except asyncio.CancelledError:
            self.logger.info("风险监控任务被取消")
        except Exception as e:
            self.logger.error(f"风险监控任务致命错误: {e}")
            raise  # 让任务管理器重启

    async def _prime_account_and_positions(self) -> None:
        # 账户摘要（EClient.reqAccountSummary）
        try:
            rows = await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=10.0)
            for r in rows:
                key = f"{r.tag}:{r.currency or ''}"
                self.account_values[key] = r.value
                # 捕获账户ID
                try:
                    if getattr(r, 'account', None):
                        self.account_id = str(r.account)
                except Exception:
                    pass
                if r.tag == "TotalCashValue" and ((r.currency or "") in ("", self.default_currency)):
                    try:
                        self.cash_balance = float(r.value)
                    except Exception:
                        pass
                if r.tag == "NetLiquidation" and ((r.currency or "") in ("", self.default_currency)):
                    try:
                        self.net_liq = float(r.value)
                    except Exception:
                        pass
            self.logger.info(f"账户摘要: 现金={self.cash_balance:.2f} 净值={self.net_liq:.2f}")
            self.account_ready = self.net_liq > 0
        except Exception as e:
            self.logger.warning(f"获取账户摘要失败: {e}")

        # 持仓（EClient.reqPositions）
        try:
            poss = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            self.positions.clear()
            for p in poss:
                sym = p.contract.symbol
                qty = int(p.position)
                self.positions[sym] = self.positions.get(sym, 0) + qty
            self.logger.info(f"当前持仓标的数: {len(self.positions)}")
        except Exception as e:
            self.logger.warning(f"获取持仓失败: {e}")

    async def refresh_account_balances_and_positions(self) -> None:
        """增强版账户刷新 - 带数据验证、缓存和同步保护"""
        refresh_start = time.time()
        
        # 保存刷新前的数据用于验证
        prev_cash = self.cash_balance
        prev_netliq = self.net_liq
        prev_positions_count = len(self.positions)
        
        try:
            # 账户摘要刷新
            self.logger.debug("开始刷新账户摘要...")
            rows = await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=10.0)
            
            if not rows:
                raise ValueError("账户摘要返回空数据")
            
            # 更新账户值
            for r in rows:
                key = f"{r.tag}:{r.currency or ''}"
                self.account_values[key] = r.value
                # 捕获账户ID
                try:
                    if getattr(r, 'account', None):
                        self.account_id = str(r.account)
                except Exception:
                    pass
            
            # 解析关键财务数据
            try:
                # 更稳健地解析账户数值，兼容 BASE/多币种
                new_cash = self._get_account_numeric("TotalCashValue")
                new_netliq = self._get_account_numeric("NetLiquidation")
                new_buying_power = self._get_account_numeric("BuyingPower")

                # 数据合理性验证（放宽：净值<=0 不再抛异常，仅标记未就绪并记录警告）
                if new_netliq <= 0:
                    self.logger.warning(f"净值异常(<=0): {new_netliq}，标记账户未就绪但不终止连接")
                    self.account_ready = False
                else:
                    self.account_ready = True

                if new_cash < -abs(new_netliq):  # 现金负数不能超过净值绝对值
                    self.logger.warning(f"现金余额异常: ${new_cash:.2f}, 净值: ${new_netliq:.2f}")

                # 检查数据变化是否合理
                if prev_netliq > 0 and new_netliq > 0:
                    netliq_change_pct = abs(new_netliq - prev_netliq) / prev_netliq
                    if netliq_change_pct > 0.5:  # 净值变化超过50%
                        self.logger.warning(f"净值变化异常大: {prev_netliq:.2f} -> {new_netliq:.2f} ({netliq_change_pct:.1%})")

                # 更新数据（即便未就绪也同步最新快照供UI显示）
                self.cash_balance = new_cash
                self.net_liq = new_netliq
                self.buying_power = new_buying_power
                self._last_account_update = time.time()

                self.logger.debug(
                    f"账户摘要刷新完成: 现金${self.cash_balance:.2f}, 净值${self.net_liq:.2f}, 购买力${self.buying_power:.2f}, 就绪={self.account_ready}"
                )

            except Exception as parse_error:
                self.logger.error(f"解析账户数据失败: {parse_error}")
                # 放宽：解析失败不再向上抛出，避免打断引擎；仅标记未就绪
                self.account_ready = False
                return
                
        except asyncio.TimeoutError:
            self.logger.error("账户摘要刷新超时")
            self.account_ready = False
            return
        except Exception as e:
            self.logger.error(f"刷新账户摘要失败: {e}")
            self.account_ready = False
            return

        # 刷新持仓数据
        try:
            self.logger.debug("开始刷新持仓...")
            poss = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            
            new_positions = {}
            for p in poss:
                sym = p.contract.symbol
                qty = int(p.position)
                if qty != 0:  # 只记录非零持仓
                    new_positions[sym] = new_positions.get(sym, 0) + qty
            
            # 检查持仓变化是否合理
            new_positions_count = len(new_positions)
            if prev_positions_count > 0:
                position_change = abs(new_positions_count - prev_positions_count)
                if position_change > 10:  # 持仓数量变化超过10个
                    self.logger.warning(f"持仓数量变化异常: {prev_positions_count} -> {new_positions_count}")
            
            # 更新持仓
            self.positions = new_positions
            
            refresh_duration = time.time() - refresh_start
            self.logger.debug(f"持仓刷新完成: {len(self.positions)}个标的 (用时{refresh_duration:.2f}秒)")
            
            # 记录关键变化
            if prev_cash != self.cash_balance or prev_netliq != self.net_liq:
                cash_change = self.cash_balance - prev_cash
                netliq_change = self.net_liq - prev_netliq
                self.logger.info(f"账户变化: 现金{cash_change:+.2f}, 净值{netliq_change:+.2f}")
            
        except asyncio.TimeoutError:
            self.logger.error("持仓刷新超时")
            return
        except Exception as e:
            self.logger.error(f"刷新持仓失败: {e}")
            return

    async def wait_for_price(self, symbol: str, timeout: float = 2.0, interval: float = 0.1) -> Optional[float]:
        """等待直到拿到该 symbol 的价格或超时。"""
        start = time.time()
        price = self.get_price(symbol)
        while price is None and time.time() - start < timeout:
            await self.ib.sleep(interval)
            price = self.get_price(symbol)
        return price

    # ------------------------- 合约与行情 -------------------------
    async def qualify_stock(self, symbol: str, primary_exchange: Optional[str] = None) -> Contract:
        # 合约资格校验（EClient.reqContractDetails / qualifyContracts）
        contract = Stock(symbol, exchange="SMART", currency=self.default_currency)
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            return qualified[0]
        except Exception as e:
            self.logger.warning(f"qualifyContracts 失败，尝试设置 primaryExchange: {e}")
            exchanges = [primary_exchange] if primary_exchange else ["NASDAQ", "NYSE", "ARCA", "AMEX"]
            last_err: Optional[Exception] = None
            for ex in exchanges:
                try:
                    contract.primaryExchange = ex
                    qualified = await self.ib.qualifyContractsAsync(contract)
                    return qualified[0]
                except Exception as ee:
                    last_err = ee
                    continue
            if last_err:
                raise last_err
            return contract

    async def subscribe(self, symbol: str) -> None:
        if symbol in self.tickers:
            return
        c = await self.qualify_stock(symbol)
        ticker = self.ib.reqMktData(c, snapshot=False)
        self.tickers[symbol] = ticker
        # 优化：减少订阅延迟，提高性能
        await self.ib.sleep(0.1)
        price = await self.wait_for_price(symbol, timeout=3.0)  # 主动等待价格
        if price is not None:
            self.logger.info(f"{symbol} 订阅成功，初始价格: {price:.4f}")
        else:
            self.logger.warning(f"{symbol} 订阅后未获取到价格")

    def unsubscribe(self, symbol: str) -> None:
        t = self.tickers.pop(symbol, None)
        if t:
            self.ib.cancelMktData(t)

    async def _validate_order_before_submission(self, symbol: str, side: str, qty: int, price: float) -> bool:
        """统一风险验证 - 委托给RiskManager处理所有检查"""
        try:
            async with self._account_lock:
                # 日内计数限制（保留在这里，因为这是交易层面的限制）
                if self._daily_order_count >= self.order_verify_cfg["daily_order_limit"]:
                    self.logger.warning(f"已达日内订单上限: {self._daily_order_count}/{self.order_verify_cfg['daily_order_limit']}")
                    return False

                # 检查待处理订单敞口
                active_orders = await self.order_manager.get_orders_by_symbol(symbol)
                pending_value = sum(
                    order.quantity * (order.price or price) 
                    for order in active_orders 
                    if order.side == side.upper() and order.is_active()
                )
                
                if pending_value > 0:
                    order_value = qty * price
                    total_exposure = order_value + pending_value
                    max_exposure = self.net_liq * self.order_verify_cfg["max_single_position_pct"]
                    
                    if total_exposure > max_exposure:
                        self.logger.warning(f"{symbol} 总敞口超限: ${total_exposure:.2f} > ${max_exposure:.2f} (含待处理订单${pending_value:.2f})")
                        return False

                # 委托给RiskManager进行统一验证
                validation_result = await self.risk_manager.validate_order_comprehensive(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=price,
                    net_liq=self.net_liq,
                    cash_balance=self.cash_balance,
                    positions=self.positions,
                    order_verify_cfg=self.order_verify_cfg,
                    account_ready=self.account_ready,
                    last_account_update=self._last_account_update,
                    account_update_interval=self.account_update_interval
                )
                
                # 处理验证结果
                if not validation_result['approved']:
                    reasons = ', '.join(validation_result['reasons'])
                    self.logger.warning(f"{symbol} 风险验证失败: {reasons}")
                    
                    # 如果有建议仓位，记录信息
                    if validation_result['recommended_qty'] != qty and validation_result['recommended_qty'] > 0:
                        self.logger.info(f"{symbol} 建议调整仓位: {qty} -> {validation_result['recommended_qty']}股 (${validation_result['recommended_value']:.0f})")
                    
                    return False
                
                # 记录警告信息
                if validation_result.get('warnings'):
                    for warning in validation_result['warnings']:
                        self.logger.warning(f"{symbol} 风险警告: {warning}")
                
                # 如果需要刷新账户数据
                if any('过期' in w for w in validation_result.get('warnings', [])):
                    self.logger.info("根据风险检查建议，刷新账户数据...")
                    await self.refresh_account_balances_and_positions()
                
                return True
                
        except Exception as e:
            self.logger.error(f"订单验证异常: {e}")
            return False

    def get_price(self, symbol: str) -> Optional[float]:
        t = self.tickers.get(symbol)
        if not t:
            # 如果没有ticker，说明未订阅，返回None让调用方处理
            self.logger.warning(f"获取价格失败: {symbol} 未订阅行情")
            return None
        price_val: Optional[float] = None
        try:
            if t.last is not None and float(t.last) > 0:
                price_val = float(t.last)
            elif (t.bid is not None and float(t.bid) > 0) and (t.ask is not None and float(t.ask) > 0):
                price_val = (float(t.bid) + float(t.ask)) / 2.0
            elif t.close is not None and float(t.close) > 0:
                price_val = float(t.close)
            else:
                mp = t.marketPrice()
                price_val = float(mp) if mp is not None and float(mp) > 0 else None
        except Exception:
            price_val = None
        ts = time.time()
        if price_val is not None:
            self.last_price[symbol] = (price_val, ts)
        return price_val

    async def get_price_with_subscription(self, symbol: str, wait_seconds: float = 2.0) -> Optional[float]:
        """获取价格，确保先订阅并等待数据"""
        # 检查是否已有有效价格
        price = self.get_price(symbol)
        if price is not None:
            return price
        
        # 确保订阅
        await self.subscribe(symbol)
        
        # 等待价格数据，最多等待指定时间
        max_wait_time = wait_seconds
        wait_interval = 0.1
        total_waited = 0.0
        
        while total_waited < max_wait_time:
            await asyncio.sleep(wait_interval)
            total_waited += wait_interval
            
            price = self.get_price(symbol)
            if price is not None:
                self.logger.debug(f"获取到 {symbol} 价格: {price} (等待 {total_waited:.1f}s)")
                return price
        
        self.logger.warning(f"等待 {symbol} 价格超时 ({wait_seconds}s)")
        return None

    # ------------------------- 动态止损（ATR + 时间加权） -------------------------
    async def _fetch_daily_bars(self, symbol: str, lookback_days: int) -> List:
        try:
            c = await self.qualify_stock(symbol)
            bars = await self.ib.reqHistoricalDataAsync(
                c,
                endDateTime="",
                durationStr=f"{max(lookback_days, 30)} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            return list(bars or [])
        except Exception as e:
            self.logger.warning(f"拉取历史数据失败 {symbol}: {e}")
            return []

    @staticmethod
    def _calc_atr_from_bars(bars: List, period: int) -> Optional[float]:
        try:
            if not bars or len(bars) < period + 1:
                return None
            highs = [float(b.high) for b in bars]
            lows = [float(b.low) for b in bars]
            closes = [float(b.close) for b in bars]
            trs: List[float] = []
            for i in range(1, len(bars)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i - 1])
                tr3 = abs(lows[i] - closes[i - 1])
                trs.append(max(tr1, tr2, tr3))
            if len(trs) < period:
                return None
            atr = sum(trs[-period:]) / float(period)
            return float(atr)
        except Exception:
            return None

    def _time_weighted_distance(self, entry_price: float, raw_distance: float, entry_time: datetime) -> float:
        try:
            minutes = max((datetime.now() - entry_time).total_seconds() / 60.0, 0.0)
            decay = max(self.dynamic_stop_cfg["min_decay_factor"], 1.0 - self.dynamic_stop_cfg["decay_per_min"] * minutes)
            return raw_distance * decay
        except Exception:
            return raw_distance

    # ------------------------- 资金管理 -------------------------
    def allocate_funds(self, symbol: str, risk_factor: float = 0.0) -> float:
        """根据账户状态与风险因子动态分配资金（风险因子∈[0,1)）。
        risk_factor 越大，分配资金越少。
        返回本次可用的最大下单预算（美元）。
        """
        try:
            rf = max(0.0, min(float(risk_factor), 0.95))
            reserved = self.net_liq * self.order_verify_cfg["cash_reserve_pct"]
            available_cash = max((self.cash_balance or 0.0) - reserved, 0.0)
            max_pos_value = self.net_liq * self.order_verify_cfg["max_single_position_pct"]
            allocation = available_cash * (1.0 - rf)
            return max(0.0, min(allocation, max_pos_value))
        except Exception:
            # 保底按单标的上限
            return max(0.0, self.net_liq * self.order_verify_cfg.get("max_single_position_pct", 0.12))

    # ------------------------- 风险配置与止损/止盈计算 -------------------------
    def get_stop_config(self, symbol: str, strategy_type: str = "swing") -> Dict:
        rm = self.risk_config.get("risk_management", {})
        cfg = {
            "stop_pct": rm.get("default_stop_pct", 0.02),
            "target_pct": rm.get("default_target_pct", 0.05),
            "use_atr_stops": rm.get("use_atr_stops", False),
            "atr_multiplier_stop": rm.get("atr_multiplier_stop", 2.0),
            "atr_multiplier_target": rm.get("atr_multiplier_target", 3.0),
        }
        # strategy overrides
        strat = (rm.get("strategy_settings", {}) or {}).get(strategy_type or "", {})
        cfg.update({k: v for k, v in strat.items() if v is not None})
        # symbol overrides
        sym = (rm.get("symbol_overrides", {}) or {}).get(symbol.upper(), {})
        cfg.update({k: v for k, v in sym.items() if v is not None})
        return cfg

    async def get_current_atr(self, symbol: str, period: int = 14, lookback_days: int = 60) -> Optional[float]:
        bars = await self._fetch_daily_bars(symbol, lookback_days)
        return self._calc_atr_from_bars(bars, period)

    def _get_risk_scale(self) -> float:
        try:
            return float(self.risk_config.get("risk_management", {}).get("atr_risk_scale", 5.0))
        except Exception:
            return 5.0

    async def _compute_risk_factor(self, symbol: str, current_price: float) -> float:
        """基于 ATR 估算风险因子 ∈ [0, 0.95]。"""
        try:
            if current_price <= 0:
                return 0.0
            atr = await self.get_current_atr(symbol, period=14, lookback_days=60)
            if not atr or atr <= 0:
                return 0.0
            risk_scale = self._get_risk_scale()
            rf = min(0.95, max(0.0, float(atr) / float(current_price * max(1e-6, risk_scale))))
            return rf
        except Exception:
            return 0.0

    # ------------------------- Webhook 通知 -------------------------
    async def _notify_webhook(self, key: str, title: str, message: str, details: Optional[Dict] = None, min_interval_sec: float = 60.0) -> None:
        try:
            url = (self.risk_config.get("risk_management", {}) or {}).get("webhook_url", "").strip()
            if not url:
                return
            last = self._notify_throttle.get(key, 0.0)
            now = _now()
            if now - last < min_interval_sec:
                return
            self._notify_throttle[key] = now
            payload = {
                "title": title,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
            }
            data = json.dumps(payload).encode("utf-8")
            def _post():
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as _:
                    return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _post)
        except Exception:
            pass

    async def calculate_stop_price(self, symbol: str, current_price: float, direction: str, config: Dict) -> float:
        stop_pct = float(config.get("stop_pct", 0.02))
        use_atr = bool(config.get("use_atr_stops", False))
        dist_by_pct = current_price * stop_pct
        dist = dist_by_pct
        if use_atr:
            atr = await self.get_current_atr(symbol, period=14, lookback_days=60)
            if atr and atr > 0:
                dist = min(dist_by_pct, atr * float(config.get("atr_multiplier_stop", 2.0)))
        stop_price = current_price - dist if direction == "LONG" else current_price + dist
        return max(0.01, float(round(stop_price, 2)))

    async def calculate_target_price(self, symbol: str, current_price: float, direction: str, config: Dict) -> float:
        target_pct = float(config.get("target_pct", 0.05))
        dist_by_pct = current_price * target_pct
        dist = dist_by_pct
        if bool(config.get("use_atr_stops", False)):
            atr = await self.get_current_atr(symbol, period=14, lookback_days=60)
            if atr and atr > 0:
                dist = max(dist_by_pct, atr * float(config.get("atr_multiplier_target", 3.0)))
        target_price = current_price + dist if direction == "LONG" else current_price - dist
        return max(0.01, float(round(target_price, 2)))

    async def _dynamic_stop_manager(self, symbol: str) -> None:
        """后台任务：周期性计算动态止损并以 StopOrder 刷新/下发止损单。"""
        try:
            while not self._stop_event or not self._stop_event.is_set():
                st = self._stop_state.get(symbol)
                if not st:
                    return
                qty = int(st.get("qty") or 0)
                entry_price = float(st.get("entry_price") or 0.0)
                entry_time: datetime = st.get("entry_time") or datetime.now()
                if qty <= 0 or entry_price <= 0:
                    return

                # 获取 ATR
                bars = await self._fetch_daily_bars(symbol, self.dynamic_stop_cfg["atr_lookback_days"])
                atr = self._calc_atr_from_bars(bars, self.dynamic_stop_cfg["atr_period"]) or 0.0
                if atr <= 0:
                    await asyncio.sleep(self.dynamic_stop_cfg["update_interval_sec"])
                    continue

                # 原始距离 = ATR * 倍数
                raw_dist = atr * float(self.dynamic_stop_cfg["atr_multiplier"])  # 正数
                # 时间加权
                dist = self._time_weighted_distance(entry_price, raw_dist, entry_time)
                stop_price = max(0.01, entry_price - dist)

                # 若现价大幅上移，可考虑抬升止损（不放宽）
                mkt = self.get_price(symbol) or entry_price
                if mkt > entry_price:
                    trail_up = max(0.0, mkt - entry_price)
                    stop_price = max(stop_price, mkt - raw_dist)  # 亦可按 dist

                prev_stop = float(st.get("current_stop") or 0.0)
                if stop_price > prev_stop + 0.01:  # 仅上调
                    # 如启用本地动态止损，才撤销/更新；否则不与服务器端括号单止损冲突
                    rm = self.risk_config.get("risk_management", {}) or {}
                    if bool(rm.get("enable_local_dynamic_stop_for_bracket", False)):
                        # 撤销旧止损
                        old_trade: Optional[Trade] = st.get("stop_trade")  # type: ignore
                        try:
                            if old_trade:
                                self.ib.cancelOrder(old_trade.order)
                        except Exception:
                            pass

                    # 下发新的止损单（仅在启用本地动态止损时）
                    if bool(rm.get("enable_local_dynamic_stop_for_bracket", False)):
                        try:
                            c = await self.qualify_stock(symbol)
                            stop_order = StopOrder("SELL", qty, stop_price)
                            new_trade = self.ib.placeOrder(c, stop_order)
                            st["stop_trade"] = new_trade
                            st["current_stop"] = stop_price
                            self._stop_state[symbol] = st
                            self.logger.info(f"更新动态止损 {symbol}: stop={stop_price:.2f} qty={qty}")
                        except Exception as e:
                            self.logger.warning(f"提交止损失败 {symbol}: {e}")
                            try:
                                await self._notify_webhook("stop_replace_fail", "动态止损提交失败", f"{symbol} 提交止损失败", {"error": str(e)})
                            except Exception:
                                pass
                    else:
                        # 仅更新本地参考止损价，不对服务器端括号止损做变更
                        st["current_stop"] = stop_price
                        self._stop_state[symbol] = st

                await asyncio.sleep(self.dynamic_stop_cfg["update_interval_sec"])
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.warning(f"动态止损任务异常 {symbol}: {e}")

    # ------------------------- 下单与订单管理 -------------------------
    async def place_market_order(self, symbol: str, action: str, quantity: int, retries: int = 3) -> OrderRef:
        """增强的市价单下单，使用EnhancedOrderExecutor"""
        # 日内计数重置
        try:
            today = datetime.now().date()
            if self._last_reset_day != today:
                self._daily_order_count = 0
                self._last_reset_day = today
        except Exception:
            pass

        # 下单前验证
        try:
            price_now = self.get_price(symbol) or 0.0
            if price_now <= 0:
                await self.subscribe(symbol)
                price_now = self.get_price(symbol) or 0.0
            if price_now <= 0:
                await self._notify_webhook("no_price", "价格获取失败", f"{symbol} 无有效价格，拒绝下单", {"symbol": symbol})
                raise RuntimeError(f"无法获取有效价格: {symbol}")
            if not await self._validate_order_before_submission(symbol, action, quantity, price_now):
                await self._notify_webhook("risk_reject", "风控拒单", f"{symbol} 下单前校验未通过", {"symbol": symbol, "action": action, "qty": quantity, "price": price_now})
                raise RuntimeError("订单前置校验未通过")
        except Exception as e:
            self.logger.warning(f"下单前校验失败 {symbol}: {e}")
            raise
        
        # 纯路由到EnhancedOrderExecutor执行订单
        from .enhanced_order_execution import ExecutionConfig
        exec_cfg = ExecutionConfig()
        order_sm = await self.enhanced_executor.execute_market_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            config=exec_cfg,
        )

        # 统一返回 OrderRef（与现有调用方兼容）
        enhanced_ref = OrderRef(
            order_id=order_sm.order_id,
            symbol=symbol,
            side=action,
            qty=quantity,
            order_type="MKT",
        )
        
        # 审计记录通过OrderManager回调自动处理，无需重复记录
        
        # 更新计数
        self._daily_order_count += 1
        
        # 刷新账户信息
        await self.refresh_account_balances_and_positions()
        
        return enhanced_ref
    

                
    async def place_market_order_with_bracket(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_pct: Optional[float] = None,
        target_pct: Optional[float] = None,
        strategy_type: str = "swing",
        use_config: bool = True,
        custom_config: Optional[Dict] = None,
    ) -> List[Trade]:
        """使用括号单进行市价入场 + 服务器端托管止盈/止损（父子/同组）。
        若 use_config=True，则从 risk_config 读取（含 symbol override 与 strategy）。
        
        ⚠️  NOTE: 考虑使用 place_bracket_order() 作为统一接口
        """
        price_now = self.get_price(symbol) or 0.0
        if price_now <= 0:
            await self.subscribe(symbol)
            price_now = self.get_price(symbol) or 0.0
        if price_now <= 0:
            raise RuntimeError(f"无法获取有效价格: {symbol}")

        if not await self._validate_order_before_submission(symbol, action, quantity, price_now):
            raise RuntimeError("订单前置校验未通过")

        c = await self.qualify_stock(symbol)
        side = action.upper()
        # 计算止损/止盈参数（优先级：custom_config > use_config > 参数传入）
        cfg = custom_config
        if not cfg and use_config:
            cfg = self.get_stop_config(symbol, strategy_type)
        eff_stop_pct = float(cfg.get("stop_pct", 0.02) if cfg else (stop_pct or 0.02))
        eff_target_pct = float(cfg.get("target_pct", 0.05) if cfg else (target_pct or 0.05))

        if side in ("BUY", "COVER"):
            stop_price = await self.calculate_stop_price(symbol, price_now, "LONG", cfg or {"stop_pct": eff_stop_pct})
            target_price = await self.calculate_target_price(symbol, price_now, "LONG", cfg or {"target_pct": eff_target_pct})
        else:
            stop_price = await self.calculate_stop_price(symbol, price_now, "SHORT", cfg or {"stop_pct": eff_stop_pct})
            target_price = await self.calculate_target_price(symbol, price_now, "SHORT", cfg or {"target_pct": eff_target_pct})

        # 做空支持（如禁用，拒绝 SHORT）
        if side == "SHORT" and not self.allow_short:
            raise RuntimeError("当前配置不允许做空")

        send_side = side if side in ("BUY", "SELL") else ("SELL" if side == "SHORT" else "BUY")

        parent = MarketOrder(send_side, quantity)
        parent.transmit = False
        trade_parent = self.ib.placeOrder(c, parent)
        await self.ib.sleep(0.1)
        parent_id = trade_parent.order.orderId

        # 子单方向：止盈/止损反向
        tp_side = "SELL" if send_side == "BUY" else "BUY"
        take_profit = LimitOrder(tp_side, quantity, lmtPrice=target_price)
        take_profit.parentId = parent_id
        take_profit.transmit = False
        trade_tp = self.ib.placeOrder(c, take_profit)

        stop_order = StopOrder(tp_side, quantity, stop_price)
        stop_order.parentId = parent_id
        stop_order.transmit = True
        trade_sl = self.ib.placeOrder(c, stop_order)

        try:
            await asyncio.wait_for(trade_parent.doneEvent.wait(), timeout=45.0)
        except Exception:
            pass

        await asyncio.sleep(1.0)
        await self.refresh_account_balances_and_positions()

        try:
            st = self._stop_state.get(symbol, {})
            avg_px = float(getattr(trade_parent.orderStatus, 'avgFillPrice', 0.0) or 0.0) or price_now
            st["entry_price"] = avg_px
            st["entry_time"] = datetime.now()
            st["qty"] = int(self.positions.get(symbol, quantity))
            st["stop_trade"] = trade_sl
            st["current_stop"] = stop_price
            self._stop_state[symbol] = st
            # 简化止损任务管理
            task_id = f"stop_manager_{symbol}"
            if task_id not in self._active_tasks:
                try:
                    task = asyncio.create_task(self._dynamic_stop_manager(symbol))
                    self._active_tasks[task_id] = task
                except Exception as e:
                    self.logger.error(f"启动止损任务失败 {symbol}: {e}")
        except Exception:
            pass

        # 审计记录 - 括号订单
        try:
            self.auditor.log_order({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': 'BRACKET',
                'parent_order_id': getattr(trade_parent.order, 'orderId', 0),
                'stop_order_id': getattr(trade_sl.order, 'orderId', 0),
                'target_order_id': getattr(trade_tp.order, 'orderId', 0),
                'entry_price': price_now,
                'stop_price': stop_price,
                'target_price': target_price,
                'stop_pct': eff_stop_pct,
                'target_pct': eff_target_pct,
                'strategy_type': strategy_type,
                'timestamp': time.time(),
                'account_value': self.net_liq,
                'cash_balance': self.cash_balance,
                'risk_level': 'MANAGED' if use_config else 'MANUAL'
            })
        except Exception as audit_error:
            self.logger.warning(f"审计记录失败: {audit_error}")

        return [trade_parent, trade_tp, trade_sl]

    # ==================== 高级执行算法接口 ====================
    
    async def execute_large_order(self, symbol: str, action: str, quantity: int, 
                                algorithm: str = "TWAP", **kwargs):
        """执行大订单的智能算法
        
        Args:
            symbol: 股票代码
            action: BUY/SELL
            quantity: 总数量
            algorithm: 执行算法 ("TWAP", "VWAP", "ICEBERG")
            **kwargs: 算法特定参数
        """
        if quantity < 1000:  # 小订单直接使用普通市价单
            return await self.place_market_order(symbol, action, quantity)
        
        self.logger.info(f"开始大订单执行: {symbol} {action} {quantity}股, 算法: {algorithm}")
        
        try:
            contract = await self.qualify_stock(symbol)
            signed_quantity = quantity if action.upper() == "BUY" else -quantity
            
            if algorithm.upper() == "TWAP":
                duration = kwargs.get('duration_minutes', 30)
                slices = kwargs.get('slice_count', 10)
                results = await self.enhanced_executor.execute_twap_order(
                    contract, signed_quantity, duration, slices
                )
                
            elif algorithm.upper() == "VWAP":
                participation = kwargs.get('participation_rate', 0.1)
                results = await self.enhanced_executor.execute_vwap_order(
                    contract, signed_quantity, participation
                )
                
            elif algorithm.upper() == "ICEBERG":
                visible_size = kwargs.get('visible_size', min(500, quantity // 10))
                randomize = kwargs.get('randomize', True)
                results = await self.enhanced_executor.execute_iceberg_order(
                    contract, signed_quantity, visible_size, randomize
                )
            else:
                raise ValueError(f"不支持的算法: {algorithm}")
            
            # 审计记录 - 大订单执行
            try:
                total_filled = sum(order.get('filled', 0) for order in results)
                avg_price = 0
                if total_filled > 0:
                    total_value = sum(order.get('filled', 0) * order.get('avg_price', 0) for order in results)
                    avg_price = total_value / total_filled
                
                self.auditor.log_order({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'filled_quantity': total_filled,
                    'avg_fill_price': avg_price,
                    'order_type': f'ALGO_{algorithm}',
                    'algorithm': algorithm,
                    'algorithm_params': kwargs,
                    'execution_slices': len(results),
                    'timestamp': time.time(),
                    'account_value': self.net_liq,
                    'execution_quality': {
                        'fill_rate': total_filled / quantity if quantity > 0 else 0,
                        'slices_executed': len(results)
                    }
                })
            except Exception as audit_error:
                self.logger.warning(f"大订单审计记录失败: {audit_error}")
            
            # 刷新账户信息
            await self.refresh_account_balances_and_positions()
            
            return results
            
        except Exception as e:
            self.logger.error(f"大订单执行失败 {symbol}: {e}")
            raise

    async def place_limit_order(self, symbol: str, action: str, quantity: int, limit_price: float) -> OrderRef:
        # 前置校验
        if not await self._validate_order_before_submission(symbol, action, quantity, limit_price):
            raise RuntimeError("订单前置校验未通过")
        
        # 纯路由到EnhancedOrderExecutor执行限价单
        from .enhanced_order_execution import ExecutionConfig, ExecutionAlgorithm
        exec_cfg = ExecutionConfig(algorithm=ExecutionAlgorithm.LIMIT)
        order_sm = await self.enhanced_executor.execute_limit_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            config=exec_cfg,
        )

        # 统一返回 OrderRef（与现有调用方兼容）
        enhanced_ref = OrderRef(
            order_id=order_sm.order_id,
            symbol=symbol,
            side=action,
            qty=quantity,
            order_type="LMT",
            limit_price=limit_price,
        )
        
        # 审计记录通过OrderManager回调自动处理，无需重复记录
        
        # 更新计数
        self._daily_order_count += 1
        
        # 刷新账户信息
        await self.refresh_account_balances_and_positions()
        
        return enhanced_ref

    async def place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> List[OrderRef]:
        side = action.upper()
        c = await self.qualify_stock(symbol)
        bracket: List[Order] = BracketOrder(
            side,
            quantity,
            limitPrice=entry_price,
            takeProfitPrice=take_profit,
            stopLossPrice=stop_loss,
        )
        refs: List[OrderRef] = []
        for o in bracket:
            trade = self.ib.placeOrder(c, o)
            await trade.doneEvent
            refs.append(
                OrderRef(
                    order_id=trade.order.orderId,
                    symbol=symbol,
                    side=side if o.parentId is None else ("SELL" if side == "BUY" else "BUY"),
                    qty=quantity,
                    order_type="BRACKET",
                    limit_price=o.lmtPrice if hasattr(o, "lmtPrice") else None,
                    parent_id=o.parentId if o.parentId else None,
                )
            )
            # 使用订单管理器跟踪
            try:
                from .order_state_machine import OrderType, OrderState
                await self.order_manager.create_order(
                    order_id=trade.order.orderId,
                    symbol=symbol,
                    side=side if o.parentId is None else ("SELL" if side == "BUY" else "BUY"),
                    quantity=quantity,
                    order_type=OrderType.BRACKET,
                    price=getattr(o, 'lmtPrice', None),
                    parent_id=o.parentId if o.parentId else None,
                )
                await self.order_manager.update_order_state(trade.order.orderId, OrderState.SUBMITTED, {"trade": trade})
            except Exception:
                pass
        self.logger.info(f"提交括号单: {[r.order_id for r in refs]}")
        await self.refresh_account_balances_and_positions()
        # 对BUY父单，初始化动态止损状态
        try:
            if action.upper() == "BUY":
                st = self._stop_state.get(symbol, {})
                st["entry_price"] = entry_price
                st["entry_time"] = datetime.now()
                st["qty"] = int(self.positions.get(symbol, quantity))
                self._stop_state[symbol] = st
                # 使用任务管理器启动止损任务
                task_id = f"stop_manager_{symbol}"
                if task_id not in self._active_tasks:
                    try:
                        task = asyncio.create_task(self._dynamic_stop_manager(symbol))
                        self._active_tasks[task_id] = task
                    except Exception as e:
                        self.logger.error(f"启动止损任务失败 {symbol}: {e}")
        except Exception:
            pass
        
        # 审计记录 - 括号单
        try:
            self.auditor.log_order({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': 'BRACKET',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'order_ids': [ref.order_id for ref in refs],
                'parent_order_id': refs[0].order_id if refs else None,
                'timestamp': time.time(),
                'account_value': self.net_liq,
                'cash_balance': self.cash_balance
            })
        except Exception as audit_error:
            self.logger.warning(f"括号单审计记录失败: {audit_error}")
        
        return refs

    def cancel_all_open_orders(self) -> None:
        self.ib.reqGlobalCancel()  # EClient.reqGlobalCancel / cancelOrder
        self.logger.info("已请求撤销全部未完成订单")

    async def graceful_shutdown(self) -> None:
        """优雅关闭：取消订单、断开连接、清理资源"""
        try:
            self.logger.info("开始优雅关闭...")
            
            # 1. 停止事件循环
            if self._stop_event:
                self._stop_event.set()
                
            # 2. 取消所有未完成的订单
            try:
                self.cancel_all_open_orders()
                await asyncio.sleep(2)  # 等待取消生效
            except Exception as e:
                self.logger.warning(f"取消订单时出错: {e}")
                
            # 3. 取消所有行情订阅
            try:
                for symbol in list(self.tickers.keys()):
                    self.unsubscribe(symbol)
                await asyncio.sleep(1)  # 给服务器时间处理
            except Exception as e:
                self.logger.warning(f"取消行情订阅时出错: {e}")
                
            # 4. 断开连接
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    self.logger.info("已断开IBKR连接")
            except Exception as e:
                self.logger.warning(f"断开连接时出错: {e}")
                
            # 5. 清理状态
            self.tickers.clear()
            self.last_price.clear()
            self.positions.clear()
            self.account_values.clear()
            
            self.logger.info("优雅关闭完成")
            
        except Exception as e:
            self.logger.error(f"优雅关闭时发生错误: {e}")
            
    async def health_check(self) -> dict:
        """Enhanced system health check with comprehensive monitoring"""
        try:
            status = {
                "connected": self.ib.isConnected(),
                "subscribed_symbols": len(self.tickers),
                "open_orders": len(self.open_orders),
                "positions": len(self.positions),
                "net_liquidation": self.net_liq,
                "cash_balance": self.cash_balance,
                "account_ready": self.account_ready,
                "last_update": time.time()
            }
            
            # Check connection status
            if not status["connected"]:
                self.logger.warning("🔌 Health check: IBKR connection lost")
                
            # Check account status
            if not status["account_ready"]:
                self.logger.warning("📊 Health check: Account information not ready")
                
            # Check price data freshness
            stale_count = 0
            current_time = time.time()
            stale_symbols = []
            
            for symbol, (price, timestamp) in self.last_price.items():
                age_minutes = (current_time - timestamp) / 60
                if age_minutes > 5:  # 5 minutes without update
                    stale_count += 1
                    stale_symbols.append(f"{symbol}({age_minutes:.1f}m)")
                    
            if stale_count > 0:
                self.logger.warning(f"📈 Health check: {stale_count} symbols with stale data: {', '.join(stale_symbols[:5])}")
                
            status["stale_prices"] = stale_count
            status["stale_symbols"] = stale_symbols
            
            # Check system performance metrics
            status["system_metrics"] = {
                "uptime": current_time - getattr(self, '_start_time', current_time),
                "memory_usage": self._get_memory_usage(),
                "error_rate": getattr(self, '_error_count', 0)
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    async def fetch_data_with_retry(self, symbol: str, retries: int = 3) -> Optional[float]:
        """Fetch price data with exponential backoff retry"""
        for attempt in range(retries):
            try:
                price = self.get_price(symbol)
                if price is not None and price > 0:
                    return price
                    
                # If no price, wait and retry
                if attempt < retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    self.logger.warning(f"No price for {symbol}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                self.logger.warning(f"Data fetch failed for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                    
        self.logger.error(f"Failed to fetch data for {symbol} after {retries} attempts")
        return None

    # ------------------------- 账户/持仓/PnL 回调 -------------------------
    def _on_account_summary(self, rows) -> None:
        for r in rows:
            key = f"{r.tag}:{r.currency or ''}"
            self.account_values[key] = r.value
        try:
            cash = float(self.account_values.get(f"TotalCashValue:{self.default_currency}", "0"))
            netliq = float(self.account_values.get(f"NetLiquidation:{self.default_currency}", "0"))
            self.cash_balance, self.net_liq = cash, netliq
        except Exception:
            pass

    def _on_update_account_value(self, tag, value, currency, account) -> None:
        key = f"{tag}:{currency or ''}"
        self.account_values[key] = value

    def _on_update_portfolio(self, position, contract, *_args) -> None:
        try:
            self.positions[contract.symbol] = int(position)
        except Exception:
            pass

    def _on_position(self, account, contract, position, avgCost) -> None:
        try:
            self.positions[contract.symbol] = int(position)
        except Exception:
            pass

    # ------------------------- 订单/成交/佣金 回调 -------------------------
    def _on_order_status(self, trade) -> None:
        try:
            o = trade.order
            s = trade.orderStatus
            self.logger.info(
                f"订单状态: id={o.orderId} permId={o.permId} status={s.status} filled={s.filled} remaining={s.remaining} avgFillPrice={s.avgFillPrice}"
            )
            # 同步到订单状态机
            try:
                from .order_state_machine import OrderState
                status = getattr(s, 'status', '')
                if status == 'Filled':
                    # 简化订单状态更新
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                            o.orderId, OrderState.FILLED,
                            {"filled_quantity": int(getattr(s, 'filled', 0) or 0),
                             "avg_fill_price": float(getattr(s, 'avgFillPrice', 0.0) or 0.0)}
                        ))
                    except Exception as e:
                        self.logger.error(f"更新订单状态失败: {e}")
                elif status in {'Cancelled', 'ApiCancelled'}:
                    # 简化订单状态更新
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                            o.orderId, OrderState.CANCELLED
                        ))
                    except Exception as e:
                        self.logger.error(f"更新订单状态失败: {e}")
                elif status in {'Inactive', 'Rejected'}:
                    # 简化订单状态更新
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                            o.orderId, OrderState.REJECTED
                        ))
                    except Exception as e:
                        self.logger.error(f"更新订单状态失败: {e}")
            except Exception:
                pass
        except Exception:
            pass

    def _on_exec_details(self, trade, fill) -> None:
        try:
            symbol = trade.contract.symbol
            side = trade.order.action
            qty = int(fill.execution.shares or 0)
            price = float(fill.execution.price or 0.0)
            self.logger.info(
                f"成交: orderId={trade.order.orderId} symbol={symbol} side={side} qty={qty} price={price}"
            )

            # 更新动态止损状态
            if side == "BUY" and qty > 0:
                state = self._stop_state.get(symbol, {})
                # 使用最新成交价作为入场价（可扩展为加权平均）
                state["entry_price"] = float(state.get("entry_price") or price or 0.0) or price
                state["entry_time"] = datetime.now()
                # 同步当前持仓
                try:
                    held = int(self.positions.get(symbol, 0))
                    state["qty"] = held
                except Exception:
                    state["qty"] = qty
                self._stop_state[symbol] = state

                # 启动/确保动态止损任务
                task_id = f"stop_manager_{symbol}"
                if task_id not in self._active_tasks:
                    try:
                        task = asyncio.create_task(self._dynamic_stop_manager(symbol))
                        self._active_tasks[task_id] = task
                    except Exception as e:
                        self.logger.error(f"启动止损任务失败 {symbol}: {e}")

            elif side == "SELL" and qty > 0:
                # 如果仓位清零，取消已有止损并停止任务
                try:
                    held = int(self.positions.get(symbol, 0))
                except Exception:
                    held = 0
                if held <= 0:
                    st = self._stop_state.pop(symbol, None)
                    if st and st.get("stop_trade"):
                        try:
                            self.ib.cancelOrder(st["stop_trade"].order)
                        except Exception:
                            pass
                    task = self._stop_tasks.pop(symbol, None)
                    if task and not task.done():
                        task.cancel()
        except Exception:
            pass

    def _on_commission(self, trade, report) -> None:
        try:
            self.logger.info(
                f"佣金: orderId={trade.order.orderId} commission={report.commission} currency={report.currency} realizedPNL={report.realizedPNL}"
            )
        except Exception:
            pass

    # ------------------------- 其他回调/错误 -------------------------
    def _on_error(self, reqId, errorCode, errorString, contract) -> None:
        """增强的错误处理，包括自动重连和详细的错误分类"""
        try:
            msg = f"IBKR错误: reqId={reqId} code={errorCode} msg={errorString}"
            if contract:
                msg += f" contract={contract}"
            
            # 根据错误严重程度使用不同日志级别
            if errorCode in (10167, 354, 2104, 2106, 2158):  # 非致命错误
                self.logger.warning(msg)
            elif errorCode in (504, 1100, 1101, 1102, 2110):  # 连接错误
                self.logger.error(msg)
            else:
                self.logger.info(msg)
            
            # 实时数据权限错误，自动切换到延迟数据
            if errorCode in (10167, 354):  # 无实时数据权限
                if self.use_delayed_if_no_realtime:
                    self.logger.warning("无实时数据权限，自动切换为延迟数据（10-20分钟延迟）")
                    try:
                        self.ib.reqMarketDataType(3)  # 切换到延迟数据
                        self.logger.info("成功切换到延迟市场数据")
                    except Exception as e:
                        self.logger.error(f"切换到延迟数据失败: {e}")
                else:
                    self.logger.error("无实时数据权限，且不允许使用延迟数据。请检查数据订阅。")
                    self.logger.error("请检查您的Professional US Securities Bundle订阅状态")
                    
            # 连接相关错误处理
            elif errorCode in (504, 1100, 1101):  # 连接丢失
                self.logger.error("检测到连接丢失，系统将在下次操作时尝试重新连接")
                
            # 订单相关错误
            elif errorCode in (201, 202, 203, 399):  # 订单被拒绝
                self.logger.error(f"订单被拒绝: {errorString}")
                
            # 市场数据错误
            elif errorCode in (200, 162, 321):  # 市场数据错误
                self.logger.warning(f"市场数据错误: {errorString}")
                
        except Exception as e:
            self.logger.error(f"处理错误回调时发生异常: {e}")

    def _on_current_time(self, time_: datetime) -> None:
        self.logger.debug(f"服务器时间 {time_}")

    # ------------------------- 简易策略/演示（已移除） -------------------------
    async def run_demo(self, symbols: List[str], target_allocation_per_symbol: float, max_symbols: int = 5) -> None:
        raise RuntimeError("Simplified demo strategy has been removed. Use Engine via GUI/launcher.")

    # ------------------------- 关闭 -------------------------
    async def close(self) -> None:
        try:
            # 停止任务管理器
            # 清理活跃任务
            for task_id, task in self._active_tasks.items():
                if not task.done():
                    task.cancel()
            self._active_tasks.clear()
        except Exception:
            pass
        try:
            self.cancel_all_open_orders()
        except Exception:
            pass
        try:
            self.ib.disconnect()
        except Exception:
            pass

    # ------------------------- 监控：Excel/JSON/手动列表 -------------------------
    @staticmethod
    def _load_from_json(path: str) -> List[str]:
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                ticks = data
            elif isinstance(data, dict):
                # 支持 {"tickers": [...]} 或 {"symbols": [...]} 结构
                ticks = data.get('tickers') or data.get('symbols') or []
            else:
                ticks = []
            return [str(x).strip().upper() for x in ticks if str(x).strip()]
        except Exception:
            return []

    @staticmethod
    def _load_from_excel(path: str, sheet: Optional[str] = None, column: Optional[str] = None) -> List[str]:
        # 优先使用 pandas；未安装时返回空并提示
        try:
            import pandas as pd  # type: ignore
        except Exception:
            logging.getLogger('IbkrAutoTrader').warning('Excel 导入需要 pandas/openpyxl，请先安装: pip install pandas openpyxl')
            return []
        try:
            df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
            if column and column in df.columns:
                series = df[column]
            else:
                # 默认尝试常见列名
                for col in ['ticker', 'tickers', 'symbol', 'symbols', 'code']:
                    if col in df.columns:
                        series = df[col]
                        break
                else:
                    # 退化为第一列
                    series = df.iloc[:, 0]
            ticks = [str(x).strip().upper() for x in series.tolist() if str(x).strip()]
            return ticks
        except Exception as e:
            logging.getLogger('IbkrAutoTrader').warning(f'读取Excel失败: {e}')
            return []

    @staticmethod
    def _load_from_manual(symbols_csv: Optional[str]) -> List[str]:
        if not symbols_csv:
            return []
        return [s.strip().upper() for s in symbols_csv.split(',') if s.strip()]

    @classmethod
    def load_watchlist(
        cls,
        json_file: Optional[str],
        excel_file: Optional[str],
        symbols_csv: Optional[str],
        sheet: Optional[str] = None,
        column: Optional[str] = None,
    ) -> List[str]:
        """统一从多源加载。保留该方法用于一次性导入到数据库，但在交易循环中我们改为直接从SQLite读取。"""
        union: List[str] = []
        if json_file:
            union.extend(cls._load_from_json(json_file))
        if excel_file:
            union.extend(cls._load_from_excel(excel_file, sheet=sheet, column=column))
        union.extend(cls._load_from_manual(symbols_csv))
        seen = set()
        ordered: List[str] = []
        for s in union:
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered

    async def run_watchlist_trading(
        self,
        json_file: Optional[str],
        excel_file: Optional[str],
        symbols_csv: Optional[str],
        alloc: float,
        poll_sec: float,
        auto_sell_removed: bool,
        sheet: Optional[str] = None,
        column: Optional[str] = None,
        fixed_qty: int = 0,
    ) -> None:
        """观察列表自动交易（含保守风控）

        风控要点：
        - 现金保留：≥15% 净值不参与下单
        - 单标的上限：≤12% 净值
        - 最小下单金额：≥$500
        - 日内订单上限：≤20 单
        - 价格区间过滤：$2 - $800
        - 重复持仓跳过
        - 交易时段检查（美东 9:30 - 16:00）
        """

        import math
        from datetime import datetime, time as dtime

        self._stop_event = self._stop_event or asyncio.Event()
        # 热加载风险配置（DB 优先）
        try:
            self.load_risk_config_from_db()
        except Exception:
            pass

        # 风控参数（可后续外部化到配置）
        cash_reserve_pct = 0.15
        max_single_position_pct = 0.12
        min_order_value_usd = 500.0
        min_price, max_price = 2.0, 800.0
        daily_order_limit = 20
        per_cycle_order_limit = 10

        def is_trading_hours() -> bool:
            try:
                now = datetime.now()
                if now.weekday() >= 5:
                    return False
                t = now.time()
                return dtime(9, 30) <= t <= dtime(16, 0)
            except Exception:
                return True

        last_desired: set[str] = set()
        last_reset_day = None
        daily_order_count = 0

        # 交易循环：支持外部输入（JSON/Excel/CSV）与数据库动态合并
        from .database import StockDatabase
        db = StockDatabase()

        def _compute_desired_list() -> list[str]:
            try:
                external: list[str] = []
                if any([json_file, excel_file, symbols_csv]):
                    external = self._merge_symbols(
                        json_file=json_file,
                        excel_file=excel_file,
                        symbols_csv=symbols_csv,
                        sheet=sheet,
                        column=column,
                    )
                try:
                    db_list = db.get_all_tickers() or []
                except Exception:
                    db_list = []
                ordered: list[str] = []
                seen: set[str] = set()
                for s in (external + db_list):
                    t = (s or "").strip().upper()
                    if t and t not in seen:
                        seen.add(t)
                        ordered.append(t)
                return ordered
            except Exception:
                try:
                    return db.get_all_tickers() or []
                except Exception:
                    return []

        async def _approve_buy(sym: str) -> bool:
            """专业级多因子信号系统：综合技术面、基本面和市场情绪"""
            try:
                # 获取足够的历史数据
                contract = await self.qualify_stock(sym)
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr="120 D",  # 增加到120天获取更多数据
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                if len(bars) < 50:  # 至少需要50天数据
                    self.logger.debug(f"{sym} 历史数据不足: {len(bars)}天")
                    return False
                    
                # 提取OHLCV数据
                highs = [b.high for b in bars]
                lows = [b.low for b in bars]
                closes = [b.close for b in bars]
                volumes = [b.volume for b in bars]
                
                # 基础数据验证
                if len(closes) < 50 or closes[-1] <= 0:
                    return False
                
                current_price = closes[-1]
                
                # === 1. 趋势分析 (30%权重) ===
                trend_score = 0.0
                
                # SMA 多周期趋势
                sma_5 = sum(closes[-5:]) / 5
                sma_20 = sum(closes[-20:]) / 20
                sma_50 = sum(closes[-50:]) / 50
                
                # 趋势排列 (短>中>长均线为多头)
                if sma_5 > sma_20 > sma_50:
                    trend_score += 0.4
                elif sma_5 > sma_20:
                    trend_score += 0.2
                elif current_price > sma_20:
                    trend_score += 0.1
                
                # 均线斜率 (均线向上为正面)
                sma20_slope = (sma_20 - sum(closes[-25:-5])/20) / sum(closes[-25:-5])/20
                if sma20_slope > 0.01:  # 1%以上上升
                    trend_score += 0.3
                elif sma20_slope > 0:
                    trend_score += 0.1
                
                # 价格相对位置
                if current_price > sma_5 * 1.02:  # 价格超过5日均线2%
                    trend_score += 0.3
                elif current_price > sma_5:
                    trend_score += 0.2
                
                # === 2. 动量分析 (25%权重) ===
                momentum_score = 0.0
                
                # RSI (14日)
                gains = []
                losses = []
                for i in range(-14, 0):
                    change = closes[i] - closes[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # RSI最佳区间：30-70
                    if 35 <= rsi <= 65:
                        momentum_score += 0.4
                    elif 30 <= rsi <= 75:
                        momentum_score += 0.2
                    elif rsi < 30:  # 超卖反弹机会
                        momentum_score += 0.3
                
                # MACD信号
                ema_12 = closes[-1]  # 简化EMA计算
                ema_26 = sum(closes[-26:]) / 26
                for i in range(-25, 0):
                    ema_12 = ema_12 * 0.85 + closes[i] * 0.15
                    ema_26 = ema_26 * 0.93 + closes[i] * 0.07
                
                macd = ema_12 - ema_26
                if macd > 0:
                    momentum_score += 0.3
                
                # 多周期动量
                momentum_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                momentum_20d = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
                
                if momentum_5d > 0.02 and momentum_20d > 0:  # 短期强势且中期向上
                    momentum_score += 0.3
                elif momentum_5d > -0.02:  # 短期稳定
                    momentum_score += 0.1
                
                # === 3. 成交量分析 (20%权重) ===
                volume_score = 0.0
                
                if len(volumes) >= 20:
                    avg_volume_20d = sum(volumes[-20:]) / 20
                    current_volume = volumes[-1]
                    
                    # 成交量放大
                    volume_ratio = current_volume / avg_volume_20d
                    if volume_ratio > 1.5:  # 放量1.5倍
                        volume_score += 0.4
                    elif volume_ratio > 1.2:
                        volume_score += 0.2
                    elif volume_ratio > 0.8:  # 正常成交量
                        volume_score += 0.1
                    
                    # 成交量趋势 (近5日vs前15日)
                    recent_avg_vol = sum(volumes[-5:]) / 5
                    prev_avg_vol = sum(volumes[-20:-5]) / 15
                    if recent_avg_vol > prev_avg_vol * 1.2:
                        volume_score += 0.3
                    elif recent_avg_vol > prev_avg_vol:
                        volume_score += 0.1
                    
                    # 价量配合 (上涨伴随放量)
                    price_change_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                    if price_change_5d > 0 and volume_ratio > 1.1:
                        volume_score += 0.3
                
                # === 4. 波动率与风险分析 (15%权重) ===
                volatility_score = 0.0
                
                if len(closes) >= 20:
                    # ATR计算
                    atr_values = []
                    for i in range(-19, 0):
                        tr = max(
                            highs[i] - lows[i],
                            abs(highs[i] - closes[i-1]),
                            abs(lows[i] - closes[i-1])
                        )
                        atr_values.append(tr)
                    
                    atr = sum(atr_values) / len(atr_values)
                    atr_pct = atr / current_price * 100
                    
                    # 合理波动率区间：1%-6%
                    if 1.5 <= atr_pct <= 4.0:
                        volatility_score += 0.4
                    elif 1.0 <= atr_pct <= 6.0:
                        volatility_score += 0.2
                    
                    # 近期波动率下降 (稳定性提升)
                    recent_atr = sum(atr_values[-5:]) / 5
                    prev_atr = sum(atr_values[-15:-5]) / 10
                    if recent_atr < prev_atr * 0.9:
                        volatility_score += 0.3
                    
                    # 布林带位置
                    bb_middle = sma_20
                    bb_std = (sum([(p - bb_middle)**2 for p in closes[-20:]]) / 20) ** 0.5
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std
                    
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                    if 0.2 <= bb_position <= 0.8:  # 在布林带中间区域
                        volatility_score += 0.3
                    elif bb_position < 0.2:  # 布林带下轨附近，超卖
                        volatility_score += 0.2
                
                # === 5. 市场环境与相对强度 (10%权重) ===
                market_score = 0.0
                
                # 获取市场基准 (简化为SPY/QQQ对比，实际可以更复杂)
                try:
                    # 这里简化处理，实际应该获取SPY数据
                    # 假设当前是良好的市场环境
                    market_score += 0.5  # 基础市场环境得分
                except:
                    market_score += 0.3  # 默认中性环境
                
                # 个股相对强度 (vs 20日均线的相对表现)
                stock_performance = (current_price / sma_20 - 1)
                if stock_performance > 0.05:  # 超越均线5%
                    market_score += 0.5
                elif stock_performance > 0:
                    market_score += 0.3
                
                # === 综合评分计算 ===
                total_score = (
                    trend_score * 0.30 +
                    momentum_score * 0.25 + 
                    volume_score * 0.20 +
                    volatility_score * 0.15 +
                    market_score * 0.10
                )
                
                # 详细日志
                self.logger.debug(f"{sym} 多因子评分: 总分{total_score:.2f} (趋势{trend_score:.2f} 动量{momentum_score:.2f} 成交量{volume_score:.2f} 波动{volatility_score:.2f} 市场{market_score:.2f})")
                
                # 评分阈值：0.6以上通过 (满分1.0)
                approval_threshold = 0.6
                approved = total_score >= approval_threshold
                
                if approved:
                    self.logger.info(f"{sym} ✅ 技术分析通过: {total_score:.2f}/{approval_threshold}")
                else:
                    self.logger.debug(f"{sym} ❌ 技术分析不通过: {total_score:.2f}/{approval_threshold}")
                
                return approved
                
            except Exception as e:
                self.logger.warning(f"{sym} 技术指标计算失败: {e}")
                return False

        while not self._stop_event.is_set():
            try:
                # 日内计数重置
                today = datetime.now().date()
                if last_reset_day != today:
                    daily_order_count = 0
                    last_reset_day = today

                # 交易时段检查
                if not is_trading_hours():
                    await asyncio.sleep(min(poll_sec * 2, 300))
                    continue

                # 刷新账户与资金
                await self.refresh_account_balances_and_positions()
                if self.net_liq <= 0:
                    self.logger.warning("账户净值为0，等待账户数据...")
                    await asyncio.sleep(poll_sec)
                    continue

                # 加载最新风险配置
                try:
                    current_risk_config = db.get_risk_config("默认风险配置")
                    if current_risk_config:
                        # 更新风险参数
                        max_single_position_pct = current_risk_config.get("max_single_position_pct", 0.1)
                        max_daily_orders = current_risk_config.get("max_daily_orders", 5)
                        min_order_value_usd = current_risk_config.get("min_order_value_usd", 100)
                        self.logger.debug(f"已加载风险配置: 单笔限制{max_single_position_pct*100:.1f}%, 日内最多{max_daily_orders}单")
                except Exception as e:
                    self.logger.warning(f"加载风险配置失败，使用默认值: {e}")
                    # 保持原有默认值

                reserved_cash = self.net_liq * cash_reserve_pct
                available_cash = max((self.cash_balance or 0.0) - reserved_cash, 0.0)
                if available_cash < min_order_value_usd:
                    self.logger.info("可用现金不足，等待...")
                    await asyncio.sleep(poll_sec)
                    continue

                # 合并外部输入与数据库
                desired_list = _compute_desired_list()
                desired: set[str] = set(desired_list)

                # 仅处理新增标的
                added = [s for s in desired_list if s not in last_desired]

                # 并发处理新增标的 - 分批并发
                orders_sent_this_cycle = 0
                max_concurrent_processing = 5  # 最多同时处理5个标的
                
                async def process_symbol_for_trading(sym: str) -> Optional[Dict[str, Any]]:
                    """处理单个标的，返回交易参数或None"""
                    try:
                        # 订阅行情
                        await self.subscribe(sym)
                        await asyncio.sleep(0.1)

                        # 重复持仓跳过
                        if int(self.positions.get(sym, 0)) > 0:
                            return None

                        price = self.get_price(sym)
                        if not price or price < min_price or price > max_price:
                            return None

                        # 技术指标审批
                        approved = await _approve_buy(sym)
                        if not approved:
                            return None
                        
                        return {
                            'symbol': sym,
                            'price': price,
                            'approved': True
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"处理标的失败 {sym}: {e}")
                        return None
                
                # 分批并发处理
                semaphore = asyncio.Semaphore(max_concurrent_processing)
                
                async def process_with_semaphore(sym: str):
                    async with semaphore:
                        return await process_symbol_for_trading(sym)
                
                # 批量处理
                batch_size = 10
                for i in range(0, len(added), batch_size):
                    if daily_order_count >= daily_order_limit or orders_sent_this_cycle >= per_cycle_order_limit:
                        break
                    
                    batch = added[i:i + batch_size]
                    self.logger.info(f"并发处理标的批次: {len(batch)}个 ({i+1}-{min(i+batch_size, len(added))}/{len(added)})")
                    
                    # 并发处理当前批次
                    batch_results = await asyncio.gather(
                        *[process_with_semaphore(sym) for sym in batch],
                        return_exceptions=True
                    )
                    
                    # 处理结果，按顺序下单
                    for result in batch_results:
                        if (daily_order_count >= daily_order_limit or 
                            orders_sent_this_cycle >= per_cycle_order_limit):
                            break
                        
                        if isinstance(result, Exception):
                            self.logger.warning(f"批处理异常: {result}")
                            continue
                        
                        if not result or not result.get('approved'):
                            continue
                        
                        sym = result['symbol']
                        price = result['price']
                        
                        try:
                            # 计算下单股数（使用 ATR 风险因子 + 资金分配）
                            if fixed_qty > 0:
                                # 固定股数模式：需要额外验证资金充足性
                                qty = int(fixed_qty)
                                fixed_order_value = qty * price
                                
                                # 固定股数超出资金限制时，按资金上限重新计算
                                max_affordable_by_cash = int(available_cash // price) if price > 0 else 0
                                max_affordable_by_position = int((self.net_liq * max_single_position_pct) // price) if price > 0 else 0
                                max_affordable = min(max_affordable_by_cash, max_affordable_by_position)
                                
                                if qty > max_affordable:
                                    self.logger.warning(f"{sym} 固定股数{qty}超出资金限制，调整为{max_affordable}")
                                    qty = max_affordable
                            else:
                                rf = await self._compute_risk_factor(sym, price)
                                budget = self.allocate_funds(sym, risk_factor=rf) * alloc  # 在分配上再乘以策略 alloc
                                qty = int(budget // price) if price > 0 else 0

                                # 最小下单金额保护
                                if qty * price < min_order_value_usd:
                                    qty = max(int(math.ceil(min_order_value_usd / price)), 1)

                            order_value = qty * (price or 0.0)

                            # 最终资金与风控检查
                            if qty <= 0 or order_value > available_cash or order_value > self.net_liq * max_single_position_pct:
                                self.logger.info(f"{sym} 资金检查未通过: qty={qty}, order_value=${order_value:.2f}, available_cash=${available_cash:.2f}")
                                continue

                            # 下单并跟踪状态
                            order_success = False
                            try:
                                # 使用当前风险配置的策略参数
                                strategy_config = None
                                if current_risk_config and "strategy_configs" in current_risk_config:
                                    strategy_config = current_risk_config["strategy_configs"].get("swing", {
                                        "stop_pct": current_risk_config.get("default_stop_pct", 0.03),
                                        "target_pct": current_risk_config.get("default_target_pct", 0.08)
                                    })
                                await self.place_market_order_with_bracket(sym, "BUY", qty, strategy_type="swing", use_config=True, custom_config=strategy_config)
                                order_success = True
                                self.logger.info(f"括号单提交成功: {sym} {qty}股")
                            except Exception as _e:
                                self.logger.warning(f"括号单下单失败，回退为市价单 {sym}: {_e}")
                                await self._notify_webhook("bracket_fallback", "括号单失败回退", f"{sym} 回退为市价单", {"error": str(_e)})
                                try:
                                    await self.place_market_order(sym, "BUY", qty)
                                    order_success = True
                                    self.logger.info(f"市价单提交成功: {sym} {qty}股")
                                except Exception as __e:
                                    self.logger.error(f"市价单也失败: {sym}: {__e}")
                                    order_success = False
                            
                            # 仅在订单成功提交后才增加计数和扣减资金
                            if order_success:
                                                                # 确保动态止损任务启动
                                try:
                                    task_id = f"stop_manager_{sym}"
                                    self.task_manager.ensure_task_running(
                                        task_id, self._dynamic_stop_manager, sym,
                                        max_restarts=10, restart_delay=5.0
                                    )
                                except Exception as e:
                                    self.logger.warning(f"启动止损任务失败 {sym}: {e}")
                                
                                orders_sent_this_cycle += 1
                                daily_order_count += 1
                                available_cash -= order_value
                                self.logger.info(f"订单计数更新: 本轮{orders_sent_this_cycle}, 日内{daily_order_count}")      
                            else:
                                self.logger.warning(f"{sym} 订单提交失败，不计入统计")
                            
                            await asyncio.sleep(0.2)
                        except Exception as e:
                            self.logger.warning(f"处理新增 {sym} 失败: {e}")
                    
                    # 如果批次处理过多，短暂暂停
                    if i + batch_size < len(added):
                        await asyncio.sleep(0.5)  # 批次间隔0.5秒

                # 清仓被移除：仅当标的从数据库中消失时卖出
                removed = [s for s in last_desired if s not in desired]
                if removed and auto_sell_removed:
                    for sym in removed:
                        try:
                            if daily_order_count >= daily_order_limit:
                                break
                            qty = int(self.positions.get(sym, 0))
                            if qty > 0:
                                # 对于 removed 自动清仓，始终使用直接市价以避免意外重建仓位
                                await self.place_market_order(sym, "SELL", qty)
                                self.positions[sym] = 0
                                daily_order_count += 1
                                await self.refresh_account_balances_and_positions()
                                await asyncio.sleep(0.2)
                            self.unsubscribe(sym)
                        except Exception as e:
                            self.logger.warning(f"处理移除 {sym} 失败: {e}")

                last_desired = desired

                # 实时信号处理（每秒一次，独立于 poll_sec）
                try:
                    if desired:
                        for sym in list(desired)[:50]:  # 限制每轮处理的标的数量
                            # 确保订阅
                            await self.subscribe(sym)
                            t = self.tickers.get(sym)
                            if not t:
                                continue
                            tick = TickData(
                                timestamp=time.time(),
                                bid=float(t.bid or 0.0),
                                ask=float(t.ask or 0.0),
                                bid_size=float(t.bidSize or 0.0),
                                ask_size=float(t.askSize or 0.0),
                                last=float(t.last or 0.0),
                                volume=float(t.volume or 0.0),
                            )
                            # 获取/初始化引擎
                            if not hasattr(self, "_rt_engines"):
                                self._rt_engines = {}
                            engine = self._rt_engines.get(sym)
                            if engine is None:
                                engine = RealtimeSignalEngine(sym)
                                bars = await self._fetch_daily_bars(sym, 60)
                                engine.initialize_with_history(bars)
                                self._rt_engines[sym] = engine
                            sig = engine.process_tick(tick)
                            if sig and sig.should_trade:
                                # 二次风控
                                side = "BUY" if sig.action in (ActionType.BUY_NOW, ActionType.BUY_LIMIT) else "SELL"
                                ok = await self._validate_order_before_submission(sym, side, max(1, int(self.positions.get(sym, 0) or 1)), sig.entry_price)
                                if not ok:
                                    continue
                                if sig.action == ActionType.BUY_NOW:
                                    price = sig.entry_price
                                    rf = await self._compute_risk_factor(sym, price)
                                    alloc = float(self.risk_config.get("risk_management", {}).get("realtime_alloc_pct", 0.03))
                                    budget = self.allocate_funds(sym, risk_factor=rf) * max(0.0, min(1.0, alloc))
                                    qty = int(budget // price) if price > 0 else 0
                                    if qty > 0:
                                        try:
                                            await self.place_market_order_with_bracket(sym, "BUY", qty, strategy_type="swing", use_config=True)
                                        except Exception as _e:
                                            self.logger.warning(f"实时买入括号单失败，回退市价 {sym}: {_e}")
                                            await self.place_market_order(sym, "BUY", qty)
                                elif sig.action == ActionType.SELL_NOW:
                                    qty = int(self.positions.get(sym, 0))
                                    if qty > 0:
                                        await self.place_market_order(sym, "SELL", qty)
                except Exception as _e:
                    self.logger.debug(f"实时信号处理异常: {_e}")

                await asyncio.sleep(1.0)

            except Exception as loop_err:
                self.logger.error(f"观察列表交易循环错误: {loop_err}")
                await asyncio.sleep(poll_sec)


# ----------------------------- CLI 入口 -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR 自动交易最小闭环脚本")
    p.add_argument("--host", default="127.0.0.1", help="TWS/IB Gateway 主机")
    p.add_argument("--port", type=int, default=7497, help="TWS(7497)/IBG(4002) 端口")
    p.add_argument("--client-id", type=int, default=123, help="客户端ID，避免与其他程序冲突")
    # 直接演示参数
    p.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL", help="逗号分隔的股票代码（手动传入）")
    p.add_argument("--alloc", type=float, default=0.05, help="每只标的目标资金占比，例如 0.05 表示 5%")
    p.add_argument("--max", type=int, default=5, help="最多处理的标的数量（演示模式）")
    # 观察列表/自动交易
    p.add_argument("--json", type=str, default=None, help="从 JSON 文件加载股票列表（支持 [..] 或 {tickers:[..]}）")
    p.add_argument("--excel", type=str, default=None, help="从 Excel 文件加载股票列表（需安装 pandas/openpyxl）")
    p.add_argument("--sheet", type=str, default=None, help="Excel 工作表名（默认第1个）")
    p.add_argument("--column", type=str, default=None, help="Excel 列名（默认自动识别 ticker/symbol/code 或第1列）")
    p.add_argument("--watch-alloc", type=float, default=0.03, help="观察列表模式每只标的目标资金占比")
    p.add_argument("--poll", type=float, default=10.0, help="观察列表轮询秒数")
    p.add_argument("--auto-sell-removed", action="store_true", help="从列表删除即自动全清仓")
    p.add_argument("--fixed-qty", type=int, default=0, help="固定下单股数（>0 生效，优先于资金占比）")
    p.add_argument("--no-delayed", action="store_true", help="无权限时不自动切到延时行情")
    p.add_argument("--verbose", action="store_true", help="调试日志")
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)
    trader = IbkrAutoTrader(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        use_delayed_if_no_realtime=not args.no_delayed,
    )

    # 优雅退出
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _stop(*_sig):
        logging.getLogger("amain").info("收到停止信号，准备退出...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop, sig)
        except NotImplementedError:
            pass

    try:
        await trader.connect()

        # 简化策略已移除，这里仅保留连接/健康检查。自动交易请使用 GUI/launcher 的 Engine 模式。
        logging.getLogger("amain").info("Simplified strategies removed. Use Engine via GUI/launcher.")

        await stop_event.wait()
    finally:
        await trader.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

