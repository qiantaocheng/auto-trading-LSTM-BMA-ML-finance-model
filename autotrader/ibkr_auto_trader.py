"""
IBKR automated trading minimal closed-loop script（connection→market data→order placement/order cancellation→reports→account/positions→risk control/tools）

Based on ib_insync wrapper for TWS API（EClient/EWrapper）。Covers common automated trading use cases：
- Connection/reconnection, market data type switching（real-time/delayed）
- Contract qualification verification, primary exchange settings
- Market data subscription and price retrieval（Ticker 事件），depthexpandable as needed
- Account summary/account updates, positions, PnL（can选）
- order placement（market/limit/bracket orders）、cancel orders, order/execution/commission reports
- Simple risk control（fund allocation ratio/position checks/order deduplication）

参考：
- IBKR Campus TWS API 文档（EClient/EWrapper）
  https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#api-introduction
- EClient 类参考: https://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html
- EWrapper 接口参考: https://interactivebrokers.github.io/tws-api/interfaceIBApi_1_1EWrapper.html
"""

# 清理：移除未使use导入
# from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
# 清理：移除未使use导入
# import sys
import time
from dataclasses import dataclass
# from dataclasses import field  # 未使use
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque
from enum import Enum
import os
import json
import sys
# 清理：移除未使use导入
# import urllib.request
# import urllib.error
from time import time as _now

# 添加Polygon数据源集成andrisk control收益平衡器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from polygon_client import polygon_client, download, Ticker
    from data_source_manager import get_data_source_manager
    from polygon_unified_factors import (
        get_polygon_unified_factors,
        enable_polygon_factors,
        enable_polygon_risk_balancer,
        disable_polygon_risk_balancer,
        check_polygon_trading_conditions,
        process_signals_with_polygon
    )
    POLYGON_INTEGRATED = True
except ImportError as e:
    logging.warning(f"Polygon集成failed: {e}")
    POLYGON_INTEGRATED = False

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


# 日志配置移至launcher.pyandbacktest_engine.pyin统一管理


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


# ----------------------------- real-time信号/数据结构 -----------------------------
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
    """轻量real-time信号引擎（每 seconds）"""
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
        # 使use统一配置管理器
        if config_manager is None:
            from .unified_config import get_unified_config
            config_manager = get_unified_config()
        
        self.config_manager = config_manager
        
        # from统一配置retrievalconnection参数，不自动分配Client ID
        conn_params = config_manager.get_connection_params(auto_allocate_client_id=False)
        self.host = conn_params['host']
        self.port = conn_params['port']
        self.client_id = conn_params['client_id']
        self.account_id = conn_params['account_id']
        self.use_delayed_if_no_realtime = conn_params['use_delayed_if_no_realtime']
        self.default_currency = "USD"

        # 允许外部传入共享connection
        self.ib = ib_client if ib_client is not None else IB()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化稳健account数据管理器
        from .account_data_manager import RobustAccountDataManager
        self.account_manager = RobustAccountDataManager(self.ib, self.account_id)

        # 兼容处理：预初始化 wrapper/_results and decoder handlers，避免老版本 ib_insync 报错
        try:
            # 确保 completedOrders/openOrders 等键存in，避免 KeyError
            if hasattr(self.ib, 'wrapper') and hasattr(self.ib.wrapper, '_results'):
                res = self.ib.wrapper._results  # type: ignore[attr-defined]
                if isinstance(res, dict):
                    res.setdefault('completedOrders', [])
                    res.setdefault('openOrders', [])
                    res.setdefault('fills', [])
            # 忽略未知消息ID（if 176）以适配not同 API 版本差异
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
        # 使use统一positions管理器
        from .unified_position_manager import get_position_manager
        self.position_manager = get_position_manager()
        
        # 兼容性属性（逐步迁移）
        self._legacy_positions: Dict[str, int] = {}  # 临when保留
        self.open_orders: Dict[int, OrderRef] = {}
        self._stop_event: Optional[asyncio.Event] = None
        
        # account状态管理增强
        self.account_ready: bool = False
        self._last_account_update: float = 0.0
        self._account_lock = asyncio.Lock()
        self.account_update_interval: float = 60.0  # 最小updates间隔60 seconds
        
        # 使use任务生命周期管理器
        from .task_lifecycle_manager import get_task_manager
        self.task_manager = get_task_manager()
        
        # 使use统一connection管理器
        from .unified_connection_manager import create_connection_manager
        self.connection_manager = create_connection_manager(self.ib, config_manager, self.logger)
        
        # 交易审计器 (需要先初始化，供OrderManager使use)
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
        
        # 简化connection恢复管理，inibkr_auto_trader内部处理重连逻辑
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_interval = 5.0
        
        # 风险管理功能集成toEngineRiskEnginein，这里只需要简单风险check
        self._daily_order_count = 0
        self._max_daily_orders = 50

        # 动态止损管理（ATR + when间加权）
        self.dynamic_stop_cfg = {
            "atr_period": 14,
            "atr_lookback_days": 60,
            "atr_multiplier": 2.0,
            "decay_per_min": 0.1 / 60.0,  # 每分钟衰减0.1/60，约每小when0.1
            "min_decay_factor": 0.1,
            "update_interval_sec": 60.0,
        }
        # symbol -> state dict {entry_price, entry_time, qty, stop_trade, current_stop}
        self._stop_state: Dict[str, Dict[str, object]] = {}
        # symbol -> asyncio.Task for updater
        self._stop_tasks: Dict[str, asyncio.Task] = {}

        # 订单验证and统计
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
        
        # Polygon统一因子集成
        self.polygon_enabled = False
        self.polygon_risk_balancer_enabled = False
        if POLYGON_INTEGRATED:
            try:
                self.polygon_unified = get_polygon_unified_factors()
                self.polygon_enabled = self.polygon_unified.is_enabled()
                self.logger.info(f"Polygon统一因子集成: {'success' if self.polygon_enabled else 'failed'}")
            except Exception as e:
                self.logger.error(f"Polygon统一因子初始化failed: {e}")
                self.polygon_unified = None
        self._notify_throttle: Dict[str, float] = {}

        # 止损/止盈配置（canfrom data/risk_config.json 读取覆盖）
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

        # 统一from全局配置同步风险限制（and RiskManager and本地验证保持一致）
        try:
            # 使use统一配置管理器替代HotConfig
            config_dict = self.config_manager._get_merged_config()
            self._sync_risk_limits_from_config({"CONFIG": config_dict})
        except Exception:
            # 配置notcanusewhen保留默认值
            pass

        # completed初始化
        try:
            self.load_risk_config_from_db()
        except Exception:
            pass
        
        # risk control-收益平衡控制器（can选）
        try:
            from .enhanced_order_execution import RRConfig, RiskRewardController
            enabled = bool(self.config_manager.get('risk_reward.enabled', False)) if self.config_manager else False
            self.rr_cfg = RRConfig(enabled=enabled)
            self.rr_controller = RiskRewardController(self.rr_cfg)
        except Exception:
            self.rr_cfg = None
            self.rr_controller = None

        # 事件绑定
        self._bind_events()

    # ------------------------- 辅助：account取值 -------------------------
    def _get_account_numeric(self, tag: str) -> float:
        """from account_values 提取某个account字段数值（增强版）"""
        try:
            # 优先使use新account数据管理器
            return self.account_manager.get_account_numeric(tag)
        except:
            # 回退to原始实现  
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
                    self.logger.debug(f"account字段{tag}未找to，当before键示例: {list(self.account_values.keys())[:5]}")
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

    def load_risk_config_from_db(self) -> None:
        try:
            from .database import StockDatabase
            db = StockDatabase()
            # 优先使use新风险配置结构
            cfg = db.get_risk_config("默认风险配置")
            if not cfg:
                # fallbackto旧配置
                cfg = db.get_risk_config() or {}
                if isinstance(cfg, dict) and "risk_management" in cfg:
                    cfg = cfg["risk_management"]
            
            if isinstance(cfg, dict):
                # updates风险配置
                if "default_stop_pct" in cfg:
                    self.risk_config["risk_management"]["default_stop_pct"] = cfg["default_stop_pct"]
                if "default_target_pct" in cfg:
                    self.risk_config["risk_management"]["default_target_pct"] = cfg["default_target_pct"]
                if "max_single_position_pct" in cfg:
                    self.risk_config["risk_management"]["max_single_position_pct"] = cfg["max_single_position_pct"]
                
                self.allow_short = bool(cfg.get("allow_short", self.allow_short))
                self.logger.info(f"from数据库加载风险配置: 止损{cfg.get('default_stop_pct', 0.02)*100:.1f}% 止盈{cfg.get('default_target_pct', 0.05)*100:.1f}%")
        except Exception as e:
            self.logger.warning(f"from数据库加载风险配置failed: {e}")
    
    @property
    def positions(self) -> Dict[str, int]:
        """兼容性属性：retrievalpositions字典"""
        return {symbol: pos.quantity for symbol, pos in self.position_manager.get_all_positions().items()}
    
    @positions.setter  
    def positions(self, value: Dict[str, int]):
        """兼容性属性：settingspositions字典（not推荐使use）"""
        self.logger.warning("直接settingspositions属性弃use，请使useposition_manager")
        self._legacy_positions = value

    def _sync_risk_limits_from_config(self, cfg: Dict[str, Any]) -> None:
        """from统一配置管理器同步风险限制，统一来源，避免冲突。
        优先级：数据库配置 > 文件配置 > 默认配置
        """
        try:
            # 资金and仓位上限
            capital = self.config_manager.get("capital", {})
            cash_reserve = capital.get("cash_reserve_pct")
            max_single_pos = capital.get("max_single_position_pct")
            if cash_reserve is not None:
                self.order_verify_cfg["cash_reserve_pct"] = float(cash_reserve)
            if max_single_pos is not None:
                self.order_verify_cfg["max_single_position_pct"] = float(max_single_pos)

            # 同步to高级风险管理器
            # 单仓限制通过统一风险管理器配置
            self.logger.debug(f"单仓限制配置: {max_single_pos*100:.1f}%" if max_single_pos else "未settings")

            risk_controls = self.config_manager.get("risk_controls", {})
            sector_limit = risk_controls.get("sector_exposure_limit")
            # 行业敞口限制通过统一风险管理器配置
            self.logger.debug(f"行业敞口限制配置: {sector_limit*100:.1f}%" if sector_limit else "未settings")
        except Exception as e:
            self.logger.warning(f"同步风险限制to配置failed: {e}")

    # ------------------------- connectionand事件 -------------------------
    def _bind_events(self) -> None:
        """绑定事件处理器"""
        ib = self.ib
        try:
            ib.errorEvent += self._on_error
            ib.orderStatusEvent += self._on_order_status
            ib.execDetailsEvent += self._on_exec_details
            ib.commissionReportEvent += self._on_commission
            ib.accountSummaryEvent += self._on_account_summary
            
            # check并绑定canuse事件
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

            self.logger.info(" 事件处理器绑定completed")
        except Exception as e:
            self.logger.warning(f" 事件绑定部分failed: {e}")
            # 继续运行，not因事件绑定failed而in断

    async def connect(self, retries: int = None, retry_delay: float = None) -> None:
        """统一connection逻辑，使use配置管理器"""
        if retries is None:
            retries = self.config_manager.get('connection.max_reconnect_attempts', 10)
        if retry_delay is None:
            retry_delay = self.config_manager.get('connection.reconnect_interval', 5.0)
            
        # 使use统一配置管理器，not再需要独立ConnectionConfig
        # from .connection_config import ConnectionManager, ConnectionConfig
        
        # 使use统一配置管理器直接connection，简化逻辑
        
        self.logger.info(f"startingconnection {self.host}:{self.port}，目标ClientID={self.client_id}，account={self.account_id}")
        
        # 使use统一connection管理器
        try:
            success = await self.connection_manager.connect()
            
            if not success:
                raise ConnectionError("connection管理器connectionfailed")
            
            self.logger.info(f"[OK] 通过connection管理器connection，ClientID={self.client_id}")
            
            # 纯交易模式：不设置市场数据类型，所有价格数据来自Polygon
            self.logger.info("运行纯交易模式：IBKR只负责交易执行，价格数据来自Polygon (~15分钟延迟)")
            
            # 等待account数据就绪
            await self._wait_for_account_data()
            
            # 清理任何遗留的市场数据订阅（纯交易模式）
            self.cleanup_unused_subscriptions()
            
            # startconnection监控and其他服务
            await self._post_connection_setup()
            
        except Exception as e:
            self.logger.error(f"connectionfailed: {e}")
            raise

    async def _wait_for_account_data(self, timeout: float = 10.0) -> bool:
        """等待account数据就绪"""
        import time
        start_time = time.time()
        
        self.logger.info("等待account数据加载...")
        
        # 首次强制刷新account数据
        try:
            await self.refresh_account_balances_and_positions()
            if self.net_liq > 0:
                self.account_ready = True
                self.logger.info(f" account数据就绪: 净值=${self.net_liq:,.2f}, 现金=${self.cash_balance:,.2f}, account={self.account_id}")
                return True
        except Exception as e:
            self.logger.debug(f"首次account数据刷新failed: {e}")
        
        # if果首次failed，再等待一下
        while time.time() - start_time < timeout:
            try:
                await asyncio.sleep(2)  # 等待数据to达
                
                # retrievalaccount值
                account_values = self.ib.accountValues()
                if account_values:
                    self.account_values = {f"{av.tag}:{av.currency}": av.value for av in account_values}
                    
                    # 解析关键account数据
                    self.net_liq = self._get_account_numeric('NetLiquidation')
                    self.cash_balance = self._get_account_numeric('TotalCashValue')
                    self.buying_power = self._get_account_numeric('BuyingPower')
                    
                    # retrievalaccountID
                    for av in account_values:
                        if av.tag == 'AccountId':
                            self.account_id = av.value
                            break
                    
                    if self.net_liq > 0:
                        self.account_ready = True
                        self.logger.info(f" account数据就绪: 净值=${self.net_liq:,.2f}, 现金=${self.cash_balance:,.2f}, account={self.account_id}")
                        return True
                        
            except Exception as e:
                self.logger.debug(f"等待account数据: {e}")
            
            await asyncio.sleep(1.0)
        
        # 即使超when也尝试使use现has数据
        if hasattr(self, 'account_id') and self.account_id:
            self.logger.info(f" account数据retrieval超when，使use现has数据: account={self.account_id}")
            return True
        
        self.logger.warning(" account数据retrieval超when，继续运行但can能影响交易")
        return False

    async def _post_connection_setup(self):
        """connectionaftersettings工作"""
        try:
            # 初始化包装器结果字典
            self._init_wrapper_results()
            
            # retrievalpositions信息
            await self._update_positions()
            
            # connection恢复功能简化to内部处理
            
            # startreal-timeaccount监控任务（使use任务生命周期管理器）
            try:
                self.task_manager.create_task(
                    self._account_monitor_task(),
                    task_id="account_monitor",
                    creator="ibkr_auto_trader",
                    description="account监控任务",
                    group="system_monitoring"
                )
            except Exception as e:
                self.logger.error(f"startaccount监控任务failed: {e}")
            
            self.logger.info(" connectionaftersettingscompleted")
            
        except Exception as e:
            self.logger.warning(f" connectionaftersettings部分failed: {e}")

    async def _update_positions(self):
        """updatespositions信息（使use统一positions管理器）"""
        try:
            positions = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            
            # 构建经纪商positions数据
            broker_positions = {}
            price_source = {}
            
            for pos in positions:
                if pos.position != 0:
                    symbol = pos.contract.symbol
                    broker_positions[symbol] = int(pos.position)
                    
                    # retrieval当beforeprice
                    current_price = self.get_price(symbol)
                    if current_price and current_price > 0:
                        price_source[symbol] = current_price
                    else:
                        # 使use平均成本作as默认price
                        price_source[symbol] = float(pos.avgCost) if pos.avgCost and pos.avgCost > 0 else 100.0
            
            # and统一positions管理器同步
            sync_result = await self.position_manager.sync_with_broker_positions(
                broker_positions, price_source
            )
            
            self.logger.info(f"positions同步completed: {len(broker_positions)} 个非零positions, "
                           f"新增{len(sync_result['added'])} updates{len(sync_result['updated'])} "
                           f"移除{len(sync_result['removed'])}")
            
        except Exception as e:
            self.logger.warning(f"updatespositions信息failed: {e}")

    def _init_wrapper_results(self):
        """初始化包装器结果字典，防止KeyError"""
        try:
            if hasattr(self.ib, 'wrapper') and hasattr(self.ib.wrapper, '_results'):
                res = self.ib.wrapper._results
                if isinstance(res, dict) and 'completedOrders' not in res:
                    res['completedOrders'] = []
                    self.logger.debug("初始化completedOrders容器")
        except Exception as e:
            self.logger.debug(f"初始化包装器结果failed: {e}")



    async def _account_monitor_task(self) -> None:
        """real-timeaccount监控任务 - 确保account数据real-time性"""
        monitor_interval = 30.0  # 30 seconds监控间隔
        
        try:
            while not self._stop_event or not self._stop_event.is_set():
                try:
                    current_time = time.time()
                    
                    # checkaccount数据is否过期
                    if current_time - self._last_account_update > self.account_update_interval:
                        async with self._account_lock:
                            self.logger.debug("account数据过期，执行自动刷新")
                            await self.refresh_account_balances_and_positions()
                    
                    # check关键指标异常
                    if self.account_ready:
                        # check净值is否异常
                        if self.net_liq <= 0:
                            self.logger.warning("account净值异常：<=0，强制刷新account数据")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                        
                        # check现金余额is否异常
                        elif self.cash_balance < 0:
                            self.logger.warning("现金余额异常：<0，强制刷新account数据")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                        
                        # checkpositions数据一致性
                        elif self.position_manager.get_portfolio_summary().total_positions == 0 and self.net_liq > self.cash_balance * 1.1:
                            self.logger.warning("positions数据can能not一致，强制刷新")
                            async with self._account_lock:
                                await self.refresh_account_balances_and_positions()
                    
                    await asyncio.sleep(monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"account监控任务异常: {e}")
                    await asyncio.sleep(monitor_interval)
                    
        except asyncio.CancelledError:
            self.logger.info("account监控任务be取消")
        except Exception as e:
            self.logger.error(f"account监控任务致命错误: {e}")
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
                    
                    # updates所haspositionsprice历史
                    for symbol in self.position_manager.get_symbols():
                        current_price = self.get_price(symbol)
                        if current_price and current_price > 0:
                            # price历史updates整合to统一风险管理器andpositions管理器
                            pass
                    
                    # 计算positions价值
                    positions_value = {}
                    total_position_value = 0.0
                    
                    for symbol, position in self.position_manager.get_all_positions().items():
                        qty = position.quantity
                        if qty > 0:
                            price = self.get_price(symbol) or 0.0
                            if price > 0:
                                value = qty * price
                                positions_value[symbol] = value
                                total_position_value += value
                    
                    if positions_value and total_position_value > 0:
                        # 评估投资组合风险
                        try:
                            # 风险评估整合to统一风险管理器
                            risk_metrics = {}  # 简化处理
                            
                            # 风险警告check
                            warnings = []
                            
                            # VaRcheck
                            if risk_metrics.portfolio_var > 0.02:  # 2%
                                warnings.append(f"组合VaR过高: {risk_metrics.portfolio_var:.2%}")
                            
                            # 相关性check
                            if risk_metrics.correlation_risk > 0.7:
                                warnings.append(f"相关性风险: {risk_metrics.correlation_risk:.2f}")
                            
                            # 集in度check
                            if risk_metrics.concentration_risk > 0.3:
                                warnings.append(f"positions集in度过高: HHI={risk_metrics.concentration_risk:.2f}")
                            
                            # 杠杆check
                            if risk_metrics.leverage_ratio > 1.2:
                                warnings.append(f"杠杆过高: {risk_metrics.leverage_ratio:.2f}x")
                            
                            # 单个positionscheck
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
                                        f"检测to{len(warnings)}个风险问题", 
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
                                # 风险报告整合to统一风险管理器
                                self.logger.debug("风险监控活跃")
                                last_risk_report = current_time
                        
                        except Exception as e:
                            self.logger.warning(f"风险评估failed: {e}")
                    
                    await asyncio.sleep(monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"风险监控任务异常: {e}")
                    await asyncio.sleep(monitor_interval)
                    
        except asyncio.CancelledError:
            self.logger.info("风险监控任务be取消")
        except Exception as e:
            self.logger.error(f"风险监控任务致命错误: {e}")
            raise  # 让任务管理器重启

    async def _prime_account_and_positions(self) -> None:
        # accountsummary（EClient.reqAccountSummary）
        try:
            rows = await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=10.0)
            for r in rows:
                key = f"{r.tag}:{r.currency or ''}"
                self.account_values[key] = r.value
                # 捕获accountID
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
            self.logger.info(f"accountsummary: 现金={self.cash_balance:.2f} 净值={self.net_liq:.2f}")
            self.account_ready = self.net_liq > 0
        except Exception as e:
            self.logger.warning(f"retrievalaccountsummaryfailed: {e}")

        # positions（EClient.reqPositions）
        try:
            poss = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            self.position_manager.clear_all_positions()
            for p in poss:
                sym = p.contract.symbol
                qty = int(p.position)
                # 通过position_managerupdatespositions
                current_price = self.get_price(sym) or p.avgCost or 100.0
                asyncio.create_task(
                    self.position_manager.update_position(sym, qty, current_price, p.avgCost)
                )
            portfolio_summary = self.position_manager.get_portfolio_summary()
            self.logger.info(f"当beforepositions标数: {portfolio_summary.total_positions}")
        except Exception as e:
            self.logger.warning(f"retrievalpositionsfailed: {e}")

    async def refresh_account_balances_and_positions(self) -> None:
        """增强版account刷新 - 带数据验证、缓存and同步保护"""
        refresh_start = time.time()
        
        # 保存刷新before数据useat验证
        prev_cash = self.cash_balance
        prev_netliq = self.net_liq
        prev_positions_count = self.position_manager.get_portfolio_summary().total_positions
        
        try:
            # accountsummary刷新
            self.logger.debug("starting刷新accountsummary...")
            rows = await asyncio.wait_for(self.ib.accountSummaryAsync(), timeout=10.0)
            
            if not rows:
                raise ValueError("accountsummary返回空数据")
            
            # updatesaccount值
            for r in rows:
                key = f"{r.tag}:{r.currency or ''}"
                self.account_values[key] = r.value
                # 捕获accountID
                try:
                    if getattr(r, 'account', None):
                        self.account_id = str(r.account)
                except Exception:
                    pass
            
            # 解析关键财务数据
            try:
                # 更稳健地解析account数值，兼容 BASE/多币种
                new_cash = self._get_account_numeric("TotalCashValue")
                new_netliq = self._get_account_numeric("NetLiquidation")
                new_buying_power = self._get_account_numeric("BuyingPower")

                # 数据合理性验证（放宽：净值<=0 not再抛异常，仅标记未就绪并记录警告）
                if new_netliq <= 0:
                    self.logger.warning(f"净值异常(<=0): {new_netliq}，标记account未就绪但not终止connection")
                    self.account_ready = False
                else:
                    self.account_ready = True
                
                if new_cash < -abs(new_netliq):  # 现金负数not能超过净值绝for值
                    self.logger.warning(f"现金余额异常: ${new_cash:.2f}, 净值: ${new_netliq:.2f}")
                
                # check数据变化is否合理
                if prev_netliq > 0 and new_netliq > 0:
                    netliq_change_pct = abs(new_netliq - prev_netliq) / prev_netliq
                    if netliq_change_pct > 0.5:  # 净值变化超过50%
                        self.logger.warning(f"净值变化异常大: {prev_netliq:.2f} -> {new_netliq:.2f} ({netliq_change_pct:.1%})")
                
                # updates数据（即便未就绪也同步最新快照供UI显示）
                self.cash_balance = new_cash
                self.net_liq = new_netliq
                self.buying_power = new_buying_power
                self._last_account_update = time.time()
                
                self.logger.debug(
                    f"accountsummary刷新completed: 现金${self.cash_balance:.2f}, 净值${self.net_liq:.2f}, 购买力${self.buying_power:.2f}, 就绪={self.account_ready}"
                )
                
            except Exception as parse_error:
                self.logger.error(f"解析account数据failed: {parse_error}")
                # 放宽：解析failednot再to上抛出，避免打断引擎；仅标记未就绪
                self.account_ready = False
                return
                
        except asyncio.TimeoutError:
            self.logger.error("accountsummary刷新超when")
            self.account_ready = False
            return
        except Exception as e:
            self.logger.error(f"刷新accountsummaryfailed: {e}")
            self.account_ready = False
            return

        # 刷新positions数据
        try:
            self.logger.debug("starting刷新positions...")
            poss = await asyncio.wait_for(self.ib.reqPositionsAsync(), timeout=10.0)
            
            new_positions = {}
            for p in poss:
                sym = p.contract.symbol
                qty = int(p.position)
                if qty != 0:  # 只记录非零positions
                    new_positions[sym] = new_positions.get(sym, 0) + qty
            
            # checkpositions变化is否合理
            new_positions_count = len(new_positions)
            if prev_positions_count > 0:
                position_change = abs(new_positions_count - prev_positions_count)
                if position_change > 10:  # positions数量变化超过10个
                    self.logger.warning(f"positions数量变化异常: {prev_positions_count} -> {new_positions_count}")
            
            # 更新positions（使用position_manager而非废弃属性）
            # self.positions = new_positions  # 废弃用法
            for symbol, quantity in new_positions.items():
                try:
                    # 获取当前价格用于position manager
                    current_price = self.get_price(symbol)
                    if current_price is None:
                        current_price = 0.0  # 临时价格，position manager应该处理这种情况
                    
                    # 在async上下文中直接使用await
                    await self.position_manager.update_position(symbol, quantity, current_price)
                except Exception as e:
                    self.logger.warning(f"更新position {symbol}失败: {e}")
            
            refresh_duration = time.time() - refresh_start
            self.logger.debug(f"positions刷新completed: {self.position_manager.get_portfolio_summary().total_positions}个标 (usewhen{refresh_duration:.2f} seconds)")
            
            # 记录关键变化
            if prev_cash != self.cash_balance or prev_netliq != self.net_liq:
                cash_change = self.cash_balance - prev_cash
                netliq_change = self.net_liq - prev_netliq
                self.logger.info(f"account变化: 现金{cash_change:+.2f}, 净值{netliq_change:+.2f}")
            
        except asyncio.TimeoutError:
            self.logger.error("positions刷新超when")
            return
        except Exception as e:
            self.logger.error(f"刷新positionsfailed: {e}")
            return

    async def wait_for_price(self, symbol: str, timeout: float = 2.0, interval: float = 0.1) -> Optional[float]:
        """等待直to拿to该 symbol priceor超when。"""
        start = time.time()
        price = self.get_price(symbol)
        while price is None and time.time() - start < timeout:
            await self.ib.sleep(interval)
            price = self.get_price(symbol)
        return price

    # ------------------------- contractandmarket data -------------------------
    async def qualify_stock(self, symbol: str, primary_exchange: Optional[str] = None) -> Contract:
        # contractqualification verification（EClient.reqContractDetails / qualifyContracts）
        contract = Stock(symbol, exchange="SMART", currency=self.default_currency)
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            return qualified[0]
        except Exception as e:
            self.logger.warning(f"qualifyContracts failed，尝试settings primaryExchange: {e}")
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

    async def prepare_symbol_for_trading(self, symbol: str) -> bool:
        """为交易准备股票合约（纯交易模式：只验证合约，不订阅数据）"""
        try:
            # 只验证合约有效性，不订阅市场数据
            c = await self.qualify_stock(symbol)
            self.logger.debug(f"{symbol} 合约验证成功，已准备交易: {c}")
            
            # 预加载Polygon价格数据
            price = self.get_price(symbol)
            if price is not None:
                self.logger.info(f"📊 {symbol} Polygon价格已加载: ${price:.4f} (~15分钟延迟)")
                return True
            else:
                self.logger.warning(f"⚠️ {symbol} Polygon价格获取失败，但合约验证成功")
                return False
                
        except Exception as e:
            self.logger.error(f"{symbol} 交易准备失败: {e}")
            return False

    def cleanup_unused_subscriptions(self) -> None:
        """清理所有不需要的市场数据订阅（纯交易模式）"""
        if hasattr(self, 'tickers') and self.tickers:
            self.logger.info(f"清理 {len(self.tickers)} 个不需要的市场数据订阅")
            for symbol, ticker in list(self.tickers.items()):
                try:
                    self.ib.cancelMktData(ticker)
                except Exception:
                    pass
            self.tickers.clear()
            self.logger.info("✅ 市场数据订阅已全部清理，运行纯交易模式")

    async def _validate_order_before_submission(self, symbol: str, side: str, qty: int, price: float) -> bool:
        """统一风险验证 - 使use统一风险管理器 - 带详细调试信息"""
        
        # 🔍 详细调试输出
        print(f"\n{'='*80}")
        print(f"🛡️  ORDER VALIDATION DEBUG - {symbol}")
        print(f"{'='*80}")
        print(f"📊 订单信息: {symbol} {side.upper()} {qty}股 @ ${price:.4f}")
        print(f"💰 订单价值: ${qty * price:,.2f}")
        
        # 📡 数据新鲜度检查
        is_suitable, reason = self.is_data_suitable_for_trading(symbol)
        data_price, data_source, data_age = self.get_data_freshness(symbol)
        
        print(f"📡 数据状态:")
        print(f"   ├─ 数据源: {data_source}")
        print(f"   ├─ 数据价格: ${data_price:.4f}" if data_price else "   ├─ 数据价格: 无")
        print(f"   ├─ 数据年龄: {data_age}秒")
        print(f"   ├─ 交易适用性: {'✅' if is_suitable else '❌'} {reason}")
        
        # 如果数据不适合交易，发出警告但不阻止（给用户选择权）
        if not is_suitable:
            print(f"   └─ ⚠️ 警告: 使用非实时数据进行交易，风险较高!")
            self.logger.warning(f"{symbol} 使用非实时数据交易: {reason}")
        
        print(f"🏦 账户状态:")
        print(f"   ├─ 净清算价值: ${self.net_liq:,.2f}")
        print(f"   ├─ 现金余额: ${self.cash_balance:,.2f}")
        print(f"   ├─ 账户就绪: {'✅' if self.account_ready else '❌'}")
        
        try:
            # 使use统一风险管理器进行验证
            from .unified_risk_manager import get_risk_manager
            risk_manager = get_risk_manager(self.config_manager)
            print(f"   └─ 风险管理器: {'✅ 已加载' if risk_manager else '❌ 加载失败'}")
            
            # retrievalaccount价值
            account_value = max(self.net_liq, 0.0)
            print(f"🎯 验证参数:")
            print(f"   ├─ 账户价值: ${account_value:,.2f}")
            print(f"   └─ 验证股票: {symbol}")
            
            # 统一风险验证
            print(f"🔄 开始统一风险验证...")
            result = await risk_manager.validate_order(symbol, side, qty, price, account_value)
            
            print(f"📋 风险验证结果:")
            print(f"   ├─ 验证结果: {'✅ 通过' if result.is_valid else '❌ 失败'}")
            if not result.is_valid:
                print(f"   ├─ 违规原因: {', '.join(result.violations)}")
                self.logger.warning(f"统一风险验证failed {symbol}: {result.violations}")
                return False

            if result.warnings:
                print(f"   ├─ 风险警告: {', '.join(result.warnings)}")
                self.logger.info(f"风险警告 {symbol}: {result.warnings}")
            
            print(f"🔒 开始交易层面验证...")
            async with self._account_lock:
                # check待处理订单敞口（交易层面check）
                active_orders = await self.order_manager.get_orders_by_symbol(symbol)
                print(f"   ├─ 当前活跃订单数: {len(active_orders)}")
                
                pending_value = sum(
                    order.quantity * (order.price or price) 
                    for order in active_orders 
                    if order.side == side.upper() and order.is_active()
                )
                print(f"   ├─ 待处理订单价值: ${pending_value:,.2f}")
                
                if pending_value > 0:
                    order_value = qty * price
                    total_exposure = order_value + pending_value
                    max_exposure = self.net_liq * self.order_verify_cfg["max_single_position_pct"]
                    
                    print(f"   ├─ 当前订单价值: ${order_value:,.2f}")
                    print(f"   ├─ 总敞口: ${total_exposure:,.2f}")
                    print(f"   ├─ 最大敞口限制: ${max_exposure:,.2f}")
                    
                    if total_exposure > max_exposure:
                        print(f"   └─ ❌ 总敞口超限!")
                        self.logger.warning(f"{symbol} 总敞口超限: ${total_exposure:.2f} > ${max_exposure:.2f} (含待处理订单${pending_value:.2f})")
                        return False
                    else:
                        print(f"   └─ ✅ 敞口检查通过")
                else:
                    print(f"   └─ ✅ 无待处理订单冲突")

                # 最终风险管理器验证
                print(f"🔍 最终风险验证...")
                validation_result = await risk_manager.validate_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    account_value=self.net_liq
                )
                
                # 处理验证结果
                print(f"📊 最终验证结果:")
                print(f"   ├─ 验证状态: {'✅ 通过' if validation_result.is_valid else '❌ 失败'}")
                
                if not validation_result.is_valid:
                    reasons = ', '.join(validation_result.violations)
                    print(f"   ├─ 失败原因: {reasons}")
                    self.logger.warning(f"{symbol} 风险验证failed: {reasons}")
                    
                    # if果has建议仓位，记录信息
                    if validation_result.recommended_size and validation_result.recommended_size != qty and validation_result.recommended_size > 0:
                        print(f"   ├─ 建议仓位: {validation_result.recommended_size}股 (当前:{qty}股)")
                        self.logger.info(f"{symbol} 建议调整仓位: {qty} -> {validation_result.recommended_size}股")
                    
                    print(f"   └─ ❌ 订单被拒绝")
                    return False
                
                # 记录警告信息
                if validation_result.warnings:
                    print(f"   ├─ 警告信息:")
                    for warning in validation_result.warnings:
                        print(f"   │  └─ ⚠️  {warning}")
                        self.logger.warning(f"{symbol} 风险警告: {warning}")
                
                # if果需要刷新account数据
                if any('过期' in w for w in validation_result.warnings):
                    print(f"   ├─ 🔄 需要刷新账户数据...")
                    self.logger.info("根据风险check建议，刷新account数据...")
                    await self.refresh_account_balances_and_positions()
                
                print(f"   └─ ✅ 所有验证通过!")
                print(f"{'='*80}")
                print(f"✅ 订单 {symbol} {side.upper()} {qty}股 验证通过，准备提交")
                print(f"{'='*80}\n")
                return True
                
        except Exception as e:
            print(f"   └─ ❌ 验证异常: {e}")
            print(f"{'='*80}")
            print(f"❌ 订单验证异常: {symbol} - {e}")
            print(f"{'='*80}\n")
            self.logger.error(f"订单验证异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_price(self, symbol: str) -> Optional[float]:
        """纯Polygon数据模式：只从Polygon获取价格数据，IBKR只负责交易执行"""
        price_val: Optional[float] = None
        data_source = None
        
        # 1. 首先检查缓存的新鲜度 (减少API调用)
        if symbol in self.last_price:
            cached_price, cached_time = self.last_price[symbol]
            cache_age_seconds = time.time() - cached_time
            
            # 如果缓存数据不超过5分钟，直接使用
            if cache_age_seconds < 300:  # 5分钟
                self.logger.debug(f"{symbol} 使用Polygon缓存: ${cached_price:.4f} ({int(cache_age_seconds/60)}分钟前)")
                return cached_price
        
        # 2. 从Polygon获取最新数据（15分钟延迟）
        try:
            from polygon_client import polygon_client
            from datetime import datetime, timedelta
            
            # 获取最近5天数据，确保有交易日数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            df = polygon_client.get_historical_bars(symbol, start_date, end_date)
            if not df.empty and ('Close' in df.columns or 'close' in df.columns):
                # 兼容不同的列名格式
                close_col = 'Close' if 'Close' in df.columns else 'close'
                latest_close = df[close_col].iloc[-1]
                data_date = df.index[-1].date()
                today = datetime.now().date()
                data_age_days = (today - data_date).days
                
                if latest_close > 0:
                    price_val = float(latest_close)
                    
                    if data_age_days == 0:
                        data_source = "POLYGON_TODAY_DELAYED_15MIN"
                        self.logger.info(f"📊 {symbol} Polygon当日延迟价格: ${price_val:.4f} (~15分钟延迟)")
                    elif data_age_days == 1:
                        data_source = "POLYGON_YESTERDAY_CLOSE"
                        self.logger.info(f"📊 {symbol} Polygon昨日收盘: ${price_val:.4f}")
                    else:
                        data_source = f"POLYGON_{data_age_days}DAYS_OLD"
                        self.logger.warning(f"⚠️ {symbol} Polygon {data_age_days}天前数据: ${price_val:.4f}")
                    
                    # 更新缓存
                    ts = time.time()
                    self.last_price[symbol] = (price_val, ts)
                    return price_val
            
        except Exception as e:
            self.logger.debug(f"Polygon历史数据获取失败 {symbol}: {e}")
        
        # 3. 最后fallback: 使用ticker详情估算价格
        try:
            from polygon_client import polygon_client
            details = polygon_client.get_ticker_details(symbol)
            if details and 'marketCap' in details and 'share_class_shares_outstanding' in details:
                market_cap = details.get('marketCap', 0)
                shares = details.get('share_class_shares_outstanding', 0)
                if market_cap > 0 and shares > 0:
                    estimated_price = market_cap / shares
                    if estimated_price > 0:
                        price_val = float(estimated_price)
                        data_source = "POLYGON_MARKET_CAP_ESTIMATE"
                        self.logger.warning(f"⚠️ {symbol} 市值估算价格(参考用): ${price_val:.4f}")
                        
                        # 更新缓存
                        ts = time.time()
                        self.last_price[symbol] = (price_val, ts)
                        return price_val
        except Exception as detail_error:
            self.logger.debug(f"Polygon市值估算失败 {symbol}: {detail_error}")
        
        # 4. 使用稍旧的缓存作为最后手段
        if symbol in self.last_price:
            cached_price, cached_time = self.last_price[symbol]
            cache_age_seconds = time.time() - cached_time
            
            # 最多使用1天的缓存
            if cache_age_seconds < 86400:  # 24小时
                cache_age_hours = int(cache_age_seconds / 3600)
                self.logger.warning(f"⚠️ {symbol} 使用陈旧缓存: ${cached_price:.4f} ({cache_age_hours}小时前)")
                return cached_price
                
        self.logger.error(f"❌ {symbol} Polygon价格获取完全失败，建议检查网络连接或API状态")
        return None

    def get_data_freshness(self, symbol: str) -> tuple[Optional[float], str, int]:
        """获取Polygon价格数据的新鲜度信息 (纯Polygon模式)
        
        Returns:
            (price, data_source, age_seconds)
        """
        # 检查缓存数据的新鲜度
        if symbol in self.last_price:
            cached_price, cached_time = self.last_price[symbol]
            age_seconds = int(time.time() - cached_time)
            
            if age_seconds < 300:  # 5分钟内 - 认为是新鲜的延迟数据
                return cached_price, "POLYGON_FRESH_DELAYED", age_seconds
            elif age_seconds < 1800:  # 30分钟内 - 可用的延迟数据
                return cached_price, "POLYGON_USABLE_DELAYED", age_seconds
            elif age_seconds < 3600:  # 1小时内 - 较旧的数据
                return cached_price, "POLYGON_STALE", age_seconds
            else:  # 超过1小时 - 过时数据
                return cached_price, "POLYGON_EXPIRED", age_seconds
        
        return None, "NO_POLYGON_DATA", 0

    def is_data_suitable_for_trading(self, symbol: str) -> tuple[bool, str]:
        """检查Polygon数据是否适合交易决策 (纯Polygon模式)
        
        Returns:
            (is_suitable, reason)
        """
        price, source, age = self.get_data_freshness(symbol)
        
        if price is None:
            return False, "无Polygon价格数据"
        
        if source == "POLYGON_FRESH_DELAYED":
            return True, f"Polygon延迟数据({age//60}分钟前，可交易)"
        elif source == "POLYGON_USABLE_DELAYED":
            return True, f"Polygon延迟数据({age//60}分钟前，谨慎交易)"
        elif source == "POLYGON_STALE":
            return False, f"Polygon数据过时({age//60}分钟前，不建议交易)"
        elif source == "POLYGON_EXPIRED":
            return False, f"Polygon数据严重过时({age//3600}小时前，拒绝交易)"
        else:
            return False, f"Polygon数据异常: {source}"

    async def get_price_with_refresh(self, symbol: str, force_refresh: bool = False) -> Optional[float]:
        """获取价格，可选择强制刷新Polygon数据"""
        
        # 如果强制刷新，清除缓存
        if force_refresh and symbol in self.last_price:
            del self.last_price[symbol]
            self.logger.debug(f"{symbol} 强制清除价格缓存，重新从Polygon获取")
        
        # 获取价格（会自动从Polygon拉取或使用缓存）
        price = self.get_price(symbol)
        return price

    # ------------------------- 动态止损（ATR + when间加权） -------------------------
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
            self.logger.warning(f"拉取历史数据failed {symbol}: {e}")
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
        """根据account状态and风险因子动态分配资金（风险因子∈[0,1)）。
        risk_factor 越大，分配资金越少。
        返回本次canuse最大order placement预算（美元）。
        """
        try:
            rf = max(0.0, min(float(risk_factor), 0.95))
            reserved = self.net_liq * self.order_verify_cfg["cash_reserve_pct"]
            available_cash = max((self.cash_balance or 0.0) - reserved, 0.0)
            max_pos_value = self.net_liq * self.order_verify_cfg["max_single_position_pct"]
            allocation = available_cash * (1.0 - rf)
            return max(0.0, min(allocation, max_pos_value))
        except Exception:
            # 保底按单标上限
            return max(0.0, self.net_liq * self.order_verify_cfg.get("max_single_position_pct", 0.12))

    # ------------------------- 风险配置and止损/止盈计算 -------------------------
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
        """基at ATR 估算风险因子 ∈ [0, 0.95]。"""
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
        """after台任务：周期性计算动态止损并以 StopOrder 刷新/下发止损单。"""
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

                # retrieval ATR
                bars = await self._fetch_daily_bars(symbol, self.dynamic_stop_cfg["atr_lookback_days"])
                atr = self._calc_atr_from_bars(bars, self.dynamic_stop_cfg["atr_period"]) or 0.0
                if atr <= 0:
                    await asyncio.sleep(self.dynamic_stop_cfg["update_interval_sec"])
                    continue

                # 原始距离 = ATR * 倍数
                raw_dist = atr * float(self.dynamic_stop_cfg["atr_multiplier"])  # 正数
                # when间加权
                dist = self._time_weighted_distance(entry_price, raw_dist, entry_time)
                stop_price = max(0.01, entry_price - dist)

                # 若现价大幅上移，can考虑抬升止损（not放宽）
                mkt = self.get_price(symbol) or entry_price
                if mkt > entry_price:
                    trail_up = max(0.0, mkt - entry_price)
                    stop_price = max(stop_price, mkt - raw_dist)  # 亦can按 dist

                prev_stop = float(st.get("current_stop") or 0.0)
                if stop_price > prev_stop + 0.01:  # 仅上调
                    # if启use本地动态止损，才撤销/updates；否则notand服务器端bracket order止损冲突
                    rm = self.risk_config.get("risk_management", {}) or {}
                    if bool(rm.get("enable_local_dynamic_stop_for_bracket", False)):
                        # 撤销旧止损
                        old_trade: Optional[Trade] = st.get("stop_trade")  # type: ignore
                        try:
                            if old_trade:
                                self.ib.cancelOrder(old_trade.order)
                        except Exception:
                            pass

                    # 下发新止损单（仅in启use本地动态止损when）
                    if bool(rm.get("enable_local_dynamic_stop_for_bracket", False)):
                        try:
                            c = await self.qualify_stock(symbol)
                            stop_order = StopOrder("SELL", qty, stop_price)
                            new_trade = self.ib.placeOrder(c, stop_order)
                            st["stop_trade"] = new_trade
                            st["current_stop"] = stop_price
                            self._stop_state[symbol] = st
                            self.logger.info(f"updates动态止损 {symbol}: stop={stop_price:.2f} qty={qty}")
                        except Exception as e:
                            self.logger.warning(f"提交止损failed {symbol}: {e}")
                            try:
                                await self._notify_webhook("stop_replace_fail", "动态止损提交failed", f"{symbol} 提交止损failed", {"error": str(e)})
                            except Exception:
                                pass
                    else:
                        # 仅updates本地参考止损价，notfor服务器端括号止损做变更
                        st["current_stop"] = stop_price
                        self._stop_state[symbol] = st

                await asyncio.sleep(self.dynamic_stop_cfg["update_interval_sec"])
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.warning(f"动态止损任务异常 {symbol}: {e}")

    # ------------------------- order placementand订单管理 -------------------------
    async def place_market_order(self, symbol: str, action: str, quantity: int, retries: int = 3) -> OrderRef:
        """增强market单order placement，使useEnhancedOrderExecutor - 带详细调试信息"""
        
        # 🔍 详细调试输出 - 订单提交流程
        print(f"\n{'='*80}")
        print(f"📈 MARKET ORDER PLACEMENT DEBUG - {symbol}")
        print(f"{'='*80}")
        print(f"🎯 订单参数: {symbol} {action.upper()} {quantity}股 (市价单)")
        print(f"🔄 重试次数: {retries}")
        print(f"📅 日期检查: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 日内计数重置
        try:
            today = datetime.now().date()
            if self._last_reset_day != today:
                self._daily_order_count = 0
                self._last_reset_day = today
                print(f"🔄 日内计数重置: {today}")
            else:
                print(f"📊 当日已下单: {self._daily_order_count}笔")
        except Exception as e:
            print(f"⚠️  日期重置异常: {e}")

        # order placementbefore验证
        print(f"\n🔍 开始订单前置验证...")
        try:
            print(f"💰 获取价格: {symbol}")
            price_now = self.get_price(symbol) or 0.0
            print(f"   ├─ 首次价格获取: ${price_now:.4f}")
            
            if price_now <= 0:
                print(f"   ├─ 价格无效，尝试刷新Polygon数据...")
                await self.prepare_symbol_for_trading(symbol)
                price_now = await self.get_price_with_refresh(symbol, force_refresh=True) or 0.0
                print(f"   ├─ 刷新后价格: ${price_now:.4f}")
                
            if price_now <= 0:
                print(f"   └─ ❌ 价格获取失败!")
                await self._notify_webhook("no_price", "priceretrievalfailed", f"{symbol} nohas效price，拒绝order placement", {"symbol": symbol})
                raise RuntimeError(f"no法retrievalhas效price: {symbol}")
            else:
                print(f"   └─ ✅ 价格获取成功: ${price_now:.4f}")
            
            print(f"🛡️ 开始风险验证: {symbol} {action} {quantity}股 @ ${price_now:.4f}")
            validation_passed = await self._validate_order_before_submission(symbol, action, quantity, price_now)
            
            if not validation_passed:
                print(f"❌ 风险验证失败，订单被拒绝!")
                await self._notify_webhook("risk_reject", "risk control拒单", f"{symbol} order placementbefore校验未通过", {"symbol": symbol, "action": action, "qty": quantity, "price": price_now})
                raise RuntimeError("订单before置校验未通过")
            else:
                print(f"✅ 风险验证通过!")
                
        except Exception as e:
            print(f"❌ 订单前置验证失败: {e}")
            self.logger.warning(f"order placementbefore校验failed {symbol}: {e}")
            print(f"{'='*80}")
            print(f"❌ 订单提交终止: {symbol} {action} {quantity}股")
            print(f"{'='*80}\n")
            raise
        
        # 纯路bytoEnhancedOrderExecutor执行订单
        print(f"\n🚀 开始执行市价单...")
        try:
            from .enhanced_order_execution import ExecutionConfig
            exec_cfg = ExecutionConfig()
            print(f"   ├─ 执行配置已加载")
            print(f"   ├─ 调用增强订单执行器...")
            
            order_sm = await self.enhanced_executor.execute_market_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                config=exec_cfg,
            )
            
            print(f"   ├─ 订单状态机创建成功: Order ID {order_sm.order_id}")
            print(f"   └─ 当前订单状态: {order_sm.state.value}")

            # 统一返回 OrderRef（and现has调use方兼容）
            enhanced_ref = OrderRef(
                order_id=order_sm.order_id,
                symbol=symbol,
                side=action,
                qty=quantity,
                order_type="MKT",
            )
            
            print(f"📋 订单引用创建: OrderRef(id={enhanced_ref.order_id})")
            
            # 审计记录通过OrderManager回调自动处理，no需重复记录
            
            # updates计数
            self._daily_order_count += 1
            print(f"📊 更新日内计数: {self._daily_order_count}笔")
            
            # 刷新account信息
            print(f"🔄 刷新账户信息...")
            await self.refresh_account_balances_and_positions()
            print(f"✅ 账户信息刷新完成")
            
            print(f"{'='*80}")
            print(f"✅ 市价单提交成功: {symbol} {action.upper()} {quantity}股")
            print(f"   ├─ 订单ID: {enhanced_ref.order_id}")
            print(f"   ├─ 订单类型: 市价单")
            print(f"   └─ 订单价格: ${price_now:.4f} (参考)")
            print(f"{'='*80}\n")
            
            return enhanced_ref
            
        except Exception as e:
            print(f"❌ 订单执行失败: {e}")
            print(f"{'='*80}")
            print(f"❌ 市价单执行失败: {symbol} {action} {quantity}股")
            print(f"   └─ 错误: {e}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()
            raise
    

                
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
        """使usebracket order进行market入场 + 服务器端托管止盈/止损（父子/同组）。
        若 use_config=True，则from risk_config 读取（含 symbol override and strategy）。
        
          NOTE: 考虑使use place_bracket_order() 作as统一接口
        """
        price_now = self.get_price(symbol) or 0.0
        if price_now <= 0:
            await self.prepare_symbol_for_trading(symbol)
            price_now = await self.get_price_with_refresh(symbol, force_refresh=True) or 0.0
        if price_now <= 0:
            raise RuntimeError(f"无法从Polygon获取{symbol}有效价格")

        if not await self._validate_order_before_submission(symbol, action, quantity, price_now):
            raise RuntimeError("订单before置校验未通过")

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

        # 做空支持（if禁use，拒绝 SHORT）
        if side == "SHORT" and not self.allow_short:
            raise RuntimeError("当before配置not允许做空")

        send_side = side if side in ("BUY", "SELL") else ("SELL" if side == "SHORT" else "BUY")

        parent = MarketOrder(send_side, quantity)
        parent.transmit = False
        trade_parent = self.ib.placeOrder(c, parent)
        await self.ib.sleep(0.1)
        parent_id = trade_parent.order.orderId

        # 子单方to：止盈/止损反to
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
            st["qty"] = self.position_manager.get_quantity(symbol)
            st["stop_trade"] = trade_sl
            st["current_stop"] = stop_price
            self._stop_state[symbol] = st
            # 使use任务生命周期管理器创建止损任务
            task_id = f"stop_manager_{symbol}"
            try:
                self.task_manager.create_task(
                    self._dynamic_stop_manager(symbol),
                    task_id=task_id,
                    creator="ibkr_auto_trader",
                    description=f"动态止损管理: {symbol}",
                    group="stop_loss_management",
                    max_lifetime=86400  # 24小when最大生存when间
                )
            except Exception as e:
                self.logger.error(f"start止损任务failed {symbol}: {e}")
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
            self.logger.warning(f"审计记录failed: {audit_error}")

        return [trade_parent, trade_tp, trade_sl]

    # ==================== 高级执行算法接口 ====================
    
    async def execute_large_order(self, symbol: str, action: str, quantity: int, 
                                algorithm: str = "TWAP", **kwargs):
        """执行大订单智能算法
        
        Args:
            symbol: 股票代码
            action: BUY/SELL
            quantity: 总数量
            algorithm: 执行算法 ("TWAP", "VWAP", "ICEBERG")
            **kwargs: 算法特定参数
        """
        if quantity < 1000:  # 小订单直接使use普通market单
            return await self.place_market_order(symbol, action, quantity)
        
        self.logger.info(f"starting大订单执行: {symbol} {action} {quantity}股, 算法: {algorithm}")
        
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
                raise ValueError(f"not支持算法: {algorithm}")
            
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
                self.logger.warning(f"大订单审计记录failed: {audit_error}")
            
            # 刷新account信息
            await self.refresh_account_balances_and_positions()
            
            return results
            
        except Exception as e:
            self.logger.error(f"大订单执行failed {symbol}: {e}")
            raise

    async def place_limit_order(self, symbol: str, action: str, quantity: int, limit_price: float) -> OrderRef:
        # before置校验
        if not await self._validate_order_before_submission(symbol, action, quantity, limit_price):
            raise RuntimeError("订单before置校验未通过")
        
        # 纯路bytoEnhancedOrderExecutor执行limit单
        from .enhanced_order_execution import ExecutionConfig, ExecutionAlgorithm
        exec_cfg = ExecutionConfig(algorithm=ExecutionAlgorithm.LIMIT)
        order_sm = await self.enhanced_executor.execute_limit_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            config=exec_cfg,
        )

        # 统一返回 OrderRef（and现has调use方兼容）
        enhanced_ref = OrderRef(
            order_id=order_sm.order_id,
            symbol=symbol,
            side=action,
            qty=quantity,
            order_type="LMT",
            limit_price=limit_price,
        )
        
        # 审计记录通过OrderManager回调自动处理，no需重复记录
        
        # updates计数
        self._daily_order_count += 1
        
        # 刷新account信息
        await self.refresh_account_balances_and_positions()
        
        return enhanced_ref

    async def plan_and_place_with_rr(
        self,
        model_signals: List[dict],
        polygon_metrics: Dict[str, dict],
        polygon_quotes: Dict[str, dict],
        portfolio_nav: float,
        current_positions: Dict[str, int],
    ) -> List[OrderRef]:
        """in延迟环境下使use RiskRewardController 规划并下limit单。
        - model_signals: [{symbol, side, expected_alpha_bps, model_price, confidence}]
        - polygon_metrics: {symbol: {prev_close, atr_14, adv_usd_20, median_spread_bps_20, sigma_15m}}
        - polygon_quotes:  {symbol: {last, tickSize}}
        """
        refs: List[OrderRef] = []
        if not hasattr(self, 'rr_controller') or not hasattr(self, 'rr_cfg') or not self.rr_cfg or not self.rr_cfg.enabled:
            self.logger.info("RiskRewardController 未启use，跳过规划")
            return refs
        try:
            from .enhanced_order_execution import Signal, Metrics, Quote
            signals: List[Signal] = []
            for s in model_signals:
                sym = s.get('symbol')
                if not sym:
                    continue
                signals.append(Signal(
                    symbol=sym,
                    side=s.get('side', 'BUY'),
                    expected_alpha_bps=float(s.get('expected_alpha_bps', 0) or 0.0),
                    model_price=s.get('model_price'),
                    confidence=float(s.get('confidence', 1.0) or 1.0)
                ))
            metrics: Dict[str, Metrics] = {}
            for sym, md in polygon_metrics.items():
                metrics[sym] = Metrics(
                    prev_close=float(md.get('prev_close', 0) or 0.0),
                    atr_14=(float(md.get('atr_14')) if md.get('atr_14') is not None else None),
                    adv_usd_20=(float(md.get('adv_usd_20')) if md.get('adv_usd_20') is not None else None),
                    median_spread_bps_20=(float(md.get('median_spread_bps_20')) if md.get('median_spread_bps_20') is not None else None),
                    sigma_15m=(float(md.get('sigma_15m')) if md.get('sigma_15m') is not None else None),
                )
            quotes: Dict[str, Quote] = {}
            for sym, q in polygon_quotes.items():
                quotes[sym] = Quote(
                    last=(float(q.get('last')) if q.get('last') is not None else None),
                    tickSize=float(q.get('tickSize', 0.01) or 0.01),
                    source='DELAYED'
                )
            planned = self.rr_controller.plan_orders(
                model_signals=signals,
                metrics=metrics,
                quotes_delayed=quotes,
                portfolio_nav=float(portfolio_nav or 0.0),
                current_positions=current_positions or {}
            )
            for p in planned:
                ref = await self.place_limit_order(p['symbol'], p['side'], int(p['quantity']), float(p['limit']))
                refs.append(ref)
            return refs
        except Exception as e:
            self.logger.error(f"RR 规划order placementfailed: {e}")
            return refs

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
            # 使use订单管理器跟踪
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
        self.logger.info(f"提交bracket order: {[r.order_id for r in refs]}")
        await self.refresh_account_balances_and_positions()
        # forBUY父单，初始化动态止损状态
        try:
            if action.upper() == "BUY":
                st = self._stop_state.get(symbol, {})
                st["entry_price"] = entry_price
                st["entry_time"] = datetime.now()
                st["qty"] = self.position_manager.get_quantity(symbol)
                self._stop_state[symbol] = st
                # 使use任务管理器start止损任务
                task_id = f"stop_manager_{symbol}"
                if task_id not in self._active_tasks:
                    try:
                        task = asyncio.create_task(self._dynamic_stop_manager(symbol))
                        self._active_tasks[task_id] = task
                    except Exception as e:
                        self.logger.error(f"start止损任务failed {symbol}: {e}")
        except Exception:
            pass
        
        # 审计记录 - bracket order
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
            self.logger.warning(f"bracket order审计记录failed: {audit_error}")
        
        return refs

    def cancel_all_open_orders(self) -> None:
        self.ib.reqGlobalCancel()  # EClient.reqGlobalCancel / cancelOrder
        self.logger.info("请求撤销全部未completed订单")

    async def graceful_shutdown(self) -> None:
        """优雅关闭：取消订单、断开connection、清理资源"""
        try:
            self.logger.info("starting优雅关闭...")
            
            # 1. 停止事件循环
            if self._stop_event:
                self._stop_event.set()
                
            # 2. 取消所has未completed订单
            try:
                self.cancel_all_open_orders()
                await asyncio.sleep(2)  # 等待取消生效
            except Exception as e:
                self.logger.warning(f"取消订单when出错: {e}")
                
            # 3. 清理市场数据订阅（纯交易模式下不需要）
            try:
                self.cleanup_unused_subscriptions()
                await asyncio.sleep(1)  # 给服务器时间处理
            except Exception as e:
                self.logger.warning(f"清理市场数据订阅时出错: {e}")
                
            # 4. 通过connection管理器断开connection
            try:
                await self.connection_manager.disconnect()
                self.logger.info("通过connection管理器断开IBKRconnection")
            except Exception as e:
                self.logger.warning(f"断开connectionwhen出错: {e}")
                
            # 5. 清理状态
            self.tickers.clear()
            self.last_price.clear()
            self.position_manager.clear_all_positions()
            self.account_values.clear()
            
            self.logger.info("优雅关闭completed")
            
        except Exception as e:
            self.logger.error(f"优雅关闭when发生错误: {e}")
            
    async def health_check(self) -> dict:
        """Enhanced system health check with comprehensive monitoring"""
        try:
            status = {
                "connected": self.ib.isConnected(),
                "subscribed_symbols": len(self.tickers),
                "open_orders": len(self.open_orders),
                "positions": self.position_manager.get_portfolio_summary().total_positions,
                "net_liquidation": self.net_liq,
                "cash_balance": self.cash_balance,
                "account_ready": self.account_ready,
                "last_update": time.time()
            }
            
            # 使useconnection管理器checkconnection状态
            if not status["connected"]:
                self.logger.warning(" Health check: IBKR connection lost")
                
                # 通过connection管理器触发重连
                try:
                    reconnect_success = await self.connection_manager.reconnect()
                    if reconnect_success:
                        self.logger.info(" 自动重连success")
                    else:
                        self.logger.error(" 自动重连failed")
                except Exception as e:
                    self.logger.error(f"重连过程异常: {e}")
                
            # Check account status
            if not status["account_ready"]:
                self.logger.warning(" Health check: Account information not ready")
                
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
                self.logger.warning(f" Health check: {stale_count} symbols with stale data: {', '.join(stale_symbols[:5])}")
                
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
            self.logger.error(f" Health check failed: {e}")
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

    # ------------------------- account/positions/PnL 回调 -------------------------
    def _on_account_summary(self, *args) -> None:
        """兼容 ib_insync  accountSummaryEvent 单 records触发and批量行。"""
        try:
            rows = []
            if len(args) == 1:
                first = args[0]
                if isinstance(first, (list, tuple)):
                    rows = list(first)
                else:
                    rows = [first]
            else:
                rows = list(args)

            for r in rows:
                try:
                    if isinstance(r, str):
                        continue
                    tag = getattr(r, 'tag', getattr(r, 'key', None))
                    value = getattr(r, 'value', None)
                    currency = getattr(r, 'currency', None)
                    if tag is None:
                        continue
                    key = f"{tag}:{currency or ''}"
                    self.account_values[key] = value
                except Exception:
                    continue

            try:
                cash = float(self.account_values.get(f"TotalCashValue:{self.default_currency}", "0") or 0)
                netliq = float(self.account_values.get(f"NetLiquidation:{self.default_currency}", "0") or 0)
                if cash > 0:
                    self.cash_balance = cash
                if netliq > 0:
                    self.net_liq = netliq
            except Exception:
                pass
        except Exception:
            pass

    def _on_update_account_value(self, *args, **_kwargs) -> None:
        """Handle both legacy and ib_insync-style account value events.

        Supported payloads:
        - (tag, value, currency, account)
        - (AccountValue(tag=..., value=..., currency=..., account=...),)
        """
        try:
            tag = value = currency = account = None
            if len(args) == 1:
                av = args[0]
                # ib_insync AccountValue dataclass/tuple
                tag = getattr(av, 'tag', getattr(av, 'key', None))
                value = getattr(av, 'value', None)
                currency = getattr(av, 'currency', None)
                account = getattr(av, 'account', None)
            elif len(args) >= 4:
                tag, value, currency, account = args[:4]
            else:
                return

            if tag is None:
                return

            key = f"{tag}:{currency or ''}"
            self.account_values[key] = value

            # Opportunistically keep cash/net liq in sync when currency matches
            try:
                cur_ok = (currency in (self.default_currency, 'BASE', None))
                if cur_ok and tag == 'TotalCashValue':
                    self.cash_balance = float(value)
                elif cur_ok and tag == 'NetLiquidation':
                    self.net_liq = float(value)
            except Exception:
                pass
        except Exception as e:
            try:
                self.logger.debug(f"account值回调处理异常: {e}")
            except Exception:
                pass

    def _on_update_portfolio(self, *args, **kwargs) -> None:
        try:
            symbol = None
            quantity = 0
            if len(args) == 1 and hasattr(args[0], 'contract'):
                item = args[0]
                c = getattr(item, 'contract', None)
                symbol = getattr(c, 'symbol', None)
                quantity = int(getattr(item, 'position', 0) or 0)
            elif len(args) >= 2:
                position, contract = args[:2]
                symbol = getattr(contract, 'symbol', None)
                quantity = int(position)
            if not symbol:
                return
            current_price = self.get_price(symbol) or 100.0
            
            # 异步updatespositions（in事件循环in执行）
            asyncio.create_task(
                self.position_manager.update_position(symbol, quantity, current_price)
            )
        except Exception as e:
            self.logger.debug(f"positions组合updatesfailed: {e}")

    def _on_position(self, *args) -> None:
        try:
            symbol = None
            quantity = 0
            avgCost = None
            if len(args) == 1 and hasattr(args[0], 'contract'):
                p = args[0]
                c = getattr(p, 'contract', None)
                symbol = getattr(c, 'symbol', None)
                quantity = int(getattr(p, 'position', 0) or 0)
                avgCost = getattr(p, 'avgCost', None)
            elif len(args) >= 4:
                _, contract, position, avgCost = args[:4]
                symbol = getattr(contract, 'symbol', None)
                quantity = int(position)
            if not symbol:
                return
            current_price = self.get_price(symbol) or (float(avgCost) if avgCost and avgCost > 0 else 100.0)
            
            # 异步updatespositions（in事件循环in执行）
            asyncio.create_task(
                self.position_manager.update_position(symbol, quantity, current_price, float(avgCost) if avgCost else None)
            )
        except Exception as e:
            self.logger.debug(f"positions事件updatesfailed: {e}")

    # ------------------------- 订单/execution/commission 回调 -------------------------
    def _on_order_status(self, trade) -> None:
        try:
            o = trade.order
            s = trade.orderStatus
            self.logger.info(
                f"订单状态: id={o.orderId} permId={o.permId} status={s.status} filled={s.filled} remaining={s.remaining} avgFillPrice={s.avgFillPrice}"
            )
            # 同步to订单状态机
            try:
                from .order_state_machine import OrderState
                status = getattr(s, 'status', '')
                if status == 'Filled':
                    # 简化订单状态updates
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                        o.orderId, OrderState.FILLED,
                        {"filled_quantity": int(getattr(s, 'filled', 0) or 0),
                         "avg_fill_price": float(getattr(s, 'avgFillPrice', 0.0) or 0.0)}
                        ))
                    except Exception as e:
                        self.logger.error(f"updates订单状态failed: {e}")
                elif status in {'Cancelled', 'ApiCancelled'}:
                    # 简化订单状态updates
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                        o.orderId, OrderState.CANCELLED
                        ))
                    except Exception as e:
                        self.logger.error(f"updates订单状态failed: {e}")
                elif status in {'Inactive', 'Rejected'}:
                    # 简化订单状态updates
                    try:
                        task = asyncio.create_task(self.order_manager.update_order_state(
                        o.orderId, OrderState.REJECTED
                        ))
                    except Exception as e:
                        self.logger.error(f"updates订单状态failed: {e}")
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
                f"execution: orderId={trade.order.orderId} symbol={symbol} side={side} qty={qty} price={price}"
            )

            # updates动态止损状态
            if side == "BUY" and qty > 0:
                state = self._stop_state.get(symbol, {})
                # 使use最新execution价作as入场价（can扩展as加权平均）
                state["entry_price"] = float(state.get("entry_price") or price or 0.0) or price
                state["entry_time"] = datetime.now()
                # 同步当beforepositions
                try:
                    held = int(self.position_manager.get_quantity(symbol))
                    state["qty"] = held
                except Exception:
                    state["qty"] = qty
                self._stop_state[symbol] = state

                # start/确保动态止损任务
                task_id = f"stop_manager_{symbol}"
                if task_id not in self._active_tasks:
                    try:
                        task = asyncio.create_task(self._dynamic_stop_manager(symbol))
                        self._active_tasks[task_id] = task
                    except Exception as e:
                        self.logger.error(f"start止损任务failed {symbol}: {e}")

            elif side == "SELL" and qty > 0:
                # if果仓位清零，取消has止损并停止任务
                try:
                    held = int(self.position_manager.get_quantity(symbol))
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

    def _on_commission(self, *args) -> None:
        try:
            report = args[-1] if args else None
            if report is None:
                return
            exec_id = getattr(report, 'execId', '')
            commission = getattr(report, 'commission', 0.0)
            currency = getattr(report, 'currency', '')
            realized = getattr(report, 'realizedPNL', 0.0)
            self.logger.info(f"commission: execId={exec_id} commission={commission} currency={currency} realizedPNL={realized}")
        except Exception:
            pass

    # ------------------------- 其他回调/错误 -------------------------
    def _on_error(self, reqId, errorCode, errorString, contract) -> None:
        """增强错误处理，包括自动重连and详细错误分类"""
        try:
            msg = f"IBKR错误: reqId={reqId} code={errorCode} msg={errorString}"
            if contract:
                msg += f" contract={contract}"
            
            # 根据错误严重程度使usenot同日志级别
            if errorCode in (10167, 354, 2104, 2106, 2158):  # 非致命错误
                self.logger.warning(msg)
            elif errorCode in (504, 1100, 1101, 1102, 2110):  # connection错误
                self.logger.error(msg)
            else:
                self.logger.info(msg)
            
            # real-time数据权限错误，自动切换to延迟数据
            if errorCode in (10167, 354):  # noreal-time数据权限
                if self.use_delayed_if_no_realtime:
                    self.logger.warning("noreal-time数据权限，自动切换as延迟数据（10-20分钟延迟）")
                    try:
                        self.ib.reqMarketDataType(3)  # 切换to延迟数据
                        self.logger.info("success切换to延迟市场数据")
                    except Exception as e:
                        self.logger.error(f"切换to延迟数据failed: {e}")
                else:
                    self.logger.error("noreal-time数据权限，且not允许使use延迟数据。请check数据subscription。")
                    self.logger.error("请check您Professional US Securities Bundlesubscription状态")
                    
            # connection相关错误处理
            elif errorCode in (504, 1100, 1101):  # connection丢失
                self.logger.error("检测toconnection丢失，系统willin下次操作when尝试重新connection")
                
            # 订单相关错误
            elif errorCode in (201, 202, 203, 399):  # 订单be拒绝
                self.logger.error(f"订单be拒绝: {errorString}")
                
            # 市场数据错误
            elif errorCode in (200, 162, 321):  # 市场数据错误
                self.logger.warning(f"市场数据错误: {errorString}")
                
        except Exception as e:
            self.logger.error(f"处理错误回调when发生异常: {e}")

    def _on_current_time(self, time_: datetime) -> None:
        self.logger.debug(f"服务器when间 {time_}")

    # ------------------------- simple策略/演示（移除） -------------------------
    async def run_demo(self, symbols: List[str], target_allocation_per_symbol: float, max_symbols: int = 5) -> None:
        raise RuntimeError("Simplified demo strategy has been removed. Use Engine via GUI/launcher.")

    # ------------------------- 关闭 -------------------------
    async def close(self) -> None:
        try:
            # 通过任务生命周期管理器清理所has任务
            self.task_manager.cancel_group("system_monitoring", "系统关闭")
            self.task_manager.cancel_group("stop_loss_management", "系统关闭")
            self.logger.info("清理所has管理任务")
        except Exception as e:
            self.logger.warning(f"任务清理failed: {e}")
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
                # 支持 {"tickers": [...]} or {"symbols": [...]} 结构
                ticks = data.get('tickers') or data.get('symbols') or []
            else:
                ticks = []
            return [str(x).strip().upper() for x in ticks if str(x).strip()]
        except Exception:
            return []

    @staticmethod
    def _load_from_excel(path: str, sheet: Optional[str] = None, column: Optional[str] = None) -> List[str]:
        # 优先使use pandas；未安装when返回空并提示
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
                    # 退化as第一列
                    series = df.iloc[:, 0]
            ticks = [str(x).strip().upper() for x in series.tolist() if str(x).strip()]
            return ticks
        except Exception as e:
            logging.getLogger('IbkrAutoTrader').warning(f'读取Excelfailed: {e}')
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
        """统一from多源加载。保留该方法useat一次性导入to数据库，但in交易循环in我们改as直接fromSQLite读取。"""
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
        """观察列表自动交易（含保守risk control）

        risk control要点：
        - 现金保留：≥15% 净值not参andorder placement
        - 单标上限：≤12% 净值
        - 最小order placement金额：≥$500
        - 日内订单上限：≤20 单
        - price区间过滤：$2 - $800
        - 重复positions跳过
        - 交易when段check（美东 9:30 - 16:00）
        """

        import math
        from datetime import datetime, time as dtime

        self._stop_event = self._stop_event or asyncio.Event()
        # 热加载风险配置（DB 优先）
        try:
            self.load_risk_config_from_db()
        except Exception:
            pass

        # risk control参数（canafter续外部化to配置）
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

        # 交易循环：支持外部输入（JSON/Excel/CSV）and数据库动态合并
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
            """专业级多因子信号系统：综合技术面、基本面and市场情绪"""
            try:
                # retrieval足够历史数据
                contract = await self.qualify_stock(sym)
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr="120 D",  # 增加to120天retrieval更多数据
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                if len(bars) < 50:  # 至少需要50天数据
                    self.logger.debug(f"{sym} 历史数据not足: {len(bars)}天")
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
                
                # 趋势排列 (短>in>长均线as多头)
                if sma_5 > sma_20 > sma_50:
                    trend_score += 0.4
                elif sma_5 > sma_20:
                    trend_score += 0.2
                elif current_price > sma_20:
                    trend_score += 0.1
                
                # 均线斜率 (均线to上as正面)
                sma20_slope = (sma_20 - sum(closes[-25:-5])/20) / sum(closes[-25:-5])/20
                if sma20_slope > 0.01:  # 1%以上上升
                    trend_score += 0.3
                elif sma20_slope > 0:
                    trend_score += 0.1
                
                # price相for位置
                if current_price > sma_5 * 1.02:  # price超过5日均线2%
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
                
                if momentum_5d > 0.02 and momentum_20d > 0:  # 短期强势且in期to上
                    momentum_score += 0.3
                elif momentum_5d > -0.02:  # 短期稳定
                    momentum_score += 0.1
                
                # === 3. execution量分析 (20%权重) ===
                volume_score = 0.0
                
                if len(volumes) >= 20:
                    avg_volume_20d = sum(volumes[-20:]) / 20
                    current_volume = volumes[-1]
                    
                    # execution量放大
                    volume_ratio = current_volume / avg_volume_20d
                    if volume_ratio > 1.5:  # 放量1.5倍
                        volume_score += 0.4
                    elif volume_ratio > 1.2:
                        volume_score += 0.2
                    elif volume_ratio > 0.8:  # 正常execution量
                        volume_score += 0.1
                    
                    # execution量趋势 (近5日vsbefore15日)
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
                
                # === 4. 波动率and风险分析 (15%权重) ===
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
                    if 0.2 <= bb_position <= 0.8:  # in布林带in间区域
                        volatility_score += 0.3
                    elif bb_position < 0.2:  # 布林带下轨附近，超卖
                        volatility_score += 0.2
                
                # === 5. 市场环境and相for强度 (10%权重) ===
                market_score = 0.0
                
                # retrieval市场基准 (简化asSPY/QQQfor比，实际can以更复杂)
                try:
                    # 这里简化处理，实际应该retrievalSPY数据
                    # 假设当beforeis良好市场环境
                    market_score += 0.5  # 基础市场环境得分
                except:
                    market_score += 0.3  # 默认in性环境
                
                # 个股相for强度 (vs 20日均线相for表现)
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
                self.logger.debug(f"{sym} 多因子评分: 总分{total_score:.2f} (趋势{trend_score:.2f} 动量{momentum_score:.2f} execution量{volume_score:.2f} 波动{volatility_score:.2f} 市场{market_score:.2f})")
                
                # 评分阈值：0.6以上通过 (满分1.0)
                approval_threshold = 0.6
                approved = total_score >= approval_threshold
                
                if approved:
                    self.logger.info(f"{sym}  技术分析通过: {total_score:.2f}/{approval_threshold}")
                else:
                    self.logger.debug(f"{sym}  技术分析not通过: {total_score:.2f}/{approval_threshold}")
                
                return approved
                
            except Exception as e:
                self.logger.warning(f"{sym} 技术指标计算failed: {e}")
                return False

        while not self._stop_event.is_set():
            try:
                # 日内计数重置
                today = datetime.now().date()
                if last_reset_day != today:
                    daily_order_count = 0
                    last_reset_day = today

                # 交易when段check
                if not is_trading_hours():
                    await asyncio.sleep(min(poll_sec * 2, 300))
                    continue

                # 刷新accountand资金
                await self.refresh_account_balances_and_positions()
                if self.net_liq <= 0:
                    self.logger.warning("account净值as0，等待account数据...")
                    await asyncio.sleep(poll_sec)
                    continue

                # 加载最新风险配置
                try:
                    current_risk_config = db.get_risk_config("默认风险配置")
                    if current_risk_config:
                        # updates风险参数
                        max_single_position_pct = current_risk_config.get("max_single_position_pct", 0.1)
                        max_daily_orders = current_risk_config.get("max_daily_orders", 5)
                        min_order_value_usd = current_risk_config.get("min_order_value_usd", 100)
                        self.logger.debug(f"Risk configuration loaded: 单笔限制{max_single_position_pct*100:.1f}%, 日内最多{max_daily_orders}单")
                except Exception as e:
                    self.logger.warning(f"加载风险配置failed，使use默认值: {e}")
                    # 保持原has默认值

                reserved_cash = self.net_liq * cash_reserve_pct
                available_cash = max((self.cash_balance or 0.0) - reserved_cash, 0.0)
                if available_cash < min_order_value_usd:
                    self.logger.info("canuse现金not足，等待...")
                    await asyncio.sleep(poll_sec)
                    continue

                # 合并外部输入and数据库
                desired_list = _compute_desired_list()
                desired: set[str] = set(desired_list)

                # 仅处理新增标
                added = [s for s in desired_list if s not in last_desired]

                # 并发处理新增标 - 分批并发
                orders_sent_this_cycle = 0
                max_concurrent_processing = 5  # 最多同when处理5个标
                
                async def process_symbol_for_trading(sym: str) -> Optional[Dict[str, Any]]:
                    """处理单个标，返回交易参数orNone"""
                    try:
                        # 准备交易合约（纯交易模式：不订阅数据）
                        await self.prepare_symbol_for_trading(sym)
                        await asyncio.sleep(0.1)

                        # 重复positions跳过
                        if int(self.position_manager.get_quantity(sym)) > 0:
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
                        self.logger.warning(f"处理标failed {sym}: {e}")
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
                    self.logger.info(f"并发处理标批次: {len(batch)}个 ({i+1}-{min(i+batch_size, len(added))}/{len(added)})")
                    
                    # 并发处理当before批次
                    batch_results = await asyncio.gather(
                        *[process_with_semaphore(sym) for sym in batch],
                        return_exceptions=True
                    )
                    
                    # 处理结果，按顺序order placement
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
                            # 计算order placement股数（使use ATR 风险因子 + 资金分配）
                            if fixed_qty > 0:
                                # 固定股数模式：需要额外验证资金充足性
                                qty = int(fixed_qty)
                                fixed_order_value = qty * price
                                
                                # 固定股数超出资金限制when，按资金上限重新计算
                                max_affordable_by_cash = int(available_cash // price) if price > 0 else 0
                                max_affordable_by_position = int((self.net_liq * max_single_position_pct) // price) if price > 0 else 0
                                max_affordable = min(max_affordable_by_cash, max_affordable_by_position)
                                
                                if qty > max_affordable:
                                    self.logger.warning(f"{sym} 固定股数{qty}超出资金限制，调整as{max_affordable}")
                                    qty = max_affordable
                            else:
                                rf = await self._compute_risk_factor(sym, price)
                                budget = self.allocate_funds(sym, risk_factor=rf) * alloc  # in分配上再乘以策略 alloc
                                qty = int(budget // price) if price > 0 else 0

                                # 最小order placement金额保护
                                if qty * price < min_order_value_usd:
                                    qty = max(int(math.ceil(min_order_value_usd / price)), 1)

                            order_value = qty * (price or 0.0)

                            # 最终资金andrisk controlcheck
                            if qty <= 0 or order_value > available_cash or order_value > self.net_liq * max_single_position_pct:
                                self.logger.info(f"{sym} 资金check未通过: qty={qty}, order_value=${order_value:.2f}, available_cash=${available_cash:.2f}")
                                continue

                            # order placement并跟踪状态
                            order_success = False
                            try:
                                # 使use当before风险配置策略参数
                                strategy_config = None
                                if current_risk_config and "strategy_configs" in current_risk_config:
                                    strategy_config = current_risk_config["strategy_configs"].get("swing", {
                                        "stop_pct": current_risk_config.get("default_stop_pct", 0.03),
                                        "target_pct": current_risk_config.get("default_target_pct", 0.08)
                                    })
                                await self.place_market_order_with_bracket(sym, "BUY", qty, strategy_type="swing", use_config=True, custom_config=strategy_config)
                                order_success = True
                                self.logger.info(f"bracket order提交success: {sym} {qty}股")
                            except Exception as _e:
                                self.logger.warning(f"bracket orderorder placementfailed，回退asmarket单 {sym}: {_e}")
                                await self._notify_webhook("bracket_fallback", "bracket orderfailed回退", f"{sym} 回退asmarket单", {"error": str(_e)})
                                try:
                                    await self.place_market_order(sym, "BUY", qty)
                                    order_success = True
                                    self.logger.info(f"market单提交success: {sym} {qty}股")
                                except Exception as __e:
                                    self.logger.error(f"market单也failed: {sym}: {__e}")
                                    order_success = False
                            
                            # 仅in订单success提交after才增加计数and扣减资金
                            if order_success:
                                                                # 确保动态止损任务start
                                try:
                                    task_id = f"stop_manager_{sym}"
                                    self.task_manager.create_task(
                                        task_id, self._dynamic_stop_manager, sym,
                                        max_restarts=10, restart_delay=5.0
                                    )
                                except Exception as e:
                                    self.logger.warning(f"start止损任务failed {sym}: {e}")
                                
                                orders_sent_this_cycle += 1
                                daily_order_count += 1
                                available_cash -= order_value
                                self.logger.info(f"订单计数updates: 本轮{orders_sent_this_cycle}, 日内{daily_order_count}")      
                            else:
                                self.logger.warning(f"{sym} 订单提交failed，not计入统计")
                            
                            await asyncio.sleep(0.2)
                        except Exception as e:
                            self.logger.warning(f"处理新增 {sym} failed: {e}")
                    
                    # if果批次处理过多，短暂暂停
                    if i + batch_size < len(added):
                        await asyncio.sleep(0.5)  # 批次间隔0.5 seconds

                # 清仓be移除：仅当标from数据库in消失when卖出
                removed = [s for s in last_desired if s not in desired]
                if removed and auto_sell_removed:
                    for sym in removed:
                        try:
                            if daily_order_count >= daily_order_limit:
                                break
                            qty = int(self.position_manager.get_quantity(sym))
                            if qty > 0:
                                # forat removed 自动清仓，始终使use直接market以避免意外重建仓位
                                await self.place_market_order(sym, "SELL", qty)
                                # 通过position_manager.update_position(sym, 0, current_price)清仓
                                daily_order_count += 1
                                await self.refresh_account_balances_and_positions()
                                await asyncio.sleep(0.2)
                            self.unsubscribe(sym)
                        except Exception as e:
                            self.logger.warning(f"处理移除 {sym} failed: {e}")

                last_desired = desired

                # real-time信号处理（每 seconds一次，独立at poll_sec）
                try:
                    if desired:
                        for sym in list(desired)[:50]:  # 限制每轮处理标数量
                            # 确保合约准备就绪（纯交易模式）
                            if not await self.prepare_symbol_for_trading(sym):
                                self.logger.warning(f"{sym} 交易准备失败，跳过")
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
                            # retrieval/初始化引擎
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
                                # 二次risk control
                                side = "BUY" if sig.action in (ActionType.BUY_NOW, ActionType.BUY_LIMIT) else "SELL"
                                ok = await self._validate_order_before_submission(sym, side, max(1, int(self.position_manager.get_quantity(sym) or 1)), sig.entry_price)
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
                                            self.logger.warning(f"real-time买入bracket orderfailed，回退market {sym}: {_e}")
                                            await self.place_market_order(sym, "BUY", qty)
                                elif sig.action == ActionType.SELL_NOW:
                                    qty = int(self.position_manager.get_quantity(sym))
                                    if qty > 0:
                                        await self.place_market_order(sym, "SELL", qty)
                except Exception as _e:
                    self.logger.debug(f"real-time信号处理异常: {_e}")

                await asyncio.sleep(1.0)

            except Exception as loop_err:
                self.logger.error(f"观察列表交易循环错误: {loop_err}")
                await asyncio.sleep(poll_sec)
    
    # =================== Polygon统一因子集成方法 ===================
    
    def enable_polygon_factors(self):
        """启usePolygon因子"""
        if POLYGON_INTEGRATED and hasattr(self, 'polygon_unified') and self.polygon_unified:
            try:
                enable_polygon_factors()
                self.polygon_enabled = True
                self.logger.info("Polygon因子启use")
            except Exception as e:
                self.logger.error(f"启usePolygon因子failed: {e}")
    
    def enable_polygon_risk_balancer(self):
        """启usePolygonrisk control收益平衡器"""
        if POLYGON_INTEGRATED and hasattr(self, 'polygon_unified') and self.polygon_unified:
            try:
                enable_polygon_risk_balancer()
                self.polygon_risk_balancer_enabled = True
                self.logger.info("Polygonrisk control收益平衡器启use")
            except Exception as e:
                self.logger.error(f"启usePolygonrisk control收益平衡器failed: {e}")
    
    def disable_polygon_risk_balancer(self):
        """禁usePolygonrisk control收益平衡器"""
        if POLYGON_INTEGRATED and hasattr(self, 'polygon_unified') and self.polygon_unified:
            try:
                disable_polygon_risk_balancer()
                self.polygon_risk_balancer_enabled = False
                self.logger.info("Polygonrisk control收益平衡器禁use")
            except Exception as e:
                self.logger.error(f"禁usePolygonrisk control收益平衡器failed: {e}")
    
    def is_polygon_risk_balancer_enabled(self) -> bool:
        """checkPolygonrisk control收益平衡器状态"""
        if POLYGON_INTEGRATED and hasattr(self, 'polygon_unified') and self.polygon_unified:
            return self.polygon_unified.is_risk_balancer_enabled()
        return False
    
    def check_polygon_trading_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        使usePolygon数据check交易 records件(替代原has超短期判断)
        """
        if not POLYGON_INTEGRATED or not hasattr(self, 'polygon_unified') or not self.polygon_unified:
            return {'can_trade': False, 'reason': 'Polygon未集成'}
        
        return check_polygon_trading_conditions(symbol)
    
    def process_signals_with_polygon_risk_control(self, signals) -> List[Dict]:
        """
        使usePolygonrisk control收益平衡器处理信号
        """
        if not POLYGON_INTEGRATED or not hasattr(self, 'polygon_unified') or not self.polygon_unified:
            self.logger.warning("Polygon未集成，使use基础信号处理")
            return self._process_signals_basic(signals)
        
        try:
            return process_signals_with_polygon(signals)
        except Exception as e:
            self.logger.error(f"Polygon信号处理failed: {e}")
            return self._process_signals_basic(signals)
    
    def _process_signals_basic(self, signals) -> List[Dict]:
        """基础信号处理(fallback)"""
        orders = []
        
        try:
            if hasattr(signals, 'to_dict'):  # pandas DataFrame
                signal_data = signals.to_dict('records')
            elif isinstance(signals, list):
                signal_data = signals
            else:
                return orders
            
            for signal in signal_data:
                symbol = signal.get('symbol', '')
                prediction = signal.get('weighted_prediction', 0)
                
                # 简单阈值过滤
                if abs(prediction) < 0.005:  # 0.5%
                    continue
                
                side = "BUY" if prediction > 0 else "SELL"
                
                orders.append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': 100,  # 固定数量
                    'order_type': 'MKT',
                    'source': 'basic_processing'
                })
            
            self.logger.info(f"基础处理生成{len(orders)}个订单")
            
        except Exception as e:
            self.logger.error(f"基础信号处理failed: {e}")
        
        return orders
    
    def get_polygon_stats(self) -> Dict[str, Any]:
        """retrievalPolygon统计信息"""
        if not POLYGON_INTEGRATED or not hasattr(self, 'polygon_unified') or not self.polygon_unified:
            return {}
        
        try:
            return self.polygon_unified.get_stats()
        except Exception as e:
            self.logger.error(f"retrievalPolygon统计failed: {e}")
            return {}
    
    def clear_polygon_cache(self):
        """清理Polygon缓存"""
        if POLYGON_INTEGRATED and hasattr(self, 'polygon_unified') and self.polygon_unified:
            try:
                self.polygon_unified.clear_cache()
                self.logger.info("Polygon缓存清理")
            except Exception as e:
                self.logger.error(f"清理Polygon缓存failed: {e}")


# ----------------------------- CLI 入口 -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IBKR automated trading minimal closed-loop script")
    p.add_argument("--host", default="127.0.0.1", help="TWS/IB Gateway 主机")
    p.add_argument("--port", type=int, default=7497, help="TWS(7497)/IBG(4002) 端口")
    p.add_argument("--client-id", type=int, default=123, help="客户端ID，避免and其他程序冲突")
    # 直接演示参数
    p.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL", help="逗号分隔股票代码（手动传入）")
    p.add_argument("--alloc", type=float, default=0.05, help="每只标目标资金ratio，例if 0.05 表示 5%")
    p.add_argument("--max", type=int, default=5, help="最多处理标数量（演示模式）")
    # 观察列表/自动交易
    p.add_argument("--json", type=str, default=None, help="from JSON 文件加载股票列表（支持 [..] or {tickers:[..]}）")
    p.add_argument("--excel", type=str, default=None, help="from Excel 文件加载股票列表（需安装 pandas/openpyxl）")
    p.add_argument("--sheet", type=str, default=None, help="Excel 工作表名（默认第1个）")
    p.add_argument("--column", type=str, default=None, help="Excel 列名（默认自动识别 ticker/symbol/code or第1列）")
    p.add_argument("--watch-alloc", type=float, default=0.03, help="观察列表模式每只标目标资金ratio")
    p.add_argument("--poll", type=float, default=10.0, help="观察列表轮询 seconds数")
    p.add_argument("--auto-sell-removed", action="store_true", help="from列表删除即自动全清仓")
    p.add_argument("--fixed-qty", type=int, default=0, help="固定order placement股数（>0 生效，优先at资金ratio）")
    p.add_argument("--no-delayed", action="store_true", help="no权限whennot自动切todelayedmarket data")
    p.add_argument("--verbose", action="store_true", help="调试日志")
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> None:
    # 日志配置移至主入口点
    # 注意：此函数仅供内部测试使use，主入口请使uselauncher.py
    from .unified_config import get_unified_config
    config_manager = get_unified_config()
    trader = IbkrAutoTrader(config_manager=config_manager)

    # 优雅退出
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _stop(*_sig):
        logging.getLogger("amain").info("收to停止信号，准备退出...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop, sig)
        except NotImplementedError:
            pass

    try:
        await trader.connect()

        # 简化策略移除，这里仅保留connection/健康check。自动交易请使use GUI/launcher  Engine 模式。
        logging.getLogger("amain").info("Simplified strategies removed. Use Engine via GUI/launcher.")

        await stop_event.wait()
    finally:
        await trader.close()


# 主入口点移至launcher.py，此文件not再需要独立运行
# if需独立测试，请使use: python autotrader/launcher.py

