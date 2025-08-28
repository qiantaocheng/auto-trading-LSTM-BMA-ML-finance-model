# 清理：移除未使use导入
# from __future__ import annotations
# import asyncio
# import logging
# from typing import Dict

import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# 改use统一配置管理器
# from .config import HotConfig
# 使用统一Polygon因子库，替代原有因子函数
from .unified_polygon_factors import get_unified_polygon_factors, zscore, atr, get_trading_signal_for_autotrader
from typing import TYPE_CHECKING

# Enhanced error handling
from .error_handling_system import (
    get_error_handler, with_error_handling, error_handling_context,
    ErrorSeverity, ErrorCategory, ErrorContext
)

if TYPE_CHECKING:
    from .ibkr_auto_trader import IbkrAutoTrader


@dataclass
class Quote:
    bid: float
    ask: float
    bidSize: float = 0.0
    askSize: float = 0.0


class DataFeed:
    def __init__(self, broker: "IbkrAutoTrader", logger) -> None:
        self.broker = broker
        self.logger = logger

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        for s in symbols:
            await self.broker.subscribe(s)
            
    async def unsubscribe_all(self) -> None:
        """取消所has数据subscription"""
        try:
            # retrieval所hassubscriptionticker并取消subscription
            if hasattr(self.broker, 'tickers'):
                symbols_to_unsubscribe = list(self.broker.tickers.keys())
                for symbol in symbols_to_unsubscribe:
                    self.broker.unsubscribe(symbol)
                self.logger.info(f"取消 {len(symbols_to_unsubscribe)} 个标数据subscription")
        except Exception as e:
            self.logger.error(f"取消数据subscriptionfailed: {e}")

    def best_quote(self, sym: str) -> Optional[Quote]:
        t = self.broker.tickers.get(sym)
        if not t:
            return None
        bid, ask = (t.bid or 0.0), (t.ask or 0.0)
        if bid == 0.0 and ask == 0.0:
            p = t.last or t.close or 0.0
            if p:
                return Quote(p, p)
            return None
        return Quote(bid, ask, t.bidSize or 0.0, t.askSize or 0.0)

    async def fetch_daily_bars(self, sym: str, lookback_days: int = 60):
        """from IB 拉取日线历史数据，useat真实信号andATR计算。"""
        try:
            contract = await self.broker.qualify_stock(sym)
            bars = await self.broker.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=f"{max(lookback_days, 30)} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            return list(bars or [])
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return []


class RiskEngine:
    """风险引擎 - 使use统一风险管理器"""
    def __init__(self, config_manager, logger) -> None:
        self.config_manager = config_manager
        self.logger = logger
        
        # 使use统一风险管理器
        from .unified_risk_manager import get_risk_manager
        self.risk_manager = get_risk_manager(config_manager)
        
        # 保持配置兼容性
        self.cfg = config_manager.get_full_config()
        
        self.logger.info("风险引擎初始化，使use统一风险管理器")

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> int:
        """增强positions计算逻辑（保持兼容性）"""
        # 安全的配置访问
        sizing = {}
        try:
            sizing = self.cfg.get("sizing", {}) if self.cfg else {}
        except (AttributeError, TypeError):
            self.logger.debug("Using default sizing config due to access error")
        
        # 验证输入参数类型和值
        try:
            effective_equity = max(float(equity or 0), 0.0)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid equity value: {equity}")
            return 0
            
        if effective_equity <= 0:
            self.logger.warning("Zero or negative equity, cannot calculate position size")
            return 0
            
        try:
            entry_price = float(entry_price or 0)
            stop_price = float(stop_price or 0)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid price values: entry={entry_price}, stop={stop_price}")
            return 0
            
        if entry_price <= 0 or stop_price <= 0:
            self.logger.warning(f"Invalid prices: entry={entry_price}, stop={stop_price}")
            return 0
        
        # 基at风险positions计算 - Use environment config first
        try:
            from .config_manager import get_config_manager
            env_manager = get_config_manager()
            risk_params = env_manager.get("trading", {}) if env_manager else {}
        except (ImportError, AttributeError):
            risk_params = {}
            self.logger.debug("Failed to load config manager, using defaults")
        
        per_trade_risk_pct = sizing.get("per_trade_risk_pct") or risk_params.get("per_trade_risk_pct", 0.02)
        # 确保风险百分比在合理范围内
        per_trade_risk_pct = max(0.001, min(per_trade_risk_pct, 0.1))  # 0.1% to 10%
        
        risk_per_trade = effective_equity * per_trade_risk_pct
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance <= 0:
            self.logger.warning("Invalid stop distance, using minimum position")
            return 1
        
        # 基at风险股数
        shares_by_risk = int(risk_per_trade // stop_distance)
        
        # 基at权益比例最大positions - Use environment config first
        max_position_pct = sizing.get("max_position_pct_of_equity") or risk_params.get("max_position_pct", 0.15)
        # 确保最大仓位百分比在合理范围内
        max_position_pct = max(0.01, min(max_position_pct, 0.5))  # 1% to 50%
        
        max_position_value = effective_equity * max_position_pct
        shares_by_equity = int(max_position_value // entry_price) if entry_price > 0 else 0
        
        # 取两者最小值，确保都是非负整数
        shares_by_risk = max(int(shares_by_risk), 0)
        shares_by_equity = max(int(shares_by_equity), 0)
        qty = min(shares_by_risk, shares_by_equity)
        qty = max(qty, 0)
        
        # 批量整理
        if sizing.get("notional_round_lots", True):
            qty = int(qty)
        
        # 记录计算过程
        self.logger.info(f"Position sizing: equity=${effective_equity:.2f}, "
                        f"risk_per_trade=${risk_per_trade:.2f}, "
                        f"stop_distance=${stop_distance:.2f}, "
                        f"shares_by_risk={shares_by_risk}, "
                        f"shares_by_equity={shares_by_equity}, "
                        f"final_qty={qty}")
        
        return qty
    
    async def validate_order(self, symbol: str, side: str, quantity: int, 
                           price: float, account_value: float) -> bool:
        """验证订单（使用统一验证器）"""
        try:
            from .unified_order_validator import get_unified_validator
            unified_validator = get_unified_validator()
            result = await unified_validator.validate_order_unified(
                symbol, side, quantity, price, account_value
            )
            
            if not result.is_valid:
                self.logger.warning(f"订单风险验证failed {symbol}: {result.reason}")
                return False
            
            if result.details:
                self.logger.info(f"订单验证详情 {symbol}: {result.details}")
            
            return True
            
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return False
    
    def update_position(self, symbol: str, quantity: int, current_price: float, 
                       entry_price: float = None):
        """updatespositions信息（新增方法）"""
        self.risk_manager.update_position(symbol, quantity, current_price, entry_price)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """retrieval风险summary（新增方法）"""
        return self.risk_manager.get_risk_summary()

    def validate_portfolio_exposure(self, equity: float, total_exposure: float) -> bool:
        """Validate that portfolio doesn't exceed maximum exposure limits"""
        try:
            if equity <= 0:
                self.logger.warning("Cannot validate exposure with zero equity")
                return False
                
            exposure_ratio = total_exposure / equity
            # 安全的配置访问，避免KeyError
            max_exposure = 0.95
            try:
                max_exposure = self.cfg.get("CONFIG", {}).get("capital", {}).get("max_portfolio_exposure", 0.95)
            except (AttributeError, KeyError, TypeError):
                self.logger.debug("Using default max_exposure due to config access error")
            
            if exposure_ratio > max_exposure:
                self.logger.warning(f"Portfolio exposure {exposure_ratio:.2%} exceeds maximum {max_exposure:.2%}")
                return False
                
            return True
            
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return False


class SignalHub:
    def __init__(self, logger=None) -> None:
        import logging
        self.logger = logger or logging.getLogger(__name__)
        
        # 🔧 指数退避配置
        self._base_delay = 1.0
        self._max_delay = 60.0
        self._consecutive_errors = 0
        import random
        self._random = random.Random()

    def mr_signal(self, closes: List[float]) -> float:
        """Enhanced mean reversion signal with proper NaN handling"""
        try:
            if len(closes) < 20:
                self.logger.warning(f"Insufficient data for signal calculation: {len(closes)} bars")
                return 0.0
                
            # Check for NaN values and handle them
            clean_closes = [p for p in closes if p is not None and not (isinstance(p, float) and p != p)]
            if len(clean_closes) < 20:
                self.logger.warning("Insufficient clean price data for signal calculation")
                return 0.0
                
            z = zscore(clean_closes, 20)
            if not z or len(z) == 0:
                return 0.0
                
            z_now = z[-1] if len(z) > 0 else 0.0
            if z_now != z_now:  # Check for NaN
                return 0.0
                
            # Enhanced signal logic with multiple thresholds
            if z_now > 2.5:
                return -1.0  # Strong sell signal
            elif z_now > 1.5:
                return -0.5  # Weak sell signal
            elif z_now < -2.5:
                return +1.0  # Strong buy signal
            elif z_now < -1.5:
                return +0.5  # Weak buy signal
            else:
                return -z_now  # Linear scaling for moderate signals
                
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return 0.0

    def calculate_momentum(self, prices: List[float], period: int = 20) -> float:
        """Calculate momentum indicator with proper error handling"""
        try:
            if len(prices) < period + 1:
                return 0.0
                
            # Calculate price returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i] > 0 and prices[i-1] > 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if len(returns) < period:
                return 0.0
                
            # Use the last 'period' returns for momentum
            recent_returns = returns[-period:]
            momentum = sum(recent_returns) / len(recent_returns)
            
            return momentum
            
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return 0.0

    def composite_score(self, factors: dict) -> float:
        """Calculate composite score from multiple factors with proper normalization"""
        try:
            # Fill NaNs with 0 for scoring
            clean_factors = {}
            for key, value in factors.items():
                if value is not None and not (isinstance(value, float) and value != value):
                    clean_factors[key] = value
                else:
                    clean_factors[key] = 0.0
            
            # Define factor weights
            weights = {
                'momentum': 0.4,
                'mean_reversion': 0.3,
                'volatility': 0.2,
                'volume': 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for factor, weight in weights.items():
                if factor in clean_factors:
                    score += clean_factors[factor] * weight
                    total_weight += weight
            
            # Normalize by actual total weight used
            if total_weight > 0:
                score = score / total_weight
            
            # Clip to [-1, 1] range
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return 0.0

    def multi_factor_signal(self, closes: List[float], highs: List[float], lows: List[float], vols: List[float]) -> float:
        """迁移自 Trader 内置多因子审批：趋势/动量/execution量/波动 四要素打分，输出[-1,1]。
        返回正值倾to买入，负值倾to卖出，绝for值as强度。
        """
        try:
            if len(closes) < 50:
                return 0.0
            # 趋势：短in长均线
            sma5 = sum(closes[-5:]) / 5
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50
            trend_score = 0.0
            if sma5 > sma20 > sma50:
                trend_score += 0.4
            elif sma5 > sma20:
                trend_score += 0.2
            elif closes[-1] > sma20:
                trend_score += 0.1
            sma20_prev = sum(closes[-25:-5]) / 20 if len(closes) >= 25 else sma20
            if sma20_prev != 0:
                slope = (sma20 - sma20_prev) / sma20_prev
                if slope > 0.01:
                    trend_score += 0.3
                elif slope > 0:
                    trend_score += 0.1
            if closes[-1] > sma5 * 1.02:
                trend_score += 0.3
            elif closes[-1] > sma5:
                trend_score += 0.2

            # 动量：基at连续收益
            momentum_score = self.calculate_momentum(closes, 20)
            # 归一：约束in[-0.2,0.2]→映射to[-0.5,0.5]
            momentum_score = max(-0.2, min(0.2, momentum_score)) * 2.5

            # execution量：20日均量for比、近5日相for提升
            volume_score = 0.0
            if len(vols) >= 20:
                v20 = sum(vols[-20:]) / 20
                v_cur = max(vols[-1], 0.0)
                ratio = v_cur / v20 if v20 > 0 else 1.0
                if ratio > 1.5:
                    volume_score += 0.4
                elif ratio > 1.2:
                    volume_score += 0.2
                elif ratio > 0.8:
                    volume_score += 0.1
                recent5 = sum(vols[-5:]) / 5
                prev15 = sum(vols[-20:-5]) / 15 if len(vols) >= 20 else recent5
                if recent5 > prev15 * 1.2:
                    volume_score += 0.3
                elif recent5 > prev15:
                    volume_score += 0.1

            # 波动：ATRratio处at适宜区间 - 安全计算
            volatility_score = 0.0
            if len(highs) >= 15 and len(lows) >= 15 and len(closes) >= 15:
                try:
                    atr_result = atr(highs[-15:], lows[-15:], closes[-15:], 14)
                    if atr_result and len(atr_result) > 0:
                        atr14 = atr_result[-1]
                        if atr14 and atr14 > 0 and closes[-1] > 0:
                            atr_pct = (atr14 / closes[-1]) * 100
                            if 1.5 <= atr_pct <= 4.0:
                                volatility_score += 0.4
                            elif 1.0 <= atr_pct <= 6.0:
                                volatility_score += 0.2
                except Exception:
                    volatility_score = 0.0

            # 汇总：权重and Trader in一致
            total = trend_score * 0.30 + momentum_score * 0.25 + volume_score * 0.20 + volatility_score * 0.15
            # 简单映射：>0.6 买，<-0.6 卖，其余in性
            if total >= 0.6:
                return +1.0
            if total <= -0.6:
                return -1.0
            return max(-1.0, min(1.0, total))
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return 0.0


class OrderRouter:
    def __init__(self, cfg: dict, logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def build_prices(self, sym: str, side: str, q: Quote) -> dict:
        """增强price构建逻辑，包括滑点处理and自适应price偏移"""
        # 安全配置访问
        try:
            cfg = self.cfg.get("orders", {})
            mode = cfg.get("smart_price_mode", "market")
        except (AttributeError, KeyError, TypeError):
            self.logger.debug("Using default order config")
            mode = "market"
        
        # 验证输入参数
        if not q or not isinstance(q, Quote):
            self.logger.warning(f"{sym} 无效报价对象，使用market单")
            return {"type": "MKT"}
        
        # 确保报价has效性 - 安全取值
        bid = getattr(q, 'bid', None) or 0.0
        ask = getattr(q, 'ask', None) or 0.0
        
        if bid <= 0 or ask <= 0:
            self.logger.warning(f"{sym} 报价无效 bid={bid} ask={ask}，使用market单")
            return {"type": "MKT"}
            
        # 验证买卖价差合理性
        if bid >= ask:
            self.logger.warning(f"{sym} 买卖价差异常 bid={bid} >= ask={ask}，使用market单")
            return {"type": "MKT"}
            
        # 安全计算中间价和价差
        try:
            mid = (bid + ask) / 2
            spread = abs(ask - bid) / mid if mid > 0 else 0.0
        except (ZeroDivisionError, ValueError):
            self.logger.warning(f"{sym} 价格计算错误，使用market单")
            return {"type": "MKT"}
        
        # 异常价差check：if果价差过大（>5%），使usemarket单防止滑点
        if spread > 0.05:
            self.logger.warning(f"{sym} 价差过大 {spread:.2%}，使usemarket单防止滑点")
            return {"type": "MKT"}
            
        # 强制market模式
        if mode == "market":
            return {"type": "MKT"}
            
        # 自适应偏移：根据价差动态调整，避免过度激进or保守
        # 价差越大，偏移越大，但限制in合理范围内
        adaptive = max(min(0.0025, 0.3 * spread), 0.0001)  # 0.01%-0.25%范围
        
        if mode == "midpoint":
            return {"type": "LMT", "limit": round(mid, 2)}
        elif mode == "aggressive":
            if side == "BUY":
                px = ask * (1 + adaptive)
            else:
                px = bid * (1 - adaptive)
            # 确保价格为正数
            px = max(px, 0.01)
            return {"type": "LMT", "limit": round(px, 2)}
        elif mode == "conservative":
            if side == "BUY":
                px = (bid + mid) / 2  # bid and mid 中间价
            else:
                px = (ask + mid) / 2  # ask and mid 中间价
            # 确保价格为正数
            px = max(px, 0.01)
            return {"type": "LMT", "limit": round(px, 2)}
        else:
            return {"type": "MKT"}

    def reprice_working_orders_if_needed(self) -> None:
        # 预留：根据报价in点偏离智能改价/撤补
        return


class Engine:
    def __init__(self, config_manager, broker: "IbkrAutoTrader") -> None:
        self.config_manager = config_manager
        self.broker = broker
        # 使use事件系统日志适配器 - 安全导入
        try:
            from .trading_logger import create_engine_logger
            self.logger = create_engine_logger("Engine", "engine")
        except ImportError:
            import logging
            self.logger = logging.getLogger("Engine")
            self.logger.warning("Failed to import trading_logger, using default logger")
        
        self.data = DataFeed(broker, self.logger)
        
        # 🔧 指数退避重试机制
        import random
        from datetime import datetime, timedelta
        self._consecutive_errors = 0
        self._last_success_time = datetime.now()
        self._base_delay = 1.0
        self._max_delay = 300.0  # 5分钟最大延迟
        self._random = random.Random()
        
        # 使use统一配置而notisHotConfig - 安全配置获取
        try:
            config_dict = config_manager.get_full_config() if config_manager else {}
        except (AttributeError, TypeError):
            config_dict = {}
            self.logger.warning("Failed to get config, using defaults")
        
        self.risk = RiskEngine(config_manager, self.logger)  # 传递config_manager
        self.router = OrderRouter(config_dict, self.logger)
        self.signal = SignalHub(self.logger)
        
        # retrieval统一positions管理器引use - 安全初始化
        try:
            from .unified_position_manager import get_position_manager
            self.position_manager = get_position_manager()
        except ImportError:
            self.position_manager = None
            self.logger.warning("Failed to import position manager")
        
        # retrieval统一风险管理器（替代旧risk_manager引use） - 安全初始化
        try:
            from .unified_risk_manager import get_risk_manager
            self.risk_manager = get_risk_manager(config_manager)
        except ImportError:
            self.risk_manager = None
            self.logger.warning("Failed to import risk manager")
        
        # 初始化统一Polygon因子库 - 安全初始化
        try:
            self.unified_factors = get_unified_polygon_factors()
            self.logger.info("统一Polygon因子库初始化完成")
        except Exception as e:
            self.unified_factors = None
            self.logger.warning(f"Failed to initialize unified factors: {e}")

    async def start(self) -> None:
        # connection IB - 安全连接
        try:
            await self.broker.connect()
        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            raise
        
        # subscription观察列表 - 安全配置获取
        try:
            uni = self.config_manager.get("scanner.universe", ["SPY"]) if self.config_manager else ["SPY"]
            await self.data.subscribe_quotes(uni)
        except Exception as e:
            self.logger.error(f"Failed to subscribe to quotes: {e}")
            # 不抛出异常，允许系统继续运行
        
    async def stop(self) -> None:
        """停止引擎，释放资源"""
        try:
            self.logger.info("正in停止引擎...")
            
            # 停止数据subscription
            if hasattr(self.data, 'unsubscribe_all'):
                await self.data.unsubscribe_all()
            
            # 清理positions管理器引use
            if hasattr(self, 'position_manager'):
                self.position_manager = None
                
            # 清理风险管理器引use  
            if hasattr(self, 'risk_manager'):
                self.risk_manager = None
                
            self.logger.info("引擎停止")
            
        except Exception as e:
            self.logger.error(f"停止引擎when出错: {e}")
        
    def _validate_account_ready(self, cfg: dict) -> bool:
        """验证account状态is否就绪"""
        try:
            # 安全配置访问
            capital_cfg = cfg.get("capital", {}) if cfg else {}
            
            # checkis否要求account就绪
            if capital_cfg.get("require_account_ready", True):
                # 安全访问broker属性
                net_liq = getattr(self.broker, 'net_liq', 0) if self.broker else 0
                if net_liq <= 0:
                    self.logger.warning("account净值as0，跳过交易信号处理")
                    return False
                    
                if not hasattr(self.broker, 'account_values') or not getattr(self.broker, 'account_values', None):
                    self.logger.warning("account值未加载，跳过交易")
                    return False
                    
            return True
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return False
            
    def _validate_capital_requirements(self, cfg: dict) -> bool:
        """验证资金要求"""
        try:
            # 安全配置访问
            capital_cfg = cfg.get("capital", {}) if cfg else {}
            cash_reserve_pct = capital_cfg.get("cash_reserve_pct", 0.10)
            
            # 安全访问broker属性
            net_liq = getattr(self.broker, 'net_liq', 0) if self.broker else 0
            cash_balance = getattr(self.broker, 'cash_balance', 0) if self.broker else 0
            
            if net_liq <= 0:
                self.logger.warning("净值为0，无法验证资金要求")
                return False
            
            min_cash_required = net_liq * cash_reserve_pct
            available_cash = max(cash_balance - min_cash_required, 0)
            
            if available_cash < 1000:  # 最少保留$1000canuse现金
                self.logger.warning(f"canuse现金not足: ${available_cash:,.2f} (需要保留{cash_reserve_pct:.1%})")
                return False
                
            # check最大投资组合敞口
            total_positions_value = 0.0
            for sym, pos_obj in self.position_manager.get_all_positions().items():
                pos = pos_obj.quantity
                if pos == 0:
                    continue
                q = self.data.best_quote(sym)
                if not q:
                    continue
                mid = (q.bid + q.ask) / 2.0 if (q.bid > 0 and q.ask > 0) else max(q.bid, q.ask)
                if mid and mid > 0:
                    total_positions_value += abs(pos) * mid
            
            max_exposure = self.broker.net_liq * cfg["capital"]["max_portfolio_exposure"]
            if total_positions_value >= max_exposure:
                self.logger.warning(f"投资组合敞口达上限: ${total_positions_value:,.2f} >= ${max_exposure:,.2f}")
                return False
                
            return True
        except Exception as e:
            context = ErrorContext(
                operation="engine",
                component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
            )
            get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
            return False
    
    def get_next_delay(self) -> float:
        """🔧 指数退避延迟计算"""
        if self._consecutive_errors == 0:
            return self._base_delay
        
        # 指数退避: 2^errors * base_delay + jitter
        exponential_delay = min(
            2 ** self._consecutive_errors * self._base_delay,
            self._max_delay
        )
        
        # 添加随机抖动避免thundering herd
        jitter = self._random.uniform(0.1, 0.3) * exponential_delay
        final_delay = exponential_delay + jitter
        
        self.logger.debug(f"Exponential backoff: {self._consecutive_errors} errors, delay={final_delay:.1f}s")
        return final_delay

    async def on_signal_and_trade(self) -> None:
        """增强信号计算and交易执行：包含account状态验证and净值check"""
        try:
            cfg = self.config_manager.get_full_config()
            uni = cfg["scanner"]["universe"]
            
            self.logger.info(f"运行信号计算and交易 - 标数量: {len(uni)}")
            
            # 强制刷新account信息，确保数据最新
            try:
                await self.broker.refresh_account_balances_and_positions()
                self.logger.info(f"account信息刷新: 净值=${self.broker.net_liq:,.2f}, 现金=${self.broker.cash_balance:,.2f}")
                
                # 🔧 成功时重置错误计数
                self._consecutive_errors = 0
                self._last_success_time = datetime.now()
                
            except Exception as e:
                # 🔧 记录连续错误
                self._consecutive_errors += 1
                context = ErrorContext(
                    operation="engine",
                    component=self.__class__.__name__ if hasattr(self, '__class__') else "unknown"
                )
                get_error_handler().handle_error(e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM)
                return
            
            # account状态验证
            if not self._validate_account_ready(cfg):
                return
                
            # 资金管理check
            if not self._validate_capital_requirements(cfg):
                self.logger.info("资金管理check未通过，跳过本轮交易")
                return
            
            orders_sent = 0
            max_new_orders = cfg["capital"].get("max_new_positions_per_day", 10)
            
            for sym in uni:
                if orders_sent >= max_new_orders:
                    self.logger.info(f"达to单日最大新订单数限制: {max_new_orders}")
                    break
                    
                q = self.data.best_quote(sym)
                if not q:
                    self.logger.debug(f"{sym} nohas效报价（nobid/ask/last/close），跳过")
                    continue
                    
                # 使用统一Polygon因子库计算信号（替代IBKR历史数据）
                try:
                    polygon_signal = self.unified_factors.get_trading_signal(sym, threshold=self.config_manager.get("signals.acceptance_threshold", 0.6))
                    
                    # 检查信号质量和可交易性
                    if not polygon_signal.get('can_trade', False):
                        reason = polygon_signal.get('delay_reason') or '信号不满足交易条件'
                        self.logger.debug(f"{sym} Polygon信号不可交易: {reason}")
                        continue
                    
                    score = polygon_signal['signal_value']
                    signal_strength = polygon_signal['signal_strength']
                    confidence = polygon_signal['confidence']
                    
                    self.logger.debug(f"{sym} Polygon信号 | 值={score:.3f}, 强度={signal_strength:.3f}, 置信度={confidence:.3f}")
                    
                    # 使用信号强度检查阈值
                    thr = self.config_manager.get("signals.acceptance_threshold", 0.6)
                    if signal_strength < thr:
                        self.logger.debug(f"{sym} Polygon信号强度不足 | {signal_strength:.3f} < {thr:.3f}")
                        continue
                    
                except Exception as e:
                    self.logger.warning(f"{sym} Polygon因子计算失败，回退到IBKR数据: {e}")
                    
                    # 回退到原有逻辑
                    bars = await self.data.fetch_daily_bars(sym, lookback_days=60)
                    if len(bars) < 60:
                        self.logger.debug(f"{sym} 历史K线不足 {len(bars)} < 60，跳过")
                        continue
                        
                    closes_all: List[float] = [float(getattr(b, "close", 0.0) or 0.0) for b in bars]
                    closes_50 = closes_all[-60:]
                    highs = [float(getattr(b, "high", 0.0) or 0.0) for b in bars][-60:]
                    lows = [float(getattr(b, "low", 0.0) or 0.0) for b in bars][-60:]
                    vols = [float(getattr(b, "volume", 0.0) or 0.0) for b in bars][-60:]
                    score = self.signal.multi_factor_signal(closes_50, highs, lows, vols)
                    
                    thr = self.config_manager.get("signals.acceptance_threshold", 0.6)
                    if abs(score) < thr:
                        self.logger.debug(f"{sym} 回退信号强度不足 | score={score:.3f}, thr={thr:.3f}，跳过")
                        continue

                side = "BUY" if score > 0 else "SELL"
                entry = q.ask if side == "BUY" else q.bid
                if not entry or entry <= 0:
                    self.logger.debug(f"{sym} no法retrievalhas效入场价（side={side}），跳过")
                    continue
                    
                # check最大单个positions限制
                max_single_position = self.broker.net_liq * cfg["capital"].get("max_single_position_pct", 0.15)
                current_position_qty = self.position_manager.get_quantity(sym)
                current_position_value = abs(current_position_qty) * entry
                if current_position_value >= max_single_position:
                    self.logger.info(f"{sym} positions达单个标上限 | 当beforepositions市值=${current_position_value:,.2f}")
                    continue
                    
                stop_pct = self.config_manager.get("trading.default_stop_loss_pct", 0.02)
                tp_pct = self.config_manager.get("trading.default_take_profit_pct", 0.05)
                
                # 风险锚定：使用Polygon数据计算ATR作为最小止损距离
                try:
                    # 优先使用Polygon数据计算ATR
                    market_data = self.unified_factors.get_market_data(sym, days=30)
                    if not market_data.empty and len(market_data) >= 14:
                        highs = market_data['High'].tolist()[-15:]
                        lows = market_data['Low'].tolist()[-15:]
                        closes = market_data['Close'].tolist()[-15:]
                        atr_values = atr(highs, lows, closes, 14)
                        atr14 = atr_values[-1] if atr_values and not math.isnan(atr_values[-1]) else None
                        
                        self.logger.debug(f"{sym} 使用Polygon数据计算ATR: {atr14}")
                    else:
                        # 回退到IBKR数据
                        if 'bars' in locals() and len(bars) >= 14:
                            hi = [float(getattr(b, "high", entry) or entry) for b in bars][-15:]
                            lo = [float(getattr(b, "low", entry) or entry) for b in bars][-15:]
                            cl = [float(getattr(b, "close", entry) or entry) for b in bars][-15:]
                            atr_values = atr(hi, lo, cl, 14)
                            atr14 = atr_values[-1] if atr_values and not math.isnan(atr_values[-1]) else None
                            self.logger.debug(f"{sym} 回退到IBKR数据计算ATR: {atr14}")
                        else:
                            atr14 = None
                    
                    atr14 = atr14 if atr14 and atr14 > 0 else None
                except Exception as e:
                    self.logger.warning(f"ATR计算失败 {sym}: {e}")
                    atr14 = None
                    
                if side == "BUY":
                    stop_px = min(entry * (1 - stop_pct), entry - (atr14 or entry * 0.02))
                else:
                    stop_px = max(entry * (1 + stop_pct), entry + (atr14 or entry * 0.02))
                    
                # 使use增强仓位计算
                qty = self.risk.position_size(self.broker.net_liq, entry, stop_px)
                
                # 应use最小仓位要求
                min_usd = cfg["sizing"].get("min_position_usd", 1000)
                min_shares = cfg["sizing"].get("min_shares", 1)
                min_qty_by_value = max(min_usd // entry, min_shares)
                
                if qty <= 0:
                    qty = min_qty_by_value
                    self.logger.info(f"{sym} 风险计算仓位as0，使use最小仓位: {qty}")
                else:
                    qty = max(qty, min_qty_by_value)

                price_plan = self.router.build_prices(sym, side, q)
                self.logger.debug(f"{sym} order placement计划 | side={side} qty={qty} plan={price_plan}")
                
                try:
                    if price_plan.get("type") == "MKT":
                        await self.broker.place_market_order(sym, side, qty)
                    else:
                        await self.broker.place_limit_order(sym, side, qty, float(price_plan["limit"]))
                    orders_sent += 1
                    self.logger.info(f"success提交订单: {side} {qty} {sym} @ {entry}")
                except Exception as e:
                    self.logger.error(f"订单提交failed {sym}: {e}")
                    continue
            
            self.logger.info(f"信号处理completed，共提交 {orders_sent} 个订单")
            
        except Exception as e:
            self.logger.error(f"信号处理异常: {e}")
            import traceback
            traceback.print_exc()

    async def on_realtime_bar(self) -> None:
        # 预留：canin此聚合 5s bar 并进行positions市值标记and工作单调价
        self.router.reprice_working_orders_if_needed()

    async def on_open_sync(self) -> None:
        # 开盘after同步accountandpositions
        await self.broker.refresh_account_balances_and_positions()
        self.logger.info("开盘同步completed：刷新accountandpositions")

    async def health_check(self) -> None:
        if not self.broker.ib.isConnected():
            self.logger.warning("connection断开，尝试重连...")
            await self.broker.connect()
        # 热updates配置
        if self.cfg.maybe_reload():
            self.logger.info("配置热updates")

