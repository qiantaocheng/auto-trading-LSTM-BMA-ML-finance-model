from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

# 已改用统一配置管理器
# from .config import HotConfig
from .factors import Bar, sma, rsi, bollinger, zscore, atr
from .ibkr_auto_trader import IbkrAutoTrader


@dataclass
class Quote:
    bid: float
    ask: float
    bidSize: float = 0.0
    askSize: float = 0.0


class DataFeed:
    def __init__(self, broker: IbkrAutoTrader, logger) -> None:
        self.broker = broker
        self.logger = logger

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        for s in symbols:
            await self.broker.subscribe(s)

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
        """从 IB 拉取日线历史数据，用于真实的信号与ATR计算。"""
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
            self.logger.warning(f"获取历史数据失败 {sym}: {e}")
            return []


class RiskEngine:
    def __init__(self, cfg: dict, logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> int:
        """Enhanced position sizing logic with proper risk management"""
        sizing = self.cfg["sizing"]
        
        # Validate inputs
        effective_equity = max(float(equity or 0), 0.0)
        if effective_equity <= 0:
            self.logger.warning("Zero or negative equity, cannot calculate position size")
            return 0
            
        if entry_price <= 0 or stop_price <= 0:
            self.logger.warning(f"Invalid prices: entry={entry_price}, stop={stop_price}")
            return 0
        
        # Calculate risk per trade based on actual equity
        risk_per_trade = effective_equity * sizing["per_trade_risk_pct"]
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance <= 0:
            self.logger.warning("Invalid stop distance, using minimum position")
            return 1
        
        # Calculate position size based on risk
        shares_by_risk = int(risk_per_trade // stop_distance)
        
        # Calculate maximum position based on equity percentage
        max_position_value = effective_equity * sizing["max_position_pct_of_equity"]
        shares_by_equity = int(max_position_value // entry_price)
        
        # Take the minimum of both constraints
        qty = min(shares_by_risk, shares_by_equity)
        qty = max(qty, 0)  # Ensure non-negative
        
        # Apply lot size rounding if configured
        if sizing.get("notional_round_lots", True):
            qty = int(qty)
        
        # Log the calculation for transparency
        self.logger.info(f"Position sizing: equity=${effective_equity:.2f}, "
                        f"risk_per_trade=${risk_per_trade:.2f}, "
                        f"stop_distance=${stop_distance:.2f}, "
                        f"shares_by_risk={shares_by_risk}, "
                        f"shares_by_equity={shares_by_equity}, "
                        f"final_qty={qty}")
        
        return qty

    def validate_portfolio_exposure(self, equity: float, total_exposure: float) -> bool:
        """Validate that portfolio doesn't exceed maximum exposure limits"""
        try:
            if equity <= 0:
                self.logger.warning("Cannot validate exposure with zero equity")
                return False
                
            exposure_ratio = total_exposure / equity
            max_exposure = self.cfg["CONFIG"]["capital"].get("max_portfolio_exposure", 0.95)
            
            if exposure_ratio > max_exposure:
                self.logger.warning(f"Portfolio exposure {exposure_ratio:.2%} exceeds maximum {max_exposure:.2%}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating portfolio exposure: {e}")
            return False


class SignalHub:
    def __init__(self, logger) -> None:
        self.logger = logger

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
                
            z_now = z[-1]
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
            self.logger.error(f"Error calculating mean reversion signal: {e}")
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
            self.logger.error(f"Error calculating momentum: {e}")
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
            
            # Normalize by actual weights used
            if total_weight > 0:
                score = score / total_weight
                
            # Clamp score to reasonable range
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return 0.0

    def multi_factor_signal(self, closes: List[float], highs: List[float], lows: List[float], vols: List[float]) -> float:
        """迁移自 Trader 内置的多因子审批：趋势/动量/成交量/波动 四要素打分，输出[-1,1]。
        返回正值倾向买入，负值倾向卖出，绝对值为强度。
        """
        try:
            if len(closes) < 50:
                return 0.0
            # 趋势：短中长均线
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
                slope = (sma20 - sma20_prev) / abs(sma20_prev)
                if slope > 0.01:
                    trend_score += 0.3
                elif slope > 0:
                    trend_score += 0.1
            if closes[-1] > sma5 * 1.02:
                trend_score += 0.3
            elif closes[-1] > sma5:
                trend_score += 0.2

            # 动量：基于连续收益
            momentum_score = self.calculate_momentum(closes, 20)
            # 归一：约束在[-0.2,0.2]→映射到[-0.5,0.5]
            momentum_score = max(-0.2, min(0.2, momentum_score)) * 2.5

            # 成交量：20日均量对比、近5日相对提升
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

            # 波动：ATR占比处于适宜区间
            from .factors import atr as atr_func
            atr14 = atr_func(highs[-15:], lows[-15:], closes[-15:], 14)[-1] if len(highs) >= 15 else None
            volatility_score = 0.0
            if atr14 and atr14 > 0 and closes[-1] > 0:
                atr_pct = (atr14 / closes[-1]) * 100
                if 1.5 <= atr_pct <= 4.0:
                    volatility_score += 0.4
                elif 1.0 <= atr_pct <= 6.0:
                    volatility_score += 0.2

            # 汇总：权重与 Trader 中一致
            total = trend_score * 0.30 + momentum_score * 0.25 + volume_score * 0.20 + volatility_score * 0.15
            # 简单映射：>0.6 买，<-0.6 卖，其余中性
            if total >= 0.6:
                return +1.0
            if total <= -0.6:
                return -1.0
            return max(-1.0, min(1.0, total))
        except Exception as e:
            self.logger.error(f"multi_factor_signal error: {e}")
            return 0.0


class OrderRouter:
    def __init__(self, cfg: dict, logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def build_prices(self, sym: str, side: str, q: Quote) -> dict:
        """增强的价格构建逻辑，包括滑点处理和自适应价格偏移"""
        cfg = self.cfg["orders"]
        mode = cfg["smart_price_mode"]
        
        # 确保报价有效性
        if not q.bid or not q.ask or q.bid <= 0 or q.ask <= 0:
            self.logger.warning(f"{sym} 报价无效 bid={q.bid} ask={q.ask}，使用市价单")
            return {"type": "MKT"}
            
        # 验证买卖价差的合理性
        if q.bid >= q.ask:
            self.logger.warning(f"{sym} 买卖价差异常 bid={q.bid} >= ask={q.ask}，使用市价单")
            return {"type": "MKT"}
            
        mid = (q.bid + q.ask) / 2
        spread = abs(q.ask - q.bid) / mid if mid > 0 else 0.0
        
        # 异常价差检查：如果价差过大（>5%），使用市价单防止滑点
        if spread > 0.05:
            self.logger.warning(f"{sym} 价差过大 {spread:.2%}，使用市价单防止滑点")
            return {"type": "MKT"}
            
        # 强制市价模式
        if mode == "market":
            return {"type": "MKT"}
            
        # 自适应偏移：根据价差动态调整，避免过度激进或保守
        # 价差越大，偏移越大，但限制在合理范围内
        adaptive = max(min(0.0025, 0.3 * spread), 0.0001)  # 0.01%-0.25%范围
        
        if mode == "midpoint":
            return {"type": "LMT", "limit": round(mid, 2)}
        elif mode == "aggressive":
            if side == "BUY":
                px = q.ask * (1 + adaptive)
            else:
                px = q.bid * (1 - adaptive)
            return {"type": "LMT", "limit": round(px, 2)}
        elif mode == "conservative":
            if side == "BUY":
                px = (q.bid + mid) / 2  # bid 和 mid 的中间价
            else:
                px = (q.ask + mid) / 2  # ask 和 mid 的中间价
            return {"type": "LMT", "limit": round(px, 2)}
        else:
            return {"type": "MKT"}

    def reprice_working_orders_if_needed(self) -> None:
        # 预留：根据报价中点偏离智能改价/撤补
        return


class Engine:
    def __init__(self, config_manager, broker: IbkrAutoTrader) -> None:
        self.config_manager = config_manager
        self.broker = broker
        # 使用事件系统的日志适配器
        from .engine_logger import create_engine_logger
        self.logger = create_engine_logger("Engine", "engine")
        self.data = DataFeed(broker, self.logger)
        
        # 使用统一配置而不是HotConfig
        config_dict = config_manager._get_merged_config()
        self.risk = RiskEngine(config_dict, self.logger)
        self.router = OrderRouter(config_dict, self.logger)
        self.signal = SignalHub(self.logger)

    async def start(self) -> None:
        # 连接 IB
        await self.broker.connect()
        # 订阅观察列表
        uni = self.config_manager.get_universe()
        await self.data.subscribe_quotes(uni)
        
    def _validate_account_ready(self, cfg: dict) -> bool:
        """验证账户状态是否就绪"""
        try:
            # 检查是否要求账户就绪
            if cfg["capital"].get("require_account_ready", True):
                if self.broker.net_liq <= 0:
                    self.logger.warning("账户净值为0，跳过交易信号处理")
                    return False
                    
                if not hasattr(self.broker, 'account_values') or not self.broker.account_values:
                    self.logger.warning("账户值未加载，跳过交易")
                    return False
                    
            return True
        except Exception as e:
            self.logger.error(f"账户状态验证失败: {e}")
            return False
            
    def _validate_capital_requirements(self, cfg: dict) -> bool:
        """验证资金要求"""
        try:
            cash_reserve_pct = cfg["capital"].get("cash_reserve_pct", 0.10)
            min_cash_required = self.broker.net_liq * cash_reserve_pct
            
            available_cash = max(self.broker.cash_balance - min_cash_required, 0)
            if available_cash < 1000:  # 最少保留$1000可用现金
                self.logger.warning(f"可用现金不足: ${available_cash:,.2f} (需要保留{cash_reserve_pct:.1%})")
                return False
                
            # 检查最大投资组合敞口
            total_positions_value = 0.0
            for sym, pos in self.broker.positions.items():
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
                self.logger.warning(f"投资组合敞口已达上限: ${total_positions_value:,.2f} >= ${max_exposure:,.2f}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"资金需求验证失败: {e}")
            return False

    async def on_signal_and_trade(self) -> None:
        """增强的信号计算与交易执行：包含账户状态验证和净值检查"""
        try:
            cfg = self.config_manager._get_merged_config()
            uni = cfg["scanner"]["universe"]
            
            self.logger.info(f"运行信号计算和交易 - 标的数量: {len(uni)}")
            
            # 强制刷新账户信息，确保数据最新
            try:
                await self.broker.refresh_account_balances_and_positions()
                self.logger.info(f"账户信息已刷新: 净值=${self.broker.net_liq:,.2f}, 现金=${self.broker.cash_balance:,.2f}")
            except Exception as e:
                self.logger.error(f"刷新账户信息失败: {e}")
                return
            
            # 账户状态验证
            if not self._validate_account_ready(cfg):
                return
                
            # 资金管理检查
            if not self._validate_capital_requirements(cfg):
                self.logger.info("资金管理检查未通过，跳过本轮交易")
                return
            
            orders_sent = 0
            max_new_orders = cfg["capital"].get("max_new_positions_per_day", 10)
            
            for sym in uni:
                if orders_sent >= max_new_orders:
                    self.logger.info(f"已达到单日最大新订单数限制: {max_new_orders}")
                    break
                    
                q = self.data.best_quote(sym)
                if not q:
                    self.logger.debug(f"{sym} 无有效报价（无bid/ask/last/close），跳过")
                    continue
                    
                # 使用真实历史数据（日线）计算信号与ATR
                bars = await self.data.fetch_daily_bars(sym, lookback_days=60)
                if len(bars) < 60:
                    # 历史数据不足（信号需要至少50根K），跳过此标的
                    self.logger.debug(f"{sym} 历史K线不足 {len(bars)} < 60，跳过")
                    continue
                    
                # 注意：信号需要50根收盘价
                closes_all: List[float] = [float(getattr(b, "close", 0.0) or 0.0) for b in bars]
                closes_50 = closes_all[-60:]

                # 迁移的多因子打分（旧策略移植）
                highs = [float(getattr(b, "high", 0.0) or 0.0) for b in bars][-60:]
                lows = [float(getattr(b, "low", 0.0) or 0.0) for b in bars][-60:]
                vols = [float(getattr(b, "volume", 0.0) or 0.0) for b in bars][-60:]
                score = self.signal.multi_factor_signal(closes_50, highs, lows, vols)
                thr = cfg["signals"]["acceptance_threshold"]
                if abs(score) < thr:
                    self.logger.debug(f"{sym} 信号强度不足 | score={score:.3f}, thr={thr:.3f}，跳过")
                    continue

                side = "BUY" if score > 0 else "SELL"
                entry = q.ask if side == "BUY" else q.bid
                if not entry or entry <= 0:
                    self.logger.debug(f"{sym} 无法获取有效入场价（side={side}），跳过")
                    continue
                    
                # 检查最大单个持仓限制
                max_single_position = self.broker.net_liq * cfg["capital"].get("max_single_position_pct", 0.15)
                current_position_value = abs(self.broker.positions.get(sym, 0)) * entry
                if current_position_value >= max_single_position:
                    self.logger.info(f"{sym} 持仓已达单个标的上限 | 当前持仓市值=${current_position_value:,.2f}")
                    continue
                    
                stop_pct = cfg["orders"]["default_stop_loss_pct"]
                tp_pct = cfg["orders"]["default_take_profit_pct"]
                
                # 风险锚定：使用历史K的 ATR 作为最小止损距离
                hi = [float(getattr(b, "high", entry) or entry) for b in bars][-15:]
                lo = [float(getattr(b, "low", entry) or entry) for b in bars][-15:]
                cl = [float(getattr(b, "close", entry) or entry) for b in bars][-15:]
                
                try:
                    atr14 = atr(hi, lo, cl, 14)[-1] if len(hi) >= 14 else None
                    atr14 = atr14 if atr14 and atr14 > 0 and not (isinstance(atr14, float) and atr14 != atr14) else None
                except Exception as e:
                    self.logger.warning(f"ATR计算失败 {sym}: {e}")
                    atr14 = None
                    
                if side == "BUY":
                    stop_px = min(entry * (1 - stop_pct), entry - (atr14 or entry * 0.02))
                else:
                    stop_px = max(entry * (1 + stop_pct), entry + (atr14 or entry * 0.02))
                    
                # 使用增强的仓位计算
                qty = self.risk.position_size(self.broker.net_liq, entry, stop_px)
                
                # 应用最小仓位要求
                min_usd = cfg["sizing"].get("min_position_usd", 1000)
                min_shares = cfg["sizing"].get("min_shares", 1)
                min_qty_by_value = max(min_usd // entry, min_shares)
                
                if qty <= 0:
                    qty = min_qty_by_value
                    self.logger.info(f"{sym} 风险计算仓位为0，使用最小仓位: {qty}")
                else:
                    qty = max(qty, min_qty_by_value)

                price_plan = self.router.build_prices(sym, side, q)
                self.logger.debug(f"{sym} 下单计划 | side={side} qty={qty} plan={price_plan}")
                
                try:
                    if price_plan.get("type") == "MKT":
                        await self.broker.place_market_order(sym, side, qty)
                    else:
                        await self.broker.place_limit_order(sym, side, qty, float(price_plan["limit"]))
                    orders_sent += 1
                    self.logger.info(f"成功提交订单: {side} {qty} {sym} @ {entry}")
                except Exception as e:
                    self.logger.error(f"订单提交失败 {sym}: {e}")
                    continue
            
            self.logger.info(f"信号处理完成，共提交 {orders_sent} 个订单")
            
        except Exception as e:
            self.logger.error(f"信号处理异常: {e}")
            import traceback
            traceback.print_exc()

    async def on_realtime_bar(self) -> None:
        # 预留：可在此聚合 5s bar 并进行持仓市值标记与工作单调价
        self.router.reprice_working_orders_if_needed()

    async def on_open_sync(self) -> None:
        # 开盘后同步账户与持仓
        await self.broker.refresh_account_balances_and_positions()
        self.logger.info("开盘同步完成：已刷新账户与持仓")

    async def health_check(self) -> None:
        if not self.broker.ib.isConnected():
            self.logger.warning("连接断开，尝试重连...")
            await self.broker.connect()
        # 热更新配置
        if self.cfg.maybe_reload():
            self.logger.info("配置已热更新")

