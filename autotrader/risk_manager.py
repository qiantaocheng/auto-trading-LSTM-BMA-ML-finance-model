#!/usr/bin/env python3
"""
专业风险管理模块 - VaR计算、相关性检查、凯利公式、风险矩阵
"""

import numpy as np
import pandas as pd
import asyncio
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_var: float  # 投资组合VaR
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    correlation_risk: float  # 相关性风险
    concentration_risk: float  # 集中度风险
    leverage_ratio: float  # 杠杆比率


@dataclass
class PositionRisk:
    """单个持仓风险"""
    symbol: str
    position_value: float
    weight: float  # 占投资组合权重
    var_contribution: float  # VaR贡献
    correlation_score: float  # 与其他持仓的平均相关性


class AdvancedRiskManager:
    """专业风险管理器"""
    
    def __init__(self, ib_client, lookback_days: int = 252):
        self.ib = ib_client
        self.lookback_days = lookback_days
        self.logger = logging.getLogger("RiskManager")
        
        # 历史价格缓存
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_days))
        self.returns_cache: Dict[str, np.ndarray] = {}
        
        # 风险限制
        self.max_portfolio_var = 0.02  # 最大组合VaR 2%
        self.max_single_position = 0.15  # 单个持仓不超过15%
        self.max_sector_exposure = 0.30  # 单个行业不超过30%
        self.max_correlation = 0.7  # 最大相关性
        self.max_drawdown_limit = 0.10  # 最大回撤限制10%
        
        # 缓存
        self._last_risk_calc = 0
        self._cached_metrics: Optional[RiskMetrics] = None
        self._correlation_matrix: Optional[np.ndarray] = None
        
    async def update_price_history(self, symbol: str, price: float):
        """更新价格历史"""
        self.price_history[symbol].append(price)
        
        # 如果有足够数据，更新收益率缓存
        if len(self.price_history[symbol]) >= 2:
            prices = list(self.price_history[symbol])
            returns = np.diff(np.log(prices))
            self.returns_cache[symbol] = returns
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """计算VaR (Historical Simulation方法)"""
        if len(returns) < 30:  # 数据不足
            return 0.0
        
        return float(np.percentile(returns, confidence_level * 100))
    
    def calculate_portfolio_var(self, positions: Dict[str, float], 
                              total_portfolio_value: float) -> float:
        """计算投资组合VaR"""
        if not positions or total_portfolio_value <= 0:
            return 0.0
        
        # 获取所有持仓的收益率
        symbols = list(positions.keys())
        returns_matrix = []
        weights = []
        
        for symbol in symbols:
            if symbol in self.returns_cache and len(self.returns_cache[symbol]) >= 30:
                returns_matrix.append(self.returns_cache[symbol][-252:])  # 最近一年
                weights.append(positions[symbol] / total_portfolio_value)
            else:
                # 数据不足，跳过
                continue
        
        if len(returns_matrix) < 2:
            return 0.0
        
        # 对齐数据长度
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_length:] for r in returns_matrix]
        
        # 转换为numpy数组
        returns_array = np.array(returns_matrix).T  # 时间 x 资产
        weights_array = np.array(weights)
        
        # 计算投资组合收益率
        portfolio_returns = np.dot(returns_array, weights_array)
        
        # 计算VaR
        return abs(self.calculate_var(portfolio_returns))
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """计算相关性矩阵"""
        if len(symbols) < 2:
            return None
        
        returns_matrix = []
        valid_symbols = []
        
        for symbol in symbols:
            if symbol in self.returns_cache and len(self.returns_cache[symbol]) >= 30:
                returns_matrix.append(self.returns_cache[symbol][-60:])  # 最近60天
                valid_symbols.append(symbol)
        
        if len(returns_matrix) < 2:
            return None
        
        # 对齐数据长度
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_length:] for r in returns_matrix]
        
        # 计算相关性矩阵
        returns_array = np.array(returns_matrix)
        correlation_matrix = np.corrcoef(returns_array)
        
        return correlation_matrix
    
    def check_correlation_risk(self, symbol: str, positions: Dict[str, float]) -> float:
        """检查新增持仓的相关性风险"""
        if not positions or symbol not in self.returns_cache:
            return 0.0
        
        existing_symbols = [s for s in positions.keys() if s in self.returns_cache and s != symbol]
        if not existing_symbols:
            return 0.0
        
        # 计算与现有持仓的平均相关性
        correlations = []
        target_returns = self.returns_cache[symbol][-60:]  # 最近60天
        
        for existing_symbol in existing_symbols:
            if existing_symbol in self.returns_cache:
                existing_returns = self.returns_cache[existing_symbol][-60:]
                
                # 对齐数据长度
                min_length = min(len(target_returns), len(existing_returns))
                if min_length >= 20:  # 至少20天数据
                    corr = np.corrcoef(
                        target_returns[-min_length:], 
                        existing_returns[-min_length:]
                    )[0, 1]
                    
                    if not np.isnan(corr):
                        # 按持仓权重加权
                        weight = positions[existing_symbol]
                        correlations.append(abs(corr) * weight)
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_kelly_criterion(self, symbol: str, win_rate: float = None, 
                                avg_win: float = None, avg_loss: float = None) -> float:
        """计算凯利公式推荐仓位"""
        # 如果没有提供参数，基于历史数据估算
        if symbol not in self.returns_cache or len(self.returns_cache[symbol]) < 50:
            return 0.05  # 默认5%
        
        returns = self.returns_cache[symbol][-252:]  # 最近一年
        
        if win_rate is None:
            win_rate = len(returns[returns > 0]) / len(returns)
        
        if avg_win is None:
            positive_returns = returns[returns > 0]
            avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0.01
        
        if avg_loss is None:
            negative_returns = returns[returns < 0]
            avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.01
        
        # 凯利公式: f = (bp - q) / b
        # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss == 0:
            return 0.02
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # 限制在合理范围内 (0-25%)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # 保守调整 (实际使用Kelly值的1/4到1/2)
        conservative_kelly = kelly_fraction * 0.35
        
        return conservative_kelly
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    async def assess_portfolio_risk(self, positions: Dict[str, float], 
                                  total_value: float) -> RiskMetrics:
        """综合评估投资组合风险"""
        # 计算VaR
        portfolio_var = self.calculate_portfolio_var(positions, total_value)
        
        # 计算相关性风险
        symbols = list(positions.keys())
        correlation_matrix = self.calculate_correlation_matrix(symbols)
        
        if correlation_matrix is not None:
            # 平均相关性作为相关性风险
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            correlation_risk = np.mean(np.abs(upper_triangle))
        else:
            correlation_risk = 0.0
        
        # 计算集中度风险 (赫芬达尔指数)
        weights = [positions[s] / total_value for s in symbols] if total_value > 0 else []
        concentration_risk = sum(w**2 for w in weights) if weights else 0.0
        
        # 简化的其他指标
        max_drawdown = 0.0  # 需要历史净值数据
        sharpe_ratio = 0.0  # 需要基准收益率
        leverage_ratio = sum(positions.values()) / total_value if total_value > 0 else 1.0
        
        return RiskMetrics(
            portfolio_var=portfolio_var,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio
        )
    
    def validate_new_position(self, symbol: str, position_value: float, 
                            current_positions: Dict[str, float], 
                            total_portfolio_value: float) -> Dict[str, Any]:
        """验证新增持仓是否符合风险控制要求"""
        validation_result = {
            'approved': True,
            'reasons': [],
            'risk_score': 0.0,
            'recommended_size': position_value
        }
        
        if total_portfolio_value <= 0:
            validation_result['approved'] = False
            validation_result['reasons'].append("投资组合价值无效")
            return validation_result
        
        # 1. 单个持仓限制检查
        position_weight = position_value / total_portfolio_value
        if position_weight > self.max_single_position:
            validation_result['approved'] = False
            validation_result['reasons'].append(f"单个持仓超限: {position_weight:.1%} > {self.max_single_position:.1%}")
            validation_result['recommended_size'] = total_portfolio_value * self.max_single_position
        
        # 2. 相关性检查
        correlation_risk = self.check_correlation_risk(symbol, current_positions)
        if correlation_risk > self.max_correlation:
            validation_result['approved'] = False
            validation_result['reasons'].append(f"相关性风险过高: {correlation_risk:.2f} > {self.max_correlation:.2f}")
            validation_result['risk_score'] += 0.3
        
        # 3. 投资组合VaR检查
        test_positions = current_positions.copy()
        test_positions[symbol] = position_value
        test_var = self.calculate_portfolio_var(test_positions, total_portfolio_value + position_value)
        
        if test_var > self.max_portfolio_var:
            validation_result['approved'] = False
            validation_result['reasons'].append(f"组合VaR超限: {test_var:.2%} > {self.max_portfolio_var:.2%}")
            validation_result['risk_score'] += 0.4
        
        # 4. 集中度检查
        test_weights = [v / (total_portfolio_value + position_value) for v in test_positions.values()]
        concentration = sum(w**2 for w in test_weights)
        if concentration > 0.3:  # HHI > 0.3 表示高度集中
            validation_result['reasons'].append(f"持仓过度集中: HHI={concentration:.2f}")
            validation_result['risk_score'] += 0.2
        
        # 5. 凯利公式建议
        kelly_size = self.calculate_kelly_criterion(symbol)
        kelly_value = total_portfolio_value * kelly_size
        
        if position_value > kelly_value * 2:  # 超过凯利建议的2倍
            validation_result['reasons'].append(f"超过凯利建议: ${position_value:.0f} > ${kelly_value:.0f}")
            validation_result['recommended_size'] = min(validation_result['recommended_size'], kelly_value)
            validation_result['risk_score'] += 0.1
        
        return validation_result
    
    async def get_position_recommendations(self, symbols: List[str], 
                                         total_budget: float) -> Dict[str, float]:
        """基于风险平价和凯利公式的仓位建议"""
        recommendations = {}
        
        if not symbols or total_budget <= 0:
            return recommendations
        
        # 计算每个标的的凯利建议仓位
        kelly_weights = {}
        total_kelly = 0
        
        for symbol in symbols:
            kelly_weight = self.calculate_kelly_criterion(symbol)
            kelly_weights[symbol] = kelly_weight
            total_kelly += kelly_weight
        
        # 归一化并分配资金
        if total_kelly > 0:
            for symbol in symbols:
                normalized_weight = kelly_weights[symbol] / total_kelly
                # 限制单个仓位不超过15%
                capped_weight = min(normalized_weight, 0.15)
                recommendations[symbol] = total_budget * capped_weight
        else:
            # 如果凯利计算失败，平均分配
            equal_weight = min(1.0 / len(symbols), 0.15)
            for symbol in symbols:
                recommendations[symbol] = total_budget * equal_weight
        
        return recommendations
    
    def get_risk_report(self, positions: Dict[str, float], 
                       total_value: float) -> Dict[str, Any]:
        """生成风险报告"""
        if not positions or total_value <= 0:
            return {"status": "无持仓数据"}
        
        # 基础统计
        position_count = len(positions)
        largest_position = max(positions.values())
        largest_weight = largest_position / total_value
        
        # 风险检查
        risk_warnings = []
        
        if largest_weight > self.max_single_position:
            risk_warnings.append(f"最大单仓超限: {largest_weight:.1%}")
        
        if position_count < 5:
            risk_warnings.append(f"持仓过度集中: 仅{position_count}个标的")
        
        # VaR计算
        portfolio_var = self.calculate_portfolio_var(positions, total_value)
        
        if portfolio_var > self.max_portfolio_var:
            risk_warnings.append(f"组合VaR超限: {portfolio_var:.2%}")
        
        # 相关性检查
        symbols = list(positions.keys())
        correlation_matrix = self.calculate_correlation_matrix(symbols)
        avg_correlation = 0.0
        
        if correlation_matrix is not None:
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            avg_correlation = np.mean(np.abs(upper_triangle))
            
            if avg_correlation > self.max_correlation:
                risk_warnings.append(f"相关性过高: {avg_correlation:.2f}")
        
        return {
            "持仓数量": position_count,
            "最大单仓权重": f"{largest_weight:.1%}",
            "投资组合VaR": f"{portfolio_var:.2%}",
            "平均相关性": f"{avg_correlation:.2f}",
            "风险警告": risk_warnings,
            "风险评级": "高风险" if len(risk_warnings) >= 3 else "中风险" if len(risk_warnings) >= 1 else "低风险"
        }
    
    async def validate_order_comprehensive(self, symbol: str, side: str, qty: int, price: float, 
                                         net_liq: float, cash_balance: float, positions: Dict[str, int],
                                         order_verify_cfg: Dict, account_ready: bool = True,
                                         last_account_update: float = 0.0, account_update_interval: float = 60.0) -> Dict[str, Any]:
        """统一的综合订单验证 - 整合所有风险检查"""
        import time
        
        validation_result = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'risk_score': 0.0,
            'recommended_qty': qty,
            'recommended_value': qty * price
        }
        
        try:
            # 1. 基础状态检查
            if not account_ready:
                validation_result['approved'] = False
                validation_result['reasons'].append("账户状态未就绪")
                return validation_result
            
            # 2. 账户数据时效性检查
            current_time = time.time()
            if current_time - last_account_update > account_update_interval:
                validation_result['warnings'].append("账户数据可能过期，建议刷新")
            
            # 3. 净值检查
            if net_liq <= 0:
                validation_result['approved'] = False
                validation_result['reasons'].append("账户净值为0")
                return validation_result
            
            # 4. 价格区间验证
            price_range = order_verify_cfg.get("price_range", [0.5, 10000])
            if price < price_range[0] or price > price_range[1]:
                validation_result['approved'] = False
                validation_result['reasons'].append(f"价格超出安全区间: ${price:.2f} (允许: ${price_range[0]}-${price_range[1]})")
                return validation_result
            
            # 5. 订单价值检查
            order_value = qty * price
            min_order_value = order_verify_cfg.get("min_order_value_usd", 100)
            if order_value < min_order_value:
                validation_result['approved'] = False
                validation_result['reasons'].append(f"订单金额过小: ${order_value:.2f} < ${min_order_value}")
                return validation_result
            
            # 6. 买单资金检查
            if side.upper() == "BUY":
                # 计算可用现金
                cash_reserve_pct = order_verify_cfg.get("cash_reserve_pct", 0.10)
                reserved = net_liq * cash_reserve_pct
                available_cash = max(cash_balance - reserved, 0.0)
                
                if order_value > available_cash:
                    validation_result['approved'] = False
                    validation_result['reasons'].append(f"可用现金不足: 需要${order_value:.2f}, 可用${available_cash:.2f}")
                    
                    # 建议调整仓位
                    if available_cash > min_order_value:
                        recommended_qty = int(available_cash / price)
                        validation_result['recommended_qty'] = recommended_qty
                        validation_result['recommended_value'] = recommended_qty * price
                    return validation_result
                
                # 单标的持仓上限检查
                max_single_position_pct = order_verify_cfg.get("max_single_position_pct", 0.15)
                max_position_value = net_liq * max_single_position_pct
                
                if order_value > max_position_value:
                    validation_result['approved'] = False
                    validation_result['reasons'].append(f"超过单标的上限: ${order_value:.2f} > ${max_position_value:.2f}")
                    
                    # 建议调整到最大允许仓位
                    recommended_qty = int(max_position_value / price)
                    validation_result['recommended_qty'] = recommended_qty
                    validation_result['recommended_value'] = recommended_qty * price
                    return validation_result
            
            # 7. 卖单持仓检查
            elif side.upper() == "SELL":
                current_position = positions.get(symbol, 0)
                if current_position < qty:
                    validation_result['approved'] = False
                    validation_result['reasons'].append(f"持仓不足: 当前{current_position}, 需要{qty}")
                    
                    # 建议调整到实际持仓
                    validation_result['recommended_qty'] = max(current_position, 0)
                    validation_result['recommended_value'] = validation_result['recommended_qty'] * price
                    return validation_result
            
            # 8. 高级风险检查（仅对买单）
            if side.upper() == "BUY":
                # 更新价格历史
                await self.update_price_history(symbol, price)
                
                # 转换持仓为价值
                current_positions_value = {}
                for pos_symbol, pos_qty in positions.items():
                    if pos_qty > 0:
                        # 简化价格获取，实际应该从ticker获取
                        pos_price = price if pos_symbol == symbol else 100.0  # 默认价格
                        current_positions_value[pos_symbol] = pos_qty * pos_price
                
                # 高级风险验证
                risk_validation = self.validate_new_position(
                    symbol=symbol,
                    position_value=order_value,
                    current_positions=current_positions_value,
                    total_portfolio_value=net_liq
                )
                
                if not risk_validation['approved']:
                    validation_result['approved'] = False
                    validation_result['reasons'].extend(risk_validation['reasons'])
                    validation_result['risk_score'] = risk_validation['risk_score']
                    
                    # 使用风险管理器的建议
                    if risk_validation['recommended_size'] < order_value:
                        recommended_qty = int(risk_validation['recommended_size'] / price)
                        validation_result['recommended_qty'] = recommended_qty
                        validation_result['recommended_value'] = recommended_qty * price
                
                # 相关性检查
                correlation_risk = self.check_correlation_risk(symbol, current_positions_value)
                if correlation_risk > 0.7:
                    validation_result['approved'] = False
                    validation_result['reasons'].append(f"相关性风险过高: {correlation_risk:.2f}")
                    validation_result['risk_score'] += 0.3
                
                # 凯利公式检查
                kelly_size = self.calculate_kelly_criterion(symbol)
                kelly_value = net_liq * kelly_size
                
                if order_value > kelly_value * 3:  # 超过凯利建议3倍
                    validation_result['warnings'].append(f"超过凯利建议: ${order_value:.0f} > ${kelly_value:.0f} (3倍)")
                    validation_result['risk_score'] += 0.2
                    
                    # 如果风险评分过高，拒绝订单
                    if validation_result['risk_score'] > 0.6:
                        validation_result['approved'] = False
                        validation_result['reasons'].append("综合风险评分过高")
            
            return validation_result
            
        except Exception as e:
            validation_result['approved'] = False
            validation_result['reasons'].append(f"风险验证异常: {str(e)}")
            return validation_result