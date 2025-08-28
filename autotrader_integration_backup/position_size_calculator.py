#!/usr/bin/env python3
"""
动态头寸规模计算器 - 基于资金百分比的股票数量计算
解决固定数量100股的问题，实现风险管理
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
        return default
    if math.isnan(numerator) or math.isinf(numerator):
        return default
    return numerator / denominator


class PositionSizeMethod(Enum):
    """头寸计算方法"""
    FIXED_PERCENTAGE = "fixed_percentage"  # 固定百分比
    KELLY_CRITERION = "kelly_criterion"    # 凯利公式
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 波动率调整
    SIGNAL_STRENGTH = "signal_strength"    # 信号强度调整


@dataclass
class PositionSizeConfig:
    """头寸计算配置"""
    target_percentage: float = 0.05  # 目标资金百分比 (5%)
    min_percentage: float = 0.04     # 最小百分比 (4%)
    max_percentage: float = 0.10     # 最大百分比 (10%)
    
    min_shares: int = 1              # 最小股数
    max_shares: int = 10000          # 最大股数
    min_order_value: float = 100.0   # 最小订单金额
    
    method: PositionSizeMethod = PositionSizeMethod.FIXED_PERCENTAGE
    
    # 高级参数
    volatility_lookback: int = 20    # 波动率计算回看期
    kelly_confidence: float = 0.5    # 凯利公式置信度折扣
    signal_scaling: bool = True      # 是否根据信号强度缩放
    
    # 🚀 增强风险管理参数
    target_portfolio_volatility: float = 0.15  # 目标组合年化波动率 (15%)
    max_single_loss_pct: float = 0.005         # 单笔最大损失占账户比例 (0.5%)
    atr_period: int = 14                       # ATR计算周期
    atr_multiplier: float = 2.0                # ATR止损倍数
    
    # 流动性约束
    max_adv_pct: float = 0.01                  # 最大占ADV20比例 (1%)
    min_dollar_volume: float = 1000000.0       # 最小日均成交额 ($1M)
    adv_lookback: int = 20                     # 平均成交量回看期
    
    # 风险预算模式
    use_risk_budget: bool = True               # 是否使用风险预算模式
    use_target_volatility: bool = True         # 是否使用目标波动率模式
    use_liquidity_constraint: bool = True      # 是否使用流动性约束


class PositionSizeCalculator:
    """动态头寸规模计算器"""
    
    def __init__(self, config: PositionSizeConfig = None):
        self.config = config or PositionSizeConfig()
        self.logger = logging.getLogger("PositionSizeCalculator")
        
        # 缓存价格数据
        self.price_cache: Dict[str, float] = {}
        self.volatility_cache: Dict[str, float] = {}
        
    def calculate_position_size(self, 
                              symbol: str,
                              current_price: float,
                              signal_strength: float,  # 统一命名：signal_strength
                              available_cash: float,
                              signal_confidence: float = 0.8,  # 统一命名：signal_confidence
                              historical_volatility: Optional[float] = None,
                              price_history: Optional[List[float]] = None,
                              volume_history: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        计算动态头寸规模
        
        Args:
            symbol: 股票代码
            current_price: 当前股价
            signal_strength: 信号强度 (-1 到 1)
            available_cash: 可用资金
            signal_confidence: 信号置信度 (0 到 1)
            historical_volatility: 历史波动率 (可选)
            
        Returns:
            包含股数、资金占比、风险指标的字典
        """
        try:
            # 输入验证
            if current_price <= 0:
                return self._create_error_result(f"Invalid price: {current_price}")
            
            if available_cash <= 0:
                return self._create_error_result(f"Invalid cash: {available_cash}")
            
            # 🚀 增强风险管理预处理
            enhanced_constraints = self._apply_enhanced_risk_management(
                symbol, current_price, signal_strength, available_cash,
                price_history, volume_history, historical_volatility
            )
            
            # 如果增强风险管理拒绝交易
            if not enhanced_constraints['can_trade']:
                return self._create_error_result(enhanced_constraints['reason'])
            
            # 根据方法选择计算策略
            if self.config.method == PositionSizeMethod.FIXED_PERCENTAGE:
                result = self._calculate_fixed_percentage(
                    symbol, current_price, signal_strength, available_cash
                )
            elif self.config.method == PositionSizeMethod.KELLY_CRITERION:
                result = self._calculate_kelly_criterion(
                    symbol, current_price, signal_strength, available_cash, 
                    signal_confidence, historical_volatility
                )
            elif self.config.method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                result = self._calculate_volatility_adjusted(
                    symbol, current_price, signal_strength, available_cash, 
                    historical_volatility
                )
            elif self.config.method == PositionSizeMethod.SIGNAL_STRENGTH:
                result = self._calculate_signal_strength_adjusted(
                    symbol, current_price, signal_strength, available_cash, 
                    signal_confidence
                )
            else:
                return self._create_error_result(f"Unknown method: {self.config.method}")
            
            # 🚀 应用增强风险约束到计算结果
            if enhanced_constraints.get('final_max_shares') is not None:
                original_shares = result.get('shares', 0)
                max_allowed_shares = enhanced_constraints['final_max_shares']
                
                if original_shares > max_allowed_shares:
                    # 应用风险约束限制
                    result['shares'] = max_allowed_shares
                    result['actual_value'] = max_allowed_shares * current_price
                    result['actual_percentage'] = (result['actual_value'] / available_cash) if available_cash > 0 else 0
                    result['risk_constrained'] = True
                    result['original_shares'] = original_shares
                    result['limiting_factor'] = enhanced_constraints.get('limiting_factor')
                    result['risk_constraints'] = enhanced_constraints
                    
                    self.logger.info(f"{symbol} 风险约束调整: {original_shares}股 → {max_allowed_shares}股 "
                                   f"(限制因素: {enhanced_constraints.get('limiting_factor')})")
                else:
                    result['risk_constrained'] = False
                    result['risk_constraints'] = enhanced_constraints
            
            # 验证结果
            validated_result = self._validate_and_adjust_position(result, available_cash)
            
            self.logger.debug(f"{symbol} 头寸计算: {validated_result}")
            return validated_result
            
        except Exception as e:
            self.logger.error(f"计算{symbol}头寸失败: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_fixed_percentage(self, symbol: str, current_price: float, 
                                  signal_strength: float, available_cash: float) -> Dict[str, Any]:
        """固定百分比方法"""
        
        # 基础目标百分比
        base_percentage = self.config.target_percentage
        
        # 根据信号强度调整 (可选)
        if self.config.signal_scaling:
            # 信号强度范围 [-1, 1] 映射到 [0.5, 1.5] 的缩放因子
            signal_scale = 0.5 + abs(signal_strength)
            adjusted_percentage = base_percentage * signal_scale
        else:
            adjusted_percentage = base_percentage
        
        # 确保在阈值范围内
        adjusted_percentage = max(self.config.min_percentage, 
                                min(self.config.max_percentage, adjusted_percentage))
        
        # 计算目标投资金额
        target_value = available_cash * adjusted_percentage
        
        # 计算股数
        target_shares = int(target_value / current_price)
        
        # 实际投资金额
        actual_value = target_shares * current_price
        actual_percentage = actual_value / available_cash if available_cash > 0 else 0
        
        return {
            'symbol': symbol,
            'shares': target_shares,
            'actual_value': actual_value,
            'actual_percentage': actual_percentage,
            'target_percentage': adjusted_percentage,
            'price': current_price,
            'method': 'fixed_percentage',
            'signal_strength': signal_strength,
            'valid': True,
            'reason': f"固定百分比 {adjusted_percentage:.1%}"
        }
    
    def _calculate_kelly_criterion(self, symbol: str, current_price: float,
                                 signal_strength: float, available_cash: float,
                                 signal_confidence: float, 
                                 historical_volatility: Optional[float]) -> Dict[str, Any]:
        """凯利公式方法"""
        
        # 估算胜率基于信号强度和置信度
        # 基础胜率50%，根据信号调整
        base_win_rate = 0.5
        signal_adjustment = signal_confidence * abs(signal_strength) * 0.2  # 最大调整20%
        win_rate = base_win_rate + signal_adjustment
        win_rate = max(0.51, min(0.85, win_rate))  # 限制在合理范围[51%, 85%]
        
        # 使用历史波动率和信号强度估算盈亏幅度
        if historical_volatility and historical_volatility > 0:
            # 基于波动率和信号强度估算盈亏幅度
            base_volatility = historical_volatility
            avg_win = base_volatility * (1.5 + abs(signal_strength) * 0.5)  # 1.5-2.0倍波动率
            avg_loss = base_volatility * (0.8 + (1 - signal_confidence) * 0.4)  # 0.8-1.2倍波动率
        else:
            # 默认值基于市场经验
            avg_win = 0.04 + abs(signal_strength) * 0.02  # 4%-6%
            avg_loss = 0.02 + (1 - signal_confidence) * 0.02  # 2%-4%
        
        # 确保avg_loss > 0避免除零错误
        avg_loss = max(avg_loss, 0.005)  # 最小0.5%损失
        
        # 正确的凯利公式: f* = (bp - q) / b = p - q/b
        # 其中 b = avg_win/avg_loss (盈亏比), p = 胜率, q = 败率
        b = safe_divide(avg_win, avg_loss, 1.0)  # 盈亏比，默认1:1
        p = win_rate           # 胜率
        q = 1 - win_rate       # 败率
        
        # Kelly fraction = (胜率 * 盈亏比 - 败率) / 盈亏比
        kelly_fraction = safe_divide(p * b - q, b, 0.0)
        
        # 防止负数或过大的Kelly值
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # 限制在0-25%
        
        # 保守调整 (通常使用25%-50%的凯利值)
        conservative_kelly = kelly_fraction * self.config.kelly_confidence
        
        # 限制在配置范围内
        adjusted_percentage = max(self.config.min_percentage,
                                min(self.config.max_percentage, conservative_kelly))
        
        # 计算股数
        target_value = available_cash * adjusted_percentage
        target_shares = int(target_value / current_price)
        
        actual_value = target_shares * current_price
        actual_percentage = actual_value / available_cash if available_cash > 0 else 0
        
        return {
            'symbol': symbol,
            'shares': target_shares,
            'actual_value': actual_value,
            'actual_percentage': actual_percentage,
            'target_percentage': adjusted_percentage,
            'price': current_price,
            'method': 'kelly_criterion',
            'signal_strength': signal_strength,
            'win_rate': win_rate,
            'kelly_fraction': kelly_fraction,
            'conservative_kelly': conservative_kelly,
            'valid': True,
            'reason': f"凯利公式 {conservative_kelly:.1%} (原始{kelly_fraction:.1%})"
        }
    
    def _calculate_volatility_adjusted(self, symbol: str, current_price: float,
                                     signal_strength: float, available_cash: float,
                                     historical_volatility: Optional[float]) -> Dict[str, Any]:
        """波动率调整方法"""
        
        # 获取或估算波动率，并进行安全检查
        if historical_volatility is None:
            # 使用缓存或默认值
            volatility = self.volatility_cache.get(symbol, 0.25)  # 默认25%年化波动率
        else:
            volatility = historical_volatility
        
        # 波动率安全检查
        if volatility is None or math.isnan(volatility) or volatility <= 0:
            self.logger.warning(f"{symbol}: 无效波动率 {volatility}，使用默认值")
            volatility = 0.25  # 25%默认年化波动率
        
        # 确保波动率在合理范围内
        volatility = max(0.05, min(2.0, volatility))  # 限制在5%-200%之间
        
        # 目标波动率 (投资组合层面)
        target_portfolio_vol = self.config.target_portfolio_volatility
        
        # 计算位置规模以匹配目标波动率
        # position_vol = position_weight * stock_vol
        # target_position_weight = target_portfolio_vol / stock_vol
        # 使用安全除法计算目标权重
        target_weight = safe_divide(target_portfolio_vol, volatility, self.config.target_percentage)
        
        # 根据信号强度调整
        signal_adjusted_weight = target_weight * abs(signal_strength)
        
        # 限制在配置范围内
        adjusted_percentage = max(self.config.min_percentage,
                                min(self.config.max_percentage, signal_adjusted_weight))
        
        # 计算股数
        target_value = available_cash * adjusted_percentage
        target_shares = int(target_value / current_price)
        
        actual_value = target_shares * current_price
        actual_percentage = actual_value / available_cash if available_cash > 0 else 0
        
        return {
            'symbol': symbol,
            'shares': target_shares,
            'actual_value': actual_value,
            'actual_percentage': actual_percentage,
            'target_percentage': adjusted_percentage,
            'price': current_price,
            'method': 'volatility_adjusted',
            'signal_strength': signal_strength,
            'volatility': volatility,
            'target_weight': target_weight,
            'valid': True,
            'reason': f"波动率调整 {adjusted_percentage:.1%} (波动率{volatility:.1%})"
        }
    
    def _calculate_signal_strength_adjusted(self, symbol: str, current_price: float,
                                          signal_strength: float, available_cash: float,
                                          signal_confidence: float) -> Dict[str, Any]:
        """信号强度调整方法"""
        
        # 基础百分比
        base_percentage = self.config.target_percentage
        
        # 信号强度调整因子 (0 到 2)
        strength_multiplier = abs(signal_strength) * 2
        
        # 置信度调整因子 (0.5 到 1.5)
        confidence_multiplier = 0.5 + signal_confidence
        
        # 综合调整
        adjusted_percentage = base_percentage * strength_multiplier * confidence_multiplier
        
        # 限制在配置范围内
        adjusted_percentage = max(self.config.min_percentage,
                                min(self.config.max_percentage, adjusted_percentage))
        
        # 计算股数
        target_value = available_cash * adjusted_percentage
        target_shares = int(target_value / current_price)
        
        actual_value = target_shares * current_price
        actual_percentage = actual_value / available_cash if available_cash > 0 else 0
        
        return {
            'symbol': symbol,
            'shares': target_shares,
            'actual_value': actual_value,
            'actual_percentage': actual_percentage,
            'target_percentage': adjusted_percentage,
            'price': current_price,
            'method': 'signal_strength',
            'signal_strength': signal_strength,
            'signal_confidence': signal_confidence,
            'strength_multiplier': strength_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'valid': True,
            'reason': f"信号强度调整 {adjusted_percentage:.1%} (强度{abs(signal_strength):.2f}, 置信度{signal_confidence:.2f})"
        }
    
    def _validate_and_adjust_position(self, result: Dict[str, Any], available_cash: float) -> Dict[str, Any]:
        """验证和调整头寸"""
        
        shares = result.get('shares', 0)
        price = result.get('price', 0)
        symbol = result.get('symbol', '')
        
        # 检查最小股数
        if shares < self.config.min_shares:
            if shares == 0:
                result.update({
                    'valid': False,
                    'reason': f"计算股数为0，跳过交易",
                    'warning': 'ZERO_SHARES'
                })
            else:
                shares = self.config.min_shares
                result.update({
                    'shares': shares,
                    'actual_value': shares * price,
                    'actual_percentage': (shares * price) / available_cash,
                    'adjusted': True,
                    'adjustment_reason': f"调整到最小股数 {self.config.min_shares}"
                })
        
        # 检查最大股数
        if shares > self.config.max_shares:
            shares = self.config.max_shares
            result.update({
                'shares': shares,
                'actual_value': shares * price,
                'actual_percentage': (shares * price) / available_cash,
                'adjusted': True,
                'adjustment_reason': f"调整到最大股数 {self.config.max_shares}"
            })
        
        # 检查最小订单金额
        order_value = shares * price
        if order_value < self.config.min_order_value:
            # 调整到最小订单金额
            min_shares = math.ceil(self.config.min_order_value / price)
            if min_shares <= self.config.max_shares:
                shares = min_shares
                result.update({
                    'shares': shares,
                    'actual_value': shares * price,
                    'actual_percentage': (shares * price) / available_cash,
                    'adjusted': True,
                    'adjustment_reason': f"调整到最小订单金额 ${self.config.min_order_value}"
                })
            else:
                result.update({
                    'valid': False,
                    'reason': f"股价过高，无法满足最小订单金额要求",
                    'warning': 'PRICE_TOO_HIGH'
                })
        
        # 检查资金充足性
        final_order_value = result.get('actual_value', 0)
        if final_order_value > available_cash:
            # 按可用资金调整
            affordable_shares = int(available_cash / price)
            if affordable_shares >= self.config.min_shares:
                result.update({
                    'shares': affordable_shares,
                    'actual_value': affordable_shares * price,
                    'actual_percentage': (affordable_shares * price) / available_cash,
                    'adjusted': True,
                    'adjustment_reason': f"调整到可用资金限制"
                })
            else:
                result.update({
                    'valid': False,
                    'reason': f"资金不足，无法购买最小股数",
                    'warning': 'INSUFFICIENT_FUNDS'
                })
        
        # 检查百分比是否在合理范围
        actual_percentage = result.get('actual_percentage', 0)
        if actual_percentage < self.config.min_percentage:
            result['warning'] = result.get('warning', '') + '_LOW_PERCENTAGE'
        elif actual_percentage > self.config.max_percentage:
            result['warning'] = result.get('warning', '') + '_HIGH_PERCENTAGE'
        
        return result
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'shares': 0,
            'actual_value': 0.0,
            'actual_percentage': 0.0,
            'valid': False,
            'error': error_message,
            'method': str(self.config.method.value)
        }
    
    def update_price_cache(self, symbol: str, price: float):
        """更新价格缓存"""
        self.price_cache[symbol] = price
    
    def update_volatility_cache(self, symbol: str, volatility: float):
        """更新波动率缓存"""
        self.volatility_cache[symbol] = volatility
    
    def get_cached_price(self, symbol: str) -> Optional[float]:
        """获取缓存价格"""
        return self.price_cache.get(symbol)
    
    def batch_calculate_positions(self, signals: List[Dict[str, Any]], 
                                available_cash: float) -> List[Dict[str, Any]]:
        """批量计算头寸"""
        results = []
        
        for signal in signals:
            symbol = signal.get('symbol', '')
            current_price = signal.get('price', 0)
            signal_strength = signal.get('weighted_prediction', 0)
            confidence = signal.get('confidence', 0.8)
            volatility = signal.get('volatility', None)
            
            if current_price <= 0:
                self.logger.warning(f"{symbol} 价格无效: {current_price}")
                continue
            
            result = self.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                signal_strength=signal_strength,
                available_cash=available_cash,
                signal_confidence=confidence,
                historical_volatility=volatility
            )
            
            if result.get('valid', False):
                results.append(result)
        
        return results
    
    def get_position_summary(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取头寸汇总"""
        if not positions:
            return {'total_positions': 0, 'total_value': 0.0, 'total_percentage': 0.0}
        
        total_value = sum(pos.get('actual_value', 0) for pos in positions)
        total_percentage = sum(pos.get('actual_percentage', 0) for pos in positions)
        
        valid_positions = [pos for pos in positions if pos.get('valid', False)]
        
        return {
            'total_positions': len(positions),
            'valid_positions': len(valid_positions),
            'total_value': total_value,
            'total_percentage': total_percentage,
            'average_position_size': total_value / len(valid_positions) if valid_positions else 0,
            'largest_position': max((pos.get('actual_value', 0) for pos in valid_positions), default=0),
            'smallest_position': min((pos.get('actual_value', 0) for pos in valid_positions), default=0)
        }
    
    def _apply_enhanced_risk_management(self, 
                                      symbol: str,
                                      current_price: float,
                                      signal_strength: float,
                                      available_cash: float,
                                      price_history: Optional[List[float]] = None,
                                      volume_history: Optional[List[float]] = None,
                                      historical_volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        应用增强风险管理约束
        
        实现以下风险管理规则：
        1. ATR-based风险预算：单笔最大损失 ≤ 账户权益的b% 
        2. 目标波动率法：头寸权重与风险预算、价格、波动和流动性挂钩
        3. 流动性约束：成交额上限 = min(shares, cap_dollar / price)
        4. Kelly半凯利：实盘取0.25-0.5 Kelly
        
        Returns:
            包含约束结果的字典
        """
        result = {
            'can_trade': True,
            'reason': '',
            'constraints': {},
            'max_shares_by_risk': None,
            'max_shares_by_liquidity': None,
            'max_shares_by_volatility': None,
            'recommended_method': self.config.method
        }
        
        try:
            # 1. ATR-based风险预算约束
            if self.config.use_risk_budget and price_history:
                atr_constraint = self._apply_atr_risk_budget(
                    symbol, current_price, available_cash, price_history
                )
                result['constraints']['atr_risk_budget'] = atr_constraint
                result['max_shares_by_risk'] = atr_constraint['max_shares']
                
                if atr_constraint['max_shares'] <= 0:
                    result['can_trade'] = False
                    result['reason'] = f"ATR风险预算限制: {atr_constraint['reason']}"
                    return result
            
            # 2. 目标波动率约束  
            if self.config.use_target_volatility:
                vol_constraint = self._apply_target_volatility_constraint(
                    symbol, current_price, signal_strength, available_cash, 
                    historical_volatility, price_history
                )
                result['constraints']['target_volatility'] = vol_constraint
                result['max_shares_by_volatility'] = vol_constraint['max_shares']
                
                if vol_constraint['max_shares'] <= 0:
                    result['can_trade'] = False
                    result['reason'] = f"目标波动率限制: {vol_constraint['reason']}"
                    return result
            
            # 3. 流动性约束
            if self.config.use_liquidity_constraint and volume_history:
                liquidity_constraint = self._apply_liquidity_constraint(
                    symbol, current_price, volume_history
                )
                result['constraints']['liquidity'] = liquidity_constraint
                result['max_shares_by_liquidity'] = liquidity_constraint['max_shares']
                
                if liquidity_constraint['max_shares'] <= 0:
                    result['can_trade'] = False
                    result['reason'] = f"流动性限制: {liquidity_constraint['reason']}"
                    return result
            
            # 4. 综合约束：取最严格的限制
            max_shares_constraints = [
                result.get('max_shares_by_risk'),
                result.get('max_shares_by_liquidity'), 
                result.get('max_shares_by_volatility')
            ]
            
            valid_constraints = [c for c in max_shares_constraints if c is not None and c > 0]
            
            if valid_constraints:
                result['final_max_shares'] = min(valid_constraints)
                # 记录限制来源
                if result['final_max_shares'] == result.get('max_shares_by_risk'):
                    result['limiting_factor'] = 'risk_budget'
                elif result['final_max_shares'] == result.get('max_shares_by_liquidity'):
                    result['limiting_factor'] = 'liquidity'
                elif result['final_max_shares'] == result.get('max_shares_by_volatility'):
                    result['limiting_factor'] = 'volatility'
            else:
                result['final_max_shares'] = None
            
            self.logger.debug(f"{symbol} 增强风险约束: "
                            f"风险={result.get('max_shares_by_risk')}, "
                            f"流动性={result.get('max_shares_by_liquidity')}, "
                            f"波动率={result.get('max_shares_by_volatility')}, "
                            f"最终={result.get('final_max_shares')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"应用增强风险管理失败 {symbol}: {e}")
            result['can_trade'] = False
            result['reason'] = f"风险管理计算错误: {str(e)}"
            return result
    
    def _apply_atr_risk_budget(self, 
                             symbol: str, 
                             current_price: float,
                             available_cash: float,
                             price_history: List[float]) -> Dict[str, Any]:
        """
        应用ATR-based风险预算
        
        公式: shares = ⌊(b * E) / (ATR14 * 美元每股)⌋
        其中: b = max_single_loss_pct, E = available_cash, ATR14 = 14日ATR
        """
        try:
            if len(price_history) < self.config.atr_period + 1:
                return {
                    'max_shares': 0,
                    'reason': '历史数据不足计算ATR',
                    'atr_value': None
                }
            
            # 计算ATR (简化版，使用价格变化)
            prices = np.array(price_history[:self.config.atr_period + 1])
            price_changes = np.abs(np.diff(prices))
            atr = np.mean(price_changes)
            
            # 风险预算计算
            max_loss_dollar = available_cash * self.config.max_single_loss_pct
            stop_distance = atr * self.config.atr_multiplier  # ATR倍数作为止损距离
            
            if stop_distance <= 0:
                return {
                    'max_shares': 0,
                    'reason': 'ATR计算结果无效',
                    'atr_value': atr
                }
            
            # 计算最大股数
            max_shares = int(max_loss_dollar / stop_distance)
            
            return {
                'max_shares': max(0, max_shares),
                'atr_value': atr,
                'stop_distance': stop_distance,
                'max_loss_dollar': max_loss_dollar,
                'reason': f'ATR={atr:.3f}, 止损距离={stop_distance:.3f}'
            }
            
        except Exception as e:
            return {
                'max_shares': 0,
                'reason': f'ATR计算失败: {str(e)}',
                'atr_value': None
            }
    
    def _apply_target_volatility_constraint(self,
                                          symbol: str,
                                          current_price: float, 
                                          signal_strength: float,
                                          available_cash: float,
                                          historical_volatility: Optional[float],
                                          price_history: Optional[List[float]]) -> Dict[str, Any]:
        """
        应用目标波动率约束
        
        目标波动率法: w_i = clip(c * s_norm_i / σ_i,ann, [-w_max, w_max])
        其中: c 由组合目标波动倒推, σ_i,ann ≈ σ_i * √252
        """
        try:
            # 获取或计算波动率
            if historical_volatility:
                volatility = historical_volatility
            elif price_history and len(price_history) > 20:
                returns = np.diff(np.log(price_history[-60:]))  # 最近60天
                daily_vol = np.std(returns, ddof=1)
                volatility = daily_vol * np.sqrt(252)  # 年化
            else:
                volatility = 0.20  # 默认20%年化波动率
            
            if volatility <= 0:
                return {
                    'max_shares': 0,
                    'reason': '波动率计算结果无效',
                    'volatility': volatility
                }
            
            # 目标组合波动率分配
            target_weight = self.config.target_portfolio_volatility / volatility
            
            # 根据信号强度调整
            signal_adjusted_weight = target_weight * abs(signal_strength)
            
            # 应用最大权重限制
            final_weight = min(signal_adjusted_weight, self.config.max_percentage)
            
            # 计算股数
            target_value = available_cash * final_weight
            max_shares = int(target_value / current_price)
            
            return {
                'max_shares': max(0, max_shares),
                'volatility': volatility,
                'target_weight': target_weight,
                'signal_adjusted_weight': signal_adjusted_weight,
                'final_weight': final_weight,
                'reason': f'波动率={volatility:.3f}, 目标权重={final_weight:.3f}'
            }
            
        except Exception as e:
            return {
                'max_shares': 0,
                'reason': f'目标波动率计算失败: {str(e)}',
                'volatility': None
            }
    
    def _apply_liquidity_constraint(self,
                                  symbol: str,
                                  current_price: float,
                                  volume_history: List[float]) -> Dict[str, Any]:
        """
        应用流动性约束
        
        成交额上限: min(shares, cap_dollar / price)
        其中: cap_dollar = max_adv_pct * ADV20 * price
        """
        try:
            if len(volume_history) < self.config.adv_lookback:
                return {
                    'max_shares': 0,
                    'reason': '成交量历史数据不足',
                    'adv': None
                }
            
            # 计算平均日成交量 (ADV)
            recent_volumes = volume_history[:self.config.adv_lookback]
            adv = np.mean([v for v in recent_volumes if v > 0])
            
            if adv <= 0:
                return {
                    'max_shares': 0,
                    'reason': '平均成交量为零',
                    'adv': adv
                }
            
            # 计算日均成交额
            daily_dollar_volume = adv * current_price
            
            # 流动性检查：是否满足最小成交额要求
            if daily_dollar_volume < self.config.min_dollar_volume:
                return {
                    'max_shares': 0,
                    'reason': f'日均成交额不足${self.config.min_dollar_volume:,.0f}',
                    'adv': adv,
                    'daily_dollar_volume': daily_dollar_volume
                }
            
            # 计算成交额上限
            max_dollar_participation = daily_dollar_volume * self.config.max_adv_pct
            max_shares = int(max_dollar_participation / current_price)
            
            return {
                'max_shares': max(0, max_shares),
                'adv': adv,
                'daily_dollar_volume': daily_dollar_volume,
                'max_participation_pct': self.config.max_adv_pct,
                'max_dollar_participation': max_dollar_participation,
                'reason': f'ADV={adv:,.0f}, 参与率{self.config.max_adv_pct:.1%}'
            }
            
        except Exception as e:
            return {
                'max_shares': 0,
                'reason': f'流动性计算失败: {str(e)}',
                'adv': None
            }


# =============================================================================
# 便捷函数
# =============================================================================

def create_position_calculator(target_percentage: float = 0.05,
                             min_percentage: float = 0.04,
                             max_percentage: float = 0.10,
                             method: str = "fixed_percentage") -> PositionSizeCalculator:
    """创建头寸计算器的便捷函数"""
    
    method_enum = PositionSizeMethod(method)
    config = PositionSizeConfig(
        target_percentage=target_percentage,
        min_percentage=min_percentage,
        max_percentage=max_percentage,
        method=method_enum
    )
    
    return PositionSizeCalculator(config)


def calculate_shares_for_percentage(price: float, 
                                  target_percentage: float,
                                  available_cash: float) -> int:
    """简单的百分比股数计算"""
    if price <= 0 or available_cash <= 0:
        return 0
    
    target_value = available_cash * target_percentage
    return int(target_value / price)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建计算器
    calculator = create_position_calculator(
        target_percentage=0.05,  # 5%目标
        min_percentage=0.04,     # 4%最小
        max_percentage=0.10,     # 10%最大
        method="fixed_percentage"
    )
    
    # 测试数据
    test_cases = [
        {'symbol': 'AAPL', 'price': 150.0, 'signal': 0.8, 'confidence': 0.9},
        {'symbol': 'MSFT', 'price': 300.0, 'signal': -0.6, 'confidence': 0.7},
        {'symbol': 'GOOGL', 'price': 2500.0, 'signal': 0.4, 'confidence': 0.8},
        {'symbol': 'TSLA', 'price': 200.0, 'signal': 0.9, 'confidence': 0.6},
    ]
    
    available_cash = 100000.0  # $100,000
    
    print("=== 动态头寸规模计算测试 ===")
    print(f"可用资金: ${available_cash:,.2f}")
    print(f"目标百分比: 5% (浮动范围 4%-10%)")
    print()
    
    results = []
    for case in test_cases:
        result = calculator.calculate_position_size(
            symbol=case['symbol'],
            current_price=case['price'],
            signal_strength=case['signal'],
            available_cash=available_cash,
            signal_confidence=case['confidence']
        )
        
        results.append(result)
        
        if result.get('valid', False):
            print(f"{case['symbol']:6s}: {result['shares']:4d}股 "
                  f"${result['actual_value']:8,.2f} ({result['actual_percentage']:5.1%}) "
                  f"- {result['reason']}")
        else:
            print(f"{case['symbol']:6s}: 跳过 - {result.get('error', 'Unknown error')}")
    
    # 汇总
    summary = calculator.get_position_summary(results)
    print(f"\n=== 汇总 ===")
    print(f"有效头寸: {summary['valid_positions']}/{summary['total_positions']}")
    print(f"总投资: ${summary['total_value']:,.2f} ({summary['total_percentage']:.1%})")
    print(f"平均头寸: ${summary['average_position_size']:,.2f}")
    
    print("\n✅ 动态头寸规模计算器测试完成")