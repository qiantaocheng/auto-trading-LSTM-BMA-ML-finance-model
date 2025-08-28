#!/usr/bin/env python3
"""
Real Risk Balancer - 真实风险平衡器实现
替换Mock风险平衡器，提供真实的投资组合风险管理
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    weight: float = 0.0
    sector: Optional[str] = None
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price - 1.0)

@dataclass
class RiskMetrics:
    """投资组合风险指标"""
    total_value: float
    concentration_risk: float  # 最大单仓占比
    sector_concentration: Dict[str, float]  # 行业集中度
    volatility_adjusted_exposure: float
    max_drawdown_risk: float
    leverage_ratio: float = 1.0

class RealRiskBalancer:
    """
    真实风险平衡器
    提供投资组合风险控制和再平衡功能
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger("RealRiskBalancer")
        
        # 风险参数配置
        self.config = {
            'max_single_position_weight': 0.15,  # 单仓最大占比15%
            'max_sector_weight': 0.30,           # 行业最大占比30%
            'rebalance_threshold': 0.05,         # 5%偏差触发再平衡
            'min_trade_size': 100,               # 最小交易数量
            'max_daily_turnover': 0.20,          # 日最大换手率20%
            'target_volatility': 0.15,           # 目标波动率15%
            'max_leverage': 1.0,                 # 最大杠杆率
            'correlation_threshold': 0.7,        # 相关性阈值
            **((config or {}))
        }
        
        # 运行时状态
        self.enabled = True
        self.last_rebalance_time = None
        self.rebalance_history = []
        self.risk_alerts = []
        
        # 投资组合数据
        self.positions: Dict[str, Position] = {}
        self.total_portfolio_value = 0.0
        self.cash_balance = 0.0
        
        self.logger.info("Real Risk Balancer initialized")
    
    def is_enabled(self) -> bool:
        """检查风险平衡器是否启用"""
        return self.enabled
    
    def enable(self):
        """启用风险平衡器"""
        self.enabled = True
        self.logger.info("Risk balancer enabled")
    
    def disable(self):
        """禁用风险平衡器"""
        self.enabled = False
        self.logger.info("Risk balancer disabled")
    
    def update_positions(self, positions_data: Dict[str, Any]) -> bool:
        """
        更新持仓数据
        
        Args:
            positions_data: 持仓数据字典
            格式: {symbol: {'quantity': int, 'entry_price': float, 'current_price': float, ...}}
        """
        try:
            self.positions.clear()
            total_value = 0.0
            
            for symbol, pos_data in positions_data.items():
                if not isinstance(pos_data, dict):
                    continue
                
                quantity = pos_data.get('quantity', 0)
                current_price = pos_data.get('current_price', 0.0)
                entry_price = pos_data.get('entry_price', current_price)
                
                if quantity == 0:
                    continue
                
                market_value = abs(quantity) * current_price
                total_value += market_value
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    sector=pos_data.get('sector', 'Unknown')
                )
                
                self.positions[symbol] = position
            
            # 计算权重
            self.total_portfolio_value = total_value
            for position in self.positions.values():
                position.weight = position.market_value / total_value if total_value > 0 else 0
            
            self.logger.debug(f"Updated {len(self.positions)} positions, total value: ${total_value:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
            return False
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算投资组合风险指标"""
        if not self.positions:
            return RiskMetrics(
                total_value=0.0,
                concentration_risk=0.0,
                sector_concentration={},
                volatility_adjusted_exposure=0.0,
                max_drawdown_risk=0.0
            )
        
        # 计算最大单仓集中度
        max_weight = max(pos.weight for pos in self.positions.values()) if self.positions else 0.0
        
        # 计算行业集中度
        sector_weights = defaultdict(float)
        for pos in self.positions.values():
            sector = pos.sector or 'Unknown'
            sector_weights[sector] += pos.weight
        
        # 计算波动率调整后的敞口（简化版本）
        vol_adjusted_exposure = sum(
            pos.weight * (1 + abs(pos.unrealized_pnl_pct) * 0.5)  # 简化的波动率估算
            for pos in self.positions.values()
        )
        
        # 计算最大回撤风险（基于当前未实现损益）
        unrealized_losses = sum(
            min(0, pos.unrealized_pnl) for pos in self.positions.values()
        )
        max_drawdown_risk = abs(unrealized_losses) / self.total_portfolio_value if self.total_portfolio_value > 0 else 0
        
        return RiskMetrics(
            total_value=self.total_portfolio_value,
            concentration_risk=max_weight,
            sector_concentration=dict(sector_weights),
            volatility_adjusted_exposure=vol_adjusted_exposure,
            max_drawdown_risk=max_drawdown_risk
        )
    
    def validate_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        验证风险限制
        
        Returns:
            Tuple[bool, List[str]]: (是否通过验证, 违规说明列表)
        """
        violations = []
        
        if not self.enabled:
            return True, []
        
        risk_metrics = self.calculate_risk_metrics()
        
        # 检查单仓集中度
        if risk_metrics.concentration_risk > self.config['max_single_position_weight']:
            violations.append(
                f"单仓集中度过高: {risk_metrics.concentration_risk:.1%} > "
                f"{self.config['max_single_position_weight']:.1%}"
            )
        
        # 检查行业集中度
        for sector, weight in risk_metrics.sector_concentration.items():
            if weight > self.config['max_sector_weight']:
                violations.append(
                    f"行业[{sector}]集中度过高: {weight:.1%} > "
                    f"{self.config['max_sector_weight']:.1%}"
                )
        
        # 检查回撤风险
        if risk_metrics.max_drawdown_risk > 0.10:  # 10%回撤警告
            violations.append(
                f"投资组合回撤风险较高: {risk_metrics.max_drawdown_risk:.1%}"
            )
        
        # 检查杠杆率
        if risk_metrics.leverage_ratio > self.config['max_leverage']:
            violations.append(
                f"杠杆率超限: {risk_metrics.leverage_ratio:.2f}x > "
                f"{self.config['max_leverage']:.2f}x"
            )
        
        is_valid = len(violations) == 0
        
        if violations:
            self.logger.warning(f"Risk limit violations: {violations}")
        
        return is_valid, violations
    
    def balance_portfolio(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        主要风险平衡方法
        
        Args:
            positions: 原始持仓数据
            
        Returns:
            Dict: 调整后的持仓建议
        """
        if not self.enabled:
            self.logger.debug("Risk balancer disabled, returning positions unchanged")
            return positions
        
        try:
            # 更新持仓数据
            if not self.update_positions(positions):
                self.logger.error("Failed to update positions, returning original")
                return positions
            
            # 验证风险限制
            is_valid, violations = self.validate_risk_limits()
            
            if is_valid:
                self.logger.debug("All risk limits passed")
                return positions
            
            # 记录风险违规
            self.risk_alerts.extend([
                {
                    'timestamp': datetime.now(),
                    'type': 'risk_violation',
                    'message': violation
                }
                for violation in violations
            ])
            
            # 生成再平衡建议
            rebalanced_positions = self._generate_rebalance_recommendations(positions)
            
            # 记录再平衡操作
            self.last_rebalance_time = datetime.now()
            self.rebalance_history.append({
                'timestamp': self.last_rebalance_time,
                'violations': violations,
                'original_positions': len(positions),
                'adjusted_positions': len(rebalanced_positions)
            })
            
            self.logger.info(f"Portfolio rebalanced due to {len(violations)} risk violations")
            return rebalanced_positions
            
        except Exception as e:
            self.logger.error(f"Portfolio balancing failed: {e}")
            return positions
    
    def _generate_rebalance_recommendations(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成再平衡建议
        
        Args:
            positions: 原始持仓
            
        Returns:
            Dict: 调整后的持仓建议
        """
        rebalanced = positions.copy()
        risk_metrics = self.calculate_risk_metrics()
        
        # 处理单仓集中度过高的情况
        if risk_metrics.concentration_risk > self.config['max_single_position_weight']:
            # 找出权重最大的持仓
            max_position = max(self.positions.values(), key=lambda p: p.weight)
            
            # 计算需要减少的数量
            target_weight = self.config['max_single_position_weight'] * 0.9  # 保留10%缓冲
            target_value = target_weight * self.total_portfolio_value
            target_quantity = int(target_value / max_position.current_price)
            
            if target_quantity < abs(max_position.quantity):
                adjustment = abs(max_position.quantity) - target_quantity
                self.logger.info(
                    f"Reducing {max_position.symbol} by {adjustment} shares "
                    f"(from {max_position.quantity} to {target_quantity})"
                )
                
                # 更新持仓建议
                if max_position.symbol in rebalanced:
                    original_qty = rebalanced[max_position.symbol].get('quantity', 0)
                    new_qty = target_quantity if original_qty > 0 else -target_quantity
                    rebalanced[max_position.symbol]['quantity'] = new_qty
        
        # 处理行业集中度过高的情况
        for sector, weight in risk_metrics.sector_concentration.items():
            if weight > self.config['max_sector_weight']:
                # 获取该行业的所有持仓
                sector_positions = [
                    pos for pos in self.positions.values() 
                    if pos.sector == sector
                ]
                
                # 按权重排序，优先调整权重最大的
                sector_positions.sort(key=lambda p: p.weight, reverse=True)
                
                reduction_needed = weight - self.config['max_sector_weight']
                reduction_value = reduction_needed * self.total_portfolio_value
                
                for pos in sector_positions:
                    if reduction_value <= 0:
                        break
                    
                    # 计算该持仓的减少量
                    pos_reduction = min(reduction_value, pos.market_value * 0.5)  # 最多减少50%
                    pos_qty_reduction = int(pos_reduction / pos.current_price)
                    
                    if pos_qty_reduction > 0 and pos.symbol in rebalanced:
                        original_qty = rebalanced[pos.symbol].get('quantity', 0)
                        sign = 1 if original_qty > 0 else -1
                        new_qty = abs(original_qty) - pos_qty_reduction
                        rebalanced[pos.symbol]['quantity'] = new_qty * sign
                        
                        reduction_value -= pos_qty_reduction * pos.current_price
                        
                        self.logger.info(
                            f"Reducing {pos.symbol} by {pos_qty_reduction} shares "
                            f"for sector[{sector}] concentration"
                        )
        
        return rebalanced
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要信息"""
        risk_metrics = self.calculate_risk_metrics()
        
        return {
            'enabled': self.enabled,
            'total_positions': len(self.positions),
            'total_value': risk_metrics.total_value,
            'concentration_risk': risk_metrics.concentration_risk,
            'max_concentration_limit': self.config['max_single_position_weight'],
            'sector_exposures': risk_metrics.sector_concentration,
            'max_sector_limit': self.config['max_sector_weight'],
            'max_drawdown_risk': risk_metrics.max_drawdown_risk,
            'recent_alerts': len([
                alert for alert in self.risk_alerts
                if (datetime.now() - alert['timestamp']) < timedelta(hours=1)
            ]),
            'last_rebalance': (
                self.last_rebalance_time.isoformat() 
                if self.last_rebalance_time else None
            ),
            'rebalance_count_today': len([
                rb for rb in self.rebalance_history
                if rb['timestamp'].date() == datetime.now().date()
            ])
        }
    
    def get_position_recommendations(self, symbol: str, target_quantity: int, 
                                   current_price: float) -> Tuple[bool, int, List[str]]:
        """
        获取新仓位建议
        
        Args:
            symbol: 股票代码
            target_quantity: 目标数量
            current_price: 当前价格
            
        Returns:
            Tuple[bool, int, List[str]]: (是否允许, 建议数量, 风险说明)
        """
        if not self.enabled:
            return True, target_quantity, []
        
        warnings = []
        suggested_quantity = target_quantity
        
        # 计算目标仓位价值和权重
        target_value = abs(target_quantity) * current_price
        target_weight = target_value / max(self.total_portfolio_value, target_value)
        
        # 检查单仓集中度
        if target_weight > self.config['max_single_position_weight']:
            max_value = self.config['max_single_position_weight'] * self.total_portfolio_value
            max_quantity = int(max_value / current_price)
            suggested_quantity = max_quantity if target_quantity > 0 else -max_quantity
            
            warnings.append(
                f"仓位过大，建议从 {target_quantity} 调整为 {suggested_quantity}"
            )
        
        # 检查最小交易数量
        if abs(suggested_quantity) < self.config['min_trade_size']:
            warnings.append(
                f"交易数量过小: {abs(suggested_quantity)} < {self.config['min_trade_size']}"
            )
            if abs(suggested_quantity) < self.config['min_trade_size'] / 2:
                return False, 0, warnings + ["交易数量过小，建议取消"]
        
        is_allowed = len([w for w in warnings if "建议取消" in w]) == 0
        
        return is_allowed, suggested_quantity, warnings


class RiskBalancerAdapter:
    """风险平衡器适配器 - 提供向后兼容接口"""
    
    def __init__(self, enabled: bool = True, config: Optional[Dict] = None):
        self.risk_balancer = RealRiskBalancer(config)
        if enabled:
            self.risk_balancer.enable()
        else:
            self.risk_balancer.disable()
        
        self.logger = logging.getLogger("RiskBalancerAdapter")
    
    def balance_portfolio(self, positions):
        """平衡投资组合 - 兼容原接口"""
        return self.risk_balancer.balance_portfolio(positions)
    
    def is_enabled(self):
        """检查是否启用"""
        return self.risk_balancer.is_enabled()
    
    def get_risk_summary(self):
        """获取风险摘要"""
        return self.risk_balancer.get_risk_summary()


def get_risk_balancer_adapter(enable_balancer: bool = True, config: Optional[Dict] = None) -> RiskBalancerAdapter:
    """
    获取真实风险平衡器适配器
    
    Args:
        enable_balancer: 是否启用风险平衡
        config: 配置参数
        
    Returns:
        RiskBalancerAdapter: 风险平衡器适配器实例
    """
    return RiskBalancerAdapter(enabled=enable_balancer, config=config)


if __name__ == "__main__":
    # 测试代码
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试风险平衡器
    balancer = RealRiskBalancer()
    
    # 测试持仓数据
    test_positions = {
        'AAPL': {
            'quantity': 1000,
            'entry_price': 150.0,
            'current_price': 155.0,
            'sector': 'Technology'
        },
        'MSFT': {
            'quantity': 800,
            'entry_price': 300.0,
            'current_price': 310.0,
            'sector': 'Technology'
        },
        'TSLA': {
            'quantity': 500,
            'entry_price': 200.0,
            'current_price': 190.0,
            'sector': 'Automotive'
        }
    }
    
    # 测试风险平衡
    result = balancer.balance_portfolio(test_positions)
    print("Balance result:", json.dumps(result, indent=2, default=str))
    
    # 测试风险摘要
    summary = balancer.get_risk_summary()
    print("Risk summary:", json.dumps(summary, indent=2, default=str))
    
    print("Real Risk Balancer test completed")