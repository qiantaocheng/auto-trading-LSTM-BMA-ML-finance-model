
# 增强风险管理器适配器
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

class EnhancedRiskManagerAdapter:
    """增强风险管理器适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('risk_manager')
        self.risk_limits = {
            'max_position_pct': 0.15,      # 单个仓位最大占比
            'max_daily_loss_pct': 0.05,    # 单日最大损失
            'max_portfolio_exposure': 0.95, # 最大投资组合暴露
            'min_cash_reserve_pct': 0.1,   # 最小现金储备
            'max_correlation_threshold': 0.7  # 最大相关性阈值
        }
        self.position_tracker = {}
        self.daily_pnl = 0.0
        self.start_nav = None
        
    def check_position_risk(self, symbol: str, quantity: int, price: float, 
                           portfolio_value: float) -> Tuple[bool, str]:
        """检查仓位风险"""
        try:
            position_value = quantity * price
            position_pct = position_value / portfolio_value
            
            # 检查单个仓位限制
            if position_pct > self.risk_limits['max_position_pct']:
                return False, f"仓位风险: {symbol} 占比 {position_pct:.2%} 超过限制 {self.risk_limits['max_position_pct']:.2%}"
            
            # 检查投资组合总暴露
            total_exposure = self._calculate_total_exposure(portfolio_value)
            if total_exposure > self.risk_limits['max_portfolio_exposure']:
                return False, f"投资组合暴露 {total_exposure:.2%} 超过限制 {self.risk_limits['max_portfolio_exposure']:.2%}"
            
            return True, "仓位风险检查通过"
            
        except Exception as e:
            self.logger.error(f"仓位风险检查错误: {e}")
            return False, f"风险检查异常: {e}"
    
    def check_daily_loss_limit(self, current_nav: float) -> Tuple[bool, str]:
        """检查日内损失限制"""
        try:
            if self.start_nav is None:
                self.start_nav = current_nav
                return True, "初始NAV已设置"
            
            daily_pnl_pct = (current_nav - self.start_nav) / self.start_nav
            
            if daily_pnl_pct < -self.risk_limits['max_daily_loss_pct']:
                return False, f"日内损失 {daily_pnl_pct:.2%} 超过限制 {self.risk_limits['max_daily_loss_pct']:.2%}"
            
            return True, f"日内PnL: {daily_pnl_pct:.2%}"
            
        except Exception as e:
            self.logger.error(f"日内损失检查错误: {e}")
            return False, f"损失检查异常: {e}"
    
    def check_cash_reserve(self, cash_available: float, total_value: float) -> Tuple[bool, str]:
        """检查现金储备"""
        try:
            cash_pct = cash_available / total_value
            
            if cash_pct < self.risk_limits['min_cash_reserve_pct']:
                return False, f"现金储备 {cash_pct:.2%} 低于最小要求 {self.risk_limits['min_cash_reserve_pct']:.2%}"
            
            return True, f"现金储备充足: {cash_pct:.2%}"
            
        except Exception as e:
            return False, f"现金检查异常: {e}"
    
    def _calculate_total_exposure(self, portfolio_value: float) -> float:
        """计算总投资组合暴露"""
        # 简化计算：假设当前暴露为已用资金比例
        total_position_value = sum(
            pos.get('value', 0) for pos in self.position_tracker.values()
        )
        return total_position_value / portfolio_value if portfolio_value > 0 else 0
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """更新仓位追踪"""
        self.position_tracker[symbol] = {
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'timestamp': datetime.now()
        }
    
    def calculate_var(self, positions: Dict[str, Any], confidence: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        try:
            # 简化VaR计算
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            # 假设日波动率为2%
            daily_volatility = 0.02
            z_score = 1.645 if confidence == 0.95 else 2.33  # 95%或99%置信度
            
            var = total_value * daily_volatility * z_score
            return var
            
        except Exception as e:
            self.logger.error(f"VaR计算错误: {e}")
            return 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        return {
            'risk_limits': self.risk_limits,
            'position_count': len(self.position_tracker),
            'daily_pnl': self.daily_pnl,
            'start_nav': self.start_nav,
            'status': 'active'
        }
    
    def update_risk_limits(self, new_limits: Dict[str, float]):
        """更新风险限制"""
        self.risk_limits.update(new_limits)
        self.logger.info(f"风险限制已更新: {new_limits}")
    
    def reset_daily_tracking(self):
        """重置日内追踪"""
        self.daily_pnl = 0.0
        self.start_nav = None
        self.logger.info("日内风险追踪已重置")

# 全局风险管理器实例
_risk_manager = None

def get_risk_manager():
    """获取风险管理器实例"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = EnhancedRiskManagerAdapter()
    return _risk_manager
