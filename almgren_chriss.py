#!/usr/bin/env python3
"""
Almgren-Chriss 交易成本模型
简化实现，支持交易APP集成
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketSnapshot:
    """市场快照数据"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: Optional[float] = None
    volatility: Optional[float] = None
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        """中间价"""
        return (self.bid + self.ask) / 2

@dataclass 
class ACParams:
    """Almgren-Chriss模型参数"""
    risk_aversion: float = 1e-6
    volatility_coeff: float = 0.5
    liquidity_coeff: float = 0.1
    time_horizon: float = 1.0
    num_periods: int = 10

@dataclass
class ExecutionBounds:
    """执行边界约束"""
    min_order_size: float = 1.0
    max_order_size: float = 1000000.0
    max_participation_rate: float = 0.1
    min_time_interval: float = 0.1

class AlmgrenChrissOptimizer:
    """Almgren-Chriss优化器"""
    
    def __init__(self, params: Optional[ACParams] = None):
        self.params = params or ACParams()
        self.model = AlmgrenChrissModel(
            risk_aversion=self.params.risk_aversion,
            volatility_coeff=self.params.volatility_coeff,
            liquidity_coeff=self.params.liquidity_coeff
        )
    
    def optimize(self, 
                 shares: float,
                 market_data: MarketSnapshot,
                 bounds: Optional[ExecutionBounds] = None) -> Dict:
        """优化执行计划"""
        bounds = bounds or ExecutionBounds()
        
        # 使用市场数据中的波动率，如果没有则使用默认值
        volatility = market_data.volatility or 0.02
        
        return self.model.calculate_optimal_trajectory(
            total_shares=shares,
            time_horizon=self.params.time_horizon,
            volatility=volatility,
            bid_ask_spread=market_data.spread / market_data.price,
            num_periods=self.params.num_periods
        )

class AlmgrenChrissModel:
    """Almgren-Chriss 交易成本优化模型"""
    
    def __init__(self, 
                 risk_aversion: float = 1e-6,
                 volatility_coeff: float = 0.5,
                 liquidity_coeff: float = 0.1):
        """
        初始化模型参数
        
        Args:
            risk_aversion: 风险厌恶系数
            volatility_coeff: 波动率系数
            liquidity_coeff: 流动性系数
        """
        self.risk_aversion = risk_aversion
        self.volatility_coeff = volatility_coeff
        self.liquidity_coeff = liquidity_coeff
    
    def calculate_optimal_trajectory(self,
                                   total_shares: float,
                                   time_horizon: float,
                                   volatility: float,
                                   bid_ask_spread: float = 0.001,
                                   num_periods: int = 10) -> Dict:
        """
        计算最优交易轨迹
        
        Args:
            total_shares: 总交易股数
            time_horizon: 交易时间窗口(小时)
            volatility: 股票日波动率
            bid_ask_spread: 买卖价差
            num_periods: 分割周期数
        
        Returns:
            包含最优交易轨迹的字典
        """
        try:
            # 时间步长
            dt = time_horizon / num_periods
            
            # 计算模型参数
            sigma = volatility * np.sqrt(dt)
            epsilon = bid_ask_spread / 2
            eta = self.liquidity_coeff * epsilon
            
            # 计算衰减参数
            kappa = np.sqrt(self.risk_aversion * sigma**2 / eta)
            tau = time_horizon
            
            # 计算最优交易速度
            if kappa * tau > 0.1:
                # 正常情况
                sinh_kt = np.sinh(kappa * tau)
                cosh_kt = np.cosh(kappa * tau)
                
                # 时间点
                times = np.linspace(0, tau, num_periods + 1)
                
                # 最优持仓轨迹
                holdings = []
                trade_rates = []
                
                for t in times:
                    remaining_time = tau - t
                    if remaining_time <= 0:
                        holding = 0
                    else:
                        holding = total_shares * np.sinh(kappa * remaining_time) / sinh_kt
                    holdings.append(holding)
                
                # 计算交易量
                for i in range(len(holdings) - 1):
                    trade_rate = holdings[i] - holdings[i + 1]
                    trade_rates.append(trade_rate)
                
            else:
                # 线性近似（小kappa情况）
                times = np.linspace(0, tau, num_periods + 1)
                holdings = [total_shares * (1 - t/tau) for t in times]
                trade_rates = [total_shares / num_periods] * num_periods
            
            # 估算交易成本
            total_cost = self._estimate_trading_cost(
                trade_rates, volatility, bid_ask_spread, dt
            )
            
            result = {
                'times': times.tolist(),
                'holdings': holdings,
                'trade_rates': trade_rates,
                'total_cost': total_cost,
                'model_params': {
                    'kappa': kappa,
                    'risk_aversion': self.risk_aversion,
                    'volatility': volatility,
                    'time_horizon': time_horizon
                }
            }
            
            logger.debug(f"AC模型计算完成: 总成本={total_cost:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"AC模型计算失败: {e}")
            # 返回简单均匀分割作为备用方案
            return self._simple_uniform_split(total_shares, num_periods)
    
    def _estimate_trading_cost(self, 
                              trade_rates: list,
                              volatility: float,
                              bid_ask_spread: float,
                              dt: float) -> float:
        """估算交易成本"""
        try:
            # 市场冲击成本
            market_impact = sum(
                self.liquidity_coeff * rate**2 * dt 
                for rate in trade_rates
            )
            
            # 价差成本
            spread_cost = sum(
                bid_ask_spread * abs(rate) * dt 
                for rate in trade_rates
            )
            
            # 时间风险成本
            risk_cost = self.risk_aversion * volatility**2 * sum(
                rate**2 * dt for rate in trade_rates
            )
            
            return market_impact + spread_cost + risk_cost
            
        except Exception:
            return 0.001  # 默认成本
    
    def _simple_uniform_split(self, total_shares: float, num_periods: int) -> Dict:
        """简单均匀分割备用方案"""
        trade_per_period = total_shares / num_periods
        
        return {
            'times': list(range(num_periods + 1)),
            'holdings': [total_shares - i * trade_per_period for i in range(num_periods + 1)],
            'trade_rates': [trade_per_period] * num_periods,
            'total_cost': 0.001,
            'model_params': {
                'method': 'uniform_split',
                'periods': num_periods
            }
        }

# 向后兼容的工厂函数
def create_almgren_chriss_model(**kwargs) -> AlmgrenChrissModel:
    """创建Almgren-Chriss模型实例"""
    return AlmgrenChrissModel(**kwargs)

# 简化的接口函数
def optimize_execution(shares: float, 
                      time_hours: float, 
                      volatility: float,
                      **kwargs) -> Dict:
    """
    优化执行轨迹的简化接口
    
    Args:
        shares: 要交易的股数
        time_hours: 执行时间(小时)
        volatility: 波动率
    
    Returns:
        最优执行计划
    """
    model = AlmgrenChrissModel()
    return model.calculate_optimal_trajectory(shares, time_hours, volatility, **kwargs)

def create_ac_plan(shares: float, 
                   market_data: MarketSnapshot,
                   params: Optional[ACParams] = None,
                   bounds: Optional[ExecutionBounds] = None) -> Dict:
    """创建AC执行计划"""
    optimizer = AlmgrenChrissOptimizer(params)
    return optimizer.optimize(shares, market_data, bounds)

class ACOptimizerInstance:
    """AC优化器实例，提供历史记录和自适应参数功能"""
    
    def __init__(self):
        self.execution_history: List[Dict] = []
        self.optimizer = AlmgrenChrissOptimizer()
    
    def save_execution_record(self, record: Dict):
        """保存执行记录"""
        self.execution_history.append({
            **record,
            'timestamp': pd.Timestamp.now()
        })
        
        # 保持最近1000条记录
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_adaptive_params(self, symbol: str, market_data: MarketSnapshot, 
                           lookback_days: int = 30) -> Tuple[float, float]:
        """获取自适应参数"""
        # 简化实现：返回默认的eta和gamma参数
        eta = 0.1  # 流动性参数
        gamma = 1e-6  # 风险厌恶参数
        
        # 基于历史数据调整
        recent_records = [
            r for r in self.execution_history 
            if r.get('symbol') == symbol and 
            (pd.Timestamp.now() - r.get('timestamp', pd.Timestamp.now())).days <= lookback_days
        ]
        
        if recent_records:
            # 基于历史执行效果调整参数
            avg_slippage = np.mean([r.get('slippage', 0) for r in recent_records])
            if avg_slippage > 0.001:  # 如果滑点较大，增加流动性参数
                eta *= 1.2
            elif avg_slippage < 0.0005:  # 如果滑点较小，减少流动性参数
                eta *= 0.8
        
        return eta, gamma
    
    def export_calibration_report(self, report_path: str):
        """导出校准报告"""
        try:
            df = pd.DataFrame(self.execution_history)
            if not df.empty:
                df.to_csv(report_path, index=False, encoding='utf-8')
                logger.info(f"AC校准报告已导出: {report_path}")
            else:
                logger.warning("没有执行历史数据可导出")
        except Exception as e:
            logger.error(f"导出校准报告失败: {e}")

# 全局优化器实例
ac_optimizer = ACOptimizerInstance()