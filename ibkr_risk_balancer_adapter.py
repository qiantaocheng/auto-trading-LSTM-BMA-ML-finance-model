#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR风控收益平衡器适配器
连接现有的IBKR交易系统和风控收益平衡器
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_reward_balancer_integrated import (
    RiskRewardBalancer, Config, Signal, TradeSide
)

try:
    from polygon_client import polygon_client
    from autotrader.ibkr_auto_trader import IBKRAutoTrader
    from autotrader.unified_position_manager import UnifiedPositionManager
    from autotrader.data_source_manager import get_data_source_manager
    from autotrader.trading_auditor_v2 import TradingAuditor
except ImportError as e:
    logging.warning(f"导入现有模块失败: {e}")

logger = logging.getLogger(__name__)

class IBKRRiskBalancerAdapter:
    """
    IBKR风控收益平衡器适配器
    将风控收益平衡器与现有的IBKR交易系统无缝集成
    """
    
    def __init__(self, enable_balancer: bool = False):
        """
        初始化适配器
        
        Args:
            enable_balancer: 是否启用风控收益平衡器
        """
        self.enable_balancer = enable_balancer
        
        # 初始化组件
        self.risk_balancer = None
        self.ibkr_trader = None
        self.position_manager = None
        self.data_source_manager = None
        self.trading_auditor = None
        
        # 配置
        self.balancer_config = Config()
        
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化数据源管理器
            self.data_source_manager = get_data_source_manager()
            
            # 初始化IBKR交易器
            if 'IBKRAutoTrader' in globals():
                self.ibkr_trader = IBKRAutoTrader()
            
            # 初始化持仓管理器
            if 'UnifiedPositionManager' in globals():
                self.position_manager = UnifiedPositionManager()
            
            # 初始化审计器
            if 'TradingAuditor' in globals():
                self.trading_auditor = TradingAuditor()
            
            # 初始化风控收益平衡器
            self.risk_balancer = RiskRewardBalancer(
                polygon_client=polygon_client if 'polygon_client' in globals() else None,
                ibkr_trader=self.ibkr_trader,
                config=self.balancer_config
            )
            
            if self.enable_balancer:
                self.risk_balancer.enable()
                
            logger.info("IBKR风控收益平衡器适配器初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
    
    def enable_risk_balancer(self):
        """启用风控收益平衡器"""
        self.enable_balancer = True
        if self.risk_balancer:
            self.risk_balancer.enable()
        logger.info("风控收益平衡器已启用")
    
    def disable_risk_balancer(self):
        """禁用风控收益平衡器"""
        self.enable_balancer = False
        if self.risk_balancer:
            self.risk_balancer.disable()
        logger.info("风控收益平衡器已禁用")
    
    def is_risk_balancer_enabled(self) -> bool:
        """检查风控收益平衡器状态"""
        return self.enable_balancer and (self.risk_balancer and self.risk_balancer.is_enabled())
    
    def update_balancer_config(self, config_dict: Dict[str, Any]):
        """更新风控收益平衡器配置"""
        try:
            # 更新配置对象
            for key, value in config_dict.items():
                if hasattr(self.balancer_config, key):
                    setattr(self.balancer_config, key, value)
                elif hasattr(self.balancer_config.guards, key):
                    setattr(self.balancer_config.guards, key, value)
                elif hasattr(self.balancer_config.sizing, key):
                    setattr(self.balancer_config.sizing, key, value)
                elif hasattr(self.balancer_config.degrade, key):
                    setattr(self.balancer_config.degrade, key, value)
                elif hasattr(self.balancer_config.throttle, key):
                    setattr(self.balancer_config.throttle, key, value)
            
            if self.risk_balancer:
                self.risk_balancer.update_config(self.balancer_config)
                
            logger.info("风控收益平衡器配置已更新")
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    def convert_bma_signals_to_risk_signals(self, bma_results: pd.DataFrame) -> List[Signal]:
        """
        将BMA模型结果转换为风控收益平衡器信号
        
        Args:
            bma_results: BMA模型输出结果
            
        Returns:
            信号列表
        """
        signals = []
        
        try:
            for _, row in bma_results.iterrows():
                symbol = row['symbol']
                
                # 确定交易方向
                prediction = row.get('weighted_prediction', 0)
                side = TradeSide.BUY if prediction > 0 else TradeSide.SELL
                
                # 计算预期alpha(转换为基点)
                expected_alpha_bps = abs(prediction) * 10000
                
                # 信心度
                confidence = row.get('confidence_score', 0.5)
                
                # 模型价格
                model_price = row.get('current_price', 100)
                
                signal = Signal(
                    symbol=symbol,
                    side=side,
                    expected_alpha_bps=expected_alpha_bps,
                    model_price=model_price,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                
                signals.append(signal)
                
            logger.info(f"转换{len(signals)}个BMA信号为风控信号")
            
        except Exception as e:
            logger.error(f"转换BMA信号失败: {e}")
            
        return signals
    
    def get_current_positions(self) -> Dict[str, int]:
        """获取当前持仓"""
        positions = {}
        
        try:
            if self.position_manager:
                # 使用持仓管理器获取持仓
                position_data = self.position_manager.get_all_positions()
                for symbol, position_info in position_data.items():
                    positions[symbol] = position_info.get('quantity', 0)
            elif self.ibkr_trader:
                # 直接从IBKR获取持仓
                positions = self.ibkr_trader.get_positions()
                
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            
        return positions
    
    def get_portfolio_nav(self) -> float:
        """获取组合净值"""
        try:
            if self.position_manager:
                return self.position_manager.get_portfolio_nav()
            elif self.ibkr_trader:
                return self.ibkr_trader.get_account_value()
            else:
                return 1000000.0  # 默认100万
                
        except Exception as e:
            logger.error(f"获取组合净值失败: {e}")
            return 1000000.0
    
    def process_trading_signals(self, signals_data: Any) -> List[Dict]:
        """
        处理交易信号的主入口
        
        Args:
            signals_data: 可能是BMA结果、信号列表等
            
        Returns:
            订单列表
        """
        if not self.is_risk_balancer_enabled():
            logger.info("风控收益平衡器未启用，使用原始交易逻辑")
            return self._process_signals_original(signals_data)
        
        try:
            # 转换为统一的信号格式
            if isinstance(signals_data, pd.DataFrame):
                signals = self.convert_bma_signals_to_risk_signals(signals_data)
            elif isinstance(signals_data, list):
                signals = signals_data  # 假设已经是Signal格式
            else:
                logger.warning(f"不支持的信号数据格式: {type(signals_data)}")
                return []
            
            if not signals:
                logger.info("没有有效信号")
                return []
            
            # 获取当前持仓和组合净值
            current_positions = self.get_current_positions()
            portfolio_nav = self.get_portfolio_nav()
            
            # 使用风控收益平衡器处理信号
            orders = self.risk_balancer.process_signals(
                signals=signals,
                current_positions=current_positions,
                portfolio_nav=portfolio_nav
            )
            
            logger.info(f"风控收益平衡器处理{len(signals)}个信号，生成{len(orders)}个订单")
            
            # 记录审计信息
            if self.trading_auditor:
                for order in orders:
                    self.trading_auditor.log_order_decision(
                        symbol=order['symbol'],
                        decision='approved_by_balancer',
                        reason=order.get('decision_reason', 'processed by risk balancer'),
                        alpha=order.get('signal_alpha', 0),
                        confidence=order.get('signal_confidence', 0)
                    )
            
            return orders
            
        except Exception as e:
            logger.error(f"风控收益平衡器处理信号失败: {e}")
            # 降级到原始处理逻辑
            return self._process_signals_original(signals_data)
    
    def _process_signals_original(self, signals_data: Any) -> List[Dict]:
        """原始信号处理逻辑(备用)"""
        try:
            # 这里实现原始的信号处理逻辑
            # 作为风控收益平衡器的fallback
            orders = []
            
            if isinstance(signals_data, pd.DataFrame):
                for _, row in signals_data.iterrows():
                    symbol = row['symbol']
                    prediction = row.get('weighted_prediction', 0)
                    
                    if abs(prediction) > 0.005:  # 0.5%阈值
                        side = "BUY" if prediction > 0 else "SELL"
                        # 简单的固定数量
                        quantity = 100
                        
                        orders.append({
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'order_type': 'MKT',
                            'source': 'original_logic'
                        })
            
            logger.info(f"原始逻辑生成{len(orders)}个订单")
            return orders
            
        except Exception as e:
            logger.error(f"原始信号处理也失败: {e}")
            return []
    
    def execute_orders(self, orders: List[Dict]) -> bool:
        """执行订单"""
        if not orders:
            return True
        
        try:
            if self.is_risk_balancer_enabled():
                # 使用风控收益平衡器的IBKR接口
                return self.risk_balancer.send_orders_to_ibkr(orders)
            else:
                # 使用原始的IBKR接口
                if self.ibkr_trader:
                    success_count = 0
                    for order in orders:
                        try:
                            if order.get('order_type') == 'LMT':
                                result = self.ibkr_trader.place_limit_order(
                                    symbol=order['symbol'],
                                    side=order['side'],
                                    quantity=order['quantity'],
                                    limit_price=order['limit_price']
                                )
                            else:
                                result = self.ibkr_trader.place_market_order(
                                    symbol=order['symbol'],
                                    side=order['side'],
                                    quantity=order['quantity']
                                )
                            
                            if result:
                                success_count += 1
                                
                        except Exception as e:
                            logger.error(f"执行订单失败: {order['symbol']} - {e}")
                    
                    logger.info(f"执行{success_count}/{len(orders)}个订单")
                    return success_count > 0
                
            return False
            
        except Exception as e:
            logger.error(f"执行订单失败: {e}")
            return False
    
    def get_balancer_stats(self) -> Dict:
        """获取风控收益平衡器统计信息"""
        if self.risk_balancer:
            return self.risk_balancer.get_stats()
        return {}
    
    def reset_balancer_stats(self):
        """重置统计信息"""
        if self.risk_balancer:
            self.risk_balancer.reset_stats()
    
    def clear_balancer_cache(self):
        """清理缓存"""
        if self.risk_balancer:
            self.risk_balancer.clear_cache()

# 全局适配器实例
_adapter_instance = None

def get_risk_balancer_adapter(enable_balancer: bool = False) -> IBKRRiskBalancerAdapter:
    """获取风控收益平衡器适配器单例"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = IBKRRiskBalancerAdapter(enable_balancer=enable_balancer)
    return _adapter_instance

def enable_risk_balancer():
    """启用风控收益平衡器的便捷函数"""
    adapter = get_risk_balancer_adapter()
    adapter.enable_risk_balancer()

def disable_risk_balancer():
    """禁用风控收益平衡器的便捷函数"""
    adapter = get_risk_balancer_adapter()
    adapter.disable_risk_balancer()

def is_risk_balancer_enabled() -> bool:
    """检查风控收益平衡器状态的便捷函数"""
    adapter = get_risk_balancer_adapter()
    return adapter.is_risk_balancer_enabled()

# 为现有系统提供的简化接口
def process_bma_signals_with_risk_control(bma_results: pd.DataFrame) -> List[Dict]:
    """
    为BMA模型提供的风控处理接口
    
    Args:
        bma_results: BMA模型输出
        
    Returns:
        订单列表
    """
    adapter = get_risk_balancer_adapter()
    return adapter.process_trading_signals(bma_results)

def execute_orders_with_risk_control(orders: List[Dict]) -> bool:
    """
    带风控的订单执行接口
    
    Args:
        orders: 订单列表
        
    Returns:
        执行是否成功
    """
    adapter = get_risk_balancer_adapter()
    return adapter.execute_orders(orders)