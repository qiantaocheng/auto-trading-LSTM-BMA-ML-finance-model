
# 实时Alpha引擎适配器
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class RealTimeAlphaAdapter:
    """实时Alpha引擎适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('alpha_engine')
        self.alpha_cache = {}
        self.last_update = None
        self.enabled = True
        
    def calculate_alpha_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算Alpha信号"""
        try:
            if not self.enabled:
                return {}
                
            signals = {}
            
            # 简化的Alpha计算逻辑
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'price' in data:
                    price = data['price']
                    volume = data.get('volume', 0)
                    
                    # 基础Alpha信号计算
                    momentum = self._calculate_momentum(symbol, price)
                    mean_reversion = self._calculate_mean_reversion(symbol, price)
                    volume_signal = self._calculate_volume_signal(symbol, volume)
                    
                    # 合成Alpha分数
                    alpha_score = (momentum * 0.4 + mean_reversion * 0.4 + volume_signal * 0.2)
                    signals[symbol] = alpha_score
                    
            self.last_update = datetime.now()
            return signals
            
        except Exception as e:
            self.logger.error(f"Alpha计算错误: {e}")
            return {}
    
    def _calculate_momentum(self, symbol: str, price: float) -> float:
        """计算动量信号"""
        # 简化实现
        if symbol not in self.alpha_cache:
            self.alpha_cache[symbol] = {'prices': []}
        
        self.alpha_cache[symbol]['prices'].append(price)
        prices = self.alpha_cache[symbol]['prices'][-20:]  # 保留最近20个价格
        
        if len(prices) >= 2:
            return (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        return 0
    
    def _calculate_mean_reversion(self, symbol: str, price: float) -> float:
        """计算均值回归信号"""
        if symbol not in self.alpha_cache:
            return 0
            
        prices = self.alpha_cache[symbol]['prices'][-20:]
        if len(prices) >= 5:
            mean_price = np.mean(prices)
            return (mean_price - price) / mean_price if mean_price != 0 else 0
        return 0
    
    def _calculate_volume_signal(self, symbol: str, volume: float) -> float:
        """计算成交量信号"""
        # 简化的成交量信号
        if volume > 1000000:  # 高成交量
            return 0.1
        elif volume < 100000:  # 低成交量
            return -0.1
        return 0
    
    def get_alpha_status(self) -> Dict[str, Any]:
        """获取Alpha引擎状态"""
        return {
            'enabled': self.enabled,
            'last_update': self.last_update,
            'cached_symbols': len(self.alpha_cache),
            'status': 'active' if self.enabled else 'disabled'
        }
    
    def enable_engine(self):
        """启用Alpha引擎"""
        self.enabled = True
        self.logger.info("实时Alpha引擎已启用")
    
    def disable_engine(self):
        """禁用Alpha引擎"""
        self.enabled = False
        self.logger.info("实时Alpha引擎已禁用")

# 全局Alpha引擎实例
_alpha_engine = None

def get_alpha_engine():
    """获取Alpha引擎实例"""
    global _alpha_engine
    if _alpha_engine is None:
        _alpha_engine = RealTimeAlphaAdapter()
    return _alpha_engine
