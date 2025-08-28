
# 统一因子管理器适配器
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class UnifiedFactorManagerAdapter:
    """统一因子管理器适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('factor_manager')
        self.factor_weights = {
            'momentum': 0.25,
            'mean_reversion': 0.30,
            'trend': 0.30,
            'volume': 0.20,
            'volatility': 0.15
        }
        self.factor_cache = {}
        
    def calculate_factors(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算因子值"""
        try:
            factors = {}
            
            if 'price' in market_data:
                price = market_data['price']
                volume = market_data.get('volume', 0)
                
                # 计算各种因子
                factors['momentum'] = self._calculate_momentum_factor(symbol, price)
                factors['mean_reversion'] = self._calculate_mean_reversion_factor(symbol, price)
                factors['trend'] = self._calculate_trend_factor(symbol, price)
                factors['volume'] = self._calculate_volume_factor(symbol, volume)
                factors['volatility'] = self._calculate_volatility_factor(symbol, price)
                
            return factors
            
        except Exception as e:
            self.logger.error(f"因子计算错误 [{symbol}]: {e}")
            return {}
    
    def _calculate_momentum_factor(self, symbol: str, price: float) -> float:
        """计算动量因子"""
        if symbol not in self.factor_cache:
            self.factor_cache[symbol] = {'prices': [], 'returns': []}
        
        cache = self.factor_cache[symbol]
        cache['prices'].append(price)
        
        # 保留最近20个价格
        cache['prices'] = cache['prices'][-20:]
        
        if len(cache['prices']) >= 2:
            recent_return = (cache['prices'][-1] - cache['prices'][-2]) / cache['prices'][-2]
            cache['returns'].append(recent_return)
            cache['returns'] = cache['returns'][-10:]  # 保留最近10个收益率
            
            # 动量 = 近期收益率的移动平均
            if len(cache['returns']) >= 3:
                return np.mean(cache['returns'][-3:])
        
        return 0.0
    
    def _calculate_mean_reversion_factor(self, symbol: str, price: float) -> float:
        """计算均值回归因子"""
        if symbol not in self.factor_cache:
            return 0.0
            
        prices = self.factor_cache[symbol]['prices']
        if len(prices) >= 10:
            # 计算价格相对于移动平均的偏离度
            ma_10 = np.mean(prices[-10:])
            deviation = (ma_10 - price) / ma_10 if ma_10 != 0 else 0
            return deviation  # 正值表示价格低于均值，预期反弹
        
        return 0.0
    
    def _calculate_trend_factor(self, symbol: str, price: float) -> float:
        """计算趋势因子"""
        if symbol not in self.factor_cache:
            return 0.0
            
        prices = self.factor_cache[symbol]['prices']
        if len(prices) >= 5:
            # 简单线性趋势
            x = np.arange(len(prices))
            trend_slope = np.polyfit(x, prices, 1)[0]
            return trend_slope / price if price != 0 else 0  # 标准化趋势斜率
        
        return 0.0
    
    def _calculate_volume_factor(self, symbol: str, volume: float) -> float:
        """计算成交量因子"""
        # 简化的成交量因子
        if volume > 2000000:
            return 0.5  # 高成交量
        elif volume > 1000000:
            return 0.2  # 中等成交量
        elif volume > 500000:
            return 0.0  # 正常成交量
        else:
            return -0.2  # 低成交量
    
    def _calculate_volatility_factor(self, symbol: str, price: float) -> float:
        """计算波动率因子"""
        if symbol not in self.factor_cache:
            return 0.0
            
        returns = self.factor_cache[symbol].get('returns', [])
        if len(returns) >= 5:
            volatility = np.std(returns)
            # 波动率因子：低波动率为正（稳定性好），高波动率为负（风险高）
            return -volatility if volatility > 0.02 else 0.1
        
        return 0.0
    
    def get_weighted_factor_score(self, factors: Dict[str, float]) -> float:
        """获取加权因子分数"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor_name, factor_value in factors.items():
            if factor_name in self.factor_weights:
                weight = self.factor_weights[factor_name]
                weighted_score += factor_value * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def update_factor_weights(self, new_weights: Dict[str, float]):
        """更新因子权重"""
        self.factor_weights.update(new_weights)
        self.logger.info(f"因子权重已更新: {new_weights}")
    
    def get_factor_status(self) -> Dict[str, Any]:
        """获取因子管理器状态"""
        return {
            'factor_weights': self.factor_weights,
            'cached_symbols': len(self.factor_cache),
            'status': 'active'
        }

# 全局因子管理器实例
_factor_manager = None

def get_factor_manager():
    """获取因子管理器实例"""
    global _factor_manager
    if _factor_manager is None:
        _factor_manager = UnifiedFactorManagerAdapter()
    return _factor_manager
