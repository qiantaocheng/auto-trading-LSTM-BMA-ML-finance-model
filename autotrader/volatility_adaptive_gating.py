#!/usr/bin/env python3
"""
波动率自适应门控系统
用于替代硬编码阈值，基于股票特征动态调整信号门槛
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class VolatilityThresholdConfig:
    """波动率自适应阈值配置"""
    
    # 基础参数
    base_k: float = 0.5                    # 基础门槛系数 k (推荐0.3-0.7)
    volatility_lookback: int = 60          # 波动率计算回望期 (60-90天)
    min_signal_threshold: float = 0.001    # 最小信号门槛
    max_signal_threshold: float = 0.02     # 最大信号门槛
    
    # ATR参数 
    atr_period: int = 14                   # ATR计算周期
    use_atr: bool = True                   # 是否使用ATR代替收益率标准差
    
    # 流动性过滤
    min_dollar_volume: float = 1000000.0   # 最小日均成交额 (100万美元)
    adv_lookback: int = 20                 # 平均成交量回望期
    enable_liquidity_filter: bool = True   # 是否启用流动性过滤
    
    # 极端波动处理
    volatility_cap: float = 0.5            # 波动率上限 (50%年化)
    volatility_floor: float = 0.01         # 波动率下限 (1%年化)
    
    # 自适应调整
    enable_adaptive_k: bool = True         # 是否启用自适应k值调整
    market_regime_lookback: int = 252      # 市场环境评估回望期


class VolatilityAdaptiveGating:
    """
    波动率自适应门控系统
    
    实现基于以下公式的动态阈值:
    - 标准化信号: s_norm = prediction / volatility  
    - 交易门槛: |s_norm| > k
    - 其中 volatility 可以是滚动标准差或ATR
    - k 值可根据市场环境自适应调整
    """
    
    def __init__(self, config: VolatilityThresholdConfig = None):
        self.config = config or VolatilityThresholdConfig()
        self.logger = logging.getLogger("VolatilityAdaptiveGating")
        
        # 缓存数据
        self.volatility_cache: Dict[str, float] = {}
        self.price_cache: Dict[str, List[float]] = {}
        self.volume_cache: Dict[str, List[float]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # 市场环境指标
        self.market_volatility: Optional[float] = None
        self.adaptive_k: float = self.config.base_k
        
    def calculate_volatility(self, 
                           symbol: str,
                           price_data: List[float],
                           use_atr: bool = None) -> float:
        """
        计算股票波动率
        
        Args:
            symbol: 股票代码
            price_data: 价格数据列表 (最新在前)
            use_atr: 是否使用ATR，None表示使用配置默认值
            
        Returns:
            日波动率 (年化)
        """
        if use_atr is None:
            use_atr = self.config.use_atr
            
        if len(price_data) < max(self.config.volatility_lookback, self.config.atr_period):
            self.logger.warning(f"{symbol} 价格数据不足，使用默认波动率")
            return 0.15  # 默认15%年化波动率
            
        try:
            prices = np.array(price_data)
            
            if use_atr:
                # 使用ATR计算波动率
                volatility = self._calculate_atr_volatility(prices)
            else:
                # 使用收益率标准差
                volatility = self._calculate_returns_volatility(prices)
            
            # 应用波动率边界
            volatility = np.clip(volatility, 
                               self.config.volatility_floor, 
                               self.config.volatility_cap)
            
            # 缓存结果
            self.volatility_cache[symbol] = volatility
            self.last_update[symbol] = datetime.now()
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"计算{symbol}波动率失败: {e}")
            return self.volatility_cache.get(symbol, 0.15)
    
    def _calculate_returns_volatility(self, prices: np.ndarray) -> float:
        """基于收益率标准差计算波动率"""
        if len(prices) < 2:
            return 0.15
            
        # 计算对数收益率
        returns = np.diff(np.log(prices[::-1]))  # 反转为时间正序
        
        # 计算滚动标准差 (最近N天)
        lookback = min(self.config.volatility_lookback, len(returns))
        recent_returns = returns[-lookback:]
        
        daily_vol = np.std(recent_returns, ddof=1)
        
        # 年化 (假设252个交易日)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def _calculate_atr_volatility(self, prices: np.ndarray) -> float:
        """基于ATR计算波动率"""
        if len(prices) < self.config.atr_period + 1:
            return 0.15
            
        # 简化版ATR计算 (假设价格为收盘价)
        # 真实ATR需要高低收数据，这里用价格变化近似
        price_changes = np.abs(np.diff(prices[::-1]))  # 反转为时间正序
        
        # 计算ATR
        lookback = min(self.config.atr_period, len(price_changes))
        recent_changes = price_changes[-lookback:]
        atr = np.mean(recent_changes)
        
        # 转换为相对波动率
        current_price = prices[0]  # 最新价格
        if current_price > 0:
            daily_vol = atr / current_price
            annual_vol = daily_vol * np.sqrt(252)
            return annual_vol
        
        return 0.15
    
    def calculate_liquidity_score(self, 
                                symbol: str,
                                volume_data: List[float],
                                price_data: List[float]) -> float:
        """
        计算流动性评分
        
        Args:
            symbol: 股票代码
            volume_data: 成交量数据
            price_data: 价格数据
            
        Returns:
            流动性评分 (0-1, 1表示流动性充足)
        """
        if not self.config.enable_liquidity_filter:
            return 1.0
            
        if len(volume_data) < self.config.adv_lookback or len(price_data) < self.config.adv_lookback:
            self.logger.warning(f"{symbol} 流动性数据不足")
            return 0.5  # 中等流动性评分
            
        try:
            # 计算平均日成交额 (ADV * Price)
            recent_volumes = volume_data[:self.config.adv_lookback]
            recent_prices = price_data[:self.config.adv_lookback]
            
            daily_dollar_volumes = [v * p for v, p in zip(recent_volumes, recent_prices) if v > 0 and p > 0]
            
            if not daily_dollar_volumes:
                return 0.0
                
            avg_dollar_volume = np.mean(daily_dollar_volumes)
            
            # 流动性评分计算
            if avg_dollar_volume >= self.config.min_dollar_volume:
                # 充足流动性：线性递增到1.0
                score = min(1.0, avg_dollar_volume / (self.config.min_dollar_volume * 2))
            else:
                # 不足流动性：线性递减
                score = avg_dollar_volume / self.config.min_dollar_volume
                
            return score
            
        except Exception as e:
            self.logger.error(f"计算{symbol}流动性评分失败: {e}")
            return 0.5
    
    def update_market_regime(self, market_returns: List[float]):
        """
        更新市场环境评估，用于自适应调整k值
        
        Args:
            market_returns: 市场指数收益率序列
        """
        if not self.config.enable_adaptive_k:
            return
            
        try:
            if len(market_returns) < self.config.market_regime_lookback:
                return
                
            # 计算市场波动率
            recent_returns = market_returns[-self.config.market_regime_lookback:]
            market_vol = np.std(recent_returns) * np.sqrt(252)
            self.market_volatility = market_vol
            
            # 根据市场波动率调整k值
            # 高波动期提高门槛，低波动期降低门槛
            base_vol = 0.15  # 基准波动率15%
            vol_ratio = market_vol / base_vol
            
            # 自适应调整 k = base_k * (0.8 + 0.4 * vol_ratio)
            # 当市场波动率较高时增加门槛，较低时降低门槛
            adjustment_factor = 0.8 + 0.4 * vol_ratio
            self.adaptive_k = self.config.base_k * adjustment_factor
            
            # 限制k值范围
            self.adaptive_k = np.clip(self.adaptive_k, 0.1, 1.5)
            
            self.logger.debug(f"市场波动率: {market_vol:.3f}, 调整后k值: {self.adaptive_k:.3f}")
            
        except Exception as e:
            self.logger.error(f"更新市场环境失败: {e}")
    
    def should_trade(self, 
                     symbol: str,
                     signal_strength: float,  # 统一命名：signal_strength代替prediction
                     price_data: List[float],
                     volume_data: Optional[List[float]] = None) -> Tuple[bool, Dict]:
        """
        主要门控判断函数
        
        Args:
            symbol: 股票代码
            signal_strength: 信号强度值 (统一命名)
            price_data: 价格数据 (最新在前)
            volume_data: 成交量数据 (可选)
            
        Returns:
            (是否可交易, 详细信息字典)
        """
        try:
            # 1. 计算股票波动率
            volatility = self.calculate_volatility(symbol, price_data)
            
            # 2. 标准化信号强度
            if volatility <= 0:
                self.logger.warning(f"{symbol} 波动率为0，跳过交易")
                return False, {'reason': 'zero_volatility', 'volatility': volatility}
            
            normalized_signal = abs(signal_strength) / volatility
            
            # 3. 获取动态阈值
            current_k = self.adaptive_k if self.config.enable_adaptive_k else self.config.base_k
            
            # 4. 流动性检查
            liquidity_score = 1.0
            if volume_data and self.config.enable_liquidity_filter:
                liquidity_score = self.calculate_liquidity_score(symbol, volume_data, price_data)
                
            # 5. 综合门控判断
            # 基础阈值检查
            passes_threshold = normalized_signal > current_k
            
            # 流动性门槛 (流动性评分低于0.3时拒绝交易)
            passes_liquidity = liquidity_score >= 0.3
            
            # 绝对最小阈值检查 (即使标准化后也不能太小)
            passes_min_threshold = abs(signal_strength) >= self.config.min_signal_threshold
            
            # 绝对最大阈值检查 (防止异常信号)
            passes_max_threshold = abs(signal_strength) <= self.config.max_signal_threshold
            
            # 综合判断
            can_trade = all([
                passes_threshold,
                passes_liquidity, 
                passes_min_threshold,
                passes_max_threshold
            ])
            
            # 详细信息
            details = {
                'symbol': symbol,
                'signal_strength': signal_strength,
                'volatility': volatility,
                'normalized_signal': normalized_signal,
                'threshold_k': current_k,
                'liquidity_score': liquidity_score,
                'passes_threshold': passes_threshold,
                'passes_liquidity': passes_liquidity,
                'passes_min_threshold': passes_min_threshold,
                'passes_max_threshold': passes_max_threshold,
                'can_trade': can_trade,
                'reason': self._get_rejection_reason(
                    passes_threshold, passes_liquidity, 
                    passes_min_threshold, passes_max_threshold
                )
            }
            
            if can_trade:
                self.logger.debug(f"{symbol} 通过门控: 信号{signal_strength:.4f}, "
                                f"标准化{normalized_signal:.3f}, 阈值{current_k:.3f}")
            else:
                self.logger.debug(f"{symbol} 未通过门控: {details['reason']}")
                
            return can_trade, details
            
        except Exception as e:
            self.logger.error(f"门控判断失败 {symbol}: {e}")
            return False, {'reason': 'calculation_error', 'error': str(e)}
    
    def _get_rejection_reason(self, 
                            passes_threshold: bool,
                            passes_liquidity: bool,
                            passes_min_threshold: bool,
                            passes_max_threshold: bool) -> str:
        """获取拒绝原因"""
        if not passes_min_threshold:
            return 'signal_too_weak'
        elif not passes_max_threshold:
            return 'signal_too_strong'
        elif not passes_threshold:
            return 'below_volatility_threshold'
        elif not passes_liquidity:
            return 'insufficient_liquidity'
        else:
            return 'approved'
    
    def should_gate_signal(self, 
                          symbol: str, 
                          signal_strength: float, 
                          price_data: List[float],
                          volume_data: Optional[List[float]] = None) -> bool:
        """
        判断是否应该门控（阻止）该信号
        
        Args:
            symbol: 股票代码
            signal_strength: 信号强度
            price_data: 价格数据
            volume_data: 成交量数据（可选）
            
        Returns:
            True = 应该门控（阻止交易）
            False = 不门控（允许交易）
        """
        try:
            # 调用主要的门控判断函数
            can_trade, info = self.should_trade(symbol, signal_strength, price_data, volume_data)
            
            # 返回相反的结果：can_trade=True时不门控(False)，can_trade=False时门控(True)
            should_gate = not can_trade
            
            if should_gate:
                reason = info.get('reason', 'unknown')
                self.logger.debug(f"🚫 门控信号 {symbol}: {reason} (强度={signal_strength:.4f})")
            else:
                self.logger.debug(f"✅ 信号通过 {symbol}: 强度={signal_strength:.4f}")
            
            return should_gate
            
        except Exception as e:
            self.logger.error(f"门控检查失败 {symbol}: {e}")
            # 发生错误时采用保守策略：门控
            return True
    
    def get_cached_volatility(self, symbol: str) -> Optional[float]:
        """获取缓存的波动率"""
        return self.volatility_cache.get(symbol)
    
    def clear_cache(self):
        """清空缓存"""
        self.volatility_cache.clear()
        self.price_cache.clear() 
        self.volume_cache.clear()
        self.last_update.clear()
    
    def get_statistics(self) -> Dict:
        """获取门控系统统计信息"""
        return {
            'cached_symbols': len(self.volatility_cache),
            'market_volatility': self.market_volatility,
            'adaptive_k': self.adaptive_k,
            'config': {
                'base_k': self.config.base_k,
                'volatility_lookback': self.config.volatility_lookback,
                'use_atr': self.config.use_atr,
                'enable_liquidity_filter': self.config.enable_liquidity_filter
            }
        }


def create_volatility_gating(base_k: float = 0.5,
                           volatility_lookback: int = 60,
                           use_atr: bool = True,
                           enable_liquidity_filter: bool = True) -> VolatilityAdaptiveGating:
    """
    创建波动率自适应门控系统的便捷函数
    
    Args:
        base_k: 基础门槛系数 (推荐0.3-0.7)
        volatility_lookback: 波动率计算回望期
        use_atr: 是否使用ATR计算波动率
        enable_liquidity_filter: 是否启用流动性过滤
        
    Returns:
        配置好的门控系统实例
    """
    config = VolatilityThresholdConfig(
        base_k=base_k,
        volatility_lookback=volatility_lookback,
        use_atr=use_atr,
        enable_liquidity_filter=enable_liquidity_filter
    )
    
    return VolatilityAdaptiveGating(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建门控系统
    gating = create_volatility_gating(
        base_k=0.5,
        volatility_lookback=60,
        use_atr=True
    )
    
    # 模拟测试数据
    import random
    
    # 模拟价格数据 (高波动股票)
    high_vol_prices = [100 + random.gauss(0, 5) for _ in range(100)]
    high_vol_volumes = [1000000 + random.randint(-200000, 500000) for _ in range(100)]
    
    # 模拟价格数据 (低波动股票) 
    low_vol_prices = [100 + random.gauss(0, 1) for _ in range(100)]
    low_vol_volumes = [2000000 + random.randint(-300000, 700000) for _ in range(100)]
    
    print("=== 波动率自适应门控测试 ===")
    
    # 测试用例
    test_cases = [
        ("AAPL_HIGH_VOL", 0.006, high_vol_prices, high_vol_volumes),
        ("AAPL_HIGH_VOL", 0.003, high_vol_prices, high_vol_volumes),
        ("MSFT_LOW_VOL", 0.006, low_vol_prices, low_vol_volumes),
        ("MSFT_LOW_VOL", 0.003, low_vol_prices, low_vol_volumes),
    ]
    
    for symbol, prediction, prices, volumes in test_cases:
        can_trade, details = gating.should_trade(symbol, prediction, prices, volumes)
        
        print(f"\n{symbol}:")
        print(f"  预测值: {prediction:.4f}")
        print(f"  波动率: {details['volatility']:.3f}")
        print(f"  标准化信号: {details['normalized_signal']:.3f}")
        print(f"  阈值: {details['threshold_k']:.3f}")
        print(f"  流动性评分: {details['liquidity_score']:.2f}")
        print(f"  可交易: {can_trade} ({details['reason']})")
    
    print(f"\n系统统计: {gating.get_statistics()}")
    print("\n✅ 波动率自适应门控测试完成")