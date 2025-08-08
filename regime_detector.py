#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市况判别器 (RegimeDetector)
实现盘前四象限判别功能

核心功能:
1. 盘前批量读取60日行情数据
2. 计算全局ADX/ATR/SMA指标
3. 基于四象限进行市况分类
4. 为因子平衡策略提供市况判断

Authors: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    """市况判别器"""
    
    def __init__(self, 
                 lookback_days: int = 60,
                 adx_threshold: float = 25.0,
                 atr_ratio_threshold: float = 0.8,
                 cache_dir: str = "regime_cache"):
        """
        初始化市况判别器
        
        Args:
            lookback_days: 回看天数，默认60天
            adx_threshold: ADX趋势判断阈值，默认25
            atr_ratio_threshold: ATR/SMA比值阈值，默认0.8
            cache_dir: 缓存目录
        """
        self.lookback_days = lookback_days
        self.adx_threshold = adx_threshold
        self.atr_ratio_threshold = atr_ratio_threshold
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 四象限配比字典
        self.regime_allocation = {
            (False, False): (0.7, 0.3),  # 无趋势+低波动：均值回归70%，趋势跟踪30%
            (False, True):  (0.6, 0.4),  # 无趋势+高波动：均值回归60%，趋势跟踪40%
            (True, False):  (0.4, 0.6),  # 有趋势+低波动：均值回归40%，趋势跟踪60%
            (True, True):   (0.3, 0.7),  # 有趋势+高波动：均值回归30%，趋势跟踪70%
        }
        
        # 标准股票池（用于全局指标计算）
        self.benchmark_symbols = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'NFLX', 'CRM', 'ADBE', 'ORCL', 'IBM', 'INTC', 'AMD'
        ]
        
        # 当前市况状态
        self.current_regime = None
        self.last_update = None
        
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """
        计算ADX指标
        
        Args:
            high: 最高价序列
            low: 最低价序列  
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            ADX值
        """
        try:
            # 计算True Range (TR)
            hl = high - low
            hc = np.abs(high - close.shift(1))
            lc = np.abs(low - close.shift(1))
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            
            # 计算Directional Movement (DM)
            dm_plus = high.diff()
            dm_minus = low.diff() * -1
            
            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            
            # 当+DM > -DM时，-DM = 0，反之亦然
            dm_plus[(dm_plus <= dm_minus)] = 0
            dm_minus[(dm_minus <= dm_plus)] = 0
            
            # 计算平滑的TR和DM
            tr_smooth = tr.ewm(span=period).mean()
            dm_plus_smooth = dm_plus.ewm(span=period).mean()
            dm_minus_smooth = dm_minus.ewm(span=period).mean()
            
            # 计算DI
            di_plus = (dm_plus_smooth / tr_smooth) * 100
            di_minus = (dm_minus_smooth / tr_smooth) * 100
            
            # 计算DX
            dx = (np.abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
            
            # 计算ADX
            adx = dx.ewm(span=period).mean()
            
            return adx.iloc[-1] if not adx.empty else 0.0
            
        except Exception as e:
            self.logger.warning(f"ADX计算错误: {e}")
            return 0.0
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """
        计算ATR指标
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            ATR值
        """
        try:
            # 计算True Range
            hl = high - low
            hc = np.abs(high - close.shift(1))
            lc = np.abs(low - close.shift(1))
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            
            # 计算ATR
            atr = tr.ewm(span=period).mean()
            
            return atr.iloc[-1] if not atr.empty else 0.0
            
        except Exception as e:
            self.logger.warning(f"ATR计算错误: {e}")
            return 0.0
    
    def calculate_sma(self, close: pd.Series, period: int = 50) -> float:
        """
        计算SMA指标
        
        Args:
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            SMA值
        """
        try:
            sma = close.rolling(window=period).mean()
            return sma.iloc[-1] if not sma.empty else 0.0
        except Exception as e:
            self.logger.warning(f"SMA计算错误: {e}")
            return 0.0
    
    def fetch_market_data(self, symbols: List[str], days: int = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取市场数据
        
        Args:
            symbols: 股票代码列表
            days: 获取天数，默认使用lookback_days
            
        Returns:
            {symbol: DataFrame} 格式的数据字典
        """
        if days is None:
            days = self.lookback_days
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # 多取一些数据以确保足够
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # 检查缓存
                cache_file = os.path.join(self.cache_dir, f"{symbol}_{end_date.strftime('%Y%m%d')}.csv")
                
                if os.path.exists(cache_file):
                    # 从缓存读取
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    self.logger.debug(f"从缓存读取 {symbol} 数据")
                else:
                    # 从yfinance获取
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        # 保存到缓存
                        df.to_csv(cache_file)
                        self.logger.debug(f"获取并缓存 {symbol} 数据")
                    else:
                        self.logger.warning(f"无法获取 {symbol} 数据")
                        continue
                
                # 确保有足够的数据
                if len(df) >= days:
                    market_data[symbol] = df.tail(days)
                else:
                    self.logger.warning(f"{symbol} 数据不足，需要{days}天，实际{len(df)}天")
                    
            except Exception as e:
                self.logger.error(f"获取 {symbol} 数据失败: {e}")
                continue
        
        return market_data
    
    def calculate_global_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算全局技术指标
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            全局指标字典
        """
        adx_values = []
        atr_values = []
        sma_values = []
        atr_ratio_values = []
        
        for symbol, df in market_data.items():
            try:
                if len(df) < 50:  # 确保有足够数据计算指标
                    continue
                
                high = df['High']
                low = df['Low']
                close = df['Close']
                
                # 计算ADX
                adx = self.calculate_adx(high, low, close)
                if adx > 0:
                    adx_values.append(adx)
                
                # 计算ATR和SMA
                atr = self.calculate_atr(high, low, close)
                sma = self.calculate_sma(close, 50)
                
                if atr > 0 and sma > 0:
                    atr_values.append(atr)
                    sma_values.append(sma)
                    atr_ratio_values.append(atr / sma)
                
            except Exception as e:
                self.logger.warning(f"计算 {symbol} 指标失败: {e}")
                continue
        
        # 计算全局指标（取平均值）
        global_indicators = {
            'ADX_val': np.mean(adx_values) if adx_values else 0.0,
            'ATR_val': np.mean(atr_values) if atr_values else 0.0,
            'SMA50_val': np.mean(sma_values) if sma_values else 0.0,
            'ATR_ratio': np.mean(atr_ratio_values) if atr_ratio_values else 0.0,
            'sample_size': len(market_data)
        }
        
        return global_indicators
    
    def detect_regime(self, force_update: bool = False) -> Dict:
        """
        检测当前市况
        
        Args:
            force_update: 是否强制更新
            
        Returns:
            市况检测结果
        """
        try:
            # 检查是否需要更新
            now = datetime.now()
            if not force_update and self.last_update:
                time_diff = now - self.last_update
                if time_diff.total_seconds() < 3600:  # 1小时内不重复更新
                    return self.current_regime
            
            self.logger.info(f"开始检测市况 (回看{self.lookback_days}天)")
            
            # 1. 获取市场数据
            market_data = self.fetch_market_data(self.benchmark_symbols)
            
            if not market_data:
                raise ValueError("无法获取市场数据")
            
            # 2. 计算全局指标
            indicators = self.calculate_global_indicators(market_data)
            
            # 3. 四象限判别
            is_trend = indicators['ADX_val'] >= self.adx_threshold
            is_high_vol = indicators['ATR_ratio'] >= self.atr_ratio_threshold
            
            # 4. 获取配比
            allocation = self.regime_allocation.get((is_trend, is_high_vol), (0.5, 0.5))
            
            # 5. 生成结果
            regime_result = {
                'timestamp': now.isoformat(),
                'regime_type': {
                    'is_trend': is_trend,
                    'is_high_vol': is_high_vol,
                    'description': self._get_regime_description(is_trend, is_high_vol)
                },
                'indicators': indicators,
                'allocation': {
                    'mean_reversion_weight': allocation[0],
                    'trend_following_weight': allocation[1]
                },
                'thresholds': {
                    'adx_threshold': self.adx_threshold,
                    'atr_ratio_threshold': self.atr_ratio_threshold
                }
            }
            
            # 6. 保存结果
            self.current_regime = regime_result
            self.last_update = now
            
            # 保存到文件
            self._save_regime_result(regime_result)
            
            self.logger.info(f"市况检测完成: {regime_result['regime_type']['description']}")
            self.logger.info(f"配比 - 均值回归: {allocation[0]:.1%}, 趋势跟踪: {allocation[1]:.1%}")
            
            return regime_result
            
        except Exception as e:
            self.logger.error(f"市况检测失败: {e}")
            # 返回默认配比
            return {
                'timestamp': datetime.now().isoformat(),
                'regime_type': {
                    'is_trend': False,
                    'is_high_vol': False,
                    'description': '默认市况（数据获取失败）'
                },
                'indicators': {},
                'allocation': {
                    'mean_reversion_weight': 0.5,
                    'trend_following_weight': 0.5
                },
                'error': str(e)
            }
    
    def _get_regime_description(self, is_trend: bool, is_high_vol: bool) -> str:
        """获取市况描述"""
        if is_trend and is_high_vol:
            return "趋势市+高波动"
        elif is_trend and not is_high_vol:
            return "趋势市+低波动"
        elif not is_trend and is_high_vol:
            return "震荡市+高波动"
        else:
            return "震荡市+低波动"
    
    def _save_regime_result(self, result: Dict):
        """保存市况检测结果"""
        try:
            # 保存当前结果
            current_file = os.path.join(self.cache_dir, "current_regime.json")
            with open(current_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 保存历史记录
            history_file = os.path.join(self.cache_dir, "regime_history.json")
            
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except:
                    history = []
            
            history.append(result)
            
            # 只保留最近100条记录
            if len(history) > 100:
                history = history[-100:]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"保存市况结果失败: {e}")
    
    def get_current_allocation(self) -> Tuple[float, float]:
        """
        获取当前配比
        
        Returns:
            (均值回归权重, 趋势跟踪权重)
        """
        if self.current_regime:
            allocation = self.current_regime['allocation']
            return allocation['mean_reversion_weight'], allocation['trend_following_weight']
        else:
            # 默认配比
            return 0.5, 0.5
    
    def load_latest_regime(self) -> Optional[Dict]:
        """从缓存加载最新市况"""
        try:
            current_file = os.path.join(self.cache_dir, "current_regime.json")
            if os.path.exists(current_file):
                with open(current_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.current_regime = result
                    return result
        except Exception as e:
            self.logger.warning(f"加载市况缓存失败: {e}")
        return None

def main():
    """测试市况检测器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    detector = RegimeDetector()
    
    print("🔍 开始市况检测...")
    result = detector.detect_regime(force_update=True)
    
    print("\n📊 市况检测结果:")
    print(f"市况类型: {result['regime_type']['description']}")
    print(f"趋势判断: {'是' if result['regime_type']['is_trend'] else '否'}")
    print(f"高波动: {'是' if result['regime_type']['is_high_vol'] else '否'}")
    
    if 'indicators' in result and result['indicators']:
        indicators = result['indicators']
        print(f"\n📈 技术指标:")
        print(f"ADX: {indicators.get('ADX_val', 0):.2f}")
        print(f"ATR比率: {indicators.get('ATR_ratio', 0):.4f}")
        print(f"样本数量: {indicators.get('sample_size', 0)}")
    
    allocation = result['allocation']
    print(f"\n💰 建议配比:")
    print(f"均值回归策略: {allocation['mean_reversion_weight']:.1%}")
    print(f"趋势跟踪策略: {allocation['trend_following_weight']:.1%}")

if __name__ == "__main__":
    main()