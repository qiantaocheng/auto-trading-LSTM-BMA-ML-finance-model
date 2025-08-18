#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微结构信号模块 - 基于订单簿(LOB)的高频信号生成
实现OFI、队列不平衡、微价、盘口斜率、TSI、VPIN等核心微结构特征

基于学术研究：
- Cont-Kukanov-Stoikov: OFI主导短时价格变动
- Gatheral: 无动态套利与凹型冲击函数  
- Easley-Lopez-O'Hara: VPIN毒性检测
"""

import numpy as np
import pandas as pd
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LOBSnapshot:
    """订单簿快照"""
    timestamp: float
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    bid_levels: Dict[float, float] = None  # {price: size}
    ask_levels: Dict[float, float] = None
    
    def __post_init__(self):
        if self.bid_levels is None:
            self.bid_levels = {}
        if self.ask_levels is None:
            self.ask_levels = {}

class LOBState:
    """
    实时订单簿状态维护
    
    功能：
    1. 维护L1-L5盘口数据
    2. 记录成交流
    3. 计算VPIN毒性指标
    """
    
    def __init__(self, symbol: str, max_levels: int = 5, 
                 trade_window_secs: int = 5, vpin_bucket_vol: int = 10000):
        self.symbol = symbol
        self.max_levels = max_levels
        self.trade_window_secs = trade_window_secs
        self.vpin_bucket_vol = vpin_bucket_vol
        
        # 当前盘口状态
        self.bid_levels = {}  # {price: size}
        self.ask_levels = {}
        self.best_bid = None
        self.best_ask = None
        
        # 历史快照用于OFI计算
        self.prev_snapshot = None
        self.snapshots_history = deque(maxlen=100)
        
        # 成交流记录 (timestamp, signed_quantity)
        # 买单为正，卖单为负
        self.trades = deque()
        
        # VPIN计算相关
        self.vpin_bucket_vol_current = 0
        self.vpin_bucket_imbalance = 0
        self.vpin_series = deque(maxlen=1000)
        
        # 统计指标
        self.update_count = 0
        self.last_update_time = 0
        
        logger.info(f"LOB状态初始化完成: {symbol}, 最大档位: {max_levels}")
    
    def on_quote_update(self, side: str, price: float, size: int):
        """处理盘口更新"""
        self.last_update_time = time.time()
        self.update_count += 1
        
        # 选择对应的盘口字典
        levels = self.bid_levels if side.lower() == 'bid' else self.ask_levels
        
        if size == 0:
            # 删除该价位
            levels.pop(price, None)
        else:
            # 更新该价位数量
            levels[price] = size
        
        # 只保留前N档，按价格排序
        if side.lower() == 'bid':
            # 买盘按价格从高到低排序
            sorted_items = sorted(levels.items(), key=lambda x: x[0], reverse=True)
            self.bid_levels = dict(sorted_items[:self.max_levels])
            self.best_bid = max(self.bid_levels.keys()) if self.bid_levels else None
        else:
            # 卖盘按价格从低到高排序
            sorted_items = sorted(levels.items(), key=lambda x: x[0])
            self.ask_levels = dict(sorted_items[:self.max_levels])
            self.best_ask = min(self.ask_levels.keys()) if self.ask_levels else None
        
        # 保存当前快照
        self._save_snapshot()
    
    def on_trade(self, price: float, quantity: int, is_buy_aggressor: bool):
        """处理成交记录"""
        timestamp = time.time()
        signed_qty = quantity if is_buy_aggressor else -quantity
        
        self.trades.append((timestamp, signed_qty))
        
        # 清理过期交易记录
        cutoff_time = timestamp - self.trade_window_secs
        while self.trades and self.trades[0][0] < cutoff_time:
            self.trades.popleft()
        
        # 更新VPIN计算
        self._update_vpin(quantity, signed_qty)
    
    def _save_snapshot(self):
        """保存当前盘口快照"""
        snapshot = LOBSnapshot(
            timestamp=self.last_update_time,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            bid_size=self.bid_levels.get(self.best_bid, 0) if self.best_bid else 0,
            ask_size=self.ask_levels.get(self.best_ask, 0) if self.best_ask else 0,
            bid_levels=self.bid_levels.copy(),
            ask_levels=self.ask_levels.copy()
        )
        
        self.snapshots_history.append(snapshot)
        self.prev_snapshot = snapshot
    
    def _update_vpin(self, volume: int, signed_volume: int):
        """更新VPIN毒性指标"""
        self.vpin_bucket_vol_current += volume
        self.vpin_bucket_imbalance += signed_volume
        
        # 当体积桶满时计算VPIN
        if self.vpin_bucket_vol_current >= self.vpin_bucket_vol:
            if self.vpin_bucket_vol_current > 0:
                vpin_value = abs(self.vpin_bucket_imbalance) / self.vpin_bucket_vol_current
                self.vpin_series.append(vpin_value)
            
            # 重置桶
            self.vpin_bucket_vol_current = 0
            self.vpin_bucket_imbalance = 0

class MicrostructureFeatures:
    """
    微结构特征计算器
    
    实现关键特征：
    1. OFI (Order Flow Imbalance) - Cont-Kukanov-Stoikov
    2. QI (Queue Imbalance) & Microprice
    3. 盘口斜率 (Book Slope)
    4. TSI (Trade Sign Imbalance)
    5. VPIN (Volume-Synchronized Probability of Informed Trading)
    """
    
    @staticmethod
    def compute_ofi_l1(prev_snapshot: LOBSnapshot, curr_snapshot: LOBSnapshot) -> float:
        """
        计算L1订单流不平衡 (OFI)
        
        基于Cont-Kukanov-Stoikov公式：
        OFI_t = Δq^bid_t * I(Δp^bid >= 0) - Δq^ask_t * I(Δp^ask <= 0)
        """
        if not prev_snapshot or not curr_snapshot:
            return 0.0
            
        ofi = 0.0
        
        # 买盘OFI计算
        if prev_snapshot.best_bid is not None and curr_snapshot.best_bid is not None:
            if curr_snapshot.best_bid > prev_snapshot.best_bid:
                # 买价上升，需求增加
                ofi += curr_snapshot.bid_size
            elif curr_snapshot.best_bid < prev_snapshot.best_bid:
                # 买价下降，需求减少
                ofi -= prev_snapshot.bid_size
            else:
                # 买价不变，数量变化
                ofi += (curr_snapshot.bid_size - prev_snapshot.bid_size)
        
        # 卖盘OFI计算
        if prev_snapshot.best_ask is not None and curr_snapshot.best_ask is not None:
            if curr_snapshot.best_ask < prev_snapshot.best_ask:
                # 卖价下降，供给增加
                ofi -= curr_snapshot.ask_size
            elif curr_snapshot.best_ask > prev_snapshot.best_ask:
                # 卖价上升，供给减少
                ofi += prev_snapshot.ask_size
            else:
                # 卖价不变，数量变化
                ofi -= (curr_snapshot.ask_size - prev_snapshot.ask_size)
        
        return float(ofi)
    
    @staticmethod
    def compute_qi_and_microprice(lob: LOBState) -> Dict[str, float]:
        """
        计算队列不平衡(QI)和微价(Microprice)
        
        QI = (Q_bid - Q_ask) / (Q_bid + Q_ask)
        Microprice = (ask * Q_bid + bid * Q_ask) / (Q_bid + Q_ask)
        """
        if not lob.best_bid or not lob.best_ask:
            return {"QI": 0.0, "microprice": 0.0, "mid_price": 0.0, "micro_dev_bps": 0.0}
        
        bid_qty = lob.bid_levels.get(lob.best_bid, 0)
        ask_qty = lob.ask_levels.get(lob.best_ask, 0)
        
        if bid_qty + ask_qty == 0:
            return {"QI": 0.0, "microprice": 0.0, "mid_price": 0.0, "micro_dev_bps": 0.0}
        
        # 队列不平衡
        qi = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        
        # 微价
        microprice = (lob.best_ask * bid_qty + lob.best_bid * ask_qty) / (bid_qty + ask_qty)
        
        # 中间价
        mid_price = (lob.best_bid + lob.best_ask) / 2
        
        # 微价偏离度(bps)
        micro_dev_bps = (microprice - mid_price) / mid_price * 10000
        
        return {
            "QI": float(qi),
            "microprice": float(microprice),
            "mid_price": float(mid_price),
            "micro_dev_bps": float(micro_dev_bps)
        }
    
    @staticmethod
    def compute_book_slope(lob: LOBState, levels: int = 5) -> Dict[str, float]:
        """
        计算盘口斜率 - 深度-价格关系的线性拟合
        
        斜率越陡 = 流动性越差 = 冲击成本越高
        """
        def calc_slope(price_size_pairs):
            if len(price_size_pairs) < 2:
                return 0.0
            
            prices = np.array([p for p, s in price_size_pairs])
            cumulative_sizes = np.cumsum([s for p, s in price_size_pairs])
            
            # 使用numpy进行线性拟合
            try:
                slope = np.polyfit(prices, cumulative_sizes, 1)[0]
                return float(slope)
            except:
                return 0.0
        
        # 买盘斜率 (价格从高到低)
        bid_pairs = sorted(lob.bid_levels.items(), reverse=True)[:levels]
        slope_bid = calc_slope(bid_pairs)
        
        # 卖盘斜率 (价格从低到高)
        ask_pairs = sorted(lob.ask_levels.items())[:levels]
        slope_ask = calc_slope(ask_pairs)
        
        return {
            "slope_bid": slope_bid,
            "slope_ask": slope_ask,
            "slope_avg": (abs(slope_bid) + abs(slope_ask)) / 2
        }
    
    @staticmethod
    def compute_tsi(lob: LOBState) -> Dict[str, float]:
        """
        计算成交符号不平衡 (Trade Sign Imbalance)
        
        TSI = sum(signed_volume) / sum(|volume|)
        正值表示买盘主导，负值表示卖盘主导
        """
        if not lob.trades:
            return {"TSI": 0.0, "trade_count": 0.0, "total_volume": 0.0}
        
        signed_volumes = [signed_vol for _, signed_vol in lob.trades]
        total_signed = sum(signed_volumes)
        total_volume = sum(abs(vol) for vol in signed_volumes)
        
        tsi = total_signed / total_volume if total_volume > 0 else 0.0
        
        return {
            "TSI": float(tsi),
            "trade_count": float(len(lob.trades)),
            "total_volume": float(total_volume)
        }
    
    @staticmethod
    def compute_vpin(lob: LOBState) -> Dict[str, float]:
        """
        计算VPIN毒性指标
        
        VPIN = 平均|买卖不平衡|/成交量
        高VPIN表示信息型交易增加，做市风险上升
        """
        if len(lob.vpin_series) < 5:
            return {"VPIN": 0.0, "VPIN_percentile": 0.0, "buckets_count": len(lob.vpin_series)}
        
        recent_vpin = np.mean(list(lob.vpin_series)[-10:])  # 最近10个桶的均值
        vpin_percentile = np.percentile(list(lob.vpin_series), 90)  # 90分位数
        
        return {
            "VPIN": float(recent_vpin),
            "VPIN_percentile": float(vpin_percentile),
            "buckets_count": len(lob.vpin_series)
        }

class MicrostructureSignalEngine:
    """
    微结构信号引擎 - 整合所有微结构特征
    """
    
    def __init__(self):
        self.lob_states = {}  # symbol -> LOBState
        self.features_cache = {}  # symbol -> latest features
        self.feature_history = defaultdict(list)  # symbol -> [features_dict, ...]
        
        logger.info("微结构信号引擎初始化完成")
    
    def add_symbol(self, symbol: str, **kwargs):
        """添加要跟踪的股票"""
        if symbol not in self.lob_states:
            self.lob_states[symbol] = LOBState(symbol, **kwargs)
            logger.info(f"添加微结构跟踪: {symbol}")
    
    def on_quote_update(self, symbol: str, side: str, price: float, size: int):
        """处理盘口更新"""
        if symbol not in self.lob_states:
            self.add_symbol(symbol)
        
        self.lob_states[symbol].on_quote_update(side, price, size)
        
        # 实时计算特征
        self._compute_features(symbol)
    
    def on_trade(self, symbol: str, price: float, quantity: int, is_buy_aggressor: bool):
        """处理成交记录"""
        if symbol not in self.lob_states:
            self.add_symbol(symbol)
        
        self.lob_states[symbol].on_trade(price, quantity, is_buy_aggressor)
        
        # 实时计算特征
        self._compute_features(symbol)
    
    def _compute_features(self, symbol: str):
        """计算并缓存微结构特征"""
        lob = self.lob_states[symbol]
        
        # 计算所有特征
        features = {
            "timestamp": time.time(),
            "symbol": symbol,
        }
        
        # OFI特征
        if len(lob.snapshots_history) >= 2:
            prev_snap = lob.snapshots_history[-2]
            curr_snap = lob.snapshots_history[-1]
            ofi = MicrostructureFeatures.compute_ofi_l1(prev_snap, curr_snap)
            features["OFI"] = ofi
        else:
            features["OFI"] = 0.0
        
        # 队列不平衡和微价
        qi_micro = MicrostructureFeatures.compute_qi_and_microprice(lob)
        features.update(qi_micro)
        
        # 盘口斜率
        slope_data = MicrostructureFeatures.compute_book_slope(lob)
        features.update(slope_data)
        
        # 成交符号不平衡
        tsi_data = MicrostructureFeatures.compute_tsi(lob)
        features.update(tsi_data)
        
        # VPIN毒性
        vpin_data = MicrostructureFeatures.compute_vpin(lob)
        features.update(vpin_data)
        
        # 计算点差
        if lob.best_bid and lob.best_ask:
            spread_bps = (lob.best_ask - lob.best_bid) / ((lob.best_ask + lob.best_bid) / 2) * 10000
            features["spread_bps"] = spread_bps
        else:
            features["spread_bps"] = 999.0  # 异常高点差
        
        # 缓存特征
        self.features_cache[symbol] = features
        self.feature_history[symbol].append(features)
        
        # 限制历史长度
        if len(self.feature_history[symbol]) > 1000:
            self.feature_history[symbol] = self.feature_history[symbol][-500:]
    
    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """获取最新的微结构特征"""
        return self.features_cache.get(symbol)
    
    def get_feature_history(self, symbol: str, lookback: int = 100) -> List[Dict[str, float]]:
        """获取特征历史"""
        history = self.feature_history.get(symbol, [])
        return history[-lookback:] if history else []
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取引擎状态摘要"""
        summary = {
            "tracked_symbols": len(self.lob_states),
            "symbols": list(self.lob_states.keys()),
            "total_updates": sum(lob.update_count for lob in self.lob_states.values()),
            "features_cache_size": len(self.features_cache)
        }
        
        # 每个股票的状态
        for symbol, lob in self.lob_states.items():
            summary[f"{symbol}_updates"] = lob.update_count
            summary[f"{symbol}_trades"] = len(lob.trades)
            summary[f"{symbol}_vpin_buckets"] = len(lob.vpin_series)
        
        return summary

# 创建全局实例
_global_microstructure_engine = None

def get_microstructure_engine() -> MicrostructureSignalEngine:
    """获取全局微结构信号引擎"""
    global _global_microstructure_engine
    if _global_microstructure_engine is None:
        _global_microstructure_engine = MicrostructureSignalEngine()
    return _global_microstructure_engine

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_microstructure_engine()
    
    # 模拟一些数据
    symbol = "AAPL"
    engine.add_symbol(symbol)
    
    # 模拟盘口更新
    engine.on_quote_update(symbol, "bid", 150.00, 1000)
    engine.on_quote_update(symbol, "ask", 150.05, 800)
    engine.on_quote_update(symbol, "bid", 150.01, 1200)
    
    # 模拟成交
    engine.on_trade(symbol, 150.05, 100, True)  # 买单主动成交
    engine.on_trade(symbol, 150.00, 200, False)  # 卖单主动成交
    
    # 获取特征
    features = engine.get_latest_features(symbol)
    print("微结构特征:", features)
    
    # 状态摘要
    status = engine.get_status_summary()
    print("引擎状态:", status)