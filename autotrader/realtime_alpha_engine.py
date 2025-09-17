#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时Alpha决策引擎 - 基于微结构信号的"α > 成本"门槛决策

核心原理:
1. 微结构信号合成短期Alpha预测
2. "能赚再上"门槛: |predicted_alpha| > total_cost 才允许交易
3. 动态参与率调整基于流动性和毒性指标
4. 与OOF校准和冲击成本模型集成

学术支撑:
- Cont-Kukanov-Stoikov: OFI线性预测短期收益
- Almgren-Chriss: 成本感知的最优执行
- Gatheral: 凹函数冲击与无套利条件
"""

import numpy as np
import pandas as pd
import logging
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass

from .microstructure_signals import MicrostructureSignalEngine, get_microstructure_engine
from .impact_model import ImpactModel, get_impact_model
from .oof_calibration import get_oof_calibrator

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """交易决策结果"""
    symbol: str
    timestamp: float
    
    # 信号与预测
    raw_alpha_bps: float              # 原始Alpha预测(bps)
    calibrated_alpha_bps: float       # OOF校准后Alpha(bps)
    confidence: float                 # 信号置信度
    
    # 成本分析
    spread_bps: float                 # 点差成本
    impact_bps: float                 # 市场冲击成本
    total_cost_bps: float            # 总交易成本
    
    # 决策结果
    is_tradable: bool                 # 是否可交易 (α > 成本)
    recommended_side: str             # 推荐方向 BUY/SELL/HOLD
    optimal_pov: float               # 最优参与率
    
    # 风险指标
    vpin_risk_level: str             # VPIN风险等级 LOW/MEDIUM/HIGH
    liquidity_risk_level: str        # 流动性风险等级
    
    # 调试信息
    features: Dict[str, float]       # 微结构特征
    decision_reason: str             # 决策原因

class MicrostructureAlphaModel:
    """
    微结构Alpha模型
    
    基于多种微结构信号合成短期Alpha预测:
    1. OFI (订单流不平衡) - 主导因子
    2. 微价偏离 - 均值回归
    3. 队列不平衡 - 方向预测
    4. 成交流向 - 动量确认
    5. VPIN毒性 - 风险调整
    """
    
    def __init__(self):
        # 模型权重 (可通过回测优化)
        self.feature_weights = {
            'OFI': 0.6,                    # OFI权重最高
            'micro_deviation': 0.4,        # 微价均值回归
            'queue_imbalance': 0.3,        # 队列不平衡门控
            'trade_sign': 0.2,             # 成交流向确认
            'momentum_consistency': 0.15    # 多信号一致性
        }
        
        # 信号标准化参数
        self.normalization_params = {
            'OFI_scale': 1e4,              # OFI标准化比例
            'micro_dev_cap': 5.0,          # 微价偏离截断(bps)
            'qi_threshold': 0.05,          # QI门控阈值
            'tsi_threshold': 0.1           # TSI门控阈值
        }
        
        # 历史信号用于动量计算
        self.signal_history = defaultdict(lambda: deque(maxlen=20))
        
        logger.info("微结构Alpha模型初始化完成")
    
    def compute_alpha_bps(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        计算短期Alpha预测 (bps) 和置信度
        
        核心逻辑:
        1. OFI → 线性价格影响 (主要驱动力)
        2. 微价偏离 → 均值回归力量
        3. 队列/成交不平衡 → 方向和强度门控
        4. 多信号一致性 → 提升置信度
        """
        alpha = 0.0
        confidence = 0.5  # 基础置信度
        
        # 1. OFI驱动的线性价格预测
        ofi = features.get('OFI', 0.0)
        ofi_normalized = np.tanh(ofi / self.normalization_params['OFI_scale'])
        ofi_alpha = ofi_normalized * 10000 * self.feature_weights['OFI']  # 转换为bps
        alpha += ofi_alpha
        
        # 2. 微价偏离的均值回归
        micro_dev = features.get('micro_dev_bps', 0.0)
        micro_dev_capped = np.clip(micro_dev, -self.normalization_params['micro_dev_cap'], 
                                  self.normalization_params['micro_dev_cap'])
        # 微价 > 中间价 → 预期向下回归 (负alpha)
        micro_alpha = -micro_dev_capped * self.feature_weights['micro_deviation']
        alpha += micro_alpha
        
        # 3. 队列不平衡门控
        qi = features.get('QI', 0.0)
        qi_gate = 1.0 if abs(qi) > self.normalization_params['qi_threshold'] else 0.5
        alpha *= qi_gate
        
        # 方向一致性检查
        qi_direction = np.sign(qi)
        alpha_direction = np.sign(alpha)
        if qi_direction * alpha_direction > 0:
            confidence += 0.1  # 方向一致，增加置信度
        
        # 4. 成交流向确认
        tsi = features.get('TSI', 0.0)
        if abs(tsi) > self.normalization_params['tsi_threshold']:
            tsi_direction = np.sign(tsi)
            if tsi_direction * alpha_direction > 0:
                confidence += 0.15  # 成交流向确认，置信度提升
                alpha *= 1.1        # 信号增强
            else:
                confidence -= 0.1   # 方向冲突，置信度下降
                alpha *= 0.8        # 信号减弱
        
        # 5. VPIN毒性调整
        vpin = features.get('VPIN', 0.0)
        if vpin > 0.7:
            alpha *= 0.5        # 高毒性环境，信号衰减
            confidence *= 0.7
        elif vpin > 0.5:
            alpha *= 0.8
            confidence *= 0.9
        
        # 6. 盘口斜率调整 (流动性指标)
        slope_avg = features.get('slope_avg', 0.0)
        if slope_avg > 0:  # 斜率陡峭 = 流动性差
            liquidity_penalty = min(1.0, slope_avg / 1000)  # 标准化
            alpha *= (1 - liquidity_penalty * 0.3)
            confidence *= (1 - liquidity_penalty * 0.2)
        
        # 7. 多周期一致性 (使用历史信号)
        self.signal_history[features.get('symbol', 'UNKNOWN')].append(alpha)
        if len(self.signal_history[features.get('symbol', 'UNKNOWN')]) >= 3:
            recent_signals = list(self.signal_history[features.get('symbol', 'UNKNOWN')])[-3:]
            direction_consistency = np.mean([np.sign(s) for s in recent_signals])
            if abs(direction_consistency) > 0.6:  # 方向一致性 > 60%
                confidence += 0.1
                alpha *= 1.05
        
        # 限制输出范围
        alpha = np.clip(alpha, -1000, 1000)  # ±1000bps
        confidence = np.clip(confidence, 0.01, 0.99)
        
        return float(alpha), float(confidence)

class RealtimeAlphaEngine:
    """
    实时Alpha决策引擎
    
    核心功能:
    1. 整合微结构信号和OOF校准
    2. 执行"α > 成本"门槛判断
    3. 动态调整参与率和执行策略
    4. 风险门控和异常检测
    """
    
    def __init__(self, max_pov: float = 0.15):
        self.max_pov = max_pov
        
        # 核心组件
        self.microstructure_engine = get_microstructure_engine()
        self.impact_model = get_impact_model()
        self.oof_calibrator = get_oof_calibrator()
        self.alpha_model = MicrostructureAlphaModel()
        
        # 决策历史
        self.decision_history = defaultdict(lambda: deque(maxlen=100))
        
        # 性能统计
        self.stats = {
            'total_decisions': 0,
            'tradable_decisions': 0,
            'alpha_gt_cost_ratio': 0.0,
            'avg_alpha_bps': 0.0,
            'avg_cost_bps': 0.0
        }
        
        logger.info(f"实时Alpha决策引擎初始化完成, 最大参与率: {max_pov}")
    
    def make_trading_decision(self, symbol: str, market_context: Dict[str, Any] = None) -> TradingDecision:
        """
        生成交易决策
        
        完整流程:
        1. 获取微结构特征
        2. 计算原始Alpha
        3. OOF校准
        4. 估算交易成本  
        5. 应用"α > 成本"门槛
        6. 优化参与率
        7. 风险检查
        """
        timestamp = time.time()
        market_context = market_context or {}
        
        try:
            # 1. 获取微结构特征（增强容错处理）
            features = self.microstructure_engine.get_latest_features(symbol)
            if not features:
                # 尝试获取历史特征作为降级
                features = self._get_fallback_features(symbol)
                if not features:
                    return self._create_hold_decision(symbol, timestamp, "无微结构数据且无历史特征")
                else:
                    self.logger.warning(f"{symbol}: 使用历史特征降级处理")

            # 验证关键特征完整性
            if not self._validate_features_completeness(features):
                # 尝试补充缺失特征
                features = self._supplement_missing_features(symbol, features)
                if not self._validate_features_completeness(features):
                    return self._create_hold_decision(symbol, timestamp, "关键特征数据不完整")

            features['symbol'] = symbol  # 添加symbol到特征中
            
            # 2. 计算原始Alpha预测
            raw_alpha_bps, raw_confidence = self.alpha_model.compute_alpha_bps(features)
            
            if abs(raw_alpha_bps) < 0.1:  # Alpha太小，不值得交易
                return self._create_hold_decision(symbol, timestamp, "Alpha信号过弱")
            
            # 3. OOF校准 (如果可用)
            try:
                # 使用微价作为参考价格
                reference_price = features.get('mid_price', 100.0)
                calibrated_alpha_bps, calibrated_confidence = self.oof_calibrator.calibrate_prediction(
                    raw_alpha_bps / 10000, raw_confidence  # OOF预期的是收益率而非bps
                )
                calibrated_alpha_bps = abs(calibrated_alpha_bps)  # 确保为正值，方向由原始信号决定
                if raw_alpha_bps < 0:
                    calibrated_alpha_bps = -calibrated_alpha_bps
            except Exception as e:
                logger.warning(f"OOF校准失败，使用原始信号: {e}")
                calibrated_alpha_bps = raw_alpha_bps
                calibrated_confidence = raw_confidence
            
            # 4. 估算交易成本
            spread_bps = features.get('spread_bps', 5.0)
            vpin = features.get('VPIN', 0.0)
            
            # 初始参与率估算 (基于信号强度)
            signal_strength = abs(calibrated_alpha_bps) / 1000  # 标准化到0-1
            base_pov = min(self.max_pov, max(0.01, signal_strength * 0.1))
            
            # 基于流动性调整参与率
            slope_avg = features.get('slope_avg', 0.0)
            if slope_avg > 0:
                liquidity_adjustment = np.clip(1.0 / (1 + slope_avg / 500), 0.3, 1.0)
                base_pov *= liquidity_adjustment
            
            # 估算总成本
            total_cost_bps = self.impact_model.estimate_cost_bps(
                symbol=symbol,
                pov=base_pov,
                spread_bps=spread_bps,
                vpin=vpin
            )
            
            # 5. 核心决策: α > 成本 门槛
            is_tradable = abs(calibrated_alpha_bps) > total_cost_bps
            
            # 6. 优化参与率 (如果可交易)
            if is_tradable:
                # 为目标alpha找最优POV，确保成本不超过alpha的80%
                target_cost = abs(calibrated_alpha_bps) * 0.8
                optimal_pov = self.impact_model.optimize_pov_for_cost_target(
                    symbol=symbol,
                    target_cost_bps=target_cost,
                    spread_bps=spread_bps,
                    vpin=vpin
                )
                optimal_pov = min(optimal_pov, self.max_pov)
                
                # 重新计算基于最优POV的成本
                final_cost_bps = self.impact_model.estimate_cost_bps(
                    symbol, optimal_pov, spread_bps, vpin
                )
            else:
                optimal_pov = 0.0
                final_cost_bps = total_cost_bps
            
            # 7. 风险等级评估
            vpin_risk = self._assess_vpin_risk(vpin)
            liquidity_risk = self._assess_liquidity_risk(features)
            
            # 8. 决策方向
            if is_tradable:
                recommended_side = "BUY" if calibrated_alpha_bps > 0 else "SELL"
                decision_reason = f"α({calibrated_alpha_bps:.1f}) > 成本({final_cost_bps:.1f})"
            else:
                recommended_side = "HOLD"
                decision_reason = f"α({calibrated_alpha_bps:.1f}) ≤ 成本({final_cost_bps:.1f})"
            
            # 构建决策对象
            decision = TradingDecision(
                symbol=symbol,
                timestamp=timestamp,
                raw_alpha_bps=raw_alpha_bps,
                calibrated_alpha_bps=calibrated_alpha_bps,
                confidence=calibrated_confidence,
                spread_bps=spread_bps,
                impact_bps=final_cost_bps - spread_bps - self.impact_model.fixed_fee_bps,
                total_cost_bps=final_cost_bps,
                is_tradable=is_tradable,
                recommended_side=recommended_side,
                optimal_pov=optimal_pov,
                vpin_risk_level=vpin_risk,
                liquidity_risk_level=liquidity_risk,
                features=features,
                decision_reason=decision_reason
            )
            
            # 更新统计
            self._update_stats(decision)
            
            # 记录决策历史
            self.decision_history[symbol].append(decision)
            
            logger.debug(f"{symbol}决策: {decision_reason}, POV={optimal_pov:.3f}, "
                        f"VPIN={vpin_risk}, 流动性={liquidity_risk}")
            
            return decision
            
        except Exception as e:
            logger.error(f"决策生成失败 {symbol}: {e}")
            return self._create_hold_decision(symbol, timestamp, f"决策异常: {e}")

    def _validate_features_completeness(self, features: Dict[str, float]) -> bool:
        """验证关键特征完整性"""
        required_features = [
            'OFI', 'QI', 'mid_price', 'spread_bps', 'VPIN',
            'TSI', 'slope_avg', 'micro_dev_bps'
        ]

        missing_features = []
        invalid_features = []

        for feature in required_features:
            if feature not in features:
                missing_features.append(feature)
            else:
                value = features[feature]
                # 检查数值有效性
                if value is None or (isinstance(value, float) and
                    (math.isnan(value) or math.isinf(value))):
                    invalid_features.append(feature)

        if missing_features:
            logger.warning(f"缺失特征: {missing_features}")
        if invalid_features:
            logger.warning(f"无效特征值: {invalid_features}")

        return len(missing_features) == 0 and len(invalid_features) == 0

    def _get_fallback_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """获取降级特征数据"""
        try:
            # 尝试从历史特征缓存获取
            history = self.microstructure_engine.get_feature_history(symbol, lookback=10)
            if history and len(history) > 0:
                # 使用最近的有效特征
                for hist_features in reversed(history):
                    if self._validate_features_completeness(hist_features):
                        logger.info(f"{symbol}: 使用历史特征 (age={time.time() - hist_features.get('timestamp', 0):.1f}s)")
                        return hist_features

                # 如果没有完全有效的历史特征，使用最近的并补充
                latest_hist = history[-1]
                if latest_hist:
                    return self._supplement_missing_features(symbol, latest_hist)

            # 如果没有历史数据，生成基础特征
            return self._generate_basic_features(symbol)

        except Exception as e:
            logger.error(f"获取降级特征失败 {symbol}: {e}")
            return None

    def _supplement_missing_features(self, symbol: str, features: Dict[str, float]) -> Dict[str, float]:
        """补充缺失的特征"""
        supplemented_features = features.copy()

        try:
            # 获取基础市场数据
            current_price = features.get('mid_price', 100.0)

            # 补充缺失的关键特征
            if 'OFI' not in supplemented_features or supplemented_features.get('OFI') is None:
                # 使用历史OFI均值或0
                hist_ofi = self._get_historical_feature_mean(symbol, 'OFI')
                supplemented_features['OFI'] = hist_ofi or 0.0
                logger.debug(f"{symbol}: 补充OFI特征 = {supplemented_features['OFI']}")

            if 'QI' not in supplemented_features:
                supplemented_features['QI'] = 0.0  # 中性队列不平衡

            if 'spread_bps' not in supplemented_features:
                # 使用历史点差均值或默认值
                hist_spread = self._get_historical_feature_mean(symbol, 'spread_bps')
                supplemented_features['spread_bps'] = hist_spread or 5.0  # 默认5bps

            if 'VPIN' not in supplemented_features:
                supplemented_features['VPIN'] = 0.3  # 中等毒性水平

            if 'TSI' not in supplemented_features:
                supplemented_features['TSI'] = 0.0  # 中性成交不平衡

            if 'slope_avg' not in supplemented_features:
                supplemented_features['slope_avg'] = 500.0  # 中等流动性

            if 'micro_dev_bps' not in supplemented_features:
                supplemented_features['micro_dev_bps'] = 0.0  # 中性微价偏离

            if 'mid_price' not in supplemented_features:
                supplemented_features['mid_price'] = current_price

            # 添加时间戳和数据质量标识
            supplemented_features['timestamp'] = time.time()
            supplemented_features['data_quality'] = 'supplemented'

            return supplemented_features

        except Exception as e:
            logger.error(f"补充特征失败 {symbol}: {e}")
            return features

    def _generate_basic_features(self, symbol: str) -> Dict[str, float]:
        """生成基础特征（最后降级选项）"""
        try:
            # 生成保守的默认特征
            basic_features = {
                'OFI': 0.0,
                'QI': 0.0,
                'mid_price': 100.0,  # 默认价格
                'spread_bps': 8.0,   # 保守的点差估计
                'VPIN': 0.4,         # 中等偏高毒性
                'TSI': 0.0,
                'slope_avg': 800.0,  # 保守的流动性估计
                'micro_dev_bps': 0.0,
                'timestamp': time.time(),
                'data_quality': 'basic_fallback'
            }

            logger.warning(f"{symbol}: 使用基础降级特征")
            return basic_features

        except Exception as e:
            logger.error(f"生成基础特征失败 {symbol}: {e}")
            return None

    def _get_historical_feature_mean(self, symbol: str, feature_name: str, lookback: int = 20) -> Optional[float]:
        """获取历史特征均值"""
        try:
            history = self.microstructure_engine.get_feature_history(symbol, lookback=lookback)
            if not history:
                return None

            values = []
            for hist_features in history:
                if feature_name in hist_features:
                    value = hist_features[feature_name]
                    if value is not None and not (isinstance(value, float) and
                        (math.isnan(value) or math.isinf(value))):
                        values.append(value)

            if values:
                return sum(values) / len(values)
            else:
                return None

        except Exception as e:
            logger.error(f"获取历史特征均值失败 {symbol} {feature_name}: {e}")
            return None

    def _create_hold_decision(self, symbol: str, timestamp: float, reason: str) -> TradingDecision:
        """创建HOLD决策"""
        return TradingDecision(
            symbol=symbol,
            timestamp=timestamp,
            raw_alpha_bps=0.0,
            calibrated_alpha_bps=0.0,
            confidence=0.0,
            spread_bps=0.0,
            impact_bps=0.0,
            total_cost_bps=0.0,
            is_tradable=False,
            recommended_side="HOLD",
            optimal_pov=0.0,
            vpin_risk_level="UNKNOWN",
            liquidity_risk_level="UNKNOWN",
            features={},
            decision_reason=reason
        )
    
    def _assess_vpin_risk(self, vpin: float) -> str:
        """评估VPIN风险等级"""
        if vpin > 0.8:
            return "HIGH"
        elif vpin > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_liquidity_risk(self, features: Dict[str, float]) -> str:
        """评估流动性风险等级"""
        spread_bps = features.get('spread_bps', 0.0)
        slope_avg = features.get('slope_avg', 0.0)
        
        # 综合评估
        if spread_bps > 10 or slope_avg > 1000:
            return "HIGH"
        elif spread_bps > 5 or slope_avg > 500:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_stats(self, decision: TradingDecision):
        """更新引擎统计"""
        self.stats['total_decisions'] += 1
        
        if decision.is_tradable:
            self.stats['tradable_decisions'] += 1
        
        self.stats['alpha_gt_cost_ratio'] = self.stats['tradable_decisions'] / self.stats['total_decisions']
        
        # 滚动平均
        alpha = 0.1  # 学习率
        self.stats['avg_alpha_bps'] = ((1-alpha) * self.stats['avg_alpha_bps'] + 
                                     alpha * abs(decision.calibrated_alpha_bps))
        self.stats['avg_cost_bps'] = ((1-alpha) * self.stats['avg_cost_bps'] + 
                                    alpha * decision.total_cost_bps)
    
    def get_decision_history(self, symbol: str, lookback: int = 10) -> List[TradingDecision]:
        """获取决策历史"""
        return list(self.decision_history[symbol])[-lookback:]
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        stats = self.stats.copy()
        stats['tracked_symbols'] = len(self.decision_history)
        stats['microstructure_status'] = self.microstructure_engine.get_status_summary()
        stats['impact_model_symbols'] = len(self.impact_model.impact_coefficients)
        return stats
    
    def batch_decide(self, symbols: List[str], market_context: Dict[str, Any] = None) -> Dict[str, TradingDecision]:
        """批量决策"""
        decisions = {}
        for symbol in symbols:
            decisions[symbol] = self.make_trading_decision(symbol, market_context)
        return decisions

# 全局实例
_global_alpha_engine = None

def get_realtime_alpha_engine() -> RealtimeAlphaEngine:
    """获取全局实时Alpha引擎"""
    global _global_alpha_engine
    if _global_alpha_engine is None:
        _global_alpha_engine = RealtimeAlphaEngine()
    return _global_alpha_engine

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = get_realtime_alpha_engine()
    
    # 模拟微结构数据
    micro_engine = engine.microstructure_engine
    symbol = "AAPL"
    
    # 添加模拟数据
    micro_engine.add_symbol(symbol)
    micro_engine.on_quote_update(symbol, "bid", 150.00, 1000)
    micro_engine.on_quote_update(symbol, "ask", 150.05, 800)
    micro_engine.on_trade(symbol, 150.05, 100, True)
    
    # 生成决策
    decision = engine.make_trading_decision(symbol)
    print(f"交易决策: {decision.recommended_side}, 可交易: {decision.is_tradable}")
    print(f"Alpha: {decision.calibrated_alpha_bps:.2f}bps, 成本: {decision.total_cost_bps:.2f}bps")
    print(f"原因: {decision.decision_reason}")
    
    # 统计
    stats = engine.get_engine_stats()
    print(f"引擎统计: {stats}")