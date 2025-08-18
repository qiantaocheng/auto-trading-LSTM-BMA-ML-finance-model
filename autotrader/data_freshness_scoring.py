#!/usr/bin/env python3
"""
数据新鲜度评分系统
用于评估和调整信号质量基于数据时效性
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


@dataclass
class FreshnessConfig:
    """数据新鲜度配置"""
    
    # 时间衰减参数
    tau_minutes: float = 15.0              # 衰减常数(分钟级: 10-30)
    max_age_minutes: float = 60.0          # 最大可接受数据年龄(分钟)
    
    # 时间戳连续性参数
    max_gap_bars: int = 2                  # 最大允许跳空bar数
    gap_penalty: float = 0.5               # 时间戳不连续时的惩罚因子
    
    # 缺失率惩罚参数
    max_missing_ratio: float = 0.3         # 最大可接受缺失率
    missing_penalty_factor: float = 1.0    # 缺失率惩罚系数
    
    # 动态阈值参数
    base_threshold: float = 0.005          # 基础阈值 k_0
    freshness_threshold_add: float = 0.010 # 新鲜度惩罚阈值 k_add
    
    # 数据源权重
    realtime_weight: float = 1.0           # 实时数据权重
    delayed_weight: float = 0.8            # 延迟数据权重
    cached_weight: float = 0.6             # 缓存数据权重
    
    # 质量控制
    min_data_points: int = 10              # 最小数据点数量
    quality_check_enabled: bool = True      # 是否启用质量检查


class DataFreshnessScoring:
    """
    数据新鲜度评分系统
    
    实现基于以下公式的新鲜度评分:
    F = exp(-Δt/τ) · q_gap · q_miss
    
    其中:
    - Δt: 最新bar距离当前的时间
    - τ: 衰减常数
    - q_gap: 时间戳连续性质量因子
    - q_miss: 缺失率惩罚因子
    """
    
    def __init__(self, config: FreshnessConfig = None):
        self.config = config or FreshnessConfig()
        self.logger = logging.getLogger("DataFreshnessScoring")
        
        # 缓存最近的新鲜度评分
        self.freshness_cache: Dict[str, Dict] = {}
        self.last_update_time: Dict[str, float] = {}
        
    def calculate_freshness_score(self,
                                symbol: str,
                                data_timestamp: datetime,
                                data_source: str = 'unknown',
                                missing_ratio: float = 0.0,
                                data_gaps: Optional[List[datetime]] = None,
                                data_quality_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        计算数据新鲜度评分
        
        Args:
            symbol: 股票代码
            data_timestamp: 数据时间戳
            data_source: 数据源类型 ('realtime', 'delayed', 'cached')
            missing_ratio: 数据缺失率 (0-1)
            data_gaps: 时间戳跳空列表
            data_quality_metrics: 额外的数据质量指标
            
        Returns:
            包含新鲜度评分和详细信息的字典
        """
        try:
            current_time = datetime.now()
            
            # 1. 计算时间衰减因子
            time_decay = self._calculate_time_decay(data_timestamp, current_time)
            
            # 2. 计算时间戳连续性因子
            gap_quality = self._calculate_gap_quality(data_gaps)
            
            # 3. 计算缺失率惩罚因子
            missing_quality = self._calculate_missing_quality(missing_ratio)
            
            # 4. 计算数据源权重
            source_weight = self._get_source_weight(data_source)
            
            # 5. 综合新鲜度评分
            freshness_score = time_decay * gap_quality * missing_quality * source_weight
            
            # 6. 质量等级评估
            quality_grade = self._assess_quality_grade(freshness_score)
            
            # 7. 计算动态阈值调整
            dynamic_threshold = self._calculate_dynamic_threshold(freshness_score)
            
            result = {
                'symbol': symbol,
                'freshness_score': freshness_score,
                'quality_grade': quality_grade,
                'timestamp': data_timestamp,
                'age_minutes': (current_time - data_timestamp).total_seconds() / 60,
                'data_source': data_source,
                
                # 组件评分
                'time_decay': time_decay,
                'gap_quality': gap_quality,
                'missing_quality': missing_quality,
                'source_weight': source_weight,
                
                # 阈值信息
                'dynamic_threshold': dynamic_threshold,
                'base_threshold': self.config.base_threshold,
                'threshold_adjustment': dynamic_threshold - self.config.base_threshold,
                
                # 质量指标
                'missing_ratio': missing_ratio,
                'data_gaps_count': len(data_gaps) if data_gaps else 0,
                'is_acceptable': self._is_data_acceptable(freshness_score, current_time - data_timestamp),
                
                # 建议
                'recommendation': self._get_recommendation(freshness_score, quality_grade)
            }
            
            # 缓存结果
            self.freshness_cache[symbol] = result
            self.last_update_time[symbol] = time.time()
            
            self.logger.debug(f"{symbol} 新鲜度评分: {freshness_score:.3f} "
                            f"(时间衰减={time_decay:.3f}, 跳空={gap_quality:.3f}, "
                            f"缺失={missing_quality:.3f}, 质量={quality_grade})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算{symbol}新鲜度评分失败: {e}")
            return self._create_default_score(symbol, data_timestamp, str(e))
    
    def _calculate_time_decay(self, data_timestamp: datetime, current_time: datetime) -> float:
        """计算时间衰减因子 exp(-Δt/τ)"""
        try:
            time_diff_minutes = (current_time - data_timestamp).total_seconds() / 60
            
            # 超过最大年龄直接返回0
            if time_diff_minutes > self.config.max_age_minutes:
                return 0.0
            
            # 指数衰减
            decay_factor = np.exp(-time_diff_minutes / self.config.tau_minutes)
            
            return max(0.0, min(1.0, decay_factor))
            
        except Exception as e:
            self.logger.warning(f"时间衰减计算失败: {e}")
            return 0.5
    
    def _calculate_gap_quality(self, data_gaps: Optional[List[datetime]]) -> float:
        """计算时间戳连续性质量因子 q_gap"""
        if not data_gaps:
            return 1.0  # 无跳空，质量满分
        
        gap_count = len(data_gaps)
        
        # 如果跳空数量超过阈值，应用惩罚
        if gap_count > self.config.max_gap_bars:
            return self.config.gap_penalty
        
        # 根据跳空数量线性衰减
        quality = 1.0 - (gap_count / self.config.max_gap_bars) * (1.0 - self.config.gap_penalty)
        
        return max(self.config.gap_penalty, quality)
    
    def _calculate_missing_quality(self, missing_ratio: float) -> float:
        """计算缺失率惩罚因子 q_miss"""
        if missing_ratio <= 0:
            return 1.0
        
        if missing_ratio >= self.config.max_missing_ratio:
            return 0.1  # 缺失率过高，严重惩罚
        
        # 线性惩罚: 1 - missing_ratio * penalty_factor
        quality = 1.0 - missing_ratio * self.config.missing_penalty_factor
        
        return max(0.1, quality)
    
    def _get_source_weight(self, data_source: str) -> float:
        """获取数据源权重"""
        weights = {
            'realtime': self.config.realtime_weight,
            'delayed': self.config.delayed_weight,
            'cached': self.config.cached_weight,
            'unknown': 0.7  # 未知来源给中等权重
        }
        
        return weights.get(data_source.lower(), 0.7)
    
    def _assess_quality_grade(self, freshness_score: float) -> str:
        """评估数据质量等级"""
        if freshness_score >= 0.9:
            return 'EXCELLENT'
        elif freshness_score >= 0.7:
            return 'GOOD'
        elif freshness_score >= 0.5:
            return 'FAIR'
        elif freshness_score >= 0.3:
            return 'POOR'
        else:
            return 'UNACCEPTABLE'
    
    def _calculate_dynamic_threshold(self, freshness_score: float) -> float:
        """
        计算动态阈值
        
        阈值公式: threshold = k_0 + (1-F) * k_add
        其中F为新鲜度评分
        """
        threshold_adjustment = (1.0 - freshness_score) * self.config.freshness_threshold_add
        return self.config.base_threshold + threshold_adjustment
    
    def _is_data_acceptable(self, freshness_score: float, age_delta: timedelta) -> bool:
        """判断数据是否可接受"""
        if freshness_score < 0.2:  # 新鲜度过低
            return False
        
        if age_delta.total_seconds() / 60 > self.config.max_age_minutes:  # 数据过旧
            return False
        
        return True
    
    def _get_recommendation(self, freshness_score: float, quality_grade: str) -> str:
        """获取使用建议"""
        if quality_grade == 'EXCELLENT':
            return 'USE_DIRECTLY'
        elif quality_grade == 'GOOD':
            return 'USE_WITH_CAUTION'
        elif quality_grade == 'FAIR':
            return 'USE_WITH_ADJUSTMENT'
        elif quality_grade == 'POOR':
            return 'AVOID_IF_POSSIBLE'
        else:
            return 'DO_NOT_USE'
    
    def _create_default_score(self, symbol: str, timestamp: datetime, error: str) -> Dict[str, Any]:
        """创建默认评分（出错时使用）"""
        return {
            'symbol': symbol,
            'freshness_score': 0.0,
            'quality_grade': 'ERROR',
            'timestamp': timestamp,
            'age_minutes': 999.0,
            'data_source': 'unknown',
            'error': error,
            'is_acceptable': False,
            'recommendation': 'DO_NOT_USE',
            'dynamic_threshold': self.config.base_threshold + self.config.freshness_threshold_add
        }
    
    def apply_freshness_to_signal(self,
                                 symbol: str,
                                 raw_signal: float,
                                 freshness_score: Optional[float] = None) -> Tuple[float, Dict]:
        """
        将新鲜度评分应用到交易信号
        
        Args:
            symbol: 股票代码
            raw_signal: 原始信号强度
            freshness_score: 新鲜度评分（如果为None则从缓存获取）
            
        Returns:
            (有效信号, 详细信息)
        """
        try:
            # 获取新鲜度评分
            if freshness_score is None:
                if symbol in self.freshness_cache:
                    freshness_info = self.freshness_cache[symbol]
                    freshness_score = freshness_info['freshness_score']
                else:
                    self.logger.warning(f"{symbol} 无新鲜度评分缓存，使用默认值0.5")
                    freshness_score = 0.5
            
            # 计算有效信号: s_eff = s_raw * F
            effective_signal = raw_signal * freshness_score
            
            # 获取动态阈值
            dynamic_threshold = self._calculate_dynamic_threshold(freshness_score)
            
            # 判断是否通过阈值
            passes_threshold = abs(effective_signal) > dynamic_threshold
            
            result = {
                'symbol': symbol,
                'raw_signal': raw_signal,
                'freshness_score': freshness_score,
                'effective_signal': effective_signal,
                'dynamic_threshold': dynamic_threshold,
                'passes_threshold': passes_threshold,
                'signal_adjustment': effective_signal - raw_signal,
                'threshold_adjustment': dynamic_threshold - self.config.base_threshold
            }
            
            self.logger.debug(f"{symbol} 信号新鲜度调整: {raw_signal:.4f} → {effective_signal:.4f} "
                            f"(F={freshness_score:.3f}, 阈值={dynamic_threshold:.4f})")
            
            return effective_signal, result
            
        except Exception as e:
            self.logger.error(f"应用新鲜度到信号失败 {symbol}: {e}")
            return raw_signal, {'error': str(e)}
    
    def get_cached_freshness(self, symbol: str) -> Optional[Dict]:
        """获取缓存的新鲜度评分"""
        return self.freshness_cache.get(symbol)
    
    def clear_stale_cache(self, max_cache_age_seconds: float = 300):
        """清理过期缓存"""
        current_time = time.time()
        stale_symbols = []
        
        for symbol, update_time in self.last_update_time.items():
            if current_time - update_time > max_cache_age_seconds:
                stale_symbols.append(symbol)
        
        for symbol in stale_symbols:
            self.freshness_cache.pop(symbol, None)
            self.last_update_time.pop(symbol, None)
        
        if stale_symbols:
            self.logger.debug(f"清理{len(stale_symbols)}个过期新鲜度缓存")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.freshness_cache:
            return {'cached_symbols': 0}
        
        scores = [info['freshness_score'] for info in self.freshness_cache.values()]
        quality_grades = [info['quality_grade'] for info in self.freshness_cache.values()]
        
        from collections import Counter
        grade_counts = Counter(quality_grades)
        
        return {
            'cached_symbols': len(self.freshness_cache),
            'average_freshness': np.mean(scores),
            'median_freshness': np.median(scores),
            'min_freshness': np.min(scores),
            'max_freshness': np.max(scores),
            'quality_distribution': dict(grade_counts),
            'config': {
                'tau_minutes': self.config.tau_minutes,
                'max_age_minutes': self.config.max_age_minutes,
                'base_threshold': self.config.base_threshold,
                'freshness_threshold_add': self.config.freshness_threshold_add
            }
        }


def create_freshness_scoring(tau_minutes: float = 15.0,
                           max_age_minutes: float = 60.0,
                           base_threshold: float = 0.005,
                           freshness_threshold_add: float = 0.010) -> DataFreshnessScoring:
    """
    创建数据新鲜度评分系统的便捷函数
    
    Args:
        tau_minutes: 衰减常数(分钟)
        max_age_minutes: 最大数据年龄(分钟)
        base_threshold: 基础阈值
        freshness_threshold_add: 新鲜度惩罚阈值
        
    Returns:
        配置好的新鲜度评分系统
    """
    config = FreshnessConfig(
        tau_minutes=tau_minutes,
        max_age_minutes=max_age_minutes,
        base_threshold=base_threshold,
        freshness_threshold_add=freshness_threshold_add
    )
    
    return DataFreshnessScoring(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建新鲜度评分系统
    freshness = create_freshness_scoring(
        tau_minutes=15.0,
        max_age_minutes=60.0,
        base_threshold=0.005,
        freshness_threshold_add=0.010
    )
    
    print("=== 数据新鲜度评分系统测试 ===")
    
    current_time = datetime.now()
    
    # 测试用例
    test_cases = [
        {
            'symbol': 'AAPL_FRESH',
            'timestamp': current_time - timedelta(minutes=2),
            'source': 'realtime',
            'missing_ratio': 0.0,
            'gaps': [],
            'note': '新鲜实时数据'
        },
        {
            'symbol': 'MSFT_DELAYED',
            'timestamp': current_time - timedelta(minutes=20),
            'source': 'delayed',
            'missing_ratio': 0.1,
            'gaps': [current_time - timedelta(minutes=15)],
            'note': '延迟数据有小幅缺失'
        },
        {
            'symbol': 'GOOGL_OLD',
            'timestamp': current_time - timedelta(minutes=45),
            'source': 'cached',
            'missing_ratio': 0.25,
            'gaps': [current_time - timedelta(minutes=40), current_time - timedelta(minutes=35)],
            'note': '旧缓存数据有较多缺失'
        },
        {
            'symbol': 'TSLA_STALE',
            'timestamp': current_time - timedelta(minutes=70),
            'source': 'cached',
            'missing_ratio': 0.4,
            'gaps': [],
            'note': '过期数据'
        },
    ]
    
    # 测试信号
    test_signals = [0.008, 0.006, 0.004, 0.012]
    
    print(f"基础阈值: {freshness.config.base_threshold:.4f}")
    print(f"新鲜度惩罚: {freshness.config.freshness_threshold_add:.4f}")
    print()
    
    for i, case in enumerate(test_cases):
        print(f"=== {case['symbol']} ({case['note']}) ===")
        
        # 计算新鲜度评分
        freshness_result = freshness.calculate_freshness_score(
            symbol=case['symbol'],
            data_timestamp=case['timestamp'],
            data_source=case['source'],
            missing_ratio=case['missing_ratio'],
            data_gaps=case['gaps']
        )
        
        print(f"时间戳: {case['timestamp'].strftime('%H:%M:%S')}")
        print(f"数据年龄: {freshness_result['age_minutes']:.1f}分钟")
        print(f"新鲜度评分: {freshness_result['freshness_score']:.3f}")
        print(f"质量等级: {freshness_result['quality_grade']}")
        print(f"动态阈值: {freshness_result['dynamic_threshold']:.4f}")
        print(f"可接受: {freshness_result['is_acceptable']}")
        
        # 应用到信号
        raw_signal = test_signals[i]
        effective_signal, signal_info = freshness.apply_freshness_to_signal(
            case['symbol'], raw_signal
        )
        
        print(f"原始信号: {raw_signal:.4f}")
        print(f"有效信号: {effective_signal:.4f}")
        print(f"通过阈值: {signal_info['passes_threshold']}")
        print()
    
    # 系统统计
    stats = freshness.get_statistics()
    print(f"=== 系统统计 ===")
    print(f"缓存符号数: {stats['cached_symbols']}")
    print(f"平均新鲜度: {stats['average_freshness']:.3f}")
    print(f"质量分布: {stats['quality_distribution']}")
    
    print("\n✅ 数据新鲜度评分系统测试完成")