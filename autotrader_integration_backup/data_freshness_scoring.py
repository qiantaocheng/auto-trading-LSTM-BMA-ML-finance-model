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
    """数据新鲜度评分系统"""
    
    def __init__(self, config: Optional[FreshnessConfig] = None):
        self.config = config or FreshnessConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 缓存最近的评分结果
        self._score_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (score, timestamp)
        self._cache_ttl = 300  # 5分钟缓存
        
        self.logger.info(f"DataFreshnessScoring initialized with tau={self.config.tau_minutes}min")
    
    def calculate_freshness_score(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        计算数据新鲜度得分
        
        Args:
            data: 包含时间序列数据的DataFrame，需要有timestamp列
            symbol: 股票代码，用于缓存和日志
            
        Returns:
            Dict包含：
            - freshness_score: 新鲜度得分 (0-1)
            - age_penalty: 年龄惩罚
            - continuity_score: 连续性得分
            - quality_score: 质量得分
            - recommendation: 建议 ('accept', 'caution', 'reject')
        """
        try:
            if data is None or len(data) == 0:
                return self._create_rejection_result("Empty data")
            
            if len(data) < self.config.min_data_points:
                return self._create_rejection_result(f"Insufficient data points: {len(data)}")
            
            # 1. 计算数据年龄惩罚
            age_penalty = self._calculate_age_penalty(data)
            
            # 2. 计算时间连续性得分
            continuity_score = self._calculate_continuity_score(data)
            
            # 3. 计算数据质量得分
            quality_score = self._calculate_quality_score(data)
            
            # 4. 综合新鲜度得分
            freshness_score = (
                (1.0 - age_penalty) * 0.4 +
                continuity_score * 0.3 +
                quality_score * 0.3
            )
            
            # 5. 生成建议
            recommendation = self._generate_recommendation(freshness_score)
            
            result = {
                'symbol': symbol,
                'freshness_score': freshness_score,
                'age_penalty': age_penalty,
                'continuity_score': continuity_score,
                'quality_score': quality_score,
                'recommendation': recommendation,
                'data_points': len(data),
                'evaluation_time': datetime.now()
            }
            
            # 缓存结果
            self._score_cache[symbol] = (freshness_score, time.time())
            
            self.logger.debug(f"Freshness score for {symbol}: {freshness_score:.3f} ({recommendation})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating freshness score for {symbol}: {e}")
            return self._create_rejection_result(f"Calculation error: {e}")
    
    def _calculate_age_penalty(self, data: pd.DataFrame) -> float:
        """计算数据年龄惩罚"""
        if 'timestamp' not in data.columns:
            # 如果没有timestamp列，尝试使用索引
            if isinstance(data.index, pd.DatetimeIndex):
                latest_timestamp = data.index.max()
            else:
                return 0.5  # 无法确定时间，给中等惩罚
        else:
            latest_timestamp = pd.to_datetime(data['timestamp']).max()
        
        # 计算数据年龄（分钟）
        current_time = pd.Timestamp.now()
        age_minutes = (current_time - latest_timestamp).total_seconds() / 60
        
        if age_minutes > self.config.max_age_minutes:
            return 1.0  # 最大惩罚
        
        # 指数衰减
        age_penalty = 1.0 - np.exp(-age_minutes / self.config.tau_minutes)
        return max(0.0, min(1.0, age_penalty))
    
    def _calculate_continuity_score(self, data: pd.DataFrame) -> float:
        """计算时间连续性得分"""
        try:
            if 'timestamp' not in data.columns:
                if isinstance(data.index, pd.DatetimeIndex):
                    timestamps = data.index
                else:
                    return 0.7  # 无法评估，给中等分数
            else:
                timestamps = pd.to_datetime(data['timestamp'])
            
            if len(timestamps) < 2:
                return 1.0
            
            # 计算时间间隔
            time_diffs = timestamps.diff().dropna()
            
            # 检测异常间隔（大于正常间隔的3倍）
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 3).sum()
            
            # 连续性得分
            gap_ratio = large_gaps / len(time_diffs)
            continuity_score = max(0.0, 1.0 - gap_ratio * self.config.gap_penalty)
            
            return continuity_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating continuity score: {e}")
            return 0.5
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """计算数据质量得分"""
        try:
            # 计算缺失值比例
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 0.5
            
            total_values = len(data) * len(numeric_columns)
            missing_values = data[numeric_columns].isnull().sum().sum()
            missing_ratio = missing_values / total_values
            
            if missing_ratio > self.config.max_missing_ratio:
                return 0.0
            
            # 质量得分
            quality_score = max(0.0, 1.0 - missing_ratio * self.config.missing_penalty_factor)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.5
    
    def _generate_recommendation(self, freshness_score: float) -> str:
        """生成使用建议"""
        if freshness_score >= 0.7:
            return 'accept'
        elif freshness_score >= 0.4:
            return 'caution'
        else:
            return 'reject'
    
    def _create_rejection_result(self, reason: str) -> Dict[str, Any]:
        """创建拒绝结果"""
        return {
            'freshness_score': 0.0,
            'age_penalty': 1.0,
            'continuity_score': 0.0,
            'quality_score': 0.0,
            'recommendation': 'reject',
            'reason': reason,
            'evaluation_time': datetime.now()
        }
    
    def get_cached_score(self, symbol: str) -> Optional[float]:
        """获取缓存的评分"""
        if symbol not in self._score_cache:
            return None
        
        score, timestamp = self._score_cache[symbol]
        if time.time() - timestamp > self._cache_ttl:
            del self._score_cache[symbol]
            return None
        
        return score
    
    def adjust_signal_threshold(self, base_threshold: float, freshness_score: float) -> float:
        """根据新鲜度调整信号阈值"""
        if freshness_score >= 0.8:
            # 数据很新鲜，可以降低阈值
            return base_threshold
        elif freshness_score >= 0.5:
            # 数据一般新鲜，保持阈值
            return base_threshold + self.config.freshness_threshold_add * 0.5
        else:
            # 数据不新鲜，提高阈值
            return base_threshold + self.config.freshness_threshold_add
    
    def clear_cache(self):
        """清空缓存"""
        self._score_cache.clear()
        self.logger.info("Freshness score cache cleared")


def create_freshness_scoring(config: Optional[FreshnessConfig] = None) -> DataFreshnessScoring:
    """创建数据新鲜度评分实例"""
    return DataFreshnessScoring(config)


# 全局实例
_global_freshness_scorer: Optional[DataFreshnessScoring] = None


def get_freshness_scorer() -> DataFreshnessScoring:
    """获取全局数据新鲜度评分实例"""
    global _global_freshness_scorer
    if _global_freshness_scorer is None:
        _global_freshness_scorer = create_freshness_scoring()
    return _global_freshness_scorer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.randn(50).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 50)
    })
    
    # 测试新鲜度评分
    scorer = create_freshness_scoring()
    result = scorer.calculate_freshness_score(test_data, "TEST")
    
    print("Freshness scoring result:")
    for key, value in result.items():
        print(f"  {key}: {value}")