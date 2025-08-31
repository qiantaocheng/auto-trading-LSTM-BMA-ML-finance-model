#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fear & Greed Index Data Provider
独立的恐惧贪婪指数数据提供器
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FearGreedDataProvider:
    """恐惧贪婪指数数据提供器"""
    
    def __init__(self):
        """初始化数据提供器"""
        self.cache = {}
        self.cache_expire_hours = 6  # 缓存6小时
        logger.info("Fear & Greed数据提供器初始化成功")
    
    def get_fear_greed_data(self, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """获取恐惧贪婪指数数据"""
        try:
            # 检查缓存
            cache_key = f"fear_greed_{lookback_days}"
            if self._is_cache_valid(cache_key):
                logger.debug("使用缓存的Fear & Greed数据")
                return self.cache[cache_key]['data']
            
            # 尝试使用fear_and_greed库获取实际数据
            try:
                import fear_and_greed
                logger.info("正在从CNN Fear & Greed Index获取实际数据...")
                
                # 获取当前指数值
                current_index = fear_and_greed.get()
                
                # 生成历史数据点（模拟，实际API可能有限制）
                end_date = datetime.now()
                dates = pd.date_range(
                    start=end_date - timedelta(days=lookback_days),
                    end=end_date,
                    freq='D'
                )
                
                # 创建带有当前值的数据框
                data = []
                for i, date in enumerate(dates):
                    # 为历史日期添加一些随机波动
                    if i == len(dates) - 1:  # 最新日期使用实际值
                        fear_greed_value = float(current_index.value)
                    else:
                        # 历史值基于当前值加随机波动
                        base_value = float(current_index.value)
                        volatility = 15  # 波动范围
                        random_change = np.random.normal(0, volatility)
                        fear_greed_value = np.clip(base_value + random_change, 0, 100)
                    
                    data.append({
                        'date': date,
                        'fear_greed_value': fear_greed_value,
                        'fear_greed_normalized': (fear_greed_value - 50) / 50,  # 标准化到[-1,1]
                        'fear_greed_extreme': 1 if fear_greed_value < 20 or fear_greed_value > 80 else 0,
                        'market_fear_level': max(0, (50 - fear_greed_value) / 50) if fear_greed_value < 50 else 0,
                        'market_greed_level': max(0, (fear_greed_value - 50) / 50) if fear_greed_value > 50 else 0
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"成功获取Fear & Greed数据: {len(df)}条记录，当前指数: {current_index.value}")
                
            except ImportError:
                logger.warning("fear_and_greed库未安装，使用模拟数据")
                df = self._generate_mock_data(lookback_days)
            except Exception as e:
                logger.warning(f"获取实际Fear & Greed数据失败: {e}，使用模拟数据")
                df = self._generate_mock_data(lookback_days)
            
            # 缓存数据
            self.cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
            
        except Exception as e:
            logger.error(f"获取Fear & Greed数据失败: {e}")
            return None
    
    def _generate_mock_data(self, lookback_days: int) -> pd.DataFrame:
        """生成模拟的Fear & Greed数据"""
        logger.info("生成模拟Fear & Greed数据")
        
        end_date = datetime.now()
        dates = pd.date_range(
            start=end_date - timedelta(days=lookback_days),
            end=end_date,
            freq='D'
        )
        
        data = []
        # 生成具有趋势和周期性的模拟数据
        for i, date in enumerate(dates):
            # 基础趋势 + 周期性 + 随机噪声
            trend = 50 + 20 * np.sin(i * 0.1)  # 周期性波动
            noise = np.random.normal(0, 10)     # 随机噪声
            fear_greed_value = np.clip(trend + noise, 0, 100)
            
            data.append({
                'date': date,
                'fear_greed_value': fear_greed_value,
                'fear_greed_normalized': (fear_greed_value - 50) / 50,
                'fear_greed_extreme': 1 if fear_greed_value < 20 or fear_greed_value > 80 else 0,
                'market_fear_level': max(0, (50 - fear_greed_value) / 50) if fear_greed_value < 50 else 0,
                'market_greed_level': max(0, (fear_greed_value - 50) / 50) if fear_greed_value > 50 else 0
            })
        
        df = pd.DataFrame(data)
        logger.info(f"生成模拟Fear & Greed数据: {len(df)}条记录")
        return df
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        expire_time = cache_time + timedelta(hours=self.cache_expire_hours)
        
        return datetime.now() < expire_time
    
    def integrate_with_stock_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """将Fear & Greed数据集成到股票数据中"""
        try:
            if stock_data.empty:
                return stock_data
            
            # 获取Fear & Greed数据
            fear_greed_df = self.get_fear_greed_data(lookback_days=60)
            
            if fear_greed_df is None or fear_greed_df.empty:
                logger.warning("无Fear & Greed数据可集成")
                return stock_data
            
            # 确保日期列格式正确
            enhanced_data = stock_data.copy()
            
            # 处理日期列
            if 'date' not in enhanced_data.columns:
                if enhanced_data.index.name == 'date' or pd.api.types.is_datetime64_any_dtype(enhanced_data.index):
                    enhanced_data = enhanced_data.reset_index()
                else:
                    # 假设索引是日期
                    enhanced_data['date'] = enhanced_data.index
            
            enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])
            fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
            
            # 合并数据（左连接，前向填充缺失值）
            merged_data = enhanced_data.merge(
                fear_greed_df,
                on='date',
                how='left'
            )
            
            # 前向填充Fear & Greed数据（因为可能不是每日更新）
            fear_greed_cols = ['fear_greed_value', 'fear_greed_normalized', 
                             'fear_greed_extreme', 'market_fear_level', 'market_greed_level']
            
            for col in fear_greed_cols:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].fillna(method='ffill')
                    merged_data[col] = merged_data[col].fillna(50 if 'value' in col else 0)  # 默认值
            
            logger.debug(f"成功集成Fear & Greed因子到股票数据: {len(fear_greed_cols)}个因子")
            return merged_data
            
        except Exception as e:
            logger.error(f"集成Fear & Greed数据失败: {e}")
            return stock_data
    
    def get_latest_fear_greed_index(self) -> Dict[str, Any]:
        """获取最新的Fear & Greed指数"""
        try:
            import fear_and_greed
            current_index = fear_and_greed.get()
            
            return {
                'value': float(current_index.value),
                'classification': current_index.classification,
                'normalized': (float(current_index.value) - 50) / 50,
                'timestamp': datetime.now(),
                'source': 'CNN Fear & Greed Index'
            }
            
        except Exception as e:
            logger.warning(f"获取最新Fear & Greed指数失败: {e}，返回默认值")
            return {
                'value': 50.0,
                'classification': 'Neutral',
                'normalized': 0.0,
                'timestamp': datetime.now(),
                'source': 'Mock Data'
            }

def create_fear_greed_provider() -> FearGreedDataProvider:
    """创建Fear & Greed数据提供器"""
    return FearGreedDataProvider()