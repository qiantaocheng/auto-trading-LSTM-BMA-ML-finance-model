#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极端新闻因子过滤模块 (Extreme News Filter with Purging Window)

核心功能：
1. 识别极端新闻事件（单日涨跌幅>阈值 或 >3倍波动率）
2. 执行窗口净化（Purging）：剔除极端事件前horizon天的样本
   原因：target是ret_fwd_10d，如果T日有极端事件，T-10到T的target都会受影响
3. 训练时过滤，预测时标记但不过滤
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ExtremeNewsFilter:
    """
    极端新闻因子过滤器（带窗口净化）
    
    设计理念：
    - 训练时：剔除极端事件及其前horizon天的样本（防止target污染）
    - 预测时：标记极端事件但不剔除（保留所有预测场景）
    """
    
    def __init__(
        self,
        threshold: float = 0.10,
        volatility_multiplier: float = 3.0,
        volatility_window: int = 20,
        horizon: int = 10,
        enabled: bool = True,
    ):
        """
        初始化极端新闻过滤器
        
        Args:
            threshold: 固定阈值（默认10%）
            volatility_multiplier: 波动率倍数（默认3倍）
            volatility_window: 波动率计算窗口（默认20天）
            horizon: 目标预测周期（默认10天，用于窗口净化）
            enabled: 是否启用过滤（默认True）
        """
        self.threshold = threshold
        self.volatility_multiplier = volatility_multiplier
        self.volatility_window = volatility_window
        self.horizon = horizon
        self.enabled = enabled
        
        logger.info(f"✅ ExtremeNewsFilter initialized:")
        logger.info(f"   threshold={threshold*100:.1f}%, volatility_multiplier={volatility_multiplier}x")
        logger.info(f"   volatility_window={volatility_window}d, horizon={horizon}d, enabled={enabled}")
    
    def _compute_daily_returns(self, df: pd.DataFrame, close_col: str = 'Close') -> pd.Series:
        """计算单日收益率"""
        if close_col not in df.columns:
            raise ValueError(f"Column '{close_col}' not found in DataFrame")
        
        # 按ticker分组计算收益率
        if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
            grouped = df.groupby(level='ticker')[close_col]
            daily_return = grouped.pct_change()
        elif 'ticker' in df.columns:
            grouped = df.groupby('ticker')[close_col]
            daily_return = grouped.pct_change()
        else:
            # 如果没有ticker信息，直接计算（不推荐）
            daily_return = df[close_col].pct_change()
            logger.warning("⚠️ No ticker grouping found, computing returns without grouping")
        
        return daily_return
    
    def _compute_rolling_volatility(self, daily_return: pd.Series, ticker_grouped: bool = True) -> pd.Series:
        """计算滚动波动率"""
        if ticker_grouped and isinstance(daily_return.index, pd.MultiIndex) and 'ticker' in daily_return.index.names:
            grouped = daily_return.groupby(level='ticker')
            rolling_std = grouped.transform(
                lambda s: s.rolling(self.volatility_window, min_periods=5).std()
            )
        elif ticker_grouped and hasattr(daily_return, 'groupby'):
            # 尝试按ticker分组
            try:
                grouped = daily_return.groupby(level='ticker')
                rolling_std = grouped.transform(
                    lambda s: s.rolling(self.volatility_window, min_periods=5).std()
                )
            except:
                rolling_std = daily_return.rolling(self.volatility_window, min_periods=5).std()
        else:
            # Fallback: 直接计算（不推荐）
            rolling_std = daily_return.rolling(self.volatility_window, min_periods=5).std()
            logger.warning("⚠️ No ticker grouping found, computing volatility without grouping")
        
        return rolling_std.fillna(0.0)
    
    def _identify_extreme_events(
        self, 
        df: pd.DataFrame, 
        close_col: str = 'Close'
    ) -> pd.Series:
        """
        识别极端新闻事件
        
        条件：abs(daily_return) > threshold OR abs(daily_return) > volatility_multiplier * rolling_std
        
        Returns:
            is_extreme: Series of boolean values (True表示极端事件)
        """
        # 计算单日收益率
        daily_return = self._compute_daily_returns(df, close_col)
        
        # 计算滚动波动率
        rolling_std = self._compute_rolling_volatility(daily_return)
        
        # 固定阈值条件
        threshold_condition = daily_return.abs() > self.threshold
        
        # 波动率倍数条件
        volatility_condition = daily_return.abs() > (self.volatility_multiplier * rolling_std)
        
        # 合并条件（OR）
        is_extreme = threshold_condition | volatility_condition
        
        # 填充NaN为False
        is_extreme = is_extreme.fillna(False)
        
        return is_extreme
    
    def _apply_purging_window(
        self, 
        df: pd.DataFrame, 
        is_extreme: pd.Series
    ) -> pd.Series:
        """
        执行窗口净化（Purging Window）
        
        核心逻辑：
        - 如果T日是极端事件，那么T-horizon到T的所有样本都应该被剔除
        - 因为target是ret_fwd_10d，T日的极端事件会影响T-horizon到T的target值
        
        Args:
            df: 原始DataFrame
            is_extreme: 极端事件标记Series
        
        Returns:
            is_polluted: Series of boolean values (True表示被污染的样本，应被剔除)
        """
        # 确保is_extreme与df对齐
        if not is_extreme.index.equals(df.index):
            # 尝试重新索引对齐
            is_extreme = is_extreme.reindex(df.index, fill_value=False)
        
        # 按ticker分组处理
        is_polluted = pd.Series(False, index=df.index)
        
        if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
            # MultiIndex情况：按ticker分组
            for ticker in df.index.get_level_values('ticker').unique():
                ticker_mask = df.index.get_level_values('ticker') == ticker
                ticker_extreme = is_extreme[ticker_mask]
                
                # 对每个ticker，使用rolling window向后看horizon+1天
                # 如果未来horizon天内有任何极端事件，当前样本被污染
                ticker_polluted = (
                    ticker_extreme
                    .rolling(window=self.horizon + 1, min_periods=1)
                    .max()
                    .shift(-self.horizon)  # 向后平移horizon天
                    .fillna(False)
                )
                
                is_polluted[ticker_mask] = ticker_polluted.values
                
        elif 'ticker' in df.columns:
            # 普通DataFrame，有ticker列
            for ticker in df['ticker'].unique():
                ticker_mask = df['ticker'] == ticker
                ticker_extreme = is_extreme[ticker_mask]
                
                # 对每个ticker，使用rolling window向后看horizon+1天
                ticker_polluted = (
                    ticker_extreme
                    .rolling(window=self.horizon + 1, min_periods=1)
                    .max()
                    .shift(-self.horizon)
                    .fillna(False)
                )
                
                is_polluted[ticker_mask] = ticker_polluted.values
        else:
            # 没有ticker分组，直接处理（不推荐）
            logger.warning("⚠️ No ticker grouping found, applying purging without grouping")
            is_polluted = (
                is_extreme
                .rolling(window=self.horizon + 1, min_periods=1)
                .max()
                .shift(-self.horizon)
                .fillna(False)
            )
        
        return is_polluted.fillna(False)
    
    def filter(
        self, 
        df: pd.DataFrame, 
        mode: str = 'train',
        close_col: str = 'Close'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        执行极端新闻过滤
        
        Args:
            df: 输入DataFrame（应包含Close列和MultiIndex或ticker列）
            mode: 'train' 或 'predict'
            close_col: 收盘价列名（默认'Close'）
        
        Returns:
            filtered_df: 过滤后的DataFrame
            is_extreme: 极端事件标记Series（用于分析）
        """
        if not self.enabled:
            logger.info("⏭️ ExtremeNewsFilter disabled, skipping filter")
            return df, pd.Series(False, index=df.index)
        
        mode = mode.lower()
        if mode not in ['train', 'predict']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'predict'")
        
        logger.info(f"🔍 Applying extreme news filter (mode={mode})...")
        
        # 1. 识别极端事件
        is_extreme = self._identify_extreme_events(df, close_col)
        extreme_count = is_extreme.sum()
        extreme_pct = extreme_count / len(df) * 100
        
        logger.info(f"   📊 Extreme events identified: {extreme_count:,} ({extreme_pct:.2f}%)")
        
        # 2. 执行窗口净化（仅在训练模式）
        if mode == 'train':
            is_polluted = self._apply_purging_window(df, is_extreme)
            polluted_count = is_polluted.sum()
            polluted_pct = polluted_count / len(df) * 100
            
            logger.info(f"   🧹 Purging window applied: {polluted_count:,} samples polluted ({polluted_pct:.2f}%)")
            
            # 过滤被污染的样本
            filtered_df = df[~is_polluted].copy()
            
            logger.info(f"   ✅ Filtered: {len(df):,} → {len(filtered_df):,} samples ({len(df)-len(filtered_df):,} removed)")
        else:
            # 预测模式：只标记，不过滤
            filtered_df = df.copy()
            filtered_df['is_extreme_news'] = is_extreme
            logger.info(f"   ✅ Prediction mode: marked {extreme_count:,} extreme events (no filtering)")
        
        return filtered_df, is_extreme
    
    def get_filter_stats(self, df: pd.DataFrame, is_extreme: pd.Series) -> dict:
        """获取过滤统计信息"""
        stats = {
            'total_samples': len(df),
            'extreme_events': int(is_extreme.sum()),
            'extreme_pct': float(is_extreme.sum() / len(df) * 100),
        }
        
        # 计算正负极端事件
        daily_return = self._compute_daily_returns(df)
        stats['positive_extreme'] = int((daily_return > self.threshold).sum())
        stats['negative_extreme'] = int((daily_return < -self.threshold).sum())
        
        # 如果有target列，计算极端事件后的target统计
        if 'target' in df.columns:
            extreme_targets = df[is_extreme]['target'].dropna()
            normal_targets = df[~is_extreme]['target'].dropna()
            
            stats['extreme_target_mean'] = float(extreme_targets.mean()) if len(extreme_targets) > 0 else np.nan
            stats['normal_target_mean'] = float(normal_targets.mean()) if len(normal_targets) > 0 else np.nan
            stats['target_diff'] = float(stats['extreme_target_mean'] - stats['normal_target_mean']) if not np.isnan(stats['extreme_target_mean']) else np.nan
        
        return stats
