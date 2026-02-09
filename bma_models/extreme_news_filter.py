#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鏋佺鏂伴椈鍥犲瓙杩囨护妯″潡 (Extreme News Filter with Purging Window)

鏍稿績鍔熻兘锛?
1. 璇嗗埆鏋佺鏂伴椈浜嬩欢锛堝崟鏃ユ定璺屽箙>闃堝€?鎴?>3鍊嶆尝鍔ㄧ巼锛?
2. 鎵ц绐楀彛鍑€鍖栵紙Purging锛夛細鍓旈櫎鏋佺浜嬩欢鍓峢orizon澶╃殑鏍锋湰
   鍘熷洜锛歵arget鏄痳et_fwd_10d锛屽鏋淭鏃ユ湁鏋佺浜嬩欢锛孴-10鍒癟鐨則arget閮戒細鍙楀奖鍝?
3. 璁粌鏃惰繃婊わ紝棰勬祴鏃舵爣璁颁絾涓嶈繃婊?
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ExtremeNewsFilter:
    """
    鏋佺鏂伴椈鍥犲瓙杩囨护鍣紙甯︾獥鍙ｅ噣鍖栵級
    
    璁捐鐞嗗康锛?
    - 璁粌鏃讹細鍓旈櫎鏋佺浜嬩欢鍙婂叾鍓峢orizon澶╃殑鏍锋湰锛堥槻姝arget姹℃煋锛?
    - 棰勬祴鏃讹細鏍囪鏋佺浜嬩欢浣嗕笉鍓旈櫎锛堜繚鐣欐墍鏈夐娴嬪満鏅級
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
        鍒濆鍖栨瀬绔柊闂昏繃婊ゅ櫒
        
        Args:
            threshold: 鍥哄畾闃堝€硷紙榛樿10%锛?
            volatility_multiplier: 娉㈠姩鐜囧€嶆暟锛堥粯璁?鍊嶏級
            volatility_window: 娉㈠姩鐜囪绠楃獥鍙ｏ紙榛樿20澶╋級
            horizon: 鐩爣棰勬祴鍛ㄦ湡锛堥粯璁?0澶╋紝鐢ㄤ簬绐楀彛鍑€鍖栵級
            enabled: 鏄惁鍚敤杩囨护锛堥粯璁rue锛?
        """
        self.threshold = threshold
        self.volatility_multiplier = volatility_multiplier
        self.volatility_window = volatility_window
        self.horizon = horizon
        self.enabled = enabled
        
        logger.info(f"鉁?ExtremeNewsFilter initialized:")
        logger.info(f"   threshold={threshold*100:.1f}%, volatility_multiplier={volatility_multiplier}x")
        logger.info(f"   volatility_window={volatility_window}d, horizon={horizon}d, enabled={enabled}")
    
    def _compute_daily_returns(self, df: pd.DataFrame, close_col: str = 'Close') -> pd.Series:
        """璁＄畻鍗曟棩鏀剁泭鐜?""
        if close_col not in df.columns:
            raise ValueError(f"Column '{close_col}' not found in DataFrame")
        
        # 鎸塼icker鍒嗙粍璁＄畻鏀剁泭鐜?
        if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
            grouped = df.groupby(level='ticker')[close_col]
            daily_return = grouped.pct_change()
        elif 'ticker' in df.columns:
            grouped = df.groupby('ticker')[close_col]
            daily_return = grouped.pct_change()
        else:
            # 濡傛灉娌℃湁ticker淇℃伅锛岀洿鎺ヨ绠楋紙涓嶆帹鑽愶級
            daily_return = df[close_col].pct_change()
            logger.warning("鈿狅笍 No ticker grouping found, computing returns without grouping")
        
        return daily_return
    
    def _compute_rolling_volatility(self, daily_return: pd.Series, ticker_grouped: bool = True) -> pd.Series:
        """璁＄畻婊氬姩娉㈠姩鐜?""
        if ticker_grouped and isinstance(daily_return.index, pd.MultiIndex) and 'ticker' in daily_return.index.names:
            grouped = daily_return.groupby(level='ticker')
            rolling_std = grouped.transform(
                lambda s: s.rolling(self.volatility_window, min_periods=5).std()
            )
        elif ticker_grouped and hasattr(daily_return, 'groupby'):
            # 灏濊瘯鎸塼icker鍒嗙粍
            try:
                grouped = daily_return.groupby(level='ticker')
                rolling_std = grouped.transform(
                    lambda s: s.rolling(self.volatility_window, min_periods=5).std()
                )
            except:
                rolling_std = daily_return.rolling(self.volatility_window, min_periods=5).std()
        else:
            # Fallback: 鐩存帴璁＄畻锛堜笉鎺ㄨ崘锛?
            rolling_std = daily_return.rolling(self.volatility_window, min_periods=5).std()
            logger.warning("鈿狅笍 No ticker grouping found, computing volatility without grouping")
        
        return rolling_std.fillna(0.0)
    
    def _identify_extreme_events(
        self, 
        df: pd.DataFrame, 
        close_col: str = 'Close'
    ) -> pd.Series:
        """
        璇嗗埆鏋佺鏂伴椈浜嬩欢
        
        鏉′欢锛歛bs(daily_return) > threshold OR abs(daily_return) > volatility_multiplier * rolling_std
        
        Returns:
            is_extreme: Series of boolean values (True琛ㄧず鏋佺浜嬩欢)
        """
        # 璁＄畻鍗曟棩鏀剁泭鐜?
        daily_return = self._compute_daily_returns(df, close_col)
        
        # 璁＄畻婊氬姩娉㈠姩鐜?
        rolling_std = self._compute_rolling_volatility(daily_return)
        
        # 鍥哄畾闃堝€兼潯浠?
        threshold_condition = daily_return.abs() > self.threshold
        
        # 娉㈠姩鐜囧€嶆暟鏉′欢
        volatility_condition = daily_return.abs() > (self.volatility_multiplier * rolling_std)
        
        # 鍚堝苟鏉′欢锛圤R锛?
        is_extreme = threshold_condition | volatility_condition
        
        # 濉厖NaN涓篎alse
        is_extreme = is_extreme.fillna(False)
        
        return is_extreme
    
    def _apply_purging_window(
        self, 
        df: pd.DataFrame, 
        is_extreme: pd.Series
    ) -> pd.Series:
        """
        鎵ц绐楀彛鍑€鍖栵紙Purging Window锛?
        
        鏍稿績閫昏緫锛?
        - 濡傛灉T鏃ユ槸鏋佺浜嬩欢锛岄偅涔圱-horizon鍒癟鐨勬墍鏈夋牱鏈兘搴旇琚墧闄?
        - 鍥犱负target鏄痳et_fwd_10d锛孴鏃ョ殑鏋佺浜嬩欢浼氬奖鍝峊-horizon鍒癟鐨則arget鍊?
        
        Args:
            df: 鍘熷DataFrame
            is_extreme: 鏋佺浜嬩欢鏍囪Series
        
        Returns:
            is_polluted: Series of boolean values (True琛ㄧず琚薄鏌撶殑鏍锋湰锛屽簲琚墧闄?
        """
        # 纭繚is_extreme涓巇f瀵归綈
        if not is_extreme.index.equals(df.index):
            # 灏濊瘯閲嶆柊绱㈠紩瀵归綈
            is_extreme = is_extreme.reindex(df.index, fill_value=False)
        
        # 鎸塼icker鍒嗙粍澶勭悊
        is_polluted = pd.Series(False, index=df.index)
        
        if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
            # MultiIndex鎯呭喌锛氭寜ticker鍒嗙粍
            for ticker in df.index.get_level_values('ticker').unique():
                ticker_mask = df.index.get_level_values('ticker') == ticker
                ticker_extreme = is_extreme[ticker_mask]
                
                # 瀵规瘡涓猼icker锛屼娇鐢╮olling window鍚戝悗鐪媓orizon+1澶?
                # 濡傛灉鏈潵horizon澶╁唴鏈変换浣曟瀬绔簨浠讹紝褰撳墠鏍锋湰琚薄鏌?
                ticker_polluted = (
                    ticker_extreme
                    .rolling(window=self.horizon + 1, min_periods=1)
                    .max()
                    .shift(-self.horizon)  # 鍚戝悗骞崇Щhorizon澶?
                    .fillna(False)
                )
                
                is_polluted[ticker_mask] = ticker_polluted.values
                
        elif 'ticker' in df.columns:
            # 鏅€欴ataFrame锛屾湁ticker鍒?
            for ticker in df['ticker'].unique():
                ticker_mask = df['ticker'] == ticker
                ticker_extreme = is_extreme[ticker_mask]
                
                # 瀵规瘡涓猼icker锛屼娇鐢╮olling window鍚戝悗鐪媓orizon+1澶?
                ticker_polluted = (
                    ticker_extreme
                    .rolling(window=self.horizon + 1, min_periods=1)
                    .max()
                    .shift(-self.horizon)
                    .fillna(False)
                )
                
                is_polluted[ticker_mask] = ticker_polluted.values
        else:
            # 娌℃湁ticker鍒嗙粍锛岀洿鎺ュ鐞嗭紙涓嶆帹鑽愶級
            logger.warning("鈿狅笍 No ticker grouping found, applying purging without grouping")
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
        鎵ц鏋佺鏂伴椈杩囨护
        
        Args:
            df: 杈撳叆DataFrame锛堝簲鍖呭惈Close鍒楀拰MultiIndex鎴杢icker鍒楋級
            mode: 'train' 鎴?'predict'
            close_col: 鏀剁洏浠峰垪鍚嶏紙榛樿'Close'锛?
        
        Returns:
            filtered_df: 杩囨护鍚庣殑DataFrame
            is_extreme: 鏋佺浜嬩欢鏍囪Series锛堢敤浜庡垎鏋愶級
        """
        if not self.enabled:
            logger.info("鈴笍 ExtremeNewsFilter disabled, skipping filter")
            return df, pd.Series(False, index=df.index)
        
        mode = mode.lower()
        if mode not in ['train', 'predict']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'predict'")
        
        logger.info(f"馃攳 Applying extreme news filter (mode={mode})...")
        
        # 1. 璇嗗埆鏋佺浜嬩欢
        is_extreme = self._identify_extreme_events(df, close_col)
        extreme_count = is_extreme.sum()
        extreme_pct = extreme_count / len(df) * 100
        
        logger.info(f"   馃搳 Extreme events identified: {extreme_count:,} ({extreme_pct:.2f}%)")
        
        # 2. 鎵ц绐楀彛鍑€鍖栵紙浠呭湪璁粌妯″紡锛?
        if mode == 'train':
            is_polluted = self._apply_purging_window(df, is_extreme)
            polluted_count = is_polluted.sum()
            polluted_pct = polluted_count / len(df) * 100
            
            logger.info(f"   馃Ч Purging window applied: {polluted_count:,} samples polluted ({polluted_pct:.2f}%)")
            
            # 杩囨护琚薄鏌撶殑鏍锋湰
            filtered_df = df[~is_polluted].copy()
            
            logger.info(f"   鉁?Filtered: {len(df):,} 鈫?{len(filtered_df):,} samples ({len(df)-len(filtered_df):,} removed)")
        else:
            # 棰勬祴妯″紡锛氬彧鏍囪锛屼笉杩囨护
            filtered_df = df.copy()
            filtered_df['is_extreme_news'] = is_extreme
            logger.info(f"   鉁?Prediction mode: marked {extreme_count:,} extreme events (no filtering)")
        
        return filtered_df, is_extreme
    
    def get_filter_stats(self, df: pd.DataFrame, is_extreme: pd.Series) -> dict:
        """鑾峰彇杩囨护缁熻淇℃伅"""
        stats = {
            'total_samples': len(df),
            'extreme_events': int(is_extreme.sum()),
            'extreme_pct': float(is_extreme.sum() / len(df) * 100),
        }
        
        # 璁＄畻姝ｈ礋鏋佺浜嬩欢
        daily_return = self._compute_daily_returns(df)
        stats['positive_extreme'] = int((daily_return > self.threshold).sum())
        stats['negative_extreme'] = int((daily_return < -self.threshold).sum())
        
        # 濡傛灉鏈塼arget鍒楋紝璁＄畻鏋佺浜嬩欢鍚庣殑target缁熻
        if 'target' in df.columns:
            extreme_targets = df[is_extreme]['target'].dropna()
            normal_targets = df[~is_extreme]['target'].dropna()
            
            stats['extreme_target_mean'] = float(extreme_targets.mean()) if len(extreme_targets) > 0 else np.nan
            stats['normal_target_mean'] = float(normal_targets.mean()) if len(normal_targets) > 0 else np.nan
            stats['target_diff'] = float(stats['extreme_target_mean'] - stats['normal_target_mean']) if not np.isnan(stats['extreme_target_mean']) else np.nan
        
        return stats

