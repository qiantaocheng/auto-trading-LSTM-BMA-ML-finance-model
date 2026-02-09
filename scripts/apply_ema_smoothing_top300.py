"""
改进的EMA平滑函数：只对连续2天都在top300的股票应用EMA (L2 B0.7)
Default weights: (0.588, 0.412) from grid search optimal config
"""

import pandas as pd
import numpy as np
from typing import Sequence


def _compute_weighted_score(score_today: float, history: list, weights: Sequence[float]) -> float:
    if pd.isna(score_today) or not weights:
        return np.nan if pd.isna(score_today) else score_today
    values = [score_today]
    for past in history:
        if len(values) >= len(weights):
            break
        values.append(past)
    usable_weights = list(weights[:len(values)])
    valid_pairs = [(w, v) for w, v in zip(usable_weights, values) if not pd.isna(v)]
    if not valid_pairs:
        return score_today
    total_w = sum(w for w, _ in valid_pairs)
    if total_w == 0:
        return score_today
    return sum(w * v for w, v in valid_pairs) / total_w


def apply_ema_smoothing_top300_filter(
    predictions_df: pd.DataFrame,
    model_name: str,
    ema_history: dict,
    weights: Sequence[float] = (0.588, 0.412),
    top_n: int = 300,
    min_days_in_top: int = 2
) -> pd.DataFrame:
    """
    应用EMA平滑(L2 B0.7)，但只对连续N天都在top300的股票应用EMA

    策略：
    1. 如果股票在过去2天都在top300：应用EMA平滑 (权重: 0.588, 0.412)
    2. 如果股票不满足条件：使用原始分数（不应用EMA）
    
    Args:
        predictions_df: 预测DataFrame，包含date, ticker, prediction列
        model_name: 模型名称
        ema_history: 历史预测分数字典 {model_name: {ticker: [S_t, S_{t-1}, S_{t-2}]}}
        weights: EMA权重 (today, t-1, t-2)，默认(0.6, 0.3, 0.1)
        top_n: Top N阈值，默认300
        min_days_in_top: 最少需要在top N的天数，默认3
    
    Returns:
        添加了prediction_smooth列的DataFrame
    """
    if model_name not in ema_history:
        ema_history[model_name] = {}

    weights = list(weights)
    max_history = max(len(weights) - 1, 0)
    max_rank_hist = max(min_days_in_top - 1, 0)

    predictions_df = predictions_df.copy()
    predictions_df['prediction_smooth'] = np.nan
    predictions_df['rank_today'] = np.nan  # 今天的排名
    predictions_df['in_top300_3days'] = False  # 是否连续3天在top300

    # 按日期排序
    predictions_df = predictions_df.sort_values('date')

    # 存储每只股票的历史排名（用于判断是否连续在top300）
    rank_history = {}  # {ticker: [rank_t, rank_t-1, ...]}
    
    for date, group in predictions_df.groupby('date'):
        # 优化：使用numpy部分排序，只计算top300的排名（减少运算量）
        # 对于其他股票，只需要知道是否在top300即可
        prediction_values = group['prediction'].values
        n_stocks = len(group)
        
        if n_stocks > top_n:
            # 使用numpy argpartition：O(N)而不是O(N log N)
            # 找到top300的索引
            top300_indices = np.argpartition(prediction_values, -top_n)[-top_n:]
            # 对top300进行排序（只排序300个，O(300 log 300)）
            top300_sorted = np.argsort(prediction_values[top300_indices])[::-1]
            top300_indices_sorted = top300_indices[top300_sorted]
            
            # 创建排名映射
            rank_map = {}
            for rank, orig_idx in enumerate(top300_indices_sorted, start=1):
                actual_idx = group.index[orig_idx]
                rank_map[actual_idx] = rank
            
            # 对于不在top300的股票，排名设为top_n+1（表示不在top300）
            for idx in group.index:
                if idx not in rank_map:
                    rank_map[idx] = top_n + 1
        else:
            # 如果股票总数少于top_n，直接排序
            group_sorted = group.sort_values('prediction', ascending=False)
            rank_map = {idx: rank + 1 for rank, idx in enumerate(group_sorted.index)}
        
        # 更新predictions_df中的排名
        for idx in group.index:
            predictions_df.loc[idx, 'rank_today'] = rank_map.get(idx, top_n + 1)
        
        # 对每只股票计算EMA
        for idx, row in group.iterrows():
            ticker = row['ticker']
            score_today = row['prediction']
            rank_today = predictions_df.loc[idx, 'rank_today']

            # 初始化排名历史
            if ticker not in rank_history:
                rank_history[ticker] = []

            # 获取历史分数和排名
            if ticker not in ema_history[model_name]:
                ema_history[model_name][ticker] = []

            history = ema_history[model_name][ticker]
            rank_hist = rank_history[ticker]

            # 判断是否连续N天在top300
            is_in_top_n_days = False
            if len(rank_hist) >= min_days_in_top - 1:
                all_ranks = [rank_today] + rank_hist[:min_days_in_top - 1]
                is_in_top_n_days = all(r <= top_n for r in all_ranks if not pd.isna(r))

            # 计算平滑分数
            if pd.isna(score_today):
                smooth_score = np.nan
            elif not is_in_top_n_days:
                smooth_score = score_today
            else:
                smooth_score = _compute_weighted_score(score_today, history, weights)

            predictions_df.loc[idx, 'prediction_smooth'] = smooth_score
            predictions_df.loc[idx, 'in_top300_3days'] = is_in_top_n_days

            # 更新历史（保留最近L-1天）
            history.insert(0, score_today)
            if len(history) > max_history:
                history.pop()

            # 更新排名历史（保留最近min_days_in_top-1天）
            rank_hist.insert(0, rank_today)
            if len(rank_hist) > max_rank_hist:
                rank_hist.pop()

    return predictions_df


def apply_ema_smoothing_top300_alternative(
    predictions_df: pd.DataFrame,
    model_name: str,
    ema_history: dict,
    weights: Sequence[float] = (0.588, 0.412),
    top_n: int = 300
) -> pd.DataFrame:
    """
    替代方案：只对今天在top300的股票应用EMA（更简单，L2 B0.7）

    这个方案更简单，只检查今天是否在top300，如果在就应用EMA
    
    Args:
        predictions_df: 预测DataFrame
        model_name: 模型名称
        ema_history: 历史记录
        weights: EMA权重
        top_n: Top N阈值
    
    Returns:
        添加了prediction_smooth列的DataFrame
    """
    if model_name not in ema_history:
        ema_history[model_name] = {}
    
    predictions_df = predictions_df.copy()
    predictions_df['prediction_smooth'] = np.nan
    
    predictions_df = predictions_df.sort_values('date')
    
    for date, group in predictions_df.groupby('date'):
        # 计算今天的排名
        group_sorted = group.sort_values('prediction', ascending=False).reset_index(drop=True)
        group_sorted['rank_today'] = group_sorted.index + 1
        
        # 创建排名映射
        rank_map = {}
        for idx, row in group_sorted.iterrows():
            orig_idx = group.index[idx]
            rank_map[orig_idx] = row['rank_today']
        
        for idx, row in group.iterrows():
            ticker = row['ticker']
            score_today = row['prediction']
            rank_today = rank_map.get(idx, np.nan)
            
            # 只对top300的股票应用EMA
            if pd.isna(rank_today) or rank_today > top_n:
                # 不在top300：使用原始分数
                smooth_score = score_today
            else:
                # 在top300：应用EMA
                if ticker not in ema_history[model_name]:
                    ema_history[model_name][ticker] = []
                
                history = ema_history[model_name][ticker]
                
                if len(history) == 0:
                    smooth_score = score_today
                elif len(history) == 1:
                    if pd.isna(history[0]):
                        smooth_score = score_today
                    else:
                        smooth_score = weights[0] * score_today + weights[1] * history[0]
                else:
                    hist_0 = history[0] if not pd.isna(history[0]) else 0.0
                    hist_1 = history[1] if not pd.isna(history[1]) else 0.0
                    smooth_score = (weights[0] * score_today + 
                                   weights[1] * hist_0 + 
                                   weights[2] * hist_1)
                
                # 更新历史
                history.insert(0, score_today)
                if len(history) > 2:
                    history.pop()
            
            predictions_df.loc[idx, 'prediction_smooth'] = smooth_score
    
    return predictions_df
