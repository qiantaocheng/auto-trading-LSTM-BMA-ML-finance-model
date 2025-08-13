#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
展示投资组合分析结果
"""

import pandas as pd
import json
import os
from datetime import datetime

def show_latest_results():
    """展示最新的投资分析结果"""
    
    # 查找最新的结果文件
    result_dir = "result"
    if not os.path.exists(result_dir):
        print("❌ 结果目录不存在")
        return
    
    # 查找最新的portfolio_details文件
    portfolio_files = [f for f in os.listdir(result_dir) if f.startswith("portfolio_details_")]
    if not portfolio_files:
        print("❌ 没有找到投资组合结果文件")
        return
    
    latest_portfolio_file = sorted(portfolio_files)[-1]
    timestamp = latest_portfolio_file.replace("portfolio_details_", "").replace(".json", "")
    
    # 读取投资组合详情
    portfolio_path = os.path.join(result_dir, latest_portfolio_file)
    with open(portfolio_path, 'r', encoding='utf-8') as f:
        portfolio_data = json.load(f)
    
    # 读取推荐数据
    recommendations_file = f"ultra_enhanced_recommendations_{timestamp}.xlsx"
    recommendations_path = os.path.join(result_dir, recommendations_file)
    
    if not os.path.exists(recommendations_path):
        print(f"❌ 推荐文件不存在: {recommendations_file}")
        return
    
    recommendations = pd.read_excel(recommendations_path)
    
    print('\n' + '='*80)
    print('           📊 BMA Ultra Enhanced 投资组合分析结果')
    print('='*80)
    
    # 时间戳
    try:
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        print(f'📅 分析时间: {dt.strftime("%Y年%m月%d日 %H:%M:%S")}')
    except:
        print(f'📅 分析时间: {timestamp}')
    
    # 投资组合概览
    metrics = portfolio_data['portfolio_metrics']
    print(f'\n🎯 投资组合概览:')
    print(f'   预期年化收益率: {metrics["expected_return"]*100:.2f}%')
    print(f'   预期年化波动率: {metrics["expected_volatility"]*100:.2f}%')
    print(f'   夏普比率: {metrics["sharpe_ratio"]:.3f}')
    print(f'   有效持仓数: {int(metrics["effective_positions"])}只股票')
    print(f'   最大单只权重: {metrics["max_weight"]*100:.1f}%')
    print(f'   净敞口: {metrics["net_exposure"]*100:.1f}%')
    
    # 投资建议详情
    print(f'\n💰 投资建议明细:')
    print('-'*80)
    print(f"{'排名':<4} {'股票':<6} {'权重':<8} {'最新价格':<12} {'预测信号':<10} {'推荐理由'}")
    print('-'*80)
    
    for _, row in recommendations.iterrows():
        rank = int(row['rank'])
        ticker = row['ticker']
        weight = f"{row['weight']*100:.1f}%"
        price = f"${row['latest_price']:.2f}"
        signal = f"{row['prediction_signal']:.3f}"
        reason = row['recommendation_reason']
        
        print(f"{rank:<4} {ticker:<6} {weight:<8} {price:<12} {signal:<10} {reason}")
    
    print('-'*80)
    
    # 权重分布
    print(f'\n📊 权重分布:')
    total_allocated = recommendations['weight'].sum()
    print(f'   总配置权重: {total_allocated*100:.1f}%')
    
    # 按权重分组
    high_weight = recommendations[recommendations['weight'] >= 0.1]
    med_weight = recommendations[(recommendations['weight'] >= 0.05) & (recommendations['weight'] < 0.1)]
    low_weight = recommendations[recommendations['weight'] < 0.05]
    
    if len(high_weight) > 0:
        print(f'   高权重配置 (≥10%): {list(high_weight["ticker"])} - 合计 {high_weight["weight"].sum()*100:.1f}%')
    if len(med_weight) > 0:
        print(f'   中等权重配置 (5-10%): {list(med_weight["ticker"])} - 合计 {med_weight["weight"].sum()*100:.1f}%')
    if len(low_weight) > 0:
        print(f'   低权重配置 (<5%): {list(low_weight["ticker"])} - 合计 {low_weight["weight"].sum()*100:.1f}%')
    
    # 信号分析
    strong_buy = recommendations[recommendations['prediction_signal'] > 0.1]
    buy = recommendations[(recommendations['prediction_signal'] > 0) & (recommendations['prediction_signal'] <= 0.1)]
    neutral = recommendations[abs(recommendations['prediction_signal']) <= 0.05]
    weak_sell = recommendations[recommendations['prediction_signal'] < -0.05]
    
    print(f'\n📈 信号分布分析:')
    if len(strong_buy) > 0:
        print(f'   🟢 强烈买入 (>0.1): {list(strong_buy["ticker"])} - 平均信号 {strong_buy["prediction_signal"].mean():.3f}')
    if len(buy) > 0:
        print(f'   🟡 买入 (0-0.1): {list(buy["ticker"])} - 平均信号 {buy["prediction_signal"].mean():.3f}')
    if len(neutral) > 0:
        print(f'   ⚪ 中性 (±0.05): {list(neutral["ticker"])} - 平均信号 {neutral["prediction_signal"].mean():.3f}')
    if len(weak_sell) > 0:
        print(f'   🔴 弱卖出 (<-0.05): {list(weak_sell["ticker"])} - 平均信号 {weak_sell["prediction_signal"].mean():.3f}')
    
    # 价格变动分析
    print(f'\n📊 近期价格表现:')
    print(f'   1日平均涨跌: {recommendations["price_change_1d"].mean()*100:.2f}%')
    print(f'   5日平均涨跌: {recommendations["price_change_5d"].mean()*100:.2f}%')
    
    recent_gainers = recommendations[recommendations['price_change_5d'] > 0]
    recent_decliners = recommendations[recommendations['price_change_5d'] <= 0]
    
    if len(recent_gainers) > 0:
        print(f'   近5日上涨股票: {list(recent_gainers["ticker"])}')
    if len(recent_decliners) > 0:
        print(f'   近5日下跌股票: {list(recent_decliners["ticker"])}')
    
    # 优化信息
    opt_info = portfolio_data.get('optimization_info', {})
    print(f'\n🔧 优化器信息:')
    print(f'   目标函数值: {opt_info.get("objective_value", "N/A")}')
    print(f'   迭代次数: {opt_info.get("iterations", "N/A")}')
    print(f'   优化状态: {opt_info.get("optimization_message", "N/A")}')
    
    # 风险提示
    print(f'\n⚠️  重要提示:')
    print(f'   1. 此为量化模型分析结果，仅供参考')
    print(f'   2. 投资有风险，决策需谨慎')
    print(f'   3. 建议结合基本面分析和市场环境')
    print(f'   4. 请根据个人风险承受能力调整仓位')
    
    print('\n' + '='*80)
    print(f'💾 详细结果已保存至: {recommendations_path}')
    print('='*80)

if __name__ == "__main__":
    show_latest_results()
