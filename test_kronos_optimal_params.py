"""
测试Kronos最优参数选择
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from kronos.auto_trainer import KronosAutoTrainer
from kronos.utils import prepare_kline_data

def test_optimal_params(symbol='AAPL'):
    """测试并显示最优训练参数"""

    # 初始化训练器
    trainer = KronosAutoTrainer('kronos/training_config.yaml')

    # 获取数据
    print(f"正在获取{symbol}数据...")
    data = prepare_kline_data(symbol, period="6mo", interval="1d")

    if data is None or data.empty:
        print(f"无法获取{symbol}数据")
        return

    print(f"成功获取{len(data)}条数据记录")

    # 分析市场状况
    market_condition = trainer.analyze_market_condition(data)
    data_quality = trainer.evaluate_data_quality(data)

    # 获取最优配置
    optimal_config = trainer.select_optimal_config(market_condition, data_quality)

    # 显示结果
    print("\n" + "="*80)
    print(f"📊 Kronos最优训练参数推荐 - {symbol}")
    print("="*80)

    print(f"\n📈 市场分析:")
    print(f"  • 市场状况: {market_condition}")
    market_descriptions = {
        'trending': '趋势明显，适合趋势跟踪策略',
        'ranging': '区间震荡，适合均值回归策略',
        'volatile': '高波动率，需要更严格的风险控制'
    }
    print(f"    说明: {market_descriptions.get(market_condition, '未知')}")

    print(f"\n💾 数据质量: {data_quality}")
    quality_descriptions = {
        'high_quality': '实时高质量数据，可使用所有策略',
        'delayed': '延迟数据，建议使用日线或更长周期',
        'limited': '数据受限，仅使用保守策略'
    }
    print(f"    说明: {quality_descriptions.get(data_quality, '未知')}")

    print(f"\n⚙️ 推荐配置:")
    print(f"  📊 数据参数:")
    print(f"    • 时间间隔: {optimal_config['data']['interval']}")
    print(f"    • 历史周期: {optimal_config['data']['period']}")
    print(f"    • 最少数据点: {optimal_config['data']['min_data_points']}")

    print(f"\n  🧠 模型参数:")
    print(f"    • 输入序列长度: {optimal_config['model']['sequence_length']} 个周期")
    print(f"    • 预测长度: {optimal_config['model']['prediction_length']} 个周期")
    print(f"    • 特征: {', '.join(optimal_config['model']['features'])}")

    print(f"\n  🎯 训练参数:")
    print(f"    • 训练轮数: {optimal_config['training']['epochs']}")
    print(f"    • 批次大小: {optimal_config['training']['batch_size']}")
    print(f"    • 学习率: {optimal_config['training']['learning_rate']}")
    print(f"    • 早停: {'启用' if optimal_config['training']['early_stopping'] else '禁用'}")
    if optimal_config['training']['early_stopping']:
        print(f"    • 耐心值: {optimal_config['training']['patience']}")
    print(f"    • 验证集比例: {optimal_config['training']['validation_split']*100:.0f}%")

    print(f"\n  🔄 重训练策略:")
    print(f"    • 定期频率: {optimal_config['retrain']['frequency']}")
    print(f"    • MAE触发阈值: {optimal_config['retrain']['trigger_mae']*100:.1f}%")
    print(f"    • 准确率触发阈值: {optimal_config['retrain']['trigger_accuracy']*100:.0f}%")

    # 计算一些额外的统计信息
    print(f"\n📊 数据统计:")
    close_prices = data['close'].values
    returns = pd.Series(close_prices).pct_change().dropna()

    print(f"  • 当前价格: ${close_prices[-1]:.2f}")
    print(f"  • 平均日收益率: {returns.mean()*100:.3f}%")
    print(f"  • 日波动率: {returns.std()*100:.2f}%")
    print(f"  • 年化波动率: {returns.std()*np.sqrt(252)*100:.1f}%")
    print(f"  • 最大单日涨幅: {returns.max()*100:.2f}%")
    print(f"  • 最大单日跌幅: {returns.min()*100:.2f}%")

    # 根据配置计算预期性能
    print(f"\n🎯 预期性能:")
    if optimal_config['data']['interval'] == '1d':
        print(f"  • 预测周期: {optimal_config['model']['prediction_length']} 天")
        print(f"  • 输入历史: {optimal_config['model']['sequence_length']} 天")
        print(f"  • 建议持仓时间: {optimal_config['model']['prediction_length']//2} - {optimal_config['model']['prediction_length']} 天")
    elif optimal_config['data']['interval'] == '4h':
        pred_days = optimal_config['model']['prediction_length'] * 4 / 24
        input_days = optimal_config['model']['sequence_length'] * 4 / 24
        print(f"  • 预测周期: {pred_days:.1f} 天")
        print(f"  • 输入历史: {input_days:.1f} 天")
        print(f"  • 建议持仓时间: {pred_days/2:.1f} - {pred_days:.1f} 天")
    elif optimal_config['data']['interval'] == '1h':
        pred_days = optimal_config['model']['prediction_length'] / 24
        input_days = optimal_config['model']['sequence_length'] / 24
        print(f"  • 预测周期: {pred_days:.1f} 天")
        print(f"  • 输入历史: {input_days:.1f} 天")
        print(f"  • 建议持仓时间: {pred_days/2:.1f} - {pred_days:.1f} 天")

    print(f"\n💡 建议:")
    if market_condition == 'trending':
        print("  • 市场趋势明显，可适当增加仓位")
        print("  • 使用趋势跟踪策略，设置移动止损")
        print("  • 关注突破信号，顺势交易")
    elif market_condition == 'ranging':
        print("  • 市场震荡，建议降低仓位")
        print("  • 在支撑位买入，阻力位卖出")
        print("  • 设置严格止损，避免假突破")
    elif market_condition == 'volatile':
        print("  • 市场波动大，控制风险敞口")
        print("  • 缩短持仓周期，快进快出")
        print("  • 使用更严格的止损策略")

    print("\n" + "="*80)

    return optimal_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试Kronos最优参数')
    parser.add_argument('--symbol', default='AAPL', help='股票代码')
    args = parser.parse_args()

    test_optimal_params(args.symbol)