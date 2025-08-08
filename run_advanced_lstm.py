#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级LSTM多日预测模型运行示例
展示如何使用完整的增强功能

Features demonstrated:
- 完整的特征工程流水线
- Optuna超参数优化
- 多模型融合
- 在线学习
- 性能评估和保存

Authors: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# 导入我们的高级LSTM模型
from lstm_multi_day_advanced import AdvancedLSTMMultiDayModel
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_factors(stock_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """创建示例技术因子"""
    factors = pd.DataFrame(index=stock_data.index)
    
    # 价格相关因子
    factors['close'] = stock_data['Close']
    factors['high'] = stock_data['High']
    factors['low'] = stock_data['Low'] 
    factors['volume'] = stock_data['Volume']
    
    # 收益率因子
    factors['returns'] = stock_data['Close'].pct_change()
    factors['returns_5d'] = stock_data['Close'].pct_change(5)
    factors['returns_10d'] = stock_data['Close'].pct_change(10)
    
    # 技术指标因子
    factors['sma_5'] = stock_data['Close'].rolling(5).mean() / stock_data['Close'] - 1
    factors['sma_10'] = stock_data['Close'].rolling(10).mean() / stock_data['Close'] - 1
    factors['sma_20'] = stock_data['Close'].rolling(20).mean() / stock_data['Close'] - 1
    
    # 波动率因子
    factors['volatility_5d'] = factors['returns'].rolling(5).std()
    factors['volatility_10d'] = factors['returns'].rolling(10).std()
    factors['volatility_20d'] = factors['returns'].rolling(20).std()
    
    # 成交量因子
    factors['volume_ratio_5d'] = stock_data['Volume'] / stock_data['Volume'].rolling(5).mean() - 1
    factors['volume_ratio_10d'] = stock_data['Volume'] / stock_data['Volume'].rolling(10).mean() - 1
    
    # RSI因子
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    factors['rsi'] = 100 - (100 / (1 + rs))
    factors['rsi_normalized'] = (factors['rsi'] - 50) / 50
    
    # MACD因子
    exp1 = stock_data['Close'].ewm(span=12).mean()
    exp2 = stock_data['Close'].ewm(span=26).mean()
    factors['macd'] = exp1 - exp2
    factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
    factors['macd_histogram'] = factors['macd'] - factors['macd_signal']
    
    # 布林带因子
    bb_middle = stock_data['Close'].rolling(20).mean()
    bb_std = stock_data['Close'].rolling(20).std()
    factors['bb_upper'] = (bb_middle + 2 * bb_std - stock_data['Close']) / stock_data['Close']
    factors['bb_lower'] = (stock_data['Close'] - bb_middle + 2 * bb_std) / stock_data['Close']
    factors['bb_width'] = (4 * bb_std) / bb_middle
    
    # 动量因子
    factors['momentum_3d'] = stock_data['Close'] / stock_data['Close'].shift(3) - 1
    factors['momentum_7d'] = stock_data['Close'] / stock_data['Close'].shift(7) - 1
    factors['momentum_14d'] = stock_data['Close'] / stock_data['Close'].shift(14) - 1
    
    # 标准化所有因子
    for col in factors.columns:
        if col != 'returns':  # 保持returns原始值作为目标变量
            factors[col] = (factors[col] - factors[col].rolling(window).mean()) / factors[col].rolling(window).std()
    
    return factors.dropna()

def create_market_cap_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """创建市值数据（示例）"""
    # 简单估算：股价 * 假设流通股本
    shares_outstanding = 1_000_000_000  # 假设10亿股
    market_cap = stock_data['Close'] * shares_outstanding
    return pd.DataFrame({'market_cap': market_cap}, index=stock_data.index)

def download_stock_data(tickers: list, start_date: str, end_date: str) -> dict:
    """下载股票数据"""
    logger.info(f"下载股票数据: {tickers}")
    
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                stock_data[ticker] = data
                logger.info(f"{ticker}: {len(data)} 个交易日")
            else:
                logger.warning(f"{ticker}: 无数据")
        except Exception as e:
            logger.error(f"下载 {ticker} 失败: {e}")
    
    return stock_data

def run_advanced_lstm_example():
    """运行高级LSTM示例"""
    logger.info("=" * 60)
    logger.info("高级LSTM多日预测模型示例")
    logger.info("=" * 60)
    
    # 1. 参数设置
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2024-08-06'
    prediction_days = 5
    
    # 2. 下载数据
    stock_data = download_stock_data(tickers, start_date, end_date)
    
    if not stock_data:
        logger.error("没有有效的股票数据")
        return
    
    # 3. 为每只股票运行高级LSTM分析
    results = {}
    
    for ticker, data in stock_data.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"分析股票: {ticker}")
        logger.info(f"{'='*40}")
        
        try:
            # 创建因子数据
            logger.info("创建技术因子...")
            factors_df = create_sample_factors(data)
            returns_df = factors_df[['returns']].copy()
            
            # 创建市值数据（示例）
            market_cap_df = create_market_cap_data(data)
            
            # 数据分割
            split_idx = int(len(factors_df) * 0.8)
            
            factors_train = factors_df.iloc[:split_idx]
            factors_test = factors_df.iloc[split_idx:]
            
            returns_train = returns_df.iloc[:split_idx]
            returns_test = returns_df.iloc[split_idx:]
            
            market_cap_train = market_cap_df.iloc[:split_idx]
            market_cap_test = market_cap_df.iloc[split_idx:]
            
            logger.info(f"训练集: {len(factors_train)} 个样本")
            logger.info(f"测试集: {len(factors_test)} 个样本")
            
            # 创建高级LSTM模型
            logger.info("创建高级LSTM模型...")
            model = AdvancedLSTMMultiDayModel(
                prediction_days=prediction_days,
                lstm_window=20,
                enable_optimization=True,  # 启用超参数优化
                enable_ensemble=True,     # 启用模型集成
                enable_online_learning=True  # 启用在线学习
            )
            
            # 高级特征工程
            logger.info("执行高级特征工程...")
            processed_factors_train = model.prepare_advanced_features(
                factors_train,
                returns_train,
                market_cap_train
            )
            
            processed_factors_test = model.feature_engineer.transform_new_data(
                factors_test.reindex(processed_factors_train.columns, axis=1).fillna(0)
            )
            
            # 创建序列数据
            logger.info("创建LSTM序列数据...")
            X_train, y_train = model.create_multi_day_sequences(
                processed_factors_train, returns_train
            )
            
            X_test, y_test = model.create_multi_day_sequences(
                processed_factors_test, returns_test
            )
            
            if X_train.shape[0] < 50:
                logger.warning(f"{ticker}: 训练样本不足，跳过")
                continue
            
            # 划分训练和验证集
            val_split = int(len(X_train) * 0.8)
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train = X_train[:val_split]
            y_train = y_train[:val_split]
            
            returns_train_seq = returns_train['returns'].values[model.lstm_window:model.lstm_window+len(X_train)]
            returns_val_seq = returns_train['returns'].values[model.lstm_window+len(X_train):model.lstm_window+len(X_train)+len(X_val)]
            
            logger.info(f"训练序列: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"验证序列: X={X_val.shape}, y={y_val.shape}")
            logger.info(f"测试序列: X={X_test.shape}, y={y_test.shape}")
            
            # 训练模型
            logger.info("开始训练高级LSTM模型...")
            model.train_advanced_model(
                X_train, y_train,
                X_val, y_val,
                returns_train_seq,
                returns_val_seq
            )
            
            # 预测
            logger.info("执行预测...")
            predictions = model.predict_advanced(X_test)
            
            # 评估结果
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            if len(predictions) > 0 and len(y_test) > 0:
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                
                # 计算方向准确率
                pred_direction = np.sign(predictions.mean(axis=1))
                actual_direction = np.sign(y_test.mean(axis=1))
                direction_accuracy = np.mean(pred_direction == actual_direction)
                
                results[ticker] = {
                    'mse': mse,
                    'mae': mae,
                    'direction_accuracy': direction_accuracy,
                    'predictions': predictions[:10],  # 保存前10个预测
                    'actuals': y_test[:10],
                    'model_performance': model.model_performance
                }
                
                logger.info(f"\n{ticker} 模型性能:")
                logger.info(f"  MSE: {mse:.6f}")
                logger.info(f"  MAE: {mae:.6f}")
                logger.info(f"  方向准确率: {direction_accuracy:.3f}")
            
            # 保存模型
            model_path = f"models/advanced_lstm_{ticker}.pkl"
            os.makedirs("models", exist_ok=True)
            model.save_model(model_path)
            logger.info(f"模型已保存: {model_path}")
            
            # 在线学习演示（使用测试集的前10个样本）
            if len(X_test) >= 10:
                logger.info("演示在线学习...")
                X_online = X_test[:10]
                y_online = y_test[:10]
                returns_online = returns_test['returns'].values[:10] if len(returns_test) >= 10 else None
                
                model.online_update(X_online, y_online, returns_online)
            
        except Exception as e:
            logger.error(f"分析 {ticker} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("汇总结果")
    logger.info(f"{'='*60}")
    
    if results:
        # 创建结果DataFrame
        result_data = []
        for ticker, metrics in results.items():
            result_data.append({
                'Ticker': ticker,
                'MSE': metrics['mse'],
                'MAE': metrics['mae'],
                'Direction_Accuracy': metrics['direction_accuracy']
            })
        
        results_df = pd.DataFrame(result_data)
        print(results_df.to_string(index=False))
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"result/advanced_lstm_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 为每只股票保存详细预测
            for ticker, metrics in results.items():
                if 'predictions' in metrics:
                    pred_df = pd.DataFrame({
                        **{f'Prediction_Day_{i+1}': metrics['predictions'][:, i] 
                           for i in range(prediction_days)},
                        **{f'Actual_Day_{i+1}': metrics['actuals'][:, i] 
                           for i in range(prediction_days)}
                    })
                    pred_df.to_excel(writer, sheet_name=f'{ticker}_Predictions', index=False)
        
        logger.info(f"结果已保存: {results_file}")
        
        # 计算平均性能
        avg_mse = results_df['MSE'].mean()
        avg_mae = results_df['MAE'].mean()
        avg_accuracy = results_df['Direction_Accuracy'].mean()
        
        logger.info(f"\n平均性能:")
        logger.info(f"  平均MSE: {avg_mse:.6f}")
        logger.info(f"  平均MAE: {avg_mae:.6f}")
        logger.info(f"  平均方向准确率: {avg_accuracy:.3f}")
        
    else:
        logger.warning("没有成功分析的股票")
    
    logger.info(f"\n{'='*60}")
    logger.info("高级LSTM多日预测分析完成")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    # 检查必要的依赖
    try:
        import optuna
        import tensorflow as tf
        logger.info("所有依赖已安装")
    except ImportError as e:
        logger.warning(f"缺少可选依赖: {e}")
        logger.info("某些高级功能可能不可用，但基础功能仍可正常使用")
    
    # 运行示例
    run_advanced_lstm_example()