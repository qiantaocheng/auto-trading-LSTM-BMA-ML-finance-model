#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM策略回测模块
基于LSTM模型输出实现单独的策略回测，用于Sharpe权重计算
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile
import os
from typing import Dict, List, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMStrategy:
    """LSTM策略回测类"""
    
    def __init__(self, transaction_cost=0.001, initial_capital=100000):
        """
        初始化LSTM策略
        
        Args:
            transaction_cost: 交易成本（包含滑点和手续费）
            initial_capital: 初始资本
        """
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.logger = logger
        
    def run_lstm_model(self, tickers: List[str], start_date: str, end_date: str) -> Dict:
        """运行LSTM模型获取信号"""
        try:
            # 创建临时ticker文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            temp_file.write("# LSTM回测股票列表\n")
            for ticker in tickers:
                temp_file.write(f"{ticker}\n")
            temp_file.close()
            
            self.logger.info(f"[LSTM策略] 运行LSTM模型，股票数量: {len(tickers)}")
            
            # 运行LSTM模型
            cmd = [
                sys.executable, 
                "lstm_multi_day_enhanced.py",
                "--ticker-file", temp_file.name,
                "--start-date", start_date,
                "--end-date", end_date
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # 清理临时文件
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
            if result.returncode != 0:
                self.logger.warning(f"[LSTM策略] LSTM模型运行失败: {result.stderr}")
                return {}
            
            # 查找LSTM结果文件
            result_files = list(Path("result").glob("*lstm_analysis_*.xlsx"))
            if not result_files:
                result_files = list(Path("result").glob("test_multi_day_lstm_analysis_*.xlsx"))
            
            if not result_files:
                self.logger.warning("[LSTM策略] 未找到LSTM结果文件")
                return {}
                
            # 使用最新的结果文件
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            
            # 解析LSTM结果
            lstm_signals = self._parse_lstm_results(latest_file)
            self.logger.info(f"[LSTM策略] 解析到 {len(lstm_signals)} 个LSTM信号")
            
            return lstm_signals
            
        except Exception as e:
            self.logger.error(f"[LSTM策略] 运行LSTM模型失败: {e}")
            return {}
    
    def _parse_lstm_results(self, excel_file: Path) -> Dict:
        """解析LSTM Excel结果文件"""
        try:
            # 读取Excel文件的所有sheet
            excel_data = pd.ExcelFile(excel_file)
            signals = {}
            
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # 查找包含股票代码的列
                ticker_cols = [col for col in df.columns if any(keyword in col.lower() 
                              for keyword in ['symbol', 'ticker', '股票', '代码'])]
                
                # 查找包含预测或评分的列
                prediction_cols = [col for col in df.columns if any(keyword in col.lower() 
                                  for keyword in ['prediction', 'score', 'rating', '预测', '评分', 'confidence', 'weighted'])]
                
                if ticker_cols and prediction_cols:
                    ticker_col = ticker_cols[0]
                    pred_col = prediction_cols[0]
                    
                    for _, row in df.iterrows():
                        ticker = str(row[ticker_col]).strip().upper()
                        prediction = float(row[pred_col]) if pd.notna(row[pred_col]) else 0.0
                        
                        if ticker and ticker != 'NAN':
                            signals[ticker] = prediction
                            
                # 如果第一个sheet没找到，尝试其他常见格式
                elif 'ticker' in df.columns and any(col in df.columns for col in ['rating', 'prediction', 'weighted_prediction']):
                    for _, row in df.iterrows():
                        ticker = str(row['ticker']).strip().upper()
                        
                        # 优先使用加权预测，然后是评级，最后是预测
                        if 'weighted_prediction' in df.columns:
                            prediction = float(row['weighted_prediction']) if pd.notna(row['weighted_prediction']) else 0.0
                        elif 'rating' in df.columns:
                            # 如果是文本评级，转换为数值
                            rating = str(row['rating']).upper()
                            rating_map = {'STRONG_BUY': 1.0, 'BUY': 0.8, 'HOLD': 0.5, 'SELL': 0.2, 'STRONG_SELL': 0.0}
                            prediction = rating_map.get(rating, 0.5)
                        else:
                            prediction = float(row['prediction']) if pd.notna(row['prediction']) else 0.0
                            
                        if ticker and ticker != 'NAN':
                            signals[ticker] = prediction
            
            return signals
            
        except Exception as e:
            self.logger.error(f"[LSTM策略] 解析LSTM结果失败: {e}")
            return {}
    
    def backtest_lstm(self, start_date: str, end_date: str, tickers: List[str]) -> pd.Series:
        """
        LSTM策略回测，返回净值序列
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期  
            tickers: 股票列表
            
        Returns:
            净值序列 (pd.Series)
        """
        try:
            self.logger.info(f"[LSTM策略] 开始回测，期间: {start_date} 到 {end_date}")
            
            # 1. 运行LSTM模型获取信号
            lstm_signals = self.run_lstm_model(tickers, start_date, end_date)
            if not lstm_signals:
                self.logger.warning("[LSTM策略] 无LSTM信号，返回平坦净值序列")
                return self._create_flat_series(start_date, end_date)
            
            # 2. 获取股价数据
            price_data = self._get_price_data(list(lstm_signals.keys()), start_date, end_date)
            if price_data.empty:
                self.logger.warning("[LSTM策略] 无价格数据，返回平坦净值序列")
                return self._create_flat_series(start_date, end_date)
            
            # 3. 实施策略回测
            net_value = self._execute_strategy(lstm_signals, price_data)
            
            self.logger.info(f"[LSTM策略] 回测完成，最终净值: {net_value.iloc[-1]:.4f}")
            return net_value
            
        except Exception as e:
            self.logger.error(f"[LSTM策略] 回测失败: {e}")
            return self._create_flat_series(start_date, end_date)
    
    def _get_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取股价数据"""
        try:
            data_frames = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        # 使用调整后收盘价
                        prices = hist['Close'].rename(ticker)
                        data_frames.append(prices)
                        
                except Exception as e:
                    self.logger.warning(f"[LSTM策略] 获取{ticker}数据失败: {e}")
                    continue
            
            if data_frames:
                price_data = pd.concat(data_frames, axis=1).fillna(method='ffill')
                self.logger.info(f"[LSTM策略] 获取到 {len(price_data.columns)} 只股票的价格数据")
                return price_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"[LSTM策略] 获取价格数据失败: {e}")
            return pd.DataFrame()
    
    def _execute_strategy(self, signals: Dict, price_data: pd.DataFrame) -> pd.Series:
        """执行策略回测"""
        try:
            # 按信号强度排序，选择前20%的股票
            # 对于LSTM，通常值越高表示越看好
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            n_select = max(1, len(sorted_signals) // 5)  # 前20%，至少1只
            selected_tickers = [ticker for ticker, _ in sorted_signals[:n_select]]
            
            self.logger.info(f"[LSTM策略] 选择 {n_select} 只股票进行回测: {selected_tickers}")
            
            # 过滤价格数据
            available_tickers = [t for t in selected_tickers if t in price_data.columns]
            if not available_tickers:
                return self._create_flat_series_from_data(price_data)
            
            strategy_data = price_data[available_tickers].copy()
            
            # 计算每日收益率
            returns = strategy_data.pct_change().fillna(0)
            
            # 等权重投资组合
            portfolio_returns = returns.mean(axis=1)
            
            # 扣除交易成本（每日重新平衡假设）
            portfolio_returns = portfolio_returns - self.transaction_cost / 252  # 年化交易成本
            
            # 计算累积净值
            net_value = (1 + portfolio_returns).cumprod()
            net_value.iloc[0] = 1.0  # 起始净值为1
            
            return net_value
            
        except Exception as e:
            self.logger.error(f"[LSTM策略] 执行策略失败: {e}")
            return self._create_flat_series_from_data(price_data)
    
    def _create_flat_series(self, start_date: str, end_date: str) -> pd.Series:
        """创建平坦的净值序列（回测失败时使用）"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            # 只保留工作日
            business_days = date_range[date_range.dayofweek < 5]
            return pd.Series(1.0, index=business_days)
        except:
            # 如果创建失败，返回最基本的序列
            return pd.Series([1.0], index=[pd.Timestamp.now()])
    
    def _create_flat_series_from_data(self, price_data: pd.DataFrame) -> pd.Series:
        """基于已有价格数据创建平坦净值序列"""
        if not price_data.empty:
            return pd.Series(1.0, index=price_data.index)
        else:
            return pd.Series([1.0], index=[pd.Timestamp.now()])


def test_lstm_strategy():
    """测试LSTM策略"""
    logger.info("=== 测试LSTM策略回测 ===")
    
    strategy = LSTMStrategy()
    
    # 测试参数
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = "2024-07-01"
    end_date = "2024-08-01"
    
    # 运行回测
    net_value = strategy.backtest_lstm(start_date, end_date, tickers)
    
    print(f"回测结果:")
    print(f"期间: {start_date} 到 {end_date}")
    print(f"初始净值: {net_value.iloc[0]:.4f}")
    print(f"最终净值: {net_value.iloc[-1]:.4f}")
    print(f"总收益: {(net_value.iloc[-1] - 1) * 100:.2f}%")
    
    return net_value


if __name__ == "__main__":
    test_lstm_strategy()