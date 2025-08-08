#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模型融合策略模块
基于Sharpe比例动态权重分配，融合BMA和LSTM策略信号
实现每周一自动权重更新和组合信号生成
"""

import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from integrated_ensemble_strategy import IntegratedEnsembleStrategy

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class EnsembleStrategy:
    """双模型融合策略类 - 重构版本"""
    
    def __init__(self, weights_file="weights.json", lookback_weeks=12):
        """
        初始化融合策略
        
        Args:
            weights_file: 权重保存文件
            lookback_weeks: Sharpe计算回望周数
        """
        self.weights_file = weights_file
        self.lookback_weeks = lookback_weeks
        self.logger = logger
        
        # 使用集成策略
        self.integrated_strategy = IntegratedEnsembleStrategy(weights_file, lookback_weeks)
        
        # 当前权重
        self.current_weights = {"w_bma": 0.5, "w_lstm": 0.5, "date": None}
        
        # 加载已保存的权重
        self._load_weights()
    
    def _load_weights(self):
        """加载已保存的权重"""
        try:
            if Path(self.weights_file).exists():
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    self.current_weights = json.load(f)
                self.logger.info(f"[融合策略] 加载权重: BMA={self.current_weights['w_bma']:.3f}, LSTM={self.current_weights['w_lstm']:.3f}")
            else:
                self.logger.info("[融合策略] 权重文件不存在，使用默认权重 (0.5, 0.5)")
        except Exception as e:
            self.logger.warning(f"[融合策略] 加载权重失败: {e}，使用默认权重")
    
    def compute_rolling_sharpe(self, net_value: pd.Series, risk_free_rate=0.0) -> float:
        """
        计算滚动Sharpe比率
        
        Args:
            net_value: 净值序列
            risk_free_rate: 无风险利率
            
        Returns:
            年化Sharpe比率
        """
        try:
            if net_value.empty or len(net_value) < 2:
                return 0.0
            
            # 计算日度收益率
            returns = net_value.pct_change().dropna()
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            # 计算年化Sharpe比率
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            return float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0
            
        except Exception as e:
            self.logger.error(f"[融合策略] 计算Sharpe比率失败: {e}")
            return 0.0
    
    def update_weights(self, tickers: List[str] = None, force_update: bool = False) -> Tuple[float, float]:
        """
        更新Sharpe权重 - 使用集成策略
        
        Args:
            tickers: 股票列表
            force_update: 是否强制更新
            
        Returns:
            (w_bma, w_lstm) 权重元组
        """
        try:
            # 设置默认股票池
            if tickers is None:
                tickers = self._get_default_tickers()
            
            # 使用集成策略更新权重
            w_bma, w_lstm = self.integrated_strategy.update_weights_by_performance(tickers, force_update)
            
            # 同步权重到当前对象
            self.current_weights = self.integrated_strategy.current_weights.copy()
            
            return w_bma, w_lstm
            
        except Exception as e:
            self.logger.error(f"[融合策略] 更新权重失败: {e}")
            return 0.5, 0.5
    
    def generate_ensemble_signals(self, tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, float]:
        """
        生成融合信号 - 使用集成策略
        
        Args:
            tickers: 股票列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            融合信号字典 {ticker: signal_strength}
        """
        try:
            # 先训练模型（如果尚未训练）
            if not hasattr(self.integrated_strategy, 'bma_strategy') or self.integrated_strategy.bma_strategy.model is None:
                self.logger.info("[融合策略] 需要先训练模型...")
                
                # 下载训练数据
                end_date_train = datetime.now().strftime('%Y-%m-%d')
                start_date_train = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                
                train_data = self.integrated_strategy.download_stock_data(tickers, start_date_train, end_date_train)
                
                if train_data:
                    self.integrated_strategy.train_integrated_models(train_data)
                else:
                    self.logger.warning("[融合策略] 无训练数据")
                    return {}
            
            # 使用集成策略生成信号
            ensemble_signals = self.integrated_strategy.generate_ensemble_signals(tickers)
            
            # 同步权重
            self.current_weights = self.integrated_strategy.current_weights.copy()
            
            self.logger.info(f"[融合策略] 生成 {len(ensemble_signals)} 个融合信号")
            
            return ensemble_signals
            
        except Exception as e:
            self.logger.error(f"[融合策略] 生成融合信号失败: {e}")
            return {}
    
    def _normalize_signal(self, signal: float, all_signals: List[float]) -> float:
        """标准化信号到 [0, 1] 范围"""
        try:
            if not all_signals or len(all_signals) <= 1:
                return 0.5
                
            min_val = min(all_signals)
            max_val = max(all_signals)
            
            if max_val == min_val:
                return 0.5
                
            normalized = (signal - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
            
        except:
            return 0.5
    
    def _get_default_tickers(self) -> List[str]:
        """获取默认股票池"""
        try:
            # 尝试从默认股票池文件加载
            default_pool_file = "default_stock_pool.json"
            if Path(default_pool_file).exists():
                with open(default_pool_file, 'r', encoding='utf-8') as f:
                    pool_data = json.load(f)
                    return pool_data.get('default_stock_pool', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
            else:
                # 使用硬编码的默认股票池
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    def get_current_weights(self) -> Dict:
        """获取当前权重信息"""
        return self.current_weights.copy()
    
    def backtest_ensemble(self, start_date: str, end_date: str, tickers: List[str] = None) -> pd.Series:
        """
        融合策略回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            tickers: 股票列表
            
        Returns:
            净值序列
        """
        try:
            if tickers is None:
                tickers = self._get_default_tickers()
                
            self.logger.info(f"[融合策略] 开始融合策略回测: {start_date} 到 {end_date}")
            
            # 获取BMA和LSTM净值序列
            nv_bma = self.bma_strategy.backtest_bma(start_date, end_date, tickers)
            nv_lstm = self.lstm_strategy.backtest_lstm(start_date, end_date, tickers)
            
            # 权重融合净值
            w_bma = self.current_weights["w_bma"]
            w_lstm = self.current_weights["w_lstm"]
            
            # 对齐时间序列
            common_dates = nv_bma.index.intersection(nv_lstm.index)
            if len(common_dates) == 0:
                self.logger.warning("[融合策略] 无法对齐BMA和LSTM时间序列")
                return pd.Series([1.0], index=[pd.Timestamp.now()])
            
            nv_bma_aligned = nv_bma.loc[common_dates]
            nv_lstm_aligned = nv_lstm.loc[common_dates]
            
            # 计算融合净值 
            ensemble_returns = (w_bma * nv_bma_aligned.pct_change() + 
                               w_lstm * nv_lstm_aligned.pct_change()).fillna(0)
            
            ensemble_nv = (1 + ensemble_returns).cumprod()
            ensemble_nv.iloc[0] = 1.0
            
            self.logger.info(f"[融合策略] 融合策略回测完成，最终净值: {ensemble_nv.iloc[-1]:.4f}")
            
            return ensemble_nv
            
        except Exception as e:
            self.logger.error(f"[融合策略] 融合策略回测失败: {e}")
            # 返回平坦序列
            return pd.Series([1.0], index=[pd.Timestamp.now()])


def test_ensemble_strategy():
    """测试融合策略"""
    logger.info("=== 测试双模型融合策略 ===")
    
    ensemble = EnsembleStrategy()
    
    # 测试参数
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # 1. 测试权重更新
    logger.info("1. 测试权重更新...")
    w_bma, w_lstm = ensemble.update_weights(tickers, force_update=True)
    print(f"更新后权重: BMA={w_bma:.3f}, LSTM={w_lstm:.3f}")
    
    # 2. 测试信号生成
    logger.info("2. 测试信号生成...")
    signals = ensemble.generate_ensemble_signals(tickers[:3])  # 使用少量股票测试
    print(f"融合信号: {signals}")
    
    # 3. 测试融合回测
    logger.info("3. 测试融合回测...")
    start_date = "2024-07-01"
    end_date = "2024-08-01"
    nv = ensemble.backtest_ensemble(start_date, end_date, tickers[:3])
    print(f"融合回测结果:")
    print(f"期间: {start_date} 到 {end_date}")
    print(f"最终净值: {nv.iloc[-1]:.4f}")
    print(f"总收益: {(nv.iloc[-1] - 1) * 100:.2f}%")
    
    return ensemble


if __name__ == "__main__":
    test_ensemble_strategy()