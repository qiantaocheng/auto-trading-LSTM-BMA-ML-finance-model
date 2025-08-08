#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号生成器
将BMA和LSTM模型的预测结果转换为IBKR自动交易器可用的JSON格式

功能:
1. 从模型预测结果生成交易信号
2. 应用风险过滤和信号强度计算
3. 生成符合IBKR交易器要求的JSON配置
4. 支持动态股票池更新
"""

import json
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class TradingSignalsGenerator:
    """交易信号生成器"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 信号过滤参数
        self.min_confidence = 0.6  # 最小信号置信度
        self.max_signals_per_day = 20  # 每日最大信号数
        self.min_prediction_threshold = 0.02  # 最小预测阈值 (2%)
        
        # 从exports/high_quality_stocks.txt加载股票池
        self.stock_universe = self._load_stock_universe()
        
        self.logger.info(f"交易信号生成器初始化完成，股票池: {len(self.stock_universe)} 只")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_stock_universe(self) -> List[str]:
        """加载股票池"""
        try:
            stock_file = "exports/high_quality_stocks.txt"
            if os.path.exists(stock_file):
                with open(stock_file, 'r', encoding='utf-8') as f:
                    stocks = [line.strip() for line in f if line.strip()]
                self.logger.info(f"从 {stock_file} 加载了 {len(stocks)} 只股票")
                return stocks
            else:
                # 使用默认股票池
                default_stocks = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                    'CRM', 'ADBE', 'ORCL', 'IBM', 'INTC', 'AMD', 'COST', 'WMT'
                ]
                self.logger.warning(f"未找到股票池文件，使用默认股票池: {len(default_stocks)} 只")
                return default_stocks
                
        except Exception as e:
            self.logger.error(f"加载股票池失败: {e}")
            return ['AAPL', 'MSFT', 'GOOGL']  # 最基本的股票
    
    def generate_signals_from_bma_results(self, bma_results_file: str) -> Dict:
        """从BMA模型结果生成交易信号"""
        try:
            self.logger.info(f"从BMA结果生成信号: {bma_results_file}")
            
            # 读取BMA结果
            if bma_results_file.endswith('.xlsx'):
                df = pd.read_excel(bma_results_file)
            elif bma_results_file.endswith('.csv'):
                df = pd.read_csv(bma_results_file)
            else:
                raise ValueError("不支持的文件格式，请使用Excel(.xlsx)或CSV(.csv)")
            
            signals = {}
            
            # 处理每只股票的预测
            for _, row in df.iterrows():
                symbol = row.get('Ticker', row.get('Symbol', ''))
                prediction = row.get('Prediction', row.get('预测值', 0))
                confidence = row.get('Confidence', row.get('置信度', 0.5))
                
                if not symbol or symbol not in self.stock_universe:
                    continue
                
                # 生成交易信号
                signal = self._create_signal_from_prediction(symbol, prediction, confidence)
                if signal:
                    signals[symbol] = signal
            
            self.logger.info(f"从BMA结果生成了 {len(signals)} 个信号")
            return signals
            
        except Exception as e:
            self.logger.error(f"从BMA结果生成信号失败: {e}")
            return {}
    
    def generate_signals_from_lstm_results(self, lstm_results_file: str) -> Dict:
        """从LSTM模型结果生成交易信号"""
        try:
            self.logger.info(f"从LSTM结果生成信号: {lstm_results_file}")
            
            # 读取LSTM结果
            if lstm_results_file.endswith('.xlsx'):
                df = pd.read_excel(lstm_results_file)
            elif lstm_results_file.endswith('.csv'):
                df = pd.read_csv(lstm_results_file)
            else:
                raise ValueError("不支持的文件格式，请使用Excel(.xlsx)或CSV(.csv)")
            
            signals = {}
            
            # 处理每只股票的预测
            for _, row in df.iterrows():
                symbol = row.get('Ticker', row.get('Symbol', ''))
                
                # LSTM可能有多日预测，取第一日
                prediction_cols = [col for col in df.columns if 'prediction' in col.lower() or '预测' in col]
                if prediction_cols:
                    prediction = row.get(prediction_cols[0], 0)
                else:
                    prediction = row.get('Prediction', row.get('预测值', 0))
                
                confidence = row.get('Confidence', row.get('置信度', 0.7))  # LSTM默认置信度稍高
                
                if not symbol or symbol not in self.stock_universe:
                    continue
                
                # 生成交易信号
                signal = self._create_signal_from_prediction(symbol, prediction, confidence)
                if signal:
                    signals[symbol] = signal
            
            self.logger.info(f"从LSTM结果生成了 {len(signals)} 个信号")
            return signals
            
        except Exception as e:
            self.logger.error(f"从LSTM结果生成信号失败: {e}")
            return {}
    
    def _create_signal_from_prediction(self, symbol: str, prediction: float, confidence: float) -> Optional[Dict]:
        """从预测值创建交易信号"""
        try:
            # 检查预测阈值
            if abs(prediction) < self.min_prediction_threshold:
                return None
            
            # 检查置信度阈值
            if confidence < self.min_confidence:
                return None
            
            # 确定交易动作
            action = "BUY" if prediction > 0 else "SELL"
            
            # 计算目标价格（简化版，实际应结合当前价格）
            # 这里假设预测值是百分比变化
            base_price = self._get_estimated_current_price(symbol)
            target_price = base_price * (1 + prediction)
            
            # 调整置信度
            adjusted_confidence = min(confidence * (abs(prediction) / 0.05), 1.0)  # 基于5%标准化
            
            signal = {
                "action": action,
                "confidence": round(adjusted_confidence, 3),
                "target_price": round(target_price, 2),
                "prediction": round(prediction, 4),
                "original_confidence": round(confidence, 3),
                "generated_at": datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"创建信号失败 {symbol}: {e}")
            return None
    
    def _get_estimated_current_price(self, symbol: str) -> float:
        """获取估计的当前价格（简化版）"""
        # 这里使用一些常见股票的大概价格
        # 实际应用中应该从实时数据源获取
        price_estimates = {
            'AAPL': 150, 'MSFT': 300, 'GOOGL': 125, 'AMZN': 100, 'TSLA': 200,
            'META': 250, 'NVDA': 400, 'NFLX': 350, 'CRM': 200, 'ADBE': 450
        }
        
        return price_estimates.get(symbol, 100)  # 默认100美元
    
    def combine_signals(self, bma_signals: Dict, lstm_signals: Dict, 
                       bma_weight: float = 0.4, lstm_weight: float = 0.6) -> Dict:
        """组合BMA和LSTM信号"""
        try:
            self.logger.info("组合BMA和LSTM信号...")
            
            combined_signals = {}
            all_symbols = set(bma_signals.keys()) | set(lstm_signals.keys())
            
            for symbol in all_symbols:
                bma_signal = bma_signals.get(symbol, {})
                lstm_signal = lstm_signals.get(symbol, {})
                
                # 如果只有一个模型有信号，使用该信号
                if not bma_signal and lstm_signal:
                    combined_signals[symbol] = lstm_signal.copy()
                elif bma_signal and not lstm_signal:
                    combined_signals[symbol] = bma_signal.copy()
                else:
                    # 两个模型都有信号，进行组合
                    combined_signal = self._combine_two_signals(symbol, bma_signal, lstm_signal, 
                                                              bma_weight, lstm_weight)
                    if combined_signal:
                        combined_signals[symbol] = combined_signal
            
            self.logger.info(f"组合后生成 {len(combined_signals)} 个信号")
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"组合信号失败: {e}")
            return {}
    
    def _combine_two_signals(self, symbol: str, bma_signal: Dict, lstm_signal: Dict, 
                           bma_weight: float, lstm_weight: float) -> Optional[Dict]:
        """组合两个信号"""
        try:
            bma_action = bma_signal.get('action')
            lstm_action = lstm_signal.get('action')
            
            # 如果动作相同，组合置信度和价格
            if bma_action == lstm_action:
                combined_confidence = (
                    bma_signal.get('confidence', 0) * bma_weight +
                    lstm_signal.get('confidence', 0) * lstm_weight
                )
                
                combined_target_price = (
                    bma_signal.get('target_price', 0) * bma_weight +
                    lstm_signal.get('target_price', 0) * lstm_weight
                )
                
                return {
                    "action": bma_action,
                    "confidence": round(combined_confidence, 3),
                    "target_price": round(combined_target_price, 2),
                    "source": "BMA+LSTM",
                    "bma_confidence": bma_signal.get('confidence', 0),
                    "lstm_confidence": lstm_signal.get('confidence', 0),
                    "generated_at": datetime.now().isoformat()
                }
            
            # 如果动作不同，选择置信度更高的
            else:
                bma_conf = bma_signal.get('confidence', 0)
                lstm_conf = lstm_signal.get('confidence', 0)
                
                if bma_conf > lstm_conf:
                    result = bma_signal.copy()
                    result['source'] = "BMA(优选)"
                else:
                    result = lstm_signal.copy()
                    result['source'] = "LSTM(优选)"
                
                result['conflict_resolved'] = True
                return result
            
        except Exception as e:
            self.logger.error(f"组合两个信号失败 {symbol}: {e}")
            return None
    
    def filter_signals(self, signals: Dict, max_signals: Optional[int] = None) -> Dict:
        """过滤和排序信号"""
        try:
            if not signals:
                return {}
            
            # 按置信度排序
            sorted_signals = dict(sorted(
                signals.items(), 
                key=lambda x: x[1].get('confidence', 0), 
                reverse=True
            ))
            
            # 限制信号数量
            if max_signals is None:
                max_signals = self.max_signals_per_day
            
            filtered_signals = dict(list(sorted_signals.items())[:max_signals])
            
            self.logger.info(f"过滤后保留 {len(filtered_signals)} 个高质量信号")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"过滤信号失败: {e}")
            return signals
    
    def generate_trading_config(self, signals: Dict, 
                              risk_management: Optional[Dict] = None) -> Dict:
        """生成完整的交易配置"""
        try:
            # 默认风险管理参数
            default_risk = {
                "max_position_size": 0.02,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
                "max_daily_trades": 50,
                "total_capital": 100000
            }
            
            if risk_management:
                default_risk.update(risk_management)
            
            # 生成配置
            config = {
                "stocks": self.stock_universe,
                "signals": signals,
                "risk_management": default_risk,
                "trading_params": {
                    "order_type": "LMT",
                    "time_in_force": "DAY",
                    "outside_rth": False
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator_version": "1.0",
                    "stock_universe_size": len(self.stock_universe),
                    "signals_count": len(signals)
                }
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"生成交易配置失败: {e}")
            return {}
    
    def save_trading_config(self, config: Dict, filename: str = "trading_signals.json"):
        """保存交易配置到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"交易配置已保存到: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存交易配置失败: {e}")
            return False
    
    def process_model_results(self, bma_file: Optional[str] = None, 
                             lstm_file: Optional[str] = None,
                             output_file: str = "trading_signals.json") -> bool:
        """处理模型结果并生成交易配置"""
        try:
            self.logger.info("🚀 开始处理模型结果...")
            
            # 生成信号
            bma_signals = {}
            lstm_signals = {}
            
            if bma_file and os.path.exists(bma_file):
                bma_signals = self.generate_signals_from_bma_results(bma_file)
            
            if lstm_file and os.path.exists(lstm_file):
                lstm_signals = self.generate_signals_from_lstm_results(lstm_file)
            
            if not bma_signals and not lstm_signals:
                self.logger.error("❌ 没有找到有效的模型结果文件")
                return False
            
            # 组合信号
            combined_signals = self.combine_signals(bma_signals, lstm_signals)
            
            # 过滤信号
            filtered_signals = self.filter_signals(combined_signals)
            
            # 生成配置
            config = self.generate_trading_config(filtered_signals)
            
            # 保存配置
            success = self.save_trading_config(config, output_file)
            
            if success:
                self.logger.info("✅ 交易信号生成完成!")
                self.logger.info(f"📊 生成了 {len(filtered_signals)} 个交易信号")
                self.logger.info(f"📁 配置文件: {output_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"处理模型结果失败: {e}")
            return False


def main():
    """主函数示例"""
    generator = TradingSignalsGenerator()
    
    # 示例：处理模型结果
    print("📈 交易信号生成器")
    print("=" * 50)
    
    # 检查是否有模型结果文件
    bma_files = ["result/bma_predictions.xlsx", "result/bma_results.xlsx"]
    lstm_files = ["result/lstm_predictions.xlsx", "result/lstm_results.xlsx", "result/multi_day_lstm_analysis.xlsx"]
    
    bma_file = None
    lstm_file = None
    
    for file in bma_files:
        if os.path.exists(file):
            bma_file = file
            break
    
    for file in lstm_files:
        if os.path.exists(file):
            lstm_file = file
            break
    
    if bma_file:
        print(f"✅ 找到BMA结果文件: {bma_file}")
    else:
        print("⚠️ 未找到BMA结果文件")
    
    if lstm_file:
        print(f"✅ 找到LSTM结果文件: {lstm_file}")
    else:
        print("⚠️ 未找到LSTM结果文件")
    
    # 处理模型结果
    success = generator.process_model_results(bma_file, lstm_file)
    
    if success:
        print("\n🎉 交易信号生成成功!")
        print("📁 生成的配置文件: trading_signals.json")
        print("🔧 现在可以使用 EnhancedIBKRAutoTrader 进行自动交易")
    else:
        print("\n❌ 交易信号生成失败")
        
        # 生成示例配置
        print("📝 生成示例配置文件...")
        example_signals = {
            "AAPL": {"action": "BUY", "confidence": 0.8, "target_price": 150.0},
            "MSFT": {"action": "BUY", "confidence": 0.75, "target_price": 300.0}
        }
        
        config = generator.generate_trading_config(example_signals)
        generator.save_trading_config(config, "example_trading_signals.json")
        print("✅ 示例配置已生成: example_trading_signals.json")


if __name__ == "__main__":
    main()