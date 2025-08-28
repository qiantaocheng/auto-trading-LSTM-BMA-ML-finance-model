#!/usr/bin/env python3
"""
基于 Polygon API 的真实数据全局训练系统
使用您的 Polygon API 密钥获取真实股票数据
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的 Polygon 客户端
from polygon_client import polygon_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_polygon_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealPolygonTrainingSystem:
    """基于 Polygon API 的真实数据训练系统"""
    
    def __init__(self):
        """初始化真实数据训练系统"""
        self.start_time = time.time()
        self.polygon = polygon_client
        
        # 训练配置
        self.config = {
            "max_workers": 8,  # 增加并发数，Polygon API支持较高并发
            "api_delay": 0.15,  # Polygon API 延迟
            "lookback_days": 60,  # 历史数据回看天数
            "technical_indicators": True,  # 计算技术指标
            "fundamental_analysis": True,  # 基本面分析
            "risk_adjustment": True,  # 风险调整
            "min_volume": 100000,  # 最小成交量过滤
            "min_price": 1.0,  # 最小价格过滤
            "max_retries": 3  # API重试次数
        }
        
        # 全局结果存储
        self.global_predictions = {}
        self.global_scores = {}
        self.market_data_cache = {}
        self.failed_stocks = []
        
        logger.info("基于 Polygon API 的真实数据训练系统初始化完成")
    
    def load_all_stocks(self) -> List[str]:
        """加载所有股票数据"""
        stock_file = "filtered_stocks_20250817_002928.txt"
        
        if not os.path.exists(stock_file):
            logger.error(f"股票文件不存在: {stock_file}")
            return []
        
        try:
            with open(stock_file, 'r') as f:
                content = f.read()
                # 解析逗号分隔和引号格式的股票代码
                import re
                stocks = re.findall(r'"([^"]+)"', content)
                
                # 去重并过滤空值
                stocks = list(set([stock.strip() for stock in stocks if stock.strip()]))
                
                # 限制为前20只股票进行测试（避免过度API调用）
                stocks = sorted(stocks)[:20]
            
            logger.info(f"成功加载 {len(stocks)} 只股票用于真实数据训练（限制测试模式）")
            return stocks
            
        except Exception as e:
            logger.error(f"加载股票列表失败: {e}")
            return []
    
    def get_real_market_data(self, ticker: str) -> Dict[str, Any]:
        """获取真实市场数据 - 使用 Polygon API"""
        try:
            logger.debug(f"正在获取 {ticker} 的真实市场数据...")
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config["lookback_days"])
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # 1. 获取历史价格数据
            historical_data = self.polygon.get_historical_bars(
                ticker, start_str, end_str, "day", 1
            )
            
            if historical_data.empty:
                logger.warning(f"无法获取 {ticker} 的历史数据")
                return None
            
            # 2. 获取股票基本信息
            ticker_info = self.polygon.get_ticker_details(ticker)
            
            # 3. 获取当前价格
            current_price = self.polygon.get_current_price(ticker)
            if current_price <= 0:
                # 如果无法获取当前价格，使用最新收盘价
                current_price = historical_data['Close'].iloc[-1] if not historical_data.empty else 0
            
            # 4. 计算技术指标
            tech_indicators = self.calculate_technical_indicators(historical_data)
            
            # 5. 基本面分析
            fundamental_metrics = self.analyze_fundamentals(ticker_info, historical_data)
            
            # 6. 风险指标
            risk_metrics = self.calculate_risk_metrics(historical_data)
            
            # 数据质量过滤
            latest_volume = historical_data['Volume'].iloc[-1] if not historical_data.empty else 0
            if current_price < self.config["min_price"] or latest_volume < self.config["min_volume"]:
                logger.info(f"{ticker} 不符合质量标准: 价格={current_price:.2f}, 成交量={latest_volume}")
                return None
            
            market_data = {
                "ticker": ticker,
                "current_price": current_price,
                "historical_data": historical_data,
                "ticker_info": ticker_info,
                "technical_indicators": tech_indicators,
                "fundamental_metrics": fundamental_metrics,
                "risk_metrics": risk_metrics,
                "data_timestamp": datetime.now().isoformat(),
                "data_quality_score": self.assess_data_quality(historical_data, ticker_info)
            }
            
            logger.debug(f"成功获取 {ticker} 的市场数据")
            return market_data
            
        except Exception as e:
            logger.error(f"获取 {ticker} 市场数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        if df.empty or len(df) < 20:
            return {}
        
        try:
            close_prices = df['Close']
            volume = df['Volume']
            high_prices = df['High']
            low_prices = df['Low']
            
            # 移动平均线
            sma_10 = close_prices.rolling(10).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # RSI
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50
            
            # 布林带
            bb_middle = close_prices.rolling(20).mean()
            bb_std = close_prices.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (close_prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # MACD
            exp1 = close_prices.ewm(span=12).mean()
            exp2 = close_prices.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal
            
            # 价格动量
            price_momentum_5d = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(df) >= 6 else 0
            price_momentum_20d = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(df) >= 21 else price_momentum_5d
            
            # 成交量指标
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # 波动率
            volatility = close_prices.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            
            return {
                "sma_10": float(sma_10),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "rsi": float(rsi),
                "bb_position": float(bb_position),
                "macd": float(macd.iloc[-1]),
                "macd_signal": float(signal.iloc[-1]),
                "macd_histogram": float(macd_histogram.iloc[-1]),
                "price_momentum_5d": float(price_momentum_5d),
                "price_momentum_20d": float(price_momentum_20d),
                "volume_ratio": float(volume_ratio),
                "volatility": float(volatility)
            }
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def analyze_fundamentals(self, ticker_info: Dict, historical_data: pd.DataFrame) -> Dict[str, float]:
        """基本面分析"""
        try:
            market_cap = ticker_info.get("market_cap", 1000000000)
            shares_outstanding = ticker_info.get("weighted_shares_outstanding", 1000000)
            
            # 当前价格
            current_price = historical_data['Close'].iloc[-1] if not historical_data.empty else 100
            
            # 市值分类
            if market_cap > 10e9:
                market_cap_score = 0.8  # 大盘股
            elif market_cap > 2e9:
                market_cap_score = 0.6  # 中盘股
            else:
                market_cap_score = 0.4  # 小盘股
            
            # 价格稳定性评分
            if not historical_data.empty and len(historical_data) >= 20:
                price_stability = 1 / (1 + historical_data['Close'].pct_change().std())
            else:
                price_stability = 0.5
            
            # 流动性评分（基于成交量）
            if not historical_data.empty:
                avg_volume = historical_data['Volume'].mean()
                liquidity_score = min(1.0, avg_volume / 1000000)  # 标准化到0-1
            else:
                liquidity_score = 0.5
            
            return {
                "market_cap": float(market_cap),
                "market_cap_score": float(market_cap_score),
                "price_stability": float(price_stability),
                "liquidity_score": float(liquidity_score),
                "shares_outstanding": float(shares_outstanding),
                "current_price": float(current_price)
            }
            
        except Exception as e:
            logger.error(f"基本面分析失败: {e}")
            return {}
    
    def calculate_risk_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """计算风险指标"""
        if historical_data.empty or len(historical_data) < 10:
            return {"volatility": 0.3, "max_drawdown": 0.1, "sharpe_ratio": 0.5}
        
        try:
            returns = historical_data['Close'].pct_change().dropna()
            
            # 年化波动率
            volatility = returns.std() * np.sqrt(252)
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # 夏普比率（假设无风险利率为0.02）
            excess_returns = returns.mean() - 0.02/252
            sharpe_ratio = excess_returns / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # VaR (95% 置信度)
            var_95 = np.percentile(returns, 5)
            
            return {
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": float(sharpe_ratio),
                "var_95": float(var_95)
            }
            
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return {"volatility": 0.3, "max_drawdown": 0.1, "sharpe_ratio": 0.5}
    
    def assess_data_quality(self, historical_data: pd.DataFrame, ticker_info: Dict) -> float:
        """评估数据质量"""
        score = 1.0
        
        # 历史数据完整性
        if historical_data.empty:
            score -= 0.5
        elif len(historical_data) < 30:
            score -= 0.2
        
        # 数据连续性检查
        if not historical_data.empty:
            null_ratio = historical_data.isnull().sum().sum() / (len(historical_data) * len(historical_data.columns))
            score -= null_ratio * 0.3
        
        # 基本信息完整性
        required_fields = ["market_cap", "name", "active"]
        missing_fields = sum(1 for field in required_fields if not ticker_info.get(field))
        score -= missing_fields * 0.1
        
        return max(0.0, min(1.0, score))
    
    def advanced_prediction_model(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """高级预测模型 - 基于真实数据"""
        try:
            tech = market_data.get("technical_indicators", {})
            fundamental = market_data.get("fundamental_metrics", {})
            risk = market_data.get("risk_metrics", {})
            
            # 技术面评分 (0-1)
            technical_score = 0.5
            
            # RSI 评分 (30-70 为好，极值为差)
            rsi = tech.get("rsi", 50)
            if 30 <= rsi <= 70:
                rsi_score = 0.8
            elif rsi < 20 or rsi > 80:
                rsi_score = 0.3
            else:
                rsi_score = 0.6
            
            # 趋势评分 (基于移动平均线)
            current_price = fundamental.get("current_price", 100)
            sma_20 = tech.get("sma_20", current_price)
            sma_50 = tech.get("sma_50", current_price)
            
            if current_price > sma_20 > sma_50:
                trend_score = 0.9  # 上升趋势
            elif current_price < sma_20 < sma_50:
                trend_score = 0.3  # 下降趋势
            else:
                trend_score = 0.6  # 横盘或混合
            
            # 动量评分
            momentum_20d = tech.get("price_momentum_20d", 0)
            momentum_score = 0.5 + momentum_20d * 0.5  # 转换为0-1分数
            momentum_score = max(0.1, min(0.9, momentum_score))
            
            # 成交量评分
            volume_ratio = tech.get("volume_ratio", 1)
            volume_score = min(0.9, 0.3 + volume_ratio * 0.3)
            
            technical_score = (rsi_score * 0.25 + trend_score * 0.35 + 
                             momentum_score * 0.25 + volume_score * 0.15)
            
            # 基本面评分 (0-1)
            fundamental_score = 0.5
            
            market_cap_score = fundamental.get("market_cap_score", 0.5)
            price_stability = fundamental.get("price_stability", 0.5)
            liquidity_score = fundamental.get("liquidity_score", 0.5)
            
            fundamental_score = (market_cap_score * 0.4 + price_stability * 0.3 + 
                                liquidity_score * 0.3)
            
            # 风险调整评分
            volatility = risk.get("volatility", 0.3)
            sharpe_ratio = risk.get("sharpe_ratio", 0.5)
            max_drawdown = risk.get("max_drawdown", 0.1)
            
            # 风险调整：低波动率、高夏普比率、低回撤为好
            risk_score = 0.5
            risk_score += (0.3 - min(0.3, volatility)) / 0.3 * 0.3  # 波动率越低越好
            risk_score += max(0, min(2, sharpe_ratio)) / 2 * 0.4   # 夏普比率越高越好
            risk_score += (0.2 - min(0.2, max_drawdown)) / 0.2 * 0.3  # 回撤越小越好
            
            # 集成预测
            weights = [0.4, 0.35, 0.25]  # 技术、基本面、风险权重
            individual_scores = [technical_score, fundamental_score, risk_score]
            final_prediction = sum(score * weight for score, weight in zip(individual_scores, weights))
            
            # 计算置信度
            data_quality = market_data.get("data_quality_score", 0.8)
            score_variance = np.var(individual_scores)
            confidence = data_quality * (1 - score_variance)
            confidence = max(0.5, min(0.99, confidence))
            
            # 综合评分
            composite_score = final_prediction * confidence * data_quality
            
            return {
                "prediction": final_prediction,
                "confidence": confidence,
                "composite_score": composite_score,
                "individual_models": {
                    "technical": technical_score,
                    "fundamental": fundamental_score,
                    "risk_adjusted": risk_score
                },
                "detailed_metrics": {
                    "rsi_score": rsi_score,
                    "trend_score": trend_score,
                    "momentum_score": momentum_score,
                    "volume_score": volume_score,
                    "data_quality": data_quality
                }
            }
            
        except Exception as e:
            logger.error(f"预测模型计算失败: {e}")
            return {
                "prediction": 0.5,
                "confidence": 0.5,
                "composite_score": 0.25,
                "individual_models": {"technical": 0.5, "fundamental": 0.5, "risk_adjusted": 0.5},
                "detailed_metrics": {}
            }
    
    def process_single_stock(self, ticker: str) -> Tuple[str, Dict[str, Any]]:
        """处理单只股票的完整流程 - 使用真实数据"""
        retry_count = 0
        last_error = None
        
        while retry_count < self.config["max_retries"]:
            try:
                logger.info(f"正在处理股票 {ticker} (尝试 {retry_count + 1}/{self.config['max_retries']})")
                
                # 获取真实市场数据
                market_data = self.get_real_market_data(ticker)
                
                if market_data is None:
                    logger.warning(f"无法获取 {ticker} 的市场数据")
                    return ticker, None
                
                # 高级预测分析
                prediction_result = self.advanced_prediction_model(market_data)
                
                # 整合结果
                stock_result = {
                    "ticker": ticker,
                    "market_data": market_data,
                    "prediction_result": prediction_result,
                    "processing_timestamp": datetime.now().isoformat(),
                    "data_source": "polygon_api",
                    "api_calls_used": 3  # 历史数据 + 股票信息 + 当前价格
                }
                
                logger.info(f"成功处理股票 {ticker}: 预测={prediction_result['prediction']:.3f}, "
                           f"置信度={prediction_result['confidence']:.3f}")
                
                return ticker, stock_result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.error(f"处理股票 {ticker} 失败 (尝试 {retry_count}): {e}")
                
                if retry_count < self.config["max_retries"]:
                    wait_time = 2 ** retry_count  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        logger.error(f"股票 {ticker} 处理最终失败: {last_error}")
        return ticker, None
    
    def run_real_data_training(self, stocks: List[str]) -> Dict[str, Any]:
        """运行真实数据并行训练"""
        logger.info(f"开始使用 Polygon API 真实数据训练 {len(stocks)} 只股票...")
        
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        total_api_calls = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(self.process_single_stock, ticker): ticker 
                for ticker in stocks
            }
            
            # 收集结果
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_name, result = future.result()
                    
                    if result is not None:
                        # 存储全局结果
                        self.global_predictions[ticker_name] = result["prediction_result"]["prediction"]
                        self.global_scores[ticker_name] = result["prediction_result"]["composite_score"]
                        self.market_data_cache[ticker_name] = result
                        
                        processed_count += 1
                        total_api_calls += result.get("api_calls_used", 3)
                    else:
                        failed_count += 1
                        self.failed_stocks.append(ticker_name)
                        
                except Exception as e:
                    logger.error(f"处理股票 {ticker} 时出错: {e}")
                    failed_count += 1
                    self.failed_stocks.append(ticker)
                
                # 进度报告
                if (processed_count + failed_count) % 25 == 0:
                    progress = (processed_count + failed_count) / len(stocks) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * len(stocks) / (processed_count + failed_count) - elapsed
                    logger.info(f"训练进度: {progress:.1f}% | 成功: {processed_count} | 失败: {failed_count} | "
                               f"用时: {elapsed/60:.1f}分钟 | 预计剩余: {eta/60:.1f}分钟")
        
        training_time = time.time() - start_time
        
        logger.info(f"真实数据训练完成: {processed_count} 成功, {failed_count} 失败, "
                   f"用时 {training_time/60:.1f} 分钟, 总API调用: {total_api_calls}")
        
        return {
            "total_processed": processed_count,
            "total_failed": failed_count,
            "failed_stocks": self.failed_stocks,
            "training_time": training_time,
            "processing_rate": processed_count / training_time if training_time > 0 else 0,
            "total_api_calls": total_api_calls,
            "avg_api_calls_per_stock": total_api_calls / processed_count if processed_count > 0 else 0
        }
    
    def analyze_real_results(self) -> Dict[str, Any]:
        """分析真实数据结果并排序"""
        logger.info("开始分析真实数据训练结果...")
        
        if not self.global_scores:
            logger.error("没有可分析的数据")
            return {}
        
        # 转换为DataFrame便于分析
        results_data = []
        for ticker, score in self.global_scores.items():
            prediction = self.global_predictions.get(ticker, 0)
            result = self.market_data_cache.get(ticker, {})
            perf = result.get("prediction_result", {})
            market_data = result.get("market_data", {})
            
            # 获取真实市场数据
            current_price = market_data.get("fundamental_metrics", {}).get("current_price", 0)
            market_cap = market_data.get("fundamental_metrics", {}).get("market_cap", 0)
            volatility = market_data.get("risk_metrics", {}).get("volatility", 0)
            data_quality = market_data.get("data_quality_score", 0)
            
            results_data.append({
                "ticker": ticker,
                "composite_score": score,
                "prediction": prediction,
                "confidence": perf.get("confidence", 0),
                "technical_score": perf.get("individual_models", {}).get("technical", 0),
                "fundamental_score": perf.get("individual_models", {}).get("fundamental", 0),
                "risk_adjusted_score": perf.get("individual_models", {}).get("risk_adjusted", 0),
                "current_price": current_price,
                "market_cap": market_cap,
                "volatility": volatility,
                "data_quality": data_quality
            })
        
        df = pd.DataFrame(results_data)
        
        # 全局统计
        global_stats = {
            "total_stocks_analyzed": len(df),
            "avg_composite_score": df["composite_score"].mean(),
            "std_composite_score": df["composite_score"].std(),
            "avg_prediction": df["prediction"].mean(),
            "avg_confidence": df["confidence"].mean(),
            "avg_data_quality": df["data_quality"].mean(),
            "avg_market_cap": df["market_cap"].mean(),
            "avg_volatility": df["volatility"].mean(),
            "score_distribution": {
                "top_10_percent": df["composite_score"].quantile(0.9),
                "top_25_percent": df["composite_score"].quantile(0.75),
                "median": df["composite_score"].median(),
                "bottom_25_percent": df["composite_score"].quantile(0.25)
            }
        }
        
        # 按综合评分排序
        df_sorted = df.sort_values("composite_score", ascending=False)
        
        # 获取前5名
        top_5 = df_sorted.head(5).to_dict("records")
        
        # 获取前20名用于详细分析
        top_20 = df_sorted.head(20).to_dict("records")
        
        logger.info(f"真实数据分析完成，识别出前5名股票")
        
        return {
            "global_statistics": global_stats,
            "top_5_stocks": top_5,
            "top_20_stocks": top_20,
            "all_results_df": df_sorted,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_source": "polygon_api_real_data"
        }
    
    def generate_real_data_report(self, training_results: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]) -> str:
        """生成真实数据训练报告"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"result/real_polygon_training_report_{timestamp}.json"
        
        # 构建完整报告（移除DataFrame）
        analysis_results_clean = analysis_results.copy()
        if "all_results_df" in analysis_results_clean:
            del analysis_results_clean["all_results_df"]
        
        final_report = {
            "system_info": {
                "training_mode": "polygon_api_real_data",
                "data_source": "polygon.io",
                "api_key_used": "FExbaO1x...Q1 (付费订阅)",
                "total_stocks": len(self.global_predictions),
                "training_time_minutes": training_results["training_time"] / 60,
                "processing_rate_per_second": training_results["processing_rate"],
                "total_api_calls": training_results["total_api_calls"],
                "avg_api_calls_per_stock": training_results["avg_api_calls_per_stock"],
                "config": self.config
            },
            "training_results": training_results,
            "analysis_results": analysis_results_clean,
            "top_5_recommendations": analysis_results.get("top_5_stocks", []),
            "failed_stocks_analysis": {
                "failed_count": len(self.failed_stocks),
                "failed_stocks": self.failed_stocks[:10],  # 只显示前10个失败的
                "failure_rate": len(self.failed_stocks) / (len(self.failed_stocks) + len(self.global_predictions)) if self.global_predictions else 1
            },
            "report_timestamp": datetime.now().isoformat()
        }
        
        # 保存报告
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"真实数据训练报告已保存: {report_file}")
        return report_file

def main():
    """主执行流程"""
    print("=" * 80)
    print("基于 Polygon API 的真实数据股票训练系统")
    print("=" * 80)
    
    # 1. 初始化系统
    print("\n1. 初始化真实数据训练系统...")
    real_system = RealPolygonTrainingSystem()
    
    # 2. 加载所有股票
    print("\n2. 加载股票数据...")
    all_stocks = real_system.load_all_stocks()
    if not all_stocks:
        print("[ERROR] 无法加载股票数据")
        return False
    
    print(f"   总股票数: {len(all_stocks)}")
    print(f"   并发线程: {real_system.config['max_workers']}")
    print(f"   API 来源: Polygon.io (付费订阅)")
    print(f"   历史数据回看: {real_system.config['lookback_days']} 天")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. 运行真实数据训练
    print("\n3. 执行基于 Polygon API 的真实数据训练...")
    print("   [警告] 这将使用真实的 API 调用，预计需要 15-30 分钟")
    
    # 自动确认执行（测试模式）
    print("\n[AUTO] 自动确认执行真实数据训练（测试模式）")
    
    training_results = real_system.run_real_data_training(all_stocks)
    
    # 4. 分析真实结果
    print("\n4. 分析真实数据训练结果...")
    analysis_results = real_system.analyze_real_results()
    
    if not analysis_results:
        print("[ERROR] 真实数据分析失败")
        return False
    
    # 5. 显示前5名结果
    print("\n5. 基于真实数据的最佳5只股票推荐:")
    print("=" * 80)
    
    top_5 = analysis_results["top_5_stocks"]
    for i, stock in enumerate(top_5, 1):
        print(f"\n[排名 {i}] {stock['ticker']}")
        print(f"   综合评分: {stock['composite_score']:.4f}")
        print(f"   预测评分: {stock['prediction']:.4f}")
        print(f"   置信度: {stock['confidence']:.4f}")
        print(f"   当前价格: ${stock['current_price']:.2f}")
        print(f"   市值: ${stock['market_cap']/1e9:.2f}B")
        print(f"   技术面评分: {stock['technical_score']:.4f}")
        print(f"   基本面评分: {stock['fundamental_score']:.4f}")
        print(f"   风险调整评分: {stock['risk_adjusted_score']:.4f}")
        print(f"   数据质量: {stock['data_quality']:.4f}")
        print(f"   波动率: {stock['volatility']:.4f}")
    
    # 6. 真实数据统计
    stats = analysis_results["global_statistics"]
    print(f"\n6. 真实数据训练统计:")
    print("=" * 80)
    print(f"   成功分析股票: {stats['total_stocks_analyzed']}")
    print(f"   失败股票数: {training_results['total_failed']}")
    print(f"   成功率: {training_results['total_processed']/(training_results['total_processed']+training_results['total_failed'])*100:.1f}%")
    print(f"   平均综合评分: {stats['avg_composite_score']:.4f}")
    print(f"   平均置信度: {stats['avg_confidence']:.4f}")
    print(f"   平均数据质量: {stats['avg_data_quality']:.4f}")
    print(f"   训练用时: {training_results['training_time']/60:.2f} 分钟")
    print(f"   总 API 调用: {training_results['total_api_calls']}")
    print(f"   平均每股 API 调用: {training_results['avg_api_calls_per_stock']:.1f}")
    print(f"   处理速度: {training_results['processing_rate']*60:.1f} 股票/分钟")
    
    # 7. 生成最终报告
    print("\n7. 生成真实数据训练报告...")
    report_file = real_system.generate_real_data_report(training_results, analysis_results)
    print(f"   报告文件: {report_file}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] 基于 Polygon API 真实数据训练完成！")
    print("已识别出最佳5只股票，数据来源于真实市场")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)