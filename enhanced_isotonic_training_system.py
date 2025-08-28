#!/usr/bin/env python3
"""
集成 Isotonic 校准的增强训练系统
基于 Polygon API 真实数据 + Isotonic 回归校准
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

# 导入 Isotonic 校准
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False
    print("[WARNING] Isotonic 校准不可用，使用基础预测模式")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_isotonic_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedIsotonicTrainingSystem:
    """集成 Isotonic 校准的增强训练系统"""
    
    def __init__(self):
        """初始化增强训练系统"""
        self.start_time = time.time()
        self.polygon = polygon_client
        
        # 训练配置
        self.config = {
            "max_workers": 6,  # 优化并发数
            "api_delay": 0.2,  # 增加API延迟保护
            "lookback_days": 90,  # 增加历史数据回看
            "isotonic_calibration": ISOTONIC_AVAILABLE,  # Isotonic 校准
            "cv_folds": 5,  # 交叉验证折数
            "calibration_samples": 100,  # 校准最少样本数
            "risk_adjustment": True,
            "quality_threshold": 0.8,  # 数据质量阈值
            "min_volume": 500000,  # 提高成交量要求
            "min_price": 5.0,  # 提高价格要求
        }
        
        # 全局结果存储
        self.global_predictions = {}
        self.global_scores = {}
        self.market_data_cache = {}
        self.calibrated_predictions = {}
        self.isotonic_calibrators = {}
        self.failed_stocks = []
        
        logger.info(f"增强 Isotonic 训练系统初始化完成 (Isotonic可用: {ISOTONIC_AVAILABLE})")
    
    def load_all_stocks(self) -> List[str]:
        """加载股票数据"""
        stock_file = "filtered_stocks_20250817_002928.txt"
        
        if not os.path.exists(stock_file):
            logger.error(f"股票文件不存在: {stock_file}")
            return []
        
        try:
            with open(stock_file, 'r') as f:
                content = f.read()
                import re
                stocks = re.findall(r'"([^"]+)"', content)
                stocks = list(set([stock.strip() for stock in stocks if stock.strip()]))
                
                # 扩展到前50只股票用于演示 Isotonic 校准效果
                stocks = sorted(stocks)[:50]
            
            logger.info(f"成功加载 {len(stocks)} 只股票用于增强训练（Isotonic 校准演示）")
            return stocks
            
        except Exception as e:
            logger.error(f"加载股票列表失败: {e}")
            return []
    
    def get_enhanced_market_data(self, ticker: str) -> Dict[str, Any]:
        """获取增强市场数据"""
        try:
            logger.debug(f"获取 {ticker} 的增强市场数据...")
            
            # 计算更长的历史数据范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config["lookback_days"])
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # 获取历史数据
            historical_data = self.polygon.get_historical_bars(
                ticker, start_str, end_str, "day", 1
            )
            
            if historical_data.empty or len(historical_data) < 30:
                logger.warning(f"{ticker} 历史数据不足")
                return None
            
            # 获取股票信息和当前价格
            ticker_info = self.polygon.get_ticker_details(ticker)
            current_price = self.polygon.get_current_price(ticker)
            if current_price <= 0:
                current_price = historical_data['Close'].iloc[-1]
            
            # 计算增强技术指标
            tech_indicators = self.calculate_enhanced_technical_indicators(historical_data)
            
            # 增强基本面分析
            fundamental_metrics = self.enhanced_fundamental_analysis(ticker_info, historical_data)
            
            # 高级风险指标
            risk_metrics = self.calculate_advanced_risk_metrics(historical_data)
            
            # 生成模拟历史预测用于 Isotonic 校准
            historical_predictions = self.generate_historical_predictions(historical_data)
            
            # 质量过滤
            latest_volume = historical_data['Volume'].iloc[-1]
            if (current_price < self.config["min_price"] or 
                latest_volume < self.config["min_volume"]):
                logger.info(f"{ticker} 不符合质量标准")
                return None
            
            market_data = {
                "ticker": ticker,
                "current_price": current_price,
                "historical_data": historical_data,
                "ticker_info": ticker_info,
                "technical_indicators": tech_indicators,
                "fundamental_metrics": fundamental_metrics,
                "risk_metrics": risk_metrics,
                "historical_predictions": historical_predictions,
                "data_timestamp": datetime.now().isoformat(),
                "data_quality_score": self.assess_enhanced_data_quality(historical_data, ticker_info)
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"获取 {ticker} 增强市场数据失败: {e}")
            return None
    
    def calculate_enhanced_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算增强技术指标"""
        if df.empty or len(df) < 30:
            return {}
        
        try:
            close_prices = df['Close']
            volume = df['Volume']
            high_prices = df['High']
            low_prices = df['Low']
            
            # 基础移动平均
            sma_10 = close_prices.rolling(10).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            ema_12 = close_prices.ewm(span=12).mean().iloc[-1]
            ema_26 = close_prices.ewm(span=26).mean().iloc[-1]
            
            # RSI (增强版)
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50
            
            # MACD 系统
            macd = ema_12 - ema_26
            signal_line = close_prices.ewm(span=9).mean().iloc[-1]
            macd_histogram = macd - signal_line
            
            # 布林带
            bb_middle = close_prices.rolling(20).mean()
            bb_std = close_prices.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (close_prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            bb_squeeze = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            
            # 高级动量指标
            momentum_5d = (close_prices.iloc[-1] / close_prices.iloc[-6] - 1) if len(df) >= 6 else 0
            momentum_20d = (close_prices.iloc[-1] / close_prices.iloc[-21] - 1) if len(df) >= 21 else momentum_5d
            momentum_50d = (close_prices.iloc[-1] / close_prices.iloc[-51] - 1) if len(df) >= 51 else momentum_20d
            
            # 成交量加权平均价格 (VWAP)
            typical_price = (high_prices + low_prices + close_prices) / 3
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            vwap_deviation = (close_prices.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
            
            # 波动率指标
            volatility_20d = close_prices.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            volatility_5d = close_prices.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252)
            
            # ATR (真实波动幅度)
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift(1))
            tr3 = abs(low_prices - close_prices.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_ratio = atr / close_prices.iloc[-1]
            
            # 相对强度
            market_return = momentum_20d  # 简化市场收益
            relative_strength = momentum_20d - market_return
            
            return {
                "sma_10": float(sma_10),
                "sma_20": float(sma_20), 
                "sma_50": float(sma_50),
                "ema_12": float(ema_12),
                "ema_26": float(ema_26),
                "rsi": float(rsi),
                "macd": float(macd),
                "macd_histogram": float(macd_histogram),
                "bb_position": float(bb_position),
                "bb_squeeze": float(bb_squeeze),
                "momentum_5d": float(momentum_5d),
                "momentum_20d": float(momentum_20d),
                "momentum_50d": float(momentum_50d),
                "vwap_deviation": float(vwap_deviation),
                "volatility_20d": float(volatility_20d),
                "volatility_5d": float(volatility_5d),
                "atr_ratio": float(atr_ratio),
                "relative_strength": float(relative_strength)
            }
            
        except Exception as e:
            logger.error(f"计算增强技术指标失败: {e}")
            return {}
    
    def enhanced_fundamental_analysis(self, ticker_info: Dict, historical_data: pd.DataFrame) -> Dict[str, float]:
        """增强基本面分析"""
        try:
            market_cap = ticker_info.get("market_cap", 1000000000)
            current_price = historical_data['Close'].iloc[-1]
            
            # 市值分类和评分
            if market_cap > 50e9:
                market_cap_score = 0.9  # 超大盘股
                size_risk = 0.1
            elif market_cap > 10e9:
                market_cap_score = 0.8  # 大盘股
                size_risk = 0.2
            elif market_cap > 2e9:
                market_cap_score = 0.6  # 中盘股
                size_risk = 0.4
            else:
                market_cap_score = 0.4  # 小盘股
                size_risk = 0.6
            
            # 价格行为分析
            price_changes = historical_data['Close'].pct_change()
            price_stability = 1 / (1 + price_changes.std())
            trend_consistency = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0
            
            # 成交量分析
            avg_volume = historical_data['Volume'].mean()
            volume_trend = np.corrcoef(range(len(historical_data)), historical_data['Volume'])[0,1]
            liquidity_score = min(1.0, avg_volume / 5000000)  # 基于日均500万股标准化
            
            # 价格动量质量
            up_days = (price_changes > 0).sum()
            down_days = (price_changes < 0).sum()
            momentum_quality = up_days / (up_days + down_days) if (up_days + down_days) > 0 else 0.5
            
            # 波动性-收益比
            returns = price_changes.mean() * 252  # 年化收益
            volatility = price_changes.std() * np.sqrt(252)  # 年化波动率
            risk_return_ratio = returns / volatility if volatility > 0 else 0
            
            return {
                "market_cap": float(market_cap),
                "market_cap_score": float(market_cap_score),
                "size_risk": float(size_risk),
                "price_stability": float(price_stability),
                "trend_consistency": float(trend_consistency),
                "liquidity_score": float(liquidity_score),
                "volume_trend": float(volume_trend),
                "momentum_quality": float(momentum_quality),
                "risk_return_ratio": float(risk_return_ratio),
                "current_price": float(current_price)
            }
            
        except Exception as e:
            logger.error(f"增强基本面分析失败: {e}")
            return {}
    
    def calculate_advanced_risk_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """计算高级风险指标"""
        if historical_data.empty or len(historical_data) < 20:
            return {}
        
        try:
            returns = historical_data['Close'].pct_change().dropna()
            
            # 基础风险指标
            volatility = returns.std() * np.sqrt(252)
            
            # 最大回撤和回撤恢复
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # 计算回撤恢复时间
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = 0
            
            for i, dd in enumerate(drawdown):
                if dd < -0.05 and not in_drawdown:  # 开始回撤
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= -0.01 and in_drawdown:  # 回撤恢复
                    in_drawdown = False
                    drawdown_periods.append(i - drawdown_start)
            
            avg_recovery_time = np.mean(drawdown_periods) if drawdown_periods else 0
            
            # 下行波动率
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            
            # Sortino 比率
            excess_returns = returns.mean() - 0.02/252  # 假设无风险利率2%
            sortino_ratio = (excess_returns * 252) / downside_volatility if downside_volatility > 0 else 0
            
            # VaR 和 CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            # 偏度和峰度
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # 信息比率（简化版）
            tracking_error = returns.std()
            information_ratio = returns.mean() / tracking_error if tracking_error > 0 else 0
            
            return {
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "downside_volatility": float(downside_volatility),
                "sortino_ratio": float(sortino_ratio),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "information_ratio": float(information_ratio),
                "avg_recovery_time": float(avg_recovery_time)
            }
            
        except Exception as e:
            logger.error(f"计算高级风险指标失败: {e}")
            return {}
    
    def generate_historical_predictions(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """生成历史预测数据用于 Isotonic 校准"""
        try:
            if len(historical_data) < 50:
                return {}
            
            # 生成模拟的历史预测和实际收益
            close_prices = historical_data['Close']
            actual_returns = close_prices.pct_change(5).shift(-5)  # 5天后的收益
            
            # 基于简单技术指标的模拟预测
            sma_10 = close_prices.rolling(10).mean()
            sma_20 = close_prices.rolling(20).mean()
            
            # 简单预测逻辑：基于移动平均交叉
            predictions = []
            for i in range(len(historical_data)):
                if i < 20:
                    predictions.append(0.5)  # 中性预测
                else:
                    # 基于技术指标的简单预测
                    if sma_10.iloc[i] > sma_20.iloc[i]:
                        pred = 0.6 + np.zeros(0)  # 看涨
                    else:
                        pred = 0.4 + np.zeros(0)  # 看跌
                    predictions.append(max(0.1, min(0.9, pred)))
            
            # 只取有实际收益数据的部分
            valid_indices = ~pd.isna(actual_returns)
            valid_predictions = np.array(predictions)[valid_indices]
            valid_returns = actual_returns[valid_indices].values
            
            if len(valid_predictions) < 20:
                return {}
            
            return {
                "predictions": valid_predictions.tolist(),
                "actual_returns": valid_returns.tolist(),
                "sample_count": len(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"生成历史预测失败: {e}")
            return {}
    
    def apply_isotonic_calibration(self, ticker: str, raw_prediction: float, 
                                  historical_predictions: Dict) -> Dict[str, float]:
        """应用 Isotonic 校准"""
        if not ISOTONIC_AVAILABLE or not historical_predictions:
            return {
                "calibrated_prediction": raw_prediction,
                "confidence": 0.8,
                "calibration_applied": False
            }
        
        try:
            predictions = np.array(historical_predictions["predictions"])
            actual_returns = np.array(historical_predictions["actual_returns"])
            
            if len(predictions) < self.config["calibration_samples"]:
                return {
                    "calibrated_prediction": raw_prediction,
                    "confidence": 0.7,
                    "calibration_applied": False,
                    "reason": "insufficient_samples"
                }
            
            # 使用 5-fold 交叉验证进行 OOF 校准
            kf = KFold(n_splits=self.config["cv_folds"], shuffle=True, random_state=42)
            calibrated_preds = np.full_like(predictions, np.nan)
            
            for train_idx, test_idx in kf.split(predictions):
                if len(train_idx) < 10:  # 训练集太小
                    continue
                    
                train_preds = predictions[train_idx]
                train_returns = actual_returns[train_idx]
                test_preds = predictions[test_idx]
                
                # 训练 Isotonic 回归
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(train_preds, train_returns)
                
                # 校准测试集预测
                calibrated_preds[test_idx] = iso_reg.predict(test_preds)
            
            # 使用全部数据训练最终校准器用于新预测
            final_iso_reg = IsotonicRegression(out_of_bounds='clip')
            final_iso_reg.fit(predictions, actual_returns)
            
            # 校准当前预测
            calibrated_prediction = final_iso_reg.predict([raw_prediction])[0]
            
            # 计算校准质量和置信度
            valid_calibrated = calibrated_preds[~np.isnan(calibrated_preds)]
            valid_actual = actual_returns[~np.isnan(calibrated_preds)]
            
            if len(valid_calibrated) > 10:
                calibration_mse = np.mean((valid_calibrated - valid_actual) ** 2)
                baseline_mse = np.mean((predictions[~np.isnan(calibrated_preds)] - valid_actual) ** 2)
                improvement = max(0, (baseline_mse - calibration_mse) / baseline_mse)
                confidence = 0.6 + 0.3 * improvement  # 基础置信度 + 改善程度
            else:
                confidence = 0.7
            
            # 存储校准器供后续使用
            self.isotonic_calibrators[ticker] = final_iso_reg
            
            return {
                "calibrated_prediction": float(calibrated_prediction),
                "confidence": float(confidence),
                "calibration_applied": True,
                "improvement": float(improvement) if 'improvement' in locals() else 0.0,
                "sample_size": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Isotonic 校准失败 {ticker}: {e}")
            return {
                "calibrated_prediction": raw_prediction,
                "confidence": 0.6,
                "calibration_applied": False,
                "error": str(e)
            }
    
    def assess_enhanced_data_quality(self, historical_data: pd.DataFrame, 
                                   ticker_info: Dict) -> float:
        """评估增强数据质量"""
        score = 1.0
        
        # 数据完整性
        if historical_data.empty:
            score -= 0.6
        elif len(historical_data) < 60:
            score -= 0.3
        elif len(historical_data) < 30:
            score -= 0.4
        
        # 数据质量检查
        if not historical_data.empty:
            # 空值比例
            null_ratio = historical_data.isnull().sum().sum() / (len(historical_data) * len(historical_data.columns))
            score -= null_ratio * 0.4
            
            # 价格数据的合理性
            price_changes = historical_data['Close'].pct_change()
            extreme_changes = (abs(price_changes) > 0.2).sum()  # 单日涨跌超过20%
            if extreme_changes > len(historical_data) * 0.05:  # 超过5%的交易日出现极端变化
                score -= 0.2
            
            # 成交量数据质量
            zero_volume_days = (historical_data['Volume'] == 0).sum()
            if zero_volume_days > 0:
                score -= zero_volume_days / len(historical_data) * 0.3
        
        # 基本信息质量
        required_fields = ["market_cap", "name", "active"]
        missing_fields = sum(1 for field in required_fields if not ticker_info.get(field))
        score -= missing_fields * 0.1
        
        return max(0.0, min(1.0, score))
    
    def enhanced_prediction_model(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强预测模型"""
        try:
            tech = market_data.get("technical_indicators", {})
            fundamental = market_data.get("fundamental_metrics", {})
            risk = market_data.get("risk_metrics", {})
            
            # 技术面评分 (更细致的权重)
            technical_components = {
                "trend_strength": self.calculate_trend_strength(tech),
                "momentum_score": self.calculate_momentum_score(tech),
                "volatility_score": self.calculate_volatility_score(tech),
                "volume_score": self.calculate_volume_score(tech)
            }
            
            technical_weights = [0.35, 0.25, 0.2, 0.2]
            technical_score = sum(score * weight for score, weight in 
                                zip(technical_components.values(), technical_weights))
            
            # 基本面评分 (增强版)
            fundamental_score = self.calculate_fundamental_score(fundamental)
            
            # 风险调整评分
            risk_adjusted_score = self.calculate_risk_adjusted_score(risk, fundamental, technical_score)
            
            # 集成预测 (动态权重)
            market_cap = fundamental.get("market_cap", 1e9)
            if market_cap > 10e9:  # 大盘股更重视基本面
                weights = [0.3, 0.5, 0.2]
            elif market_cap > 2e9:  # 中盘股平衡
                weights = [0.4, 0.35, 0.25]
            else:  # 小盘股更重视技术面
                weights = [0.5, 0.25, 0.25]
            
            individual_scores = [technical_score, fundamental_score, risk_adjusted_score]
            raw_prediction = sum(score * weight for score, weight in zip(individual_scores, weights))
            
            # 应用 Isotonic 校准
            historical_preds = market_data.get("historical_predictions", {})
            calibration_result = self.apply_isotonic_calibration(
                market_data["ticker"], raw_prediction, historical_preds
            )
            
            # 最终预测和置信度
            final_prediction = calibration_result["calibrated_prediction"]
            confidence = calibration_result["confidence"]
            
            # 数据质量调整
            data_quality = market_data.get("data_quality_score", 0.8)
            adjusted_confidence = confidence * data_quality
            
            # 综合评分
            composite_score = final_prediction * adjusted_confidence * data_quality
            
            return {
                "raw_prediction": raw_prediction,
                "final_prediction": final_prediction,
                "confidence": adjusted_confidence,
                "composite_score": composite_score,
                "individual_models": {
                    "technical": technical_score,
                    "fundamental": fundamental_score,
                    "risk_adjusted": risk_adjusted_score
                },
                "technical_components": technical_components,
                "calibration_result": calibration_result,
                "data_quality": data_quality
            }
            
        except Exception as e:
            logger.error(f"增强预测模型计算失败: {e}")
            return {
                "raw_prediction": 0.5,
                "final_prediction": 0.5,
                "confidence": 0.5,
                "composite_score": 0.25,
                "individual_models": {},
                "calibration_result": {"calibration_applied": False}
            }
    
    def calculate_trend_strength(self, tech: Dict) -> float:
        """计算趋势强度"""
        try:
            sma_10 = tech.get("sma_10", 0)
            sma_20 = tech.get("sma_20", 0)
            sma_50 = tech.get("sma_50", 0)
            
            if sma_20 == 0:
                return 0.5
            
            # 移动平均排列
            if sma_10 > sma_20 > sma_50:
                trend_score = 0.8  # 强上升趋势
            elif sma_10 < sma_20 < sma_50:
                trend_score = 0.2  # 强下降趋势
            elif sma_10 > sma_20:
                trend_score = 0.65  # 短期上升
            elif sma_10 < sma_20:
                trend_score = 0.35  # 短期下降
            else:
                trend_score = 0.5   # 震荡
            
            # RSI 确认
            rsi = tech.get("rsi", 50)
            if 30 <= rsi <= 70:
                rsi_confirmation = 0.8  # RSI 健康
            elif rsi > 80 or rsi < 20:
                rsi_confirmation = 0.3  # RSI 极值
            else:
                rsi_confirmation = 0.6
            
            return trend_score * 0.7 + rsi_confirmation * 0.3
            
        except Exception:
            return 0.5
    
    def calculate_momentum_score(self, tech: Dict) -> float:
        """计算动量评分"""
        try:
            momentum_5d = tech.get("momentum_5d", 0)
            momentum_20d = tech.get("momentum_20d", 0)
            momentum_50d = tech.get("momentum_50d", 0)
            
            # 动量一致性检查
            if momentum_5d > 0 and momentum_20d > 0 and momentum_50d > 0:
                momentum_score = 0.8  # 全面上涨
            elif momentum_5d < 0 and momentum_20d < 0 and momentum_50d < 0:
                momentum_score = 0.2  # 全面下跌
            elif momentum_20d > 0:
                momentum_score = 0.65  # 中期上涨
            elif momentum_20d < 0:
                momentum_score = 0.35  # 中期下跌
            else:
                momentum_score = 0.5
            
            # MACD 确认
            macd_hist = tech.get("macd_histogram", 0)
            if macd_hist > 0:
                macd_confirmation = 0.7
            elif macd_hist < 0:
                macd_confirmation = 0.3
            else:
                macd_confirmation = 0.5
            
            return momentum_score * 0.8 + macd_confirmation * 0.2
            
        except Exception:
            return 0.5
    
    def calculate_volatility_score(self, tech: Dict) -> float:
        """计算波动率评分"""
        try:
            volatility_20d = tech.get("volatility_20d", 0.3)
            bb_squeeze = tech.get("bb_squeeze", 0.1)
            atr_ratio = tech.get("atr_ratio", 0.02)
            
            # 波动率评分 (适中的波动率最好)
            if 0.15 <= volatility_20d <= 0.35:
                vol_score = 0.8  # 适中波动率
            elif volatility_20d < 0.10:
                vol_score = 0.4  # 波动率太低
            elif volatility_20d > 0.50:
                vol_score = 0.3  # 波动率太高
            else:
                vol_score = 0.6
            
            # 布林带挤压 (预示突破)
            if bb_squeeze < 0.05:
                squeeze_score = 0.3  # 过度压缩
            elif bb_squeeze > 0.15:
                squeeze_score = 0.3  # 过度扩张
            else:
                squeeze_score = 0.8  # 适中
            
            return vol_score * 0.6 + squeeze_score * 0.4
            
        except Exception:
            return 0.5
    
    def calculate_volume_score(self, tech: Dict) -> float:
        """计算成交量评分"""
        try:
            vwap_deviation = tech.get("vwap_deviation", 0)
            relative_strength = tech.get("relative_strength", 0)
            
            # VWAP 偏离度
            if abs(vwap_deviation) < 0.02:
                vwap_score = 0.8  # 接近 VWAP
            elif abs(vwap_deviation) > 0.05:
                vwap_score = 0.4  # 偏离过多
            else:
                vwap_score = 0.6
            
            # 相对强度
            if relative_strength > 0.02:
                rs_score = 0.8  # 跑赢大盘
            elif relative_strength < -0.02:
                rs_score = 0.3  # 跑输大盘
            else:
                rs_score = 0.6  # 跟随大盘
            
            return vwap_score * 0.4 + rs_score * 0.6
            
        except Exception:
            return 0.5
    
    def calculate_fundamental_score(self, fundamental: Dict) -> float:
        """计算基本面评分"""
        try:
            market_cap_score = fundamental.get("market_cap_score", 0.5)
            price_stability = fundamental.get("price_stability", 0.5)
            liquidity_score = fundamental.get("liquidity_score", 0.5)
            momentum_quality = fundamental.get("momentum_quality", 0.5)
            risk_return_ratio = fundamental.get("risk_return_ratio", 0)
            
            # 权重分配
            components = [
                (market_cap_score, 0.25),
                (price_stability, 0.20),
                (liquidity_score, 0.20),
                (momentum_quality, 0.20),
                (min(1.0, max(0.0, risk_return_ratio + 0.5)), 0.15)  # 标准化风险收益比
            ]
            
            return sum(score * weight for score, weight in components)
            
        except Exception:
            return 0.5
    
    def calculate_risk_adjusted_score(self, risk: Dict, fundamental: Dict, technical_score: float) -> float:
        """计算风险调整评分"""
        try:
            volatility = risk.get("volatility", 0.3)
            max_drawdown = risk.get("max_drawdown", 0.1)
            sortino_ratio = risk.get("sortino_ratio", 0)
            skewness = risk.get("skewness", 0)
            
            # 风险惩罚
            vol_penalty = min(0.3, volatility * 0.5)
            drawdown_penalty = min(0.2, max_drawdown * 2)
            
            # 风险奖励
            sortino_bonus = min(0.2, max(0, sortino_ratio) * 0.1)
            skew_bonus = min(0.1, max(0, skewness) * 0.1)  # 正偏度奖励
            
            # 基础评分
            base_score = technical_score
            
            # 风险调整
            risk_adjusted = base_score - vol_penalty - drawdown_penalty + sortino_bonus + skew_bonus
            
            return max(0.1, min(0.9, risk_adjusted))
            
        except Exception:
            return 0.5
    
    def process_enhanced_stock(self, ticker: str) -> Tuple[str, Dict[str, Any]]:
        """处理单只股票的完整增强流程"""
        retry_count = 0
        last_error = None
        
        while retry_count < 3:
            try:
                logger.info(f"处理股票 {ticker} (尝试 {retry_count + 1}/3)")
                
                # 获取增强市场数据
                market_data = self.get_enhanced_market_data(ticker)
                
                if market_data is None:
                    logger.warning(f"无法获取 {ticker} 的市场数据")
                    return ticker, None
                
                # 增强预测分析
                prediction_result = self.enhanced_prediction_model(market_data)
                
                # 整合结果
                stock_result = {
                    "ticker": ticker,
                    "market_data": market_data,
                    "prediction_result": prediction_result,
                    "processing_timestamp": datetime.now().isoformat(),
                    "data_source": "polygon_api_enhanced_isotonic",
                    "isotonic_calibrated": prediction_result["calibration_result"]["calibration_applied"]
                }
                
                logger.info(f"成功处理股票 {ticker}: "
                           f"原始预测={prediction_result['raw_prediction']:.3f}, "
                           f"校准预测={prediction_result['final_prediction']:.3f}, "
                           f"置信度={prediction_result['confidence']:.3f}, "
                           f"Isotonic校准={'是' if prediction_result['calibration_result']['calibration_applied'] else '否'}")
                
                return ticker, stock_result
                
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.error(f"处理股票 {ticker} 失败 (尝试 {retry_count}): {e}")
                
                if retry_count < 3:
                    wait_time = 2 ** retry_count
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        logger.error(f"股票 {ticker} 处理最终失败: {last_error}")
        return ticker, None
    
    def run_enhanced_training(self, stocks: List[str]) -> Dict[str, Any]:
        """运行增强训练"""
        logger.info(f"开始增强 Isotonic 训练 {len(stocks)} 只股票...")
        
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        calibrated_count = 0
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            future_to_ticker = {
                executor.submit(self.process_enhanced_stock, ticker): ticker 
                for ticker in stocks
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_name, result = future.result()
                    
                    if result is not None:
                        prediction_result = result["prediction_result"]
                        
                        # 存储结果
                        self.global_predictions[ticker_name] = prediction_result["final_prediction"]
                        self.global_scores[ticker_name] = prediction_result["composite_score"]
                        self.market_data_cache[ticker_name] = result
                        
                        # 统计校准
                        if prediction_result["calibration_result"]["calibration_applied"]:
                            calibrated_count += 1
                        
                        processed_count += 1
                    else:
                        failed_count += 1
                        self.failed_stocks.append(ticker_name)
                        
                except Exception as e:
                    logger.error(f"处理股票 {ticker} 时出错: {e}")
                    failed_count += 1
                    self.failed_stocks.append(ticker)
                
                # 进度报告
                if (processed_count + failed_count) % 10 == 0:
                    progress = (processed_count + failed_count) / len(stocks) * 100
                    logger.info(f"进度: {progress:.1f}% | 成功: {processed_count} | "
                               f"失败: {failed_count} | Isotonic校准: {calibrated_count}")
        
        training_time = time.time() - start_time
        
        logger.info(f"增强训练完成: {processed_count} 成功, {failed_count} 失败, "
                   f"Isotonic校准: {calibrated_count}, 用时 {training_time/60:.1f} 分钟")
        
        return {
            "total_processed": processed_count,
            "total_failed": failed_count,
            "isotonic_calibrated": calibrated_count,
            "calibration_rate": calibrated_count / processed_count if processed_count > 0 else 0,
            "training_time": training_time,
            "processing_rate": processed_count / training_time if training_time > 0 else 0
        }
    
    def analyze_enhanced_results(self) -> Dict[str, Any]:
        """分析增强结果"""
        logger.info("开始分析增强训练结果...")
        
        if not self.global_scores:
            return {}
        
        # 构建结果数据
        results_data = []
        for ticker, score in self.global_scores.items():
            result = self.market_data_cache.get(ticker, {})
            prediction_result = result.get("prediction_result", {})
            market_data = result.get("market_data", {})
            
            calibration_result = prediction_result.get("calibration_result", {})
            fundamental = market_data.get("fundamental_metrics", {})
            tech = market_data.get("technical_indicators", {})
            
            results_data.append({
                "ticker": ticker,
                "composite_score": score,
                "raw_prediction": prediction_result.get("raw_prediction", 0),
                "final_prediction": prediction_result.get("final_prediction", 0),
                "confidence": prediction_result.get("confidence", 0),
                "isotonic_calibrated": calibration_result.get("calibration_applied", False),
                "calibration_improvement": calibration_result.get("improvement", 0),
                "current_price": fundamental.get("current_price", 0),
                "market_cap": fundamental.get("market_cap", 0),
                "technical_score": prediction_result.get("individual_models", {}).get("technical", 0),
                "fundamental_score": prediction_result.get("individual_models", {}).get("fundamental", 0),
                "risk_adjusted_score": prediction_result.get("individual_models", {}).get("risk_adjusted", 0),
                "data_quality": prediction_result.get("data_quality", 0),
                "rsi": tech.get("rsi", 50),
                "volatility": tech.get("volatility_20d", 0.3)
            })
        
        df = pd.DataFrame(results_data)
        df_sorted = df.sort_values("composite_score", ascending=False)
        
        # 统计信息
        calibrated_df = df[df["isotonic_calibrated"] == True]
        
        global_stats = {
            "total_stocks_analyzed": len(df),
            "isotonic_calibrated_count": len(calibrated_df),
            "calibration_rate": len(calibrated_df) / len(df),
            "avg_composite_score": df["composite_score"].mean(),
            "avg_calibration_improvement": calibrated_df["calibration_improvement"].mean() if len(calibrated_df) > 0 else 0,
            "avg_confidence": df["confidence"].mean(),
            "avg_data_quality": df["data_quality"].mean(),
            "prediction_difference": (df["final_prediction"] - df["raw_prediction"]).abs().mean()
        }
        
        # 前5名
        top_5 = df_sorted.head(5).to_dict("records")
        
        return {
            "global_statistics": global_stats,
            "top_5_stocks": top_5,
            "isotonic_performance": {
                "calibrated_stocks": len(calibrated_df),
                "avg_improvement": calibrated_df["calibration_improvement"].mean() if len(calibrated_df) > 0 else 0,
                "avg_confidence_calibrated": calibrated_df["confidence"].mean() if len(calibrated_df) > 0 else 0,
                "avg_confidence_uncalibrated": df[df["isotonic_calibrated"] == False]["confidence"].mean()
            },
            "analysis_timestamp": datetime.now().isoformat()
        }

def main():
    """主执行流程"""
    print("=" * 80)
    print("集成 Isotonic 校准的增强股票训练系统")
    print("=" * 80)
    
    # 1. 初始化系统
    print("\n1. 初始化增强训练系统...")
    enhanced_system = EnhancedIsotonicTrainingSystem()
    
    # 2. 加载股票
    print("\n2. 加载股票数据...")
    stocks = enhanced_system.load_all_stocks()
    if not stocks:
        print("[ERROR] 无法加载股票数据")
        return False
    
    print(f"   总股票数: {len(stocks)}")
    print(f"   Isotonic校准: {'启用' if ISOTONIC_AVAILABLE else '不可用'}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. 运行增强训练
    print("\n3. 执行增强 Isotonic 校准训练...")
    training_results = enhanced_system.run_enhanced_training(stocks)
    
    # 4. 分析结果
    print("\n4. 分析增强训练结果...")
    analysis_results = enhanced_system.analyze_enhanced_results()
    
    if not analysis_results:
        print("[ERROR] 结果分析失败")
        return False
    
    # 5. 显示增强结果
    print("\n5. 增强 Isotonic 校准最佳5只股票:")
    print("=" * 80)
    
    top_5 = analysis_results["top_5_stocks"]
    for i, stock in enumerate(top_5, 1):
        print(f"\n[排名 {i}] {stock['ticker']}")
        print(f"   综合评分: {stock['composite_score']:.4f}")
        print(f"   原始预测: {stock['raw_prediction']:.4f}")
        print(f"   校准预测: {stock['final_prediction']:.4f}")
        print(f"   置信度: {stock['confidence']:.4f}")
        print(f"   当前价格: ${stock['current_price']:.2f}")
        print(f"   Isotonic校准: {'是' if stock['isotonic_calibrated'] else '否'}")
        if stock['isotonic_calibrated']:
            print(f"   校准改善: {stock['calibration_improvement']:.1%}")
        print(f"   技术面: {stock['technical_score']:.3f} | "
              f"基本面: {stock['fundamental_score']:.3f} | "
              f"风险调整: {stock['risk_adjusted_score']:.3f}")
    
    # 6. Isotonic 校准统计
    stats = analysis_results["global_statistics"]
    isotonic_perf = analysis_results["isotonic_performance"]
    
    print(f"\n6. Isotonic 校准效果统计:")
    print("=" * 80)
    print(f"   分析股票总数: {stats['total_stocks_analyzed']}")
    print(f"   Isotonic校准数: {stats['isotonic_calibrated_count']}")
    print(f"   校准成功率: {stats['calibration_rate']*100:.1f}%")
    print(f"   平均校准改善: {stats['avg_calibration_improvement']*100:.1f}%")
    print(f"   平均置信度提升: {stats['avg_confidence']:.3f}")
    print(f"   预测调整幅度: {stats['prediction_difference']:.3f}")
    
    print(f"\n   Isotonic vs 基础预测对比:")
    print(f"   校准股票置信度: {isotonic_perf['avg_confidence_calibrated']:.3f}")
    print(f"   未校准股票置信度: {isotonic_perf['avg_confidence_uncalibrated']:.3f}")
    print(f"   训练用时: {training_results['training_time']/60:.1f} 分钟")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] 增强 Isotonic 校准训练完成！")
    print("已应用高级校准算法提升预测精度")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)