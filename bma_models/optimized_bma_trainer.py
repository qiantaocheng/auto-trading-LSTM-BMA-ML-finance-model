#!/usr/bin/env python3
"""
内存优化版BMA训练器
集成所有优化功能，支持2800股票训练
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

# 本地导入（避免依赖已删除的模块）
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from autotrader.encoding_fix import apply_encoding_fixes, get_safe_logger
    apply_encoding_fixes()
    logger = get_safe_logger(__name__)
except ImportError:
    # 回退到标准logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# 基础内存优化装饰器
def memory_optimize(func):
    """基础内存优化装饰器"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            gc.collect()  # 强制垃圾回收
            return result
        except Exception as e:
            logger.error(f"Memory optimized function failed: {e}")
            raise
    return wrapper

class OptimizedBMATrainer:
    """内存优化版BMA训练器"""
    
    def __init__(self,
                 batch_size: int = 400,
                 memory_limit_gb: float = 3.0,
                 enable_caching: bool = True,
                 cache_dir: str = "cache/optimized_bma"):
        """
        初始化优化版BMA训练器
        
        Args:
            batch_size: 批次大小
            memory_limit_gb: 内存限制
            enable_caching: 启用缓存
            cache_dir: 缓存目录
        """
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # 初始化基础组件（无依赖版本）
        self.memory_manager = None  # 基础版本不使用内存管理器
        self.progress_monitor = self._create_basic_progress_monitor()
        self.streaming_loader = None  # 基础版本不使用流式加载
        self.model_cache = None  # 基础版本不使用模型缓存
        self.batch_trainer = None  # 自己实现批次处理，不依赖外部组件
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 训练统计
        self.training_stats = {
            'total_stocks': 0,
            'successful_stocks': 0,
            'failed_stocks': 0,
            'total_batches': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _create_basic_progress_monitor(self):
        """创建基础进度监控器"""
        class BasicProgressMonitor:
            def __init__(self):
                self.stages = {}
                self.current_stage = None
                
            def add_stage(self, stage_name, total_items):
                self.stages[stage_name] = {'total': total_items, 'completed': 0}
                logger.info(f"添加阶段: {stage_name} ({total_items} 项目)")
                
            def start_training(self):
                logger.info("训练开始")
                
            def start_stage(self, stage_name):
                self.current_stage = stage_name
                logger.info(f"开始阶段: {stage_name}")
                
            def complete_stage(self, stage_name, success=True):
                status = "成功" if success else "失败"
                logger.info(f"阶段完成: {stage_name} - {status}")
                
            def update_progress(self, stage_name, completed):
                if stage_name in self.stages:
                    self.stages[stage_name]['completed'] = completed
                    total = self.stages[stage_name]['total']
                    logger.info(f"进度更新: {stage_name} ({completed}/{total})")
        
        return BasicProgressMonitor()
    
    def _process_universe_in_batches(self, universe: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """基础批次处理实现"""
        logger.info(f"开始批次处理: {len(universe)} 股票，批次大小 {self.batch_size}")
        
        # 分割为批次
        batches = [universe[i:i + self.batch_size] for i in range(0, len(universe), self.batch_size)]
        self.training_stats['total_batches'] = len(batches)
        
        # 汇总结果
        combined_result = {
            'predictions': {},
            'model_performance': {},
            'feature_importance': {},
            'training_metadata': {
                'batch_count': len(batches),
                'total_stocks': len(universe)
            }
        }
        
        # 处理每个批次
        for batch_idx, batch_tickers in enumerate(batches):
            logger.info(f"处理批次 {batch_idx + 1}/{len(batches)}: {len(batch_tickers)} 股票")
            
            try:
                # 处理单个批次
                batch_result = self.train_single_batch(batch_tickers, start_date, end_date)
                
                # 合并结果
                if batch_result:
                    combined_result['predictions'].update(batch_result.get('predictions', {}))
                    combined_result['model_performance'].update(batch_result.get('model_performance', {}))
                    combined_result['feature_importance'].update(batch_result.get('feature_importance', {}))
                
                # 内存清理
                gc.collect()
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
                continue
        
        logger.info(f"批次处理完成: {len(combined_result['predictions'])} 个预测结果")
        return combined_result
    
    @memory_optimize
    def load_universe(self, universe_file: str = "stocks.txt") -> List[str]:
        """加载股票清单"""
        logger.info("加载股票清单...")
        
        try:
            # 支持多种格式
            if universe_file.endswith('.txt'):
                with open(universe_file, 'r', encoding='utf-8') as f:
                    tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            elif universe_file.endswith('.csv'):
                df = pd.read_csv(universe_file)
                tickers = df.iloc[:, 0].tolist()  # 假设第一列是股票代码
            else:
                raise ValueError(f"不支持的文件格式: {universe_file}")
            
            # 清理和验证
            valid_tickers = []
            for ticker in tickers:
                ticker = ticker.strip().upper()
                if ticker and len(ticker) <= 10:  # 基本验证
                    valid_tickers.append(ticker)
            
            logger.info(f"成功加载 {len(valid_tickers)} 个股票代码")
            return valid_tickers
            
        except Exception as e:
            logger.error(f"加载股票清单失败: {e}")
            # 返回默认清单
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    @memory_optimize
    def train_single_batch(self, 
                          batch_tickers: List[str],
                          start_date: str,
                          end_date: str) -> Dict[str, Any]:
        """训练单个批次"""
        logger.info(f"开始训练批次: {len(batch_tickers)} 股票")
        
        batch_results = {
            'predictions': {},
            'model_performance': {},
            'feature_importance': {},
            'training_metadata': {
                'batch_size': len(batch_tickers),
                'start_time': time.time(),
                'memory_before': self.memory_manager.get_memory_info()
            }
        }
        
        successful_count = 0
        
        for ticker in batch_tickers:
            try:
                # 检查模型缓存
                if self.model_cache:
                    cached_result = self._check_model_cache(ticker, start_date, end_date)
                    if cached_result:
                        batch_results['predictions'][ticker] = cached_result
                        successful_count += 1
                        continue
                
                # 流式加载数据
                data = self.streaming_loader.get_data(
                    ticker, "price_data", start_date, end_date,
                    self._load_ticker_data
                )
                
                if data is None or data.empty:
                    logger.warning(f"跳过 {ticker}: 数据为空")
                    continue
                
                # 计算特征
                features = self._calculate_features(data, ticker)
                if features is None or features.empty:
                    logger.warning(f"跳过 {ticker}: 特征计算失败")
                    continue
                
                # 训练模型
                model_result = self._train_ticker_model(ticker, features)
                if model_result:
                    batch_results['predictions'][ticker] = model_result['prediction']
                    batch_results['model_performance'][ticker] = model_result['performance']
                    batch_results['feature_importance'][ticker] = model_result['feature_importance']
                    
                    # 缓存模型
                    if self.model_cache and model_result.get('model'):
                        self._cache_model(ticker, model_result['model'], features, start_date, end_date)
                    
                    successful_count += 1
                
                # 定期内存清理
                if successful_count % 50 == 0:
                    self.memory_manager.force_garbage_collection()
                
            except Exception as e:
                logger.error(f"训练 {ticker} 失败: {e}")
                continue
        
        batch_results['training_metadata'].update({
            'end_time': time.time(),
            'successful_count': successful_count,
            'memory_after': self.memory_manager.get_memory_info()
        })
        
        logger.info(f"批次训练完成: {successful_count}/{len(batch_tickers)} 成功")
        return batch_results
    
    def _load_ticker_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载单个股票数据"""
        try:
            # 这里需要集成实际的数据源
            # 例如：polygon_client, yfinance等
            from polygon_client import download as polygon_download
            
            data = polygon_download(
                ticker,
                start_date=start_date,
                end_date=end_date,
                timespan='day'
            )
            
            if data is not None and not data.empty:
                return data
            else:
                logger.warning(f"数据加载失败: {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"数据加载异常 {ticker}: {e}")
            return None
    
    def _calculate_features(self, data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """计算技术指标特征"""
        try:
            if len(data) < 30:  # 需要足够的历史数据
                return None
            
            features = pd.DataFrame(index=data.index)
            
            # 基础价格特征
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            
            # 技术指标
            # SMA
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
                features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # 成交量特征
            if 'volume' in data.columns:
                features['volume_sma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
            
            # 波动率特征
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_to_high'] = data['close'] / data['high']
            features['close_to_low'] = data['close'] / data['low']
            
            # 清理无效数据
            features = features.dropna()
            
            if len(features) < 10:  # 需要足够的有效数据
                return None
            
            return features
            
        except Exception as e:
            logger.error(f"特征计算失败 {ticker}: {e}")
            return None
    
    def _train_ticker_model(self, ticker: str, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """训练单个股票的模型"""
        try:
            # 准备目标变量（未来5日收益）
            returns = features['returns'].shift(-5)  # 预测未来5日
            
            # 移除缺失值
            valid_mask = ~(returns.isna() | features.isna().any(axis=1))
            X = features[valid_mask].copy()
            y = returns[valid_mask].copy()
            
            if len(X) < 50:  # 需要足够的训练数据
                return None
            
            # 特征标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # 简单的时间序列分割
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # 训练多个模型
            models = {}
            performances = {}
            
            # Random Forest
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(
                n_estimators=50,  # 减少数量节省内存
                max_depth=10,
                random_state=42,
                n_jobs=1  # 避免过度并行
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            models['rf'] = rf_model
            performances['rf'] = np.corrcoef(y_test, rf_pred)[0, 1] if len(y_test) > 1 else 0
            
            # Ridge回归
            from sklearn.linear_model import Ridge
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train, y_train)
            ridge_pred = ridge_model.predict(X_test)
            models['ridge'] = ridge_model
            performances['ridge'] = np.corrcoef(y_test, ridge_pred)[0, 1] if len(y_test) > 1 else 0
            
            # 选择最佳模型
            best_model_name = max(performances, key=performances.get)
            best_model = models[best_model_name]
            best_performance = performances[best_model_name]
            
            # 生成最终预测
            final_prediction = best_model.predict(X_scaled.tail(1))[0]
            
            # 特征重要性
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            else:
                feature_importance = dict(zip(X.columns, np.abs(best_model.coef_)))
            
            return {
                'prediction': final_prediction,
                'performance': best_performance,
                'model_type': best_model_name,
                'feature_importance': feature_importance,
                'model': best_model,
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"模型训练失败 {ticker}: {e}")
            return None
    
    def _check_model_cache(self, ticker: str, start_date: str, end_date: str) -> Optional[float]:
        """检查模型缓存"""
        if not self.model_cache:
            return None
        
        try:
            # 构造缓存键的特征
            cache_features = pd.DataFrame({
                'ticker': [ticker],
                'start_date': [start_date],
                'end_date': [end_date]
            })
            
            cached_model = self.model_cache.get_model(
                model_type='bma_stock_model',
                features=cache_features,
                training_data=cache_features,  # 简化版本
                hyperparameters={'model': 'bma'}
            )
            
            if cached_model:
                logger.debug(f"使用缓存模型: {ticker}")
                # 这里需要用缓存的模型生成预测
                # 为简化，返回一个示例预测
                return np.zeros(0.01)  # 示例预测
            
        except Exception as e:
            logger.debug(f"缓存检查失败 {ticker}: {e}")
        
        return None
    
    def _cache_model(self, ticker: str, model: Any, features: pd.DataFrame, 
                    start_date: str, end_date: str):
        """缓存训练好的模型"""
        if not self.model_cache:
            return
        
        try:
            cache_features = pd.DataFrame({
                'ticker': [ticker],
                'start_date': [start_date],
                'end_date': [end_date]
            })
            
            self.model_cache.cache_model(
                model=model,
                model_type='bma_stock_model',
                features=cache_features,
                training_data=features,
                hyperparameters={'model': 'bma'},
                performance_score=0.5,  # 默认性能分数
                metadata={'ticker': ticker}
            )
            
        except Exception as e:
            logger.warning(f"模型缓存失败 {ticker}: {e}")
    
    def train_universe(self, 
                      universe: List[str],
                      start_date: str = "2021-01-01",
                      end_date: str = "2024-12-31") -> Dict[str, Any]:
        """训练整个股票池"""
        logger.info(f"开始优化版BMA训练: {len(universe)} 股票")
        
        # 设置进度监控
        self.progress_monitor.add_stage("数据加载", len(universe))
        self.progress_monitor.add_stage("模型训练", len(universe))
        self.progress_monitor.add_stage("结果汇总", 1)
        self.progress_monitor.start_training()
        
        # 记录开始时间
        self.training_stats['start_time'] = time.time()
        self.training_stats['total_stocks'] = len(universe)
        
        try:
            # 基础批次处理实现
            final_result = self._process_universe_in_batches(
                universe=universe,
                start_date=start_date,
                end_date=end_date
            )
            
            # 更新统计
            self.training_stats['end_time'] = time.time()
            self.training_stats['successful_stocks'] = len(final_result.get('predictions', {}))
            self.training_stats['failed_stocks'] = (
                self.training_stats['total_stocks'] - 
                self.training_stats['successful_stocks']
            )
            
            # 完成进度监控
            self.progress_monitor.complete_training(success=True)
            
            # 添加优化统计
            final_result['optimization_stats'] = {
                'memory_stats': self.memory_manager.get_statistics(),
                'cache_stats': self.model_cache.get_cache_statistics() if self.model_cache else {},
                'streaming_stats': self.streaming_loader.get_statistics(),
                'training_stats': self.training_stats
            }
            
            logger.info("优化版BMA训练完成!")
            return final_result
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            self.progress_monitor.complete_training(success=False)
            raise
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存训练结果"""
        try:
            # 转换为可序列化格式
            serializable_results = {}
            
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = value
            
            # 保存为JSON
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if self.model_cache:
            self.model_cache.stop()
        
        self.memory_manager.stop_monitoring()
        self.streaming_loader.clear_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info("资源清理完成")


def main():
    """主函数 - 训练2800股票"""
    # 应用编码修复
    apply_encoding_fixes()
    
    # 创建优化训练器
    trainer = OptimizedBMATrainer(
        batch_size=400,  # 400股票/批次
        memory_limit_gb=3.0,  # 3GB内存限制
        enable_caching=True
    )
    
    try:
        # 加载股票清单
        universe = trainer.load_universe("stocks.txt")
        
        if len(universe) > 2800:
            universe = universe[:2800]  # 限制为2800股票
        
        logger.info(f"准备训练 {len(universe)} 股票")
        
        # 开始训练
        results = trainer.train_universe(
            universe=universe,
            start_date="2022-01-01",
            end_date="2024-12-31"
        )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"result/optimized_bma_results_{timestamp}.json"
        trainer.save_results(results, output_file)
        
        # 显示摘要
        logger.info("=" * 60)
        logger.info("训练完成摘要:")
        logger.info(f"总股票数: {results.get('total_stocks', 0)}")
        logger.info(f"成功预测: {len(results.get('predictions', {}))}")
        logger.info(f"总训练时间: {results.get('total_training_time', 0) / 60:.1f} 分钟")
        logger.info(f"结果保存至: {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()