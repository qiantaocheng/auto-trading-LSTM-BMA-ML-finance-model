#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Ultra Enhanced - Main Application
使用原始完整版本的BMA Ultra Enhanced模型
支持实时数据获取、模型训练、预测和报告生成
"""

import os
import sys
import logging
import warnings
import argparse
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bma_app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# =====================================
# 导入原始BMA Ultra Enhanced模型
# =====================================
try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    logger.info("✅ 成功导入原始BMA Ultra Enhanced模型")
except ImportError as e:
    logger.error(f"❌ 无法导入BMA模型: {e}")
    sys.exit(1)

class BMAApplication:
    """BMA应用主类 - 使用原始完整版本"""
    
    def __init__(self, config_path: str = "bma_models/unified_config.yaml"):
        """
        初始化BMA应用
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.model = None
        self.last_training_time = None
        self.last_predictions = None
        
        logger.info("="*60)
        logger.info("BMA Ultra Enhanced Application Starting")
        logger.info("Using ORIGINAL complete version")
        logger.info("="*60)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化BMA模型"""
        try:
            logger.info("初始化BMA Ultra Enhanced模型...")
            self.model = UltraEnhancedQuantitativeModel(
                config_path=self.config_path,
                enable_optimization=True,
                enable_v6_enhancements=True
            )
            logger.info("✅ 模型初始化成功")
            
            # 验证关键组件
            self._verify_components()
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def _verify_components(self):
        """验证模型组件"""
        components = {
            'Polygon Client': hasattr(self.model, 'polygon_client') or (hasattr(self.model, 'alpha_engine') and hasattr(self.model.alpha_engine, 'polygon_client')),
            'Alpha Engine': hasattr(self.model, 'alpha_engine'),
            'Index Aligner': hasattr(self.model, 'index_aligner'),
            'Regime Detector': hasattr(self.model, 'regime_detector'),
            'Memory Manager': hasattr(self.model, 'memory_manager'),
            'CV Splitter': hasattr(self.model, 'cv_splitter')
        }
        
        logger.info("\n组件验证结果:")
        for name, available in components.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {name}: {'可用' if available else '不可用'}")
    
    def get_market_data(self, tickers: List[str], days_back: int = 30) -> pd.DataFrame:
        """
        获取市场数据
        
        Args:
            tickers: 股票代码列表
            days_back: 获取多少天前的数据
            
        Returns:
            市场数据DataFrame
        """
        logger.info(f"\n获取 {len(tickers)} 只股票的市场数据...")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            # 使用模型的数据获取方法
            data = self.model.get_data_and_features(tickers, start_date, end_date)
            
            if data is not None and not data.empty:
                logger.info(f"✅ 数据获取成功: {data.shape}")
                return data
            else:
                logger.error("❌ 未能获取到数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ 数据获取失败: {e}")
            return pd.DataFrame()
    
    def train_model(self, data: pd.DataFrame = None, tickers: List[str] = None, 
                   days_back: int = 252) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data: 训练数据（可选）
            tickers: 股票代码列表（如果没有提供data）
            days_back: 获取多少天前的数据
            
        Returns:
            训练结果
        """
        logger.info("\n" + "="*60)
        logger.info("开始模型训练")
        logger.info("="*60)
        
        try:
            # 如果没有提供数据，则获取数据
            if data is None:
                if tickers is None:
                    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                    logger.info(f"使用默认股票列表: {tickers}")
                
                data = self.get_market_data(tickers, days_back)
                
                if data.empty:
                    logger.error("无法获取训练数据")
                    return {'success': False, 'error': '无法获取训练数据'}
            
            # 准备训练数据
            logger.info("准备训练数据...")
            
            # 分离特征和目标
            # 检查数据结构并准备X和y
            if 'returns' in data.columns:
                # 创建未来收益作为目标
                y = data.groupby('ticker')['returns'].shift(-10) if 'ticker' in data.columns else data['returns'].shift(-10)
                X = data.drop(columns=['returns'])
            elif 'close' in data.columns or 'Close' in data.columns:
                # 基于收盘价计算收益
                close_col = 'close' if 'close' in data.columns else 'Close'
                returns = data.groupby('ticker')[close_col].pct_change() if 'ticker' in data.columns else data[close_col].pct_change()
                y = returns.shift(-10)
                X = data
            else:
                # 使用所有数据作为特征，创建随机目标（仅用于测试）
                logger.warning("未找到合适的目标列，使用最后一列作为目标")
                y = data.iloc[:, -1]
                X = data.iloc[:, :-1]
            
            # 清理NaN值
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            logger.info(f"训练数据准备完成: X={X_clean.shape}, y={len(y_clean)}")
            
            # 训练模型
            results = self.model.train_enhanced_models(X_clean, y_clean, validation_split=0.2)
            
            self.last_training_time = datetime.now()
            
            if results.get('success'):
                logger.info("✅ 模型训练成功")
                self._print_training_summary(results)
            else:
                logger.error("❌ 模型训练失败")
                if 'error' in results:
                    logger.error(f"错误: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def generate_predictions(self, data: pd.DataFrame = None, tickers: List[str] = None) -> pd.DataFrame:
        """
        生成预测
        
        Args:
            data: 预测数据（可选）
            tickers: 股票代码列表（如果没有提供data）
            
        Returns:
            预测结果DataFrame
        """
        logger.info("\n生成预测...")
        
        try:
            # 如果没有提供数据，则获取最新数据
            if data is None:
                if tickers is None:
                    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                
                data = self.get_market_data(tickers, days_back=30)
                
                if data.empty:
                    logger.error("无法获取预测数据")
                    return pd.DataFrame()
            
            # 生成预测
            predictions = self.model.generate_enhanced_predictions(data, use_ensemble=True)
            
            if not predictions.empty:
                logger.info(f"✅ 预测生成成功: {predictions.shape}")
                self.last_predictions = predictions
                
                # 打印预测摘要
                self._print_prediction_summary(predictions)
            else:
                logger.error("❌ 预测生成失败")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 预测过程出错: {e}")
            return pd.DataFrame()
    
    def run_complete_analysis(self, tickers: List[str], days_back: int = 252, 
                            top_n: int = 10) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Args:
            tickers: 股票代码列表
            days_back: 获取多少天前的数据
            top_n: 返回前N个推荐
            
        Returns:
            分析结果
        """
        logger.info("\n" + "="*60)
        logger.info("运行完整分析流程")
        logger.info("="*60)
        
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # 调用模型的完整分析方法
            results = self.model.run_complete_analysis(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                top_n=top_n
            )
            
            if results.get('success'):
                logger.info("✅ 分析完成")
                
                # 打印推荐
                if 'recommendations' in results:
                    self._print_recommendations(results['recommendations'])
            else:
                logger.error("❌ 分析失败")
                if 'error' in results:
                    logger.error(f"错误: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 分析过程出错: {e}")
            return {'success': False, 'error': str(e)}
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """打印训练摘要"""
        logger.info("\n训练结果摘要:")
        logger.info("-" * 40)
        
        if 'training_time' in results:
            logger.info(f"训练时间: {results['training_time']:.2f}秒")
        
        if 'feature_count' in results:
            logger.info(f"特征数量: {results['feature_count']}")
        
        if 'training_samples' in results:
            logger.info(f"训练样本: {results['training_samples']}")
        
        if 'validation_samples' in results:
            logger.info(f"验证样本: {results['validation_samples']}")
        
        # 打印各模型结果
        if 'traditional_models' in results and results['traditional_models']:
            models = results['traditional_models'].get('models', {})
            logger.info(f"\n传统模型: {len(models)}个")
            for name in models.keys():
                logger.info(f"  - {name}")
        
        if 'regime_models' in results and results['regime_models']:
            models = results['regime_models'].get('models', {})
            logger.info(f"\n制度感知模型: {len(models)}个")
        
        if 'stacking_models' in results and results['stacking_models']:
            logger.info(f"\nStacking集成: 已启用")
    
    def _print_prediction_summary(self, predictions: pd.DataFrame):
        """打印预测摘要"""
        logger.info("\n预测结果摘要:")
        logger.info("-" * 40)
        
        for col in predictions.columns:
            if 'pred' in col.lower():
                mean_pred = predictions[col].mean()
                std_pred = predictions[col].std()
                logger.info(f"{col}: 均值={mean_pred:.4f}, 标准差={std_pred:.4f}")
    
    def _print_recommendations(self, recommendations: List[Dict]):
        """打印股票推荐"""
        logger.info("\n股票推荐 (Top Picks):")
        logger.info("-" * 40)
        
        for i, rec in enumerate(recommendations, 1):
            ticker = rec.get('ticker', 'N/A')
            score = rec.get('score', 0)
            logger.info(f"{i}. {ticker:<6} Score: {score:.4f}")
    
    def save_model(self, filepath: str = "models/bma_model.pkl"):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save_model(filepath)
            logger.info(f"✅ 模型已保存至: {filepath}")
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
    
    def load_model(self, filepath: str = "models/bma_model.pkl"):
        """加载模型"""
        try:
            self.model.load_model(filepath)
            logger.info(f"✅ 模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        try:
            return self.model.get_model_summary()
        except Exception as e:
            logger.error(f"获取模型摘要失败: {e}")
            return {}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BMA Ultra Enhanced Application')
    parser.add_argument('--mode', type=str, default='analysis',
                       choices=['train', 'predict', 'analysis', 'backtest'],
                       help='运行模式')
    parser.add_argument('--tickers', type=str, nargs='+',
                       default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'],
                       help='股票代码列表')
    parser.add_argument('--days', type=int, default=252,
                       help='获取多少天前的数据')
    parser.add_argument('--top-n', type=int, default=10,
                       help='返回前N个推荐')
    parser.add_argument('--config', type=str, default='bma_models/unified_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--save-model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--load-model', type=str,
                       help='加载已保存的模型路径')
    
    args = parser.parse_args()
    
    # 创建应用实例
    app = BMAApplication(config_path=args.config)
    
    # 如果指定了加载模型
    if args.load_model:
        app.load_model(args.load_model)
    
    # 根据模式执行
    if args.mode == 'train':
        logger.info("执行模式: 训练")
        results = app.train_model(tickers=args.tickers, days_back=args.days)
        
        if args.save_model and results.get('success'):
            app.save_model()
    
    elif args.mode == 'predict':
        logger.info("执行模式: 预测")
        predictions = app.generate_predictions(tickers=args.tickers)
        
        if not predictions.empty:
            # 保存预测结果
            predictions.to_csv('predictions.csv')
            logger.info("预测结果已保存至 predictions.csv")
    
    elif args.mode == 'analysis':
        logger.info("执行模式: 完整分析")
        results = app.run_complete_analysis(
            tickers=args.tickers,
            days_back=args.days,
            top_n=args.top_n
        )
        
        # 保存分析结果
        if results.get('success'):
            with open('analysis_results.json', 'w') as f:
                # 转换不可序列化的对象
                save_results = {
                    'success': results.get('success'),
                    'analysis_time': results.get('analysis_time'),
                    'recommendations': results.get('recommendations', []),
                    'date_range': results.get('date_range'),
                    'tickers': results.get('tickers')
                }
                json.dump(save_results, f, indent=2, default=str)
            logger.info("分析结果已保存至 analysis_results.json")
    
    elif args.mode == 'backtest':
        logger.info("执行模式: 回测")
        logger.warning("回测功能正在开发中...")
    
    # 打印模型摘要
    summary = app.get_model_summary()
    if summary:
        logger.info("\n模型摘要:")
        logger.info(f"  总模型数: {summary.get('total_models', 0)}")
        logger.info(f"  传统模型: {summary.get('traditional_models', [])}")
        logger.info(f"  制度模型: {summary.get('regime_models', [])}")
        logger.info(f"  元学习器: {summary.get('meta_learners', [])}")
    
    logger.info("\n" + "="*60)
    logger.info("BMA Application 执行完成")
    logger.info("="*60)

if __name__ == "__main__":
    main()