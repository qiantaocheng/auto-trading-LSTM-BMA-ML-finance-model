#!/usr/bin/env python3
"""
简化BMA模型
专注核心功能：数据预处理 -> 特征工程 -> ML训练/预测
删除所有OOF、CV、回测相关复杂逻辑
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

# 设置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
try:
    from .index_aligner import IndexAligner, create_index_aligner
    from .enhanced_alpha_strategies import AlphaStrategiesEngine
    from polygon_client import PolygonClient
    INDEX_ALIGNER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 核心模块导入失败: {e}")
    IndexAligner = None
    AlphaStrategiesEngine = None
    PolygonClient = None
    INDEX_ALIGNER_AVAILABLE = False

# 导入简化ML管理器
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple_ml_manager import SimpleMLManager, SimpleMLConfig
    SIMPLE_ML_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 简化ML管理器导入失败: {e}")
    SimpleMLManager = None
    SimpleMLConfig = None
    SIMPLE_ML_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedBMAModel:
    """简化BMA模型 - 专注数据流到ML"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化简化模型"""
        self.config = config or {}
        self.logger = logger
        
        # 初始化组件
        self.data_client = None
        self.alpha_engine = None
        self.index_aligner = None
        self.ml_manager = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 数据客户端
            if PolygonClient:
                # 尝试使用环境变量或默认API密钥
                api_key = os.getenv('POLYGON_API_KEY', 'demo_key')
                self.data_client = PolygonClient(api_key)
                self.logger.info("✅ 数据客户端初始化成功")
            else:
                self.logger.warning("❌ 数据客户端不可用")
                
            # Alpha引擎
            if AlphaStrategiesEngine:
                self.alpha_engine = AlphaStrategiesEngine()
                self.logger.info("✅ Alpha引擎初始化成功")
            else:
                self.logger.warning("❌ Alpha引擎不可用")
                
            # 索引对齐器
            if INDEX_ALIGNER_AVAILABLE:
                self.index_aligner = create_index_aligner(horizon=10)
                self.logger.info("✅ 索引对齐器初始化成功")
            else:
                self.logger.warning("❌ 索引对齐器不可用")
                
            # ML管理器
            if SIMPLE_ML_AVAILABLE:
                ml_config = SimpleMLConfig()
                self.ml_manager = SimpleMLManager(ml_config)
                self.logger.info("✅ ML管理器初始化成功")
            else:
                self.logger.warning("❌ ML管理器不可用")
                
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
    
    def get_data(self, tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        if not self.data_client:
            self.logger.error("数据客户端不可用")
            return None
            
        try:
            # 获取数据
            data = self.data_client.get_stock_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"获取数据成功: {data.shape}")
                return data
            else:
                self.logger.warning("获取数据为空")
                return None
                
        except Exception as e:
            self.logger.error(f"获取数据失败: {e}")
            return None
    
    def create_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """创建特征"""
        if not self.alpha_engine:
            self.logger.error("Alpha引擎不可用")
            return None
            
        try:
            # 使用Alpha引擎创建特征
            features = self.alpha_engine.compute_all_alphas(data)
            
            if features is not None and not features.empty:
                self.logger.info(f"特征创建成功: {features.shape}")
                return features
            else:
                self.logger.warning("特征创建结果为空")
                return None
                
        except Exception as e:
            self.logger.error(f"特征创建失败: {e}")
            return None
    
    def prepare_ml_data(self, features: pd.DataFrame, target_column: str = 'target') -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """准备ML训练数据"""
        if features is None or features.empty:
            self.logger.error("特征数据为空")
            return None
            
        try:
            # 分离特征和目标
            if target_column in features.columns:
                y = features[target_column]
                X = features.drop(columns=[target_column])
            else:
                # 如果没有目标列，创建简单的未来收益作为目标
                if 'close' in features.columns:
                    # 创建未来收益作为目标
                    if isinstance(features.index, pd.MultiIndex):
                        # MultiIndex处理
                        features_sorted = features.sort_index(level=0)
                        y = features_sorted.groupby(level=1)['close'].pct_change(periods=10).shift(-10)
                    else:
                        y = features['close'].pct_change(periods=10).shift(-10)
                    X = features
                else:
                    self.logger.error("无法创建目标变量：缺少close列")
                    return None
            
            # 删除缺失值
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) == 0:
                self.logger.error("清理后数据为空")
                return None
                
            self.logger.info(f"ML数据准备完成: X={X_clean.shape}, y={len(y_clean)}")
            return X_clean, y_clean
            
        except Exception as e:
            self.logger.error(f"ML数据准备失败: {e}")
            return None
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, Any]]:
        """训练ML模型"""
        if not self.ml_manager:
            self.logger.error("ML管理器不可用")
            return None
            
        try:
            # 验证数据
            validation = self.ml_manager.validate_data_for_ml(X, y)
            if not validation['ready_for_training']:
                self.logger.error(f"数据不适合训练: {validation['issues']}")
                return None
                
            # 准备数据
            X_prepared, y_prepared = self.ml_manager.prepare_data_for_ml(X, y)
            
            # 获取模型
            models = self.ml_manager.get_simple_ml_models()
            
            if not models:
                self.logger.error("没有可用的ML模型")
                return None
                
            # 训练模型
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_prepared, y_prepared)
                    trained_models[name] = model
                    self.logger.info(f"✅ {name}模型训练完成")
                except Exception as model_e:
                    self.logger.error(f"❌ {name}模型训练失败: {model_e}")
                    
            if trained_models:
                self.logger.info(f"成功训练{len(trained_models)}个模型")
                return trained_models
            else:
                self.logger.error("所有模型训练失败")
                return None
                
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return None
    
    def predict(self, models: Dict[str, Any], X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """使用训练好的模型进行预测"""
        if not models or not self.ml_manager:
            self.logger.error("模型或ML管理器不可用")
            return None
            
        try:
            # 准备预测数据
            X_prepared, _ = self.ml_manager.prepare_data_for_ml(X, pd.Series(index=X.index))
            
            # 进行预测
            predictions = {}
            for name, model in models.items():
                try:
                    pred = model.predict(X_prepared)
                    predictions[name] = pred
                    self.logger.info(f"✅ {name}模型预测完成")
                except Exception as pred_e:
                    self.logger.error(f"❌ {name}模型预测失败: {pred_e}")
                    
            if predictions:
                # 将预测结果组合成DataFrame
                pred_df = pd.DataFrame(predictions, index=X.index)
                
                # 计算集成预测（简单平均）
                pred_df['ensemble'] = pred_df.mean(axis=1)
                
                self.logger.info(f"预测完成: {pred_df.shape}")
                return pred_df
            else:
                self.logger.error("所有模型预测失败")
                return None
                
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return None
    
    def run_simple_pipeline(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """运行简化的完整流水线"""
        self.logger.info("🚀 开始运行简化BMA流水线")
        
        results = {
            'success': False,
            'data': None,
            'features': None,
            'models': None,
            'predictions': None,
            'errors': []
        }
        
        try:
            # 1. 获取数据
            self.logger.info("📊 步骤1: 获取数据")
            data = self.get_data(tickers, start_date, end_date)
            if data is None:
                results['errors'].append("数据获取失败")
                return results
            results['data'] = data
            
            # 2. 创建特征
            self.logger.info("🔧 步骤2: 创建特征")
            features = self.create_features(data)
            if features is None:
                results['errors'].append("特征创建失败")
                return results
            results['features'] = features
            
            # 3. 准备ML数据
            self.logger.info("📋 步骤3: 准备ML数据")
            ml_data = self.prepare_ml_data(features)
            if ml_data is None:
                results['errors'].append("ML数据准备失败")
                return results
            X, y = ml_data
            
            # 4. 训练模型
            self.logger.info("🤖 步骤4: 训练ML模型")
            models = self.train_models(X, y)
            if models is None:
                results['errors'].append("模型训练失败")
                return results
            results['models'] = models
            
            # 5. 生成预测
            self.logger.info("🔮 步骤5: 生成预测")
            predictions = self.predict(models, X)
            if predictions is None:
                results['errors'].append("预测生成失败")
                return results
            results['predictions'] = predictions
            
            results['success'] = True
            self.logger.info("✅ 简化BMA流水线运行完成")
            
        except Exception as e:
            self.logger.error(f"流水线运行失败: {e}")
            results['errors'].append(str(e))
            
        return results

def create_simplified_bma_model(config: Dict[str, Any] = None) -> SimplifiedBMAModel:
    """创建简化BMA模型"""
    return SimplifiedBMAModel(config)

if __name__ == "__main__":
    # 测试简化模型
    model = create_simplified_bma_model()
    
    # 运行测试流水线
    test_tickers = ['AAPL', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    results = model.run_simple_pipeline(test_tickers, start_date, end_date)
    
    print("\n" + "="*60)
    print("简化BMA模型测试结果")
    print("="*60)
    print(f"成功: {results['success']}")
    if results['errors']:
        print(f"错误: {results['errors']}")
    if results['data'] is not None:
        print(f"数据: {results['data'].shape}")
    if results['features'] is not None:
        print(f"特征: {results['features'].shape}")
    if results['models']:
        print(f"模型: {list(results['models'].keys())}")
    if results['predictions'] is not None:
        print(f"预测: {results['predictions'].shape}")