#!/usr/bin/env python3
"""
深度验证Alpha特征降维集成到ML训练流程
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deep_verify_alpha_integration():
    """深度验证Alpha特征集成的每个环节"""
    try:
        logger.info("=== 深度验证Alpha特征降维集成 ===")
        
        # 1. 验证Alpha引擎本身
        logger.info("第1步: 验证Alpha引擎...")
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        model = UltraEnhancedQuantitativeModel()
        
        # 检查Alpha引擎状态
        alpha_engine_available = hasattr(model, 'alpha_engine') and model.alpha_engine is not None
        logger.info(f"Alpha引擎可用: {alpha_engine_available}")
        
        if alpha_engine_available:
            logger.info(f"Alpha函数数量: {len(model.alpha_engine.alpha_functions)}")
            logger.info(f"Config中的因子: {len(model.alpha_engine.config.get('alphas', []))}")
            
            # 测试Alpha计算功能
            test_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'open': np.random.randn(100) * 10 + 100,
                'high': np.random.randn(100) * 10 + 105,
                'low': np.random.randn(100) * 10 + 95, 
                'close': np.random.randn(100) * 10 + 100,
                'volume': np.random.randint(1000000, 5000000, 100)
            })
            
            try:
                alpha_result = model.alpha_engine.compute_all_alphas(test_data)
                logger.info(f"Alpha计算成功: {alpha_result.shape if alpha_result is not None else 'None'}")
                if alpha_result is not None:
                    alpha_cols = [col for col in alpha_result.columns if col not in ['date', 'ticker']]
                    logger.info(f"生成的Alpha特征: {len(alpha_cols)}个")
                    logger.info(f"Alpha特征示例: {alpha_cols[:5]}")
                else:
                    logger.warning("Alpha计算返回None")
            except Exception as e:
                logger.error(f"Alpha计算失败: {e}")
        else:
            logger.error("Alpha引擎不可用 - 这是问题的根源")
            return False
        
        # 2. 验证Alpha摘要处理器
        logger.info("\n第2步: 验证Alpha摘要处理器...")
        alpha_processor_available = hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor is not None
        logger.info(f"Alpha摘要处理器可用: {alpha_processor_available}")
        
        if not alpha_processor_available:
            logger.warning("Alpha摘要处理器不可用 - 降维功能无法工作")
        
        # 3. 创建完整的测试数据并验证特征创建
        logger.info("\n第3步: 验证完整特征创建流程...")
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        test_data = []
        for date in dates[::7]:  # 每周一次
            for ticker in tickers:
                test_data.append({
                    'date': date,
                    'ticker': ticker,
                    'open': 100 + np.random.randn() * 5,
                    'high': 105 + np.random.randn() * 5,
                    'low': 95 + np.random.randn() * 5,
                    'close': 100 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                    'target': np.random.randn() * 0.02,
                    'COUNTRY': 'US'
                })
        
        feature_data = pd.DataFrame(test_data)
        logger.info(f"测试数据创建: {feature_data.shape}")
        
        # 准备股票数据
        stock_data = {}
        for ticker in tickers:
            ticker_data = feature_data[feature_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            stock_data[ticker] = ticker_data[['date', 'open', 'high', 'low', 'close', 'volume', 'COUNTRY']]
        
        # 4. 测试传统特征创建
        logger.info("\n第4步: 测试传统特征创建...")
        try:
            traditional_features = model.create_traditional_features(stock_data)
            if traditional_features is not None and not traditional_features.empty:
                traditional_cols = len([col for col in traditional_features.columns 
                                      if col not in ['ticker', 'date', 'target']])
                logger.info(f"传统特征成功: {traditional_features.shape}, 特征列数: {traditional_cols}")
            else:
                logger.error("传统特征创建失败")
                return False
        except Exception as e:
            logger.error(f"传统特征创建异常: {e}")
            return False
        
        # 5. 测试Alpha特征集成
        logger.info("\n第5步: 测试Alpha特征集成...")
        try:
            alpha_result = model._integrate_alpha_summary_features(traditional_features, stock_data)
            if alpha_result is not None and not alpha_result.empty:
                integrated_cols = len([col for col in alpha_result.columns 
                                     if col not in ['ticker', 'date', 'target']])
                added_alpha_features = integrated_cols - traditional_cols
                logger.info(f"Alpha集成结果: {alpha_result.shape}")
                logger.info(f"总特征列数: {integrated_cols}")
                logger.info(f"新增Alpha特征: {added_alpha_features}个")
                
                # 检查Alpha特征名称
                alpha_feature_names = [col for col in alpha_result.columns 
                                     if any(x in col.lower() for x in ['alpha_pc', 'alpha_composite', 'alpha_summary'])]
                logger.info(f"Alpha特征名称: {alpha_feature_names}")
                
                if added_alpha_features > 0:
                    logger.info("SUCCESS: Alpha特征成功添加!")
                    integrated_features = alpha_result
                else:
                    logger.warning("WARNING: Alpha特征未添加，使用传统特征")
                    integrated_features = traditional_features
            else:
                logger.warning("Alpha集成返回None，使用传统特征")
                integrated_features = traditional_features
        except Exception as e:
            logger.error(f"Alpha特征集成失败: {e}")
            import traceback
            traceback.print_exc()
            integrated_features = traditional_features
        
        # 6. 验证ML训练流程中的特征使用
        logger.info("\n第6步: 验证ML训练流程中的特征使用...")
        try:
            # 模拟训练流程中的特征提取
            feature_cols = [col for col in integrated_features.columns 
                           if col not in ['ticker', 'date', 'target', 'COUNTRY']]
            
            X = integrated_features[feature_cols]
            y = integrated_features['target']
            
            logger.info(f"ML训练特征矩阵: X.shape = {X.shape}")
            logger.info(f"ML训练目标变量: y.shape = {y.shape}")
            logger.info(f"特征列数: {len(feature_cols)}")
            
            # 检查是否包含Alpha特征
            alpha_features_in_X = [col for col in feature_cols 
                                 if any(x in col.lower() for x in ['alpha_pc', 'alpha_composite', 'alpha_summary'])]
            logger.info(f"X中的Alpha特征: {len(alpha_features_in_X)}个")
            logger.info(f"Alpha特征名称: {alpha_features_in_X}")
            
            # 验证数据质量
            nan_count = X.isnull().sum().sum()
            inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            
            logger.info(f"数据质量检查: NaN={nan_count}, Inf={inf_count}")
            
            if len(alpha_features_in_X) > 0:
                logger.info("🎉 SUCCESS: Alpha特征已成功集成到ML训练流程!")
                return True
            else:
                logger.warning("⚠️ WARNING: ML训练流程中未发现Alpha特征")
                # 但传统特征工作正常也算部分成功
                if X.shape[1] > 10:  # 至少有一些特征
                    logger.info("✓ 传统特征正常工作，系统基本可用")
                    return "partial"
                else:
                    logger.error("❌ 特征数量不足，系统不可用")
                    return False
                    
        except Exception as e:
            logger.error(f"ML训练流程验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"验证过程失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = deep_verify_alpha_integration()
    
    print("\n" + "="*60)
    print("深度验证结果:")
    print("="*60)
    
    if result is True:
        print("🎉 完全成功: Alpha特征降维已完全集成到ML训练流程")
        print("✓ Alpha引擎工作正常")  
        print("✓ 降维处理成功")
        print("✓ 特征自动包含在ML训练中")
    elif result == "partial":
        print("⚠️ 部分成功: 系统基本可用但Alpha特征集成未完全工作")
        print("✓ 传统特征正常")
        print("⚠️ Alpha特征集成需要进一步调试")
    else:
        print("❌ 失败: Alpha特征降维集成存在严重问题")
        print("需要检查和修复关键组件")
    print("="*60)