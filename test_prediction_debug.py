#!/usr/bin/env python3
"""
测试BMA Enhanced预测生成的实际问题
重现完整分析流程并定位具体失败点
"""

import sys
import os
sys.path.append('bma_models')
sys.path.append('autotrader')

import logging
import importlib.util
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_prediction_with_longer_timeframe():
    """测试更长时间范围的数据，避免数据不足问题"""
    
    try:
        # 导入BMA Enhanced模型
        spec = importlib.util.spec_from_file_location(
            'bma_ultra_enhanced', 
            'bma_models/量化模型_bma_ultra_enhanced.py'
        )
        bma_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bma_module)
        
        model = bma_module.UltraEnhancedQuantitativeModel('alphas_config.yaml')
        logger.info("BMA Enhanced模型初始化成功")
        
        # 使用更长的时间范围确保数据充足
        test_tickers = ['AAPL', 'MSFT']
        start_date = '2022-01-01'  # 扩展到2年多数据
        end_date = '2024-08-01'
        
        logger.info(f"开始测试 - 股票: {test_tickers}, 时间: {start_date} to {end_date}")
        
        # 运行完整分析
        result = model.run_complete_analysis(
            tickers=test_tickers,
            start_date=start_date,
            end_date=end_date,
            top_n=10
        )
        
        logger.info("完整分析调用完成")
        
        # 详细检查结果
        if result:
            logger.info(f"结果类型: {type(result)}")
            logger.info(f"结果键: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
            
            success = result.get('success', False)
            logger.info(f"成功状态: {success}")
            
            if success:
                predictions = result.get('predictions')
                logger.info(f"预测结果类型: {type(predictions)}")
                
                if predictions is not None:
                    if hasattr(predictions, 'shape'):
                        logger.info(f"预测形状: {predictions.shape}")
                    if hasattr(predictions, 'empty'):
                        logger.info(f"预测是否为空: {predictions.empty}")
                    if hasattr(predictions, 'columns'):
                        logger.info(f"预测列: {list(predictions.columns)}")
                        
                    logger.info("✅ 预测生成成功！")
                else:
                    logger.error("❌ 预测为None")
                    
            else:
                error_msg = result.get('error', '无错误信息')
                logger.error(f"❌ 分析失败: {error_msg}")
                
        else:
            logger.error("❌ 结果为空")
            
    except Exception as e:
        logger.error(f"测试异常: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("=== 开始BMA Enhanced预测生成测试 ===")
    test_prediction_with_longer_timeframe()
    logger.info("=== 测试完成 ===")