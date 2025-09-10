#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA预测结果Excel输出模块
按预测收益率从高到低排序输出
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BMAExcelExporter:
    """BMA结果Excel导出器"""
    
    def __init__(self, output_dir: str = "D:/trade/results"):
        """
        初始化导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_predictions(self, 
                          predictions: np.ndarray,
                          feature_data: pd.DataFrame,
                          model_info: Dict[str, Any],
                          filename: Optional[str] = None) -> str:
        """
        导出预测结果到Excel
        
        Args:
            predictions: 预测收益率数组
            feature_data: 原始特征数据 (包含ticker, date等)
            model_info: 模型信息字典
            filename: 输出文件名 (可选)
            
        Returns:
            输出文件路径
        """
        
        # 创建结果DataFrame
        results_df = feature_data.copy()
        results_df['predicted_return'] = predictions
        results_df['predicted_return_pct'] = predictions * 100  # 转换为百分比
        
        # 按预测收益率从高到低排序
        results_df = results_df.sort_values('predicted_return', ascending=False).reset_index(drop=True)
        
        # 添加排名列
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # 重新排列列的顺序
        main_cols = ['rank', 'ticker', 'predicted_return_pct', 'predicted_return']
        if 'date' in results_df.columns:
            main_cols.insert(2, 'date')
        
        other_cols = [col for col in results_df.columns if col not in main_cols]
        results_df = results_df[main_cols + other_cols]
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"BMA_Predictions_{timestamp}.xlsx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # 写入预测结果
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # 写入模型信息
            model_info_df = pd.DataFrame([
                ['模型类型', model_info.get('model_type', 'BMA Ultra Enhanced')],
                ['训练时间', model_info.get('training_time', 'N/A')],
                ['样本数量', model_info.get('n_samples', 'N/A')],
                ['特征数量', model_info.get('n_features', 'N/A')],
                ['最佳模型', model_info.get('best_model', 'N/A')],
                ['CV分数', model_info.get('cv_score', 'N/A')],
                ['IC分数', model_info.get('ic_score', 'N/A')],
                ['R²分数', model_info.get('r2_score', 'N/A')],
                ['导出时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ], columns=['指标', '数值'])
            
            model_info_df.to_excel(writer, sheet_name='Model_Info', index=False)
            
            # 写入统计摘要
            summary_stats = pd.DataFrame([
                ['总股票数', len(results_df)],
                ['预测收益率均值(%)', f"{results_df['predicted_return_pct'].mean():.4f}"],
                ['预测收益率中位数(%)', f"{results_df['predicted_return_pct'].median():.4f}"],
                ['预测收益率标准差(%)', f"{results_df['predicted_return_pct'].std():.4f}"],
                ['最高预测收益率(%)', f"{results_df['predicted_return_pct'].max():.4f}"],
                ['最低预测收益率(%)', f"{results_df['predicted_return_pct'].min():.4f}"],
                ['前10%股票数量', len(results_df) // 10],
                ['前20%股票数量', len(results_df) // 5]
            ], columns=['统计项', '数值'])
            
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            # 如果有股票代码，按代码分组统计
            if 'ticker' in results_df.columns:
                ticker_stats = results_df.groupby('ticker').agg({
                    'predicted_return_pct': ['mean', 'std', 'count'],
                    'rank': 'mean'
                }).round(4)
                
                ticker_stats.columns = ['平均预测收益率(%)', '收益率标准差(%)', '记录数', '平均排名']
                ticker_stats = ticker_stats.sort_values('平均预测收益率(%)', ascending=False)
                ticker_stats.to_excel(writer, sheet_name='By_Ticker')
        
        logger.info(f"预测结果已导出至: {filepath}")
        logger.info(f"共导出 {len(results_df)} 条预测记录")
        logger.info(f"预测收益率范围: {results_df['predicted_return_pct'].min():.4f}% 到 {results_df['predicted_return_pct'].max():.4f}%")
        
        return filepath
    
    def export_top_k_predictions(self, 
                                predictions: np.ndarray,
                                feature_data: pd.DataFrame,
                                model_info: Dict[str, Any],
                                k: int = 50,
                                filename: Optional[str] = None) -> str:
        """
        导出Top-K预测结果
        
        Args:
            predictions: 预测收益率数组
            feature_data: 原始特征数据
            model_info: 模型信息字典
            k: 保留前K个结果
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        
        # 创建完整结果DataFrame
        results_df = feature_data.copy()
        results_df['predicted_return'] = predictions
        results_df['predicted_return_pct'] = predictions * 100
        
        # 按预测收益率从高到低排序，保留前K个
        results_df = results_df.sort_values('predicted_return', ascending=False).head(k).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"BMA_Top{k}_Predictions_{timestamp}.xlsx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 重新排列列顺序
        main_cols = ['rank', 'ticker', 'predicted_return_pct', 'predicted_return']
        if 'date' in results_df.columns:
            main_cols.insert(2, 'date')
        
        other_cols = [col for col in results_df.columns if col not in main_cols]
        results_df = results_df[main_cols + other_cols]
        
        # 导出Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name=f'Top_{k}_Predictions', index=False)
            
            # 模型信息
            model_info_df = pd.DataFrame([
                ['选择数量', f'Top {k}'],
                ['模型类型', model_info.get('model_type', 'BMA Ultra Enhanced')],
                ['训练样本数', model_info.get('n_samples', 'N/A')],
                ['特征数量', model_info.get('n_features', 'N/A')],
                ['最佳基础模型', model_info.get('best_model', 'N/A')],
                ['模型性能(IC)', model_info.get('ic_score', 'N/A')],
                ['导出时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ], columns=['项目', '值'])
            
            model_info_df.to_excel(writer, sheet_name='Selection_Info', index=False)
        
        logger.info(f"Top-{k} 预测结果已导出至: {filepath}")
        logger.info(f"最高预测收益率: {results_df['predicted_return_pct'].iloc[0]:.4f}%")
        logger.info(f"第{k}名预测收益率: {results_df['predicted_return_pct'].iloc[-1]:.4f}%")
        
        return filepath

# 便捷函数
def export_bma_predictions_to_excel(predictions: np.ndarray,
                                   feature_data: pd.DataFrame, 
                                   model_info: Dict[str, Any],
                                   output_dir: str = "D:/trade/predictions",
                                   top_k: Optional[int] = None,
                                   filename: Optional[str] = None) -> str:
    """
    便捷函数：导出BMA预测结果到Excel
    
    Args:
        predictions: 预测收益率
        feature_data: 特征数据
        model_info: 模型信息
        output_dir: 输出目录
        top_k: 如果指定，只导出前K个结果
        filename: 自定义文件名
        
    Returns:
        输出文件路径
    """
    
    exporter = BMAExcelExporter(output_dir)
    
    if top_k:
        return exporter.export_top_k_predictions(
            predictions, feature_data, model_info, k=top_k, filename=filename
        )
    else:
        return exporter.export_predictions(
            predictions, feature_data, model_info, filename=filename
        )