#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Ultra Enhanced模型生产环境稳健特征选择集成
实际可用的集成代码，可以直接加入到主模型中
"""

from robust_feature_selection import RobustFeatureSelector
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BMAEnhancedWithRobustFeatures:
    """
    集成稳健特征选择的BMA模型增强版
    """
    
    def __init__(self, base_model):
        """
        初始化增强版模型
        
        Args:
            base_model: 原始BMA模型实例
        """
        self.base_model = base_model
        self.feature_selector = None
        self.selected_features = None
        self.feature_selection_enabled = True
        
        # 特征选择配置
        self.robust_config = {
            'target_features': 16,      # 目标特征数
            'ic_window': 126,          # 6个月IC窗口
            'min_ic_mean': 0.005,      # 最小IC均值
            'min_ic_ir': 0.2,          # 最小IC信息比率
            'max_correlation': 0.6,     # 最大特征相关性
            'reselection_period': 180,  # 重选周期(天)
        }
        
        # 记录上次特征选择的时间
        self.last_selection_date = None
    
    def should_reselect_features(self, current_date):
        """
        判断是否需要重新进行特征选择
        
        Args:
            current_date: 当前日期
            
        Returns:
            bool: 是否需要重新选择
        """
        if self.last_selection_date is None:
            return True
        
        days_since_last = (current_date - self.last_selection_date).days
        return days_since_last >= self.robust_config['reselection_period']
    
    def apply_robust_feature_selection(self, X, y, dates, force_reselect=False):
        """
        应用稳健特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            force_reselect: 强制重新选择
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        try:
            current_date = dates.max() if hasattr(dates, 'max') else pd.Timestamp.now()
            
            # 检查是否需要重新选择特征
            if (force_reselect or 
                self.feature_selector is None or 
                self.should_reselect_features(current_date)):
                
                logger.info("开始稳健特征选择...")
                logger.info(f"输入特征数: {X.shape[1]}, 样本数: {len(X)}")
                
                # 创建特征选择器
                selector = RobustFeatureSelector(
                    target_features=self.robust_config['target_features'],
                    ic_window=self.robust_config['ic_window'],
                    min_ic_mean=self.robust_config['min_ic_mean'],
                    min_ic_ir=self.robust_config['min_ic_ir'],
                    max_correlation=self.robust_config['max_correlation']
                )
                
                # 执行特征选择
                X_selected = selector.fit_transform(X, y, dates)
                
                # 保存选择器和结果
                self.feature_selector = selector
                self.selected_features = selector.selected_features_
                self.last_selection_date = current_date
                
                logger.info(f"✅ 特征选择完成: {X.shape[1]} -> {X_selected.shape[1]} 特征")
                logger.info(f"选择的特征: {self.selected_features}")
                
                # 生成选择报告
                self._log_selection_report(selector)
                
                return X_selected
            
            else:
                # 使用已保存的特征选择
                if self.selected_features and all(col in X.columns for col in self.selected_features):
                    X_selected = X[self.selected_features]
                    logger.info(f"使用已选择的特征: {len(self.selected_features)}个")
                    return X_selected
                else:
                    logger.warning("已选择的特征不完整，使用原始特征")
                    return X
            
        except Exception as e:
            logger.error(f"稳健特征选择失败: {e}")
            logger.warning("回退到使用原始特征")
            return X
    
    def _log_selection_report(self, selector):
        """记录特征选择报告"""
        if selector.feature_stats_:
            report = selector.get_feature_report()
            selected_stats = report[report['selected']]
            
            if len(selected_stats) > 0:
                avg_ic = selected_stats['ic_mean'].mean()
                avg_ic_ir = selected_stats['ic_ir'].mean()
                ic_range = (selected_stats['ic_mean'].min(), selected_stats['ic_mean'].max())
                
                logger.info(f"📊 特征选择质量报告:")
                logger.info(f"  - 平均IC: {avg_ic:.4f}")
                logger.info(f"  - 平均IC_IR: {avg_ic_ir:.4f}")
                logger.info(f"  - IC范围: {ic_range[0]:.4f} - {ic_range[1]:.4f}")
                
                # 记录计算效率提升
                compression_ratio = len(selected_stats) / len(report)
                efficiency_gain = (1 - compression_ratio**2) * 100
                logger.info(f"  - 维度压缩: {compression_ratio:.1%}")
                logger.info(f"  - 计算效率提升: ~{efficiency_gain:.1f}%")
    
    def enhanced_create_traditional_features(self, *args, **kwargs):
        """
        增强版特征创建，集成稳健特征选择
        
        Returns:
            pd.DataFrame: 优化后的特征数据
        """
        logger.info("🎯 创建传统特征（集成稳健特征选择）")
        
        # 调用原始的特征创建方法
        feature_data = self.base_model.create_traditional_features(*args, **kwargs)
        
        if not self.feature_selection_enabled:
            logger.info("稳健特征选择已禁用，使用原始特征")
            return feature_data
        
        try:
            # 提取特征和目标
            feature_cols = [col for col in feature_data.columns 
                           if col not in ['target', 'date', 'ticker']]
            
            if len(feature_cols) <= self.robust_config['target_features']:
                logger.info(f"原始特征数({len(feature_cols)})已少于目标数({self.robust_config['target_features']})，跳过特征选择")
                return feature_data
            
            # 准备数据
            clean_data = feature_data.dropna()
            if len(clean_data) == 0:
                logger.warning("清理后数据为空，跳过特征选择")
                return feature_data
            
            X = clean_data[feature_cols].fillna(0)
            y = clean_data['target'].fillna(0)
            dates = clean_data['date']
            
            # 应用稳健特征选择
            X_selected = self.apply_robust_feature_selection(X, y, dates)
            
            # 重构feature_data
            optimized_data = clean_data[['target', 'date', 'ticker']].copy()
            for col in X_selected.columns:
                optimized_data[col] = X_selected[col]
            
            logger.info(f"✅ 特征优化完成: {feature_data.shape} -> {optimized_data.shape}")
            return optimized_data
            
        except Exception as e:
            logger.error(f"特征选择集成失败: {e}")
            logger.warning("回退到原始特征")
            return feature_data
    
    def enhanced_train_enhanced_models(self, current_ticker=None):
        """
        增强版模型训练，使用优化的特征
        
        Args:
            current_ticker: 当前股票代码
            
        Returns:
            训练结果
        """
        logger.info("🚀 开始增强版模型训练（稳健特征优化）")
        
        # 确保特征已经过选择优化
        if hasattr(self.base_model, 'feature_data') and self.base_model.feature_data is not None:
            original_shape = self.base_model.feature_data.shape
            
            # 如果特征数据未经优化，先进行优化
            feature_cols = [col for col in self.base_model.feature_data.columns 
                           if col not in ['target', 'date', 'ticker']]
            
            if (self.feature_selection_enabled and 
                len(feature_cols) > self.robust_config['target_features'] and
                (self.selected_features is None or 
                 not all(col in feature_cols for col in self.selected_features))):
                
                logger.info("特征数据需要优化，应用稳健特征选择")
                self.base_model.feature_data = self.enhanced_create_traditional_features()
            
        # 调用原始的模型训练方法
        training_results = self.base_model.train_enhanced_models(current_ticker)
        
        # 添加特征选择信息到训练结果
        if self.feature_selector:
            training_results['robust_feature_selection'] = {
                'enabled': True,
                'selected_features': self.selected_features,
                'feature_count_original': len(self.base_model.feature_data.columns) - 3,  # 减去target, date, ticker
                'feature_count_selected': len(self.selected_features) if self.selected_features else 0,
                'compression_ratio': len(self.selected_features) / (len(self.base_model.feature_data.columns) - 3) if self.selected_features else 1.0,
                'last_selection_date': self.last_selection_date.isoformat() if self.last_selection_date else None
            }
        else:
            training_results['robust_feature_selection'] = {'enabled': False}
        
        return training_results
    
    def enhanced_generate_predictions(self, *args, **kwargs):
        """
        增强版预测生成，确保使用相同的特征
        
        Returns:
            预测结果
        """
        # 在预测时确保使用相同的特征选择
        if (self.feature_selector and 
            hasattr(self.base_model, 'feature_data') and 
            self.base_model.feature_data is not None):
            
            feature_cols = [col for col in self.base_model.feature_data.columns 
                           if col not in ['target', 'date', 'ticker']]
            
            if self.selected_features and all(col in feature_cols for col in self.selected_features):
                # 只保留选择的特征
                selected_data = self.base_model.feature_data[['target', 'date', 'ticker'] + self.selected_features].copy()
                original_data = self.base_model.feature_data
                self.base_model.feature_data = selected_data
                
                try:
                    # 生成预测
                    predictions = self.base_model.generate_enhanced_predictions(*args, **kwargs)
                    return predictions
                finally:
                    # 恢复原始数据
                    self.base_model.feature_data = original_data
        
        # 如果没有特征选择或出错，使用原始方法
        return self.base_model.generate_enhanced_predictions(*args, **kwargs)
    
    def get_feature_selection_status(self):
        """
        获取特征选择状态
        
        Returns:
            Dict: 特征选择状态信息
        """
        return {
            'enabled': self.feature_selection_enabled,
            'selector_available': self.feature_selector is not None,
            'selected_features_count': len(self.selected_features) if self.selected_features else 0,
            'selected_features': self.selected_features,
            'last_selection_date': self.last_selection_date.isoformat() if self.last_selection_date else None,
            'config': self.robust_config
        }
    
    def set_feature_selection_config(self, **config):
        """
        更新特征选择配置
        
        Args:
            **config: 配置参数
        """
        self.robust_config.update(config)
        logger.info(f"稳健特征选择配置已更新: {config}")
    
    def disable_feature_selection(self):
        """禁用特征选择"""
        self.feature_selection_enabled = False
        logger.info("稳健特征选择已禁用")
    
    def enable_feature_selection(self):
        """启用特征选择"""
        self.feature_selection_enabled = True
        logger.info("稳健特征选择已启用")


def create_enhanced_bma_model(original_model):
    """
    创建集成稳健特征选择的增强版BMA模型
    
    Args:
        original_model: 原始BMA模型实例
        
    Returns:
        BMAEnhancedWithRobustFeatures: 增强版模型
    """
    enhanced_model = BMAEnhancedWithRobustFeatures(original_model)
    
    # 替换原始模型的方法
    enhanced_model.create_traditional_features = enhanced_model.enhanced_create_traditional_features
    enhanced_model.train_enhanced_models = enhanced_model.enhanced_train_enhanced_models
    enhanced_model.generate_enhanced_predictions = enhanced_model.enhanced_generate_predictions
    
    logger.info("✅ BMA模型已增强，集成稳健特征选择功能")
    logger.info(f"📊 配置: 目标{enhanced_model.robust_config['target_features']}个特征，"
                f"IC窗口{enhanced_model.robust_config['ic_window']}天，"
                f"重选周期{enhanced_model.robust_config['reselection_period']}天")
    
    return enhanced_model


# 使用示例
if __name__ == "__main__":
    print("""
    稳健特征选择生产集成示例:
    
    # 1. 集成到现有BMA模型
    from bma_robust_feature_production import create_enhanced_bma_model
    
    # 原始模型
    original_bma = UltraEnhancedQuantitativeModel()
    
    # 创建增强版
    enhanced_bma = create_enhanced_bma_model(original_bma)
    
    # 2. 使用增强版模型（自动应用特征选择）
    result = enhanced_bma.run_complete_analysis(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2024-01-01',
        end_date='2024-12-01'
    )
    
    # 3. 查看特征选择状态
    status = enhanced_bma.get_feature_selection_status()
    print(f"选择的特征数: {status['selected_features_count']}")
    
    # 4. 自定义配置
    enhanced_bma.set_feature_selection_config(
        target_features=20,
        min_ic_mean=0.01,
        reselection_period=90
    )
    
    # 5. 临时禁用/启用
    enhanced_bma.disable_feature_selection()  # 禁用
    enhanced_bma.enable_feature_selection()   # 启用
    """)
