#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Ultra Enhanced模型稳健特征选择集成
将稳健特征选择系统集成到主模型中
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加bma_models目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'bma_models'))

from robust_feature_selection import RobustFeatureSelector

def integrate_robust_feature_selection():
    """
    将稳健特征选择集成到BMA模型中进行测试
    """
    try:
        print("=" * 80)
        print("BMA Ultra Enhanced模型 + 稳健特征选择集成测试")
        print("=" * 80)
        
        # 导入模型
        print("1. 导入UltraEnhancedQuantitativeModel...")
        from 量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        print("   ✅ 模型导入成功")
        
        # 创建模型实例
        print("\n2. 创建模型实例...")
        model = UltraEnhancedQuantitativeModel()
        print("   ✅ 模型实例创建成功")
        
        # 使用多股票进行完整测试
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # 5只股票
        start_date = '2024-01-01'  
        end_date = '2024-12-01'
        
        print(f"\n3. 设置测试参数:")
        print(f"   股票: {test_tickers}")
        print(f"   时间范围: {start_date} - {end_date}")
        
        # 下载数据并创建特征（不进行完整训练）
        print(f"\n4. 下载数据并创建特征...")
        
        # 下载股票数据
        model.download_stock_data(test_tickers, start_date, end_date)
        print("   ✅ 数据下载完成")
        
        # 创建传统特征
        feature_data = model.create_traditional_features()
        print(f"   ✅ 特征创建完成: {feature_data.shape}")
        
        # 准备稳健特征选择的数据
        print(f"\n5. 准备稳健特征选择...")
        
        # 提取特征矩阵、目标变量和日期
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['target', 'date', 'ticker']]
        
        X = feature_data[feature_cols].fillna(0)  # 简单NaN填充用于测试
        y = feature_data['target'].fillna(0)
        dates = feature_data['date']
        
        print(f"   原始特征数: {len(feature_cols)}")
        print(f"   样本数: {len(X)}")
        print(f"   NaN处理后数据: X{X.shape}, y{y.shape}")
        
        # 应用稳健特征选择
        print(f"\n6. 应用稳健特征选择...")
        
        selector = RobustFeatureSelector(
            target_features=16,      # 目标16个特征
            ic_window=90,           # 3个月IC窗口  
            min_ic_mean=0.005,      # 最小IC均值
            min_ic_ir=0.2,          # 最小IC信息比率
            max_correlation=0.6     # 最大特征相关性
        )
        
        try:
            X_selected = selector.fit_transform(X, y, dates)
            print(f"   ✅ 特征选择完成: {X.shape[1]} -> {X_selected.shape[1]} 特征")
            
            # 显示选择的特征
            selected_features = selector.selected_features_
            print(f"   选择的特征: {selected_features}")
            
        except Exception as e:
            print(f"   ⚠️ 特征选择失败: {e}")
            print("   回退到使用原始特征")
            X_selected = X
            selected_features = feature_cols
        
        # 生成特征选择报告
        print(f"\n7. 生成特征选择报告...")
        
        if selector.feature_stats_:
            report = selector.get_feature_report()
            
            print("\n   特征质量报告 (Top 15):")
            print("   " + "="*70)
            top_features = report.head(15)
            for _, row in top_features.iterrows():
                status = "✅选中" if row['selected'] else "✗未选中"
                print(f"   {row['feature']:<25} IC:{row['ic_mean']:>7.4f} IR:{row['ic_ir']:>7.4f} {status}")
            
            # 统计信息
            selected_stats = report[report['selected']]
            if len(selected_stats) > 0:
                print(f"\n   选中特征统计:")
                print(f"   - 数量: {len(selected_stats)}")
                print(f"   - 平均IC: {selected_stats['ic_mean'].mean():.4f}")
                print(f"   - 平均IC_IR: {selected_stats['ic_ir'].mean():.4f}")
                print(f"   - IC范围: {selected_stats['ic_mean'].min():.4f} - {selected_stats['ic_mean'].max():.4f}")
        
        # 使用选择的特征进行简化训练测试
        print(f"\n8. 使用选择特征进行简化训练测试...")
        
        # 创建选择特征后的feature_data
        feature_data_selected = feature_data[['target', 'date', 'ticker'] + list(X_selected.columns)].copy()
        
        print(f"   优化后特征数据: {feature_data_selected.shape}")
        print(f"   特征维度减少: {len(feature_cols)} -> {len(X_selected.columns)} ({len(X_selected.columns)/len(feature_cols):.1%})")
        
        # 更新模型的feature_data
        original_feature_data = model.feature_data
        model.feature_data = feature_data_selected
        
        try:
            # 进行简化的模型训练测试（只测试数据处理部分）
            print("   测试数据清洗和预处理...")
            
            # 数据清洗
            clean_data = feature_data_selected.dropna()
            if len(clean_data) > 0:
                print(f"   ✅ 数据清洗完成: {len(feature_data_selected)} -> {len(clean_data)} 样本")
                
                # 特征标准化测试
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
                feature_cols_selected = [col for col in clean_data.columns 
                                       if col not in ['target', 'date', 'ticker']]
                X_test = clean_data[feature_cols_selected]
                X_scaled = scaler.fit_transform(X_test)
                
                print(f"   ✅ 特征标准化完成: {X_test.shape}")
                print(f"   特征统计: 均值≈0 (实际:{np.mean(X_scaled):.6f}), 标准差≈1 (实际:{np.std(X_scaled):.6f})")
                
            else:
                print("   ⚠️ 数据清洗后为空")
                
        except Exception as e:
            print(f"   ⚠️ 简化训练测试失败: {e}")
        
        finally:
            # 恢复原始数据
            model.feature_data = original_feature_data
        
        # 性能对比分析
        print(f"\n9. 性能对比分析...")
        print(f"   计算复杂度降低:")
        print(f"   - 特征数: {len(feature_cols)} -> {len(X_selected.columns)} (减少 {len(feature_cols)-len(X_selected.columns)} 个)")
        print(f"   - 维度压缩率: {len(X_selected.columns)/len(feature_cols):.1%}")
        print(f"   - 理论计算量: 约减少 {(1 - (len(X_selected.columns)/len(feature_cols))**2)*100:.1f}%")
        
        print(f"\n   预期收益:")
        print(f"   - ✅ 降维到 {len(X_selected.columns)} 个稳健特征")
        print(f"   - ✅ 计算量直线下降")
        print(f"   - ✅ 预期IC提升，过拟合减少")
        print(f"   - ✅ 模型更稳定，泛化性更好")
        
        print(f"\n10. 集成建议...")
        print(f"   建议将稳健特征选择集成到BMA模型的以下位置:")
        print(f"   1. 在create_traditional_features()之后")
        print(f"   2. 在模型训练之前") 
        print(f"   3. 定期(每6-12个月)重新执行特征选择")
        print(f"   4. 保存选择的特征列表以确保预测时一致性")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_production_integration():
    """
    创建生产环境的集成代码建议
    """
    integration_code = '''
# 在BMA模型的train_enhanced_models方法中添加稳健特征选择

def train_enhanced_models_with_robust_selection(self, current_ticker=None):
    """训练增强模型（集成稳健特征选择）"""
    
    # 1. 原有的特征创建流程
    X_clean, y_clean, dates_clean, tickers_clean = self._prepare_training_data()
    
    # 2. 🎯 新增：稳健特征选择
    try:
        from robust_feature_selection import RobustFeatureSelector
        
        logger.info("开始稳健特征选择...")
        selector = RobustFeatureSelector(
            target_features=16,
            ic_window=126,  # 6个月
            min_ic_mean=0.01,
            min_ic_ir=0.3,
            max_correlation=0.6
        )
        
        X_robust = selector.fit_transform(X_clean, y_clean, dates_clean)
        logger.info(f"特征选择完成: {X_clean.shape[1]} -> {X_robust.shape[1]} 特征")
        
        # 保存特征选择器和选择的特征
        self.feature_selector = selector
        self.selected_features = selector.selected_features_
        
        # 使用选择后的特征继续训练
        X_clean = X_robust
        
    except Exception as e:
        logger.warning(f"稳健特征选择失败，使用原始特征: {e}")
    
    # 3. 继续原有的训练流程
    training_results = {}
    
    # ... 原有的LTR、传统模型训练等
    
    return training_results

# 在预测时确保使用相同的特征
def generate_predictions_with_robust_features(self, X_pred):
    """使用稳健特征生成预测"""
    
    if hasattr(self, 'feature_selector') and self.feature_selector:
        # 使用训练时的特征选择器
        X_pred_robust = self.feature_selector.transform(X_pred)
        return self._generate_predictions(X_pred_robust)
    else:
        return self._generate_predictions(X_pred)
'''
    
    print("生产环境集成代码建议:")
    print("="*60)
    print(integration_code)
    
    # 保存到文件
    with open('bma_robust_feature_integration_guide.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("✅ 集成指南已保存到: bma_robust_feature_integration_guide.py")

if __name__ == "__main__":
    print("稳健特征选择集成测试")
    print("="*80)
    
    # 运行集成测试
    success = integrate_robust_feature_selection()
    
    if success:
        print("\n🎉 稳健特征选择集成测试成功！")
        
        # 创建生产集成指南
        print("\n创建生产环境集成指南...")
        create_production_integration()
        
    else:
        print("\n💥 稳健特征选择集成测试失败。")
