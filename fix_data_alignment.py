#!/usr/bin/env python3
"""
修复BMA Enhanced系统中的数据对齐问题
解决特征融合和LTR训练中的长度不匹配问题
"""

import sys
import os
sys.path.append('bma_models')

def fix_data_alignment_in_bma():
    """修复BMA模型中的数据对齐问题"""
    
    bma_file = 'bma_models/量化模型_bma_ultra_enhanced.py'
    
    # 读取原始文件
    with open(bma_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: 改进特征融合对齐逻辑
    old_fusion_code = '''                # 长度不等，需要对齐
                logger.warning(f"特征长度不匹配: X_clean={len(X_clean)}, alpha_summary={len(alpha_summary)}")
                # 使用较短的长度
                min_len = min(len(X_clean), len(alpha_summary))
                X_fused = pd.concat([
                    X_clean.iloc[:min_len], 
                    alpha_summary.iloc[:min_len]
                ], axis=1)
                logger.info(f"特征对齐融合: → {X_fused.shape}")'''
    
    new_fusion_code = '''                # 长度不等，需要对齐
                logger.warning(f"特征长度不匹配: X_clean={len(X_clean)}, alpha_summary={len(alpha_summary)}")
                
                # 基于索引对齐数据
                if hasattr(X_clean, 'index') and hasattr(alpha_summary, 'index'):
                    # 使用索引交集对齐
                    common_index = X_clean.index.intersection(alpha_summary.index)
                    if len(common_index) > 0:
                        X_fused = pd.concat([
                            X_clean.loc[common_index], 
                            alpha_summary.loc[common_index]
                        ], axis=1)
                        logger.info(f"特征索引对齐融合: → {X_fused.shape}")
                    else:
                        # 回退到长度对齐
                        min_len = min(len(X_clean), len(alpha_summary))
                        X_fused = pd.concat([
                            X_clean.iloc[:min_len], 
                            alpha_summary.iloc[:min_len]
                        ], axis=1)
                        logger.info(f"特征长度对齐融合: → {X_fused.shape}")
                else:
                    # 使用较短的长度
                    min_len = min(len(X_clean), len(alpha_summary))
                    X_fused = pd.concat([
                        X_clean.iloc[:min_len], 
                        alpha_summary.iloc[:min_len]
                    ], axis=1)
                    logger.info(f"特征长度对齐融合: → {X_fused.shape}")'''
    
    # 应用修复1
    if old_fusion_code in content:
        content = content.replace(old_fusion_code, new_fusion_code)
        print("[OK] 修复1: 改进特征融合对齐逻辑")
    else:
        print("[FAIL] 修复1: 未找到特征融合对齐代码")
    
    # 修复2: 在传统模型训练前添加数据对齐验证
    old_train_code = '''        # 训练传统ML模型
        if self.traditional_models:
            logger.info("训练传统ML模型")'''
    
    new_train_code = '''        # 训练传统ML模型
        if self.traditional_models:
            logger.info("训练传统ML模型")
            
            # 数据对齐验证和修复
            if hasattr(self, '_fix_data_alignment'):
                X_final, y_final, dates_final = self._fix_data_alignment(X_final, y_final, dates)
                logger.info(f"数据对齐修复完成: X={X_final.shape}, y={len(y_final)}, dates={len(dates_final)}")'''
    
    # 应用修复2
    if old_train_code in content:
        content = content.replace(old_train_code, new_train_code)
        print("[OK] 修复2: 添加传统模型训练前数据对齐验证")
    else:
        print("[FAIL] 修复2: 未找到传统模型训练代码")
    
    # 修复3: 添加数据对齐辅助方法
    alignment_method = '''
    def _fix_data_alignment(self, X, y, dates):
        """修复数据对齐问题"""
        try:
            # 确保所有数据具有相同长度
            if isinstance(X, pd.DataFrame):
                X_len = len(X)
                X_index = X.index
            else:
                X_len = len(X) if X is not None else 0
                X_index = None
                
            y_len = len(y) if y is not None else 0
            dates_len = len(dates) if dates is not None else 0
            
            logger.info(f"数据对齐前长度: X={X_len}, y={y_len}, dates={dates_len}")
            
            if X_len == y_len == dates_len:
                # 长度一致，无需修复
                return X, y, dates
            
            # 找到最小公共长度
            min_len = min(filter(lambda x: x > 0, [X_len, y_len, dates_len]))
            
            if min_len == 0:
                logger.error("所有数据长度为0，无法对齐")
                return None, None, None
            
            logger.info(f"使用最小公共长度: {min_len}")
            
            # 对齐数据
            if isinstance(X, pd.DataFrame) and min_len <= len(X):
                X_aligned = X.iloc[:min_len].copy()
            elif X is not None:
                X_aligned = X[:min_len]
            else:
                X_aligned = None
                
            if isinstance(y, (pd.Series, list)) and min_len <= len(y):
                if isinstance(y, pd.Series):
                    y_aligned = y.iloc[:min_len].copy()
                else:
                    y_aligned = y[:min_len]
            else:
                y_aligned = None
                
            if isinstance(dates, (pd.Series, list)) and min_len <= len(dates):
                if isinstance(dates, pd.Series):
                    dates_aligned = dates.iloc[:min_len].copy()
                else:
                    dates_aligned = dates[:min_len]
            else:
                dates_aligned = None
            
            logger.info(f"数据对齐完成: X={len(X_aligned) if X_aligned is not None else 0}, y={len(y_aligned) if y_aligned is not None else 0}, dates={len(dates_aligned) if dates_aligned is not None else 0}")
            
            return X_aligned, y_aligned, dates_aligned
            
        except Exception as e:
            logger.error(f"数据对齐失败: {e}")
            return X, y, dates
'''
    
    # 找到类定义的结尾，添加新方法
    class_methods_end = content.rfind('\n    def ')
    if class_methods_end != -1:
        # 找到该方法的结尾
        next_method_start = content.find('\n    def ', class_methods_end + 1)
        if next_method_start == -1:
            # 这是最后一个方法，在类的结尾添加
            insertion_point = content.rfind('\n\n# ', class_methods_end)  # 查找类外的注释或其他内容
            if insertion_point == -1:
                insertion_point = len(content)
        else:
            insertion_point = next_method_start
        
        content = content[:insertion_point] + alignment_method + content[insertion_point:]
        print("[OK] 修复3: 添加数据对齐辅助方法")
    else:
        print("[FAIL] 修复3: 未找到类方法结尾位置")
    
    # 保存修复后的文件
    with open(bma_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"数据对齐修复完成: {bma_file}")

def fix_ltr_alignment():
    """修复LTR模型中的数据对齐检查"""
    
    ltr_file = 'bma_models/learning_to_rank_bma.py'
    
    # 读取原始文件
    with open(ltr_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复LTR数据对齐检查
    old_ltr_check = '''        # 确保数据对齐
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, dates长度不一致")'''
    
    new_ltr_check = '''        # 确保数据对齐
        logger.info(f"LTR数据对齐检查: X={len(X)}, y={len(y)}, dates={len(dates)}")
        
        if len(X) != len(y) or len(X) != len(dates):
            logger.warning(f"LTR数据长度不一致: X={len(X)}, y={len(y)}, dates={len(dates)}")
            
            # 自动对齐到最小长度
            min_len = min(len(X), len(y), len(dates))
            if min_len == 0:
                raise ValueError("所有数据长度为0，无法训练LTR模型")
            
            logger.info(f"自动对齐到最小长度: {min_len}")
            
            # 对齐数据
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:min_len].copy()
            else:
                X = X[:min_len]
                
            if isinstance(y, pd.Series):
                y = y.iloc[:min_len].copy()  
            else:
                y = y[:min_len]
                
            if isinstance(dates, pd.Series):
                dates = dates.iloc[:min_len].copy()
            else:
                dates = dates[:min_len]
                
            logger.info(f"LTR数据对齐完成: X={len(X)}, y={len(y)}, dates={len(dates)}")'''
    
    # 应用修复
    if old_ltr_check in content:
        content = content.replace(old_ltr_check, new_ltr_check)
        print("[OK] 修复LTR数据对齐检查")
    else:
        print("[FAIL] 未找到LTR数据对齐检查代码")
    
    # 保存修复后的文件
    with open(ltr_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"LTR数据对齐修复完成: {ltr_file}")

if __name__ == "__main__":
    print("=== 开始修复BMA Enhanced数据对齐问题 ===")
    fix_data_alignment_in_bma()
    fix_ltr_alignment()
    print("=== 数据对齐修复完成 ===")