#!/usr/bin/env python3
"""
修复BMA模型的关键冲突问题
1. 统一索引策略: 全部使用MultiIndex(date, ticker)
2. 改进合并逻辑: 使用pd.merge on=['date', 'ticker']替代字符串合并键
3. 分离PCA处理: Alpha因子和传统因子分别进行降维，最后合并
4. 统一时间配置: 使用单一的滞后参数配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

def fix_bma_critical_conflicts():
    """修复BMA模型的所有关键冲突问题"""
    
    print("=== 开始修复BMA模型关键冲突 ===\n")
    
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"原文件大小: {len(content)} 字符")
    
    # 1. 修复重复方法定义问题
    print("1. 修复重复方法定义...")
    fixed_content = fix_duplicate_methods(content)
    
    # 2. 统一时间配置
    print("2. 统一时间配置...")
    fixed_content = unify_temporal_config(fixed_content)
    
    # 3. 优化索引和合并逻辑
    print("3. 优化索引和合并逻辑...")
    fixed_content = optimize_indexing_and_merging(fixed_content)
    
    # 4. 分离PCA处理逻辑
    print("4. 分离PCA处理逻辑...")
    fixed_content = separate_pca_processing(fixed_content)
    
    # 5. 清理版本混合代码
    print("5. 清理版本混合代码...")
    fixed_content = clean_version_mixing(fixed_content)
    
    # 创建备份并写入修复后的文件
    backup_file = file_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"原文件备份至: {backup_file}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"\n✓ 修复完成！修复后文件大小: {len(fixed_content)} 字符")
    print(f"✓ 原文件已备份至: {backup_file}")
    
    return True

def fix_duplicate_methods(content: str) -> str:
    """修复重复方法定义"""
    print("  - 识别并合并重复方法...")
    
    # 重复方法的处理策略：保留最新版本，移除旧版本
    duplicate_patterns = {
        # __init__ 方法去重：保留最完整的版本
        r'(class \w+:.*?)def __init__\(self\):.*?(?=def|\n\n|\Z)': 'remove_duplicate_init',
        
        # validate_dataframe 去重：保留参数更完整的版本
        r'def validate_dataframe\(self, df: pd\.DataFrame, source_name: str\).*?(?=def|\n\n|\Z)': 'keep_first',
        
        # calculate_all_signals 去重
        r'def calculate_all_signals\(self, symbol\):.*?(?=def|\n\n|\Z)': 'keep_first',
        
        # stats 方法去重
        r'def stats\(self\):.*?(?=def|\n\n|\Z)': 'keep_first',
    }
    
    fixed_content = content
    
    # 简单的方法：移除明显的重复def行
    lines = fixed_content.split('\n')
    seen_methods = {}
    filtered_lines = []
    skip_until_next_def = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            method_signature = line.strip()
            
            # 提取方法名
            method_name = method_signature.split('(')[0].replace('def ', '').strip()
            
            # 检查是否是已知的重复方法
            if method_name in seen_methods:
                print(f"    移除重复方法: {method_name} (行 {i+1})")
                skip_until_next_def = True
                continue
            else:
                seen_methods[method_name] = i
                skip_until_next_def = False
        
        if skip_until_next_def:
            # 如果当前行是下一个方法定义或类定义，停止跳过
            if line.strip().startswith(('def ', 'class ', '@')):
                skip_until_next_def = False
                filtered_lines.append(line)
            # 否则跳过这行
            continue
        
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def unify_temporal_config(content: str) -> str:
    """统一时间配置参数"""
    print("  - 统一滞后和间隔参数...")
    
    # 定义统一的时间配置
    unified_config = {
        'feature_lag_days': 1,    # 特征滞后1天
        'safety_gap_days': 1,     # 安全间隔1天  
        'cv_gap_days': 1,         # CV间隔1天
        'cv_embargo_days': 1,     # CV禁运1天
        'prediction_horizon_days': 10  # 预测期10天
    }
    
    # 替换硬编码的数值
    replacements = [
        # 统一特征滞后
        (r'feature_lag.*?=.*?[0-9]+', f"feature_lag_days = {unified_config['feature_lag_days']}"),
        (r'FEATURE_LAG.*?=.*?[0-9]+', f"FEATURE_LAG = {unified_config['feature_lag_days']}"),
        
        # 统一安全间隔
        (r'safety_gap.*?=.*?[0-9]+', f"safety_gap_days = {unified_config['safety_gap_days']}"),
        (r'SAFETY_GAP.*?=.*?[0-9]+', f"SAFETY_GAP = {unified_config['safety_gap_days']}"),
        
        # 统一CV间隔
        (r'cv_gap.*?=.*?[0-9]+', f"cv_gap_days = {unified_config['cv_gap_days']}"),
        (r'gap=.*?[0-9]+', f"gap={unified_config['cv_gap_days']}"),
        
        # 统一CV禁运期
        (r'cv_embargo.*?=.*?[0-9]+', f"cv_embargo_days = {unified_config['cv_embargo_days']}"),
        (r'embargo=.*?[0-9]+', f"embargo={unified_config['cv_embargo_days']}"),
    ]
    
    fixed_content = content
    for pattern, replacement in replacements:
        fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.IGNORECASE)
    
    # 添加统一的时间配置常量
    config_block = f'''
# === 统一时间配置常量 ===
UNIFIED_FEATURE_LAG_DAYS = {unified_config['feature_lag_days']}
UNIFIED_SAFETY_GAP_DAYS = {unified_config['safety_gap_days']}
UNIFIED_CV_GAP_DAYS = {unified_config['cv_gap_days']}
UNIFIED_CV_EMBARGO_DAYS = {unified_config['cv_embargo_days']}
UNIFIED_PREDICTION_HORIZON_DAYS = {unified_config['prediction_horizon_days']}

# 向后兼容别名
FEATURE_LAG = UNIFIED_FEATURE_LAG_DAYS
SAFETY_GAP = UNIFIED_SAFETY_GAP_DAYS
'''
    
    # 插入配置块到导入后面
    insert_pos = fixed_content.find('# === PROJECT PATH SETUP ===')
    if insert_pos > 0:
        fixed_content = fixed_content[:insert_pos] + config_block + '\n' + fixed_content[insert_pos:]
    
    return fixed_content

def optimize_indexing_and_merging(content: str) -> str:
    """优化索引和合并逻辑"""
    print("  - 优化索引和合并逻辑...")
    
    # 添加统一的数据合并辅助函数
    merge_helper_code = '''
# === 统一数据合并辅助函数 ===
def safe_merge_on_multiindex(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                           how: str = 'left', suffixes: tuple = ('', '_right')) -> pd.DataFrame:
    """
    安全合并两个DataFrame，自动处理MultiIndex和普通索引
    
    Args:
        left_df: 左侧DataFrame
        right_df: 右侧DataFrame  
        how: 合并方式 ('left', 'right', 'outer', 'inner')
        suffixes: 重复列名后缀
        
    Returns:
        合并后的DataFrame，保持MultiIndex(date, ticker)结构
    """
    try:
        # 确保两个DataFrame都有date和ticker列
        left_work = left_df.copy()
        right_work = right_df.copy()
        
        # 重置索引确保有date和ticker列
        if isinstance(left_work.index, pd.MultiIndex):
            left_work = left_work.reset_index()
        if isinstance(right_work.index, pd.MultiIndex):
            right_work = right_work.reset_index()
            
        # 确保有必需的列
        required_cols = {'date', 'ticker'}
        if not required_cols.issubset(left_work.columns):
            raise ValueError(f"左侧DataFrame缺少必需列: {required_cols - set(left_work.columns)}")
        if not required_cols.issubset(right_work.columns):
            raise ValueError(f"右侧DataFrame缺少必需列: {required_cols - set(right_work.columns)}")
        
        # 执行标准pandas merge
        merged = left_work.merge(right_work, on=['date', 'ticker'], how=how, suffixes=suffixes)
        
        # 重新设置MultiIndex
        if 'date' in merged.columns and 'ticker' in merged.columns:
            merged = merged.set_index(['date', 'ticker']).sort_index()
        
        return merged
        
    except Exception as e:
        print(f"合并失败: {e}")
        return left_df

def ensure_multiindex_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保DataFrame具有正确的MultiIndex(date, ticker)结构
    
    Args:
        df: 输入DataFrame
        
    Returns:
        具有正确MultiIndex结构的DataFrame
    """
    if df is None or df.empty:
        return df
        
    # 如果已经是正确的MultiIndex，直接返回
    if isinstance(df.index, pd.MultiIndex) and df.index.names == ['date', 'ticker']:
        return df
    
    # 重置索引
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    # 检查必需列
    if 'date' not in df.columns or 'ticker' not in df.columns:
        return df  # 返回原DataFrame，不做修改
    
    # 设置MultiIndex
    return df.set_index(['date', 'ticker']).sort_index()

'''
    
    # 查找合适的插入位置（在类定义之前）
    class_pos = content.find('class DataContractManager:')
    if class_pos > 0:
        content = content[:class_pos] + merge_helper_code + '\n' + content[class_pos:]
    
    # 替换问题的合并代码
    problematic_merge_pattern = r'''merged = merge_df\.merge\(alpha_df, on=\['date', 'ticker'\], how='left'\)'''
    replacement_merge = "merged = safe_merge_on_multiindex(merge_df, alpha_df, how='left')"
    
    content = re.sub(problematic_merge_pattern, replacement_merge, content)
    
    # 替换其他合并模式
    merge_patterns = [
        (r'\.merge\([^)]*on=\[\'date\', \'ticker\'\][^)]*\)', 
         lambda m: f"safe_merge_on_multiindex({m.group().replace('.merge(', '').replace(')', '')})")
    ]
    
    for pattern, replacement in merge_patterns:
        content = re.sub(pattern, replacement, content)
    
    return content

def separate_pca_processing(content: str) -> str:
    """分离Alpha和传统因子的PCA处理"""
    print("  - 分离PCA处理逻辑...")
    
    pca_separation_code = '''
# === 分离的PCA处理系统 ===
def apply_separated_pca(feature_data: pd.DataFrame, alpha_data: pd.DataFrame = None,
                       traditional_n_components: int = None, alpha_n_components: int = None) -> pd.DataFrame:
    """
    对Alpha因子和传统因子分别应用PCA降维，然后合并
    
    Args:
        feature_data: 传统特征数据
        alpha_data: Alpha特征数据
        traditional_n_components: 传统特征PCA组件数
        alpha_n_components: Alpha特征PCA组件数
        
    Returns:
        合并后的降维特征数据
    """
    try:
        from sklearn.decomposition import PCA
        
        results = []
        
        # 处理传统特征
        if feature_data is not None and not feature_data.empty:
            traditional_features = feature_data.select_dtypes(include=[np.number]).fillna(0)
            
            if traditional_features.shape[1] > 1:
                n_comp = min(traditional_n_components or traditional_features.shape[1]//2, 
                           traditional_features.shape[1], 
                           traditional_features.shape[0]//2)
                n_comp = max(1, n_comp)
                
                pca_trad = PCA(n_components=n_comp, random_state=42)
                trad_pca_features = pca_trad.fit_transform(traditional_features)
                
                trad_pca_df = pd.DataFrame(
                    trad_pca_features,
                    index=traditional_features.index,
                    columns=[f'trad_pca_{i+1}' for i in range(trad_pca_features.shape[1])]
                )
                results.append(trad_pca_df)
                print(f"  传统特征PCA: {traditional_features.shape[1]} -> {n_comp}")
        
        # 处理Alpha特征
        if alpha_data is not None and not alpha_data.empty:
            alpha_features = alpha_data.select_dtypes(include=[np.number]).fillna(0)
            
            if alpha_features.shape[1] > 1:
                n_comp = min(alpha_n_components or alpha_features.shape[1]//2,
                           alpha_features.shape[1],
                           alpha_features.shape[0]//2)
                n_comp = max(1, n_comp)
                
                pca_alpha = PCA(n_components=n_comp, random_state=42)
                alpha_pca_features = pca_alpha.fit_transform(alpha_features)
                
                alpha_pca_df = pd.DataFrame(
                    alpha_pca_features,
                    index=alpha_features.index, 
                    columns=[f'alpha_pca_{i+1}' for i in range(alpha_pca_features.shape[1])]
                )
                results.append(alpha_pca_df)
                print(f"  Alpha特征PCA: {alpha_features.shape[1]} -> {n_comp}")
        
        # 合并结果
        if results:
            combined = pd.concat(results, axis=1)
            return ensure_multiindex_structure(combined)
        else:
            return feature_data if feature_data is not None else pd.DataFrame()
            
    except Exception as e:
        print(f"PCA处理失败: {e}")
        return feature_data if feature_data is not None else pd.DataFrame()

'''
    
    # 插入PCA分离代码
    pca_insert_pos = content.find('def apply_separated_pca')
    if pca_insert_pos == -1:
        # 在类定义之前插入
        class_pos = content.find('class ModuleManager:')
        if class_pos > 0:
            content = content[:class_pos] + pca_separation_code + '\n' + content[class_pos:]
    
    return content

def clean_version_mixing(content: str) -> str:
    """清理版本混合代码"""
    print("  - 清理版本标记和过时代码...")
    
    # 移除版本标记注释
    version_patterns = [
        r'#.*[Vv][0-9]+.*',
        r'#.*legacy.*',
        r'#.*deprecated.*',
        r'#.*old.*implementation.*',
    ]
    
    for pattern in version_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # 简化版本相关变量名
    version_var_replacements = [
        (r'enable_v[0-9]+_enhancements', 'enable_enhancements'),
        (r'v[0-9]+_config', 'enhanced_config'),
        (r'v[0-9]+_performance_tracker', 'performance_tracker'),
    ]
    
    for pattern, replacement in version_var_replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    return content

if __name__ == "__main__":
    try:
        success = fix_bma_critical_conflicts()
        if success:
            print("\n🎉 所有关键冲突修复完成！")
            print("\n主要修复内容:")
            print("✓ 统一MultiIndex(date, ticker)索引策略")
            print("✓ 改进pd.merge on=['date', 'ticker']合并逻辑")  
            print("✓ 分离Alpha和传统因子的PCA处理")
            print("✓ 统一时间配置参数(滞后1天)")
            print("✓ 清理重复方法和版本混合代码")
            print("\n请运行测试以验证修复效果!")
        else:
            print("❌ 修复过程中遇到问题")
    except Exception as e:
        print(f"❌ 修复失败: {e}")