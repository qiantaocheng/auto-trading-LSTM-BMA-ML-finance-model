#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳健特征选择系统
基于滚动IC分析，保留少量高质量、互异的特征(K≈12-20)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

class RobustFeatureSelector:
    """
    稳健特征选择器
    
    通过滚动IC分析保留高质量、低相关性的特征
    """
    
    def __init__(self, 
                 target_features: int = 16,  # 目标特征数量
                 ic_window: int = 126,       # IC计算窗口(约6个月)
                 min_ic_mean: float = 0.01,  # 最小IC均值
                 min_ic_ir: float = 0.3,     # 最小IC信息比率
                 max_correlation: float = 0.6  # 最大特征间相关性
                 ):
        self.target_features = target_features
        self.ic_window = ic_window
        self.min_ic_mean = min_ic_mean
        self.min_ic_ir = min_ic_ir
        self.max_correlation = max_correlation
        
        # 保存选择的特征
        self.selected_features_ = None
        self.feature_stats_ = None
        self.correlation_matrix_ = None
        self.feature_clusters_ = None
        
    def rolling_ic(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> Dict[str, Tuple[float, float]]:
        """
        计算滚动IC统计
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            
        Returns:
            Dict[特征名, (IC均值, IC标准差)]
        """
        print(f"计算滚动IC统计，窗口: {self.ic_window}天")
        
        out = {}
        
        for col in X.columns:
            try:
                # 构建分析数据框
                df = pd.DataFrame({
                    'x': X[col], 
                    'y': y, 
                    'd': dates
                }).dropna()
                
                if len(df) < self.ic_window:
                    print(f"⚠️ 特征 {col}: 数据不足({len(df)} < {self.ic_window})")
                    continue
                
                # 直接计算滚动相关性（简化版）
                # 对于时间序列数据，直接计算滚动相关性
                df_sorted = df.sort_values('d')
                
                if len(df_sorted) < self.ic_window:
                    print(f"⚠️ 特征 {col}: 数据不足({len(df_sorted)} < {self.ic_window})")
                    continue
                
                # 计算滚动Spearman相关性
                rolling_ic_list = []
                for i in range(self.ic_window, len(df_sorted) + 1):
                    window_data = df_sorted.iloc[i-self.ic_window:i]
                    if len(window_data) >= 10:  # 至少需要10个点计算相关性
                        try:
                            ic = spearmanr(window_data['x'], window_data['y'])[0]
                            if not np.isnan(ic):
                                rolling_ic_list.append(ic)
                        except:
                            continue
                
                if len(rolling_ic_list) == 0:
                    print(f"⚠️ 特征 {col}: 无法计算有效IC")
                    continue
                
                daily_ic = pd.Series(rolling_ic_list)
                
                if len(daily_ic) < self.ic_window // 2:
                    print(f"⚠️ 特征 {col}: 有效IC天数不足({len(daily_ic)})")
                    continue
                
                # 计算IC统计
                ic_mean = daily_ic.mean()
                ic_std = daily_ic.std()
                
                out[col] = (ic_mean, ic_std)
                
                # 详细统计
                ic_ir = ic_mean / (ic_std + 1e-8)
                print(f"✓ 特征 {col}: IC均值={ic_mean:.4f}, IC标准差={ic_std:.4f}, IC_IR={ic_ir:.4f}")
                
            except Exception as e:
                print(f"❌ 特征 {col} IC计算失败: {e}")
                continue
        
        print(f"滚动IC计算完成，有效特征: {len(out)}/{len(X.columns)}")
        return out
    
    def filter_by_ic_quality(self, ic_stats: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        根据IC质量过滤特征
        
        Args:
            ic_stats: IC统计结果
            
        Returns:
            候选特征列表
        """
        candidates = []
        
        for feature, (ic_mean, ic_std) in ic_stats.items():
            # 计算IC信息比率
            ic_ir = ic_mean / (ic_std + 1e-8) if ic_std > 1e-6 else 0
            
            # 过滤条件
            if ic_mean > self.min_ic_mean and ic_ir > self.min_ic_ir:
                candidates.append(feature)
                print(f"✓ 候选特征 {feature}: IC={ic_mean:.4f}, IC_IR={ic_ir:.4f}")
            else:
                print(f"✗ 过滤特征 {feature}: IC={ic_mean:.4f}, IC_IR={ic_ir:.4f} (不符合标准)")
        
        print(f"IC质量过滤: {len(candidates)}/{len(ic_stats)} 特征通过")
        return candidates
    
    def remove_redundant_features(self, X: pd.DataFrame, candidates: List[str], 
                                ic_stats: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        去除冗余特征：按相关性聚类，每簇选择IC最高的特征
        
        Args:
            X: 特征矩阵
            candidates: 候选特征列表
            ic_stats: IC统计结果
            
        Returns:
            最终选择的特征列表
        """
        if len(candidates) <= self.target_features:
            print(f"候选特征数({len(candidates)})已少于目标数({self.target_features})，直接返回")
            return candidates
        
        print(f"开始去冗余，候选特征: {len(candidates)}")
        
        # 计算候选特征的相关性矩阵
        X_candidates = X[candidates].fillna(0)
        corr_matrix = X_candidates.corr().abs()
        self.correlation_matrix_ = corr_matrix
        
        # 使用层次聚类基于相关性聚类
        # 距离 = 1 - |correlation|
        distance_matrix = 1 - corr_matrix
        
        # 进行层次聚类
        linkage_matrix = linkage(distance_matrix.values, method='average')
        
        # 动态确定聚类数，确保每簇的平均相关性不超过阈值
        best_clusters = self.target_features
        for n_clusters in range(self.target_features, len(candidates)):
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # 检查簇内最大相关性
            max_intra_corr = 0
            for cluster_id in np.unique(cluster_labels):
                cluster_features = [candidates[i] for i in range(len(candidates)) if cluster_labels[i] == cluster_id]
                if len(cluster_features) > 1:
                    cluster_corr = corr_matrix.loc[cluster_features, cluster_features]
                    max_corr_in_cluster = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].max()
                    max_intra_corr = max(max_intra_corr, max_corr_in_cluster)
            
            if max_intra_corr <= self.max_correlation:
                best_clusters = n_clusters
                break
        
        # 使用最佳聚类数进行聚类
        cluster_labels = fcluster(linkage_matrix, best_clusters, criterion='maxclust')
        
        print(f"层次聚类完成，聚类数: {best_clusters}")
        
        # 每个簇选择IC最高的特征
        selected_features = []
        cluster_info = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_features = [candidates[i] for i in range(len(candidates)) if cluster_labels[i] == cluster_id]
            
            # 选择该簇中IC均值最高的特征
            best_feature = max(cluster_features, key=lambda f: ic_stats[f][0])
            selected_features.append(best_feature)
            
            # 保存簇信息
            cluster_info[cluster_id] = {
                'features': cluster_features,
                'selected': best_feature,
                'ic_stats': {f: ic_stats[f] for f in cluster_features}
            }
            
            print(f"簇 {cluster_id} (共{len(cluster_features)}个特征): 选择 {best_feature} (IC={ic_stats[best_feature][0]:.4f})")
        
        self.feature_clusters_ = cluster_info
        
        print(f"去冗余完成: {len(candidates)} -> {len(selected_features)} 特征")
        return selected_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> 'RobustFeatureSelector':
        """
        拟合特征选择器
        
        Args:
            X: 特征矩阵
            y: 目标变量  
            dates: 日期序列
            
        Returns:
            self
        """
        print("=" * 60)
        print("稳健特征选择系统")
        print("=" * 60)
        print(f"输入: {X.shape[1]} 个特征, {len(X)} 个样本")
        print(f"目标: 选择 {self.target_features} 个稳健特征")
        print(f"标准: IC>{self.min_ic_mean}, IC_IR>{self.min_ic_ir}, 相关性<{self.max_correlation}")
        
        # 1. 计算滚动IC统计
        ic_stats = self.rolling_ic(X, y, dates)
        self.feature_stats_ = ic_stats
        
        if len(ic_stats) == 0:
            raise ValueError("没有特征通过IC计算")
        
        # 2. 根据IC质量过滤特征
        candidates = self.filter_by_ic_quality(ic_stats)
        
        if len(candidates) == 0:
            raise ValueError("没有特征通过IC质量过滤")
        
        # 3. 去除冗余特征
        selected = self.remove_redundant_features(X, candidates, ic_stats)
        
        self.selected_features_ = selected
        
        print("\n" + "=" * 60)
        print(f"特征选择完成: {X.shape[1]} -> {len(selected)} 特征")
        print(f"最终特征: {selected}")
        print("=" * 60)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换特征矩阵，只保留选择的特征
        
        Args:
            X: 输入特征矩阵
            
        Returns:
            转换后的特征矩阵
        """
        if self.selected_features_ is None:
            raise ValueError("特征选择器未拟合")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """
        拟合并转换
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            
        Returns:
            转换后的特征矩阵
        """
        return self.fit(X, y, dates).transform(X)
    
    def get_feature_report(self) -> pd.DataFrame:
        """
        获取特征选择报告
        
        Returns:
            特征报告DataFrame
        """
        if self.feature_stats_ is None:
            raise ValueError("特征选择器未拟合")
        
        report_data = []
        
        for feature, (ic_mean, ic_std) in self.feature_stats_.items():
            ic_ir = ic_mean / (ic_std + 1e-8)
            selected = feature in (self.selected_features_ or [])
            
            # 找到特征所属的簇
            cluster_id = None
            if self.feature_clusters_:
                for cid, cinfo in self.feature_clusters_.items():
                    if feature in cinfo['features']:
                        cluster_id = cid
                        break
            
            report_data.append({
                'feature': feature,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'selected': selected,
                'cluster': cluster_id
            })
        
        report_df = pd.DataFrame(report_data)
        return report_df.sort_values('ic_ir', ascending=False)


def test_robust_feature_selection():
    """测试稳健特征选择"""
    
    # 创建模拟数据
    # np.random.seed removed
    n_samples = 1000
    n_features = 50
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 创建基础信号
    base_signal = np.cumsum(np.zeros(n_samples) * 0.01)  # 模拟价格走势
    
    # 创建一些有预测能力的特征
    X = pd.DataFrame()
    
    # 高质量特征 (与未来收益正相关)
    for i in range(10):
        # 创建与未来信号相关的特征
        feature = np.roll(base_signal, -5) + np.zeros(n_samples) * 0.5  # 前瞻5天的信号加噪声
        X[f'good_feature_{i}'] = feature
    
    # 中等质量特征 (弱相关)
    for i in range(15):
        feature = np.roll(base_signal, -2) + np.zeros(n_samples) * 1.0  # 前瞻2天但噪声更大
        X[f'medium_feature_{i}'] = feature
    
    # 低质量特征 (无相关性)
    for i in range(25):
        X[f'bad_feature_{i}'] = np.zeros(n_samples)
    
    # 创建目标变量（未来收益）
    future_return = np.diff(base_signal, prepend=base_signal[0])  # 收益率
    y = pd.Series(future_return + np.zeros(n_samples) * 0.2)  # 加入噪声
    
    print("测试数据创建完成")
    print(f"特征数: {X.shape[1]}, 样本数: {len(X)}")
    
    # 运行特征选择
    selector = RobustFeatureSelector(
        target_features=15,
        ic_window=60,  # 2个月窗口
        min_ic_mean=0.005,
        min_ic_ir=0.2,
        max_correlation=0.7
    )
    
    X_selected = selector.fit_transform(X, y, dates)
    
    print(f"\n最终选择的特征: {list(X_selected.columns)}")
    
    # 生成报告
    report = selector.get_feature_report()
    
    print("\n特征选择报告 (Top 20):")
    print(report.head(20).to_string(index=False))
    
    print(f"\n选择的特征统计:")
    selected_report = report[report['selected']]
    print(f"平均IC: {selected_report['ic_mean'].mean():.4f}")
    print(f"平均IC_IR: {selected_report['ic_ir'].mean():.4f}")
    print(f"IC范围: {selected_report['ic_mean'].min():.4f} - {selected_report['ic_mean'].max():.4f}")


if __name__ == "__main__":
    test_robust_feature_selection()
