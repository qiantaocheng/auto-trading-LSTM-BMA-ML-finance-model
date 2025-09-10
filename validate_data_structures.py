#!/usr/bin/env python3
"""
数据结构验证脚本 - 检测BMA模型中的所有数据结构问题
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import traceback
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataStructureValidator:
    """数据结构验证器 - 检测所有潜在的数据结构问题"""
    
    def __init__(self):
        self.issues = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        self.validation_results = {}
        
    def validate_multiindex(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """验证MultiIndex结构"""
        logger.info(f"\n{'='*60}")
        logger.info(f"验证 {name} 的MultiIndex结构")
        
        passed = True
        
        # 检查是否为MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            self.issues['critical'].append(f"{name}: 不是MultiIndex格式")
            logger.error(f"❌ {name} 不是MultiIndex格式")
            passed = False
        else:
            logger.info(f"✅ {name} 是MultiIndex格式")
            
            # 检查MultiIndex的levels
            if df.index.nlevels != 2:
                self.issues['high'].append(f"{name}: MultiIndex应该有2个levels，实际有{df.index.nlevels}个")
                passed = False
            
            # 检查level names
            expected_names = ['date', 'ticker']
            actual_names = list(df.index.names)
            if actual_names != expected_names:
                self.issues['medium'].append(f"{name}: MultiIndex names应该是{expected_names}，实际是{actual_names}")
                logger.warning(f"⚠️ MultiIndex names不标准: {actual_names}")
        
        # 检查重复索引
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            self.issues['critical'].append(f"{name}: 存在{dup_count}个重复索引")
            logger.error(f"❌ 发现{dup_count}个重复索引")
            passed = False
        
        # 检查索引排序
        if isinstance(df.index, pd.MultiIndex):
            if not df.index.is_monotonic_increasing:
                self.issues['medium'].append(f"{name}: 索引未排序")
                logger.warning(f"⚠️ 索引未按升序排序")
        
        return passed
    
    def validate_data_types(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """验证数据类型一致性"""
        logger.info(f"\n验证 {name} 的数据类型")
        
        passed = True
        
        # 检查object类型列
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            self.issues['high'].append(f"{name}: 包含object类型列: {object_cols}")
            logger.warning(f"⚠️ 发现object类型列: {object_cols}")
            passed = False
        
        # 检查混合类型
        for col in df.columns:
            if df[col].dtype == 'object':
                continue
            # 检查是否有混合类型
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                self.issues['medium'].append(f"{name}.{col}: 包含混合数据类型")
                passed = False
        
        # 检查数值列的dtype一致性
        numeric_dtypes = df.select_dtypes(include=[np.number]).dtypes
        if len(numeric_dtypes.unique()) > 1:
            self.issues['low'].append(f"{name}: 数值列使用了多种dtype: {numeric_dtypes.unique()}")
            logger.info(f"ℹ️ 数值列dtype不统一: {numeric_dtypes.unique()}")
        
        return passed
    
    def validate_nan_handling(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """验证NaN处理"""
        logger.info(f"\n验证 {name} 的NaN处理")
        
        passed = True
        
        # 检查NaN数量
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            nan_percentage = (nan_count / (df.shape[0] * df.shape[1])) * 100
            
            if nan_percentage > 50:
                self.issues['critical'].append(f"{name}: NaN比例过高 ({nan_percentage:.2f}%)")
                logger.error(f"❌ NaN比例过高: {nan_percentage:.2f}%")
                passed = False
            elif nan_percentage > 10:
                self.issues['high'].append(f"{name}: 包含较多NaN ({nan_percentage:.2f}%)")
                logger.warning(f"⚠️ 包含{nan_count}个NaN ({nan_percentage:.2f}%)")
            else:
                self.issues['low'].append(f"{name}: 包含{nan_count}个NaN ({nan_percentage:.2f}%)")
                logger.info(f"ℹ️ 包含{nan_count}个NaN ({nan_percentage:.2f}%)")
        
        # 检查inf值
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.issues['high'].append(f"{name}: 包含{inf_count}个inf值")
            logger.error(f"❌ 发现{inf_count}个inf值")
            passed = False
        
        return passed
    
    def validate_temporal_consistency(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """验证时间序列一致性"""
        logger.info(f"\n验证 {name} 的时间序列一致性")
        
        passed = True
        
        if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
            dates = df.index.get_level_values('date')
            
            # 检查日期类型
            if not isinstance(dates[0], (pd.Timestamp, datetime)):
                self.issues['high'].append(f"{name}: 日期不是datetime类型")
                logger.error(f"❌ 日期不是datetime类型: {type(dates[0])}")
                passed = False
            
            # 检查日期范围
            date_range = dates.max() - dates.min()
            if date_range.days < 30:
                self.issues['medium'].append(f"{name}: 日期范围过短 ({date_range.days}天)")
                logger.warning(f"⚠️ 日期范围仅{date_range.days}天")
            
            # 检查日期连续性
            unique_dates = sorted(dates.unique())
            expected_dates = pd.date_range(unique_dates[0], unique_dates[-1], freq='D')
            missing_dates = set(expected_dates) - set(unique_dates)
            if len(missing_dates) > len(unique_dates) * 0.3:  # 缺失超过30%
                self.issues['medium'].append(f"{name}: 缺失{len(missing_dates)}个日期")
                logger.warning(f"⚠️ 时间序列不连续，缺失{len(missing_dates)}个日期")
        
        return passed
    
    def validate_cross_sectional_consistency(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """验证横截面一致性"""
        logger.info(f"\n验证 {name} 的横截面一致性")
        
        passed = True
        
        if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names and 'ticker' in df.index.names:
            # 检查每个日期的股票数量
            ticker_counts = df.groupby(level='date').size()
            
            min_tickers = ticker_counts.min()
            max_tickers = ticker_counts.max()
            
            if min_tickers < 2:
                self.issues['high'].append(f"{name}: 某些日期股票数量过少 (最少{min_tickers}只)")
                logger.error(f"❌ 某些日期仅有{min_tickers}只股票")
                passed = False
            
            # 检查股票数量变化
            ticker_std = ticker_counts.std()
            if ticker_std > ticker_counts.mean() * 0.5:  # 标准差超过均值的50%
                self.issues['medium'].append(f"{name}: 横截面股票数量变化过大 (std={ticker_std:.2f})")
                logger.warning(f"⚠️ 横截面股票数量不稳定: {min_tickers} ~ {max_tickers}")
        
        return passed
    
    def validate_dimension_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """验证训练集和测试集维度一致性"""
        logger.info(f"\n{'='*60}")
        logger.info("验证训练集和测试集维度一致性")
        
        passed = True
        
        # 检查列数
        if train_df.shape[1] != test_df.shape[1]:
            self.issues['critical'].append(f"维度不匹配: 训练集{train_df.shape[1]}列, 测试集{test_df.shape[1]}列")
            logger.error(f"❌ 列数不匹配: {train_df.shape[1]} vs {test_df.shape[1]}")
            passed = False
        
        # 检查列名
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        if missing_in_test:
            self.issues['critical'].append(f"测试集缺失列: {missing_in_test}")
            logger.error(f"❌ 测试集缺失列: {missing_in_test}")
            passed = False
        
        if extra_in_test:
            self.issues['high'].append(f"测试集多余列: {extra_in_test}")
            logger.warning(f"⚠️ 测试集多余列: {extra_in_test}")
        
        # 检查列顺序
        if list(train_df.columns) != list(test_df.columns):
            self.issues['medium'].append("列顺序不一致")
            logger.warning("⚠️ 列顺序不一致")
        
        return passed
    
    def validate_cv_temporal_safety(self, cv_gap: int, embargo: int, horizon: int) -> bool:
        """验证CV时间安全性"""
        logger.info(f"\n{'='*60}")
        logger.info("验证CV时间安全参数")
        
        passed = True
        
        # 检查gap是否足够
        if cv_gap < horizon - 1:
            self.issues['critical'].append(f"CV gap不足: {cv_gap} < {horizon-1}")
            logger.error(f"❌ CV gap不足以防止数据泄漏: gap={cv_gap}, 需要>={horizon-1}")
            passed = False
        
        # 检查embargo是否合理
        if embargo < horizon:
            self.issues['high'].append(f"CV embargo不足: {embargo} < {horizon}")
            logger.warning(f"⚠️ CV embargo可能不足: embargo={embargo}, horizon={horizon}")
        
        # 检查总隔离期
        total_isolation = cv_gap + embargo
        if total_isolation < horizon * 1.5:
            self.issues['medium'].append(f"总隔离期偏短: {total_isolation} < {horizon * 1.5}")
            logger.info(f"ℹ️ 总隔离期={total_isolation}天")
        
        logger.info(f"时间配置: gap={cv_gap}, embargo={embargo}, horizon={horizon}")
        
        return passed
    
    def run_comprehensive_validation(self, feature_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """运行综合验证"""
        logger.info("="*80)
        logger.info("开始数据结构综合验证")
        logger.info("="*80)
        
        # 如果没有提供数据，创建模拟数据
        if feature_data is None:
            feature_data = self.create_test_data()
        
        # 1. 验证MultiIndex
        self.validate_multiindex(feature_data, "feature_data")
        
        # 2. 验证数据类型
        self.validate_data_types(feature_data, "feature_data")
        
        # 3. 验证NaN处理
        self.validate_nan_handling(feature_data, "feature_data")
        
        # 4. 验证时间序列一致性
        self.validate_temporal_consistency(feature_data, "feature_data")
        
        # 5. 验证横截面一致性
        self.validate_cross_sectional_consistency(feature_data, "feature_data")
        
        # 6. 模拟训练/测试分割并验证
        if len(feature_data) > 100:
            split_point = int(len(feature_data) * 0.8)
            train_data = feature_data.iloc[:split_point]
            test_data = feature_data.iloc[split_point:]
            self.validate_dimension_consistency(train_data, test_data)
        
        # 7. 验证CV时间安全性
        self.validate_cv_temporal_safety(cv_gap=9, embargo=10, horizon=10)
        
        # 生成报告
        return self.generate_report()
    
    def create_test_data(self) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'feature1': np.random.randn(len(index)),
            'feature2': np.random.randn(len(index)),
            'feature3': np.random.randn(len(index)),
            'target': np.random.randn(len(index))
        }, index=index)
        
        # 添加一些问题
        data.iloc[10:20, 0] = np.nan  # 添加NaN
        data.iloc[50, 1] = np.inf     # 添加inf
        
        return data
    
    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        logger.info("\n" + "="*80)
        logger.info("数据结构验证报告")
        logger.info("="*80)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        logger.info(f"\n发现问题总数: {total_issues}")
        logger.info(f"  - 关键问题: {len(self.issues['critical'])}")
        logger.info(f"  - 高风险问题: {len(self.issues['high'])}")
        logger.info(f"  - 中风险问题: {len(self.issues['medium'])}")
        logger.info(f"  - 低风险问题: {len(self.issues['low'])}")
        
        if self.issues['critical']:
            logger.error("\n🔴 关键问题（必须立即修复）:")
            for issue in self.issues['critical']:
                logger.error(f"  • {issue}")
        
        if self.issues['high']:
            logger.warning("\n🟠 高风险问题:")
            for issue in self.issues['high']:
                logger.warning(f"  • {issue}")
        
        if self.issues['medium']:
            logger.info("\n🟡 中风险问题:")
            for issue in self.issues['medium']:
                logger.info(f"  • {issue}")
        
        if self.issues['low']:
            logger.info("\n🟢 低风险问题:")
            for issue in self.issues['low']:
                logger.info(f"  • {issue}")
        
        # 生成修复建议
        if self.issues['critical'] or self.issues['high']:
            logger.info("\n" + "="*60)
            logger.info("修复建议:")
            logger.info("1. 优先修复所有关键问题")
            logger.info("2. 统一使用MultiIndex(date, ticker)格式")
            logger.info("3. 确保CV参数满足: gap >= horizon-1")
            logger.info("4. 实施严格的NaN处理策略")
            logger.info("5. 添加数据验证断言")
        
        return {
            'total_issues': total_issues,
            'issues_by_severity': self.issues,
            'validation_timestamp': datetime.now().isoformat(),
            'passed': len(self.issues['critical']) == 0
        }


def main():
    """主函数"""
    validator = DataStructureValidator()
    
    # 尝试加载真实数据
    try:
        from bma_models.polygon_client import PolygonDataProvider
        provider = PolygonDataProvider()
        
        # 获取一些测试数据
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        feature_data = provider.get_training_data(
            tickers=tickers,
            start_date='2024-01-01',
            end_date='2024-03-01'
        )
        
        logger.info("使用真实数据进行验证")
        
    except Exception as e:
        logger.warning(f"无法加载真实数据: {e}")
        logger.info("使用模拟数据进行验证")
        feature_data = None
    
    # 运行验证
    report = validator.run_comprehensive_validation(feature_data)
    
    # 保存报告
    import json
    with open('data_structure_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n验证报告已保存到: data_structure_validation_report.json")
    
    # 返回是否通过
    return report['passed']


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)