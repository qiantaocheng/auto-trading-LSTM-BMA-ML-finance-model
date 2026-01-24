"""
诊断为什么会有多个预测 - 检查数据流中每个环节的重复情况
"""
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bma_models.simple_25_factor_engine import Simple17FactorEngine
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_duplicates(df, name, check_index=True, check_date_ticker=True):
    """检查DataFrame中的重复情况"""
    print(f"\n{'='*80}")
    print(f"检查: {name}")
    print(f"{'='*80}")
    print(f"Shape: {df.shape}")
    
    if check_index and isinstance(df.index, pd.MultiIndex):
        duplicates = df.index.duplicated()
        dup_count = duplicates.sum()
        print(f"索引重复数: {dup_count}")
        
        if dup_count > 0:
            print(f"⚠️ 发现 {dup_count} 个重复索引!")
            dup_indices = df.index[duplicates]
            print(f"重复索引示例 (前10个):")
            for idx in dup_indices[:10]:
                print(f"  {idx}")
            
            # 检查每个日期的重复ticker
            if check_date_ticker:
                date_level = df.index.get_level_values('date')
                ticker_level = df.index.get_level_values('ticker')
                
                for date in sorted(date_level.unique())[-5:]:  # 检查最后5个日期
                    date_mask = date_level == date
                    date_tickers = ticker_level[date_mask]
                    dup_tickers = date_tickers[date_tickers.duplicated()]
                    if len(dup_tickers) > 0:
                        print(f"  ⚠️ Date {date}: {len(dup_tickers)} 个重复ticker: {dup_tickers.unique()[:10].tolist()}")
    
    elif check_index:
        duplicates = df.index.duplicated()
        dup_count = duplicates.sum()
        print(f"索引重复数: {dup_count}")
        if dup_count > 0:
            print(f"⚠️ 发现 {dup_count} 个重复索引!")
    
    return dup_count

def main():
    """主诊断流程"""
    print("="*80)
    print("诊断: 为什么会有多个预测")
    print("="*80)
    
    # 1. 初始化引擎
    print("\n[步骤1] 初始化Simple17FactorEngine...")
    engine = Simple17FactorEngine(lookback_days=120)
    
    # 2. 获取市场数据
    print("\n[步骤2] 获取市场数据...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    market_data = engine.fetch_market_data(symbols, use_optimized_downloader=True)
    
    if market_data.empty:
        print("❌ 无法获取市场数据")
        return
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Market data columns: {list(market_data.columns)}")
    
    # 检查market_data的重复
    check_duplicates(market_data, "market_data (fetch_market_data返回)", check_index=False)
    
    # 检查market_data中是否有重复的(date, ticker)组合
    if 'date' in market_data.columns and 'ticker' in market_data.columns:
        date_ticker_combos = market_data.groupby(['date', 'ticker']).size()
        dup_combos = date_ticker_combos[date_ticker_combos > 1]
        if len(dup_combos) > 0:
            print(f"⚠️ market_data中有 {len(dup_combos)} 个重复的(date, ticker)组合!")
            print(f"重复组合示例 (前10个):")
            for (date, ticker), count in dup_combos.head(10).items():
                print(f"  ({date}, {ticker}): {count} 次")
    
    # 3. 计算因子
    print("\n[步骤3] 计算所有因子...")
    all_feature_data = engine.compute_all_17_factors(market_data, mode='predict')
    
    if all_feature_data.empty:
        print("❌ 无法计算因子")
        return
    
    # 检查all_feature_data的重复
    dup_count = check_duplicates(all_feature_data, "all_feature_data (compute_all_17_factors返回)")
    
    # 4. 模拟Direct Predict的数据提取
    print("\n[步骤4] 模拟Direct Predict的数据提取...")
    if isinstance(all_feature_data.index, pd.MultiIndex):
        pred_date = all_feature_data.index.get_level_values('date').max()
        date_mask = all_feature_data.index.get_level_values('date') <= pred_date
        date_feature_data = all_feature_data[date_mask].copy()
        
        dup_count_date = check_duplicates(date_feature_data, "date_feature_data (提取后)")
        
        # 5. 模拟predict_with_snapshot的数据准备
        print("\n[步骤5] 模拟predict_with_snapshot的数据准备...")
        try:
            # 初始化模型（不需要训练）
            model = UltraEnhancedQuantitativeModel()
            
            # 调用_prepare_standard_data_format
            X, y, dates, tickers = model._prepare_standard_data_format(date_feature_data)
            
            dup_count_X = check_duplicates(X, "X (_prepare_standard_data_format返回)")
            
            # 6. 总结
            print("\n" + "="*80)
            print("诊断总结")
            print("="*80)
            print(f"1. market_data重复: {'需要检查date/ticker列' if 'date' in market_data.columns else 'N/A'}")
            print(f"2. all_feature_data重复: {dup_count} 个")
            print(f"3. date_feature_data重复: {dup_count_date} 个")
            print(f"4. X重复: {dup_count_X} 个")
            
            if dup_count > 0:
                print(f"\n⚠️ 根本原因: all_feature_data (compute_all_17_factors返回) 有 {dup_count} 个重复索引")
                print("   建议: 在compute_all_17_factors返回前添加去重逻辑")
            elif dup_count_date > 0:
                print(f"\n⚠️ 根本原因: date_feature_data (提取后) 有 {dup_count_date} 个重复索引")
                print("   建议: 在提取date_feature_data后立即去重")
            elif dup_count_X > 0:
                print(f"\n⚠️ 根本原因: X (_prepare_standard_data_format返回) 有 {dup_count_X} 个重复索引")
                print("   建议: 检查_prepare_standard_data_format的去重逻辑")
            else:
                print("\n✅ 未发现重复索引，问题可能在其他环节")
                
        except Exception as e:
            print(f"\n❌ 模拟predict_with_snapshot失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ all_feature_data不是MultiIndex格式")

if __name__ == "__main__":
    main()
