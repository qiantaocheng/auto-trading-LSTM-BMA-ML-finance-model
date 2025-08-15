
# 内存优化版BMA运行器
def run_memory_optimized_bma():
    """内存优化版BMA回测"""
    import gc
    from bma_walkforward_enhanced import EnhancedBMAWalkForward, ENHANCED_STOCK_POOL
    
    # 1. 限制股票数量
    limited_stocks = ENHANCED_STOCK_POOL[:15]  # 只用15只股票
    
    # 2. 创建优化配置的回测器
    backtest = EnhancedBMAWalkForward(
        initial_capital=100000,    # 减少初始资金
        max_positions=10,          # 减少最大持仓
        training_window_months=3,  # 减少训练窗口
        min_training_samples=60    # 减少最小样本
    )
    
    print(f"内存优化设置:")
    print(f"- 股票数量: {len(limited_stocks)}")
    print(f"- 最大持仓: 10")
    print(f"- 训练窗口: 3个月")
    
    try:
        # 3. 运行回测
        results = backtest.run_enhanced_walkforward_backtest(
            tickers=limited_stocks,
            start_date="2023-01-01",  # 缩短时间范围
            end_date="2024-06-01"
        )
        
        print("回测完成!")
        if results and 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"总收益率: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"Sharpe比率: {metrics.get('sharpe_ratio', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"回测失败: {e}")
        return None
    finally:
        # 4. 强制清理内存
        gc.collect()

if __name__ == "__main__":
    run_memory_optimized_bma()
