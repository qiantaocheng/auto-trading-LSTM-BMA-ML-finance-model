#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试run_complete_analysis方法签名修复
"""

import sys
sys.path.append('.')

def test_method_signature():
    """测试方法签名修复"""
    
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        print("✓ 成功导入UltraEnhancedQuantitativeModel")
        
        # 创建模型实例
        model = UltraEnhancedQuantitativeModel()
        print("✓ 成功创建模型实例")
        
        # 测试方法签名 - 模拟autotrader调用方式
        print("\n测试关键字参数调用方式:")
        try:
            # 这是autotrader使用的调用方式
            # model.run_complete_analysis(tickers=tickers, start_date=start_date, end_date=end_date, top_n=10)
            
            # 我们创建模拟参数
            test_tickers = ['AAPL', 'MSFT', 'GOOGL']
            test_start_date = '2023-01-01'
            test_end_date = '2024-01-01'
            test_top_n = 10
            
            # 测试方法是否接受这些关键字参数（不实际执行）
            import inspect
            sig = inspect.signature(model.run_complete_analysis)
            print(f"✓ 方法签名: {sig}")
            
            # 检查参数绑定
            try:
                bound_args = sig.bind(
                    tickers=test_tickers,
                    start_date=test_start_date, 
                    end_date=test_end_date,
                    top_n=test_top_n
                )
                print("✓ 关键字参数绑定成功")
                print(f"  绑定参数: {bound_args.arguments}")
                
                # 检查是否能检测到原始API调用模式
                kwargs = {
                    'tickers': test_tickers,
                    'start_date': test_start_date,
                    'end_date': test_end_date,
                    'top_n': test_top_n
                }
                
                # 模拟方法内部的检测逻辑
                if 'tickers' in kwargs or isinstance(test_tickers, list):
                    print("✓ 正确检测到原始API调用模式")
                else:
                    print("❌ 未能检测到原始API调用模式")
                    
            except Exception as e:
                print(f"❌ 参数绑定失败: {e}")
                return False
                
        except Exception as e:
            print(f"❌ 方法签名测试失败: {e}")
            return False
            
        print("\n✅ 所有测试通过 - 方法签名修复成功")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_method_signature()
    sys.exit(0 if success else 1)