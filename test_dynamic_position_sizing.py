#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态仓位管理系统测试
验证反马丁格尔和金字塔加仓策略
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dynamic_position_sizing import (
    DynamicPositionSizing, DynamicSizingConfig, TradeResult,
    PositionSizingMode, TradingState
)

async def test_basic_position_sizing():
    """测试基础仓位计算"""
    print("🔍 测试基础仓位计算...")
    
    config = DynamicSizingConfig(
        base_risk_pct=0.02,  # 2%基础风险
        max_exposure_pct=0.08,  # 8%最大敞口
        win_streak_trigger=3,  # 3连胜触发
        addon_aggressive_factor=0.2  # 20%激进度
    )
    
    dps = DynamicPositionSizing(config)
    
    # 测试初始仓位
    initial_size = dps.calculate_position_size("AAPL", 1.0)
    print(f"初始仓位大小: {initial_size:.2%}")
    assert initial_size == 0.02, f"期望2%，实际{initial_size:.2%}"
    
    # 模拟连续盈利
    for i in range(4):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"pos_{i}",
            entry_price=150.0,
            exit_price=153.0,
            quantity=100,
            pnl=300.0,
            pnl_pct=0.02,
            position_size_pct=0.02
        )
        dps.record_trade_result(trade)
    
    # 检查连胜后的仓位增加
    enhanced_size = dps.calculate_position_size("AAPL", 1.0)
    print(f"连胜后仓位大小: {enhanced_size:.2%}")
    print(f"连胜次数: {dps.win_streak}")
    assert enhanced_size > initial_size, "连胜后仓位应该增加"
    
    print("✅ 基础仓位计算测试通过")

async def test_addon_mechanism():
    """测试加仓机制"""
    print("\n🔍 测试加仓机制...")
    
    config = DynamicSizingConfig(
        base_risk_pct=0.02,
        win_streak_trigger=2,  # 降低门槛便于测试
        addon_size_ratio=0.5,  # 50%加仓比例
        max_addon_levels=3
    )
    
    dps = DynamicPositionSizing(config)
    
    # 建立连胜记录
    for i in range(3):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"winning_pos_{i}",
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=0.033,
            position_size_pct=0.02
        )
        dps.record_trade_result(trade)
    
    # 创建基础持仓
    base_trade = TradeResult(
        timestamp=datetime.now(),
        symbol="AAPL",
        action="OPEN",
        position_id="base_pos",
        entry_price=150.0,
        exit_price=None,
        quantity=100,
        pnl=0.0,
        pnl_pct=0.0,
        position_size_pct=0.02
    )
    dps.record_trade_result(base_trade)
    
    # 测试加仓触发
    current_price = 152.0  # 持仓盈利
    market_data = {
        'volume_ratio': 1.5,  # 满足成交量条件
        'price_change_pct': 0.025  # 满足动量条件
    }
    
    can_add = dps.check_addon_trigger("base_pos", current_price, market_data)
    print(f"可以加仓: {can_add}")
    assert can_add, "应该触发加仓条件"
    
    # 计算加仓大小
    addon_size = dps.calculate_position_size("AAPL", 1.0, True, "base_pos")
    print(f"加仓大小: {addon_size:.2%}")
    assert addon_size > 0, "加仓大小应该大于0"
    assert addon_size == 0.01, f"期望1%加仓，实际{addon_size:.2%}"  # 2% * 50% = 1%
    
    print("✅ 加仓机制测试通过")

async def test_cooldown_mechanism():
    """测试冷静期机制"""
    print("\n🔍 测试冷静期机制...")
    
    config = DynamicSizingConfig(
        base_risk_pct=0.02,
        loss_streak_cooldown=2,  # 2连败触发冷静期
        cooldown_duration_hours=1  # 1小时冷静期
    )
    
    dps = DynamicPositionSizing(config)
    
    # 模拟连续亏损
    for i in range(3):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"losing_pos_{i}",
            entry_price=150.0,
            exit_price=147.0,
            quantity=100,
            pnl=-300.0,
            pnl_pct=-0.02,
            position_size_pct=0.02
        )
        dps.record_trade_result(trade)
    
    print(f"连败次数: {dps.loss_streak}")
    print(f"交易状态: {dps.trading_state}")
    assert dps.trading_state == TradingState.COOLDOWN, "应该进入冷静期"
    
    # 测试冷静期的仓位计算
    cooldown_size = dps.calculate_position_size("AAPL", 1.0)
    print(f"冷静期仓位大小: {cooldown_size:.2%}")
    assert cooldown_size == 0.02, "冷静期应该使用基础仓位"
    
    print("✅ 冷静期机制测试通过")

async def test_position_management_signals():
    """测试仓位管理信号"""
    print("\n🔍 测试仓位管理信号...")
    
    config = DynamicSizingConfig(
        base_risk_pct=0.02,
        win_streak_trigger=2
    )
    
    dps = DynamicPositionSizing(config)
    
    # 建立连胜
    for i in range(3):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"win_pos_{i}",
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=0.033,
            position_size_pct=0.02
        )
        dps.record_trade_result(trade)
    
    # 创建持仓
    dps.record_trade_result(TradeResult(
        timestamp=datetime.now(),
        symbol="AAPL",
        action="OPEN",
        position_id="test_pos",
        entry_price=150.0,
        exit_price=None,
        quantity=100,
        pnl=0.0,
        pnl_pct=0.0,
        position_size_pct=0.02
    ))
    
    # 测试不同价格下的信号
    test_cases = [
        (147.0, "触发止损"),  # 低于止损价 (150 * 0.98 = 147)
        (160.0, "触发止盈"),  # 高于止盈价
        (152.0, "加仓机会")   # 盈利状态
    ]
    
    for price, expected in test_cases:
        signal = dps.get_position_management_signal("test_pos", price)
        print(f"价格 {price}: {signal['action']} - {signal['reason']}")
        
        if "止损" in expected:
            assert signal['action'] == 'CLOSE', f"期望CLOSE，实际{signal['action']}"
        elif "止盈" in expected:
            assert signal['action'] == 'CLOSE', f"期望CLOSE，实际{signal['action']}"
        elif "加仓" in expected:
            assert signal['action'] in ['ADD', 'HOLD'], f"期望ADD或HOLD，实际{signal['action']}"
    
    print("✅ 仓位管理信号测试通过")

async def test_performance_tracking():
    """测试性能跟踪"""
    print("\n🔍 测试性能跟踪...")
    
    dps = DynamicPositionSizing()
    
    # 模拟一系列交易
    trades = [
        (300.0, True),   # 盈利
        (-150.0, False), # 亏损
        (450.0, True),   # 盈利
        (200.0, True),   # 盈利
        (-100.0, False), # 亏损
        (350.0, True),   # 盈利
    ]
    
    for i, (pnl, is_addon) in enumerate(trades):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"perf_pos_{i}",
            entry_price=150.0,
            exit_price=150.0 + pnl/100,
            quantity=100,
            pnl=pnl,
            pnl_pct=pnl/(150.0*100),
            position_size_pct=0.02,
            is_addon=is_addon and i > 2  # 后面的交易可能是加仓
        )
        dps.record_trade_result(trade)
    
    # 获取性能统计
    status = dps.get_system_status()
    stats = status['performance_stats']
    
    print(f"总交易数: {stats['total_trades']}")
    print(f"盈利交易: {stats['winning_trades']}")
    print(f"胜率: {stats['win_rate']:.1%}")
    print(f"总盈亏: ${stats['total_pnl']:.2f}")
    print(f"加仓交易数: {stats['addon_trades']}")
    
    assert stats['total_trades'] == 6, f"期望6笔交易，实际{stats['total_trades']}"
    assert stats['winning_trades'] == 4, f"期望4笔盈利，实际{stats['winning_trades']}"
    assert abs(stats['win_rate'] - 4/6) < 0.01, f"胜率计算错误"
    
    print("✅ 性能跟踪测试通过")

async def test_integration_with_factor_balanced_system():
    """测试与因子平衡系统的集成"""
    print("\n🔍 测试与因子平衡系统集成...")
    
    try:
        from factor_balanced_trading_system import FactorBalancedTradingSystem, SystemConfig
        
        # 创建启用动态仓位的系统配置
        config = SystemConfig(
            enable_dynamic_sizing=True,
            dynamic_base_risk=0.025,
            dynamic_win_streak_trigger=2
        )
        
        system = FactorBalancedTradingSystem(config)
        await system.initialize_system()
        
        # 测试动态仓位大小获取
        position_size = system.get_dynamic_position_size("AAPL", 1.0)
        print(f"系统动态仓位大小: {position_size:.2%}")
        
        # 测试交易记录
        system.record_trade_execution(
            symbol="AAPL",
            action="OPEN",
            quantity=100,
            price=150.0,
            position_id="integration_test",
            position_size_pct=0.025
        )
        
        # 测试加仓机会检查
        addon_signal = system.check_addon_opportunity("integration_test", 152.0)
        print(f"加仓信号: {addon_signal}")
        
        # 获取系统摘要，验证动态仓位状态
        summary = system.get_system_summary()
        if 'dynamic_sizing' in summary:
            print(f"动态仓位系统状态: {summary['dynamic_sizing']}")
            assert summary['dynamic_sizing']['enabled'], "动态仓位应该启用"
        
        await system.stop_system()
        print("✅ 因子平衡系统集成测试通过")
        
    except ImportError:
        print("⚠️ 因子平衡系统不可用，跳过集成测试")

async def main():
    """主测试函数"""
    print("🚀 动态仓位管理系统测试")
    print("=" * 60)
    
    test_functions = [
        test_basic_position_sizing,
        test_addon_mechanism,
        test_cooldown_mechanism,
        test_position_management_signals,
        test_performance_tracking,
        test_integration_with_factor_balanced_system
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！动态仓位管理系统运行正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)