#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR API连接测试脚本
测试IBKR API的基本连接功能
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IBKR相关导入
try:
    from ib_insync import *
    import ib_insync as ibs
    IBKR_AVAILABLE = True
    print("✅ ib_insync已成功导入")
except ImportError as e:
    print(f"❌ ib_insync导入失败: {e}")
    IBKR_AVAILABLE = False

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    IBAPI_AVAILABLE = True
    print("✅ ibapi已成功导入")
except ImportError as e:
    print(f"❌ ibapi导入失败: {e}")
    IBAPI_AVAILABLE = False

class IBKRConnectionTest:
    """IBKR连接测试类"""
    
    def __init__(self):
        self.ib = None
        self.connected = False
        self.test_results = []
        
    async def test_basic_connection(self):
        """测试基本连接"""
        print("\n🔌 开始测试IBKR基本连接...")
        
        try:
            # 创建IB连接
            self.ib = IB()
            
            # 设置事件处理
            self.ib.errorEvent += self.on_error
            self.ib.connectedEvent += self.on_connected
            self.ib.disconnectedEvent += self.on_disconnected
            
            # 连接参数
            host = '127.0.0.1'
            port = 4002  # TWS Paper Trading
            client_id = 50310           
            print(f"📡 尝试连接到 {host}:{port} (Client ID: {client_id})")
            
            # 尝试连接
            await self.ib.connectAsync(host, port, client_id)
            
            if self.ib.isConnected():
                self.connected = True
                print("✅ 基本连接成功!")
                self.test_results.append(("基本连接", "✅ 成功"))
                
                # 测试获取账户信息
                await self.test_account_info()
                
                # 测试获取服务器时间
                await self.test_server_time()
                
                # 测试获取合约详情
                await self.test_contract_details()
                
                # 断开连接
                await self.ib.disconnectAsync()
                print("🔌 连接已断开")
                
            else:
                print("❌ 基本连接失败")
                self.test_results.append(("基本连接", "❌ 失败"))
                
        except Exception as e:
            print(f"❌ 连接测试异常: {e}")
            self.test_results.append(("基本连接", f"❌ 异常: {e}"))
    
    async def test_account_info(self):
        """测试账户信息获取"""
        print("\n💰 测试账户信息获取...")
        
        try:
            # 请求账户信息
            accounts = self.ib.managedAccounts()
            if accounts:
                print(f"✅ 获取到账户: {accounts}")
                self.test_results.append(("账户信息", f"✅ 成功 - 账户: {accounts}"))
            else:
                print("⚠️ 未获取到账户信息")
                self.test_results.append(("账户信息", "⚠️ 未获取到账户"))
                
        except Exception as e:
            print(f"❌ 账户信息测试失败: {e}")
            self.test_results.append(("账户信息", f"❌ 失败: {e}"))
    
    async def test_server_time(self):
        """测试服务器时间获取"""
        print("\n⏰ 测试服务器时间获取...")
        
        try:
            # 请求服务器时间
            server_time = self.ib.reqCurrentTime()
            if server_time:
                print(f"✅ 服务器时间: {server_time}")
                self.test_results.append(("服务器时间", f"✅ 成功 - {server_time}"))
            else:
                print("⚠️ 未获取到服务器时间")
                self.test_results.append(("服务器时间", "⚠️ 未获取到时间"))
                
        except Exception as e:
            print(f"❌ 服务器时间测试失败: {e}")
            self.test_results.append(("服务器时间", f"❌ 失败: {e}"))
    
    async def test_contract_details(self):
        """测试合约详情获取"""
        print("\n📋 测试合约详情获取...")
        
        try:
            # 创建AAPL合约
            contract = Stock('AAPL', 'SMART', 'USD')
            
            # 请求合约详情
            details = self.ib.reqContractDetails(contract)
            
            if details:
                print(f"✅ 获取到合约详情: {len(details)} 个结果")
                for i, detail in enumerate(details[:2]):  # 只显示前2个
                    print(f"  详情 {i+1}: {detail.contract.symbol} - {detail.contract.exchange}")
                self.test_results.append(("合约详情", f"✅ 成功 - {len(details)} 个结果"))
            else:
                print("⚠️ 未获取到合约详情")
                self.test_results.append(("合约详情", "⚠️ 未获取到详情"))
                
        except Exception as e:
            print(f"❌ 合约详情测试失败: {e}")
            self.test_results.append(("合约详情", f"❌ 失败: {e}"))
    
    def on_connected(self):
        """连接成功回调"""
        print("📡 连接事件触发")
    
    def on_disconnected(self):
        """断开连接回调"""
        print("📡 断开连接事件触发")
    
    def on_error(self, reqId, errorCode, errorString, contract):
        """错误回调"""
        print(f"❌ IBKR错误: ID={reqId}, 代码={errorCode}, 消息={errorString}")
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "="*50)
        print("📊 IBKR API连接测试总结")
        print("="*50)
        
        for test_name, result in self.test_results:
            print(f"{test_name:<15}: {result}")
        
        print("="*50)
        
        # 统计结果
        success_count = sum(1 for _, result in self.test_results if "✅" in result)
        total_count = len(self.test_results)
        
        print(f"总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        if success_count == total_count:
            print("🎉 所有测试通过！IBKR API连接正常")
        elif success_count > 0:
            print("⚠️ 部分测试通过，请检查失败的测试项")
        else:
            print("❌ 所有测试失败，请检查IBKR连接配置")

async def main():
    """主函数"""
    print("🚀 开始IBKR API连接测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not IBKR_AVAILABLE:
        print("❌ IBKR API不可用，无法进行测试")
        return
    
    # 创建测试实例
    tester = IBKRConnectionTest()
    
    # 执行测试
    await tester.test_basic_connection()
    
    # 打印测试总结
    tester.print_test_summary()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main()) 