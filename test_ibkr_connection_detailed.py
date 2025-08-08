#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR API详细连接测试脚本
测试不同端口和连接配置
"""

import asyncio
import sys
import os
import socket
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

class DetailedIBKRConnectionTest:
    """详细IBKR连接测试类"""
    
    def __init__(self):
        self.test_results = []
        
        # 不同的连接配置
        self.connection_configs = [
            {
                'name': 'TWS Paper Trading',
                'host': '127.0.0.1',
                'port': 4002,
                'description': 'TWS模拟交易端口'
            },
            {
                'name': 'TWS Live Trading',
                'host': '127.0.0.1',
                'port': 7497,
                'description': 'TWS实盘交易端口'
            },
            {
                'name': 'IB Gateway Paper',
                'host': '127.0.0.1',
                'port': 4001,
                'description': 'IB Gateway模拟交易端口'
            },
            {
                'name': 'IB Gateway Live',
                'host': '127.0.0.1',
                'port': 7496,
                'description': 'IB Gateway实盘交易端口'
            }
        ]
    
    def test_port_connectivity(self, host, port):
        """测试端口连通性"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"端口测试异常: {e}")
            return False
    
    async def test_single_connection(self, config):
        """测试单个连接配置"""
        print(f"\n🔌 测试 {config['name']} ({config['description']})")
        print(f"   地址: {config['host']}:{config['port']}")
        
        # 首先测试端口连通性
        port_open = self.test_port_connectivity(config['host'], config['port'])
        if not port_open:
            print(f"❌ 端口 {config['port']} 不可达")
            self.test_results.append((config['name'], "❌ 端口不可达"))
            return False
        
        print(f"✅ 端口 {config['port']} 可达")
        
        # 尝试IBKR连接
        try:
            ib = IB()
            
            # 设置事件处理
            ib.errorEvent += lambda reqId, errorCode, errorString, contract: self.on_error(reqId, errorCode, errorString, contract)
            
            # 尝试连接
            await ib.connectAsync(config['host'], config['port'], 9999)
            
            if ib.isConnected():
                print(f"✅ {config['name']} 连接成功!")
                
                # 测试基本功能
                try:
                    accounts = ib.managedAccounts()
                    if accounts:
                        print(f"   📊 账户: {accounts}")
                    else:
                        print("   ⚠️ 未获取到账户信息")
                except Exception as e:
                    print(f"   ❌ 账户信息获取失败: {e}")
                
                # 断开连接
                await ib.disconnectAsync()
                self.test_results.append((config['name'], "✅ 连接成功"))
                return True
            else:
                print(f"❌ {config['name']} 连接失败")
                self.test_results.append((config['name'], "❌ 连接失败"))
                return False
                
        except Exception as e:
            print(f"❌ {config['name']} 连接异常: {e}")
            self.test_results.append((config['name'], f"❌ 异常: {e}"))
            return False
    
    async def test_all_connections(self):
        """测试所有连接配置"""
        print("🚀 开始详细IBKR连接测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success_count = 0
        for config in self.connection_configs:
            if await self.test_single_connection(config):
                success_count += 1
        
        return success_count
    
    def on_error(self, reqId, errorCode, errorString, contract):
        """错误回调"""
        print(f"   ❌ IBKR错误: ID={reqId}, 代码={errorCode}, 消息={errorString}")
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 IBKR API详细连接测试总结")
        print("="*60)
        
        for test_name, result in self.test_results:
            print(f"{test_name:<20}: {result}")
        
        print("="*60)
        
        # 统计结果
        success_count = sum(1 for _, result in self.test_results if "✅" in result)
        total_count = len(self.test_results)
        
        print(f"总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        if success_count == 0:
            print("\n❌ 所有连接测试失败")
            print("\n🔧 故障排除建议:")
            print("1. 确保TWS或IB Gateway正在运行")
            print("2. 检查TWS/IB Gateway的API设置:")
            print("   - 启用API连接")
            print("   - 允许来自本地主机的连接")
            print("   - 检查端口设置")
            print("3. 检查防火墙设置")
            print("4. 确认使用的是正确的端口:")
            print("   - TWS Paper: 4002")
            print("   - TWS Live: 7497")
            print("   - IB Gateway Paper: 4001")
            print("   - IB Gateway Live: 7496")
        elif success_count == total_count:
            print("\n🎉 所有连接测试通过！")
        else:
            print(f"\n⚠️ {success_count}/{total_count} 个连接测试通过")
    
    def print_connection_guide(self):
        """打印连接指南"""
        print("\n" + "="*60)
        print("📖 IBKR连接配置指南")
        print("="*60)
        print("1. TWS (Trader Workstation) 设置:")
        print("   - 打开TWS")
        print("   - 进入 Edit > Global Configuration")
        print("   - 选择 API > Settings")
        print("   - 启用 'Enable ActiveX and Socket Clients'")
        print("   - 设置 Socket port: 4002 (Paper) 或 7497 (Live)")
        print("   - 启用 'Allow connections from localhost'")
        print("   - 点击 'OK' 保存设置")
        print()
        print("2. IB Gateway 设置:")
        print("   - 打开IB Gateway")
        print("   - 进入 Configuration > API")
        print("   - 启用 'Enable ActiveX and Socket Clients'")
        print("   - 设置 Socket port: 4001 (Paper) 或 7496 (Live)")
        print("   - 启用 'Allow connections from localhost'")
        print()
        print("3. 防火墙设置:")
        print("   - 确保防火墙允许Python访问网络")
        print("   - 允许TWS/IB Gateway通过防火墙")
        print()
        print("4. 测试连接:")
        print("   - 运行此测试脚本")
        print("   - 检查哪个端口可以成功连接")
        print("   - 在交易系统中使用成功的端口配置")

async def main():
    """主函数"""
    if not IBKR_AVAILABLE:
        print("❌ IBKR API不可用，无法进行测试")
        return
    
    # 创建测试实例
    tester = DetailedIBKRConnectionTest()
    
    # 执行测试
    success_count = await tester.test_all_connections()
    
    # 打印测试总结
    tester.print_test_summary()
    
    # 如果所有测试都失败，显示连接指南
    if success_count == 0:
        tester.print_connection_guide()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main()) 