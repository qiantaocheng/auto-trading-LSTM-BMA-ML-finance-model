#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBKR连接问题诊断脚本
找出连接超时的具体原因
"""

import asyncio
import sys
import os
import socket
import time
from datetime import datetime

# IBKR相关导入
try:
    from ib_insync import *
    import ib_insync as ibs
    IBKR_AVAILABLE = True
    print("✅ ib_insync已成功导入")
except ImportError as e:
    print(f"❌ ib_insync导入失败: {e}")
    IBKR_AVAILABLE = False

class IBKRDiagnostic:
    """IBKR连接诊断类"""
    
    def __init__(self):
        self.diagnostic_results = []
        
    def check_port_status(self, host, port):
        """检查端口状态"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return "OPEN"
            else:
                return "CLOSED"
        except Exception as e:
            return f"ERROR: {e}"
    
    def check_tws_process(self):
        """检查TWS进程"""
        import psutil
        
        tws_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'tws' in proc.info['name'].lower() or 'ib' in proc.info['name'].lower():
                    tws_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return tws_processes
    
    async def test_connection_with_timeout(self, host, port, client_id, timeout=10):
        """测试连接并记录超时"""
        print(f"\n🔍 测试连接 {host}:{port} (Client ID: {client_id})")
        print(f"   超时设置: {timeout}秒")
        
        start_time = time.time()
        
        try:
            ib = IB()
            
            # 设置事件处理
            ib.errorEvent += lambda reqId, errorCode, errorString, contract: self.on_error(reqId, errorCode, errorString, contract)
            
            # 尝试连接
            await asyncio.wait_for(
                ib.connectAsync(host, port, client_id),
                timeout=timeout
            )
            
            if ib.isConnected():
                elapsed = time.time() - start_time
                print(f"✅ 连接成功! 耗时: {elapsed:.2f}秒")
                await ib.disconnectAsync()
                return True, elapsed
            else:
                elapsed = time.time() - start_time
                print(f"❌ 连接失败 耗时: {elapsed:.2f}秒")
                return False, elapsed
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"⏰ 连接超时! 耗时: {elapsed:.2f}秒")
            return False, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 连接异常: {e} 耗时: {elapsed:.2f}秒")
            return False, elapsed
    
    def on_error(self, reqId, errorCode, errorString, contract):
        """错误回调"""
        print(f"   ❌ IBKR错误: ID={reqId}, 代码={errorCode}, 消息={errorString}")
    
    async def run_diagnostic(self):
        """运行完整诊断"""
        print("🔍 开始IBKR连接问题诊断")
        print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 检查端口状态
        print("\n📊 步骤1: 检查端口状态")
        ports_to_check = [4002, 7497, 4001, 7496]
        
        for port in ports_to_check:
            status = self.check_port_status('127.0.0.1', port)
            print(f"   端口 {port}: {status}")
            self.diagnostic_results.append(("端口检查", f"端口{port}: {status}"))
        
        # 2. 检查TWS进程
        print("\n📊 步骤2: 检查TWS进程")
        tws_processes = self.check_tws_process()
        
        if tws_processes:
            print(f"✅ 发现 {len(tws_processes)} 个IBKR相关进程:")
            for proc in tws_processes:
                print(f"   - PID: {proc['pid']}, 名称: {proc['name']}")
            self.diagnostic_results.append(("进程检查", f"发现{len(tws_processes)}个IBKR进程"))
        else:
            print("❌ 未发现IBKR相关进程")
            self.diagnostic_results.append(("进程检查", "未发现IBKR进程"))
        
        # 3. 测试不同Client ID
        print("\n📊 步骤3: 测试不同Client ID")
        client_ids = [9999, 50310, 1, 2, 3]
        
        for client_id in client_ids:
            success, elapsed = await self.test_connection_with_timeout('127.0.0.1', 4002, client_id, 5)
            status = "成功" if success else "失败"
            self.diagnostic_results.append(("Client ID测试", f"ID {client_id}: {status} ({elapsed:.2f}s)"))
        
        # 4. 测试不同超时设置
        print("\n📊 步骤4: 测试不同超时设置")
        timeouts = [3, 5, 10, 15]
        
        for timeout in timeouts:
            success, elapsed = await self.test_connection_with_timeout('127.0.0.1', 4002, 50310, timeout)
            status = "成功" if success else "失败"
            self.diagnostic_results.append(("超时测试", f"{timeout}s: {status} ({elapsed:.2f}s)"))
        
        # 5. 检查网络配置
        print("\n📊 步骤5: 检查网络配置")
        try:
            # 检查本地回环
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 4002))
            sock.close()
            
            if result == 0:
                print("✅ 本地回环网络正常")
                self.diagnostic_results.append(("网络配置", "本地回环正常"))
            else:
                print("❌ 本地回环网络异常")
                self.diagnostic_results.append(("网络配置", "本地回环异常"))
        except Exception as e:
            print(f"❌ 网络配置检查失败: {e}")
            self.diagnostic_results.append(("网络配置", f"检查失败: {e}"))
    
    def print_diagnostic_summary(self):
        """打印诊断总结"""
        print("\n" + "="*60)
        print("📊 IBKR连接问题诊断总结")
        print("="*60)
        
        for test_name, result in self.diagnostic_results:
            print(f"{test_name:<15}: {result}")
        
        print("="*60)
        
        # 分析问题
        print("\n🔍 问题分析:")
        
        # 检查是否有TWS进程
        has_tws = any("进程检查" in item[0] and "发现" in item[1] for item in self.diagnostic_results)
        
        if not has_tws:
            print("❌ 主要问题: TWS/IB Gateway未运行")
            print("   解决方案: 启动TWS或IB Gateway")
        else:
            print("✅ TWS进程正在运行")
            
            # 检查端口状态
            port_4002_open = any("端口4002" in item[1] and "OPEN" in item[1] for item in self.diagnostic_results)
            
            if port_4002_open:
                print("✅ 端口4002已开放")
                print("⚠️ 可能的问题:")
                print("   1. API连接未在TWS中启用")
                print("   2. Client ID冲突")
                print("   3. 防火墙阻止连接")
                print("   4. TWS需要重启")
            else:
                print("❌ 端口4002未开放")
                print("   解决方案: 在TWS中启用API连接")
        
        print("\n🔧 建议的解决步骤:")
        print("1. 确保TWS正在运行")
        print("2. 在TWS中: Edit > Global Configuration > API > Settings")
        print("3. 启用 'Enable ActiveX and Socket Clients'")
        print("4. 设置 Socket port: 4002")
        print("5. 启用 'Allow connections from localhost'")
        print("6. 重启TWS")
        print("7. 重新运行连接测试")

async def main():
    """主函数"""
    if not IBKR_AVAILABLE:
        print("❌ IBKR API不可用，无法进行诊断")
        return
    
    # 创建诊断实例
    diagnostic = IBKRDiagnostic()
    
    # 运行诊断
    await diagnostic.run_diagnostic()
    
    # 打印诊断总结
    diagnostic.print_diagnostic_summary()

if __name__ == "__main__":
    # 运行异步诊断
    asyncio.run(main()) 