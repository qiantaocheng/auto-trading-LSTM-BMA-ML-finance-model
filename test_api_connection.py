#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API连接测试工具
测试IBKR API连接和交易功能

功能:
1. 测试IBKR TWS/Gateway连接
2. 验证账户信息获取
3. 测试股票数据获取
4. 验证下单权限（模拟）

Author: AI Assistant
Version: 1.0
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIConnectionTester:
    """API连接测试类"""
    
    def __init__(self):
        self.ib_client = None
        self.connection_status = False
        self.account_info = {}
        self.test_results = {}
        
    def test_ibkr_imports(self) -> Dict[str, bool]:
        """测试IBKR相关模块导入"""
        logger.info("=== 测试IBKR模块导入 ===")
        
        results = {
            'ibapi': False,
            'ib_insync': False,
            'yfinance': False,
            'pandas': False,
            'numpy': False
        }
        
        # 测试ibapi
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
            from ibapi.contract import Contract
            from ibapi.order import Order
            results['ibapi'] = True
            logger.info("✅ ibapi 导入成功")
        except ImportError as e:
            logger.warning(f"❌ ibapi 导入失败: {e}")
        
        # 测试ib_insync
        try:
            from ib_insync import IB, Stock, util
            results['ib_insync'] = True
            logger.info("✅ ib_insync 导入成功")
        except ImportError as e:
            logger.warning(f"❌ ib_insync 导入失败: {e}")
        
        # 测试yfinance
        try:
            import yfinance as yf
            results['yfinance'] = True
            logger.info("✅ yfinance 导入成功")
        except ImportError as e:
            logger.warning(f"❌ yfinance 导入失败: {e}")
        
        # 测试pandas
        try:
            import pandas as pd
            results['pandas'] = True
            logger.info("✅ pandas 导入成功")
        except ImportError as e:
            logger.warning(f"❌ pandas 导入失败: {e}")
        
        # 测试numpy
        try:
            import numpy as np
            results['numpy'] = True
            logger.info("✅ numpy 导入成功")
        except ImportError as e:
            logger.warning(f"❌ numpy 导入失败: {e}")
        
        return results
    
    def test_ib_insync_connection(self, host='127.0.0.1', port=7497, client_id=1) -> bool:
        """测试ib_insync连接"""
        logger.info(f"=== 测试ib_insync连接 {host}:{port} ===")
        
        try:
            from ib_insync import IB
            
            self.ib_client = IB()
            
            # 尝试连接
            logger.info(f"尝试连接到 {host}:{port} (Client ID: {client_id})")
            self.ib_client.connect(host, port, clientId=client_id)
            
            # 检查连接状态
            if self.ib_client.isConnected():
                logger.info("✅ 连接成功")
                self.connection_status = True
                return True
            else:
                logger.error("❌ 连接失败")
                return False
                
        except ImportError:
            logger.error("❌ ib_insync 不可用")
            return False
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")
            return False
    
    def test_account_info(self) -> Dict:
        """测试账户信息获取"""
        logger.info("=== 测试账户信息获取 ===")
        
        if not self.ib_client or not self.connection_status:
            logger.error("❌ 未建立有效连接")
            return {}
        
        try:
            # 获取账户信息
            accounts = self.ib_client.managedAccounts()
            logger.info(f"管理的账户: {accounts}")
            
            if accounts:
                account = accounts[0]
                
                # 获取账户价值
                account_values = self.ib_client.accountValues()
                
                account_info = {}
                for value in account_values:
                    if value.account == account:
                        account_info[value.tag] = {
                            'value': value.value,
                            'currency': value.currency
                        }
                
                # 提取关键信息
                key_info = {}
                important_keys = [
                    'NetLiquidation', 'TotalCashValue', 'AvailableFunds',
                    'BuyingPower', 'GrossPositionValue'
                ]
                
                for key in important_keys:
                    if key in account_info:
                        key_info[key] = account_info[key]
                
                logger.info("✅ 账户信息获取成功")
                logger.info(f"关键账户信息: {key_info}")
                
                self.account_info = key_info
                return key_info
            else:
                logger.error("❌ 没有找到管理的账户")
                return {}
                
        except Exception as e:
            logger.error(f"❌ 获取账户信息失败: {e}")
            return {}
    
    def test_stock_data(self, symbols=['AAPL', 'MSFT', 'GOOGL']) -> Dict:
        """测试股票数据获取"""
        logger.info(f"=== 测试股票数据获取 {symbols} ===")
        
        results = {}
        
        if self.ib_client and self.connection_status:
            # 使用IBKR获取数据
            results.update(self._test_ibkr_stock_data(symbols))
        
        # 使用yfinance作为备用
        results.update(self._test_yfinance_stock_data(symbols))
        
        return results
    
    def _test_ibkr_stock_data(self, symbols) -> Dict:
        """使用IBKR获取股票数据"""
        results = {}
        
        try:
            from ib_insync import Stock
            
            for symbol in symbols:
                try:
                    # 创建股票合约
                    stock = Stock(symbol, 'SMART', 'USD')
                    
                    # 请求合约详情
                    self.ib_client.qualifyContracts(stock)
                    
                    # 请求市场数据
                    ticker = self.ib_client.reqMktData(stock)
                    time.sleep(1)  # 等待数据
                    
                    if ticker.last and ticker.last > 0:
                        results[f'{symbol}_IBKR'] = {
                            'price': ticker.last,
                            'bid': ticker.bid,
                            'ask': ticker.ask,
                            'volume': ticker.volume,
                            'source': 'IBKR',
                            'timestamp': datetime.now().isoformat()
                        }
                        logger.info(f"✅ {symbol} IBKR数据: ${ticker.last}")
                    else:
                        logger.warning(f"⚠️ {symbol} IBKR数据无效")
                    
                    # 取消数据订阅
                    self.ib_client.cancelMktData(stock)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol} IBKR数据获取失败: {e}")
        
        except Exception as e:
            logger.error(f"❌ IBKR数据测试失败: {e}")
        
        return results
    
    def _test_yfinance_stock_data(self, symbols) -> Dict:
        """使用yfinance获取股票数据"""
        results = {}
        
        try:
            import yfinance as yf
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period='1d')
                    
                    if not hist.empty and 'regularMarketPrice' in info:
                        results[f'{symbol}_YF'] = {
                            'price': info.get('regularMarketPrice', 0),
                            'bid': info.get('bid', 0),
                            'ask': info.get('ask', 0),
                            'volume': info.get('volume', 0),
                            'market_cap': info.get('marketCap', 0),
                            'source': 'yfinance',
                            'timestamp': datetime.now().isoformat()
                        }
                        logger.info(f"✅ {symbol} yfinance数据: ${info.get('regularMarketPrice', 0)}")
                    else:
                        logger.warning(f"⚠️ {symbol} yfinance数据无效")
                
                except Exception as e:
                    logger.error(f"❌ {symbol} yfinance数据获取失败: {e}")
        
        except ImportError:
            logger.error("❌ yfinance 不可用")
        except Exception as e:
            logger.error(f"❌ yfinance数据测试失败: {e}")
        
        return results
    
    def test_order_permissions(self) -> Dict:
        """测试下单权限（不实际下单）"""
        logger.info("=== 测试下单权限 ===")
        
        if not self.ib_client or not self.connection_status:
            logger.error("❌ 未建立有效连接")
            return {'order_test': False, 'reason': 'No connection'}
        
        try:
            from ib_insync import Stock, LimitOrder
            
            # 创建测试股票和订单
            test_stock = Stock('AAPL', 'SMART', 'USD')
            self.ib_client.qualifyContracts(test_stock)
            
            # 创建一个极低价格的限价买单（不会成交）
            test_order = LimitOrder('BUY', 1, 0.01)
            
            # 尝试提交订单（但立即取消）
            trade = self.ib_client.placeOrder(test_stock, test_order)
            
            # 立即取消订单
            if trade:
                self.ib_client.cancelOrder(test_order)
                logger.info("✅ 下单权限测试成功（订单已取消）")
                return {
                    'order_test': True,
                    'reason': 'Order placed and cancelled successfully',
                    'order_id': test_order.orderId if hasattr(test_order, 'orderId') else 'N/A'
                }
            else:
                logger.error("❌ 下单测试失败")
                return {'order_test': False, 'reason': 'Failed to place order'}
        
        except Exception as e:
            logger.error(f"❌ 下单权限测试失败: {e}")
            return {'order_test': False, 'reason': str(e)}
    
    def run_full_test(self, host='127.0.0.1', port=7497, client_id=1) -> Dict:
        """运行完整的API连接测试"""
        logger.info("🚀 开始API连接测试")
        
        test_results = {
            'test_time': datetime.now().isoformat(),
            'connection_params': {
                'host': host,
                'port': port,
                'client_id': client_id
            }
        }
        
        # 1. 测试模块导入
        test_results['imports'] = self.test_ibkr_imports()
        
        # 2. 测试连接
        connection_success = False
        if test_results['imports']['ib_insync']:
            connection_success = self.test_ib_insync_connection(host, port, client_id)
        
        test_results['connection'] = {
            'success': connection_success,
            'method': 'ib_insync' if test_results['imports']['ib_insync'] else 'none'
        }
        
        # 3. 测试账户信息
        if connection_success:
            test_results['account'] = self.test_account_info()
        else:
            test_results['account'] = {}
        
        # 4. 测试股票数据
        test_results['stock_data'] = self.test_stock_data()
        
        # 5. 测试下单权限
        if connection_success:
            test_results['order_permissions'] = self.test_order_permissions()
        else:
            test_results['order_permissions'] = {'order_test': False, 'reason': 'No connection'}
        
        # 断开连接
        if self.ib_client and self.connection_status:
            try:
                self.ib_client.disconnect()
                logger.info("连接已断开")
            except:
                pass
        
        return test_results
    
    def generate_report(self, test_results: Dict) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("API连接测试报告")
        report.append("=" * 60)
        report.append(f"测试时间: {test_results.get('test_time', 'Unknown')}")
        report.append(f"连接参数: {test_results.get('connection_params', {})}")
        report.append("")
        
        # 模块导入结果
        report.append("📦 模块导入测试:")
        imports = test_results.get('imports', {})
        for module, success in imports.items():
            status = "✅" if success else "❌"
            report.append(f"  {status} {module}")
        report.append("")
        
        # 连接结果
        report.append("🔗 连接测试:")
        connection = test_results.get('connection', {})
        if connection.get('success', False):
            report.append(f"  ✅ 连接成功 (使用: {connection.get('method', 'unknown')})")
        else:
            report.append(f"  ❌ 连接失败")
        report.append("")
        
        # 账户信息
        report.append("💰 账户信息:")
        account = test_results.get('account', {})
        if account:
            for key, value in account.items():
                report.append(f"  {key}: {value}")
        else:
            report.append("  ❌ 未获取到账户信息")
        report.append("")
        
        # 股票数据
        report.append("📊 股票数据测试:")
        stock_data = test_results.get('stock_data', {})
        if stock_data:
            for symbol, data in stock_data.items():
                price = data.get('price', 0)
                source = data.get('source', 'unknown')
                report.append(f"  ✅ {symbol}: ${price} (来源: {source})")
        else:
            report.append("  ❌ 未获取到股票数据")
        report.append("")
        
        # 下单权限
        report.append("📝 下单权限测试:")
        order_test = test_results.get('order_permissions', {})
        if order_test.get('order_test', False):
            report.append(f"  ✅ 下单权限正常")
        else:
            reason = order_test.get('reason', 'Unknown')
            report.append(f"  ❌ 下单权限异常: {reason}")
        
        report.append("")
        report.append("=" * 60)
        
        # 总结
        connection_ok = connection.get('success', False)
        data_ok = len(stock_data) > 0
        order_ok = order_test.get('order_test', False)
        
        if connection_ok and data_ok and order_ok:
            report.append("🎉 测试结果: 所有功能正常，可以进行实盘交易")
        elif connection_ok and data_ok:
            report.append("⚠️ 测试结果: 连接和数据获取正常，但下单功能异常")
        elif data_ok:
            report.append("⚠️ 测试结果: 数据获取正常，但IBKR连接异常")
        else:
            report.append("❌ 测试结果: 多项功能异常，请检查配置")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API连接测试工具')
    parser.add_argument('--host', default='127.0.0.1', help='IBKR TWS/Gateway主机地址')
    parser.add_argument('--port', type=int, default=7497, help='IBKR TWS/Gateway端口')
    parser.add_argument('--client-id', type=int, default=1, help='客户端ID')
    parser.add_argument('--output', help='输出报告文件路径')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = APIConnectionTester()
    
    # 运行测试
    print("开始运行API连接测试...")
    test_results = tester.run_full_test(args.host, args.port, args.client_id)
    
    # 生成报告
    report = tester.generate_report(test_results)
    print(report)
    
    # 保存报告
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到: {args.output}")
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    # 保存JSON结果
    json_file = 'api_test_results.json'
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"详细结果已保存到: {json_file}")
    except Exception as e:
        print(f"保存详细结果失败: {e}")


if __name__ == "__main__":
    main()