#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IBKR持久化连接客户端
实现自动重连、事件驱动市场数据订阅和完整订单生命周期跟踪
"""

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import BarData, TickerId
    IBKR_AVAILABLE = True
except ImportError:
    print("IBKR API不可用")
    IBKR_AVAILABLE = False
    
    # 创建占位符类
    class EWrapper:
        pass
    
    class EClient:
        def __init__(self, wrapper):
            pass
    
    class Contract:
        pass
    
    class Order:
        pass
    
    class BarData:
        pass
    
    TickerId = int


class PersistentIBKRClient(EWrapper, EClient):
    """持久化IBKR客户端 - 支持自动重连和事件驱动"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 50310):
        if not IBKR_AVAILABLE:
            raise ImportError("IBKR API不可用")
            
        EClient.__init__(self, self)
        
        # 连接参数
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # 连接状态
        self.connected = False
        self.connection_thread = None
        self.auto_reconnect = True
        self.reconnect_interval = 5  # 秒
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0
        
        # 订单管理
        self.nextOrderId = None
        self.orders = {}  # orderId -> order_info
        self.order_status = {}  # orderId -> status_info
        self.executions = {}  # execId -> execution_info
        self.commission_reports = {}  # execId -> commission_info
        self.open_orders = {}  # orderId -> open_order_info
        
        # 市场数据管理
        self.market_data = {}  # reqId -> tick_data
        self.historical_data = {}  # reqId -> bar_data_list
        self.data_ready = {}  # reqId -> bool
        self.subscribed_contracts = {}  # reqId -> contract
        self.subscription_callbacks = {}  # reqId -> callback_function
        
        # 账户数据
        self.account_info = {}
        self.positions_data = []
        self.portfolio_data = []
        
        # 事件回调
        self.event_callbacks = {
            'connection_lost': [],
            'connection_restored': [],
            'order_filled': [],
            'order_cancelled': [],
            'market_data_update': [],
            'position_update': []
        }
        
        # 日志
        self.logger = self._setup_logger()
        
        # 启动连接
        self._start_connection()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('PersistentIBKR')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _start_connection(self):
        """启动连接"""
        self.logger.info(f"启动IBKR连接: {self.host}:{self.port} (ClientID: {self.client_id})")
        self._connect()
    
    def _connect(self):
        """建立连接"""
        try:
            self.connect(self.host, self.port, self.client_id)
            self.connection_thread = threading.Thread(target=self.run, daemon=True)
            self.connection_thread.start()
            
            # 等待连接确认
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < 10:
                time.sleep(0.1)
            
            if self.connected:
                self.logger.info("IBKR连接成功")
                self.reconnect_attempts = 0
                self._trigger_event('connection_restored')
            else:
                self.logger.error("IBKR连接超时")
                self._schedule_reconnect()
                
        except Exception as e:
            self.logger.error(f"IBKR连接失败: {e}")
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """安排重连"""
        if not self.auto_reconnect:
            return
            
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            self.logger.error(f"达到最大重连次数({self.max_reconnect_attempts})，停止重连")
            return
        
        self.logger.info(f"将在{self.reconnect_interval}秒后尝试第{self.reconnect_attempts}次重连")
        
        def delayed_reconnect():
            time.sleep(self.reconnect_interval)
            if not self.connected and self.auto_reconnect:
                self.logger.info(f"尝试重连({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                self._connect()
        
        threading.Thread(target=delayed_reconnect, daemon=True).start()
    
    # ====== EWrapper回调方法 ======
    
    def connectAck(self):
        """连接确认"""
        self.connected = True
        self.logger.info("收到连接确认")
    
    def connectionClosed(self):
        """连接关闭"""
        self.connected = False
        self.logger.warning("IBKR连接已关闭")
        self._trigger_event('connection_lost')
        
        if self.auto_reconnect:
            self._schedule_reconnect()
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """错误处理"""
        if errorCode in [502, 503, 504]:  # 连接相关错误
            self.logger.error(f"连接错误 {errorCode}: {errorString}")
            self.connected = False
            if self.auto_reconnect:
                self._schedule_reconnect()
        elif errorCode == 2104:  # 市场数据farm连接
            self.logger.info(f"市场数据服务器连接: {errorString}")
        elif errorCode == 2106:  # 历史数据farm连接
            self.logger.info(f"历史数据服务器连接: {errorString}")
        else:
            self.logger.warning(f"错误 {errorCode} (ReqId: {reqId}): {errorString}")
    
    def nextValidId(self, orderId: int):
        """下一个有效订单ID"""
        self.nextOrderId = orderId
        self.logger.info(f"下一个订单ID: {orderId}")
    
    # ====== 市场数据回调 ======
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """实时价格更新"""
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        
        self.market_data[reqId][tickType] = {
            'price': price,
            'timestamp': datetime.now()
        }
        
        # 触发市场数据更新事件
        self._trigger_event('market_data_update', {
            'reqId': reqId,
            'tickType': tickType,
            'price': price,
            'contract': self.subscribed_contracts.get(reqId)
        })
        
        # 调用订阅回调
        if reqId in self.subscription_callbacks:
            try:
                self.subscription_callbacks[reqId](reqId, tickType, price)
            except Exception as e:
                self.logger.error(f"市场数据回调错误: {e}")
    
    def tickSize(self, reqId: TickerId, tickType: int, size: int):
        """实时数量更新"""
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        
        self.market_data[reqId][f"{tickType}_size"] = {
            'size': size,
            'timestamp': datetime.now()
        }
    
    def historicalData(self, reqId: int, bar: BarData):
        """历史数据"""
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'count': bar.count,
            'wap': bar.wap
        })
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """历史数据结束"""
        self.data_ready[reqId] = True
        self.logger.info(f"历史数据接收完成 (ReqId: {reqId}), 数据量: {len(self.historical_data.get(reqId, []))}")
    
    # ====== 订单管理回调 ======
    
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                   clientId: int, whyHeld: str, mktCapPrice: float):
        """订单状态更新"""
        self.order_status[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'permId': permId,
            'parentId': parentId,
            'lastFillPrice': lastFillPrice,
            'clientId': clientId,
            'whyHeld': whyHeld,
            'mktCapPrice': mktCapPrice,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"订单状态更新 {orderId}: {status}, 已成交: {filled}, 剩余: {remaining}")
        
        # 触发事件
        if status == 'Filled':
            self._trigger_event('order_filled', {
                'orderId': orderId,
                'filled': filled,
                'avgFillPrice': avgFillPrice
            })
        elif status == 'Cancelled':
            self._trigger_event('order_cancelled', {'orderId': orderId})
    
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
        """未成交订单"""
        self.open_orders[orderId] = {
            'contract': contract,
            'order': order,
            'orderState': orderState,
            'timestamp': datetime.now()
        }
    
    def execDetails(self, reqId: int, contract: Contract, execution):
        """执行详情"""
        self.executions[execution.execId] = {
            'orderId': execution.orderId,
            'contract': contract,
            'execution': execution,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"执行详情: 订单{execution.orderId}, 数量: {execution.shares}, 价格: {execution.price}")
    
    def commissionReport(self, commissionReport):
        """佣金报告"""
        self.commission_reports[commissionReport.execId] = {
            'commission': commissionReport.commission,
            'currency': commissionReport.currency,
            'realizedPNL': commissionReport.realizedPNL,
            'timestamp': datetime.now()
        }
    
    # ====== 账户和持仓回调 ======
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """账户摘要"""
        if account not in self.account_info:
            self.account_info[account] = {}
        
        self.account_info[account][tag] = {
            'value': value,
            'currency': currency,
            'timestamp': datetime.now()
        }
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """持仓更新"""
        position_info = {
            'account': account,
            'symbol': contract.symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'position': position,
            'avgCost': avgCost,
            'timestamp': datetime.now()
        }
        
        self.positions_data.append(position_info)
        
        # 触发持仓更新事件
        if position != 0:  # 只对非零持仓触发事件
            self._trigger_event('position_update', position_info)
    
    def positionEnd(self):
        """持仓数据结束"""
        self.logger.info(f"收到持仓数据，共{len(self.positions_data)}个持仓")
    
    # ====== 公共API方法 ======
    
    def subscribe_market_data(self, contract: Contract, callback: Optional[Callable] = None) -> int:
        """订阅市场数据"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法订阅市场数据")
            return -1
        
        req_id = len(self.subscribed_contracts) + 1000
        self.subscribed_contracts[req_id] = contract
        
        if callback:
            self.subscription_callbacks[req_id] = callback
        
        # 订阅实时数据
        self.reqMktData(req_id, contract, "", False, False, [])
        
        self.logger.info(f"订阅市场数据: {contract.symbol} (ReqId: {req_id})")
        return req_id
    
    def unsubscribe_market_data(self, req_id: int):
        """取消订阅市场数据"""
        if req_id in self.subscribed_contracts:
            self.cancelMktData(req_id)
            del self.subscribed_contracts[req_id]
            if req_id in self.subscription_callbacks:
                del self.subscription_callbacks[req_id]
            self.logger.info(f"取消市场数据订阅 (ReqId: {req_id})")
    
    def place_order(self, contract: Contract, order: Order) -> int:
        """下单"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法下单")
            return -1
        
        if self.nextOrderId is None:
            self.logger.error("订单ID未初始化，无法下单")
            return -1
        
        order_id = self.nextOrderId
        self.nextOrderId += 1
        
        # 记录订单信息
        self.orders[order_id] = {
            'contract': contract,
            'order': order,
            'timestamp': datetime.now(),
            'status': 'Submitted'
        }
        
        # 发送订单
        self.placeOrder(order_id, contract, order)
        
        self.logger.info(f"提交订单 {order_id}: {order.action} {order.totalQuantity} {contract.symbol}")
        return order_id
    
    def cancel_order(self, order_id: int):
        """取消订单"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法取消订单")
            return
        
        self.cancelOrder(order_id, "")
        self.logger.info(f"请求取消订单: {order_id}")
    
    def get_historical_data(self, contract: Contract, duration: str = "1 Y", 
                           bar_size: str = "1 day", what_to_show: str = "TRADES") -> List[Dict]:
        """获取历史数据"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法获取历史数据")
            return []
        
        req_id = len(self.historical_data) + 2000
        self.data_ready[req_id] = False
        
        # 请求历史数据
        self.reqHistoricalData(
            req_id, contract, "", duration, bar_size, 
            what_to_show, 1, 1, False, []
        )
        
        # 等待数据完成
        timeout = 30  # 30秒超时
        start_time = time.time()
        while not self.data_ready.get(req_id, False) and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.data_ready.get(req_id, False):
            return self.historical_data.get(req_id, [])
        else:
            self.logger.error(f"获取历史数据超时 (ReqId: {req_id})")
            return []
    
    def get_current_positions(self) -> List[Dict]:
        """获取当前持仓"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法获取持仓")
            return []
        
        self.positions_data = []  # 清空旧数据
        self.reqPositions()
        
        # 等待数据
        time.sleep(2)
        
        return [pos for pos in self.positions_data if pos['position'] != 0]
    
    def get_account_summary(self, account: str = "All") -> Dict:
        """获取账户摘要"""
        if not self.connected:
            self.logger.error("未连接到IBKR，无法获取账户信息")
            return {}
        
        req_id = 9001
        tags = "NetLiquidation,TotalCashValue,AvailableFunds,BuyingPower"
        
        self.reqAccountSummary(req_id, account, tags)
        
        # 等待数据
        time.sleep(2)
        
        return self.account_info
    
    # ====== 事件管理 ======
    
    def add_event_listener(self, event_name: str, callback: Callable):
        """添加事件监听器"""
        if event_name in self.event_callbacks:
            self.event_callbacks[event_name].append(callback)
    
    def remove_event_listener(self, event_name: str, callback: Callable):
        """移除事件监听器"""
        if event_name in self.event_callbacks and callback in self.event_callbacks[event_name]:
            self.event_callbacks[event_name].remove(callback)
    
    def _trigger_event(self, event_name: str, data: Dict = None):
        """触发事件"""
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    if data:
                        callback(data)
                    else:
                        callback()
                except Exception as e:
                    self.logger.error(f"事件回调错误 ({event_name}): {e}")
    
    # ====== 连接管理 ======
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected and self.isConnected()
    
    def stop_auto_reconnect(self):
        """停止自动重连"""
        self.auto_reconnect = False
        self.logger.info("已停止自动重连")
    
    def start_auto_reconnect(self):
        """启动自动重连"""
        self.auto_reconnect = True
        self.logger.info("已启动自动重连")
    
    def disconnect(self):
        """断开连接"""
        self.auto_reconnect = False  # 停止自动重连
        if self.connected:
            EClient.disconnect(self)
        self.logger.info("主动断开IBKR连接")
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """创建股票合约"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def create_market_order(self, action: str, quantity: float) -> Order:
        """创建市价单"""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "MKT"
        return order
    
    def create_limit_order(self, action: str, quantity: float, limit_price: float) -> Order:
        """创建限价单"""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "LMT"
        order.lmtPrice = limit_price
        return order
    
    def create_stop_order(self, action: str, quantity: float, stop_price: float) -> Order:
        """创建止损单"""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "STP"
        order.auxPrice = stop_price
        return order


def main():
    """测试持久化客户端"""
    if not IBKR_AVAILABLE:
        print("IBKR API不可用，无法运行测试")
        return
    
    # 创建客户端
    client = PersistentIBKRClient()
    
    # 添加事件监听器
    def on_connection_restored():
        print("连接已恢复")
        
        # 订阅市场数据
        aapl_contract = client.create_stock_contract("AAPL")
        
        def market_data_callback(req_id, tick_type, price):
            print(f"AAPL价格更新: TickType={tick_type}, Price={price}")
        
        client.subscribe_market_data(aapl_contract, market_data_callback)
    
    def on_connection_lost():
        print("连接已断开")
    
    def on_order_filled(data):
        print(f"订单成交: {data}")
    
    client.add_event_listener('connection_restored', on_connection_restored)
    client.add_event_listener('connection_lost', on_connection_lost)
    client.add_event_listener('order_filled', on_order_filled)
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("正在停止...")
        client.disconnect()


if __name__ == "__main__":
    main()