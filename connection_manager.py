#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced IBKR Connection Manager with Auto-Reconnection and Heartbeat
实现自动重连、心跳检测和连接状态监控
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    print("Warning: ib_insync not available")


class AlertManager:
    """告警管理器 - 支持邮件、短信和GUI通知"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 邮件设置
        self.email_config = config.get('email', {})
        self.sms_config = config.get('sms', {})
        
        # 告警限制 - 防止频繁告警
        self.alert_cooldown = config.get('alert_cooldown_minutes', 5)
        self.last_alerts = {}
    
    def send_alert(self, alert_type: str, title: str, message: str, severity: str = 'warning'):
        """发送告警通知"""
        # 检查告警冷却期
        now = datetime.now()
        last_alert_time = self.last_alerts.get(alert_type)
        
        if last_alert_time and (now - last_alert_time).total_seconds() < self.alert_cooldown * 60:
            self.logger.debug(f"Alert {alert_type} in cooldown period, skipping")
            return
        
        self.last_alerts[alert_type] = now
        
        try:
            # 发送邮件告警
            if self.email_config.get('enabled', False):
                self._send_email_alert(title, message, severity)
            
            # 发送短信告警 (需要第三方服务)
            if self.sms_config.get('enabled', False) and severity in ['critical', 'error']:
                self._send_sms_alert(title, message)
            
            # GUI告警 (可以通过文件或消息队列实现)
            self._send_gui_alert(title, message, severity)
            
            self.logger.info(f"Alert sent: {alert_type} - {title}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, title: str, message: str, severity: str):
        """发送邮件告警"""
        if not self.email_config.get('smtp_server'):
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"[IBKR Trading {severity.upper()}] {title}"
            
            body = f"""
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
级别: {severity.upper()}
标题: {title}

详细信息:
{message}

---
IBKR自动交易系统
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config.get('smtp_port', 587))
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    def _send_sms_alert(self, title: str, message: str):
        """发送短信告警 (需要集成第三方SMS服务)"""
        # 这里可以集成阿里云短信、腾讯云短信等服务
        pass
    
    def _send_gui_alert(self, title: str, message: str, severity: str):
        """GUI告警 - 写入告警文件供GUI程序读取"""
        try:
            alert_file = "alerts/current_alerts.json"
            os.makedirs("alerts", exist_ok=True)
            
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'title': title,
                'message': message,
                'severity': severity,
                'acknowledged': False
            }
            
            # 简单实现：写入JSON文件
            import json
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alert_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"GUI alert failed: {e}")


class ConnectionManager:
    """增强的连接管理器 - 自动重连、心跳检测、状态监控"""
    
    def __init__(self, config: Dict[str, Any], alert_manager: AlertManager):
        self.config = config
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        # 连接配置
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 4002)
        self.client_id = config.get('client_id', 1)
        
        # 重连配置
        self.reconnect_enabled = config.get('auto_reconnect', True)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('reconnect_delay_seconds', 30)
        
        # 心跳配置
        self.heartbeat_interval = config.get('heartbeat_interval_seconds', 30)
        self.heartbeat_timeout = config.get('heartbeat_timeout_seconds', 10)
        
        # 状态变量
        self.ib = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_count = 0
        self.last_heartbeat = None
        self.trading_suspended = False
        
        # 线程
        self.heartbeat_thread = None
        self.reconnect_thread = None
        
        # 回调函数
        self.on_connected_callback = None
        self.on_disconnected_callback = None
        self.on_reconnected_callback = None
        self.on_connection_failed_callback = None
    
    def set_callbacks(self, 
                     on_connected: Optional[Callable] = None,
                     on_disconnected: Optional[Callable] = None, 
                     on_reconnected: Optional[Callable] = None,
                     on_connection_failed: Optional[Callable] = None):
        """设置连接状态回调函数"""
        self.on_connected_callback = on_connected
        self.on_disconnected_callback = on_disconnected
        self.on_reconnected_callback = on_reconnected
        self.on_connection_failed_callback = on_connection_failed
    
    def connect(self) -> bool:
        """建立IBKR连接"""
        if not IB_INSYNC_AVAILABLE:
            self.logger.error("ib_insync not available")
            return False
        
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=20)
            
            if self.ib.isConnected():
                self.is_connected = True
                self.is_running = True
                self.reconnect_count = 0
                self.last_heartbeat = datetime.now()
                
                self.logger.info(f"Connected to IBKR at {self.host}:{self.port} (client_id={self.client_id})")
                
                # 启动心跳检测
                self._start_heartbeat()
                
                # 触发连接成功回调
                if self.on_connected_callback:
                    try:
                        self.on_connected_callback(self.ib)
                    except Exception as e:
                        self.logger.error(f"Connected callback error: {e}")
                
                return True
            else:
                self.logger.error("Failed to connect to IBKR")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self._handle_connection_error(str(e))
            return False
    
    def disconnect(self):
        """断开连接"""
        self.is_running = False
        
        if self.ib and self.ib.isConnected():
            try:
                self.ib.disconnect()
                self.logger.info("Disconnected from IBKR")
            except Exception as e:
                self.logger.error(f"Disconnect error: {e}")
        
        self.is_connected = False
        self._stop_heartbeat()
    
    def _start_heartbeat(self):
        """启动心跳检测线程"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        self.logger.info("Heartbeat monitoring started")
    
    def _stop_heartbeat(self):
        """停止心跳检测"""
        # 心跳线程会通过is_running标志自动停止
        pass
    
    def _heartbeat_loop(self):
        """心跳检测循环"""
        while self.is_running:
            try:
                if self.ib and self.ib.isConnected():
                    # 发送心跳请求 - 可以是简单的账户信息查询
                    start_time = time.time()
                    
                    # 使用简单的连接检查作为心跳
                    connected = self.ib.isConnected()
                    
                    if connected:
                        response_time = time.time() - start_time
                        self.last_heartbeat = datetime.now()
                        
                        if response_time > self.heartbeat_timeout:
                            self.logger.warning(f"Slow heartbeat response: {response_time:.2f}s")
                    else:
                        self.logger.warning("Heartbeat failed - connection lost")
                        self._handle_disconnection()
                else:
                    self.logger.warning("No connection available for heartbeat")
                    self._handle_disconnection()
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                self._handle_disconnection()
            
            time.sleep(self.heartbeat_interval)
    
    def _handle_disconnection(self):
        """处理连接断开"""
        if not self.is_connected:
            return  # 已经在处理断开
        
        self.is_connected = False
        self.trading_suspended = True
        
        self.logger.error("Connection lost to IBKR")
        
        # 发送断线告警
        self.alert_manager.send_alert(
            'connection_lost',
            'IBKR连接断开',
            f'与IBKR的连接在 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} 断开\n'
            f'交易已暂停，正在尝试重连...',
            'error'
        )
        
        # 触发断开回调
        if self.on_disconnected_callback:
            try:
                self.on_disconnected_callback()
            except Exception as e:
                self.logger.error(f"Disconnected callback error: {e}")
        
        # 启动重连
        if self.reconnect_enabled:
            self._start_reconnection()
    
    def _start_reconnection(self):
        """启动重连线程"""
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return
        
        self.reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self.reconnect_thread.start()
    
    def _reconnect_loop(self):
        """重连循环"""
        while self.is_running and not self.is_connected and self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            
            self.logger.info(f"Attempting reconnection {self.reconnect_count}/{self.max_reconnect_attempts}")
            
            try:
                # 关闭旧连接
                if self.ib:
                    try:
                        self.ib.disconnect()
                    except:
                        pass
                
                # 等待重连延迟
                time.sleep(self.reconnect_delay)
                
                # 尝试重连
                if self.connect():
                    self.logger.info("Reconnection successful")
                    self.trading_suspended = False
                    
                    # 发送重连成功告警
                    self.alert_manager.send_alert(
                        'reconnection_success',
                        'IBKR重连成功',
                        f'与IBKR的连接在第 {self.reconnect_count} 次尝试后成功恢复\n'
                        f'交易已恢复正常',
                        'info'
                    )
                    
                    # 触发重连成功回调
                    if self.on_reconnected_callback:
                        try:
                            self.on_reconnected_callback(self.ib)
                        except Exception as e:
                            self.logger.error(f"Reconnected callback error: {e}")
                    
                    return
                else:
                    self.logger.warning(f"Reconnection attempt {self.reconnect_count} failed")
                    
            except Exception as e:
                self.logger.error(f"Reconnection error: {e}")
        
        # 重连失败
        if not self.is_connected:
            self.logger.error(f"Failed to reconnect after {self.max_reconnect_attempts} attempts")
            
            # 发送重连失败告警
            self.alert_manager.send_alert(
                'reconnection_failed',
                'IBKR重连失败',
                f'经过 {self.max_reconnect_attempts} 次尝试后仍无法连接到IBKR\n'
                f'交易已暂停，请检查网络连接和TWS/Gateway状态\n'
                f'手动干预可能是必要的',
                'critical'
            )
            
            # 触发连接失败回调
            if self.on_connection_failed_callback:
                try:
                    self.on_connection_failed_callback()
                except Exception as e:
                    self.logger.error(f"Connection failed callback error: {e}")
    
    def _handle_connection_error(self, error_msg: str):
        """处理连接错误"""
        self.alert_manager.send_alert(
            'connection_error',
            'IBKR连接错误',
            f'连接IBKR时发生错误: {error_msg}\n'
            f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'error'
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        return {
            'connected': self.is_connected,
            'trading_suspended': self.trading_suspended,
            'reconnect_count': self.reconnect_count,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'host': self.host,
            'port': self.port,
            'client_id': self.client_id
        }
    
    def is_trading_allowed(self) -> bool:
        """检查是否允许交易"""
        return self.is_connected and not self.trading_suspended
    
    def force_reconnect(self):
        """强制重连"""
        self.logger.info("Force reconnection requested")
        self.reconnect_count = 0  # 重置重连计数
        self._handle_disconnection()


# 使用示例和测试
if __name__ == "__main__":
    # 配置
    config = {
        'host': '127.0.0.1',
        'port': 4002,
        'client_id': 1,
        'auto_reconnect': True,
        'max_reconnect_attempts': 5,
        'reconnect_delay_seconds': 30,
        'heartbeat_interval_seconds': 30,
        'heartbeat_timeout_seconds': 10,
        'email': {
            'enabled': False,  # 设置为True并填写邮件配置来启用
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'from_email': 'your_email@gmail.com',
            'to_email': 'alert_recipient@gmail.com'
        },
        'alert_cooldown_minutes': 5
    }
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建管理器
    alert_manager = AlertManager(config)
    conn_manager = ConnectionManager(config, alert_manager)
    
    # 设置回调
    def on_connected(ib):
        print("Connected callback triggered")
        
    def on_disconnected():
        print("Disconnected callback triggered")
        
    def on_reconnected(ib):
        print("Reconnected callback triggered")
        
    def on_connection_failed():
        print("Connection failed callback triggered")
    
    conn_manager.set_callbacks(
        on_connected=on_connected,
        on_disconnected=on_disconnected,
        on_reconnected=on_reconnected,
        on_connection_failed=on_connection_failed
    )
    
    # 测试连接
    if conn_manager.connect():
        print("Connection established")
        
        # 运行一段时间进行测试
        try:
            time.sleep(60)  # 运行1分钟
        except KeyboardInterrupt:
            pass
        
        conn_manager.disconnect()
    else:
        print("Failed to connect")