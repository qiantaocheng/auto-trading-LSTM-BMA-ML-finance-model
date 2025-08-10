#!/usr/bin/env python3
"""
交易审计器 v2 - 增强版，包含完整的合规检查功能
"""

import os
import sqlite3
import json
import time
import logging
from typing import Dict, List, Any, Optional
from collections import deque


class TradingAuditor:
    """交易审计器 - 记录、监控和合规检查"""
    
    def __init__(self, log_directory: str = "audit_logs", db_path: str = "trading_audit.db"):
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)
        self.db_path = db_path
        self.logger = logging.getLogger("TradingAuditor")
        
        # 合规检查缓存
        self._recent_orders = deque(maxlen=1000)  # 最近1000笔订单
        self._daily_stats = {}
        
        self._init_db()
        self._init_compliance_rules()
    
    def _init_db(self):
        """初始化审计数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 审计日志表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        symbol TEXT,
                        order_id INTEGER,
                        data_json TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 合规警告表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        symbol TEXT,
                        alert_type TEXT NOT NULL,
                        severity TEXT DEFAULT 'WARNING',
                        details TEXT,
                        order_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 交易统计表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_stats (
                        date TEXT PRIMARY KEY,
                        total_orders INTEGER DEFAULT 0,
                        total_volume REAL DEFAULT 0,
                        total_value REAL DEFAULT 0,
                        largest_order REAL DEFAULT 0,
                        compliance_alerts INTEGER DEFAULT 0,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("审计数据库初始化完成")
                
        except Exception as e:
            self.logger.error(f"审计数据库初始化失败: {e}")
    
    def _init_compliance_rules(self):
        """初始化合规规则"""
        self.compliance_rules = {
            # 大额订单阈值
            'large_order_threshold': 100000,  # $100K
            'large_order_pct_threshold': 0.15,  # 15% of account value
            
            # 频率限制
            'max_orders_per_5min': 10,
            'max_orders_per_symbol_daily': 8,
            'max_orders_per_hour': 25,
            
            # 集中度限制
            'max_single_position_pct': 0.2,  # 20%
            'max_sector_concentration': 0.3,  # 30%
            
            # 风险参数
            'max_daily_loss_pct': 0.05,  # 5%
            'max_leverage': 2.0,
            
            # 特殊标记
            'watch_symbols': ['TSLA', 'GME', 'AMC'],  # 特别关注的股票
            'restricted_hours': [],  # 限制交易时间段
        }
    
    def log_order(self, order_data: Dict[str, Any]):
        """记录订单到数据库和日志文件"""
        try:
            # 数据库记录
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_log (timestamp, event_type, symbol, order_id, data_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    time.time(), 
                    'ORDER', 
                    order_data.get('symbol'), 
                    order_data.get('order_id'),
                    json.dumps(order_data, ensure_ascii=False, default=str)
                ))
                conn.commit()
            
            # 结构化日志文件
            log_entry = {
                'type': 'ORDER',
                'timestamp': time.time(),
                'data': order_data
            }
            self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
            
            # 执行合规检查
            self._perform_compliance_checks(order_data)
            
            # 更新统计
            self._update_daily_stats(order_data)
            
        except Exception as e:
            self.logger.error(f"审计日志记录失败: {e}")
    
    def _perform_compliance_checks(self, order_data: Dict[str, Any]):
        """执行合规检查"""
        warnings = []
        severity = 'WARNING'
        
        current_time = time.time()
        symbol = order_data.get('symbol', '')
        order_value = order_data.get('quantity', 0) * order_data.get('price', 0)
        account_value = order_data.get('account_value', 0)
        
        # 1. 大额订单检查
        if order_value > self.compliance_rules['large_order_threshold']:
            warnings.append(f"大额订单: ${order_value:,.0f}")
            severity = 'HIGH'
        
        # 2. 大比例订单检查
        if account_value > 0:
            order_pct = order_value / account_value
            if order_pct > self.compliance_rules['large_order_pct_threshold']:
                warnings.append(f"大比例订单: {order_pct:.1%}")
                severity = 'HIGH'
        
        # 3. 高频交易检查
        recent_5min = [
            order for order in self._recent_orders 
            if current_time - order.get('timestamp', 0) < 300
        ]
        if len(recent_5min) > self.compliance_rules['max_orders_per_5min']:
            warnings.append(f"高频交易: 5分钟内{len(recent_5min)}笔订单")
            severity = 'HIGH'
        
        # 4. 单股票集中度检查
        if symbol:
            symbol_orders_today = [
                order for order in self._recent_orders 
                if (order.get('symbol') == symbol and 
                    current_time - order.get('timestamp', 0) < 86400)
            ]
            if len(symbol_orders_today) > self.compliance_rules['max_orders_per_symbol_daily']:
                warnings.append(f"单股集中: {symbol} 今日{len(symbol_orders_today)}笔")
        
        # 5. 特别关注股票检查
        if symbol in self.compliance_rules['watch_symbols']:
            warnings.append(f"关注股票: {symbol}")
        
        # 6. 手动风险参数检查
        if order_data.get('risk_level') == 'MANUAL':
            warnings.append("手动风险参数")
        
        # 7. 算法交易检查
        if order_data.get('order_type', '').startswith('ALGO_'):
            algorithm = order_data.get('algorithm', 'UNKNOWN')
            warnings.append(f"算法交易: {algorithm}")
        
        # 8. 时间段检查
        import datetime
        current_hour = datetime.datetime.now().hour
        if current_hour in self.compliance_rules['restricted_hours']:
            warnings.append(f"受限时段交易: {current_hour}时")
        
        # 记录合规结果
        if warnings:
            self._log_compliance_alert(order_data, warnings, severity)
        
        # 缓存订单
        order_data['timestamp'] = current_time
        self._recent_orders.append(order_data)
    
    def _log_compliance_alert(self, order_data: Dict[str, Any], warnings: List[str], severity: str):
        """记录合规警告"""
        try:
            alert_data = {
                'timestamp': time.time(),
                'symbol': order_data.get('symbol'),
                'order_id': order_data.get('order_id'),
                'warnings': warnings,
                'severity': severity,
                'order_summary': {
                    'action': order_data.get('action'),
                    'quantity': order_data.get('quantity'),
                    'value': order_data.get('quantity', 0) * order_data.get('price', 0)
                }
            }
            
            # 写入数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO compliance_alerts 
                    (timestamp, symbol, alert_type, severity, details, order_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert_data['timestamp'],
                    alert_data['symbol'],
                    'MULTIPLE_WARNINGS',
                    severity,
                    '; '.join(warnings),
                    json.dumps(alert_data, ensure_ascii=False, default=str)
                ))
                conn.commit()
            
            # 日志输出
            warning_msg = f"合规警告 [{alert_data['symbol']}]: {'; '.join(warnings)}"
            
            if severity == 'HIGH':
                self.logger.critical(f"🚨 严重合规问题: {warning_msg}")
            else:
                self.logger.warning(f"⚠️ {warning_msg}")
            
        except Exception as e:
            self.logger.error(f"合规警告记录失败: {e}")
    
    def _update_daily_stats(self, order_data: Dict[str, Any]):
        """更新日常统计"""
        try:
            import datetime
            today = datetime.date.today().isoformat()
            
            order_value = order_data.get('quantity', 0) * order_data.get('price', 0)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取当日统计
                cursor.execute("SELECT * FROM trading_stats WHERE date = ?", (today,))
                row = cursor.fetchone()
                
                if row:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE trading_stats SET 
                        total_orders = total_orders + 1,
                        total_volume = total_volume + ?,
                        total_value = total_value + ?,
                        largest_order = MAX(largest_order, ?),
                        updated_at = CURRENT_TIMESTAMP
                        WHERE date = ?
                    """, (order_data.get('quantity', 0), order_value, order_value, today))
                else:
                    # 创建新记录
                    cursor.execute("""
                        INSERT INTO trading_stats 
                        (date, total_orders, total_volume, total_value, largest_order)
                        VALUES (?, 1, ?, ?, ?)
                    """, (today, order_data.get('quantity', 0), order_value, order_value))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"统计更新失败: {e}")
    
    def get_compliance_report(self, days: int = 1) -> Dict[str, Any]:
        """生成合规报告"""
        try:
            cutoff_time = time.time() - (days * 86400)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 合规警告统计
                cursor.execute("""
                    SELECT alert_type, severity, COUNT(*) as count
                    FROM compliance_alerts 
                    WHERE timestamp > ?
                    GROUP BY alert_type, severity
                """, (cutoff_time,))
                
                alerts = cursor.fetchall()
                
                # 交易统计
                cursor.execute("""
                    SELECT COUNT(*) as total_orders, 
                           SUM(total_value) as total_value,
                           MAX(largest_order) as largest_order
                    FROM trading_stats 
                    WHERE date > date('now', '-{} days')
                """.format(days))
                
                stats = cursor.fetchone()
                
                return {
                    'period_days': days,
                    'compliance_alerts': [
                        {'type': alert[0], 'severity': alert[1], 'count': alert[2]}
                        for alert in alerts
                    ],
                    'trading_stats': {
                        'total_orders': stats[0] or 0,
                        'total_value': stats[1] or 0,
                        'largest_order': stats[2] or 0
                    },
                    'compliance_score': self._calculate_compliance_score(alerts, stats),
                    'generated_at': time.time()
                }
                
        except Exception as e:
            self.logger.error(f"合规报告生成失败: {e}")
            return {'error': str(e)}
    
    def _calculate_compliance_score(self, alerts, stats) -> float:
        """计算合规评分 (0-100)"""
        try:
            total_orders = stats[0] or 1
            
            # 基础分100分
            score = 100.0
            
            # 根据警告扣分
            for alert in alerts:
                if alert[1] == 'HIGH':
                    score -= alert[2] * 10  # 高级警告扣10分
                else:
                    score -= alert[2] * 2   # 普通警告扣2分
            
            # 根据警告率额外扣分
            alert_rate = sum(alert[2] for alert in alerts) / total_orders
            if alert_rate > 0.1:  # 超过10%警告率
                score -= (alert_rate - 0.1) * 100
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0  # 默认中等评分
    
    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """记录风险事件"""
        try:
            risk_log = {
                'type': 'RISK_EVENT',
                'event_type': event_type,
                'timestamp': time.time(),
                'details': details
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_log (timestamp, event_type, data_json)
                    VALUES (?, ?, ?)
                """, (risk_log['timestamp'], 'RISK_EVENT', json.dumps(risk_log, ensure_ascii=False, default=str)))
                conn.commit()
            
            self.logger.warning(f"风险事件: {event_type} - {details}")
            
        except Exception as e:
            self.logger.error(f"风险事件记录失败: {e}")
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """清理旧记录"""
        try:
            cutoff_time = time.time() - (days_to_keep * 86400)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理旧的审计日志
                cursor.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff_time,))
                audit_deleted = cursor.rowcount
                
                # 清理旧的合规警告
                cursor.execute("DELETE FROM compliance_alerts WHERE timestamp < ?", (cutoff_time,))
                alert_deleted = cursor.rowcount
                
                conn.commit()
                
            self.logger.info(f"清理完成: 删除{audit_deleted}条审计记录, {alert_deleted}条合规警告")
            
        except Exception as e:
            self.logger.error(f"记录清理失败: {e}")


# 兼容性：别名
TradingAuditorV2 = TradingAuditor