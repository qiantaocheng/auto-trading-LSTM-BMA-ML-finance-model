#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冲击成本模型 - 基于Almgren-Chriss和Gatheral理论
实现凹函数冲击成本估算与实时滑点校正

理论基础:
- Almgren-Chriss: 临时冲击 + 永久冲击 + 价格风险
- Gatheral: 无动态套利条件下的凹型冲击函数
- 实证: 冲击成本 ≈ eta * sqrt(POV) 且具有衰减性
"""

import numpy as np
import pandas as pd
import logging
import sqlite3
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ExecutionRecord:
    """执行记录"""
    timestamp: float
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    executed_price: float
    reference_price: float  # 决策时的参考价格
    pov: float  # 参与率
    spread_bps: float
    realized_slippage_bps: float
    market_impact_bps: float  # 扣除点差后的净冲击
    trade_id: str

class ImpactModel:
    """
    冲击成本模型
    
    核心功能:
    1. 基于sqrt(POV)的凹函数冲击估算
    2. 实时滑点数据校正模型参数  
    3. 分股票/流动性分组的自适应参数
    4. 与VPIN毒性指标联动的动态调整
    """
    
    def __init__(self, db_path: str = "impact_model.db"):
        self.db_path = db_path
        
        # 冲击系数 eta: symbol -> impact_coefficient
        self.impact_coefficients = defaultdict(lambda: 5e-4)  # 默认值
        
        # 按流动性分组的系数 (高/中/低流动性)
        self.liquidity_group_coefficients = {
            'high': 2e-4,    # 高流动性股票
            'medium': 5e-4,  # 中等流动性
            'low': 1e-3      # 低流动性股票
        }
        
        # 固定费用 (bps)
        self.fixed_fee_bps = 0.2
        
        # 历史执行记录
        self.execution_history = deque(maxlen=10000)
        
        # 模型统计
        self.calibration_stats = defaultdict(dict)
        
        # 初始化数据库
        self._init_database()
        self._load_coefficients()
        
        logger.info("冲击成本模型初始化完成")
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 执行记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    executed_price REAL NOT NULL,
                    reference_price REAL NOT NULL,
                    pov REAL NOT NULL,
                    spread_bps REAL NOT NULL,
                    realized_slippage_bps REAL NOT NULL,
                    market_impact_bps REAL NOT NULL,
                    trade_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 冲击系数表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS impact_coefficients (
                    symbol TEXT PRIMARY KEY,
                    eta REAL NOT NULL,
                    last_calibration_date DATE,
                    sample_count INTEGER,
                    r_squared REAL,
                    liquidity_group TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("冲击成本数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def _load_coefficients(self):
        """从数据库加载冲击系数"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT symbol, eta FROM impact_coefficients')
            results = cursor.fetchall()
            
            for symbol, eta in results:
                self.impact_coefficients[symbol] = eta
            
            conn.close()
            logger.info(f"加载了{len(results)}个股票的冲击系数")
        except Exception as e:
            logger.warning(f"加载冲击系数失败: {e}")
    
    def estimate_cost_bps(self, symbol: str, pov: float, spread_bps: float, 
                         vpin: float = None, liquidity_group: str = None) -> float:
        """
        估算总交易成本 (bps)
        
        公式: 成本 = 点差 + 冲击函数(POV) + 固定费用 + VPIN调整
        
        Args:
            symbol: 股票代码
            pov: 参与率 (0-1)
            spread_bps: 当前点差 (bps)
            vpin: VPIN毒性指标 (可选)
            liquidity_group: 流动性分组 (可选)
        
        Returns:
            总成本 (bps)
        """
        # 获取冲击系数
        if liquidity_group and liquidity_group in self.liquidity_group_coefficients:
            eta = self.liquidity_group_coefficients[liquidity_group]
        else:
            eta = self.impact_coefficients[symbol]
        
        # 凹函数冲击: eta * sqrt(POV)
        # Gatheral证明平方根形式满足无动态套利条件
        pov_clamped = max(pov, 1e-6)  # 避免除零
        impact_bps = eta * np.sqrt(pov_clamped) * 10000
        
        # VPIN毒性调整
        vpin_multiplier = 1.0
        if vpin is not None:
            if vpin > 0.8:
                vpin_multiplier = 2.0    # 高毒性环境，冲击翻倍
            elif vpin > 0.6:
                vpin_multiplier = 1.5    # 中等毒性，冲击增加50%
            elif vpin > 0.4:
                vpin_multiplier = 1.2    # 轻微毒性，冲击增加20%
        
        # 总成本
        total_cost_bps = spread_bps + impact_bps * vpin_multiplier + self.fixed_fee_bps
        
        return float(total_cost_bps)
    
    def record_execution(self, symbol: str, side: str, quantity: int,
                        executed_price: float, reference_price: float, 
                        pov: float, spread_bps: float, trade_id: str = None):
        """
        记录执行结果并更新模型
        
        Args:
            symbol: 股票代码
            side: 交易方向
            quantity: 数量
            executed_price: 实际成交价
            reference_price: 决策时参考价
            pov: 参与率
            spread_bps: 点差
            trade_id: 交易ID
        """
        try:
            # 计算实际滑点
            if side.upper() == "BUY":
                slippage_bps = (executed_price - reference_price) / reference_price * 10000
            else:
                slippage_bps = (reference_price - executed_price) / reference_price * 10000
            
            # 净市场冲击 = 总滑点 - 点差 - 固定费用
            market_impact_bps = max(0, slippage_bps - spread_bps - self.fixed_fee_bps)
            
            # 创建执行记录
            record = ExecutionRecord(
                timestamp=time.time(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                executed_price=executed_price,
                reference_price=reference_price,
                pov=pov,
                spread_bps=spread_bps,
                realized_slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                trade_id=trade_id or f"{symbol}_{int(time.time())}"
            )
            
            # 保存到历史
            self.execution_history.append(record)
            
            # 保存到数据库
            self._save_execution_to_db(record)
            
            # 更新冲击系数
            self._update_impact_coefficient(symbol, pov, market_impact_bps)
            
            logger.debug(f"记录执行: {symbol} {side} {quantity}, "
                        f"滑点={slippage_bps:.2f}bps, 净冲击={market_impact_bps:.2f}bps")
            
        except Exception as e:
            logger.error(f"记录执行失败: {e}")
    
    def _save_execution_to_db(self, record: ExecutionRecord):
        """保存执行记录到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO execution_records 
                (timestamp, symbol, side, quantity, executed_price, reference_price,
                 pov, spread_bps, realized_slippage_bps, market_impact_bps, trade_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp, record.symbol, record.side, record.quantity,
                record.executed_price, record.reference_price, record.pov,
                record.spread_bps, record.realized_slippage_bps, 
                record.market_impact_bps, record.trade_id
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存执行记录到数据库失败: {e}")
    
    def _update_impact_coefficient(self, symbol: str, pov: float, 
                                  market_impact_bps: float, learning_rate: float = 0.1):
        """
        使用实际滑点数据更新冲击系数
        
        基于回归: market_impact_bps = eta * sqrt(POV) * 10000
        求解: eta = market_impact_bps / (sqrt(POV) * 10000)
        """
        try:
            if pov <= 1e-6 or market_impact_bps < 0:
                return
            
            # 计算隐含的eta
            sqrt_pov = np.sqrt(pov)
            implied_eta = market_impact_bps / (sqrt_pov * 10000)
            
            # 使用指数加权移动平均更新
            current_eta = self.impact_coefficients[symbol]
            new_eta = (1 - learning_rate) * current_eta + learning_rate * implied_eta
            
            # 限制在合理范围内
            new_eta = np.clip(new_eta, 1e-5, 5e-3)
            self.impact_coefficients[symbol] = float(new_eta)
            
            logger.debug(f"更新{symbol}冲击系数: {current_eta:.6f} -> {new_eta:.6f}")
            
        except Exception as e:
            logger.error(f"更新冲击系数失败: {e}")
    
    def calibrate_by_symbol(self, symbol: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        对特定股票进行回归校准
        
        使用最近N天的执行数据拟合: impact = eta * sqrt(POV)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取最近的执行数据
            cutoff_time = time.time() - lookback_days * 86400
            query = '''
                SELECT pov, market_impact_bps 
                FROM execution_records 
                WHERE symbol = ? AND timestamp >= ? AND market_impact_bps > 0
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql(query, conn, params=(symbol, cutoff_time))
            conn.close()
            
            if len(df) < 5:
                logger.warning(f"{symbol}执行样本不足({len(df)}个)，无法校准")
                return {"success": False, "reason": "样本不足"}
            
            # 准备回归数据
            X = np.sqrt(df['pov'].values).reshape(-1, 1)
            y = df['market_impact_bps'].values / 10000  # 转换回eta单位
            
            # 简单线性回归 (通过原点)
            eta_fitted = np.sum(X.flatten() * y) / np.sum(X.flatten() ** 2)
            
            # 计算R²
            y_pred = eta_fitted * X.flatten()
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 更新系数
            old_eta = self.impact_coefficients[symbol]
            self.impact_coefficients[symbol] = float(np.clip(eta_fitted, 1e-5, 5e-3))
            
            # 保存校准结果
            self._save_calibration_result(symbol, eta_fitted, len(df), r_squared)
            
            result = {
                "success": True,
                "symbol": symbol,
                "old_eta": old_eta,
                "new_eta": eta_fitted,
                "r_squared": r_squared,
                "sample_count": len(df),
                "lookback_days": lookback_days
            }
            
            logger.info(f"{symbol}校准完成: eta={eta_fitted:.6f}, R²={r_squared:.3f}, 样本={len(df)}")
            return result
            
        except Exception as e:
            logger.error(f"{symbol}校准失败: {e}")
            return {"success": False, "reason": str(e)}
    
    def _save_calibration_result(self, symbol: str, eta: float, sample_count: int, r_squared: float):
        """保存校准结果到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO impact_coefficients 
                (symbol, eta, last_calibration_date, sample_count, r_squared)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, eta, datetime.now().date(), sample_count, r_squared))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存校准结果失败: {e}")
    
    def batch_calibrate(self, symbols: List[str] = None, lookback_days: int = 30) -> Dict[str, Any]:
        """批量校准多个股票"""
        if symbols is None:
            # 获取所有有执行记录的股票
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT symbol FROM execution_records')
                symbols = [row[0] for row in cursor.fetchall()]
                conn.close()
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                return {"success": False, "reason": str(e)}
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.calibrate_by_symbol(symbol, lookback_days)
        
        # 统计
        successful = sum(1 for r in results.values() if r.get("success"))
        logger.info(f"批量校准完成: {successful}/{len(symbols)}个股票成功")
        
        return {
            "success": True,
            "total_symbols": len(symbols),
            "successful_calibrations": successful,
            "results": results
        }
    
    def get_impact_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """获取冲击成本统计"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                # 单个股票统计
                query = '''
                    SELECT COUNT(*) as count, 
                           AVG(market_impact_bps) as avg_impact,
                           STDDEV(market_impact_bps) as std_impact,
                           MIN(market_impact_bps) as min_impact,
                           MAX(market_impact_bps) as max_impact,
                           AVG(pov) as avg_pov
                    FROM execution_records 
                    WHERE symbol = ? AND timestamp >= ?
                '''
                cutoff = time.time() - 30 * 86400  # 30天
                df = pd.read_sql(query, conn, params=(symbol, cutoff))
                
                stats = {
                    "symbol": symbol,
                    "eta": self.impact_coefficients[symbol],
                    "executions_30d": int(df.iloc[0]['count']) if not df.empty else 0,
                    "avg_impact_bps": float(df.iloc[0]['avg_impact']) if not df.empty and df.iloc[0]['avg_impact'] else 0,
                    "std_impact_bps": float(df.iloc[0]['std_impact']) if not df.empty and df.iloc[0]['std_impact'] else 0,
                    "avg_pov": float(df.iloc[0]['avg_pov']) if not df.empty and df.iloc[0]['avg_pov'] else 0
                }
            else:
                # 全局统计
                query = '''
                    SELECT symbol, COUNT(*) as count, AVG(market_impact_bps) as avg_impact
                    FROM execution_records 
                    WHERE timestamp >= ?
                    GROUP BY symbol
                    ORDER BY count DESC
                '''
                cutoff = time.time() - 30 * 86400
                df = pd.read_sql(query, conn, params=(cutoff,))
                
                stats = {
                    "total_symbols": len(df),
                    "total_executions": int(df['count'].sum()) if not df.empty else 0,
                    "avg_impact_bps": float(df['avg_impact'].mean()) if not df.empty else 0,
                    "top_symbols": df.head(10).to_dict('records') if not df.empty else []
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            return {"error": str(e)}
    
    def optimize_pov_for_cost_target(self, symbol: str, target_cost_bps: float, 
                                   spread_bps: float, vpin: float = None) -> float:
        """
        为目标成本找到最优参与率
        
        逆向求解: POV = ((target - spread - fee) / (eta * vpin_mult))^2
        """
        try:
            eta = self.impact_coefficients[symbol]
            
            # VPIN调整
            vpin_multiplier = 1.0
            if vpin is not None:
                if vpin > 0.8:
                    vpin_multiplier = 2.0
                elif vpin > 0.6:
                    vpin_multiplier = 1.5
                elif vpin > 0.4:
                    vpin_multiplier = 1.2
            
            # 可用于冲击的成本预算
            available_for_impact = target_cost_bps - spread_bps - self.fixed_fee_bps
            
            if available_for_impact <= 0:
                return 0.001  # 最小参与率
            
            # 求解 POV: impact = eta * sqrt(POV) * 10000 * vpin_mult
            # POV = (impact / (eta * 10000 * vpin_mult))^2
            pov = (available_for_impact / (eta * 10000 * vpin_multiplier)) ** 2
            
            # 限制在合理范围
            pov = np.clip(pov, 0.001, 0.5)
            
            return float(pov)
            
        except Exception as e:
            logger.error(f"优化POV失败: {e}")
            return 0.05  # 默认5%参与率

# 全局实例
_global_impact_model = None

def get_impact_model() -> ImpactModel:
    """获取全局冲击成本模型"""
    global _global_impact_model
    if _global_impact_model is None:
        _global_impact_model = ImpactModel()
    return _global_impact_model

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model = get_impact_model()
    
    # 测试成本估算
    symbol = "AAPL"
    cost = model.estimate_cost_bps(symbol, pov=0.05, spread_bps=1.0, vpin=0.3)
    print(f"估算成本: {cost:.2f}bps")
    
    # 模拟执行记录
    model.record_execution(
        symbol=symbol,
        side="BUY",
        quantity=1000,
        executed_price=150.02,
        reference_price=150.00,
        pov=0.05,
        spread_bps=1.0
    )
    
    # 校准
    result = model.calibrate_by_symbol(symbol)
    print("校准结果:", result)
    
    # 统计
    stats = model.get_impact_statistics(symbol)
    print("统计数据:", stats)