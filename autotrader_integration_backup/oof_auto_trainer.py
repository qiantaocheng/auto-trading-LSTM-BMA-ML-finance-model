#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOF自动训练模块 - 启动时基于投资股票池自动校准IsotonicRegression

功能:
1. 启动时自动获取投资股票池
2. 为股票池中的股票自动训练OOF校准器
3. 定期更新和重新校准
4. 监控校准器性能和质量
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set, Any
from pathlib import Path

from .oof_calibration import get_oof_calibrator
from .database import StockDatabase

logger = logging.getLogger(__name__)

class OOFAutoTrainer:
    """
    OOF自动训练器
    
    核心功能:
    1. 获取当前投资股票池
    2. 自动训练IsotonicRegression校准器
    3. 定期重新校准和性能监控
    4. 质量控制和异常处理
    """
    
    def __init__(self, db_path: str = "trading_database.db"):
        self.db_path = db_path
        self.oof_calibrator = get_oof_calibrator()
        
        # 训练配置
        self.min_samples_for_training = 50  # 最少样本数
        self.retrain_interval_hours = 24    # 24小时重新训练
        self.last_training_time = {}        # symbol -> timestamp
        
        # 性能监控
        self.training_stats = {}
        self.training_history = []
        
        logger.info("OOF自动训练器初始化完成")
    
    def get_investment_universe(self) -> List[str]:
        """获取当前投资股票池"""
        try:
            # 方法1: 从数据库获取活跃交易的股票
            stock_db = StockDatabase(db_path=self.db_path)
            
            # 获取最近30天有交易记录的股票
            cutoff_date = datetime.now() - timedelta(days=30)
            
            try:
                # 尝试从StockDatabase获取股票列表
                active_symbols = set(stock_db.get_all_tickers())
            except Exception as e:
                logger.warning(f"从数据库获取股票失败: {e}")
                active_symbols = set()
                
            # 如果没有找到活跃股票，使用默认股票池
            if not active_symbols:
                logger.warning("未找到活跃交易股票，使用默认股票池")
                active_symbols = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'}
            
            result = list(active_symbols)[:50]  # 限制最多50只股票
            logger.info(f"获取投资股票池: {len(result)}只股票")
            return result
                
        except Exception as e:
            logger.error(f"获取投资股票池失败: {e}")
            # 返回默认股票池
            # 🔒 移除硬编码，返回安全的默认值
            return ['SPY']  # 使用ETF作为安全默认值
    
    async def auto_train_on_startup(self, max_concurrent: int = 5):
        """启动时自动训练所有投资股票的OOF校准器"""
        logger.info("🚀 开始启动时OOF自动训练...")
        
        # 获取投资股票池
        universe = self.get_investment_universe()
        if not universe:
            logger.warning("投资股票池为空，跳过OOF训练")
            return
        
        # 并行训练
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def train_single_symbol(symbol: str):
            async with semaphore:
                try:
                    result = await self._train_symbol_async(symbol)
                    if result.get('success'):
                        logger.info(f"✅ {symbol} OOF训练成功: "
                                  f"R²={result.get('r_squared', 0):.3f}, "
                                  f"样本={result.get('sample_count', 0)}")
                    else:
                        logger.warning(f"⚠️ {symbol} OOF训练失败: {result.get('reason', 'Unknown')}")
                    return result
                except Exception as e:
                    logger.error(f"❌ {symbol} OOF训练异常: {e}")
                    return {'success': False, 'reason': str(e)}
        
        # 执行并行训练
        start_time = time.time()
        tasks = [train_single_symbol(symbol) for symbol in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        total_time = time.time() - start_time
        
        logger.info(f"🎯 OOF启动训练完成: {successful}/{len(universe)}个股票成功, "
                   f"耗时{total_time:.1f}秒")
        
        # 保存训练统计
        self.training_history.append({
            'timestamp': datetime.now(),
            'type': 'startup_training',
            'total_symbols': len(universe),
            'successful_symbols': successful,
            'duration_seconds': total_time,
            'symbols': universe
        })
        
        return successful > 0
    
    async def _train_symbol_async(self, symbol: str) -> Dict[str, Any]:
        """异步训练单个股票的OOF校准器"""
        try:
            # 在线程池中执行同步训练
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.oof_calibrator.calibrate_by_symbol,
                symbol,
                30  # 30天回望期
            )
            
            # 记录训练时间
            self.last_training_time[symbol] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"异步训练{symbol}失败: {e}")
            return {'success': False, 'reason': str(e)}
    
    def schedule_periodic_retraining(self):
        """安排定期重新训练"""
        async def periodic_trainer():
            while True:
                try:
                    await asyncio.sleep(900)  # 每15分钟检查一次（优化：从1小时缩短）
                    
                    # 检查哪些股票需要重新训练
                    universe = self.get_investment_universe()
                    current_time = time.time()
                    
                    symbols_to_retrain = []
                    for symbol in universe:
                        last_training = self.last_training_time.get(symbol, 0)
                        hours_since_training = (current_time - last_training) / 3600
                        
                        if hours_since_training >= self.retrain_interval_hours:
                            symbols_to_retrain.append(symbol)
                    
                    if symbols_to_retrain:
                        logger.info(f"⏰ 定期重新训练: {len(symbols_to_retrain)}只股票")
                        
                        # 异步重新训练
                        tasks = [self._train_symbol_async(symbol) for symbol in symbols_to_retrain]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
                        logger.info(f"✅ 定期重新训练完成: {successful}/{len(symbols_to_retrain)}个成功")
                    
                except Exception as e:
                    logger.error(f"定期重新训练出错: {e}")
                    await asyncio.sleep(30)  # 出错后等待30秒再继续（优化：从5分钟缩短）
        
        # 在后台启动定期训练任务
        asyncio.create_task(periodic_trainer())
        logger.info("📅 定期重新训练任务已启动")
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态摘要"""
        universe = self.get_investment_universe()
        current_time = time.time()
        
        # 统计训练状态
        trained_count = 0
        outdated_count = 0
        never_trained_count = 0
        
        for symbol in universe:
            if symbol in self.last_training_time:
                hours_since = (current_time - self.last_training_time[symbol]) / 3600
                if hours_since < self.retrain_interval_hours:
                    trained_count += 1
                else:
                    outdated_count += 1
            else:
                never_trained_count += 1
        
        # 获取校准器统计
        calibrator_stats = self.oof_calibrator.get_calibrator_stats()
        
        return {
            'universe_size': len(universe),
            'trained_symbols': trained_count,
            'outdated_symbols': outdated_count,
            'never_trained_symbols': never_trained_count,
            'training_coverage': trained_count / len(universe) if universe else 0,
            'calibrator_available': calibrator_stats.get('calibrator_available', False),
            'total_training_sessions': len(self.training_history),
            'last_startup_training': self.training_history[-1] if self.training_history else None,
            'retrain_interval_hours': self.retrain_interval_hours
        }
    
    def force_retrain_all(self) -> bool:
        """强制重新训练所有股票"""
        try:
            universe = self.get_investment_universe()
            
            # 批量训练
            results = self.oof_calibrator.batch_calibrate(universe, lookback_days=30)
            
            if results.get('success'):
                successful = results.get('successful_calibrations', 0)
                total = results.get('total_symbols', 0)
                
                # 更新训练时间
                current_time = time.time()
                for symbol in universe:
                    self.last_training_time[symbol] = current_time
                
                logger.info(f"🔄 强制重新训练完成: {successful}/{total}个股票成功")
                return successful > 0
            else:
                logger.error(f"强制重新训练失败: {results.get('reason', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"强制重新训练异常: {e}")
            return False

# 全局实例
_global_oof_trainer = None

def get_oof_auto_trainer() -> OOFAutoTrainer:
    """获取全局OOF自动训练器"""
    global _global_oof_trainer
    if _global_oof_trainer is None:
        _global_oof_trainer = OOFAutoTrainer()
    return _global_oof_trainer

async def startup_oof_training():
    """启动时执行OOF训练 - 供外部调用"""
    trainer = get_oof_auto_trainer()
    success = await trainer.auto_train_on_startup()
    
    if success:
        # 启动定期重新训练
        trainer.schedule_periodic_retraining()
    
    return success

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test_oof_training():
        trainer = get_oof_auto_trainer()
        
        # 测试获取股票池
        universe = trainer.get_investment_universe()
        print(f"投资股票池: {universe}")
        
        # 测试启动训练
        success = await trainer.auto_train_on_startup()
        print(f"启动训练成功: {success}")
        
        # 查看状态
        status = trainer.get_training_status()
        print(f"训练状态: {status}")
    
    asyncio.run(test_oof_training())