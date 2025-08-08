#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模型融合策略调度器
每周一自动更新Sharpe权重，生成融合信号，并执行交易
"""

import schedule
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import threading
import sys

from ensemble_strategy import EnsembleStrategy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ensemble_scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnsembleScheduler:
    """融合策略调度器"""
    
    def __init__(self, config_file="ensemble_config.json"):
        """
        初始化调度器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.ensemble = EnsembleStrategy()
        self.is_running = False
        self.config = self._load_config()
        
        # 确保日志目录存在
        Path("logs").mkdir(exist_ok=True)
        
        self.logger = logger
        
    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "enabled": True,
            "update_time": "09:00",  # 每周一上午9点
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
            "lookback_weeks": 12,
            "auto_trading": False,  # 是否自动执行交易
            "max_positions": 10,    # 最大持仓数量
            "signal_threshold": 0.6 # 信号阈值
        }
        
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置
                    default_config.update(config)
                    return default_config
            else:
                # 创建默认配置文件
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)
                return default_config
        except Exception as e:
            self.logger.warning(f"[调度器] 加载配置失败: {e}，使用默认配置")
            return default_config
    
    def weekly_update_job(self):
        """周度更新任务"""
        try:
            self.logger.info("[调度器] === 开始周度融合策略更新 ===")
            
            # 检查是否启用
            if not self.config.get("enabled", True):
                self.logger.info("[调度器] 调度器已禁用，跳过更新")
                return
            
            # 检查是否是交易日
            today = datetime.now()
            if today.weekday() != 0:  # 0 = Monday
                # 如果今天不是周一，检查是否是节后第一个交易日
                if not self._is_trading_day(today):
                    self.logger.info("[调度器] 今日非交易日，跳过更新")
                    return
            
            tickers = self.config.get("tickers", self.ensemble._get_default_tickers())
            
            # 1. 更新权重
            self.logger.info("[调度器] 步骤1: 更新Sharpe权重...")
            w_bma, w_lstm = self.ensemble.update_weights(tickers, force_update=True)
            
            self.logger.info(f"[调度器] 权重更新完成: BMA={w_bma:.3f}, LSTM={w_lstm:.3f}")
            
            # 2. 生成融合信号
            self.logger.info("[调度器] 步骤2: 生成融合信号...")
            signals = self.ensemble.generate_ensemble_signals(tickers)
            
            if signals:
                # 保存信号到文件
                signal_file = f"ensemble_signals_{today.strftime('%Y%m%d')}.json"
                with open(signal_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "date": today.strftime('%Y-%m-%d'),
                        "signals": signals,
                        "weights": {"w_bma": w_bma, "w_lstm": w_lstm},
                        "tickers_count": len(signals)
                    }, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"[调度器] 融合信号已保存到: {signal_file}")
                
                # 3. 生成交易建议
                trading_recommendations = self._generate_trading_recommendations(signals)
                self.logger.info(f"[调度器] 生成 {len(trading_recommendations)} 个交易建议")
                
                # 4. 自动交易（如果启用）
                if self.config.get("auto_trading", False):
                    self.logger.info("[调度器] 步骤3: 执行自动交易...")
                    self._execute_auto_trading(trading_recommendations)
                else:
                    self.logger.info("[调度器] 自动交易未启用，仅生成建议")
                    self._save_trading_recommendations(trading_recommendations)
            else:
                self.logger.warning("[调度器] 未生成融合信号，跳过交易步骤")
            
            # 5. 记录完成状态
            self._log_completion_status(w_bma, w_lstm, len(signals) if signals else 0)
            
            self.logger.info("[调度器] === 周度融合策略更新完成 ===")
            
        except Exception as e:
            self.logger.error(f"[调度器] 周度更新失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_trading_recommendations(self, signals: Dict[str, float]) -> List[Dict]:
        """生成交易建议"""
        try:
            threshold = self.config.get("signal_threshold", 0.6)
            max_positions = self.config.get("max_positions", 10)
            
            # 按信号强度排序
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for ticker, signal in sorted_signals[:max_positions]:
                if signal >= threshold:
                    recommendations.append({
                        "ticker": ticker,
                        "signal": signal,
                        "action": "BUY",
                        "confidence": "HIGH" if signal >= 0.8 else "MEDIUM"
                    })
                elif signal <= (1 - threshold):
                    recommendations.append({
                        "ticker": ticker,
                        "signal": signal,
                        "action": "SELL",
                        "confidence": "HIGH" if signal <= 0.2 else "MEDIUM"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"[调度器] 生成交易建议失败: {e}")
            return []
    
    def _save_trading_recommendations(self, recommendations: List[Dict]):
        """保存交易建议"""
        try:
            today = datetime.now()
            rec_file = f"trading_recommendations_{today.strftime('%Y%m%d')}.json"
            
            rec_data = {
                "date": today.strftime('%Y-%m-%d'),
                "time": today.strftime('%H:%M:%S'),
                "recommendations": recommendations,
                "total_count": len(recommendations),
                "buy_count": len([r for r in recommendations if r["action"] == "BUY"]),
                "sell_count": len([r for r in recommendations if r["action"] == "SELL"])
            }
            
            with open(rec_file, 'w', encoding='utf-8') as f:
                json.dump(rec_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[调度器] 交易建议已保存到: {rec_file}")
            
        except Exception as e:
            self.logger.error(f"[调度器] 保存交易建议失败: {e}")
    
    def _execute_auto_trading(self, recommendations: List[Dict]):
        """执行自动交易（预留接口）"""
        try:
            self.logger.info("[调度器] 自动交易功能暂未实现，仅记录建议")
            
            # 这里可以接入实际的交易系统
            # 比如调用 quantitative_trading_manager.py 的交易功能
            
            for rec in recommendations:
                self.logger.info(f"[调度器] 交易建议: {rec['action']} {rec['ticker']} "
                               f"(信号: {rec['signal']:.3f}, 置信度: {rec['confidence']})")
            
            # 保存交易建议
            self._save_trading_recommendations(recommendations)
            
        except Exception as e:
            self.logger.error(f"[调度器] 执行自动交易失败: {e}")
    
    def _is_trading_day(self, date: datetime) -> bool:
        """检查是否是交易日"""
        # 简单实现：周一到周五
        # 实际应用中可以考虑节假日
        return date.weekday() < 5
    
    def _log_completion_status(self, w_bma: float, w_lstm: float, signals_count: int):
        """记录完成状态"""
        try:
            status = {
                "completion_time": datetime.now().isoformat(),
                "weights": {"w_bma": w_bma, "w_lstm": w_lstm},
                "signals_generated": signals_count,
                "status": "SUCCESS"
            }
            
            with open("last_ensemble_update.json", 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"[调度器] 记录状态失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        try:
            self.logger.info("[调度器] 启动双模型融合策略调度器")
            
            # 解析更新时间
            update_time = self.config.get("update_time", "09:00")
            
            # 注册周一定时任务
            schedule.every().monday.at(update_time).do(self.weekly_update_job)
            
            # 也可以注册每日检查任务（处理节假日情况）
            schedule.every().day.at(update_time).do(self._daily_check)
            
            self.logger.info(f"[调度器] 已注册周一 {update_time} 的定时任务")
            
            self.is_running = True
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            self.logger.info("[调度器] 收到停止信号，正在关闭...")
            self.stop_scheduler()
        except Exception as e:
            self.logger.error(f"[调度器] 调度器运行失败: {e}")
    
    def _daily_check(self):
        """每日检查任务"""
        today = datetime.now()
        
        # 只在周一或节假日后第一个交易日执行
        if today.weekday() == 0 or self._should_run_today():
            self.weekly_update_job()
    
    def _should_run_today(self) -> bool:
        """判断今天是否应该运行更新"""
        try:
            # 检查上次更新时间
            if Path("last_ensemble_update.json").exists():
                with open("last_ensemble_update.json", 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    last_update = datetime.fromisoformat(status["completion_time"])
                    
                    # 如果距离上次更新超过7天，且今天是交易日，则运行
                    days_since_update = (datetime.now() - last_update).days
                    return days_since_update >= 7 and self._is_trading_day(datetime.now())
            else:
                # 如果没有记录，且今天是交易日，则运行
                return self._is_trading_day(datetime.now())
                
        except Exception:
            return False
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        self.logger.info("[调度器] 调度器已停止")
    
    def run_once(self):
        """手动运行一次更新"""
        self.logger.info("[调度器] 手动执行一次融合策略更新")
        self.weekly_update_job()


def run_scheduler_daemon():
    """以守护进程模式运行调度器"""
    scheduler = EnsembleScheduler()
    
    def signal_handler(signum, frame):
        logger.info(f"[调度器] 收到信号 {signum}，正在关闭...")
        scheduler.stop_scheduler()
        sys.exit(0)
    
    # 注册信号处理
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动调度器
    scheduler.start_scheduler()


def test_scheduler():
    """测试调度器"""
    logger.info("=== 测试融合策略调度器 ===")
    
    scheduler = EnsembleScheduler()
    
    # 手动运行一次
    scheduler.run_once()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "daemon":
            run_scheduler_daemon()
        elif sys.argv[1] == "test":
            test_scheduler()
        else:
            print("用法: python ensemble_scheduler.py [daemon|test]")
    else:
        # 默认运行测试
        test_scheduler()