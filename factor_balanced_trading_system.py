#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子平衡交易系统 (FactorBalancedTradingSystem)
完整集成的四象限市况判别+因子平衡策略系统

核心功能:
1. 盘前四象限市况判别（ADX/ATR/SMA）
2. A/B双策略并行管理（均值回归+趋势跟踪）
3. 动态配比调整（基础权重+表现权重）
4. 定期再平衡（周五收盘后+盘前调整）
5. 风险控制和冷静期管理
6. 动态加仓机制

Authors: AI Assistant
Version: 1.0
"""

import asyncio
import logging
import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# 导入所有组件
from regime_detector import RegimeDetector
from allocation_manager import AllocationManager, AllocationConfig
from dual_strategy_system import DualStrategySystem
from rebalancing_scheduler import RebalancingScheduler

# 导入增强版交易策略
try:
    from ibkr_trading_strategy_enhanced import EnhancedMeanReversionStrategy
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("[WARNING] IBKR策略不可用，将使用模拟模式")

# 导入动态仓位管理系统
try:
    from dynamic_position_sizing import DynamicPositionSizing, DynamicSizingConfig, TradeResult
    DYNAMIC_SIZING_AVAILABLE = True
except ImportError:
    DYNAMIC_SIZING_AVAILABLE = False
    print("[WARNING] 动态仓位管理系统不可用")

@dataclass
class SystemConfig:
    """系统配置数据类"""
    # 基础配置
    total_capital: float = 100000.0
    enable_real_trading: bool = False
    
    # IBKR配置
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 4002
    ibkr_client_id: int = 50310
    ibkr_account: str = ""
    
    # 风险配置
    max_position_size: float = 0.05
    max_portfolio_exposure: float = 0.95
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.06
    daily_loss_limit: float = 0.02
    drawdown_limit: float = 0.10
    
    # 市况判别配置
    adx_threshold: float = 25.0
    atr_ratio_threshold: float = 0.8
    regime_lookback_days: int = 60
    
    # 配比管理配置
    sharpe_blend_ratio: float = 0.5
    performance_lookback: int = 60
    cooldown_days: int = 5
    
    # 动态加仓配置
    dynamic_boost_threshold: float = 1.0
    dynamic_boost_amount: float = 0.05
    max_allocation_adjustment: float = 0.20
    
    # 动态仓位管理配置
    enable_dynamic_sizing: bool = True
    dynamic_base_risk: float = 0.02
    dynamic_max_exposure: float = 0.08
    dynamic_win_streak_trigger: int = 3
    dynamic_addon_aggressive: float = 0.2

class FactorBalancedTradingSystem:
    """因子平衡交易系统主类"""
    
    def __init__(self, config: SystemConfig = None):
        """
        初始化因子平衡交易系统
        
        Args:
            config: 系统配置
        """
        self.config = config or SystemConfig()
        self.logger = logging.getLogger(__name__)
        
        # 设置日志
        self._setup_logging()
        
        # 组件初始化
        self.regime_detector = None
        self.allocation_manager = None
        self.dual_strategy_system = None
        self.rebalancing_scheduler = None
        self.dynamic_position_sizing = None
        
        # 系统状态
        self.system_running = False
        self.initialization_complete = False
        
        # 历史记录
        self.trading_history = []
        self.system_events = []
        
        # 线程管理
        self.background_tasks = []
        
        self.logger.info("因子平衡交易系统初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_dir = "factor_balanced_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    async def initialize_system(self):
        """初始化系统组件"""
        try:
            self.logger.info("🚀 开始初始化因子平衡交易系统")
            
            # 1. 初始化市况检测器
            self.logger.info("初始化市况检测器...")
            self.regime_detector = RegimeDetector(
                lookback_days=self.config.regime_lookback_days,
                adx_threshold=self.config.adx_threshold,
                atr_ratio_threshold=self.config.atr_ratio_threshold
            )
            
            # 2. 初始化配比管理器
            self.logger.info("初始化配比管理器...")
            self.allocation_manager = AllocationManager(
                performance_lookback=self.config.performance_lookback,
                sharpe_blend_ratio=self.config.sharpe_blend_ratio
            )
            
            # 3. 初始化双策略系统
            self.logger.info("初始化双策略系统...")
            dual_system_config = {
                'total_capital': self.config.total_capital,
                'enable_enhanced_mode': True,
                'enable_real_trading': self.config.enable_real_trading,
                'ibkr_host': self.config.ibkr_host,
                'ibkr_port': self.config.ibkr_port,
                'ibkr_client_id': self.config.ibkr_client_id,
                'ibkr_account': self.config.ibkr_account,
                'max_position_size': self.config.max_position_size,
                'max_portfolio_exposure': self.config.max_portfolio_exposure,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct
            }
            
            self.dual_strategy_system = DualStrategySystem(dual_system_config)
            
            # 4. 初始化再平衡调度器
            self.logger.info("初始化再平衡调度器...")
            scheduler_config = {
                'cooldown_days': self.config.cooldown_days,
                'daily_loss_limit': self.config.daily_loss_limit,
                'drawdown_limit': self.config.drawdown_limit
            }
            
            self.rebalancing_scheduler = RebalancingScheduler(
                self.dual_strategy_system, 
                scheduler_config
            )
            
            # 5. 初始化动态仓位管理系统
            if self.config.enable_dynamic_sizing and DYNAMIC_SIZING_AVAILABLE:
                self.logger.info("初始化动态仓位管理系统...")
                dynamic_config = DynamicSizingConfig(
                    base_risk_pct=self.config.dynamic_base_risk,
                    max_exposure_pct=self.config.dynamic_max_exposure,
                    win_streak_trigger=self.config.dynamic_win_streak_trigger,
                    addon_aggressive_factor=self.config.dynamic_addon_aggressive
                )
                self.dynamic_position_sizing = DynamicPositionSizing(dynamic_config)
                self.logger.info("✅ 动态仓位管理系统初始化完成")
            else:
                self.logger.warning("动态仓位管理系统未启用或不可用")
            
            # 6. 执行初始市况检测和配比设置
            await self._initial_allocation_setup()
            
            self.initialization_complete = True
            self.logger.info("✅ 因子平衡交易系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    async def _initial_allocation_setup(self):
        """初始配比设置"""
        try:
            self.logger.info("执行初始配比设置...")
            
            # 1. 检测当前市况
            regime_result = self.regime_detector.detect_regime(force_update=True)
            
            # 2. 生成初始配比
            allocation_config = self.allocation_manager.generate_allocation_config(regime_result)
            
            # 3. 应用配比到双策略系统
            await self.dual_strategy_system._apply_allocation(allocation_config)
            
            # 4. 保存配比配置
            self.allocation_manager.save_allocation_config(allocation_config)
            
            self.logger.info(f"初始配比设置完成: A={allocation_config.final_allocation_A:.1%}, B={allocation_config.final_allocation_B:.1%}")
            self.logger.info(f"市况类型: {regime_result['regime_type']['description']}")
            
        except Exception as e:
            self.logger.error(f"初始配比设置失败: {e}")
            raise
    
    async def start_system(self):
        """启动交易系统"""
        try:
            if not self.initialization_complete:
                await self.initialize_system()
            
            self.logger.info("🚀 启动因子平衡交易系统")
            
            # 1. 启动双策略系统
            await self.dual_strategy_system.start_system()
            
            # 2. 启动再平衡调度器
            self.rebalancing_scheduler.start_scheduler()
            
            # 3. 启动系统监控
            await self._start_system_monitoring()
            
            self.system_running = True
            
            # 记录系统启动事件
            self._log_system_event("system_started", {"timestamp": datetime.now().isoformat()})
            
            self.logger.info("✅ 因子平衡交易系统已启动")
            
        except Exception as e:
            self.logger.error(f"启动交易系统失败: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """停止交易系统"""
        try:
            self.logger.info("🛑 停止因子平衡交易系统")
            
            self.system_running = False
            
            # 1. 停止再平衡调度器
            if self.rebalancing_scheduler:
                self.rebalancing_scheduler.stop_scheduler()
            
            # 2. 停止双策略系统
            if self.dual_strategy_system:
                await self.dual_strategy_system.stop_system()
            
            # 3. 停止系统监控
            await self._stop_system_monitoring()
            
            # 记录系统停止事件
            self._log_system_event("system_stopped", {"timestamp": datetime.now().isoformat()})
            
            self.logger.info("✅ 因子平衡交易系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易系统失败: {e}")
    
    async def _start_system_monitoring(self):
        """启动系统监控"""
        try:
            # 创建监控任务
            monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            self.background_tasks.append(monitoring_task)
            
            self.logger.info("系统监控已启动")
            
        except Exception as e:
            self.logger.error(f"启动系统监控失败: {e}")
    
    async def _stop_system_monitoring(self):
        """停止系统监控"""
        try:
            # 取消所有后台任务
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务完成
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            self.logger.info("系统监控已停止")
            
        except Exception as e:
            self.logger.error(f"停止系统监控失败: {e}")
    
    async def _system_monitoring_loop(self):
        """系统监控主循环"""
        self.logger.info("系统监控循环启动")
        
        while self.system_running:
            try:
                # 1. 收集系统状态
                await self._collect_system_status()
                
                # 2. 保存系统状态
                await self._save_system_state()
                
                # 3. 检查系统健康状态
                await self._check_system_health()
                
                # 监控间隔
                await asyncio.sleep(300)  # 5分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"系统监控循环错误: {e}")
                await asyncio.sleep(600)  # 错误后等待10分钟
        
        self.logger.info("系统监控循环结束")
    
    async def _collect_system_status(self):
        """收集系统状态"""
        try:
            # 收集各组件状态
            system_status = {
                'timestamp': datetime.now().isoformat(),
                'system_running': self.system_running,
                'dual_strategy_status': self.dual_strategy_system.get_system_status() if self.dual_strategy_system else {},
                'scheduler_status': self.rebalancing_scheduler.get_scheduler_status() if self.rebalancing_scheduler else {},
                'regime_info': self.regime_detector.current_regime if self.regime_detector else {},
                'allocation_info': {
                    'current_allocation': self.allocation_manager.get_current_allocation() if self.allocation_manager else (0.5, 0.5),
                    'last_update': datetime.now().isoformat()
                }
            }
            
            # 保存状态
            self.current_system_status = system_status
            
        except Exception as e:
            self.logger.error(f"收集系统状态失败: {e}")
    
    async def _save_system_state(self):
        """保存系统状态"""
        try:
            state_file = "factor_balanced_system_state.json"
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_system_status, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
    
    async def _check_system_health(self):
        """检查系统健康状态"""
        try:
            # 检查各组件状态
            if self.dual_strategy_system:
                dual_status = self.dual_strategy_system.get_system_status()
                
                # 检查策略健康状态
                active_strategies = dual_status.get('active_strategies', 0)
                total_strategies = dual_status.get('strategy_count', 0)
                
                if active_strategies < total_strategies:
                    self.logger.warning(f"策略健康检查: {active_strategies}/{total_strategies} 策略活跃")
            
            # 检查调度器状态
            if self.rebalancing_scheduler:
                scheduler_status = self.rebalancing_scheduler.get_scheduler_status()
                
                if scheduler_status.get('status') != 'running':
                    self.logger.warning(f"调度器状态异常: {scheduler_status.get('status')}")
            
        except Exception as e:
            self.logger.error(f"系统健康检查失败: {e}")
    
    def _log_system_event(self, event_type: str, event_data: Dict):
        """记录系统事件"""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'data': event_data
            }
            
            self.system_events.append(event)
            
            # 保存事件日志
            events_file = "factor_balanced_events.json"
            
            events = []
            if os.path.exists(events_file):
                try:
                    with open(events_file, 'r', encoding='utf-8') as f:
                        events = json.load(f)
                except:
                    events = []
            
            events.append(event)
            
            # 只保留最近1000个事件
            if len(events) > 1000:
                events = events[-1000:]
            
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"记录系统事件失败: {e}")
    
    def get_system_summary(self) -> Dict:
        """获取系统摘要"""
        try:
            summary = {
                'system_info': {
                    'running': self.system_running,
                    'initialized': self.initialization_complete,
                    'start_time': getattr(self, 'start_time', None),
                    'uptime_hours': self._calculate_uptime()
                },
                'capital_info': {
                    'total_capital': self.config.total_capital,
                    'current_pnl': 0,  # TODO: 从双策略系统获取
                    'daily_pnl': 0     # TODO: 从双策略系统获取
                },
                'allocation_info': {
                    'current_allocation': self.allocation_manager.get_current_allocation() if self.allocation_manager else (0.5, 0.5),
                    'regime_type': self.regime_detector.current_regime.get('regime_type', {}).get('description', 'Unknown') if self.regime_detector and self.regime_detector.current_regime else 'Unknown'
                },
                'risk_info': {
                    'cooldown_active': self.rebalancing_scheduler.cooldown_active if self.rebalancing_scheduler else False,
                    'max_position_size': self.config.max_position_size,
                    'daily_loss_limit': self.config.daily_loss_limit,
                    'drawdown_limit': self.config.drawdown_limit
                }
            }
            
            # 添加动态仓位管理状态
            if self.dynamic_position_sizing:
                dps_status = self.dynamic_position_sizing.get_system_status()
                summary['dynamic_sizing'] = {
                    'enabled': True,
                    'win_streak': dps_status['win_streak'],
                    'loss_streak': dps_status['loss_streak'],
                    'trading_state': dps_status['trading_state'],
                    'current_exposure': dps_status['current_exposure'],
                    'is_in_cooldown': dps_status['is_in_cooldown'],
                    'performance_stats': dps_status['performance_stats']
                }
            else:
                summary['dynamic_sizing'] = {'enabled': False}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取系统摘要失败: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self) -> float:
        """计算系统运行时间（小时）"""
        if hasattr(self, 'start_time') and self.start_time:
            uptime = datetime.now() - self.start_time
            return uptime.total_seconds() / 3600
        return 0.0
    
    async def execute_manual_rebalancing(self, reason: str = "手动触发") -> bool:
        """手动执行再平衡"""
        try:
            self.logger.info(f"手动执行再平衡: {reason}")
            
            if self.rebalancing_scheduler:
                success = await self.rebalancing_scheduler._execute_rebalancing(reason)
                
                if success:
                    self.logger.info("手动再平衡执行成功")
                    self._log_system_event("manual_rebalancing", {"reason": reason, "success": True})
                else:
                    self.logger.error("手动再平衡执行失败")
                    self._log_system_event("manual_rebalancing", {"reason": reason, "success": False})
                
                return success
            else:
                self.logger.error("再平衡调度器未初始化")
                return False
                
        except Exception as e:
            self.logger.error(f"手动再平衡失败: {e}")
            return False
    
    def force_cooldown(self, reason: str = "手动触发"):
        """强制触发冷静期"""
        try:
            if self.rebalancing_scheduler:
                self.rebalancing_scheduler._trigger_cooldown(reason)
                self.logger.info(f"已强制触发冷静期: {reason}")
                self._log_system_event("force_cooldown", {"reason": reason})
            else:
                self.logger.error("再平衡调度器未初始化")
                
        except Exception as e:
            self.logger.error(f"强制触发冷静期失败: {e}")
    
    def record_trade_execution(self, symbol: str, action: str, quantity: int, 
                             price: float, position_id: str, pnl: float = 0.0, 
                             position_size_pct: float = 0.0, is_addon: bool = False):
        """记录交易执行到动态仓位管理系统"""
        try:
            if not self.dynamic_position_sizing:
                return
            
            trade_result = TradeResult(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                position_id=position_id,
                entry_price=price if action in ["OPEN", "ADD"] else None,
                exit_price=price if action == "CLOSE" else None,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=pnl / (price * quantity) if price * quantity > 0 else 0.0,
                position_size_pct=position_size_pct,
                is_addon=is_addon,
                reason=f"系统执行-{action}"
            )
            
            self.dynamic_position_sizing.record_trade_result(trade_result)
            self.logger.info(f"交易记录已更新到动态仓位系统: {symbol} {action}")
            
        except Exception as e:
            self.logger.error(f"记录交易执行失败: {e}")
    
    def get_dynamic_position_size(self, symbol: str, signal_strength: float = 1.0, 
                                 is_addon: bool = False, base_position_id: str = None) -> float:
        """获取动态仓位大小"""
        try:
            if not self.dynamic_position_sizing:
                return self.config.max_position_size  # 使用默认仓位
            
            return self.dynamic_position_sizing.calculate_position_size(
                symbol, signal_strength, is_addon, base_position_id
            )
            
        except Exception as e:
            self.logger.error(f"获取动态仓位大小失败: {e}")
            return self.config.max_position_size * 0.5  # 返回安全的小仓位
    
    def check_addon_opportunity(self, position_id: str, current_price: float, 
                               market_data: Dict = None) -> Dict:
        """检查加仓机会"""
        try:
            if not self.dynamic_position_sizing:
                return {'action': 'HOLD', 'reason': '动态仓位管理未启用'}
            
            return self.dynamic_position_sizing.get_position_management_signal(
                position_id, current_price, market_data
            )
            
        except Exception as e:
            self.logger.error(f"检查加仓机会失败: {e}")
            return {'action': 'HOLD', 'reason': f'错误: {e}'}

# 全局实例（单例模式）
_factor_balanced_system = None

def get_factor_balanced_system(config: SystemConfig = None) -> FactorBalancedTradingSystem:
    """获取因子平衡交易系统实例（单例模式）"""
    global _factor_balanced_system
    
    if _factor_balanced_system is None:
        _factor_balanced_system = FactorBalancedTradingSystem(config)
    
    return _factor_balanced_system

async def main():
    """主函数 - 系统测试"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 因子平衡交易系统测试")
    print("=" * 50)
    
    # 创建测试配置
    config = SystemConfig(
        total_capital=100000.0,
        enable_real_trading=False,  # 测试模式
        adx_threshold=25.0,
        atr_ratio_threshold=0.8,
        cooldown_days=1  # 测试用短冷静期
    )
    
    # 创建系统实例
    system = FactorBalancedTradingSystem(config)
    
    try:
        # 启动系统
        print("启动系统...")
        await system.start_system()
        
        # 显示系统摘要
        summary = system.get_system_summary()
        print("\n📊 系统摘要:")
        print(f"系统状态: {'运行中' if summary['system_info']['running'] else '已停止'}")
        print(f"总资金: ${summary['capital_info']['total_capital']:,.2f}")
        print(f"当前配比: A={summary['allocation_info']['current_allocation'][0]:.1%}, B={summary['allocation_info']['current_allocation'][1]:.1%}")
        print(f"市况类型: {summary['allocation_info']['regime_type']}")
        print(f"冷静期: {'是' if summary['risk_info']['cooldown_active'] else '否'}")
        
        # 运行系统一段时间
        print("\n🔄 系统运行中...")
        for i in range(6):  # 运行6次，每次30秒
            await asyncio.sleep(30)
            summary = system.get_system_summary()
            print(f"运行状态检查 {i+1}: 配比 A={summary['allocation_info']['current_allocation'][0]:.1%}, B={summary['allocation_info']['current_allocation'][1]:.1%}")
        
        # 测试手动再平衡
        print("\n🔧 测试手动再平衡...")
        success = await system.execute_manual_rebalancing("测试手动触发")
        print(f"手动再平衡结果: {'成功' if success else '失败'}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试出现错误: {e}")
    finally:
        # 停止系统
        print("\n🛑 停止系统...")
        await system.stop_system()
        print("✅ 因子平衡交易系统测试完成")

if __name__ == "__main__":
    asyncio.run(main())