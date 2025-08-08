#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金配比管理器 (AllocationManager)
实现因子平衡策略的资金配比管理

核心功能:
1. 根据四象限字典和月度Sharpe回测结果管理资金配比
2. 生成和管理parameters.json配置文件
3. 月度策略表现评估和权重调整
4. 动态加仓和冷静期管理
5. 版本记录和历史跟踪

Authors: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import shutil
from pathlib import Path

@dataclass
class StrategyPerformance:
    """策略表现数据类"""
    strategy_name: str
    returns: List[float]
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_return_per_trade: float
    volatility: float
    
@dataclass
class AllocationConfig:
    """配比配置数据类"""
    timestamp: str
    regime_type: str
    base_allocation_A: float  # 均值回归策略基础配比
    base_allocation_B: float  # 趋势跟踪策略基础配比
    performance_weight_A: float  # 基于表现的权重A
    performance_weight_B: float  # 基于表现的权重B
    final_allocation_A: float  # 最终配比A
    final_allocation_B: float  # 最终配比B
    total_capital: float
    max_position_size: float
    cooldown_active: bool
    cooldown_reason: str
    version: str

class AllocationManager:
    """资金配比管理器"""
    
    def __init__(self, 
                 config_dir: str = "allocation_config",
                 history_dir: str = "allocation_history",
                 performance_lookback: int = 60,
                 sharpe_blend_ratio: float = 0.5):
        """
        初始化配比管理器
        
        Args:
            config_dir: 配置文件目录
            history_dir: 历史记录目录
            performance_lookback: 表现回看天数
            sharpe_blend_ratio: Sharpe权重与基础权重的混合比例
        """
        self.config_dir = config_dir
        self.history_dir = history_dir
        self.performance_lookback = performance_lookback
        self.sharpe_blend_ratio = sharpe_blend_ratio
        
        # 创建目录
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 配置文件路径
        self.parameters_file = os.path.join(config_dir, "parameters.json")
        self.performance_file = os.path.join(config_dir, "performance_history.json")
        self.allocation_history_file = os.path.join(history_dir, "allocation_history.json")
        
        # 默认配置
        self.default_config = {
            'total_capital': 100000.0,
            'max_position_size': 0.05,  # 单个仓位最大5%
            'max_portfolio_exposure': 0.95,  # 最大总仓位95%
            'stop_loss_pct': 0.02,  # 2%止损
            'take_profit_pct': 0.06,  # 6%止盈
            'daily_loss_limit': 0.02,  # 日内2%亏损限制
            'drawdown_limit': 0.10,  # 10%回撤限制
            'cooldown_days': 5,  # 冷静期天数
            'dynamic_boost_threshold': 1.0,  # 动态加仓Sharpe阈值
            'dynamic_boost_amount': 0.05,  # 动态加仓幅度5%
            'max_allocation_adjustment': 0.20  # 最大配比调整幅度20%
        }
        
        # 加载现有配置
        self.current_config = self._load_current_config()
    
    def _load_current_config(self) -> Optional[AllocationConfig]:
        """加载当前配置"""
        try:
            if os.path.exists(self.parameters_file):
                with open(self.parameters_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 过滤出AllocationConfig需要的字段
                    config_fields = {
                        'timestamp', 'regime_type', 'base_allocation_A', 'base_allocation_B',
                        'performance_weight_A', 'performance_weight_B', 'final_allocation_A',
                        'final_allocation_B', 'total_capital', 'max_position_size',
                        'cooldown_active', 'cooldown_reason', 'version'
                    }
                    filtered_data = {k: v for k, v in data.items() if k in config_fields}
                    return AllocationConfig(**filtered_data)
        except Exception as e:
            self.logger.warning(f"加载配置失败: {e}")
        return None
    
    def calculate_strategy_performance(self, 
                                     strategy_name: str, 
                                     trades_data: List[Dict],
                                     lookback_days: int = None) -> StrategyPerformance:
        """
        计算策略表现
        
        Args:
            strategy_name: 策略名称
            trades_data: 交易数据列表
            lookback_days: 回看天数
            
        Returns:
            策略表现对象
        """
        if lookback_days is None:
            lookback_days = self.performance_lookback
        
        # 过滤时间范围内的交易
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [
            trade for trade in trades_data 
            if datetime.fromisoformat(trade.get('timestamp', '1970-01-01')) >= cutoff_date
        ]
        
        if not recent_trades:
            # 返回默认表现
            return StrategyPerformance(
                strategy_name=strategy_name,
                returns=[],
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_return_per_trade=0.0,
                volatility=0.0
            )
        
        # 计算收益率
        returns = [trade.get('pnl_pct', 0.0) for trade in recent_trades]
        
        # 计算各项指标
        total_trades = len(returns)
        win_trades = [r for r in returns if r > 0]
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_return = np.mean(returns) if returns else 0.0
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # 计算Sharpe比率 (假设无风险利率为0)
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        
        # 计算最大回撤
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            returns=returns,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_return_per_trade=avg_return,
            volatility=volatility
        )
    
    def calculate_performance_weights(self, 
                                    perf_A: StrategyPerformance, 
                                    perf_B: StrategyPerformance) -> Tuple[float, float]:
        """
        基于表现计算权重
        
        Args:
            perf_A: 策略A表现
            perf_B: 策略B表现
            
        Returns:
            (权重A, 权重B)
        """
        sharpe_A = max(perf_A.sharpe_ratio, 0.01)  # 避免负数或零
        sharpe_B = max(perf_B.sharpe_ratio, 0.01)
        
        # 基于Sharpe比率计算权重
        total_sharpe = sharpe_A + sharpe_B
        weight_A = sharpe_A / total_sharpe
        weight_B = sharpe_B / total_sharpe
        
        # 避免极端权重
        weight_A = np.clip(weight_A, 0.2, 0.8)
        weight_B = 1.0 - weight_A
        
        return weight_A, weight_B
    
    def generate_allocation_config(self, 
                                 regime_result: Dict,
                                 strategy_performances: Dict[str, StrategyPerformance] = None,
                                 force_cooldown: bool = False,
                                 cooldown_reason: str = "") -> AllocationConfig:
        """
        生成配比配置
        
        Args:
            regime_result: 市况检测结果
            strategy_performances: 策略表现字典
            force_cooldown: 是否强制冷静期
            cooldown_reason: 冷静期原因
            
        Returns:
            配比配置对象
        """
        try:
            # 1. 获取基础配比（来自四象限）
            allocation = regime_result.get('allocation', {})
            base_alloc_A = allocation.get('mean_reversion_weight', 0.5)
            base_alloc_B = allocation.get('trend_following_weight', 0.5)
            
            # 2. 计算表现权重
            if strategy_performances and len(strategy_performances) >= 2:
                strategy_names = list(strategy_performances.keys())
                perf_A = strategy_performances[strategy_names[0]]
                perf_B = strategy_performances[strategy_names[1]]
                
                perf_weight_A, perf_weight_B = self.calculate_performance_weights(perf_A, perf_B)
            else:
                # 默认表现权重
                perf_weight_A, perf_weight_B = 0.5, 0.5
            
            # 3. 混合基础权重和表现权重
            final_alloc_A = (base_alloc_A * (1 - self.sharpe_blend_ratio) + 
                           perf_weight_A * self.sharpe_blend_ratio)
            final_alloc_B = 1.0 - final_alloc_A
            
            # 4. 应用配比限制
            max_adj = self.default_config['max_allocation_adjustment']
            final_alloc_A = np.clip(final_alloc_A, 0.5 - max_adj, 0.5 + max_adj)
            final_alloc_B = 1.0 - final_alloc_A
            
            # 5. 检查冷静期
            cooldown_active = force_cooldown or self._check_cooldown_conditions()
            
            if cooldown_active and not cooldown_reason:
                cooldown_reason = "自动触发冷静期"
            
            # 6. 生成配置
            config = AllocationConfig(
                timestamp=datetime.now().isoformat(),
                regime_type=regime_result.get('regime_type', {}).get('description', 'Unknown'),
                base_allocation_A=base_alloc_A,
                base_allocation_B=base_alloc_B,
                performance_weight_A=perf_weight_A,
                performance_weight_B=perf_weight_B,
                final_allocation_A=final_alloc_A if not cooldown_active else 0.1,  # 冷静期减少配比
                final_allocation_B=final_alloc_B if not cooldown_active else 0.1,
                total_capital=self.default_config['total_capital'],
                max_position_size=self.default_config['max_position_size'],
                cooldown_active=cooldown_active,
                cooldown_reason=cooldown_reason,
                version=self._generate_version()
            )
            
            return config
            
        except Exception as e:
            self.logger.error(f"生成配比配置失败: {e}")
            # 返回默认配置
            return self._get_default_allocation_config()
    
    def _check_cooldown_conditions(self) -> bool:
        """检查是否需要触发冷静期"""
        try:
            # 读取最近的交易记录
            performance_history = self._load_performance_history()
            
            if not performance_history:
                return False
            
            # 检查日内亏损
            today = datetime.now().date()
            today_trades = [
                p for p in performance_history 
                if datetime.fromisoformat(p['date']).date() == today
            ]
            
            if today_trades:
                daily_pnl = sum(t.get('total_pnl', 0) for t in today_trades)
                daily_loss_limit = self.default_config['daily_loss_limit'] * self.default_config['total_capital']
                
                if daily_pnl <= -daily_loss_limit:
                    self.logger.warning(f"触发日内亏损限制: {daily_pnl:.2f}")
                    return True
            
            # 检查组合回撤
            recent_pnl = [p.get('total_pnl', 0) for p in performance_history[-30:]]  # 最近30天
            if len(recent_pnl) > 1:
                cumulative_pnl = np.cumsum(recent_pnl)
                running_max = np.maximum.accumulate(cumulative_pnl)
                current_drawdown = (cumulative_pnl[-1] - running_max[-1]) / self.default_config['total_capital']
                
                if current_drawdown <= -self.default_config['drawdown_limit']:
                    self.logger.warning(f"触发组合回撤限制: {current_drawdown:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"冷静期检查失败: {e}")
            return False
    
    def _load_performance_history(self) -> List[Dict]:
        """加载表现历史"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载表现历史失败: {e}")
        return []
    
    def _generate_version(self) -> str:
        """生成版本号"""
        now = datetime.now()
        return f"v{now.strftime('%Y%m%d_%H%M%S')}"
    
    def _get_default_allocation_config(self) -> AllocationConfig:
        """获取默认配比配置"""
        return AllocationConfig(
            timestamp=datetime.now().isoformat(),
            regime_type="默认配置",
            base_allocation_A=0.5,
            base_allocation_B=0.5,
            performance_weight_A=0.5,
            performance_weight_B=0.5,
            final_allocation_A=0.5,
            final_allocation_B=0.5,
            total_capital=self.default_config['total_capital'],
            max_position_size=self.default_config['max_position_size'],
            cooldown_active=False,
            cooldown_reason="",
            version=self._generate_version()
        )
    
    def save_allocation_config(self, config: AllocationConfig):
        """保存配比配置"""
        try:
            # 1. 保存当前配置到parameters.json
            config_dict = asdict(config)
            config_dict.update(self.default_config)  # 合并默认配置
            
            with open(self.parameters_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # 2. 备份历史版本
            backup_file = os.path.join(self.history_dir, f"parameters_{config.version}.json")
            shutil.copy2(self.parameters_file, backup_file)
            
            # 3. 更新配比历史
            self._update_allocation_history(config)
            
            self.current_config = config
            
            self.logger.info(f"保存配比配置: A={config.final_allocation_A:.1%}, B={config.final_allocation_B:.1%}")
            
        except Exception as e:
            self.logger.error(f"保存配比配置失败: {e}")
    
    def _update_allocation_history(self, config: AllocationConfig):
        """更新配比历史"""
        try:
            history = []
            if os.path.exists(self.allocation_history_file):
                with open(self.allocation_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append(asdict(config))
            
            # 只保留最近200条记录
            if len(history) > 200:
                history = history[-200:]
            
            with open(self.allocation_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"更新配比历史失败: {e}")
    
    def check_dynamic_boost(self, strategy_performances: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """
        检查动态加仓条件
        
        Args:
            strategy_performances: 策略表现字典
            
        Returns:
            策略加仓幅度字典
        """
        boost_adjustments = {}
        
        try:
            for strategy_name, performance in strategy_performances.items():
                # 检查连续盈利和Sharpe比率
                if (performance.sharpe_ratio > self.default_config['dynamic_boost_threshold'] and
                    performance.win_rate > 0.6 and
                    len(performance.returns) >= 10):
                    
                    # 检查最近两周是否连续盈利
                    recent_returns = performance.returns[-10:]  # 最近10个交易日
                    if all(r > 0 for r in recent_returns):
                        boost_adjustments[strategy_name] = self.default_config['dynamic_boost_amount']
                        self.logger.info(f"策略 {strategy_name} 满足动态加仓条件: Sharpe={performance.sharpe_ratio:.2f}")
                    
        except Exception as e:
            self.logger.warning(f"动态加仓检查失败: {e}")
        
        return boost_adjustments
    
    def apply_dynamic_adjustments(self, 
                                 config: AllocationConfig, 
                                 boost_adjustments: Dict[str, float]) -> AllocationConfig:
        """
        应用动态调整
        
        Args:
            config: 原始配置
            boost_adjustments: 加仓调整字典
            
        Returns:
            调整后的配置
        """
        if not boost_adjustments:
            return config
        
        try:
            # 假设策略A是均值回归，策略B是趋势跟踪
            strategy_map = {
                'mean_reversion': 'final_allocation_A',
                'trend_following': 'final_allocation_B'
            }
            
            for strategy_name, boost_amount in boost_adjustments.items():
                if strategy_name in strategy_map:
                    attr_name = strategy_map[strategy_name]
                    current_value = getattr(config, attr_name)
                    new_value = min(current_value + boost_amount, 0.8)  # 最大80%
                    setattr(config, attr_name, new_value)
                    
                    # 调整另一个策略的配比
                    other_attr = 'final_allocation_B' if attr_name == 'final_allocation_A' else 'final_allocation_A'
                    setattr(config, other_attr, 1.0 - new_value)
                    
                    self.logger.info(f"应用动态加仓: {strategy_name} +{boost_amount:.1%}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"应用动态调整失败: {e}")
            return config
    
    def load_parameters(self) -> Dict:
        """加载parameters.json文件"""
        try:
            if os.path.exists(self.parameters_file):
                with open(self.parameters_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 返回默认参数
                default_config = asdict(self._get_default_allocation_config())
                default_config.update(self.default_config)
                return default_config
        except Exception as e:
            self.logger.error(f"加载参数文件失败: {e}")
            return {}
    
    def get_current_allocation(self) -> Tuple[float, float]:
        """获取当前配比"""
        if self.current_config:
            return self.current_config.final_allocation_A, self.current_config.final_allocation_B
        else:
            return 0.5, 0.5

def main():
    """测试配比管理器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    manager = AllocationManager()
    
    # 模拟市况结果
    regime_result = {
        'regime_type': {'description': '趋势市+高波动'},
        'allocation': {'mean_reversion_weight': 0.3, 'trend_following_weight': 0.7}
    }
    
    # 模拟策略表现
    strategy_performances = {
        'mean_reversion': StrategyPerformance(
            strategy_name='mean_reversion',
            returns=[0.01, -0.005, 0.015, 0.008, 0.012],
            sharpe_ratio=1.2,
            max_drawdown=-0.02,
            win_rate=0.8,
            total_trades=5,
            avg_return_per_trade=0.008,
            volatility=0.01
        ),
        'trend_following': StrategyPerformance(
            strategy_name='trend_following',
            returns=[0.02, 0.015, -0.01, 0.025, 0.018],
            sharpe_ratio=1.5,
            max_drawdown=-0.015,
            win_rate=0.8,
            total_trades=5,
            avg_return_per_trade=0.014,
            volatility=0.015
        )
    }
    
    print("🔧 生成配比配置...")
    config = manager.generate_allocation_config(regime_result, strategy_performances)
    
    print(f"\n📊 配比结果:")
    print(f"市况类型: {config.regime_type}")
    print(f"基础配比 - A: {config.base_allocation_A:.1%}, B: {config.base_allocation_B:.1%}")
    print(f"表现权重 - A: {config.performance_weight_A:.1%}, B: {config.performance_weight_B:.1%}")
    print(f"最终配比 - A: {config.final_allocation_A:.1%}, B: {config.final_allocation_B:.1%}")
    print(f"冷静期: {'是' if config.cooldown_active else '否'}")
    
    # 检查动态加仓
    boost_adjustments = manager.check_dynamic_boost(strategy_performances)
    if boost_adjustments:
        print(f"\n🚀 动态加仓建议: {boost_adjustments}")
        config = manager.apply_dynamic_adjustments(config, boost_adjustments)
        print(f"调整后配比 - A: {config.final_allocation_A:.1%}, B: {config.final_allocation_B:.1%}")
    
    # 保存配置
    manager.save_allocation_config(config)
    print(f"\n💾 配置已保存: {config.version}")

if __name__ == "__main__":
    main()