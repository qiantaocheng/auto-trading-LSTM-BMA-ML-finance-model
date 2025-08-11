#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoTrader 回测分析器
提供专业级的绩效分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """绩效指标汇总"""
    # 收益指标
    total_return: float
    annual_return: float
    monthly_returns: List[float]
    
    # 风险指标
    annual_volatility: float
    max_drawdown: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    
    # 风险调整收益
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 交易统计
    win_rate: float
    profit_factor: float  # 盈利因子
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # 其他指标
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0


class BacktestAnalyzer:
    """回测分析器"""
    
    def __init__(self, backtest_results: Dict[str, Any], benchmark_symbol: str = "SPY"):
        self.results = backtest_results
        self.benchmark_symbol = benchmark_symbol
        self.logger = logging.getLogger("BacktestAnalyzer")
        
        # 转换数据
        self.performance_df = pd.DataFrame(backtest_results['detailed_performance'])
        self.trades_df = pd.DataFrame(backtest_results['trades']) if backtest_results['trades'] else pd.DataFrame()
        
        if not self.performance_df.empty:
            self.performance_df['date'] = pd.to_datetime(self.performance_df['date'])
            self.performance_df.set_index('date', inplace=True)
        
        if not self.trades_df.empty:
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
    
    def calculate_comprehensive_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """计算全面的绩效指标"""
        if self.performance_df.empty:
            return PerformanceMetrics(0, 0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        returns = pd.Series(self.performance_df['daily_return'].values, 
                          index=self.performance_df.index)
        
        # 基础收益指标
        total_return = self.results['returns']['total_return']
        annual_return = self.results['returns']['annual_return']
        annual_vol = self.results['returns']['annual_volatility']
        
        # 月度收益率
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1).tolist()
        
        # 风险指标
        max_drawdown = abs(self.results['returns']['max_drawdown'])
        var_95 = np.percentile(returns.dropna(), 5)
        var_99 = np.percentile(returns.dropna(), 1)
        
        # 下行波动率（用于Sortino比率）
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 风险调整收益
        sharpe_ratio = self.results['returns']['sharpe_ratio']
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 交易统计
        win_rate = self.results['returns']['win_rate']
        profit_factor, avg_win, avg_loss = self._calculate_profit_factor()
        max_consec_wins, max_consec_losses = self._calculate_consecutive_stats(returns)
        
        # 相对基准的指标
        beta, alpha, info_ratio, tracking_error = 0, 0, 0, 0
        if benchmark_returns is not None:
            beta, alpha, info_ratio, tracking_error = self._calculate_relative_metrics(
                returns, benchmark_returns
            )
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=monthly_returns,
            annual_volatility=annual_vol,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            beta=beta,
            alpha=alpha,
            information_ratio=info_ratio,
            tracking_error=tracking_error
        )
    
    def _calculate_profit_factor(self) -> Tuple[float, float, float]:
        """计算盈利因子"""
        if self.trades_df.empty:
            return 0, 0, 0
        
        # 计算每笔交易的盈亏
        buy_trades = self.trades_df[self.trades_df['action'] == 'BUY'].copy()
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL'].copy()
        
        if buy_trades.empty or sell_trades.empty:
            return 0, 0, 0
        
        # 简化：假设按时间配对买卖交易
        trade_pnl = []
        for symbol in self.trades_df['symbol'].unique():
            symbol_buys = buy_trades[buy_trades['symbol'] == symbol].sort_values('timestamp')
            symbol_sells = sell_trades[sell_trades['symbol'] == symbol].sort_values('timestamp')
            
            for i, buy in symbol_buys.iterrows():
                # 找到对应的卖出交易
                matching_sells = symbol_sells[symbol_sells['timestamp'] > buy['timestamp']]
                if not matching_sells.empty:
                    sell = matching_sells.iloc[0]
                    pnl = (sell['price'] - buy['price']) * min(buy['shares'], sell['shares'])
                    trade_pnl.append(pnl)
        
        if not trade_pnl:
            return 0, 0, 0
        
        wins = [p for p in trade_pnl if p > 0]
        losses = [p for p in trade_pnl if p < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / sum(abs(l) for l in losses) if losses else float('inf')
        
        return profit_factor, avg_win, avg_loss
    
    def _calculate_consecutive_stats(self, returns: pd.Series) -> Tuple[int, int]:
        """计算连续盈亏统计"""
        if returns.empty:
            return 0, 0
        
        # 将收益转换为涨跌标志
        signals = (returns > 0).astype(int)
        
        # 计算连续序列
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for signal in signals:
            if signal == 1:  # 盈利
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:  # 亏损
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        return max_wins, max_losses
    
    def _calculate_relative_metrics(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> Tuple[float, float, float, float]:
        """计算相对基准的指标"""
        try:
            # 对齐数据
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if aligned_data.empty:
                return 0, 0, 0, 0
            
            port_ret = aligned_data.iloc[:, 0]
            bench_ret = aligned_data.iloc[:, 1]
            
            # Beta和Alpha
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            benchmark_var = np.var(bench_ret)
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            alpha = port_ret.mean() - beta * bench_ret.mean()
            alpha = alpha * 252  # 年化
            
            # 信息比率和跟踪误差
            excess_returns = port_ret - bench_ret
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            return beta, alpha, information_ratio, tracking_error
            
        except Exception as e:
            self.logger.warning(f"计算相对指标失败: {e}")
            return 0, 0, 0, 0
    
    def create_comprehensive_report(self, save_path: Optional[str] = None) -> None:
        """创建综合回测报告"""
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. 净值曲线图
        ax1 = plt.subplot(4, 2, 1)
        self._plot_equity_curve(ax1)
        
        # 2. 回撤图
        ax2 = plt.subplot(4, 2, 2)
        self._plot_drawdown(ax2)
        
        # 3. 月度收益热力图
        ax3 = plt.subplot(4, 2, 3)
        self._plot_monthly_returns_heatmap(ax3)
        
        # 4. 收益分布直方图
        ax4 = plt.subplot(4, 2, 4)
        self._plot_returns_distribution(ax4)
        
        # 5. 滚动夏普比率
        ax5 = plt.subplot(4, 2, 5)
        self._plot_rolling_sharpe(ax5)
        
        # 6. 持仓权重变化
        ax6 = plt.subplot(4, 2, 6)
        self._plot_position_weights(ax6)
        
        # 7. 交易统计
        ax7 = plt.subplot(4, 2, 7)
        self._plot_trade_statistics(ax7)
        
        # 8. 绩效指标总结
        ax8 = plt.subplot(4, 2, 8)
        self._plot_performance_summary(ax8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"报告已保存到: {save_path}")
        
        plt.show()
    
    def _plot_equity_curve(self, ax):
        """绘制净值曲线"""
        if self.performance_df.empty:
            return
        
        equity = self.performance_df['total_value']
        initial_value = self.results['portfolio']['initial_capital']
        
        # 绘制净值曲线
        ax.plot(equity.index, equity / initial_value, linewidth=2, label='策略净值', color='navy')
        
        # 添加基准线
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='初始值')
        
        ax.set_title('策略净值曲线', fontsize=14, fontweight='bold')
        ax.set_ylabel('净值 (初始值=1)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_drawdown(self, ax):
        """绘制回撤图"""
        if self.performance_df.empty:
            return
        
        returns = pd.Series(self.performance_df['daily_return'].values)
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        
        ax.fill_between(drawdowns.index, drawdowns, 0, alpha=0.3, color='red', label='回撤')
        ax.plot(drawdowns.index, drawdowns, linewidth=1, color='darkred')
        
        ax.set_title('回撤曲线', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤幅度', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标注最大回撤
        max_dd_idx = drawdowns.idxmin()
        max_dd_val = drawdowns.min()
        ax.annotate(f'最大回撤: {max_dd_val:.2%}', 
                   xy=(max_dd_idx, max_dd_val), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_monthly_returns_heatmap(self, ax):
        """绘制月度收益热力图"""
        if self.performance_df.empty:
            return
        
        returns = pd.Series(self.performance_df['daily_return'].values,
                          index=self.performance_df.index)
        
        # 计算月度收益
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建年月矩阵
        monthly_returns.index = monthly_returns.index.to_period('M')
        years = sorted(set(monthly_returns.index.year))
        months = range(1, 13)
        
        heatmap_data = np.full((len(years), 12), np.nan)
        
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                period = pd.Period(f'{year}-{month:02d}')
                if period in monthly_returns.index:
                    heatmap_data[i, j] = monthly_returns[period]
        
        # 绘制热力图
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        
        # 设置标签
        ax.set_xticks(range(12))
        ax.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                           '7月', '8月', '9月', '10月', '11月', '12月'])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        
        # 添加数值标签
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.1%}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('月度收益率', rotation=270, labelpad=15)
    
    def _plot_returns_distribution(self, ax):
        """绘制收益分布直方图"""
        if self.performance_df.empty:
            return
        
        returns = pd.Series(self.performance_df['daily_return'].values)
        returns = returns.dropna()
        
        # 绘制直方图
        ax.hist(returns, bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black')
        
        # 添加正态分布拟合曲线
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = ((np.pi * sigma) * np.sqrt(2.0)) ** (-1.0) * np.exp(-0.5 * (1.0 / sigma * (x - mu)) ** 2.0)
        ax.plot(x, y, 'r-', linewidth=2, label=f'正态分布 (μ={mu:.3f}, σ={sigma:.3f})')
        
        # 标注关键统计量
        ax.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'均值: {mu:.3f}')
        ax.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.8, label=f'+1σ: {mu+sigma:.3f}')
        ax.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.8, label=f'-1σ: {mu-sigma:.3f}')
        
        ax.set_title('日收益率分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('日收益率', fontsize=12)
        ax.set_ylabel('概率密度', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_sharpe(self, ax):
        """绘制滚动夏普比率"""
        if self.performance_df.empty:
            return
        
        returns = pd.Series(self.performance_df['daily_return'].values,
                          index=self.performance_df.index)
        
        # 计算60日滚动夏普比率
        rolling_sharpe = returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        ax.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='purple')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='优秀 (>1)')
        ax.axhline(y=2, color='blue', linestyle='--', alpha=0.7, label='卓越 (>2)')
        
        ax.set_title('60日滚动夏普比率', fontsize=14, fontweight='bold')
        ax.set_ylabel('夏普比率', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_position_weights(self, ax):
        """绘制持仓权重变化"""
        if self.performance_df.empty:
            return
        
        # 提取持仓权重数据
        dates = []
        position_data = {}
        
        for _, row in self.performance_df.iterrows():
            if 'positions' in row and row['positions']:
                dates.append(row.name)
                for symbol, weight in row['positions'].items():
                    if symbol not in position_data:
                        position_data[symbol] = []
                    position_data[symbol].append(weight)
        
        if not position_data:
            ax.text(0.5, 0.5, '无持仓数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('持仓权重变化', fontsize=14, fontweight='bold')
            return
        
        # 选择权重最大的前10个股票
        avg_weights = {symbol: np.mean(weights) for symbol, weights in position_data.items()}
        top_symbols = sorted(avg_weights.keys(), key=lambda x: avg_weights[x], reverse=True)[:10]
        
        # 绘制堆叠面积图
        weights_matrix = np.zeros((len(dates), len(top_symbols)))
        for i, symbol in enumerate(top_symbols):
            if symbol in position_data:
                weights_matrix[:, i] = position_data[symbol][:len(dates)]
        
        ax.stackplot(dates, *weights_matrix.T, labels=top_symbols, alpha=0.8)
        
        ax.set_title('主要持仓权重变化', fontsize=14, fontweight='bold')
        ax.set_ylabel('权重', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_trade_statistics(self, ax):
        """绘制交易统计"""
        if self.trades_df.empty:
            ax.text(0.5, 0.5, '无交易数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('交易统计', fontsize=14, fontweight='bold')
            return
        
        # 按月统计交易次数
        trade_counts = self.trades_df.groupby(self.trades_df['timestamp'].dt.to_period('M')).size()
        
        # 绘制柱状图
        ax.bar(range(len(trade_counts)), trade_counts.values, alpha=0.7, color='steelblue')
        
        # 设置x轴标签
        ax.set_xticks(range(len(trade_counts)))
        ax.set_xticklabels([str(period) for period in trade_counts.index], rotation=45)
        
        ax.set_title('月度交易次数', fontsize=14, fontweight='bold')
        ax.set_ylabel('交易次数', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加平均线
        avg_trades = trade_counts.mean()
        ax.axhline(y=avg_trades, color='red', linestyle='--', alpha=0.7, 
                  label=f'平均: {avg_trades:.1f}')
        ax.legend()
    
    def _plot_performance_summary(self, ax):
        """绘制绩效指标总结"""
        metrics = self.calculate_comprehensive_metrics()
        
        # 准备数据
        labels = ['总收益率', '年化收益', '年化波动', '夏普比率', '最大回撤', '胜率']
        values = [metrics.total_return, metrics.annual_return, metrics.annual_volatility,
                 metrics.sharpe_ratio, metrics.max_drawdown, metrics.win_rate]
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for label, value in zip(labels, values):
            if label in ['总收益率', '年化收益', '年化波动', '最大回撤', '胜率']:
                formatted_value = f'{value:.2%}'
            else:
                formatted_value = f'{value:.3f}'
            table_data.append([label, formatted_value])
        
        # 添加更多指标
        additional_metrics = [
            ['Sortino比率', f'{metrics.sortino_ratio:.3f}'],
            ['Calmar比率', f'{metrics.calmar_ratio:.3f}'],
            ['盈利因子', f'{metrics.profit_factor:.2f}'],
            ['平均盈利', f'${metrics.avg_win:.2f}'],
            ['平均亏损', f'${metrics.avg_loss:.2f}'],
            ['最大连胜', f'{metrics.max_consecutive_wins}'],
            ['最大连亏', f'{metrics.max_consecutive_losses}'],
            ['VaR (95%)', f'{metrics.var_95:.2%}']
        ]
        
        table_data.extend(additional_metrics)
        
        table = ax.table(cellText=table_data,
                        colLabels=['指标', '数值'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('绩效指标汇总', fontsize=14, fontweight='bold', pad=20)
    
    def export_detailed_report(self, file_path: str) -> None:
        """导出详细报告到Excel"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 基础信息
                summary_data = {
                    '回测期间': [f"{self.results['period']['start_date']} 至 {self.results['period']['end_date']}"],
                    '交易天数': [self.results['period']['trading_days']],
                    '初始资金': [f"${self.results['portfolio']['initial_capital']:,.2f}"],
                    '最终资产': [f"${self.results['portfolio']['final_value']:,.2f}"],
                    '总收益率': [f"{self.results['returns']['total_return']:.2%}"],
                    '年化收益率': [f"{self.results['returns']['annual_return']:.2%}"],
                    '年化波动率': [f"{self.results['returns']['annual_volatility']:.2%}"],
                    '夏普比率': [f"{self.results['returns']['sharpe_ratio']:.3f}"],
                    '最大回撤': [f"{self.results['returns']['max_drawdown']:.2%}"],
                    '胜率': [f"{self.results['returns']['win_rate']:.2%}"]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='回测摘要', index=False)
                
                # 每日绩效
                self.performance_df.to_excel(writer, sheet_name='每日绩效')
                
                # 交易记录
                if not self.trades_df.empty:
                    self.trades_df.to_excel(writer, sheet_name='交易记录', index=False)
                
                # 绩效指标
                metrics = self.calculate_comprehensive_metrics()
                metrics_data = {
                    '指标': ['总收益率', '年化收益率', '年化波动率', '夏普比率', 'Sortino比率', 
                           'Calmar比率', '最大回撤', '胜率', '盈利因子', 'VaR(95%)', 'VaR(99%)'],
                    '数值': [metrics.total_return, metrics.annual_return, metrics.annual_volatility,
                           metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio,
                           metrics.max_drawdown, metrics.win_rate, metrics.profit_factor,
                           metrics.var_95, metrics.var_99]
                }
                pd.DataFrame(metrics_data).to_excel(writer, sheet_name='详细指标', index=False)
            
            self.logger.info(f"详细报告已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")


# 使用示例
def analyze_backtest_results(results: Dict[str, Any], save_dir: str = "./backtest_reports"):
    """分析回测结果的便捷函数"""
    import os
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建分析器
    analyzer = BacktestAnalyzer(results)
    
    # 生成综合报告
    report_path = os.path.join(save_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    analyzer.create_comprehensive_report(save_path=report_path)
    
    # 导出详细数据
    excel_path = os.path.join(save_dir, f"backtest_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    analyzer.export_detailed_report(excel_path)
    
    # 计算并打印关键指标
    metrics = analyzer.calculate_comprehensive_metrics()
    
    print("\n" + "="*60)
    print("回测分析结果")
    print("="*60)
    print(f"总收益率: {metrics.total_return:.2%}")
    print(f"年化收益率: {metrics.annual_return:.2%}")
    print(f"年化波动率: {metrics.annual_volatility:.2%}")
    print(f"夏普比率: {metrics.sharpe_ratio:.3f}")
    print(f"Sortino比率: {metrics.sortino_ratio:.3f}")
    print(f"Calmar比率: {metrics.calmar_ratio:.3f}")
    print(f"最大回撤: {metrics.max_drawdown:.2%}")
    print(f"胜率: {metrics.win_rate:.2%}")
    print(f"盈利因子: {metrics.profit_factor:.2f}")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    print(f"最大连胜: {metrics.max_consecutive_wins}")
    print(f"最大连亏: {metrics.max_consecutive_losses}")
    print("="*60)
    
    return analyzer
