import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveCategoryBacktest:
    """
    全面的多category回测和可视化系统
    支持所有投资决策类别的深度分析
    """
    
    def __init__(self, 
                 excel_file: str,
                 start_date: str = "2020-01-01",
                 end_date: str = "2024-12-31",
                 initial_capital: float = 1000000,
                 result_dir: str = "category_analysis_results"):
        
        self.excel_file = excel_file
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.result_dir = result_dir
        
        # 创建结果目录
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # 数据存储
        self.df = None
        self.price_data = {}
        self.category_results = {}
        self.performance_metrics = {}
        
        # 类别定义
        self.investment_categories = {
            'BUY': {'threshold': 0.7, 'description': '强烈推荐买入', 'color': '#2E8B57'},
            'HOLD': {'threshold': 0.4, 'description': '持有观望', 'color': '#FFD700'},
            'SELL': {'threshold': 0.0, 'description': '建议卖出', 'color': '#DC143C'}
        }
        
        # 评估指标类别
        self.evaluation_categories = {
            'risk_adjusted_return': [
                'Strategy_Sharpe', 'Strategy_Sortino', 'Strategy_Calmar',
                'Information_Ratio', 'Strategy_Omega', 'Strategy_TailRatio'
            ],
            'risk_metrics': [
                'Strategy_MaxDrawdown', 'Strategy_VaR_5%', 'Strategy_CVaR_5%',
                'Comprehensive_Risk_Score'
            ],
            'return_metrics': [
                'Monthly_Return', 'Total_Period_Return', 'Annualized_Return',
                'OutOfSample_Sharpe', 'OutOfSample_Sortino'
            ],
            'fundamental_metrics': [
                'Beta', 'Revenue_Growth', 'Net_Profit_Growth', 'EPS', 'PE_Ratio',
                'Gross_Margin', 'ROE'
            ],
            'technical_indicators': [
                'RSI (last day)', 'MA50 (last day)', 'MA20'
            ],
            'factor_scores': [
                'Market_Factor', 'Size_Factor', 'Value_Factor', 'Momentum_Factor',
                'Investment_Factor', 'Final_Score', 'Factor_Score'
            ]
        }
        
        print(f"[GROWTH] 初始化全面Category回测系统")
        print(f"[DATA] 数据文件: {self.excel_file}")
        print(f"[DATE] 回测期间: {self.start_date} ~ {self.end_date}")
        print(f"初始资金: ${self.initial_capital:,.0f}")
        print(f"[FILE] 结果目录: {self.result_dir}")
        
    def get_price_column(self, data):
        """获取合适的价格列名"""
        if 'Adj Close' in data.columns:
            return 'Adj Close'
        elif 'Close' in data.columns:
            return 'Close'
        else:
            return None
        
    def load_data(self):
        """加载Excel数据"""
        try:
            self.df = pd.read_excel(self.excel_file)
            print(f"[OK] 成功加载数据: {len(self.df)} 只股票")
            
            # 分析推荐分布
            if 'Recommendation' in self.df.columns:
                rec_counts = self.df['Recommendation'].value_counts()
                print(f"[CHART] 推荐分布:")
                for rec, count in rec_counts.items():
                    pct = count / len(self.df) * 100
                    print(f"   {rec}: {count}只 ({pct:.1f}%)")
            
            # 检查关键列
            required_columns = ['Ticker', 'Final_Score']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                print(f"[WARNING] 缺少必要列: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 数据加载失败: {e}")
            return False
    
    def categorize_stocks(self):
        """根据不同标准对股票进行分类"""
        print("\n[TARGET] 开始股票分类...")
        
        # 1. 基于推荐的分类
        if 'Recommendation' in self.df.columns:
            self.df['Category_Recommendation'] = self.df['Recommendation']
        else:
            # 基于Final_Score分类
            score_quantiles = self.df['Final_Score'].quantile([0.3, 0.7])
            self.df['Category_Recommendation'] = self.df['Final_Score'].apply(
                lambda x: 'BUY' if x >= score_quantiles[0.7] else 
                         'HOLD' if x >= score_quantiles[0.3] else 'SELL'
            )
        
        # 2. 基于ML预测的分类
        if 'ML_Recommendation' in self.df.columns:
            self.df['Category_ML'] = self.df['ML_Recommendation']
        else:
            self.df['Category_ML'] = self.df['Category_Recommendation']
        
        # 3. 基于风险调整收益的分类
        risk_adj_columns = [col for col in self.evaluation_categories['risk_adjusted_return'] 
                           if col in self.df.columns]
        if risk_adj_columns:
            self.df['Risk_Adjusted_Score'] = self.df[risk_adj_columns].mean(axis=1)
            ra_quantiles = self.df['Risk_Adjusted_Score'].quantile([0.3, 0.7])
            self.df['Category_RiskAdjusted'] = self.df['Risk_Adjusted_Score'].apply(
                lambda x: 'BUY' if x >= ra_quantiles[0.7] else 
                         'HOLD' if x >= ra_quantiles[0.3] else 'SELL'
            )
        
        # 4. 基于收益率的分类
        return_columns = [col for col in self.evaluation_categories['return_metrics'] 
                         if col in self.df.columns]
        if return_columns:
            self.df['Return_Score'] = self.df[return_columns].mean(axis=1)
            ret_quantiles = self.df['Return_Score'].quantile([0.3, 0.7])
            self.df['Category_Return'] = self.df['Return_Score'].apply(
                lambda x: 'BUY' if x >= ret_quantiles[0.7] else 
                         'HOLD' if x >= ret_quantiles[0.3] else 'SELL'
            )
        
        # 5. 基于因子得分的分类
        factor_columns = [col for col in self.evaluation_categories['factor_scores'] 
                         if col in self.df.columns]
        if factor_columns:
            self.df['Factor_Composite_Score'] = self.df[factor_columns].mean(axis=1)
            factor_quantiles = self.df['Factor_Composite_Score'].quantile([0.3, 0.7])
            self.df['Category_Factor'] = self.df['Factor_Composite_Score'].apply(
                lambda x: 'BUY' if x >= factor_quantiles[0.7] else 
                         'HOLD' if x >= factor_quantiles[0.3] else 'SELL'
            )
        
        # 统计各类别分布
        category_columns = [col for col in self.df.columns if col.startswith('Category_')]
        for col in category_columns:
            print(f"\n[DATA] {col} 分布:")
            counts = self.df[col].value_counts()
            for cat, count in counts.items():
                pct = count / len(self.df) * 100
                print(f"   {cat}: {count}只 ({pct:.1f}%)")
        
        return True
    
    def download_price_data(self):
        """下载历史价格数据"""
        print("\n[EMOJI] 下载历史价格数据...")
        
        tickers = self.df['Ticker'].unique()
        successful_downloads = 0
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if data is not None and not data.empty:
                    # 标准化数据格式
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    
                    self.price_data[ticker] = data
                    successful_downloads += 1
                    
            except Exception as e:
                print(f"  [WARNING] {ticker} 下载失败: {e}")
        
        print(f"[OK] 成功下载 {successful_downloads}/{len(tickers)} 只股票价格数据")
        return successful_downloads > 0
    
    def calculate_category_performance(self):
        """计算各类别的表现"""
        print("\n[DATA] 计算各类别表现...")
        
        category_columns = [col for col in self.df.columns if col.startswith('Category_')]
        
        for category_col in category_columns:
            print(f"\n[EMOJI] 分析 {category_col}...")
            
            category_name = category_col.replace('Category_', '')
            self.category_results[category_name] = {}
            
            for category in ['BUY', 'HOLD', 'SELL']:
                category_stocks = self.df[self.df[category_col] == category]['Ticker'].tolist()
                
                if not category_stocks:
                    continue
                
                # 计算该类别的组合表现
                category_returns = []
                category_prices = []
                
                for ticker in category_stocks:
                    if ticker in self.price_data:
                        # 尝试获取Adj Close，如果没有则使用Close
                        price_data = self.price_data[ticker]
                        if 'Adj Close' in price_data.columns:
                            prices = price_data['Adj Close']
                        elif 'Close' in price_data.columns:
                            prices = price_data['Close']
                        else:
                            continue
                            
                        if len(prices) > 1:
                            returns = prices.pct_change().dropna()
                            category_returns.append(returns)
                            category_prices.append(prices)
                
                if category_returns:
                    # 等权重组合收益
                    portfolio_returns = pd.concat(category_returns, axis=1).mean(axis=1)
                    portfolio_prices = pd.concat(category_prices, axis=1).mean(axis=1)
                    
                    # 计算性能指标
                    performance = self.calculate_performance_metrics(portfolio_returns, portfolio_prices)
                    performance['stock_count'] = len(category_stocks)
                    performance['tickers'] = category_stocks
                    
                    self.category_results[category_name][category] = performance
                    
                    print(f"  {category}: {len(category_stocks)}只股票, "
                          f"收益率: {performance['total_return']:.2%}, "
                          f"夏普比率: {performance['sharpe_ratio']:.3f}")
        
        return True
    
    def calculate_performance_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict:
        """计算性能指标"""
        
        # 基本收益指标
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 风险指标
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # VaR和CVaR
        var_5 = returns.quantile(0.05)
        cvar_5 = returns[returns <= var_5].mean()
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'avg_return': returns.mean(),
            'median_return': returns.median()
        }
    
    def create_comprehensive_visualizations(self):
        """创建全面的可视化图表"""
        print("\n[DATA] 创建综合可视化图表...")
        
        # 1. 类别分布对比图
        self.plot_category_distribution()
        
        # 2. 各类别收益率对比
        self.plot_category_returns()
        
        # 3. 风险收益散点图
        self.plot_risk_return_scatter()
        
        # 4. 各类别累计收益曲线
        self.plot_cumulative_returns()
        
        # 5. 性能指标雷达图
        self.plot_performance_radar()
        
        # 6. 分类准确性分析
        self.plot_classification_accuracy()
        
        # 7. 因子分析热力图
        self.plot_factor_heatmap()
        
        # 8. 综合仪表板
        self.create_comprehensive_dashboard()
        
        print("[OK] 所有可视化图表已生成")
    
    def plot_category_distribution(self):
        """绘制类别分布图"""
        category_columns = [col for col in self.df.columns if col.startswith('Category_')]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(category_columns):
            if i < len(axes):
                counts = self.df[col].value_counts()
                colors = [self.investment_categories[cat]['color'] for cat in counts.index]
                
                ax = axes[i]
                wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, 
                                                 autopct='%1.1f%%', colors=colors)
                ax.set_title(f'{col.replace("Category_", "")} 分布', fontsize=14, fontweight='bold')
                
                # 添加数量标签
                for j, (cat, count) in enumerate(counts.items()):
                    ax.text(0, -1.3 - j*0.1, f'{cat}: {count}只', 
                           ha='center', fontsize=10)
        
        # 隐藏多余的子图
        for i in range(len(category_columns), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各类别股票分布对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_returns(self):
        """绘制各类别收益率对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown']
        metric_names = ['总收益率', '年化收益率', '夏普比率', '最大回撤']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            categories = []
            buy_values = []
            hold_values = []
            sell_values = []
            
            for category_name, results in self.category_results.items():
                categories.append(category_name)
                buy_values.append(results.get('BUY', {}).get(metric, 0))
                hold_values.append(results.get('HOLD', {}).get(metric, 0))
                sell_values.append(results.get('SELL', {}).get(metric, 0))
            
            x = np.arange(len(categories))
            width = 0.25
            
            ax.bar(x - width, buy_values, width, label='BUY', color='#2E8B57', alpha=0.8)
            ax.bar(x, hold_values, width, label='HOLD', color='#FFD700', alpha=0.8)
            ax.bar(x + width, sell_values, width, label='SELL', color='#DC143C', alpha=0.8)
            
            ax.set_xlabel('分类方法')
            ax.set_ylabel(name)
            ax.set_title(f'{name} 类别对比')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, (b, h, s) in enumerate(zip(buy_values, hold_values, sell_values)):
                if metric == 'max_drawdown':
                    ax.text(j-width, b-0.001, f'{b:.1%}', ha='center', va='top', fontsize=8)
                    ax.text(j, h-0.001, f'{h:.1%}', ha='center', va='top', fontsize=8)
                    ax.text(j+width, s-0.001, f'{s:.1%}', ha='center', va='top', fontsize=8)
                else:
                    ax.text(j-width, b+0.001, f'{b:.1%}', ha='center', va='bottom', fontsize=8)
                    ax.text(j, h+0.001, f'{h:.1%}', ha='center', va='bottom', fontsize=8)
                    ax.text(j+width, s+0.001, f'{s:.1%}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('各类别性能指标对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/category_returns_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_risk_return_scatter(self):
        """绘制风险收益散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (category_name, results) in enumerate(self.category_results.items()):
            if i < len(axes):
                ax = axes[i]
                
                for category in ['BUY', 'HOLD', 'SELL']:
                    if category in results:
                        perf = results[category]
                        ax.scatter(perf['volatility'], perf['annualized_return'], 
                                 s=100, alpha=0.7, 
                                 c=self.investment_categories[category]['color'],
                                 label=f'{category} ({perf["stock_count"]}只)')
                
                ax.set_xlabel('年化波动率')
                ax.set_ylabel('年化收益率')
                ax.set_title(f'{category_name} 风险收益分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 添加效率前沿参考线
                x_line = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
                ax.plot(x_line, x_line, '--', alpha=0.5, color='gray', label='夏普比率=1')
        
        # 隐藏多余子图
        for i in range(len(self.category_results), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各分类方法风险收益分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cumulative_returns(self):
        """绘制累计收益曲线"""
        if not self.price_data:
            return
        
        fig, axes = plt.subplots(len(self.category_results), 1, figsize=(14, 4*len(self.category_results)))
        if len(self.category_results) == 1:
            axes = [axes]
        
        for i, (category_name, results) in enumerate(self.category_results.items()):
            ax = axes[i]
            
            for category in ['BUY', 'HOLD', 'SELL']:
                if category in results:
                    tickers = results[category]['tickers']
                    
                    # 计算等权重组合的累计收益
                    portfolio_prices = []
                    for ticker in tickers:
                        if ticker in self.price_data:
                            # 尝试获取Adj Close，如果没有则使用Close
                            price_data = self.price_data[ticker]
                            if 'Adj Close' in price_data.columns:
                                prices = price_data['Adj Close']
                            elif 'Close' in price_data.columns:
                                prices = price_data['Close']
                            else:
                                continue
                                
                            if len(prices) > 1:
                                normalized_prices = prices / prices.iloc[0]
                                portfolio_prices.append(normalized_prices)
                    
                    if portfolio_prices:
                        portfolio_cum_returns = pd.concat(portfolio_prices, axis=1).mean(axis=1)
                        ax.plot(portfolio_cum_returns.index, portfolio_cum_returns.values, 
                               label=f'{category} ({len(tickers)}只)', 
                               color=self.investment_categories[category]['color'],
                               linewidth=2)
            
            ax.set_title(f'{category_name} 累计收益曲线')
            ax.set_xlabel('时间')
            ax.set_ylabel('累计收益')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('各类别累计收益曲线对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_radar(self):
        """绘制性能指标雷达图"""
        from math import pi
        
        # 选择关键指标
        metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'win_rate']
        metric_names = ['总收益', '夏普比率', 'Sortino比率', '胜率']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        for i, (category_name, results) in enumerate(self.category_results.items()):
            if i < len(axes):
                ax = axes[i]
                
                # 角度设置
                angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
                angles += angles[:1]  # 闭合
                
                # 修复matplotlib兼容性问题
                try:
                    ax.set_theta_offset(pi / 2)
                    ax.set_theta_direction(-1)
                except AttributeError:
                    # 新版本matplotlib的设置方式
                    ax.set_theta_zero_location('N')  # 0度位置在顶部
                    ax.set_theta_direction(-1)  # 顺时针方向
                
                # 绘制每个类别
                for category in ['BUY', 'HOLD', 'SELL']:
                    if category in results:
                        values = []
                        for metric in metrics:
                            value = results[category].get(metric, 0)
                            # 标准化到0-1范围
                            if metric == 'total_return':
                                values.append(max(0, min(1, (value + 0.5) / 1.0)))
                            elif metric in ['sharpe_ratio', 'sortino_ratio']:
                                values.append(max(0, min(1, (value + 1) / 3.0)))
                            else:  # win_rate
                                values.append(value)
                        
                        values += values[:1]  # 闭合
                        
                        ax.plot(angles, values, 'o-', linewidth=2, 
                               label=f'{category}', 
                               color=self.investment_categories[category]['color'])
                        ax.fill(angles, values, alpha=0.25, 
                               color=self.investment_categories[category]['color'])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metric_names)
                ax.set_ylim(0, 1)
                ax.set_title(f'{category_name} 性能雷达图')
                ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
                ax.grid(True)
        
        # 隐藏多余子图
        for i in range(len(self.category_results), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('各类别性能指标雷达图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_accuracy(self):
        """绘制分类准确性分析"""
        category_columns = [col for col in self.df.columns if col.startswith('Category_')]
        
        if len(category_columns) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 两两比较分类方法
        comparisons = []
        for i in range(len(category_columns)):
            for j in range(i+1, len(category_columns)):
                comparisons.append((category_columns[i], category_columns[j]))
        
        for i, (cat1, cat2) in enumerate(comparisons[:4]):
            if i < len(axes):
                ax = axes[i]
                
                # 创建混淆矩阵
                cm = confusion_matrix(self.df[cat1], self.df[cat2], 
                                    labels=['BUY', 'HOLD', 'SELL'])
                
                # 绘制热力图
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['BUY', 'HOLD', 'SELL'],
                           yticklabels=['BUY', 'HOLD', 'SELL'])
                
                ax.set_title(f'{cat1.replace("Category_", "")} vs {cat2.replace("Category_", "")}')
                ax.set_xlabel(cat2.replace("Category_", ""))
                ax.set_ylabel(cat1.replace("Category_", ""))
                
                # 计算一致性
                accuracy = accuracy_score(self.df[cat1], self.df[cat2])
                ax.text(0.5, -0.1, f'一致性: {accuracy:.1%}', 
                       transform=ax.transAxes, ha='center', fontsize=10)
        
        # 隐藏多余子图
        for i in range(len(comparisons), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('分类方法一致性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/classification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_factor_heatmap(self):
        """绘制因子分析热力图"""
        # 选择数值型列
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # 排除不相关的列
        exclude_columns = ['Unnamed: 0'] if 'Unnamed: 0' in numeric_columns else []
        plot_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if len(plot_columns) < 2:
            return
        
        # 计算相关性矩阵
        corr_matrix = self.df[plot_columns].corr()
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('因子相关性热力图', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/factor_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, height_ratios=[1, 1, 1, 1, 1, 1])
        
        # 1. 标题
        fig.suptitle('全面Category分析仪表板', fontsize=24, fontweight='bold', y=0.98)
        
        # 2. 总体统计
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_summary_stats(ax1)
        
        # 3. 收益率分布
        ax2 = fig.add_subplot(gs[1, :2])
        self.plot_return_distribution(ax2)
        
        # 4. 风险指标分布
        ax3 = fig.add_subplot(gs[1, 2:])
        self.plot_risk_distribution(ax3)
        
        # 5. 行业分布（如果有行业数据）
        ax4 = fig.add_subplot(gs[2, :2])
        self.plot_sector_distribution(ax4)
        
        # 6. 市值分布
        ax5 = fig.add_subplot(gs[2, 2:])
        self.plot_market_cap_distribution(ax5)
        
        # 7. 时间序列表现
        ax6 = fig.add_subplot(gs[3, :])
        self.plot_time_series_performance(ax6)
        
        # 8. 评估指标箱线图
        ax7 = fig.add_subplot(gs[4, :])
        self.plot_metrics_boxplot(ax7)
        
        # 9. 投资建议总结
        ax8 = fig.add_subplot(gs[5, :])
        self.plot_investment_summary(ax8)
        
        plt.tight_layout()
        plt.savefig(f'{self.result_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_summary_stats(self, ax):
        """绘制总体统计信息"""
        ax.axis('off')
        
        # 计算总体统计
        total_stocks = len(self.df)
        avg_score = self.df['Final_Score'].mean()
        
        # 各类别统计
        if 'Category_Recommendation' in self.df.columns:
            buy_count = len(self.df[self.df['Category_Recommendation'] == 'BUY'])
            hold_count = len(self.df[self.df['Category_Recommendation'] == 'HOLD'])
            sell_count = len(self.df[self.df['Category_Recommendation'] == 'SELL'])
            
            text = f"""
            [DATA] 分析总览
            
            总股票数量: {total_stocks} 只
            平均得分: {avg_score:.2f}
            
            推荐分布:
            [GROWTH] BUY: {buy_count} 只 ({buy_count/total_stocks:.1%})
            [DATA] HOLD: {hold_count} 只 ({hold_count/total_stocks:.1%})
            [WARNING] SELL: {sell_count} 只 ({sell_count/total_stocks:.1%})
            """
            
            ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=14, 
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def plot_return_distribution(self, ax):
        """绘制收益率分布"""
        if 'Monthly_Return' in self.df.columns:
            self.df['Monthly_Return'].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title('月度收益率分布')
            ax.set_xlabel('月度收益率')
            ax.set_ylabel('频数')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无收益率数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
    
    def plot_risk_distribution(self, ax):
        """绘制风险指标分布"""
        if 'Strategy_MaxDrawdown' in self.df.columns:
            self.df['Strategy_MaxDrawdown'].hist(bins=30, alpha=0.7, ax=ax, color='red')
            ax.set_title('最大回撤分布')
            ax.set_xlabel('最大回撤')
            ax.set_ylabel('频数')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无风险指标数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
    
    def plot_sector_distribution(self, ax):
        """绘制行业分布"""
        # 这里可以根据实际数据添加行业分布
        ax.text(0.5, 0.5, '行业分布\n(需要行业数据)', transform=ax.transAxes, 
               ha='center', va='center', fontsize=14)
    
    def plot_market_cap_distribution(self, ax):
        """绘制市值分布"""
        # 这里可以根据实际数据添加市值分布
        ax.text(0.5, 0.5, '市值分布\n(需要市值数据)', transform=ax.transAxes, 
               ha='center', va='center', fontsize=14)
    
    def plot_time_series_performance(self, ax):
        """绘制时间序列表现"""
        if not self.price_data:
            ax.text(0.5, 0.5, '需要价格数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            return
        
        # 计算市场基准（使用所有股票的平均表现）
        all_returns = []
        for ticker, data in self.price_data.items():
            # 尝试获取Adj Close，如果没有则使用Close
            price_column = None
            if 'Adj Close' in data.columns:
                price_column = 'Adj Close'
            elif 'Close' in data.columns:
                price_column = 'Close'
            
            if price_column and len(data) > 1:
                returns = data[price_column].pct_change().dropna()
                all_returns.append(returns)
        
        if all_returns:
            market_returns = pd.concat(all_returns, axis=1).mean(axis=1)
            cumulative_market = (1 + market_returns).cumprod()
            
            ax.plot(cumulative_market.index, cumulative_market.values, 
                   label='市场平均', color='black', linewidth=2)
            ax.set_title('时间序列表现')
            ax.set_xlabel('时间')
            ax.set_ylabel('累计收益')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_metrics_boxplot(self, ax):
        """绘制评估指标箱线图"""
        metrics = ['Final_Score', 'Monthly_Return', 'Strategy_Sharpe']
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        if available_metrics:
            data_for_plot = []
            labels = []
            
            for metric in available_metrics:
                data_for_plot.append(self.df[metric].dropna())
                labels.append(metric)
            
            ax.boxplot(data_for_plot, labels=labels)
            ax.set_title('关键指标分布')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无可用指标数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
    
    def plot_investment_summary(self, ax):
        """绘制投资建议总结"""
        ax.axis('off')
        
        # 生成投资建议
        summary_text = self.generate_investment_summary()
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def generate_investment_summary(self) -> str:
        """生成投资建议总结"""
        summary = "[CHART] 投资建议总结\n\n"
        
        # 基于分析结果生成建议
        if self.category_results:
            best_category = None
            best_return = -float('inf')
            
            for category_name, results in self.category_results.items():
                if 'BUY' in results:
                    buy_return = results['BUY'].get('total_return', 0)
                    if buy_return > best_return:
                        best_return = buy_return
                        best_category = category_name
            
            if best_category:
                summary += f"[CORE] 最佳分类方法: {best_category}\n"
                summary += f"[FINANCE] BUY类别预期收益: {best_return:.2%}\n\n"
        
        # 风险提示
        summary += "[WARNING] 风险提示:\n"
        summary += "• 过往表现不代表未来收益\n"
        summary += "• 建议分散投资降低风险\n"
        summary += "• 定期评估调整投资组合\n"
        
        return summary
    
    def generate_comprehensive_report(self):
        """生成全面的分析报告"""
        print("\n[DATA] 生成全面分析报告...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建Excel报告
        with pd.ExcelWriter(f'{self.result_dir}/comprehensive_analysis_{timestamp}.xlsx') as writer:
            
            # 1. 原始数据
            self.df.to_excel(writer, sheet_name='原始数据', index=False)
            
            # 2. 类别分析结果
            category_summary = []
            for category_name, results in self.category_results.items():
                for cat_type, metrics in results.items():
                    row = {
                        '分类方法': category_name,
                        '类别': cat_type,
                        '股票数量': metrics.get('stock_count', 0),
                        '总收益率': metrics.get('total_return', 0),
                        '年化收益率': metrics.get('annualized_return', 0),
                        '夏普比率': metrics.get('sharpe_ratio', 0),
                        'Sortino比率': metrics.get('sortino_ratio', 0),
                        '最大回撤': metrics.get('max_drawdown', 0),
                        '胜率': metrics.get('win_rate', 0),
                        'VaR_5%': metrics.get('var_5', 0),
                        'CVaR_5%': metrics.get('cvar_5', 0)
                    }
                    category_summary.append(row)
            
            if category_summary:
                pd.DataFrame(category_summary).to_excel(writer, sheet_name='类别分析', index=False)
            
            # 3. 推荐股票列表
            if 'Category_Recommendation' in self.df.columns:
                buy_stocks = self.df[self.df['Category_Recommendation'] == 'BUY'].sort_values(
                    'Final_Score', ascending=False)
                buy_stocks.to_excel(writer, sheet_name='BUY推荐', index=False)
                
                hold_stocks = self.df[self.df['Category_Recommendation'] == 'HOLD'].sort_values(
                    'Final_Score', ascending=False)
                hold_stocks.to_excel(writer, sheet_name='HOLD推荐', index=False)
                
                sell_stocks = self.df[self.df['Category_Recommendation'] == 'SELL'].sort_values(
                    'Final_Score', ascending=True)
                sell_stocks.to_excel(writer, sheet_name='SELL推荐', index=False)
            
            # 4. 因子分析
            factor_analysis = self.perform_factor_analysis()
            if factor_analysis is not None:
                factor_analysis.to_excel(writer, sheet_name='因子分析', index=False)
        
        print(f"[OK] 全面分析报告已生成: comprehensive_analysis_{timestamp}.xlsx")
        return f'{self.result_dir}/comprehensive_analysis_{timestamp}.xlsx'
    
    def perform_factor_analysis(self):
        """执行因子分析"""
        try:
            # 选择数值型列进行因子分析
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            
            # 排除不相关的列
            exclude_columns = ['Unnamed: 0'] if 'Unnamed: 0' in numeric_columns else []
            analysis_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            if len(analysis_columns) < 3:
                return None
            
            # 计算因子重要性
            factor_importance = []
            
            for col in analysis_columns:
                correlation_with_score = self.df[col].corr(self.df['Final_Score'])
                std_dev = self.df[col].std()
                mean_val = self.df[col].mean()
                
                factor_importance.append({
                    '因子名称': col,
                    '与总分相关性': correlation_with_score,
                    '标准差': std_dev,
                    '均值': mean_val,
                    '重要性得分': abs(correlation_with_score) * std_dev
                })
            
            df_factors = pd.DataFrame(factor_importance)
            df_factors = df_factors.sort_values('重要性得分', ascending=False)
            
            return df_factors
            
        except Exception as e:
            print(f"因子分析失败: {e}")
            return None
    
    def run_comprehensive_analysis(self):
        """运行全面分析"""
        print("[GROWTH] 开始全面Category分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 股票分类
        if not self.categorize_stocks():
            return False
        
        # 3. 下载价格数据
        if not self.download_price_data():
            print("[WARNING] 无法下载价格数据，跳过时间序列分析")
        
        # 4. 计算各类别表现
        if not self.calculate_category_performance():
            return False
        
        # 5. 创建可视化
        self.create_comprehensive_visualizations()
        
        # 6. 生成报告
        report_file = self.generate_comprehensive_report()
        
        print(f"\n[OK] 全面Category分析完成！")
        print(f"[DATA] 分析结果保存在: {self.result_dir}")
        print(f"[CHART] 详细报告: {report_file}")
        
        return True

def main():
    """主函数"""
    print("启动全面Category回测分析系统")
    print("=" * 60)
    
    # 检查命令行参数
    import sys
    
    if len(sys.argv) > 1:
        # 使用命令行指定的文件
        data_file = sys.argv[1]
        print(f"[FILE] 使用指定数据文件: {data_file}")
        
        if not os.path.exists(data_file):
            print(f"[ERROR] 文件不存在: {data_file}")
            return
    else:
        # 查找最新的量化分析文件
        import glob
        files = glob.glob("*.xlsx")
        quant_files = [f for f in files if 'quantitative_analysis' in f and not f.startswith('~$')]
        
        if not quant_files:
            print("[ERROR] 未找到量化分析文件")
            return
        
        # 使用最新文件
        data_file = max(quant_files, key=os.path.getmtime)
        print(f"[FILE] 自动选择最新文件: {data_file}")
    
    # 创建分析系统
    analyzer = ComprehensiveCategoryBacktest(
        excel_file=data_file,
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    
    # 运行分析
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n[EMOJI] 分析完成！")
        print("[DATA] 请查看生成的图表和报告文件")
    else:
        print("\n[ERROR] 分析失败")

if __name__ == "__main__":
    main() 