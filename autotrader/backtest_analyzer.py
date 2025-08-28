"""
Backtest analysis and reporting module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from .backtest_engine import BacktestResult

logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """Analyzes backtest results and generates reports"""
    
    def __init__(self):
        self.results: List[BacktestResult] = []
    
    def add_result(self, result: BacktestResult):
        """Add a backtest result for analysis"""
        self.results.append(result)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report for all results"""
        if not self.results:
            return {}
        
        summary = {
            'total_backtests': len(self.results),
            'average_return': np.mean([r.total_return for r in self.results]),
            'average_sharpe': np.mean([r.sharpe_ratio for r in self.results]),
            'average_max_drawdown': np.mean([r.max_drawdown for r in self.results]),
            'average_win_rate': np.mean([r.win_rate for r in self.results]),
            'best_performing': self._get_best_performing(),
            'worst_performing': self._get_worst_performing(),
        }
        
        return summary
    
    def _get_best_performing(self) -> Dict[str, Any]:
        """Get best performing backtest"""
        if not self.results:
            return {}
        
        best = max(self.results, key=lambda x: x.total_return)
        return {
            'total_return': best.total_return,
            'sharpe_ratio': best.sharpe_ratio,
            'max_drawdown': best.max_drawdown,
            'total_trades': best.total_trades
        }
    
    def _get_worst_performing(self) -> Dict[str, Any]:
        """Get worst performing backtest"""
        if not self.results:
            return {}
        
        worst = min(self.results, key=lambda x: x.total_return)
        return {
            'total_return': worst.total_return,
            'sharpe_ratio': worst.sharpe_ratio,
            'max_drawdown': worst.max_drawdown,
            'total_trades': worst.total_trades
        }
    
    def analyze_risk_metrics(self) -> Dict[str, Any]:
        """Analyze risk metrics across all backtests"""
        if not self.results:
            return {}
        
        returns_list = []
        for result in self.results:
            returns_list.extend(result.daily_returns.tolist())
        
        all_returns = pd.Series(returns_list)
        
        return {
            'volatility': all_returns.std() * np.sqrt(252),
            'skewness': all_returns.skew(),
            'kurtosis': all_returns.kurtosis(),
            'var_95': all_returns.quantile(0.05),
            'var_99': all_returns.quantile(0.01),
            'max_daily_loss': all_returns.min(),
            'max_daily_gain': all_returns.max()
        }
    
    def generate_performance_comparison(self) -> pd.DataFrame:
        """Generate performance comparison table"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for i, result in enumerate(self.results):
            data.append({
                'Backtest': f'Test_{i+1}',
                'Total Return (%)': result.total_return * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown (%)': result.max_drawdown * 100,
                'Win Rate (%)': result.win_rate * 100,
                'Total Trades': result.total_trades
            })
        
        return pd.DataFrame(data)


def analyze_backtest_results(results: List[BacktestResult]) -> Dict[str, Any]:
    """Analyze backtest results and return comprehensive analysis"""
    analyzer = BacktestAnalyzer()
    
    for result in results:
        analyzer.add_result(result)
    
    analysis = {
        'summary': analyzer.generate_summary_report(),
        'risk_metrics': analyzer.analyze_risk_metrics(),
        'performance_comparison': analyzer.generate_performance_comparison().to_dict('records')
    }
    
    logger.info(f"Analysis complete for {len(results)} backtest results")
    return analysis


def export_analysis_to_excel(analysis: Dict[str, Any], filename: str):
    """Export analysis results to Excel file"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([analysis['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk metrics sheet
            risk_df = pd.DataFrame([analysis['risk_metrics']])
            risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
            
            # Performance comparison sheet
            perf_df = pd.DataFrame(analysis['performance_comparison'])
            perf_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)
            
        logger.info(f"Analysis exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export analysis to Excel: {e}")


def generate_backtest_report(results: List[BacktestResult], output_file: Optional[str] = None) -> str:
    """Generate a comprehensive backtest report"""
    analysis = analyze_backtest_results(results)
    
    report = []
    report.append("=== BACKTEST ANALYSIS REPORT ===\n")
    
    # Summary section
    summary = analysis['summary']
    report.append("SUMMARY:")
    report.append(f"Total Backtests: {summary.get('total_backtests', 0)}")
    report.append(f"Average Return: {summary.get('average_return', 0):.2%}")
    report.append(f"Average Sharpe Ratio: {summary.get('average_sharpe', 0):.2f}")
    report.append(f"Average Max Drawdown: {summary.get('average_max_drawdown', 0):.2%}")
    report.append(f"Average Win Rate: {summary.get('average_win_rate', 0):.2%}\n")
    
    # Risk metrics section
    risk = analysis['risk_metrics']
    report.append("RISK METRICS:")
    report.append(f"Annualized Volatility: {risk.get('volatility', 0):.2%}")
    report.append(f"VaR (95%): {risk.get('var_95', 0):.2%}")
    report.append(f"VaR (99%): {risk.get('var_99', 0):.2%}")
    report.append(f"Max Daily Loss: {risk.get('max_daily_loss', 0):.2%}")
    report.append(f"Max Daily Gain: {risk.get('max_daily_gain', 0):.2%}\n")
    
    report_text = "\n".join(report)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_text