#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周度LSTM运行器 - 专为Trading Manager集成设计

功能：
- 每周一开盘前自动运行
- 生成标准化交易信号
- 完全兼容现有Trading Manager
- 保留所有原有功能

使用方法：
1. 手动运行: python weekly_lstm_runner.py
2. 定时任务: 每周日晚或周一早运行
3. Trading Manager调用: 直接import使用

Author: AI Assistant
"""

import sys
import os
import logging
from datetime import datetime, date
import json

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入LSTM交易系统
try:
    from lstm_multi_day_trading_system import WeeklyTradingSystemLSTM, MULTI_DAY_TICKER_LIST, run_weekly_trading_analysis
except ImportError as e:
    print(f"导入LSTM系统失败: {e}")
    print("请确保 lstm_multi_day_trading_system.py 在同目录下")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/weekly_runner_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeeklyLSTMRunner:
    """周度LSTM运行器"""
    
    def __init__(self):
        self.system = WeeklyTradingSystemLSTM(
            prediction_days=5,
            lstm_window=20,
            enable_optimization=False  # 周度运行关闭以提高速度
        )
        
        # 确保输出目录存在
        os.makedirs('weekly_trading_signals', exist_ok=True)
        os.makedirs('result', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def should_run_full_analysis(self) -> bool:
        """判断是否需要运行完整分析"""
        today = date.today()
        weekday = today.weekday()  # 0=Monday
        
        # 周一或周日运行完整分析
        return weekday in [0, 6]
    
    def run_weekly_analysis(self, force_retrain: bool = False) -> dict:
        """运行周度分析"""
        try:
            logger.info("="*80)
            logger.info("启动周度LSTM分析")
            logger.info("="*80)
            
            today = date.today()
            is_full_analysis = self.should_run_full_analysis() or force_retrain
            
            logger.info(f"分析日期: {today}")
            logger.info(f"分析类型: {'完整分析' if is_full_analysis else '快速分析'}")
            
            # 运行分析
            result = self.system.run_weekly_analysis(
                ticker_list=MULTI_DAY_TICKER_LIST,
                days_history=365,
                retrain_model=is_full_analysis
            )
            
            if result['status'] == 'success':
                logger.info("分析成功完成")
                
                # 创建Trading Manager兼容的状态文件
                self._create_status_file(result)
                
                # 输出摘要
                self._print_summary(result)
                
            return result
            
        except Exception as e:
            logger.error(f"分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
    
    def _create_status_file(self, result: dict):
        """创建Trading Manager状态文件"""
        try:
            status_file = 'weekly_trading_signals/lstm_status.json'
            
            status = {
                'last_run': datetime.now().isoformat(),
                'status': result['status'],
                'stocks_analyzed': result.get('total_stocks_analyzed', 0),
                'signals_generated': {
                    'buy': result.get('buy_signals', 0),
                    'sell': result.get('sell_signals', 0),
                    'hold': result.get('hold_signals', 0)
                },
                'latest_signal_file': result['files_generated'].get('json_file'),
                'model_performance': result.get('model_performance', {}),
                'next_recommended_run': self._get_next_run_time()
            }
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"状态文件已更新: {status_file}")
            
        except Exception as e:
            logger.error(f"创建状态文件失败: {e}")
    
    def _get_next_run_time(self) -> str:
        """获取下次推荐运行时间"""
        from datetime import timedelta
        
        today = date.today()
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:  # 今天是周一
            days_until_monday = 7
            
        next_run = today + timedelta(days=days_until_monday)
        return next_run.isoformat()
    
    def _print_summary(self, result: dict):
        """输出结果摘要"""
        print("\n" + "="*60)
        print("周度LSTM分析摘要")
        print("="*60)
        print(f"状态: {result['status'].upper()}")
        print(f"时间: {result['timestamp']}")
        print(f"分析股票: {result['total_stocks_analyzed']} 只")
        print(f"交易信号: 买入 {result['buy_signals']}, 卖出 {result['sell_signals']}, 持有 {result['hold_signals']}")
        print(f"平均置信度: {result['average_confidence']:.2f}")
        
        if result.get('top_recommendation'):
            top = result['top_recommendation']
            print(f"最佳推荐: {top['ticker']} - {top['rating']} (预期收益: {top['expected_return']*100:.2f}%)")
        
        print("\n生成文件:")
        files = result['files_generated']
        if files.get('excel_file'):
            print(f"  Excel报告: {files['excel_file']}")
        if files.get('json_file'):
            print(f"  交易信号: {files['json_file']}")
        if files.get('csv_file'):
            print(f"  CSV文件: {files['csv_file']}")
        
        print(f"\n总共生成 {files.get('total_signals', 0)} 个交易信号")
        print("="*60)
    
    def get_latest_signals(self) -> dict:
        """获取最新交易信号（供Trading Manager调用）"""
        try:
            status_file = 'weekly_trading_signals/lstm_status.json'
            
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                
                signal_file = status.get('latest_signal_file')
                if signal_file and os.path.exists(signal_file):
                    with open(signal_file, 'r', encoding='utf-8') as f:
                        signals = json.load(f)
                    
                    return {
                        'status': 'success',
                        'signals': signals,
                        'last_update': status['last_run']
                    }
            
            return {'status': 'no_signals', 'message': '没有找到最新信号'}
            
        except Exception as e:
            logger.error(f"获取信号失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def check_system_health(self) -> dict:
        """检查系统健康状态"""
        health = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 检查TensorFlow
            try:
                import tensorflow as tf
                health['checks']['tensorflow'] = {
                    'status': 'ok',
                    'version': tf.__version__
                }
            except ImportError:
                health['checks']['tensorflow'] = {
                    'status': 'missing',
                    'message': 'TensorFlow not installed'
                }
                health['status'] = 'warning'
            
            # 检查数据目录
            required_dirs = ['weekly_trading_signals', 'result', 'logs', 'models/weekly_cache']
            for dir_path in required_dirs:
                if os.path.exists(dir_path):
                    health['checks'][f'dir_{dir_path}'] = {'status': 'ok'}
                else:
                    health['checks'][f'dir_{dir_path}'] = {'status': 'missing'}
                    os.makedirs(dir_path, exist_ok=True)
            
            # 检查缓存模型
            model_cache = 'models/weekly_cache/weekly_lstm_model.h5'
            if os.path.exists(model_cache):
                model_age = (datetime.now().timestamp() - os.path.getmtime(model_cache)) / (24*3600)
                health['checks']['model_cache'] = {
                    'status': 'ok',
                    'age_days': round(model_age, 1),
                    'needs_refresh': model_age > 7
                }
            else:
                health['checks']['model_cache'] = {'status': 'missing'}
            
            # 检查最新信号
            signals = self.get_latest_signals()
            health['checks']['latest_signals'] = {
                'status': signals['status'],
                'last_update': signals.get('last_update', 'never')
            }
            
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='周度LSTM运行器')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='强制重新训练模型')
    parser.add_argument('--check-health', action='store_true',
                       help='检查系统健康状态')
    parser.add_argument('--get-signals', action='store_true',
                       help='获取最新交易信号')
    
    args = parser.parse_args()
    
    runner = WeeklyLSTMRunner()
    
    if args.check_health:
        # 检查系统健康
        health = runner.check_system_health()
        print(json.dumps(health, indent=2, ensure_ascii=False))
        
    elif args.get_signals:
        # 获取最新信号
        signals = runner.get_latest_signals()
        print(json.dumps(signals, indent=2, ensure_ascii=False))
        
    else:
        # 运行分析
        result = runner.run_weekly_analysis(force_retrain=args.force_retrain)
        
        if result['status'] == 'success':
            print("\n✅ 周度LSTM分析成功完成")
            print("🔗 系统已准备好与Trading Manager集成")
            
            # 显示集成说明
            print("\n" + "="*60)
            print("Trading Manager集成说明")
            print("="*60)
            print("1. 交易信号文件: weekly_trading_signals/weekly_signals_*.json")
            print("2. 状态文件: weekly_trading_signals/lstm_status.json")
            print("3. Python接口: runner.get_latest_signals()")
            print("4. 建议运行频率: 每周一开盘前")
            print("="*60)
            
        else:
            print(f"\n❌ 分析失败: {result.get('message')}")
            sys.exit(1)


# Trading Manager集成接口
def get_weekly_lstm_signals():
    """Trading Manager调用接口 - 获取最新LSTM信号"""
    runner = WeeklyLSTMRunner()
    return runner.get_latest_signals()


def run_weekly_lstm_analysis(force_retrain=False):
    """Trading Manager调用接口 - 运行LSTM分析"""
    runner = WeeklyLSTMRunner()
    return runner.run_weekly_analysis(force_retrain=force_retrain)


def check_lstm_system_status():
    """Trading Manager调用接口 - 检查LSTM系统状态"""
    runner = WeeklyLSTMRunner()
    return runner.check_system_health()


if __name__ == "__main__":
    main()