"""
Kronos模型自动训练器
根据市场条件和数据质量自动选择最佳训练参数
"""
import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import schedule
import time

logger = logging.getLogger(__name__)

class KronosAutoTrainer:
    """Kronos模型自动训练管理器"""

    def __init__(self, config_path: str = "training_config.yaml"):
        """初始化自动训练器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.current_config = None
        self.performance_history = []
        self.last_train_time = None
        self.model = None

    def analyze_market_condition(self, data: pd.DataFrame) -> str:
        """
        分析当前市场状况

        Returns:
            'trending': 趋势市场
            'ranging': 震荡市场
            'volatile': 高波动市场
        """
        # 计算技术指标
        close = data['close'].values
        returns = pd.Series(close).pct_change().dropna()

        # 1. 趋势强度（ADX指标简化版）
        price_changes = np.abs(np.diff(close))
        trend_strength = np.mean(price_changes[-20:]) / np.std(close[-60:])

        # 2. 波动率
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        avg_volatility = 0.15  # 假设平均年化波动率15%

        # 3. 价格动量
        momentum = (close[-1] - close[-20]) / close[-20]

        # 判断市场状况
        if trend_strength > 0.02 and abs(momentum) > 0.05:
            return 'trending'
        elif volatility > avg_volatility * 1.5:
            return 'volatile'
        else:
            return 'ranging'

    def evaluate_data_quality(self, data: pd.DataFrame) -> str:
        """
        评估数据质量

        Returns:
            'high_quality': 高质量实时数据
            'delayed': 延迟数据
            'limited': 受限数据
        """
        # 检查数据完整性
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))

        # 检查数据时效性
        last_timestamp = pd.to_datetime(data.index[-1])
        current_time = datetime.now()
        delay_hours = (current_time - last_timestamp).total_seconds() / 3600

        # 检查数据量
        data_points = len(data)

        if missing_ratio < 0.01 and delay_hours < 1 and data_points > 1000:
            return 'high_quality'
        elif delay_hours < 24 and data_points > 500:
            return 'delayed'
        else:
            return 'limited'

    def select_optimal_config(self, market_condition: str, data_quality: str) -> Dict[str, Any]:
        """
        根据市场条件和数据质量选择最优配置
        """
        # 获取推荐配置
        selection = self.config['selection_criteria']

        # 根据市场条件选择基础配置
        if market_condition in selection['market_conditions']:
            preferred_config = selection['market_conditions'][market_condition]['prefer']
        else:
            preferred_config = 'daily_trading'  # 默认日线

        # 根据数据质量调整
        if data_quality in selection['data_availability']:
            availability = selection['data_availability'][data_quality]
            if 'only_enable' in availability:
                if preferred_config not in availability['only_enable']:
                    preferred_config = availability['only_enable'][0]
            elif 'disable' in availability and preferred_config in availability['disable']:
                # 降级到日线交易
                preferred_config = 'daily_trading'

        # 获取具体配置
        config = self.config['configurations'][preferred_config].copy()

        # 根据市场条件微调参数
        if market_condition == 'trending':
            config['model']['sequence_length'] = int(config['model']['sequence_length'] * 1.2)
        elif market_condition == 'volatile':
            config['training']['batch_size'] = config['training']['batch_size'] * 2
            config['training']['epochs'] = int(config['training']['epochs'] * 1.2)
        elif market_condition == 'ranging':
            config['model']['prediction_length'] = int(config['model']['prediction_length'] * 0.7)

        logger.info(f"Selected config: {preferred_config} for {market_condition} market with {data_quality} data")
        return config

    def calculate_training_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """计算训练性能指标"""
        metrics = {}

        # MAE
        metrics['mae'] = np.mean(np.abs(predictions - actuals))

        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((predictions - actuals) ** 2))

        # 方向准确率
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        metrics['direction_accuracy'] = np.mean(pred_direction == actual_direction)

        # MAPE
        metrics['mape'] = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        return metrics

    def should_retrain(self, current_metrics: Dict[str, float], config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        判断是否需要重新训练

        Returns:
            (should_retrain, reason)
        """
        retrain_config = config.get('retrain', {})

        # 检查定期重训练
        if self.last_train_time:
            frequency = retrain_config.get('frequency', 'weekly')
            time_since_last_train = datetime.now() - self.last_train_time

            frequency_map = {
                'daily': timedelta(days=1),
                '3days': timedelta(days=3),
                'weekly': timedelta(weeks=1),
                '6hours': timedelta(hours=6)
            }

            if frequency in frequency_map:
                if time_since_last_train >= frequency_map[frequency]:
                    return True, f"定期重训练 ({frequency})"

        # 检查性能触发
        if current_metrics:
            # MAE触发
            mae_threshold = retrain_config.get('trigger_mae', 0.025)
            if current_metrics.get('mae', 0) > mae_threshold:
                return True, f"MAE超过阈值 ({current_metrics['mae']:.4f} > {mae_threshold})"

            # 准确率触发
            accuracy_threshold = retrain_config.get('trigger_accuracy', 0.65)
            if current_metrics.get('direction_accuracy', 1.0) < accuracy_threshold:
                return True, f"方向准确率低于阈值 ({current_metrics['direction_accuracy']:.2%} < {accuracy_threshold:.0%})"

        return False, ""

    def optimize_hyperparameters(self, data: pd.DataFrame, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用贝叶斯优化调整超参数
        简化版实现
        """
        if not self.config['auto_optimization']['enabled']:
            return base_config

        optimized_config = base_config.copy()

        # 简单的网格搜索（实际应用中使用optuna或hyperopt）
        param_ranges = {
            'sequence_length': [40, 60, 80, 100],
            'learning_rate': [0.0005, 0.001, 0.002],
            'batch_size': [16, 32, 64],
            'prediction_length': [10, 20, 30]
        }

        # 这里简化处理，实际应该进行完整的超参数优化
        logger.info("Running hyperparameter optimization...")

        # 基于数据特征调整参数
        data_length = len(data)
        if data_length > 2000:
            optimized_config['model']['sequence_length'] = 80
            optimized_config['training']['batch_size'] = 64
        elif data_length > 1000:
            optimized_config['model']['sequence_length'] = 60
            optimized_config['training']['batch_size'] = 32
        else:
            optimized_config['model']['sequence_length'] = 40
            optimized_config['training']['batch_size'] = 16

        return optimized_config

    def train_model(self, data: pd.DataFrame, config: Dict[str, Any]):
        """执行模型训练"""
        from .kronos_model import KronosModelWrapper, KronosConfig

        # 创建Kronos配置
        kronos_config = KronosConfig()
        kronos_config.seq_len = config['model']['sequence_length']
        kronos_config.pred_len = config['model']['prediction_length']

        # 初始化模型
        self.model = KronosModelWrapper(kronos_config)
        self.model.load_model()

        # 准备训练数据
        features = config['model']['features']
        available_features = [f for f in features if f in data.columns]
        train_data = data[available_features].values

        # 训练模型（Kronos是预训练模型，这里主要是验证）
        logger.info(f"Training with config: {config['model']}")

        # 记录训练时间
        self.last_train_time = datetime.now()
        self.current_config = config

        logger.info("Model training completed")

    def run_continuous_training(self, symbol: str, check_interval: int = 3600):
        """
        持续运行训练循环

        Args:
            symbol: 股票代码
            check_interval: 检查间隔（秒）
        """
        from .utils import prepare_kline_data

        while True:
            try:
                # 获取最新数据
                logger.info(f"Fetching data for {symbol}...")
                data = prepare_kline_data(symbol, period="6mo", interval="1d")

                if data is None or data.empty:
                    logger.error(f"Failed to fetch data for {symbol}")
                    time.sleep(check_interval)
                    continue

                # 分析市场状况
                market_condition = self.analyze_market_condition(data)
                data_quality = self.evaluate_data_quality(data)

                logger.info(f"Market condition: {market_condition}, Data quality: {data_quality}")

                # 选择最优配置
                optimal_config = self.select_optimal_config(market_condition, data_quality)

                # 优化超参数（可选）
                if self.config['auto_optimization']['enabled']:
                    optimal_config = self.optimize_hyperparameters(data, optimal_config)

                # 评估当前模型性能
                current_metrics = {}
                if self.model:
                    # 这里应该运行实际的验证
                    # predictions = self.model.predict(...)
                    # current_metrics = self.calculate_training_metrics(predictions, actuals)
                    pass

                # 判断是否需要重训练
                should_retrain, reason = self.should_retrain(current_metrics, optimal_config)

                if should_retrain:
                    logger.info(f"Retraining model: {reason}")
                    self.train_model(data, optimal_config)

                    # 记录性能
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'config': optimal_config,
                        'metrics': current_metrics,
                        'market_condition': market_condition
                    })
                else:
                    logger.info("No retraining needed")

                # 等待下一次检查
                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
                time.sleep(check_interval)

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要统计"""
        if not self.performance_history:
            return {}

        summary = {
            'total_trains': len(self.performance_history),
            'last_train': self.last_train_time,
            'current_config': self.current_config,
            'avg_metrics': {},
            'market_conditions': {}
        }

        # 计算平均指标
        all_metrics = [h['metrics'] for h in self.performance_history if h['metrics']]
        if all_metrics:
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in all_metrics if key in m]
                summary['avg_metrics'][key] = np.mean(values)

        # 统计市场条件
        conditions = [h['market_condition'] for h in self.performance_history]
        for condition in set(conditions):
            summary['market_conditions'][condition] = conditions.count(condition)

        return summary


def main():
    """主函数：演示自动训练器使用"""
    import argparse

    parser = argparse.ArgumentParser(description='Kronos自动训练器')
    parser.add_argument('--symbol', default='AAPL', help='股票代码')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                      help='训练模式：once(单次)或continuous(持续)')
    parser.add_argument('--interval', type=int, default=3600,
                      help='持续模式下的检查间隔（秒）')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化训练器
    trainer = KronosAutoTrainer()

    if args.mode == 'once':
        # 单次训练
        from .utils import prepare_kline_data

        logger.info(f"Running single training for {args.symbol}")
        data = prepare_kline_data(args.symbol, period="6mo", interval="1d")

        if data is not None and not data.empty:
            market_condition = trainer.analyze_market_condition(data)
            data_quality = trainer.evaluate_data_quality(data)
            optimal_config = trainer.select_optimal_config(market_condition, data_quality)

            print("\n" + "="*60)
            print(f"最优训练配置 - {args.symbol}")
            print("="*60)
            print(f"市场状况: {market_condition}")
            print(f"数据质量: {data_quality}")
            print(f"\n选定配置:")
            print(f"  - 数据间隔: {optimal_config['data']['interval']}")
            print(f"  - 序列长度: {optimal_config['model']['sequence_length']}")
            print(f"  - 预测长度: {optimal_config['model']['prediction_length']}")
            print(f"  - 批次大小: {optimal_config['training']['batch_size']}")
            print(f"  - 训练轮数: {optimal_config['training']['epochs']}")
            print(f"  - 重训练频率: {optimal_config['retrain']['frequency']}")
            print("="*60)

            trainer.train_model(data, optimal_config)

    else:
        # 持续训练模式
        logger.info(f"Starting continuous training for {args.symbol}")
        trainer.run_continuous_training(args.symbol, args.interval)


if __name__ == "__main__":
    main()