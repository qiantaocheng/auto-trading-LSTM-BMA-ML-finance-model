"""
Triple-Barrier + Meta-Label + OOF Isotonic 标签工程
==================================================

完全替换旧标签产线，输出校准后的expected_alpha_bps和confidence
下游ibkr_auto_trader.plan_and_place_with_rr()无需修改
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
# 🚫 已删除TimeSeriesSplit导入 - 使用统一CV工厂
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class TripleBarrierLabeling:
    """三重障碍标签生成器"""
    
    def __init__(self, tp_sigma: float = 2.0, sl_sigma: float = 2.0, 
                 max_holding_days: int = 5, min_ret_threshold: float = 0.0001):
        self.tp_sigma = tp_sigma          # 止盈倍数
        self.sl_sigma = sl_sigma          # 止损倍数  
        self.max_holding_days = max_holding_days  # 最大持有期
        self.min_ret_threshold = min_ret_threshold  # 最小收益阈值
        
    def compute_daily_volatility(self, adj_close: pd.Series, lookback: int = 20) -> pd.Series:
        """计算每日波动率"""
        returns = adj_close.pct_change()
        return returns.rolling(window=lookback, min_periods=10).std()
        
    def triple_barrier(self, adj_close: pd.Series, vol: pd.Series, 
                      base_signal: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        三重障碍标签生成
        
        Args:
            adj_close: 复权收盘价序列
            vol: 日波动率序列
            base_signal: 基础信号方向(+1/-1/0)，可选
            
        Returns:
            DataFrame with columns: ['event_end', 'actual_ret', 'label_dir', 'barrier_type']
        """
        results = []
        
        for i in range(len(adj_close) - self.max_holding_days):
            t_start = adj_close.index[i]
            px_start = adj_close.iloc[i]
            vol_t = vol.iloc[i]
            
            if pd.isna(px_start) or pd.isna(vol_t) or vol_t <= 0:
                continue
                
            # 设置障碍
            tp_threshold = self.tp_sigma * vol_t  # 止盈阈值
            sl_threshold = -self.sl_sigma * vol_t  # 止损阈值
            
            # 寻找首次触发的障碍
            event_end = None
            actual_ret = 0.0
            barrier_type = 'time'  # 默认时间到期
            
            for j in range(1, min(self.max_holding_days + 1, len(adj_close) - i)):
                t_current = adj_close.index[i + j]
                px_current = adj_close.iloc[i + j]
                
                if pd.isna(px_current):
                    continue
                    
                ret = (px_current - px_start) / px_start
                
                # 检查止盈
                if ret >= tp_threshold:
                    event_end = t_current
                    actual_ret = ret
                    barrier_type = 'tp'
                    break
                    
                # 检查止损
                if ret <= sl_threshold:
                    event_end = t_current  
                    actual_ret = ret
                    barrier_type = 'sl'
                    break
            
            # 如果没有触发，使用时间到期
            if event_end is None:
                end_idx = min(i + self.max_holding_days, len(adj_close) - 1)
                event_end = adj_close.index[end_idx]
                px_end = adj_close.iloc[end_idx]
                if not pd.isna(px_end):
                    actual_ret = (px_end - px_start) / px_start
                    
            # 生成方向标签
            if abs(actual_ret) < self.min_ret_threshold:
                label_dir = 0  # 无显著收益
            elif actual_ret > 0:
                label_dir = 1  # 上涨
            else:
                label_dir = -1  # 下跌
                
            # 如果有基础信号，考虑信号方向
            if base_signal is not None and not pd.isna(base_signal.iloc[i]):
                signal_dir = base_signal.iloc[i]
                # 只有信号方向与实际方向一致时才标记为正确
                if signal_dir != 0 and np.sign(signal_dir) != np.sign(actual_ret):
                    label_dir = 0  # 方向错误
                    
            results.append({
                'start_date': t_start,
                'event_end': event_end,
                'actual_ret': actual_ret,
                'label_dir': label_dir,
                'barrier_type': barrier_type,
                'vol_used': vol_t
            })
            
        return pd.DataFrame(results).set_index('start_date')

class MetaLabelGenerator:
    """Meta-Label生成器 - "该不该执行"的二分类"""
    
    def __init__(self, min_ret_for_exec: float = 0.0005):
        self.min_ret_for_exec = min_ret_for_exec
        
    def make_meta_label(self, base_signal: pd.Series, barrier_labels: pd.DataFrame,
                       strategy_type: str = 'directional') -> pd.Series:
        """
        生成Meta-Label
        
        Args:
            base_signal: 基础信号强度/方向
            barrier_labels: 三重障碍标签结果
            strategy_type: 策略类型 ('directional', 'mean_reverting', 'momentum')
            
        Returns:
            Series: 1表示应该执行，0表示不应该执行
        """
        meta_labels = pd.Series(0, index=base_signal.index, name='meta_label')
        
        for date in barrier_labels.index:
            if date not in base_signal.index:
                continue
                
            signal = base_signal.loc[date]
            barrier_info = barrier_labels.loc[date]
            
            actual_ret = barrier_info['actual_ret']
            label_dir = barrier_info['label_dir']
            
            # 基本条件：有显著信号
            if pd.isna(signal) or abs(signal) < 0.001:
                continue
                
            should_execute = False
            
            if strategy_type == 'directional':
                # 方向性策略：信号方向与实际收益方向一致且收益足够
                if (np.sign(signal) == np.sign(actual_ret) and 
                    abs(actual_ret) >= self.min_ret_for_exec):
                    should_execute = True
                    
            elif strategy_type == 'mean_reverting':
                # 均值回归：信号与实际收益反向且收益足够
                if (np.sign(signal) == -np.sign(actual_ret) and 
                    abs(actual_ret) >= self.min_ret_for_exec):
                    should_execute = True
                    
            elif strategy_type == 'momentum':
                # 动量策略：考虑信号强度与收益幅度
                signal_strength = abs(signal)
                if (np.sign(signal) == np.sign(actual_ret) and 
                    actual_ret * signal_strength > self.min_ret_for_exec):
                    should_execute = True
                    
            meta_labels.loc[date] = 1 if should_execute else 0
            
        return meta_labels

class OOFIsotonicCalibrator:
    """OOF Isotonic校准器 - 仅用OOF数据拟合"""
    
    def __init__(self, n_splits: int = 5, test_size_days: int = 63):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.iso_meta = IsotonicRegression(out_of_bounds='clip')
        self.iso_ret = IsotonicRegression(out_of_bounds='clip') 
        self.oof_metrics = {}
        
    def time_series_oof_split(self, X: pd.DataFrame, gap_days: int = 2) -> List[Tuple]:
        """时间序列OOF分割"""
        dates = X.index.get_level_values(0).unique().sort_values()
        
        splits = []
        total_days = len(dates)
        fold_size = total_days // self.n_splits
        
        for i in range(self.n_splits):
            test_start_idx = i * fold_size
            test_end_idx = min((i + 1) * fold_size, total_days)
            
            # 训练集：测试集之前的数据，留gap避免泄露
            train_end_idx = max(0, test_start_idx - gap_days)
            
            if train_end_idx > 0:
                train_dates = dates[:train_end_idx]
                test_dates = dates[test_start_idx:test_end_idx]
                
                train_mask = X.index.get_level_values(0).isin(train_dates)
                test_mask = X.index.get_level_values(0).isin(test_dates)
                
                splits.append((train_mask, test_mask))
                
        return splits
        
    def fit_and_calibrate(self, X: pd.DataFrame, y_ret: pd.Series, y_meta: pd.Series,
                         base_model_ret=None, base_model_meta=None) -> Dict:
        """
        OOF训练和校准
        
        Args:
            X: 特征矩阵 (MultiIndex: date, symbol)
            y_ret: 收益目标
            y_meta: Meta-Label目标
            base_model_ret: 收益预测模型
            base_model_meta: Meta-Label分类模型
            
        Returns:
            Dict: 校准结果和OOF预测
        """
        if base_model_ret is None:
            base_model_ret = LinearRegression()
        if base_model_meta is None:
            base_model_meta = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            
        # 对齐数据
        common_idx = X.index.intersection(y_ret.index).intersection(y_meta.index)
        X_aligned = X.loc[common_idx]
        y_ret_aligned = y_ret.loc[common_idx]
        y_meta_aligned = y_meta.loc[common_idx]
        
        # 存储OOF预测
        oof_ret_pred = pd.Series(np.nan, index=common_idx)
        oof_meta_pred = pd.Series(np.nan, index=common_idx)
        
        splits = self.time_series_oof_split(X_aligned)
        
        logger.info(f"开始OOF训练，{len(splits)}个fold")
        
        for fold_idx, (train_mask, test_mask) in enumerate(splits):
            try:
                X_train, X_test = X_aligned[train_mask], X_aligned[test_mask]
                y_ret_train, y_ret_test = y_ret_aligned[train_mask], y_ret_aligned[test_mask]
                y_meta_train, y_meta_test = y_meta_aligned[train_mask], y_meta_aligned[test_mask]
                
                # 移除NaN
                valid_train = ~(pd.isna(y_ret_train) | pd.isna(y_meta_train))
                valid_test = ~(pd.isna(y_ret_test) | pd.isna(y_meta_test))
                
                if valid_train.sum() < 10 or valid_test.sum() < 5:
                    continue
                    
                X_train_clean = X_train[valid_train].fillna(0)
                X_test_clean = X_test[valid_test].fillna(0)
                
                # 训练收益模型
                base_model_ret.fit(X_train_clean, y_ret_train[valid_train])
                ret_pred = base_model_ret.predict(X_test_clean)
                oof_ret_pred[y_ret_test[valid_test].index] = ret_pred
                
                # 训练Meta模型
                base_model_meta.fit(X_train_clean, y_meta_train[valid_train])
                if hasattr(base_model_meta, 'predict_proba'):
                    meta_pred = base_model_meta.predict_proba(X_test_clean)[:, 1]
                else:
                    meta_pred = base_model_meta.predict(X_test_clean)
                oof_meta_pred[y_meta_test[valid_test].index] = meta_pred
                
                logger.info(f"Fold {fold_idx+1}/{len(splits)} 完成")
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx+1} 训练失败: {e}")
                continue
        
        # 仅用OOF预测拟合Isotonic回归
        valid_oof = ~(pd.isna(oof_ret_pred) | pd.isna(oof_meta_pred) | 
                      pd.isna(y_ret_aligned) | pd.isna(y_meta_aligned))
        
        if valid_oof.sum() < 50:
            logger.warning("OOF有效样本不足，使用简单校准")
            # 使用简单线性校准作为回退
            self.iso_meta = lambda x: np.clip(x, 0, 1)
            self.iso_ret = lambda x: x
        else:
            oof_ret_clean = oof_ret_pred[valid_oof]
            oof_meta_clean = oof_meta_pred[valid_oof] 
            y_ret_clean = y_ret_aligned[valid_oof]
            y_meta_clean = y_meta_aligned[valid_oof]
            
            # 拟合Isotonic - 仅用OOF数据！
            self.iso_meta.fit(oof_meta_clean, y_meta_clean)
            self.iso_ret.fit(oof_ret_clean, y_ret_clean)
            
            # 计算校准指标
            meta_calibrated = self.iso_meta.predict(oof_meta_clean)
            ret_calibrated = self.iso_ret.predict(oof_ret_clean)
            
            from sklearn.metrics import roc_auc_score, mean_squared_error
            try:
                meta_auc = roc_auc_score(y_meta_clean, meta_calibrated)
                ret_mse = mean_squared_error(y_ret_clean, ret_calibrated)
                
                self.oof_metrics = {
                    'meta_auc_calibrated': meta_auc,
                    'ret_mse_calibrated': ret_mse,
                    'n_oof_samples': valid_oof.sum(),
                    'calibration_r2_meta': np.corrcoef(meta_calibrated, y_meta_clean)[0,1]**2,
                    'calibration_r2_ret': np.corrcoef(ret_calibrated, y_ret_clean)[0,1]**2
                }
                logger.info(f"OOF校准完成: Meta AUC={meta_auc:.3f}, Ret MSE={ret_mse:.4f}")
            except Exception as e:
                logger.warning(f"校准指标计算失败: {e}")
        
        return {
            'oof_ret_pred': oof_ret_pred,
            'oof_meta_pred': oof_meta_pred,
            'iso_meta': self.iso_meta,
            'iso_ret': self.iso_ret,
            'oof_metrics': self.oof_metrics
        }
    
    def predict_calibrated(self, ret_pred: np.ndarray, meta_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用校准器预测"""
        ret_calibrated = self.iso_ret.predict(ret_pred)
        meta_calibrated = self.iso_meta.predict(meta_pred)
        return ret_calibrated, meta_calibrated

class EnhancedLabelingPipeline:
    """增强标签产线 - 集成三重障碍 + Meta + OOF校准"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'tp_sigma': 2.0,
            'sl_sigma': 2.0, 
            'max_holding_days': 5,
            'min_ret_threshold': 0.0005,
            'min_ret_for_exec': 0.001,
            'strategy_type': 'directional',
            'vol_lookback': 20,
            'oof_n_splits': 5,
            'require_oof_validation': True
        }
        
        self.config = {**default_config, **(config or {})}
        
        self.barrier_labeler = TripleBarrierLabeling(
            tp_sigma=self.config['tp_sigma'],
            sl_sigma=self.config['sl_sigma'],
            max_holding_days=self.config['max_holding_days'],
            min_ret_threshold=self.config['min_ret_threshold']
        )
        
        self.meta_labeler = MetaLabelGenerator(
            min_ret_for_exec=self.config['min_ret_for_exec']
        )
        
        self.oof_calibrator = OOFIsotonicCalibrator(
            n_splits=self.config['oof_n_splits']
        )
        
    def generate_labels_and_train(self, adj_close: pd.DataFrame, features: pd.DataFrame,
                                 base_signals: Optional[pd.DataFrame] = None) -> Dict:
        """
        生成标签并训练校准器
        
        Args:
            adj_close: 复权价格 (DataFrame: date x symbol)
            features: 特征矩阵 (MultiIndex: (date, symbol) x features)
            base_signals: 基础信号 (DataFrame: date x symbol)
            
        Returns:
            Dict: 训练结果和校准器
        """
        logger.info("开始增强标签生成流程")
        
        all_labels = []
        all_meta_labels = []
        
        for symbol in adj_close.columns:
            try:
                # 单个股票的价格和信号
                symbol_prices = adj_close[symbol].dropna()
                symbol_signal = base_signals[symbol].dropna() if base_signals is not None else None
                
                if len(symbol_prices) < 100:  # 需要足够的历史数据
                    continue
                    
                # 计算波动率
                vol = self.barrier_labeler.compute_daily_volatility(
                    symbol_prices, self.config['vol_lookback']
                )
                
                # 生成三重障碍标签
                barrier_labels = self.barrier_labeler.triple_barrier(
                    symbol_prices, vol, symbol_signal
                )
                
                if len(barrier_labels) == 0:
                    continue
                    
                # 生成Meta标签
                if symbol_signal is not None:
                    meta_labels = self.meta_labeler.make_meta_label(
                        symbol_signal, barrier_labels, self.config['strategy_type']
                    )
                else:
                    # 如果没有基础信号，用收益绝对值作为执行标准
                    meta_labels = (barrier_labels['actual_ret'].abs() >= 
                                 self.config['min_ret_for_exec']).astype(int)
                
                # 添加symbol信息
                barrier_labels['symbol'] = symbol
                meta_labels_df = pd.DataFrame({
                    'symbol': symbol,
                    'meta_label': meta_labels
                }, index=meta_labels.index)
                
                all_labels.append(barrier_labels)
                all_meta_labels.append(meta_labels_df)
                
                logger.info(f"{symbol}: 生成{len(barrier_labels)}个标签")
                
            except Exception as e:
                logger.warning(f"处理{symbol}时出错: {e}")
                continue
        
        if not all_labels:
            raise ValueError("没有生成任何有效标签")
            
        # 合并所有标签
        combined_labels = pd.concat(all_labels, ignore_index=False)
        combined_meta = pd.concat(all_meta_labels, ignore_index=False)
        
        # 重建MultiIndex用于与特征对齐
        label_index = pd.MultiIndex.from_arrays([
            combined_labels.index, combined_labels['symbol']
        ], names=['date', 'symbol'])
        
        meta_index = pd.MultiIndex.from_arrays([
            combined_meta.index, combined_meta['symbol']  
        ], names=['date', 'symbol'])
        
        y_ret = pd.Series(combined_labels['actual_ret'].values, index=label_index)
        y_meta = pd.Series(combined_meta['meta_label'].values, index=meta_index)
        
        # OOF训练和校准
        logger.info("开始OOF训练和校准")
        calibration_result = self.oof_calibrator.fit_and_calibrate(
            features, y_ret, y_meta
        )
        
        # 验证校准质量
        if self.config['require_oof_validation']:
            metrics = calibration_result['oof_metrics']
            if (metrics.get('meta_auc_calibrated', 0) < 0.52 or 
                metrics.get('n_oof_samples', 0) < 100):
                logger.warning("OOF校准质量不足，请检查数据或模型")
        
        return {
            'barrier_labels': combined_labels,
            'y_ret': y_ret,
            'y_meta': y_meta,
            'calibration_result': calibration_result,
            'config': self.config,
            'pipeline': self
        }
    
    def make_signal_payload(self, symbol: str, ret_pred: float, meta_pred: float,
                           px_ref: float) -> Dict:
        """
        生成信号载荷，输出到下游接口
        
        Args:
            symbol: 股票代码
            ret_pred: 原始收益预测
            meta_pred: 原始Meta预测
            px_ref: 参考价格
            
        Returns:
            Dict: 符合plan_and_place_with_rr接口的信号载荷
        """
        # 使用校准器校准预测
        ret_calibrated, meta_calibrated = self.oof_calibrator.predict_calibrated(
            np.array([ret_pred]), np.array([meta_pred])
        )
        
        expected_alpha_bps = float(ret_calibrated[0] * 10000)  # 转换为bps
        confidence = float(np.clip(meta_calibrated[0], 0.01, 0.99))  # 校准后的执行概率
        
        # 确定交易方向
        side = "BUY" if expected_alpha_bps > 0 else "SELL"
        
        return {
            "symbol": symbol,
            "side": side,
            "expected_alpha_bps": abs(expected_alpha_bps),  # 下游期望正值
            "confidence": confidence,
            "reference_price": px_ref,
            "signal_timestamp": datetime.now(),
            "calibration_source": "oof_isotonic"
        }

# 工厂函数
def create_enhanced_labeling_pipeline(config: Optional[Dict] = None) -> EnhancedLabelingPipeline:
    """创建增强标签产线"""
    return EnhancedLabelingPipeline(config)

# 使用示例
def example_usage():
    """使用示例"""
    # 配置
    config = {
        'tp_sigma': 2.5,
        'sl_sigma': 2.0,
        'max_holding_days': 5,
        'strategy_type': 'directional',
        'require_oof_validation': True
    }
    
    # 创建产线
    pipeline = create_enhanced_labeling_pipeline(config)
    
    # 假设有数据
    # adj_close: DataFrame (date x symbol)
    # features: MultiIndex DataFrame ((date, symbol) x features)
    # base_signals: DataFrame (date x symbol) - 可选
    
    # 训练
    # result = pipeline.generate_labels_and_train(adj_close, features, base_signals)
    
    # 实时预测
    # signal_payload = pipeline.make_signal_payload("AAPL", 0.015, 0.75, 150.0)
    # 然后传给: trader.plan_and_place_with_rr(**signal_payload)
    
    return pipeline