#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected Prediction Exporter
Fixes the critical issues with Excel output format
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from bma_models.simple_25_factor_engine import T5_ALPHA_FACTORS

logger = logging.getLogger(__name__)

class CorrectedPredictionExporter:
    """Corrected prediction exporter with proper data handling"""
    
    def __init__(self, output_dir: str = "D:/trade/results"):
        """Initialize with proper output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_predictions(self,
                         predictions: Union[np.ndarray, pd.Series],
                         dates: Union[pd.Series, np.ndarray],
                         tickers: Union[pd.Series, np.ndarray],
                         model_info: Optional[Dict[str, Any]] = None,
                         filename: Optional[str] = None,
                         constant_threshold: float = 1e-10,
                         lambda_predictions_df: Optional[pd.DataFrame] = None,
                         ridge_predictions_df: Optional[pd.DataFrame] = None,
                         final_predictions_df: Optional[pd.DataFrame] = None,
                         kronos_top35_df: Optional[pd.DataFrame] = None,
                         only_core_sheets: bool = False,
                         minimal_t5_only: bool = False,
                         professional_t5_mode: bool = False) -> str:
        """
        Export predictions with correct format

        Args:
            predictions: Model predictions (return values)
            dates: Date array corresponding to predictions
            tickers: Ticker array corresponding to predictions
            model_info: Optional model metadata
            filename: Optional output filename
            constant_threshold: Threshold for constant prediction detection (default 1e-10)

        Returns:
            Output file path
        """

        try:
            # Coerce None to empty containers to avoid NoneType len errors
            if predictions is None:
                predictions = np.array([])
            if dates is None:
                dates = []
            if tickers is None:
                tickers = []

            # Safe length alignment: slice all to the minimum common length
            try:
                n_pred = len(predictions)
            except Exception:
                # In case predictions is a scalar
                predictions = np.array([predictions])
                n_pred = len(predictions)

            try:
                n_dates = len(dates)
            except Exception:
                dates = [dates]
                n_dates = len(dates)

            try:
                n_tickers = len(tickers)
            except Exception:
                tickers = [tickers]
                n_tickers = len(tickers)

            n = min(int(n_pred), int(n_dates), int(n_tickers))
            if n <= 0:
                # Create minimal dummy to avoid export failure
                predictions = np.array([0.0])
                dates = [datetime.now()]
                tickers = ["NO_DATA"]
                n = 1
            else:
                # Slice all to same length
                if hasattr(predictions, 'values'):
                    predictions = predictions[:n]
                else:
                    predictions = np.array(predictions)[:n]
                if hasattr(dates, 'values'):
                    dates = dates[:n]
                else:
                    dates = list(dates)[:n]
                if hasattr(tickers, 'values'):
                    tickers = tickers[:n]
                else:
                    tickers = list(tickers)[:n]

            logger.info(f"Processing {len(predictions)} predictions for export")

            # Check for constant predictions (warning mode with file marking)
            if hasattr(predictions, 'values'):
                pred_array = predictions.values
            else:
                pred_array = np.array(predictions)

            pred_std = np.std(pred_array)
            is_constant = pred_std < constant_threshold

            if is_constant:
                logger.warning(f"⚠️ CONSTANT PREDICTIONS DETECTED: std={pred_std:.2e} < threshold={constant_threshold}")
                logger.warning("Prediction values appear to be constant. This may indicate a model issue.")

                # Add warning to model_info for file marking
                if model_info is None:
                    model_info = {}
                model_info['WARNING_CONSTANT_PREDICTIONS'] = True
                model_info['prediction_std'] = float(pred_std)

                # Add _CONSTANT suffix to filename
                if filename and not '_CONSTANT' in filename:
                    base, ext = os.path.splitext(filename)
                    filename = f"{base}_CONSTANT{ext}"
            
            # Convert inputs to numpy arrays to avoid index conflicts
            if hasattr(dates, 'values'):
                dates = dates.values
            if hasattr(tickers, 'values'):
                tickers = tickers.values
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            
            # Create results dataframe（生产：导出信号z-score，避免误解为收益率）
            results_df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'ticker': tickers,
                'signal': predictions,
                'signal_zscore': predictions
            })
            
            # Remove any NaN predictions
            initial_count = len(results_df)
            results_df = results_df.dropna(subset=['signal'])
            final_count = len(results_df)
            
            if final_count < initial_count:
                logger.warning(f"Dropped {initial_count - final_count} rows with NaN predictions")
            
            if results_df.empty:
                logger.warning("No valid predictions to export - creating dummy data")
                # Create dummy data to avoid export failure
                results_df = pd.DataFrame({
                    'date': [pd.Timestamp.now()],
                    'ticker': ['NO_DATA'],
                    'signal': [0.0],
                    'signal_zscore': [0.0]
                })
            
            # CRITICAL FIX: Get only one prediction per ticker (the latest date)
            logger.info(f"Before deduplication: {len(results_df)} predictions for {results_df['ticker'].nunique()} tickers")
            
            # Get the latest prediction for each ticker
            results_df = results_df.sort_values(['ticker', 'date'])
            
            # Reset index to avoid conflicts with groupby
            if 'ticker' in results_df.index.names:
                results_df = results_df.reset_index()
            
            latest_predictions = results_df.groupby('ticker').last().reset_index()
            
            logger.info(f"After deduplication: {len(latest_predictions)} predictions (one per ticker)")
            
            # Sort by signal (descending)
            latest_predictions = latest_predictions.sort_values('signal', ascending=False)
            latest_predictions = latest_predictions.reset_index(drop=True)
            
            # Add ranking
            latest_predictions['rank'] = range(1, len(latest_predictions) + 1)
            
            results_df = latest_predictions
            
            # Reorder columns（简化为信号域）
            results_df = results_df[['rank', 'ticker', 'date', 'signal', 'signal_zscore']]
            
            # Format date column
            results_df['date'] = results_df['date'].dt.strftime('%Y-%m-%d')
            
            # Round numeric columns
            results_df['signal'] = results_df['signal'].round(6)
            results_df['signal_zscore'] = results_df['signal_zscore'].round(6)
            
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"bma_predictions_{timestamp}.xlsx"
            
            output_path = os.path.join(self.output_dir, filename)
            
            # 仅输出单表：Final_Predictions
            final_df = results_df[['date', 'ticker', 'signal']].copy()
            final_df.columns = ['Date', 'Ticker', 'Final_Score']
            final_df = final_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
            final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
            # 合并 Kronos 过滤通过信息（如有）
            try:
                kronos_pass_map = None
                if kronos_top35_df is not None and isinstance(kronos_top35_df, pd.DataFrame) and not kronos_top35_df.empty:
                    kdf = kronos_top35_df.copy()
                    pass_col = None
                    for c in ['kronos_pass', 'Kronos_Pass', 'pass', 'passed']:
                        if c in kdf.columns:
                            pass_col = c
                            break
                    if pass_col is not None and 'ticker' in kdf.columns:
                        kronos_pass_map = (
                            kdf.set_index('ticker')[pass_col]
                            .astype(str)
                            .str.upper()
                            .str[0]
                            .to_dict()
                        )
                # 默认未测试标记
                final_df['Kronos_Pass'] = 'N/A'
                if kronos_pass_map:
                    final_df['Kronos_Pass'] = final_df['Ticker'].map(kronos_pass_map).fillna('N/A')
                # 列顺序统一
                final_df = final_df[['Rank', 'Date', 'Ticker', 'Final_Score', 'Kronos_Pass']]
            except Exception as e:
                logger.warning(f"集成Kronos通过列失败（继续导出）: {e}")
            # 统一日期格式
            try:
                final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
            except Exception:
                pass

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Final_Predictions', index=False)
                # Optional: Kronos filter sheets (supports T+10 or T+5 variants)
                try:
                    if kronos_top35_df is not None and isinstance(kronos_top35_df, pd.DataFrame) and not kronos_top35_df.empty:
                        out_df = kronos_top35_df.copy()

                        # Determine variant by available return column
                        if 'kronos_t5_return' in out_df.columns:
                            # T+5 variant with full details
                            # Expected columns: rank, ticker, bma_rank, model_score, t0_price, t5_price, kronos_t5_return, kronos_pass, reason
                            display_cols = []
                            rename_map = {}

                            # Add columns in order with proper renaming
                            if 'rank' in out_df.columns:
                                display_cols.append('rank')
                                rename_map['rank'] = 'Rank'
                            if 'bma_rank' in out_df.columns:
                                display_cols.append('bma_rank')
                                rename_map['bma_rank'] = 'BMA_Rank'
                            if 'ticker' in out_df.columns:
                                display_cols.append('ticker')
                                rename_map['ticker'] = 'Ticker'
                            if 'model_score' in out_df.columns:
                                display_cols.append('model_score')
                                rename_map['model_score'] = 'BMA_Score'
                            if 't0_price' in out_df.columns:
                                display_cols.append('t0_price')
                                rename_map['t0_price'] = 'Current_Price'
                            if 't5_price' in out_df.columns:
                                display_cols.append('t5_price')
                                rename_map['t5_price'] = 'T+5_Predicted_Price'
                            if 'kronos_t5_return' in out_df.columns:
                                display_cols.append('kronos_t5_return')
                                rename_map['kronos_t5_return'] = 'T+5_Return_%'
                            if 'kronos_pass' in out_df.columns:
                                display_cols.append('kronos_pass')
                                rename_map['kronos_pass'] = 'Kronos_Pass'
                            if 'reason' in out_df.columns:
                                display_cols.append('reason')
                                rename_map['reason'] = 'Status'

                            # Select and rename columns
                            if display_cols:
                                out_df = out_df[display_cols].rename(columns=rename_map)

                            # Sort by BMA rank
                            if 'BMA_Rank' in out_df.columns:
                                out_df = out_df.sort_values('BMA_Rank')
                            elif 'Rank' in out_df.columns:
                                out_df = out_df.sort_values('Rank')

                            out_df.to_excel(writer, sheet_name='Kronos_T5_Filter', index=False)
                            logger.info(f"📊 Kronos T+5过滤表已写入: {len(out_df)} 条记录")
                        else:
                            # T+10 variant (legacy support)
                            expected_cols = ['rank', 'ticker', 'model_score', 'kronos_t10_return']
                            for c in expected_cols:
                                if c not in out_df.columns:
                                    out_df[c] = np.nan
                            out_df = out_df[expected_cols]
                            out_df = out_df.rename(columns={
                                'rank': 'Rank',
                                'ticker': 'Ticker',
                                'model_score': 'Model_Score',
                                'kronos_t10_return': 'Kronos_T+10_Return'
                            })
                            out_df.to_excel(writer, sheet_name='Kronos_T10_Pos_Top35', index=False)
                except Exception as e:
                    logger.warning(f"写入Kronos过滤表失败: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            logger.info(f"Predictions exported to: {output_path}")
            logger.info(f"Exported {len(results_df)} predictions")
            logger.info(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
            logger.info(f"Signal range: {results_df['signal'].min():.6f} to {results_df['signal'].max():.6f}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export predictions: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_10d_rebalance_plan(self, results_df: pd.DataFrame, holding_period_days: int = 10, top_k: int = 10,
                                  min_signal: float = 0.0, weighting: str = 'proportional') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a clear 10-day rebalance plan from the latest-day predictions.
        Assumes 'results_df' contains one latest row per ticker with columns ['rank','ticker','date','signal','signal_zscore'].
        """
        if results_df.empty:
            return pd.DataFrame(columns=['as_of_date','planned_exit_date','ticker','rank','signal','weight','action']), pd.DataFrame()
        # As-of date (latest date among predictions)
        as_of_date = pd.to_datetime(results_df['date']).max()
        # Filter to positive signals first; fallback to top-K if none
        df = results_df.copy()
        df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
        df = df.dropna(subset=['signal'])

        if df.empty:
            return pd.DataFrame(columns=['as_of_date','planned_exit_date','ticker','rank','signal','weight','action']), pd.DataFrame()

        pos = df[df['signal'] > min_signal].sort_values('signal', ascending=False)
        if pos.empty:
            selected = df.sort_values('signal', ascending=False).head(max(1, top_k)).copy()
        else:
            selected = pos.head(max(1, top_k)).copy()
        # Compute weights
        if weighting == 'proportional':
            raw = selected['signal'].clip(lower=0.0).values
            total = float(raw.sum())
            if total <= 0:
                weights = np.ones(len(selected)) / len(selected)
                w_method = 'equal'
            else:
                weights = raw / total
                w_method = 'proportional_positive_signal'
        else:
            weights = np.ones(len(selected)) / len(selected)
            w_method = 'equal'
        selected = selected.reset_index(drop=True)
        selected['weight'] = np.round(weights, 6)
        # Planned exit date: add business days
        try:
            from pandas.tseries.offsets import BDay
            exit_date = (as_of_date + BDay(holding_period_days)).date()
        except Exception:
            exit_date = (as_of_date + pd.Timedelta(days=holding_period_days)).date()
        plan_df = selected[['ticker','rank','signal','weight']].copy()
        plan_df.insert(0, 'as_of_date', as_of_date.strftime('%Y-%m-%d'))
        plan_df.insert(1, 'planned_exit_date', pd.to_datetime(exit_date).strftime('%Y-%m-%d'))
        plan_df['action'] = f'Buy/Hold {holding_period_days}B days'
        # Plan summary
        expected_portfolio_signal = float(np.sum(plan_df['signal'].values * plan_df['weight'].values))
        plan_summary = pd.DataFrame([
            ['计划生成日期', as_of_date.strftime('%Y-%m-%d')],
            ['持有期(交易日)', holding_period_days],
            ['持仓数量', len(plan_df)],
            ['加权方式', w_method],
            ['最小纳入信号阈值', min_signal],
            ['预期组合信号(加权)', round(expected_portfolio_signal, 6)]
        ], columns=['指标','数值'])
        return plan_df, plan_summary
    
    def _create_summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        
        pred_returns = results_df['signal']
        
        summary_data = [
            ['总预测数量', len(results_df)],
            ['股票数量', results_df['ticker'].nunique()],
            ['日期范围', f"{results_df['date'].min()} to {results_df['date'].max()}"],
            ['信号 - 平均值', f"{pred_returns.mean():.6f}"],
            ['信号 - 中位数', f"{pred_returns.median():.6f}"],
            ['信号 - 标准差', f"{pred_returns.std():.6f}"],
            ['信号 - 最大值', f"{pred_returns.max():.6f}"],
            ['信号 - 最小值', f"{pred_returns.min():.6f}"],
            ['正预测数量', (pred_returns > 0).sum()],
            ['负预测数量', (pred_returns < 0).sum()],
            ['零预测数量', (pred_returns == 0).sum()],
            ['前10%数量', max(1, len(results_df) // 10)],
            ['前20%数量', max(1, len(results_df) // 5)]
        ]
        
        return pd.DataFrame(summary_data, columns=['指标', '数值'])
    
    def _create_model_info(self, model_info: Optional[Dict], n_predictions: int) -> pd.DataFrame:
        """Create model information sheet"""
        
        if model_info is None:
            model_info = {}
        
        info_data = [
            ['模型类型', model_info.get('model_type', 'BMA Enhanced Model')],
            ['训练样本数', model_info.get('n_samples', 'N/A')],
            ['特征数量', model_info.get('n_features', 'N/A')],
            ['训练时间', model_info.get('training_time', 'N/A')],
            ['CV分数', model_info.get('cv_score', 'N/A')],
            ['IC分数', model_info.get('ic_score', 'N/A')],
            ['模型权重', str(model_info.get('model_weights', 'N/A'))],
            ['预测数量', n_predictions],
            ['导出时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['导出版本', 'Corrected Exporter v1.0']
        ]
        
        return pd.DataFrame(info_data, columns=['参数', '数值'])

    def _create_bottom_10(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create bottom 10 stocks sheet"""
        # Get the bottom 10 stocks by signal
        bottom_10 = results_df.tail(10).copy()

        # Reverse the order so worst is first
        bottom_10 = bottom_10.sort_values('signal', ascending=True)

        # Add a bottom rank column
        bottom_10['bottom_rank'] = range(1, len(bottom_10) + 1)

        # Reorder columns
        bottom_10 = bottom_10[['bottom_rank', 'ticker', 'date', 'signal', 'signal_zscore', 'rank']]

        return bottom_10

    def _create_detailed_model_data(self, model_info: Optional[Dict], results_df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed model data sheet with additional statistics"""

        if model_info is None:
            model_info = {}

        # Extract predictions for additional stats
        pred_returns = results_df['signal']

        # Calculate quantile statistics
        quantiles = pred_returns.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

        # Build detailed model data
        model_data = []

        # Basic model information
        model_data.append(['模型类型', model_info.get('model_type', 'BMA Enhanced Model')])
        model_data.append(['模型版本', model_info.get('model_version', 'v3.0')])
        model_data.append(['训练样本数', model_info.get('n_samples', 'N/A')])
        model_data.append(['特征数量', model_info.get('n_features', 'N/A')])
        model_data.append(['训练时间', model_info.get('training_time', 'N/A')])

        # Performance metrics
        model_data.append(['CV分数', model_info.get('cv_score', 'N/A')])
        model_data.append(['IC分数', model_info.get('ic_score', 'N/A')])
        model_data.append(['Sharpe比率', model_info.get('sharpe_ratio', 'N/A')])
        model_data.append(['最大回撤', model_info.get('max_drawdown', 'N/A')])

        # Model weights if available
        if 'model_weights' in model_info and isinstance(model_info['model_weights'], dict):
            for model_name, weight in model_info['model_weights'].items():
                model_data.append([f'权重-{model_name}', f'{weight:.4f}'])

        # Prediction statistics
        model_data.append(['预测均值', f'{pred_returns.mean():.6f}'])
        model_data.append(['预测标准差', f'{pred_returns.std():.6f}'])
        model_data.append(['预测偏度', f'{pred_returns.skew():.6f}'])
        model_data.append(['预测峰度', f'{pred_returns.kurtosis():.6f}'])

        # Quantiles
        for q_val, q_stat in quantiles.items():
            model_data.append([f'分位数 {q_val:.0%}', f'{q_stat:.6f}'])

        # Signal distribution
        model_data.append(['正信号数量', (pred_returns > 0).sum()])
        model_data.append(['负信号数量', (pred_returns < 0).sum()])
        model_data.append(['零信号数量', (pred_returns == 0).sum()])
        model_data.append(['正信号比例', f'{(pred_returns > 0).mean():.2%}'])

        # Additional metadata
        model_data.append(['预测期(T+N)', model_info.get('prediction_horizon', 'T+5')])
        model_data.append(['数据频率', model_info.get('data_frequency', 'Daily')])
        model_data.append(['特征工程', model_info.get('feature_engineering', 'Enabled')])
        model_data.append(['交叉验证折数', model_info.get('cv_folds', 5)])
        model_data.append(['导出时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        return pd.DataFrame(model_data, columns=['指标', '数值'])

    def _create_all_t5_predictions(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create sheet with all stocks' T+5 predictions"""
        # Copy all predictions
        all_predictions = results_df.copy()

        # Add additional columns for T+5 analysis
        all_predictions['prediction_horizon'] = 'T+5'

        # Categorize signals into buckets
        conditions = [
            (all_predictions['signal'] >= all_predictions['signal'].quantile(0.9)),
            (all_predictions['signal'] >= all_predictions['signal'].quantile(0.7)),
            (all_predictions['signal'] >= all_predictions['signal'].quantile(0.3)),
            (all_predictions['signal'] >= all_predictions['signal'].quantile(0.1)),
            (all_predictions['signal'] < all_predictions['signal'].quantile(0.1))
        ]
        choices = ['强烈买入', '买入', '中性', '卖出', '强烈卖出']
        all_predictions['signal_category'] = np.select(conditions, choices, default='中性')

        # Add percentile rank
        all_predictions['percentile_rank'] = all_predictions['signal'].rank(pct=True).round(4)

        # Add absolute signal strength
        all_predictions['signal_strength'] = all_predictions['signal'].abs()

        # Sort by signal (descending)
        all_predictions = all_predictions.sort_values('signal', ascending=False)

        # Reorder columns
        all_predictions = all_predictions[[
            'rank', 'ticker', 'date', 'signal', 'signal_zscore',
            'signal_category', 'percentile_rank', 'signal_strength', 'prediction_horizon'
        ]]

        # Round numeric columns
        all_predictions['percentile_rank'] = all_predictions['percentile_rank'].round(4)
        all_predictions['signal_strength'] = all_predictions['signal_strength'].round(6)

        return all_predictions

    def _create_factor_contribution(self, model_info: Optional[Dict]) -> pd.DataFrame:
        """Create factor contribution analysis sheet"""

        if model_info is None:
            model_info = {}

        # Get factor contributions if available
        factor_contributions = model_info.get('factor_contributions', {})

        # Check for dual-model contributions (stacking + lambda ranker)
        stacking_contributions = model_info.get('stacking_contributions', {})
        lambda_contributions = model_info.get('lambda_contributions', {})

        if not factor_contributions and not stacking_contributions and not lambda_contributions:
            factor_contributions = {factor: 0.0 for factor in T5_ALPHA_FACTORS}

        # Determine if we have dual-model contributions
        has_dual_models = bool(stacking_contributions and lambda_contributions)

        if has_dual_models:
            # Create dual-column factor contribution table
            factor_data = []

            # Get all unique factors from both models
            all_factors = set(stacking_contributions.keys()) | set(lambda_contributions.keys())

            # Sort by combined absolute contribution
            combined_contributions = {}
            for factor in all_factors:
                stacking_val = stacking_contributions.get(factor, 0)
                lambda_val = lambda_contributions.get(factor, 0)
                combined_contributions[factor] = abs(stacking_val) + abs(lambda_val)

            sorted_factors = sorted(combined_contributions.items(),
                                   key=lambda x: x[1], reverse=True)

            # Calculate totals for percentage calculations
            total_stacking = sum(abs(v) for v in stacking_contributions.values())
            total_lambda = sum(abs(v) for v in lambda_contributions.values())

            for rank, (factor_name, _) in enumerate(sorted_factors, 1):
                stacking_contrib = stacking_contributions.get(factor_name, 0)
                lambda_contrib = lambda_contributions.get(factor_name, 0)

                # Determine factor category
                category = self._get_factor_category(factor_name)

                factor_data.append([
                    rank,
                    factor_name,
                    category,
                    f'{stacking_contrib:.6f}',
                    f'{lambda_contrib:.6f}',
                    f'{(abs(stacking_contrib)/total_stacking*100):.2f}%' if total_stacking > 0 else '0.00%',
                    f'{(abs(lambda_contrib)/total_lambda*100):.2f}%' if total_lambda > 0 else '0.00%',
                    '正向' if (stacking_contrib + lambda_contrib) > 0 else '负向'
                ])

            # Create DataFrame with dual columns
            factor_df = pd.DataFrame(factor_data, columns=[
                '排名', '因子名称', '因子类别',
                'Stacking贡献', 'LambdaRank贡献',
                'Stacking占比', 'LambdaRank占比', '整体方向'
            ])

        else:
            # Single-model contribution table (legacy format)
            factor_data = []

            # Use available single model or fallback to default
            active_contributions = factor_contributions or stacking_contributions or lambda_contributions

            # Sort factors by absolute contribution
            sorted_factors = sorted(active_contributions.items(),
                                   key=lambda x: abs(x[1]),
                                   reverse=True)

            total_contribution = sum(abs(v) for v in active_contributions.values())
            cumulative_contribution = 0

            for rank, (factor_name, contribution) in enumerate(sorted_factors, 1):
                abs_contribution = abs(contribution)
                cumulative_contribution += abs_contribution

                # Determine factor category
                category = self._get_factor_category(factor_name)

                # Direction of contribution
                direction = '正向' if contribution > 0 else '负向'

                factor_data.append([
                    rank,
                    factor_name,
                    category,
                    f'{contribution:.6f}',
                    direction,
                    f'{abs_contribution:.6f}',
                    f'{(abs_contribution/total_contribution*100):.2f}%' if total_contribution > 0 else '0.00%',
                    f'{(cumulative_contribution/total_contribution*100):.2f}%' if total_contribution > 0 else '0.00%'
                ])

            # Create DataFrame
            factor_df = pd.DataFrame(factor_data, columns=[
                '排名', '因子名称', '因子类别', '贡献值', '贡献方向',
                '绝对贡献', '贡献占比', '累计贡献占比'
            ])

        # Add summary statistics at the bottom (adaptive to dual-column format)
        if has_dual_models:
            # Dual-model summary
            summary_df = pd.DataFrame([
                ['', '', '', '', '', '', '', ''],
                ['汇总统计', '', '', '', '', '', '', ''],
                ['总因子数', str(len(all_factors)), '', '', '', '', '', ''],
                ['Stacking正向因子', str(sum(1 for v in stacking_contributions.values() if v > 0)), '', '', '', '', '', ''],
                ['Stacking负向因子', str(sum(1 for v in stacking_contributions.values() if v < 0)), '', '', '', '', '', ''],
                ['LambdaRank正向因子', str(sum(1 for v in lambda_contributions.values() if v > 0)), '', '', '', '', '', ''],
                ['LambdaRank负向因子', str(sum(1 for v in lambda_contributions.values() if v < 0)), '', '', '', '', '', ''],
                ['Stacking总贡献', f'{total_stacking:.6f}', '', '', '', '', '', ''],
                ['LambdaRank总贡献', f'{total_lambda:.6f}', '', '', '', '', '', '']
            ], columns=factor_df.columns)
        else:
            # Single-model summary
            active_contributions = factor_contributions or stacking_contributions or lambda_contributions
            summary_df = pd.DataFrame([
                ['', '', '', '', '', '', '', ''],
                ['汇总统计', '', '', '', '', '', '', ''],
                ['总因子数', str(len(active_contributions)), '', '', '', '', '', ''],
                ['正向因子数', str(sum(1 for v in active_contributions.values() if v > 0)), '', '', '', '', '', ''],
                ['负向因子数', str(sum(1 for v in active_contributions.values() if v < 0)), '', '', '', '', '', ''],
                ['总绝对贡献', f'{total_contribution:.6f}', '', '', '', '', '', ''],
                ['平均贡献', f'{(total_contribution/len(active_contributions)):.6f}' if active_contributions else '0', '', '', '', '', '', ''],
                ['最大正贡献', f'{max(active_contributions.values()):.6f}' if active_contributions else '0', '', '', '', '', '', ''],
                ['最大负贡献', f'{min(active_contributions.values()):.6f}' if active_contributions else '0', '', '', '', '', '', '']
            ], columns=factor_df.columns)

        # Combine main data with summary
        result_df = pd.concat([factor_df, summary_df], ignore_index=True)

        return result_df

    def _get_factor_category(self, factor_name: str) -> str:
        """Determine factor category based on factor name"""
        if 'momentum' in factor_name:
            return '动量因子'
        elif 'volatility' in factor_name or 'beta' in factor_name or 'ivol' in factor_name or 'atr' in factor_name:
            return '风险因子'
        elif any(x in factor_name for x in ['rsi', 'macd', 'ma', 'volume', 'bollinger', 'squeeze', 'obv']):
            return '技术因子'
        elif any(x in factor_name for x in ['value', 'book', 'earnings_yield', 'pe']):
            return '价值因子'
        elif any(x in factor_name for x in ['quality', 'roe', 'roa', 'margin', 'profitability']):
            return '质量因子'
        elif any(x in factor_name for x in ['growth', 'investment']):
            return '成长因子'
        elif 'sentiment' in factor_name:
            return '情绪因子'
        elif 'size' in factor_name:
            return '规模因子'
        elif any(x in factor_name for x in ['liquidity', 'near_52w', 'reversal', 'spike', 'accel', 'gap', 'lottery', 'streak', 'efficiency']):
            return '其他因子'
        else:
            return '其他因子'

# Test function to validate the exporter
def test_corrected_exporter():
    """Test the corrected exporter with sample data"""
    
    logger.info("Testing corrected prediction exporter...")
    
    # Create sample data
    n_samples = 100
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Generate all combinations
    sample_data = []
    for date in dates:
        for ticker in tickers:
            sample_data.append({
                'date': date,
                'ticker': ticker,
                'prediction': np.random.normal(0.02, 0.15)  # 2% mean, 15% volatility
            })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Extract arrays
    predictions = sample_df['prediction'].values
    dates = sample_df['date'].values
    tickers = sample_df['ticker'].values
    
    # Create model info
    model_info = {
        'model_type': 'BMA Enhanced (Test)',
        'n_samples': len(predictions),
        'n_features': 25,
        'training_time': '45.2s',
        'cv_score': 0.034,
        'ic_score': 0.025,
        'model_weights': {'elastic_net': 0.45, 'xgboost': 0.21, 'lightgbm': 0.34}
    }
    
    # Export
    exporter = CorrectedPredictionExporter()
    output_path = exporter.export_predictions(
        predictions=predictions,
        dates=dates, 
        tickers=tickers,
        model_info=model_info,
        filename='test_corrected_predictions.xlsx'
    )
    
    logger.info(f"Test export completed: {output_path}")
    
    # Validate the output
    test_df = pd.read_excel(output_path, sheet_name='Predictions')
    
    logger.info("Validation Results:")
    logger.info(f"  Rows: {len(test_df)}")
    logger.info(f"  Columns: {list(test_df.columns)}")
    logger.info(f"  Top 5 predictions:")
    for i, row in test_df.head().iterrows():
        logger.info(f"    {row['rank']}. {row['ticker']} ({row['date']}): signal={row['signal']:.6f}")
    
    return output_path

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_corrected_exporter()