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

logger = logging.getLogger(__name__)

class CorrectedPredictionExporter:
    """Corrected prediction exporter with proper data handling"""
    
    def __init__(self, output_dir: str = "D:/trade/production_results"):
        """Initialize with proper output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_predictions(self,
                          predictions: Union[np.ndarray, pd.Series],
                          dates: Union[pd.Series, np.ndarray],
                          tickers: Union[pd.Series, np.ndarray],
                          model_info: Optional[Dict[str, Any]] = None,
                          filename: Optional[str] = None,
                          constant_threshold: float = 1e-10) -> str:
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
            # Validate inputs
            if len(predictions) != len(dates) or len(predictions) != len(tickers):
                raise ValueError(f"Length mismatch: predictions({len(predictions)}), dates({len(dates)}), tickers({len(tickers)})")

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
                raise ValueError("No valid predictions to export")
            
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
            
            # Create summary statistics
            summary_stats = self._create_summary_stats(results_df)
            
            # Create model info sheet
            model_sheet = self._create_model_info(model_info, len(results_df))
            
            # Build 10-day rebalance plan (top-K portfolio held for 10 business days)
            plan_df, plan_summary = self._build_10d_rebalance_plan(results_df, holding_period_days=10, top_k=max(10, len(results_df) // 5))

            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main predictions sheet
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Summary sheet
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                
                # Model info sheet
                model_sheet.to_excel(writer, sheet_name='Model_Info', index=False)
                
                # Top picks sheet (top 20%)
                top_picks = results_df.head(max(10, len(results_df) // 5))
                top_picks.to_excel(writer, sheet_name='Top_Picks', index=False)
                
                # 10-day rebalance plan
                plan_df.to_excel(writer, sheet_name='10D_Rebalance_Plan', index=False)
                plan_summary.to_excel(writer, sheet_name='Plan_Summary', index=False)
            
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