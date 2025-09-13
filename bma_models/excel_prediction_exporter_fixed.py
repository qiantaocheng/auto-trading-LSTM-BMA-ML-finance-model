#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] Fixed BMA Excel Exporter - Resolves all data format issues

⚠️ WARNING: This module is DEPRECATED and will be removed in future versions.
Please use bma_models.corrected_prediction_exporter.CorrectedPredictionExporter instead.

This module is kept only for backward compatibility.
Migration guide:
  Old: from bma_models.excel_prediction_exporter_fixed import BMAExcelExporterFixed
  New: from bma_models.corrected_prediction_exporter import CorrectedPredictionExporter
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, Optional, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class BMAExcelExporterFixed:
    """[DEPRECATED] Fixed BMA Excel Exporter with comprehensive error handling"""

    def __init__(self, output_dir: str = "D:/trade/results"):
        """Initialize exporter with validation"""
        import warnings
        warnings.warn(
            "BMAExcelExporterFixed is deprecated. Please use CorrectedPredictionExporter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.output_dir = output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            self.output_dir = "."  # Fallback to current directory
        
    def _validate_inputs(self, predictions: Union[np.ndarray, list], 
                        feature_data: pd.DataFrame, 
                        model_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate inputs before processing"""
        try:
            # Check predictions
            if predictions is None:
                return False, "Predictions is None"
            
            if isinstance(predictions, (list, tuple)):
                predictions = np.array(predictions)
            
            if not isinstance(predictions, np.ndarray):
                return False, "Predictions must be numpy array or list"
            
            if len(predictions) == 0:
                return False, "Predictions array is empty"
            
            # Check feature_data
            if feature_data is None or feature_data.empty:
                return False, "Feature data is empty"
            
            if len(predictions) != len(feature_data):
                return False, f"Length mismatch: predictions({len(predictions)}) != feature_data({len(feature_data)})"
            
            # Check model_info
            if not isinstance(model_info, dict):
                return False, "Model info must be dictionary"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _prepare_results_dataframe(self, predictions: np.ndarray, 
                                  feature_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare results DataFrame with error handling"""
        try:
            # Create base results DataFrame
            results_df = feature_data.copy()
            results_df = results_df.reset_index(drop=True)  # Ensure clean index
            
            # Add predictions
            results_df['predicted_return'] = predictions[:len(results_df)]
            results_df['predicted_return_pct'] = results_df['predicted_return'] * 100
            
            # Sort by predictions (handle NaN values)
            results_df = results_df.sort_values('predicted_return', ascending=False, na_position='last')
            results_df = results_df.reset_index(drop=True)
            
            # Add ranking
            results_df['rank'] = range(1, len(results_df) + 1)
            
            # Reorganize columns
            main_cols = ['rank', 'predicted_return_pct', 'predicted_return']
            
            # Add ticker column if it exists
            if 'ticker' in results_df.columns:
                main_cols.insert(1, 'ticker')
            
            # Add date column if it exists
            if 'date' in results_df.columns:
                main_cols.insert(-2, 'date')
            
            # Get remaining columns
            other_cols = [col for col in results_df.columns if col not in main_cols]
            
            # Reorder columns
            final_cols = main_cols + other_cols
            results_df = results_df[[col for col in final_cols if col in results_df.columns]]
            
            return results_df
            
        except Exception as e:
            logger.error(f"Failed to prepare results DataFrame: {e}")
            # Fallback: simple DataFrame
            return pd.DataFrame({
                'rank': range(1, len(predictions) + 1),
                'predicted_return': predictions,
                'predicted_return_pct': predictions * 100
            })
    
    def _create_model_info_sheet(self, model_info: Dict[str, Any]) -> pd.DataFrame:
        """Create model info sheet with safe access"""
        try:
            info_data = [
                ['模型类型', model_info.get('model_type', 'BMA Enhanced Fixed')],
                ['训练时间', model_info.get('training_time', 'N/A')],
                ['样本数量', model_info.get('n_samples', 'N/A')],
                ['特征数量', model_info.get('n_features', 'N/A')],
                ['最佳模型', model_info.get('best_model', 'N/A')],
                ['CV分数', model_info.get('cv_score', 'N/A')],
                ['IC分数', model_info.get('ic_score', 'N/A')],
                ['R²分数', model_info.get('r2_score', 'N/A')],
                ['导出时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            return pd.DataFrame(info_data, columns=['指标', '数值'])
            
        except Exception as e:
            logger.error(f"Failed to create model info sheet: {e}")
            return pd.DataFrame({
                '指标': ['错误'],
                '数值': [f'创建失败: {e}']
            })
    
    def _create_summary_sheet(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics sheet"""
        try:
            if results_df.empty:
                return pd.DataFrame({'统计项': ['无数据'], '数值': ['N/A']})
            
            pred_col = 'predicted_return_pct'
            if pred_col not in results_df.columns:
                return pd.DataFrame({'统计项': ['错误'], '数值': ['缺少预测列']})
            
            predictions = results_df[pred_col].dropna()
            
            if len(predictions) == 0:
                return pd.DataFrame({'统计项': ['无有效预测'], '数值': ['N/A']})
            
            summary_data = [
                ['总股票数', len(results_df)],
                ['有效预测数', len(predictions)],
                ['预测收益率均值(%)', f"{predictions.mean():.4f}"],
                ['预测收益率中位数(%)', f"{predictions.median():.4f}"],
                ['预测收益率标准差(%)', f"{predictions.std():.4f}"],
                ['最高预测收益率(%)', f"{predictions.max():.4f}"],
                ['最低预测收益率(%)', f"{predictions.min():.4f}"],
                ['前10%股票数量', max(1, len(results_df) // 10)],
                ['前20%股票数量', max(1, len(results_df) // 5)]
            ]
            
            return pd.DataFrame(summary_data, columns=['统计项', '数值'])
            
        except Exception as e:
            logger.error(f"Failed to create summary sheet: {e}")
            return pd.DataFrame({
                '统计项': ['错误'],
                '数值': [f'创建失败: {e}']
            })
    
    def export_predictions(self, 
                          predictions: Union[np.ndarray, list],
                          feature_data: pd.DataFrame,
                          model_info: Dict[str, Any],
                          filename: Optional[str] = None) -> str:
        """
        Export predictions to Excel with comprehensive error handling
        
        Args:
            predictions: Prediction array/list
            feature_data: Original feature data
            model_info: Model information dictionary
            filename: Output filename (optional)
            
        Returns:
            Output file path
        """
        
        try:
            # Validate inputs
            is_valid, error_msg = self._validate_inputs(predictions, feature_data, model_info)
            if not is_valid:
                raise ValueError(f"Input validation failed: {error_msg}")
            
            # Convert predictions to numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"BMA_Fixed_Predictions_{timestamp}.xlsx"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Prepare results
            results_df = self._prepare_results_dataframe(predictions, feature_data)
            model_info_df = self._create_model_info_sheet(model_info)
            summary_df = self._create_summary_sheet(results_df)
            
            # Write to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main predictions sheet
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Model info sheet
                model_info_df.to_excel(writer, sheet_name='Model_Info', index=False)
                
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # By ticker analysis (if ticker column exists)
                if 'ticker' in results_df.columns and len(results_df) > 0:
                    try:
                        ticker_stats = results_df.groupby('ticker').agg({
                            'predicted_return_pct': ['mean', 'std', 'count'],
                            'rank': 'mean'
                        }).round(4)
                        
                        ticker_stats.columns = ['平均预测收益率(%)', '收益率标准差(%)', '记录数', '平均排名']
                        ticker_stats = ticker_stats.sort_values('平均预测收益率(%)', ascending=False)
                        ticker_stats.to_excel(writer, sheet_name='By_Ticker')
                    except Exception as e:
                        logger.warning(f"Failed to create ticker analysis: {e}")
            
            # Log success
            logger.info(f"预测结果已导出至: {filepath}")
            logger.info(f"共导出 {len(results_df)} 条预测记录")
            
            if len(predictions) > 0:
                logger.info(f"预测收益率范围: {predictions.min():.4f} 到 {predictions.max():.4f}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            
            # Try fallback export
            try:
                fallback_filename = f"BMA_Fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                fallback_path = os.path.join(self.output_dir, fallback_filename)
                
                # Simple CSV export
                pd.DataFrame({
                    'predicted_return': predictions,
                    'rank': range(1, len(predictions) + 1)
                }).to_csv(fallback_path, index=False)
                
                logger.info(f"Fallback CSV export successful: {fallback_path}")
                return fallback_path
                
            except Exception as fallback_error:
                logger.error(f"Fallback export also failed: {fallback_error}")
                raise Exception(f"Both main and fallback exports failed. Main: {e}, Fallback: {fallback_error}")


# Convenience function
def export_bma_predictions_fixed(predictions: Union[np.ndarray, list],
                                feature_data: pd.DataFrame, 
                                model_info: Dict[str, Any],
                                output_dir: str = "D:/trade/predictions",
                                filename: Optional[str] = None) -> str:
    """
    Fixed convenience function for BMA prediction export
    
    Args:
        predictions: Prediction values
        feature_data: Feature DataFrame
        model_info: Model information
        output_dir: Output directory
        filename: Custom filename
        
    Returns:
        Output file path
    """
    
    try:
        exporter = BMAExcelExporterFixed(output_dir)
        return exporter.export_predictions(predictions, feature_data, model_info, filename)
    except Exception as e:
        logger.error(f"Fixed export function failed: {e}")
        raise


def test_fixed_excel_export():
    """Test the fixed Excel export functionality"""
    try:
        print("Testing Fixed Excel Export...")
        
        # Create test data
        n_samples = 100
        predictions = np.random.randn(n_samples) * 0.02
        
        # Create proper ticker list
        base_tickers = ['AAPL', 'MSFT', 'GOOGL']
        ticker_list = (base_tickers * (n_samples // len(base_tickers) + 1))[:n_samples]
        
        feature_data = pd.DataFrame({
            'ticker': ticker_list,
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'Close': np.random.uniform(100, 200, n_samples),
            'Volume': np.random.randint(1000000, 10000000, n_samples)
        })
        
        model_info = {
            'model_type': 'BMA Fixed Test',
            'training_time': 45.2,
            'n_samples': n_samples,
            'n_features': 4,
            'best_model': 'Ridge',
            'r2_score': 0.123
        }
        
        # Test export
        output_file = export_bma_predictions_fixed(
            predictions=predictions,
            feature_data=feature_data,
            model_info=model_info,
            output_dir="D:/trade/test_results",
            filename="fixed_excel_test.xlsx"
        )
        
        if os.path.exists(output_file):
            print(f"[OK] Fixed Excel export test PASSED: {output_file}")
            return True
        else:
            print("[FAIL] Fixed Excel export test FAILED: File not created")
            return False

    except Exception as e:
        print(f"[ERROR] Fixed Excel export test ERROR: {e}")
        return False


if __name__ == "__main__":
    # Test the fixed export
    success = test_fixed_excel_export()
    print(f"Test result: {'PASSED' if success else 'FAILED'}")