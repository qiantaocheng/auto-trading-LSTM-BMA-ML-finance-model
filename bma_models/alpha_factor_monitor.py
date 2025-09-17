#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Factor Calculation Monitor
Comprehensive monitoring for alpha factor calculations and MultiIndex handling
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import traceback

class AlphaFactorMonitor:
    """Monitor alpha factor calculations and MultiIndex operations"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the monitoring system"""
        self.logger = logger or logging.getLogger(__name__)
        self.factor_timings = {}
        self.factor_status = {}
        self.multiindex_validations = []
        self.calculation_start_time = None
        self.total_factors = 25
        
        # Factor groups for organized monitoring
        self.factor_groups = {
            'momentum': ['rsi', 'macd', 'stoch', 'williams_r', 'roc'],
            'mean_reversion': ['bollinger_position', 'rsi_divergence', 'price_vs_ma'],
            'volume': ['volume_ma_ratio', 'volume_price_trend', 'on_balance_volume'],
            'volatility': ['atr', 'volatility_ratio', 'garch_vol'],
            'trend': ['adx', 'aroon', 'trend_strength', 'breakout'],
            'statistical': ['zscore', 'skewness', 'kurtosis', 'correlation'],
            'technical': ['support_resistance', 'fibonacci', 'pivot_points']
        }
    
    def start_factor_calculation_monitoring(self):
        """Start monitoring the factor calculation process"""
        self.calculation_start_time = time.time()
        self.factor_timings = {}
        self.factor_status = {}
        
        self.logger.info("=" * 80)
        self.logger.info("🔍 ALPHA FACTOR CALCULATION MONITORING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Total factors to calculate: {self.total_factors}")
        self.logger.info(f"🏗️ Factor groups: {list(self.factor_groups.keys())}")
        self.logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
    
    def log_factor_start(self, factor_name: str, group: str = "unknown"):
        """Log the start of a factor calculation"""
        start_time = time.time()
        self.factor_timings[factor_name] = {'start': start_time, 'group': group}
        self.factor_status[factor_name] = 'calculating'
        
        # Find which factors are completed
        completed = len([f for f, s in self.factor_status.items() if s == 'completed'])
        progress = (completed / self.total_factors) * 100
        
        self.logger.info(f"⚙️ [{completed}/{self.total_factors}] [{progress:5.1f}%] Calculating {factor_name} ({group} group)...")
    
    def log_factor_completion(self, factor_name: str, data_shape: Tuple[int, int], 
                            validation_result: Dict[str, Any]):
        """Log the completion of a factor calculation"""
        if factor_name not in self.factor_timings:
            self.logger.warning(f"⚠️ Factor {factor_name} completion logged without start")
            return
        
        end_time = time.time()
        duration = end_time - self.factor_timings[factor_name]['start']
        self.factor_timings[factor_name]['end'] = end_time
        self.factor_timings[factor_name]['duration'] = duration
        self.factor_status[factor_name] = 'completed'
        
        # Validation status
        is_valid = validation_result.get('is_valid', False)
        nan_count = validation_result.get('nan_count', 0)
        inf_count = validation_result.get('inf_count', 0)
        range_info = validation_result.get('range', 'N/A')
        
        status = "✅" if is_valid else "❌"
        
        completed = len([f for f, s in self.factor_status.items() if s == 'completed'])
        progress = (completed / self.total_factors) * 100
        
        self.logger.info(f"{status} [{completed}/{self.total_factors}] [{progress:5.1f}%] {factor_name} completed in {duration:.2f}s")
        self.logger.info(f"    📊 Shape: {data_shape}, NaN: {nan_count}, Inf: {inf_count}, Range: {range_info}")
    
    def log_factor_error(self, factor_name: str, error: Exception):
        """Log a factor calculation error"""
        if factor_name in self.factor_timings:
            end_time = time.time()
            duration = end_time - self.factor_timings[factor_name]['start']
            self.factor_timings[factor_name]['duration'] = duration
        
        self.factor_status[factor_name] = 'failed'
        
        completed = len([f for f, s in self.factor_status.items() if s == 'completed'])
        failed = len([f for f, s in self.factor_status.items() if s == 'failed'])
        progress = (completed / self.total_factors) * 100
        
        self.logger.error(f"❌ [{completed}/{self.total_factors}] [{progress:5.1f}%] {factor_name} FAILED: {str(error)}")
        self.logger.error(f"    📋 Error traceback: {traceback.format_exc()}")
    
    def validate_multiindex_structure(self, df: pd.DataFrame, stage: str, 
                                    expected_levels: List[str] = None) -> Dict[str, Any]:
        """Validate MultiIndex DataFrame structure"""
        
        validation_start = time.time()
        
        validation = {
            'stage': stage,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_multiindex': isinstance(df.index, pd.MultiIndex),
            'shape': df.shape,
            'is_valid': True,
            'issues': []
        }
        
        if expected_levels is None:
            expected_levels = ['date', 'ticker']
        
        try:
            if isinstance(df.index, pd.MultiIndex):
                validation['index_levels'] = df.index.names
                validation['level_counts'] = [len(df.index.get_level_values(i).unique()) 
                                            for i in range(df.index.nlevels)]
                
                # Check if expected levels are present
                missing_levels = [level for level in expected_levels if level not in df.index.names]
                if missing_levels:
                    validation['issues'].append(f"Missing index levels: {missing_levels}")
                    validation['is_valid'] = False
                
                # Check for proper ordering
                if df.index.names != expected_levels:
                    validation['issues'].append(f"Index level order: {df.index.names} != expected {expected_levels}")
                
                # Check for duplicated index entries
                if df.index.duplicated().any():
                    dup_count = df.index.duplicated().sum()
                    validation['issues'].append(f"Duplicated index entries: {dup_count}")
                    validation['is_valid'] = False
                
            else:
                validation['issues'].append("DataFrame does not have MultiIndex")
                validation['is_valid'] = False
                validation['index_type'] = str(type(df.index))
            
            # Check data quality
            if df.empty:
                validation['issues'].append("DataFrame is empty")
                validation['is_valid'] = False
            
            # Check for excessive NaN values
            nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if nan_ratio > 0.8:
                validation['issues'].append(f"High NaN ratio: {nan_ratio:.2%}")
                validation['is_valid'] = False
            
            validation['nan_ratio'] = nan_ratio
            validation['validation_time'] = time.time() - validation_start
            
        except Exception as e:
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
            validation['validation_time'] = time.time() - validation_start
        
        # Log validation results
        status = "✅" if validation['is_valid'] else "❌"
        self.logger.info(f"{status} MultiIndex Validation [{stage}]: Shape {validation['shape']}")
        
        if validation['is_valid']:
            if 'index_levels' in validation:
                self.logger.info(f"    📊 Index levels: {validation['index_levels']}")
                self.logger.info(f"    📈 Level counts: {validation['level_counts']}")
            self.logger.info(f"    💧 NaN ratio: {validation['nan_ratio']:.2%}")
        else:
            self.logger.error(f"    ❌ Issues found: {validation['issues']}")
        
        self.multiindex_validations.append(validation)
        return validation
    
    def log_export_preparation(self, data: Dict[str, Any]):
        """Log the preparation of data for export"""
        self.logger.info("=" * 60)
        self.logger.info("📤 EXPORT PREPARATION MONITORING")
        self.logger.info("=" * 60)
        
        for key, value in data.items():
            if hasattr(value, 'shape'):
                self.logger.info(f"📊 {key}: Shape {value.shape}, Type {type(value)}")
                if hasattr(value, 'index') and isinstance(value.index, pd.MultiIndex):
                    self.logger.info(f"    📋 MultiIndex levels: {value.index.names}")
            elif hasattr(value, '__len__'):
                self.logger.info(f"📊 {key}: Length {len(value)}, Type {type(value)}")
            else:
                self.logger.info(f"📊 {key}: Value {value}, Type {type(value)}")
    
    def log_data_conversion(self, stage: str, before_type: str, after_type: str, 
                          before_shape: Tuple, after_shape: Tuple):
        """Log data type conversion during export"""
        self.logger.info(f"🔄 Data Conversion [{stage}]: {before_type} -> {after_type}")
        self.logger.info(f"    📏 Shape: {before_shape} -> {after_shape}")
        
        if before_shape != after_shape:
            self.logger.warning(f"⚠️ Shape changed during conversion!")
    
    def generate_factor_calculation_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of factor calculations"""
        
        total_time = time.time() - self.calculation_start_time if self.calculation_start_time else 0
        
        # Status summary
        status_counts = {}
        for status in ['completed', 'failed', 'calculating']:
            status_counts[status] = len([f for f, s in self.factor_status.items() if s == status])
        
        # Timing analysis
        completed_factors = {f: timing for f, timing in self.factor_timings.items() 
                           if 'duration' in timing}
        
        if completed_factors:
            durations = [timing['duration'] for timing in completed_factors.values()]
            avg_time = np.mean(durations)
            total_calc_time = sum(durations)
            slowest_factor = max(completed_factors.items(), key=lambda x: x[1]['duration'])
            fastest_factor = min(completed_factors.items(), key=lambda x: x[1]['duration'])
        else:
            avg_time = total_calc_time = 0
            slowest_factor = fastest_factor = None
        
        # Group analysis
        group_analysis = {}
        for group, factors in self.factor_groups.items():
            group_factors = [f for f in factors if f in self.factor_status]
            if group_factors:
                group_analysis[group] = {
                    'total': len(group_factors),
                    'completed': len([f for f in group_factors if self.factor_status[f] == 'completed']),
                    'failed': len([f for f in group_factors if self.factor_status[f] == 'failed']),
                    'avg_time': np.mean([self.factor_timings[f]['duration'] 
                                       for f in group_factors 
                                       if f in self.factor_timings and 'duration' in self.factor_timings[f]])
                }
        
        report = {
            'total_time': total_time,
            'status_summary': status_counts,
            'calculation_time': total_calc_time,
            'average_factor_time': avg_time,
            'slowest_factor': slowest_factor,
            'fastest_factor': fastest_factor,
            'group_analysis': group_analysis,
            'multiindex_validations': len(self.multiindex_validations),
            'valid_multiindex_stages': len([v for v in self.multiindex_validations if v['is_valid']]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.logger.info("=" * 80)
        self.logger.info("📋 ALPHA FACTOR CALCULATION REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️ Total monitoring time: {total_time:.2f}s")
        self.logger.info(f"⚙️ Factor calculation time: {total_calc_time:.2f}s")
        self.logger.info(f"📊 Status: ✅{status_counts['completed']} ❌{status_counts['failed']} ⏳{status_counts['calculating']}")
        self.logger.info(f"🎯 Average factor time: {avg_time:.2f}s")
        
        if slowest_factor:
            self.logger.info(f"🐌 Slowest factor: {slowest_factor[0]} ({slowest_factor[1]['duration']:.2f}s)")
        if fastest_factor:
            self.logger.info(f"⚡ Fastest factor: {fastest_factor[0]} ({fastest_factor[1]['duration']:.2f}s)")
        
        self.logger.info(f"📋 MultiIndex validations: {len(self.multiindex_validations)} total, {report['valid_multiindex_stages']} valid")
        
        for group, analysis in group_analysis.items():
            completion_rate = (analysis['completed'] / analysis['total']) * 100
            self.logger.info(f"📦 {group.capitalize()}: {analysis['completed']}/{analysis['total']} ({completion_rate:.1f}%) avg:{analysis['avg_time']:.2f}s")
        
        self.logger.info("=" * 80)
        
        return report

def create_factor_validation_result(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Create validation result for factor data"""
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        values = data.values
        shape = data.shape
    else:
        values = data
        shape = data.shape if hasattr(data, 'shape') else (len(data),)
    
    # Basic validation
    nan_count = np.isnan(values).sum() if np.issubdtype(values.dtype, np.number) else 0
    inf_count = np.isinf(values).sum() if np.issubdtype(values.dtype, np.number) else 0
    
    # Range analysis for numeric data
    if np.issubdtype(values.dtype, np.number) and len(values) > 0:
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            range_info = f"{finite_values.min():.4f} to {finite_values.max():.4f}"
        else:
            range_info = "No finite values"
    else:
        range_info = "Non-numeric data"
    
    # Determine validity
    total_elements = np.prod(shape)
    is_valid = (
        total_elements > 0 and
        nan_count / total_elements < 0.95 and  # Less than 95% NaN
        inf_count == 0  # No infinite values
    )
    
    return {
        'is_valid': is_valid,
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'range': range_info,
        'total_elements': total_elements,
        'valid_ratio': 1.0 - (nan_count / total_elements) if total_elements > 0 else 0.0
    }

if __name__ == "__main__":
    # Test the monitoring system
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    monitor = AlphaFactorMonitor()
    monitor.start_factor_calculation_monitoring()
    
    # Simulate factor calculations
    import time
    test_factors = ['rsi', 'macd', 'bollinger_position', 'volume_ma_ratio', 'atr']
    
    for i, factor in enumerate(test_factors):
        group = list(monitor.factor_groups.keys())[i % len(monitor.factor_groups)]
        
        monitor.log_factor_start(factor, group)
        time.sleep(0.1)  # Simulate calculation
        
        # Create test data
        test_data = pd.DataFrame({
            'factor_value': np.random.randn(100)
        }, index=pd.MultiIndex.from_product([
            pd.date_range('2024-01-01', periods=20),
            ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        ], names=['date', 'ticker']))
        
        validation = create_factor_validation_result(test_data)
        monitor.log_factor_completion(factor, test_data.shape, validation)
        
        # Test MultiIndex validation
        monitor.validate_multiindex_structure(test_data, f"After {factor} calculation")
    
    # Generate final report
    report = monitor.generate_factor_calculation_report()