#!/usr/bin/env python3
"""
Numerical Stability Utilities for Alpha Factor Computation
Provides safe mathematical operations to prevent overflow, underflow, and division by zero
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

def safe_divide(numerator: Union[float, pd.Series, np.ndarray], 
                denominator: Union[float, pd.Series, np.ndarray], 
                fill_value: float = 0.0,
                eps: float = 1e-12) -> Union[float, pd.Series, np.ndarray]:
    """
    Safe division operation that handles division by zero and very small numbers
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero or very small
        eps: Minimum absolute value for denominator to avoid division by zero
    
    Returns:
        Result of safe division
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        # For Series/arrays, use vectorized operations
        mask = np.abs(denominator) < eps
        safe_denominator = denominator.copy()
        
        if isinstance(safe_denominator, pd.Series):
            safe_denominator.loc[mask] = eps
            result = numerator / safe_denominator
            result.loc[mask] = fill_value
        else:
            safe_denominator[mask] = eps
            result = numerator / safe_denominator
            result[mask] = fill_value
            
        return result
    else:
        # For scalar values
        if abs(denominator) < eps:
            return fill_value
        return numerator / denominator

def safe_log(x: Union[float, pd.Series, np.ndarray], 
             fill_value: float = 0.0,
             eps: float = 1e-12) -> Union[float, pd.Series, np.ndarray]:
    """
    Safe logarithm operation that handles negative and zero values
    
    Args:
        x: Input values
        fill_value: Value to use for non-positive inputs
        eps: Minimum positive value for logarithm computation
    
    Returns:
        Result of safe logarithm
    """
    if isinstance(x, (pd.Series, np.ndarray)):
        # For Series/arrays
        safe_x = np.maximum(x, eps)
        result = np.log(safe_x)
        
        if isinstance(x, pd.Series):
            mask = x <= 0
            result.loc[mask] = fill_value
        else:
            mask = x <= 0
            result[mask] = fill_value
            
        return result
    else:
        # For scalar values
        if x <= 0:
            return fill_value
        return np.log(max(x, eps))

def safe_sqrt(x: Union[float, pd.Series, np.ndarray], 
              fill_value: float = 0.0) -> Union[float, pd.Series, np.ndarray]:
    """
    Safe square root operation that handles negative values
    
    Args:
        x: Input values
        fill_value: Value to use for negative inputs
    
    Returns:
        Result of safe square root
    """
    if isinstance(x, (pd.Series, np.ndarray)):
        safe_x = np.maximum(x, 0)
        result = np.sqrt(safe_x)
        
        if isinstance(x, pd.Series):
            mask = x < 0
            result.loc[mask] = fill_value
        else:
            mask = x < 0
            result[mask] = fill_value
            
        return result
    else:
        if x < 0:
            return fill_value
        return np.sqrt(max(x, 0))

def winsorize(data: Union[pd.Series, np.ndarray], 
              lower_percentile: float = 0.01, 
              upper_percentile: float = 0.99) -> Union[pd.Series, np.ndarray]:
    """
    Winsorize data by capping extreme values at specified percentiles
    
    Args:
        data: Input data to winsorize
        lower_percentile: Lower percentile for winsorization (0-1)
        upper_percentile: Upper percentile for winsorization (0-1)
    
    Returns:
        Winsorized data
    """
    if isinstance(data, pd.Series):
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        lower_bound = np.percentile(data, lower_percentile * 100)
        upper_bound = np.percentile(data, upper_percentile * 100)
        return np.clip(data, lower_bound, upper_bound)

def standardize(data: Union[pd.Series, np.ndarray], 
                robust: bool = False) -> Union[pd.Series, np.ndarray]:
    """
    Standardize data (z-score normalization)
    
    Args:
        data: Input data to standardize
        robust: If True, use median and MAD instead of mean and std
    
    Returns:
        Standardized data
    """
    if robust:
        # Robust standardization using median and MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return safe_divide(data - median, mad * 1.4826, fill_value=0.0)  # 1.4826 is the MAD scaling factor
    else:
        # Standard z-score normalization
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return safe_divide(data - mean, std, fill_value=0.0)

def handle_inf_nan(data: Union[pd.Series, np.ndarray], 
                   fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """
    Replace infinite and NaN values with a specified fill value
    
    Args:
        data: Input data
        fill_value: Value to use for inf/nan replacement
    
    Returns:
        Data with inf/nan values replaced
    """
    if isinstance(data, pd.Series):
        return data.replace([np.inf, -np.inf, np.nan], fill_value)
    else:
        result = data.copy()
        mask = ~np.isfinite(result)
        result[mask] = fill_value
        return result

# Alias for backward compatibility
safe_reciprocal = lambda x, fill_value=0.0: safe_divide(1.0, x, fill_value=fill_value)