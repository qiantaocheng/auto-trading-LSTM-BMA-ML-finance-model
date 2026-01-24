#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Stacking LambdaRank - Advanced Meta-Learner with Signal Trajectory Analysis

This module implements a "Temporal Stacking" architecture that feeds the meta-learner
not just today's base model predictions (T), but also their trajectory from T-1 and T-2
to identify signal momentum and stability.

Core Concept:
    Traditional meta-learners only see current predictions. This architecture captures
    how predictions evolve over time, enabling detection of:
    - Signal momentum (is conviction increasing?)
    - Signal stability (is the model confused?)
    - Rank acceleration (are top picks rising or falling in rank?)

Key Features:
1. Feature Engineering: T-1, T-2 lags with proper IPO/new ticker handling
2. Pre-Normalization: Z-Score per date before trajectory calculation
3. Trajectory Features: Momentum_3d, Volatility_3d, Rank_Acceleration
4. Anti-Overfitting LambdaRank: Extremely simple trees (num_leaves=8, max_depth=3)
5. Temporal Safety Validation: Strict leakage checks
6. Fault-Tolerant Inference: Cold-start fallback mechanism

Author: BMA Ultra Enhanced Quantitative System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import PurgedCV to prevent data leakage
try:
    from bma_models.unified_purged_cv_factory import create_unified_cv
    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def zscore_normalize_cross_sectional(df: pd.DataFrame,
                                      pred_cols: List[str],
                                      eps: float = 1e-8) -> pd.DataFrame:
    """
    Apply Z-Score normalization (StandardScaler) per date to base model predictions.

    WHY THIS MATTERS:
    - pred_elastic might be in range [-0.01, 0.01]
    - pred_lambdarank might be in range [-5, 5]
    - Without normalization, "Volatility" features would be dominated by lambdarank
    - After normalization, all models have comparable scale for trajectory features

    Args:
        df: DataFrame with MultiIndex(date, ticker) and prediction columns
        pred_cols: List of prediction column names to normalize
        eps: Small constant to prevent division by zero

    Returns:
        DataFrame with normalized prediction columns (original columns replaced)
    """
    logger.info(f"🔄 Applying Z-Score normalization to {len(pred_cols)} prediction columns...")

    df_norm = df.copy()

    def _normalize_group(group):
        """Normalize each column within the group (same date)"""
        for col in pred_cols:
            if col in group.columns:
                values = group[col].values
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > eps:
                    group[col] = (values - mean_val) / (std_val + eps)
                else:
                    # All same value - set to 0 (neutral)
                    group[col] = 0.0
        return group

    # Group by date and normalize
    df_norm = df_norm.groupby(level='date', group_keys=False).apply(_normalize_group)

    logger.info(f"✅ Z-Score normalization complete")
    return df_norm


def build_temporal_features(df: pd.DataFrame,
                            pred_cols: List[str],
                            lookback: int = 3,
                            handle_ipo_method: str = 'backfill') -> pd.DataFrame:
    """
    Construct T-1, T-2, T-3 lag features and derived trajectory features from OOF predictions.

    This is the core feature engineering function for temporal stacking. It creates:
    1. Lag features (T-1, T-2, T-3 predictions)
    2. Momentum features (signal strength change)
    3. Volatility features (model uncertainty)
    4. Rank acceleration features (position change)

    Args:
        df: DataFrame with MultiIndex(date, ticker) containing OOF predictions
            CRUCIAL: These must be OOF predictions (validation set results),
            strictly devoid of look-ahead bias
        pred_cols: List of prediction column names (e.g., ['pred_elastic', 'pred_lambdarank'])
        lookback: Number of days to look back (default=3 for T-1, T-2, T-3)
        handle_ipo_method: How to handle missing history for new tickers
            - 'backfill': Use current prediction (assumes flat momentum)
            - 'cross_median': Use daily cross-sectional median
            - 'zero': Fill with 0 (NOT recommended - see docstring)

    Returns:
        DataFrame with original columns + lag features + trajectory features

    IPO Handling Logic:
        When a ticker appears for the first time (IPO or newly added), it has no T-1/T-2 history.
        If we fill with 0, the model interprets this as "prediction dropped from X to 0",
        which is a FALSE signal of momentum crash. Instead:
        - 'backfill': Assumes prediction was same yesterday → momentum = 0
        - 'cross_median': Uses typical prediction for that day → neutral signal
    """
    logger.info(f"🔧 Building temporal features with {lookback}-day lookback...")
    logger.info(f"   Prediction columns: {pred_cols}")
    logger.info(f"   IPO handling: {handle_ipo_method}")

    # Step A: Pre-Normalization (CRUCIAL - must happen before lag calculation)
    df_norm = zscore_normalize_cross_sectional(df, pred_cols)

    # Step B: Generate lag features per ticker
    df_result = df_norm.copy()

    # Get unique tickers and process each
    # We need to unstack to ticker level for proper shift operations
    logger.info("   Generating lag features...")

    for col in pred_cols:
        col_short = col.replace('pred_', '')  # Shorter name for lag columns

        # Create lag features by shifting within each ticker
        for lag in range(1, lookback + 1):
            lag_col_name = f'{col_short}_lag{lag}'

            # Shift by ticker (using unstack/stack or groupby)
            # This ensures we get the same ticker's prediction from lag days ago
            df_result[lag_col_name] = df_result.groupby(level='ticker')[col].shift(lag)

        # Step B.1: Handle IPO/New Tickers (NaN from shift)
        for lag in range(1, lookback + 1):
            lag_col_name = f'{col_short}_lag{lag}'

            if handle_ipo_method == 'backfill':
                # Use current prediction as proxy (flat momentum assumption)
                mask = df_result[lag_col_name].isna()
                df_result.loc[mask, lag_col_name] = df_result.loc[mask, col]

            elif handle_ipo_method == 'cross_median':
                # Use daily cross-sectional median
                def _fill_with_daily_median(group):
                    daily_median = group[col].median()
                    group[lag_col_name] = group[lag_col_name].fillna(daily_median)
                    return group
                df_result = df_result.groupby(level='date', group_keys=False).apply(_fill_with_daily_median)

            elif handle_ipo_method == 'zero':
                # NOT recommended but supported
                df_result[lag_col_name] = df_result[lag_col_name].fillna(0)
                logger.warning(f"   ⚠️ Using zero-fill for {lag_col_name} - this may create false momentum signals!")

    # Step C: Calculate Trajectory Features
    logger.info("   Calculating trajectory features...")

    for col in pred_cols:
        col_short = col.replace('pred_', '')

        # C.1: Momentum_3d = Pred_T - Pred_T_lag3
        # Positive = signal strengthening, Negative = signal weakening
        if lookback >= 3:
            df_result[f'{col_short}_momentum_3d'] = (
                df_result[col] - df_result[f'{col_short}_lag3']
            )
        elif lookback >= 1:
            # Fallback to available lags
            df_result[f'{col_short}_momentum_{lookback}d'] = (
                df_result[col] - df_result[f'{col_short}_lag{lookback}']
            )

        # C.2: Volatility_3d = Rolling std of [T, T-1, T-2]
        # High volatility = model is "confused" or signal is unstable
        lag_cols_for_vol = [col] + [f'{col_short}_lag{i}' for i in range(1, min(lookback, 3) + 1)]
        existing_cols = [c for c in lag_cols_for_vol if c in df_result.columns]
        if len(existing_cols) >= 2:
            df_result[f'{col_short}_volatility_3d'] = df_result[existing_cols].std(axis=1)

        # C.3: Rank Acceleration
        # Convert raw scores to daily ranks, then calculate rank change
        df_result[f'{col_short}_rank_pct'] = df_result.groupby(level='date')[col].rank(pct=True)
        if f'{col_short}_lag1' in df_result.columns:
            # Need to rank the lagged value within its original day context
            # This is approximated by ranking current lag1 within current day
            df_result[f'{col_short}_lag1_rank_pct'] = df_result.groupby(level='date')[f'{col_short}_lag1'].rank(pct=True)
            df_result[f'{col_short}_rank_accel'] = (
                df_result[f'{col_short}_rank_pct'] - df_result[f'{col_short}_lag1_rank_pct']
            )

    # Report statistics
    n_features_added = len(df_result.columns) - len(df_norm.columns)
    nan_ratio = df_result.isna().sum().sum() / (len(df_result) * len(df_result.columns))

    logger.info(f"✅ Temporal feature engineering complete:")
    logger.info(f"   Original columns: {len(df_norm.columns)}")
    logger.info(f"   New columns added: {n_features_added}")
    logger.info(f"   Total columns: {len(df_result.columns)}")
    logger.info(f"   NaN ratio: {nan_ratio:.4%}")

    return df_result


# =============================================================================
# TEMPORAL SAFETY VALIDATION
# =============================================================================

def validate_temporal_integrity(df: pd.DataFrame,
                                pred_cols: List[str],
                                correlation_warn_threshold: float = 0.99) -> Dict[str, Any]:
    """
    Rigorous safety checks to ensure no future data leakage in temporal features.

    This function runs BEFORE training to catch common mistakes:
    1. Verifies that Lag_1 of today equals Pred_T of yesterday for the same ticker
    2. Checks correlation between Pred_T and Lag_1 (should be high but < 1.0)
    3. Validates temporal ordering of dates

    Args:
        df: DataFrame with temporal features (must have MultiIndex(date, ticker))
        pred_cols: List of prediction column names
        correlation_warn_threshold: Warn if Pred_T vs Lag_1 correlation > this

    Returns:
        Dict with validation results:
        - 'passed': bool - True if all checks pass
        - 'checks': Dict of individual check results
        - 'warnings': List of warning messages
        - 'errors': List of error messages
    """
    logger.info("🔍 Running temporal integrity validation...")

    results = {
        'passed': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }

    # Check 1: Verify DataFrame structure
    if not isinstance(df.index, pd.MultiIndex):
        results['errors'].append("DataFrame must have MultiIndex(date, ticker)")
        results['passed'] = False
        return results

    if df.index.names != ['date', 'ticker']:
        results['warnings'].append(f"Expected index names ['date', 'ticker'], got {df.index.names}")

    # Check 2: Temporal ordering
    dates = df.index.get_level_values('date').unique()
    dates_sorted = sorted(dates)
    if not all(dates[i] <= dates[i+1] for i in range(len(dates)-1)):
        results['warnings'].append("Dates are not in chronological order in index")
    results['checks']['temporal_ordering'] = True

    # Check 3: Lag consistency validation
    # For each ticker, verify that Lag_1[date=T] == Pred[date=T-1]
    logger.info("   Validating lag consistency...")

    sample_size = min(100, len(df))  # Sample for performance
    sample_idx = np.random.choice(len(df), sample_size, replace=False)

    lag_errors = 0
    for col in pred_cols:
        col_short = col.replace('pred_', '')
        lag1_col = f'{col_short}_lag1'

        if lag1_col not in df.columns:
            continue

        # Check a sample of records
        for idx in sample_idx[:20]:  # Check 20 samples
            row = df.iloc[idx]
            date = df.index.get_level_values('date')[idx]
            ticker = df.index.get_level_values('ticker')[idx]

            # Find previous day for this ticker
            try:
                ticker_data = df.xs(ticker, level='ticker')
                ticker_dates = ticker_data.index.get_level_values('date')
                date_pos = list(ticker_dates).index(date)

                if date_pos > 0:
                    prev_date = ticker_dates[date_pos - 1]
                    prev_pred = ticker_data.loc[prev_date, col]
                    curr_lag1 = row[lag1_col]

                    # Allow for floating point tolerance
                    if not np.isnan(prev_pred) and not np.isnan(curr_lag1):
                        if not np.isclose(prev_pred, curr_lag1, rtol=1e-5):
                            lag_errors += 1
            except (KeyError, ValueError):
                continue

    if lag_errors > 0:
        results['errors'].append(f"Lag consistency check failed: {lag_errors} mismatches found")
        results['passed'] = False
    else:
        results['checks']['lag_consistency'] = True
        logger.info("   ✓ Lag consistency check passed")

    # Check 4: Correlation analysis
    logger.info("   Analyzing Pred_T vs Lag_1 correlation...")

    for col in pred_cols:
        col_short = col.replace('pred_', '')
        lag1_col = f'{col_short}_lag1'

        if lag1_col not in df.columns:
            continue

        valid_mask = ~(df[col].isna() | df[lag1_col].isna())
        if valid_mask.sum() < 100:
            continue

        corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, lag1_col])
        results['checks'][f'{col}_lag1_correlation'] = float(corr)

        if corr > correlation_warn_threshold:
            results['warnings'].append(
                f"High correlation ({corr:.4f}) between {col} and {lag1_col}. "
                f"This might indicate shift failure OR naturally stable signal."
            )
            logger.warning(f"   ⚠️ {col} vs {lag1_col} correlation = {corr:.4f}")
        else:
            logger.info(f"   ✓ {col} vs {lag1_col} correlation = {corr:.4f}")

    # Summary
    if results['passed'] and not results['errors']:
        logger.info("✅ Temporal integrity validation PASSED")
    else:
        logger.error("❌ Temporal integrity validation FAILED")
        for err in results['errors']:
            logger.error(f"   ERROR: {err}")

    for warn in results['warnings']:
        logger.warning(f"   WARNING: {warn}")

    return results


# =============================================================================
# INFERENCE STATE MANAGER (FAULT TOLERANT)
# =============================================================================

@dataclass
class InferenceStateRecord:
    """Record of predictions for a single date"""
    date: str
    predictions: Dict[str, Dict[str, float]]  # {ticker: {pred_col: value}}
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute MD5 checksum for data integrity verification"""
        data_str = json.dumps(self.predictions, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]


class TemporalInferenceState:
    """
    Fault-tolerant state manager for T-1/T-2 prediction history during live trading.

    In live trading, we need yesterday's and day-before-yesterday's predictions
    to compute temporal features. This class manages persistent storage with:
    - Automatic file rotation (keeps last N days)
    - Corruption detection via checksums
    - Graceful fallback on failure (use current predictions as lags)

    COLD START / FALLBACK MECHANISM:
        If load_lags() fails (file missing/corrupted), the system DOES NOT CRASH.
        Instead, it returns current day's predictions as lags (implying 0 momentum)
        and logs a CRITICAL warning. This ensures the trading loop completes even
        if the state file is lost.

    Usage:
        state_mgr = TemporalInferenceState(state_dir='./inference_state')

        # At end of trading day:
        state_mgr.save_state(date='2024-01-15', preds_df=today_predictions)

        # At start of next trading day:
        lags = state_mgr.load_lags(
            date='2024-01-16',
            current_tickers=['AAPL', 'MSFT'],
            fallback_predictions=current_predictions
        )
    """

    def __init__(self,
                 state_dir: Union[str, Path] = './cache/temporal_inference_state',
                 max_history_days: int = 5,
                 pred_cols: Optional[List[str]] = None):
        """
        Initialize state manager.

        Args:
            state_dir: Directory to store state files
            max_history_days: Number of days of history to retain
            pred_cols: Expected prediction column names
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_days = max_history_days
        self.pred_cols = pred_cols or ['pred_elastic', 'pred_xgb', 'pred_lambdarank', 'pred_catboost']

        logger.info(f"📁 TemporalInferenceState initialized: {self.state_dir}")

    def _get_state_path(self, date: str) -> Path:
        """Get file path for a specific date's state"""
        # Normalize date format
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = pd.to_datetime(date).strftime('%Y%m%d')
        return self.state_dir / f'predictions_{date_str}.parquet'

    def save_state(self, date: str, preds_df: pd.DataFrame) -> bool:
        """
        Save today's predictions to persistent storage.

        Args:
            date: Date string (YYYY-MM-DD or YYYYMMDD)
            preds_df: DataFrame with predictions (must have ticker in index or as column)

        Returns:
            True if save successful, False otherwise
        """
        try:
            logger.info(f"💾 Saving inference state for {date}...")

            # Ensure we have the right columns
            save_cols = [c for c in self.pred_cols if c in preds_df.columns]
            if not save_cols:
                logger.error(f"No prediction columns found in DataFrame")
                return False

            # Extract relevant data
            if isinstance(preds_df.index, pd.MultiIndex):
                # Already has MultiIndex - extract just the ticker level
                df_to_save = preds_df[save_cols].copy()
            else:
                df_to_save = preds_df[save_cols].copy()

            # Add metadata
            df_to_save['_save_timestamp'] = datetime.now().isoformat()
            df_to_save['_date'] = str(date)

            # Save to parquet
            state_path = self._get_state_path(date)
            df_to_save.to_parquet(state_path, compression='gzip')

            logger.info(f"   ✓ Saved {len(df_to_save)} records to {state_path.name}")

            # Cleanup old files
            self._cleanup_old_states()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save state for {date}: {e}")
            return False

    def load_lags(self,
                  date: str,
                  current_tickers: List[str],
                  fallback_predictions: Optional[pd.DataFrame] = None,
                  lookback: int = 2) -> pd.DataFrame:
        """
        Retrieve T-1, T-2 predictions for temporal feature construction.

        FAULT TOLERANCE:
            If loading fails, returns fallback_predictions as lags (0 momentum).
            This ensures the trading loop never crashes due to missing state.

        Args:
            date: Current date
            current_tickers: List of tickers we need predictions for
            fallback_predictions: Current predictions to use if history unavailable
            lookback: Number of lag days to load (default=2 for T-1, T-2)

        Returns:
            DataFrame with columns: ticker, lag1_*, lag2_*, etc.
        """
        logger.info(f"📖 Loading lag predictions for {date}...")

        lags_data = {}
        missing_days = []

        # Convert date to datetime for arithmetic
        target_date = pd.to_datetime(date)

        for lag_num in range(1, lookback + 1):
            # Calculate the date we need (accounting for weekends/holidays is approximate)
            # In production, this should use a trading calendar
            lag_date = target_date - pd.Timedelta(days=lag_num)

            # Try to load state file
            state_path = self._get_state_path(lag_date)

            if state_path.exists():
                try:
                    lag_df = pd.read_parquet(state_path)

                    # Extract predictions for current tickers
                    for col in self.pred_cols:
                        if col in lag_df.columns:
                            col_short = col.replace('pred_', '')
                            lags_data[f'{col_short}_lag{lag_num}'] = lag_df[col]

                    logger.info(f"   ✓ Loaded T-{lag_num} from {state_path.name}")

                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to read {state_path.name}: {e}")
                    missing_days.append(lag_num)
            else:
                logger.warning(f"   ⚠️ State file not found: {state_path.name}")
                missing_days.append(lag_num)

        # FALLBACK MECHANISM
        if missing_days and fallback_predictions is not None:
            logger.critical(
                f"🚨 COLD START: Missing state for T-{missing_days}. "
                f"Using current predictions as fallback (0 momentum assumption)."
            )

            # Use current predictions as lags (implies flat momentum)
            for lag_num in missing_days:
                for col in self.pred_cols:
                    if col in fallback_predictions.columns:
                        col_short = col.replace('pred_', '')
                        lags_data[f'{col_short}_lag{lag_num}'] = fallback_predictions[col]

        # Construct result DataFrame
        if lags_data:
            result = pd.DataFrame(lags_data)
            # Filter to current tickers if needed
            if hasattr(result.index, 'get_level_values'):
                try:
                    ticker_level = result.index.get_level_values('ticker')
                    mask = ticker_level.isin(current_tickers)
                    result = result[mask]
                except:
                    pass

            logger.info(f"   ✓ Loaded lags for {len(result)} records")
            return result
        else:
            logger.error("❌ No lag data available and no fallback provided")
            return pd.DataFrame()

    def _cleanup_old_states(self):
        """Remove state files older than max_history_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days + 2)

            for state_file in self.state_dir.glob('predictions_*.parquet'):
                # Extract date from filename
                try:
                    date_str = state_file.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')

                    if file_date < cutoff_date:
                        state_file.unlink()
                        logger.debug(f"   Cleaned up old state: {state_file.name}")
                except:
                    continue

        except Exception as e:
            logger.warning(f"State cleanup failed: {e}")

    def get_available_history(self) -> List[str]:
        """List available state files"""
        files = list(self.state_dir.glob('predictions_*.parquet'))
        dates = []
        for f in files:
            try:
                date_str = f.stem.split('_')[1]
                dates.append(date_str)
            except:
                continue
        return sorted(dates)


# =============================================================================
# TEMPORAL STACKING LAMBDARANK MODEL
# =============================================================================

class TemporalStackingLambdaRank:
    """
    Temporal Stacking LambdaRank - Meta-learner with signal trajectory analysis.

    This model takes first-layer OOF predictions, constructs temporal features
    (lags, momentum, volatility), and trains a LightGBM LambdaRank model focused
    on Top-K ranking optimization.

    Key Design Decisions:

    1. ANTI-OVERFITTING PARAMETERS
       - num_leaves=8: Extremely simple trees force generalization
       - max_depth=3: Shallow trees prevent memorization
       - feature_fraction=0.6: Forces model to use trajectory features, not just current
       - lambdarank_truncation_level=60: Focus only on top 60 stocks

    2. PRE-NORMALIZATION
       - Z-Score per date before lag calculation
       - Ensures trajectory features are comparable across models

    3. IPO HANDLING
       - Backfill with current prediction (flat momentum assumption)
       - NOT zero-fill which creates false momentum signals

    4. TEMPORAL SAFETY
       - Automatic leakage validation before training
       - PurgedCV with gap/embargo for proper time-series splits
    """

    def __init__(self,
                 base_pred_cols: Tuple[str, ...] = ('pred_elastic', 'pred_xgb', 'pred_lambdarank', 'pred_catboost'),
                 lookback_days: int = 3,
                 n_quantiles: int = 32,
                 label_gain_power: float = 2.5,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = 500,
                 early_stopping_rounds: int = 50,
                 use_purged_cv: bool = True,
                 cv_n_splits: int = 5,
                 cv_gap_days: int = 10,
                 cv_embargo_days: int = 10,
                 random_state: int = 42,
                 ipo_handling: str = 'backfill'):
        """
        Initialize Temporal Stacking LambdaRank model.

        Args:
            base_pred_cols: First-layer prediction columns to use
            lookback_days: Number of days for lag features (default=3 for T-1,T-2,T-3)
            n_quantiles: Number of quantile levels for rank conversion
            label_gain_power: Power for label gain (higher = more focus on top ranks)
            lgb_params: Custom LightGBM parameters (overrides defaults)
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Rounds for early stopping
            use_purged_cv: Use PurgedCV for temporal safety (required=True)
            cv_n_splits: Number of CV folds
            cv_gap_days: Gap days between train/test (should match prediction horizon)
            cv_embargo_days: Embargo days after test fold
            random_state: Random seed for reproducibility
            ipo_handling: How to handle new tickers ('backfill', 'cross_median', 'zero')
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for TemporalStackingLambdaRank")

        self.base_pred_cols = list(base_pred_cols)
        self.lookback_days = lookback_days
        self.n_quantiles = n_quantiles
        self.label_gain_power = label_gain_power
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.use_purged_cv = use_purged_cv
        self.cv_n_splits = cv_n_splits
        self.cv_gap_days = cv_gap_days
        self.cv_embargo_days = cv_embargo_days
        self.random_state = random_state
        self.ipo_handling = ipo_handling

        # Generate label_gain sequence
        if label_gain_power == 1.0:
            self.label_gain = list(range(n_quantiles))
        else:
            self.label_gain = [
                (i / (n_quantiles - 1)) ** label_gain_power * (n_quantiles - 1)
                for i in range(n_quantiles)
            ]

        # ANTI-OVERFITTING LightGBM parameters (as specified in requirements)
        default_lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'label_gain': self.label_gain,

            # ANTI-OVERFITTING CONSTRAINTS
            'num_leaves': 8,                    # ★ Keep trees extremely simple
            'max_depth': 3,                     # ★ Very shallow
            'learning_rate': 0.005,             # ★ Slow learning
            'feature_fraction': 0.6,            # ★ Force model to use lags/diffs
            'bagging_fraction': 0.7,
            'bagging_freq': 3,
            'min_data_in_leaf': 200,

            # LambdaRank specific
            'lambdarank_truncation_level': 60,  # ★ Focus only on Top 60
            'sigmoid': 1.0,

            # Regularization
            'lambda_l1': 0.1,
            'lambda_l2': 5.0,

            # Other
            'verbose': -1,
            'random_state': random_state,
            'force_col_wise': True
        }

        self.lgb_params = default_lgb_params.copy()
        if lgb_params:
            self.lgb_params.update(lgb_params)

        # Model state
        self.model = None
        self.scaler = StandardScaler()
        self.fitted_ = False
        self.feature_names_ = None
        self._temporal_feature_cols = None

        # Inference state manager
        self.inference_state = TemporalInferenceState(
            pred_cols=self.base_pred_cols
        )

        logger.info("🏗️ TemporalStackingLambdaRank initialized:")
        logger.info(f"   Base predictions: {self.base_pred_cols}")
        logger.info(f"   Lookback days: {self.lookback_days}")
        logger.info(f"   Anti-overfitting: num_leaves={self.lgb_params['num_leaves']}, "
                   f"max_depth={self.lgb_params['max_depth']}, "
                   f"feature_fraction={self.lgb_params['feature_fraction']}")
        logger.info(f"   LambdaRank: truncation_level={self.lgb_params['lambdarank_truncation_level']}, "
                   f"label_gain_power={self.label_gain_power}")

    def _prepare_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare temporal features from OOF predictions"""
        # Build temporal features
        df_temporal = build_temporal_features(
            df=df,
            pred_cols=[c for c in self.base_pred_cols if c in df.columns],
            lookback=self.lookback_days,
            handle_ipo_method=self.ipo_handling
        )

        return df_temporal

    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify which columns to use as features"""
        # Include: original predictions + lag features + trajectory features
        feature_cols = []

        for col in df.columns:
            # Skip target and metadata columns
            if col.startswith('ret_fwd') or col.startswith('target'):
                continue
            if col.startswith('_'):  # Metadata columns
                continue
            if col in ['date', 'ticker', 'Date', 'Ticker']:
                continue

            # Include prediction columns and their derived features
            for base in self.base_pred_cols:
                base_short = base.replace('pred_', '')
                if col == base or col.startswith(base_short):
                    feature_cols.append(col)
                    break

        return list(set(feature_cols))

    def _convert_to_rank_labels(self,
                                 df: pd.DataFrame,
                                 target_col: str) -> Tuple[np.ndarray, List[int]]:
        """
        Convert continuous target to rank labels, grouped by date.

        Returns:
            labels: Array of integer rank labels [0, n_quantiles-1]
            group_sizes: List of group sizes (one per trading day)
        """
        labels = np.zeros(len(df), dtype=np.int32)
        group_sizes = []

        dates = df.index.get_level_values('date').unique()

        current_idx = 0
        for date in dates:
            date_mask = df.index.get_level_values('date') == date
            date_data = df.loc[date_mask, target_col]
            n_samples = len(date_data)
            group_sizes.append(n_samples)

            if n_samples <= 1:
                labels[current_idx:current_idx + n_samples] = self.n_quantiles // 2
            else:
                # Rank within day
                rank_pct = date_data.rank(pct=True, method='average').values
                day_labels = np.floor(rank_pct * self.n_quantiles).astype(int)
                day_labels[day_labels == self.n_quantiles] = self.n_quantiles - 1
                labels[current_idx:current_idx + n_samples] = day_labels

            current_idx += n_samples

        return labels, group_sizes

    def fit(self,
            oof_df: pd.DataFrame,
            target_col: str = 'ret_fwd_10d') -> 'TemporalStackingLambdaRank':
        """
        Train the Temporal Stacking LambdaRank model.

        Args:
            oof_df: DataFrame with MultiIndex(date, ticker) containing:
                    - OOF predictions from first-layer models
                    - Target variable (future returns)
            target_col: Name of target column

        Returns:
            self (fitted model)
        """
        logger.info("🚀 Training TemporalStackingLambdaRank...")

        # Validate input
        if not isinstance(oof_df.index, pd.MultiIndex):
            raise ValueError("oof_df must have MultiIndex(date, ticker)")

        if target_col not in oof_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Check for required prediction columns
        available_pred_cols = [c for c in self.base_pred_cols if c in oof_df.columns]
        if not available_pred_cols:
            raise ValueError(f"No prediction columns found. Expected: {self.base_pred_cols}")

        logger.info(f"   Using prediction columns: {available_pred_cols}")

        # Step 1: Build temporal features
        logger.info("Step 1: Building temporal features...")
        df_temporal = self._prepare_temporal_features(oof_df)

        # Add target column back
        df_temporal[target_col] = oof_df[target_col]

        # Step 2: Run temporal integrity validation
        logger.info("Step 2: Validating temporal integrity...")
        validation_result = validate_temporal_integrity(df_temporal, available_pred_cols)
        if not validation_result['passed']:
            for err in validation_result['errors']:
                logger.error(f"   {err}")
            raise ValueError("Temporal integrity validation failed. Check logs for details.")

        # Step 3: Identify feature columns
        self._temporal_feature_cols = self._identify_feature_columns(df_temporal)
        logger.info(f"   Using {len(self._temporal_feature_cols)} temporal features")

        # Step 4: Prepare training data
        logger.info("Step 3: Preparing training data...")

        # Drop rows with NaN in features or target
        valid_mask = ~(
            df_temporal[self._temporal_feature_cols].isna().any(axis=1) |
            df_temporal[target_col].isna()
        )
        df_valid = df_temporal[valid_mask].copy()

        logger.info(f"   Valid samples: {len(df_valid)} / {len(df_temporal)} ({len(df_valid)/len(df_temporal)*100:.1f}%)")

        X = df_valid[self._temporal_feature_cols].values
        y_labels, group_sizes = self._convert_to_rank_labels(df_valid, target_col)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names_ = self._temporal_feature_cols

        # Step 5: Train with PurgedCV or direct
        if self.use_purged_cv and PURGED_CV_AVAILABLE:
            logger.info("Step 4: Training with PurgedCV...")
            self.model = self._train_with_cv(X_scaled, y_labels, df_valid, group_sizes)
        else:
            logger.info("Step 4: Training without CV (direct)...")
            train_dataset = lgb.Dataset(
                X_scaled,
                label=y_labels,
                group=group_sizes,
                feature_name=self.feature_names_
            )
            self.model = lgb.train(
                self.lgb_params,
                train_dataset,
                num_boost_round=self.num_boost_round,
                callbacks=[lgb.log_evaluation(period=0)]
            )

        self.fitted_ = True

        # Report training results
        logger.info("✅ TemporalStackingLambdaRank training complete:")
        logger.info(f"   Best iteration: {self.model.best_iteration}")
        logger.info(f"   Feature count: {len(self.feature_names_)}")

        # Top feature importances
        importance = dict(zip(self.feature_names_, self.model.feature_importance()))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"   Top features: {dict(top_features)}")

        return self

    def _train_with_cv(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       df: pd.DataFrame,
                       group_sizes: List[int]) -> lgb.Booster:
        """Train with PurgedCV for temporal safety"""
        cv = create_unified_cv(
            n_splits=self.cv_n_splits,
            gap=self.cv_gap_days,
            embargo=self.cv_embargo_days
        )

        dates = df.index.get_level_values('date')
        unique_dates = sorted(dates.unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        sample_groups = [date_to_idx[d] for d in dates]

        cv_splits = list(cv.split(X, y, groups=sample_groups))

        best_iteration = self.num_boost_round
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Recalculate group sizes for folds
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            train_groups = [np.sum(train_dates == d) for d in train_dates.unique()]
            val_groups = [np.sum(val_dates == d) for d in val_dates.unique()]

            train_dataset = lgb.Dataset(X_train, label=y_train, group=train_groups)
            val_dataset = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_dataset)

            try:
                model = lgb.train(
                    self.lgb_params,
                    train_dataset,
                    num_boost_round=self.num_boost_round,
                    valid_sets=[val_dataset],
                    valid_names=['val'],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                if model.best_iteration > 0:
                    best_iteration = min(best_iteration, model.best_iteration)
                    cv_scores.append(model.best_score.get('val', {}).get('ndcg@10', 0))

            except Exception as e:
                logger.warning(f"CV fold {fold+1} failed: {e}")
                continue

        if cv_scores:
            logger.info(f"   CV NDCG@10: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # Final training on all data
        logger.info(f"   Final training with {best_iteration} rounds...")
        train_dataset = lgb.Dataset(X, label=y, group=group_sizes, feature_name=self.feature_names_)

        return lgb.train(
            self.lgb_params,
            train_dataset,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using trained model.

        Args:
            df: DataFrame with OOF predictions (same format as training)

        Returns:
            DataFrame with 'temporal_score' column
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Build temporal features
        df_temporal = self._prepare_temporal_features(df)

        # Check for required features
        missing_features = [f for f in self.feature_names_ if f not in df_temporal.columns]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
            # Fill missing with 0 (neutral)
            for f in missing_features:
                df_temporal[f] = 0.0

        # Extract features and predict
        X = df_temporal[self.feature_names_].values
        valid_mask = ~np.isnan(X).any(axis=1)

        X_valid = X[valid_mask]
        X_scaled = self.scaler.transform(X_valid)

        predictions = np.full(len(X), np.nan)
        predictions[valid_mask] = self.model.predict(X_scaled)

        result = pd.DataFrame({
            'temporal_score': predictions
        }, index=df_temporal.index)

        # Add rank within each day
        result['temporal_rank'] = result.groupby(level='date')['temporal_score'].rank(
            method='average', ascending=False
        )

        return result

    def predict_with_state(self,
                           current_predictions: pd.DataFrame,
                           current_date: str) -> pd.DataFrame:
        """
        Generate predictions using inference state manager for lags.

        This is the recommended method for live trading, as it properly
        handles historical state management.

        Args:
            current_predictions: Today's first-layer predictions
            current_date: Current trading date

        Returns:
            DataFrame with temporal scores
        """
        # Load historical lags
        tickers = (
            current_predictions.index.get_level_values('ticker').unique().tolist()
            if isinstance(current_predictions.index, pd.MultiIndex)
            else current_predictions.index.tolist()
        )

        lags = self.inference_state.load_lags(
            date=current_date,
            current_tickers=tickers,
            fallback_predictions=current_predictions,
            lookback=self.lookback_days
        )

        # Merge current predictions with lags
        df_combined = current_predictions.copy()
        for col in lags.columns:
            df_combined[col] = lags[col]

        # Generate predictions
        return self.predict(df_combined)

    def save_inference_state(self, date: str, predictions: pd.DataFrame) -> bool:
        """Save current predictions for future lag calculation"""
        return self.inference_state.save_state(date, predictions)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'model_type': 'TemporalStackingLambdaRank',
            'fitted': self.fitted_,
            'base_pred_cols': self.base_pred_cols,
            'lookback_days': self.lookback_days,
            'n_quantiles': self.n_quantiles,
            'label_gain_power': self.label_gain_power,
            'lgb_params': {k: v for k, v in self.lgb_params.items() if k != 'label_gain'},
            'ipo_handling': self.ipo_handling
        }

        if self.fitted_:
            info['best_iteration'] = self.model.best_iteration
            info['n_features'] = len(self.feature_names_)
            info['feature_names'] = self.feature_names_

            # Feature importance
            importance = dict(zip(self.feature_names_, self.model.feature_importance()))
            info['feature_importance'] = dict(sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )[:20])

        return info


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_temporal_stacker(config: Optional[Dict[str, Any]] = None) -> TemporalStackingLambdaRank:
    """
    Factory function to create TemporalStackingLambdaRank with config.

    Args:
        config: Optional configuration dict. If None, uses defaults.

    Returns:
        Configured TemporalStackingLambdaRank instance
    """
    default_config = {
        'base_pred_cols': ('pred_elastic', 'pred_xgb', 'pred_lambdarank', 'pred_catboost'),
        'lookback_days': 3,
        'n_quantiles': 32,
        'label_gain_power': 2.5,
        'num_boost_round': 500,
        'early_stopping_rounds': 50,
        'cv_n_splits': 5,
        'cv_gap_days': 10,
        'cv_embargo_days': 10,
        'ipo_handling': 'backfill'
    }

    if config:
        default_config.update(config)

    return TemporalStackingLambdaRank(**default_config)


def run_temporal_stacking_training(data_path: str,
                                    target_col: str = 'ret_fwd_10d',
                                    output_dir: Optional[str] = None,
                                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    End-to-end temporal stacking training pipeline.

    Args:
        data_path: Path to parquet file with OOF predictions
        target_col: Target column name
        output_dir: Directory to save model and results
        config: Optional configuration overrides

    Returns:
        Dict with training results and metrics
    """
    logger.info(f"🚀 Running temporal stacking training pipeline...")
    logger.info(f"   Data: {data_path}")

    # Load data
    df = pd.read_parquet(data_path)

    # Ensure MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.set_index(['date', 'ticker'])
        else:
            raise ValueError("Data must have date and ticker columns or MultiIndex")

    # Create and train model
    model = create_temporal_stacker(config)
    model.fit(df, target_col=target_col)

    # Save model if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model info
        info = model.get_model_info()
        with open(output_path / 'temporal_stacker_info.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"   Model info saved to {output_path}")

    return {
        'model': model,
        'info': model.get_model_info(),
        'status': 'success'
    }
