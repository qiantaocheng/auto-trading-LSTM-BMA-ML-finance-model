#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Prediction with EWMA Smoothing and Excel Ranking Report
==============================================================
Uses exponential weighted average (recent days more important) to smooth predictions:
Score*(t) = 0.5*Score(t) + 0.3*Score(t-1) + 0.2*Score(t-2)

Or uses half-life = 3 days for EWMA.

Generates Excel ranking report with:
- Raw predictions for last 3 days
- EWMA smoothed predictions
- Rankings
- Model breakdowns
"""

import sys
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from bma_models.model_registry import load_manifest
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def calculate_ewma_smoothed_scores(predictions_df: pd.DataFrame, 
                                   weights: tuple = (0.5, 0.3, 0.2),
                                   use_half_life: bool = False,
                                   half_life_days: float = 3.0) -> pd.DataFrame:
    """
    Calculate EWMA smoothed scores from last 3 days of predictions
    
    Args:
        predictions_df: DataFrame with MultiIndex (date, ticker) and 'score' column
        weights: Tuple of weights for (today, yesterday, day_before_yesterday)
        use_half_life: If True, use half-life formula instead of fixed weights
        half_life_days: Half-life in days (for EWMA calculation)
    
    Returns:
        DataFrame with smoothed scores
    """
    if not isinstance(predictions_df.index, pd.MultiIndex):
        raise ValueError("predictions_df must have MultiIndex (date, ticker)")
    
    if 'score' not in predictions_df.columns:
        raise ValueError("predictions_df must have 'score' column")
    
    # Get unique dates (sorted)
    dates = sorted(predictions_df.index.get_level_values('date').unique())
    
    if len(dates) < 3:
        logger.warning(f"Only {len(dates)} days available, need at least 3 days for EWMA smoothing")
        logger.warning("Returning original scores without smoothing")
        return predictions_df.copy()
    
    # Calculate weights based on half-life if requested
    if use_half_life:
        # EWMA with half-life: alpha = 1 - exp(-ln(2) / half_life)
        alpha = 1 - np.exp(-np.log(2) / half_life_days)
        # Weights decay exponentially: w_t = alpha * (1-alpha)^(n-t)
        weights = (
            alpha,  # Today (t=0)
            alpha * (1 - alpha),  # Yesterday (t=1)
            alpha * (1 - alpha) ** 2  # Day before yesterday (t=2)
        )
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = tuple(w / total for w in weights)
        logger.info(f"Using half-life={half_life_days} days, calculated weights: {weights}")
    else:
        logger.info(f"Using fixed weights: {weights}")
    
    # Create result DataFrame
    result_df = predictions_df.copy()
    result_df['score_raw'] = result_df['score'].copy()
    result_df['score_smoothed'] = np.nan
    
    # Process each ticker
    tickers = predictions_df.index.get_level_values('ticker').unique()
    
    for ticker in tickers:
        ticker_data = predictions_df.xs(ticker, level='ticker', drop_level=False)
        
        # Get last 3 days
        last_3_dates = dates[-3:]
        
        for i, current_date in enumerate(dates):
            if current_date < last_3_dates[0]:
                # Not enough history, use raw score
                result_df.loc[(current_date, ticker), 'score_smoothed'] = result_df.loc[(current_date, ticker), 'score']
                continue
            
            # Get scores for last 3 days (including today)
            scores = []
            available_dates = []
            
            for j, date_offset in enumerate([0, -1, -2]):
                target_date_idx = i + date_offset
                if target_date_idx >= 0 and target_date_idx < len(dates):
                    target_date = dates[target_date_idx]
                    if (target_date, ticker) in predictions_df.index:
                        scores.append(predictions_df.loc[(target_date, ticker), 'score'])
                        available_dates.append(target_date)
            
            if len(scores) == 0:
                # No data available
                result_df.loc[(current_date, ticker), 'score_smoothed'] = result_df.loc[(current_date, ticker), 'score']
            elif len(scores) == 1:
                # Only today available
                result_df.loc[(current_date, ticker), 'score_smoothed'] = scores[0]
            else:
                # Calculate weighted average
                # Use weights in reverse order (most recent first)
                if len(scores) == 2:
                    # Only today and yesterday
                    smoothed = weights[0] * scores[0] + weights[1] * scores[1]
                else:
                    # All 3 days available
                    smoothed = weights[0] * scores[0] + weights[1] * scores[1] + weights[2] * scores[2]
                
                result_df.loc[(current_date, ticker), 'score_smoothed'] = smoothed
    
    # Use smoothed score as final score
    result_df['score'] = result_df['score_smoothed'].fillna(result_df['score_raw'])
    
    return result_df


def generate_excel_ranking_report(predictions_df: pd.DataFrame,
                                 output_path: str,
                                 model_name: str = "MetaRankerStacker") -> None:
    """
    Generate Excel ranking report with predictions and rankings
    
    Args:
        predictions_df: DataFrame with MultiIndex (date, ticker) and score columns
        output_path: Path to output Excel file
        model_name: Name of the model for report title
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    logger.info(f"📊 Generating Excel ranking report: {output_path}")
    
    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Ranking Report"
    
    # Get latest date
    dates = sorted(predictions_df.index.get_level_values('date').unique())
    latest_date = dates[-1] if len(dates) > 0 else None
    
    # Title
    title_cell = ws['A1']
    title_cell.value = f"{model_name} - EWMA Smoothed Predictions Ranking Report"
    title_cell.font = Font(size=16, bold=True)
    ws.merge_cells('A1:F1')
    
    # Date info
    date_cell = ws['A2']
    date_cell.value = f"Report Date: {latest_date.strftime('%Y-%m-%d') if latest_date else 'N/A'}"
    date_cell.font = Font(size=12)
    ws.merge_cells('A2:F2')
    
    # EWMA info
    ewma_cell = ws['A3']
    ewma_cell.value = "EWMA Weights: Today=0.5, Yesterday=0.3, Day Before=0.2 (or half-life=3 days)"
    ewma_cell.font = Font(size=10, italic=True)
    ws.merge_cells('A3:F3')
    
    # Headers
    headers = ['Rank', 'Ticker', 'Smoothed Score', 'Raw Score (Today)', 'Raw Score (Yesterday)', 'Raw Score (Day Before)', 'Score Change']
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=4, column=col_idx)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    
    # Get latest date predictions
    if latest_date:
        try:
            latest_predictions = predictions_df.xs(latest_date, level='date', drop_level=False)
        except KeyError:
            logger.warning(f"Date {latest_date} not found in predictions, using all available data")
            latest_predictions = predictions_df.groupby(level='ticker').last()
        
        # Ensure we have a 'score' column
        if 'score' not in latest_predictions.columns:
            if 'score_smoothed' in latest_predictions.columns:
                latest_predictions['score'] = latest_predictions['score_smoothed']
            elif 'score_raw' in latest_predictions.columns:
                latest_predictions['score'] = latest_predictions['score_raw']
            else:
                logger.warning("No score column found, using first numeric column")
                numeric_cols = latest_predictions.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    latest_predictions['score'] = latest_predictions[numeric_cols[0]]
                else:
                    raise ValueError("No score column found in predictions")
        
        # Sort by smoothed score (descending)
        latest_predictions = latest_predictions.sort_values('score', ascending=False)
        
        # Get previous days for comparison
        prev_dates = dates[-3:] if len(dates) >= 3 else dates
        
        row = 5
        for rank, (idx, row_data) in enumerate(latest_predictions.iterrows(), start=1):
            ticker = idx[1] if isinstance(idx, tuple) else idx
            
            # Get scores for last 3 days
            scores_today = row_data.get('score_raw', np.nan)
            if np.isnan(scores_today):
                scores_today = row_data.get('score', np.nan)
            
            scores_yesterday = np.nan
            scores_day_before = np.nan
            
            if len(prev_dates) >= 2:
                yesterday_date = prev_dates[-2]
                try:
                    if (yesterday_date, ticker) in predictions_df.index:
                        scores_yesterday = predictions_df.loc[(yesterday_date, ticker), 'score_raw']
                        if np.isnan(scores_yesterday):
                            scores_yesterday = predictions_df.loc[(yesterday_date, ticker), 'score']
                except (KeyError, IndexError):
                    pass
            
            if len(prev_dates) >= 3:
                day_before_date = prev_dates[-3]
                try:
                    if (day_before_date, ticker) in predictions_df.index:
                        scores_day_before = predictions_df.loc[(day_before_date, ticker), 'score_raw']
                        if np.isnan(scores_day_before):
                            scores_day_before = predictions_df.loc[(day_before_date, ticker), 'score']
                except (KeyError, IndexError):
                    pass
            
            # Calculate score change
            score_change = np.nan
            if not np.isnan(scores_today) and not np.isnan(scores_yesterday):
                score_change = scores_today - scores_yesterday
            
            # Get smoothed score
            smoothed_score = row_data.get('score', np.nan)
            if np.isnan(smoothed_score):
                smoothed_score = row_data.get('score_smoothed', scores_today)
            
            # Write row
            ws.cell(row=row, column=1).value = rank
            ws.cell(row=row, column=2).value = ticker
            ws.cell(row=row, column=3).value = smoothed_score if not np.isnan(smoothed_score) else None
            ws.cell(row=row, column=4).value = scores_today if not np.isnan(scores_today) else None
            ws.cell(row=row, column=5).value = scores_yesterday if not np.isnan(scores_yesterday) else None
            ws.cell(row=row, column=6).value = scores_day_before if not np.isnan(scores_day_before) else None
            ws.cell(row=row, column=7).value = score_change if not np.isnan(score_change) else None
            
            # Format cells
            for col in range(1, 8):
                cell = ws.cell(row=row, column=col)
                cell.alignment = Alignment(horizontal="center" if col == 1 else "left" if col == 2 else "right", vertical="center")
                if col == 1:  # Rank
                    cell.font = Font(bold=True)
                if col == 7 and not np.isnan(score_change):  # Score change
                    if score_change > 0:
                        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif score_change < 0:
                        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            
            row += 1
    else:
        logger.warning("No dates found in predictions")
    
    # Auto-adjust column widths
    for col in range(1, 8):
        max_length = 0
        column = get_column_letter(col)
        for cell in ws[column]:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column].width = adjusted_width
    
    # Add summary sheet
    ws_summary = wb.create_sheet("Summary")
    ws_summary['A1'] = "Summary Statistics"
    ws_summary['A1'].font = Font(size=14, bold=True)
    
    if latest_date and len(latest_predictions) > 0:
        ws_summary['A3'] = "Total Tickers"
        ws_summary['B3'] = len(latest_predictions)
        ws_summary['A4'] = "Top 10 Avg Score"
        ws_summary['B4'] = latest_predictions.head(10)['score'].mean()
        ws_summary['A5'] = "Top 20 Avg Score"
        ws_summary['B5'] = latest_predictions.head(20)['score'].mean()
        ws_summary['A6'] = "Bottom 10 Avg Score"
        ws_summary['B6'] = latest_predictions.tail(10)['score'].mean()
    
    # Save workbook
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    logger.info(f"✅ Excel report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Direct prediction with EWMA smoothing and Excel ranking report")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated ticker symbols")
    parser.add_argument("--snapshot", type=str, default="latest", help="Snapshot ID or 'latest'")
    parser.add_argument("--days", type=int, default=3, help="Number of days to fetch for EWMA (default: 3)")
    parser.add_argument("--weights", type=str, default="0.5,0.3,0.2", help="EWMA weights (today,yesterday,day_before) or 'half-life'")
    parser.add_argument("--half-life", type=float, default=None, help="Half-life in days (overrides weights if provided)")
    parser.add_argument("--output", type=str, default="results/ewma_ranking_report.xlsx", help="Output Excel file path")
    parser.add_argument("--as-of-date", type=str, default=None, help="Prediction date (YYYY-MM-DD), default: today")
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided")
    
    logger.info("=" * 100)
    logger.info("DIRECT PREDICTION WITH EWMA SMOOTHING")
    logger.info("=" * 100)
    logger.info(f"Tickers: {', '.join(tickers)}")
    logger.info(f"Days: {args.days}")
    
    # Parse weights
    use_half_life = False
    weights = (0.5, 0.3, 0.2)
    
    if args.half_life:
        use_half_life = True
        half_life_days = args.half_life
        logger.info(f"Using half-life: {half_life_days} days")
    elif args.weights.lower() == 'half-life' or args.weights.lower() == 'halflife':
        use_half_life = True
        half_life_days = 3.0
        logger.info(f"Using half-life: {half_life_days} days (default)")
    else:
        try:
            weight_list = [float(w.strip()) for w in args.weights.split(',')]
            if len(weight_list) == 3:
                weights = tuple(weight_list)
                logger.info(f"Using fixed weights: {weights}")
            else:
                logger.warning(f"Invalid weights format, using default (0.5, 0.3, 0.2)")
        except Exception as e:
            logger.warning(f"Failed to parse weights: {e}, using default")
    
    # Determine date range
    if args.as_of_date:
        end_date = pd.to_datetime(args.as_of_date)
    else:
        end_date = pd.Timestamp.today()
    
    # Need at least 3 days of data for EWMA (plus lookback for features)
    # Fetch data from (end_date - days - lookback) to end_date
    MIN_LOOKBACK_DAYS = 280  # For feature calculation
    start_date = (end_date - pd.Timedelta(days=args.days + MIN_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date_str}")
    logger.info(f"Prediction period: Last {args.days} days")
    
    # Initialize model
    model = UltraEnhancedQuantitativeModel()
    
    # Get predictions for last N days using predict_with_snapshot
    logger.info("📡 Fetching data and generating predictions...")
    
    all_predictions = []
    
    # Get predictions for all days at once (more efficient)
    # Fetch data for the entire period, then extract predictions for each day
    logger.info(f"🔮 Fetching data and generating predictions for last {args.days} days...")
    
    try:
        # Use predict_with_snapshot with auto-fetch for the entire period
        results = model.predict_with_snapshot(
            feature_data=None,  # Auto-fetch
            snapshot_id=None if args.snapshot == "latest" else args.snapshot,
            universe_tickers=tickers,
            as_of_date=end_date,
            prediction_days=args.days  # Get predictions for all days
        )
        
        if results.get('success', False):
            predictions = results.get('predictions')
            if predictions is not None and len(predictions) > 0:
                # Convert to DataFrame
                if isinstance(predictions, pd.Series):
                    pred_df = predictions.to_frame('score')
                elif isinstance(predictions, pd.DataFrame):
                    pred_df = predictions.copy()
                    if 'score' not in pred_df.columns and len(pred_df.columns) == 1:
                        pred_df.columns = ['score']
                else:
                    logger.error(f"Unexpected predictions type: {type(predictions)}")
                    raise ValueError("Predictions must be Series or DataFrame")
                
                # Ensure MultiIndex with date and ticker
                if not isinstance(pred_df.index, pd.MultiIndex):
                    logger.warning("Predictions don't have MultiIndex, attempting to infer dates...")
                    # If no MultiIndex, assume all predictions are for the latest date
                    pred_df.index = pd.MultiIndex.from_arrays(
                        [[end_date] * len(pred_df), pred_df.index],
                        names=['date', 'ticker']
                    )
                
                # Check if we have multiple dates
                dates_in_pred = sorted(pred_df.index.get_level_values('date').unique())
                logger.info(f"✅ Found predictions for {len(dates_in_pred)} dates: {dates_in_pred}")
                
                # If we only have one date but need multiple days, try to get historical predictions
                if len(dates_in_pred) < args.days:
                    logger.info(f"⚠️ Only {len(dates_in_pred)} date(s) found, fetching historical predictions...")
                    # Fetch predictions for previous days
                    for day_offset in range(args.days - 1, 0, -1):  # From oldest to newest (excluding today)
                        pred_date = end_date - pd.Timedelta(days=day_offset)
                        pred_date_str = pred_date.strftime('%Y-%m-%d')
                        
                        # Skip if we already have this date
                        if pred_date in dates_in_pred:
                            continue
                        
                        logger.info(f"🔮 Fetching predictions for {pred_date_str}...")
                        try:
                            hist_results = model.predict_with_snapshot(
                                feature_data=None,
                                snapshot_id=None if args.snapshot == "latest" else args.snapshot,
                                universe_tickers=tickers,
                                as_of_date=pred_date,
                                prediction_days=1
                            )
                            
                            if hist_results.get('success', False):
                                hist_pred = hist_results.get('predictions')
                                if hist_pred is not None and len(hist_pred) > 0:
                                    if isinstance(hist_pred, pd.Series):
                                        hist_df = hist_pred.to_frame('score')
                                    else:
                                        hist_df = hist_pred.copy()
                                    
                                    # Ensure MultiIndex
                                    if not isinstance(hist_df.index, pd.MultiIndex):
                                        hist_df.index = pd.MultiIndex.from_arrays(
                                            [[pred_date] * len(hist_df), hist_df.index],
                                            names=['date', 'ticker']
                                        )
                                    else:
                                        # Update date level
                                        new_index = pd.MultiIndex.from_arrays([
                                            [pred_date] * len(hist_df),
                                            hist_df.index.get_level_values('ticker')
                                        ], names=['date', 'ticker'])
                                        hist_df.index = new_index
                                    
                                    pred_df = pd.concat([pred_df, hist_df], axis=0)
                                    logger.info(f"✅ Added {len(hist_df)} predictions for {pred_date_str}")
                        except Exception as e:
                            logger.warning(f"Failed to fetch predictions for {pred_date_str}: {e}")
                
                all_predictions.append(pred_df)
                logger.info(f"✅ Combined predictions: {len(pred_df)} total rows")
            else:
                raise RuntimeError("No predictions returned")
        else:
            raise RuntimeError(f"Prediction failed: {results.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        raise
    
    if not all_predictions:
        raise RuntimeError("No predictions generated for any date")
    
    # Combine all predictions (if we have multiple DataFrames)
    if len(all_predictions) == 1:
        combined_predictions = all_predictions[0]
    else:
        combined_predictions = pd.concat(all_predictions, axis=0)
    
    logger.info(f"✅ Combined {len(combined_predictions)} predictions")
    logger.info(f"   Date range: {combined_predictions.index.get_level_values('date').min()} to {combined_predictions.index.get_level_values('date').max()}")
    logger.info(f"   Unique tickers: {combined_predictions.index.get_level_values('ticker').nunique()}")
    
    # Apply EWMA smoothing
    logger.info("📊 Applying EWMA smoothing...")
    smoothed_predictions = calculate_ewma_smoothed_scores(
        combined_predictions,
        weights=weights,
        use_half_life=use_half_life,
        half_life_days=args.half_life if args.half_life else 3.0
    )
    
    # Generate Excel report
    logger.info("📊 Generating Excel ranking report...")
    generate_excel_ranking_report(
        smoothed_predictions,
        args.output,
        model_name="MetaRankerStacker"
    )
    
    logger.info("=" * 100)
    logger.info("✅ COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Excel report: {args.output}")
    logger.info(f"Total predictions: {len(smoothed_predictions)}")
    logger.info(f"Unique tickers: {smoothed_predictions.index.get_level_values('ticker').nunique()}")
    logger.info(f"Date range: {smoothed_predictions.index.get_level_values('date').min()} to {smoothed_predictions.index.get_level_values('date').max()}")


if __name__ == "__main__":
    main()
