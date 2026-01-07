#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Search Report Generator
=============================

读取网格搜索结果并生成详细的Excel报告。

Report Structure:
- Sheet 1: Summary (所有模型最佳参数汇总)
- Sheet 2: ElasticNet Top 10
- Sheet 3: XGBoost Top 10
- Sheet 4: CatBoost Top 10
- Sheet 5: LambdaRank Top 10
- Sheet 6: Ridge Top 10
- Sheet 7: All Results (完整结果)
- Sheet 8: Parameter Analysis (参数影响分析)

Usage:
    python scripts/grid_search_report.py \
        --input-dir results/grid_search_20231205 \
        --output-file results/grid_search_report.xlsx
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_grid_search_results(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all grid search results from directory.

    Args:
        input_dir: Directory containing grid search results

    Returns:
        Dictionary mapping model_name -> results DataFrame
    """
    results = {}

    # Look for *_grid_search_final.csv files
    for csv_file in input_dir.glob("*_grid_search_final.csv"):
        model_name = csv_file.stem.replace('_grid_search_final', '')
        df = pd.read_csv(csv_file)
        results[model_name] = df
        logger.info(f"Loaded {len(df)} results for {model_name}")

    if not results:
        raise FileNotFoundError(f"No grid search results found in {input_dir}")

    return results


def create_summary_sheet(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create summary sheet with best parameters for each model.

    Args:
        results: Dictionary of model results

    Returns:
        Summary DataFrame
    """
    summary_rows = []

    for model_name, df in results.items():
        if df.empty or 'top20_avg_return' not in df.columns:
            continue

        # Find best combination
        best_idx = df['top20_avg_return'].idxmax()
        best_row = df.loc[best_idx]

        # Extract parameter columns (exclude meta columns)
        meta_cols = ['model', 'combination_id', 'params', 'snapshot_id',
                     'top20_avg_return', 'train_success', 'backtest_success', 'timestamp']
        param_cols = [col for col in df.columns if col not in meta_cols]

        # Build summary row
        summary_row = {
            'Model': model_name.upper(),
            'Best_Top20_Return': best_row['top20_avg_return'],
            'Combination_ID': best_row['combination_id'],
            'Snapshot_ID': best_row['snapshot_id'],
            'Total_Combinations': len(df),
            'Successful_Runs': df['train_success'].sum(),
            'Success_Rate': f"{df['train_success'].sum() / len(df) * 100:.1f}%"
        }

        # Add best parameters
        for param_col in param_cols:
            summary_row[f'Best_{param_col}'] = best_row[param_col]

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Best_Top20_Return', ascending=False)

    return summary_df


def create_top_n_sheet(df: pd.DataFrame, model_name: str, n: int = 10) -> pd.DataFrame:
    """
    Create top N results for a specific model.

    Args:
        df: Model results DataFrame
        model_name: Model name
        n: Number of top results

    Returns:
        Top N DataFrame
    """
    if df.empty or 'top20_avg_return' not in df.columns:
        return pd.DataFrame()

    # Sort by Top20% return
    top_n_df = df.sort_values('top20_avg_return', ascending=False).head(n).copy()

    # Add rank column
    top_n_df.insert(0, 'Rank', range(1, len(top_n_df) + 1))

    # Reorder columns for better readability
    priority_cols = ['Rank', 'top20_avg_return', 'combination_id', 'snapshot_id']
    other_cols = [col for col in top_n_df.columns if col not in priority_cols]
    top_n_df = top_n_df[priority_cols + other_cols]

    return top_n_df


def create_parameter_analysis(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze the impact of each parameter on Top20% return.

    Args:
        results: Dictionary of model results

    Returns:
        Parameter analysis DataFrame
    """
    analysis_rows = []

    for model_name, df in results.items():
        if df.empty or 'top20_avg_return' not in df.columns:
            continue

        # Identify parameter columns
        meta_cols = ['model', 'combination_id', 'params', 'snapshot_id',
                     'top20_avg_return', 'train_success', 'backtest_success', 'timestamp']
        param_cols = [col for col in df.columns if col not in meta_cols]

        # Analyze each parameter
        for param_col in param_cols:
            # Group by parameter value and compute statistics
            grouped = df.groupby(param_col)['top20_avg_return'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()

            grouped.columns = [param_col, 'Mean_Return', 'Std_Return',
                             'Min_Return', 'Max_Return', 'N_Combinations']

            # Add model and parameter name
            grouped.insert(0, 'Model', model_name.upper())
            grouped.insert(1, 'Parameter', param_col)

            # Rename first column to 'Value'
            grouped = grouped.rename(columns={param_col: 'Value'})

            analysis_rows.append(grouped)

    if not analysis_rows:
        return pd.DataFrame()

    analysis_df = pd.concat(analysis_rows, ignore_index=True)

    # Sort by Model and Mean_Return (descending)
    analysis_df = analysis_df.sort_values(['Model', 'Mean_Return'], ascending=[True, False])

    return analysis_df


def format_excel_output(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame
):
    """
    Format Excel sheet with styling.

    Args:
        writer: ExcelWriter object
        sheet_name: Sheet name
        df: DataFrame to write
    """
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Get workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })

    number_format = workbook.add_format({
        'num_format': '0.0000',
        'align': 'right'
    })

    percent_format = workbook.add_format({
        'num_format': '0.00%',
        'align': 'right'
    })

    # Apply header format
    for col_num, value in enumerate(df.columns):
        worksheet.write(0, col_num, value, header_format)

    # Auto-adjust column width
    for col_num, column in enumerate(df.columns):
        max_length = max(
            df[column].astype(str).map(len).max(),
            len(str(column))
        )
        worksheet.set_column(col_num, col_num, min(max_length + 2, 50))

    # Apply number format to numeric columns
    for col_num, column in enumerate(df.columns):
        if df[column].dtype in [np.float64, np.float32]:
            if 'return' in column.lower() or 'rate' in column.lower():
                for row_num in range(1, len(df) + 1):
                    worksheet.write(row_num, col_num, df.iloc[row_num - 1][column], number_format)


def generate_comprehensive_report(
    input_dir: Path,
    output_file: Path,
    top_n: int = 10
):
    """
    Generate comprehensive grid search report in Excel format.

    Args:
        input_dir: Directory containing grid search results
        output_file: Output Excel file path
        top_n: Number of top results per model
    """
    logger.info("=" * 80)
    logger.info("📊 Generating Comprehensive Grid Search Report")
    logger.info("=" * 80)

    # Load all results
    results = load_grid_search_results(input_dir)

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

        # Sheet 1: Summary
        logger.info("Creating Summary sheet...")
        summary_df = create_summary_sheet(results)
        format_excel_output(writer, 'Summary', summary_df)

        # Sheets 2-6: Top N for each model
        model_order = ['elastic_net', 'xgboost', 'catboost', 'lambdarank', 'ridge']
        for model_name in model_order:
            if model_name in results:
                logger.info(f"Creating {model_name.upper()} Top {top_n} sheet...")
                top_df = create_top_n_sheet(results[model_name], model_name, n=top_n)
                format_excel_output(writer, f'{model_name.upper()}_Top{top_n}', top_df)

        # Sheet 7: All Results Combined
        logger.info("Creating All Results sheet...")
        all_results = pd.concat([df for df in results.values()], ignore_index=True)
        all_results = all_results.sort_values('top20_avg_return', ascending=False)
        format_excel_output(writer, 'All_Results', all_results)

        # Sheet 8: Parameter Analysis
        logger.info("Creating Parameter Analysis sheet...")
        param_analysis_df = create_parameter_analysis(results)
        format_excel_output(writer, 'Parameter_Analysis', param_analysis_df)

    logger.info("=" * 80)
    logger.info(f"✅ Report generated successfully: {output_file}")
    logger.info("=" * 80)

    # Print summary statistics
    logger.info("\n📈 Summary Statistics:")
    logger.info(f"Total models searched: {len(results)}")
    for model_name, df in results.items():
        best_return = df['top20_avg_return'].max()
        logger.info(f"  {model_name.upper()}: {len(df)} combinations, Best Top20% Return: {best_return:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive grid search report'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing grid search results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output Excel file path'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top results per model (default: 10)'
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        generate_comprehensive_report(
            input_dir=input_dir,
            output_file=output_file,
            top_n=args.top_n
        )
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
