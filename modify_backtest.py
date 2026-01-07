#!/usr/bin/env python3
"""Script to modify comprehensive_model_backtest.py to output weekly results"""

with open('scripts/comprehensive_model_backtest.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modification 1: Change calculate_group_returns signature and return
content = content.replace(
    'def calculate_group_returns(self, predictions: pd.DataFrame, quantile: float = 0.2) -> Dict:',
    'def calculate_group_returns(self, predictions: pd.DataFrame, quantile: float = 0.2) -> Tuple[Dict, pd.DataFrame]:'
)

content = content.replace(
    '            results.append({\n                \'date\': date,\n                \'top_return\': top_return,\n                \'bottom_return\': bottom_return,\n                \'all_return\': all_return,\n                \'long_short\': top_return - bottom_return\n            })',
    '            results.append({\n                \'date\': date,\n                \'top_20_return\': top_return,\n                \'bottom_20_return\': bottom_return,\n                \'all_return\': all_return,\n                \'long_short\': top_return - bottom_return,\n                \'n_stocks\': len(valid_group)\n            })'
)

content = content.replace(
    '        if len(results) == 0:\n            return {}',
    '        if len(results) == 0:\n            return {}, pd.DataFrame()'
)

content = content.replace(
    '            \'avg_top_return\': results_df[\'top_return\'].mean(),\n            \'avg_bottom_return\': results_df[\'bottom_return\'].mean(),',
    '            \'avg_top_return\': results_df[\'top_20_return\'].mean(),\n            \'avg_bottom_return\': results_df[\'bottom_20_return\'].mean(),'
)

content = content.replace(
    '        return summary',
    '        return summary, results_df'
)

# Modification 2: Update generate_report to handle tuple return
content = content.replace(
    '            # Calculate group returns\n            group_returns = self.calculate_group_returns(predictions, quantile=0.2)',
    '            # Calculate group returns\n            group_returns, weekly_details = self.calculate_group_returns(predictions, quantile=0.2)'
)

content = content.replace(
    '    def generate_report(self, all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:',
    '    def generate_report(self, all_results: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:'
)

content = content.replace(
    '        return report_df',
    '        # Store weekly details for each model\n        weekly_details_dict = {}\n        for model_name, predictions in all_results.items():\n            _, weekly_df = self.calculate_group_returns(predictions, quantile=0.2)\n            weekly_details_dict[model_name] = weekly_df\n        \n        return report_df, weekly_details_dict'
)

# Modification 3: Update run_backtest to return weekly details
content = content.replace(
    '    def run_backtest(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:',
    '    def run_backtest(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, pd.DataFrame]]:'
)

content = content.replace(
    '        # Generate report\n        report_df = self.generate_report(all_results)\n\n        return all_results, report_df',
    '        # Generate report\n        report_df, weekly_details_dict = self.generate_report(all_results)\n\n        return all_results, report_df, weekly_details_dict'
)

# Modification 4: Update main() to save weekly details
content = content.replace(
    '    # Run backtest\n    all_results, report_df = backtest.run_backtest()',
    '    # Run backtest\n    all_results, report_df, weekly_details_dict = backtest.run_backtest()'
)

content = content.replace(
    '    # Save detailed predictions\n    for model_name, predictions in all_results.items():\n        pred_path = os.path.join(output_dir, f"{model_name}_predictions_{datetime.now().strftime(\'%Y%m%d_%H%M%S\')}.parquet")\n        predictions.to_parquet(pred_path, index=False)\n        logger.info(f"📄 {model_name} 预测结果已保存: {pred_path}")',
    '    # Save weekly details for each model\n    timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')\n    for model_name, weekly_df in weekly_details_dict.items():\n        weekly_path = os.path.join(output_dir, f"{model_name}_weekly_returns_{timestamp}.csv")\n        weekly_df.to_csv(weekly_path, index=False)\n        logger.info(f"📊 {model_name} 每周收益已保存: {weekly_path}")\n    \n    # Save detailed predictions\n    for model_name, predictions in all_results.items():\n        pred_path = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.parquet")\n        predictions.to_parquet(pred_path, index=False)\n        logger.info(f"📄 {model_name} 预测结果已保存: {pred_path}")'
)

# Write the modified content
with open('scripts/comprehensive_model_backtest.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Script modified successfully!")
print("Changes made:")
print("1. calculate_group_returns now returns (summary, weekly_df)")
print("2. generate_report now returns (report_df, weekly_details_dict)")
print("3. run_backtest now returns (all_results, report_df, weekly_details_dict)")
print("4. main() now saves weekly details to CSV files")
