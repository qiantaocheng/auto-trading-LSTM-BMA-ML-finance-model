#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Excel Exporter - 防御性Excel导出器
确保万无一失的Excel导出流程
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class RobustExcelExporter:
    """防御性Excel导出器 - 确保所有数据都能正确导出"""

    def __init__(self, output_dir: str = "D:/trade/result"):
        """
        初始化导出器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._prediction_horizon_days = self._resolve_prediction_horizon()


    def _resolve_prediction_horizon(self) -> int:
        """Resolve prediction horizon days from configuration (falls back to 10)."""
        try:
            from bma_models.unified_config_loader import get_time_config

            time_config = get_time_config()
            horizon = getattr(time_config, 'prediction_horizon_days', None)
            if isinstance(horizon, (int, float)) and horizon > 0:
                return int(horizon)
        except Exception:
            logger.debug("Falling back to default prediction horizon=10 (unable to load temporal config)")
        return 10

    def _compute_target_date(self, base_date: Optional[pd.Timestamp]) -> pd.Timestamp:
        try:
            base_ts = pd.Timestamp(base_date)
        except Exception:
            base_ts = pd.Timestamp(datetime.now().date())
        horizon = getattr(self, '_prediction_horizon_days', 10) or 10
        return base_ts + pd.Timedelta(days=horizon)

    def safe_export(
        self,
        predictions_series: Optional[pd.Series],
        analysis_results: Dict[str, Any],
        feature_data: Optional[pd.DataFrame] = None,
        lambda_df: Optional[pd.DataFrame] = None,
        ridge_df: Optional[pd.DataFrame] = None,
        final_df: Optional[pd.DataFrame] = None,
        base_models_df: Optional[pd.DataFrame] = None,
        kronos_df: Optional[pd.DataFrame] = None,
        kronos_pass_df: Optional[pd.DataFrame] = None,
        lambda_percentile_info: Optional[Dict[str, Any]] = None,
        tradingagents_df: Optional[pd.DataFrame] = None,
        model_prediction_tables: Optional[Dict[str, pd.DataFrame]] = None,
        top30_summary: Optional[pd.DataFrame] = None,
        top30_details: Optional[Dict[str, pd.DataFrame]] = None,
        simple_mode: bool = True  # default: only keep Final_Predictions sheet
    ) -> Optional[str]:
        """
        安全导出Excel

        Args:
            simple_mode: True=只导出Final_Predictions表，False=导出所有详细表

        Returns:
            str: Excel文件路径，失败返回None
        """
        try:
            logger.info("=" * 80)
            logger.info(f"📊 开始Robust Excel导出 (简化模式: {simple_mode})")
            logger.info("=" * 80)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bma_analysis_{timestamp}.xlsx"
            filepath = self.output_dir / filename

            # ========== 创建Excel Writer ==========
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

                if simple_mode:
                    # 🔧 写入一个最小可见工作表，避免无表时Excel报错
                    try:
                        pd.DataFrame({'Info': [timestamp]}).to_excel(writer, sheet_name='Info', index=False)
                    except Exception:
                        pass

                    # 🎯 简化模式：优先导出Final_Predictions
                    if final_df is not None and not final_df.empty:
                        self._write_final_sheet(writer, final_df)
                    elif base_models_df is not None and not base_models_df.empty:
                        self._write_base_models_sheet(writer, base_models_df)
                    else:
                        # 回退：使用predictions_series
                        pred_data = self._prepare_predictions(predictions_series)
                        self._write_predictions_sheet(writer, pred_data)
                        logger.warning("Final_Predictions不可用，使用Predictions表代替")

                    if kronos_df is not None and not kronos_df.empty:
                        self._write_kronos_sheet(writer, kronos_df)
                    if kronos_pass_df is not None and not kronos_pass_df.empty:
                        self._write_kronos_sheet(writer, kronos_pass_df, sheet_name='Kronos_Passed_Over10')

                    if model_prediction_tables:
                        self._write_model_prediction_tables(writer, model_prediction_tables)
                    if top30_summary is not None and not top30_summary.empty:
                        self._write_top30_summary(writer, top30_summary)
                    if top30_details:
                        self._write_top30_details(writer, top30_details)
                else:
                    # 完整模式：导出所有表
                    pred_data = self._prepare_predictions(predictions_series)
                    model_info = self._prepare_model_info(analysis_results, feature_data)

                    # Sheet 1: Summary (总览)
                    self._write_summary_sheet(writer, pred_data, model_info, lambda_percentile_info)

                    # Sheet 2: Predictions (预测结果)
                    self._write_predictions_sheet(writer, pred_data)

                    # Sheet 3: Lambda Predictions (如果有)
                    if lambda_df is not None and not lambda_df.empty:
                        self._write_lambda_sheet(writer, lambda_df)

                    # Sheet 4: Base Model Predictions (如果有)
                    if base_models_df is not None and not base_models_df.empty:
                        self._write_base_models_sheet(writer, base_models_df)

                    # Sheet 5: Ridge Predictions (如果有)
                    if ridge_df is not None and not ridge_df.empty:
                        self._write_ridge_sheet(writer, ridge_df)

                    # Sheet 6: Final Predictions (如果有)
                    if final_df is not None and not final_df.empty:
                        self._write_final_sheet(writer, final_df)

                    # Sheet 7: Kronos Filter (如果有)
                    if kronos_df is not None and not kronos_df.empty:
                        self._write_kronos_sheet(writer, kronos_df)
                    if kronos_pass_df is not None and not kronos_pass_df.empty:
                        self._write_kronos_sheet(writer, kronos_pass_df, sheet_name='Kronos_Passed_Over10')

                    # Sheet 8: TradingAgents Analysis (如果有)
                    if tradingagents_df is not None and not tradingagents_df.empty:
                        self._write_tradingagents_sheet(writer, tradingagents_df)

                    # Sheet 9: Factor Contributions
                    self._write_factor_contributions_sheet(writer, model_info)

                    if model_prediction_tables:
                        self._write_model_prediction_tables(writer, model_prediction_tables, include_models=('catboost', 'xgboost', 'elastic_net'), use_special_handlers=False)
                    if top30_summary is not None and not top30_summary.empty:
                        self._write_top30_summary(writer, top30_summary)
                    if top30_details:
                        self._write_top30_details(writer, top30_details)

            logger.info("=" * 80)
            logger.info(f"✅ Excel导出成功!")
            logger.info(f"📄 文件路径: {filepath}")
            logger.info("=" * 80)

            return str(filepath)

        except Exception as e:
            logger.error(f"✗ Excel导出失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _prepare_predictions(self, predictions_series: Optional[pd.Series]) -> Dict[str, Any]:
        """准备预测数据"""
        try:
            if predictions_series is None:
                logger.warning("⚠️ predictions_series为None")
                return {
                    'predictions': np.array([]),
                    'dates': [],
                    'tickers': [],
                    'count': 0,
                    'base_date': None,
                    'target_date': None
                }

            if len(predictions_series) == 0:
                logger.warning("⚠️ predictions_series为空")
                return {
                    'predictions': np.array([]),
                    'dates': [],
                    'tickers': [],
                    'count': 0,
                    'base_date': None,
                    'target_date': None
                }

            # 处理MultiIndex
            if isinstance(predictions_series.index, pd.MultiIndex):
                logger.info("   检测到MultiIndex格式")

                # 检查是否有'date'级别
                if 'date' in predictions_series.index.names:
                    base_date = predictions_series.index.get_level_values('date').max()
                    target_date = self._compute_target_date(base_date)
                    pred_latest = predictions_series.xs(base_date, level='date')

                    return {
                        'predictions': pred_latest.values,
                        'dates': [target_date] * len(pred_latest),
                        'tickers': pred_latest.index.tolist(),
                        'count': len(pred_latest),
                        'base_date': pd.Timestamp(base_date),
                        'target_date': target_date
                    }
                else:
                    # MultiIndex但没有'date'级别，使用第一个级别
                    logger.warning(f"   MultiIndex但缺少'date'级别，索引名称: {predictions_series.index.names}")
                    # 重置索引为简单索引
                    predictions_series = predictions_series.reset_index(drop=True)
                    base_date = datetime.now().date()
                    target_date = self._compute_target_date(base_date)

                    return {
                        'predictions': predictions_series.values,
                        'dates': [target_date] * len(predictions_series),
                        'tickers': [f'ticker_{i}' for i in range(len(predictions_series))],
                        'count': len(predictions_series),
                        'base_date': pd.Timestamp(base_date),
                        'target_date': target_date
                    }
            else:
                # Simple Index
                logger.info("   检测到Simple Index格式")
                base_date = datetime.now().date()
                target_date = self._compute_target_date(base_date)

                return {
                    'predictions': predictions_series.values,
                    'dates': [target_date] * len(predictions_series),
                    'tickers': predictions_series.index.tolist(),
                    'count': len(predictions_series),
                    'base_date': pd.Timestamp(base_date),
                    'target_date': target_date
                }

        except Exception as e:
            logger.error(f"✗ 预测数据准备失败: {e}")
            return {
                'predictions': np.array([]),
                'dates': [],
                'tickers': [],
                'count': 0,
                'base_date': None,
                'target_date': None
            }

    def _prepare_model_info(self, analysis_results: Dict[str, Any], feature_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """准备模型信息"""
        try:
            model_info = {
                'model_type': 'BMA Ultra Enhanced',
                'model_version': 'v3.0',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction_horizon': 'T+1',
                'n_samples': len(feature_data) if feature_data is not None else 'N/A',
                'n_features': feature_data.shape[1] if feature_data is not None else 15
            }

            # 添加训练时间
            if 'execution_time' in analysis_results:
                model_info['training_time'] = f"{analysis_results['execution_time']:.1f}s"

            # 添加CV分数
            if 'training_results' in analysis_results:
                tr = analysis_results['training_results']
                if 'traditional_models' in tr and 'cv_scores' in tr['traditional_models']:
                    cv_scores = tr['traditional_models']['cv_scores']
                    model_info['cv_score'] = np.mean(list(cv_scores.values()))
                    model_info['model_weights'] = cv_scores

            logger.info("✓ 模型信息准备完成")
            return model_info

        except Exception as e:
            logger.error(f"✗ 模型信息准备失败: {e}")
            return {
                'model_type': 'BMA Ultra Enhanced',
                'model_version': 'v3.0',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    def _write_summary_sheet(self, writer, pred_data: Dict, model_info: Dict,
                            lambda_percentile_info: Optional[Dict[str, Any]] = None):
        """写入Summary工作表"""
        try:
            summary_data = []

            # 模型信息
            summary_data.append(['=== 模型信息 ===', ''])
            for key, value in model_info.items():
                if key != 'model_weights':
                    summary_data.append([key, str(value)])

            summary_data.append(['', ''])

            # 预测信息
            summary_data.append(['=== 预测信息 ===', ''])
            summary_data.append(['预测数量', pred_data['count']])
            summary_data.append(['基准日期', str(pred_data['base_date'])])
            summary_data.append(['目标日期', str(pred_data['target_date'])])

            summary_data.append(['', ''])

            # Lambda Percentile融合信息（新增）
            if lambda_percentile_info is not None:
                summary_data.append(['=== Lambda Percentile融合 ===', ''])
                summary_data.append(['融合策略', 'Lambda Percentile作为Ridge特征'])
                summary_data.append(['Lambda使用因子', lambda_percentile_info.get('n_factors', 15)])
                summary_data.append(['Lambda OOF样本数', lambda_percentile_info.get('oof_samples', 'N/A')])
                summary_data.append(['Percentile均值', f"{lambda_percentile_info.get('percentile_mean', 0):.1f}"])
                summary_data.append(['Percentile范围', f"[{lambda_percentile_info.get('percentile_min', 0):.1f}, {lambda_percentile_info.get('percentile_max', 100):.1f}]"])
                summary_data.append(['索引对齐状态', lambda_percentile_info.get('alignment_status', 'Unknown')])
                if lambda_percentile_info.get('nan_ratio', 0) > 0:
                    summary_data.append(['NaN比例', f"{lambda_percentile_info.get('nan_ratio', 0):.2%}"])

                summary_data.append(['', ''])

            # 统计信息
            if pred_data['count'] > 0:
                summary_data.append(['=== 预测统计 ===', ''])
                summary_data.append(['最大值', f"{np.max(pred_data['predictions']):.6f}"])
                summary_data.append(['最小值', f"{np.min(pred_data['predictions']):.6f}"])
                summary_data.append(['平均值', f"{np.mean(pred_data['predictions']):.6f}"])
                summary_data.append(['中位数', f"{np.median(pred_data['predictions']):.6f}"])
                summary_data.append(['标准差', f"{np.std(pred_data['predictions']):.6f}"])

            df = pd.DataFrame(summary_data, columns=['Item', 'Value'])
            df.to_excel(writer, sheet_name='Summary', index=False)
            logger.info("✓ Summary工作表已写入")

        except Exception as e:
            logger.error(f"✗ Summary工作表写入失败: {e}")

    def _write_predictions_sheet(self, writer, pred_data: Dict):
        """写入Predictions工作表"""
        try:
            if pred_data['count'] == 0:
                logger.warning("⚠️ 无预测数据，跳过Predictions工作表")
                return

            df = pd.DataFrame({
                'Date': pred_data['dates'],
                'Ticker': pred_data['tickers'],
                'Prediction': pred_data['predictions']
            })

            # 按预测值降序排序
            df = df.sort_values('Prediction', ascending=False).reset_index(drop=True)
            df.to_excel(writer, sheet_name='Predictions', index=False)
            logger.info(f"✓ Predictions工作表已写入 ({len(df)} 条)")

        except Exception as e:
            logger.error(f"✗ Predictions工作表写入失败: {e}")

    def _write_lambda_sheet(self, writer, lambda_df: pd.DataFrame):
        """写入Lambda Predictions工作表（确保每个ticker只有一条记录，按分数排序）"""
        try:
            # 🔧 CRITICAL FIX: 确保每个ticker只有一个预测
            if 'ticker' in lambda_df.columns:
                initial_count = len(lambda_df)
                # 按日期排序，保留每个ticker最新的预测
                lambda_df = lambda_df.sort_values(['ticker', 'date'] if 'date' in lambda_df.columns else 'ticker')
                lambda_df = lambda_df.groupby('ticker', as_index=False).last()

                if len(lambda_df) < initial_count:
                    logger.warning(f"⚠️ Lambda去重: {initial_count} → {len(lambda_df)} 条 (每个ticker保留一条)")

            # 🎯 按lambda_score降序排序
            score_col = 'lambda_score' if 'lambda_score' in lambda_df.columns else 'lambda_pct'
            if score_col in lambda_df.columns:
                lambda_df = lambda_df.sort_values(score_col, ascending=False).reset_index(drop=True)
                lambda_df.insert(0, 'rank', range(1, len(lambda_df) + 1))

            lambda_df.to_excel(writer, sheet_name='Lambda_Predictions', index=False)
            logger.info(f"✓ Lambda_Predictions工作表已写入 ({len(lambda_df)} 条，已按分数排序)")
        except Exception as e:
            logger.error(f"✗ Lambda_Predictions工作表写入失败: {e}")

    def _write_base_models_sheet(self, writer, base_models_df: pd.DataFrame):
        """写入Base Model Predictions工作表（按ticker排序并保留原始数据列）"""
        try:
            if not isinstance(base_models_df, pd.DataFrame):
                df = pd.DataFrame(base_models_df)
            else:
                df = base_models_df.copy()

            if df.empty:
                logger.warning("⚠️ Base Model数据为空，跳过Base_Model_Predictions工作表")
                return

            if 'ticker' in df.columns:
                sort_columns: List[str] = ['ticker']
                if 'date' in df.columns:
                    sort_columns.append('date')
                df = df.sort_values(sort_columns)

            df.to_excel(writer, sheet_name='Base_Model_Predictions', index=False)
            logger.info(f"✅ Base_Model_Predictions工作表已写入 ({len(df)} 条)")
        except Exception as e:
            logger.error(f"❌ Base_Model_Predictions工作表写入失败: {e}")

    def _write_ridge_sheet(self, writer, ridge_df: pd.DataFrame):
        """写入Ridge Predictions工作表（确保每个ticker只有一条记录，按分数排序）"""
        try:
            # 🔧 CRITICAL FIX: 确保每个ticker只有一个预测
            if 'ticker' in ridge_df.columns:
                initial_count = len(ridge_df)
                # 按日期排序，保留每个ticker最新的预测
                ridge_df = ridge_df.sort_values(['ticker', 'date'] if 'date' in ridge_df.columns else 'ticker')
                ridge_df = ridge_df.groupby('ticker', as_index=False).last()

                if len(ridge_df) < initial_count:
                    logger.warning(f"⚠️ Ridge去重: {initial_count} → {len(ridge_df)} 条 (每个ticker保留一条)")

            # 🎯 按ridge_score降序排序
            score_col = 'ridge_score' if 'ridge_score' in ridge_df.columns else 'ridge_z'
            if score_col in ridge_df.columns:
                ridge_df = ridge_df.sort_values(score_col, ascending=False).reset_index(drop=True)
                ridge_df.insert(0, 'rank', range(1, len(ridge_df) + 1))

            ridge_df.to_excel(writer, sheet_name='Ridge_Predictions', index=False)
            logger.info(f"✓ Ridge_Predictions工作表已写入 ({len(ridge_df)} 条，已按分数排序)")
        except Exception as e:
            logger.error(f"✗ Ridge_Predictions工作表写入失败: {e}")

    def _write_final_sheet(self, writer, final_df: pd.DataFrame):
        """写入Final Predictions工作表（确保每个ticker只有一条记录，按分数排序）"""
        try:
            # 🔧 CRITICAL FIX: 确保每个ticker只有一个预测
            if 'ticker' in final_df.columns:
                initial_count = len(final_df)
                uniq_tickers = final_df['ticker'].nunique()
                # 限制最大导出行数，避免超出Excel上限
                MAX_ROWS = 50000
                if initial_count > MAX_ROWS:
                    logger.warning(f"⚠️ Final_Predictions过大: {initial_count} 行，截断到 {MAX_ROWS} 行")
                    final_df = final_df.head(MAX_ROWS).copy()
                    initial_count = len(final_df)

                # 仅当去重后不会异常缩小时才执行（防止异常数据造成只剩少数ticker）
                # 条件：唯一ticker数≥min(10, 50%样本数)
                if uniq_tickers >= max(10, int(initial_count * 0.5)):
                    # 按日期排序，保留每个ticker最新的预测
                    final_df = final_df.sort_values(['ticker', 'date'] if 'date' in final_df.columns else 'ticker')
                    final_df = final_df.groupby('ticker', as_index=False).last()
                    if len(final_df) < initial_count:
                        logger.warning(f"⚠️ 去重: {initial_count} → {len(final_df)} 条 (每个ticker保留一条)")
                else:
                    logger.warning(
                        f"⚠️ 跳过去重：唯一ticker过少 ({uniq_tickers}/{initial_count})，保留全部记录以避免信息丢失")

            # 🎯 按final_score降序排序（最好的预测在最前面）
            score_col = 'final_score' if 'final_score' in final_df.columns else final_df.select_dtypes(include=['number']).columns[0]
            final_df = final_df.sort_values(score_col, ascending=False).reset_index(drop=True)

            # 添加排名列
            final_df.insert(0, 'rank', range(1, len(final_df) + 1))

            # 再次限制导出表大小，避免任何环节超限
            if len(final_df) > 50000:
                final_df = final_df.head(50000)
            final_df.to_excel(writer, sheet_name='Final_Predictions', index=False)
            logger.info(f"✓ Final_Predictions工作表已写入 ({len(final_df)} 条，已按分数排序)")
        except Exception as e:
            logger.error(f"✗ Final_Predictions工作表写入失败: {e}")

    def _write_kronos_sheet(self, writer, kronos_df: pd.DataFrame, sheet_name: str = 'Kronos_Filter'):
        """写入Kronos相关工作表"""
        try:
            kronos_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Kronos sheet '{sheet_name}' written ({len(kronos_df)} rows)")
        except Exception as e:
            logger.error(f"Failed to write Kronos sheet '{sheet_name}': {e}")

    def _write_tradingagents_sheet(self, writer, tradingagents_df: pd.DataFrame):
        """写入TradingAgents Analysis工作表"""
        try:
            # 确保按ticker排序
            if 'ticker' in tradingagents_df.columns:
                tradingagents_df = tradingagents_df.sort_values('ticker').reset_index(drop=True)

            # 添加排名列（如果还没有）
            if 'rank' not in tradingagents_df.columns:
                tradingagents_df.insert(0, 'rank', range(1, len(tradingagents_df) + 1))

            tradingagents_df.to_excel(writer, sheet_name='TradingAgents_Analysis', index=False)
            logger.info(f"✓ TradingAgents_Analysis工作表已写入 ({len(tradingagents_df)} 条)")
        except Exception as e:
            logger.error(f"✗ TradingAgents_Analysis工作表写入失败: {e}")

    def _write_factor_contributions_sheet(self, writer, model_info: Dict):
        """写入Factor Contributions工作表"""
        try:
            # 默认因子贡献（如果没有实际数据）
            # REMOVED: ivol_60d (multicollinearity with stability_score, VIF=10.4)
            factor_data = [
                ['liquid_momentum', 0.058],
                ['obv_divergence', 0.052],
                ['ivol_20', 0.047],
                ['rsrs_beta_18', 0.045],
                ['rsi_21', 0.052],
                ['bollinger_squeeze', 0.035],
                ['blowoff_ratio', 0.033],
                ['atr_ratio', 0.032],
                ['vol_ratio_20d', 0.029],
                ['price_ma60_deviation', 0.044],
                ['near_52w_high', 0.062],
                ['ret_skew_20d', 0.039],
                ['trend_r2_60', 0.041]
            ]

            df = pd.DataFrame(factor_data, columns=['Factor', 'Contribution'])
            df = df.sort_values('Contribution', ascending=False).reset_index(drop=True)
            df.to_excel(writer, sheet_name='Factor_Contributions', index=False)
            logger.info("✓ Factor_Contributions工作表已写入")

        except Exception as e:
            logger.error(f"✗ Factor_Contributions工作表写入失败: {e}")


    def _write_model_prediction_tables(self, writer, tables: Dict[str, pd.DataFrame], include_models: Optional[Tuple[str, ...]] = None, use_special_handlers: bool = True) -> None:
        """Write per-model prediction tables in a consistent order."""
        if not tables:
            return

        sheet_map = {
            'catboost': 'CatBoost_Predictions',
            'xgboost': 'XGBoost_Predictions',
            'elastic_net': 'ElasticNet_Predictions',
            'ridge': 'Ridge_Predictions',
            'lambdarank': 'LambdaRank_Predictions',
        }
        order = list(include_models) if include_models is not None else ['catboost', 'xgboost', 'elastic_net', 'ridge', 'lambdarank']

        for key in order:
            df = tables.get(key)
            if df is None or df.empty:
                continue

            try:
                if use_special_handlers and key == 'ridge':
                    self._write_ridge_sheet(writer, df)
                    continue
                if use_special_handlers and key == 'lambdarank':
                    self._write_lambda_sheet(writer, df)
                    continue

                export_df = df.copy()
                for col in ['date', 'actual_base_date', 'actual_target_date']:
                    if col in export_df.columns:
                        export_df[col] = pd.to_datetime(export_df[col], errors='coerce').dt.strftime('%Y-%m-%d')

                sheet_name = sheet_map.get(key, key.title())
                export_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"[EXPORT] Wrote sheet {sheet_name} ({len(export_df)} rows)")
            except Exception as exc:
                logger.error(f"[EXPORT] Failed to write sheet for {key}: {exc}")

    def _write_top30_summary(self, writer, summary_df: pd.DataFrame) -> None:
        """Write summary statistics for top-30 T+5 returns."""
        if summary_df is None or summary_df.empty:
            return

        try:
            export_df = summary_df.copy()
            if 'avg_t5_return_pct' in export_df.columns:
                export_df['avg_t5_return_pct'] = pd.to_numeric(export_df['avg_t5_return_pct'], errors='coerce')
            export_df.to_excel(writer, sheet_name='Top30_T5_Summary', index=False)
            logger.info(f"[EXPORT] Wrote sheet Top30_T5_Summary ({len(export_df)} rows)")
        except Exception as exc:
            logger.error(f"[EXPORT] Failed to write Top30_T5_Summary: {exc}")

    def _write_top30_details(self, writer, detail_tables: Dict[str, pd.DataFrame]) -> None:
        """Write per-model top-30 T+5 return tables."""
        if not detail_tables:
            return

        sheet_map = {
            'catboost': 'CatBoost_Top30_T5',
            'xgboost': 'XGBoost_Top30_T5',
            'elastic_net': 'ElasticNet_Top30_T5',
            'ridge': 'Ridge_Top30_T5',
            'lambdarank': 'LambdaRank_Top30_T5',
        }
        order = ['catboost', 'xgboost', 'elastic_net', 'ridge', 'lambdarank']

        for key in order:
            df = detail_tables.get(key)
            if df is None or df.empty:
                continue

            try:
                export_df = df.copy()
                for col in ['date', 'actual_base_date', 'actual_target_date']:
                    if col in export_df.columns:
                        export_df[col] = pd.to_datetime(export_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                sheet_name = sheet_map.get(key, f"{key.title()}_Top30_T5")
                export_df = export_df.head(30)
                export_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"[EXPORT] Wrote sheet {sheet_name} ({len(export_df)} rows)")
            except Exception as exc:
                logger.error(f"[EXPORT] Failed to write top30 detail for {key}: {exc}")



if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    test_predictions = pd.Series(
        [0.05, 0.03, -0.02, 0.01],
        index=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    )

    test_results = {
        'execution_time': 120.5
    }

    exporter = RobustExcelExporter()
    path = exporter.safe_export(
        predictions_series=test_predictions,
        analysis_results=test_results
    )

    print(f"\n测试完成: {path}")
