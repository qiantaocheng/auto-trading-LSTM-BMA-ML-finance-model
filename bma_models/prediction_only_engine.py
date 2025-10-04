#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction-Only Engine
======================
Load trained models from snapshot and predict without retraining.

Workflow:
1. Load model snapshots (ElasticNet, XGBoost, CatBoost, LambdaRank, Ridge)
2. Fetch market data for input stocks
3. Calculate 17 alpha factors using Simple17FactorEngine
4. Generate predictions using all 5 models
5. Combine predictions using Ridge stacker
6. Return top recommendations

No training - only prediction using saved model weights.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PredictionOnlyEngine:
    """预测引擎 - 仅使用已保存的模型快照进行预测"""

    def __init__(self, snapshot_id: Optional[str] = None):
        """
        初始化预测引擎

        Args:
            snapshot_id: 模型快照ID，如果为None则使用最新快照
        """
        self.snapshot_id = snapshot_id
        self.manifest = None
        self.models = {}
        self.ridge_stacker = None
        self.lambda_rank_stacker = None
        self.lambda_percentile_transformer = None
        self.feature_names = None

        # Load models from snapshot
        self._load_models()

    def _load_models(self):
        """从快照加载所有模型"""
        from bma_models.model_registry import load_manifest, load_models_from_snapshot

        logger.info("=" * 80)
        logger.info("📦 加载模型快照")
        logger.info("=" * 80)

        # Load manifest
        self.manifest = load_manifest(self.snapshot_id)
        self.snapshot_id = self.manifest['snapshot_id']
        self.feature_names = self.manifest.get('feature_names', [])

        logger.info(f"快照ID: {self.snapshot_id}")
        logger.info(f"创建时间: {datetime.fromtimestamp(self.manifest['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"标签: {self.manifest.get('tag', 'N/A')}")

        # Load all models
        loaded = load_models_from_snapshot(self.snapshot_id)

        self.models = loaded['models']
        self.ridge_stacker = loaded.get('ridge_stacker')
        self.lambda_rank_stacker = loaded.get('lambda_rank_stacker')
        self.lambda_percentile_transformer = loaded.get('lambda_percentile_transformer')

        # Log loaded models
        loaded_model_names = []
        if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
            loaded_model_names.append("ElasticNet")
        if 'xgboost' in self.models and self.models['xgboost'] is not None:
            loaded_model_names.append("XGBoost")
        if 'catboost' in self.models and self.models['catboost'] is not None:
            loaded_model_names.append("CatBoost")
        if self.lambda_rank_stacker is not None:
            loaded_model_names.append("LambdaRank")
        if self.ridge_stacker is not None:
            loaded_model_names.append("Ridge")

        logger.info(f"✅ 成功加载 {len(loaded_model_names)}/5 个模型: {', '.join(loaded_model_names)}")
        logger.info("=" * 80)

    def predict(self,
                tickers: List[str],
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                top_n: int = 10) -> Dict[str, Any]:
        """
        预测股票

        Args:
            tickers: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)，默认为30天前
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            top_n: 返回前N个推荐

        Returns:
            预测结果字典
        """
        logger.info("=" * 80)
        logger.info("🎯 开始预测")
        logger.info("=" * 80)
        logger.info(f"输入股票: {len(tickers)} 只")
        logger.info(f"股票列表: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")

        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        logger.info(f"数据时间范围: {start_date} 至 {end_date}")

        # Step 1: Fetch market data and calculate factors
        feature_data = self._get_features(tickers, start_date, end_date)

        if feature_data is None or len(feature_data) == 0:
            logger.error("❌ 无法获取特征数据")
            return {'success': False, 'error': '无法获取特征数据'}

        logger.info(f"✅ 特征数据准备完成: {feature_data.shape}")

        # Step 2: Generate predictions from all models
        predictions = self._generate_predictions(feature_data)

        if predictions is None:
            logger.error("❌ 预测失败")
            return {'success': False, 'error': '预测失败'}

        logger.info(f"✅ 预测完成: {len(predictions)} 只股票")

        # Step 3: Get latest predictions (most recent date)
        latest_predictions = self._get_latest_predictions(predictions, tickers)

        # Step 4: Create recommendations
        recommendations = self._create_recommendations(latest_predictions, top_n)

        logger.info("=" * 80)
        logger.info(f"🏆 Top {len(recommendations)} 推荐:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec['ticker']}: {rec['score']:.6f}")
        logger.info("=" * 80)

        return {
            'success': True,
            'snapshot_id': self.snapshot_id,
            'predictions': predictions,
            'recommendations': recommendations,
            'tickers': tickers,
            'n_stocks': len(tickers),
            'date_range': f"{start_date} to {end_date}"
        }

    def _get_features(self,
                     tickers: List[str],
                     start_date: str,
                     end_date: str) -> Optional[pd.DataFrame]:
        """获取特征数据"""
        logger.info("📡 获取市场数据并计算因子...")

        try:
            # Initialize Simple17FactorEngine
            from bma_models.simple_25_factor_engine import Simple17FactorEngine

            # Calculate lookback days
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            lookback_days = (end_dt - start_dt).days + 50

            engine = Simple17FactorEngine(lookback_days=lookback_days)

            # Fetch market data
            market_data = engine.fetch_market_data(
                symbols=tickers,
                use_optimized_downloader=True,
                start_date=start_date,
                end_date=end_date
            )

            if market_data.empty:
                logger.error("❌ 无法获取市场数据")
                return None

            logger.info(f"✅ 市场数据: {market_data.shape}")

            # Calculate all 17 factors
            feature_data = engine.compute_all_17_factors(market_data)

            if feature_data.empty:
                logger.error("❌ 因子计算失败")
                return None

            logger.info(f"✅ 因子计算完成: {feature_data.shape}")
            logger.info(f"   因子列: {list(feature_data.columns)}")

            # Drop rows with missing target (if exists)
            if 'target' in feature_data.columns:
                feature_data = feature_data.drop(columns=['target'])

            # Drop Close column (not a feature)
            if 'Close' in feature_data.columns:
                feature_data = feature_data.drop(columns=['Close'])

            return feature_data

        except Exception as e:
            logger.error(f"❌ 特征计算失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _generate_predictions(self, feature_data: pd.DataFrame) -> Optional[pd.Series]:
        """使用所有模型生成预测"""
        logger.info("🔮 生成模型预测...")

        try:
            # Ensure feature_data has MultiIndex (date, ticker)
            if not isinstance(feature_data.index, pd.MultiIndex):
                logger.error("❌ feature_data必须有MultiIndex (date, ticker)")
                return None

            # Extract features (exclude metadata columns)
            metadata_cols = ['date', 'ticker', 'target', 'Close']
            feature_cols = [col for col in feature_data.columns if col not in metadata_cols]

            X = feature_data[feature_cols].copy()
            dates = feature_data.index.get_level_values('date')
            tickers = feature_data.index.get_level_values('ticker')

            logger.info(f"特征矩阵: {X.shape}")
            logger.info(f"使用特征: {feature_cols}")

            # Fill NaN with 0 (models were trained with this)
            X = X.fillna(0)

            # Generate first layer predictions
            first_layer_predictions = {}

            # ElasticNet
            if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
                try:
                    enet_pred = self.models['elastic_net'].predict(X)
                    first_layer_predictions['elastic_net'] = pd.Series(enet_pred, index=feature_data.index)
                    logger.info(f"✅ ElasticNet预测: {len(enet_pred)}")
                except Exception as e:
                    logger.warning(f"ElasticNet预测失败: {e}")

            # XGBoost
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                try:
                    xgb_pred = self.models['xgboost'].predict(X)
                    first_layer_predictions['xgboost'] = pd.Series(xgb_pred, index=feature_data.index)
                    logger.info(f"✅ XGBoost预测: {len(xgb_pred)}")
                except Exception as e:
                    logger.warning(f"XGBoost预测失败: {e}")

            # CatBoost
            if 'catboost' in self.models and self.models['catboost'] is not None:
                try:
                    cat_pred = self.models['catboost'].predict(X)
                    first_layer_predictions['catboost'] = pd.Series(cat_pred, index=feature_data.index)
                    logger.info(f"✅ CatBoost预测: {len(cat_pred)}")
                except Exception as e:
                    logger.warning(f"CatBoost预测失败: {e}")

            # LambdaRank
            if self.lambda_rank_stacker is not None:
                try:
                    # LambdaRank expects specific input format
                    lambda_input = X.copy()
                    lambda_input.index = feature_data.index
                    lambda_pred = self.lambda_rank_stacker.predict(lambda_input)

                    # Apply Lambda Percentile Transformer if available
                    if self.lambda_percentile_transformer is not None:
                        try:
                            lambda_pred = self.lambda_percentile_transformer.transform(lambda_pred)
                            logger.info("✅ Lambda Percentile Transformer应用成功")
                        except Exception as e:
                            logger.warning(f"Lambda Percentile Transformer应用失败: {e}")

                    first_layer_predictions['lambdarank'] = lambda_pred
                    logger.info(f"✅ LambdaRank预测: {len(lambda_pred)}")
                except Exception as e:
                    logger.warning(f"LambdaRank预测失败: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            if len(first_layer_predictions) == 0:
                logger.error("❌ 所有第一层模型预测失败")
                return None

            logger.info(f"✅ 第一层预测完成: {len(first_layer_predictions)} 个模型")

            # Combine using Ridge stacker
            if self.ridge_stacker is not None:
                try:
                    # Create stacking features
                    stacking_df = pd.DataFrame(index=feature_data.index)
                    for name, pred in first_layer_predictions.items():
                        stacking_df[name] = pred

                    # Ridge predict
                    final_predictions = self.ridge_stacker.predict(stacking_df)
                    logger.info(f"✅ Ridge Stacker融合完成: {len(final_predictions)}")

                    return final_predictions

                except Exception as e:
                    logger.warning(f"Ridge Stacker融合失败: {e}")
                    logger.warning("使用简单平均作为后备")
                    import traceback
                    logger.debug(traceback.format_exc())

            # Fallback: simple average
            stacked = pd.DataFrame(first_layer_predictions)
            final_predictions = stacked.mean(axis=1)
            logger.info(f"✅ 使用简单平均: {len(final_predictions)}")

            return final_predictions

        except Exception as e:
            logger.error(f"❌ 预测生成失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_latest_predictions(self,
                               predictions: pd.Series,
                               tickers: List[str]) -> pd.DataFrame:
        """获取最新日期的预测"""
        # Get latest date
        if isinstance(predictions.index, pd.MultiIndex):
            dates = predictions.index.get_level_values('date')
            latest_date = dates.max()

            # Filter to latest date and requested tickers
            mask = predictions.index.get_level_values('date') == latest_date
            latest_pred = predictions[mask]

            # Create DataFrame
            pred_df = pd.DataFrame({
                'ticker': latest_pred.index.get_level_values('ticker'),
                'score': latest_pred.values
            })
        else:
            # If not MultiIndex, assume single date
            pred_df = pd.DataFrame({
                'ticker': predictions.index,
                'score': predictions.values
            })

        # Filter to requested tickers only
        pred_df = pred_df[pred_df['ticker'].isin(tickers)]

        # Sort by score descending
        pred_df = pred_df.sort_values('score', ascending=False)

        return pred_df

    def _create_recommendations(self,
                               pred_df: pd.DataFrame,
                               top_n: int) -> List[Dict[str, Any]]:
        """创建推荐列表"""
        recommendations = []

        for i, (idx, row) in enumerate(pred_df.head(top_n).iterrows(), 1):
            recommendations.append({
                'rank': i,
                'ticker': row['ticker'],
                'score': float(row['score']),
                'prediction_signal': float(row['score']),
            })

        return recommendations


def create_prediction_engine(snapshot_id: Optional[str] = None) -> PredictionOnlyEngine:
    """工厂函数：创建预测引擎"""
    return PredictionOnlyEngine(snapshot_id=snapshot_id)
