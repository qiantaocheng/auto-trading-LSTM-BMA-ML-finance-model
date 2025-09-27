#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV-Bagging Inference Engine
Ensures training-inference consistency by using CV fold models for prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CVBaggingInference:
    """
    CV-Bagging inference engine to maintain training-inference consistency

    Problem:
    - Training: Ridge uses OOF predictions from CV folds
    - Inference: Ridge uses full-retrained model predictions
    - Solution: Use CV fold models to generate OOF-like predictions for inference
    """

    def __init__(self,
                 cv_folds: int = 5,
                 model_save_dir: str = "cv_models"):
        """
        Initialize CV-Bagging inference engine

        Args:
            cv_folds: Number of CV folds used in training
            model_save_dir: Directory to save/load CV fold models
        """
        self.cv_folds = cv_folds
        self.model_save_dir = Path(model_save_dir)
        self.fold_models = {}
        self.fold_mappings = {}

        # Ensure model directory exists
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def save_fold_models(self,
                        fold_models: Dict[int, Dict[str, Any]],
                        fold_mappings: Dict[int, np.ndarray],
                        experiment_id: str = "default") -> bool:
        """
        Save CV fold models and their data mappings

        Args:
            fold_models: {fold_idx: {model_name: model_object}}
            fold_mappings: {fold_idx: array_of_sample_indices}
            experiment_id: Unique identifier for this experiment

        Returns:
            Success status
        """
        try:
            # Save fold models
            models_path = self.model_save_dir / f"fold_models_{experiment_id}.pkl"
            with open(models_path, 'wb') as f:
                pickle.dump(fold_models, f)

            # Save fold mappings
            mappings_path = self.model_save_dir / f"fold_mappings_{experiment_id}.pkl"
            with open(mappings_path, 'wb') as f:
                pickle.dump(fold_mappings, f)

            self.fold_models = fold_models
            self.fold_mappings = fold_mappings

            logger.info(f"CV fold models saved: {len(fold_models)} folds")
            logger.info(f"Saved to: {models_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save CV fold models: {e}")
            return False

    def load_fold_models(self, experiment_id: str = "default") -> bool:
        """
        Load CV fold models and mappings

        Args:
            experiment_id: Experiment identifier

        Returns:
            Success status
        """
        try:
            # Load fold models
            models_path = self.model_save_dir / f"fold_models_{experiment_id}.pkl"
            if not models_path.exists():
                logger.error(f"CV fold models not found: {models_path}")
                return False

            with open(models_path, 'rb') as f:
                self.fold_models = pickle.load(f)

            # Load fold mappings
            mappings_path = self.model_save_dir / f"fold_mappings_{experiment_id}.pkl"
            with open(mappings_path, 'rb') as f:
                self.fold_mappings = pickle.load(f)

            logger.info(f"CV fold models loaded: {len(self.fold_models)} folds")
            return True

        except Exception as e:
            logger.error(f"Failed to load CV fold models: {e}")
            return False

    def generate_oof_like_predictions(self,
                                    X: pd.DataFrame,
                                    sample_indices: Optional[np.ndarray] = None) -> Dict[str, pd.Series]:
        """
        Generate OOF-like predictions using CV fold models

        For each sample, use models from folds that didn't see this sample during training

        Args:
            X: Features for prediction
            sample_indices: Original sample indices (for fold mapping)

        Returns:
            OOF-like predictions for each model
        """
        if not self.fold_models:
            raise ValueError("CV fold models not loaded. Call load_fold_models() first.")

        n_samples = len(X)
        if sample_indices is None:
            sample_indices = np.arange(n_samples)

        # Initialize prediction containers
        model_names = list(next(iter(self.fold_models.values())).keys())
        predictions = {name: np.full(n_samples, np.nan) for name in model_names}

        logger.info(f"Generating OOF-like predictions for {n_samples} samples")

        # For each sample, find which folds didn't train on it
        for i, sample_idx in enumerate(sample_indices):
            # Find folds that didn't see this sample
            available_folds = []
            for fold_idx, fold_indices in self.fold_mappings.items():
                if sample_idx not in fold_indices:  # This fold didn't train on this sample
                    available_folds.append(fold_idx)

            if not available_folds:
                # Fallback: use all folds (shouldn't happen with proper CV)
                available_folds = list(self.fold_models.keys())
                logger.warning(f"Sample {sample_idx} not found in any fold mapping, using all folds")

            # Average predictions from available folds
            for model_name in model_names:
                fold_predictions = []

                for fold_idx in available_folds:
                    if fold_idx in self.fold_models and model_name in self.fold_models[fold_idx]:
                        model = self.fold_models[fold_idx][model_name]
                        try:
                            # Single sample prediction
                            pred = model.predict(X.iloc[i:i+1])
                            if hasattr(pred, '__len__') and len(pred) > 0:
                                fold_predictions.append(pred[0])
                            else:
                                fold_predictions.append(pred)
                        except Exception as e:
                            logger.warning(f"Prediction failed for fold {fold_idx}, model {model_name}: {e}")

                # Average fold predictions
                if fold_predictions:
                    predictions[model_name][i] = np.mean(fold_predictions)
                else:
                    logger.warning(f"No valid predictions for sample {i}, model {model_name}")

        # Convert to pandas Series with proper index
        result = {}
        for model_name, preds in predictions.items():
            result[model_name] = pd.Series(preds, index=X.index)

        # Log statistics
        for model_name, pred_series in result.items():
            valid_preds = pred_series.dropna()
            logger.info(f"{model_name}: {len(valid_preds)}/{len(pred_series)} valid predictions")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded CV models"""
        if not self.fold_models:
            return {"status": "No models loaded"}

        info = {
            "n_folds": len(self.fold_models),
            "model_names": list(next(iter(self.fold_models.values())).keys()) if self.fold_models else [],
            "total_samples_mapped": sum(len(indices) for indices in self.fold_mappings.values()),
            "fold_sizes": {f"fold_{i}": len(indices) for i, indices in self.fold_mappings.items()}
        }

        return info

def integrate_cv_bagging_with_main_model():
    """
    Integration guide for main model

    This function shows how to modify the main model to use CV-bagging inference
    """

    integration_guide = """
    INTEGRATION STEPS FOR CV-BAGGING INFERENCE:

    1. MODIFY TRAINING PHASE:
    ========================

    In _unified_model_training() or similar:

    # After CV training, save fold models
    cv_bagger = CVBaggingInference(cv_folds=5)

    # Collect fold models during CV
    fold_models = {}
    fold_mappings = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Train models on this fold
        fold_models[fold_idx] = {
            'elastic_net': fitted_elastic_net,
            'xgboost': fitted_xgboost,
            'catboost': fitted_catboost
        }
        fold_mappings[fold_idx] = train_idx  # Samples this fold was trained on

    # Save for inference
    cv_bagger.save_fold_models(fold_models, fold_mappings, experiment_id="production")

    2. MODIFY INFERENCE PHASE:
    =========================

    In run_complete_analysis():

    # Replace single model predictions with CV-bagging
    cv_bagger = CVBaggingInference()
    cv_bagger.load_fold_models(experiment_id="production")

    # Generate OOF-like predictions
    oof_like_preds = cv_bagger.generate_oof_like_predictions(X)

    # Use these for Ridge stacker (consistent with training)
    first_layer_preds = pd.DataFrame({
        'pred_elastic': oof_like_preds['elastic_net'],
        'pred_xgb': oof_like_preds['xgboost'],
        'pred_catboost': oof_like_preds['catboost']
    })

    3. BENEFITS:
    ============

    ✓ Training-inference consistency
    ✓ Ridge sees same distribution as training
    ✓ Proper generalization (no data leakage)
    ✓ Maintains CV's benefit in production
    ✓ Better calibration and reliability

    4. MONITORING:
    ==============

    - Compare OOF vs CV-bagging predictions distribution
    - Monitor KS test between training and inference
    - Track Ridge performance stability
    """

    print(integration_guide)
    return integration_guide

if __name__ == "__main__":
    # Show integration guide
    integrate_cv_bagging_with_main_model()

    # Example usage
    print("\n" + "="*60)
    print("CV-BAGGING INFERENCE EXAMPLE")
    print("="*60)

    # Create example data
    cv_bagger = CVBaggingInference(cv_folds=3)

    # Simulate some fold models and mappings
    # (In practice, these come from actual CV training)
    fold_models = {
        0: {'model_a': None, 'model_b': None},  # Placeholder models
        1: {'model_a': None, 'model_b': None},
        2: {'model_a': None, 'model_b': None}
    }

    fold_mappings = {
        0: np.array([0, 1, 2, 3]),      # Fold 0 trained on samples 0-3
        1: np.array([4, 5, 6, 7]),      # Fold 1 trained on samples 4-7
        2: np.array([8, 9, 10, 11])     # Fold 2 trained on samples 8-11
    }

    print(f"Example setup: {len(fold_models)} folds")
    print(f"Sample mapping: {fold_mappings}")

    print("\nFor inference on sample 5:")
    print("- Sample 5 was in fold 1's training set")
    print("- So use models from folds 0 and 2 for prediction")
    print("- Average their outputs for final prediction")

    print("\nThis ensures no data leakage and maintains OOF-like characteristics!")