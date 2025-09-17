#!/usr/bin/env python3
"""
Unified Training Monitor - Comprehensive training metrics tracking and monitoring
Solves Problem 4: No Training Monitoring
"""

import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch/fold"""
    epoch: int
    fold: int
    train_loss: float
    val_loss: float
    train_score: float
    val_score: float
    ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class CVFoldMetrics:
    """Metrics for a single CV fold"""
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train_samples: int
    n_val_samples: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    feature_stats: Dict[str, Any]
    data_quality: Dict[str, Any]
    leakage_check: Dict[str, bool]

    def check_for_leakage(self) -> bool:
        """Check if there's potential data leakage"""
        # Check if validation performance is suspiciously high
        if self.val_metrics.get('ic', 0) > 0.15:
            logger.warning(f"Fold {self.fold_id}: Suspiciously high validation IC: {self.val_metrics['ic']}")
            return True

        # Check if val score >> train score
        val_score = self.val_metrics.get('score', 0)
        train_score = self.train_metrics.get('score', 0)
        if val_score > train_score * 1.5:
            logger.warning(f"Fold {self.fold_id}: Val score ({val_score}) >> Train score ({train_score})")
            return True

        return False


class UnifiedTrainingMonitor:
    """
    Unified training monitoring system with real-time metrics tracking,
    data leakage detection, and performance visualization
    """

    def __init__(self, experiment_name: str = None, save_dir: str = "training_logs"):
        """
        Initialize training monitor

        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save training logs and metrics
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training history storage
        self.training_history = []
        self.cv_fold_metrics = []
        self.epoch_metrics = defaultdict(list)

        # Real-time monitoring
        self.current_epoch = 0
        self.current_fold = 0
        self.training_start_time = None
        self.fold_start_time = None

        # Performance tracking
        self.best_metrics = {
            'best_val_score': -np.inf,
            'best_val_ic': -np.inf,
            'best_sharpe': -np.inf,
            'best_epoch': 0,
            'best_fold': 0
        }

        # Data quality monitoring
        self.data_quality_issues = []
        self.leakage_warnings = []

        # Feature importance tracking
        self.feature_importance_history = defaultdict(list)

        # Initialize logging
        self._setup_logging()

        logger.info(f"Training Monitor initialized for experiment: {self.experiment_name}")

    def _setup_logging(self):
        """Setup file logging for training metrics"""
        log_file = self.save_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def start_training(self):
        """Mark the start of training"""
        self.training_start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"Training started for experiment: {self.experiment_name}")
        logger.info("=" * 80)

    def start_fold(self, fold_id: int, train_dates: Tuple[str, str], val_dates: Tuple[str, str],
                   n_train: int, n_val: int):
        """
        Mark the start of a CV fold

        Args:
            fold_id: Fold identifier
            train_dates: (start_date, end_date) for training set
            val_dates: (start_date, end_date) for validation set
            n_train: Number of training samples
            n_val: Number of validation samples
        """
        self.current_fold = fold_id
        self.fold_start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting CV Fold {fold_id}")
        logger.info(f"Train period: {train_dates[0]} to {train_dates[1]} ({n_train} samples)")
        logger.info(f"Val period: {val_dates[0]} to {val_dates[1]} ({n_val} samples)")

        # Check for temporal overlap (data leakage)
        if train_dates[1] >= val_dates[0]:
            warning = f"WARNING: Potential data leakage! Train end ({train_dates[1]}) >= Val start ({val_dates[0]})"
            logger.error(warning)
            self.leakage_warnings.append(warning)

        logger.info(f"{'='*60}\n")

    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float,
                         train_score: float, val_score: float, **kwargs):
        """
        Log metrics for a training epoch

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_score: Training score (e.g., R2, accuracy)
            val_score: Validation score
            **kwargs: Additional metrics (ic, sharpe, etc.)
        """
        self.current_epoch = epoch

        metrics = TrainingMetrics(
            epoch=epoch,
            fold=self.current_fold,
            train_loss=train_loss,
            val_loss=val_loss,
            train_score=train_score,
            val_score=val_score,
            **kwargs
        )

        self.training_history.append(metrics)
        self.epoch_metrics[self.current_fold].append(metrics)

        # Update best metrics
        if val_score > self.best_metrics['best_val_score']:
            self.best_metrics['best_val_score'] = val_score
            self.best_metrics['best_epoch'] = epoch
            self.best_metrics['best_fold'] = self.current_fold

        # Log to console and file
        logger.info(f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"Train Score: {train_score:.4f} | Val Score: {val_score:.4f}")

        # Check for overfitting
        if val_loss > train_loss * 1.5:
            logger.warning(f"Potential overfitting: Val loss ({val_loss:.4f}) >> Train loss ({train_loss:.4f})")

        # Check for underfitting
        if train_score < 0.01:
            logger.warning(f"Potential underfitting: Train score too low ({train_score:.4f})")

    def log_fold_complete(self, fold_id: int, fold_metrics: Dict[str, Any]):
        """
        Log completion of a CV fold

        Args:
            fold_id: Fold identifier
            fold_metrics: Dictionary containing fold-level metrics
        """
        fold_duration = time.time() - self.fold_start_time if self.fold_start_time else 0

        logger.info(f"\n{'='*60}")
        logger.info(f"CV Fold {fold_id} Complete")
        logger.info(f"Duration: {fold_duration:.2f} seconds")
        logger.info(f"Final Val Score: {fold_metrics.get('val_score', 'N/A')}")
        logger.info(f"Final Val IC: {fold_metrics.get('val_ic', 'N/A')}")
        logger.info(f"{'='*60}\n")

        # Store fold metrics
        self.cv_fold_metrics.append(fold_metrics)

        # Save intermediate results
        self.save_metrics()

    def log_feature_importance(self, importance_dict: Dict[str, float], fold: Optional[int] = None):
        """
        Log feature importance scores

        Args:
            importance_dict: Dictionary of feature names to importance scores
            fold: Optional fold identifier
        """
        fold = fold or self.current_fold

        # Store history
        for feature, importance in importance_dict.items():
            self.feature_importance_history[feature].append(importance)

        # Log top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"\nTop 10 Important Features (Fold {fold}):")
        for i, (feature, importance) in enumerate(sorted_features, 1):
            logger.info(f"  {i:2d}. {feature:30s}: {importance:.4f}")

    def check_data_quality(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Check data quality and detect potential issues

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'train_nulls': X_train.isnull().sum().sum(),
            'val_nulls': X_val.isnull().sum().sum(),
            'train_inf': np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum(),
            'val_inf': np.isinf(X_val.select_dtypes(include=[np.number])).sum().sum(),
            'train_constant_features': (X_train.std() == 0).sum(),
            'val_constant_features': (X_val.std() == 0).sum(),
            'feature_distribution_shift': self._check_distribution_shift(X_train, X_val),
            'target_distribution_shift': self._check_target_shift(y_train, y_val)
        }

        # Log warnings
        if quality_report['train_nulls'] > 0:
            logger.warning(f"Training data contains {quality_report['train_nulls']} null values")

        if quality_report['train_constant_features'] > 0:
            logger.warning(f"Training data has {quality_report['train_constant_features']} constant features")

        if quality_report['feature_distribution_shift'] > 0.1:
            logger.warning(f"Significant feature distribution shift detected: {quality_report['feature_distribution_shift']:.3f}")

        return quality_report

    def _check_distribution_shift(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> float:
        """Check for distribution shift between train and validation sets"""
        try:
            # Use KS test for numerical features
            shifts = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                if col in X_val.columns:
                    from scipy.stats import ks_2samp
                    statistic, _ = ks_2samp(X_train[col].dropna(), X_val[col].dropna())
                    shifts.append(statistic)

            return np.mean(shifts) if shifts else 0.0
        except Exception as e:
            logger.warning(f"Could not check distribution shift: {e}")
            return 0.0

    def _check_target_shift(self, y_train: pd.Series, y_val: pd.Series) -> float:
        """Check for target distribution shift"""
        try:
            from scipy.stats import ks_2samp
            statistic, _ = ks_2samp(y_train.dropna(), y_val.dropna())
            return statistic
        except Exception as e:
            logger.warning(f"Could not check target shift: {e}")
            return 0.0

    def plot_training_curves(self, save: bool = True, show: bool = False):
        """
        Plot training curves for loss and metrics

        Args:
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if not self.training_history:
            logger.warning("No training history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Convert history to DataFrame
        df = pd.DataFrame([m.to_dict() for m in self.training_history])

        # Plot loss curves
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold]
            axes[0, 0].plot(fold_df['epoch'], fold_df['train_loss'], label=f'Fold {fold} Train', alpha=0.7)
            axes[0, 0].plot(fold_df['epoch'], fold_df['val_loss'], label=f'Fold {fold} Val', alpha=0.7, linestyle='--')

        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot score curves
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold]
            axes[0, 1].plot(fold_df['epoch'], fold_df['train_score'], label=f'Fold {fold} Train', alpha=0.7)
            axes[0, 1].plot(fold_df['epoch'], fold_df['val_score'], label=f'Fold {fold} Val', alpha=0.7, linestyle='--')

        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Training and Validation Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot IC evolution
        if 'ic' in df.columns:
            for fold in df['fold'].unique():
                fold_df = df[df['fold'] == fold]
                axes[1, 0].plot(fold_df['epoch'], fold_df['ic'], label=f'Fold {fold}', alpha=0.7)

            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IC')
            axes[1, 0].set_title('Information Coefficient Evolution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot feature importance
        if self.feature_importance_history:
            # Get average importance for top features
            avg_importance = {feat: np.mean(scores) for feat, scores in self.feature_importance_history.items()}
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]

            features, scores = zip(*top_features)
            axes[1, 1].barh(range(len(features)), scores)
            axes[1, 1].set_yticks(range(len(features)))
            axes[1, 1].set_yticklabels(features)
            axes[1, 1].set_xlabel('Average Importance')
            axes[1, 1].set_title('Top 15 Feature Importance')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Training Monitor: {self.experiment_name}')
        plt.tight_layout()

        if save:
            plot_path = self.save_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            logger.info(f"Training curves saved to {plot_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics_file = self.save_dir / 'training_metrics.json'

        metrics_data = {
            'experiment_name': self.experiment_name,
            'training_history': [m.to_dict() for m in self.training_history],
            'cv_fold_metrics': self.cv_fold_metrics,
            'best_metrics': self.best_metrics,
            'data_quality_issues': self.data_quality_issues,
            'leakage_warnings': self.leakage_warnings,
            'feature_importance': dict(self.feature_importance_history)
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        logger.info(f"Metrics saved to {metrics_file}")

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report of training

        Returns:
            Dictionary containing training summary
        """
        if not self.training_history:
            return {"error": "No training history available"}

        df = pd.DataFrame([m.to_dict() for m in self.training_history])

        total_duration = time.time() - self.training_start_time if self.training_start_time else 0

        summary = {
            'experiment_name': self.experiment_name,
            'total_duration_seconds': total_duration,
            'total_duration_readable': f"{total_duration/60:.2f} minutes",
            'total_epochs': len(self.training_history),
            'total_folds': df['fold'].nunique(),
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'mean_val_score': df.groupby('fold')['val_score'].last().mean(),
                'std_val_score': df.groupby('fold')['val_score'].last().std(),
                'mean_val_ic': df.groupby('fold')['ic'].last().mean() if 'ic' in df else None,
                'std_val_ic': df.groupby('fold')['ic'].last().std() if 'ic' in df else None,
            },
            'convergence': {
                'converged': self._check_convergence(df),
                'overfitting': self._check_overfitting(df),
                'underfitting': self._check_underfitting(df)
            },
            'data_quality': {
                'total_issues': len(self.data_quality_issues),
                'leakage_warnings': len(self.leakage_warnings)
            }
        }

        return summary

    def _check_convergence(self, df: pd.DataFrame) -> bool:
        """Check if training has converged"""
        # Check if validation loss has plateaued
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold].tail(10)
            if len(fold_df) >= 5:
                recent_losses = fold_df['val_loss'].values
                if np.std(recent_losses) < 0.001:
                    return True
        return False

    def _check_overfitting(self, df: pd.DataFrame) -> bool:
        """Check for overfitting"""
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold].tail(1)
            if len(fold_df) > 0:
                row = fold_df.iloc[0]
                if row['val_loss'] > row['train_loss'] * 1.5:
                    return True
        return False

    def _check_underfitting(self, df: pd.DataFrame) -> bool:
        """Check for underfitting"""
        for fold in df['fold'].unique():
            fold_df = df[df['fold'] == fold].tail(1)
            if len(fold_df) > 0:
                row = fold_df.iloc[0]
                if row['train_score'] < 0.01:
                    return True
        return False

    def print_summary(self):
        """Print training summary to console"""
        summary = self.get_summary_report()

        print("\n" + "="*80)
        print(f"TRAINING SUMMARY: {summary['experiment_name']}")
        print("="*80)
        print(f"Duration: {summary['total_duration_readable']}")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Total Folds: {summary['total_folds']}")
        print("\nBest Metrics:")
        for key, value in summary['best_metrics'].items():
            print(f"  {key}: {value}")
        print("\nFinal Metrics:")
        for key, value in summary['final_metrics'].items():
            if value is not None:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print("\nTraining Status:")
        for key, value in summary['convergence'].items():
            print(f"  {key}: {value}")
        print("\nData Quality:")
        for key, value in summary['data_quality'].items():
            print(f"  {key}: {value}")

        if self.leakage_warnings:
            print("\n⚠️ DATA LEAKAGE WARNINGS:")
            for warning in self.leakage_warnings:
                print(f"  - {warning}")

        print("="*80 + "\n")


def create_training_monitor(experiment_name: Optional[str] = None) -> UnifiedTrainingMonitor:
    """
    Factory function to create a training monitor

    Args:
        experiment_name: Optional experiment name

    Returns:
        UnifiedTrainingMonitor instance
    """
    return UnifiedTrainingMonitor(experiment_name=experiment_name)


# Example usage integration
def example_training_with_monitoring():
    """Example of how to integrate the monitor with training code"""

    # Create monitor
    monitor = create_training_monitor("example_experiment")
    monitor.start_training()

    # Simulated CV loop
    for fold in range(3):
        # Start fold
        monitor.start_fold(
            fold_id=fold,
            train_dates=("2023-01-01", "2023-06-30"),
            val_dates=("2023-07-01", "2023-09-30"),
            n_train=1000,
            n_val=300
        )

        # Simulated training loop
        for epoch in range(10):
            # Simulate metrics
            train_loss = np.random.random() * (0.5 - epoch * 0.04)
            val_loss = train_loss * (1 + np.random.random() * 0.2)
            train_score = min(0.9, epoch * 0.08 + np.random.random() * 0.1)
            val_score = train_score * (0.9 + np.random.random() * 0.1)
            ic = np.random.random() * 0.1

            # Log metrics
            monitor.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_score=train_score,
                val_score=val_score,
                ic=ic
            )

        # Log feature importance
        feature_importance = {f"feature_{i}": np.random.random() for i in range(20)}
        monitor.log_feature_importance(feature_importance, fold)

        # Complete fold
        monitor.log_fold_complete(fold, {
            'val_score': val_score,
            'val_ic': ic
        })

    # Generate plots and summary
    monitor.plot_training_curves(save=True)
    monitor.print_summary()
    monitor.save_metrics()


if __name__ == "__main__":
    # Run example
    example_training_with_monitoring()