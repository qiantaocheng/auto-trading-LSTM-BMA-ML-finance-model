#!/usr/bin/env python3
"""
Model Version Control System
Manages model versioning, tracking, and deployment
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    created_at: str
    model_hash: str
    config_hash: str
    performance_metrics: Dict[str, float]
    feature_count: int
    training_samples: int
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ModelVersionControl:
    """Model version control and tracking system"""
    
    def __init__(self, base_path: str = "model_versions"):
        self.base_path = base_path
        self.versions_file = os.path.join(base_path, "versions.json")
        self.models_dir = os.path.join(base_path, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing versions
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load existing model versions"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                return {k: ModelVersion(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load versions file: {e}")
        return {}
    
    def _save_versions(self):
        """Save versions to file"""
        try:
            with open(self.versions_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.versions.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions file: {e}")
    
    def _calculate_model_hash(self, model) -> str:
        """Calculate hash of model parameters"""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.md5(model_bytes).hexdigest()
        except Exception:
            return "unknown"
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration"""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def register_model(self, 
                      model, 
                      config: Dict[str, Any], 
                      performance_metrics: Dict[str, float],
                      feature_count: int,
                      training_samples: int,
                      description: str = "",
                      tags: List[str] = None) -> str:
        """Register a new model version"""
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        
        # Calculate hashes
        model_hash = self._calculate_model_hash(model)
        config_hash = self._calculate_config_hash(config)
        
        # Create version object
        version = ModelVersion(
            version=version_id,
            created_at=datetime.now().isoformat(),
            model_hash=model_hash,
            config_hash=config_hash,
            performance_metrics=performance_metrics,
            feature_count=feature_count,
            training_samples=training_samples,
            description=description,
            tags=tags or []
        )
        
        # Save model file
        model_file = os.path.join(self.models_dir, f"{version_id}.pkl")
        try:
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'config': config,
                    'metadata': asdict(version)
                }, f)
        except Exception as e:
            logger.error(f"Failed to save model file: {e}")
            return None
        
        # Register version
        self.versions[version_id] = version
        self._save_versions()
        
        logger.info(f"Registered model version {version_id}")
        return version_id
    
    def load_model(self, version_id: str):
        """Load a specific model version"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        model_file = os.path.join(self.models_dir, f"{version_id}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file for version {version_id} not found")
        
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
            return data['model'], data['config']
        except Exception as e:
            logger.error(f"Failed to load model {version_id}: {e}")
            raise
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest model version"""
        if not self.versions:
            return None
        
        latest = max(self.versions.items(), key=lambda x: x[1].created_at)
        return latest[0]
    
    def get_best_version(self, metric: str = 'ic') -> Optional[str]:
        """Get the best performing model version"""
        if not self.versions:
            return None
        
        best = None
        best_score = float('-inf')
        
        for version_id, version in self.versions.items():
            if metric in version.performance_metrics:
                score = version.performance_metrics[metric]
                if score > best_score:
                    best_score = score
                    best = version_id
        
        return best
    
    def list_versions(self) -> List[ModelVersion]:
        """List all model versions"""
        return list(self.versions.values())
    
    def delete_version(self, version_id: str) -> bool:
        """Delete a model version"""
        if version_id not in self.versions:
            return False
        
        # Delete model file
        model_file = os.path.join(self.models_dir, f"{version_id}.pkl")
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
            except Exception as e:
                logger.error(f"Failed to delete model file: {e}")
        
        # Remove from versions
        del self.versions[version_id]
        self._save_versions()
        
        logger.info(f"Deleted model version {version_id}")
        return True
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """Keep only the N most recent versions"""
        if len(self.versions) <= keep_count:
            return
        
        # Sort by creation date
        sorted_versions = sorted(
            self.versions.items(), 
            key=lambda x: x[1].created_at, 
            reverse=True
        )
        
        # Delete old versions
        for version_id, _ in sorted_versions[keep_count:]:
            self.delete_version(version_id)
    
    def get_version_info(self, version_id: str) -> Optional[ModelVersion]:
        """Get information about a specific version"""
        return self.versions.get(version_id)
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("One or both versions not found")
        
        v1 = self.versions[version1]
        v2 = self.versions[version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'feature_count_diff': v2.feature_count - v1.feature_count,
            'training_samples_diff': v2.training_samples - v1.training_samples,
            'performance_comparison': {}
        }
        
        # Compare performance metrics
        all_metrics = set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys())
        for metric in all_metrics:
            val1 = v1.performance_metrics.get(metric, 0)
            val2 = v2.performance_metrics.get(metric, 0)
            comparison['performance_comparison'][metric] = {
                'version1': val1,
                'version2': val2,
                'improvement': val2 - val1
            }
        
        return comparison

# Global instance
model_version_control = ModelVersionControl()

def register_model(*args, **kwargs):
    """Convenience function for model registration"""
    return model_version_control.register_model(*args, **kwargs)

def load_latest_model():
    """Load the latest model version"""
    latest_version = model_version_control.get_latest_version()
    if latest_version:
        return model_version_control.load_model(latest_version)
    return None, None

def load_best_model(metric: str = 'ic'):
    """Load the best performing model"""
    best_version = model_version_control.get_best_version(metric)
    if best_version:
        return model_version_control.load_model(best_version)
    return None, None