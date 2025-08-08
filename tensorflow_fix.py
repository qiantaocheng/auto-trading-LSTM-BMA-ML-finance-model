"""
TensorFlow GPU Configuration and Optimization Utilities
Provides GPU setup, memory monitoring, and model optimization functions
"""

import os
import sys
import warnings
import logging
from typing import Optional, Union, Dict, Any
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_gpu(force_gpu: bool = False, memory_growth: bool = True, 
                  memory_limit: Optional[int] = None) -> bool:
    """
    Configure TensorFlow GPU settings
    
    Args:
        force_gpu: Force GPU usage even if not optimal
        memory_growth: Enable memory growth to prevent OOM
        memory_limit: Set memory limit in MB
        
    Returns:
        bool: True if GPU is available and configured
    """
    try:
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPU devices found")
            return False
            
        logger.info(f"Found {len(gpus)} GPU device(s)")
        
        # Configure GPU memory growth
        if memory_growth:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU: {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Failed to set memory growth for {gpu}: {e}")
        
        # Set memory limit if specified
        if memory_limit is not None:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"Set GPU memory limit to {memory_limit}MB")
            except RuntimeError as e:
                logger.warning(f"Failed to set memory limit: {e}")
        
        # Test GPU availability
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(test_tensor)
            logger.info(f"GPU test successful: {result.numpy()}")
            
        return True
        
    except ImportError:
        logger.error("TensorFlow not available")
        return False
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        return False

def set_gpu_strategy():
    """
    Set up TensorFlow distributed strategy for GPU
    
    Returns:
        tf.distribute.Strategy: Configured strategy
    """
    try:
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPU available, using CPU strategy")
            return tf.distribute.OneDeviceStrategy("/cpu:0")
        
        # Use MirroredStrategy for multiple GPUs, OneDeviceStrategy for single GPU
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            logger.info("Using OneDeviceStrategy with single GPU")
            
        return strategy
        
    except ImportError:
        logger.error("TensorFlow not available")
        return None
    except Exception as e:
        logger.error(f"Failed to set GPU strategy: {e}")
        return None

def monitor_gpu_memory():
    """
    Monitor GPU memory usage
    
    Returns:
        dict: Memory usage information
    """
    try:
        import tensorflow as tf
        
        # Get GPU memory info
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return {"error": "No GPU available"}
        
        memory_info = {}
        for i, gpu in enumerate(gpus):
            try:
                # This is a simplified version - in practice you might use nvidia-ml-py
                memory_info[f"GPU_{i}"] = {
                    "device": str(gpu),
                    "status": "available"
                }
            except Exception as e:
                memory_info[f"GPU_{i}"] = {
                    "device": str(gpu),
                    "status": f"error: {e}"
                }
        
        logger.info(f"GPU memory info: {memory_info}")
        return memory_info
        
    except ImportError:
        logger.error("TensorFlow not available")
        return {"error": "TensorFlow not available"}
    except Exception as e:
        logger.error(f"Failed to monitor GPU memory: {e}")
        return {"error": str(e)}

def safe_load_model(model_path: str, custom_objects: Optional[Dict[str, Any]] = None):
    """
    Safely load a TensorFlow model with error handling
    
    Args:
        model_path: Path to the model file
        custom_objects: Custom objects for model loading
        
    Returns:
        tf.keras.Model: Loaded model or None if failed
    """
    try:
        import tensorflow as tf
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Try loading with custom objects
        if custom_objects:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(model_path)
            
        logger.info(f"Successfully loaded model from {model_path}")
        return model
        
    except ImportError:
        logger.error("TensorFlow not available")
        return None
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def compile_model_with_gpu_optimization(model, optimizer='adam', loss='mse', 
                                      metrics=None, mixed_precision=False):
    """
    Compile model with GPU optimizations
    
    Args:
        model: TensorFlow model to compile
        optimizer: Optimizer to use
        loss: Loss function
        metrics: List of metrics
        mixed_precision: Enable mixed precision training
        
    Returns:
        tf.keras.Model: Compiled model
    """
    try:
        import tensorflow as tf
        
        # Enable mixed precision if requested
        if mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Enabled mixed precision training")
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or ['mae']
        )
        
        logger.info("Model compiled with GPU optimizations")
        return model
        
    except ImportError:
        logger.error("TensorFlow not available")
        return model
    except Exception as e:
        logger.error(f"Failed to compile model with GPU optimizations: {e}")
        return model

def clear_gpu_memory():
    """
    Clear GPU memory and garbage collect
    """
    try:
        import tensorflow as tf
        
        # Clear TensorFlow memory
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("GPU memory cleared")
        
    except ImportError:
        logger.warning("TensorFlow not available for memory clearing")
    except Exception as e:
        logger.error(f"Failed to clear GPU memory: {e}")

# Initialize GPU configuration on module import
if __name__ == "__main__":
    # Test GPU configuration
    gpu_available = configure_gpu()
    print(f"GPU available: {gpu_available}")
    
    if gpu_available:
        strategy = set_gpu_strategy()
        print(f"GPU strategy: {type(strategy).__name__}")
        
        memory_info = monitor_gpu_memory()
        print(f"Memory info: {memory_info}") 