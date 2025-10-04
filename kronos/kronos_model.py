import os
import sys
import subprocess
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass
class KronosConfig:
    model_size: str = "base"  # Default to base for better compatibility on CPU
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.95
    max_context_length: int = 2048  # Large model context length
    pred_len: int = 30
    num_paths: int = 1
    use_cache: bool = True
    # Strict mode controls
    allow_fallback: bool = False
    allow_model_downgrade: bool = False
    attempt_dependency_install_on_failure: bool = False

class KronosModelWrapper:
    def __init__(self, config: Optional[KronosConfig] = None):
        self.config = config or KronosConfig()
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self.is_loaded = False
        self._deps_installed = False

    def _locate_repo_root_with_model(self, root_path: str) -> Optional[str]:
        """Locate the repository root that contains a usable 'model' module.

        Accepts either:
          - a file 'model.py' at repo root, or
          - a package directory 'model' with an __init__.py at repo root.
        Returns the repo root path (parent directory to import from), or None.
        """
        try:
            # Direct root checks
            if os.path.isfile(os.path.join(root_path, 'model.py')):
                return root_path
            if os.path.isdir(os.path.join(root_path, 'model')) and \
               os.path.isfile(os.path.join(root_path, 'model', '__init__.py')):
                return root_path

            # Recursive search
            for dirpath, dirnames, filenames in os.walk(root_path):
                # case: .../something/model.py at dirpath → repo root is dirpath
                if 'model.py' in filenames:
                    return dirpath
                # case: .../something/model/__init__.py → repo root is dirpath
                if 'model' in dirnames:
                    candidate = os.path.join(dirpath, 'model')
                    if os.path.isfile(os.path.join(candidate, '__init__.py')):
                        return dirpath
        except Exception:
            return None
        return None

    def _ensure_kronos_repo(self) -> Optional[str]:
        """Ensure the original Kronos repo code is available locally; clone or download if missing.

        Returns the directory path that contains model.py, or None on failure.
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))
        candidates = [
            os.path.join(base_dir, 'kronos_original_repo'),
            os.path.join(base_dir, 'kronos_original')
        ]

        # Use existing path if available
        for cand in candidates:
            if os.path.isdir(cand):
                found_root = self._locate_repo_root_with_model(cand)
                if found_root:
                    return found_root

        # Attempt to clone or download the repository
        repo_url = os.environ.get('KRONOS_REPO_URL', 'https://github.com/shiyu-coder/Kronos')
        target_root = candidates[0]
        try:
            os.makedirs(target_root, exist_ok=True)
        except Exception:
            pass

        # Try git clone first
        try:
            if not os.listdir(target_root):
                subprocess.run(['git', 'clone', '--depth', '1', repo_url, target_root], check=True)
        except Exception:
            # Fallback to zip download
            try:
                import requests, zipfile, io
                zip_url = repo_url.rstrip('/') + '/archive/refs/heads/main.zip'
                resp = requests.get(zip_url, timeout=60)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(target_root)
            except Exception as _:
                return None

        # Locate repo root with model module
        found_root = self._locate_repo_root_with_model(target_root)
        return found_root

    def load_model(self) -> bool:
        try:
            # Ensure original Kronos repository code is present and importable
            kronos_repo_root = self._ensure_kronos_repo()
            if not kronos_repo_root:
                raise RuntimeError("Original Kronos repo code not available (model.py not found)")
            if kronos_repo_root not in sys.path:
                sys.path.insert(0, kronos_repo_root)

            logger.info(f"Loading Kronos model: {self.config.model_size}")

            # Map to available models
            model_mapping = {
                "base": "NeoQuasar/Kronos-base",
                "large": "NeoQuasar/Kronos-large"
            }

            tokenizer_name = "NeoQuasar/Kronos-Tokenizer-base"
            model_name = model_mapping.get(self.config.model_size, "NeoQuasar/Kronos-base")

            # Use original Kronos classes (if available in local kronos repository)
            Kronos = None
            KronosTokenizer = None
            KronosPredictor = None
            try:
                # First try regular import
                from model import Kronos as _Kronos, KronosTokenizer as _KronosTokenizer, KronosPredictor as _KronosPredictor  # type: ignore
                Kronos, KronosTokenizer, KronosPredictor = _Kronos, _KronosTokenizer, _KronosPredictor
            except Exception as imp_err:
                # Fallback: direct import via file path
                try:
                    import importlib.util
                    model_file = os.path.join(kronos_repo_root, 'model.py')
                    if not os.path.isfile(model_file):
                        # Try package directory
                        model_pkg_init = os.path.join(kronos_repo_root, 'model', '__init__.py')
                        if not os.path.isfile(model_pkg_init):
                            raise FileNotFoundError(model_file)
                        model_file = model_pkg_init
                    spec = importlib.util.spec_from_file_location('kronos_model_impl', model_file)
                    if spec is None or spec.loader is None:
                        raise ImportError('Cannot create import spec for model.py')
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    Kronos = getattr(mod, 'Kronos', None)
                    KronosTokenizer = getattr(mod, 'KronosTokenizer', None)
                    KronosPredictor = getattr(mod, 'KronosPredictor', None)
                    if Kronos is None or KronosTokenizer is None or KronosPredictor is None:
                        raise ImportError('model.py does not expose required classes')
                except Exception as imp2_err:
                    raise RuntimeError(f"Unable to import Kronos classes: {imp_err} | {imp2_err}")

            logger.info(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)

            logger.info(f"Loading model: {model_name}")
            self.model = Kronos.from_pretrained(model_name)

            self.model.eval()

            # Use original KronosPredictor
            self.predictor = KronosPredictor(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.config.device
            )

            self.is_loaded = True
            logger.info("Original Kronos model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Kronos model: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Optional model downgrade (disabled by default)
            if self.config.allow_model_downgrade and self.config.model_size != "base":
                logger.info("Current model failed, trying base model (allowed by config)...")
                self.config.model_size = "base"
                return self.load_model()

            # Optional dependency installation (disabled by default)
            if self.config.attempt_dependency_install_on_failure and not self._deps_installed:
                try:
                    logger.info("Attempting to install missing dependencies (allowed by config)...")
                    self._install_dependencies()
                    self._deps_installed = True
                    logger.info("Dependencies installed, retrying model load...")
                    return self.load_model()
                except Exception as install_error:
                    logger.error(f"Dependency installation failed: {install_error}")
                    self._deps_installed = True  # Mark as attempted to avoid retries
                    return False

            logger.error("Strict mode: model load failed; no fallback, downgrade, or auto-install will be attempted")
            return False


    def predict(self,
                data: np.ndarray,
                timestamps: Optional[List] = None,
                pred_len: Optional[int] = None) -> Dict[str, Any]:

        if not self.is_loaded:
            if not self.load_model():
                return {
                    "status": "error",
                    "error": "Kronos model not loaded; strict mode prohibits fallback",
                    "predictions": None
                }

        try:
            pred_len = pred_len or self.config.pred_len

            # Convert numpy array to DataFrame for original Kronos
            import pandas as pd
            from datetime import datetime, timedelta

            # Create column names - original Kronos expects 'amount' column too
            cols = ['open', 'high', 'low', 'close', 'volume']
            expected_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']

            if data.shape[1] < len(cols):
                # Add volume column if missing
                volume_col = np.ones((data.shape[0], 1)) * 100000  # Default volume
                data = np.concatenate([data, volume_col], axis=1)

            if data.shape[1] < len(expected_cols):
                # Add amount column (volume * average price)
                avg_price = np.mean(data[:, :4], axis=1, keepdims=True)  # Average of OHLC
                amount_col = data[:, 4:5] * avg_price  # volume * avg_price
                data = np.concatenate([data, amount_col], axis=1)

            df = pd.DataFrame(data, columns=expected_cols)

            # Generate timestamps if not provided or derive from provided ones
            if timestamps is None:
                # Create dummy timestamps with daily frequency (legacy fallback)
                base_time = datetime.now()
                x_timestamp_index = pd.date_range(
                    start=base_time - timedelta(days=len(df)),
                    periods=len(df),
                    freq='D'
                )
                y_timestamp_index = pd.date_range(
                    start=x_timestamp_index[-1] + timedelta(days=1),
                    periods=pred_len,
                    freq='D'
                )
                x_timestamps = pd.Series(x_timestamp_index)
                y_timestamps = pd.Series(y_timestamp_index)
            else:
                # Use provided historical timestamps, infer frequency for future timestamps
                x_index = pd.to_datetime(pd.Index(timestamps))
                if len(x_index) != len(df):
                    # Align: take the most recent len(df) timestamps
                    x_index = x_index[-len(df):]

                # Try to infer a frequency; if None, compute median delta
                inferred = pd.infer_freq(x_index)
                if inferred is not None:
                    y_index = pd.date_range(start=x_index[-1] + pd.tseries.frequencies.to_offset(inferred),
                                             periods=pred_len,
                                             freq=inferred)
                else:
                    # Compute typical delta
                    deltas = (x_index[1:] - x_index[:-1]).to_series().dt.total_seconds()
                    if len(deltas) == 0:
                        delta = 24 * 3600  # default 1 day in seconds
                    else:
                        delta = float(deltas.median())
                    delta_td = pd.to_timedelta(delta, unit='s')
                    y_index = pd.DatetimeIndex([x_index[-1] + delta_td * (i + 1) for i in range(pred_len)])

                x_timestamps = pd.Series(x_index)
                y_timestamps = pd.Series(y_index)

            # Use original KronosPredictor interface when available; otherwise fallback
            if self.predictor is not None:
                predictions = self.predictor.predict(
                    df=df,
                    x_timestamp=x_timestamps,
                    y_timestamp=y_timestamps,
                    pred_len=pred_len,
                    T=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    sample_count=self.config.num_paths
                )
                # Convert back to numpy array - keep only original columns
                pred_array = predictions[cols].values
            else:
                return {
                    "status": "error",
                    "error": "Kronos predictor unavailable after load; strict mode prohibits fallback",
                    "predictions": None
                }

            return {
                "status": "success",
                "predictions": pred_array,
                "config": {
                    "model_size": self.config.model_size,
                    "pred_len": pred_len,
                    "temperature": self.config.temperature
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "predictions": None
            }

    def batch_predict(self, data_list: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for data in data_list:
            result = self.predict(data, **kwargs)
            results.append(result)
        return results

    def _install_dependencies(self) -> None:
        try:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch", "accelerate", "einops"], check=True)
            logger.info("Dependencies installed successfully")
        except Exception as e:
            logger.error(f"Failed to install dependencies: {str(e)}")

    def _predict_fallback(self, data: np.ndarray, pred_len: int) -> np.ndarray:
        """Deterministic statistical fallback: project last returns and typical range.

        Returns array of shape (pred_len, 5): [open, high, low, close, volume]
        """
        try:
            # Ensure we have at least OHLCV columns
            arr = data
            num_cols = arr.shape[1]
            # If fewer than 5 columns, pad volume with mean 1e6
            if num_cols < 5:
                vol_col = np.full((arr.shape[0], 1), 1_000_000.0)
                arr = np.concatenate([arr, vol_col], axis=1)
            # Columns
            OPEN, HIGH, LOW, CLOSE, VOL = 0, 1, 2, 3, 4
            close = arr[:, CLOSE]
            volume = arr[:, VOL]
            # Compute simple mean return (avoid division by zero)
            close_safe = np.clip(close, 1e-6, None)
            rets = np.diff(close_safe) / close_safe[:-1]
            mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
            # Typical intrabar range as fraction of close
            highs = arr[:, HIGH]
            lows = arr[:, LOW]
            range_frac = np.mean((highs - lows) / np.clip(close_safe, 1e-6, None)) if arr.shape[0] > 0 else 0.02
            range_frac = float(np.clip(range_frac, 0.005, 0.08))
            # Volume projection: moving average
            vol_ma = float(np.mean(volume[-min(20, len(volume)) :])) if len(volume) > 0 else 1_000_000.0
            # Start from last close
            last_close = float(close[-1]) if len(close) else 100.0
            preds = []
            current_close = last_close
            for _ in range(pred_len):
                # Project next close deterministically
                current_close = current_close * (1.0 + mean_ret)
                # Open near previous close
                open_p = current_close * (1.0 - 0.1 * range_frac)
                high_p = current_close * (1.0 + 0.5 * range_frac)
                low_p = current_close * (1.0 - 0.5 * range_frac)
                close_p = current_close
                vol_p = vol_ma
                preds.append([open_p, high_p, low_p, close_p, vol_p])
            return np.array(preds, dtype=float)
        except Exception:
            # Absolute fallback: flat projection
            last_close = float(data[-1, 3]) if data.size and data.shape[1] >= 4 else 100.0
            vol = float(data[-1, 4]) if data.size and data.shape[1] >= 5 else 1_000_000.0
            return np.array([[last_close, last_close, last_close, last_close, vol] for _ in range(pred_len)], dtype=float)