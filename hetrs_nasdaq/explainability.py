from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.repro import set_global_seed


@dataclass(frozen=True)
class ExplainConfig:
    seed: int = 42
    outdir: str = "results/hetrs_nasdaq/explain"
    max_points: int = 250  # cap for plotting


def _require_pf():
    try:
        from pytorch_forecasting import TemporalFusionTransformer  # type: ignore

        return TemporalFusionTransformer
    except Exception as e:  # pragma: no cover
        raise ImportError("Explainability requires pytorch-forecasting installed.") from e


def extract_tft_interpretation(
    df_features: pd.DataFrame,
    checkpoint_path: str,
    seed: int = 42,
) -> dict:
    """
    Best-effort TFT interpretability extraction.

    PyTorch Forecasting provides interpretability helpers for TFT via:
      - model.predict(..., mode="raw", return_x=True)
      - model.interpret_output(raw_predictions, reduction="sum")

    Output:
      - variable_importance (if available)
      - attention (if available)
    """
    from hetrs_nasdaq.tft_model import prepare_tft_dataframe, build_datasets

    TemporalFusionTransformer = _require_pf()
    set_global_seed(seed)

    df_tft = prepare_tft_dataframe(df_features, horizon_days=5)
    training, _, _, _, val_cutoff = build_datasets(df_tft, train_ratio=0.7, val_ratio=0.1)

    predict_frame = df_tft[df_tft.time_idx > val_cutoff]
    # Use dataset from training for consistent encoders/scalers
    from pytorch_forecasting import TimeSeriesDataSet  # type: ignore

    predict_ds = TimeSeriesDataSet.from_dataset(training, predict_frame, predict=False, stop_randomization=True)
    loader = predict_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, weights_only=False)

    pred = model.predict(
        loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )

    raw = pred.output  # typically dict-like for mode="raw"
    x = getattr(pred, "x", None)

    result: dict = {"has_raw": raw is not None, "has_x": x is not None}

    # Variable importance / attention
    try:
        interpretation = model.interpret_output(raw, reduction="sum")  # type: ignore[arg-type]
        result["interpretation_keys"] = list(interpretation.keys())
        # Common keys: 'static_variables', 'encoder_variables', 'decoder_variables', 'attention'
        for k, v in interpretation.items():
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            if isinstance(v, (np.ndarray, list)):
                result[k] = v
    except Exception as e:
        result["interpretation_error"] = str(e)

    return result


def plot_model_interpretation(
    interpretation: dict,
    outdir: str,
    prefix: str = "tft",
) -> list[str]:
    """
    Save heatmaps for variable importance / attention if present.
    """
    from matplotlib import pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Encoder variable importance heatmap (if present)
    if "encoder_variables" in interpretation:
        arr = np.asarray(interpretation["encoder_variables"])
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(arr.reshape(1, -1), aspect="auto")
        ax.set_title("TFT Encoder Variable Importance (summed)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = str(Path(outdir) / f"{prefix}_encoder_var_importance.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    if "attention" in interpretation:
        att = np.asarray(interpretation["attention"])
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(att, aspect="auto")
        ax.set_title("TFT Attention Weights (summed)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = str(Path(outdir) / f"{prefix}_attention.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # Persist raw interpretation dump (small keys only)
    meta_path = Path(outdir) / f"{prefix}_interpretation_meta.json"
    safe = {k: v for k, v in interpretation.items() if isinstance(v, (str, int, float, bool, list, dict))}
    meta_path.write_text(json.dumps(safe, ensure_ascii=False, indent=2))
    saved.append(str(meta_path))

    return saved


