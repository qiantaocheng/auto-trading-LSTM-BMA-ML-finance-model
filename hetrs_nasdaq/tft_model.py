from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.repro import set_global_seed


def _require_tft():
    try:
        # pytorch-forecasting >= 1.5 uses the new Lightning package namespace
        import lightning.pytorch as pl  # type: ignore
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # type: ignore
        from pytorch_forecasting.data import GroupNormalizer  # type: ignore
        from pytorch_forecasting.metrics import QuantileLoss  # type: ignore

        return pl, TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer, QuantileLoss
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Module 3 requires pytorch-forecasting + pytorch-lightning + torch.\n"
            "Install: pip install -r requirements-hetrs_nasdaq.txt"
        ) from e


@dataclass(frozen=True)
class TFTPaths:
    outdir: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.outdir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.outdir / "artifacts"


def prepare_tft_dataframe(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """
    Prepare single-series dataframe for PyTorch Forecasting.
    - Adds time_idx (0..n-1)
    - Adds group_id constant
    - Adds day_of_week
    - Creates target = future horizon return (sum of log returns over horizon, approx)
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex (date).")

    out = out.sort_index()
    out["time_idx"] = np.arange(len(out), dtype=int)
    out["group_id"] = "QQQ"
    out["day_of_week"] = out.index.dayofweek.astype(int)

    # Target: forward 5-day return (simple return compounded)
    close = out["Close"].astype(float)
    out["target_return_5d"] = close.pct_change(horizon_days).shift(-horizon_days)

    out = out.dropna(axis=0, how="any")
    return out


def build_datasets(
    df_tft: pd.DataFrame,
    min_encoder_length: int = 60,
    max_prediction_length: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    pl, _, TimeSeriesDataSet, GroupNormalizer, _ = _require_tft()

    max_time = int(df_tft["time_idx"].max())
    train_cutoff = int(np.floor(max_time * train_ratio))
    val_cutoff = int(np.floor(max_time * (train_ratio + val_ratio)))

    training = TimeSeriesDataSet(
        df_tft[df_tft.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="target_return_5d",
        group_ids=["group_id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=min_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "day_of_week"],
        time_varying_unknown_reals=[
            "close_ffd",
            "vol_gk",
            "regime_prob_0",
            "regime_prob_1",
            "regime_prob_2",
            "vix",
            "us10y",
        ],
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_tft[(df_tft.time_idx > train_cutoff) & (df_tft.time_idx <= val_cutoff)],
        predict=True,
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        df_tft[df_tft.time_idx > val_cutoff],
        predict=True,
        stop_randomization=True,
    )

    return training, validation, test, train_cutoff, val_cutoff


def train_tft(
    df_features: pd.DataFrame,
    outdir: str,
    max_epochs: int = 30,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict:
    pl, TemporalFusionTransformer, TimeSeriesDataSet, _, QuantileLoss = _require_tft()
    set_global_seed(seed)
    try:
        pl.seed_everything(seed, workers=True)
    except Exception:
        pass

    paths = TFTPaths(outdir=Path(outdir))
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_tft = prepare_tft_dataframe(df_features, horizon_days=5)
    training, validation, _, train_cutoff, val_cutoff = build_datasets(
        df_tft, train_ratio=train_ratio, val_ratio=val_ratio
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=6, verbose=True, mode="min"
    )
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=str(paths.checkpoint_dir),
        filename="tft-{epoch:02d}-{val_loss:.5f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        callbacks=[early_stop, ckpt],
        gradient_clip_val=0.1,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
        logger=False,
        enable_progress_bar=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=80,
        output_size=3,  # quantiles
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=3,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = ckpt.best_model_path
    meta = {
        "best_checkpoint": best_path,
        "train_cutoff_time_idx": int(train_cutoff),
        "val_cutoff_time_idx": int(val_cutoff),
        "rows_after_dropna": int(len(df_tft)),
        "min_encoder_length": 60,
        "max_prediction_length": 5,
        "quantiles": [0.1, 0.5, 0.9],
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
    }

    (paths.artifacts_dir / "tft_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def predict_tft_quantiles(
    features_df: pd.DataFrame,
    checkpoint_path: str,
    out_path: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> pd.DataFrame:
    """
    Generate real TFT quantile predictions:
    - Produces per-date predictions for 5-day forward return (target) aligned to each encoder end date.
    - Output columns: tft_p10, tft_p50, tft_p90
    """
    pl, TemporalFusionTransformer, TimeSeriesDataSet, _, _ = _require_tft()
    set_global_seed(seed)
    try:
        pl.seed_everything(seed, workers=True)
    except Exception:
        pass

    df_tft = prepare_tft_dataframe(features_df, horizon_days=5)
    training, _, _, train_cutoff, val_cutoff = build_datasets(df_tft, train_ratio=train_ratio, val_ratio=val_ratio)

    # Build a rolling prediction dataset on the (strict) holdout region.
    # IMPORTANT: predict=True would only predict the final point of the series; we want predictions for all possible
    # windows in the holdout region, so we use predict=False.
    predict_frame = df_tft[df_tft.time_idx > val_cutoff]
    predict_ds = TimeSeriesDataSet.from_dataset(
        training,
        predict_frame,
        predict=False,
        stop_randomization=True,
    )

    # torch>=2.6 defaults to "weights_only=True" for torch.load(), which can block
    # loading checkpoints containing pickled objects (e.g., normalizers). This checkpoint
    # is produced locally by this project, so we explicitly opt into full load.
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, weights_only=False)
    loader = predict_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    # pytorch-forecasting returns a Prediction object
    pred = model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        trainer_kwargs={
            "logger": False,  # avoid tensorboard initialization issues
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )
    q = pred.output  # type: ignore[attr-defined]
    x = getattr(pred, "x", None)
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    q = np.asarray(q)

    # Use the FIRST decoder step (t+1) for each sample as the actionable next-period forecast.
    p10 = q[:, 0, 0]
    p50 = q[:, 0, 1]
    p90 = q[:, 0, 2]

    # Align to timestamps using decoder_time_idx from x (preferred)
    if x is not None and isinstance(x, dict) and "decoder_time_idx" in x:
        # decoder_time_idx shape: (n_samples, prediction_length)
        decoder_t = np.asarray(x["decoder_time_idx"])[:, 0].astype(int)
        # map time_idx -> timestamp
        tmp = df_tft.reset_index()
        date_col = tmp.columns[0]  # first column is the index after reset_index()
        map_idx = tmp[["time_idx", date_col]].set_index("time_idx")[date_col]
        ts = map_idx.reindex(decoder_t).values
        out = pd.DataFrame({"tft_p10": p10, "tft_p50": p50, "tft_p90": p90}, index=pd.to_datetime(ts))
        out = out[~out.index.isna()].sort_index()
    else:
        # Conservative fallback: align to the end of df_tft (test window only)
        out = pd.DataFrame({"tft_p10": p10, "tft_p50": p50, "tft_p90": p90}, index=predict_frame.index[: len(p10)])

    if out_path is not None:
        from pathlib import Path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path)
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Temporal Fusion Transformer (HETRS-NASDAQ).")
    p.add_argument("--in", dest="inp", required=True, help="Input features parquet")
    p.add_argument("--outdir", required=True, help="Output directory for checkpoints/artifacts")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--predict-out",
        default=None,
        help="Optional parquet path to save quantile predictions after training (tft_p10/p50/p90).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    df = pd.read_parquet(args.inp)
    meta = train_tft(
        df_features=df,
        outdir=args.outdir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"[tft_model] done. best_checkpoint={meta['best_checkpoint']}")

    if args.predict_out:
        pred = predict_tft_quantiles(
            features_df=df,
            checkpoint_path=meta["best_checkpoint"],
            out_path=args.predict_out,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        print(f"[tft_model] saved predictions: {args.predict_out} rows={len(pred)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


