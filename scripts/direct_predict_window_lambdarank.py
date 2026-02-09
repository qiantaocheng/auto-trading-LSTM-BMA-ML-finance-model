#!/usr/bin/env python3
"""Download 11-factor window and run LambdaRank predictions with EMA variants."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bma_models.simple_25_factor_engine import (
    Simple17FactorEngine,
    TOP_FEATURE_SET,
    MAX_CLOSE_THRESHOLD,
)
from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

DEFAULT_TICKER_FILE = Path(r"D:/trade/data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet")
SNAPSHOT_ID = "e03c3d05-f9a3-419e-86fa-fd79ae4ecb7b"
SNAPSHOT_ROOT = Path("D:/trade/cache/model_snapshots")
DEFAULT_OUTPUT_DATA = Path("results/direct_predict_window_features.parquet")
DEFAULT_PREDICTION_OUTPUT = Path("results/direct_predict_window_predictions.parquet")
DEFAULT_OVERLAP = 5
DEFAULT_UNIQUE = 15
EMA_WEIGHTS = (0.588, 0.412)


def _load_snapshot_features(snapshot_id: str) -> List[str]:
    manifest_paths = list(SNAPSHOT_ROOT.glob(f"**/{snapshot_id}/manifest.json"))
    if manifest_paths:
        manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        features = manifest.get("feature_names")
        if features:
            return list(features)
    return list(TOP_FEATURE_SET)


def _load_universe(ticker_file: Path) -> List[str]:
    if not ticker_file.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")
    df = pd.read_parquet(ticker_file)
    if isinstance(df.index, pd.MultiIndex):
        tickers = df.index.get_level_values("ticker").unique().tolist()
    elif "ticker" in df.columns:
        tickers = df["ticker"].unique().tolist()
    else:
        raise ValueError("Ticker file must have ticker column or MultiIndex")
    tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
    if not tickers:
        raise ValueError("No tickers discovered in ticker universe file")
    return tickers


def _parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _derive_fetch_window(start_date: pd.Timestamp, end_date: pd.Timestamp, lookback: int, horizon: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_fetch = start_date - pd.Timedelta(days=lookback)
    end_fetch = end_date + pd.Timedelta(days=horizon + 5)
    return start_fetch, end_fetch


def _ensure_feature_columns(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    for col in features:
        if col not in frame.columns:
            frame[col] = 0.0
    ordered = list(features) + [c for c in frame.columns if c not in features]
    return frame[ordered]


def fetch_feature_window(
    tickers: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_days: int,
    horizon: int,
    features: Sequence[str],
) -> pd.DataFrame:
    start_fetch, end_fetch = _derive_fetch_window(start_date, end_date, lookback_days, horizon)
    engine = Simple17FactorEngine(
        lookback_days=lookback_days + horizon + 30,
        mode="predict",
        horizon=horizon,
    )
    market_data = engine.fetch_market_data(
        symbols=list(tickers),
        use_optimized_downloader=True,
        start_date=start_fetch.strftime("%Y-%m-%d"),
        end_date=end_fetch.strftime("%Y-%m-%d"),
    )
    if market_data.empty:
        raise RuntimeError("No market data retrieved for requested window")
    close_col = None
    for candidate in ("Close", "close", "adj_close"):
        if candidate in market_data.columns:
            close_col = candidate
            break
    if close_col:
        valid_mask = (
            pd.to_numeric(market_data[close_col], errors="coerce").notna()
            & (market_data[close_col] > 0)
            & (market_data[close_col] <= MAX_CLOSE_THRESHOLD)
        )
        removed = (~valid_mask).sum()
        if removed:
            market_data = market_data[valid_mask]
            print(f"Filtered {removed} rows with invalid {close_col} values before factor computation")
        if market_data.empty:
            raise RuntimeError("No market data left after removing invalid close prices")
    factors_df = engine.compute_all_17_factors(market_data, mode="predict")
    factors_df = _ensure_feature_columns(factors_df, features)
    factors_df = factors_df.sort_index()
    mask = (factors_df.index.get_level_values("date") >= start_date) & (
        factors_df.index.get_level_values("date") <= end_date
    )
    window_df = factors_df[mask].copy()
    if window_df.empty:
        raise RuntimeError("No factor rows remain inside requested date window")
    return window_df


def _apply_feature_overrides(feature_list: Sequence[str]) -> None:
    override = json.dumps({
        "elastic_net": feature_list,
        "xgboost": feature_list,
        "catboost": feature_list,
        "lightgbm_ranker": feature_list,
        "lambdarank": feature_list,
    })
    os.environ["BMA_FEATURE_OVERRIDES"] = override


def _apply_ema(predictions_df: pd.DataFrame, columns: Sequence[str], weights: Sequence[float]) -> pd.DataFrame:
    if predictions_df.empty or not isinstance(predictions_df.index, pd.MultiIndex):
        return predictions_df
    weights = tuple(w for w in weights if w > 0)
    if not weights:
        return predictions_df
    norm = sum(weights)
    weights = tuple(w / norm for w in weights)
    dates = sorted(predictions_df.index.get_level_values("date").unique())
    if len(dates) < 2:
        return predictions_df
    df = predictions_df.copy()
    tickers = df.index.get_level_values("ticker").unique()
    for col in columns:
        if col not in df.columns:
            continue
        raw_col = f"{col}_raw"
        ema_col = f"{col}_ema"
        if raw_col not in df.columns:
            df[raw_col] = df[col]
        df[ema_col] = np.nan
        for ticker in tickers:
            try:
                series = df.xs(ticker, level="ticker")[raw_col].sort_index()
            except KeyError:
                continue
            if series.dropna().empty:
                continue
            values = series.values
            idx_dates = list(series.index)
            for idx_pos, current_date in enumerate(idx_dates):
                weighted_sum = 0.0
                weight_total = 0.0
                for offset, weight in enumerate(weights):
                    prev_pos = idx_pos - offset
                    if prev_pos < 0:
                        break
                    prev_value = values[prev_pos]
                    if pd.isna(prev_value):
                        continue
                    weighted_sum += weight * float(prev_value)
                    weight_total += weight
                if weight_total:
                    df.loc[(current_date, ticker), ema_col] = weighted_sum / weight_total
        df[col] = df[ema_col].fillna(df[col])
    return df


def run_predictions(
    factors_df: pd.DataFrame,
    tickers: Sequence[str],
    dates: Iterable[pd.Timestamp],
    snapshot_id: str,
    horizon: int,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    model = UltraEnhancedQuantitativeModel()
    frames = []
    for current_date in dates:
        mask = factors_df.index.get_level_values("date") <= current_date
        feature_slice = factors_df.loc[mask, feature_cols].copy()
        if feature_slice.empty:
            continue
        results = model.predict_with_snapshot(
            feature_data=feature_slice,
            snapshot_id=snapshot_id,
            universe_tickers=list(tickers),
            as_of_date=current_date,
            prediction_days=horizon,
        )
        predictions_raw = results.get("predictions_raw")
        if predictions_raw is None:
            predictions_raw = results.get("predictions")
        if predictions_raw is None:
            continue
        if isinstance(predictions_raw, pd.Series):
            pred_df = predictions_raw.to_frame("score").copy()
        else:
            pred_df = predictions_raw.copy()
            if "score" not in pred_df.columns and pred_df.shape[1] == 1:
                pred_df.columns = ["score"]
        if not isinstance(pred_df.index, pd.MultiIndex):
            pred_df["date"] = current_date
            idx_name = pred_df.index.name or "ticker"
            pred_df = pred_df.reset_index().set_index(["date", idx_name])
        base_predictions = results.get("base_predictions")
        if isinstance(base_predictions, pd.DataFrame):
            aligned = base_predictions.reindex(pred_df.index)
            if "pred_lambdarank" in aligned.columns:
                pred_df["score_lambdarank"] = aligned["pred_lambdarank"]
            if "pred_catboost" in aligned.columns:
                pred_df["score_catboost"] = aligned["pred_catboost"]
        pred_df["as_of_date"] = current_date
        frames.append(pred_df)
    if not frames:
        raise RuntimeError("LambdaRank prediction loop returned no data")
    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def _format_top(df: pd.DataFrame, date: pd.Timestamp, score_col: str, top_n: int, unique: bool) -> pd.DataFrame:
    try:
        day_slice = df.xs(date, level="date", drop_level=False)
    except KeyError:
        return pd.DataFrame()
    work = day_slice.reset_index(level="ticker")
    work = work.rename(columns={"ticker": "ticker"})
    work = work.dropna(subset=[score_col])
    work = work.sort_values(score_col, ascending=False)
    if unique:
        work = work.drop_duplicates(subset=["ticker"], keep="first")
    return work.head(top_n)[["ticker", score_col]]


def _display_day_results(
    date: pd.Timestamp,
    df_raw: pd.DataFrame,
    df_ema: pd.DataFrame,
    score_col: str,
    top_overlap: int,
    top_unique: int,
) -> None:
    stamp = date.strftime("%Y-%m-%d")
    print(f"\n===== {stamp} =====")
    raw_overlap = _format_top(df_raw, date, score_col, top_overlap, unique=False)
    raw_unique = _format_top(df_raw, date, score_col, top_unique, unique=True)
    ema_overlap = _format_top(df_ema, date, score_col, top_overlap, unique=False)
    ema_unique = _format_top(df_ema, date, score_col, top_unique, unique=True)
    print(f"Raw LambdaRank top {top_overlap} (overlapping):")
    print(raw_overlap.to_string(index=False) if not raw_overlap.empty else "  <no data>")
    print(f"Raw LambdaRank top {top_unique} (unique tickers):")
    print(raw_unique.to_string(index=False) if not raw_unique.empty else "  <no data>")
    print(f"EMA LambdaRank top {top_overlap} (overlapping):")
    print(ema_overlap.to_string(index=False) if not ema_overlap.empty else "  <no data>")
    print(f"EMA LambdaRank top {top_unique} (unique tickers):")
    print(ema_unique.to_string(index=False) if not ema_unique.empty else "  <no data>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 11-factor window and run LambdaRank predictions")
    parser.add_argument("--start-date", default="2026-01-09", help="Start date (inclusive)")
    parser.add_argument("--end-date", default="2026-01-24", help="End date (inclusive)")
    parser.add_argument("--lookback-days", type=int, default=310, help="History window before start-date")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon")
    parser.add_argument("--tickers-file", type=str, default=str(DEFAULT_TICKER_FILE), help="Ticker universe parquet")
    parser.add_argument("--snapshot-id", type=str, default=SNAPSHOT_ID, help="Snapshot ID to load")
    parser.add_argument("--save-data", type=str, default=str(DEFAULT_OUTPUT_DATA), help="Where to save factor window parquet")
    parser.add_argument("--save-preds", type=str, default=str(DEFAULT_PREDICTION_OUTPUT), help="Where to save raw predictions parquet")
    parser.add_argument("--top-overlap", type=int, default=DEFAULT_OVERLAP, help="Top-N with overlaps")
    parser.add_argument("--top-unique", type=int, default=DEFAULT_UNIQUE, help="Top-N unique tickers")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be >= start-date")

    features = _load_snapshot_features(args.snapshot_id)
    _apply_feature_overrides(features)

    tickers = _load_universe(Path(args.tickers_file))

    window = fetch_feature_window(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback_days=args.lookback_days,
        horizon=args.horizon,
        features=features,
    )

    save_data_path = Path(args.save_data)
    save_data_path.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(save_data_path)
    print(f"Saved feature window to {save_data_path} (rows={len(window)})")

    unique_dates = sorted(window.index.get_level_values("date").unique())
    predictions = run_predictions(
        factors_df=window,
        tickers=tickers,
        dates=unique_dates,
        snapshot_id=args.snapshot_id,
        horizon=args.horizon,
        feature_cols=features,
    )

    save_preds_path = Path(args.save_preds)
    save_preds_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(save_preds_path)
    print(f"Saved raw predictions to {save_preds_path} (rows={len(predictions)})")

    score_col = "score_lambdarank" if "score_lambdarank" in predictions.columns else "score"
    predictions_ema = _apply_ema(predictions, [score_col], EMA_WEIGHTS)

    for date in unique_dates:
        _display_day_results(
            date=date,
            df_raw=predictions,
            df_ema=predictions_ema,
            score_col=score_col,
            top_overlap=args.top_overlap,
            top_unique=args.top_unique,
        )


if __name__ == "__main__":
    main()
