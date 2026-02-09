#!/usr/bin/env python
"""Full Direct Prediction bridge for TraderApp.

Replicates the autotrader/app.py _direct_predict_snapshot() flow:
1. Load tickers from polygon_full_features_T5.parquet
2. Load feature list from snapshot manifest
3. Fetch market data from Polygon API
4. Compute 15 alpha factors
5. Run predict_with_snapshot (ElasticNet, XGBoost, CatBoost, LambdaRank + MetaRankerStacker)
6. Apply EMA smoothing
7. Generate Excel report
8. Store predictions to monitoring.db
9. Output JSON result to stdout

Progress updates are emitted on stderr as JSON lines.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ensure D:\trade is on sys.path so bma_models / scripts are importable
ROOT = Path(__file__).resolve().parents[2]  # D:\trade
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also ensure scripts directory is importable
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Constants (replicated from autotrader/app.py lines 55-70)
# ---------------------------------------------------------------------------
TICKER_DATA_PATH = ROOT / "data" / "factor_exports" / "polygon_full_features_T5.parquet"
DEFAULT_SNAPSHOT_ID = "b35a35db-352b-43d8-ace8-4a54674c1da5"
EMA_WEIGHTS = (0.41, 0.28, 0.19, 0.12)
MAX_CLOSE = 10000.0
MIN_LOOKBACK = 280
HORIZON = 10


def emit_progress(step: str, progress: int, detail: str = "") -> None:
    """Emit a JSON progress line on stderr for C# to read."""
    msg = json.dumps({"step": step, "progress": progress, "detail": detail})
    print(msg, file=sys.stderr, flush=True)


def load_snapshot_features(snapshot_id: str) -> list[str]:
    """Load the feature list from the snapshot manifest.

    Replicates app.py _get_direct_predict_features() (lines 6248-6277).
    """
    features = None
    try:
        snapshots_root = ROOT / "cache" / "model_snapshots"
        manifest_paths = list(snapshots_root.glob(f"**/{snapshot_id}/manifest.json"))
        if manifest_paths:
            manifest_data = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            manifest_features = manifest_data.get("feature_names")
            if manifest_features:
                features = manifest_features
    except Exception:
        pass

    if not features:
        try:
            from bma_models.simple_25_factor_engine import TOP_FEATURE_SET
            features = list(TOP_FEATURE_SET)
        except Exception:
            features = [
                "momentum_10d", "ivol_20", "hist_vol_20", "rsi_21",
                "near_52w_high", "atr_ratio", "vol_ratio_20d",
                "5_days_reversal", "trend_r2_60", "liquid_momentum",
            ]
    return features


def determine_base_date(market_data, initial_base_date):
    """Find the last trading day with valid close data.

    Replicates app.py lines 3354-3454.
    """
    import pandas as pd

    close_col = None
    for candidate in ("Close", "close", "adj_close"):
        if candidate in market_data.columns:
            close_col = candidate
            break

    if isinstance(market_data.index, pd.MultiIndex) and close_col:
        all_dates = market_data.index.get_level_values("date").unique()
        all_dates = sorted([pd.Timestamp(d) for d in all_dates])

        for date in reversed(all_dates):
            try:
                date_data = market_data.xs(date, level="date", drop_level=False)
            except KeyError:
                continue
            if not date_data.empty and close_col in date_data.columns:
                valid_count = date_data[close_col].notna().sum()
                if valid_count > 0:
                    return pd.Timestamp(date).normalize()

        return pd.Timestamp(all_dates[-1]).normalize() if all_dates else initial_base_date
    elif "date" in market_data.columns:
        last_date = pd.to_datetime(market_data["date"]).max()
        return pd.Timestamp(last_date).normalize()

    return initial_base_date


def apply_ema(predictions_df, weights=EMA_WEIGHTS, score_columns=None):
    """Apply EWMA smoothing to score columns.

    Replicates app.py _apply_direct_predict_ema() (lines 6279-6353).
    """
    import numpy as np
    import pandas as pd

    if predictions_df is None or len(predictions_df) == 0:
        return predictions_df

    if not isinstance(predictions_df.index, pd.MultiIndex):
        return predictions_df

    score_columns = score_columns or ["score"]
    numeric_weights = tuple(float(w) for w in weights if isinstance(w, (int, float)) and float(w) > 0)
    if not numeric_weights:
        return predictions_df

    weight_sum = sum(numeric_weights)
    normalized_weights = tuple(w / weight_sum for w in numeric_weights)

    unique_dates = sorted(predictions_df.index.get_level_values("date").unique())
    if len(unique_dates) < 2:
        return predictions_df

    df = predictions_df.sort_index(level=["date", "ticker"]).copy()
    tickers = df.index.get_level_values("ticker").unique()

    for column in score_columns:
        if column not in df.columns:
            continue

        raw_col = f"{column}_raw"
        if raw_col not in df.columns:
            df[raw_col] = df[column]

        ema_col = f"{column}_ema"
        df[ema_col] = np.nan

        for ticker in tickers:
            try:
                ticker_series = df.xs(ticker, level="ticker")[raw_col].sort_index()
            except KeyError:
                continue
            if ticker_series.dropna().empty:
                continue

            values = ticker_series.values
            date_index = list(ticker_series.index)

            for idx_pos, current_date in enumerate(date_index):
                weighted_sum = 0.0
                weight_total = 0.0
                for offset, weight in enumerate(normalized_weights):
                    prev_idx = idx_pos - offset
                    if prev_idx < 0:
                        break
                    prev_value = values[prev_idx]
                    if __import__("pandas").isna(prev_value):
                        continue
                    weighted_sum += weight * float(prev_value)
                    weight_total += weight

                if weight_total:
                    smoothed_value = weighted_sum / weight_total
                    df.loc[(current_date, ticker), ema_col] = smoothed_value

        df[column] = df[ema_col].fillna(df[column])

    return df


def store_to_db(recs: list[dict], snapshot_id: str) -> None:
    """Store predictions to monitoring.db.

    Replicates app.py lines 4457-4509.
    """
    try:
        db_path = str(ROOT / "data" / "monitoring.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS direct_predictions (
                ts INTEGER,
                snapshot_id TEXT,
                ticker TEXT,
                score REAL
            )
        """)
        ts = int(time.time())
        rows = [
            (ts, snapshot_id, r.get("ticker"), float(r.get("score", 0.0)))
            for r in recs
            if r.get("ticker")
        ]
        if rows:
            cur.executemany(
                "INSERT INTO direct_predictions (ts, snapshot_id, ticker, score) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        conn.close()
    except Exception as e:
        emit_progress("DB warning", -1, f"Failed to store to monitoring.db: {e}")


def run_prediction(snapshot_id: str) -> None:
    """Run the full direct prediction pipeline."""
    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import BDay

    # ------------------------------------------------------------------
    # STEP 1: Load tickers from parquet
    # ------------------------------------------------------------------
    emit_progress("Loading tickers", 5, str(TICKER_DATA_PATH))
    if not TICKER_DATA_PATH.exists():
        raise FileNotFoundError(f"Ticker data file not found: {TICKER_DATA_PATH}")

    df_tickers = pd.read_parquet(TICKER_DATA_PATH)
    if isinstance(df_tickers.index, pd.MultiIndex):
        tickers = sorted(df_tickers.index.get_level_values("ticker").unique().tolist())
    elif "ticker" in df_tickers.columns:
        tickers = sorted(df_tickers["ticker"].unique().tolist())
    else:
        raise ValueError("Cannot extract tickers from parquet file")

    if not tickers:
        raise ValueError("No tickers found in parquet file")
    emit_progress("Tickers loaded", 10, f"Found {len(tickers)} tickers")

    # ------------------------------------------------------------------
    # STEP 2: Load feature list from snapshot manifest
    # ------------------------------------------------------------------
    emit_progress("Loading snapshot manifest", 12, f"Snapshot: {snapshot_id}")
    required_features = load_snapshot_features(snapshot_id)
    emit_progress("Manifest loaded", 15, f"{len(required_features)} features")

    # ------------------------------------------------------------------
    # STEP 3: Get prediction horizon
    # ------------------------------------------------------------------
    prediction_horizon = HORIZON
    try:
        from bma_models.unified_config_loader import get_time_config
        time_config = get_time_config()
        prediction_horizon = getattr(time_config, "prediction_horizon_days", HORIZON)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # STEP 4: Initialize factor engine
    # ------------------------------------------------------------------
    total_lookback = MIN_LOOKBACK + 30
    emit_progress("Initializing factor engine", 18, f"Lookback={total_lookback}, horizon=T+{prediction_horizon}")

    from bma_models.simple_25_factor_engine import Simple17FactorEngine

    engine = Simple17FactorEngine(
        lookback_days=total_lookback,
        mode="predict",
        horizon=prediction_horizon,
    )

    # ------------------------------------------------------------------
    # STEP 5: Fetch market data from Polygon API (slowest step)
    # ------------------------------------------------------------------
    today = pd.Timestamp.today()
    initial_base_date = today - BDay(1)
    start_date = initial_base_date - pd.Timedelta(days=total_lookback)
    emit_progress(
        "Fetching market data", 20,
        f"{len(tickers)} tickers, {start_date.date()} to {today.date()}",
    )

    market_data = engine.fetch_market_data(
        symbols=tickers,
        use_optimized_downloader=True,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    if market_data.empty:
        raise ValueError(f"No market data returned for {len(tickers)} tickers")
    emit_progress("Market data fetched", 45, f"Shape: {market_data.shape}")

    # ------------------------------------------------------------------
    # STEP 5b: Filter invalid close prices
    # ------------------------------------------------------------------
    close_col = None
    for candidate in ("Close", "close", "adj_close"):
        if candidate in market_data.columns:
            close_col = candidate
            break
    if close_col:
        close_values = market_data[close_col]
        invalid_mask = (
            (~pd.to_numeric(close_values, errors="coerce").notna())
            | (close_values <= 0)
            | (close_values > MAX_CLOSE)
        )
        if invalid_mask.any():
            market_data = market_data[~invalid_mask]
            if market_data.empty:
                raise ValueError("All market data rows removed due to invalid close prices")

    # ------------------------------------------------------------------
    # STEP 6: Determine base_date (last valid trading day)
    # ------------------------------------------------------------------
    emit_progress("Determining base date", 48, "Finding last trading day with close data")
    base_date = determine_base_date(market_data, initial_base_date)
    emit_progress("Base date determined", 50, f"Base: {base_date.strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------
    # STEP 7: Compute alpha factors
    # ------------------------------------------------------------------
    emit_progress("Computing alpha factors", 52, "Running Simple17FactorEngine")
    all_feature_data = engine.compute_all_17_factors(market_data, mode="predict")
    if all_feature_data.empty:
        raise ValueError("Factor computation returned empty data")

    # Select only required features
    feature_columns = [col for col in required_features if col in all_feature_data.columns]
    missing_cols = sorted(set(required_features) - set(feature_columns))
    if not feature_columns:
        raise ValueError(
            f"None of the required snapshot features are available. "
            f"Missing: {missing_cols}, Available: {list(all_feature_data.columns)}"
        )
    all_feature_data = all_feature_data[feature_columns]

    # Handle Sato factors if required
    needs_sato = any(name.startswith("feat_sato") for name in required_features)
    if needs_sato:
        if "feat_sato_momentum_10d" not in all_feature_data.columns or "feat_sato_divergence_10d" not in all_feature_data.columns:
            try:
                from scripts.sato_factor_calculation import calculate_sato_factors

                sato_data = all_feature_data.copy()
                if "adj_close" not in sato_data.columns and "Close" in sato_data.columns:
                    sato_data["adj_close"] = sato_data["Close"]

                has_vol_ratio = "vol_ratio_20d" in sato_data.columns
                if "Volume" not in sato_data.columns:
                    if has_vol_ratio:
                        sato_data["Volume"] = 1_000_000 * sato_data["vol_ratio_20d"].fillna(1.0).clip(lower=0.1, upper=10.0)
                    else:
                        sato_data["Volume"] = 1_000_000

                sato_factors_df = calculate_sato_factors(
                    df=sato_data,
                    price_col="adj_close",
                    volume_col="Volume",
                    vol_ratio_col="vol_ratio_20d",
                    lookback_days=10,
                    vol_window=20,
                    use_vol_ratio_directly=has_vol_ratio,
                )
                if not isinstance(sato_factors_df.index, pd.MultiIndex):
                    if "date" in sato_factors_df.columns and "ticker" in sato_factors_df.columns:
                        sato_factors_df = sato_factors_df.set_index(["date", "ticker"])

                all_feature_data["feat_sato_momentum_10d"] = sato_factors_df["feat_sato_momentum_10d"].reindex(all_feature_data.index).fillna(0.0)
                all_feature_data["feat_sato_divergence_10d"] = sato_factors_df["feat_sato_divergence_10d"].reindex(all_feature_data.index).fillna(0.0)
            except Exception:
                if "feat_sato_momentum_10d" not in all_feature_data.columns:
                    all_feature_data["feat_sato_momentum_10d"] = 0.0
                if "feat_sato_divergence_10d" not in all_feature_data.columns:
                    all_feature_data["feat_sato_divergence_10d"] = 0.0

    emit_progress("Factors computed", 62, f"Feature data shape: {all_feature_data.shape}")

    # ------------------------------------------------------------------
    # STEP 8: Standardize MultiIndex format
    # ------------------------------------------------------------------
    emit_progress("Standardizing data format", 64, "Normalizing MultiIndex")

    if not isinstance(all_feature_data.index, pd.MultiIndex):
        raise ValueError(f"all_feature_data must have MultiIndex, got: {type(all_feature_data.index)}")

    date_level = all_feature_data.index.get_level_values("date")
    if isinstance(date_level, pd.DatetimeIndex):
        date_normalized = date_level.tz_localize(None).normalize() if date_level.tz is not None else date_level.normalize()
    else:
        date_converted = pd.to_datetime(date_level)
        if isinstance(date_converted, pd.DatetimeIndex):
            date_normalized = date_converted.tz_localize(None).normalize() if date_converted.tz is not None else date_converted.normalize()
        else:
            date_normalized = date_converted.dt.tz_localize(None).dt.normalize() if date_converted.dt.tz is not None else date_converted.dt.normalize()

    ticker_level = all_feature_data.index.get_level_values("ticker").astype(str).str.strip().str.upper()
    all_feature_data.index = pd.MultiIndex.from_arrays(
        [date_normalized, ticker_level], names=["date", "ticker"]
    )

    # Remove duplicates
    if all_feature_data.index.duplicated().any():
        all_feature_data = all_feature_data[~all_feature_data.index.duplicated(keep="first")]
        all_feature_data = all_feature_data.groupby(level=["date", "ticker"]).first()

    emit_progress("Data standardized", 66, f"Shape: {all_feature_data.shape}")

    # ------------------------------------------------------------------
    # STEP 9: Run ML model prediction
    # ------------------------------------------------------------------
    emit_progress("Running ML models", 68, "Loading snapshot models")

    # Filter to data up to base_date
    date_mask = all_feature_data.index.get_level_values("date") <= base_date
    date_feature_data = all_feature_data[date_mask].copy()

    # Deduplicate
    if date_feature_data.index.duplicated().any():
        date_feature_data = date_feature_data[~date_feature_data.index.duplicated(keep="first")]
    date_feature_data = date_feature_data.groupby(level=["date", "ticker"]).first()

    if date_feature_data.empty:
        raise ValueError(f"No feature data available for base date {base_date.strftime('%Y-%m-%d')}")

    emit_progress("Predicting", 72, f"snapshot={snapshot_id}, horizon=T+{prediction_horizon}")

    from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

    model = UltraEnhancedQuantitativeModel()
    results = model.predict_with_snapshot(
        feature_data=date_feature_data,
        snapshot_id=snapshot_id,
        universe_tickers=tickers,
        as_of_date=base_date,
        prediction_days=prediction_horizon,
    )

    if not results.get("success", False):
        error_msg = results.get("error", "Unknown error")
        raise RuntimeError(f"Prediction failed: {error_msg}")

    emit_progress("Models completed", 82, "Processing predictions")

    # ------------------------------------------------------------------
    # STEP 10: Build pred_df with base model scores
    # ------------------------------------------------------------------
    predictions_raw = results.get("predictions_raw") or results.get("predictions")
    base_predictions = results.get("base_predictions")
    pred_date = base_date

    if predictions_raw is None:
        raise ValueError("No predictions returned from model")

    # Convert to DataFrame
    if isinstance(predictions_raw, pd.Series):
        # Remove duplicate indices
        if isinstance(predictions_raw.index, pd.MultiIndex) and predictions_raw.index.duplicated().any():
            predictions_raw = predictions_raw[~predictions_raw.index.duplicated(keep="first")]
        pred_df = predictions_raw.to_frame("score")
    else:
        pred_df = predictions_raw.copy()
        if "score" not in pred_df.columns and len(pred_df.columns) > 0:
            pred_df.columns = ["score"]

    # Ensure MultiIndex with date and ticker
    if not isinstance(pred_df.index, pd.MultiIndex):
        tickers_from_index = pred_df.index.tolist()
        dates_list = [pred_date] * len(tickers_from_index)
        pred_df.index = pd.MultiIndex.from_arrays(
            [dates_list, tickers_from_index], names=["date", "ticker"]
        )
    else:
        new_index = pd.MultiIndex.from_arrays(
            [[pred_date] * len(pred_df), pred_df.index.get_level_values("ticker")],
            names=["date", "ticker"],
        )
        pred_df.index = new_index

    # Remove duplicate indices
    if pred_df.index.duplicated().any():
        pred_df = pred_df[~pred_df.index.duplicated(keep="first")]
    if isinstance(pred_df.index, pd.MultiIndex):
        pred_df = pred_df.groupby(level=["date", "ticker"]).first()

    # Add base model predictions
    if base_predictions is not None and isinstance(base_predictions, pd.DataFrame):
        if isinstance(base_predictions.index, pd.MultiIndex) and base_predictions.index.duplicated().any():
            base_predictions = base_predictions[~base_predictions.index.duplicated(keep="first")]

        if isinstance(base_predictions.index, pd.MultiIndex):
            base_predictions_aligned = base_predictions.reindex(pred_df.index)
        else:
            base_predictions_aligned = base_predictions.reindex(pred_df.index.get_level_values("ticker"))
            base_predictions_aligned.index = pred_df.index

        if base_predictions_aligned.index.duplicated().any():
            base_predictions_aligned = base_predictions_aligned[~base_predictions_aligned.index.duplicated(keep="first")]

        for src, dst in [
            ("pred_lambdarank", "score_lambdarank"),
            ("pred_catboost", "score_catboost"),
            ("pred_elastic", "score_elastic"),
            ("pred_xgb", "score_xgb"),
        ]:
            if src in base_predictions_aligned.columns:
                pred_df[dst] = base_predictions_aligned[src]

    emit_progress("Predictions processed", 85, f"{len(pred_df)} ticker predictions")

    # ------------------------------------------------------------------
    # STEP 11: Apply EMA smoothing
    # ------------------------------------------------------------------
    emit_progress("Applying EMA smoothing", 87, f"Weights: {EMA_WEIGHTS}")
    combined_predictions = pred_df.copy()
    final_predictions = apply_ema(
        combined_predictions,
        weights=EMA_WEIGHTS,
        score_columns=["score", "score_lambdarank"],
    )
    emit_progress("EMA applied", 89, "Generating rankings")

    # ------------------------------------------------------------------
    # STEP 12: Get latest date predictions & build recommendations
    # ------------------------------------------------------------------
    dates_in_pred = sorted(final_predictions.index.get_level_values("date").unique())
    latest_date = dates_in_pred[-1] if dates_in_pred else None

    recs: list[dict] = []
    if latest_date is not None:
        try:
            latest_predictions = final_predictions.xs(latest_date, level="date", drop_level=False)
        except KeyError:
            latest_predictions = final_predictions

        # Remove duplicate tickers
        if isinstance(latest_predictions.index, pd.MultiIndex):
            ticker_level = latest_predictions.index.get_level_values("ticker")
            if ticker_level.duplicated().any():
                latest_predictions = latest_predictions[~ticker_level.duplicated(keep="first")]

        latest_predictions = latest_predictions.sort_values("score", ascending=False)

        for idx, row in latest_predictions.iterrows():
            ticker = idx[1] if isinstance(idx, tuple) else idx
            score = float(row.get("score", 0.0))
            ema4 = float(row.get("score_ema", score)) if "score_ema" in row.index else score
            recs.append({"ticker": str(ticker), "score": score, "ema4": ema4})

    # ------------------------------------------------------------------
    # STEP 13: Generate Excel report
    # ------------------------------------------------------------------
    output_dir = ROOT / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = output_dir / f"direct_predict_top30_{timestamp}.xlsx"
    emit_progress("Generating Excel", 92, str(excel_path))

    try:
        from direct_predict_ewma_excel import generate_excel_ranking_report

        generate_excel_ranking_report(
            final_predictions,
            str(excel_path),
            model_name="MetaRankerStacker",
            top_n=30,
        )
        excel_path_str = str(excel_path)
    except Exception as e:
        emit_progress("Excel warning", 93, f"Excel generation failed: {e}")
        excel_path_str = ""

    # ------------------------------------------------------------------
    # STEP 14: Store predictions to monitoring.db
    # ------------------------------------------------------------------
    emit_progress("Storing predictions", 95, "Writing to monitoring.db")
    store_to_db(recs, snapshot_id)

    # ------------------------------------------------------------------
    # STEP 15: Build final JSON payload for C# bridge
    # ------------------------------------------------------------------
    top20_list = [
        {"ticker": r["ticker"], "score": r["score"], "ema4": r["ema4"]}
        for r in recs[:20]
    ]
    top10_list = [r["ticker"] for r in recs[:10]]

    payload = {
        "run_id": datetime.utcnow().isoformat() + "Z",
        "as_of": base_date.strftime("%Y-%m-%d"),
        "excel_path": excel_path_str,
        "top20": top20_list,
        "top10": top10_list,
    }

    emit_progress("Complete", 100, f"Top 20 generated, Excel at {excel_path_str}")
    json.dump(payload, sys.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct prediction bridge")
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()

    snapshot_id = (args.snapshot or "").strip() or DEFAULT_SNAPSHOT_ID

    # Change working directory to ROOT so relative paths in bma_models work
    os.chdir(str(ROOT))

    try:
        run_prediction(snapshot_id)
    except Exception as exc:
        emit_progress("FAILED", -1, str(exc))
        error_payload = {
            "run_id": datetime.utcnow().isoformat() + "Z",
            "as_of": datetime.utcnow().date().isoformat(),
            "excel_path": "",
            "top20": [],
            "top10": [],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        json.dump(error_payload, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
