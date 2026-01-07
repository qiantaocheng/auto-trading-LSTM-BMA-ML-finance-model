#!/usr/bin/env python3
"""Adapter that exposes HETRS Nasdaq signals to the AutoTrader stack."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .config_helpers import get_config_manager


logger = logging.getLogger(__name__)


def _as_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    return Path(value)


@dataclass
class HetrsConfig:
    symbols: Iterable[str]
    min_abs_position: float
    timeseries_csv: Optional[Path]
    results_root: Path
    learning_summary: Optional[Path] = None
    entry_position_abs: Optional[float] = None
    exit_position_abs: Optional[float] = None


class HetrsNasdaqSignalProvider:
    """Loads the latest HETRS Nasdaq backtest output and converts it to signals."""

    def __init__(self, cfg: HetrsConfig):
        self.cfg = cfg
        self.covered_symbols = {str(sym).upper() for sym in cfg.symbols}
        self._cache_df: Optional[pd.DataFrame] = None
        self._cache_mtime: Optional[float] = None
        self._cache_path: Optional[Path] = None
        self.entry_position_abs = float(cfg.entry_position_abs or cfg.min_abs_position or 0.2)
        self.exit_position_abs = float(cfg.exit_position_abs or 0.0)
        self._load_learning_thresholds(cfg.learning_summary)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_signal(self, symbol: str, threshold: float = 0.3,
                   as_of: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        if not symbol:
            return None
        symbol_upper = symbol.upper()
        if symbol_upper not in self.covered_symbols:
            return None

        df = self._load_timeseries()
        if df is None or df.empty:
            return None

        target_ts = self._normalize_timestamp(as_of)
        row = self._select_row(df, target_ts)
        if row is None:
            return None

        raw_position = float(row.get('position', 0.0))
        position = max(raw_position, 0.0)
        strength = position
        entry_thr = max(float(threshold or 0.0), self.entry_position_abs)
        exit_thr = max(0.0, self.exit_position_abs)
        meets = strength >= entry_thr
        exit_flag = strength <= exit_thr
        side = 'BUY' if position > 0 else 'SELL'
        confidence = min(0.5 + 0.4 * strength, 0.95)

        metadata = {
            'position': position,
            'raw_position': raw_position,
            'turnover': _safe_float(row.get('turnover')),
            'trade_cost': _safe_float(row.get('trade_cost')),
            'r_port': _safe_float(row.get('r_port')),
            'equity': _safe_float(row.get('equity')),
            'timeseries_path': str(self._cache_path) if self._cache_path else None,
            'entry_threshold': entry_thr,
            'exit_threshold': exit_thr,
            'exit_signal': exit_flag,
        }

        return {
            'symbol': symbol_upper,
            'signal_value': position,
            'signal_strength': strength,
            'confidence': confidence,
            'side': side,
            'can_trade': meets,
            'meets_threshold': meets,
            'meets_confidence': True,
            'threshold': entry_thr,
            'timestamp': row.name.to_pydatetime(),
            'can_trade_delayed': True,
            'data_quality': 1.0,
            'delay_reason': None if meets else f'|position|<{entry_thr:.3f}',
            'source': 'HETRS_NASDAQ',
            'metadata': metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_row(self, df: pd.DataFrame, target_ts: datetime) -> Optional[pd.Series]:
        if df.empty:
            return None
        eligible = df.loc[df.index <= target_ts]
        if eligible.empty:
            # use earliest row
            return df.iloc[0]
        return eligible.iloc[-1]

    def _load_timeseries(self) -> Optional[pd.DataFrame]:
        path = self._resolve_timeseries_path()
        if path is None or not path.exists():
            return None
        mtime = path.stat().st_mtime
        if self._cache_df is not None and self._cache_mtime == mtime:
            return self._cache_df

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to load HETRS timeseries %s: %s", path, exc)
            return None

        date_col = None
        for candidate in ('date', 'Date', 'Unnamed: 0'):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            logger.warning("HETRS timeseries %s missing date column", path)
            return None

        df = df.rename(columns={date_col: 'date'})
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as exc:
            logger.warning("Failed to parse dates in %s: %s", path, exc)
            return None

        df = df.set_index('date').sort_index()
        self._cache_df = df
        self._cache_mtime = mtime
        self._cache_path = path
        logger.info("Loaded HETRS Nasdaq timeseries (%d rows) from %s", len(df), path)
        return df

    def _resolve_timeseries_path(self) -> Optional[Path]:
        if self.cfg.timeseries_csv and self.cfg.timeseries_csv.exists():
            return self.cfg.timeseries_csv

        root = self.cfg.results_root
        if not root.exists():
            return None

        latest_meta: Optional[Path] = None
        latest_mtime = float('-inf')
        for child in root.iterdir():
            if not child.is_dir():
                continue
            meta = child / 'meta_timeseries.csv'
            if not meta.exists():
                continue
            mtime = meta.stat().st_mtime
            if mtime > latest_mtime:
                latest_meta = meta
                latest_mtime = mtime

        return latest_meta

    @staticmethod
    def _normalize_timestamp(as_of: Optional[datetime]) -> datetime:
        if as_of is None:
            as_of = datetime.utcnow()
        return pd.Timestamp(as_of).tz_localize(None).to_pydatetime()

    def _load_learning_thresholds(self, summary_path: Optional[Path]) -> None:
        path = summary_path or Path('result/hetrs_learning/learning_summary.json')
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.debug("Failed to parse HETRS learning summary %s: %s", path, exc)
            return
        events = data.get('event_thresholds') or {}
        entry = events.get('enter_long') or {}
        exit_info = events.get('exit_long') or {}
        if 'hetrs_position_abs' in entry:
            self.entry_position_abs = float(entry['hetrs_position_abs'])
        if 'hetrs_position_abs' in exit_info:
            self.exit_position_abs = float(exit_info['hetrs_position_abs'])
        logger.info(
            "HETRS thresholds updated from %s entry>=%.3f exit<=%.3f",
            path,
            self.entry_position_abs,
            self.exit_position_abs,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


_provider: Optional[HetrsNasdaqSignalProvider] = None


def get_hetrs_signal_provider() -> Optional[HetrsNasdaqSignalProvider]:
    """Return a singleton HETRS signal provider if configuration is available."""
    global _provider
    if _provider is not None:
        return _provider

    cfg = _build_config()
    if cfg is None:
        logger.info("HETRS Nasdaq signal provider not configured")
        return None

    _provider = HetrsNasdaqSignalProvider(cfg)
    return _provider


def _build_config() -> Optional[HetrsConfig]:
    cfg = {}
    mgr = get_config_manager()
    if mgr and hasattr(mgr, 'get'):
        try:
            cfg = mgr.get('signals', {}).get('hetrs', {}) or {}
        except Exception:
            cfg = {}

    env_csv = os.environ.get('HETRS_TIMESERIES_CSV')
    env_root = os.environ.get('HETRS_RESULTS_DIR')
    env_summary = os.environ.get('HETRS_LEARNING_SUMMARY')

    symbols = cfg.get('symbols') or ['QQQ', 'NDX', 'NDX100']
    min_abs = float(cfg.get('min_abs_position', 0.15))
    csv_path = _as_path(cfg.get('timeseries_csv'))
    if env_csv:
        csv_path = Path(env_csv)

    results_root = _as_path(cfg.get('results_dir')) or Path('results/hetrs_nasdaq')
    if env_root:
        results_root = Path(env_root)

    if not results_root.exists() and (not csv_path or not csv_path.exists()):
        logger.warning("HETRS results directory %s not found", results_root)
        return None

    summary_path = _as_path(cfg.get('learning_summary')) if isinstance(cfg, dict) else None
    if env_summary:
        summary_path = Path(env_summary)
    if summary_path and not summary_path.exists():
        logger.warning("HETRS learning summary %s not found", summary_path)
        summary_path = None

    return HetrsConfig(
        symbols=symbols,
        min_abs_position=min_abs,
        timeseries_csv=csv_path,
        results_root=results_root,
        learning_summary=summary_path,
        entry_position_abs=float(cfg.get('entry_position_abs')) if isinstance(cfg, dict) and cfg.get('entry_position_abs') is not None else None,
        exit_position_abs=float(cfg.get('exit_position_abs')) if isinstance(cfg, dict) and cfg.get('exit_position_abs') is not None else None,
    )


__all__ = [
    'HetrsNasdaqSignalProvider',
    'get_hetrs_signal_provider',
]
