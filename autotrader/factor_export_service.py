#!/usr/bin/env python3
"""Shared utilities for exporting polygon factor datasets with MultiIndex support."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Dict, Any, Optional, Callable

import pandas as pd

from autotrader.database import StockDatabase
from bma_models.simple_25_factor_engine import Simple17FactorEngine
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

logger = logging.getLogger(__name__)
StatusCallback = Callable[[str], None]


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield list(seq[idx: idx + size])


def default_date_window(years: int = 5) -> tuple[str, str]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years + 30)
    return start.isoformat(), end.isoformat()


def normalize_factor_frame(df: pd.DataFrame, keep_multiindex: bool = True) -> pd.DataFrame:
    """
    Normalize factor DataFrame format.

    Args:
        df: Input factor DataFrame
        keep_multiindex: Keep MultiIndex format (default True for ML training)

    Returns:
        Normalized DataFrame (MultiIndex or flat)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()

    # If MultiIndex and need to keep, validate and return directly
    if keep_multiindex and isinstance(result.index, pd.MultiIndex):
        index_names_lower = [name.lower() if name else '' for name in result.index.names]

        if 'date' in index_names_lower and ('ticker' in index_names_lower or 'symbol' in index_names_lower):
            # Standardize index names: symbol -> ticker (与训练流程保持一致)
            new_names = []
            for name in result.index.names:
                if name and name.lower() == 'symbol':
                    new_names.append('ticker')
                elif name and name.lower() == 'date':
                    new_names.append('date')
                else:
                    new_names.append(name)

            result.index.names = new_names
            logger.info(f"Keeping MultiIndex: {result.index.names}, shape={result.shape}")
            return result
        else:
            logger.warning(f"MultiIndex invalid: {result.index.names}, converting to flat")

    # Convert to flat format
    if isinstance(result.index, pd.MultiIndex) or result.index.name:
        result = result.reset_index()

    # Standardize column names
    symbol_col = None
    for candidate in ('symbol', 'ticker'):
        if candidate in result.columns:
            symbol_col = candidate
            break
    if not symbol_col:
        raise ValueError('factor dataframe missing symbol/ticker column')
    if symbol_col != 'symbol':
        result.rename(columns={symbol_col: 'symbol'}, inplace=True)

    if 'as_of_date' not in result.columns:
        if 'date' in result.columns:
            result.rename(columns={'date': 'as_of_date'}, inplace=True)
        elif 'timestamp' in result.columns:
            result.rename(columns={'timestamp': 'as_of_date'}, inplace=True)

    if 'as_of_date' in result.columns:
        result['as_of_date'] = pd.to_datetime(result['as_of_date'], errors='coerce')

    logger.info(f"Flat format: {list(result.columns)[:5]}..., shape={result.shape}")
    return result


def load_symbols(limit: Optional[int] = None) -> List[str]:
    with StockDatabase() as db:
        symbols = db.get_all_tickers()
    symbols = [s.strip().upper() for s in symbols if s]
    if limit is not None:
        symbols = symbols[:limit]
    return symbols


def export_polygon_factors(
    max_symbols: int = 2600,
    batch_size: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 5,
    output_dir: str | Path = 'data/factor_exports',
    log_level: str = 'INFO',
    status_callback: Optional[StatusCallback] = None,
    symbols: Optional[Sequence[str]] = None,
    pool_name: Optional[str] = None,
    mode: str = 'train',
    keep_multiindex: bool = True,
    horizon: int = 5,
) -> Dict[str, Any]:
    """Export polygon-based factor dataset with MultiIndex support."""
    logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not start_date or not end_date:
        start_date, end_date = default_date_window(years)

    if symbols:
        symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
        if not symbols:
            raise RuntimeError('Provided symbol list is empty')
        if max_symbols:
            symbols = symbols[:max_symbols]
    else:
        symbols = load_symbols(max_symbols)
    if not symbols:
        raise RuntimeError('No symbols available from StockDatabase')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if status_callback:
        pool_desc = pool_name or '数据库股票池'
        mode_desc = '训练数据(含target,已dropna)' if mode == 'train' else '因子数据(含target,可能有NaN)'
        format_desc = 'MultiIndex格式' if keep_multiindex else 'Flat格式'
        status_callback(
            f'准备导出{mode_desc} 【{pool_desc}】\n'
            f'  股票数: {len(symbols)} | 区间: {start_date} 到 {end_date}\n'
            f'  模式: {mode.upper()} | 格式: {format_desc}'
        )

    model = UltraEnhancedQuantitativeModel(preserve_state=False)
    model.simple_25_engine = Simple17FactorEngine(mode=mode, lookback_days=252 * years, horizon=horizon)

    manifest: List[Dict[str, Any]] = []
    total_samples = 0

    for batch_id, batch in enumerate(chunked(symbols, batch_size), start=1):
        try:
            if status_callback:
                status_callback(f'批次 {batch_id}: 处理 {len(batch)} 只股票')

            feature_df = model.get_data_and_features(batch, start_date, end_date, mode=mode)

            if feature_df is None or feature_df.empty:
                logger.warning(f'Batch {batch_id} produced no data')
                continue

            normalized = normalize_factor_frame(feature_df, keep_multiindex=keep_multiindex)

            if normalized.empty:
                logger.warning(f'Batch {batch_id} normalized to empty')
                continue

            shard = output_path / f'polygon_factors_batch_{batch_id:04d}.parquet'
            if keep_multiindex:
                normalized.to_parquet(shard)
            else:
                normalized.to_parquet(shard, index=False)

            batch_samples = len(normalized)
            total_samples += batch_samples

            manifest.append({
                'batch_id': batch_id,
                'file': str(shard),
                'symbols': batch,
                'sample_count': batch_samples,
            })

            if status_callback:
                status_callback(f'批次 {batch_id} 完成: {batch_samples}样本')

        except Exception as exc:
            logger.exception(f'Batch {batch_id} failed')
            if status_callback:
                status_callback(f'批次 {batch_id} 失败: {exc}')

    manifest_path = None
    if manifest:
        manifest_path = output_path / 'manifest.parquet'
        pd.DataFrame(manifest).to_parquet(manifest_path, index=False)

        readme_path = output_path / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""MultiIndex因子数据导出
==================
导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
股票池: {pool_name or 'database'}
数据区间: {start_date} 到 {end_date}
导出模式: {mode.upper()}
数据格式: {'MultiIndex' if keep_multiindex else 'Flat'}

统计: {len(manifest)}批次, {total_samples:,}样本

读取示例:
import pandas as pd
df = pd.read_parquet('polygon_factors_batch_0001.parquet')
# df.index.names = ['date', 'symbol']
X = df.drop(columns=['Close', 'target'], errors='ignore')
y = df['target']
""")

        if status_callback:
            status_callback(f'完成！{len(manifest)}批次, {total_samples:,}样本')

    return {
        'output_dir': str(output_path),
        'manifest': str(manifest_path) if manifest_path else None,
        'batch_count': len(manifest),
        'total_samples': total_samples,
        'mode': mode,
        'keep_multiindex': keep_multiindex,
    }
