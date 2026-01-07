#!/usr/bin/env python3
"""Centralised helpers for resolving project data locations."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_FILENAME = "trading_system.db"


def ensure_data_dir() -> Path:
    """Create the shared data directory if needed and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def resolve_data_path(filename: str | Path | None = None) -> Path:
    """Return an absolute path under the shared data directory."""
    ensure_data_dir()
    path = Path(filename) if filename else Path(DEFAULT_DB_FILENAME)
    if not path.is_absolute():
        path = DATA_DIR / path
    return path


__all__ = ["DATA_DIR", "DEFAULT_DB_FILENAME", "ensure_data_dir", "resolve_data_path"]
