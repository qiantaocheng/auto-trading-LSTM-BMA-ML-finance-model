#!/usr/bin/env python3
"""Thin helpers around optional configuration loader dependencies."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable


@lru_cache(maxsize=1)
def _resolve_loader() -> Callable[[], Any]:
    """Return the external `get_config_manager` loader or a safe fallback."""
    try:  # pragma: no cover - optional dependency
        from bma_models.unified_config_loader import get_config_manager as loader  # type: ignore
        return loader
    except Exception:
        return lambda: None


def get_config_manager() -> Any:  # pragma: no cover - simple delegation
    """Safely fetch the unified config manager if the dependency is installed."""
    loader = _resolve_loader()
    try:
        return loader()
    except Exception:
        return None


__all__ = ["get_config_manager"]
