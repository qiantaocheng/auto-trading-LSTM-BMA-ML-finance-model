#!/usr/bin/env python
"""Fetch US market calendar info from Polygon (Massive domain)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

API_KEY = os.environ.get("POLYGON_API_KEY", "isFExbaO1xdmrV6f6p3zHCxk8IArjeowQ1")
BASE_URL = os.environ.get("POLYGON_BASE", "https://api.polygon.io")


def fetch(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}?apiKey={API_KEY}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    now = fetch("/v1/marketstatus/now")
    upcoming = fetch("/v1/marketstatus/upcoming")
    payload = {
        "current": now,
        "upcoming": upcoming,
    }
    json.dump(payload, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
