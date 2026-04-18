"""
cache_utils.py
--------------
Handles persistent signal caching to signals_cache.json.
Used by strategy.py (write) and app.py (read metadata).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals_cache.json")

_IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60  # UTC+5:30


def _now_ist() -> datetime:
    """Return current time in IST (UTC+5:30)."""
    utc_now = datetime.now(timezone.utc)
    from datetime import timedelta, timezone as tz
    ist_tz = tz(timedelta(seconds=_IST_OFFSET_SECONDS), name="IST")
    return utc_now.astimezone(ist_tz)


def is_market_open() -> bool:
    """
    Returns True if NSE market is currently open.
    Market hours: Monday–Friday, 09:15 – 15:30 IST.
    """
    now = _now_ist()
    # Weekday: 0=Monday ... 4=Friday
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def save_signals_cache(signals: List[Dict]) -> None:
    """
    Persist signals to signals_cache.json with a timestamp.
    Only called when a non-empty live scan result is available.
    """
    if not signals:
        return
    payload: Dict[str, Any] = {
        "last_updated": _now_ist().strftime("%Y-%m-%d %H:%M"),
        "signals": signals,
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        # Non-fatal: log and continue
        print(f"[cache_utils] WARNING: could not write cache: {exc}")


def load_signals_cache() -> Optional[Dict[str, Any]]:
    """
    Load the persisted cache from disk.
    Returns dict with keys 'signals' and 'last_updated', or None if unavailable.
    """
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("signals"), list):
            return data
    except Exception as exc:
        print(f"[cache_utils] WARNING: could not read cache: {exc}")
    return None


def get_cache_meta() -> Dict[str, Any]:
    """
    Return lightweight metadata about the cache without loading all signals.
    Useful for health/status endpoints.
    """
    cached = load_signals_cache()
    if cached is None:
        return {"cache_available": False, "last_updated": None, "signal_count": 0}
    return {
        "cache_available": True,
        "last_updated": cached.get("last_updated"),
        "signal_count": len(cached.get("signals", [])),
    }
