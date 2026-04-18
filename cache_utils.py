from __future__ import annotations

import json
import os
from datetime import datetime, time as dtime
from typing import Any, Dict, Optional

# ── TIMEZONE ────────────────────────────────────────────────────────────────
try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
    def _now_ist() -> datetime:
        return datetime.now(_IST)
except ImportError:
    from datetime import timezone, timedelta
    _IST_OFFSET = timedelta(hours=5, minutes=30)
    def _now_ist() -> datetime:  # type: ignore[misc]
        return datetime.utcnow() + _IST_OFFSET

# ── CONSTANTS ────────────────────────────────────────────────────────────────
CACHE_FILE = "signals_cache.json"
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def is_market_open() -> bool:
    """Returns True during NSE trading hours on weekdays."""
    now = _now_ist()
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    t = now.time().replace(tzinfo=None)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def load_signals_cache() -> Optional[Dict[str, Any]]:
    """Load cached signals from disk. Returns None if unavailable."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_signals_cache(signals: list) -> None:
    """Persist signals to disk with a timestamp."""
    try:
        payload = {
            "signals": signals,
            "last_updated": _now_ist().strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass
