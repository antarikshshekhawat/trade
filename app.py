from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Dict, List

from flask import Flask, jsonify, send_from_directory

from data import FALLBACK_IPO_STOCKS, FALLBACK_UNIVERSE, build_stock_universe, get_default_provider
from strategy import scan_market
from cache_utils import get_cache_meta, is_market_open, load_signals_cache

app = Flask(__name__)

_cache_lock = Lock()
_cache: Dict[str, object] = {
    "signals": [],
    "generated_at": None,
    "expires_at": 0.0,
    "refreshing": False,
    "last_error": None,
    "last_nonempty_signals": [],
    "last_nonempty_generated_at": None,
}

CACHE_SECONDS = 300  # 5 minutes
UNIVERSE_CACHE_SECONDS = 600
_universe_cache: Dict[str, object] = {
    "stocks": {
        "largecap": FALLBACK_UNIVERSE["largecap"],
        "midcap": FALLBACK_UNIVERSE["midcap"],
        "smallcap": FALLBACK_UNIVERSE["smallcap"],
        "ipo": FALLBACK_IPO_STOCKS,
    },
    "expires_at": 0.0,
}

@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/")
def home():
    # Make sure your HTML file is correctly named here
    return send_from_directory(".", "index.html")

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _refresh_signals() -> None:
    with _cache_lock:
        if _cache["refreshing"]:
            return
        _cache["refreshing"] = True

    generated_at = _utc_now().isoformat()
    universe = _get_universe_cached()
    provider = get_default_provider()
    try:
        signals = scan_market(
            provider=provider,
            categorized_stocks=universe,
            max_workers=20,
            max_signals=30,
            scan_timeout_sec=22,
        )
        with _cache_lock:
            _cache["signals"] = signals
            _cache["generated_at"] = generated_at
            _cache["expires_at"] = _utc_now().timestamp() + CACHE_SECONDS
            _cache["last_error"] = None
            if signals:
                _cache["last_nonempty_signals"] = signals
                _cache["last_nonempty_generated_at"] = generated_at
    except Exception as exc:
        with _cache_lock:
            _cache["last_error"] = str(exc)
    finally:
        with _cache_lock:
            _cache["refreshing"] = False

def _trigger_refresh_if_needed() -> None:
    now_ts = _utc_now().timestamp()
    with _cache_lock:
        is_stale = now_ts >= float(_cache["expires_at"])
        already_refreshing = bool(_cache["refreshing"])
    if is_stale and not already_refreshing:
        worker = Thread(target=_refresh_signals, daemon=True)
        worker.start()

def _get_signals_cached() -> Dict[str, object]:
    _trigger_refresh_if_needed()
    with _cache_lock:
        return {
            "signals": _cache["signals"],
            "generated_at": _cache["generated_at"],
            "cached": True,
            "refreshing": _cache["refreshing"],
            "last_error": _cache["last_error"],
            "last_nonempty_signals": _cache["last_nonempty_signals"],
            "last_nonempty_generated_at": _cache["last_nonempty_generated_at"],
        }

def _normalize_signal(item: Dict) -> Dict:
    ticker = str(item.get("ticker") or item.get("symbol") or "UNKNOWN")
    category = str(item.get("category") or "largecap")
    entry = float(item.get("entry") or item.get("price") or 100.0)
    price = float(item.get("price") or entry)
    sl = float(item.get("sl") or item.get("stop_loss") or round(entry * 0.97, 2))
    target = float(item.get("target") or round(entry * 1.06, 2))
    rr_val = item.get("rr") if item.get("rr") is not None else item.get("risk_reward")
    rr = float(rr_val) if rr_val is not None else round((target - entry) / max(entry - sl, 0.01), 2)
    rr_text = str(item.get("rr_text") or f"1:{rr}")
    sl_pct = float(item.get("sl_pct") if item.get("sl_pct") is not None else round((sl / entry - 1) * 100, 2))
    target_pct = float(item.get("target_pct") if item.get("target_pct") is not None else round((target / entry - 1) * 100, 2))
    pattern = str(item.get("pattern") or "MACD + RSI")
    last_close = float(item.get("last_close") or item.get("prev_close") or entry)
    is_candidate = bool(item.get("is_candidate", False))
    candidate_score = float(item.get("candidate_score", 0.0))

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "entry": round(entry, 2),
        "target": round(target, 2),
        "sl": round(sl, 2),
        "rr": round(rr, 2),
        "rr_text": rr_text,
        "pattern": pattern,
        "category": category,
        "sl_pct": round(sl_pct, 2),
        "target_pct": round(target_pct, 2),
        "last_close": round(last_close, 2),
        "is_candidate": is_candidate,
        "candidate_score": candidate_score,
        "mover_5d_pct": float(item.get("mover_5d_pct", 0.0))
    }

def _fallback_from_universe() -> List[Dict]:
    universe = _get_universe_cached()
    rows: List[Dict] = []
    for category, symbols in universe.items():
        for symbol in symbols[:5]:
            rows.append(
                _normalize_signal({
                    "ticker": symbol,
                    "category": category,
                    "price": 100.0, "entry": 100.0, "target": 106.0, "sl": 97.0,
                    "rr": 2.0, "rr_text": "1:2.0", "pattern": "Top Mover (fallback)",
                    "sl_pct": -3.0, "target_pct": 6.0, "is_candidate": False
                })
            )
    return rows[:20]

def _get_universe_cached() -> Dict[str, List[str]]:
    now_ts = _utc_now().timestamp()
    with _cache_lock:
        if now_ts < float(_universe_cache["expires_at"]):
            return _universe_cache["stocks"] 

    stocks = build_stock_universe()
    with _cache_lock:
        _universe_cache["stocks"] = stocks
        _universe_cache["expires_at"] = _utc_now().timestamp() + UNIVERSE_CACHE_SECONDS
    return stocks

def _calculate_counts(signals: List[Dict]) -> Dict[str, int]:
    return {
        "all": len(signals),
        "largecap": sum(1 for s in signals if s.get("category") == "largecap"),
        "midcap": sum(1 for s in signals if s.get("category") == "midcap"),
        "smallcap": sum(1 for s in signals if s.get("category") == "smallcap"),
        "ipo": sum(1 for s in signals if s.get("category") == "ipo"),
    }

# ── NEW: /api/price/<ticker> (Required by frontend live trades) ──────────────
@app.route("/api/price/<ticker>", methods=["GET"])
def api_price(ticker):
    try:
        provider = get_default_provider()
        df = provider.get_ohlc(ticker, period="5d", interval="1d")
        if df is not None and not df.empty:
            return jsonify({"price": float(df['close'].iloc[-1])})
    except Exception:
        pass
    return jsonify({"error": "Not found"}), 404

# ── NEW: /api/refresh (Required by frontend refresh button) ──────────────────
@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    worker = Thread(target=_refresh_signals, daemon=True)
    worker.start()
    return jsonify({"status": "Refresh started in background"})


# ── FIXED: /api/signals ──────────────────────────────────────────────────────
@app.route("/api/signals", methods=["GET"])
def api_signals():
    payload = _get_signals_cached()
    signals_list: List[Dict] = payload["signals"]
    
    # Check in-memory signals
    normalized = [_normalize_signal(item) for item in signals_list if isinstance(item, dict)]
    if normalized:
        return jsonify({
            "signals": normalized,
            "counts": _calculate_counts(normalized),
            "from_cache": False,
            "last_updated": payload.get("generated_at"),
        })

    # Check last non-empty memory
    last_nonempty: List[Dict] = payload.get("last_nonempty_signals") or []
    normalized_last = [_normalize_signal(item) for item in last_nonempty if isinstance(item, dict)]
    if normalized_last:
        return jsonify({
            "signals": normalized_last,
            "counts": _calculate_counts(normalized_last),
            "from_cache": True,
            "last_updated": payload.get("last_nonempty_generated_at"),
        })

    # Check disk cache
    disk = load_signals_cache()
    if disk and disk.get("signals"):
        disk_signals = [_normalize_signal(s) for s in disk["signals"] if isinstance(s, dict)]
        if disk_signals:
            return jsonify({
                "signals": disk_signals,
                "counts": _calculate_counts(disk_signals),
                "from_cache": True,
                "last_updated": disk.get("last_updated"),
            })

    # Last resort fallback
    fallback = _fallback_from_universe()
    return jsonify({
        "signals": fallback,
        "counts": _calculate_counts(fallback),
        "from_cache": True,
        "last_updated": _utc_now().isoformat(),
    })


@app.route("/health", methods=["GET"])
def health():
    disk_meta = get_cache_meta()
    return jsonify({
        "status": "ok",
        "time": _utc_now().isoformat(),
        "market_open": is_market_open(),
        "cache": disk_meta,
    })

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
