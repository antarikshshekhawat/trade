from __future__ import annotations



from datetime import datetime, timezone

from threading import Lock, Thread

from typing import Dict, List



from flask import Flask, jsonify, request, send_from_directory



from data import (
    FALLBACK_IPO_STOCKS,
    FALLBACK_UNIVERSE,
    build_stock_universe,
    fetch_last_prices_nse,
    get_default_provider,
)
from sector_indices import get_sector_performance

from strategy import scan_market



# ── NEW: cache helpers ────────────────────────────────────────────────────────

from cache_utils import get_cache_meta, is_market_open, load_signals_cache

# ─────────────────────────────────────────────────────────────────────────────



app = Flask(__name__)



@app.route("/")

def home():

    return send_from_directory(".", "nse_alpha_scanner.html")

@app.route("/scan")

def scan():

    universe = _get_universe_cached()

    provider = get_default_provider()

    signals = scan_market(

        provider=provider,

        categorized_stocks=universe,

        max_workers=10,

        max_signals=30,

        scan_timeout_sec=120,

        max_symbols_to_scan=120,

    )

    return jsonify(signals)



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



CACHE_SECONDS = 12

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





def _utc_now() -> datetime:

    return datetime.now(timezone.utc)


_sector_lock = Lock()

_sector_cache: Dict[str, object] = {"payload": None, "expires_at": 0.0}

SECTOR_CACHE_SECONDS = 35.0


def _get_sectors_payload() -> Dict[str, object]:

    now_ts = _utc_now().timestamp()

    with _sector_lock:

        if _sector_cache["payload"] is not None and now_ts < float(_sector_cache["expires_at"]):

            return _sector_cache["payload"]  # type: ignore[return-value]

    payload = get_sector_performance()

    with _sector_lock:

        _sector_cache["payload"] = payload

        _sector_cache["expires_at"] = _utc_now().timestamp() + SECTOR_CACHE_SECONDS

    return payload


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

            max_workers=10,

            max_signals=30,

            scan_timeout_sec=120,

            max_symbols_to_scan=120,

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

    target_pct = float(

        item.get("target_pct") if item.get("target_pct") is not None else round((target / entry - 1) * 100, 2)

    )



    pattern = str(item.get("pattern") or "MACD + RSI")



    # ✅ ADD THIS LINE

    last_close = float(item.get("last_close") or item.get("prev_close") or entry)



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



        # ✅ ADD THIS FIELD

        "last_close": round(last_close, 2),

    }





def _fallback_from_universe() -> List[Dict]:

    universe = _get_universe_cached()

    rows: List[Dict] = []

    for category, symbols in universe.items():

        for symbol in symbols[:5]:

            rows.append(

                _normalize_signal(

                    {

                        "ticker": symbol,

                        "category": category,

                        "price": 100.0,

                        "entry": 100.0,

                        "target": 106.0,

                        "sl": 97.0,

                        "rr": 2.0,

                        "rr_text": "1:2.0",

                        "pattern": "Top Mover (fallback)",

                        "sl_pct": -3.0,

                        "target_pct": 6.0,

                    }

                )

            )

    return rows[:20]





def _get_universe_cached() -> Dict[str, List[str]]:

    now_ts = _utc_now().timestamp()

    with _cache_lock:

        if now_ts < float(_universe_cache["expires_at"]):

            return _universe_cache["stocks"]  # type: ignore[return-value]



    stocks = build_stock_universe()

    with _cache_lock:

        _universe_cache["stocks"] = stocks

        _universe_cache["expires_at"] = _utc_now().timestamp() + UNIVERSE_CACHE_SECONDS

    return stocks





# ─────────────────────────────────────────────────────────────────────────────

# Helper: build cache-status metadata block for every /signals response

# ─────────────────────────────────────────────────────────────────────────────

def _build_cache_status(signals_list: List[Dict]) -> Dict:

    """

    Returns a metadata dict that tells the frontend:

      - whether the data is live or from disk cache

      - when the signals were last saved to disk

      - whether the market is currently open

    """

    market_open = is_market_open()

    from_cache = any(bool(s.get("_from_cache")) for s in signals_list)



    last_updated = None

    for sig in signals_list:

        ts = sig.get("_cache_last_updated")

        if ts:

            last_updated = ts

            break

    if last_updated is None:

        disk_meta = get_cache_meta()

        last_updated = disk_meta.get("last_updated")



    return {

        "market_open": market_open,

        "from_cache": from_cache,

        "last_updated": last_updated,

        "status_label": (

            "Live signals"

            if (market_open and not from_cache)

            else "Using last market signals"

        ),

    }





@app.route("/stocks", methods=["GET"])

def stocks():

    universe = _get_universe_cached()

    counts = {category: len(items) for category, items in universe.items()}

    return jsonify(

        {

            "stocks": universe,

            "counts": counts,

            "generated_at": _utc_now().isoformat(),

        }

    )





# ─────────────────────────────────────────────────────────────────────────────

# UPDATED /signals endpoint — now wraps signals in an envelope with

# cache_status so the frontend can show "Using last market signals" badge

# ─────────────────────────────────────────────────────────────────────────────

def _signals_handler():

    payload = _get_signals_cached()

    signals_list: List[Dict] = payload["signals"]  # type: ignore[assignment]



    # ── Priority 1: live in-memory signals ───────────────────────────────────

    normalized = [_normalize_signal(item) for item in signals_list if isinstance(item, dict)]

    if normalized:

        cache_status = _build_cache_status(signals_list)

        return jsonify({

            "signals": normalized,

            "cache_status": cache_status,

            "generated_at": payload.get("generated_at"),

        })



    # ── Priority 2: last non-empty in-memory signals ─────────────────────────

    last_nonempty: List[Dict] = payload.get("last_nonempty_signals") or []  # type: ignore[assignment]

    normalized_last = [_normalize_signal(item) for item in last_nonempty if isinstance(item, dict)]

    if normalized_last:

        cache_status = _build_cache_status(last_nonempty)

        cache_status["from_cache"] = True

        cache_status["status_label"] = "Using last market signals"

        return jsonify({

            "signals": normalized_last,

            "cache_status": cache_status,

            "generated_at": payload.get("last_nonempty_generated_at"),

        })



    # ── Priority 3: persistent disk cache (signals_cache.json) ───────────────

    disk = load_signals_cache()

    if disk and disk.get("signals"):

        disk_signals = [_normalize_signal(s) for s in disk["signals"] if isinstance(s, dict)]

        if disk_signals:

            return jsonify({

                "signals": disk_signals,

                "cache_status": {

                    "market_open": is_market_open(),

                    "from_cache": True,

                    "last_updated": disk.get("last_updated"),

                    "status_label": "Using last market signals",

                },

                "generated_at": disk.get("last_updated"),

            })



    # ── Priority 4: universe-based fallback (last resort) ────────────────────

    fallback = _fallback_from_universe()

    return jsonify({

        "signals": fallback,

        "cache_status": {

            "market_open": is_market_open(),

            "from_cache": True,

            "last_updated": None,

            "status_label": "Using last market signals",

        },

        "generated_at": _utc_now().isoformat(),

    })


@app.route("/signals", methods=["GET"])
@app.route("/api/signals", methods=["GET"])
def signals():
    return _signals_handler()


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    worker = Thread(target=_refresh_signals, daemon=True)
    worker.start()
    return jsonify({"ok": True})


@app.route("/sectors", methods=["GET"])
@app.route("/api/sectors", methods=["GET"])
def api_sectors():
    return jsonify(_get_sectors_payload())


@app.route("/api/quotes", methods=["GET"])
def api_quotes():
    raw = request.args.get("tickers", "") or ""
    parts = [p.strip().upper().replace(".NS", "") for p in raw.split(",") if p.strip()]
    prices = fetch_last_prices_nse(parts)
    return jsonify({"prices": prices, "generated_at": _utc_now().isoformat()})


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
