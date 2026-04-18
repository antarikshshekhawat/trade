from __future__ import annotations

import threading
import time
import logging
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from data import MarketDataProvider
from strategy import scan_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── DATA PROVIDER (shared across threads) ────────────────────────────────────
provider = MarketDataProvider(cache_size=400)

# ── STOCK UNIVERSE ────────────────────────────────────────────────────────────
# Large-cap: Nifty 50 & Nifty Next 50 selections
LARGE_CAP = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "SBIN", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "LT",
    "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA",
    "WIPRO", "ULTRACEMCO", "M&M", "NTPC", "POWERGRID",
    "TECHM", "HCLTECH", "BAJAJ-AUTO", "EICHERMOT", "INDUSINDBK",
    "JSWSTEEL", "TATASTEEL", "COALINDIA", "APOLLOHOSP", "NESTLEIND",
    "HINDUNILVR", "DIVISLAB", "CIPLA", "DRREDDY", "ONGC",
    "IOC", "BPCL", "TATACONSUM", "BRITANNIA", "HEROMOTOCO",
    "GRASIM", "SHREECEM", "ADANIENT", "ADANIPORTS", "TATAPOWER",
    "NHPC", "RECLTD", "PFC", "HAL", "BEL",
]

# Mid-cap: Nifty Midcap 150 selections
MID_CAP = [
    "ABCAPITAL", "ASHOKLEY", "ASTRAL", "AUBANK", "BALKRISIND",
    "BATAINDIA", "BERGEPAINT", "BHEL", "CAMS", "CHOLAFIN",
    "COFORGE", "CONCOR", "CROMPTON", "DEEPAKNTR", "DIXON",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "FORTIS", "GODREJPROP",
    "GRANULES", "HAL", "HONASA", "HUDCO", "IDFCFIRSTB",
    "IEXINDIA", "INDHOTEL", "IRFC", "KALYANKJIL", "KPIL",
    "LAURUSLABS", "LTFH", "LUPIN", "MAXHEALTH", "METROPOLIS",
    "MPHASIS", "MUTHOOTFIN", "NCC", "OBEROIRLTY", "PAGEIND",
    "PATANJALI", "PETRONET", "PIIND", "POLYCAB", "PRESTIGE",
    "TRENT", "VARUNBEV", "VOLTAS", "ZYDUSLIFE", "NAUKRI",
]

# Small-cap selections
SMALL_CAP = [
    "SHAKTIPUMP", "FIRSTSOURCE", "GMDS", "WAAREEENER",
    "APLAPOLLO", "APTUS", "BIKAJI", "BLUESTARCO", "CHAMBLFERT",
    "CRAFTSMAN", "CUMMINSIND", "ELGIEQUIP", "ENGINERSIN",
    "GLAXO", "GNFC", "GSFC", "HFCL", "INOXWIND",
    "ITI", "JYOTHYLAB", "KANSAINER", "KAYNES", "LICI",
    "LINDEINDIA", "NATIONALUM", "NIACL", "NLCINDIA",
    "NUVAMA", "OIL", "PCBL", "PHOENIXLTD", "PRINCEPIPES",
    "RHIM", "SAFARI", "SAILONG", "SOLARINDS", "SUPPETRO",
    "TATACHEM", "THERMAX", "TIINDIA", "TRITURBINE", "ZENSARTECH",
]

# Recent IPOs (update periodically)
IPO = [
    "NSDL", "NYKAA", "PAYTM", "ZOMATO", "POLICYBZR",
    "DELHIVERY", "MAPMYINDIA", "IDEAFORGE", "NETWEB",
]

CATEGORIZED_STOCKS = {
    "largecap": LARGE_CAP,
    "midcap":   MID_CAP,
    "smallcap": SMALL_CAP,
    "ipo":      IPO,
}

# ── SIGNAL CACHE (in-memory) ─────────────────────────────────────────────────
_scan_lock    = threading.Lock()
_cached_signals: list = []
_last_scan_ts: float  = 0.0
SCAN_TTL_SEC          = 300          # re-scan every 5 min during market hours


def _do_scan() -> list:
    global _cached_signals, _last_scan_ts
    try:
        results = scan_market(
            provider=provider,
            categorized_stocks=CATEGORIZED_STOCKS,
            max_workers=10,
            max_signals=60,
            scan_timeout_sec=45,
        )
        with _scan_lock:
            _cached_signals = results
            _last_scan_ts   = time.time()
        return results
    except Exception as e:
        logger.error("Scan failed: %s", e)
        return _cached_signals


def _get_signals(force: bool = False) -> list:
    """Return cached signals or trigger a fresh scan."""
    age = time.time() - _last_scan_ts
    if force or not _cached_signals or age > SCAN_TTL_SEC:
        return _do_scan()
    return _cached_signals


# ── BACKGROUND SCANNER ────────────────────────────────────────────────────────

def _background_scanner():
    """Periodically refreshes signals in the background."""
    time.sleep(5)                    # give app a moment to start
    while True:
        try:
            _do_scan()
        except Exception as e:
            logger.error("Background scan error: %s", e)
        time.sleep(SCAN_TTL_SEC)


_bg_thread = threading.Thread(target=_background_scanner, daemon=True)
_bg_thread.start()


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signals")
def api_signals():
    category = request.args.get("category", "all").lower()
    signals  = _get_signals()

    if category != "all":
        signals = [s for s in signals if s.get("category", "").lower() == category]

    # Build category counts for the header bar
    counts: Dict[str, int] = {"all": len(_cached_signals)}
    for cat_key in ("largecap", "midcap", "smallcap", "ipo"):
        counts[cat_key] = sum(
            1 for s in _cached_signals if s.get("category", "").lower() == cat_key
        )

    from_cache     = any(s.get("_from_cache") for s in signals)
    last_updated   = next(
        (s.get("_cache_last_updated", "") for s in signals if s.get("_from_cache")),
        "",
    )

    return jsonify({
        "signals":      signals,
        "counts":       counts,
        "from_cache":   from_cache,
        "last_updated": last_updated,
        "total":        len(signals),
    })


@app.route("/api/price/<symbol>")
def api_price(symbol: str):
    """Return latest price for a single symbol (for trade P&L updates)."""
    price = provider.get_current_price(symbol.upper())
    if price is None:
        return jsonify({"error": "price unavailable"}), 404
    return jsonify({"symbol": symbol.upper(), "price": price})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Force a fresh market scan."""
    provider.clear_cache()
    results = _do_scan()
    return jsonify({"ok": True, "count": len(results)})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
