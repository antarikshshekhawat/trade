from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait
from io import StringIO
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

import pandas as pd
import pandas_ta as ta

from data import MarketDataProvider

# ── NEW: import persistence helpers ──────────────────────────────────────────
from cache_utils import (
    is_market_open,
    load_signals_cache,
    save_signals_cache,
)
# ─────────────────────────────────────────────────────────────────────────────

NSE_ALL_STOCKS_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
PRIORITY_SYMBOLS = [
    "SHAKTIPUMP",
    "NSDL",
    "WAAREEENER",
    "GMDS",
    "FIRSTSOURCE",
]


def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["vol_sma20"] = ta.sma(df["volume"], length=20)
    # Previous 20-day high (exclude current candle to detect breakout)
    df["high20_prev"] = df["high"].rolling(20).max().shift(1)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is None or macd.empty:
        df["macdh"] = pd.NA
    else:
        hist_col = [col for col in macd.columns if "MACDh" in col]
        if hist_col:
            df["macdh"] = macd[hist_col[0]]
        else:
            df["macdh"] = pd.NA
    return df.dropna().copy()


def _fetch_all_nse_symbols() -> List[str]:
    try:
        with urlopen(NSE_ALL_STOCKS_CSV, timeout=10) as response:
            csv_text = response.read().decode("utf-8", errors="ignore")
        frame = pd.read_csv(StringIO(csv_text))
        if "SYMBOL" not in frame.columns:
            return []
        symbols = (
            frame["SYMBOL"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".NS", "", regex=False)
            .tolist()
        )
        return sorted(set([s for s in symbols if s]))
    except Exception:
        return []


def _build_category_map(categorized_stocks: Dict[str, List[str]]) -> Dict[str, str]:
    category_map: Dict[str, str] = {}
    for category, symbols in categorized_stocks.items():
        for symbol in symbols:
            clean = str(symbol).strip().upper().replace(".NS", "")
            if clean:
                category_map[clean] = category
    return category_map


def _rr_format(entry: float, sl: float, target: float) -> Tuple[Optional[float], Optional[str]]:
    risk = entry - sl
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None, None
    rr = reward / risk
    return round(rr, 2), f"1:{round(rr, 2)}"


def _build_signal(ticker: str, category: str, row: pd.Series, pattern: str) -> Dict:
    entry = float(row["close"])
    atr = float(row["atr14"])
    sl = round(entry - 1.5 * atr, 2)
    target = round(entry + 3.0 * atr, 2)
    rr, rr_text = _rr_format(entry, sl, target)
    sl_pct = round((sl / entry - 1) * 100, 2)
    target_pct = round((target / entry - 1) * 100, 2)

    return {
        "ticker": ticker,
        "category": category,
        "pattern": pattern,
        "entry": round(entry, 2),
        "sl": sl,
        "sl_pct": sl_pct,
        "target": target,
        "target_pct": target_pct,
        "upside_pct": round(target_pct, 2),
        "rr": rr,
        "rr_text": rr_text,
        "price": round(entry, 2),
        "rsi": round(float(row["rsi14"]), 2),
        "ema20": round(float(row["ema20"]), 2),
        "ema50": round(float(row["ema50"]), 2),
        "atr14": round(float(row["atr14"]), 2),
        "macd_hist": round(float(row["macdh"]), 4),
    }


def _score_candidate(row: pd.Series) -> float:
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    macdh = float(row["macdh"])
    rsi14 = float(row["rsi14"])
    close = float(row["close"])

    score = 0.0
    if ema20 > ema50:
        score += 1.5
    if macdh > 0:
        score += 1.5
    if 40 <= rsi14 <= 70:
        score += 1.0

    score += min(max(macdh * 10, 0), 2)
    ema_gap = ema20 - ema50
    score += min(max(ema_gap / max(close, 1) * 100, 0), 2)
    return round(score, 4)


def _mover_score(df: pd.DataFrame) -> Optional[float]:
    if df.empty or len(df) < 8:
        return None
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-6])
    if prev <= 0:
        return None
    return round((last / prev - 1) * 100, 3)


def scan_symbol(provider: MarketDataProvider, symbol: str, category: str) -> Optional[Dict]:
    try:
        frame = provider.get_ohlc(symbol=symbol, period="8mo", interval="1d")
        if frame.empty or len(frame) < 35:
            return None

        df = _prepare_indicators(frame)
        if len(df) < 3:
            return None

        recent = df.iloc[-4:]
        prev_row = df.iloc[-2]
        last_row = df.iloc[-1]

        prev_macdh = float(prev_row["macdh"])
        last_macdh = float(last_row["macdh"])
        min_recent_macdh = float(recent["macdh"].astype(float).min())
        macd_cross_up = (prev_macdh < 0 and last_macdh > 0) or (
            min_recent_macdh < 0 and last_macdh > 0
        )
        last_rsi = float(last_row["rsi14"])
        ema20 = float(last_row["ema20"])
        ema50 = float(last_row["ema50"])
        rsi_ok = 35 <= last_rsi <= 80
        ema_trend_ok = ema20 > ema50

        last_close = float(last_row["close"])
        high20_prev = float(last_row["high20_prev"])
        last_volume = float(last_row["volume"])
        vol_sma20 = float(last_row["vol_sma20"])
        breakout_ok = last_close >= high20_prev and vol_sma20 > 0 and last_volume >= 2.0 * vol_sma20

        if breakout_ok:
            breakout = _build_signal(
                ticker=symbol,
                category=category,
                row=last_row,
                pattern="Momentum Breakout",
            )
            breakout["candidate_score"] = float(_score_candidate(last_row)) + 2.0
            breakout["is_candidate"] = True
            breakout["mover_5d_pct"] = float(_mover_score(frame) or 0.0)
            return breakout

        if macd_cross_up and rsi_ok and ema_trend_ok:
            return _build_signal(
                ticker=symbol,
                category=category,
                row=last_row,
                pattern="MACD Cross + RSI Zone + EMA Trend",
            )

        if ema_trend_ok and last_macdh > 0 and 38 <= last_rsi <= 72:
            candidate = _build_signal(
                ticker=symbol,
                category=category,
                row=last_row,
                pattern="Momentum Trend Candidate",
            )
            candidate["candidate_score"] = float(_score_candidate(last_row))
            candidate["is_candidate"] = True
            mover = _mover_score(frame)
            if mover is not None:
                candidate["mover_5d_pct"] = float(mover)
            return candidate
    except Exception:
        return None
    return None


def _build_priority_watch(provider: MarketDataProvider, symbol: str, category: str) -> Optional[Dict]:
    try:
        frame = provider.get_ohlc(symbol=symbol, period="8mo", interval="1d")
        if frame.empty or len(frame) < 35:
            return None
        df = _prepare_indicators(frame)
        if df.empty:
            return None
        row = df.iloc[-1]
        item = _build_signal(
            ticker=symbol,
            category=category,
            row=row,
            pattern="Priority Momentum Watch",
        )
        item["is_candidate"] = True
        item["candidate_score"] = float(_score_candidate(row)) + 1.0
        item["mover_5d_pct"] = float(_mover_score(frame) or 0.0)
        return item
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED scan_market — with persistence + market-awareness
# ─────────────────────────────────────────────────────────────────────────────

def scan_market(
    provider: MarketDataProvider,
    categorized_stocks: Dict[str, List[str]],
    max_workers: int = 20,
    max_signals: int = 30,
    scan_timeout_sec: int = 25,
) -> List[Dict]:
    """
    Scan the market for signals.

    Behaviour
    ---------
    1. If the market is CLOSED  → skip live scan, return persisted cache.
    2. If the market is OPEN    → run the live scan.
         • Success   → save to signals_cache.json, return live signals.
         • Empty/err → load signals_cache.json, return cached signals.
    3. If both live scan AND cache are empty → mover fallback (original).
    4. Absolute last resort     → watchlist skeleton (original).

    Each returned signal dict carries two optional meta keys that app.py
    forwards to the frontend:
        _from_cache          : bool  – True when served from disk cache
        _cache_last_updated  : str   – "YYYY-MM-DD HH:MM" timestamp
    """
    market_open = is_market_open()

    # ── STEP 1: Market closed → serve disk cache immediately ─────────────────
    if not market_open:
        cached = load_signals_cache()
        if cached and cached.get("signals"):
            signals = cached["signals"]
            for sig in signals:
                sig["_from_cache"] = True
                sig["_cache_last_updated"] = cached.get("last_updated", "")
            return signals
        # Cache is missing too – fall through to attempt a live scan anyway
        # (data.py will return the last daily close even outside market hours)

    # ── STEP 2: Attempt live scan ─────────────────────────────────────────────
# ✅ USE ONLY YOUR STRUCTURED UNIVERSE
all_symbols = []

for category, stocks in categorized_stocks.items():
    for s in stocks:
        all_symbols.append((s, category))

# Add priority stocks as smallcap if not present
for s in PRIORITY_SYMBOLS:
    all_symbols.append((s, "smallcap"))

    if not all_symbols:
        cached = load_signals_cache()
        if cached and cached.get("signals"):
            for sig in cached["signals"]:
                sig["_from_cache"] = True
                sig["_cache_last_updated"] = cached.get("last_updated", "")
            return cached["signals"]
        return []

    category_map = _build_category_map(categorized_stocks)
    ipo_set = set(
        str(symbol).strip().upper().replace(".NS", "")
        for symbol in categorized_stocks.get("ipo", [])
    )

    signals: List[Dict] = []
    movers: List[Tuple[float, Dict]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
jobs = [exe.submit(scan_symbol, provider, s, cat) for (s, cat) in all_symbols]
        ]
        done_jobs = jobs
        try:
            done_jobs, pending_jobs = wait(jobs, timeout=scan_timeout_sec)
            for future in pending_jobs:
                future.cancel()
        except TimeoutError:
            done_jobs = []

        for future in done_jobs:
            try:
                result = future.result()
                if result:
                    signals.append(result)
            except Exception:
                continue

    signals.sort(
        key=lambda item: (
            int(not item.get("is_candidate", False)),
            float(item.get("candidate_score", float(item.get("macd_hist", 0.0)))),
            float(item.get("price", 0.0)),
        ),
        reverse=True,
    )
    for item in signals:
        ticker = str(item.get("ticker", "")).upper()
        if ticker in ipo_set:
            item["category"] = "ipo"

    top = signals[:max_signals]

    if top:
        existing = {str(item.get("ticker", "")).upper() for item in top}
        for symbol in PRIORITY_SYMBOLS:
            if len(top) >= max_signals:
                break
            if symbol in existing:
                continue
            watch = _build_priority_watch(
                provider,
                symbol,
                "ipo" if symbol in ipo_set else category_map.get(symbol, "smallcap"),
            )
            if watch:
                top.append(watch)
                existing.add(symbol)
        top = top[:max_signals]

        # ── STEP 3: Persist fresh signals ────────────────────────────────────
        save_signals_cache(top)
        return top

    # ── STEP 4: Live scan empty → load disk cache ─────────────────────────────
    cached = load_signals_cache()
    if cached and cached.get("signals"):
        cached_signals = cached["signals"]
        for sig in cached_signals:
            sig["_from_cache"] = True
            sig["_cache_last_updated"] = cached.get("last_updated", "")
        return cached_signals

    # ── STEP 5: Mover fallback (original logic, unchanged) ───────────────────
    for symbol in all_symbols:
        try:
            frame = provider.get_ohlc(symbol=symbol, period="2mo", interval="1d")
            score = _mover_score(frame)
            if score is None:
                continue
            df = _prepare_indicators(frame)
            if df.empty:
                continue
            last_row = df.iloc[-1]
            category = category_map.get(symbol, "smallcap")
            if symbol in ipo_set:
                category = "ipo"
            item = _build_signal(
                ticker=symbol,
                category=category,
                row=last_row,
                pattern="Top Mover (fallback)",
            )
            item["mover_5d_pct"] = float(score)
            movers.append((float(score), item))
        except Exception:
            continue

    movers.sort(key=lambda x: float(x[0]), reverse=True)
    fallback = [item for _, item in movers[:max_signals]]
    if fallback:
        # Persist movers as well so next off-hours request has something useful
        save_signals_cache(fallback)
        return fallback

    # ── STEP 6: Absolute last resort – watchlist skeleton (original) ─────────
    ordered_symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s not in PRIORITY_SYMBOLS]
    sliced = ordered_symbols[: min(20, len(ordered_symbols))]
    return [
        {
            "ticker": symbol,
            "category": "ipo" if symbol in ipo_set else category_map.get(symbol, "smallcap"),
            "pattern": "Watchlist (data unavailable)",
            "price": 100.0,
            "entry": 100.0,
            "target": 106.0,
            "sl": 97.0,
            "rr": 2.0,
            "rr_text": "1:2.0",
            "sl_pct": -3.0,
            "target_pct": 6.0,
        }
        for symbol in sliced
    ]
