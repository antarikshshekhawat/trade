from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait
from io import StringIO
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

import pandas as pd
import pandas_ta as ta

from data import MarketDataProvider
from cache_utils import is_market_open, load_signals_cache, save_signals_cache

NSE_ALL_STOCKS_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

PRIORITY_SYMBOLS = [
    "SHAKTIPUMP", "NSDL", "WAAREEENER", "GMDS", "FIRSTSOURCE",
]


# ---------------- INDICATORS ---------------- #

def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["vol_sma20"] = ta.sma(df["volume"], length=20)
    df["high20_prev"] = df["high"].rolling(20).max().shift(1)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macdh"] = macd.iloc[:, 2] if macd is not None else pd.NA

    return df.dropna().copy()


# ---------------- SYMBOL FETCH ---------------- #

def _fetch_all_nse_symbols() -> List[str]:
    try:
        with urlopen(NSE_ALL_STOCKS_CSV, timeout=10) as response:
            csv_text = response.read().decode("utf-8", errors="ignore")
        frame = pd.read_csv(StringIO(csv_text))
        symbols = frame["SYMBOL"].dropna().astype(str).str.upper().tolist()
        return sorted(set(symbols))
    except Exception:
        print("⚠ NSE fetch failed, using fallback universe")
        return []


def _build_category_map(categorized_stocks: Dict[str, List[str]]) -> Dict[str, str]:
    mp = {}
    for cat, arr in categorized_stocks.items():
        for s in arr:
            mp[s.upper()] = cat
    return mp


# ---------------- SIGNAL BUILD ---------------- #

def _rr(entry, sl, target):
    risk = entry - sl
    reward = target - entry
    if risk <= 0:
        return None, None
    val = reward / risk
    return round(val, 2), f"1:{round(val, 2)}"


def _build_signal(ticker, category, row, pattern):
    entry = float(row["close"])
    atr = float(row["atr14"])

    sl = round(entry - 1.5 * atr, 2)
    target = round(entry + 3 * atr, 2)

    rr, rr_text = _rr(entry, sl, target)

    return {
        "ticker": ticker,
        "category": category,
        "pattern": pattern,
        "entry": round(entry, 2),
        "sl": sl,
        "target": target,
        "rr": rr,
        "rr_text": rr_text,
        "price": round(entry, 2),
    }


def _score(row):
    score = 0
    if row["ema20"] > row["ema50"]:
        score += 2
    if row["macdh"] > 0:
        score += 2
    if 40 <= row["rsi14"] <= 70:
        score += 1
    return score


# ---------------- MAIN LOGIC ---------------- #

def scan_symbol(provider, symbol, category):
    try:
        df = provider.get_ohlc(symbol)
        if df.empty or len(df) < 40:
            return None

        df = _prepare_indicators(df)
        last = df.iloc[-1]

        breakout = last["close"] >= last["high20_prev"] and last["volume"] > 2 * last["vol_sma20"]

        if breakout:
            s = _build_signal(symbol, category, last, "Breakout")
            s["score"] = _score(last) + 2
            return s

        if last["ema20"] > last["ema50"] and last["macdh"] > 0:
            s = _build_signal(symbol, category, last, "Trend")
            s["score"] = _score(last)
            return s

    except Exception:
        return None

    return None


# ---------------- FINAL SCAN ---------------- #

def scan_market(
    provider: MarketDataProvider,
    categorized_stocks: Dict[str, List[str]],
    max_workers: int = 40,
    max_signals: int = 100,
    scan_timeout_sec: int = 30,
) -> List[Dict]:

    # MARKET CLOSED → USE CACHE
    if not is_market_open():
        cache = load_signals_cache()
        if cache:
            return cache.get("signals", [])

    all_symbols = _fetch_all_nse_symbols()

    if not all_symbols:
        for arr in categorized_stocks.values():
            all_symbols.extend(arr)

    all_symbols = list(set(all_symbols + PRIORITY_SYMBOLS))

    category_map = _build_category_map(categorized_stocks)

    signals = []

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        jobs = [exe.submit(scan_symbol, provider, s, category_map.get(s, "smallcap")) for s in all_symbols]

        done, _ = wait(jobs, timeout=scan_timeout_sec)

        for f in done:
            try:
                r = f.result()
                if r:
                    signals.append(r)
            except:
                pass

    # SORT
    signals.sort(key=lambda x: x.get("score", 0), reverse=True)

    # ---------------- BALANCED OUTPUT ---------------- #

    large = [s for s in signals if s["category"] == "largecap"]
    mid = [s for s in signals if s["category"] == "midcap"]
    small = [s for s in signals if s["category"] == "smallcap"]
    ipo = [s for s in signals if s["category"] == "ipo"]

    balanced = (
        large[:25] +
        mid[:25] +
        small[:25] +
        ipo[:25]
    )

    # fallback fill
    if len(balanced) < max_signals:
        rest = [s for s in signals if s not in balanced]
        balanced += rest[: max_signals - len(balanced)]

    final = balanced[:max_signals]

    # SAVE CACHE
    if final:
        save_signals_cache(final)

    return final
