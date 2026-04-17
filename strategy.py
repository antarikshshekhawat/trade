from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, wait
from io import StringIO
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

from data import MarketDataProvider

# ── PERSISTENCE HELPERS ──────────────────────────────────────────────────────
from cache_utils import (
    is_market_open,
    load_signals_cache,
    save_signals_cache,
)

NSE_ALL_STOCKS_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
PRIORITY_SYMBOLS = [
    "SHAKTIPUMP",
    "NSDL",
    "WAAREEENER",
    "GMDS",
    "FIRSTSOURCE",
]

# ── CORE UTILITIES ───────────────────────────────────────────────────────────

def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and cleans data."""
    if frame.empty:
        return frame
    
    df = frame.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["vol_sma20"] = ta.sma(df["volume"], length=20)
    
    df["high20_prev"] = df["high"].rolling(window=20).max().shift(1)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        hist_col = [col for col in macd.columns if "MACDh" in col]
        df["macdh"] = macd[hist_col[0]] if hist_col else pd.NA
    else:
        df["macdh"] = pd.NA
        
    return df.dropna().copy()


def _score_candidate(row: pd.Series) -> float:
    """Improved scoring function with base score and technical weights."""
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    macdh = float(row["macdh"])
    rsi14 = float(row["rsi14"])
    
    score = 50.0

    if ema20 > ema50:
        score += 15
    if macdh > 0:
        score += 15
    if 45 <= rsi14 <= 65:
        score += 10

    score += min(max(macdh * 50, 0), 10)

    return round(score, 2)


def _rr_format(entry: float, sl: float, target: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculates Risk/Reward ratio and formats as string."""
    risk = abs(entry - sl)
    reward = abs(target - entry)
    if risk <= 0 or reward <= 0:
        return 1.0, "1:1"
    rr = reward / risk
    return round(rr, 2), f"1:{round(rr, 2)}"


def _build_signal(ticker: str, category: str, row: pd.Series, pattern: str) -> Dict:
    """Constructs a signal dictionary from technical data."""
    entry = float(row["close"])
    atr = float(row["atr14"])
    sl = round(entry - 1.5 * atr, 2)
    target = round(entry + 3.0 * atr, 2)
    rr, rr_text = _rr_format(entry, sl, target)
    sl_pct = round(((sl / entry) - 1) * 100, 2)
    target_pct = round(((target / entry) - 1) * 100, 2)

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


def _mover_score(df: pd.DataFrame) -> Optional[float]:
    """Calculates 5-day price movement percentage."""
    if df.empty or len(df) < 8:
        return None
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-6])
    return round((last / prev - 1) * 100, 3) if prev > 0 else None

# ── SCANNING LOGIC ───────────────────────────────────────────────────────────

def scan_symbol(provider: MarketDataProvider, symbol: str, category: str) -> Optional[Dict]:
    """Analyzes symbol and returns signal or safe fallback for non-signals."""
    try:
        frame = provider.get_ohlc(symbol=symbol, period="4mo", interval="1d")
        if frame is None or frame.empty or len(frame) < 35:
            return None

        df = _prepare_indicators(frame)
        if df.empty or len(df) < 3:
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
        
        # 1. Momentum Breakout
        if last_close >= high20_prev and vol_sma20 > 0 and last_volume >= 2.0 * vol_sma20:
            breakout = _build_signal(symbol, category, last_row, "Momentum Breakout")
            breakout["candidate_score"] = float(_score_candidate(last_row)) + 20.0
            breakout["is_candidate"] = True
            breakout["mover_5d_pct"] = float(_mover_score(frame) or 0.0)
            return breakout

        # 2. MACD Reversal
        if macd_cross_up and rsi_ok and ema_trend_ok:
            return _build_signal(symbol, category, last_row, "MACD Cross + RSI Zone + EMA Trend")

        # 3. Trend Continuation Candidate
        if ema_trend_ok and last_macdh > 0 and 38 <= last_rsi <= 72:
            candidate = _build_signal(symbol, category, last_row, "Momentum Trend Candidate")
            candidate["candidate_score"] = float(_score_candidate(last_row))
            candidate["is_candidate"] = True
            mover = _mover_score(frame)
            if mover is not None:
                candidate["mover_5d_pct"] = float(mover)
            return candidate

        # --- SAFE FALLBACK FOR ALL OTHER STOCKS ---
        try:
            current_close = float(last_row["close"])
            current_score = float(_score_candidate(last_row))
        except:
            return None

        return {
            "ticker": symbol,
            "category": category,
            "pattern": "No Strong Signal",
            "price": round(current_close, 2),
            "entry": round(current_close, 2),
            "target": round(current_close * 1.02, 2),
            "sl": round(current_close * 0.98, 2),
            "rr": 1.0,
            "rr_text": "1:1",
            "sl_pct": -2.0,
            "target_pct": 2.0,
            "candidate_score": current_score,
            "is_candidate": False
        }
            
    except Exception:
        return None


def scan_market(
    provider: MarketDataProvider,
    categorized_stocks: Dict[str, List[str]],
    max_workers: int = 5,
    max_signals: int = 30,
    scan_timeout_sec: int = 25,
) -> List[Dict]:
    """Orchestrates market-wide scanning returning all stocks ranked by score."""
    
    if not is_market_open():
        cached = load_signals_cache()
        if cached and cached.get("signals"):
            for sig in cached["signals"]:
                sig["_from_cache"] = True
                sig["_cache_last_updated"] = cached.get("last_updated", "")
            return cached["signals"]

    symbol_tasks = {}
    for category, stocks in categorized_stocks.items():
        for s in stocks:
            clean_s = str(s).strip().upper().replace(".NS", "")
            if clean_s: symbol_tasks[clean_s] = category

    for s in PRIORITY_SYMBOLS:
        clean_s = str(s).strip().upper().replace(".NS", "")
        if clean_s not in symbol_tasks:
            symbol_tasks[clean_s] = "smallcap"

    if not symbol_tasks:
        return []

    ipo_set = set(str(s).strip().upper().replace(".NS", "") for s in categorized_stocks.get("ipo", []))
    signals = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(scan_symbol, provider, s, cat): s 
            for s, cat in symbol_tasks.items()
        }
        done, pending = wait(future_to_symbol.keys(), timeout=scan_timeout_sec)
        for f in pending: f.cancel()
        for f in done:
            try:
                res = f.result()
                if res: signals.append(res)
            except: continue

    # Rank: Candidates first, then by descending score
    signals.sort(
        key=lambda x: (int(x.get("is_candidate", False)), float(x.get("candidate_score", 0))),
        reverse=True,
    )

    for item in signals:
        if item["ticker"] in ipo_set:
            item["category"] = "ipo"

    top_signals = signals[:100]

    if top_signals:
        save_signals_cache(top_signals)
        return top_signals

    return []
