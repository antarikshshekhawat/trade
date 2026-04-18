from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Tuple

from data import MarketDataProvider
from cache_utils import is_market_open, load_signals_cache, save_signals_cache

# ── CONFIG ───────────────────────────────────────────────────────────────────
PRIORITY_SYMBOLS = [
    "SHAKTIPUMP", "NSDL", "WAAREEENER", "GMDS", "FIRSTSOURCE",
]

# Minimum requirements for a signal to be returned
MIN_RR            = 2.0      # risk-reward ratio floor
MIN_RSI           = 35
MAX_RSI           = 78
MIN_DATA_BARS     = 55       # need at least this many clean bars
SL_ATR_MULT       = 1.5      # stop-loss = entry - SL_ATR_MULT * ATR
TARGET_ATR_MULT   = 3.0      # target   = entry + TARGET_ATR_MULT * ATR
MIN_PRICE         = 20.0     # filter out penny stocks
MIN_AVG_VOLUME    = 50_000   # avoid illiquid names


# ── INDICATOR ENGINE ─────────────────────────────────────────────────────────

def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates all technical indicators on a clean copy of OHLCV data."""
    if frame.empty:
        return frame

    df = frame.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    # ── Trend
    df["ema9"]  = ta.ema(df["close"], length=9)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)

    # ── Momentum
    df["rsi14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        hist_col = [c for c in macd.columns if "MACDh" in c]
        df["macdh"] = macd[hist_col[0]] if hist_col else pd.NA
    else:
        df["macdh"] = pd.NA

    # ── Volatility / Risk
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        bb_u = [c for c in cols if "BBU" in c]
        bb_l = [c for c in cols if "BBL" in c]
        bb_m = [c for c in cols if "BBM" in c]
        if bb_u: df["bb_upper"] = bb[bb_u[0]]
        if bb_l: df["bb_lower"] = bb[bb_l[0]]
        if bb_m: df["bb_mid"]   = bb[bb_m[0]]
    else:
        df["bb_upper"] = pd.NA
        df["bb_lower"] = pd.NA
        df["bb_mid"]   = pd.NA

    # Supertrend (10,3)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    if st is not None and not st.empty:
        st_dir = [c for c in st.columns if "SUPERTd" in c]
        df["st_uptrend"] = st[st_dir[0]].map(lambda x: x == 1) if st_dir else False
    else:
        df["st_uptrend"] = False

    # ── Volume
    df["vol_sma20"]   = ta.sma(df["volume"], length=20)
    df["high20_prev"] = df["high"].rolling(window=20).max().shift(1)
    df["low10_prev"]  = df["low"].rolling(window=10).min().shift(1)

    return df.dropna(subset=["ema20", "ema50", "rsi14", "atr14", "macdh"]).copy()


# ── SCORING ──────────────────────────────────────────────────────────────────

def _score_candidate(row: pd.Series) -> float:
    """
    Quality score 0-100.
    Higher = stronger technical setup.
    """
    ema20  = float(row["ema20"])
    ema50  = float(row["ema50"])
    macdh  = float(row["macdh"])
    rsi14  = float(row["rsi14"])
    score  = 40.0

    # Trend alignment
    if ema20 > ema50:
        score += 15
    try:
        if bool(row.get("st_uptrend", False)):
            score += 10
    except Exception:
        pass

    # Momentum
    if macdh > 0:
        score += 12
    score += min(max(macdh * 40, 0), 8)   # MACD histogram strength bonus

    # RSI sweet spot
    if 45 <= rsi14 <= 65:
        score += 10
    elif 38 <= rsi14 <= 72:
        score += 5

    # Volume bonus (set by caller)
    return round(min(score, 100.0), 2)


# ── SIGNAL BUILDER ───────────────────────────────────────────────────────────

def _rr_format(entry: float, sl: float, target: float) -> Tuple[float, str]:
    risk   = abs(entry - sl)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0, "N/A"
    rr = round(reward / risk, 2)
    return rr, f"1:{rr}"


def _build_signal(
    ticker: str,
    category: str,
    row: pd.Series,
    pattern: str,
    sl_mult: float = SL_ATR_MULT,
    tgt_mult: float = TARGET_ATR_MULT,
) -> Dict:
    entry  = float(row["close"])
    atr    = float(row["atr14"])
    sl     = round(entry - sl_mult * atr, 2)
    target = round(entry + tgt_mult * atr, 2)
    rr, rr_text = _rr_format(entry, sl, target)
    sl_pct     = round(((sl / entry) - 1) * 100, 2)
    target_pct = round(((target / entry) - 1) * 100, 2)

    return {
        "ticker":      ticker,
        "category":    category,
        "pattern":     pattern,
        "entry":       round(entry, 2),
        "price":       round(entry, 2),
        "sl":          sl,
        "sl_pct":      sl_pct,
        "target":      target,
        "target_pct":  target_pct,
        "upside_pct":  round(target_pct, 2),
        "rr":          rr,
        "rr_text":     rr_text,
        "rsi":         round(float(row["rsi14"]), 2),
        "ema20":       round(float(row["ema20"]), 2),
        "ema50":       round(float(row["ema50"]), 2),
        "atr14":       round(float(row["atr14"]), 2),
        "macd_hist":   round(float(row["macdh"]), 4),
        "is_candidate": False,
        "candidate_score": 0.0,
        "mover_5d_pct": 0.0,
    }


def _mover_score(df: pd.DataFrame) -> Optional[float]:
    """5-session price change %."""
    if df.empty or len(df) < 8:
        return None
    close = df["close"].astype(float)
    last, prev = float(close.iloc[-1]), float(close.iloc[-6])
    return round((last / prev - 1) * 100, 3) if prev > 0 else None


# ── PER-SYMBOL SCAN ──────────────────────────────────────────────────────────

def scan_symbol(provider: MarketDataProvider, symbol: str, category: str) -> Optional[Dict]:
    """
    Evaluate one symbol.  Returns a signal dict if a qualifying setup is found,
    or None if no strong signal exists (weak setups are intentionally discarded).
    """
    try:
        frame = provider.get_ohlc(symbol=symbol, period="6mo", interval="1d")
        if frame is None or frame.empty or len(frame) < MIN_DATA_BARS:
            return None

        df = _prepare_indicators(frame)
        if df.empty or len(df) < 5:
            return None

        last_row   = df.iloc[-1]
        prev_row   = df.iloc[-2]
        recent4    = df.iloc[-4:]

        # ── Extract key values ────────────────────────────────────────────────
        last_close  = float(last_row["close"])
        last_rsi    = float(last_row["rsi14"])
        ema20       = float(last_row["ema20"])
        ema50       = float(last_row["ema50"])
        last_macdh  = float(last_row["macdh"])
        prev_macdh  = float(prev_row["macdh"])
        high20_prev = float(last_row["high20_prev"])
        last_volume = float(last_row["volume"])
        vol_sma20   = float(last_row["vol_sma20"])
        atr         = float(last_row["atr14"])
        st_up       = bool(last_row.get("st_uptrend", False))

        # ── Basic quality gates ───────────────────────────────────────────────
        if last_close < MIN_PRICE:
            return None
        if vol_sma20 < MIN_AVG_VOLUME:
            return None
        if not (MIN_RSI <= last_rsi <= MAX_RSI):
            return None
        if atr <= 0:
            return None

        # Pre-compute derived flags
        ema_uptrend   = ema20 > ema50
        vol_ratio     = (last_volume / vol_sma20) if vol_sma20 > 0 else 0
        min_rec_macdh = float(recent4["macdh"].astype(float).min())
        macd_cross_up = (
            (prev_macdh < 0 and last_macdh > 0) or
            (min_rec_macdh < 0 and last_macdh > 0)
        )

        # ── SIGNAL 1: Momentum Breakout ───────────────────────────────────────
        # Price breaks 20-day high with strong volume surge AND trend is up
        if (
            last_close >= high20_prev
            and vol_ratio >= 2.0
            and ema_uptrend
            and last_macdh > 0
        ):
            sig = _build_signal(symbol, category, last_row, "Momentum Breakout")
            sig["candidate_score"] = min(_score_candidate(last_row) + 20.0, 100.0)
            sig["is_candidate"]    = True
            sig["mover_5d_pct"]    = float(_mover_score(frame) or 0.0)
            sig["vol_ratio"]       = round(vol_ratio, 2)
            return sig

        # ── SIGNAL 2: MACD Bullish Cross ─────────────────────────────────────
        # MACD histogram crosses from negative to positive territory
        # + confirmed by upward EMA structure + Supertrend bullish
        if (
            macd_cross_up
            and ema_uptrend
            and st_up
            and 40 <= last_rsi <= 72
        ):
            sig = _build_signal(symbol, category, last_row, "MACD Cross + Supertrend")
            sig["candidate_score"] = _score_candidate(last_row) + 10.0
            sig["is_candidate"]    = True
            sig["mover_5d_pct"]    = float(_mover_score(frame) or 0.0)
            return sig

        # ── SIGNAL 3: EMA Pullback in Uptrend ────────────────────────────────
        # Price dips near EMA20 in a strong uptrend (EMA20 > EMA50 > price > EMA20 * 0.97)
        # RSI not overbought, MACD positive = continuation buy
        near_ema20 = (last_close >= ema20 * 0.97) and (last_close <= ema20 * 1.03)
        if (
            ema_uptrend
            and near_ema20
            and last_macdh > 0
            and 38 <= last_rsi <= 60
            and st_up
        ):
            sig = _build_signal(symbol, category, last_row, "EMA20 Pullback Buy")
            sig["candidate_score"] = _score_candidate(last_row) + 5.0
            sig["is_candidate"]    = True
            sig["mover_5d_pct"]    = float(_mover_score(frame) or 0.0)
            return sig

        # ── SIGNAL 4: High-Quality Trend Continuation ─────────────────────────
        # All four conditions must be green: EMA uptrend + Supertrend + MACD pos + RSI sweet spot
        # Require at least 1.5x volume to confirm participation
        if (
            ema_uptrend
            and st_up
            and last_macdh > 0
            and 45 <= last_rsi <= 68
            and vol_ratio >= 1.5
        ):
            sig = _build_signal(symbol, category, last_row, "Trend Continuation")
            sig["candidate_score"] = _score_candidate(last_row)
            sig["is_candidate"]    = True
            sig["mover_5d_pct"]    = float(_mover_score(frame) or 0.0)
            sig["vol_ratio"]       = round(vol_ratio, 2)
            return sig

        # No qualifying signal — return nothing (no weak fallback)
        return None

    except Exception:
        return None


# ── MARKET SCAN ORCHESTRATOR ─────────────────────────────────────────────────

def scan_market(
    provider: MarketDataProvider,
    categorized_stocks: Dict[str, List[str]],
    max_workers: int  = 8,
    max_signals: int  = 60,
    scan_timeout_sec: int = 40,
) -> List[Dict]:
    """
    Scan all provided symbols in parallel.
    Returns up to `max_signals` results sorted by score (best first).
    Serves from cache when the market is closed and cache is fresh.
    """

    # ── Cache path (market closed) ────────────────────────────────────────────
    if not is_market_open():
        cached = load_signals_cache()
        if cached and cached.get("signals"):
            for sig in cached["signals"]:
                sig["_from_cache"]          = True
                sig["_cache_last_updated"]  = cached.get("last_updated", "")
            return cached["signals"]

    # ── Build symbol→category map ─────────────────────────────────────────────
    symbol_tasks: Dict[str, str] = {}
    for category, stocks in categorized_stocks.items():
        for s in stocks:
            clean = str(s).strip().upper().replace(".NS", "")
            if clean:
                symbol_tasks[clean] = category

    for s in PRIORITY_SYMBOLS:
        clean = str(s).strip().upper().replace(".NS", "")
        if clean not in symbol_tasks:
            symbol_tasks[clean] = "smallcap"

    if not symbol_tasks:
        return []

    ipo_set = {
        str(s).strip().upper().replace(".NS", "")
        for s in categorized_stocks.get("ipo", [])
    }

    # ── Parallel scan ─────────────────────────────────────────────────────────
    signals: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(scan_symbol, provider, sym, cat): sym
            for sym, cat in symbol_tasks.items()
        }
        done, pending = wait(future_map.keys(), timeout=scan_timeout_sec)
        for f in pending:
            f.cancel()
        for f in done:
            try:
                result = f.result()
                if result:
                    # ── Post-filter: enforce minimum RR ───────────────────────
                    if float(result.get("rr", 0)) >= MIN_RR:
                        signals.append(result)
            except Exception:
                continue

    # ── Rank by candidate flag then score ────────────────────────────────────
    signals.sort(
        key=lambda x: (
            int(x.get("is_candidate", False)),
            float(x.get("candidate_score", 0)),
        ),
        reverse=True,
    )

    # ── Patch IPO category ────────────────────────────────────────────────────
    for item in signals:
        if item["ticker"] in ipo_set:
            item["category"] = "ipo"

    top = signals[:max_signals]

    if top:
        save_signals_cache(top)

    return top
