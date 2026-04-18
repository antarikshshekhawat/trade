from __future__ import annotations

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

EXCHANGE_SUFFIX = ".NS"
_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


class MarketDataProvider:
    """
    Fetches OHLCV data from Yahoo Finance for NSE-listed stocks.

    Symbols are accepted without the .NS suffix; it is appended automatically.
    An in-process LRU-style cache reduces redundant network calls when the same
    symbol is requested by multiple scan workers.
    """

    def __init__(self, cache_size: int = 300) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_size = cache_size

    # ── INTERNAL HELPERS ─────────────────────────────────────────────────────

    def _to_ticker(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        # Strip existing suffixes to normalise
        for suffix in (".NS", ".BO", ".BSE"):
            s = s.replace(suffix, "")
        return s + EXCHANGE_SUFFIX

    def _evict_if_full(self) -> None:
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)), None)

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse MultiIndex columns produced by yfinance ≥0.2."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _rename_and_select(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Lower-case columns, rename adj_close, select required cols."""
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        if "adj_close" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"adj_close": "close"})
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            logger.debug("Missing columns %s", missing)
            return None
        return df[_REQUIRED_COLS].copy()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def get_ohlc(
        self,
        symbol: str,
        period: str = "4mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame with columns [open, high, low, close, volume]
        indexed by date, sorted ascending.  Returns None on any failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance is not installed. Run: pip install yfinance")
            return None

        ticker = self._to_ticker(symbol)
        cache_key = f"{ticker}|{period}|{interval}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            raw: pd.DataFrame = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:
            logger.debug("Download failed for %s: %s", ticker, exc)
            return None

        if raw is None or raw.empty:
            return None

        raw = self._flatten_columns(raw)
        df = self._rename_and_select(raw)
        if df is None:
            return None

        # Drop rows where close price is missing / zero
        df = df[df["close"].notna() & (df["close"] > 0)]
        if len(df) < 10:
            return None

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        self._evict_if_full()
        self._cache[cache_key] = df.copy()
        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Return the latest available closing price for a symbol.
        Tries fast_info first, falls back to a 5-day OHLC fetch.
        """
        try:
            import yfinance as yf
            ticker = self._to_ticker(symbol)
            t = yf.Ticker(ticker)
            fi = t.fast_info
            for attr in ("last_price", "regular_market_price"):
                val = getattr(fi, attr, None)
                if val and val > 0:
                    return round(float(val), 2)
        except Exception:
            pass

        # Fallback
        df = self.get_ohlc(symbol, period="5d", interval="1d")
        if df is not None and not df.empty:
            return round(float(df["close"].iloc[-1]), 2)
        return None

    def clear_cache(self) -> None:
        self._cache.clear()
