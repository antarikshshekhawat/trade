from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List
import time
import warnings

import pandas as pd
import yfinance as yf
from nsepython import nse_quote_ltp  # Kept for compatibility if used elsewhere

# ─────────────────────────────────────────────────────────────
# INDEX DATA SOURCES
# ─────────────────────────────────────────────────────────────

INDEX_URLS = {
    "largecap": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "midcap": "https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "smallcap": "https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
}

FALLBACK_UNIVERSE = {
    "largecap": [
        "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","LT","SBIN",
        "ITC","AXISBANK","BAJFINANCE","ASIANPAINT","KOTAKBANK",
        "MARUTI","HCLTECH","WIPRO","ULTRACEMCO","SUNPHARMA",
        "TITAN","NTPC","POWERGRID"
    ],
    "midcap": [
        "POLYCAB","PERSISTENT","COFORGE","MPHASIS","BHEL",
        "NHPC","IDFCFIRSTB","LUPIN","INDHOTEL","SUPREMEIND"
    ],
    "smallcap": [
        "IRB","JUBLINGREA","FSL","KNRCON","RKFORGE",
        "RAIN","TRITURBINE","FCL","WELCORP","KPIGREEN"
    ],
}

FALLBACK_IPO_STOCKS = [
    "HYUNDAI","BAJAJHFL","OLALEC","PREMIERENE","UNIECOM","TBO",
    "AWFIS","KRN","VRAJ","GODIGIT","SWIGGY","MOBIKWIK",
    "NSDL","WAAREEENER","JUNIPER","AZAD","KAYNES","TATATECH",
    "MANKIND","DOMS","IREDA","MEDANTA","LATENTVIEW","NYKAA"
]

ALL_NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_ALL_NSE_CACHE: Dict[str, object] = {"symbols": [], "expires_at": 0.0}

# MEMORY CACHE TO STOP SWITCHBACKING
_LTP_CACHE: Dict[str, float] = {}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".NS", "")

def _to_nse_ticker(symbol: str) -> str:
    s = _clean_symbol(symbol)
    return f"{s}.NS" if s else ""

# ─────────────────────────────────────────────────────────────
# LOAD SYMBOLS FROM NSE
# ─────────────────────────────────────────────────────────────

def _load_index_symbols(url: str) -> List[str]:
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_csv = response.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(StringIO(raw_csv))

        for col in ["Symbol", "SYMBOL", "Ticker", "ticker"]:
            if col in df.columns:
                symbols = [_clean_symbol(x) for x in df[col].dropna()]
                return sorted(list(set(filter(None, symbols))))
    except Exception as e:
        print(f"[DATA ERROR] Failed to load index from {url}: {e}")

    return []

def load_all_nse_symbols() -> List[str]:
    now = time.time()
    if _ALL_NSE_CACHE["symbols"] and now < float(_ALL_NSE_CACHE["expires_at"]):
        return _ALL_NSE_CACHE["symbols"]  # type: ignore[return-value]

    symbols: List[str] = []
    try:
        req = urllib.request.Request(
            ALL_NSE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            raw_csv = response.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(raw_csv))
        for col in ["SYMBOL", "Symbol", "symbol"]:
            if col in df.columns:
                symbols = sorted(list(set(_clean_symbol(x) for x in df[col].dropna() if _clean_symbol(x))))
                break
    except Exception as e:
        print(f"[DATA ERROR] Failed to load full NSE list: {e}")

    if not symbols:
        uni = build_stock_universe()
        tmp: List[str] = []
        for vals in uni.values():
            tmp.extend(vals)
        symbols = sorted(list(set(_clean_symbol(x) for x in tmp if _clean_symbol(x))))

    _ALL_NSE_CACHE["symbols"] = symbols
    _ALL_NSE_CACHE["expires_at"] = now + (6 * 3600)
    return symbols

# ─────────────────────────────────────────────────────────────
# BUILD STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────

def build_stock_universe() -> Dict[str, List[str]]:
    universe = {}
    for category, url in INDEX_URLS.items():
        live = _load_index_symbols(url)
        if live:
            universe[category] = live
        else:
            print(f"[WARNING] Using fallback for {category}")
            universe[category] = FALLBACK_UNIVERSE[category]
    universe["ipo"] = FALLBACK_IPO_STOCKS
    return universe

@dataclass
class StockRecord:
    symbol: str
    category: str

def flatten_universe(universe: Dict[str, List[str]]) -> List[StockRecord]:
    records = []
    for category, symbols in universe.items():
        for symbol in symbols:
            s = _clean_symbol(symbol)
            if s:
                records.append(StockRecord(s, category))
    return records

# ─────────────────────────────────────────────────────────────
# DATA PROVIDERS
# ─────────────────────────────────────────────────────────────

class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        pass

class YFinanceDataProvider(MarketDataProvider):
    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return yf.download(
                ticker, period=period, interval=interval, progress=False, threads=False,
            )

    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        ticker = _to_nse_ticker(symbol)
        if not ticker: return pd.DataFrame()
        for attempt in range(3):
            try:
                df = self._download(ticker, period, interval)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(c[0]).lower() for c in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]
                required = ["open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required): return pd.DataFrame()
                df = df[required].dropna()
                if df.empty: return pd.DataFrame()
                return df
            except Exception as e:
                print(f"[YFINANCE ERROR] {ticker} attempt {attempt+1}: {e}")
                time.sleep(1)
        return pd.DataFrame()

class BrokerRealtimeProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period="1d", interval="1m") -> pd.DataFrame:
        return get_default_provider().get_ohlc(symbol, period, interval)

# ─────────────────────────────────────────────────────────────
# ULTIMATE TRADINGVIEW RAW API FETCHER WITH CACHE
# ─────────────────────────────────────────────────────────────

def fetch_last_prices_nse(symbols: List[str]) -> Dict[str, float]:
    """
    High-speed BULK Live Price Fetcher.
    Uses TradingView first. Caches the result to prevent switchbacking.
    """
    out: Dict[str, float] = {}
    valid_symbols = list(set([_clean_symbol(str(s)) for s in symbols if _clean_symbol(str(s))][:100]))
    if not valid_symbols:
        return out

    # 1. Query TradingView Servers Directly
    try:
        tv_tickers = [f"NSE:{s}" for s in valid_symbols]
        url = "https://scanner.tradingview.com/india/scan"
        payload = {
            "symbols": { "tickers": tv_tickers, "query": { "types": [] } },
            "columns": ["close"] 
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            for item in resp_data.get('data', []):
                ticker_name = item.get('s', '').replace('NSE:', '')
                close_price = item.get('d', [0])[0]
                if close_price:
                    out[ticker_name] = round(float(close_price), 2)
                    _LTP_CACHE[ticker_name] = out[ticker_name] # LOCK PRICE IN MEMORY
    except Exception as e:
        print(f"[LIVE FETCH ERROR] TV Bulk fetch failed: {e}")

    # 2. Check Cache before falling back to yfinance (STOPS SWITCHBACKING)
    missing_symbols = [s for s in valid_symbols if s not in out]
    yfinance_needed = []
    
    for s in missing_symbols:
        if s in _LTP_CACHE:
            out[s] = _LTP_CACHE[s] # Restore stable price from cache
        else:
            yfinance_needed.append(s)

    # 3. YFinance Fallback ONLY for unseen symbols
    if yfinance_needed:
        tickers_bo = " ".join([f"{s}.BO" for s in yfinance_needed]) 
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_bo = yf.download(tickers_bo, period="1d", interval="1m", progress=False)

            if not df_bo.empty:
                if isinstance(df_bo.columns, pd.MultiIndex):
                    for symbol in yfinance_needed:
                        ticker = f"{symbol}.BO"
                        if ('Close', ticker) in df_bo.columns:
                            series = df_bo['Close'][ticker].dropna()
                            if not series.empty:
                                price = round(float(series.iloc[-1]), 2)
                                out[symbol] = price
                                _LTP_CACHE[symbol] = price
                else:
                    if 'Close' in df_bo.columns:
                        series = df_bo['Close'].dropna()
                        if not series.empty:
                            price = round(float(series.iloc[-1]), 2)
                            out[yfinance_needed[0]] = price
                            _LTP_CACHE[yfinance_needed[0]] = price
        except Exception:
            pass
            
    return out

# ─────────────────────────────────────────────────────────────
# DEFAULT PROVIDER
# ─────────────────────────────────────────────────────────────

def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
