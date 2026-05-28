from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List
import time
import warnings

import pandas as pd
import yfinance as yf
from nsepython import nse_quote_ltp  # Kept for compatibility

# ─────────────────────────────────────────────────────────────
# INDEX DATA SOURCES
# ─────────────────────────────────────────────────────────────

INDEX_URLS = {
    "largecap": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "midcap": "https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "smallcap": "https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
}

# EXPANDED UNIVERSE FOR PYTHONANYWHERE BLOCK BYPASS (150+ Stocks)
FALLBACK_UNIVERSE = {
    "largecap": [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "INFY", "LICI", "ITC", "HINDUNILVR",
        "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "TATAMOTORS", "M&M", "KOTAKBANK", "ONGC", "TATASTEEL",
        "COALINDIA", "NTPC", "AXISBANK", "POWERGRID", "ASIANPAINT", "BAJAJFINSV", "TITAN", "ADANIPORTS", "ULTRACEMCO", "WIPRO",
        "JSWSTEEL", "ZOMATO", "GRASIM", "TECHM", "BAJAJ-AUTO", "HINDALCO", "TRENT", "LTIM", "NESTLEIND", "SIEMENS",
        "DRREDDY", "HAL", "IOC", "CIPLA", "INDUSINDBK", "EICHERMOT", "APOLLOHOSP", "PIDILITIND", "BRITANNIA", "BEL"
    ],
    "midcap": [
        "POLYCAB", "PERSISTENT", "COFORGE", "MPHASIS", "BHEL", "NHPC", "IDFCFIRSTB", "LUPIN", "INDHOTEL", "SUPREMEIND",
        "ASTRAL", "CGPOWER", "CUMMINSIND", "DIXON", "ESCORTS", "GODREJPROP", "KPITTECH", "MAXHEALTH", "MAZDOCK", "OFSS",
        "PAGEIND", "PAYTM", "PIIND", "PRESTIGE", "RECLTD", "SAIL", "TATACOMM", "TORNTPOWER", "TVSMOTOR",
        "UBL", "UCOBANK", "VOLTAS", "YESBANK", "ZEEL", "APOLLOTYRE", "ASHOKLEY", "BALKRISIND", "BANDHANBNK", "BANKBARODA"
    ],
    "smallcap": [
        "IRB", "JUBLINGREA", "FSL", "KNRCON", "RKFORGE", "RAIN", "TRITURBINE", "FCL", "WELCORP", "KPIGREEN",
        "ANGELONE", "ANURAS", "BEML", "BLS", "BSOFT", "CDSL", "CEATLTD", "CENTURYPLY", "CERA", "CHAMBLFERT",
        "CHEMPLASTS", "CHOLAFIN", "CITYUNION", "CLEAN", "COCHINSHIP", "CREDITACC", "CROMPTON", "CSBBANK", "CYIENT", "DATAPATTNS",
        "DEEPAKNTR", "DELHIVERY", "DEVYANI", "ECLERX", "EIDPARRY", "EQUITASBNK", "ERIS", "EXIDEIND", "FACT", "FINEORG"
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
        print(f"[DATA ERROR] Failed to load full NSE list (PythonAnywhere Blocked): {e}")

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
                ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
            )

    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        ticker = _to_nse_ticker(symbol)

        if not ticker:
            return pd.DataFrame()

        for attempt in range(3):
            try:
                df = self._download(ticker, period, interval)

                if df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(c[0]).lower() for c in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]

                required = ["open", "high", "low", "close", "volume"]

                if not all(col in df.columns for col in required):
                    return pd.DataFrame()

                df = df[required].dropna()

                if df.empty:
                    return pd.DataFrame()

                return df

            except Exception as e:
                print(f"[YFINANCE ERROR] {ticker} attempt {attempt+1}: {e}")
                time.sleep(1)

        return pd.DataFrame()

class BrokerRealtimeProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period="1d", interval="1m") -> pd.DataFrame:
        return get_default_provider().get_ohlc(symbol, period, interval)

def fetch_last_prices_nse(symbols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    valid_symbols = list(set([_clean_symbol(str(s)) for s in symbols if _clean_symbol(str(s))][:100]))
    if not valid_symbols:
        return out

    tickers_ns = " ".join([f"{s}.NS" for s in valid_symbols])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_ns = yf.download(tickers_ns, period="1d", interval="1m", progress=False)

        if not df_ns.empty:
            if isinstance(df_ns.columns, pd.MultiIndex):
                for symbol in valid_symbols:
                    ticker = f"{symbol}.NS"
                    if ('Close', ticker) in df_ns.columns:
                        series = df_ns['Close'][ticker].dropna()
                        if not series.empty:
                            out[symbol] = round(float(series.iloc[-1]), 2)
            else:
                if 'Close' in df_ns.columns:
                    series = df_ns['Close'].dropna()
                    if not series.empty:
                        out[valid_symbols[0]] = round(float(series.iloc[-1]), 2)
    except Exception as e:
        print(f"[LIVE FETCH ERROR] Bulk YF fetch failed: {e}")
            
    return out

# ─────────────────────────────────────────────────────────────
# DEFAULT PROVIDER
# ─────────────────────────────────────────────────────────────

def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
