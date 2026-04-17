from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List
from urllib.request import urlopen

import pandas as pd
import yfinance as yf


# ================================
# INDEX DATA SOURCES
# ================================
INDEX_URLS = {
    "largecap": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "midcap": "https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "smallcap": "https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
}


# ================================
# FALLBACK (SAFE + EXTENDED)
# ================================
FALLBACK_UNIVERSE = {
    "largecap": [
        "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","LT","SBIN",
        "ITC","AXISBANK","BAJFINANCE","ASIANPAINT","KOTAKBANK",
        "MARUTI","HCLTECH","WIPRO","ULTRACEMCO","SUNPHARMA",
        "TITAN","NTPC","POWERGRID","ADANIENT","ADANIPORTS",
        "BAJAJFINSV","TECHM","GRASIM","JSWSTEEL","HDFCLIFE"
    ],

    "midcap": [
        "POLYCAB","PERSISTENT","COFORGE","MPHASIS","BHEL","NHPC",
        "IDFCFIRSTB","LUPIN","INDHOTEL","SUPREMEIND",
        "CUMMINSIND","DIXON","AUBANK","ASTRAL","PIIND",
        "PAGEIND","SRF","NAVINFLUOR","ALKEM","GODREJPROP"
    ],

    "smallcap": [
        "IRB","JUBLINGREA","FSL","KNRCON","RKFORGE","RAIN",
        "TRITURBINE","FCL","WELCORP","KPIGREEN",
        "RVNL","IRFC","NBCC","HFCL","SJVN",
        "MAHSEAMLES","KSB","GMRINFRA","RITES","BEML"
    ],
}


FALLBACK_IPO_STOCKS = [
    "TATATECH","AWL","MEDANTA","MANKIND","IREDA",
    "DOMS","ZOMATO","NYKAA","PAYTM","LATENTVIEW"
]


# ================================
# HELPERS
# ================================
def _clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".NS", "")


def _to_nse_ticker(symbol: str) -> str:
    symbol = _clean_symbol(symbol)
    return f"{symbol}.NS" if symbol else ""


# ================================
# FETCH INDEX DATA
# ================================
def _load_index_symbols(url: str) -> List[str]:
    try:
        with urlopen(url, timeout=10) as response:
            raw_csv = response.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(StringIO(raw_csv))

        for col in ["Symbol", "SYMBOL", "Ticker"]:
            if col in df.columns:
                return sorted(set(
                    _clean_symbol(x) for x in df[col].dropna().tolist()
                ))

    except Exception as e:
        print(f"[ERROR] Failed to load {url} -> {e}")
        return []

    return []


# ================================
# BUILD STOCK UNIVERSE
# ================================
def build_stock_universe() -> Dict[str, List[str]]:
    universe = {}

    for category, url in INDEX_URLS.items():
        symbols = _load_index_symbols(url)

        if symbols:
            print(f"[DATA] Loaded {len(symbols)} {category}")
            universe[category] = symbols
        else:
            print(f"[DATA] Using fallback for {category}")
            universe[category] = FALLBACK_UNIVERSE[category]

    universe["ipo"] = FALLBACK_IPO_STOCKS

    return universe


# ================================
# DATA STRUCTURE
# ================================
@dataclass
class StockRecord:
    symbol: str
    category: str


def flatten_universe(universe: Dict[str, List[str]]) -> List[StockRecord]:
    records = []

    for category, symbols in universe.items():
        for s in symbols:
            cleaned = _clean_symbol(s)
            if cleaned:
                records.append(StockRecord(cleaned, category))

    return records


# ================================
# DATA PROVIDERS
# ================================
class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlc(self, symbol: str, period: str = "8mo", interval: str = "1d") -> pd.DataFrame:
        pass


class YFinanceDataProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period: str = "8mo", interval: str = "1d") -> pd.DataFrame:
        ticker = _to_nse_ticker(symbol)

        if not ticker:
            return pd.DataFrame()

        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False
            )
        except Exception as e:
            print(f"[ERROR] yf failed for {symbol}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns=str.lower)

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        return df[list(required)].dropna()


class BrokerRealtimeProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        raise NotImplementedError("Use broker API later")


# ================================
# DEFAULT PROVIDER
# ================================
def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
