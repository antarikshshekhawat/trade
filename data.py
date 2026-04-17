from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List
from urllib.request import urlopen

import pandas as pd
import yfinance as yf


INDEX_URLS = {
    "largecap": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "midcap": "https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "smallcap": "https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
}


# ✅ FIXED: comma added after largecap list
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
    "TATATECH","AWL","MEDANTA","MANKIND",
    "IREDA","DOMS","ZOMATO","NYKAA","PAYTM","LATENTVIEW"
]


def _clean_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not symbol:
        return ""
    return symbol.replace(".NS", "")


def _to_nse_ticker(symbol: str) -> str:
    symbol = _clean_symbol(symbol)
    if not symbol:
        return ""
    return f"{symbol}.NS"


def _load_index_symbols(url: str) -> List[str]:
    try:
        with urlopen(url, timeout=5) as response:
            raw_csv = response.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(StringIO(raw_csv))

        for col in ["Symbol", "SYMBOL", "Ticker", "ticker"]:
            if col in df.columns:
                symbols = [_clean_symbol(x) for x in df[col].dropna()]
                return sorted(list(set(filter(None, symbols))))
    except Exception:
        return []

    return []


def build_stock_universe() -> Dict[str, List[str]]:
    universe = {}

    for category, url in INDEX_URLS.items():
        live = _load_index_symbols(url)

        if live:
            universe[category] = live
        else:
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


class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        pass


class YFinanceDataProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        ticker = _to_nse_ticker(symbol)

        if not ticker:
            return pd.DataFrame()

        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # normalize columns
        df.columns = [c.lower() for c in df.columns]

        required = ["open", "high", "low", "close", "volume"]

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df = df[required].dropna()

        return df


class BrokerRealtimeProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        raise NotImplementedError("Add broker API later")


def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
