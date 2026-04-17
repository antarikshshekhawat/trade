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

FALLBACK_UNIVERSE = {
"largecap": [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","LT","SBIN",
    "ITC","AXISBANK","BAJFINANCE","ASIANPAINT","KOTAKBANK",
    "MARUTI","HCLTECH","WIPRO","ULTRACEMCO","SUNPHARMA",
    "TITAN","NTPC","POWERGRID"
]
    "midcap": [
        "POLYCAB",
        "PERSISTENT",
        "COFORGE",
        "MPHASIS",
        "BHEL",
        "NHPC",
        "IDFCFIRSTB",
        "LUPIN",
        "INDHOTEL",
        "SUPREMEIND",
    ],
    "smallcap": [
        "IRB",
        "JUBLINGREA",
        "FSL",
        "KNRCON",
        "RKFORGE",
        "RAIN",
        "TRITURBINE",
        "FCL",
        "WELCORP",
        "KPIGREEN",
    ],
}

FALLBACK_IPO_STOCKS = [
    "TATATECH",
    "AWL",
    "MEDANTA",
    "MANKIND",
    "IREDA",
    "DOMS",
    "ZOMATO",
    "NYKAA",
    "PAYTM",
    "LATENTVIEW",
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
        frame = pd.read_csv(StringIO(raw_csv))
        candidate_columns = ["Symbol", "SYMBOL", "Ticker", "ticker"]
        for column in candidate_columns:
            if column in frame.columns:
                symbols = [_clean_symbol(item) for item in frame[column].dropna().tolist()]
                return sorted(set([item for item in symbols if item]))
    except Exception:
        return []
    return []


def build_stock_universe() -> Dict[str, List[str]]:
    universe = {}
    for category, url in INDEX_URLS.items():
        live_symbols = _load_index_symbols(url)
        if live_symbols:
            universe[category] = live_symbols
        else:
            universe[category] = FALLBACK_UNIVERSE[category]
    universe["ipo"] = FALLBACK_IPO_STOCKS
    return universe


@dataclass
class StockRecord:
    symbol: str
    category: str


def flatten_universe(universe: Dict[str, List[str]]) -> List[StockRecord]:
    records: List[StockRecord] = []
    for category, symbols in universe.items():
        for symbol in symbols:
            cleaned = _clean_symbol(symbol)
            if cleaned:
                records.append(StockRecord(symbol=cleaned, category=category))
    return records


class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlc(self, symbol: str, period: str = "8mo", interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError


class YFinanceDataProvider(MarketDataProvider):
    def get_ohlc(self, symbol: str, period: str = "8mo", interval: str = "1d") -> pd.DataFrame:
        ticker = _to_nse_ticker(symbol)
        if not ticker:
            return pd.DataFrame()
        try:
            frame = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame()

        if frame.empty:
            return pd.DataFrame()

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[0]).lower() for col in frame.columns]
        else:
            frame = frame.rename(columns=str.lower)

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(frame.columns):
            return pd.DataFrame()

        frame = frame[list(required_cols)].dropna().copy()
        if frame.empty:
            return pd.DataFrame()
        return frame


class BrokerRealtimeProvider(MarketDataProvider):
    """
    Placeholder provider for future broker integrations:
    - Zerodha Kite historical + LTP streams
    - Angel One SmartAPI
    - Upstox
    """

    def get_ohlc(self, symbol: str, period: str = "8mo", interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError("Implement broker API bridge for live market data.")


def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
