from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List
from urllib.request import urlopen
import time

import pandas as pd
import yfinance as yf


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
    "TATATECH","AWL","MEDANTA","MANKIND",
    "IREDA","DOMS","ZOMATO","NYKAA","PAYTM","LATENTVIEW"
]


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
        with urlopen(url, timeout=5) as response:
            raw_csv = response.read().decode("utf-8", errors="ignore")

        df = pd.read_csv(StringIO(raw_csv))

        for col in ["Symbol", "SYMBOL", "Ticker", "ticker"]:
            if col in df.columns:
                symbols = [_clean_symbol(x) for x in df[col].dropna()]
                return sorted(list(set(filter(None, symbols))))
    except Exception as e:
        print(f"[DATA ERROR] Failed to load index from {url}: {e}")

    return []


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

        # 🔁 Retry mechanism
        for attempt in range(3):
            try:
                df = self._download(ticker, period, interval)

                if df.empty:
                    continue

                # ✅ Fix multi-index columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

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
    def get_ohlc(self, symbol: str, period="8mo", interval="1d") -> pd.DataFrame:
        raise NotImplementedError("Add broker API here later")


# ─────────────────────────────────────────────────────────────
# DEFAULT PROVIDER
# ─────────────────────────────────────────────────────────────

def get_default_provider() -> MarketDataProvider:
    return YFinanceDataProvider()
