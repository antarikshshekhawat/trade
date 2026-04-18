"""
Nifty sector index snapshots via Yahoo Finance (approximate sector heat).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import yfinance as yf

# Major Indian sector / thematic indices (Yahoo symbols — some may fail if renamed)
SECTOR_INDICES: List[tuple[str, str]] = [
    ("^NSEI", "Nifty 50"),
    ("^NSEBANK", "Bank"),
    ("^CNXIT", "IT"),
    ("^CNXPHARMA", "Pharma"),
    ("^CNXAUTO", "Auto"),
    ("^CNXFMCG", "FMCG"),
    ("^CNXMETAL", "Metal"),
    ("^CNXENERGY", "Energy"),
    ("^CNXREALTY", "Realty"),
    ("^CNXMEDIA", "Media"),
    ("^CNXFINANCE", "Financial Services"),
]


def get_sector_performance() -> Dict[str, Any]:
    sectors: List[Dict[str, Any]] = []
    for ysym, label in SECTOR_INDICES:
        try:
            t = yf.Ticker(ysym)
            hist = t.history(period="10d", interval="1d")
            if hist is None or hist.empty or len(hist) < 2:
                continue
            close = hist["Close"]
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            pct = ((last / prev) - 1.0) * 100.0 if prev else 0.0
            sectors.append(
                {
                    "symbol": ysym,
                    "name": label,
                    "change_pct": round(pct, 3),
                    "last": round(last, 2),
                }
            )
        except Exception:
            continue

    sectors.sort(key=lambda x: x["change_pct"], reverse=True)
    hot = sectors[0] if sectors else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hot": hot,
        "sectors": sectors,
    }
