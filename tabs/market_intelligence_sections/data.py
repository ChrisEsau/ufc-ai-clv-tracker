from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.common.paths import MARKET_INTELLIGENCE_HISTORY_PATH, MARKET_SIGNALS_PATH
from utils.data_loader import load_parquet


@dataclass
class MarketIntelligenceData:
    signals: pd.DataFrame
    history: pd.DataFrame


def _safe_load(path) -> pd.DataFrame:
    df = load_parquet(path)
    if df is None:
        return pd.DataFrame()
    return df.copy()


def load_market_intelligence_data() -> MarketIntelligenceData:
    return MarketIntelligenceData(
        signals=_safe_load(MARKET_SIGNALS_PATH),
        history=_safe_load(MARKET_INTELLIGENCE_HISTORY_PATH),
    )
