from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common.paths import BETTING_BOARD_PATH, BETTING_OUTCOMES_PATH
from utils.betting_outcomes_adapter import betting_outcomes_to_legacy_board


@st.cache_data
def load_parquet(path):
    path = Path(path)

    # Backend-only Betting Board migration:
    # keep the existing Streamlit UI reading BETTING_BOARD_PATH, but allow the
    # new generic outcome-level artifact to drive it when available.
    if path == Path(BETTING_BOARD_PATH) and Path(BETTING_OUTCOMES_PATH).exists():
        try:
            betting_outcomes = pd.read_parquet(BETTING_OUTCOMES_PATH)
            adapted = betting_outcomes_to_legacy_board(betting_outcomes)
            if not adapted.empty:
                return adapted
        except Exception:
            pass

    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
