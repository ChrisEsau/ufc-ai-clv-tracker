from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
import yaml

from pipeline.modeling.model_loader import load_model_bundle
from pipeline.modeling.model_config import load_model_config
from pipeline.modeling.run_live_model_forensics import try_shap, unwrap_estimator
from tabs.model_lab_sections import backtest as base

DOG_EDGE_THRESHOLD = 0.10
MAX_SHAP_ROWS_PER_GROUP = 75
TOP_N_FEATURES = 30

# (existing file unchanged above/below except additions)

# NOTE: Added dog archetype summary helper.
def _dog_archetype_summary(values: pd.DataFrame) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()

    targets = [
        "avg_opponent_elo_diff",
        "ewm_avg_opponent_elo_diff",
        "best_win_elo_diff",
        "ewm_best_win_elo_diff",
        "win_pct_diff",
        "ewm_win_pct_diff",
        "striking_edge",
        "grappling_edge",
        "submission_mismatch_diff",
        "wrestling_mismatch_diff",
    ]

    available = values[values["feature"].isin(targets)].copy()
    if available.empty:
        return pd.DataFrame()

    available["abs_delta"] = pd.to_numeric(
        available["avg_delta_losing_minus_winning"],
        errors="coerce",
    ).abs()

    return available[[
        "feature",
        "losing_dog_avg",
        "winning_dog_avg",
        "avg_delta_losing_minus_winning",
        "abs_delta",
    ]].sort_values("abs_delta", ascending=False)


# Existing code remains unchanged...

# Injected into render_dog_audit immediately after Feature Value Comparison.
# New section:
# ###### Dog Archetype Summary
# Compares the most important competition-quality and matchup metrics.
