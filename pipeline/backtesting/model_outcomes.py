from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.modeling.prediction_formatter import format_prediction_outcomes


def build_backtest_model_outcomes(
    *,
    feature_df: pd.DataFrame,
    probabilities: pd.Series | np.ndarray,
    model_config: dict[str, Any],
    prediction_run_id: str,
    prediction_timestamp: str,
) -> pd.DataFrame:
    """Build backtest model outcomes with the live prediction formatter."""

    fight_df = build_formatter_fight_df(feature_df)
    return format_prediction_outcomes(
        fight_df=fight_df,
        probabilities=np.asarray(probabilities, dtype="float64"),
        model_config=model_config,
        prediction_run_id=prediction_run_id,
        prediction_timestamp=prediction_timestamp,
    )


def build_formatter_fight_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map historical feature-view metadata to prediction formatter columns."""

    out = pd.DataFrame(index=df.index)
    out["event_id"] = df["event_id"] if "event_id" in df.columns else pd.NA
    out["event_name"] = df["event_name"] if "event_name" in df.columns else pd.NA
    out["fight_id"] = df["fight_id"].astype(str)
    out["red_fighter"] = df["r_name"] if "r_name" in df.columns else df.get("red_fighter", pd.NA)
    out["blue_fighter"] = df["b_name"] if "b_name" in df.columns else df.get("blue_fighter", pd.NA)
    out["red_fighter_id"] = df["r_id"] if "r_id" in df.columns else df.get("red_fighter_id", pd.NA)
    out["blue_fighter_id"] = df["b_id"] if "b_id" in df.columns else df.get("blue_fighter_id", pd.NA)
    return out
