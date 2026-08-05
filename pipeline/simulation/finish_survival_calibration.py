"""Sequential round-survival calibration for finish-hazard providers.

The multiclass finish model is calibrated for event class probabilities across
observed fighter-round rows. A full fight simulator also needs the sequence of
round hazards to reproduce when finishes occur. Small conditional hazard errors
compound across rounds and can materially bias expected fight duration even when
aggregate decision/KO/submission rates are accurate.

This module calibrates only terminal-event mass by scheduled-round/round group.
It preserves the model's conditional KO/submission and red/blue proportions. The
schedule for a target holdout year is estimated exclusively from earlier
walk-forward predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import FINISH_CLASSES


class FinishSurvivalCalibrationError(RuntimeError):
    """Raised when survival calibration inputs violate the contract."""


@dataclass(frozen=True)
class FinishSurvivalCalibrationResult:
    predictions: pd.DataFrame
    schedule: pd.DataFrame


CALIBRATED_COLUMNS = tuple(
    f"calibrated_prob_{name}" for name in FINISH_CLASSES
)
TERMINAL_COLUMNS = tuple(
    column for column in CALIBRATED_COLUMNS if not column.endswith("no_finish")
)


def _clip_probability(value: float, low: float = 1e-5, high: float = 1.0 - 1e-5) -> float:
    return float(np.clip(float(value), low, high))


def _odds(value: float) -> float:
    probability = _clip_probability(value)
    return probability / (1.0 - probability)


def _terminal_probability(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(TERMINAL_COLUMNS)].sum(axis=1).to_numpy(dtype=float)


def _validate_predictions(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = [
        "fight_id",
        "round",
        "total_rounds",
        "model_name",
        *CALIBRATED_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FinishSurvivalCalibrationError(
            f"{label} is missing required columns: {missing}"
        )
    out = frame.copy()
    for column in ("round", "total_rounds"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in CALIBRATED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["round", "total_rounds", *CALIBRATED_COLUMNS]].isna().any().any():
        raise FinishSurvivalCalibrationError(
            f"{label} contains missing survival values"
        )
    out["round"] = out["round"].astype(int)
    out["total_rounds"] = out["total_rounds"].astype(int)
    probabilities = out[list(CALIBRATED_COLUMNS)].to_numpy(dtype=float)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise FinishSurvivalCalibrationError(
            f"{label} probability rows must be nonnegative and sum to one"
        )
    return out


def fit_finish_survival_schedule(
    walk_forward_predictions: pd.DataFrame,
    model_name: str,
    target_year: int,
    group_prior_rows: float = 200.0,
    factor_low: float = 0.25,
    factor_high: float = 4.0,
) -> pd.DataFrame:
    """Estimate terminal-odds factors from completed prior holdout years only."""
    if group_prior_rows < 0 or not np.isfinite(group_prior_rows):
        raise FinishSurvivalCalibrationError(
            "group_prior_rows must be finite and nonnegative"
        )
    if factor_low <= 0 or factor_high <= factor_low:
        raise FinishSurvivalCalibrationError("Invalid survival factor bounds")

    frame = _validate_predictions(
        walk_forward_predictions, "Walk-forward finish predictions"
    )
    for column in ("test_year", "finish_class"):
        if column not in frame.columns:
            raise FinishSurvivalCalibrationError(
                f"Walk-forward finish predictions are missing {column!r}"
            )
    frame["test_year"] = pd.to_numeric(frame["test_year"], errors="coerce")
    if frame["test_year"].isna().any():
        raise FinishSurvivalCalibrationError(
            "Walk-forward predictions contain invalid test_year values"
        )
    prior = frame.loc[
        frame["model_name"].astype(str).eq(str(model_name))
        & frame["test_year"].lt(int(target_year))
    ].copy()
    if prior.empty:
        raise FinishSurvivalCalibrationError(
            f"No prior walk-forward rows exist for {model_name}/{target_year}"
        )
    prior["actual_terminal"] = prior["finish_class"].astype(str).ne(
        "no_finish"
    ).astype(float)
    prior["predicted_terminal"] = _terminal_probability(prior)

    parent_rows: dict[int, dict[str, float]] = {}
    for round_number, group in prior.groupby("round", sort=True):
        parent_rows[int(round_number)] = {
            "rows": float(len(group)),
            "actual_rate": float(group["actual_terminal"].mean()),
            "predicted_rate": float(group["predicted_terminal"].mean()),
        }
    global_actual = float(prior["actual_terminal"].mean())
    global_predicted = float(prior["predicted_terminal"].mean())

    schedule_rows: list[dict[str, object]] = []
    for (total_rounds, round_number), group in prior.groupby(
        ["total_rounds", "round"], sort=True
    ):
        parent = parent_rows.get(
            int(round_number),
            {
                "rows": float(len(prior)),
                "actual_rate": global_actual,
                "predicted_rate": global_predicted,
            },
        )
        rows = float(len(group))
        actual_count = float(group["actual_terminal"].sum())
        predicted_count = float(group["predicted_terminal"].sum())
        actual_rate = (actual_count + group_prior_rows * parent["actual_rate"]) / (
            rows + group_prior_rows
        )
        predicted_rate = (
            predicted_count + group_prior_rows * parent["predicted_rate"]
        ) / (rows + group_prior_rows)
        factor = float(
            np.clip(
                _odds(actual_rate) / _odds(predicted_rate),
                factor_low,
                factor_high,
            )
        )
        schedule_rows.append(
            {
                "model_name": str(model_name),
                "target_year": int(target_year),
                "total_rounds": int(total_rounds),
                "round": int(round_number),
                "prior_rows": int(rows),
                "parent_round_rows": int(parent["rows"]),
                "actual_terminal_rate": float(actual_rate),
                "predicted_terminal_rate": float(predicted_rate),
                "terminal_odds_factor": factor,
                "calibration_source": "prior_walk_forward_round_survival",
            }
        )
    schedule = pd.DataFrame(schedule_rows)
    if schedule.empty or schedule.duplicated(["total_rounds", "round"]).any():
        raise FinishSurvivalCalibrationError(
            "Survival schedule is empty or contains duplicate round groups"
        )
    return schedule


def apply_finish_survival_schedule(
    counterfactual_predictions: pd.DataFrame,
    schedule: pd.DataFrame,
) -> FinishSurvivalCalibrationResult:
    """Adjust terminal mass while preserving conditional finish-class mix."""
    predictions = _validate_predictions(
        counterfactual_predictions, "Counterfactual finish predictions"
    )
    required_schedule = [
        "model_name",
        "target_year",
        "total_rounds",
        "round",
        "terminal_odds_factor",
        "calibration_source",
    ]
    missing = [column for column in required_schedule if column not in schedule.columns]
    if missing:
        raise FinishSurvivalCalibrationError(
            f"Survival schedule is missing required columns: {missing}"
        )
    schedule_frame = schedule.copy()
    for column in ("target_year", "total_rounds", "round", "terminal_odds_factor"):
        schedule_frame[column] = pd.to_numeric(
            schedule_frame[column], errors="coerce"
        )
    if schedule_frame[["total_rounds", "round", "terminal_odds_factor"]].isna().any().any():
        raise FinishSurvivalCalibrationError(
            "Survival schedule contains invalid numeric values"
        )
    schedule_frame["total_rounds"] = schedule_frame["total_rounds"].astype(int)
    schedule_frame["round"] = schedule_frame["round"].astype(int)
    if schedule_frame.duplicated(["model_name", "total_rounds", "round"]).any():
        raise FinishSurvivalCalibrationError(
            "Survival schedule contains duplicate model/round groups"
        )

    merged = predictions.merge(
        schedule_frame[
            [
                "model_name",
                "total_rounds",
                "round",
                "terminal_odds_factor",
                "calibration_source",
            ]
        ],
        on=["model_name", "total_rounds", "round"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_survival"),
    )
    if merged["terminal_odds_factor"].isna().any():
        sample = merged.loc[
            merged["terminal_odds_factor"].isna(),
            ["model_name", "total_rounds", "round"],
        ].drop_duplicates().head(10)
        raise FinishSurvivalCalibrationError(
            "Counterfactual rows are missing survival factors: "
            + sample.to_dict(orient="records").__repr__()
        )

    terminal = _terminal_probability(merged)
    factors = merged["terminal_odds_factor"].to_numpy(dtype=float)
    adjusted_terminal = np.empty(len(merged), dtype=float)
    for index, (probability, factor) in enumerate(zip(terminal, factors)):
        probability = _clip_probability(probability)
        logit = log(probability / (1.0 - probability)) + log(float(factor))
        adjusted_terminal[index] = 1.0 / (1.0 + exp(-logit))

    terminal_matrix = merged[list(TERMINAL_COLUMNS)].to_numpy(dtype=float)
    terminal_mix = terminal_matrix / np.clip(
        terminal.reshape(-1, 1), 1e-12, None
    )
    adjusted_matrix = terminal_mix * adjusted_terminal.reshape(-1, 1)
    merged["calibrated_prob_no_finish"] = 1.0 - adjusted_terminal
    for index, column in enumerate(TERMINAL_COLUMNS):
        merged[column] = adjusted_matrix[:, index]
    merged["survival_calibration_applied"] = 1
    merged["survival_calibration_source"] = merged[
        "calibration_source_survival"
    ]
    merged = merged.drop(columns=["calibration_source_survival"])

    probabilities = merged[list(CALIBRATED_COLUMNS)].to_numpy(dtype=float)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise FinishSurvivalCalibrationError(
            "Survival-calibrated probabilities violate the simplex"
        )
    return FinishSurvivalCalibrationResult(
        predictions=merged,
        schedule=schedule_frame,
    )
