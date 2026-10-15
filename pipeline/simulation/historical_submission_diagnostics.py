"""Historical submission-side and grappling failure diagnostics.

This module is evaluation-only. It combines three leakage-safe research outputs:

1. full-fight simulator probabilities;
2. counterfactual round-level finish-provider hazards;
3. shifted pre-fight fighter states reconstructed from completed prior fights.

Realized winner, method, and finish round are joined only after all probabilities
and pre-fight states have been constructed. No simulator, model, feature schema,
or production path is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.simulation.finish_hazard_model import FINISH_CLASSES
from pipeline.simulation.historical_simulator_replay import (
    build_fighter_fight_history,
    build_holdout_matchups,
    population_priors,
)


class HistoricalSubmissionDiagnosticError(RuntimeError):
    """Raised when submission diagnostic inputs violate the audit contract."""


@dataclass(frozen=True)
class HistoricalSubmissionDiagnosticResult:
    fight_diagnostics: pd.DataFrame
    error_classes: pd.DataFrame
    calibration: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    summary: Mapping[str, object]


PROBABILITY_COLUMNS = tuple(
    f"calibrated_prob_{name}" for name in FINISH_CLASSES
)
SIMULATOR_REQUIRED_COLUMNS = (
    "fight_id",
    "scheduled_rounds",
    "red_fighter_id",
    "blue_fighter_id",
    "red_prior_fights",
    "blue_prior_fights",
    "actual_winner_corner",
    "actual_method",
    "sim_red_win_probability",
    "sim_decision_probability",
    "sim_ko_tko_probability",
    "sim_submission_probability",
)
FINISH_REQUIRED_COLUMNS = (
    "fight_id",
    "round",
    "total_rounds",
    *PROBABILITY_COLUMNS,
)


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HistoricalSubmissionDiagnosticError(
            f"{label} is missing required columns: {missing}"
        )


def _normalize_probability_frame(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    _require_columns(frame, FINISH_REQUIRED_COLUMNS, label)
    out = frame.copy()
    out["fight_id"] = out["fight_id"].astype(str)
    for column in ("round", "total_rounds", *PROBABILITY_COLUMNS):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["round", "total_rounds", *PROBABILITY_COLUMNS]].isna().any().any():
        raise HistoricalSubmissionDiagnosticError(
            f"{label} contains missing probability or round values"
        )
    out["round"] = out["round"].astype(int)
    out["total_rounds"] = out["total_rounds"].astype(int)
    if out.duplicated(["fight_id", "round"]).any():
        raise HistoricalSubmissionDiagnosticError(
            f"{label} contains duplicate fight-round keys"
        )
    probabilities = out[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise HistoricalSubmissionDiagnosticError(
            f"{label} probability rows must be nonnegative and sum to one"
        )
    return out


def aggregate_finish_hazards(
    finish_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Convert conditional round hazards into unconditional fight probabilities."""
    frame = _normalize_probability_frame(
        finish_predictions,
        "Survival finish predictions",
    )
    rows: list[dict[str, object]] = []
    for fight_id, group in frame.groupby("fight_id", sort=False):
        ordered = group.sort_values("round").reset_index(drop=True)
        total_rounds_values = ordered["total_rounds"].unique()
        if len(total_rounds_values) != 1:
            raise HistoricalSubmissionDiagnosticError(
                f"Fight {fight_id} has inconsistent scheduled rounds"
            )
        total_rounds = int(total_rounds_values[0])
        expected_rounds = list(range(1, total_rounds + 1))
        actual_rounds = ordered["round"].astype(int).tolist()
        if actual_rounds != expected_rounds:
            raise HistoricalSubmissionDiagnosticError(
                f"Fight {fight_id} has incomplete counterfactual round coverage: "
                f"{actual_rounds!r}"
            )

        survival_before = 1.0
        masses = {
            "red_ko_tko": 0.0,
            "red_submission": 0.0,
            "blue_ko_tko": 0.0,
            "blue_submission": 0.0,
        }
        submission_round_mass: list[tuple[int, float]] = []
        for row in ordered.itertuples(index=False):
            no_finish = float(getattr(row, "calibrated_prob_no_finish"))
            red_ko = survival_before * float(
                getattr(row, "calibrated_prob_red_ko_tko")
            )
            red_sub = survival_before * float(
                getattr(row, "calibrated_prob_red_submission")
            )
            blue_ko = survival_before * float(
                getattr(row, "calibrated_prob_blue_ko_tko")
            )
            blue_sub = survival_before * float(
                getattr(row, "calibrated_prob_blue_submission")
            )
            masses["red_ko_tko"] += red_ko
            masses["red_submission"] += red_sub
            masses["blue_ko_tko"] += blue_ko
            masses["blue_submission"] += blue_sub
            submission_round_mass.append((int(row.round), red_sub + blue_sub))
            survival_before *= no_finish

        decision = float(survival_before)
        total_probability = decision + sum(masses.values())
        if not np.isclose(total_probability, 1.0, atol=1e-6):
            raise HistoricalSubmissionDiagnosticError(
                f"Unconditional finish mass for {fight_id} sums to {total_probability}"
            )

        total_submission = float(
            masses["red_submission"] + masses["blue_submission"]
        )
        if total_submission > 1e-12:
            red_submission_share = float(
                masses["red_submission"] / total_submission
            )
            expected_submission_round = float(
                sum(round_number * mass for round_number, mass in submission_round_mass)
                / total_submission
            )
            peak_submission_round = int(
                max(submission_round_mass, key=lambda value: value[1])[0]
            )
        else:
            red_submission_share = 0.5
            expected_submission_round = np.nan
            peak_submission_round = 1

        identity = ordered.iloc[0]
        output: dict[str, object] = {
            "fight_id": str(fight_id),
            "provider_total_rounds": total_rounds,
            "provider_decision_probability": decision,
            "provider_red_ko_tko_probability": masses["red_ko_tko"],
            "provider_blue_ko_tko_probability": masses["blue_ko_tko"],
            "provider_red_submission_probability": masses["red_submission"],
            "provider_blue_submission_probability": masses["blue_submission"],
            "provider_total_submission_probability": total_submission,
            "provider_conditional_red_submission_share": red_submission_share,
            "provider_expected_submission_round": expected_submission_round,
            "provider_peak_submission_round": peak_submission_round,
        }
        for column in (
            "event_id",
            "event_name",
            "date",
            "division",
            "title_fight",
            "red_fighter_id",
            "blue_fighter_id",
            "model_name",
            "model_version",
        ):
            if column in ordered.columns:
                output[f"provider_{column}"] = identity[column]
        rows.append(output)

    result = pd.DataFrame(rows)
    if result.empty or result["fight_id"].duplicated().any():
        raise HistoricalSubmissionDiagnosticError(
            "Finish hazard aggregation produced an empty or duplicated fight table"
        )
    return result


def _actual_finish_rounds(training_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(training_df, ("fight_id", "round"), "Training table")
    frame = training_df[["fight_id", "round"]].copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    if frame["round"].isna().any():
        raise HistoricalSubmissionDiagnosticError(
            "Training table contains invalid round values"
        )
    return (
        frame.groupby("fight_id", as_index=False)["round"]
        .max()
        .rename(columns={"round": "actual_finish_round"})
        .astype({"actual_finish_round": int})
    )


def _holdout_state_frame(
    training_df: pd.DataFrame,
    fight_ids: set[str],
    test_year: int,
) -> pd.DataFrame:
    history = build_fighter_fight_history(training_df)
    priors = population_priors(history, test_year=test_year)
    records = build_holdout_matchups(
        history,
        test_year=test_year,
        priors=priors,
    )
    history_lookup = history.loc[
        history["date"].dt.year.eq(int(test_year))
        & history["fight_id"].astype(str).isin(fight_ids)
    ].copy()
    identity_by_fight: dict[str, pd.Series] = {}
    for fight_id, group in history_lookup.groupby("fight_id", sort=False):
        identity_by_fight[str(fight_id)] = group.iloc[0]

    rows: list[dict[str, object]] = []
    for record in records:
        matchup = record["matchup"]
        fight_id = str(matchup.fight_id)
        if fight_id not in fight_ids:
            continue
        identity = identity_by_fight.get(fight_id)
        row: dict[str, object] = {
            "fight_id": fight_id,
            "state_red_submission_threat": matchup.red.submission_threat,
            "state_blue_submission_threat": matchup.blue.submission_threat,
            "state_red_submission_defense": matchup.red.submission_defense,
            "state_blue_submission_defense": matchup.blue.submission_defense,
            "state_red_td_attempts_per_15": matchup.red.td_attempts_per_15,
            "state_blue_td_attempts_per_15": matchup.blue.td_attempts_per_15,
            "state_red_td_accuracy": matchup.red.td_accuracy,
            "state_blue_td_accuracy": matchup.blue.td_accuracy,
            "state_red_td_defense": matchup.red.td_defense,
            "state_blue_td_defense": matchup.blue.td_defense,
            "state_red_control_seconds_per_takedown": (
                matchup.red.control_seconds_per_takedown
            ),
            "state_blue_control_seconds_per_takedown": (
                matchup.blue.control_seconds_per_takedown
            ),
            "state_red_phase_imposition": matchup.red.phase_imposition,
            "state_blue_phase_imposition": matchup.blue.phase_imposition,
        }
        if identity is not None:
            for column in ("division", "title_fight", "event_name"):
                if column in identity.index:
                    row[column] = identity[column]
        rows.append(row)

    state = pd.DataFrame(rows)
    resolved = set(state.get("fight_id", pd.Series(dtype=str)).astype(str))
    if resolved != fight_ids:
        missing = sorted(fight_ids - resolved)
        extra = sorted(resolved - fight_ids)
        raise HistoricalSubmissionDiagnosticError(
            f"Holdout state alignment failed; missing={missing[:10]}, extra={extra[:10]}"
        )
    if state["fight_id"].duplicated().any():
        raise HistoricalSubmissionDiagnosticError(
            "Holdout state table contains duplicate fights"
        )
    return state


def _experience_bucket(red_prior: int, blue_prior: int) -> str:
    minimum = min(int(red_prior), int(blue_prior))
    if minimum <= 0:
        return "0_cold_start"
    if minimum <= 2:
        return "1_2_prior"
    if minimum <= 5:
        return "3_5_prior"
    return "6_plus_prior"


def _predicted_method(row: pd.Series) -> str:
    probabilities = {
        "decision": float(row["sim_decision_probability"]),
        "ko_tko": float(row["sim_ko_tko_probability"]),
        "submission": float(row["sim_submission_probability"]),
    }
    return max(probabilities, key=probabilities.get)


def _normalize_title(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "title"
    if text in {"false", "0", "no", "n"}:
        return "non_title"
    return "unknown"


def _binary_log_loss(actual: np.ndarray, predicted: np.ndarray) -> float:
    clipped = np.clip(predicted.astype(float), 1e-9, 1.0 - 1e-9)
    targets = actual.astype(float)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    )


def _calibration_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_bins = pd.cut(
        diagnostics["provider_total_submission_probability"],
        bins=[-1e-12, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0],
        labels=["0_10", "10_20", "20_30", "30_40", "40_50", "50_100"],
        include_lowest=True,
    )
    total = diagnostics.assign(_band=total_bins)
    for band, group in total.groupby("_band", observed=True, sort=True):
        rows.append(
            {
                "calibration_type": "total_submission_probability",
                "band": str(band),
                "fights": int(len(group)),
                "mean_probability": float(
                    group["provider_total_submission_probability"].mean()
                ),
                "observed_rate": float(group["actual_is_submission"].mean()),
                "absolute_gap": float(
                    abs(
                        group["provider_total_submission_probability"].mean()
                        - group["actual_is_submission"].mean()
                    )
                ),
            }
        )

    submissions = diagnostics.loc[diagnostics["actual_is_submission"].eq(1)].copy()
    if not submissions.empty:
        side_bins = pd.cut(
            submissions["provider_conditional_red_submission_share"],
            bins=[-1e-12, 0.35, 0.45, 0.55, 0.65, 1.0],
            labels=["0_35", "35_45", "45_55", "55_65", "65_100"],
            include_lowest=True,
        )
        side = submissions.assign(_band=side_bins)
        for band, group in side.groupby("_band", observed=True, sort=True):
            rows.append(
                {
                    "calibration_type": "conditional_red_submission_share",
                    "band": str(band),
                    "fights": int(len(group)),
                    "mean_probability": float(
                        group["provider_conditional_red_submission_share"].mean()
                    ),
                    "observed_rate": float(
                        group["actual_winner_corner"].eq("red").mean()
                    ),
                    "absolute_gap": float(
                        abs(
                            group[
                                "provider_conditional_red_submission_share"
                            ].mean()
                            - group["actual_winner_corner"].eq("red").mean()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _subgroup_metrics(
    diagnostics: pd.DataFrame,
    minimum_group_size: int,
) -> pd.DataFrame:
    submissions = diagnostics.loc[diagnostics["actual_is_submission"].eq(1)].copy()
    segments = (
        "error_class",
        "experience_bucket",
        "scheduled_rounds",
        "division_segment",
        "title_segment",
        "actual_winner_corner",
        "predicted_method",
        "side_winner_alignment",
    )
    rows: list[dict[str, object]] = []
    for segment in segments:
        for subgroup, group in submissions.groupby(segment, dropna=False, sort=True):
            expected_round = group["provider_expected_submission_round"]
            valid_round = expected_round.notna()
            rows.append(
                {
                    "segment": segment,
                    "subgroup": str(subgroup),
                    "fights": int(len(group)),
                    "meets_minimum_group_size": bool(
                        len(group) >= int(minimum_group_size)
                    ),
                    "submission_method_detection_rate": float(
                        group["method_correct"].mean()
                    ),
                    "submission_side_accuracy": float(
                        group["submission_side_correct"].mean()
                    ),
                    "simulator_winner_accuracy": float(
                        group["winner_correct"].mean()
                    ),
                    "mean_total_submission_probability": float(
                        group["provider_total_submission_probability"].mean()
                    ),
                    "mean_actual_winner_conditional_submission_share": float(
                        group[
                            "provider_actual_winner_conditional_submission_share"
                        ].mean()
                    ),
                    "expected_submission_round_mae": (
                        float(
                            np.mean(
                                np.abs(
                                    expected_round.loc[valid_round].to_numpy(dtype=float)
                                    - group.loc[
                                        valid_round, "actual_finish_round"
                                    ].to_numpy(dtype=float)
                                )
                            )
                        )
                        if valid_round.any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _recommendation(summary: Mapping[str, float]) -> tuple[str, str]:
    fights = int(summary["actual_submission_fights"])
    method_detection = float(summary["submission_method_detection_rate"])
    side_accuracy = float(summary["submission_side_accuracy"])
    winner_accuracy = float(summary["simulator_winner_accuracy_on_submissions"])
    disagreement = float(summary["submission_side_vs_simulator_winner_disagreement_rate"])

    if fights < 30:
        return (
            "insufficient_submission_sample",
            "Expand the corrected holdout before selecting a new submission component.",
        )
    if method_detection < 0.45 and side_accuracy < 0.55:
        return (
            "hierarchical_finish_model",
            "Both submission detection and conditional red/blue allocation are weak; "
            "separate decision-vs-finish, method, and side stages before adding new "
            "grappling mechanics.",
        )
    if method_detection < 0.45:
        return (
            "hierarchical_method_detection",
            "Conditional side allocation is more reliable than submission detection; "
            "separate finish type from finish side.",
        )
    if side_accuracy < 0.55:
        return (
            "submission_side_allocation_model",
            "Total submission recognition is usable but red/blue allocation is weak; "
            "fit a dedicated conditional submission-side model.",
        )
    if side_accuracy - winner_accuracy >= 0.10 or disagreement >= 0.25:
        return (
            "stateful_grappling_and_scoring_provider",
            "The finish provider identifies the submission side materially better than "
            "the simulator identifies the overall winner; audit takedown, control, "
            "grappling scoring, and decision interaction next.",
        )
    return (
        "submission_probability_calibration",
        "Method and side ordering are broadly usable; focus on calibration and finish-round timing.",
    )


def audit_submission_failures(
    simulator_predictions: pd.DataFrame,
    survival_finish_predictions: pd.DataFrame,
    training_df: pd.DataFrame,
    test_year: int = 2026,
    minimum_group_size: int = 10,
) -> HistoricalSubmissionDiagnosticResult:
    """Diagnose submission detection, side allocation, and winner interaction."""
    if minimum_group_size <= 0:
        raise HistoricalSubmissionDiagnosticError(
            "minimum_group_size must be positive"
        )
    _require_columns(
        simulator_predictions,
        SIMULATOR_REQUIRED_COLUMNS,
        "Simulator predictions",
    )
    simulator = simulator_predictions.copy()
    simulator["fight_id"] = simulator["fight_id"].astype(str)
    if simulator.empty or simulator["fight_id"].duplicated().any():
        raise HistoricalSubmissionDiagnosticError(
            "Simulator predictions must contain unique fight rows"
        )
    numeric_columns = (
        "scheduled_rounds",
        "red_prior_fights",
        "blue_prior_fights",
        "sim_red_win_probability",
        "sim_decision_probability",
        "sim_ko_tko_probability",
        "sim_submission_probability",
    )
    for column in numeric_columns:
        simulator[column] = pd.to_numeric(simulator[column], errors="coerce")
    if simulator[list(numeric_columns)].isna().any().any():
        raise HistoricalSubmissionDiagnosticError(
            "Simulator predictions contain invalid numeric values"
        )

    provider = aggregate_finish_hazards(survival_finish_predictions)
    simulator_fights = set(simulator["fight_id"])
    provider_fights = set(provider["fight_id"])
    if simulator_fights != provider_fights:
        raise HistoricalSubmissionDiagnosticError(
            "Simulator and finish-provider fight sets do not match; "
            f"simulator_only={sorted(simulator_fights - provider_fights)[:10]}, "
            f"provider_only={sorted(provider_fights - simulator_fights)[:10]}"
        )

    state = _holdout_state_frame(
        training_df,
        fight_ids=simulator_fights,
        test_year=test_year,
    )
    finish_round = _actual_finish_rounds(training_df)
    diagnostics = simulator.merge(
        provider,
        on="fight_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        state,
        on="fight_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        finish_round,
        on="fight_id",
        how="left",
        validate="one_to_one",
    )
    if diagnostics["actual_finish_round"].isna().any():
        raise HistoricalSubmissionDiagnosticError(
            "Actual finish round was unavailable for one or more audit fights"
        )
    diagnostics["actual_finish_round"] = diagnostics[
        "actual_finish_round"
    ].astype(int)

    diagnostics["actual_is_submission"] = diagnostics["actual_method"].astype(
        str
    ).eq("submission").astype(int)
    diagnostics["predicted_method"] = diagnostics.apply(_predicted_method, axis=1)
    diagnostics["predicted_submission_side"] = np.where(
        diagnostics["provider_red_submission_probability"].ge(
            diagnostics["provider_blue_submission_probability"]
        ),
        "red",
        "blue",
    )
    diagnostics["predicted_winner_corner"] = np.where(
        diagnostics["sim_red_win_probability"].ge(0.5),
        "red",
        "blue",
    )
    diagnostics["method_correct"] = diagnostics["predicted_method"].eq(
        "submission"
    )
    diagnostics["submission_side_correct"] = diagnostics[
        "predicted_submission_side"
    ].eq(diagnostics["actual_winner_corner"])
    diagnostics["winner_correct"] = diagnostics["predicted_winner_corner"].eq(
        diagnostics["actual_winner_corner"]
    )
    diagnostics["side_winner_alignment"] = np.where(
        diagnostics["predicted_submission_side"].eq(
            diagnostics["predicted_winner_corner"]
        ),
        "aligned",
        "disagree",
    )
    diagnostics["experience_bucket"] = [
        _experience_bucket(red, blue)
        for red, blue in zip(
            diagnostics["red_prior_fights"],
            diagnostics["blue_prior_fights"],
        )
    ]
    diagnostics["division_segment"] = diagnostics.get(
        "division", pd.Series("unknown", index=diagnostics.index)
    ).fillna("unknown").astype(str)
    diagnostics["title_segment"] = diagnostics.get(
        "title_fight", pd.Series(np.nan, index=diagnostics.index)
    ).map(_normalize_title)

    red_actual_winner = diagnostics["actual_winner_corner"].eq("red")
    diagnostics["provider_actual_winner_submission_probability"] = np.where(
        red_actual_winner,
        diagnostics["provider_red_submission_probability"],
        diagnostics["provider_blue_submission_probability"],
    )
    diagnostics["provider_actual_loser_submission_probability"] = np.where(
        red_actual_winner,
        diagnostics["provider_blue_submission_probability"],
        diagnostics["provider_red_submission_probability"],
    )
    diagnostics[
        "provider_actual_winner_conditional_submission_share"
    ] = np.where(
        red_actual_winner,
        diagnostics["provider_conditional_red_submission_share"],
        1.0 - diagnostics["provider_conditional_red_submission_share"],
    )
    diagnostics["provider_simulator_submission_probability_gap"] = (
        diagnostics["provider_total_submission_probability"]
        - diagnostics["sim_submission_probability"]
    )

    for name in (
        "submission_threat",
        "submission_defense",
        "td_attempts_per_15",
        "td_accuracy",
        "td_defense",
        "control_seconds_per_takedown",
        "phase_imposition",
    ):
        red_column = f"state_red_{name}"
        blue_column = f"state_blue_{name}"
        diagnostics[f"state_actual_winner_{name}_advantage"] = np.where(
            red_actual_winner,
            diagnostics[red_column] - diagnostics[blue_column],
            diagnostics[blue_column] - diagnostics[red_column],
        )

    submissions = diagnostics["actual_is_submission"].eq(1)
    method_correct = diagnostics["method_correct"]
    side_correct = diagnostics["submission_side_correct"]
    diagnostics["error_class"] = "not_actual_submission"
    diagnostics.loc[
        submissions & method_correct & side_correct,
        "error_class",
    ] = "correct_method_correct_side"
    diagnostics.loc[
        submissions & method_correct & ~side_correct,
        "error_class",
    ] = "correct_method_wrong_side"
    diagnostics.loc[
        submissions & ~method_correct & side_correct,
        "error_class",
    ] = "wrong_method_correct_side"
    diagnostics.loc[
        submissions & ~method_correct & ~side_correct,
        "error_class",
    ] = "wrong_method_wrong_side"

    submission_rows = diagnostics.loc[submissions].copy()
    if submission_rows.empty:
        raise HistoricalSubmissionDiagnosticError(
            "The audit cohort contains no actual submission fights"
        )
    error_classes = (
        submission_rows["error_class"]
        .value_counts(dropna=False)
        .rename_axis("error_class")
        .reset_index(name="fights")
    )
    error_classes["rate"] = error_classes["fights"] / len(submission_rows)

    all_actual = diagnostics["actual_is_submission"].to_numpy(dtype=float)
    all_probability = diagnostics[
        "provider_total_submission_probability"
    ].to_numpy(dtype=float)
    side_actual_red = submission_rows["actual_winner_corner"].eq("red").to_numpy(
        dtype=float
    )
    side_probability = submission_rows[
        "provider_conditional_red_submission_share"
    ].to_numpy(dtype=float)
    valid_expected_round = submission_rows[
        "provider_expected_submission_round"
    ].notna()
    expected_round_mae = (
        float(
            np.mean(
                np.abs(
                    submission_rows.loc[
                        valid_expected_round,
                        "provider_expected_submission_round",
                    ].to_numpy(dtype=float)
                    - submission_rows.loc[
                        valid_expected_round,
                        "actual_finish_round",
                    ].to_numpy(dtype=float)
                )
            )
        )
        if valid_expected_round.any()
        else np.nan
    )

    metric_summary: dict[str, float] = {
        "cohort_fights": int(len(diagnostics)),
        "actual_submission_fights": int(len(submission_rows)),
        "actual_submission_rate": float(diagnostics["actual_is_submission"].mean()),
        "provider_mean_submission_probability": float(
            diagnostics["provider_total_submission_probability"].mean()
        ),
        "provider_submission_brier": float(
            np.mean(np.square(all_probability - all_actual))
        ),
        "provider_submission_binary_log_loss": _binary_log_loss(
            all_actual,
            all_probability,
        ),
        "submission_method_detection_rate": float(
            submission_rows["method_correct"].mean()
        ),
        "submission_side_accuracy": float(
            submission_rows["submission_side_correct"].mean()
        ),
        "submission_side_brier": float(
            np.mean(np.square(side_probability - side_actual_red))
        ),
        "submission_side_binary_log_loss": _binary_log_loss(
            side_actual_red,
            side_probability,
        ),
        "simulator_winner_accuracy_on_submissions": float(
            submission_rows["winner_correct"].mean()
        ),
        "submission_side_vs_simulator_winner_disagreement_rate": float(
            submission_rows["side_winner_alignment"].eq("disagree").mean()
        ),
        "provider_simulator_submission_probability_gap_mae": float(
            diagnostics[
                "provider_simulator_submission_probability_gap"
            ].abs().mean()
        ),
        "expected_submission_round_mae": expected_round_mae,
    }
    recommended_action, rationale = _recommendation(metric_summary)
    summary: dict[str, object] = {
        "status": "evaluation_only",
        "test_year": int(test_year),
        "diagnostic_target": "submission_side_and_grappling_failure",
        "probability_source": "counterfactual_round_survival_finish_provider",
        "state_source": "shifted_prefight_historical_fighter_state",
        "realized_labels_role": "scoring_only",
        "metrics": metric_summary,
        "error_classes": error_classes.to_dict(orient="records"),
        "recommended_next_component": recommended_action,
        "recommendation_rationale": rationale,
    }
    sort_columns = [
        column for column in ("date", "fight_id") if column in diagnostics.columns
    ]
    return HistoricalSubmissionDiagnosticResult(
        fight_diagnostics=diagnostics.sort_values(sort_columns).reset_index(drop=True),
        error_classes=error_classes,
        calibration=_calibration_table(diagnostics),
        subgroup_metrics=_subgroup_metrics(
            diagnostics,
            minimum_group_size=minimum_group_size,
        ),
        summary=summary,
    )
