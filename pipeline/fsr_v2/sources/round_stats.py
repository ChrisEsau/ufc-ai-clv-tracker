"""Round-stat loading, reciprocal pairing, and shared exposure derivation."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH
from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.sources.master import load_master

IDENTITY_COLUMNS = [
    "event_id", "event_date", "fight_id", "fighter_id", "fighter_name",
    "opponent_id", "opponent_name", "round",
]
STAT_COLUMNS = [
    "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
    "sub_att", "rev", "ctrl_sec", "head_landed", "head_attempted",
    "body_landed", "body_attempted", "leg_landed", "leg_attempted",
    "distance_landed", "distance_attempted", "clinch_landed",
    "clinch_attempted", "ground_landed", "ground_attempted",
]
ROUND_COLUMNS = IDENTITY_COLUMNS + STAT_COLUMNS


def load_round_stats(path: Path = ROUND_STATS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"FSR V2 round source is missing: {path}")
    frame = pd.read_parquet(path, columns=ROUND_COLUMNS)
    missing = set(ROUND_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Round source missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    if frame["event_date"].isna().any():
        raise ValueError("Round source contains invalid event_date values")
    key = ["fight_id", "round", "fighter_id"]
    if frame.duplicated(key).any():
        raise ValueError("Round source contains duplicate fighter-round rows")
    return frame


def _fight_elapsed_seconds(master: pd.DataFrame) -> pd.Series:
    """Normalize the observed mixed match-time semantics.

    Rows through part of the source history store cumulative fight seconds;
    newer rows can store seconds within the finishing round. A value that is
    less than or equal to 300 in round 2+ cannot be cumulative, so completed
    rounds are added. The decision is exposed in ``match_time_interpretation``.
    """
    completed = (master["finish_round"] - 1).clip(lower=0) * 300
    within_final = (master["finish_round"] > 1) & (master["match_time_sec"] <= 300)
    return master["match_time_sec"] + np.where(within_final, completed, 0)


def build_paired_rounds(
    rounds: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
    config: FSRV2Config | None = None,
) -> pd.DataFrame:
    config = config or FSRV2Config()
    rounds = load_round_stats() if rounds is None else rounds.copy()
    master = load_master() if master is None else master.copy()
    master["fight_elapsed_seconds"] = _fight_elapsed_seconds(master)
    master["match_time_interpretation"] = np.where(
        (master["finish_round"] > 1) & (master["match_time_sec"] <= 300),
        "final_round_seconds", "cumulative_seconds",
    )
    frame = rounds.merge(master, on="fight_id", how="left", validate="many_to_one")
    if frame["finish_round"].isna().any():
        raise ValueError("Master metadata is missing for round-stat fights")
    frame["round_elapsed_seconds"] = np.where(
        frame["round"] < frame["finish_round"],
        config.maximum_round_seconds,
        frame["fight_elapsed_seconds"]
        - config.maximum_round_seconds * (frame["finish_round"] - 1),
    )
    if not frame["round_elapsed_seconds"].between(1, config.maximum_round_seconds).all():
        raise ValueError("Normalized round elapsed time falls outside 1..300 seconds")

    opponent = frame[["fight_id", "round", "fighter_id"] + STAT_COLUMNS].rename(
        columns={"fighter_id": "opponent_id", **{c: f"opponent_{c}" for c in STAT_COLUMNS}}
    )
    paired = frame.merge(
        opponent, on=["fight_id", "round", "opponent_id"], how="left", validate="one_to_one"
    )
    if paired["opponent_ctrl_sec"].isna().any():
        raise ValueError("Round source contains non-reciprocal fighter/opponent rows")
    paired["combined_control_seconds_raw"] = paired["ctrl_sec"] + paired["opponent_ctrl_sec"]
    paired["ground_exposure_seconds"] = np.minimum(
        paired["round_elapsed_seconds"], paired["combined_control_seconds_raw"]
    )
    paired["standing_exposure_seconds"] = (
        paired["round_elapsed_seconds"] - paired["ground_exposure_seconds"]
    )
    paired["td_tendency_exposure_seconds"] = (
        paired["round_elapsed_seconds"] - paired["opponent_ctrl_sec"]
    ).clip(lower=0)
    paired["td_suppression_exposure_seconds"] = (
        paired["round_elapsed_seconds"] - paired["ctrl_sec"]
    ).clip(lower=0)
    paired["submission_finish"] = (
        paired["method"].str.contains("Submission", case=False, na=False)
        & paired["fighter_id"].eq(paired["winner_id"])
    )
    paired["opponent_submission_finish"] = (
        paired["method"].str.contains("Submission", case=False, na=False)
        & paired["opponent_id"].eq(paired["winner_id"])
    )
    paired["effective_submission_attempts"] = np.maximum(
        paired["sub_att"], paired["submission_finish"].astype(int)
    )
    paired["opponent_effective_submission_attempts"] = np.maximum(
        paired["opponent_sub_att"], paired["opponent_submission_finish"].astype(int)
    )
    # Qualification is shared by both reciprocal rows.  Control alone is not
    # evidence of ground position because UFCStats control includes clinch.
    paired["explicit_true_ground_evidence"] = (
        (paired["td_landed"] > 0) | (paired["opponent_td_landed"] > 0)
        | (paired["ground_attempted"] > 0) | (paired["opponent_ground_attempted"] > 0)
        | (paired["sub_att"] > 0) | (paired["opponent_sub_att"] > 0)
        | (paired["rev"] > 0) | (paired["opponent_rev"] > 0)
    )
    paired["explicit_true_ground_activity"] = (
        (paired["ground_attempted"] > 0) | (paired["opponent_ground_attempted"] > 0)
        | (paired["sub_att"] > 0) | (paired["opponent_sub_att"] > 0)
        | (paired["rev"] > 0) | (paired["opponent_rev"] > 0)
    )
    paired["qualified_control_inflicted_seconds"] = np.where(
        paired["explicit_true_ground_evidence"], paired["ctrl_sec"], 0.0
    )
    paired["qualified_control_suffered_seconds"] = np.where(
        paired["explicit_true_ground_evidence"], paired["opponent_ctrl_sec"], 0.0
    )
    paired["ground_exposure_fallback_used"] = (
        (paired["ground_exposure_seconds"] == 0) &
        paired["explicit_true_ground_activity"]
    )
    paired["modeled_ground_exposure_seconds"] = np.where(
        paired["ground_exposure_fallback_used"],
        config.zero_control_ground_fallback_seconds,
        np.where(paired["explicit_true_ground_evidence"], paired["ground_exposure_seconds"], 0.0),
    )
    paired["inferred_ground_entry"] = (
        (paired["ctrl_sec"] >= config.zero_td_control_threshold_seconds)
        & (paired["td_landed"] == 0)
        & paired["explicit_true_ground_evidence"]
    )
    paired["ground_entries"] = paired["td_landed"] + paired["inferred_ground_entry"].astype(int)
    paired["opponent_inferred_ground_entry"] = (
        (paired["opponent_ctrl_sec"] >= config.zero_td_control_threshold_seconds)
        & (paired["opponent_td_landed"] == 0)
        & paired["explicit_true_ground_evidence"]
    )
    paired["opponent_ground_entries"] = (
        paired["opponent_td_landed"] + paired["opponent_inferred_ground_entry"].astype(int)
    )
    return paired.sort_values(["event_date", "fight_id", "round", "fighter_id"]).reset_index(drop=True)
