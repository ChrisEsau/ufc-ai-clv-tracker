"""FSR V3 direct-feature construction for Event Clock MC V2.

This is the V3 analogue of Event Clock V1 ``prototype_stage1.build_feature_rows``.
It deliberately reuses the frozen V1 age/retention mechanics while changing
only the FSR semantic transforms that were revalidated in FSR V3.
"""

from __future__ import annotations

from math import exp

import numpy as np
import pandas as pd

from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    TAKEDOWN_ATTACKER_AGE_CENTER_YEARS,
    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR,
)
from pipeline.simulation.event_mc_v1.single_fight import fighter_age_years
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage1_flow_replay import (
    stage1_observed_duration_seconds,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    FighterPathTraits,
    derive_runtime_inputs,
    initialize_fighter_path_traits,
)


# Same V1 direct-model inputs where the trait survived unchanged, with
# ground_striking_defense removed and V3 ground mean-structure fields added.
FSR_V3_ATTRS = (
    "standing_striking_tendency",
    "standing_striking_suppression",
    "standing_striking_offense",
    "standing_striking_defense",
    "standing_accuracy_baseline",
    "takedown_tendency",
    "takedown_suppression",
    "takedown_offense",
    "takedown_defense",
    "takedown_completion_baseline",
    "escape_offense",
    "escape_defense",
    "escape_population_mean_seconds",
    "ground_striking_tendency",
    "ground_striking_suppression",
    "ground_striking_offense",
    "ground_accuracy_baseline",
    "ground_striking_burst_baseline",
    "ground_striking_population_slope_15m",
    "submission_tendency",
    "submission_suppression",
    "submission_offense",
    "submission_defense",
)


BASE_DIRECT_FEATURES = (
    "scheduled_rounds",
    "fighter_age",
    "opponent_age",
    "effective_standing_rate",
    "effective_td_rate",
    "effective_ground_rate",
    "ground_burst_attempts",
    "td_completion_matchup",
    "standing_accuracy_matchup",
    "ground_accuracy_matchup",
    "retention_mean_base",
    "successful_td_pressure",
    "control_pressure",
    "age_edge",
)


def direct_feature_columns_v3() -> list[str]:
    cols = list(BASE_DIRECT_FEATURES)
    for attr in FSR_V3_ATTRS:
        cols.extend((f"self_{attr}", f"opp_{attr}"))
    return cols


def _finite(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _exact_snapshot(
    fsr: pd.DataFrame,
    *,
    fight_id: str,
    event_date: pd.Timestamp,
    fighter_id: str,
    corner: str,
) -> dict:
    matched = fsr[
        fsr["fight_id"].astype(str).eq(str(fight_id))
        & fsr["event_date"].eq(event_date)
        & fsr["fighter_id"].astype(str).eq(str(fighter_id))
    ]
    if len(matched) != 1:
        raise ValueError(
            f"canonical FSR V3 must resolve exactly one {corner} row for "
            f"fight={fight_id!r}, date={event_date.date()}, fighter_id={fighter_id!r}; "
            f"found {len(matched)}"
        )
    row = matched.iloc[0].to_dict()
    if "ground_striking_defense" in row:
        raise ValueError("ground_striking_defense must not be present in FSR V3")
    return row


def _mean_path_traits(record: dict) -> FighterPathTraits:
    return initialize_fighter_path_traits(
        record,
        None,
        rng=np.random.default_rng(0),
        sample_epistemic=False,
    )


def _trait_value(record: dict, traits: FighterPathTraits, attr: str) -> float:
    if attr in traits.values:
        return _finite(traits.values[attr])
    return _finite(record.get(attr))


def _directional_feature_row(
    *,
    master_row,
    event_date: pd.Timestamp,
    duration: float,
    side: str,
    fighter_record: dict,
    fighter_traits: FighterPathTraits,
    opponent_record: dict,
    opponent_traits: FighterPathTraits,
    fighter_age: float,
    opponent_age: float,
) -> dict:
    runtime = derive_runtime_inputs(fighter_traits, opponent_traits)
    row = {
        "fight_id": str(master_row["fight_id"]),
        "event_date": event_date,
        "side": side,
        "fighter_id": str(fighter_record["fighter_id"]),
        "opponent_id": str(opponent_record["fighter_id"]),
        "fighter_name": str(fighter_record.get("fighter_name", "")),
        "opponent_name": str(opponent_record.get("fighter_name", "")),
        "duration": float(duration),
        "scheduled_rounds": float(master_row["total_rounds"]),
        "fighter_age": _finite(fighter_age),
        "opponent_age": _finite(opponent_age),
    }

    for attr in FSR_V3_ATTRS:
        row[f"self_{attr}"] = _trait_value(fighter_record, fighter_traits, attr)
        row[f"opp_{attr}"] = _trait_value(opponent_record, opponent_traits, attr)

    # Compatibility feature names retained so frozen V1 model classes can be
    # reused.  These columns have V3 semantics and require a V3-fitted bundle.
    row["effective_standing_rate"] = runtime.standing_rate_15m
    row["effective_td_rate"] = runtime.takedown_rate_15m
    row["effective_ground_rate"] = runtime.ground_slope_rate_15m_own_control
    row["ground_burst_attempts"] = runtime.ground_burst_attempts

    # Preserve exact V1 TD age translation; do not mutate persisted FSR state.
    if np.isfinite(row["fighter_age"]):
        td_age_offset = TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR * (
            row["fighter_age"] - TAKEDOWN_ATTACKER_AGE_CENTER_YEARS
        )
    else:
        td_age_offset = 0.0

    p = float(np.clip(runtime.takedown_completion, 1e-9, 1.0 - 1e-9))
    td_logit = np.log(p / (1.0 - p)) + td_age_offset
    row["td_completion_matchup"] = float(1.0 / (1.0 + np.exp(-td_logit)))
    row["standing_accuracy_matchup"] = runtime.standing_accuracy
    row["ground_accuracy_matchup"] = runtime.ground_accuracy

    bottom_population_mean = _finite(opponent_record.get("escape_population_mean_seconds"))
    bottom_escape_offense = _trait_value(opponent_record, opponent_traits, "escape_offense")
    top_escape_defense = _trait_value(fighter_record, fighter_traits, "escape_defense")
    retention_mean = bottom_population_mean * exp(
        -bottom_escape_offense + top_escape_defense
    )
    row["retention_mean_base"] = retention_mean
    row["successful_td_pressure"] = (
        row["effective_td_rate"] * row["td_completion_matchup"]
    )
    row["control_pressure"] = row["successful_td_pressure"] * retention_mean
    row["age_edge"] = (
        row["fighter_age"] - row["opponent_age"]
        if np.isfinite(row["fighter_age"]) and np.isfinite(row["opponent_age"])
        else np.nan
    )
    return row


def build_sampled_fight_feature_rows_v3(
    master_row,
    *,
    red_record: dict,
    blue_record: dict,
    red_traits: FighterPathTraits,
    blue_traits: FighterPathTraits,
) -> pd.DataFrame:
    """Build scheduled-horizon direct features from one path's fixed FSR draws."""
    event_date = pd.Timestamp(master_row["event_date"]).normalize()
    duration = float(master_row["total_rounds"]) * 300.0
    red_age = fighter_age_years(master_row.get("r_dob"), event_date)
    blue_age = fighter_age_years(master_row.get("b_dob"), event_date)
    rows = [
        _directional_feature_row(
            master_row=master_row,
            event_date=event_date,
            duration=duration,
            side="red",
            fighter_record=red_record,
            fighter_traits=red_traits,
            opponent_record=blue_record,
            opponent_traits=blue_traits,
            fighter_age=red_age,
            opponent_age=blue_age,
        ),
        _directional_feature_row(
            master_row=master_row,
            event_date=event_date,
            duration=duration,
            side="blue",
            fighter_record=blue_record,
            fighter_traits=blue_traits,
            opponent_record=red_record,
            opponent_traits=red_traits,
            fighter_age=blue_age,
            opponent_age=red_age,
        ),
    ]
    return pd.DataFrame(rows)


def build_feature_rows_v3(
    master: pd.DataFrame,
    fsr: pd.DataFrame,
    *,
    scheduled_duration: bool = False,
) -> pd.DataFrame:
    """Build one V3 direct-model row per fighter-fight.

    Training uses observed historical exposure, matching V1. Forward target
    construction passes ``scheduled_duration=True`` and never uses finish time.
    """
    frame = fsr.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    rows: list[dict] = []
    skipped = 0

    for _, master_row in master.iterrows():
        fight_id = str(master_row["fight_id"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        try:
            red_record = _exact_snapshot(
                frame, fight_id=fight_id, event_date=event_date,
                fighter_id=str(master_row["r_id"]), corner="red",
            )
            blue_record = _exact_snapshot(
                frame, fight_id=fight_id, event_date=event_date,
                fighter_id=str(master_row["b_id"]), corner="blue",
            )
            red_traits = _mean_path_traits(red_record)
            blue_traits = _mean_path_traits(blue_record)
        except Exception:
            skipped += 1
            continue

        duration = (
            float(master_row["total_rounds"]) * 300.0
            if scheduled_duration
            else float(stage1_observed_duration_seconds(master_row))
        )
        red_age = fighter_age_years(master_row.get("r_dob"), event_date)
        blue_age = fighter_age_years(master_row.get("b_dob"), event_date)
        rows.extend(
            [
                _directional_feature_row(
                    master_row=master_row,
                    event_date=event_date,
                    duration=duration,
                    side="red",
                    fighter_record=red_record,
                    fighter_traits=red_traits,
                    opponent_record=blue_record,
                    opponent_traits=blue_traits,
                    fighter_age=red_age,
                    opponent_age=blue_age,
                ),
                _directional_feature_row(
                    master_row=master_row,
                    event_date=event_date,
                    duration=duration,
                    side="blue",
                    fighter_record=blue_record,
                    fighter_traits=blue_traits,
                    opponent_record=red_record,
                    opponent_traits=red_traits,
                    fighter_age=blue_age,
                    opponent_age=red_age,
                ),
            ]
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        missing = [
            column for column in direct_feature_columns_v3()
            if column not in result.columns
        ]
        if missing:
            raise RuntimeError(f"V3 direct feature builder missing columns: {missing}")
    print(
        f"V3 feature fights built: {result['fight_id'].nunique() if not result.empty else 0} | "
        f"fighter-fight rows: {len(result)} | skipped fights: {skipped}"
    )
    return result
