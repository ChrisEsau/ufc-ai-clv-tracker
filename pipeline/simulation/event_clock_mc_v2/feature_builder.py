"""FSR V3 direct-feature construction for Event Clock MC V2."""
from __future__ import annotations

from math import exp
import numpy as np
import pandas as pd

from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    TAKEDOWN_ATTACKER_AGE_CENTER_YEARS,
    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR,
)
from pipeline.simulation.event_mc_v1.single_fight import fighter_age_years
from pipeline.simulation.event_mc_v1.diagnostics.stage1_flow_replay import (
    stage1_observed_duration_seconds,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    FighterPathTraits,
    derive_runtime_inputs,
    initialize_fighter_path_traits,
)
from pipeline.simulation.event_clock_mc_v2.reach_translation import (
    directional_reach_inputs,
)

FSR_V3_ATTRS = (
    "standing_striking_tendency", "standing_striking_suppression",
    "standing_striking_offense", "standing_striking_defense",
    "standing_accuracy_baseline", "takedown_tendency", "takedown_suppression",
    "takedown_offense", "takedown_defense", "takedown_completion_baseline",
    "escape_offense", "escape_defense", "escape_population_mean_seconds",
    "ground_striking_tendency", "ground_striking_suppression",
    "ground_striking_offense", "ground_accuracy_baseline",
    "ground_striking_burst_baseline", "ground_striking_population_slope_15m",
    "submission_tendency", "submission_suppression", "submission_offense",
    "submission_defense",
)
BASE_DIRECT_FEATURES = (
    "scheduled_rounds", "fighter_age", "opponent_age",
    "effective_standing_rate", "effective_td_rate", "effective_ground_rate",
    "ground_burst_attempts", "td_completion_matchup",
    "standing_accuracy_matchup", "ground_accuracy_matchup",
    "retention_mean_base", "successful_td_pressure", "control_pressure", "age_edge",
)


def direct_feature_columns_v3():
    cols = list(BASE_DIRECT_FEATURES)
    for attr in FSR_V3_ATTRS:
        cols += [f"self_{attr}", f"opp_{attr}"]
    return cols


def _finite(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _exact_snapshot(fsr, fight_id, event_date, fighter_id, corner):
    matched = fsr[
        fsr["fight_id"].astype(str).eq(str(fight_id))
        & fsr["event_date"].eq(event_date)
        & fsr["fighter_id"].astype(str).eq(str(fighter_id))
    ]
    if len(matched) != 1:
        raise ValueError(
            f"FSR V3 expected one {corner} row for fight={fight_id}, fighter={fighter_id}; "
            f"found {len(matched)}"
        )
    row = matched.iloc[0].to_dict()
    if "ground_striking_defense" in row:
        raise ValueError("ground_striking_defense is rejected in FSR V3")
    return row


def _mean_traits(record):
    return initialize_fighter_path_traits(
        record, None, rng=np.random.default_rng(0), sample_epistemic=False
    )


def _trait_value(record, traits, attr):
    return _finite(traits.values[attr] if attr in traits.values else record.get(attr))


def _directional_row(master_row, event_date, duration, side, fighter, fighter_traits,
                     opponent, opponent_traits, fighter_age, opponent_age):
    runtime = derive_runtime_inputs(fighter_traits, opponent_traits)
    row = {
        "fight_id": str(master_row["fight_id"]), "event_date": event_date,
        "side": side, "fighter_id": str(fighter["fighter_id"]),
        "opponent_id": str(opponent["fighter_id"]),
        "fighter_name": str(fighter.get("fighter_name", "")),
        "opponent_name": str(opponent.get("fighter_name", "")),
        "duration": float(duration), "scheduled_rounds": float(master_row["total_rounds"]),
        "fighter_age": _finite(fighter_age), "opponent_age": _finite(opponent_age),
    }
    # Reach is a fight-specific physical matchup translation, not an FSR trait
    # and not a fitted direct-model feature.  We carry it alongside the model
    # features so inference can apply the validated post-model distance-volume
    # adjustment without altering persisted FSR state or the frozen model schema.
    row.update(directional_reach_inputs(master_row, side))

    for attr in FSR_V3_ATTRS:
        row[f"self_{attr}"] = _trait_value(fighter, fighter_traits, attr)
        row[f"opp_{attr}"] = _trait_value(opponent, opponent_traits, attr)

    row["effective_standing_rate"] = runtime.standing_rate_15m
    row["effective_td_rate"] = runtime.takedown_rate_15m
    row["effective_ground_rate"] = runtime.ground_slope_rate_15m_own_control
    row["ground_burst_attempts"] = runtime.ground_burst_attempts

    age_offset = 0.0
    if np.isfinite(row["fighter_age"]):
        age_offset = TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR * (
            row["fighter_age"] - TAKEDOWN_ATTACKER_AGE_CENTER_YEARS
        )
    p = np.clip(runtime.takedown_completion, 1e-9, 1 - 1e-9)
    td_logit = np.log(p / (1 - p)) + age_offset
    row["td_completion_matchup"] = float(1 / (1 + np.exp(-td_logit)))
    row["standing_accuracy_matchup"] = runtime.standing_accuracy
    row["ground_accuracy_matchup"] = runtime.ground_accuracy

    retention = _finite(opponent.get("escape_population_mean_seconds")) * exp(
        -_trait_value(opponent, opponent_traits, "escape_offense")
        + _trait_value(fighter, fighter_traits, "escape_defense")
    )
    row["retention_mean_base"] = retention
    row["successful_td_pressure"] = row["effective_td_rate"] * row["td_completion_matchup"]
    row["control_pressure"] = row["successful_td_pressure"] * retention
    row["age_edge"] = (
        row["fighter_age"] - row["opponent_age"]
        if np.isfinite(row["fighter_age"]) and np.isfinite(row["opponent_age"])
        else np.nan
    )
    return row


def build_sampled_fight_feature_rows_v3(master_row, *, red_record, blue_record,
                                         red_traits: FighterPathTraits,
                                         blue_traits: FighterPathTraits):
    event_date = pd.Timestamp(master_row["event_date"]).normalize()
    duration = float(master_row["total_rounds"]) * 300.0
    red_age = fighter_age_years(master_row.get("r_dob"), event_date)
    blue_age = fighter_age_years(master_row.get("b_dob"), event_date)
    return pd.DataFrame([
        _directional_row(master_row, event_date, duration, "red", red_record, red_traits,
                         blue_record, blue_traits, red_age, blue_age),
        _directional_row(master_row, event_date, duration, "blue", blue_record, blue_traits,
                         red_record, red_traits, blue_age, red_age),
    ])


def build_feature_rows_v3(master, fsr, *, scheduled_duration=False):
    frame = fsr.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    rows, skipped = [], 0
    for _, master_row in master.iterrows():
        fight_id = str(master_row["fight_id"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        try:
            red = _exact_snapshot(frame, fight_id, event_date, master_row["r_id"], "red")
            blue = _exact_snapshot(frame, fight_id, event_date, master_row["b_id"], "blue")
            red_traits, blue_traits = _mean_traits(red), _mean_traits(blue)
        except Exception:
            skipped += 1
            continue
        duration = (
            float(master_row["total_rounds"]) * 300.0 if scheduled_duration
            else float(stage1_observed_duration_seconds(master_row))
        )
        red_age = fighter_age_years(master_row.get("r_dob"), event_date)
        blue_age = fighter_age_years(master_row.get("b_dob"), event_date)
        rows += [
            _directional_row(master_row, event_date, duration, "red", red, red_traits,
                             blue, blue_traits, red_age, blue_age),
            _directional_row(master_row, event_date, duration, "blue", blue, blue_traits,
                             red, red_traits, blue_age, red_age),
        ]
    result = pd.DataFrame(rows)
    if not result.empty:
        missing = [c for c in direct_feature_columns_v3() if c not in result.columns]
        if missing:
            raise RuntimeError(f"V3 feature builder missing columns: {missing}")
    print(
        f"V3 feature fights built: {result['fight_id'].nunique() if not result.empty else 0} | "
        f"fighter-fight rows: {len(result)} | skipped fights: {skipped}"
    )
    return result
