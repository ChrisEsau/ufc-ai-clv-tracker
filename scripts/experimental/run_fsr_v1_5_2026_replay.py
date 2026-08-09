"""Large-cohort historical replay for the provisional FSR/MC V1.5 stack.

Shadow/research only.

This script evaluates the current V1.5 simulator on completed fights from one
holdout year (2026 by default).  It is deliberately separate from production
prediction and wagering paths.

Leakage boundary
----------------
- Locked FSR V1.1 ratings are replayed chronologically and snapshotted BEFORE
  each target fight. Same-date updates are applied simultaneously.
- RFS Phase Baseline / Dynamic Response inputs come from the existing
  leakage-safe PRE-fight state row for that target fight.
- Population calibration for cardio/activity uses only rows dated strictly
  before the target fight.
- Actual winner, method, finish time, and target-fight statistics are attached
  only after simulator inputs have been constructed.

Current simulator checkpoint
----------------------------
- V1.4 style-aware phase transitions.
- x1.75 two-stage multi-attempt takedown chains.
- V1.5 fighter-specific distance/clinch/ground activity style.
- V1.5 control-seconds tendency with FSR opponent adjustment.
- V1.3 KO/TKO hazards.
- Submission attempt generation 1.00x and base hazard 0.12.
- Current cardio/dynamic state, scoring, and judging.

Outputs
-------
data/simulation/rfs_mc_v2_shared_state/v1_5/replay_<year>/
    fight_predictions.csv
    metrics.csv
    aggregate_calibration.csv
    confidence_bands.csv
    experience_bands.csv
    top_misses.csv
    excluded_fights.csv

Example
-------
PYTHONPATH=. python \
  scripts/experimental/run_fsr_v1_5_2026_replay.py \
  --year 2026 --max-fights 300 --simulations 250
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, log
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    FightResultBranch,
    resolve_final_fight_result,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_contracts import FinishMethod
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    resolve_fighter_parameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TAKEDOWN_EVENTS,
    TransitionEvent,
)

from scripts.experimental import fsr_cardio_v1 as cardio
from scripts.experimental import fsr_locked_families_v1 as equations_v1
from scripts.experimental import fsr_locked_families_v1_1 as equations_v1_1
from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_v1_5_compare as v1_5
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style
from scripts.experimental import run_fsr_v1_4_two_fight_full_validation as v1_4


MASTER_PATH = Path("data/master/ufc_master.parquet")
ROUND_PATH = base.ROUND_PATH
RFS_PATH = equations_v1.RFS_PATH

DEFAULT_YEAR = 2026
DEFAULT_MAX_FIGHTS = 300
DEFAULT_SIMULATIONS = 250
DEFAULT_SEED = 2026081000

METHODS = ("decision", "ko_tko", "submission")
EXACT_OUTCOMES = (
    "red_decision",
    "blue_decision",
    "red_ko_tko",
    "blue_ko_tko",
    "red_submission",
    "blue_submission",
    "draw",
)

PROFILE_PREFIXES = (
    "rfs_traj_",
    "rfs_open_",
    "rfs_phase_base_",
    "rfs_phase_interact_",
    "rfs_dynamic_response_",
    "rfs_finish_state_",
)
CURRENT_FIGHT_PREFIXES = (
    "rfs_traj_fight_",
    "rfs_open_fight_",
    "rfs_phase_base_fight_",
    "rfs_phase_interact_fight_",
    "rfs_dynamic_response_fight_",
    "rfs_finish_state_fight_",
)

OUTPUT_STAT_FIELDS = (
    "sig_attempted",
    "sig_landed",
    "td_attempted",
    "td_landed",
    "control_seconds",
    "knockdowns",
    "submission_attempts",
)


@dataclass(frozen=True)
class CohortFight:
    fight_id: str
    date: pd.Timestamp
    scheduled_rounds: int
    red_fighter_id: str
    red_fighter_name: str
    blue_fighter_id: str
    blue_fighter_name: str
    actual_winner_corner: str
    actual_method: str
    actual_finish_round: int
    actual_fight_time_seconds: float


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _finite_nonnegative(value: object) -> float | None:
    parsed = _finite(value)
    if parsed is None or parsed < 0.0:
        return None
    return parsed


def _method_family(value: object) -> str | None:
    """Normalize UFC result labels into decision / KO-TKO / submission."""

    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if "decision" in text:
        return "decision"
    if "submission" in text or text.startswith("sub"):
        return "submission"
    if "ko/tko" in text or "tko" in text or text == "ko" or "knockout" in text:
        return "ko_tko"
    return None


def _profile_columns(history: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in history.columns
        if column.startswith(PROFILE_PREFIXES)
        and not column.startswith(CURRENT_FIGHT_PREFIXES)
    ]
    if not columns:
        raise RuntimeError("RFS history contains no leakage-safe profile columns")
    return columns


def _prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and standardize RFS history, master labels, and round statistics."""

    for path in (RFS_PATH, MASTER_PATH, ROUND_PATH):
        if not path.exists():
            raise RuntimeError(f"Required replay input not found: {path}")

    rfs = pd.read_parquet(RFS_PATH)
    master = pd.read_parquet(MASTER_PATH)
    rounds = pd.read_parquet(ROUND_PATH)

    date_col = "date" if "date" in rfs.columns else "event_date"
    rfs[date_col] = pd.to_datetime(rfs[date_col], errors="raise")
    if date_col != "date":
        rfs["date"] = rfs[date_col]
    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    rfs["corner"] = rfs["corner"].astype(str).str.strip().str.lower()

    master["fight_id"] = master["fight_id"].astype(str)
    if "date" in master.columns:
        master["date"] = pd.to_datetime(master["date"], errors="coerce")
    master = repair_elapsed_match_time(master)

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    rounds["corner"] = rounds["corner"].astype(str).str.strip().str.lower()
    rounds["event_date"] = pd.to_datetime(rounds["event_date"], errors="raise")

    if rfs.duplicated(["fight_id", "fighter_id"]).any():
        duplicates = int(rfs.duplicated(["fight_id", "fighter_id"]).sum())
        raise RuntimeError(
            "RFS history must be one row per fighter-fight for replay; "
            f"found {duplicates} duplicate keys"
        )

    return rfs, master, rounds


def build_locked_prefight_snapshots(rfs: pd.DataFrame) -> pd.DataFrame:
    """Replay locked FSR V1.1 once and snapshot every PRE-fight rating card.

    This mirrors ``fsr_locked_families_v1_1.py`` but retains each chronological
    PRE-fight state rather than rebuilding the entire history separately for
    hundreds of target fights.
    """

    equations_v1.validate_columns(rfs)
    df = rfs.copy().sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = defaultdict(
        lambda: {skill: equations_v1.BASE_RATING for skill in equations_v1.SKILLS}
    )
    update_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {skill: 0 for skill in equations_v1.SKILLS}
    )
    fight_counts: dict[str, int] = defaultdict(int)
    pools = {key: [] for key in equations_v1.POOL_KEYS}

    weighted_observation_sum: dict[str, float] = defaultdict(float)
    quality_sum: dict[str, float] = defaultdict(float)
    snapshots: list[dict[str, object]] = []

    for fight_date, date_rows in df.groupby("date", sort=True):
        # Snapshot every fighter before ANY same-date results are incorporated.
        for fight_id, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue
            for row in fight.itertuples(index=False):
                fighter_id = str(row.fighter_id)
                _ = ratings[fighter_id]
                snapshot: dict[str, object] = {
                    "fight_id": str(fight_id),
                    "date": pd.Timestamp(fight_date),
                    "fighter_id": fighter_id,
                    "fighter_name": str(row.fighter_name),
                    "prior_ufc_fights": int(fight_counts[fighter_id]),
                }
                snapshot.update(
                    {skill: float(ratings[fighter_id][skill]) for skill in equations_v1.SKILLS}
                )
                snapshots.append(snapshot)

        date_deltas: dict[str, dict[str, float]] = defaultdict(
            lambda: {skill: 0.0 for skill in equations_v1.SKILLS}
        )
        date_updates: dict[str, dict[str, int]] = defaultdict(
            lambda: {skill: 0 for skill in equations_v1.SKILLS}
        )
        date_fights: dict[str, int] = defaultdict(int)
        date_weighted_observation_sum: dict[str, float] = defaultdict(float)
        date_quality_sum: dict[str, float] = defaultdict(float)

        for _, fight in date_rows.groupby("fight_id", sort=False):
            if len(fight) != 2:
                continue

            first = fight.iloc[0]
            second = fight.iloc[1]
            for row, opponent_row in ((first, second), (second, first)):
                fighter_id = str(row["fighter_id"])
                opponent_id = str(opponent_row["fighter_id"])
                _ = ratings[fighter_id]
                _ = ratings[opponent_id]

                bundle = equations_v1.observation_bundle(row, opponent_row, pools)

                for skill in equations_v1.SKILLS:
                    observation, quality = bundle[skill]
                    baseline = equations_v1_1.population_baseline(
                        weighted_observation_sum,
                        quality_sum,
                        skill,
                    )
                    expected = equations_v1_1.expected_probability(
                        ratings,
                        fighter_id,
                        opponent_id,
                        skill,
                        baseline,
                    )

                    if observation is None or quality <= 0.0:
                        delta = 0.0
                    else:
                        delta = (
                            equations_v1.k_factor(update_counts[fighter_id][skill])
                            * quality
                            * (float(observation) - expected)
                        )
                        date_updates[fighter_id][skill] += 1
                        date_weighted_observation_sum[skill] += (
                            quality * float(observation)
                        )
                        date_quality_sum[skill] += quality

                    date_deltas[fighter_id][skill] += delta

                date_fights[fighter_id] += 1

        for fighter_id, skill_deltas in date_deltas.items():
            for skill, delta in skill_deltas.items():
                ratings[fighter_id][skill] = equations_v1.clamp(
                    ratings[fighter_id][skill] + delta,
                    equations_v1.MIN_RATING,
                    equations_v1.MAX_RATING,
                )
                update_counts[fighter_id][skill] += date_updates[fighter_id][skill]

        for fighter_id, count in date_fights.items():
            fight_counts[fighter_id] += count

        for skill in equations_v1.SKILLS:
            weighted_observation_sum[skill] += date_weighted_observation_sum[skill]
            quality_sum[skill] += date_quality_sum[skill]

        equations_v1.append_date_to_pools(date_rows, pools)

    snapshots_df = pd.DataFrame(snapshots)
    if snapshots_df.empty:
        raise RuntimeError("Locked FSR replay produced no pre-fight snapshots")
    if snapshots_df.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("Locked FSR snapshots contain duplicate fighter-fight keys")

    return snapshots_df


def _locked_card(snapshot: pd.Series) -> tuple[dict[str, float], int]:
    """Translate one V1.1 snapshot into the current generic adapter aliases."""

    locked = {skill: float(snapshot[skill]) for skill in equations_v1.SKILLS}
    card = {
        "distance_volume": 50.0,
        "distance_accuracy": locked["distance_precision"],
        "distance_defense": locked["distance_defense"],
        "td_initiative": locked["wrestling_entry"],
        "td_completion": locked["wrestling_conversion"],
        "td_defense": locked["td_defense"],
        "control_imposition": locked["control_imposition"],
        "control_resistance": locked["control_resistance"],
        "submission_pressure": locked["submission_pressure"],
        "submission_conversion": locked["submission_conversion"],
        "submission_resistance": locked["submission_resistance"],
        "finishing_power": locked["striking_power"],
        "chin_resistance": locked["chin_resistance"],
        "damage_absorption": locked["damage_resistance"],
    }
    card.update(locked)
    return card, int(snapshot["prior_ufc_fights"])


def _attach_cardio(
    card: dict[str, float],
    *,
    profile: dict[str, object],
    prior_fight_count: int,
    population: pd.DataFrame,
) -> dict[str, float]:
    """Attach the same leakage-safe cardio bridge used by single-fight V1.5."""

    resolved = resolve_fighter_parameters(
        profile=profile,
        prior_fight_count=prior_fight_count,
        population_history=population,
    )

    out = dict(card)
    for target in cardio.CARDIO_TARGETS:
        estimate = resolved.estimates[target]
        suffix = target.removeprefix("dynamic.")
        engine_value = float(estimate.shrunk_estimate)
        out[f"{suffix}_engine"] = engine_value
        out[f"{suffix}_rating"] = cardio.unit_to_card_rating(engine_value)
        out[f"{suffix}_reliability"] = float(estimate.reliability)
    return out


def _attach_activity_style(card: dict[str, float], row: pd.Series) -> dict[str, float]:
    """Attach the exact V1.4 + V1.5 Phase Baseline PRE-fight style fields."""

    out = dict(card)

    for short_name, column in style.STYLE_COLUMNS.items():
        value = _finite(row.get(column))
        out[f"style_{short_name}"] = float("nan") if value is None else value

    for short_name, column in v1_5.V1_5_STYLE_COLUMNS.items():
        value = _finite_nonnegative(row.get(column))
        out[f"style_{short_name}"] = float("nan") if value is None else value

    return out


def _build_cohort(
    rfs: pd.DataFrame,
    master: pd.DataFrame,
    *,
    year: int,
    max_fights: int | None,
) -> tuple[list[CohortFight], pd.DataFrame]:
    """Select completed scoreable fights and retain transparent exclusions."""

    master_by_fight = {
        str(fight_id): group.iloc[0]
        for fight_id, group in master.groupby("fight_id", sort=False)
    }

    candidates = rfs.loc[rfs["date"].dt.year.eq(int(year))].copy()
    records: list[CohortFight] = []
    excluded: list[dict[str, object]] = []

    for fight_id, fight in candidates.groupby("fight_id", sort=False):
        fight_id = str(fight_id)
        reason = None

        if len(fight) != 2 or set(fight["corner"]) != {"red", "blue"}:
            reason = "incomplete_red_blue_rfs_pair"
        elif fight_id not in master_by_fight:
            reason = "missing_master_result"

        if reason is not None:
            excluded.append({"fight_id": fight_id, "reason": reason})
            continue

        result = master_by_fight[fight_id]
        red = fight.loc[fight["corner"].eq("red")].iloc[0]
        blue = fight.loc[fight["corner"].eq("blue")].iloc[0]

        winner_id = None if pd.isna(result.get("winner_id")) else str(result.get("winner_id"))
        fighter_ids = {str(red["fighter_id"]), str(blue["fighter_id"])}
        if winner_id not in fighter_ids:
            excluded.append({"fight_id": fight_id, "reason": "no_scoreable_winner"})
            continue

        method = _method_family(result.get("method"))
        if method is None:
            excluded.append({"fight_id": fight_id, "reason": "unsupported_method"})
            continue

        scheduled_raw = result.get("total_rounds", red.get("total_rounds"))
        scheduled = _finite(scheduled_raw)
        if scheduled is None or int(scheduled) not in {3, 5}:
            excluded.append({"fight_id": fight_id, "reason": "unsupported_scheduled_rounds"})
            continue
        scheduled_rounds = int(scheduled)

        finish_round_raw = _finite(result.get("finish_round"))
        finish_round = (
            scheduled_rounds if method == "decision" and finish_round_raw is None
            else int(finish_round_raw) if finish_round_raw is not None
            else 0
        )

        fight_time = _finite(result.get("match_time_sec"))
        if fight_time is None or fight_time <= 0.0:
            excluded.append({"fight_id": fight_id, "reason": "missing_fight_time"})
            continue

        records.append(
            CohortFight(
                fight_id=fight_id,
                date=pd.Timestamp(red["date"]),
                scheduled_rounds=scheduled_rounds,
                red_fighter_id=str(red["fighter_id"]),
                red_fighter_name=str(red["fighter_name"]),
                blue_fighter_id=str(blue["fighter_id"]),
                blue_fighter_name=str(blue["fighter_name"]),
                actual_winner_corner=(
                    "red" if winner_id == str(red["fighter_id"]) else "blue"
                ),
                actual_method=method,
                actual_finish_round=finish_round,
                actual_fight_time_seconds=float(fight_time),
            )
        )

    records.sort(key=lambda item: (item.date, item.fight_id))
    if max_fights is not None and len(records) > int(max_fights):
        records = records[-int(max_fights):]

    return records, pd.DataFrame(excluded)


def _actual_fight_stats(rounds: pd.DataFrame, fight_id: str) -> dict[str, float]:
    fight = rounds.loc[rounds["fight_id"].eq(str(fight_id))].copy()
    if fight.empty:
        raise RuntimeError(f"No round statistics found for fight {fight_id}")

    def metric(frame: pd.DataFrame, *columns: str) -> float:
        for column in columns:
            if column in frame.columns:
                return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
        return float("nan")

    output: dict[str, float] = {}
    for side in ("red", "blue"):
        selected = fight.loc[fight["corner"].eq(side)]
        output[f"{side}_sig_attempted"] = metric(selected, "sig_str_attempted")
        output[f"{side}_sig_landed"] = metric(selected, "sig_str_landed")
        output[f"{side}_td_attempted"] = metric(selected, "td_attempted")
        output[f"{side}_td_landed"] = metric(selected, "td_landed")
        output[f"{side}_control_seconds"] = metric(selected, "ctrl_sec", "control_seconds")
        output[f"{side}_knockdowns"] = metric(selected, "kd", "knockdowns")
        output[f"{side}_submission_attempts"] = metric(selected, "sub_att", "submission_attempts")
    return output


def _empty_path_stats() -> dict[str, float]:
    return {
        f"{side}_{metric}": 0.0
        for side in ("red", "blue")
        for metric in OUTPUT_STAT_FIELDS
    }


def _accumulate_path_stats(path, totals: dict[str, float]) -> None:
    """Accumulate physically simulated fight totals from one complete path."""

    for segment in path.segments:
        for side, activity in (("red", segment.activity.red), ("blue", segment.activity.blue)):
            distance_attempted = float(getattr(activity, "sig_str_attempted", 0))
            distance_landed = float(getattr(activity, "sig_str_landed", 0))
            clinch_attempted = float(getattr(activity, "clinch_str_attempted", 0))
            clinch_landed = float(getattr(activity, "clinch_str_landed", 0))
            ground_attempted = float(getattr(activity, "ground_str_attempted", 0))
            ground_landed = float(getattr(activity, "ground_str_landed", 0))

            totals[f"{side}_sig_attempted"] += (
                distance_attempted + clinch_attempted + ground_attempted
            )
            totals[f"{side}_sig_landed"] += (
                distance_landed + clinch_landed + ground_landed
            )
            totals[f"{side}_control_seconds"] += float(
                getattr(activity, "control_seconds", 0)
            )
            totals[f"{side}_knockdowns"] += float(getattr(activity, "knockdowns", 0))
            totals[f"{side}_submission_attempts"] += float(
                getattr(activity, "submission_attempts", 0)
            )

        transition = segment.transition
        if transition is None or transition.event not in TAKEDOWN_EVENTS:
            continue

        if transition.actor is FighterSide.RED:
            side = "red"
        elif transition.actor is FighterSide.BLUE:
            side = "blue"
        else:
            raise RuntimeError("Takedown terminal transition has no actor")

        totals[f"{side}_td_attempted"] += float(transition.attempt_count)
        if transition.event is TransitionEvent.TAKEDOWN:
            totals[f"{side}_td_landed"] += 1.0


def _simulate_fight(
    *,
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
    scheduled_rounds: int,
    simulations: int,
    seed_start: int,
) -> dict[str, float]:
    """Run one fight once, aggregating results and physical totals together."""

    red_transition = base.build_transition(red_card)
    blue_transition = base.build_transition(blue_card)
    red_phase = base.build_phase(red_card, blue_card, baselines)
    blue_phase = base.build_phase(blue_card, red_card, baselines)
    red_dynamic = base.build_dynamic(red_card)
    blue_dynamic = base.build_dynamic(blue_card)

    candidate = base.Candidate(
        landed_ko_hazard=base.V1_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=base.V1_KNOCKDOWN_BONUS_HAZARD,
    )
    dynamic_cal = base.state_calibration(candidate)
    phase_cal = base.phase_effect_calibration(candidate)
    transition_cal = base.zero_transition_effect_calibration()
    finish_cal = base.finish_calibration(candidate)

    winner_counts = Counter()
    method_counts = Counter()
    exact_counts = Counter()
    finish_round_counts = Counter()
    path_stats = _empty_path_stats()
    total_fight_time = 0.0

    for index in range(simulations):
        path = base.run_finish_enabled_dynamic_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,
            dynamic_state_calibration=dynamic_cal,
            phase_effect_calibration=phase_cal,
            transition_effect_calibration=transition_cal,
            finish_probability_calibration=finish_cal,
            scheduled_rounds=scheduled_rounds,
            seed=seed_start + index,
            red_intrinsic_power_multiplier=base.intrinsic_power_multiplier(
                red_card["finishing_power"]
            ),
            blue_intrinsic_power_multiplier=base.intrinsic_power_multiplier(
                blue_card["finishing_power"]
            ),
            red_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(red_card["chin_resistance"])
            ),
            blue_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(blue_card["chin_resistance"])
            ),
            shared_path_calibration=v1_4.V1_4_CALIBRATION,
        )
        result = resolve_final_fight_result(path)
        _accumulate_path_stats(path, path_stats)

        if result.winner is FighterSide.RED:
            winner = "red"
        elif result.winner is FighterSide.BLUE:
            winner = "blue"
        else:
            winner = "draw"
        winner_counts[winner] += 1

        if result.branch is FightResultBranch.FINISH:
            if result.finish is None:
                raise RuntimeError("Finish branch has no finish payload")
            if result.finish.method is FinishMethod.KO_TKO:
                method = "ko_tko"
            elif result.finish.method is FinishMethod.SUBMISSION:
                method = "submission"
            else:
                raise RuntimeError("Unsupported simulated finish method")
            method_counts[method] += 1
            exact_counts[f"{winner}_{method}"] += 1
            finish_round_counts[int(result.finish.round_number)] += 1
            total_fight_time += (
                (int(result.finish.round_number) - 1) * 300.0
                + float(result.finish.elapsed_seconds_in_round)
            )
        elif result.branch is FightResultBranch.SCHEDULED_DISTANCE:
            method_counts["decision"] += 1
            exact_counts[f"{winner}_decision" if winner != "draw" else "draw"] += 1
            total_fight_time += float(scheduled_rounds * 300)
        else:
            raise RuntimeError("Unsupported simulated result branch")

    total = float(simulations)
    output: dict[str, float] = {
        "sim_red_win_probability": winner_counts["red"] / total,
        "sim_blue_win_probability": winner_counts["blue"] / total,
        "sim_draw_probability": winner_counts["draw"] / total,
        "sim_decision_probability": method_counts["decision"] / total,
        "sim_ko_tko_probability": method_counts["ko_tko"] / total,
        "sim_submission_probability": method_counts["submission"] / total,
        "sim_fight_time_seconds": total_fight_time / total,
    }

    for exact in EXACT_OUTCOMES:
        output[f"sim_{exact}_probability"] = exact_counts[exact] / total

    for round_number in range(1, scheduled_rounds + 1):
        output[f"sim_finish_round_{round_number}_probability"] = (
            finish_round_counts[round_number] / total
        )

    for key, value in path_stats.items():
        output[f"sim_{key}"] = value / total

    return output


def _actual_exact_outcome(record: CohortFight) -> str:
    return f"{record.actual_winner_corner}_{record.actual_method}"


def _score_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score winner/method/timing/activity and build calibration summaries."""

    df = predictions.copy()
    epsilon = 1e-9

    non_draw_mass = df["sim_red_win_probability"] + df["sim_blue_win_probability"]
    df["sim_red_win_probability_conditional"] = np.where(
        non_draw_mass.gt(0),
        df["sim_red_win_probability"] / non_draw_mass,
        0.5,
    )
    df["predicted_winner_corner"] = np.where(
        df["sim_red_win_probability"] >= df["sim_blue_win_probability"],
        "red",
        "blue",
    )
    df["winner_correct"] = df["predicted_winner_corner"].eq(df["actual_winner_corner"])
    df["actual_red_win"] = df["actual_winner_corner"].eq("red").astype(float)
    df["predicted_winner_confidence"] = np.maximum(
        df["sim_red_win_probability_conditional"],
        1.0 - df["sim_red_win_probability_conditional"],
    )
    df["actual_winner_probability_raw"] = np.where(
        df["actual_winner_corner"].eq("red"),
        df["sim_red_win_probability"],
        df["sim_blue_win_probability"],
    )

    method_columns = [f"sim_{method}_probability" for method in METHODS]
    method_matrix = df[method_columns].to_numpy(dtype=float)
    method_matrix = np.clip(method_matrix, epsilon, None)
    method_matrix = method_matrix / method_matrix.sum(axis=1, keepdims=True)
    method_index = {method: index for index, method in enumerate(METHODS)}
    actual_method_index = np.array([method_index[value] for value in df["actual_method"]])
    df["predicted_method"] = np.array(METHODS)[np.argmax(method_matrix, axis=1)]
    df["method_correct"] = df["predicted_method"].eq(df["actual_method"])
    df["actual_method_probability"] = method_matrix[
        np.arange(len(df)), actual_method_index
    ]

    exact_columns = [f"sim_{value}_probability" for value in EXACT_OUTCOMES]
    exact_matrix = df[exact_columns].to_numpy(dtype=float)
    exact_matrix = np.clip(exact_matrix, epsilon, None)
    exact_matrix = exact_matrix / exact_matrix.sum(axis=1, keepdims=True)
    exact_index = {value: index for index, value in enumerate(EXACT_OUTCOMES)}
    actual_exact = df["actual_exact_outcome"].tolist()
    actual_exact_index = np.array([exact_index[value] for value in actual_exact])
    df["predicted_exact_outcome"] = np.array(EXACT_OUTCOMES)[np.argmax(exact_matrix, axis=1)]
    df["exact_outcome_correct"] = df["predicted_exact_outcome"].eq(df["actual_exact_outcome"])
    df["actual_exact_outcome_probability"] = exact_matrix[
        np.arange(len(df)), actual_exact_index
    ]

    metrics: list[dict[str, object]] = []

    def add(name: str, value: float) -> None:
        metrics.append({"metric": name, "value": float(value)})

    red_probability = df["sim_red_win_probability_conditional"].to_numpy(dtype=float)
    actual_red = df["actual_red_win"].to_numpy(dtype=float)
    add("fights", len(df))
    add("winner_accuracy", df["winner_correct"].mean())
    add("winner_brier_conditional", np.mean((red_probability - actual_red) ** 2))
    add("winner_log_loss_raw", -np.mean(np.log(np.clip(df["actual_winner_probability_raw"], epsilon, 1.0))))
    add("mean_probability_actual_winner", df["actual_winner_probability_raw"].mean())
    add("mean_sim_draw_probability", df["sim_draw_probability"].mean())
    add("method_accuracy", df["method_correct"].mean())
    add("method_log_loss", -np.mean(np.log(np.clip(df["actual_method_probability"], epsilon, 1.0))))
    add("exact_winner_method_accuracy", df["exact_outcome_correct"].mean())
    add("exact_winner_method_log_loss", -np.mean(np.log(np.clip(df["actual_exact_outcome_probability"], epsilon, 1.0))))

    actual_distance = df["actual_method"].eq("decision").astype(float).to_numpy()
    add(
        "goes_distance_brier",
        np.mean((df["sim_decision_probability"].to_numpy(dtype=float) - actual_distance) ** 2),
    )
    add(
        "fight_time_mae_seconds",
        np.mean(np.abs(df["sim_fight_time_seconds"] - df["actual_fight_time_seconds"])),
    )
    add(
        "fight_time_bias_seconds",
        np.mean(df["sim_fight_time_seconds"] - df["actual_fight_time_seconds"]),
    )

    for metric in OUTPUT_STAT_FIELDS:
        errors = []
        signed = []
        for side in ("red", "blue"):
            residual = (
                df[f"sim_{side}_{metric}"].to_numpy(dtype=float)
                - df[f"actual_{side}_{metric}"].to_numpy(dtype=float)
            )
            errors.extend(np.abs(residual).tolist())
            signed.extend(residual.tolist())
        add(f"fighter_{metric}_mae", np.mean(errors))
        add(f"fighter_{metric}_bias", np.mean(signed))

    metrics_df = pd.DataFrame(metrics)

    aggregate_rows = []
    for method in METHODS:
        aggregate_rows.append(
            {
                "quantity": f"method_{method}_rate",
                "actual": float(df["actual_method"].eq(method).mean()),
                "simulated": float(df[f"sim_{method}_probability"].mean()),
            }
        )
    aggregate_rows.extend(
        [
            {
                "quantity": "red_win_rate",
                "actual": float(df["actual_winner_corner"].eq("red").mean()),
                "simulated": float(df["sim_red_win_probability"].mean()),
            },
            {
                "quantity": "draw_rate",
                "actual": 0.0,
                "simulated": float(df["sim_draw_probability"].mean()),
            },
            {
                "quantity": "mean_fight_time_seconds",
                "actual": float(df["actual_fight_time_seconds"].mean()),
                "simulated": float(df["sim_fight_time_seconds"].mean()),
            },
        ]
    )
    for metric in OUTPUT_STAT_FIELDS:
        actual_values = np.concatenate(
            [df[f"actual_red_{metric}"].to_numpy(float), df[f"actual_blue_{metric}"].to_numpy(float)]
        )
        sim_values = np.concatenate(
            [df[f"sim_red_{metric}"].to_numpy(float), df[f"sim_blue_{metric}"].to_numpy(float)]
        )
        aggregate_rows.append(
            {
                "quantity": f"mean_fighter_{metric}",
                "actual": float(np.mean(actual_values)),
                "simulated": float(np.mean(sim_values)),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df["bias"] = aggregate_df["simulated"] - aggregate_df["actual"]

    confidence_bins = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.000001]
    confidence_labels = ["50-55", "55-60", "60-70", "70-80", "80-90", "90-100"]
    df["confidence_band"] = pd.cut(
        df["predicted_winner_confidence"],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True,
        right=False,
    )
    confidence_df = (
        df.groupby("confidence_band", observed=True)
        .agg(
            fights=("fight_id", "size"),
            mean_confidence=("predicted_winner_confidence", "mean"),
            winner_accuracy=("winner_correct", "mean"),
            mean_actual_winner_probability=("actual_winner_probability_raw", "mean"),
        )
        .reset_index()
    )

    minimum_prior = df[["red_prior_fights", "blue_prior_fights"]].min(axis=1)
    df["experience_band"] = pd.cut(
        minimum_prior,
        bins=[-1, 0, 2, 5, float("inf")],
        labels=["0_prior", "1-2_prior", "3-5_prior", "6+_prior"],
    )
    experience_df = (
        df.groupby("experience_band", observed=True)
        .agg(
            fights=("fight_id", "size"),
            winner_accuracy=("winner_correct", "mean"),
            mean_confidence=("predicted_winner_confidence", "mean"),
            method_accuracy=("method_correct", "mean"),
            exact_outcome_accuracy=("exact_outcome_correct", "mean"),
        )
        .reset_index()
    )

    return df, metrics_df, aggregate_df, confidence_df, experience_df


def _print_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    confidence: pd.DataFrame,
    experience: pd.DataFrame,
) -> None:
    values = dict(zip(metrics["metric"], metrics["value"]))

    print()
    print("=" * 118)
    print("FSR / MC V1.5 — LARGE 2026 HISTORICAL REPLAY")
    print("=" * 118)
    print(f"Scoreable fights              : {int(values['fights'])}")
    print(f"Winner accuracy               : {100.0 * values['winner_accuracy']:.2f}%")
    print(f"Winner Brier (non-draw norm)  : {values['winner_brier_conditional']:.4f}")
    print(f"Winner log loss (raw mass)    : {values['winner_log_loss_raw']:.4f}")
    print(f"Mean P(actual winner)         : {100.0 * values['mean_probability_actual_winner']:.2f}%")
    print(f"Mean simulated draw           : {100.0 * values['mean_sim_draw_probability']:.2f}%")
    print(f"Method accuracy               : {100.0 * values['method_accuracy']:.2f}%")
    print(f"Method log loss               : {values['method_log_loss']:.4f}")
    print(f"Winner + method accuracy      : {100.0 * values['exact_winner_method_accuracy']:.2f}%")
    print(f"Winner + method log loss      : {values['exact_winner_method_log_loss']:.4f}")
    print(f"Goes-distance Brier           : {values['goes_distance_brier']:.4f}")
    print(f"Fight-time MAE                : {values['fight_time_mae_seconds']:.1f} sec")
    print()
    print("Physical fighter-level MAE")
    print("-" * 70)
    for metric in OUTPUT_STAT_FIELDS:
        print(f"{metric:<24}: {values[f'fighter_{metric}_mae']:.3f}")

    print()
    print("Aggregate calibration")
    print("-" * 90)
    print(aggregate.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("Winner confidence calibration")
    print("-" * 90)
    if confidence.empty:
        print("No confidence bands available")
    else:
        print(confidence.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("Experience bands")
    print("-" * 90)
    if experience.empty:
        print("No experience bands available")
    else:
        print(experience.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    misses = predictions.loc[~predictions["winner_correct"]].copy()
    misses = misses.sort_values("predicted_winner_confidence", ascending=False).head(20)
    print()
    print("Most confident wrong winner picks")
    print("-" * 118)
    if misses.empty:
        print("None")
    else:
        display = misses[
            [
                "date",
                "red_fighter_name",
                "blue_fighter_name",
                "predicted_winner_corner",
                "predicted_winner_confidence",
                "actual_winner_corner",
                "actual_method",
                "red_prior_fights",
                "blue_prior_fights",
            ]
        ].copy()
        print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--max-fights", type=int, default=DEFAULT_MAX_FIGHTS)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.simulations <= 0:
        raise ValueError("--simulations must be positive")
    if args.max_fights <= 0:
        raise ValueError("--max-fights must be positive")

    # Install the current V1.5 shadow adapter exactly once. The benchmark then
    # constructs cards in memory rather than invoking per-fight subprocesses.
    v1_5.install_v1_5()

    print("Loading replay inputs...")
    rfs, master, rounds = _prepare_inputs()
    profile_columns = _profile_columns(rfs)

    print("Building one chronological leakage-safe FSR V1.1 replay...")
    snapshots = build_locked_prefight_snapshots(rfs)
    snapshot_index = snapshots.set_index(["fight_id", "fighter_id"], verify_integrity=True)
    rfs_index = rfs.set_index(["fight_id", "fighter_id"], verify_integrity=True)

    cohort, excluded = _build_cohort(
        rfs,
        master,
        year=args.year,
        max_fights=args.max_fights,
    )
    if not cohort:
        raise RuntimeError(f"No scoreable fights found for {args.year}")

    print(
        f"Selected {len(cohort)} scoreable fights from {args.year} "
        f"(requested max {args.max_fights})."
    )

    population_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    baseline_cache: dict[pd.Timestamp, dict[str, float]] = {}
    rows: list[dict[str, object]] = []

    for index, fight in enumerate(cohort):
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(cohort):
            print(
                f"[{index + 1:>3}/{len(cohort)}] "
                f"{fight.date.date()} | {fight.red_fighter_name} vs {fight.blue_fighter_name}"
            )

        target_date = pd.Timestamp(fight.date)
        if target_date not in population_cache:
            population = rfs.loc[
                (rfs["date"] < target_date)
                & (
                    pd.to_numeric(
                        rfs["rfs_traj_prior_fight_count"], errors="coerce"
                    ).fillna(0).gt(0)
                )
            ].copy()
            if population.empty:
                raise RuntimeError(f"No leakage-safe RFS population before {target_date}")
            population_cache[target_date] = population

        if target_date not in baseline_cache:
            baseline_cache[target_date] = base.population_baselines(rounds, target_date)

        population = population_cache[target_date]
        baselines = baseline_cache[target_date]

        fighter_cards: dict[str, tuple[dict[str, float], int, pd.Series]] = {}
        for fighter_id in (fight.red_fighter_id, fight.blue_fighter_id):
            key = (fight.fight_id, fighter_id)
            if key not in snapshot_index.index:
                raise RuntimeError(f"Missing locked FSR snapshot for {key}")
            if key not in rfs_index.index:
                raise RuntimeError(f"Missing RFS target state for {key}")

            snapshot = snapshot_index.loc[key]
            rfs_row = rfs_index.loc[key]
            card, prior_fights = _locked_card(snapshot)
            profile = {column: rfs_row[column] for column in profile_columns}
            rfs_prior_fights = int(
                pd.to_numeric(rfs_row["rfs_traj_prior_fight_count"], errors="raise")
            )
            card = _attach_cardio(
                card,
                profile=profile,
                prior_fight_count=rfs_prior_fights,
                population=population,
            )
            card = _attach_activity_style(card, rfs_row)
            fighter_cards[fighter_id] = (card, prior_fights, rfs_row)

        red_card, red_prior, red_rfs = fighter_cards[fight.red_fighter_id]
        blue_card, blue_prior, blue_rfs = fighter_cards[fight.blue_fighter_id]

        seed_start = int(args.seed + index * 100003)
        simulated = _simulate_fight(
            red_card=red_card,
            blue_card=blue_card,
            baselines=baselines,
            scheduled_rounds=fight.scheduled_rounds,
            simulations=args.simulations,
            seed_start=seed_start,
        )
        actual_stats = _actual_fight_stats(rounds, fight.fight_id)

        row: dict[str, object] = {
            "fight_id": fight.fight_id,
            "date": fight.date,
            "scheduled_rounds": fight.scheduled_rounds,
            "red_fighter_id": fight.red_fighter_id,
            "red_fighter_name": fight.red_fighter_name,
            "blue_fighter_id": fight.blue_fighter_id,
            "blue_fighter_name": fight.blue_fighter_name,
            "red_prior_fights": red_prior,
            "blue_prior_fights": blue_prior,
            "actual_winner_corner": fight.actual_winner_corner,
            "actual_method": fight.actual_method,
            "actual_finish_round": fight.actual_finish_round,
            "actual_fight_time_seconds": fight.actual_fight_time_seconds,
            "actual_exact_outcome": _actual_exact_outcome(fight),
            "simulations": args.simulations,
            "seed_start": seed_start,
            "red_rfs_distance_attempts_per_round": red_card.get("style_distance_attempts_per_round"),
            "blue_rfs_distance_attempts_per_round": blue_card.get("style_distance_attempts_per_round"),
            "red_rfs_ground_attempts_per_round": red_card.get("style_ground_attempts_per_round"),
            "blue_rfs_ground_attempts_per_round": blue_card.get("style_ground_attempts_per_round"),
            "red_rfs_control_seconds_per_round": red_card.get("style_control_seconds_per_round"),
            "blue_rfs_control_seconds_per_round": blue_card.get("style_control_seconds_per_round"),
            "red_fsr_control_imposition": red_card["control_imposition"],
            "blue_fsr_control_imposition": blue_card["control_imposition"],
            "red_fsr_striking_power": red_card["finishing_power"],
            "blue_fsr_striking_power": blue_card["finishing_power"],
        }
        row.update({f"actual_{key}": value for key, value in actual_stats.items()})
        row.update(simulated)
        rows.append(row)

    raw_predictions = pd.DataFrame(rows)
    predictions, metrics, aggregate, confidence, experience = _score_predictions(
        raw_predictions
    )

    output_dir = Path(
        f"data/simulation/rfs_mc_v2_shared_state/v1_5/replay_{args.year}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / "fight_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_calibration.csv", index=False)
    confidence.to_csv(output_dir / "confidence_bands.csv", index=False)
    experience.to_csv(output_dir / "experience_bands.csv", index=False)
    excluded.to_csv(output_dir / "excluded_fights.csv", index=False)

    top_misses = predictions.loc[~predictions["winner_correct"]].copy()
    top_misses = top_misses.sort_values(
        "predicted_winner_confidence", ascending=False
    ).head(50)
    top_misses.to_csv(output_dir / "top_misses.csv", index=False)

    _print_summary(predictions, metrics, aggregate, confidence, experience)

    print()
    print("Saved replay artifacts:")
    for name in (
        "fight_predictions.csv",
        "metrics.csv",
        "aggregate_calibration.csv",
        "confidence_bands.csv",
        "experience_bands.csv",
        "top_misses.csv",
        "excluded_fights.csv",
    ):
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
