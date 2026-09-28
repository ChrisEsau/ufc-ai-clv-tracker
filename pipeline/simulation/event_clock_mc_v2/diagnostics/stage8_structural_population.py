"""Stage 8 structural population validation for the causal Event Clock V2 engine.

Diagnostic only. Uses historical FSR V3 prefight rows and realized UFCStats outputs.
No simulator mechanics, policy, timing, FSR, or production entrypoint is modified.

Historical phase-time denominators do not exist in the repository, so simulated
standing/clinch/ground exposure is reported as simulator-only structure rather
than compared to fabricated historical phase shares.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineConfig,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_latest_profiles,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
    prior_snapshot_count,
)

from .stage6_real_causal_path import _capabilities, _mechanics

ROUND_STATS = Path("data/fight_details/ufc_round_stats.parquet")
MASTER = Path("data/master/ufc_master.parquet")

STANDING_STRIKES = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
CLINCH_STRIKES = {ActionFamily.CLINCH_STRIKE}
GROUND_STRIKES = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD_ACTIONS = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}


def pick_col(df: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise RuntimeError(f"missing any of columns {names}; available={list(df.columns)}")
    return None


def numeric_sum(frame: pd.DataFrame, *aliases: str) -> float:
    col = pick_col(frame, *aliases, required=False)
    if col is None:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum())


def actual_side_totals(frame: pd.DataFrame) -> dict[str, float]:
    sig_att = numeric_sum(frame, "sig_str_attempted", "sig_str_att", "significant_strikes_attempted", "sig_attempted")
    sig_land = numeric_sum(frame, "sig_str_landed", "sig_str_land", "significant_strikes_landed", "sig_landed")
    clinch_att = numeric_sum(frame, "clinch_attempted", "clinch_att", "clinch_sig_str_attempted", "clinch_sig_att")
    clinch_land = numeric_sum(frame, "clinch_landed", "clinch_land", "clinch_sig_str_landed", "clinch_sig_land")
    ground_att = numeric_sum(frame, "ground_attempted", "ground_att", "ground_sig_str_attempted", "ground_sig_att")
    ground_land = numeric_sum(frame, "ground_landed", "ground_land", "ground_sig_str_landed", "ground_sig_land")
    td_att = numeric_sum(frame, "td_attempted", "td_att", "takedowns_attempted")
    td_land = numeric_sum(frame, "td_landed", "td_land", "takedowns_landed")
    control = numeric_sum(frame, "control_seconds", "control_time_sec", "ctrl_seconds", "control")
    sub_att = numeric_sum(frame, "sub_attempts", "sub_att", "submission_attempts")

    # Distance is only derived when both component fields exist. Do not invent it.
    distance_att = float("nan") if any(np.isnan(x) for x in (sig_att, clinch_att, ground_att)) else max(0.0, sig_att - clinch_att - ground_att)
    distance_land = float("nan") if any(np.isnan(x) for x in (sig_land, clinch_land, ground_land)) else max(0.0, sig_land - clinch_land - ground_land)
    return {
        "sig_att": sig_att,
        "sig_land": sig_land,
        "distance_att": distance_att,
        "distance_land": distance_land,
        "clinch_att": clinch_att,
        "clinch_land": clinch_land,
        "ground_att": ground_att,
        "ground_land": ground_land,
        "td_att": td_att,
        "td_land": td_land,
        "control_seconds": control,
        "sub_att": sub_att,
    }


def elapsed_seconds(row: pd.Series) -> float:
    for col in ("actual_elapsed_seconds", "fight_elapsed_seconds", "elapsed_seconds"):
        if col in row and pd.notna(row[col]):
            value = float(row[col])
            if value > 0:
                return value
    finish_round = row.get("finish_round", row.get("round", np.nan))
    match_time = row.get("match_time_sec", row.get("time_seconds", np.nan))
    if pd.notna(finish_round) and pd.notna(match_time):
        return max(0.0, (float(finish_round) - 1.0) * 300.0 + float(match_time))
    rounds = float(row.get("total_rounds", 3) or 3)
    return rounds * 300.0


def side_rows(round_stats: pd.DataFrame, fight_id: str, fighter_id: str, corner: str) -> pd.DataFrame:
    fid_col = pick_col(round_stats, "fight_id", "bout_id")
    fr = round_stats[round_stats[fid_col].astype(str).eq(str(fight_id))]
    fighter_col = pick_col(fr, "fighter_id", required=False)
    if fighter_col is not None:
        rows = fr[fr[fighter_col].astype(str).eq(str(fighter_id))]
        if not rows.empty:
            return rows
    corner_col = pick_col(fr, "corner", required=False)
    if corner_col is not None:
        rows = fr[fr[corner_col].astype(str).str.lower().eq(corner.lower())]
        if not rows.empty:
            return rows
    raise RuntimeError(f"cannot map round rows fight={fight_id} fighter={fighter_id} corner={corner}")


def mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=25)
    ap.add_argument("--seed-base", type=int, default=20260825)
    ap.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage8_structural_population.json"))
    args = ap.parse_args()
    if args.fights < 1 or args.paths_per_fight < 1:
        raise ValueError("fights and paths-per-fight must be positive")

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(["_event_date", "fight_id"], ascending=[False, False])

    round_stats = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(round_stats, "fight_id", "bout_id")
    available_fights = set(round_stats[rs_fight_col].astype(str))

    snapshots = load_prefight_snapshots()
    latest = load_latest_profiles()
    reference = CapabilityReference.from_latest(latest)

    chosen: list[tuple[pd.Series, pd.Series, pd.Series]] = []
    skipped = Counter()
    for _, row in master.iterrows():
        if len(chosen) >= args.fights:
            break
        fight_id = str(row["fight_id"])
        if fight_id not in available_fights:
            skipped["no_round_stats"] += 1
            continue
        try:
            red_id = str(row["r_id"])
            blue_id = str(row["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(
                snapshots,
                event_date=row["_event_date"],
                fight_id=fight_id,
                fighter_ids=(red_id, blue_id),
            )
            side_rows(round_stats, fight_id, red_id, "red")
            side_rows(round_stats, fight_id, blue_id, "blue")
        except Exception:
            skipped["incomplete_historical_inputs"] += 1
            continue
        chosen.append((row, red_fsr, blue_fsr))

    if len(chosen) < args.fights:
        raise RuntimeError(f"only {len(chosen)} complete historical fights found; requested {args.fights}; skipped={dict(skipped)}")

    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()
    fight_rows: list[dict[str, object]] = []
    pooled_actual = defaultdict(float)
    pooled_sim = defaultdict(float)
    pooled_seconds_actual = 0.0
    pooled_seconds_sim = 0.0
    phase_seconds = defaultdict(float)
    segment_durations = defaultdict(list)
    transition_counts = Counter()
    total_events = 0
    total_sim_paths = 0
    illegal_actions = 0
    timeline_mismatches = 0

    for fight_index, (mr, red_fsr, blue_fsr) in enumerate(chosen):
        fight_id = str(mr["fight_id"])
        red_id, blue_id = str(mr["r_id"]), str(mr["b_id"])
        horizon = elapsed_seconds(mr)
        if horizon <= 0:
            raise RuntimeError(f"invalid horizon fight={fight_id}: {horizon}")

        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        inputs = EngineInputs(
            red=FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
            blue=FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
        )
        rounds = max(1, int(math.ceil(horizon / 300.0)))
        config = EngineConfig(number_of_rounds=rounds)

        actual = {
            "red": actual_side_totals(side_rows(round_stats, fight_id, red_id, "red")),
            "blue": actual_side_totals(side_rows(round_stats, fight_id, blue_id, "blue")),
        }
        for side in ("red", "blue"):
            for key, value in actual[side].items():
                if np.isfinite(value):
                    pooled_actual[key] += value
            pooled_seconds_actual += horizon

        per_path: list[dict[str, float]] = []
        for path_index in range(args.paths_per_fight):
            seed = args.seed_base + fight_index * 10000 + path_index
            result = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=config)
            exposure = defaultdict(float)
            control = defaultdict(float)
            counts = Counter()
            for segment in result.timeline_segments:
                exposure[segment.phase.value] += segment.duration
                phase_seconds[segment.phase.value] += segment.duration
                segment_durations[segment.phase.value].append(segment.duration)
                if segment.controller is not None:
                    control[segment.controller.value] += segment.duration
            if not np.isclose(sum(exposure.values()), result.reported_through_seconds, atol=1e-9):
                timeline_mismatches += 1

            for event in result.events:
                total_events += 1
                action = event.selected_action
                side = event.actor.value
                if event.source_phase is Phase.GROUND and action in STANDING_STRIKES:
                    illegal_actions += 1
                if action in STANDING_STRIKES:
                    counts[f"{side}_standing_att"] += 1
                    if event.outcome.value == "landed": counts[f"{side}_standing_land"] += 1
                elif action in CLINCH_STRIKES:
                    counts[f"{side}_clinch_att"] += 1
                    if event.outcome.value == "landed": counts[f"{side}_clinch_land"] += 1
                elif action in GROUND_STRIKES:
                    counts[f"{side}_ground_att"] += 1
                    if event.outcome.value == "landed": counts[f"{side}_ground_land"] += 1
                if action in TD_ACTIONS:
                    counts[f"{side}_td_att"] += 1
                    if event.transition_kind is not None and event.resulting_phase is Phase.GROUND:
                        counts[f"{side}_td_land"] += 1
                if action is ActionFamily.CLINCH_ENTRY:
                    counts[f"{side}_clinch_entries"] += 1
                if action is ActionFamily.SUBMISSION_ATTACK:
                    counts[f"{side}_sub_att"] += 1
                if event.transition_kind is not None:
                    transition_counts[event.transition_kind.value] += 1

            row = {"events": float(len(result.events)), **{k: float(v) for k, v in counts.items()}}
            row.update({"red_control_seconds": control["red"], "blue_control_seconds": control["blue"]})
            row.update({f"phase_{k}_seconds": v for k, v in exposure.items()})
            per_path.append(row)
            pooled_seconds_sim += horizon
            total_sim_paths += 1

        keys = set().union(*(p.keys() for p in per_path))
        sim_mean = {key: mean([float(p.get(key, 0.0)) for p in per_path]) for key in keys}
        for side in ("red", "blue"):
            pooled_sim["standing_att"] += sim_mean.get(f"{side}_standing_att", 0.0)
            pooled_sim["standing_land"] += sim_mean.get(f"{side}_standing_land", 0.0)
            pooled_sim["clinch_att"] += sim_mean.get(f"{side}_clinch_att", 0.0)
            pooled_sim["clinch_land"] += sim_mean.get(f"{side}_clinch_land", 0.0)
            pooled_sim["ground_att"] += sim_mean.get(f"{side}_ground_att", 0.0)
            pooled_sim["ground_land"] += sim_mean.get(f"{side}_ground_land", 0.0)
            pooled_sim["td_att"] += sim_mean.get(f"{side}_td_att", 0.0)
            pooled_sim["td_land"] += sim_mean.get(f"{side}_td_land", 0.0)
            pooled_sim["control_seconds"] += sim_mean.get(f"{side}_control_seconds", 0.0)
            pooled_sim["sub_att"] += sim_mean.get(f"{side}_sub_att", 0.0)

        fight_rows.append({
            "fight_id": fight_id,
            "event_date": str(mr["_event_date"].date()),
            "red_name": str(mr.get("r_name", red_id)),
            "blue_name": str(mr.get("b_name", blue_id)),
            "horizon_seconds": horizon,
            "red_prior_ufc_fights": prior_snapshot_count(snapshots, fighter_id=red_id, event_date=mr["_event_date"]),
            "blue_prior_ufc_fights": prior_snapshot_count(snapshots, fighter_id=blue_id, event_date=mr["_event_date"]),
            "red_takedown_capability": red_cap.takedown,
            "blue_takedown_capability": blue_cap.takedown,
            "actual": actual,
            "sim_mean": sim_mean,
        })

    if illegal_actions:
        raise AssertionError(f"illegal ground/standing actions: {illegal_actions}")
    if timeline_mismatches:
        raise AssertionError(f"timeline exposure mismatches: {timeline_mismatches}")

    def per15(total: float, seconds: float) -> float:
        return float(total / seconds * 900.0) if seconds > 0 else float("nan")

    comparisons = {
        # Model standing strike attempt is closest to UFCStats distance significant-strike attempt,
        # not definition-identical. Clinch and ground are similarly compared to phase sig strikes.
        "standing_vs_actual_distance_sig_att_per15": {
            "sim": per15(pooled_sim["standing_att"], pooled_seconds_actual),
            "actual": per15(pooled_actual["distance_att"], pooled_seconds_actual),
        },
        "clinch_strike_vs_actual_clinch_sig_att_per15": {
            "sim": per15(pooled_sim["clinch_att"], pooled_seconds_actual),
            "actual": per15(pooled_actual["clinch_att"], pooled_seconds_actual),
        },
        "ground_strike_vs_actual_ground_sig_att_per15": {
            "sim": per15(pooled_sim["ground_att"], pooled_seconds_actual),
            "actual": per15(pooled_actual["ground_att"], pooled_seconds_actual),
        },
        "takedown_attempts_per15": {
            "sim": per15(pooled_sim["td_att"], pooled_seconds_actual),
            "actual": per15(pooled_actual["td_att"], pooled_seconds_actual),
        },
        "takedowns_landed_per15": {
            "sim": per15(pooled_sim["td_land"], pooled_seconds_actual),
            "actual": per15(pooled_actual["td_land"], pooled_seconds_actual),
        },
        "control_seconds_per15": {
            "sim": per15(pooled_sim["control_seconds"], pooled_seconds_actual),
            "actual": per15(pooled_actual["control_seconds"], pooled_seconds_actual),
        },
        "submission_attempts_per15": {
            "sim": per15(pooled_sim["sub_att"], pooled_seconds_actual),
            "actual": per15(pooled_actual["sub_att"], pooled_seconds_actual),
        },
    }

    total_phase = sum(phase_seconds.values())
    payload = {
        "diagnostic": "Stage 8 structural population validation",
        "fights": args.fights,
        "paths_per_fight": args.paths_per_fight,
        "total_simulated_paths": total_sim_paths,
        "seed_base": args.seed_base,
        "selection": "most recent fights with round stats + exactly two canonical historical FSR V3 prefight rows; no maturity filter",
        "historical_comparator_semantics": {
            "standing": "modeled standing strike attempts vs derived UFCStats distance significant-strike attempts; closest comparator, not definition-identical",
            "clinch": "modeled clinch strike attempts vs UFCStats clinch significant-strike attempts",
            "ground": "modeled ground/bottom strike attempts vs UFCStats ground significant-strike attempts",
            "phase_time": "NO historical phase-time denominator available; simulated phase shares are simulator-only diagnostics",
        },
        "unresolved_stage6_placeholders": {
            "clinch_capability": 0.35,
            "submission_capability": 0.30,
            "escape_capability": 0.40,
            "reversal_capability": 0.30,
            "submission_success_probability": 0.0,
            "ground_escape_probability": 0.40,
            "ground_reversal_probability": 0.30,
        },
        "invariants": {
            "illegal_ground_standing_actions": illegal_actions,
            "timeline_exposure_mismatches": timeline_mismatches,
        },
        "activity": {
            "mean_events_per_minute_combined": total_events / pooled_seconds_sim * 60.0,
        },
        "simulated_phase_share": {k: v / total_phase for k, v in phase_seconds.items()},
        "mean_segment_duration_seconds": {k: mean(v) for k, v in segment_durations.items()},
        "transition_counts": dict(sorted(transition_counts.items())),
        "actual_vs_simulated": comparisons,
        "skipped_candidates": dict(skipped),
        "fight_detail": fight_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    print("=" * 100)
    print("STAGE 8 STRUCTURAL POPULATION VALIDATION")
    print("=" * 100)
    print(f"fights={args.fights} paths_per_fight={args.paths_per_fight} total_paths={total_sim_paths}")
    print("selection:", payload["selection"])
    print("invariants:", json.dumps(payload["invariants"], indent=2))
    print("simulated_phase_share:", json.dumps(payload["simulated_phase_share"], indent=2))
    print("mean_segment_duration_seconds:", json.dumps(payload["mean_segment_duration_seconds"], indent=2))
    print("actual_vs_simulated:", json.dumps(comparisons, indent=2))
    print("mean_events_per_minute_combined:", payload["activity"]["mean_events_per_minute_combined"])
    print("selected fights:")
    for row in fight_rows:
        print(row["event_date"], row["red_name"], "vs", row["blue_name"], "horizon", row["horizon_seconds"])
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
