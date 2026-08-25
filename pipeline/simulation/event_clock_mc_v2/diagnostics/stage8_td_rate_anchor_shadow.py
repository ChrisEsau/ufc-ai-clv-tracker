"""Shadow Stage 8 test of an absolute FSR TD-rate anchor for brain action choice.

This is NOT production policy. It tests one causal hypothesis without tuning:
validated matchup-effective FSR V3 TD attempts/15 should determine the baseline
share of brain action opportunities allocated to takedown attempts, while TD
completion remains mechanics.

At standing/clinch action opportunities:
    P(TD action) = matchup_td_rate_per15 / neutral_brain_opportunities_per15
All non-TD legal actions retain their existing Stage 5 relative probabilities.
Ground policy is untouched. Dynamic decision context remains neutral, matching
Stage 8 structural-population scope.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext, action_probabilities
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext, expected_action_delay
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import FightState, Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineConfig, EngineFunctions, EngineInputs, FighterEngineInputs, run_causal_path
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows, load_latest_profiles, load_prefight_snapshots
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import CapabilityReference
from .stage6_real_causal_path import _capabilities, _mechanics
from .stage8_structural_population import MASTER, ROUND_STATS, actual_side_totals, elapsed_seconds, pick_col, side_rows

STANDING_STRIKES = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
GROUND_STRIKES = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD_ACTIONS = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}


class RateAnchoredChooser:
    def __init__(self, td_rate_per15: dict[Side, float], opportunities_per15: float) -> None:
        self.td_rate_per15 = td_rate_per15
        self.opportunities_per15 = opportunities_per15

    def __call__(self, state, actor, capabilities, context, rng, config) -> ActionFamily:
        rows = action_probabilities(state, actor, capabilities, context, config)
        actions = [row.action_family for row in rows]
        probs = np.asarray([row.probability for row in rows], dtype=float)
        target = None
        if state.phase is Phase.STANDING:
            target = ActionFamily.TAKEDOWN_ENTRY
        elif state.phase is Phase.CLINCH:
            target = ActionFamily.CLINCH_TAKEDOWN

        if target is not None and target in actions:
            index = actions.index(target)
            desired = float(np.clip(self.td_rate_per15[actor] / self.opportunities_per15, 0.0, 0.95))
            non_target = 1.0 - probs[index]
            if non_target <= 0.0:
                probs[:] = 0.0
                probs[index] = 1.0
            else:
                scale = (1.0 - desired) / non_target
                probs *= scale
                probs[index] = desired
            probs /= probs.sum()
        return actions[int(rng.choice(len(actions), p=probs))]


def per15(total: float, seconds: float) -> float:
    return float(total * 900.0 / seconds) if seconds > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=25)
    ap.add_argument("--paths-per-fight", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=20260825)
    ap.add_argument("--output", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/stage8_td_rate_anchor_shadow.json"))
    args = ap.parse_args()

    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = pick_col(master, "date", "event_date")
    master["_event_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    master = master.dropna(subset=["_event_date"]).sort_values(["_event_date", "fight_id"], ascending=[False, False])
    round_stats = pd.read_parquet(ROUND_STATS).copy()
    rs_fight_col = pick_col(round_stats, "fight_id", "bout_id")
    available = set(round_stats[rs_fight_col].astype(str))
    snapshots = load_prefight_snapshots()
    latest = load_latest_profiles()
    reference = CapabilityReference.from_latest(latest)

    neutral_timing = BrainTimingContext()
    neutral_decision = BrainDecisionContext()
    opportunities_per15 = 900.0 / expected_action_delay(FightState(), neutral_timing)

    selected = []
    for _, fight in master.iterrows():
        if len(selected) >= args.fights:
            break
        fight_id = str(fight["fight_id"])
        if fight_id not in available:
            continue
        try:
            red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
            red_fsr, blue_fsr = historical_fighter_rows(snapshots, event_date=fight["_event_date"], fight_id=fight_id, fighter_ids=(red_id, blue_id))
            side_rows(round_stats, fight_id, red_id, "red")
            side_rows(round_stats, fight_id, blue_id, "blue")
        except Exception:
            continue
        selected.append((fight, red_fsr, blue_fsr))
    if len(selected) < args.fights:
        raise RuntimeError(f"only {len(selected)} complete fights; requested {args.fights}")

    actual = defaultdict(float)
    sim = defaultdict(float)
    actual_seconds = 0.0
    sim_seconds = 0.0
    phase_seconds = defaultdict(float)
    total_paths = 0
    illegal = 0
    exposure_mismatch = 0
    fight_rows = []

    for fight_index, (fight, red_fsr, blue_fsr) in enumerate(selected):
        fight_id = str(fight["fight_id"])
        red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
        horizon = elapsed_seconds(fight)
        red_cap, red_runtime = _capabilities(red_fsr, blue_fsr, reference)
        blue_cap, blue_runtime = _capabilities(blue_fsr, red_fsr, reference)
        inputs = EngineInputs(
            FighterEngineInputs(red_cap, neutral_timing, neutral_decision, _mechanics(red_runtime)),
            FighterEngineInputs(blue_cap, neutral_timing, neutral_decision, _mechanics(blue_runtime)),
        )
        chooser = RateAnchoredChooser(
            {Side.RED: red_runtime.takedown_rate_15m, Side.BLUE: blue_runtime.takedown_rate_15m},
            opportunities_per15,
        )
        functions = EngineFunctions(action_chooser=chooser)
        rounds = max(1, int(math.ceil(horizon / 300.0)))
        config = EngineConfig(number_of_rounds=rounds)

        fight_actual = {}
        for side, fighter_id in (("red", red_id), ("blue", blue_id)):
            values = actual_side_totals(side_rows(round_stats, fight_id, fighter_id, side))
            fight_actual[side] = values
            actual["standing_att"] += values["distance_att"]
            actual["ground_att"] += values["ground_att"]
            actual["td_att"] += values["td_att"]
            actual["td_land"] += values["td_land"]
            actual["sub_att"] += values["sub_att"]
            actual_seconds += horizon

        path_counts = []
        for path_index in range(args.paths_per_fight):
            result = run_causal_path(inputs, seed=args.seed_base + fight_index * 10000 + path_index, horizon_seconds=horizon, config=config, functions=functions)
            counts = Counter()
            exposure = defaultdict(float)
            for segment in result.timeline_segments:
                exposure[segment.phase.value] += segment.duration
                phase_seconds[segment.phase.value] += segment.duration
            if not np.isclose(sum(exposure.values()), result.reported_through_seconds, atol=1e-9):
                exposure_mismatch += 1
            for event in result.events:
                action = event.selected_action
                if event.source_phase is Phase.GROUND and action in STANDING_STRIKES:
                    illegal += 1
                if action in STANDING_STRIKES:
                    counts["standing_att"] += 1
                if action in GROUND_STRIKES:
                    counts["ground_att"] += 1
                if action in TD_ACTIONS:
                    counts["td_att"] += 1
                    if event.transition_kind is not None and event.resulting_phase is Phase.GROUND:
                        counts["td_land"] += 1
                if action is ActionFamily.SUBMISSION_ATTACK:
                    counts["sub_att"] += 1
            path_counts.append(counts)
            sim_seconds += horizon
            total_paths += 1

        for key in ("standing_att", "ground_att", "td_att", "td_land", "sub_att"):
            sim[key] += float(np.mean([row[key] for row in path_counts]))
        fight_rows.append({
            "fight_id": fight_id,
            "red_name": str(fight.get("r_name", red_id)),
            "blue_name": str(fight.get("b_name", blue_id)),
            "horizon_seconds": horizon,
            "red_fsr_td_rate_per15": float(red_runtime.takedown_rate_15m),
            "blue_fsr_td_rate_per15": float(blue_runtime.takedown_rate_15m),
        })

    if illegal or exposure_mismatch:
        raise AssertionError({"illegal": illegal, "timeline_mismatches": exposure_mismatch})

    phase_total = sum(phase_seconds.values())
    payload = {
        "diagnostic": "Stage 8 shadow FSR TD-rate anchor",
        "production_policy_changed": False,
        "fights": len(selected),
        "paths_per_fight": args.paths_per_fight,
        "total_paths": total_paths,
        "neutral_opportunities_per15": opportunities_per15,
        "invariants": {"illegal_ground_standing_actions": illegal, "timeline_exposure_mismatches": exposure_mismatch},
        "simulated_phase_share": {key: value / phase_total for key, value in phase_seconds.items()},
        "actual_vs_shadow_per15": {
            key: {"actual": per15(actual[key], actual_seconds), "shadow": per15(sim[key], actual_seconds)}
            for key in ("standing_att", "ground_att", "td_att", "td_land", "sub_att")
        },
        "fights_detail": fight_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 100)
    print("STAGE 8 SHADOW TD RATE ANCHOR")
    print("=" * 100)
    print(json.dumps(payload, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
