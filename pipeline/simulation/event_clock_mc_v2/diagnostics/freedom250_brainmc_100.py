from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v2.brain.intent_priors import BrainIntentPriors
from pipeline.simulation.event_clock_mc_v2.brain.policy import BrainDecisionContext
from pipeline.simulation.event_clock_mc_v2.brain.timing import BrainTimingContext
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import (
    EngineConfig,
    EngineFunctions,
    EngineInputs,
    FighterEngineInputs,
    run_causal_path,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture
from pipeline.simulation.event_clock_mc_v2.physiology_adapter import (
    age_years_on_date,
    fighter_mechanics_from_prefight,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage6_real_causal_path import _capabilities
from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_intent_prior_shadow import IntentPriorChooser
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed

EVENT_DATE = pd.Timestamp("2026-06-14")
PATHS = 100
OUTPUT = Path("artifacts/freedom250-brainmc-100/results.json")

STAND = {ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER}
GROUND = {ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE}
TD = {ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN}
EXPECTED_NAMES = {
    "Ilia Topuria", "Justin Gaethje",
    "Alex Pereira", "Ciryl Gane",
    "Sean O'Malley", "Aiemann Zahabi",
    "Josh Hokit", "Derrick Lewis",
    "Mauricio Ruffy", "Michael Chandler",
    "Bo Nickal", "Kyle Daukaus",
    "Diego Lopes", "Steve Garcia",
}


def _norm(name: str) -> str:
    return (
        str(name)
        .casefold()
        .replace("’", "'")
        .replace("á", "a").replace("ã", "a").replace("â", "a")
        .replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    )


def _round_from_time(seconds: float, total_rounds: int) -> int:
    return min(total_rounds, max(1, int((max(seconds, 1e-9) - 1e-9) // 300) + 1))


def _mean(counter: Counter, key: str) -> float:
    return float(counter[key] / PATHS)


def main() -> None:
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
    card = master.loc[master["date"].eq(EVENT_DATE)].copy()
    if len(card) != 7:
        raise RuntimeError(f"Expected 7 Freedom 250 fights on {EVENT_DATE.date()}, found {len(card)}")

    card_names = set(card["r_name"].astype(str)) | set(card["b_name"].astype(str))
    expected_norm = {_norm(x) for x in EXPECTED_NAMES}
    actual_norm = {_norm(x) for x in card_names}
    if expected_norm != actual_norm:
        raise RuntimeError(
            "Freedom 250 name mismatch. "
            f"missing={sorted(expected_norm-actual_norm)} extra={sorted(actual_norm-expected_norm)}"
        )

    snapshots = load_prefight_snapshots()
    reference = CapabilityReference.from_prefight_before(snapshots, EVENT_DATE)
    results = []

    for _, fight in card.sort_values("fight_id").iterrows():
        bout_id = str(fight["fight_id"])
        red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])
        red_name, blue_name = str(fight["r_name"]), str(fight["b_name"])
        rounds = int(fight["total_rounds"])
        horizon = float(rounds * 300)

        red, blue = historical_fighter_rows(
            snapshots,
            event_date=EVENT_DATE,
            fight_id=bout_id,
            fighter_ids=(red_id, blue_id),
        )
        rc, rr = _capabilities(red, blue, reference)
        bc, br = _capabilities(blue, red, reference)
        red_age = age_years_on_date(fight.get("r_dob"), EVENT_DATE)
        blue_age = age_years_on_date(fight.get("b_dob"), EVENT_DATE)
        red_mech = fighter_mechanics_from_prefight(red, rr, age_years=red_age)
        blue_mech = fighter_mechanics_from_prefight(blue, br, age_years=blue_age)

        inputs = EngineInputs(
            FighterEngineInputs(rc, BrainTimingContext(), BrainDecisionContext(), red_mech),
            FighterEngineInputs(bc, BrainTimingContext(), BrainDecisionContext(), blue_mech),
            ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
        )
        chooser = IntentPriorChooser(
            {
                Side.RED: BrainIntentPriors(rr.standing_rate_15m, rr.takedown_rate_15m, 0.06, 3.0, 0.3),
                Side.BLUE: BrainIntentPriors(br.standing_rate_15m, br.takedown_rate_15m, 0.06, 3.0, 0.3),
            }
        )
        funcs = EngineFunctions(action_chooser=chooser)
        cfg = EngineConfig(number_of_rounds=rounds)

        path_counts = Counter()
        red_stats, blue_stats = Counter(), Counter()
        phase_seconds = Counter()
        finish_times = []
        finish_rounds = Counter()
        final_stamina = {"red": [], "blue": []}

        for path_id in range(PATHS):
            seed = derive_path_seed(SEED_SET_VERSION, bout_id, path_id)
            out = run_causal_path(
                inputs,
                seed=seed,
                horizon_seconds=horizon,
                config=cfg,
                functions=funcs,
            )
            for seg in out.timeline_segments:
                phase_seconds[seg.phase.value] += seg.duration
            final_stamina["red"].append(out.final_state.physiology.red.stamina)
            final_stamina["blue"].append(out.final_state.physiology.blue.stamina)

            for event in out.events:
                stats = red_stats if event.actor is Side.RED else blue_stats
                a = event.selected_action
                landed = event.outcome.value == "landed"
                if a in STAND:
                    stats["standing_att"] += 1
                    stats["standing_land"] += int(landed)
                elif a is ActionFamily.CLINCH_STRIKE:
                    stats["clinch_att"] += 1
                    stats["clinch_land"] += int(landed)
                elif a in GROUND:
                    stats["ground_att"] += 1
                    stats["ground_land"] += int(landed)
                elif a in TD:
                    stats["td_att"] += 1
                    stats["td_land"] += int(event.resulting_phase.value == "ground")
                elif a is ActionFamily.SUBMISSION_ATTACK:
                    stats["sub_att"] += 1
                stats["kds"] += int(event.knockdown)

            if out.termination is None:
                path_counts["decision_unresolved"] += 1
                path_counts["over_1_5"] += int(horizon >= 450)
                path_counts["over_2_5"] += int(horizon >= 750)
                if rounds >= 5:
                    path_counts["over_3_5"] += 1
                    path_counts["over_4_5"] += 1
                continue

            side = out.termination.winner.value
            method = out.termination.finish_method.value
            t = float(out.reported_through_seconds)
            finish_times.append(t)
            r = _round_from_time(t, rounds)
            finish_rounds[r] += 1
            path_counts[f"{side}_finish_win"] += 1
            path_counts[f"{side}_{method}"] += 1
            path_counts[method] += 1
            path_counts["under_1_5"] += int(t < 450)
            path_counts["over_1_5"] += int(t >= 450)
            path_counts["under_2_5"] += int(t < 750)
            path_counts["over_2_5"] += int(t >= 750)
            if rounds >= 5:
                path_counts["under_3_5"] += int(t < 1050)
                path_counts["over_3_5"] += int(t >= 1050)
                path_counts["under_4_5"] += int(t < 1350)
                path_counts["over_4_5"] += int(t >= 1350)

        decision_mass = path_counts["decision_unresolved"] / PATHS
        red_finish = path_counts["red_finish_win"] / PATHS
        blue_finish = path_counts["blue_finish_win"] / PATHS
        total_phase = float(sum(phase_seconds.values()))

        def stats_payload(c: Counter, stamina_values: list[float]) -> dict:
            return {
                "standing_attempts": _mean(c, "standing_att"),
                "standing_landed": _mean(c, "standing_land"),
                "clinch_attempts": _mean(c, "clinch_att"),
                "clinch_landed": _mean(c, "clinch_land"),
                "td_attempts": _mean(c, "td_att"),
                "td_landed": _mean(c, "td_land"),
                "ground_attempts": _mean(c, "ground_att"),
                "ground_landed": _mean(c, "ground_land"),
                "submission_attempts": _mean(c, "sub_att"),
                "knockdowns": _mean(c, "kds"),
                "mean_final_stamina": float(np.mean(stamina_values)),
            }

        props = {
            "fight_ko_tko": path_counts["ko_tko"] / PATHS,
            "fight_submission": path_counts["submission"] / PATHS,
            "goes_distance_unresolved": decision_mass,
            "red_ko_tko": path_counts["red_ko_tko"] / PATHS,
            "blue_ko_tko": path_counts["blue_ko_tko"] / PATHS,
            "red_submission": path_counts["red_submission"] / PATHS,
            "blue_submission": path_counts["blue_submission"] / PATHS,
            "under_1_5": path_counts["under_1_5"] / PATHS,
            "over_1_5": path_counts["over_1_5"] / PATHS,
            "under_2_5": path_counts["under_2_5"] / PATHS,
            "over_2_5": path_counts["over_2_5"] / PATHS,
        }
        if rounds >= 5:
            props.update(
                {
                    "under_3_5": path_counts["under_3_5"] / PATHS,
                    "over_3_5": path_counts["over_3_5"] / PATHS,
                    "under_4_5": path_counts["under_4_5"] / PATHS,
                    "over_4_5": path_counts["over_4_5"] / PATHS,
                }
            )
        for r in range(1, rounds + 1):
            props[f"finish_round_{r}"] = finish_rounds[r] / PATHS

        results.append(
            {
                "bout_id": bout_id,
                "date": EVENT_DATE.date().isoformat(),
                "red": red_name,
                "blue": blue_name,
                "scheduled_rounds": rounds,
                "paths": PATHS,
                "architecture": "empirical_event2",
                "moneyline": {
                    "available": False,
                    "reason": "Brain MC has no approved decision-judging layer; decision paths are intentionally unresolved.",
                    "red_finish_win_mass": red_finish,
                    "blue_finish_win_mass": blue_finish,
                    "decision_unresolved_mass": decision_mass,
                    "red_win_probability_bounds": [red_finish, red_finish + decision_mass],
                    "blue_win_probability_bounds": [blue_finish, blue_finish + decision_mass],
                },
                "props": props,
                "aggregate": {
                    "mean_duration_seconds": float(
                        (sum(finish_times) + path_counts["decision_unresolved"] * horizon) / PATHS
                    ),
                    "phase_share": {
                        "standing": phase_seconds["standing"] / total_phase if total_phase else 0.0,
                        "clinch": phase_seconds["clinch"] / total_phase if total_phase else 0.0,
                        "ground": phase_seconds["ground"] / total_phase if total_phase else 0.0,
                    },
                    "red": stats_payload(red_stats, final_stamina["red"]),
                    "blue": stats_payload(blue_stats, final_stamina["blue"]),
                },
            }
        )

    payload = {
        "source_sha": "d42b214c2583b6ce1409eb3ab5e6507aee2827ae",
        "event": "UFC Freedom 250",
        "event_date": EVENT_DATE.date().isoformat(),
        "paths_per_fight": PATHS,
        "ko_kd_architecture": "empirical_event2",
        "moneyline_limitation": "No approved Brain MC decision judge exists at this SHA; finish win mass and exact decision-unresolved mass are reported instead of fabricated ML.",
        "fights": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
