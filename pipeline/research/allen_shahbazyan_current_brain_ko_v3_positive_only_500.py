"""Research-only Allen-Shahbazyan 500-path diagnostic.

Combines the current Allen Brain opportunity/grappling/submission research stack with
canonical KO V3 positive-only defender susceptibility hazard architecture.

IMPORTANT:
- No production changes.
- Does NOT restore the old KO nine-fight STANDING_ATTEMPT_SCALE=0.25 override.
- KO is checked once per landed significant strike.
- If KO does not occur, KD is checked separately.
- KD cannot directly finish and does not trigger a second KO loop.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.research import allen_shahbazyan_ground_opportunity_submission_trace as sub_shadow
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.research.ko_v3_from_scratch_shadow import fit_prefight_hazards
from pipeline.research import ko_v3_from_scratch_stage1 as s1
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import ko_v3_positive_only_defense_nine_fight_cohort as ko_pos
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod

PATHS = 500
OUTDIR = Path("data/research/allen_shahbazyan_current_brain_ko_v3_positive_only_500")


def main():
    # Current research Brain stack from the successful one-path / 500-path submission work.
    sub_mod.RATE_PER_15_BY_SIDE = sub_mod._build_submission_rates()
    base_trace.action_probabilities_with_intent_priors = sub_shadow._ground_opportunity_submission_probs
    timing._prefight_td_decomposition()
    timing.CLINCH_RATE_BY_SIDE = timing._build_clinch_rates()
    base_trace._standing_rates_no_reset = timing._new_timing_rates
    timing.target._standing_rates_no_reset = timing._new_timing_rates

    pressure_mod.FIGHT_ID = base_trace.FIGHT_ID
    pressure_mod.PATHS = PATHS
    fight, inputs, priors, horizon, cfg = pressure_mod.build_setup()
    names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
    target_date = getattr(fight, "date", None) or getattr(fight, "event_date", None)
    if target_date is None:
        raise RuntimeError("fight date unavailable")
    cutoff = pd.Timestamp(target_date).normalize()

    # Same expected-control escape architecture as the current Allen stack.
    control_model = timing.target._expected_control_model(target_date, names)

    # Canonical KO V3 positive-only defender hazard inputs.
    ff, _ = s1.load_raw_fighter_fights()
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff)).copy()
    frame["fight_id"] = frame.fight_id.astype(str)
    beta_att, beta_def, age_fit = ko_pos.cohort.fit_age_slopes(frame, cutoff)
    total_by_id = ko_pos.positive_only_total_hazards_for_fight(
        frame, base_trace.FIGHT_ID, cutoff, beta_att, beta_def
    )
    kd_by_id = fit_prefight_hazards(fight_id=base_trace.FIGHT_ID)
    side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
    total_by_side = {s: total_by_id[fid] for s, fid in side_to_id.items()}
    kd_by_side = {s: kd_by_id[fid] for s, fid in side_to_id.items()}

    # Exact KO V3 one-roll KO then separate KD resolver. Disable old hurt increment.
    original_resolver = physiology_mod.resolve_empirical_ko_kd
    original_hurt = physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    ko_resolver = ko_pos.cohort.Resolver(total_by_side, kd_by_side)
    physiology_mod.resolve_empirical_ko_kd = ko_resolver
    physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

    results = []
    counts = Counter()
    try:
        for path_id in range(PATHS):
            brain = base_trace.TraceBrain(inputs, priors, horizon)
            seed = derive_path_seed(SEED_SET_VERSION, base_trace.FIGHT_ID, path_id)
            escape_resolver = timing.target.ExpectedControlEscapeResolver(control_model, seed)
            funcs = EngineFunctions(
                timing_sampler=brain.timing_sampler,
                action_chooser=brain.action_chooser,
                mechanics_resolver=escape_resolver,
            )
            out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
            if out.termination is None:
                raise RuntimeError(f"path {path_id} ended without termination")
            winner = names[out.termination.winner]
            method = out.termination.finish_method.value
            counts[(winner, method)] += 1
            sub_attempts = Counter()
            for e in out.events:
                if e.submission_attempt:
                    sub_attempts[names[e.actor]] += 1
            results.append({
                "path_id": path_id,
                "seed": seed,
                "winner": winner,
                "method": method,
                "reported_through_seconds": float(out.reported_through_seconds),
                "allen_submission_attempts": int(sub_attempts[names[Side.RED]]),
                "shahbazyan_submission_attempts": int(sub_attempts[names[Side.BLUE]]),
            })
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt

    methods = sorted({m for (_, m) in counts})
    rows = []
    for side in (Side.RED, Side.BLUE):
        fighter = names[side]
        wins = sum(v for (w, _), v in counts.items() if w == fighter)
        h = total_by_side[side]
        row = {
            "fighter": fighter,
            "wins": wins,
            "ml_probability": wins / PATHS,
            "population_ko_per_sig": h["population_ko_per_sig"],
            "raw_attacker_ko_per_sig": h["raw_attacker_ko_per_sig"],
            "raw_defender_ko_loss_per_sig": h["raw_defender_ko_loss_per_sig"],
            "shrunk_attacker_ko_per_sig": h["shrunk_attacker_ko_per_sig"],
            "shrunk_defender_ko_loss_per_sig": h["shrunk_defender_ko_loss_per_sig"],
            "defender_positive_only_logit_delta": h["defender_positive_only_logit_delta"],
            "age_logodds_delta": h["age_logodds_delta"],
            "total_ko_per_landed": h["total_ko_per_landed"],
            "kd_per_landed": float(kd_by_side[side].kd_per_landed),
            "landed_ko_resolutions": int(ko_resolver.landed[side]),
            "ko_finishes": int(ko_resolver.kos[side]),
            "knockdowns": int(ko_resolver.kds[side]),
        }
        for method in ("decision", "ko_tko", "submission"):
            row[f"{method}_wins"] = counts[(fighter, method)]
            row[f"{method}_probability"] = counts[(fighter, method)] / PATHS
        rows.append(row)

    summary = pd.DataFrame(rows)
    paths = pd.DataFrame(results)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    paths.to_csv(OUTDIR / "paths.csv", index=False)
    payload = {
        "study": "Allen-Shahbazyan current Brain + KO V3 positive-only defender 500",
        "production_changed": False,
        "paths": PATHS,
        "fight_id": base_trace.FIGHT_ID,
        "seed_set": SEED_SET_VERSION,
        "standing_attempt_scale_override": None,
        "ko_architecture": "O50 + positive-only D50 + chronological age; one KO roll then separate KD",
        "kd_can_finish": False,
        "post_kd_finish_loop": False,
        "ground_submission_hazard_multiplier": sub_shadow.GROUND_HAZARD_MULTIPLIER,
        "age_fit": age_fit,
        "summary": rows,
        "mean_submission_attempts": {
            names[Side.RED]: float(paths["allen_submission_attempts"].mean()),
            names[Side.BLUE]: float(paths["shahbazyan_submission_attempts"].mean()),
        },
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
