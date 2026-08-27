"""Research-only 500-path Allen-Shahbazyan MC using the ground-opportunity submission shadow."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.research import allen_shahbazyan_ground_opportunity_submission_trace as shadow
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod

PATHS = 500
OUTDIR = Path("data/research/allen_shahbazyan_ground_opportunity_submission_500")


def main():
    # Install the exact same research-only stack used by the successful one-path shadow.
    sub_mod.RATE_PER_15_BY_SIDE = sub_mod._build_submission_rates()
    base_trace.action_probabilities_with_intent_priors = shadow._ground_opportunity_submission_probs
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

    # Match the successful one-path stack: empirical expected-control duration
    # threshold resolver, not the older Weibull helper.
    control_model = timing.target._expected_control_model(target_date, names)

    results = []
    counts = Counter()
    for path_id in range(PATHS):
        brain = base_trace.TraceBrain(inputs, priors, horizon)
        seed = derive_path_seed(SEED_SET_VERSION, base_trace.FIGHT_ID, path_id)
        resolver = timing.target.ExpectedControlEscapeResolver(control_model, seed)
        funcs = EngineFunctions(
            timing_sampler=brain.timing_sampler,
            action_chooser=brain.action_chooser,
            mechanics_resolver=resolver,
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

    rows = []
    methods = sorted({m for (_, m) in counts})
    for fighter in (names[Side.RED], names[Side.BLUE]):
        wins = sum(v for (w, _), v in counts.items() if w == fighter)
        row = {
            "fighter": fighter,
            "wins": wins,
            "ml_probability": wins / PATHS,
        }
        for method in methods:
            row[f"{method}_wins"] = counts[(fighter, method)]
            row[f"{method}_probability"] = counts[(fighter, method)] / PATHS
        rows.append(row)

    summary = pd.DataFrame(rows)
    paths = pd.DataFrame(results)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    paths.to_csv(OUTDIR / "paths.csv", index=False)
    payload = {
        "study": "Allen-Shahbazyan 500-path ground-opportunity submission shadow",
        "production_changed": False,
        "paths": PATHS,
        "fight_id": base_trace.FIGHT_ID,
        "seed_set": SEED_SET_VERSION,
        "ground_hazard_multiplier": shadow.GROUND_HAZARD_MULTIPLIER,
        "methods_observed": methods,
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
