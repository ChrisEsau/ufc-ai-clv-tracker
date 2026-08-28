"""Research-only dump of the exact judge-scored outputs for Allen-Shahbazyan.

Replays the same current Brain + time-KO shadow setup and reports the five
round-level judge inputs for decision paths:
- landed significant strikes
- knockdowns
- completed takedowns
- submission attempts
- control seconds

Production mechanics are unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from pipeline.research import allen_shahbazyan_time_ko_clock_2000 as base
from pipeline.research import allen_shahbazyan_ground_opportunity_submission_trace as sub_shadow
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod

PATHS = 2000
OUTDIR = Path("data/research/allen_shahbazyan_decision_scored_outputs_2000")
SIGNIFICANT = frozenset({
    ActionFamily.STAND_ATTACK, ActionFamily.STAND_COUNTER, ActionFamily.CLINCH_STRIKE,
    ActionFamily.GROUND_STRIKE, ActionFamily.BOTTOM_STRIKE,
})
TAKEDOWNS = frozenset({ActionFamily.TAKEDOWN_ENTRY, ActionFamily.CLINCH_TAKEDOWN})


def round_absolute_stats(out, round_number: int, round_length: float = 300.0):
    start, end = (round_number - 1) * round_length, round_number * round_length
    stats = {side: dict(sig=0, kd=0, td=0, sub=0, ctrl=0.0) for side in Side}
    for event in out.events:
        if not start <= event.timestamp_seconds < end:
            continue
        row = stats[event.actor]
        if event.selected_action in SIGNIFICANT and event.outcome.value == "landed":
            row["sig"] += 1
        row["kd"] += int(event.knockdown)
        if event.selected_action in TAKEDOWNS and event.transition_kind is not None:
            row["td"] += 1
        if event.selected_action is ActionFamily.SUBMISSION_ATTACK:
            row["sub"] += 1
    for segment in out.timeline_segments:
        overlap = max(0.0, min(segment.end_time, end) - max(segment.start_time, start))
        if overlap and segment.controller is not None:
            stats[segment.controller]["ctrl"] += overlap
    return stats


def main():
    # Recreate the exact Brain research stack used by the 2,000-path time-KO run.
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
    control_model = timing.target._expected_control_model(getattr(fight, "date", None), names)
    cutoff, p0, baseline_piece, clock = base._time_clock_inputs()
    hazards_by_side = {side: clock[names[side]]["hazards_per_second"] for side in Side}

    original_resolver = physiology_mod.resolve_empirical_ko_kd
    original_hurt = physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    physiology_mod.resolve_empirical_ko_kd = base.NoKOKDResolver()
    physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

    rows = []
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
            if out.termination is None or out.termination.finish_method.value != "decision":
                continue

            base_winner = names[out.termination.winner]
            # Same independent clock stream as the time-KO harness.
            clock_rng = np.random.default_rng((int(seed) ^ 0x6B4F434C4F434B) & ((1 << 63) - 1))
            sampled = []
            for side in Side:
                t = base._sample_piecewise_event_time(clock_rng, hazards_by_side[side], float(horizon))
                if t is not None:
                    sampled.append((float(t), side))
            sampled.sort(key=lambda x: x[0])
            survives_time_ko = not sampled or sampled[0][0] >= float(out.reported_through_seconds)

            for rnd in range(1, cfg.number_of_rounds + 1):
                stats = round_absolute_stats(out, rnd, cfg.round_length_seconds)
                red, blue = stats[Side.RED], stats[Side.BLUE]
                p_red = float(out.decision.round_probabilities[rnd - 1]) if out.decision else np.nan
                row = {
                    "path_id": path_id,
                    "seed": int(seed),
                    "base_decision_winner": base_winner,
                    "survives_time_ko": bool(survives_time_ko),
                    "round": rnd,
                    "p_allen_round": p_red,
                }
                for key in ("sig", "kd", "td", "sub", "ctrl"):
                    row[f"allen_{key}"] = red[key]
                    row[f"shah_{key}"] = blue[key]
                    row[f"{key}_diff_allen_minus_shah"] = red[key] - blue[key]
                rows.append(row)
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt

    detail = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTDIR / "decision_round_scored_outputs.csv", index=False)

    # Aggregate fight totals (sum rounds) by base decision winner and by final surviving decision winner.
    totals = detail.groupby(["path_id", "base_decision_winner", "survives_time_ko"], as_index=False).agg({
        **{f"allen_{k}": "sum" for k in ("sig","kd","td","sub","ctrl")},
        **{f"shah_{k}": "sum" for k in ("sig","kd","td","sub","ctrl")},
        "p_allen_round": "mean",
    })
    totals.to_csv(OUTDIR / "decision_path_scored_totals.csv", index=False)

    summary_rows = []
    for scope, df in [("base_decisions", totals), ("final_decisions_after_time_ko", totals[totals.survives_time_ko])]:
        for winner, g in df.groupby("base_decision_winner"):
            row = {"scope": scope, "decision_winner": winner, "paths": int(len(g)), "mean_p_allen_round": float(g.p_allen_round.mean())}
            for k in ("sig","kd","td","sub","ctrl"):
                row[f"mean_allen_{k}"] = float(g[f"allen_{k}"].mean())
                row[f"mean_shah_{k}"] = float(g[f"shah_{k}"].mean())
                row[f"mean_diff_{k}_allen_minus_shah"] = float((g[f"allen_{k}"] - g[f"shah_{k}"]).mean())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTDIR / "summary.csv", index=False)

    payload = {
        "study": "Allen-Shahbazyan exact judge-scored decision outputs 2000",
        "production_changed": False,
        "paths": PATHS,
        "fight_id": base_trace.FIGHT_ID,
        "cutoff": str(cutoff.date()),
        "judge_features": ["landed significant strikes", "knockdowns", "completed takedowns", "submission attempts", "control seconds"],
        "summary": summary.to_dict(orient="records"),
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
