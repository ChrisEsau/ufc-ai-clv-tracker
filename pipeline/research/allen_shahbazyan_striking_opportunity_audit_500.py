"""Research-only Allen-Shahbazyan striking opportunity audit over 500 paths.

Keeps the current Allen Brain/grappling/submission stack and canonical KO V3
positive-only hazard frozen. Measures where KO hazard opportunities come from:
strike attempts, landings, phase allocation, and phase exposure.
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
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import ActionOutcome
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.diagnostics import ko_v3_positive_only_defense_nine_fight_cohort as ko_pos
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod

PATHS = 500
ROUND_STATS = Path("data/fight_details/ufc_round_stats.parquet")
OUTDIR = Path("data/research/allen_shahbazyan_striking_opportunity_audit_500")
STRIKE_FAMILIES = {
    ActionFamily.STAND_ATTACK: "standing",
    ActionFamily.STAND_COUNTER: "standing",
    ActionFamily.CLINCH_STRIKE: "clinch",
    ActionFamily.GROUND_STRIKE: "ground",
    ActionFamily.BOTTOM_STRIKE: "ground",
}


def actual_rows(fight_id: str):
    df = pd.read_parquet(ROUND_STATS)
    x = df[df.fight_id.astype(str).eq(str(fight_id))].copy()
    rows = []
    for name, g in x.groupby("fighter_name", sort=False):
        row = {
            "fighter": str(name),
            "actual_sig_attempted": float(g.sig_str_attempted.sum()),
            "actual_sig_landed": float(g.sig_str_landed.sum()),
            "actual_accuracy": float(g.sig_str_landed.sum() / g.sig_str_attempted.sum()) if g.sig_str_attempted.sum() else 0.0,
            "actual_distance_attempted": float(g.distance_attempted.sum()),
            "actual_distance_landed": float(g.distance_landed.sum()),
            "actual_clinch_attempted": float(g.clinch_attempted.sum()),
            "actual_clinch_landed": float(g.clinch_landed.sum()),
            "actual_ground_attempted": float(g.ground_attempted.sum()),
            "actual_ground_landed": float(g.ground_landed.sum()),
        }
        rows.append(row)
    return rows


def main():
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
    cutoff = pd.Timestamp(target_date).normalize()
    control_model = timing.target._expected_control_model(target_date, names)

    ff, _ = s1.load_raw_fighter_fights()
    frame = s1.build_matchup_frame(s1.build_prefight_states(ff)).copy()
    frame["fight_id"] = frame.fight_id.astype(str)
    beta_att, beta_def, age_fit = ko_pos.cohort.fit_age_slopes(frame, cutoff)
    total_by_id = ko_pos.positive_only_total_hazards_for_fight(frame, base_trace.FIGHT_ID, cutoff, beta_att, beta_def)
    kd_by_id = fit_prefight_hazards(fight_id=base_trace.FIGHT_ID)
    side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
    total_by_side = {s: total_by_id[fid] for s, fid in side_to_id.items()}
    kd_by_side = {s: kd_by_id[fid] for s, fid in side_to_id.items()}

    original_resolver = physiology_mod.resolve_empirical_ko_kd
    original_hurt = physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    ko_resolver = ko_pos.cohort.Resolver(total_by_side, kd_by_side)
    physiology_mod.resolve_empirical_ko_kd = ko_resolver
    physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

    path_rows = []
    agg = {s: Counter() for s in Side}
    phase_seconds = Counter()
    try:
        for path_id in range(PATHS):
            brain = base_trace.TraceBrain(inputs, priors, horizon)
            seed = derive_path_seed(SEED_SET_VERSION, base_trace.FIGHT_ID, path_id)
            escape = timing.target.ExpectedControlEscapeResolver(control_model, seed)
            funcs = EngineFunctions(brain.timing_sampler, brain.action_chooser, escape)
            out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
            per = {s: Counter() for s in Side}
            for seg in out.timeline_segments:
                phase_seconds[seg.phase.value] += float(seg.duration)
            for e in out.events:
                family = e.selected_action
                if family not in STRIKE_FAMILIES:
                    continue
                side = e.actor
                phase = STRIKE_FAMILIES[family]
                per[side][f"{phase}_attempted"] += 1
                per[side]["sig_attempted"] += 1
                if e.outcome is ActionOutcome.LANDED:
                    per[side][f"{phase}_landed"] += 1
                    per[side]["sig_landed"] += 1
            row = {"path_id": path_id, "reported_through_seconds": float(out.reported_through_seconds), "method": out.termination.finish_method.value}
            for side in Side:
                prefix = "allen" if side is Side.RED else "shahbazyan"
                for key in ("sig_attempted","sig_landed","standing_attempted","standing_landed","clinch_attempted","clinch_landed","ground_attempted","ground_landed"):
                    row[f"{prefix}_{key}"] = int(per[side][key])
                    agg[side][key] += per[side][key]
            path_rows.append(row)
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt

    total_exposure = sum(float(r["reported_through_seconds"]) for r in path_rows)
    summary = []
    actual = {r["fighter"]: r for r in actual_rows(base_trace.FIGHT_ID)}
    for side in Side:
        fighter = names[side]
        a = agg[side]
        sim_att = a["sig_attempted"] / PATHS
        sim_land = a["sig_landed"] / PATHS
        row = {
            "fighter": fighter,
            "paths": PATHS,
            "mean_simulated_fight_seconds": total_exposure / PATHS,
            "sim_sig_attempted_per_path": sim_att,
            "sim_sig_landed_per_path": sim_land,
            "sim_accuracy": sim_land / sim_att if sim_att else 0.0,
            "sim_sig_attempted_per_15m_exposure": a["sig_attempted"] / total_exposure * 900.0,
            "sim_sig_landed_per_15m_exposure": a["sig_landed"] / total_exposure * 900.0,
            "sim_standing_attempted_per_path": a["standing_attempted"] / PATHS,
            "sim_standing_landed_per_path": a["standing_landed"] / PATHS,
            "sim_clinch_attempted_per_path": a["clinch_attempted"] / PATHS,
            "sim_clinch_landed_per_path": a["clinch_landed"] / PATHS,
            "sim_ground_attempted_per_path": a["ground_attempted"] / PATHS,
            "sim_ground_landed_per_path": a["ground_landed"] / PATHS,
            "ko_hazard_per_landed": float(total_by_side[side]["total_ko_per_landed"]),
            "ko_hazard_evaluations": int(ko_resolver.landed[side]),
            "ko_finishes": int(ko_resolver.kos[side]),
            **actual.get(fighter, {}),
        }
        summary.append(row)

    phase_summary = [{"phase": p, "total_seconds": float(v), "mean_seconds_per_path": float(v / PATHS), "share_of_sim_exposure": float(v / total_exposure)} for p, v in phase_seconds.items()]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(path_rows).to_csv(OUTDIR / "paths.csv", index=False)
    pd.DataFrame(summary).to_csv(OUTDIR / "fighter_summary.csv", index=False)
    pd.DataFrame(phase_summary).to_csv(OUTDIR / "phase_summary.csv", index=False)
    payload = {
        "study": "Allen-Shahbazyan 500-path striking opportunity audit",
        "production_changed": False,
        "ko_hazard_changed": False,
        "brain_opportunity_changed": False,
        "paths": PATHS,
        "actual": list(actual.values()),
        "fighter_summary": summary,
        "phase_summary": phase_summary,
        "age_fit": age_fit,
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
