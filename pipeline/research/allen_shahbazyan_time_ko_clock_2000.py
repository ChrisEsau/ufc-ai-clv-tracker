"""Research-only Allen-Shahbazyan Brain MC with a time-based KO/TKO clock.

Architecture under test:
- Current Allen Brain opportunity/grappling/submission research stack is frozen.
- Strike-level KO/TKO termination is disabled.
- Empirical KD generation is retained (with the existing research hurt increment disabled,
  matching the prior canonical KO V3 comparison harness).
- KO/TKO is an independent piecewise-exponential competing-risk clock using the
  OOS-selected fighter offense/opponent vulnerability time-survival model.
- If a sampled KO time occurs before the underlying Brain termination, it overrides
  that later outcome. Otherwise the underlying submission/decision stands.

No production mechanics are changed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research import allen_shahbazyan_ground_opportunity_submission_trace as sub_shadow
from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.research import ko_time_survival_oos as surv
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import physiology as physiology_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical

PATHS = 2000
PRIOR_EVENTS = 2.0
OUTDIR = Path("data/research/allen_shahbazyan_time_ko_clock_2000")


def _time_clock_inputs():
    ff = surv.add_prefight(surv.load_fighter_fights())
    target = ff[ff.fight_id.astype(str).eq(base_trace.FIGHT_ID)].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected two target fighter rows, got {len(target)}")
    cutoff = pd.Timestamp(target.event_date.iloc[0]).normalize()
    train = ff[ff.event_date < cutoff].copy()
    p0, piece = surv.train_baselines(train)
    prior_sec = PRIOR_EVENTS / p0
    by_name = {}
    for r in target.itertuples(index=False):
        att_rate = (float(r.prior_ko_win) + PRIOR_EVENTS) / (float(r.prior_seconds) + prior_sec)
        def_rate = (float(r.opp_prior_ko_loss) + PRIOR_EVENTS) / (float(r.opp_prior_seconds) + prior_sec)
        rr = float(np.clip(att_rate * def_rate / (p0 * p0), 0.05, 20.0))
        hazards = np.asarray(piece, float) * rr
        by_name[str(r.fighter_name)] = {
            "prior_ko_wins": float(r.prior_ko_win),
            "prior_seconds": float(r.prior_seconds),
            "opponent_prior_ko_losses": float(r.opp_prior_ko_loss),
            "opponent_prior_seconds": float(r.opp_prior_seconds),
            "attacker_rate_per_minute": float(att_rate * 60.0),
            "defender_vulnerability_per_minute": float(def_rate * 60.0),
            "rate_ratio": rr,
            "hazards_per_second": hazards,
        }
    return cutoff, float(p0), np.asarray(piece, float), by_name


def _sample_piecewise_event_time(rng: np.random.Generator, hazards: np.ndarray, horizon: float) -> float | None:
    """Sample first event time from a piecewise-constant hazard on 5-minute intervals."""
    threshold = float(rng.exponential(1.0))
    cumulative = 0.0
    start = 0.0
    for h in np.asarray(hazards, float):
        if start >= horizon:
            break
        dt = min(300.0, horizon - start)
        mass = float(h) * dt
        if threshold <= cumulative + mass and h > 0:
            return float(start + (threshold - cumulative) / float(h))
        cumulative += mass
        start += dt
    return None


class NoKOKDResolver:
    """Disable strike-level KO while retaining the existing empirical KD process."""
    def __call__(self, *, state, attacker_side, attacker, defender, rng):
        target = state.physiology.fighter(attacker_side.opponent)
        prior = int(target.knockdowns_suffered)
        stamina = float(state.physiology.fighter(attacker_side).stamina)
        p_kd = ko_kd_empirical.kd_probability(
            attacker,
            defender,
            prior_defender_kds=prior,
            elapsed_seconds=float(state.fight_time_seconds),
            attacker_stamina=stamina,
        )
        kd = bool(rng.random() < p_kd)
        return ko_kd_empirical.EmpiricalKOKDResult(
            ko_probability=0.0,
            ko_tko=False,
            kd_probability=float(p_kd),
            knockdown=kd,
            prior_defender_kds=prior,
        )


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
    control_model = timing.target._expected_control_model(getattr(fight, "date", None), names)

    cutoff, p0, baseline_piece, clock = _time_clock_inputs()
    hazards_by_side = {side: clock[names[side]]["hazards_per_second"] for side in (Side.RED, Side.BLUE)}

    original_resolver = physiology_mod.resolve_empirical_ko_kd
    original_hurt = physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT
    physiology_mod.resolve_empirical_ko_kd = NoKOKDResolver()
    physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = 0.0

    counts = Counter()
    base_counts = Counter()
    clock_counts = Counter()
    paths = []
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
            base_winner = names[out.termination.winner]
            base_method = out.termination.finish_method.value
            base_time = float(out.reported_through_seconds)
            base_counts[(base_winner, base_method)] += 1

            # Separate deterministic RNG stream for the KO clocks so Brain path generation is unchanged.
            clock_rng = np.random.default_rng((int(seed) ^ 0x6B4F434C4F434B) & ((1 << 63) - 1))
            sampled = []
            for side in (Side.RED, Side.BLUE):
                t = _sample_piecewise_event_time(clock_rng, hazards_by_side[side], float(horizon))
                if t is not None:
                    sampled.append((float(t), side))
            sampled.sort(key=lambda x: x[0])

            if sampled and sampled[0][0] < base_time:
                final_time, final_side = sampled[0]
                final_winner = names[final_side]
                final_method = "ko_tko"
                clock_counts[final_winner] += 1
                clock_triggered = True
            else:
                final_time = base_time
                final_winner = base_winner
                final_method = base_method
                clock_triggered = False

            counts[(final_winner, final_method)] += 1
            paths.append({
                "path_id": path_id,
                "seed": int(seed),
                "base_winner": base_winner,
                "base_method": base_method,
                "base_end_seconds": base_time,
                "allen_clock_time": next((t for t,s in sampled if s is Side.RED), np.nan),
                "shahbazyan_clock_time": next((t for t,s in sampled if s is Side.BLUE), np.nan),
                "clock_triggered": clock_triggered,
                "winner": final_winner,
                "method": final_method,
                "end_seconds": final_time,
            })
    finally:
        physiology_mod.resolve_empirical_ko_kd = original_resolver
        physiology_mod.EMPIRICAL_KD_HURT_ACUTE_INCREMENT = original_hurt

    rows = []
    for side in (Side.RED, Side.BLUE):
        fighter = names[side]
        wins = sum(v for (w,_),v in counts.items() if w == fighter)
        c = clock[fighter]
        row = {
            "fighter": fighter,
            "wins": wins,
            "ml_probability": wins / PATHS,
            "decision_wins": counts[(fighter, "decision")],
            "decision_probability": counts[(fighter, "decision")] / PATHS,
            "ko_tko_wins": counts[(fighter, "ko_tko")],
            "ko_tko_probability": counts[(fighter, "ko_tko")] / PATHS,
            "submission_wins": counts[(fighter, "submission")],
            "submission_probability": counts[(fighter, "submission")] / PATHS,
            "ko_clock_triggered_wins": clock_counts[fighter],
            "population_hazard_per_minute": p0 * 60.0,
            "attacker_rate_per_minute": c["attacker_rate_per_minute"],
            "defender_vulnerability_per_minute": c["defender_vulnerability_per_minute"],
            "matchup_rate_ratio": c["rate_ratio"],
            **{f"r{i+1}_hazard_per_second": float(c["hazards_per_second"][i]) for i in range(5)},
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    path_df = pd.DataFrame(paths)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    path_df.to_csv(OUTDIR / "paths.csv", index=False)
    payload = {
        "study": "Allen-Shahbazyan current Brain + piecewise time KO competing clock 2000",
        "production_changed": False,
        "paths": PATHS,
        "fight_id": base_trace.FIGHT_ID,
        "cutoff": str(cutoff.date()),
        "ko_architecture": "piecewise time survival clock; strike KO disabled; KD retained",
        "prior_events": PRIOR_EVENTS,
        "baseline_hazard_per_second": baseline_piece.tolist(),
        "summary": rows,
        "base_outcomes_without_ko_clock": [
            {"winner": w, "method": m, "count": int(n), "probability": n / PATHS}
            for (w,m),n in sorted(base_counts.items())
        ],
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\nSUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
