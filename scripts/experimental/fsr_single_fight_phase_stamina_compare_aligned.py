"""Paired single-fight comparison: recovery baseline vs phase-stamina V3.2."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v3_2_phase_stamina as phase_stamina
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 250
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)


def _run(sim_cls, red, blue, *, seeds, rounds, r_age, b_age):
    totals = [
        dict(ko=0.0, r1_ko=0.0, sig_att=0.0, sig_landed=0.0, damage=0.0,
             kd=0.0, max_strike=0.0, final_stamina=0.0,
             final_fatigue_penalty=0.0, final_effective_power=0.0,
             finish_round_sum=0.0, finishes=0.0),
        dict(ko=0.0, r1_ko=0.0, sig_att=0.0, sig_landed=0.0, damage=0.0,
             kd=0.0, max_strike=0.0, final_stamina=0.0,
             final_fatigue_penalty=0.0, final_effective_power=0.0,
             finish_round_sum=0.0, finishes=0.0),
    ]
    for seed in seeds:
        sim = sim_cls(
            red, blue, collapse=STRONG_COLLAPSE, rounds=rounds, seed=int(seed),
            red_age=r_age, blue_age=b_age,
        )
        result = sim.run()
        for i, stats in enumerate(sim.stats):
            assert isinstance(stats, damage.DamageFighterStats)
            totals[i]["sig_att"] += stats.sig_att
            totals[i]["sig_landed"] += stats.sig_landed
            totals[i]["damage"] += stats.damage_dealt
            totals[i]["kd"] += stats.knockdowns_scored
            totals[i]["max_strike"] += stats.max_single_strike_damage
            if isinstance(sim, phase_stamina.StaticFSRMCKOTKOV32PhaseStamina):
                totals[i]["final_stamina"] += sim.stamina_state[i].fraction
                totals[i]["final_fatigue_penalty"] += sim.fatigue_penalty(i)
                totals[i]["final_effective_power"] += float(sim._effective_profile(i)["striking_power"])
            else:
                totals[i]["final_stamina"] += 1.0
                totals[i]["final_effective_power"] += float(red["striking_power"] if i == 0 else blue["striking_power"])
        if result.finish is not None:
            w = int(result.finish.winner)
            totals[w]["ko"] += 1.0
            totals[w]["finishes"] += 1.0
            totals[w]["finish_round_sum"] += float(result.finish.round)
            if int(result.finish.round) == 1:
                totals[w]["r1_ko"] += 1.0
    n = float(len(seeds))
    for row in totals:
        finishes = row["finishes"]
        mean_finish_round = row["finish_round_sum"] / finishes if finishes > 0 else np.nan
        for key in (
            "ko", "r1_ko", "sig_att", "sig_landed", "damage", "kd", "max_strike",
            "final_stamina", "final_fatigue_penalty", "final_effective_power"
        ):
            row[key] /= n
        row["mean_finish_round"] = mean_finish_round
        del row["finish_round_sum"]
        del row["finishes"]
    return totals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    cohort, pairs = cohort32.build_aligned_cohort()
    match = cohort[cohort["bout_id"].astype(str).eq(str(args.bout_id))]
    if len(match) != 1:
        raise ValueError(f"Expected one aligned cohort bout for {args.bout_id}; found {len(match)}")
    bout = match.iloc[0]
    red, blue = pairs[str(args.bout_id)]
    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None
    seeds = np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

    baseline = _run(recovery.StaticFSRMCKOTKOV2RoundRecovery, red, blue, seeds=seeds,
                    rounds=args.rounds, r_age=r_age, b_age=b_age)
    candidate = _run(phase_stamina.StaticFSRMCKOTKOV32PhaseStamina, red, blue, seeds=seeds,
                     rounds=args.rounds, r_age=r_age, b_age=b_age)

    names = [base._display_name(red), base._display_name(blue)]
    print("\n" + "=" * 132)
    print("ALIGNED SINGLE-FIGHT PHASE-STAMINA V3.2 COMPARISON")
    print("=" * 132)
    print(f"bout_id: {args.bout_id}")
    print(f"fight: {names[0]} vs {names[1]}")
    print(f"event_date: {bout['event_date']}")
    print(f"actual KO/TKO: {int(bout['actual_ko_tko'])}; actual R1 KO: {int(bout['actual_r1_ko'])}")
    print(f"paths: {args.paths}; horizon: {args.rounds} rounds")

    print("\nFSR-32 STAMINA CONTRACT")
    fields = ["striking_power", fsr32.STAMINA_CAPACITY,
              fsr32.STAMINA_DEPLETION_RESISTANCE,
              fsr32.STAMINA_PERFORMANCE_RESILIENCE,
              fsr32.STAMINA_RECOVERY_ABILITY]
    print(pd.DataFrame({names[0]: [red[f] for f in fields], names[1]: [blue[f] for f in fields]},
                       index=fields).to_string(float_format=lambda x: f"{float(x):.4f}"))

    rows = []
    for label, pair in (("baseline_recovery", baseline), ("phase_stamina_v3_2", candidate)):
        for i, row in enumerate(pair):
            rows.append({"variant": label, "fighter": names[i], **row})
    frame = pd.DataFrame(rows)
    print("\nPAIRED RESULT")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nV3.2 DELTAS VS BASELINE")
    for i, name in enumerate(names):
        b = baseline[i]
        c = candidate[i]
        print(
            f"{name}: KO {b['ko']:.1%}->{c['ko']:.1%}; "
            f"R1 KO {b['r1_ko']:.1%}->{c['r1_ko']:.1%}; "
            f"sig att {b['sig_att']:.1f}->{c['sig_att']:.1f}; "
            f"landed {b['sig_landed']:.1f}->{c['sig_landed']:.1f}; "
            f"damage {b['damage']:.1f}->{c['damage']:.1f}; "
            f"KD {b['kd']:.3f}->{c['kd']:.3f}; "
            f"max strike {b['max_strike']:.2f}->{c['max_strike']:.2f}; "
            f"mean finish round {b['mean_finish_round']:.2f}->{c['mean_finish_round']:.2f}; "
            f"final stamina={c['final_stamina']:.1%}; "
            f"fatigue penalty={c['final_fatigue_penalty']:.2f}; "
            f"effective power={c['final_effective_power']:.2f}"
        )


if __name__ == "__main__":
    main()
