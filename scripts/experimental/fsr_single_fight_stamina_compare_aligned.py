"""Paired single-fight comparison: current recovery engine vs FSR-32 stamina V3.

Uses identical aligned historical profiles, locked ages, strong KD collapse, and
path seeds for both variants.  Intended as the first validation step before the
full population audit.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v3_stamina as stamina
from scripts.experimental import fsr_static_mc_v0 as base


DEFAULT_PATHS = 250
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _run_variant(
    sim_cls,
    red: pd.Series,
    blue: pd.Series,
    *,
    seeds: np.ndarray,
    rounds: int,
    r_age: float | None,
    b_age: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    totals = [
        {
            "ko": 0.0,
            "r1_ko": 0.0,
            "sig_att": 0.0,
            "sig_landed": 0.0,
            "damage": 0.0,
            "kd": 0.0,
            "max_strike": 0.0,
            "final_stamina": 0.0,
            "final_output_mult": 0.0,
            "final_power_mult": 0.0,
        },
        {
            "ko": 0.0,
            "r1_ko": 0.0,
            "sig_att": 0.0,
            "sig_landed": 0.0,
            "damage": 0.0,
            "kd": 0.0,
            "max_strike": 0.0,
            "final_stamina": 0.0,
            "final_output_mult": 0.0,
            "final_power_mult": 0.0,
        },
    ]

    for seed in seeds:
        sim = sim_cls(
            red,
            blue,
            collapse=STRONG_COLLAPSE,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()
        for i, stats in enumerate(sim.stats):
            assert isinstance(stats, damage.DamageFighterStats)
            totals[i]["sig_att"] += stats.sig_att
            totals[i]["sig_landed"] += stats.sig_landed
            totals[i]["damage"] += stats.damage_dealt
            totals[i]["kd"] += stats.knockdowns_scored
            totals[i]["max_strike"] += stats.max_single_strike_damage
            if isinstance(sim, stamina.StaticFSRMCKOTKOV3Stamina):
                totals[i]["final_stamina"] += sim.stamina_state[i].fraction
                totals[i]["final_output_mult"] += sim.stamina_output_multiplier(i)
                totals[i]["final_power_mult"] += sim.stamina_power_multiplier(i)
            else:
                totals[i]["final_stamina"] += 1.0
                totals[i]["final_output_mult"] += 1.0
                totals[i]["final_power_mult"] += 1.0

        if result.finish is not None:
            winner = int(result.finish.winner)
            totals[winner]["ko"] += 1.0
            if int(result.finish.round) == 1:
                totals[winner]["r1_ko"] += 1.0

    n = float(len(seeds))
    for row in totals:
        for key in row:
            row[key] /= n
    return totals[0], totals[1]


def _profile_table(red: pd.Series, blue: pd.Series) -> pd.DataFrame:
    fields = [
        "striking_power",
        fsr32.STAMINA_CAPACITY,
        fsr32.STAMINA_DEPLETION_RESISTANCE,
        fsr32.STAMINA_PERFORMANCE_RESILIENCE,
        fsr32.STAMINA_RECOVERY_ABILITY,
        "knockdown_resistance",
        "damage_durability",
    ]
    return pd.DataFrame(
        {
            base._display_name(red): [red[f] for f in fields],
            base._display_name(blue): [blue[f] for f in fields],
        },
        index=fields,
    )


def main() -> None:
    args = _parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    match = cohort[cohort["bout_id"].astype(str).eq(str(args.bout_id))]
    if len(match) != 1:
        raise ValueError(f"Expected one aligned cohort bout for {args.bout_id}; found {len(match)}")
    bout = match.iloc[0]
    red, blue = pairs[str(args.bout_id)]

    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

    baseline_rows = _run_variant(
        recovery.StaticFSRMCKOTKOV2RoundRecovery,
        red,
        blue,
        seeds=seeds,
        rounds=args.rounds,
        r_age=r_age,
        b_age=b_age,
    )
    stamina_rows = _run_variant(
        stamina.StaticFSRMCKOTKOV3Stamina,
        red,
        blue,
        seeds=seeds,
        rounds=args.rounds,
        r_age=r_age,
        b_age=b_age,
    )

    names = [base._display_name(red), base._display_name(blue)]
    print("\n" + "=" * 124)
    print("ALIGNED SINGLE-FIGHT STAMINA COMPARISON")
    print("=" * 124)
    print(f"bout_id: {args.bout_id}")
    print(f"fight: {names[0]} vs {names[1]}")
    print(f"event_date: {bout['event_date']}")
    print(f"actual KO/TKO: {int(bout['actual_ko_tko'])}; actual R1 KO: {int(bout['actual_r1_ko'])}")
    print(f"paths: {args.paths}; horizon: {args.rounds} rounds")

    print("\nFSR-32 FIGHTER CONTRACT")
    print(_profile_table(red, blue).to_string(float_format=lambda x: f"{float(x):.4f}"))

    rows = []
    for variant, pair in (("baseline_recovery", baseline_rows), ("stamina_v3", stamina_rows)):
        for i, row in enumerate(pair):
            rows.append({"variant": variant, "fighter": names[i], **row})
    result = pd.DataFrame(rows)
    print("\nPAIRED 250-PATH RESULT")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSTAMINA DELTAS VS BASELINE")
    for i, name in enumerate(names):
        before = baseline_rows[i]
        after = stamina_rows[i]
        print(
            f"{name}: KO {before['ko']:.1%}->{after['ko']:.1%}; "
            f"R1 KO {before['r1_ko']:.1%}->{after['r1_ko']:.1%}; "
            f"sig att {before['sig_att']:.1f}->{after['sig_att']:.1f}; "
            f"damage {before['damage']:.1f}->{after['damage']:.1f}; "
            f"KD {before['kd']:.3f}->{after['kd']:.3f}; "
            f"final stamina={after['final_stamina']:.1%}; "
            f"final output={after['final_output_mult']:.1%}; "
            f"final power={after['final_power_mult']:.1%}"
        )


if __name__ == "__main__":
    main()
