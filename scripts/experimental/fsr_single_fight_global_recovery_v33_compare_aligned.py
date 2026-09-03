"""Paired single-fight comparison: neutral-recovery baseline vs global V3.3."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_single_fight_phase_stamina_compare_aligned as compare
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=250)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260810)
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

    # The legacy recovery baseline requires recovery_ability. The new FSR-32
    # artifact intentionally removes that field, so supply the former neutral
    # population value only to the frozen comparison baseline.
    baseline_red = red.copy(deep=True)
    baseline_blue = blue.copy(deep=True)
    baseline_red["recovery_ability"] = 50.0
    baseline_blue["recovery_ability"] = 50.0

    baseline = compare._run(
        recovery.StaticFSRMCKOTKOV2RoundRecovery,
        baseline_red, baseline_blue,
        seeds=seeds, rounds=args.rounds, r_age=r_age, b_age=b_age,
    )
    candidate = compare._run(
        v33.StaticFSRMCKOTKOV33GlobalRecovery,
        red, blue,
        seeds=seeds, rounds=args.rounds, r_age=r_age, b_age=b_age,
    )

    names = [base._display_name(red), base._display_name(blue)]
    print("\n" + "=" * 132)
    print("ALIGNED SINGLE-FIGHT GLOBAL-RECOVERY V3.3 COMPARISON")
    print("=" * 132)
    print(f"bout_id: {args.bout_id}")
    print(f"fight: {names[0]} vs {names[1]}")
    print(f"event_date: {bout['event_date']}")
    print(f"actual KO/TKO: {int(bout['actual_ko_tko'])}; actual R1 KO: {int(bout['actual_r1_ko'])}")
    print(f"paths: {args.paths}; horizon: {args.rounds} rounds")
    print("baseline comparison recovery_ability fixed at neutral 50")
    print(
        f"V3.3 global recovery: damage={v33.GLOBAL_DAMAGE_RECOVERY_FRACTION:.0%} of missing; "
        f"stamina={v33.GLOBAL_STAMINA_RECOVERY_FRACTION:.0%} of missing"
    )

    fields = [
        "striking_power",
        fsr32.STAMINA_CAPACITY,
        fsr32.STAMINA_DEPLETION_RESISTANCE,
        fsr32.STAMINA_PERFORMANCE_RESILIENCE,
    ]
    print("\nFSR-32 STAMINA CONTRACT")
    print(pd.DataFrame({names[0]: [red[f] for f in fields], names[1]: [blue[f] for f in fields]},
                       index=fields).to_string(float_format=lambda x: f"{float(x):.4f}"))

    rows = []
    for label, pair in (("baseline_recovery_neutral50", baseline), ("global_recovery_v3_3", candidate)):
        for i, row in enumerate(pair):
            rows.append({"variant": label, "fighter": names[i], **row})
    frame = pd.DataFrame(rows)
    print("\nPAIRED RESULT")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nV3.3 DELTAS VS BASELINE")
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
