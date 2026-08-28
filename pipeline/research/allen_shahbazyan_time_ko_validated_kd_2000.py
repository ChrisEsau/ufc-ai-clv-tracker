"""Research-only Allen-Shahbazyan rerun with OOS-selected static KD hazard.

Purpose:
- Keep the current Brain timing/grappling/submission stack frozen.
- Keep the piecewise time-based KO/TKO competing clock frozen.
- Replace only the legacy dynamic in-fight KD equation with the already OOS-selected
  prefight KD hazard from ko_v3_from_scratch_shadow:
    EWM95 attacker KD creation + defender KD susceptibility, strength 200,
    exposure, age, division.
- No same-fight prior-KD escalation term is applied because Stage 2 does not
  identify acute within-fight KD escalation from aggregate round data.

This is a research shadow. Production mechanics are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.research import allen_shahbazyan_time_ko_clock_2000 as base
from pipeline.research import allen_shahbazyan_decision_scored_outputs_2000 as scored
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.research.ko_v3_from_scratch_shadow import fit_prefight_hazards
from pipeline.simulation.event_clock_mc_v2.causal.state import Side
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod
from pipeline.simulation.event_clock_mc_v2.mechanics import ko_kd_empirical

OUTDIR = Path("data/research/allen_shahbazyan_time_ko_validated_kd_2000")


class StaticValidatedKDResolver:
    """One static prefight KD probability per landed significant strike by side."""

    p_by_side: dict[Side, float] = {}

    def __call__(self, *, state, attacker_side, attacker, defender, rng):
        del attacker, defender
        target = state.physiology.fighter(attacker_side.opponent)
        prior = int(target.knockdowns_suffered)
        p_kd = float(self.p_by_side[attacker_side])
        kd = bool(rng.random() < p_kd)
        return ko_kd_empirical.EmpiricalKOKDResult(
            ko_probability=0.0,
            ko_tko=False,
            kd_probability=p_kd,
            knockdown=kd,
            prior_defender_kds=prior,
        )


def main():
    # Resolve exact side -> fighter id mapping once.
    pressure_mod.FIGHT_ID = base_trace.FIGHT_ID
    fight, _, _, _, _ = pressure_mod.build_setup()
    side_to_id = {Side.RED: str(fight.r_id), Side.BLUE: str(fight.b_id)}
    side_to_name = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}

    hazards = fit_prefight_hazards(fight_id=base_trace.FIGHT_ID)
    StaticValidatedKDResolver.p_by_side = {
        side: float(hazards[fid].kd_per_landed) for side, fid in side_to_id.items()
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "study": "Allen-Shahbazyan time KO + OOS-selected static KD hazard",
        "production_changed": False,
        "fight_id": base_trace.FIGHT_ID,
        "kd_architecture": "static prefight KD hazard per landed significant strike",
        "within_fight_prior_kd_escalation": False,
        "source": "ko_v3_from_scratch_shadow.fit_prefight_hazards",
        "fighters": {
            side_to_name[side]: {
                "fighter_id": side_to_id[side],
                "kd_per_landed": StaticValidatedKDResolver.p_by_side[side],
                "population_kd_per_landed": float(hazards[side_to_id[side]].kd_population_hazard),
                "raw_audit": hazards[side_to_id[side]].raw_audit,
            }
            for side in Side
        },
    }
    (OUTDIR / "kd_hazard_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("VALIDATED STATIC KD HAZARDS")
    print(json.dumps(audit, indent=2))

    original_cls = base.NoKOKDResolver
    original_base_out = base.OUTDIR
    original_scored_out = scored.OUTDIR
    try:
        # Both harnesses instantiate base.NoKOKDResolver(), so patching this one
        # research seam preserves every other Brain mechanic and every seed.
        base.NoKOKDResolver = StaticValidatedKDResolver

        base.OUTDIR = OUTDIR / "sim"
        base.main()

        scored.OUTDIR = OUTDIR / "scored"
        scored.main()
    finally:
        base.NoKOKDResolver = original_cls
        base.OUTDIR = original_base_out
        scored.OUTDIR = original_scored_out


if __name__ == "__main__":
    main()
