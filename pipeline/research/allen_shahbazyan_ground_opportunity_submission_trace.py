"""Research-only Allen-Shahbazyan one-path shadow using the OOS ground-opportunity submission hazard.

Only submission-attempt opportunity mapping changes relative to
allen_shahbazyan_new_timing_trace. Production mechanics and submission conversion
are untouched. The same matched path/seed and current 1.0x TD timing stack are
retained.
"""
from __future__ import annotations

import math
import numpy as np

from pipeline.research import allen_shahbazyan_new_timing_trace as timing
from pipeline.research import allen_shahbazyan_fighter_level_submission_trace as sub_mod
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.simulation.event_clock_mc_v2.brain.policy import ActionProbability
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase

TOTAL_EXPOSURE_SCALE = 0.8876304179003159
GROUND_OPPORTUNITY_SCALE = 2.1691969230906825
GROUND_HAZARD_MULTIPLIER = GROUND_OPPORTUNITY_SCALE / TOTAL_EXPOSURE_SCALE
GROUND_MEAN_DELAY_SECONDS = 4.4
REMOVED = {ActionFamily.IMPROVE_POSITION, ActionFamily.ADVANCE_POSITION}


def _ground_opportunity_submission_probs(state, actor, capabilities, context, priors, config):
    rows = list(sub_mod._original_probs(state, actor, capabilities, context, priors, config))
    if state.phase is not Phase.GROUND or not any(
        r.action_family is ActionFamily.SUBMISSION_ATTACK for r in rows
    ):
        return tuple(rows)

    total_time_rate_15 = float(sub_mod.RATE_PER_15_BY_SIDE[actor])
    ground_hazard_per_second = (
        total_time_rate_15 / 900.0 * GROUND_HAZARD_MULTIPLIER
    )
    p_sub = float(np.clip(
        1.0 - math.exp(-ground_hazard_per_second * GROUND_MEAN_DELAY_SECONDS),
        1e-6,
        0.95,
    ))

    kept_non_sub = [
        r for r in rows
        if r.action_family not in REMOVED
        and r.action_family is not ActionFamily.SUBMISSION_ATTACK
    ]
    denom = sum(r.probability for r in kept_non_sub)
    if denom <= 0:
        raise RuntimeError("no non-submission ground probability mass")

    out = []
    for r in rows:
        if r.action_family in REMOVED:
            prob = 0.0
        elif r.action_family is ActionFamily.SUBMISSION_ATTACK:
            prob = p_sub
        else:
            prob = (1.0 - p_sub) * r.probability / denom
        out.append(ActionProbability(r.action_family, r.utility, float(prob)))
    return tuple(out)


def main():
    sub_mod.RATE_PER_15_BY_SIDE = sub_mod._build_submission_rates()
    print("SUBMISSION_GROUND_OPPORTUNITY_SHADOW", {
        "total_exposure_scale": TOTAL_EXPOSURE_SCALE,
        "ground_opportunity_scale": GROUND_OPPORTUNITY_SCALE,
        "ground_hazard_multiplier": GROUND_HAZARD_MULTIPLIER,
        "production_changed": False,
        "submission_conversion_changed": False,
    })
    base_trace.action_probabilities_with_intent_priors = _ground_opportunity_submission_probs

    # timing.main() would reinstall the old submission chooser, so reproduce its
    # setup while retaining the new submission mapping.
    timing._prefight_td_decomposition()
    timing.CLINCH_RATE_BY_SIDE = timing._build_clinch_rates()
    base_trace._standing_rates_no_reset = timing._new_timing_rates
    timing.target._standing_rates_no_reset = timing._new_timing_rates
    timing.target.main()


if __name__ == "__main__":
    main()
