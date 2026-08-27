"""Research-only Allen-Shahbazyan trace with fighter-level submission intent.

Submission attempt propensity is no longer derived from separate top/bottom
utilities. A single point-in-time matchup submission-attempt rate is computed
for each fighter from the OOS-validated submission_tendency x opponent
submission_suppression model. When SUBMISSION_ATTACK is legal on the ground,
that fighter-level rate is converted to one position-invariant per-ground-action
probability using the structural 4.4-second ground action clock. Remaining legal
ground actions retain their relative Brain probabilities.

This is an architecture diagnostic, not a claim that the absolute ground-time
hazard is fully calibrated. Production code is unchanged.
"""
from __future__ import annotations

import math
import numpy as np

from pipeline.research import allen_shahbazyan_no_pressure_trace as no_pressure
from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as base_trace
from pipeline.research.submission_attempt_opportunity_oos_validation import build_frame, fit_scale
from pipeline.simulation.event_clock_mc_v2.brain.policy import ActionProbability
from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.diagnostics import leavitt_brito_dynamic_pressure_shadow as pressure_mod

GROUND_MEAN_DELAY_SECONDS = 4.0 * 1.10
REMOVED = {ActionFamily.IMPROVE_POSITION, ActionFamily.ADVANCE_POSITION}
_original_probs = base_trace.action_probabilities_with_intent_priors
RATE_PER_15_BY_SIDE: dict[Side, float] = {}


def _fighter_level_submission_probs(state, actor, capabilities, context, priors, config):
    rows = list(_original_probs(state, actor, capabilities, context, priors, config))
    if state.phase is not Phase.GROUND or not any(r.action_family is ActionFamily.SUBMISSION_ATTACK for r in rows):
        return tuple(rows)

    rate_15 = float(RATE_PER_15_BY_SIDE[actor])
    # Poisson probability of >=1 submission-attempt event during one structural
    # ground action-clock interval. Same fighter-level value whether top/bottom.
    p_sub = float(np.clip(1.0 - math.exp(-(rate_15 / 900.0) * GROUND_MEAN_DELAY_SECONDS), 1e-6, 0.95))

    kept_non_sub = [r for r in rows if r.action_family not in REMOVED and r.action_family is not ActionFamily.SUBMISSION_ATTACK]
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


def _build_submission_rates():
    pressure_mod.FIGHT_ID = base_trace.FIGHT_ID
    fight, _, _, _, _ = pressure_mod.build_setup()
    target_date = getattr(fight, "date", None) or getattr(fight, "event_date", None)
    frame = build_frame()
    train = frame[frame.event_date < target_date].copy()
    scale = fit_scale(train, "rate_matchup")
    fight_rows = frame[frame.fight_id.astype(str) == str(base_trace.FIGHT_ID)].copy()
    if len(fight_rows) != 2:
        raise RuntimeError(f"expected 2 fighter rows for target fight, got {len(fight_rows)}")
    red_id, blue_id = str(fight.r_id), str(fight.b_id)
    by_id = {str(r.fighter_id): r for r in fight_rows.itertuples(index=False)}
    rates = {
        Side.RED: float(scale * float(by_id[red_id].rate_matchup) * 900.0),
        Side.BLUE: float(scale * float(by_id[blue_id].rate_matchup) * 900.0),
    }
    names = {Side.RED: str(fight.r_name), Side.BLUE: str(fight.b_name)}
    print("FIGHTER_LEVEL_SUBMISSION_RATES_PER_15", {names[s]: rates[s] for s in Side})
    print("FIGHTER_LEVEL_SUBMISSION_P_PER_GROUND_ACTION", {
        names[s]: 1.0 - math.exp(-(rates[s] / 900.0) * GROUND_MEAN_DELAY_SECONDS) for s in Side
    })
    return rates


def main():
    global RATE_PER_15_BY_SIDE
    RATE_PER_15_BY_SIDE = _build_submission_rates()
    base_trace.action_probabilities_with_intent_priors = _fighter_level_submission_probs
    no_pressure.target._standing_rates_no_reset = no_pressure._standing_rates_no_pressure_no_reset
    no_pressure.main()


if __name__ == "__main__":
    main()
