"""Measurement-only loss decomposition for Leavitt-Brito intent-rate shadow.

Runs the already-defined intent-rate Brain shadow with the canonical 500 matched
seeds and reports where Brito's remaining losses come from. No production code,
mechanics, judging, FSR, or shadow behavior is changed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json

from pipeline.simulation.event_clock_mc_v2.causal.state import Phase, Side
from pipeline.simulation.event_clock_mc_v2.engine import EngineFunctions, run_causal_path
from pipeline.simulation.event_clock_mc_v2.mechanics.resolution import FinishMethod
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.calibration import SEED_SET_VERSION
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_dynamic_pressure_shadow import (
    FIGHT_ID,
    PATHS,
    STAND,
    GROUND,
    TD,
    STRIKES,
    build_setup,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.leavitt_brito_intent_rate_shadow import (
    IntentRateBrain,
)


def _path_stats(out):
    s = {side: Counter() for side in Side}
    control = Counter()
    for seg in out.timeline_segments:
        if seg.phase is Phase.GROUND and seg.controller in (Side.RED, Side.BLUE):
            control[seg.controller] += seg.duration
    for ev in out.events:
        side = ev.actor
        a = ev.selected_action
        if a in STRIKES:
            s[side]["strike_attempts"] += 1
            if ev.outcome.value == "landed":
                s[side]["strikes_landed"] += 1
        if a in STAND:
            s[side]["standing_attempts"] += 1
            if ev.outcome.value == "landed":
                s[side]["standing_landed"] += 1
        if a in GROUND:
            s[side]["ground_attempts"] += 1
            if ev.outcome.value == "landed":
                s[side]["ground_landed"] += 1
        if a in TD:
            s[side]["td_attempts"] += 1
            if ev.resulting_phase is Phase.GROUND and ev.resulting_controller is side:
                s[side]["td_success"] += 1
        if ev.knockdown:
            s[side]["kd"] += 1
        if ev.submission_attempt:
            s[side]["sub_attempts"] += 1
            if ev.submission_success:
                s[side]["sub_success"] += 1
    for side in Side:
        s[side]["control_seconds"] = control[side]
    return s


def _mean(counter, n):
    return {k: float(v) / n for k, v in sorted(counter.items())} if n else {}


def main():
    fight, inputs, priors, horizon, cfg = build_setup()
    brain = IntentRateBrain(inputs, priors, horizon)
    funcs = EngineFunctions(timing_sampler=brain.timing_sampler, action_chooser=brain.action_chooser)

    method_winner = Counter()
    cohort_n = Counter()
    cohort_totals = {label: {side: Counter() for side in Side} for label in ("brito_win", "brito_loss", "decision")}
    edge_sign = {label: Counter() for label in ("brito_win", "brito_loss", "decision")}
    decision_winner = Counter()

    for path_id in range(PATHS):
        seed = derive_path_seed(SEED_SET_VERSION, FIGHT_ID, path_id)
        out = run_causal_path(inputs, seed=seed, horizon_seconds=horizon, config=cfg, functions=funcs)
        if out.termination is None:
            continue
        winner = out.termination.winner
        method = out.termination.finish_method.value
        method_winner[(winner.value, method)] += 1
        label = "brito_win" if winner is Side.BLUE else "brito_loss"
        cohort_n[label] += 1
        stats = _path_stats(out)
        for side in Side:
            cohort_totals[label][side].update(stats[side])

        # Signs of the principal realized edges on each path, from Brito perspective.
        for metric in ("standing_landed", "strikes_landed", "td_success", "control_seconds", "kd", "sub_attempts"):
            delta = stats[Side.BLUE][metric] - stats[Side.RED][metric]
            edge_sign[label][f"{metric}::positive"] += int(delta > 0)
            edge_sign[label][f"{metric}::tied"] += int(delta == 0)
            edge_sign[label][f"{metric}::negative"] += int(delta < 0)

        if out.termination.finish_method is FinishMethod.DECISION:
            cohort_n["decision"] += 1
            decision_winner[winner.value] += 1
            for side in Side:
                cohort_totals["decision"][side].update(stats[side])
            for metric in ("standing_landed", "strikes_landed", "td_success", "control_seconds", "kd", "sub_attempts"):
                delta = stats[Side.BLUE][metric] - stats[Side.RED][metric]
                edge_sign["decision"][f"{metric}::positive"] += int(delta > 0)
                edge_sign["decision"][f"{metric}::tied"] += int(delta == 0)
                edge_sign["decision"][f"{metric}::negative"] += int(delta < 0)

    payload = {
        "diagnostic": "Leavitt-Brito intent-rate remaining-loss decomposition",
        "fight_id": FIGHT_ID,
        "paths": PATHS,
        "seed_set": SEED_SET_VERSION,
        "production_changed": False,
        "method_winner_counts": {
            f"{side}::{method}": n for (side, method), n in sorted(method_winner.items())
        },
        "decision_winner_counts": dict(decision_winner),
        "cohorts": {},
    }
    for label in ("brito_win", "brito_loss", "decision"):
        n = cohort_n[label]
        payload["cohorts"][label] = {
            "paths": n,
            "brito_mean": _mean(cohort_totals[label][Side.BLUE], n),
            "leavitt_mean": _mean(cohort_totals[label][Side.RED], n),
            "brito_edge_sign_counts": dict(sorted(edge_sign[label].items())),
        }

    print("LEAVITT_BRITO_INTENT_RATE_LOSS_DECOMPOSITION")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
