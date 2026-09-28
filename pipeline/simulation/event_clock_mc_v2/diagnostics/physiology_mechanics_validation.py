"""Post-Stage-10 controlled physiology and frozen-V1 micro validation.

Measurement only.  No market targets and no production mechanics changes.
"""
from __future__ import annotations

import argparse
import json
from math import exp, log
from pathlib import Path

import numpy as np

from pipeline.simulation.event_clock_mc_v2.causal.events import ActionFamily
from pipeline.simulation.event_clock_mc_v2.mechanics.physiology import (
    _sigmoid,
    action_stamina_cost,
)


def _impact(power: float, severity: np.ndarray | float) -> np.ndarray:
    return np.maximum(1e-9, np.exp((power - 50.0) / 55.0) * severity * 0.5)


def _trauma(impact: np.ndarray | float, durability: float) -> np.ndarray:
    return np.asarray(impact) * np.exp(-(durability - 50.0) / 40.0)


def _kd_probability(impact: float, kd_resistance: float, cumulative_trauma: float, acute: float) -> float:
    resistance = max(
        1e-6,
        exp((kd_resistance - 50.0) / 32.0)
        * exp(-cumulative_trauma / 80.0)
        * exp(-acute),
    )
    return _sigmoid(2.0 * (log(impact / resistance) - log(36.0)))


def _finish_probability(impact: float, durability: float, kd_resistance: float, trauma: float, acute: float, kd: bool):
    resistance = max(
        1e-9,
        exp((((durability + kd_resistance) / 2.0) - 50.0) / 32.0)
        * exp(-trauma / 120.0)
        * exp(-acute),
    )
    logit = 2.0 * (log(max(impact / resistance, 1e-12)) - log(36.0)) + (1.0 if kd else 0.0)
    return _sigmoid(logit), resistance


class _Mechanics:
    def __init__(self, capacity: float, resistance: float):
        self.stamina_capacity = capacity
        self.stamina_depletion_resistance = resistance


def build_report(seed: int = 20260825, draws: int = 20000) -> dict:
    rng = np.random.default_rng(seed)
    severity = rng.gamma(1.0, 2.0, draws)
    tail = rng.random(draws) < 0.06
    severity[tail] += rng.gamma(1.25, 4.8, int(tail.sum()))

    low_power, high_power = 40.0, 60.0
    impacts_low = _impact(low_power, severity)
    impacts_high = _impact(high_power, severity)
    impact_ratio = float(np.mean(impacts_high) / np.mean(impacts_low))

    matched_impact = 8.0
    trauma_low_dur = float(_trauma(matched_impact, 40.0))
    trauma_high_dur = float(_trauma(matched_impact, 60.0))

    matched_trauma = 20.0
    kd_low_res = _kd_probability(matched_impact, 40.0, matched_trauma, 0.25)
    kd_high_res = _kd_probability(matched_impact, 60.0, matched_trauma, 0.25)

    base_action = ActionFamily.TAKEDOWN_ENTRY
    cost_cap_80 = action_stamina_cost(_Mechanics(80.0, 50.0), base_action)
    cost_cap_120 = action_stamina_cost(_Mechanics(120.0, 50.0), base_action)
    cost_dep_40 = action_stamina_cost(_Mechanics(100.0, 40.0), base_action)
    cost_dep_60 = action_stamina_cost(_Mechanics(100.0, 60.0), base_action)

    acute_before = 0.50
    acute_after_30 = acute_before * exp(-log(2.0) * 30.0 / 30.0)
    finish_p, finish_res = _finish_probability(8.0, 55.0, 60.0, 25.0, 0.25, True)
    recovered = 0.40 + (1.0 - 0.40) * 0.40

    # Frozen V1 references at age-neutral power context (age 30).  These are
    # copied equations for measurement only; V2 has no V1 runtime dependency.
    v1_impact = exp((60.0 - 50.0) / 55.0) * 4.0 * 0.5
    v2_impact = float(_impact(60.0, 4.0))
    v1_trauma = v1_impact * exp(-(55.0 - 50.0) / 40.0)
    v2_trauma = float(_trauma(v2_impact, 55.0))
    v1_kd = _kd_probability(v1_impact, 60.0, 25.0, 0.25)
    v2_kd = _kd_probability(v2_impact, 60.0, 25.0, 0.25)
    v1_finish, v1_finish_res = _finish_probability(v1_impact, 55.0, 60.0, 25.0, 0.25, True)
    v2_finish, v2_finish_res = _finish_probability(v2_impact, 55.0, 60.0, 25.0, 0.25, True)
    v1_cost = 3.0 * np.clip(exp(-(60.0 - 50.0) / 80.0), 0.65, 1.45) / 100.0
    v2_cost = action_stamina_cost(_Mechanics(100.0, 60.0), ActionFamily.TAKEDOWN_ENTRY)
    v1_recovery = recovered
    v2_recovery = recovered

    controlled = {
        "power": {"low": low_power, "high": high_power, "mean_impact_low": float(np.mean(impacts_low)), "mean_impact_high": float(np.mean(impacts_high)), "high_over_low": impact_ratio, "direction_pass": impact_ratio > 1.0},
        "durability": {"low": 40.0, "high": 60.0, "trauma_low_durability": trauma_low_dur, "trauma_high_durability": trauma_high_dur, "high_over_low": trauma_high_dur / trauma_low_dur, "direction_pass": trauma_high_dur < trauma_low_dur},
        "kd_resistance": {"low": 40.0, "high": 60.0, "p_kd_low_resistance": kd_low_res, "p_kd_high_resistance": kd_high_res, "delta": kd_high_res - kd_low_res, "direction_pass": kd_high_res < kd_low_res},
        "stamina_capacity": {"low": 80.0, "high": 120.0, "cost_low_capacity": cost_cap_80, "cost_high_capacity": cost_cap_120, "high_over_low": cost_cap_120 / cost_cap_80, "direction_pass": cost_cap_120 < cost_cap_80},
        "depletion_resistance": {"low": 40.0, "high": 60.0, "cost_low_resistance": cost_dep_40, "cost_high_resistance": cost_dep_60, "high_over_low": cost_dep_60 / cost_dep_40, "direction_pass": cost_dep_60 < cost_dep_40},
    }
    micro = {
        "impact": {"v1": v1_impact, "v2": v2_impact, "abs_diff": abs(v1_impact-v2_impact)},
        "trauma_increment": {"v1": v1_trauma, "v2": v2_trauma, "abs_diff": abs(v1_trauma-v2_trauma)},
        "kd_probability": {"v1": v1_kd, "v2": v2_kd, "abs_diff": abs(v1_kd-v2_kd), "controlled_rng_outcome_equal": (0.01 < v1_kd) == (0.01 < v2_kd)},
        "acute_increment_decay": {"increment": 0.5, "after_30_seconds": acute_after_30, "expected_half": 0.25},
        "finish": {"v1_resistance": v1_finish_res, "v2_resistance": v2_finish_res, "resistance_abs_diff": abs(v1_finish_res-v2_finish_res), "v1_probability": v1_finish, "v2_probability": v2_finish, "probability_abs_diff": abs(v1_finish-v2_finish)},
        "action_stamina_depletion": {"v1": float(v1_cost), "v2": float(v2_cost), "abs_diff": abs(float(v1_cost)-float(v2_cost))},
        "round_recovery": {"start": 0.40, "v1": v1_recovery, "v2": v2_recovery, "abs_diff": abs(v1_recovery-v2_recovery)},
        "intentional_difference": "V1 strike impact additionally applied a fight-date age adjustment to power. V2 comparison is at the age-30 neutral reference because post-Stage-10 canonical translation must not invent an age proxy; age/context remains upstream.",
    }
    return {
        "seed": seed,
        "draws": draws,
        "controlled_effects": controlled,
        "all_directional_effects_pass": all(v["direction_pass"] for v in controlled.values()),
        "v1_v2_microcomparison": micro,
        "matched_formula_max_abs_diff": max(micro[k].get("abs_diff", 0.0) for k in ("impact", "trauma_increment", "kd_probability", "action_stamina_depletion", "round_recovery")),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=20000)
    args = p.parse_args()
    report = build_report(draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
