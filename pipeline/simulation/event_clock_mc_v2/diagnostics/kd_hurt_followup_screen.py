"""Research-only men's KD/hurt follow-up screen.

Keeps the validated continuous consequence-side power curve fixed:

    power_offset(t) = clip(35 - t / 12, -40, 35)

and tests whether a short, temporary post-KD KO-vulnerability window can move
finishes earlier without materially inflating whole-fight KO share.

The frozen shadow model already gives every subsequent strike a permanent
prior-KD KO logit bonus.  This diagnostic adds only an EXTRA temporary bonus
for a survived KD.  Strike budgets, event scheduling, FSR, wrestling,
submissions, judging, and frozen V1 source remain unchanged.

Arms:
  control       : no extra hurt-window bonus
  hurt_20s_b075 : +0.75 KO logit for 20 seconds after survived KD
  hurt_20s_b125 : +1.25 KO logit for 20 seconds after survived KD

This is a falsification screen, not a production proposal.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import (
    EventClockShadowKOKDModel as FrozenShadow,
    ShadowKOKDCalibration,
    ShadowStrikeConsequence,
)
from pipeline.simulation.event_clock_mc_v1 import run_event_or_fight as frozen_runner
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import shared_power_decay_grid as shared
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles, Side

INTERCEPT = 35.0
DENOMINATOR = 12.0
LOWER_CAP = -40.0
UPPER_CAP = 35.0

_MODE = "control"
ARMS = {
    "control": (0.0, 0.0),
    "hurt_20s_b075": (20.0, 0.75),
    "hurt_20s_b125": (20.0, 1.25),
}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + exp(-x))


class HurtFollowupPowerShadow(FrozenShadow):
    """Selected continuous power curve plus optional temporary post-KD hurt state."""

    def __init__(self, profiles: MatchupProfiles, calibration: ShadowKOKDCalibration = ShadowKOKDCalibration()):
        super().__init__(profiles=profiles, calibration=calibration)
        object.__setattr__(self, "_hurt_until", {"red": -1.0, "blue": -1.0})

    @staticmethod
    def _power_offset(state) -> float:
        t = float(state.fight_time_seconds)
        return float(np.clip(INTERCEPT - t / DENOMINATOR, LOWER_CAP, UPPER_CAP))

    def _shifted_model(self, state) -> FrozenShadow:
        offset = self._power_offset(state)

        def shifted(side: Side):
            profile = self.profiles.fighter(side)
            return replace(profile, striking_power=float(profile.striking_power) + offset)

        return FrozenShadow(
            profiles=MatchupProfiles(red=shifted(Side.RED), blue=shifted(Side.BLUE)),
            calibration=self.calibration,
        )

    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        global _MODE
        window_seconds, hurt_bonus = ARMS[_MODE]
        defender = attacker.opponent
        model = self._shifted_model(state)

        p_ko = model.ko_probability(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
        )
        now = float(state.fight_time_seconds)
        if hurt_bonus > 0.0 and now <= float(self._hurt_until[defender.value]):
            p_ko = _sigmoid(_logit(p_ko) + hurt_bonus)

        ko_tko = bool(rng.random() < p_ko)
        if ko_tko:
            return ShadowStrikeConsequence(
                attacker=attacker,
                defender=defender,
                ko_probability=p_ko,
                ko_tko=True,
                kd_probability=0.0,
                knockdown=False,
                prior_defender_kds=int(prior_defender_kds),
            )

        p_kd = model.kd_probability(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
        )
        knockdown = bool(rng.random() < p_kd)
        if knockdown and window_seconds > 0.0:
            self._hurt_until[defender.value] = now + window_seconds

        return ShadowStrikeConsequence(
            attacker=attacker,
            defender=defender,
            ko_probability=p_ko,
            ko_tko=False,
            kd_probability=p_kd,
            knockdown=knockdown,
            prior_defender_kds=int(prior_defender_kds),
        )


def _thin(cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(cohort) <= n:
        return cohort.reset_index(drop=True)
    idx = np.linspace(0, len(cohort) - 1, num=n, dtype=int)
    return cohort.iloc[np.unique(idx)].reset_index(drop=True)


def _install_summary_wrapper() -> None:
    original = canonical.summarize_fight

    def wrapped(fight_id, pair, rows, master_row):
        out = original(fight_id, pair, rows, master_row)
        paths = pd.DataFrame(rows)
        nondec = paths["method"].ne("DEC")
        for threshold in (300.0, 600.0, 900.0):
            out[f"p_nondec_by_{int(threshold)}"] = float(
                (nondec & paths["elapsed"].le(threshold)).mean()
            )
        return out

    canonical.summarize_fight = wrapped


def _historical_targets(target_n: int) -> tuple[float, dict[int, float], float]:
    cohorts = []
    for division in shared.MEN_DIVISIONS:
        cohort, _ = wc_audit.select_cohort(division, target_n)
        cohorts.append(cohort)
    all_cohort = pd.concat(cohorts, ignore_index=True)
    method = all_cohort["method"].map(wc_audit.normalize_method)
    elapsed = pd.to_numeric(all_cohort["match_time_sec"], errors="raise")
    ko_share = float(method.eq("KO_TKO").mean())
    finish = {
        threshold: float(((method != "DEC") & elapsed.le(float(threshold))).mean())
        for threshold in (300, 600, 900)
    }
    return ko_share, finish, float(elapsed.mean())


def _run_arm(target_n: int, sim_n_per_division: int, paths: int, seed: int, arm: str) -> pd.DataFrame:
    global _MODE
    _MODE = arm
    frozen_runner.EventClockShadowKOKDModel = HurtFollowupPowerShadow
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort, _ = wc_audit.select_cohort(division, target_n)
        cohort = _thin(cohort, sim_n_per_division)
        print(f"ARM {arm} | {division} | fights={len(cohort)} paths={paths}")
        summary = canonical._simulate_c(cohort, paths, seed + i * 100_000_000)
        summary["division"] = division
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _summarize(summary: pd.DataFrame, hist_ko: float, hist_finish: dict[int, float], hist_elapsed: float, paths: int, arm: str) -> dict:
    y = summary["actual_winner"].eq("red").astype(float).to_numpy()
    p = summary["p_red_win"].to_numpy(float)
    winner_p = np.where(y > 0.5, p, 1.0 - p)
    rec = {
        "arm": arm,
        "n_fights": int(len(summary)),
        "paths_per_fight": int(paths),
        "ml_accuracy": float(summary["ml_correct"].mean()),
        "ml_brier": float(np.mean((p - y) ** 2)),
        "ml_logloss": float(-np.mean(np.log(np.clip(winner_p, 1e-9, 1.0)))),
        "method_accuracy": float(summary["method_correct"].mean()),
        "historical_ko_share": hist_ko,
        "simulated_ko_share": float(summary["p_fight_ko_tko"].mean()),
        "ko_share_bias": float(summary["p_fight_ko_tko"].mean() - hist_ko),
        "historical_mean_elapsed": hist_elapsed,
        "simulated_mean_elapsed": float(summary["sim_mean_elapsed"].mean()),
        "duration_relative_bias": float(summary["sim_mean_elapsed"].mean() / hist_elapsed - 1.0),
    }
    for threshold in (300, 600, 900):
        sim = float(summary[f"p_nondec_by_{threshold}"].mean())
        rec[f"hist_nondec_by_{threshold}"] = hist_finish[threshold]
        rec[f"sim_nondec_by_{threshold}"] = sim
        rec[f"bias_nondec_by_{threshold}"] = sim - hist_finish[threshold]
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--sim-n-per-division", type=int, default=20)
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/kd_hurt_followup_screen"),
    )
    args = parser.parse_args()

    _install_summary_wrapper()
    hist_ko, hist_finish, hist_elapsed = _historical_targets(args.target_n)

    metrics = []
    summaries = []
    for arm in ARMS:
        summary = _run_arm(args.target_n, args.sim_n_per_division, args.paths, args.seed, arm)
        summary["arm"] = arm
        summaries.append(summary)
        metrics.append(_summarize(summary, hist_ko, hist_finish, hist_elapsed, args.paths, arm))

    metrics_df = pd.DataFrame(metrics)
    summary_df = pd.concat(summaries, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.out_dir / "arm_metrics.csv", index=False)
    summary_df.to_csv(args.out_dir / "fight_summaries.csv", index=False)

    print("\nHISTORICAL MEN TARGETS")
    print(f"KO share={hist_ko:.5f} | mean elapsed={hist_elapsed:.2f}s")
    for threshold in (300, 600, 900):
        print(f"nondecision by {threshold}s={hist_finish[threshold]:.5f}")
    print("\nARM METRICS")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
