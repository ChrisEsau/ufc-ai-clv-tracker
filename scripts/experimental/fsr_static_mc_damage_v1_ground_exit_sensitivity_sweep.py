"""Shadow sensitivity sweep for ground-exit persistence.

Research-only. This script does not modify the baseline simulator module.
It subclasses Damage V1 and overrides only the ground-exit base probability
for candidate values, then reruns the same historical 300-bout time-matched
cohort.

Primary goal: test whether longer ground persistence improves the currently
observed control/ground opportunity deficits without creating unacceptable
side effects in phase mix, strike volume, takedowns, or KD behavior.

Candidate values are 30-second ground-exit probabilities:
    0.20 (baseline), 0.17, 0.14, 0.11

The script reports, by candidate:
- true MC ground residence seconds per ground entry;
- MC ground-control seconds per landed TD;
- control share;
- distance/clinch/ground phase-time share;
- distance/clinch/ground significant-strike attempts/min;
- total significant-strike attempts/min and landed/min;
- TD attempts/min and landed/min;
- any-KD probability and expected total KD;
- matchup-level Spearman correlations against historical control, TD activity,
  and phase-specific strike shares where available.

No candidate is promoted or written back to simulator constants.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_predictive_value as prior
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_time_matched as tm
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_phase_mix_diagnostic as phase_diag


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_ground_exit_sensitivity_sweep.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810
DEFAULT_CANDIDATES = (0.20, 0.17, 0.14, 0.11)
PHASES = ("DISTANCE", "CLINCH", "GROUND")


class GroundExitSensitivitySim(phase_diag.InstrumentedDamageV1):
    """Damage V1 with only the ground-exit base probability overridden."""

    def __init__(self, *args, ground_exit_base_30s: float, **kwargs):
        self.ground_exit_base_30s = float(ground_exit_base_30s)
        self.ground_entries = 0
        super().__init__(*args, **kwargs)

    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        before = self.stats[attacker].td_landed
        note = super()._attempt_takedown(attacker, source_phase)
        if self.stats[attacker].td_landed > before:
            self.ground_entries += 1
        return note

    def _ground_exit_hazard(self, controller: int) -> float:
        bottom = self._other(controller)
        escape_edge = (
            damage.base._value(self.fighters[bottom], "control_resistance")
            - damage.base._value(self.fighters[controller], "control_imposition")
        ) / damage.base.RATING_SCALE
        reversal_edge = (
            damage.base._value(self.fighters[bottom], "reversal_ability")
            - damage.base._value(self.fighters[controller], "control_imposition")
        ) / damage.base.RATING_SCALE
        modifier = exp(float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5)))
        base_10s = damage.base._rescale_interval_prob(
            self.ground_exit_base_30s,
            damage.base.CALIBRATION_INTERVAL_SECONDS,
            damage.base.SEGMENT_SECONDS,
        )
        return damage.base._prob(base_10s * modifier, high=0.90)


def _reset_round(sim: GroundExitSensitivitySim) -> None:
    sim.phase = "DISTANCE"
    sim.ground_controller = None
    sim.clinch_controller = None
    sim.clinch_initiator = None


def _run_full_segment(sim: GroundExitSensitivitySim) -> None:
    phase = sim.phase
    sim.phase_seconds[phase] += damage.base.SEGMENT_SECONDS
    for stats in sim.stats:
        stats.phase_segments[phase] += 1
    sim._generate_striking(phase)
    if phase == "DISTANCE":
        sim._distance_transition()
    elif phase == "CLINCH":
        sim._clinch_transition()
    else:
        sim._ground_transition()


def _run_partial_segment(sim: GroundExitSensitivitySim, seconds: float) -> None:
    if seconds <= 0:
        return
    fraction = seconds / damage.base.SEGMENT_SECONDS
    sim._advance_damage_timers()
    phase = sim.phase
    sim.phase_seconds[phase] += seconds
    if phase == "GROUND" and sim.ground_controller is not None:
        top = sim.ground_controller
        bottom = sim._other(top)
        sim._generate_strikes_for_fighter(top, "GROUND", rate_multiplier=fraction)
        sim._generate_strikes_for_fighter(
            bottom,
            "GROUND",
            rate_multiplier=damage.base.BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER * fraction,
        )
    else:
        for fighter in (0, 1):
            sim._generate_strikes_for_fighter(fighter, phase, rate_multiplier=fraction)


def _simulate_one(red, blue, *, elapsed_sec: float, rounds: int, seed: int, ground_exit_base_30s: float):
    maximum = rounds * 300.0
    elapsed_sec = float(np.clip(elapsed_sec, 0.0, maximum))
    sim = GroundExitSensitivitySim(
        red,
        blue,
        rounds=rounds,
        seed=seed,
        ground_exit_base_30s=ground_exit_base_30s,
    )
    full = int(elapsed_sec // damage.base.SEGMENT_SECONDS)
    rem = elapsed_sec - full * damage.base.SEGMENT_SECONDS
    current_round = 0
    for idx in range(full):
        round_no = idx // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
            current_round = round_no
        _run_full_segment(sim)
    if rem > 1e-9:
        round_no = full // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
        _run_partial_segment(sim, rem)
    return sim


def _actual_data(round_stats_path: Path, master_path: Path, bout_ids: set[str]) -> pd.DataFrame:
    actual_exposure = prior._actual_exposure(round_stats_path, master_path, bout_ids)
    phase = phase_diag._actual_phase_data(round_stats_path, bout_ids)
    out = actual_exposure.merge(phase, on="bout_id", how="left", validate="one_to_one")
    elapsed_min = out["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    out["actual_ctrl_share"] = out["ctrl_sec"] / out["actual_elapsed_sec"].clip(lower=1.0)
    out["actual_td_att_per_min"] = out["td_attempted"] / elapsed_min
    td_landed = pd.read_parquet(round_stats_path).copy()
    td_landed["fight_id"] = td_landed["fight_id"].astype(str)
    td_landed = td_landed[td_landed["fight_id"].isin(bout_ids)]
    td_agg = td_landed.groupby("fight_id", as_index=False)["td_landed"].sum().rename(columns={"fight_id":"bout_id"})
    out = out.merge(td_agg, on="bout_id", how="left", validate="one_to_one")
    out["actual_td_landed_per_min"] = out["td_landed"].fillna(0.0) / elapsed_min
    return out


def _run_candidate(validation: pd.DataFrame, pairs, *, candidate: float, paths_per_bout: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + int(round(candidate * 10000)))
    rows = []
    total_paths = len(validation) * paths_per_bout
    done = 0
    for bout_no, (_, bout) in enumerate(validation.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        elapsed = float(bout["actual_elapsed_sec"])
        elapsed_min = max(elapsed, 1.0) / 60.0
        rounds = prior._rounds_for_bout(bout)
        path_rows = []
        for _ in range(paths_per_bout):
            sim = _simulate_one(
                red, blue,
                elapsed_sec=elapsed,
                rounds=rounds,
                seed=int(rng.integers(0, 2**31 - 1)),
                ground_exit_base_30s=candidate,
            )
            total_att = sum(sim.phase_sig_att.values())
            total_land = sum(sim.phase_sig_landed.values())
            td_att = sim.stats[0].td_att + sim.stats[1].td_att
            td_land = sim.stats[0].td_landed + sim.stats[1].td_landed
            ground_ctrl = sim.stats[0].ground_control_seconds + sim.stats[1].ground_control_seconds
            total_ctrl = sim.stats[0].control_seconds + sim.stats[1].control_seconds
            total_kd = sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored
            row = {
                "sig_att": total_att,
                "sig_land": total_land,
                "td_att": td_att,
                "td_land": td_land,
                "ground_ctrl": ground_ctrl,
                "total_ctrl": total_ctrl,
                "ground_entries": sim.ground_entries,
                "total_kd": total_kd,
            }
            for p in PHASES:
                row[f"{p.lower()}_sec"] = sim.phase_seconds[p]
                row[f"{p.lower()}_att"] = sim.phase_sig_att[p]
            path_rows.append(row)
            done += 1
            if done % 1000 == 0 or done == total_paths:
                print(
                    f"[ground-exit sweep {candidate:.2f}] paths {done:,}/{total_paths:,}; "
                    f"bouts_started={bout_no:,}/{len(validation):,}",
                    flush=True,
                )
        pf = pd.DataFrame(path_rows)
        mean = pf.mean(numeric_only=True)
        sec_total = mean[["distance_sec","clinch_sec","ground_sec"]].sum()
        out = {
            "bout_id": bout_id,
            "ground_exit_base_30s": candidate,
            "sim_paths": paths_per_bout,
            "sim_sig_attempted_per_min": mean["sig_att"] / elapsed_min,
            "sim_sig_landed_per_min": mean["sig_land"] / elapsed_min,
            "sim_td_att_per_min": mean["td_att"] / elapsed_min,
            "sim_td_landed_per_min": mean["td_land"] / elapsed_min,
            "sim_control_share": mean["total_ctrl"] / max(elapsed,1.0),
            "sim_ground_control_sec_per_td_landed": mean["ground_ctrl"] / mean["td_land"] if mean["td_land"] > 0 else np.nan,
            "sim_ground_residence_sec_per_entry": mean["ground_sec"] / mean["ground_entries"] if mean["ground_entries"] > 0 else np.nan,
            "sim_p_any_kd": float((pf["total_kd"] > 0).mean()),
            "sim_expected_total_kd": float(pf["total_kd"].mean()),
        }
        for p in PHASES:
            key = p.lower()
            out[f"sim_{key}_time_share"] = mean[f"{key}_sec"] / sec_total if sec_total else np.nan
            out[f"sim_{key}_attempts_per_min"] = mean[f"{key}_att"] / elapsed_min
        rows.append(out)
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand, g in frame.groupby("ground_exit_base_30s", sort=False):
        rows.append({
            "ground_exit_base_30s": cand,
            "control_share": g["sim_control_share"].mean(),
            "ground_time_share": g["sim_ground_time_share"].mean(),
            "distance_time_share": g["sim_distance_time_share"].mean(),
            "clinch_time_share": g["sim_clinch_time_share"].mean(),
            "ground_res_sec_per_entry": g["sim_ground_residence_sec_per_entry"].mean(),
            "ground_ctrl_sec_per_td": g["sim_ground_control_sec_per_td_landed"].mean(),
            "td_att_per_min": g["sim_td_att_per_min"].mean(),
            "td_land_per_min": g["sim_td_landed_per_min"].mean(),
            "sig_att_per_min": g["sim_sig_attempted_per_min"].mean(),
            "sig_land_per_min": g["sim_sig_landed_per_min"].mean(),
            "distance_att_per_min": g["sim_distance_attempts_per_min"].mean(),
            "clinch_att_per_min": g["sim_clinch_attempts_per_min"].mean(),
            "ground_att_per_min": g["sim_ground_attempts_per_min"].mean(),
            "p_any_kd": g["sim_p_any_kd"].mean(),
            "expected_total_kd": g["sim_expected_total_kd"].mean(),
            "control_spearman": g["actual_ctrl_share"].corr(g["sim_control_share"], method="spearman"),
            "td_att_spearman": g["actual_td_att_per_min"].corr(g["sim_td_att_per_min"], method="spearman"),
            "td_land_spearman": g["actual_td_landed_per_min"].corr(g["sim_td_landed_per_min"], method="spearman"),
            "ground_share_spearman": g["actual_ground_att_share"].corr(g["sim_ground_time_share"], method="spearman"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep shadow ground-exit persistence candidates")
    ap.add_argument("--validation", type=Path, default=prior.VALIDATION_PATH)
    ap.add_argument("--fsr-path", type=Path, default=prior.FSR_PATH)
    ap.add_argument("--round-stats", type=Path, default=Path(prior.ROUND_STATS_PATH))
    ap.add_argument("--master", type=Path, default=Path(prior.MASTER_PATH))
    ap.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--candidates", nargs="*", type=float, default=list(DEFAULT_CANDIDATES))
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    validation = prior._load_validation(args.validation)
    bout_ids = set(validation["bout_id"].astype(str))
    pairs, style = prior._load_fsr_pairs(args.fsr_path, bout_ids)
    actual = _actual_data(args.round_stats, args.master, bout_ids)
    validation = validation.merge(actual, on="bout_id", how="left", validate="one_to_one")

    all_rows = []
    print(f"[ground-exit sweep] bouts={len(validation):,}; candidates={args.candidates}", flush=True)
    for candidate in args.candidates:
        if not 0 < candidate < 1:
            raise ValueError(f"candidate must be in (0,1), got {candidate}")
        sim = _run_candidate(
            validation,
            pairs,
            candidate=float(candidate),
            paths_per_bout=args.paths_per_bout,
            seed=args.seed,
        )
        merged = validation.merge(sim, on="bout_id", how="left", validate="one_to_one")
        all_rows.append(merged)

    out = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    summary = _summary(out)
    print("\n" + "=" * 140)
    print("GROUND EXIT SENSITIVITY SWEEP — SHADOW ONLY")
    print("=" * 140)
    print("Historical targets/reference from current 300-bout cohort:")
    print(f"control share={out.drop_duplicates('bout_id')['actual_ctrl_share'].mean():.4f}")
    print(f"TD att/min={out.drop_duplicates('bout_id')['actual_td_att_per_min'].mean():.4f}")
    print(f"TD landed/min={out.drop_duplicates('bout_id')['actual_td_landed_per_min'].mean():.4f}")
    print("historical control sec / landed TD proxy from prior ground-chain audit ~= 133.62s")
    print("\nCANDIDATE COMPARISON")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nINTERPRETATION")
    print("- Prefer candidates that improve persistence/control without materially degrading matchup ranking.")
    print("- Watch total and phase-specific strike volume plus KD behavior for side effects.")
    print("- This is a sensitivity test only; no candidate changes the baseline simulator constant.")
    print(f"\n[ground-exit sweep] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
