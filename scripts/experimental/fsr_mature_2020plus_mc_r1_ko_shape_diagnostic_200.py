"""Diagnostics-only R1 KO/KD shape audit for the current research MC.

This script does NOT change simulator constants or FSR artifacts.
It uses the current calibration exactly as configured by the -8.80 comparison run:
- contact sigma = 0.80
- power magnitude scale = 75
- base damage at power 50 = 1.18
- KD base logit = -8.80
- KD shock coefficient = 100
- KD depletion coefficient = 0
- collapse scale = 2.0
- collapse curvature = 20.0

Purpose
-------
1. Compare simulated and historical R1 KD/KO rates by combined significant-strike
   exposure bucket on the exact same first 200 mature 2020+ bouts.
2. Log simulated landed-strike severity so we can inspect the contact-quality and
   shock distributions associated with ordinary strikes, surviving KDs, and KOs.
3. Split R1 simulated KO mechanisms into:
   - direct fresh strike
   - terminal KD-collapse
   - direct follow-up strike after a surviving KD

Research only. No production simulator, FSR value, or calibration constant is modified.
"""
from __future__ import annotations

import argparse
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase88_curve20_scale2_200 as run88  # noqa: F401
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as current
from scripts.experimental import historical_sigstr_kd_ko_exposure_2020plus_mature as hist
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

OUTPUT_DIR = Path("data/experimental")
DETAIL_PATH = OUTPUT_DIR / "mc_r1_ko_shape_200_strikes.csv"
PATH_PATH = OUTPUT_DIR / "mc_r1_ko_shape_200_paths.csv"
EXPOSURE_PATH = OUTPUT_DIR / "mc_r1_ko_shape_200_by_exposure.csv"
HIST_EXPOSURE_PATH = OUTPUT_DIR / "historical_same200_r1_by_exposure.csv"
SEVERITY_PATH = OUTPUT_DIR / "mc_r1_ko_shape_200_severity_summary.csv"

EXPOSURE_BINS = hist.EXPOSURE_BINS
EXPOSURE_LABELS = hist.EXPOSURE_LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostics-only R1 KO/KD shape audit")
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--round-stats", type=Path, default=hist.ROUND_STATS_PATH)
    return p.parse_args()


def _power_multiplier(power: float) -> float:
    return float(exp((float(power) - 50.0) / current.POWER_MAGNITUDE_SCALE))


class R1DiagnosticSim(current.AuditSim):
    """Current simulator with observational strike logging only."""

    def __init__(self, *args, **kwargs) -> None:
        self.strike_log: list[dict[str, Any]] = []
        self.ko_mechanism: str | None = None
        super().__init__(*args, **kwargs)

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        surviving_kds = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = bool(state.recent_knockdown)
            reservoir_before = float(state.reservoir_current)
            reservoir_capacity = float(state.reservoir_capacity)
            reservoir_fraction_before = float(state.reservoir_fraction)

            effective_power = base._value(self.fighters[attacker], "striking_power")
            sigma = current.CONTACT_SIGMA
            contact_quality = float(
                self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma)
            )
            raw_damage = max(
                0.0,
                current.BASE_DAMAGE_AT_POWER_50
                * contact_quality
                * _power_multiplier(effective_power),
            )
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            shock_fraction = effective_damage / reservoir_capacity
            kd_probability = self._knockdown_probability(defender, effective_damage)

            log_row: dict[str, Any] = {
                "attacker": attacker,
                "defender": defender,
                "effective_power": float(effective_power),
                "contact_quality": contact_quality,
                "raw_damage": float(raw_damage),
                "effective_damage": float(effective_damage),
                "shock_fraction": float(shock_fraction),
                "kd_probability": float(kd_probability),
                "reservoir_capacity": reservoir_capacity,
                "reservoir_before": reservoir_before,
                "reservoir_fraction_before": reservoir_fraction_before,
                "recent_kd_before": recent_kd_before,
                "event_type": "ordinary",
                "collapse_fraction": 0.0,
                "collapse_damage": 0.0,
            }

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, effective_damage
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, effective_damage
            )
            total_damage += effective_damage

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.direct_strike_finishes += 1
                self.ko_mechanism = (
                    "followup_after_surviving_kd" if recent_kd_before else "direct_fresh_strike"
                )
                log_row["event_type"] = self.ko_mechanism
                log_row["reservoir_after"] = float(state.reservoir_current)
                self.strike_log.append(log_row)
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=False,
                    recent_kd_before=recent_kd_before,
                )
                break

            knockdown = self.rng.random() < kd_probability
            if not knockdown:
                log_row["reservoir_after"] = float(state.reservoir_current)
                self.strike_log.append(log_row)
                continue

            collapse_fraction = self._kd_collapse_fraction(shock_fraction)
            collapse_damage = min(
                collapse_fraction * state.reservoir_capacity,
                state.reservoir_current,
            )
            state.reservoir_current = max(0.0, state.reservoir_current - collapse_damage)
            attacker_stats.damage_dealt += collapse_damage
            defender_stats.damage_absorbed += collapse_damage
            total_damage += collapse_damage

            log_row["collapse_fraction"] = float(collapse_fraction)
            log_row["collapse_damage"] = float(collapse_damage)

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.terminal_collapse_finishes += 1
                self.ko_mechanism = "terminal_kd_collapse"
                log_row["event_type"] = "terminal_kd_collapse"
                log_row["reservoir_after"] = float(state.reservoir_current)
                self.strike_log.append(log_row)
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=True,
                    recent_kd_before=recent_kd_before,
                )
                break

            attacker_stats.knockdowns_scored += 1
            defender_stats.knockdowns_absorbed += 1
            state.recent_knockdown_segments = max(
                state.recent_knockdown_segments,
                damage.RECENT_KD_SEGMENTS,
            )
            surviving_kds += 1
            log_row["event_type"] = "surviving_kd"
            log_row["reservoir_after"] = float(state.reservoir_current)
            self.strike_log.append(log_row)

        return total_damage, surviving_kds


def _summarize_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in EXPOSURE_LABELS:
        g = frame[frame["sig_str_bin"].astype(str).eq(label)]
        n = len(g)
        sig = int(g["sig_str_landed"].sum()) if n else 0
        kd = int(g["kd"].sum()) if n else 0
        ko_count = int(g["ko_tko"].sum()) if n else 0
        rows.append(
            {
                "sig_str_bin": label,
                "fight_rounds": n,
                "sig_str_landed": sig,
                "mean_sig_str_landed": float(g["sig_str_landed"].mean()) if n else np.nan,
                "knockdowns": kd,
                "mean_kd_per_round": float(g["kd"].mean()) if n else np.nan,
                "p_any_kd": float(g["any_kd"].mean()) if n else np.nan,
                "ko_tko_finishes": ko_count,
                "p_ko_tko": float(g["ko_tko"].mean()) if n else np.nan,
                "kd_per_100_sig_landed": 100.0 * kd / sig if sig else np.nan,
                "ko_per_1000_sig_landed": 1000.0 * ko_count / sig if sig else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _severity_summary(strikes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = [
        "ordinary",
        "surviving_kd",
        "terminal_kd_collapse",
        "direct_fresh_strike",
        "followup_after_surviving_kd",
    ]
    for event_type in order:
        g = strikes[strikes["event_type"].eq(event_type)]
        if g.empty:
            continue
        row: dict[str, Any] = {"event_type": event_type, "strikes": len(g)}
        for col in ("contact_quality", "raw_damage", "effective_damage", "shock_fraction", "kd_probability", "reservoir_fraction_before"):
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(s.mean())
            row[f"{col}_p50"] = float(s.quantile(0.50))
            row[f"{col}_p90"] = float(s.quantile(0.90))
            row[f"{col}_p95"] = float(s.quantile(0.95))
            row[f"{col}_p99"] = float(s.quantile(0.99))
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_same_bouts_r1(cohort: pd.DataFrame, round_stats_path: Path) -> pd.DataFrame:
    rs = hist._load_round_stats(round_stats_path)
    rounds = hist._build_fight_rounds(rs, cohort)
    r1 = rounds[rounds["round"].eq(1)].copy()
    return r1


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    # Importing run88 above intentionally applies ONLY its established -8.80
    # comparison setting to the shared current research module.
    if current.KD_BASE_LOGIT != -8.80:
        raise RuntimeError(f"Expected current KD base -8.80, got {current.KD_BASE_LOGIT}")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    historical_r1 = _historical_same_bouts_r1(cohort, args.round_stats)

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(
        0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64
    )

    strike_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    completed = 0

    print("\n" + "=" * 150)
    print("R1 KO/KD SHAPE DIAGNOSTIC — CURRENT CONSTANTS — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"contact sigma={current.CONTACT_SIGMA:.2f}; power scale={current.POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={current.KD_BASE_LOGIT:.2f}; shock={current.KD_SHOCK_COEFFICIENT:.0f}; depletion={current.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={current.COLLAPSE_SCALE:.1f}; curvature={current.COLLAPSE_CURVATURE:.1f}")
    print("Diagnostics only: no constants changed.")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for path_idx, seed in enumerate(seed_matrix[bout_idx]):
            sim = R1DiagnosticSim(
                red,
                blue,
                rounds=1,
                seed=int(seed),
                red_age=r_age,
                blue_age=b_age,
            )
            path = sim.run()
            sig_landed = int(sim.stats[0].sig_landed) + int(sim.stats[1].sig_landed)
            kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
            ko_tko = int(path.finish is not None)

            path_row = {
                "bout_id": bout_id,
                "path_idx": path_idx,
                "seed": int(seed),
                "sig_str_landed": sig_landed,
                "kd": kd,
                "any_kd": int(kd > 0),
                "ko_tko": ko_tko,
                "ko_mechanism": sim.ko_mechanism or "none",
                "terminal_collapse_ko": int(sim.ko_mechanism == "terminal_kd_collapse"),
                "direct_fresh_ko": int(sim.ko_mechanism == "direct_fresh_strike"),
                "followup_ko": int(sim.ko_mechanism == "followup_after_surviving_kd"),
            }
            path_rows.append(path_row)

            for strike_idx, row in enumerate(sim.strike_log):
                out = dict(row)
                out.update(
                    {
                        "bout_id": bout_id,
                        "path_idx": path_idx,
                        "seed": int(seed),
                        "strike_idx": strike_idx,
                        "path_sig_str_landed": sig_landed,
                        "path_kd": kd,
                        "path_ko_tko": ko_tko,
                        "path_ko_mechanism": sim.ko_mechanism or "none",
                    }
                )
                strike_rows.append(out)

        completed += args.paths
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    paths = pd.DataFrame(path_rows)
    strikes = pd.DataFrame(strike_rows)

    paths["sig_str_bin"] = pd.cut(
        paths["sig_str_landed"],
        bins=EXPOSURE_BINS,
        labels=EXPOSURE_LABELS,
        ordered=True,
    )
    historical_r1["sig_str_bin"] = pd.cut(
        historical_r1["sig_str_landed"],
        bins=EXPOSURE_BINS,
        labels=EXPOSURE_LABELS,
        ordered=True,
    )

    sim_exposure = _summarize_exposure(paths)
    hist_exposure = _summarize_exposure(historical_r1)
    severity = _severity_summary(strikes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strikes.to_csv(DETAIL_PATH, index=False)
    paths.to_csv(PATH_PATH, index=False)
    sim_exposure.to_csv(EXPOSURE_PATH, index=False)
    hist_exposure.to_csv(HIST_EXPOSURE_PATH, index=False)
    severity.to_csv(SEVERITY_PATH, index=False)

    sim_sig = int(paths["sig_str_landed"].sum())
    sim_kd = int(paths["kd"].sum())
    sim_ko = int(paths["ko_tko"].sum())
    hist_sig = int(historical_r1["sig_str_landed"].sum())
    hist_kd = int(historical_r1["kd"].sum())
    hist_ko = int(historical_r1["ko_tko"].sum())

    print("\nR1 OVERALL — SIMULATED vs EXACT SAME HISTORICAL BOUTS")
    print("----------------------------------------------------")
    overall = pd.DataFrame(
        [
            {
                "source": "simulated",
                "fight_rounds": len(paths),
                "mean_sig_str_landed": paths["sig_str_landed"].mean(),
                "kd_per_round": paths["kd"].mean(),
                "p_any_kd": paths["any_kd"].mean(),
                "p_ko_tko": paths["ko_tko"].mean(),
                "kd_per_100_sig": 100.0 * sim_kd / sim_sig,
                "ko_per_1000_sig": 1000.0 * sim_ko / sim_sig,
            },
            {
                "source": "historical",
                "fight_rounds": len(historical_r1),
                "mean_sig_str_landed": historical_r1["sig_str_landed"].mean(),
                "kd_per_round": historical_r1["kd"].mean(),
                "p_any_kd": historical_r1["any_kd"].mean(),
                "p_ko_tko": historical_r1["ko_tko"].mean(),
                "kd_per_100_sig": 100.0 * hist_kd / hist_sig,
                "ko_per_1000_sig": 1000.0 * hist_ko / hist_sig,
            },
        ]
    )
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSIMULATED R1 KO MECHANISM MIX")
    print("-----------------------------")
    ko_paths = paths[paths["ko_tko"].eq(1)]
    if ko_paths.empty:
        print("<no R1 KOs>")
    else:
        mech = (
            ko_paths.groupby("ko_mechanism", as_index=False)
            .size()
            .rename(columns={"size": "ko_paths"})
        )
        mech["share_of_r1_ko"] = mech["ko_paths"] / len(ko_paths)
        print(mech.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSIMULATED R1 BY SIGNIFICANT-STRIKE EXPOSURE")
    print("-------------------------------------------")
    print(sim_exposure.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHISTORICAL SAME-200 R1 BY SIGNIFICANT-STRIKE EXPOSURE")
    print("------------------------------------------------------")
    print(hist_exposure.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSIMULATED STRIKE SEVERITY BY EVENT TYPE")
    print("---------------------------------------")
    print(severity.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOUTPUTS")
    print("-------")
    print(DETAIL_PATH)
    print(PATH_PATH)
    print(EXPOSURE_PATH)
    print(HIST_EXPOSURE_PATH)
    print(SEVERITY_PATH)
    print("\nResearch only: diagnostics; no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
