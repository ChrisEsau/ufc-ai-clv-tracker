"""Shock-driven KD reservoir-collapse sensitivity sweep for KO/TKO V2.

This diagnostic leaves the locked KD probability model untouched. It tests only
how much *additional reservoir collapse* occurs after a confirmed knockdown,
with the collapse tied nonlinearly to the shock fraction that caused the KD.

Architecture under test
-----------------------
landed strike
    -> raw Damage V1 strike damage
    -> reservoir depletion
    -> locked KD check

if KD:
    -> additional reservoir collapse based on KD-causing shock severity
    -> existing recent-KD state activates
    -> later follow-up strikes retain the provisional 2x multiplier

if reservoir reaches zero:
    -> deterministic KO/TKO

No generic strike-level KO hazard is introduced.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_v0 as base


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_kd_collapse_sweep.parquet"
)
DEFAULT_MATCHUPS = 500
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810


@dataclass(frozen=True)
class CollapseCandidate:
    name: str
    collapse_scale: float
    shock_curvature: float


CANDIDATES = (
    CollapseCandidate("none", 0.0, 1.0),
    CollapseCandidate("mild", 2.0, 1.5),
    CollapseCandidate("moderate", 3.5, 1.75),
    CollapseCandidate("strong", 5.0, 2.0),
    CollapseCandidate("very_strong", 7.0, 2.25),
)


class StaticFSRMCKOTKOV2KDCollapse(ko.StaticFSRMCKOTKOV2):
    """KO V2 plus shock-tied reservoir collapse on confirmed KDs."""

    def __init__(self, *args, collapse: CollapseCandidate, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.collapse = collapse
        self.kd_collapse_damage_dealt = [0.0, 0.0]

    def _kd_collapse_fraction(self, shock_fraction: float) -> float:
        """Map KD shock to additional fraction of capacity lost.

        The mapping is zero when collapse_scale is zero, otherwise nonlinear so
        larger KD-causing shocks generate disproportionately larger collapse.
        """
        if self.collapse.collapse_scale <= 0.0:
            return 0.0
        shock = max(0.0, float(shock_fraction))
        frac = self.collapse.collapse_scale * (
            shock + self.collapse.shock_curvature * shock * shock
        )
        return float(np.clip(frac, 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)

            raw_damage = self._draw_strike_damage(attacker)
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)

            # Keep the existing locked KD probability architecture exactly as-is.
            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown = self.rng.random() < p_kd

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

            if knockdown:
                shock_fraction = effective_damage / state.reservoir_capacity
                collapse_fraction = self._kd_collapse_fraction(shock_fraction)
                collapse_damage = collapse_fraction * state.reservoir_capacity

                # KD collapse is additional state loss, not a second copy of the
                # strike. It represents the acute physiological collapse after a
                # confirmed KD and is explicitly tied to that KD's shock severity.
                collapse_damage = min(collapse_damage, state.reservoir_current)
                state.reservoir_current = max(0.0, state.reservoir_current - collapse_damage)
                self.kd_collapse_damage_dealt[attacker] += collapse_damage

                attacker_stats.damage_dealt += collapse_damage
                defender_stats.damage_absorbed += collapse_damage
                total_damage += collapse_damage

                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments, damage.RECENT_KD_SEGMENTS
                )
                knockdowns += 1

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=bool(knockdown),
                    recent_kd_before=bool(recent_kd_before),
                )
                break

        return total_damage, knockdowns


def _latest_profiles() -> pd.DataFrame:
    return damage.load_profiles(damage.FSR_PATH).reset_index(drop=True)


def _choose_matchups(
    profiles: pd.DataFrame,
    matchup_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _ in range(matchup_count):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _run_candidate(
    profiles: pd.DataFrame,
    matchups: list[tuple[int, int]],
    candidate: CollapseCandidate,
    *,
    paths_per_matchup: int,
    rounds: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total_paths = len(matchups) * paths_per_matchup
    path_no = 0

    print(
        f"[KD collapse sweep] candidate={candidate.name} "
        f"scale={candidate.collapse_scale:.2f} curvature={candidate.shock_curvature:.2f}",
        flush=True,
    )

    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]

        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = StaticFSRMCKOTKOV2KDCollapse(
                red,
                blue,
                collapse=candidate,
                rounds=rounds,
                seed=path_seed,
            )
            path = sim.run()
            finish = path.finish

            rows.append(
                {
                    "candidate": candidate.name,
                    "collapse_scale": candidate.collapse_scale,
                    "shock_curvature": candidate.shock_curvature,
                    "matchup_index": matchup_index,
                    "path_index": path_index,
                    "finished": int(finish is not None),
                    "finish_round": finish.round if finish is not None else np.nan,
                    "finish_kd_on_strike": (
                        int(finish.knockdown_on_strike) if finish is not None else 0
                    ),
                    "finish_recent_kd_before": (
                        int(finish.recent_kd_before) if finish is not None else 0
                    ),
                    "red_kd": sim.stats[0].knockdowns_scored,
                    "blue_kd": sim.stats[1].knockdowns_scored,
                    "red_collapse_damage": sim.kd_collapse_damage_dealt[0],
                    "blue_collapse_damage": sim.kd_collapse_damage_dealt[1],
                    "red_power": base._value(red, "striking_power"),
                    "blue_power": base._value(blue, "striking_power"),
                    "red_kd_resistance": base._value(red, "knockdown_resistance"),
                    "blue_kd_resistance": base._value(blue, "knockdown_resistance"),
                    "red_durability": base._value(red, "damage_durability"),
                    "blue_durability": base._value(blue, "damage_durability"),
                    "finish_winner": finish.winner if finish is not None else np.nan,
                    "finish_loser": finish.loser if finish is not None else np.nan,
                }
            )

            path_no += 1
            if path_no % 1000 == 0 or path_no == total_paths:
                finish_count = sum(r["finished"] for r in rows)
                print(
                    f"[KD collapse sweep] {candidate.name}: "
                    f"paths {path_no:,}/{total_paths:,}; "
                    f"finishes={finish_count:,} ({finish_count / path_no:.2%})",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _fighter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        for side, opp, idx in (("red", "blue", 0), ("blue", "red", 1)):
            rows.append(
                {
                    "candidate": r["candidate"],
                    "durability": r[f"{side}_durability"],
                    "power_edge": r[f"{side}_power"] - r[f"{opp}_kd_resistance"],
                    "ko_win": int(r["finished"] == 1 and r["finish_winner"] == idx),
                    "ko_loss": int(r["finished"] == 1 and r["finish_loser"] == idx),
                }
            )
    return pd.DataFrame(rows)


def _print_summary(frame: pd.DataFrame) -> None:
    fighter = _fighter_rows(frame)
    rows: list[dict[str, Any]] = []

    for candidate, g in frame.groupby("candidate", sort=False):
        finished = g[g["finished"] == 1]
        fg = fighter[fighter["candidate"] == candidate]

        low_dur = fg[fg["durability"] <= fg["durability"].quantile(0.20)]
        high_dur = fg[fg["durability"] >= fg["durability"].quantile(0.80)]
        low_edge = fg[fg["power_edge"] <= fg["power_edge"].quantile(0.20)]
        high_edge = fg[fg["power_edge"] >= fg["power_edge"].quantile(0.80)]

        rows.append(
            {
                "candidate": candidate,
                "collapse_scale": g["collapse_scale"].iloc[0],
                "shock_curvature": g["shock_curvature"].iloc[0],
                "ko_finish_rate": g["finished"].mean(),
                "mean_finish_round": finished["finish_round"].mean() if len(finished) else np.nan,
                "round1_share": (finished["finish_round"] == 1).mean() if len(finished) else np.nan,
                "round2_share": (finished["finish_round"] == 2).mean() if len(finished) else np.nan,
                "round3_share": (finished["finish_round"] == 3).mean() if len(finished) else np.nan,
                "finish_on_kd_strike": finished["finish_kd_on_strike"].mean() if len(finished) else np.nan,
                "finish_during_recent_kd": finished["finish_recent_kd_before"].mean() if len(finished) else np.nan,
                "loser_had_any_kd": (
                    (
                        np.where(
                            finished["finish_loser"] == 0,
                            finished["blue_kd"],
                            finished["red_kd"],
                        )
                        >= 1
                    ).mean()
                    if len(finished)
                    else np.nan
                ),
                "mean_total_collapse_damage": (
                    g["red_collapse_damage"] + g["blue_collapse_damage"]
                ).mean(),
                "low_dur_ko_loss": low_dur["ko_loss"].mean(),
                "high_dur_ko_loss": high_dur["ko_loss"].mean(),
                "low_edge_ko_win": low_edge["ko_win"].mean(),
                "high_edge_ko_win": high_edge["ko_win"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print("KO/TKO V2 — SHOCK-DRIVEN KD RESERVOIR COLLAPSE SWEEP")
    print("=" * 140)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation:")
    print("- Prefer candidates that move KO rate/timing toward history through KD-linked finishes.")
    print("- Avoid candidates that flatten durability or power-vs-KD-resistance separation.")
    print("- This sweep changes no locked KD probability constants.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    profiles = _latest_profiles()
    rng = np.random.default_rng(args.seed)
    matchups = _choose_matchups(profiles, args.matchups, rng)

    frames = []
    for i, candidate in enumerate(CANDIDATES):
        frames.append(
            _run_candidate(
                profiles,
                matchups,
                candidate,
                paths_per_matchup=args.paths_per_matchup,
                rounds=args.rounds,
                seed=args.seed + 10000 * i,
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame)
    print(f"\n[KD collapse sweep] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
