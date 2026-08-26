"""Run 10,000 full-fight shadow MC paths for Uros Medic vs Daniel Rodriguez.

Purpose
-------
Use the exact leakage-safe pre-fight FSR pair from their 2026-08-01 bout and
measure directional KO/KD danger across a full scheduled 3-round fight.

This is diagnostic-only. It changes no FSR values or simulator constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_ko_tko_v2_2020plus_mature_r1_severity_decomposition as severity
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_damage_v1 as damage

BOUT_ID = "68ae50dbf98dc15f"
DEFAULT_PATHS = 10_000
DEFAULT_SEED = 20260810
HEARTBEAT = 1_000
ROUNDS = 3
SHOCK_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15)
STRONG = next(c for c in collapse.CANDIDATES if c.name == "strong")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "medic_rodriguez_fullfight_10000.parquet"
)


class FullFightTraceSimulator(collapse.StaticFSRMCKOTKOV2KDCollapse):
    """Strong-collapse full-fight simulator with strike-level trace capture."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["rounds"] = ROUNDS
        super().__init__(*args, **kwargs)
        self.strike_trace: list[dict[str, float | int | bool]] = []
        self.current_round = 1

    def run(self):
        # Parent run owns the path mechanics. We infer round from the segment
        # context attached to finish and from cumulative event count afterward.
        return super().run()

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
            reservoir_capacity = float(state.reservoir_capacity)

            raw_damage = self._draw_strike_damage(attacker)
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)
            shock_fraction = effective_damage / reservoir_capacity
            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown = bool(self.rng.random() < p_kd)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage, effective_damage
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage, effective_damage
            )
            total_damage += effective_damage

            collapse_damage = 0.0
            if knockdown:
                collapse_fraction = self._kd_collapse_fraction(shock_fraction)
                collapse_damage = min(
                    collapse_fraction * reservoir_capacity,
                    state.reservoir_current,
                )
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

            self.strike_trace.append(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "raw_damage": float(raw_damage),
                    "effective_damage": float(effective_damage),
                    "shock_fraction": float(shock_fraction),
                    "p_kd": float(p_kd),
                    "knockdown": knockdown,
                    "recent_kd_before": bool(recent_kd_before),
                    "reservoir_fraction_before": reservoir_before / reservoir_capacity,
                    "reservoir_fraction_after": float(state.reservoir_current) / reservoir_capacity,
                    "collapse_damage": float(collapse_damage),
                }
            )

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=knockdown,
                    recent_kd_before=bool(recent_kd_before),
                )
                break

        return total_damage, knockdowns


def _load_pair(fsr_path: Path, master_path: Path):
    modern = severity.modern
    master = modern._load_master(master_path)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(fsr_path, candidate)
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    if BOUT_ID not in pairs:
        raise KeyError(f"Bout {BOUT_ID} not found in leakage-safe FSR pair set")
    bout = cohort.loc[cohort["bout_id"].eq(BOUT_ID)].iloc[0]
    red, blue = pairs[BOUT_ID]
    return bout, red, blue


def _name(profile: pd.Series) -> str:
    from scripts.experimental import fsr_static_mc_v0 as base
    return base._display_name(profile)


def _run(paths: int, seed: int, red: pd.Series, blue: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for i in range(paths):
        path_seed = int(rng.integers(0, 2**31 - 1))
        sim = FullFightTraceSimulator(red, blue, collapse=STRONG, seed=path_seed)
        path = sim.run()
        trace = pd.DataFrame(sim.strike_trace)

        finish = path.finish
        winner = int(finish.winner) if finish is not None else -1
        finish_round = int(finish.round) if finish is not None and finish.round is not None else 0

        row: dict[str, object] = {
            "path_index": i,
            "seed": path_seed,
            "ko_tko": int(finish is not None),
            "winner": winner,
            "finish_round": finish_round,
            "medic_ko": int(finish is not None and winner == 0),
            "rodriguez_ko": int(finish is not None and winner == 1),
            "medic_r1_ko": int(finish is not None and winner == 0 and finish_round == 1),
            "rodriguez_r1_ko": int(finish is not None and winner == 1 and finish_round == 1),
            "medic_kd": int(sim.stats[0].knockdowns_scored),
            "rodriguez_kd": int(sim.stats[1].knockdowns_scored),
            "medic_any_kd": int(sim.stats[0].knockdowns_scored > 0),
            "rodriguez_any_kd": int(sim.stats[1].knockdowns_scored > 0),
            "medic_sig_att": int(sim.stats[0].sig_att),
            "rodriguez_sig_att": int(sim.stats[1].sig_att),
            "medic_sig_landed": int(sim.stats[0].sig_landed),
            "rodriguez_sig_landed": int(sim.stats[1].sig_landed),
            "medic_damage": float(sim.stats[0].damage_dealt),
            "rodriguez_damage": float(sim.stats[1].damage_dealt),
        }

        for fighter, prefix in ((0, "medic"), (1, "rodriguez")):
            ft = trace.loc[trace["attacker"].eq(fighter)] if len(trace) else pd.DataFrame()
            row[f"{prefix}_max_shock"] = float(ft["shock_fraction"].max()) if len(ft) else 0.0
            row[f"{prefix}_mean_shock"] = float(ft["shock_fraction"].mean()) if len(ft) else 0.0
            for threshold in SHOCK_THRESHOLDS:
                label = int(round(threshold * 100))
                row[f"{prefix}_any_shock_ge_{label}"] = int(
                    len(ft) > 0 and bool((ft["shock_fraction"] >= threshold).any())
                )

        rows.append(row)

        if (i + 1) % HEARTBEAT == 0 or (i + 1) == paths:
            partial = pd.DataFrame(rows)
            print(
                f"[Medic-Rodriguez 10k] {i + 1:,}/{paths:,} | "
                f"Medic KO={partial['medic_ko'].mean():.2%} | "
                f"Rodriguez KO={partial['rodriguez_ko'].mean():.2%} | "
                f"distance={1.0 - partial['ko_tko'].mean():.2%}",
                flush=True,
            )

    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame, red_name: str, blue_name: str) -> None:
    print("\n" + "=" * 110)
    print("MEDIC VS RODRIGUEZ — 10,000 FULL-FIGHT STATIC MC PATHS")
    print("=" * 110)
    print(f"paths: {len(df):,}; scheduled rounds: {ROUNDS}")
    print(f"P({red_name} KO/TKO)      = {df['medic_ko'].mean():.2%}")
    print(f"P({blue_name} KO/TKO) = {df['rodriguez_ko'].mean():.2%}")
    print(f"P(any KO/TKO)             = {df['ko_tko'].mean():.2%}")
    print(f"P(no KO/TKO by R3 end)    = {1.0 - df['ko_tko'].mean():.2%}")
    print(f"P({red_name} R1 KO/TKO)   = {df['medic_r1_ko'].mean():.2%}")
    print(f"P({blue_name} R1 KO/TKO)  = {df['rodriguez_r1_ko'].mean():.2%}")

    print("\nKO/TKO FINISH ROUND")
    for rnd in (1, 2, 3):
        print(f"R{rnd}: {(df['finish_round'] == rnd).mean():.2%} of all paths")

    print("\nDIRECTIONAL KD / STRIKING")
    for prefix, name in (("medic", red_name), ("rodriguez", blue_name)):
        print(
            f"{name}: P(any KD)={df[f'{prefix}_any_kd'].mean():.2%}; "
            f"mean KD={df[f'{prefix}_kd'].mean():.3f}; "
            f"sig att={df[f'{prefix}_sig_att'].mean():.2f}; "
            f"sig landed={df[f'{prefix}_sig_landed'].mean():.2f}; "
            f"damage={df[f'{prefix}_damage'].mean():.2f}; "
            f"max shock={df[f'{prefix}_max_shock'].mean():.4f}"
        )

    print("\nSEVERE-SHOCK PATH INCIDENCE")
    for threshold in SHOCK_THRESHOLDS:
        label = int(round(threshold * 100))
        print(
            f">={label}% shock: "
            f"{red_name}={df[f'medic_any_shock_ge_{label}'].mean():.2%}; "
            f"{blue_name}={df[f'rodriguez_any_shock_ge_{label}'].mean():.2%}"
        )

    medic = df['medic_ko'].mean()
    rod = df['rodriguez_ko'].mean()
    if rod > 0:
        print(f"\nDirectional KO ratio ({red_name}/{blue_name}) = {medic / rod:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=severity.modern.FSR_PATH)
    parser.add_argument("--master", type=Path, default=severity.modern.MASTER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    bout, red, blue = _load_pair(args.fsr_path, args.master)
    red_name = _name(red)
    blue_name = _name(blue)
    print(
        f"[Medic-Rodriguez 10k] bout={BOUT_ID}; date={pd.Timestamp(bout['event_date']).date()}; "
        f"matchup={red_name} vs {blue_name}; paths={args.paths:,}; rounds={ROUNDS}",
        flush=True,
    )

    df = _run(args.paths, args.seed, red, blue)
    _print_summary(df, red_name, blue_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\n[Medic-Rodriguez 10k] wrote {args.output}")
    print("No simulator constants or FSR values were changed.")


if __name__ == "__main__":
    main()
