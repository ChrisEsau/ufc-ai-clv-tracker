"""Population sanity audit for reservoir-exhaustion KO/TKO V2.

This is descriptive, not calibration. It checks whether the new finish
architecture produces plausible pathway shape before historical tuning:
- how often reservoir exhaustion occurs;
- finish-round distribution;
- whether finishes are concentrated after knockdowns;
- whether durability and power-vs-KD-resistance rank outcomes sensibly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_ground017 as ground017
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_v0 as base


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_reservoir_population_audit.parquet"
)
DEFAULT_MATCHUPS = 500
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_SEED = 20260810


def _run(matchups: int, paths_per_matchup: int, rounds: int, seed: int) -> pd.DataFrame:
    profiles = damage.load_profiles(damage.FSR_PATH).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total = matchups * paths_per_matchup
    done = 0

    for matchup_index in range(matchups):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        red = profiles.iloc[int(a)]
        blue = profiles.iloc[int(b)]

        for path_index in range(paths_per_matchup):
            sim = ko.StaticFSRMCKOTKOV2(
                red,
                blue,
                rounds=rounds,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            path = sim.run()
            finish = path.finish
            loser = finish.loser if finish is not None else None
            winner = finish.winner if finish is not None else None

            rows.append(
                {
                    "matchup_index": matchup_index,
                    "path_index": path_index,
                    "finished": int(finish is not None),
                    "finish_round": finish.round if finish else np.nan,
                    "finish_segment": finish.segment if finish else np.nan,
                    "finish_recent_kd_before": int(finish.recent_kd_before) if finish else 0,
                    "finish_kd_on_strike": int(finish.knockdown_on_strike) if finish else 0,
                    "winner": winner if finish else np.nan,
                    "loser": loser if finish else np.nan,
                    "loser_kd_absorbed": sim.stats[loser].knockdowns_absorbed if finish else np.nan,
                    "winner_power": base._value(sim.fighters[winner], "striking_power") if finish else np.nan,
                    "loser_kd_resistance": base._value(sim.fighters[loser], "knockdown_resistance") if finish else np.nan,
                    "loser_durability": base._value(sim.fighters[loser], "damage_durability") if finish else np.nan,
                    "red_power": base._value(red, "striking_power"),
                    "blue_power": base._value(blue, "striking_power"),
                    "red_kd_resistance": base._value(red, "knockdown_resistance"),
                    "blue_kd_resistance": base._value(blue, "knockdown_resistance"),
                    "red_durability": base._value(red, "damage_durability"),
                    "blue_durability": base._value(blue, "damage_durability"),
                    "red_kd_scored": sim.stats[0].knockdowns_scored,
                    "blue_kd_scored": sim.stats[1].knockdowns_scored,
                    "red_sig_landed": sim.stats[0].sig_landed,
                    "blue_sig_landed": sim.stats[1].sig_landed,
                    "red_reservoir_fraction": sim.damage_state[0].reservoir_fraction,
                    "blue_reservoir_fraction": sim.damage_state[1].reservoir_fraction,
                }
            )
            done += 1
            if done % 1000 == 0 or done == total:
                finishes = sum(int(r["finished"]) for r in rows)
                print(
                    f"[KO V2 reservoir audit] paths {done:,}/{total:,}; "
                    f"finishes={finishes:,} ({finishes/done:.2%})",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _fighter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, r in frame.iterrows():
        for i, side in enumerate(("red", "blue")):
            opp = "blue" if side == "red" else "red"
            rows.append(
                {
                    "durability": float(r[f"{side}_durability"]),
                    "power_edge": float(r[f"{side}_power"] - r[f"{opp}_kd_resistance"]),
                    "ko_win": float(r["finished"] == 1 and r["winner"] == i),
                    "ko_loss": float(r["finished"] == 1 and r["loser"] == i),
                    "kd_scored": float(r[f"{side}_kd_scored"]),
                }
            )
    return pd.DataFrame(rows)


def _print_summary(frame: pd.DataFrame, rounds: int) -> None:
    finished = frame[frame["finished"].eq(1)].copy()
    print("\n" + "=" * 110)
    print("RESERVOIR-EXHAUSTION KO/TKO V2 — POPULATION SANITY AUDIT")
    print("=" * 110)
    print(f"paths: {len(frame):,}; scheduled rounds: {rounds}")
    print(f"ground-exit shadow base/30s: {ground017.GROUND_EXIT_BASE_30S_SHADOW:.3f}")
    print(f"post-KD follow-up multiplier: {ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER:.2f}x")
    print(f"KO/TKO finish probability: {frame['finished'].mean():.3%}")

    if finished.empty:
        print("No finishes occurred. Reservoir/damage scale or post-KD follow-up will need diagnosis before calibration.")
        return

    print(f"mean finish round: {finished['finish_round'].mean():.3f}")
    for r in range(1, rounds + 1):
        print(f"round {r} share of finishes: {(finished['finish_round'] == r).mean():.3%}")
    print(f"finish strike occurs during recent-KD state: {finished['finish_recent_kd_before'].mean():.3%}")
    print(f"finish strike itself causes KD: {finished['finish_kd_on_strike'].mean():.3%}")
    print(f"loser absorbed >=1 KD before/by finish: {(finished['loser_kd_absorbed'] >= 1).mean():.3%}")
    print(f"loser absorbed >=2 KDs before/by finish: {(finished['loser_kd_absorbed'] >= 2).mean():.3%}")
    print(f"mean loser durability: {finished['loser_durability'].mean():.3f}")
    print(
        "mean winner power - loser KD resistance: "
        f"{(finished['winner_power'] - finished['loser_kd_resistance']).mean():+.3f}"
    )

    fighter = _fighter_rows(frame)
    fighter["durability_q"] = pd.qcut(
        fighter["durability"].rank(method="first"), 5,
        labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    )
    durability = fighter.groupby("durability_q", observed=True).agg(
        fighter_paths=("ko_loss", "size"),
        mean_durability=("durability", "mean"),
        ko_loss_rate=("ko_loss", "mean"),
    ).reset_index()
    print("\nDURABILITY QUINTILES")
    print(durability.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fighter["edge_q"] = pd.qcut(
        fighter["power_edge"].rank(method="first"), 5,
        labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    )
    edge = fighter.groupby("edge_q", observed=True).agg(
        fighter_paths=("ko_win", "size"),
        mean_power_edge=("power_edge", "mean"),
        ko_win_rate=("ko_win", "mean"),
        mean_kd_scored=("kd_scored", "mean"),
    ).reset_index()
    print("\nPOWER - OPPONENT KD RESISTANCE QUINTILES")
    print(edge.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION")
    print("- This is architecture sanity only, not final KO-rate calibration.")
    print("- Prefer finishes concentrated after KD/recent-KD pathways rather than generic fresh-state accumulation.")
    print("- Durability should reduce KO loss; power-vs-KD-resistance edge should increase KO win/KD activity.")
    print("- Do not tune exact finish rate until submission, fatigue/dynamic state, and judging pieces are connected.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit reservoir-exhaustion KO/TKO V2")
    ap.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    ap.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    frame = _run(args.matchups, args.paths_per_matchup, args.rounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame, args.rounds)
    print(f"\n[KO V2 reservoir audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
