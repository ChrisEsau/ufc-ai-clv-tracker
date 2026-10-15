"""Population audit for shadow Static FSR MC KO/TKO V1.

Purpose
-------
Exercise the provisional KO/TKO finish layer across many FSR-28 matchups before
any finish-rate calibration. This script is descriptive only: it measures the
shape of the finish mechanics and writes a path-level parquet for deeper review.

Primary questions
-----------------
- What fraction of 3-round paths end by KO/TKO?
- How often does a finish occur on a KD-producing strike?
- How often is the defender already in the recent-KD state?
- At what reservoir fraction do stoppages occur?
- Do catastrophic finishes occur while meaningful reservoir remains?
- Do lower-durability fighters lose by KO/TKO more often?
- Does the attacker power - defender KD-resistance edge separate KO/TKO wins?
- Are repeated-KD paths concentrated among stoppages without becoming absurd?

Boundary
--------
The numeric KO/TKO constants are provisional. Do not interpret the rates from
this script as calibrated UFC targets and do not modify constants from this
script alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v1 as ko
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v1_population_audit.parquet"
)

DEFAULT_MATCHUPS = 500
DEFAULT_PATHS_PER_MATCHUP = 20
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260809


def _rank_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    rank = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        rank,
        bins=np.linspace(0.0, 1.0, len(labels) + 1),
        labels=labels,
        include_lowest=True,
    )


def _latest_profiles(path: Path) -> pd.DataFrame:
    profiles = damage.load_profiles(path).copy()
    needed = {
        "fighter_id",
        "striking_power",
        "knockdown_resistance",
        "damage_durability",
    }
    missing = sorted(needed - set(profiles.columns))
    if missing:
        raise ValueError(f"latest FSR-28 profiles missing audit columns: {missing}")
    return profiles.reset_index(drop=True)


def _choose_matchups(
    profiles: pd.DataFrame,
    matchup_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if len(profiles) < 2:
        raise ValueError("Need at least two latest fighter profiles")
    pairs: list[tuple[int, int]] = []
    for _ in range(matchup_count):
        a, b = rng.choice(len(profiles), size=2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def _finish_pathway(finish: ko.FinishResult | None) -> str:
    if finish is None:
        return "no_finish"
    if finish.knockdown_on_strike and finish.recent_kd_before:
        return "KD_on_strike + recent_KD"
    if finish.knockdown_on_strike:
        return "KD_on_finish_strike"
    if finish.recent_kd_before:
        return "recent_KD_followup"
    if finish.reservoir_fraction_after <= 0.25:
        return "low_reservoir_accumulation"
    return "higher_reservoir_catastrophic"


def _run_population(
    profiles: pd.DataFrame,
    matchup_count: int,
    paths_per_matchup: int,
    rounds: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    matchups = _choose_matchups(profiles, matchup_count, rng)
    rows: list[dict[str, Any]] = []

    total_paths = matchup_count * paths_per_matchup
    path_counter = 0

    for matchup_index, (red_i, blue_i) in enumerate(matchups, start=1):
        red = profiles.iloc[red_i]
        blue = profiles.iloc[blue_i]

        for path_index in range(paths_per_matchup):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = ko.StaticFSRMCKOTKOV1(red, blue, rounds=rounds, seed=path_seed)
            path = sim.run()
            finish = path.finish

            finish_winner = finish.winner if finish is not None else None
            finish_loser = finish.loser if finish is not None else None

            row: dict[str, Any] = {
                "matchup_index": matchup_index,
                "path_index": path_index,
                "path_seed": path_seed,
                "red_id": str(sim.fighters[0]["fighter_id"]),
                "blue_id": str(sim.fighters[1]["fighter_id"]),
                "finished": int(finish is not None),
                "finish_pathway": _finish_pathway(finish),
                "finish_winner": finish_winner,
                "finish_loser": finish_loser,
                "finish_round": finish.round if finish is not None else np.nan,
                "finish_segment": finish.segment if finish is not None else np.nan,
                "finish_probability": finish.probability if finish is not None else np.nan,
                "finish_strike_damage": finish.strike_damage if finish is not None else np.nan,
                "finish_shock_fraction": finish.shock_fraction if finish is not None else np.nan,
                "finish_reservoir_fraction": (
                    finish.reservoir_fraction_after if finish is not None else np.nan
                ),
                "finish_kd_on_strike": (
                    int(finish.knockdown_on_strike) if finish is not None else 0
                ),
                "finish_recent_kd_before": (
                    int(finish.recent_kd_before) if finish is not None else 0
                ),
                "red_kd_scored": sim.stats[0].knockdowns_scored,
                "blue_kd_scored": sim.stats[1].knockdowns_scored,
                "red_kd_absorbed": sim.stats[0].knockdowns_absorbed,
                "blue_kd_absorbed": sim.stats[1].knockdowns_absorbed,
                "red_sig_landed": sim.stats[0].sig_landed,
                "blue_sig_landed": sim.stats[1].sig_landed,
                "red_damage_dealt": sim.stats[0].damage_dealt,
                "blue_damage_dealt": sim.stats[1].damage_dealt,
                "red_reservoir_fraction": sim.damage_state[0].reservoir_fraction,
                "blue_reservoir_fraction": sim.damage_state[1].reservoir_fraction,
                "red_power": base._value(sim.fighters[0], "striking_power"),
                "blue_power": base._value(sim.fighters[1], "striking_power"),
                "red_kd_resistance": base._value(sim.fighters[0], "knockdown_resistance"),
                "blue_kd_resistance": base._value(sim.fighters[1], "knockdown_resistance"),
                "red_durability": base._value(sim.fighters[0], "damage_durability"),
                "blue_durability": base._value(sim.fighters[1], "damage_durability"),
            }

            if finish is not None:
                attacker = finish.winner
                defender = finish.loser
                row.update(
                    {
                        "winner_power": base._value(sim.fighters[attacker], "striking_power"),
                        "loser_kd_resistance": base._value(
                            sim.fighters[defender], "knockdown_resistance"
                        ),
                        "loser_durability": base._value(
                            sim.fighters[defender], "damage_durability"
                        ),
                        "winner_power_minus_loser_kd_resistance": (
                            base._value(sim.fighters[attacker], "striking_power")
                            - base._value(sim.fighters[defender], "knockdown_resistance")
                        ),
                        "loser_kds_absorbed": sim.stats[defender].knockdowns_absorbed,
                        "winner_kds_scored": sim.stats[attacker].knockdowns_scored,
                    }
                )
            else:
                row.update(
                    {
                        "winner_power": np.nan,
                        "loser_kd_resistance": np.nan,
                        "loser_durability": np.nan,
                        "winner_power_minus_loser_kd_resistance": np.nan,
                        "loser_kds_absorbed": np.nan,
                        "winner_kds_scored": np.nan,
                    }
                )

            rows.append(row)
            path_counter += 1
            if path_counter % 1000 == 0 or path_counter == total_paths:
                finishes = sum(int(r["finished"]) for r in rows)
                print(
                    f"[KO/TKO V1 audit] paths {path_counter:,}/{total_paths:,}; "
                    f"finishes={finishes:,} ({finishes / path_counter:.2%})",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _print_overall(frame: pd.DataFrame, rounds: int) -> None:
    finished = frame[frame["finished"] == 1].copy()

    print("\n" + "=" * 124)
    print("STATIC FSR MC KO/TKO V1 — POPULATION AUDIT")
    print("=" * 124)
    print(f"paths: {len(frame):,}")
    print(f"scheduled rounds: {rounds}")
    print(f"KO/TKO finish probability: {frame['finished'].mean():.3%}")
    print(f"KO/TKO finishes: {len(finished):,}")

    if finished.empty:
        print("No finishes drawn; pathway summaries unavailable.")
        return

    print(f"mean finish round: {finished['finish_round'].mean():.3f}")
    print(f"round 1 share: {(finished['finish_round'] == 1).mean():.3%}")
    print(f"round 2 share: {(finished['finish_round'] == 2).mean():.3%}")
    print(f"round 3+ share: {(finished['finish_round'] >= 3).mean():.3%}")
    print(f"mean loser reservoir at finish: {finished['finish_reservoir_fraction'].mean():.3%}")
    print(f"median loser reservoir at finish: {finished['finish_reservoir_fraction'].median():.3%}")
    print(f"p10 loser reservoir at finish: {finished['finish_reservoir_fraction'].quantile(0.10):.3%}")
    print(f"p90 loser reservoir at finish: {finished['finish_reservoir_fraction'].quantile(0.90):.3%}")
    print(f"finish at zero reservoir: {(finished['finish_reservoir_fraction'] <= 0).mean():.3%}")
    print(f"finish above 25% reservoir: {(finished['finish_reservoir_fraction'] > 0.25).mean():.3%}")
    print(f"finish above 50% reservoir: {(finished['finish_reservoir_fraction'] > 0.50).mean():.3%}")
    print(f"KD on finish strike: {finished['finish_kd_on_strike'].mean():.3%}")
    print(f"recent KD before finish strike: {finished['finish_recent_kd_before'].mean():.3%}")
    print(f"loser had >=1 KD absorbed: {(finished['loser_kds_absorbed'] >= 1).mean():.3%}")
    print(f"loser had >=2 KD absorbed: {(finished['loser_kds_absorbed'] >= 2).mean():.3%}")

    print("\nFINISH PATHWAYS")
    pathway = (
        finished.groupby("finish_pathway", observed=True)
        .size()
        .rename("finishes")
        .reset_index()
    )
    pathway["share"] = pathway["finishes"] / len(finished)
    print(pathway.sort_values("finishes", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nRESERVOIR AT FINISH")
    reservoir_bucket = pd.cut(
        finished["finish_reservoir_fraction"],
        bins=[-1e-9, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0000001],
        labels=["zero", "0-10%", "10-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True,
    )
    reservoir_rows = (
        finished.assign(reservoir_bucket=reservoir_bucket)
        .groupby("reservoir_bucket", observed=True, sort=False)
        .agg(
            finishes=("finished", "size"),
            mean_finish_probability=("finish_probability", "mean"),
            kd_on_finish_strike=("finish_kd_on_strike", "mean"),
            recent_kd_before=("finish_recent_kd_before", "mean"),
        )
        .reset_index()
    )
    reservoir_rows["finish_share"] = reservoir_rows["finishes"] / len(finished)
    print(reservoir_rows.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def _fighter_exposure_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        for fighter, opponent in (("red", "blue"), ("blue", "red")):
            idx = 0 if fighter == "red" else 1
            lost = int(r["finished"] == 1 and r["finish_loser"] == idx)
            won = int(r["finished"] == 1 and r["finish_winner"] == idx)
            rows.append(
                {
                    "durability": r[f"{fighter}_durability"],
                    "power_minus_opp_kd_resistance": (
                        r[f"{fighter}_power"] - r[f"{opponent}_kd_resistance"]
                    ),
                    "ko_loss": lost,
                    "ko_win": won,
                    "sig_landed": r[f"{fighter}_sig_landed"],
                    "kd_scored": r[f"{fighter}_kd_scored"],
                }
            )
    return pd.DataFrame(rows)


def _print_trait_summaries(frame: pd.DataFrame) -> None:
    fighter = _fighter_exposure_rows(frame)

    print("\nDURABILITY QUINTILES — KO/TKO LOSS")
    fighter["durability_bucket"] = _rank_bucket(
        fighter["durability"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket, g in fighter.groupby("durability_bucket", observed=True, sort=False):
        rows.append(
            {
                "bucket": str(bucket),
                "fighter_paths": len(g),
                "mean_durability": g["durability"].mean(),
                "ko_loss_probability": g["ko_loss"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nPOWER - OPPONENT KD RESISTANCE QUINTILES — KO/TKO WIN")
    fighter["edge_bucket"] = _rank_bucket(
        fighter["power_minus_opp_kd_resistance"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket, g in fighter.groupby("edge_bucket", observed=True, sort=False):
        rows.append(
            {
                "bucket": str(bucket),
                "fighter_paths": len(g),
                "mean_edge": g["power_minus_opp_kd_resistance"].mean(),
                "ko_win_probability": g["ko_win"].mean(),
                "mean_kd_scored": g["kd_scored"].mean(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Population audit for shadow Static FSR MC KO/TKO V1"
    )
    parser.add_argument("--matchups", type=int, default=DEFAULT_MATCHUPS)
    parser.add_argument("--paths-per-matchup", type=int, default=DEFAULT_PATHS_PER_MATCHUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"[KO/TKO V1 audit] loading profiles from {args.fsr_path}", flush=True)
    profiles = _latest_profiles(args.fsr_path)
    total_paths = args.matchups * args.paths_per_matchup
    print(
        f"[KO/TKO V1 audit] profiles={len(profiles):,}; matchups={args.matchups:,}; "
        f"paths/matchup={args.paths_per_matchup:,}; total paths={total_paths:,}",
        flush=True,
    )

    frame = _run_population(
        profiles,
        matchup_count=args.matchups,
        paths_per_matchup=args.paths_per_matchup,
        rounds=args.rounds,
        seed=args.seed,
    )

    _print_overall(frame, args.rounds)
    _print_trait_summaries(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"\n[KO/TKO V1 audit] wrote {args.output}", flush=True)
    print(
        "\nAUDIT BOUNDARY: inspect finish-path directions and plausibility only. "
        "KO/TKO constants remain provisional and are not calibrated targets.",
        flush=True,
    )


if __name__ == "__main__":
    main()
