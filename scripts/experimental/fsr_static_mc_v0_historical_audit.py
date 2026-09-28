"""Leakage-safe historical replay audit for FSR Static MC V0.

For each historical UFC fight with two pre-fight FSR-26 snapshots, this script:

1. uses only the FSR snapshot attached to that historical fight;
2. simulates the matchup repeatedly with the static 10-second V0 engine;
3. averages simulated fighter-fight outputs;
4. compares those outputs with realized RFS fight-level outcomes.

This is a path/calibration audit, not a winner-prediction backtest. Finishes,
judging, fatigue, damage state, adversity, recovery, and urgency remain disabled.

Important data limitation
-------------------------
UFCStats does not provide literal distance/clinch/ground occupancy seconds in the
classic round feed. The available RFS phase-share targets are strike-attempt
shares. Therefore simulated phase occupancy vs realized phase-attempt share is a
proxy comparison. TD attempts/round, TD completion, and control seconds/round are
more direct historical comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental.fsr_static_mc_v0 import StaticFSRMCV0

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")

REALIZED_COLUMNS = {
    "distance_share": "rfs_phase_base_fight_distance_attempt_share",
    "clinch_share": "rfs_phase_base_fight_clinch_attempt_share",
    "ground_share": "rfs_phase_base_fight_ground_attempt_share",
    "td_attempts_per_round": "rfs_phase_base_fight_td_attempts_per_round",
    "td_completion": "rfs_phase_base_fight_td_completion_rate",
    "control_seconds_per_round": "rfs_phase_base_fight_control_seconds_per_round",
}


def _prepare_frames(fsr_path: Path, rfs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[historical audit] loading FSR-26 from {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path).copy()
    print(f"[historical audit] loaded {len(fsr):,} FSR rows", flush=True)

    print(f"[historical audit] loading RFS history from {rfs_path}", flush=True)
    rfs = pd.read_parquet(rfs_path).copy()
    print(f"[historical audit] loaded {len(rfs):,} RFS rows", flush=True)

    for frame in (fsr, rfs):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)

    missing = [c for c in REALIZED_COLUMNS.values() if c not in rfs.columns]
    if missing:
        raise ValueError(f"RFS history missing required audit columns: {missing}")

    # Restrict to fights with exactly two pre-fight fighter rows. This protects
    # the simulator matchup grain and avoids malformed historical records.
    counts = fsr.groupby("fight_id")["fighter_id"].nunique()
    valid_fights = counts[counts == 2].index
    fsr = fsr[fsr["fight_id"].isin(valid_fights)].copy()

    return fsr, rfs


def _simulate_one_fight(
    pair: pd.DataFrame,
    *,
    sims_per_fight: int,
    rounds: int,
    base_seed: int,
) -> list[dict[str, float | str]]:
    pair = pair.reset_index(drop=True)
    red = pair.iloc[0]
    blue = pair.iloc[1]

    accum = [
        {
            "distance_share": 0.0,
            "clinch_share": 0.0,
            "ground_share": 0.0,
            "td_attempts_per_round": 0.0,
            "td_completion_num": 0.0,
            "td_completion_den": 0.0,
            "control_seconds_per_round": 0.0,
        }
        for _ in range(2)
    ]

    for sim_i in range(sims_per_fight):
        # Deterministic but distinct seed per fight/simulation.
        sim = StaticFSRMCV0(
            red,
            blue,
            rounds=rounds,
            seed=base_seed + sim_i,
        )
        path = sim.run()

        for i, stats in enumerate(path.stats):
            total_segments = max(sum(stats.phase_segments.values()), 1)
            accum[i]["distance_share"] += stats.phase_segments["DISTANCE"] / total_segments
            accum[i]["clinch_share"] += stats.phase_segments["CLINCH"] / total_segments
            accum[i]["ground_share"] += stats.phase_segments["GROUND"] / total_segments
            accum[i]["td_attempts_per_round"] += stats.td_att / rounds
            accum[i]["td_completion_num"] += stats.td_landed
            accum[i]["td_completion_den"] += stats.td_att
            accum[i]["control_seconds_per_round"] += stats.control_seconds / rounds

    rows: list[dict[str, float | str]] = []
    for i in range(2):
        den = accum[i]["td_completion_den"]
        rows.append(
            {
                "fight_id": str(pair.iloc[i]["fight_id"]),
                "fighter_id": str(pair.iloc[i]["fighter_id"]),
                "sim_distance_share": accum[i]["distance_share"] / sims_per_fight,
                "sim_clinch_share": accum[i]["clinch_share"] / sims_per_fight,
                "sim_ground_share": accum[i]["ground_share"] / sims_per_fight,
                "sim_td_attempts_per_round": accum[i]["td_attempts_per_round"] / sims_per_fight,
                # Pool TD attempts across paths rather than averaging path-level
                # percentages, which avoids unstable 0/0 and 1-attempt paths.
                "sim_td_completion": accum[i]["td_completion_num"] / den if den else 0.0,
                "sim_control_seconds_per_round": accum[i]["control_seconds_per_round"] / sims_per_fight,
            }
        )
    return rows


def _safe_spearman(frame: pd.DataFrame, a: str, b: str) -> float:
    x = pd.to_numeric(frame[a], errors="coerce")
    y = pd.to_numeric(frame[b], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"a": x[mask], "b": y[mask]}).corr(method="spearman").iloc[0, 1])


def _print_metric_table(audit: pd.DataFrame) -> None:
    metric_pairs = [
        ("distance share*", "sim_distance_share", "real_distance_share"),
        ("clinch share*", "sim_clinch_share", "real_clinch_share"),
        ("ground share*", "sim_ground_share", "real_ground_share"),
        ("TD attempts/round", "sim_td_attempts_per_round", "real_td_attempts_per_round"),
        ("TD completion", "sim_td_completion", "real_td_completion"),
        ("control sec/round", "sim_control_seconds_per_round", "real_control_seconds_per_round"),
    ]

    rows = []
    for label, sim_col, real_col in metric_pairs:
        sim = pd.to_numeric(audit[sim_col], errors="coerce")
        real = pd.to_numeric(audit[real_col], errors="coerce")
        mask = sim.notna() & real.notna()
        mae = float(np.mean(np.abs(sim[mask] - real[mask]))) if mask.any() else np.nan
        rows.append(
            {
                "metric": label,
                "rows": int(mask.sum()),
                "sim_mean": float(sim[mask].mean()) if mask.any() else np.nan,
                "real_mean": float(real[mask].mean()) if mask.any() else np.nan,
                "mean_bias": float((sim[mask] - real[mask]).mean()) if mask.any() else np.nan,
                "mae": mae,
                "spearman": _safe_spearman(audit.loc[mask], sim_col, real_col),
            }
        )

    out = pd.DataFrame(rows)
    print("\nHISTORICAL FIGHTER-FIGHT AUDIT")
    print("=" * 110)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n* phase comparisons are simulated occupancy vs realized strike-attempt share proxies, not literal historical phase time.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay historical fights through FSR Static MC V0")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--rfs-path", type=Path, default=RFS_PATH)
    parser.add_argument("--sims-per-fight", type=int, default=25)
    parser.add_argument("--max-fights", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/simulation/rfs_mc_v2_shared_state/fsr_static_mc_v0_historical_audit.parquet"),
    )
    args = parser.parse_args()

    if args.sims_per_fight <= 0:
        raise SystemExit("--sims-per-fight must be positive")
    if args.max_fights <= 0:
        raise SystemExit("--max-fights must be positive")

    fsr, rfs = _prepare_frames(args.fsr_path, args.rfs_path)

    fight_ids = fsr["fight_id"].drop_duplicates().tolist()
    if len(fight_ids) > args.max_fights:
        # Deterministic evenly-spaced sample across chronological/artifact order,
        # rather than taking only the newest or oldest fights.
        idx = np.linspace(0, len(fight_ids) - 1, args.max_fights, dtype=int)
        fight_ids = [fight_ids[i] for i in idx]

    print(
        f"[historical audit] replaying {len(fight_ids):,} fights x "
        f"{args.sims_per_fight:,} paths ({len(fight_ids) * args.sims_per_fight:,} total paths)",
        flush=True,
    )

    simulated_rows: list[dict[str, float | str]] = []
    grouped = fsr.set_index("fight_id", drop=False)
    for fight_i, fight_id in enumerate(fight_ids, 1):
        pair = grouped.loc[[fight_id]].copy()
        if len(pair) != 2:
            continue
        fight_seed = args.seed + fight_i * 100_003
        simulated_rows.extend(
            _simulate_one_fight(
                pair,
                sims_per_fight=args.sims_per_fight,
                rounds=args.rounds,
                base_seed=fight_seed,
            )
        )
        if fight_i == 1 or fight_i % 50 == 0 or fight_i == len(fight_ids):
            print(f"[historical audit] fight {fight_i:,}/{len(fight_ids):,}", flush=True)

    sim = pd.DataFrame(simulated_rows)

    realized = rfs[["fight_id", "fighter_id", *REALIZED_COLUMNS.values()]].copy()
    realized = realized.rename(columns={v: f"real_{k}" for k, v in REALIZED_COLUMNS.items()})
    audit = sim.merge(realized, on=["fight_id", "fighter_id"], how="inner", validate="one_to_one")

    print(f"[historical audit] matched fighter-fight rows: {len(audit):,}", flush=True)
    _print_metric_table(audit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output, index=False)
    print(f"\n[historical audit] wrote {args.output}", flush=True)

    print(
        "\nINTERPRETATION: population means test absolute calibration; Spearman tests whether "
        "fighter/matchup-specific FSR differences rank realized historical behavior. "
        "Finishes and dynamic state remain disabled in this audit."
    )


if __name__ == "__main__":
    main()
