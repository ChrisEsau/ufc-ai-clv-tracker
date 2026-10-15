"""Finalist validation for FSR Static MC V0 control persistence.

This shadow-only research script compares the two control-calibration finalists:

- ground exit 30s = 0.20, clinch separation 30s = 0.25
- ground exit 30s = 0.25, clinch separation 30s = 0.20

It does not modify the simulator's committed constants. For each finalist it
temporarily injects equivalent 10-second hazards, replays the same leakage-safe
full-distance historical cohort, and scores the resulting fighter totals.

The primary calibration cohort is three-round full-distance fights. Five-round
fights are retained as a secondary sanity check because that cohort is much smaller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.experimental.fsr_static_mc_v0 as mc
from scripts.experimental.fsr_static_mc_v0_decision_stats_audit import (
    FSR_PATH,
    RFS_PATH,
    _attach_realized,
    _decision_fight_table,
    _load_frames,
    _simulate_fight,
)

# Finalists selected from the broad control-persistence sweep.
CONTROL_FINALISTS_30S = (
    (0.20, 0.25),
    (0.25, 0.20),
)

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_v0_control_calibration_sweep.parquet"
)


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _relative_bias(sim_mean: float, real_mean: float) -> float:
    if not np.isfinite(real_mean) or abs(real_mean) < 1e-12:
        return float("nan")
    return (sim_mean - real_mean) / real_mean


def _cohort_summary(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Summarize one scheduled-round cohort using fighter-fight means."""
    out: dict[str, float] = {}

    metrics = (
        "sig_att",
        "sig_landed",
        "td_att",
        "td_landed",
        "control_seconds",
        "sub_att",
        "reversals",
    )
    for metric in metrics:
        sim_mean = _mean(frame, f"sim_{metric}")
        real_mean = _mean(frame, f"real_{metric}")
        out[f"{prefix}_{metric}_sim_mean"] = sim_mean
        out[f"{prefix}_{metric}_real_mean"] = real_mean
        out[f"{prefix}_{metric}_rel_bias"] = _relative_bias(sim_mean, real_mean)

    out[f"{prefix}_clinch_control_sim_mean"] = _mean(
        frame, "sim_clinch_control_seconds"
    )
    out[f"{prefix}_ground_control_sim_mean"] = _mean(
        frame, "sim_ground_control_seconds"
    )

    # Control is the primary objective. TD and significant-strike attempt
    # volume protect already-useful calibration from collateral damage.
    control = abs(out[f"{prefix}_control_seconds_rel_bias"])
    td = abs(out[f"{prefix}_td_att_rel_bias"])
    sig = abs(out[f"{prefix}_sig_att_rel_bias"])
    sub = abs(out[f"{prefix}_sub_att_rel_bias"])
    rev = abs(out[f"{prefix}_reversals_rel_bias"])
    out[f"{prefix}_calibration_score"] = (
        3.0 * control
        + 1.5 * td
        + 1.0 * sig
        + 0.5 * sub
        + 0.5 * rev
    )
    return out


def _selected_fights(
    fsr: pd.DataFrame,
    decisions: pd.DataFrame,
    max_fights: int,
) -> tuple[list[str], dict[str, int]]:
    decision_ids = decisions["fight_id"].drop_duplicates().tolist()
    fsr_counts = fsr.groupby("fight_id")["fighter_id"].nunique()
    valid_fsr = set(fsr_counts[fsr_counts == 2].index.astype(str))
    decision_ids = [fight_id for fight_id in decision_ids if fight_id in valid_fsr]

    if len(decision_ids) > max_fights:
        idx = np.linspace(0, len(decision_ids) - 1, max_fights, dtype=int)
        decision_ids = [decision_ids[i] for i in idx]

    rounds_map = (
        decisions[["fight_id", "scheduled_rounds"]]
        .drop_duplicates("fight_id")
        .set_index("fight_id")["scheduled_rounds"]
        .to_dict()
    )
    return decision_ids, rounds_map


def _run_candidate(
    *,
    fsr: pd.DataFrame,
    decisions: pd.DataFrame,
    fight_ids: list[str],
    rounds_map: dict[str, int],
    sims_per_fight: int,
    seed: int,
    ground_exit_30s: float,
    clinch_separate_30s: float,
) -> pd.DataFrame:
    # Convert the research source priors to the simulator's 10-second clock.
    mc.GROUND_EXIT_BASE = mc._rescale_interval_prob(
        ground_exit_30s,
        mc.CALIBRATION_INTERVAL_SECONDS,
        mc.SEGMENT_SECONDS,
    )
    mc.CLINCH_SEPARATE_BASE = mc._rescale_interval_prob(
        clinch_separate_30s,
        mc.CALIBRATION_INTERVAL_SECONDS,
        mc.SEGMENT_SECONDS,
    )

    grouped = fsr.set_index("fight_id", drop=False)
    rows: list[dict[str, float | str | int]] = []
    for fight_i, fight_id in enumerate(fight_ids, 1):
        pair = grouped.loc[[fight_id]].copy()
        if len(pair) != 2:
            continue
        rounds = int(rounds_map[fight_id])
        rows.extend(
            _simulate_fight(
                pair,
                rounds=rounds,
                sims_per_fight=sims_per_fight,
                seed=seed + fight_i * 100_003,
            )
        )
        if fight_i == 1 or fight_i % 100 == 0 or fight_i == len(fight_ids):
            print(
                f"[control finalists] fight {fight_i:,}/{len(fight_ids):,}",
                flush=True,
            )

    sim = pd.DataFrame(rows)
    selected = decisions[decisions["fight_id"].isin(fight_ids)].copy()
    return _attach_realized(sim, selected)


def _print_ranked(results: pd.DataFrame) -> None:
    columns = [
        "rank",
        "ground_exit_30s",
        "clinch_separate_30s",
        "score",
        "r3_control_seconds_sim_mean",
        "r3_control_seconds_real_mean",
        "r3_control_seconds_rel_bias",
        "r3_clinch_control_sim_mean",
        "r3_ground_control_sim_mean",
        "r3_td_att_sim_mean",
        "r3_td_att_real_mean",
        "r3_sig_att_sim_mean",
        "r3_sig_att_real_mean",
        "r3_sub_att_sim_mean",
        "r3_sub_att_real_mean",
        "r3_reversals_sim_mean",
        "r3_reversals_real_mean",
        "r5_control_seconds_sim_mean",
        "r5_control_seconds_real_mean",
        "r5_control_seconds_rel_bias",
    ]
    available = [c for c in columns if c in results.columns]
    print("\nCONTROL CALIBRATION FINALISTS — RANKED")
    print("=" * 190)
    print(
        results[available]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate two control-persistence finalists for Static MC V0"
    )
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--rfs-path", type=Path, default=RFS_PATH)
    parser.add_argument("--max-fights", type=int, default=500)
    parser.add_argument("--sims-per-fight", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4109)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.max_fights <= 0 or args.sims_per_fight <= 0:
        raise SystemExit("--max-fights and --sims-per-fight must be positive")

    fsr, rfs = _load_frames(args.fsr_path, args.rfs_path)
    decisions = _decision_fight_table(rfs)
    fight_ids, rounds_map = _selected_fights(fsr, decisions, args.max_fights)

    n3 = sum(int(rounds_map[fid]) == 3 for fid in fight_ids)
    n5 = sum(int(rounds_map[fid]) == 5 for fid in fight_ids)
    combinations = len(CONTROL_FINALISTS_30S)
    paths_per_candidate = len(fight_ids) * args.sims_per_fight
    print(
        f"[control finalists] cohort={len(fight_ids):,} fights "
        f"({n3:,} three-round, {n5:,} five-round)",
        flush=True,
    )
    print(
        f"[control finalists] {combinations} finalists x {paths_per_candidate:,} paths "
        f"= {combinations * paths_per_candidate:,} total paths",
        flush=True,
    )

    original_ground = mc.GROUND_EXIT_BASE
    original_clinch = mc.CLINCH_SEPARATE_BASE
    result_rows: list[dict[str, float]] = []

    try:
        for candidate_no, (ground_exit, clinch_sep) in enumerate(
            CONTROL_FINALISTS_30S,
            1,
        ):
            print(
                f"[control finalists] candidate {candidate_no}/{combinations}: "
                f"ground_exit_30s={ground_exit:.2f}, "
                f"clinch_separate_30s={clinch_sep:.2f}",
                flush=True,
            )
            audit = _run_candidate(
                fsr=fsr,
                decisions=decisions,
                fight_ids=fight_ids,
                rounds_map=rounds_map,
                sims_per_fight=args.sims_per_fight,
                seed=args.seed,
                ground_exit_30s=ground_exit,
                clinch_separate_30s=clinch_sep,
            )

            row: dict[str, float] = {
                "ground_exit_30s": ground_exit,
                "clinch_separate_30s": clinch_sep,
                "fighter_rows": float(len(audit)),
            }
            r3 = audit[audit["scheduled_rounds"] == 3]
            r5 = audit[audit["scheduled_rounds"] == 5]
            if len(r3):
                row.update(_cohort_summary(r3, "r3"))
            if len(r5):
                row.update(_cohort_summary(r5, "r5"))

            # Three-round performance is primary. Five-round performance is a
            # smaller secondary guardrail rather than an equal objective.
            r3_score = row.get("r3_calibration_score", np.nan)
            r5_score = row.get("r5_calibration_score", np.nan)
            if np.isfinite(r5_score):
                row["score"] = float(r3_score + 0.25 * r5_score)
            else:
                row["score"] = float(r3_score)
            result_rows.append(row)
    finally:
        # Never leave the imported simulator module mutated after research.
        mc.GROUND_EXIT_BASE = original_ground
        mc.CLINCH_SEPARATE_BASE = original_clinch

    results = pd.DataFrame(result_rows).sort_values(
        ["score", "ground_exit_30s", "clinch_separate_30s"]
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    _print_ranked(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    print(f"\n[control finalists] wrote {args.output}", flush=True)

    best = results.iloc[0]
    print("\nBEST VALIDATION FINALIST")
    print("-" * 80)
    print(f"ground exit 30s base      : {best['ground_exit_30s']:.2f}")
    print(f"clinch separation 30s base: {best['clinch_separate_30s']:.2f}")
    print(f"weighted score            : {best['score']:.4f}")
    print(
        f"3R control                : {best['r3_control_seconds_sim_mean']:.2f}s sim vs "
        f"{best['r3_control_seconds_real_mean']:.2f}s real"
    )
    print(
        f"3R TD attempts            : {best['r3_td_att_sim_mean']:.3f} sim vs "
        f"{best['r3_td_att_real_mean']:.3f} real"
    )
    print(
        f"3R sig attempts           : {best['r3_sig_att_sim_mean']:.2f} sim vs "
        f"{best['r3_sig_att_real_mean']:.2f} real"
    )

    print(
        "\nNOTE: finalist validation only. This script does not promote or change "
        "the committed simulator priors."
    )


if __name__ == "__main__":
    main()
