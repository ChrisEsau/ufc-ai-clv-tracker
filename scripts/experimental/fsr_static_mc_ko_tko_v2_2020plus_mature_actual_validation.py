"""Validate the strong KO/TKO V2 simulator on the modern mature-fighter cohort.

Cohort contract
---------------
- UFC bouts dated 2020-01-01 or later.
- Both fighters had at least 3 prior UFC fights before the bout.
- Leakage-safe pre-fight FSR snapshots only.

This is the same cohort contract used by
``fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature.py``.
The default is intentionally only 25 Monte Carlo paths per bout for a fast
architectural diagnostic. Use --paths-per-bout for higher precision later.

No simulator constants or FSR values are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_2020plus_mature_actual_validation.parquet"
)
DEFAULT_PATHS_PER_BOUT = 25
DEFAULT_SEED = 20260810
HEARTBEAT_PATHS = 1000
STRONG = next(c for c in collapse.CANDIDATES if c.name == "strong")


def _scheduled_rounds(value: object) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(parsed):
        rounds = int(round(float(parsed)))
        if rounds in (3, 5):
            return rounds
    return 3


def _attach_actual_metadata(cohort: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Attach winner/method/scheduled-round metadata without changing cohort rows."""
    keep = [c for c in ["fight_id", "winner_id", "method", "total_rounds", "r_id", "b_id"] if c in master.columns]
    required = {"fight_id", "winner_id", "method", "r_id", "b_id"}
    missing = sorted(required - set(keep))
    if missing:
        raise ValueError(f"UFC master missing simulator-validation columns: {missing}")

    meta = master[keep].copy()
    meta["fight_id"] = meta["fight_id"].astype(str)
    meta = meta.drop_duplicates("fight_id", keep="last")
    out = cohort.merge(
        meta,
        left_on="bout_id",
        right_on="fight_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_master"),
    )
    if out["winner_id"].isna().any():
        print(
            f"[2020+ mature MC] warning: missing winner_id rows={int(out['winner_id'].isna().sum())}",
            flush=True,
        )
    out["actual_winner_id"] = out["winner_id"].where(out["winner_id"].notna(), None)
    return out


def _run_paths(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(cohort) * paths_per_bout
    path_counter = 0
    ko_counter = 0
    r1_ko_counter = 0

    print(
        f"[2020+ mature MC] candidate=strong; bouts={len(cohort):,}; "
        f"paths_per_bout={paths_per_bout}; total_paths={total_paths:,}",
        flush=True,
    )

    for bout_index, (_, bout) in enumerate(cohort.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        rounds = _scheduled_rounds(bout.get("total_rounds"))

        for path_index in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = collapse.StaticFSRMCKOTKOV2KDCollapse(
                red,
                blue,
                collapse=STRONG,
                rounds=rounds,
                seed=path_seed,
            )
            path = sim.run()
            finish = path.finish
            did_ko = finish is not None
            finish_round = int(finish.round) if did_ko and finish.round is not None else np.nan

            winner_id = None
            if did_ko:
                ko_counter += 1
                if finish_round == 1:
                    r1_ko_counter += 1
                winner_row = red if finish.winner == 0 else blue
                winner_id = str(winner_row["fighter_id"])

            rows.append(
                {
                    "bout_id": bout_id,
                    "path_index": path_index,
                    "path_seed": path_seed,
                    "scheduled_rounds": rounds,
                    "mc_ko_tko": int(did_ko),
                    "mc_finish_round": finish_round,
                    "mc_ko_winner_id": winner_id,
                }
            )

            path_counter += 1
            if path_counter % HEARTBEAT_PATHS == 0 or path_counter == total_paths:
                print(
                    f"[2020+ mature MC] paths {path_counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_index:,}/{len(cohort):,}; "
                    f"KO={ko_counter/path_counter:.2%}; R1-KO={r1_ko_counter/path_counter:.2%}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _aggregate(paths: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bout_id, g in paths.groupby("bout_id", sort=False):
        ko = g[g["mc_ko_tko"].eq(1)]
        row: dict[str, object] = {
            "bout_id": str(bout_id),
            "mc_paths": len(g),
            "mc_p_ko_tko": float(g["mc_ko_tko"].mean()),
            "mc_p_r1_ko_tko": float((g["mc_finish_round"] == 1).mean()),
        }

        for round_no in range(1, 6):
            row[f"mc_p_ko_round_{round_no}_unconditional"] = float(
                (g["mc_finish_round"] == round_no).mean()
            )
            row[f"mc_p_round_{round_no}_given_ko"] = (
                float((ko["mc_finish_round"] == round_no).mean()) if len(ko) else 0.0
            )

        if len(ko):
            round_values = pd.to_numeric(ko["mc_finish_round"], errors="coerce").dropna()
            row["mc_expected_ko_round_conditional"] = float(round_values.mean()) if len(round_values) else np.nan
            row["mc_predicted_ko_round_conditional"] = (
                int(round_values.value_counts().idxmax()) if len(round_values) else np.nan
            )
            winner_probs = ko["mc_ko_winner_id"].dropna().astype(str).value_counts(normalize=True)
            row["mc_predicted_ko_winner_id"] = str(winner_probs.index[0]) if len(winner_probs) else None
            row["mc_predicted_ko_winner_prob_conditional"] = float(winner_probs.iloc[0]) if len(winner_probs) else np.nan
        else:
            row["mc_expected_ko_round_conditional"] = np.nan
            row["mc_predicted_ko_round_conditional"] = np.nan
            row["mc_predicted_ko_winner_id"] = None
            row["mc_predicted_ko_winner_prob_conditional"] = np.nan

        rows.append(row)

    agg = pd.DataFrame(rows)
    out = agg.merge(cohort, on="bout_id", how="left", validate="one_to_one")
    out["predicted_ko_winner_correct"] = (
        out["mc_predicted_ko_winner_id"].notna()
        & out["actual_winner_id"].notna()
        & out["mc_predicted_ko_winner_id"].astype(str).eq(out["actual_winner_id"].astype(str))
    ).astype(int)
    return out


def _safe_auc(y: pd.Series, p: pd.Series) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), p.astype(float)))


def _print_summary(frame: pd.DataFrame, paths_per_bout: int) -> None:
    y_ko = frame["actual_ko_tko"].astype(int)
    y_r1 = frame["actual_r1_ko"].astype(int)
    p_ko = frame["mc_p_ko_tko"].astype(float)
    p_r1 = frame["mc_p_r1_ko_tko"].astype(float)
    actual_ko = frame[frame["actual_ko_tko"].eq(1)].copy()

    print("\n" + "=" * 124)
    print("2020+ MATURE-FIGHTER KO/TKO V2 — STRONG-COLLAPSE ACTUAL VALIDATION")
    print("=" * 124)
    print(f"bouts: {len(frame):,}")
    print(f"date range: {frame['event_date'].min().date()} -> {frame['event_date'].max().date()}")
    print(f"minimum prior UFC fights: both fighters >= {modern.MIN_PRIOR_UFC_FIGHTS}")
    print(f"MC paths per bout: {paths_per_bout}")

    print("\nANY KO/TKO")
    print(f"actual rate: {y_ko.mean():.2%} ({int(y_ko.sum()):,}/{len(frame):,})")
    print(f"mean MC p(KO): {p_ko.mean():.2%}")
    print(f"ROC-AUC: {_safe_auc(y_ko, p_ko):.4f}")
    print(f"Brier: {brier_score_loss(y_ko, p_ko):.6f}")

    print("\nROUND-1 KO/TKO")
    print(f"actual rate: {y_r1.mean():.2%} ({int(y_r1.sum()):,}/{len(frame):,})")
    print(f"mean MC p(R1 KO): {p_r1.mean():.2%}")
    print(f"ROC-AUC: {_safe_auc(y_r1, p_r1):.4f}")
    print(f"Brier: {brier_score_loss(y_r1, p_r1):.6f}")

    print("\nACTUAL VS MC KO ROUND DISTRIBUTION")
    rows = []
    actual_dist = actual_ko["actual_finish_round"].value_counts(normalize=True)
    for round_no in range(1, 6):
        rows.append(
            {
                "round": round_no,
                "actual_KO_share": float(actual_dist.get(round_no, 0.0)),
                "mean_MC_P_round_given_KO_actual_KO_bouts": float(
                    actual_ko[f"mc_p_round_{round_no}_given_ko"].mean()
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    timing = actual_ko.dropna(
        subset=["actual_finish_round", "mc_expected_ko_round_conditional", "mc_predicted_ko_round_conditional"]
    ).copy()
    if len(timing):
        timing["mode_error"] = (
            timing["mc_predicted_ko_round_conditional"] - timing["actual_finish_round"]
        ).abs()
        timing["expected_error"] = (
            timing["mc_expected_ko_round_conditional"] - timing["actual_finish_round"]
        ).abs()
        print("\nFINISH TIMING — ACTUAL KO BOUTS")
        print(f"exact modal-round hit: {(timing['mode_error'] == 0).mean():.2%}")
        print(f"modal round within +/-1: {(timing['mode_error'] <= 1).mean():.2%}")
        print(f"modal-round MAE: {timing['mode_error'].mean():.3f}")
        print(f"conditional expected-round MAE: {timing['expected_error'].mean():.3f}")

    side = actual_ko[
        actual_ko["mc_predicted_ko_winner_id"].notna() & actual_ko["actual_winner_id"].notna()
    ]
    if len(side):
        print("\nKO WINNER SIDE — ACTUAL KO BOUTS")
        print(
            f"correct: {side['predicted_ko_winner_correct'].mean():.2%} "
            f"({int(side['predicted_ko_winner_correct'].sum())}/{len(side)})"
        )

    print("\nREFERENCE FSR-ONLY OOF AUC FROM SAME COHORT")
    print("any KO, E_all_traits_interactions: 0.6292")
    print("R1 KO,  E_all_traits_interactions: 0.6372")
    print("- With only 25 paths/bout, per-bout probabilities are coarse; use this run for architecture/calibration direction.")
    print("- Rerun a finalist at >=100 paths/bout before treating individual-fight probabilities as stable.")
    print("- No simulator constants or FSR values are changed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate strong KO/TKO V2 on 2020+ bouts where both fighters had >=3 prior UFC fights"
    )
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=modern.FSR_PATH)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.paths_per_bout <= 0:
        raise ValueError("--paths-per-bout must be positive")

    master = modern._load_master(args.master)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(args.fsr_path, candidate)
    cohort = _attach_actual_metadata(cohort, master)

    print(
        f"[2020+ mature MC] eligible matched bouts={len(cohort):,}; "
        f"date={cohort['event_date'].min().date()} -> {cohort['event_date'].max().date()}",
        flush=True,
    )

    paths = _run_paths(
        cohort,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    result = _aggregate(paths, cohort)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    _print_summary(result, args.paths_per_bout)
    print(f"\n[2020+ mature MC] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
