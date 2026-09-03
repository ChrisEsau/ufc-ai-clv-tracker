"""Diagnose winner-signal direction in the FSR/MC V1.5 2026 replay.

Shadow/research only.  This script does not run simulations and does not change
FSR, RFS, or Monte Carlo behavior.  It joins the already-generated 2026 replay
predictions to the cached leakage-safe PRE-fight locked-FSR V1.1 snapshots.

Questions answered
------------------
1. Is the simulator's winner probability directionally useful at all?
2. Does inverting the simulator pick improve accuracy?
3. Which individual persistent FSR skill differences point toward winners?
4. Does a simple equal-weight persistent-FSR composite have winner signal?
5. Which FSR deltas are most associated with the simulator's red probability?
6. What do the most confident wrong picks look like in persistent FSR space?

No target-fight statistics are used to construct the cached FSR ratings.  The
actual result is used only as the evaluation label in this diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_locked_families_v1 as fsr


DEFAULT_PREDICTIONS = Path(
    "data/simulation/rfs_mc_v2_shared_state/v1_5/replay_2026/"
    "fight_predictions.csv"
)
DEFAULT_SNAPSHOTS = Path(
    "data/simulation/rfs_mc_v2_shared_state/v1_5/replay_cache/"
    "locked_fsr_v1_1_prefight_snapshots.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state/v1_5/replay_2026/"
    "winner_signal_audit"
)


def _binary_auc(score: pd.Series, actual: pd.Series) -> float:
    """Return rank-based binary AUC without an sklearn/scipy dependency."""

    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "actual": pd.to_numeric(actual, errors="coerce"),
        }
    ).dropna()

    positive = frame.loc[frame["actual"].eq(1.0)]
    negative = frame.loc[frame["actual"].eq(0.0)]
    n_pos = len(positive)
    n_neg = len(negative)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = frame["score"].rank(method="average")
    positive_rank_sum = float(ranks.loc[frame["actual"].eq(1.0)].sum())

    return (
        positive_rank_sum
        - n_pos * (n_pos + 1) / 2.0
    ) / float(n_pos * n_neg)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    if len(frame) < 3:
        return float("nan")
    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")

    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _load_inputs(
    predictions_path: Path,
    snapshots_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions_path.exists():
        raise RuntimeError(f"Predictions not found: {predictions_path}")
    if not snapshots_path.exists():
        raise RuntimeError(f"FSR snapshot cache not found: {snapshots_path}")

    predictions = pd.read_csv(predictions_path)
    snapshots = pd.read_parquet(snapshots_path)

    required_predictions = {
        "fight_id",
        "red_fighter_id",
        "red_fighter_name",
        "blue_fighter_id",
        "blue_fighter_name",
        "actual_winner_corner",
        "sim_red_win_probability",
        "sim_blue_win_probability",
        "sim_draw_probability",
        "red_prior_fights",
        "blue_prior_fights",
    }
    missing_predictions = sorted(
        required_predictions - set(predictions.columns)
    )
    if missing_predictions:
        raise RuntimeError(
            "Replay predictions are missing required columns: "
            f"{missing_predictions}"
        )

    required_snapshots = {
        "fight_id",
        "fighter_id",
        "prior_ufc_fights",
        *fsr.SKILLS,
    }
    missing_snapshots = sorted(required_snapshots - set(snapshots.columns))
    if missing_snapshots:
        raise RuntimeError(
            "FSR snapshot cache is missing required columns: "
            f"{missing_snapshots}"
        )

    predictions["fight_id"] = predictions["fight_id"].astype(str)
    predictions["red_fighter_id"] = predictions["red_fighter_id"].astype(str)
    predictions["blue_fighter_id"] = predictions["blue_fighter_id"].astype(str)
    predictions["actual_winner_corner"] = (
        predictions["actual_winner_corner"].astype(str).str.lower()
    )

    snapshots["fight_id"] = snapshots["fight_id"].astype(str)
    snapshots["fighter_id"] = snapshots["fighter_id"].astype(str)

    if predictions["fight_id"].duplicated().any():
        raise RuntimeError("Replay predictions contain duplicate fight_id values")
    if snapshots.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("FSR snapshot cache contains duplicate fighter-fight keys")

    return predictions, snapshots


def _attach_fsr_cards(
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    selected = snapshots[
        [
            "fight_id",
            "fighter_id",
            "prior_ufc_fights",
            *fsr.SKILLS,
        ]
    ].copy()

    red = selected.rename(
        columns={
            "fighter_id": "red_fighter_id",
            "prior_ufc_fights": "snapshot_red_prior_fights",
            **{skill: f"red_fsr_{skill}" for skill in fsr.SKILLS},
        }
    )
    blue = selected.rename(
        columns={
            "fighter_id": "blue_fighter_id",
            "prior_ufc_fights": "snapshot_blue_prior_fights",
            **{skill: f"blue_fsr_{skill}" for skill in fsr.SKILLS},
        }
    )

    joined = predictions.merge(
        red,
        on=["fight_id", "red_fighter_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        blue,
        on=["fight_id", "blue_fighter_id"],
        how="left",
        validate="one_to_one",
    )

    fsr_columns = [
        f"{side}_fsr_{skill}"
        for side in ("red", "blue")
        for skill in fsr.SKILLS
    ]
    if joined[fsr_columns].isna().any().any():
        missing = joined.loc[
            joined[fsr_columns].isna().any(axis=1),
            ["fight_id", "red_fighter_name", "blue_fighter_name"],
        ]
        raise RuntimeError(
            "Some replay fights could not be joined to cached FSR cards:\n"
            + missing.head(20).to_string(index=False)
        )

    for skill in fsr.SKILLS:
        joined[f"fsr_delta_{skill}"] = (
            joined[f"red_fsr_{skill}"]
            - joined[f"blue_fsr_{skill}"]
        )

    joined["actual_red_win"] = joined["actual_winner_corner"].eq("red").astype(float)

    non_draw_mass = (
        joined["sim_red_win_probability"]
        + joined["sim_blue_win_probability"]
    )
    joined["sim_red_win_probability_conditional"] = np.where(
        non_draw_mass.gt(0.0),
        joined["sim_red_win_probability"] / non_draw_mass,
        0.5,
    )

    joined["sim_pick_red"] = (
        joined["sim_red_win_probability"]
        >= joined["sim_blue_win_probability"]
    )
    joined["sim_winner_correct"] = joined["sim_pick_red"].eq(
        joined["actual_red_win"].astype(bool)
    )
    joined["sim_inverted_correct"] = (~joined["sim_pick_red"]).eq(
        joined["actual_red_win"].astype(bool)
    )
    joined["sim_pick_confidence"] = np.maximum(
        joined["sim_red_win_probability_conditional"],
        1.0 - joined["sim_red_win_probability_conditional"],
    )

    # Locked FSR skills all use the same 50-centered rating scale, so the mean
    # rating difference is a transparent equal-weight diagnostic composite.
    delta_columns = [f"fsr_delta_{skill}" for skill in fsr.SKILLS]
    joined["fsr_equal_weight_delta"] = joined[delta_columns].mean(axis=1)
    joined["fsr_skill_vote_margin"] = (
        (joined[delta_columns] > 0.0).sum(axis=1)
        - (joined[delta_columns] < 0.0).sum(axis=1)
    )

    joined["fsr_equal_weight_pick_red"] = joined[
        "fsr_equal_weight_delta"
    ].ge(0.0)
    joined["fsr_vote_pick_red"] = joined["fsr_skill_vote_margin"].ge(0.0)

    return joined


def _overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    actual_red = frame["actual_red_win"]
    sim_probability = frame["sim_red_win_probability_conditional"]

    empirical_red_rate = float(actual_red.mean())
    flat_half_brier = float(np.mean((0.5 - actual_red) ** 2))
    empirical_rate_brier = float(
        np.mean((empirical_red_rate - actual_red) ** 2)
    )

    rows = [
        ("fights", float(len(frame))),
        ("actual_red_win_rate", empirical_red_rate),
        ("simulator_pick_accuracy", float(frame["sim_winner_correct"].mean())),
        ("inverted_simulator_pick_accuracy", float(frame["sim_inverted_correct"].mean())),
        ("always_red_accuracy", empirical_red_rate),
        ("always_blue_accuracy", 1.0 - empirical_red_rate),
        (
            "simulator_winner_auc",
            _binary_auc(sim_probability, actual_red),
        ),
        (
            "simulator_winner_brier_conditional",
            float(np.mean((sim_probability - actual_red) ** 2)),
        ),
        ("flat_50_brier", flat_half_brier),
        ("empirical_red_rate_brier_hindsight", empirical_rate_brier),
        (
            "fsr_equal_weight_accuracy",
            float(
                frame["fsr_equal_weight_pick_red"].eq(
                    actual_red.astype(bool)
                ).mean()
            ),
        ),
        (
            "fsr_equal_weight_auc",
            _binary_auc(frame["fsr_equal_weight_delta"], actual_red),
        ),
        (
            "fsr_skill_vote_accuracy",
            float(
                frame["fsr_vote_pick_red"].eq(
                    actual_red.astype(bool)
                ).mean()
            ),
        ),
        (
            "corr_fsr_equal_weight_to_sim_red_probability",
            _safe_corr(
                frame["fsr_equal_weight_delta"],
                sim_probability,
            ),
        ),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def _skill_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actual_red = frame["actual_red_win"]
    sim_probability = frame["sim_red_win_probability_conditional"]

    for skill in fsr.SKILLS:
        delta = frame[f"fsr_delta_{skill}"]
        non_tie = delta.ne(0.0)
        directional_accuracy = float("nan")
        if non_tie.any():
            pick_red = delta.loc[non_tie].gt(0.0)
            directional_accuracy = float(
                pick_red.eq(
                    actual_red.loc[non_tie].astype(bool)
                ).mean()
            )

        rows.append(
            {
                "skill": skill,
                "fights": int(len(frame)),
                "non_tie_fights": int(non_tie.sum()),
                "higher_rating_pick_accuracy": directional_accuracy,
                "winner_auc": _binary_auc(delta, actual_red),
                "corr_delta_to_actual_red_win": _safe_corr(delta, actual_red),
                "corr_delta_to_sim_red_probability": _safe_corr(
                    delta,
                    sim_probability,
                ),
                "mean_delta_when_red_wins": float(
                    delta.loc[actual_red.eq(1.0)].mean()
                ),
                "mean_delta_when_blue_wins": float(
                    delta.loc[actual_red.eq(0.0)].mean()
                ),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["winner_auc", "higher_rating_pick_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def _experience_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    minimum_prior = frame[["red_prior_fights", "blue_prior_fights"]].min(axis=1)
    work = frame.copy()
    work["experience_band"] = pd.cut(
        minimum_prior,
        bins=[-1, 0, 2, 5, float("inf")],
        labels=["0_prior", "1-2_prior", "3-5_prior", "6+_prior"],
    )

    rows = []
    for band, group in work.groupby("experience_band", observed=True):
        rows.append(
            {
                "experience_band": str(band),
                "fights": int(len(group)),
                "simulator_accuracy": float(group["sim_winner_correct"].mean()),
                "inverted_accuracy": float(group["sim_inverted_correct"].mean()),
                "simulator_auc": _binary_auc(
                    group["sim_red_win_probability_conditional"],
                    group["actual_red_win"],
                ),
                "fsr_equal_weight_accuracy": float(
                    group["fsr_equal_weight_pick_red"].eq(
                        group["actual_red_win"].astype(bool)
                    ).mean()
                ),
                "fsr_equal_weight_auc": _binary_auc(
                    group["fsr_equal_weight_delta"],
                    group["actual_red_win"],
                ),
                "mean_sim_confidence": float(group["sim_pick_confidence"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _print_report(
    overall: pd.DataFrame,
    skills: pd.DataFrame,
    experience: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    values = dict(zip(overall["metric"], overall["value"]))

    print()
    print("=" * 118)
    print("FSR / MC V1.5 — WINNER SIGNAL DIRECTION AUDIT")
    print("=" * 118)
    print(f"Fights                              : {int(values['fights'])}")
    print(f"Actual red win rate                 : {100 * values['actual_red_win_rate']:.2f}%")
    print(f"Simulator pick accuracy             : {100 * values['simulator_pick_accuracy']:.2f}%")
    print(f"Inverted simulator pick accuracy    : {100 * values['inverted_simulator_pick_accuracy']:.2f}%")
    print(f"Simulator winner AUC                : {values['simulator_winner_auc']:.4f}")
    print(f"Simulator conditional Brier         : {values['simulator_winner_brier_conditional']:.4f}")
    print(f"Flat 50% Brier                      : {values['flat_50_brier']:.4f}")
    print(f"Equal-weight FSR accuracy           : {100 * values['fsr_equal_weight_accuracy']:.2f}%")
    print(f"Equal-weight FSR AUC                : {values['fsr_equal_weight_auc']:.4f}")
    print(f"FSR skill-vote accuracy             : {100 * values['fsr_skill_vote_accuracy']:.2f}%")
    print(
        "Corr(FSR equal-weight delta, sim P(red)): "
        f"{values['corr_fsr_equal_weight_to_sim_red_probability']:.4f}"
    )

    print()
    print("Persistent FSR skill direction")
    print("-" * 118)
    display = skills[
        [
            "skill",
            "higher_rating_pick_accuracy",
            "winner_auc",
            "corr_delta_to_actual_red_win",
            "corr_delta_to_sim_red_probability",
            "mean_delta_when_red_wins",
            "mean_delta_when_blue_wins",
        ]
    ].copy()
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("Experience-band signal")
    print("-" * 118)
    print(experience.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    wrong = enriched.loc[~enriched["sim_winner_correct"]].copy()
    wrong = wrong.sort_values("sim_pick_confidence", ascending=False).head(20)

    print()
    print("Most confident wrong picks with equal-weight FSR direction")
    print("-" * 118)
    if wrong.empty:
        print("None")
    else:
        print(
            wrong[
                [
                    "date",
                    "red_fighter_name",
                    "blue_fighter_name",
                    "actual_winner_corner",
                    "sim_red_win_probability",
                    "sim_blue_win_probability",
                    "sim_pick_confidence",
                    "fsr_equal_weight_delta",
                    "fsr_skill_vote_margin",
                    "red_prior_fights",
                    "blue_prior_fights",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=DEFAULT_SNAPSHOTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    predictions, snapshots = _load_inputs(
        args.predictions,
        args.snapshots,
    )
    enriched = _attach_fsr_cards(predictions, snapshots)
    overall = _overall_metrics(enriched)
    skills = _skill_metrics(enriched)
    experience = _experience_metrics(enriched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_dir / "predictions_with_fsr_cards.csv", index=False)
    overall.to_csv(args.output_dir / "overall_signal_metrics.csv", index=False)
    skills.to_csv(args.output_dir / "fsr_skill_signal.csv", index=False)
    experience.to_csv(args.output_dir / "experience_signal.csv", index=False)

    wrong = enriched.loc[~enriched["sim_winner_correct"]].copy()
    wrong = wrong.sort_values("sim_pick_confidence", ascending=False)
    wrong.to_csv(args.output_dir / "confident_wrong_with_fsr.csv", index=False)

    _print_report(overall, skills, experience, enriched)

    print()
    print("Saved winner-signal audit:")
    for name in (
        "overall_signal_metrics.csv",
        "fsr_skill_signal.csv",
        "experience_signal.csv",
        "predictions_with_fsr_cards.csv",
        "confident_wrong_with_fsr.csv",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
