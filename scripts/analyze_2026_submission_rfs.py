"""Analyze leakage-safe pre-fight RFS traits for 2026 submissions.

The script compares:

1. Submission winners vs winners of non-submission fights.
2. Submitted losers vs losers of non-submission fights.
3. Winner-minus-loser RFS differences in submission vs non-submission fights.

Only prior-state RFS features are included. Current-fight observation columns
containing "_fight_" are excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
)


YEAR = 2026
SOURCE = "last3"
MIN_PRIOR_FIGHTS = 3

OUTPUT_DIR = Path(
    "data/simulation/submission_rfs_analysis"
)

FAMILY_PATHS = {
    "trajectory": ROUND_FIGHTER_STATE_HISTORY_PATH,
    "suppression": ROUND_FIGHTER_SUPPRESSION_P0_2_HISTORY_PATH,
    "wrestling": ROUND_FIGHTER_WRESTLING_P0_3_HISTORY_PATH,
    "defense": ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
}

METADATA_COLUMNS = {
    "event_id",
    "fight_id",
    "fighter_id",
    "opponent_id",
    "fighter_name",
    "opponent_name",
    "event_name",
    "date",
    "corner",
}

NON_ANALYSIS_TOKENS = (
    "_fight_",
    "_has_state",
    "prior_fight_count",
    "prior_valid",
    "rounds_observed",
)


def prepare_fights(master: pd.DataFrame) -> pd.DataFrame:
    """Create one oriented winner/loser row per completed 2026 fight."""

    required = {
        "fight_id",
        "date",
        "r_id",
        "r_name",
        "b_id",
        "b_name",
        "winner_id",
        "method",
    }

    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(
            f"Master is missing required columns: {sorted(missing)}"
        )

    fights = master.copy()
    fights["date"] = pd.to_datetime(
        fights["date"],
        errors="coerce",
    )

    fights = fights.loc[
        fights["date"].dt.year.eq(YEAR)
    ].copy()

    for column in [
        "fight_id",
        "r_id",
        "b_id",
        "winner_id",
    ]:
        fights[column] = (
            fights[column]
            .astype("string")
            .str.strip()
        )

    # Protect against accidental duplicate fight rows.
    fights = (
        fights.sort_values(["date", "fight_id"])
        .drop_duplicates("fight_id", keep="last")
        .reset_index(drop=True)
    )

    winner_is_red = fights["winner_id"].eq(
        fights["r_id"]
    )
    winner_is_blue = fights["winner_id"].eq(
        fights["b_id"]
    )

    fights = fights.loc[
        winner_is_red | winner_is_blue
    ].copy()

    winner_is_red = fights["winner_id"].eq(
        fights["r_id"]
    )

    fights["winner_name"] = fights["b_name"].where(
        ~winner_is_red,
        fights["r_name"],
    )
    fights["loser_id"] = fights["r_id"].where(
        ~winner_is_red,
        fights["b_id"],
    )
    fights["loser_name"] = fights["r_name"].where(
        ~winner_is_red,
        fights["b_name"],
    )

    fights["is_submission"] = (
        fights["method"]
        .astype("string")
        .str.contains(
            r"submission|\bsub\b",
            case=False,
            na=False,
            regex=True,
        )
    )

    return fights


def numeric_rfs_columns(
    history: pd.DataFrame,
) -> list[str]:
    """Return numeric prior-state RFS columns from one family."""

    feature_columns: list[str] = []

    for column in history.columns:
        if column in METADATA_COLUMNS:
            continue

        if not column.startswith("rfs_"):
            continue

        # Target-fight observations are leakage and are excluded.
        if "_fight_" in column:
            continue

        converted = pd.to_numeric(
            history[column],
            errors="coerce",
        )

        if converted.notna().any():
            history[column] = converted
            feature_columns.append(column)

    return feature_columns


def attach_family(
    fights: pd.DataFrame,
    *,
    family_name: str,
    path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach winner and loser prior-state features for one family."""

    if not path.exists():
        print(
            f"WARNING: {family_name} artifact missing: {path}"
        )
        return fights, []

    history = pd.read_parquet(path).copy()

    required = {"fight_id", "fighter_id"}
    missing = required - set(history.columns)

    if missing:
        raise RuntimeError(
            f"{family_name} history missing keys: "
            f"{sorted(missing)}"
        )

    history["fight_id"] = (
        history["fight_id"]
        .astype("string")
        .str.strip()
    )
    history["fighter_id"] = (
        history["fighter_id"]
        .astype("string")
        .str.strip()
    )

    duplicate_count = int(
        history.duplicated(
            ["fight_id", "fighter_id"]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"{family_name} has {duplicate_count} duplicate "
            "fight_id/fighter_id rows"
        )

    features = numeric_rfs_columns(history)

    keep = [
        "fight_id",
        "fighter_id",
        *features,
    ]
    family = history[keep].copy()

    winner = family.rename(
        columns={
            "fighter_id": "winner_id",
            **{
                feature: f"winner__{feature}"
                for feature in features
            },
        }
    )

    loser = family.rename(
        columns={
            "fighter_id": "loser_id",
            **{
                feature: f"loser__{feature}"
                for feature in features
            },
        }
    )

    out = fights.merge(
        winner,
        on=["fight_id", "winner_id"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        loser,
        on=["fight_id", "loser_id"],
        how="left",
        validate="one_to_one",
    )

    winner_coverage = int(
        out[
            [f"winner__{feature}" for feature in features]
        ]
        .notna()
        .any(axis=1)
        .sum()
    )

    loser_coverage = int(
        out[
            [f"loser__{feature}" for feature in features]
        ]
        .notna()
        .any(axis=1)
        .sum()
    )

    print(
        f"{family_name:12s} | "
        f"features={len(features):3d} | "
        f"winner coverage={winner_coverage:3d}/{len(out)} | "
        f"loser coverage={loser_coverage:3d}/{len(out)}"
    )

    return out, features


def source_features(
    features: list[str],
) -> list[str]:
    """Select meaningful features for the requested RFS source."""

    selected: list[str] = []

    for feature in features:
        if SOURCE != "all":
            source_token = f"_{SOURCE}_"
            if source_token not in feature:
                continue

        if any(
            token in feature
            for token in NON_ANALYSIS_TOKENS
        ):
            continue

        selected.append(feature)

    return sorted(set(selected))


def mann_whitney_auc(
    positive: pd.Series,
    control: pd.Series,
) -> float:
    """Calculate univariate AUC using average ranks."""

    positive = positive.dropna().astype(float)
    control = control.dropna().astype(float)

    if positive.empty or control.empty:
        return float("nan")

    combined = pd.concat(
        [positive, control],
        ignore_index=True,
    )

    ranks = combined.rank(
        method="average"
    )

    positive_rank_sum = float(
        ranks.iloc[: len(positive)].sum()
    )

    u_statistic = (
        positive_rank_sum
        - len(positive)
        * (len(positive) + 1)
        / 2
    )

    return float(
        u_statistic
        / (len(positive) * len(control))
    )


def standardized_mean_difference(
    positive: pd.Series,
    control: pd.Series,
) -> float:
    """Calculate pooled-standard-deviation effect size."""

    positive = positive.dropna().astype(float)
    control = control.dropna().astype(float)

    if len(positive) < 2 or len(control) < 2:
        return float("nan")

    positive_var = float(
        positive.var(ddof=1)
    )
    control_var = float(
        control.var(ddof=1)
    )

    pooled_variance = (
        (len(positive) - 1) * positive_var
        + (len(control) - 1) * control_var
    ) / (
        len(positive)
        + len(control)
        - 2
    )

    if not np.isfinite(pooled_variance):
        return float("nan")

    if pooled_variance <= 0:
        return 0.0

    return float(
        (
            positive.mean()
            - control.mean()
        )
        / np.sqrt(pooled_variance)
    )


def feature_summary(
    *,
    feature: str,
    positive: pd.Series,
    control: pd.Series,
) -> dict[str, object] | None:
    """Summarize separation and consistency for one feature."""

    positive = pd.to_numeric(
        positive,
        errors="coerce",
    ).dropna()

    control = pd.to_numeric(
        control,
        errors="coerce",
    ).dropna()

    # Avoid ranking extremely sparse features.
    if len(positive) < 8 or len(control) < 30:
        return None

    positive_mean = float(positive.mean())
    control_mean = float(control.mean())

    positive_median = float(positive.median())
    control_median = float(control.median())

    smd = standardized_mean_difference(
        positive,
        control,
    )

    auc = mann_whitney_auc(
        positive,
        control,
    )

    if positive_median >= control_median:
        direction = "higher"
        same_side_pct = float(
            positive.ge(control_median).mean()
        )
    else:
        direction = "lower"
        same_side_pct = float(
            positive.le(control_median).mean()
        )

    auc_strength = (
        max(auc, 1.0 - auc)
        if np.isfinite(auc)
        else float("nan")
    )

    signal_score = (
        abs(smd)
        * auc_strength
        * same_side_pct
        if (
            np.isfinite(smd)
            and np.isfinite(auc_strength)
        )
        else float("nan")
    )

    common_candidate = bool(
        np.isfinite(smd)
        and abs(smd) >= 0.30
        and auc_strength >= 0.58
        and same_side_pct >= 0.65
    )

    return {
        "feature": feature,
        "submission_n": len(positive),
        "control_n": len(control),
        "submission_mean": positive_mean,
        "control_mean": control_mean,
        "submission_median": positive_median,
        "control_median": control_median,
        "direction": direction,
        "same_side_pct": same_side_pct,
        "standardized_mean_difference": smd,
        "auc": auc,
        "auc_strength": auc_strength,
        "signal_score": signal_score,
        "common_candidate": common_candidate,
    }


def build_comparison(
    fights: pd.DataFrame,
    *,
    features: list[str],
    role: str,
) -> pd.DataFrame:
    """Build winner, loser, or matchup-difference comparison."""

    rows: list[dict[str, object]] = []

    submission_mask = fights["is_submission"]
    control_mask = ~fights["is_submission"]

    for feature in features:
        winner_column = f"winner__{feature}"
        loser_column = f"loser__{feature}"

        if (
            winner_column not in fights.columns
            or loser_column not in fights.columns
        ):
            continue

        if role == "winner":
            values = fights[winner_column]
        elif role == "loser":
            values = fights[loser_column]
        elif role == "matchup_diff":
            values = (
                pd.to_numeric(
                    fights[winner_column],
                    errors="coerce",
                )
                - pd.to_numeric(
                    fights[loser_column],
                    errors="coerce",
                )
            )
        else:
            raise ValueError(
                f"Unknown comparison role: {role}"
            )

        summary = feature_summary(
            feature=feature,
            positive=values.loc[submission_mask],
            control=values.loc[control_mask],
        )

        if summary is not None:
            summary["role"] = role
            rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "common_candidate",
                "signal_score",
            ],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def print_top(
    title: str,
    table: pd.DataFrame,
    *,
    limit: int = 15,
) -> None:
    """Print the highest-ranked candidate RFS signals."""

    print()
    print(title)
    print("-" * len(title))

    if table.empty:
        print("No features met minimum coverage.")
        return

    columns = [
        "feature",
        "submission_n",
        "submission_median",
        "control_median",
        "direction",
        "same_side_pct",
        "standardized_mean_difference",
        "auc_strength",
        "common_candidate",
    ]

    printable = table[columns].head(limit).copy()

    printable["same_side_pct"] *= 100.0

    print(
        printable.to_string(
            index=False,
            formatters={
                "submission_median": "{:.4f}".format,
                "control_median": "{:.4f}".format,
                "same_side_pct": "{:.1f}%".format,
                "standardized_mean_difference": (
                    "{:+.3f}".format
                ),
                "auc_strength": "{:.3f}".format,
            },
        )
    )


def main() -> None:
    """Run the 2026 submission RFS analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_parquet(MASTER_PATH)
    fights = prepare_fights(master)

    submission_count = int(
        fights["is_submission"].sum()
    )

    total_fights = len(fights)
    submission_rate = (
        submission_count / total_fights
        if total_fights
        else 0.0
    )

    print("=" * 80)
    print("2026 SUBMISSION RFS ANALYSIS")
    print("=" * 80)
    print(f"Completed fights:     {total_fights}")
    print(f"Submission finishes: {submission_count}")
    print(
        f"Submission rate:     "
        f"{100.0 * submission_rate:.2f}%"
    )
    print(f"RFS source:          {SOURCE}")
    print()

    all_features: list[str] = []

    for family_name, path in FAMILY_PATHS.items():
        fights, features = attach_family(
            fights,
            family_name=family_name,
            path=Path(path),
        )
        all_features.extend(features)

    all_features = sorted(set(all_features))
    analysis_features = source_features(all_features)

    print()
    print(
        f"Selected {len(analysis_features)} "
        f"{SOURCE} RFS features"
    )

    trajectory_count_feature = (
        "rfs_traj_prior_fight_count"
    )

    winner_count_column = (
        f"winner__{trajectory_count_feature}"
    )
    loser_count_column = (
        f"loser__{trajectory_count_feature}"
    )

    cohorts: dict[str, pd.DataFrame] = {
        "all": fights.copy(),
    }

    if (
        winner_count_column in fights.columns
        and loser_count_column in fights.columns
    ):
        experienced_mask = (
            pd.to_numeric(
                fights[winner_count_column],
                errors="coerce",
            ).ge(MIN_PRIOR_FIGHTS)
            & pd.to_numeric(
                fights[loser_count_column],
                errors="coerce",
            ).ge(MIN_PRIOR_FIGHTS)
        )

        cohorts["experienced"] = (
            fights.loc[experienced_mask].copy()
        )

    fight_columns = [
        column
        for column in [
            "date",
            "fight_id",
            "event_name",
            "winner_id",
            "winner_name",
            "loser_id",
            "loser_name",
            "method",
            "is_submission",
            "weight_class",
            "gender",
        ]
        if column in fights.columns
    ]

    fights[fight_columns].to_csv(
        OUTPUT_DIR / "2026_fight_cohort.csv",
        index=False,
    )

    for cohort_name, cohort in cohorts.items():
        cohort_submission_count = int(
            cohort["is_submission"].sum()
        )

        print()
        print("=" * 80)
        print(
            f"COHORT: {cohort_name.upper()} | "
            f"fights={len(cohort)} | "
            f"submissions={cohort_submission_count}"
        )
        print("=" * 80)

        winner_table = build_comparison(
            cohort,
            features=analysis_features,
            role="winner",
        )

        loser_table = build_comparison(
            cohort,
            features=analysis_features,
            role="loser",
        )

        matchup_table = build_comparison(
            cohort,
            features=analysis_features,
            role="matchup_diff",
        )

        winner_table.to_csv(
            OUTPUT_DIR
            / f"{YEAR}_{SOURCE}_{cohort_name}_"
            "submission_winner_features.csv",
            index=False,
        )

        loser_table.to_csv(
            OUTPUT_DIR
            / f"{YEAR}_{SOURCE}_{cohort_name}_"
            "submitted_loser_features.csv",
            index=False,
        )

        matchup_table.to_csv(
            OUTPUT_DIR
            / f"{YEAR}_{SOURCE}_{cohort_name}_"
            "submission_matchup_diffs.csv",
            index=False,
        )

        print_top(
            "SUBMISSION WINNER COMMON FEATURES",
            winner_table,
        )

        print_top(
            "SUBMITTED FIGHTER COMMON FEATURES",
            loser_table,
        )

        print_top(
            "SUBMISSION MATCHUP DIFFERENCES",
            matchup_table,
        )

    print()
    print(f"Artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
