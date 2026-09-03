"""Audit winner/loser behavior across raw RFS population percentiles.

Reads the existing 750-fight compression-audit output. No simulator or
resolver behavior is modified.

For every target:
- convert each matchup into one winner and one loser observation
- compare winner vs loser raw percentile distributions
- calculate TRAIN and DEVELOPMENT AUC
- build raw-percentile decile win rates
- learn direction from TRAIN only
- fit a TRAIN-only monotonic percentile -> win-probability curve
- evaluate that curve on DEVELOPMENT

2025+ remains untouched because it was never present in the input audit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    "historical_parameter_raw_percentiles_v1.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state"
)

FIGHTER_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_raw_percentile_fighter_rows_v1.csv"
)

BIN_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_raw_percentile_win_bins_v1.csv"
)

CURVE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_raw_percentile_curves_v1.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_raw_percentile_curve_summary_v1.csv"
)

FAMILY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_raw_percentile_family_summary_v1.csv"
)


BIN_EDGES = np.linspace(0.0, 1.0, 11)

# These are diagnostic thresholds, not production promotion rules.
MIN_TRAIN_FIGHTS = 300
MIN_DEVELOPMENT_FIGHTS = 150


def _auc(
    outcome: pd.Series,
    score: pd.Series,
) -> float:
    """ROC AUC using the rank-sum definition."""

    frame = pd.DataFrame(
        {
            "outcome": pd.to_numeric(
                outcome,
                errors="coerce",
            ),
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
        }
    ).dropna()

    positives = frame["outcome"] == 1
    negatives = frame["outcome"] == 0

    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = frame["score"].rank(
        method="average"
    )

    positive_rank_sum = float(
        ranks.loc[positives].sum()
    )

    u = (
        positive_rank_sum
        - n_pos * (n_pos + 1) / 2.0
    )

    return float(
        u / (n_pos * n_neg)
    )


def _weighted_isotonic(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted non-decreasing isotonic fit using PAVA."""

    blocks: list[dict[str, float | int]] = []

    for index, (value, weight) in enumerate(
        zip(values, weights, strict=True)
    ):
        blocks.append(
            {
                "start": index,
                "end": index,
                "value": float(value),
                "weight": float(weight),
            }
        )

        while (
            len(blocks) >= 2
            and float(blocks[-2]["value"])
            > float(blocks[-1]["value"])
        ):
            right = blocks.pop()
            left = blocks.pop()

            total_weight = (
                float(left["weight"])
                + float(right["weight"])
            )

            combined_value = (
                float(left["value"])
                * float(left["weight"])
                + float(right["value"])
                * float(right["weight"])
            ) / total_weight

            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "value": combined_value,
                    "weight": total_weight,
                }
            )

    result = np.empty(
        len(values),
        dtype=float,
    )

    for block in blocks:
        result[
            int(block["start"]):
            int(block["end"]) + 1
        ] = float(block["value"])

    return result


def _to_fighter_rows(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Turn every fight-target row into Red and Blue fighter observations."""

    required = {
        "fight_id",
        "date",
        "split",
        "target",
        "family",
        "red_win",
        "red_raw_percentile",
        "blue_raw_percentile",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Input audit missing required columns: "
            f"{sorted(missing)}"
        )

    common = raw[
        [
            "fight_id",
            "date",
            "split",
            "target",
            "family",
        ]
    ].copy()

    red = common.copy()
    red["corner"] = "RED"
    red["percentile"] = pd.to_numeric(
        raw["red_raw_percentile"],
        errors="coerce",
    )
    red["won"] = pd.to_numeric(
        raw["red_win"],
        errors="coerce",
    )

    blue = common.copy()
    blue["corner"] = "BLUE"
    blue["percentile"] = pd.to_numeric(
        raw["blue_raw_percentile"],
        errors="coerce",
    )
    blue["won"] = (
        1
        - pd.to_numeric(
            raw["red_win"],
            errors="coerce",
        )
    )

    fighters = pd.concat(
        [red, blue],
        ignore_index=True,
    )

    fighters = fighters.loc[
        fighters["percentile"].notna()
        & fighters["won"].isin([0, 1])
        & fighters["split"].isin(
            ["TRAIN", "DEVELOPMENT"]
        )
    ].copy()

    fighters["percentile"] = fighters[
        "percentile"
    ].clip(
        lower=0.0,
        upper=1.0,
    )

    fighters["won"] = fighters[
        "won"
    ].astype(int)

    return fighters


def _raw_bins(
    fighters: pd.DataFrame,
) -> pd.DataFrame:
    """Build literal raw-percentile win-rate bins."""

    work = fighters.copy()

    work["percentile_bin"] = pd.cut(
        work["percentile"],
        bins=BIN_EDGES,
        include_lowest=True,
        right=False,
    )

    rows = []

    for (
        split,
        family,
        target,
        percentile_bin,
    ), group in work.groupby(
        [
            "split",
            "family",
            "target",
            "percentile_bin",
        ],
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                "split": split,
                "family": family,
                "target": target,
                "percentile_bin": str(
                    percentile_bin
                ),
                "bin_low": float(
                    percentile_bin.left
                ),
                "bin_high": float(
                    min(
                        percentile_bin.right,
                        1.0,
                    )
                ),
                "fighter_observations": len(
                    group
                ),
                "fights": group[
                    "fight_id"
                ].nunique(),
                "mean_percentile": float(
                    group["percentile"].mean()
                ),
                "win_rate": float(
                    group["won"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def _fit_target_curve(
    train: pd.DataFrame,
    *,
    direction: int,
) -> pd.DataFrame:
    """Fit a monotonic TRAIN curve for one target.

    Direction is learned from TRAIN AUC:
      +1 means higher percentile historically helps.
      -1 means lower percentile historically helps.

    For fitting only, lower-helpful targets are reflected:
      oriented_percentile = 1 - raw_percentile
    """

    work = train.copy()

    if direction == 1:
        work["oriented_percentile"] = work[
            "percentile"
        ]
    else:
        work["oriented_percentile"] = (
            1.0
            - work["percentile"]
        )

    work["curve_bin"] = pd.cut(
        work["oriented_percentile"],
        bins=BIN_EDGES,
        include_lowest=True,
        right=False,
    )

    grouped = (
        work
        .groupby(
            "curve_bin",
            observed=True,
            sort=True,
        )
        .agg(
            fighter_observations=(
                "won",
                "size",
            ),
            fights=(
                "fight_id",
                "nunique",
            ),
            mean_oriented_percentile=(
                "oriented_percentile",
                "mean",
            ),
            empirical_win_rate=(
                "won",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped = grouped.loc[
        grouped[
            "fighter_observations"
        ] > 0
    ].copy()

    if grouped.empty:
        return grouped

    grouped[
        "fitted_win_rate"
    ] = _weighted_isotonic(
        grouped[
            "empirical_win_rate"
        ].to_numpy(dtype=float),
        grouped[
            "fighter_observations"
        ].to_numpy(dtype=float),
    )

    return grouped


def _predict_curve(
    percentile: pd.Series,
    *,
    direction: int,
    curve: pd.DataFrame,
) -> np.ndarray:
    """Interpolate TRAIN isotonic curve onto new fighter percentiles."""

    raw = pd.to_numeric(
        percentile,
        errors="coerce",
    ).to_numpy(dtype=float)

    if direction == 1:
        oriented = raw
    else:
        oriented = (
            1.0 - raw
        )

    x = curve[
        "mean_oriented_percentile"
    ].to_numpy(dtype=float)

    y = curve[
        "fitted_win_rate"
    ].to_numpy(dtype=float)

    if len(x) == 0:
        return np.full(
            len(oriented),
            0.5,
        )

    if len(x) == 1:
        return np.full(
            len(oriented),
            y[0],
        )

    return np.interp(
        oriented,
        x,
        y,
        left=y[0],
        right=y[-1],
    )


def _brier(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Binary Brier score."""

    return float(
        np.mean(
            (
                predicted
                - actual
            )
            ** 2
        )
    )


def _log_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Binary log loss."""

    predicted = np.clip(
        predicted,
        1e-6,
        1.0 - 1e-6,
    )

    return float(
        -np.mean(
            actual
            * np.log(predicted)
            + (
                1 - actual
            )
            * np.log(
                1.0 - predicted
            )
        )
    )


def main() -> None:
    """Run the independent raw-percentile winner/loser audit."""

    print("=" * 78)
    print("RFS RAW PERCENTILE WINNER / LOSER CURVE AUDIT")
    print("=" * 78)
    print()
    print("Input:", INPUT_PATH)
    print()

    raw = pd.read_csv(
        INPUT_PATH
    )

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce",
    )

    fighters = _to_fighter_rows(
        raw
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fighters.to_csv(
        FIGHTER_OUTPUT_PATH,
        index=False,
    )

    bins = _raw_bins(
        fighters
    )

    bins.to_csv(
        BIN_OUTPUT_PATH,
        index=False,
    )

    summary_rows = []
    curve_rows = []

    for target in sorted(
        fighters["target"].unique()
    ):
        target_df = fighters.loc[
            fighters["target"]
            == target
        ].copy()

        family = str(
            target_df["family"].iloc[0]
        )

        train = target_df.loc[
            target_df["split"]
            == "TRAIN"
        ].copy()

        development = target_df.loc[
            target_df["split"]
            == "DEVELOPMENT"
        ].copy()

        if train.empty:
            continue

        train_auc = _auc(
            train["won"],
            train["percentile"],
        )

        development_auc = _auc(
            development["won"],
            development["percentile"],
        )

        direction = (
            1
            if (
                np.isfinite(train_auc)
                and train_auc >= 0.5
            )
            else -1
        )

        direction_label = (
            "HIGHER_HELPFUL"
            if direction == 1
            else "LOWER_HELPFUL"
        )

        train_oriented_auc = (
            train_auc
            if direction == 1
            else 1.0 - train_auc
        )

        development_oriented_auc = (
            development_auc
            if direction == 1
            else 1.0 - development_auc
        )

        curve = _fit_target_curve(
            train,
            direction=direction,
        )

        development_prediction = (
            _predict_curve(
                development["percentile"],
                direction=direction,
                curve=curve,
            )
            if not development.empty
            else np.array([])
        )

        development_actual = (
            development["won"]
            .to_numpy(dtype=float)
        )

        if len(development_actual):
            development_brier = _brier(
                development_actual,
                development_prediction,
            )

            development_log_loss = _log_loss(
                development_actual,
                development_prediction,
            )
        else:
            development_brier = np.nan
            development_log_loss = np.nan

        winners = train.loc[
            train["won"] == 1,
            "percentile",
        ]

        losers = train.loc[
            train["won"] == 0,
            "percentile",
        ]

        development_winners = development.loc[
            development["won"] == 1,
            "percentile",
        ]

        development_losers = development.loc[
            development["won"] == 0,
            "percentile",
        ]

        train_fights = train[
            "fight_id"
        ].nunique()

        development_fights = development[
            "fight_id"
        ].nunique()

        direction_consistent = bool(
            np.isfinite(
                development_oriented_auc
            )
            and development_oriented_auc
            >= 0.5
        )

        development_brier_gain = (
            0.25 - development_brier
            if np.isfinite(
                development_brier
            )
            else np.nan
        )

        curve_range = (
            float(
                curve[
                    "fitted_win_rate"
                ].max()
                - curve[
                    "fitted_win_rate"
                ].min()
            )
            if not curve.empty
            else np.nan
        )

        # A conservative diagnostic label only.
        curve_promising = bool(
            train_fights
            >= MIN_TRAIN_FIGHTS
            and development_fights
            >= MIN_DEVELOPMENT_FIGHTS
            and direction_consistent
            and np.isfinite(
                development_brier_gain
            )
            and development_brier_gain
            > 0.0
            and development_oriented_auc
            >= 0.52
        )

        summary_rows.append(
            {
                "family": family,
                "target": target,
                "train_fights": train_fights,
                "development_fights": development_fights,
                "train_auc": train_auc,
                "development_auc": development_auc,
                "direction": direction_label,
                "train_oriented_auc": train_oriented_auc,
                "development_oriented_auc": development_oriented_auc,
                "direction_consistent": direction_consistent,
                "train_winner_mean_percentile": float(
                    winners.mean()
                ),
                "train_loser_mean_percentile": float(
                    losers.mean()
                ),
                "train_winner_minus_loser": float(
                    winners.mean()
                    - losers.mean()
                ),
                "development_winner_mean_percentile": float(
                    development_winners.mean()
                ),
                "development_loser_mean_percentile": float(
                    development_losers.mean()
                ),
                "development_winner_minus_loser": float(
                    development_winners.mean()
                    - development_losers.mean()
                ),
                "curve_range": curve_range,
                "development_brier": development_brier,
                "development_brier_gain_vs_050": development_brier_gain,
                "development_log_loss": development_log_loss,
                "curve_promising": curve_promising,
            }
        )

        for row in curve.itertuples(
            index=False
        ):
            oriented_mid = float(
                row.mean_oriented_percentile
            )

            raw_mid = (
                oriented_mid
                if direction == 1
                else 1.0 - oriented_mid
            )

            curve_rows.append(
                {
                    "family": family,
                    "target": target,
                    "direction": direction_label,
                    "raw_percentile": raw_mid,
                    "oriented_percentile": oriented_mid,
                    "train_fighter_observations": int(
                        row.fighter_observations
                    ),
                    "train_fights": int(
                        row.fights
                    ),
                    "train_empirical_win_rate": float(
                        row.empirical_win_rate
                    ),
                    "train_fitted_win_rate": float(
                        row.fitted_win_rate
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        [
            "curve_promising",
            "development_brier_gain_vs_050",
            "development_oriented_auc",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    curves = pd.DataFrame(
        curve_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    curves.to_csv(
        CURVE_OUTPUT_PATH,
        index=False,
    )

    family_summary = (
        summary
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            targets=(
                "target",
                "count",
            ),
            promising_targets=(
                "curve_promising",
                "sum",
            ),
            direction_consistent_targets=(
                "direction_consistent",
                "sum",
            ),
            mean_train_oriented_auc=(
                "train_oriented_auc",
                "mean",
            ),
            mean_development_oriented_auc=(
                "development_oriented_auc",
                "mean",
            ),
            mean_development_brier=(
                "development_brier",
                "mean",
            ),
            mean_development_brier_gain_vs_050=(
                "development_brier_gain_vs_050",
                "mean",
            ),
        )
        .sort_values(
            "mean_development_brier_gain_vs_050",
            ascending=False,
        )
    )

    family_summary.to_csv(
        FAMILY_OUTPUT_PATH,
        index=False,
    )

    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    print(
        "Unique fights:",
        fighters["fight_id"].nunique(),
    )
    print(
        "Fighter-target observations:",
        len(fighters),
    )
    print(
        "Targets:",
        fighters["target"].nunique(),
    )
    print()

    print("=" * 78)
    print("TOP RAW-PERCENTILE CURVE CANDIDATES")
    print("=" * 78)

    columns = [
        "family",
        "target",
        "direction",
        "train_auc",
        "development_auc",
        "development_oriented_auc",
        "train_winner_minus_loser",
        "development_winner_minus_loser",
        "development_brier",
        "development_brier_gain_vs_050",
        "curve_promising",
    ]

    print(
        summary[
            columns
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("FAMILY SUMMARY")
    print("=" * 78)

    print(
        family_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(FIGHTER_OUTPUT_PATH)
    print(BIN_OUTPUT_PATH)
    print(CURVE_OUTPUT_PATH)
    print(SUMMARY_OUTPUT_PATH)
    print(FAMILY_OUTPUT_PATH)
    print()
    print("No simulator parameters were changed.")
    print("2025+ remains untouched.")
    print("RAW PERCENTILE WINNER / LOSER AUDIT COMPLETE")


if __name__ == "__main__":
    main()
