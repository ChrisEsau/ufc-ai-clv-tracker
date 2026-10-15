"""Build empirical RFS parameter compression curves from historical fights.

This script is diagnostic only. It does not modify simulator behavior.

For every eligible historical matchup it:

1. Loads the exact leakage-safe pre-fight RFS profiles.
2. Uses only population rows strictly before the fight date.
3. Resolves the same 37 parameters used by RFS MC V2.
4. Recovers each target's population-relative score BEFORE compression.
5. Measures whether the parameter advantage belonged to the winner.
6. Builds empirical win-rate curves as percentile gaps increase.
7. Fits a simple power-curve shape to each calibration family.

Chronological contract
----------------------
TRAIN:
    through 2022-12-31

DEVELOPMENT:
    2023-01-01 through 2024-12-31

HOLDOUT:
    2025+ is deliberately NOT analyzed.

The fitted curve shape comes from TRAIN only.
DEVELOPMENT is used only to check whether that shape generalizes.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.historical_matchup_loader import (
    HistoricalMatchupLoadError,
    load_historical_matchup,
)
from pipeline.simulation.rfs_mc_v2_shared_state.rfs_parameter_resolver import (
    ALL_TARGETS,
    DIRECT_RULES,
    RFSParameterResolutionError,
    _calibration_family,
    _direct_estimate,
    _latent_raw_score,
    _percentile,
    _population_series,
    _state_candidates,
    resolve_fighter_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

SHADOW_HISTORY_PATH = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state/"
    "historical_fighter_state.parquet"
)

CANONICAL_HISTORY_PATH = (
    REPO_ROOT
    / "data/features/round_fighter_state_history.parquet"
)

MASTER_PATH = (
    REPO_ROOT
    / "data/master/ufc_master.parquet"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "data/simulation/rfs_mc_v2_shared_state"
)

RAW_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_raw_percentiles_v1.csv"
)

ORIENTATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_orientation_v1.csv"
)

BIN_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_family_curve_bins_v1.csv"
)

FIT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_family_curve_fits_v1.csv"
)

PLOT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "historical_parameter_compression_curves_v1.png"
)


MIN_PRIOR_FIGHTS = 3

# Smoke-test limit. Set to None for the full historical audit.
TRAIN_FIGHT_LIMIT = 500
DEVELOPMENT_FIGHT_LIMIT = 250

TRAIN_END = pd.Timestamp("2022-12-31")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")

# A target whose TRAIN AUC is almost exactly 0.50 does not provide a
# sufficiently reliable direction for curve fitting.
MIN_DIRECTION_EDGE = 0.01

# Percentile-gap bins.
GAP_BIN_EDGES = np.array(
    [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.000001,
    ],
    dtype=float,
)


def _history_path() -> Path:
    """Prefer the simulator shadow artifact."""

    if SHADOW_HISTORY_PATH.exists():
        return SHADOW_HISTORY_PATH

    return CANONICAL_HISTORY_PATH


def _finite(value: Any) -> float | None:
    """Return value as finite float or None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _split_name(date: pd.Timestamp) -> str:
    """Return chronological audit split."""

    if date <= TRAIN_END:
        return "TRAIN"

    if date <= DEVELOPMENT_END:
        return "DEVELOPMENT"

    return "HOLDOUT"


def _auc(
    outcome: pd.Series,
    score: pd.Series,
) -> float:
    """Calculate ROC AUC using the Mann-Whitney rank statistic."""

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

    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())

    if (
        n_positive == 0
        or n_negative == 0
    ):
        return float("nan")

    ranks = frame["score"].rank(
        method="average"
    )

    positive_rank_sum = float(
        ranks.loc[positives].sum()
    )

    u = (
        positive_rank_sum
        - n_positive
        * (n_positive + 1)
        / 2.0
    )

    return float(
        u
        / (
            n_positive
            * n_negative
        )
    )


def _invoke_private(
    function: Callable[..., Any],
    *,
    target: str,
    profile: dict[str, object],
    prior_fight_count: int,
    population_history: pd.DataFrame,
) -> Any:
    """Invoke a resolver helper without hard-coding optional arguments.

    The private helpers are part of this shadow diagnostic's implementation
    contract. This adapter lets the diagnostic tolerate harmless signature
    differences while still failing loudly if a genuinely required argument
    is unknown.
    """

    available = {
        "target": target,
        "profile": profile,
        "prior_fight_count": prior_fight_count,

        # Resolver private helpers use the keyword "population".
        "population": population_history,

        # Keep aliases for diagnostic compatibility.
        "population_history": population_history,
        "history_df": population_history,
        "population_df": population_history,
    }

    signature = inspect.signature(
        function
    )

    kwargs: dict[str, object] = {}

    for name, parameter in signature.parameters.items():
        if name in available:
            kwargs[name] = available[name]
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            raise RuntimeError(
                f"Cannot invoke {function.__name__}: "
                f"unknown required argument {name!r}"
            )

    return function(
        **kwargs
    )


def _empirical_percentile(
    value: float,
    population: pd.Series,
) -> float | None:
    """Calculate percentile rank with midpoint treatment for ties."""

    values = pd.to_numeric(
        population,
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(values)
    ]

    if values.empty:
        return None

    array = values.to_numpy(
        dtype=float
    )

    below = int(
        np.sum(array < value)
    )

    equal = int(
        np.sum(
            np.isclose(
                array,
                value,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    )

    percentile = (
        below
        + 0.5 * equal
    ) / len(array)

    return float(
        np.clip(
            percentile,
            0.0,
            1.0,
        )
    )


def _direct_precompression_percentile(
    *,
    target: str,
    profile: dict[str, object],
    prior_fight_count: int,
    population_history: pd.DataFrame,
) -> float | None:
    """Recover the exact direct-target percentile before compression.

    This mirrors _normalized_direct_estimate():
        direct reliability shrinkage
        -> rank against leakage-safe physical population
        -> percentile BEFORE family compression.
    """

    base = _direct_estimate(
        target=target,
        profile=profile,
        population=population_history,
        prior_fight_count=prior_fight_count,
    )

    suffix, scale, lower, upper = DIRECT_RULES[target]

    pop, _ = _population_series(
        population_history,
        target,
        suffix,
    )

    scaled_population = (
        pd.to_numeric(
            pop,
            errors="coerce",
        )
        .dropna()
        * float(scale)
    )

    if lower is not None:
        scaled_population = scaled_population.clip(
            lower=lower
        )

    if upper is not None:
        scaled_population = scaled_population.clip(
            upper=upper
        )

    if scaled_population.empty:
        return None

    value = _finite(
        base.shrunk_estimate
    )

    if value is None:
        return None

    # This is intentionally identical to the resolver's percentile.
    percentile = float(
        (
            scaled_population
            <= value
        ).mean()
    )

    return float(
        np.clip(
            percentile,
            0.0,
            1.0,
        )
    )



def _latent_precompression_percentile(
    *,
    target: str,
    profile: dict[str, object],
    prior_fight_count: int,
    population_history: pd.DataFrame,
) -> float | None:
    """Return the raw latent population-relative score before compression."""

    result = _invoke_private(
        _latent_raw_score,
        target=target,
        profile=profile,
        prior_fight_count=prior_fight_count,
        population_history=population_history,
    )

    if isinstance(
        result,
        tuple,
    ):
        score = result[0]
    else:
        score = result

    score = _finite(
        score
    )

    if score is None:
        return None

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def _precompression_percentile(
    *,
    target: str,
    profile: dict[str, object],
    prior_fight_count: int,
    population_history: pd.DataFrame,
) -> float | None:
    """Resolve one target's population-relative value before compression."""

    if target in DIRECT_RULES:
        return _direct_precompression_percentile(
            target=target,
            profile=profile,
            prior_fight_count=prior_fight_count,
            population_history=population_history,
        )

    return _latent_precompression_percentile(
        target=target,
        profile=profile,
        prior_fight_count=prior_fight_count,
        population_history=population_history,
    )


def _target_orientations(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Learn parameter direction from TRAIN only."""

    train = raw_df.loc[
        raw_df["split"]
        == "TRAIN"
    ]

    rows: list[
        dict[str, object]
    ] = []

    for target, group in train.groupby(
        "target",
        sort=True,
    ):
        auc = _auc(
            group["red_win"],
            group["raw_gap"],
        )

        if not np.isfinite(
            auc
        ):
            direction = 0
            edge = np.nan
        else:
            edge = abs(
                auc - 0.5
            )

            direction = (
                1
                if auc >= 0.5
                else -1
            )

        rows.append(
            {
                "target": target,
                "family": group[
                    "family"
                ].iloc[0],
                "train_fights": len(group),
                "train_auc": auc,
                "train_predictive_edge": edge,
                "direction": direction,
                "direction_label": (
                    "HIGHER_HELPFUL"
                    if direction == 1
                    else (
                        "LOWER_HELPFUL"
                        if direction == -1
                        else "UNKNOWN"
                    )
                ),
                "curve_eligible": bool(
                    np.isfinite(edge)
                    and edge
                    >= MIN_DIRECTION_EDGE
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _add_oriented_outcome(
    raw_df: pd.DataFrame,
    orientation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach TRAIN-derived direction and favored-side result."""

    merged = raw_df.merge(
        orientation_df[
            [
                "target",
                "train_auc",
                "train_predictive_edge",
                "direction",
                "direction_label",
                "curve_eligible",
            ]
        ],
        on="target",
        how="left",
        validate="many_to_one",
    )

    merged[
        "oriented_gap"
    ] = (
        merged["raw_gap"]
        * merged["direction"]
    )

    merged[
        "gap_magnitude"
    ] = merged[
        "oriented_gap"
    ].abs()

    # After orientation, positive means Red owns the historically helpful
    # side of the parameter and negative means Blue owns it.
    favored_red = (
        merged["oriented_gap"]
        > 0
    )

    favored_blue = (
        merged["oriented_gap"]
        < 0
    )

    favored_won = np.where(
        favored_red,
        merged["red_win"],
        np.where(
            favored_blue,
            1 - merged["red_win"],
            np.nan,
        ),
    )

    merged[
        "favored_won"
    ] = favored_won

    return merged


def _curve_bins(
    oriented_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build empirical family win-rate curves."""

    usable = oriented_df.loc[
        oriented_df[
            "curve_eligible"
        ]
        & oriented_df[
            "favored_won"
        ].notna()
        & oriented_df[
            "gap_magnitude"
        ].notna()
    ].copy()

    usable[
        "gap_bin"
    ] = pd.cut(
        usable[
            "gap_magnitude"
        ],
        bins=GAP_BIN_EDGES,
        right=False,
        include_lowest=True,
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        split,
        family,
        gap_bin,
    ), group in usable.groupby(
        [
            "split",
            "family",
            "gap_bin",
        ],
        observed=True,
        sort=True,
    ):
        interval = gap_bin

        rows.append(
            {
                "split": split,
                "family": family,
                "gap_bin": str(
                    interval
                ),
                "gap_low": float(
                    interval.left
                ),
                "gap_high": float(
                    min(
                        interval.right,
                        1.0,
                    )
                ),
                "gap_midpoint": float(
                    (
                        interval.left
                        + min(
                            interval.right,
                            1.0,
                        )
                    )
                    / 2.0
                ),
                "observations": len(
                    group
                ),
                "targets": group[
                    "target"
                ].nunique(),
                "mean_gap": float(
                    group[
                        "gap_magnitude"
                    ].mean()
                ),
                "favored_win_rate": float(
                    group[
                        "favored_won"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _weighted_isotonic(
    y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Simple weighted PAVA for a non-decreasing empirical curve."""

    blocks: list[
        dict[str, float | int]
    ] = []

    for index, (
        value,
        weight,
    ) in enumerate(
        zip(
            y,
            weights,
            strict=True,
        )
    ):
        blocks.append(
            {
                "start": index,
                "end": index,
                "weight": float(weight),
                "value": float(value),
            }
        )

        while (
            len(blocks) >= 2
            and float(
                blocks[-2]["value"]
            )
            > float(
                blocks[-1]["value"]
            )
        ):
            right = blocks.pop()
            left = blocks.pop()

            combined_weight = (
                float(left["weight"])
                + float(right["weight"])
            )

            combined_value = (
                float(left["value"])
                * float(left["weight"])
                + float(right["value"])
                * float(right["weight"])
            ) / combined_weight

            blocks.append(
                {
                    "start": int(
                        left["start"]
                    ),
                    "end": int(
                        right["end"]
                    ),
                    "weight": combined_weight,
                    "value": combined_value,
                }
            )

    result = np.empty(
        len(y),
        dtype=float,
    )

    for block in blocks:
        result[
            int(block["start"]):
            int(block["end"]) + 1
        ] = float(
            block["value"]
        )

    return result


def _fit_power_curve(
    train_bins: pd.DataFrame,
) -> dict[str, float]:
    """Fit p(gap) = 0.5 + amplitude * gap ** gamma.

    gamma controls SHAPE.

    amplitude measures the historical win-probability edge at a hypothetical
    full 1.0 percentile gap. It is NOT the simulator compression cap.
    """

    work = train_bins.loc[
        train_bins["observations"]
        > 0
    ].copy()

    if len(work) < 3:
        return {
            "gamma": np.nan,
            "win_edge_amplitude": np.nan,
            "train_curve_rmse": np.nan,
        }

    x = work[
        "mean_gap"
    ].to_numpy(
        dtype=float
    )

    y = work[
        "favored_win_rate"
    ].to_numpy(
        dtype=float
    )

    weights = work[
        "observations"
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        x
    )

    x = x[order]
    y = y[order]
    weights = weights[order]

    # The theoretical edge at zero parameter difference is 50/50.
    anchor_weight = max(
        25.0,
        float(
            np.median(
                weights
            )
        ),
    )

    x_with_anchor = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            x,
        ]
    )

    y_with_anchor = np.concatenate(
        [
            np.array(
                [0.5]
            ),
            y,
        ]
    )

    w_with_anchor = np.concatenate(
        [
            np.array(
                [anchor_weight]
            ),
            weights,
        ]
    )

    isotonic_y = _weighted_isotonic(
        y_with_anchor,
        w_with_anchor,
    )

    # Drop the zero anchor for the power fit.
    x_fit = x_with_anchor[1:]
    y_fit = isotonic_y[1:]
    w_fit = w_with_anchor[1:]

    empirical_edge = np.maximum(
        y_fit - 0.5,
        0.0,
    )

    best_gamma = np.nan
    best_amplitude = np.nan
    best_error = np.inf

    for gamma in np.linspace(
        0.20,
        3.00,
        281,
    ):
        basis = np.power(
            x_fit,
            gamma,
        )

        denominator = float(
            np.sum(
                w_fit
                * basis
                * basis
            )
        )

        if denominator <= 0:
            continue

        amplitude = float(
            np.sum(
                w_fit
                * basis
                * empirical_edge
            )
            / denominator
        )

        amplitude = float(
            np.clip(
                amplitude,
                0.0,
                0.50,
            )
        )

        predicted = (
            0.5
            + amplitude
            * basis
        )

        mse = float(
            np.average(
                (
                    predicted
                    - y_fit
                )
                ** 2,
                weights=w_fit,
            )
        )

        if mse < best_error:
            best_error = mse
            best_gamma = float(
                gamma
            )
            best_amplitude = amplitude

    return {
        "gamma": best_gamma,
        "win_edge_amplitude": best_amplitude,
        "train_curve_rmse": float(
            np.sqrt(
                best_error
            )
        ),
    }


def _evaluate_curve(
    bins: pd.DataFrame,
    *,
    gamma: float,
    amplitude: float,
) -> float:
    """Weighted RMSE of TRAIN-fitted curve on another split."""

    if (
        not np.isfinite(gamma)
        or not np.isfinite(amplitude)
        or bins.empty
    ):
        return float(
            "nan"
        )

    work = bins.loc[
        bins["observations"]
        > 0
    ].copy()

    if work.empty:
        return float(
            "nan"
        )

    prediction = (
        0.5
        + amplitude
        * np.power(
            work[
                "mean_gap"
            ].to_numpy(
                dtype=float
            ),
            gamma,
        )
    )

    actual = work[
        "favored_win_rate"
    ].to_numpy(
        dtype=float
    )

    weights = work[
        "observations"
    ].to_numpy(
        dtype=float
    )

    return float(
        np.sqrt(
            np.average(
                (
                    prediction
                    - actual
                )
                ** 2,
                weights=weights,
            )
        )
    )


def _fit_family_curves(
    bins_df: pd.DataFrame,
    orientation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit family curve shapes using TRAIN only."""

    rows: list[
        dict[str, object]
    ] = []

    families = sorted(
        bins_df[
            "family"
        ].dropna().unique()
    )

    for family in families:
        train_bins = bins_df.loc[
            (
                bins_df[
                    "family"
                ]
                == family
            )
            & (
                bins_df[
                    "split"
                ]
                == "TRAIN"
            )
        ].copy()

        development_bins = bins_df.loc[
            (
                bins_df[
                    "family"
                ]
                == family
            )
            & (
                bins_df[
                    "split"
                ]
                == "DEVELOPMENT"
            )
        ].copy()

        fit = _fit_power_curve(
            train_bins
        )

        family_orientation = orientation_df.loc[
            orientation_df[
                "family"
            ]
            == family
        ]

        rows.append(
            {
                "family": family,
                "targets_total": len(
                    family_orientation
                ),
                "targets_curve_eligible": int(
                    family_orientation[
                        "curve_eligible"
                    ].sum()
                ),
                "train_observations": int(
                    train_bins[
                        "observations"
                    ].sum()
                ),
                "development_observations": int(
                    development_bins[
                        "observations"
                    ].sum()
                ),
                "gamma": fit[
                    "gamma"
                ],
                "win_edge_amplitude": fit[
                    "win_edge_amplitude"
                ],
                "train_curve_rmse": fit[
                    "train_curve_rmse"
                ],
                "development_curve_rmse": _evaluate_curve(
                    development_bins,
                    gamma=fit[
                        "gamma"
                    ],
                    amplitude=fit[
                        "win_edge_amplitude"
                    ],
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "development_curve_rmse",
        ascending=True,
    )


def _plot_curves(
    bins_df: pd.DataFrame,
    fits_df: pd.DataFrame,
) -> None:
    """Save one comparison figure of empirical family curves."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib unavailable; skipping PNG curve plot."
        )
        return

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    x_curve = np.linspace(
        0.0,
        1.0,
        200,
    )

    for fit in fits_df.itertuples(
        index=False
    ):
        gamma = float(
            fit.gamma
        )

        amplitude = float(
            fit.win_edge_amplitude
        )

        if (
            not np.isfinite(gamma)
            or not np.isfinite(amplitude)
        ):
            continue

        y_curve = (
            0.5
            + amplitude
            * np.power(
                x_curve,
                gamma,
            )
        )

        axis.plot(
            x_curve,
            y_curve,
            label=(
                f"{fit.family} "
                f"(gamma={gamma:.2f})"
            ),
        )

        development = bins_df.loc[
            (
                bins_df[
                    "family"
                ]
                == fit.family
            )
            & (
                bins_df[
                    "split"
                ]
                == "DEVELOPMENT"
            )
        ]

        if not development.empty:
            axis.scatter(
                development[
                    "mean_gap"
                ],
                development[
                    "favored_win_rate"
                ],
                s=25,
                alpha=0.6,
            )

    axis.axhline(
        0.50,
        linewidth=1,
        linestyle="--",
    )

    axis.set_xlabel(
        "Pre-compression population percentile gap"
    )

    axis.set_ylabel(
        "Historically favored fighter win probability"
    )

    axis.set_title(
        "RFS Parameter Family Compression Curves"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.40,
        0.75,
    )

    axis.legend(
        fontsize=8
    )

    figure.tight_layout()

    figure.savefig(
        PLOT_OUTPUT_PATH,
        dpi=160,
    )

    plt.close(
        figure
    )


def main() -> None:
    """Run compression-curve audit."""

    history_path = _history_path()

    print("=" * 78)
    print("RFS MC V2 PARAMETER COMPRESSION CURVE AUDIT")
    print("=" * 78)
    print()
    print("History :", history_path)
    print("Outcomes:", MASTER_PATH)
    print("Targets :", len(ALL_TARGETS))
    print("Train through       :", TRAIN_END.date())
    print("Development through :", DEVELOPMENT_END.date())
    print("2025+                : RESERVED HOLDOUT")
    print()

    history = pd.read_parquet(
        history_path
    ).copy()

    outcomes = pd.read_parquet(
        MASTER_PATH
    )

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )

    required = {
        "fight_id",
        "date",
        "rfs_traj_prior_fight_count",
    }

    missing = (
        required
        - set(
            history.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Historical RFS artifact missing required columns: "
            f"{sorted(missing)}"
        )

    fight_index = (
        history[
            [
                "fight_id",
                "date",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["fight_id"]
        )
        .sort_values(
            [
                "date",
                "fight_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    holdout_count = int(
        (
            fight_index[
                "date"
            ]
            > DEVELOPMENT_END
        ).sum()
    )

    fight_index = fight_index.loc[
        fight_index[
            "date"
        ]
        <= DEVELOPMENT_END
    ]

    rows: list[
        dict[str, object]
    ] = []

    skipped_loader = 0
    skipped_resolver = 0
    missing_parameter_observations = 0
    eligible_fights = 0

    eligible_by_split = {
        "TRAIN": 0,
        "DEVELOPMENT": 0,
    }

    # Fights are processed chronologically, so retain only the most recent
    # event-date population. Keeping every historical population would create
    # many increasingly large copies of the 686-column RFS history artifact.
    cached_population_date: pd.Timestamp | None = None
    cached_population: pd.DataFrame | None = None

    for record in fight_index.itertuples(
        index=False
    ):
        record_split = _split_name(
            pd.Timestamp(record.date)
        )

        if (
            record_split == "TRAIN"
            and eligible_by_split["TRAIN"] >= TRAIN_FIGHT_LIMIT
        ) or (
            record_split == "DEVELOPMENT"
            and eligible_by_split["DEVELOPMENT"] >= DEVELOPMENT_FIGHT_LIMIT
        ):
            continue

        fight_id = str(
            record.fight_id
        )

        try:
            matchup = load_historical_matchup(
                history,
                outcomes,
                fight_id,
                min_prior_fights=MIN_PRIOR_FIGHTS,
            )
        except (
            HistoricalMatchupLoadError,
            ValueError,
            KeyError,
        ):
            skipped_loader += 1
            continue

        red_id = str(
            matchup.red.fighter_id
        )

        blue_id = str(
            matchup.blue.fighter_id
        )

        winner_id = (
            None
            if matchup.actual.winner_id is None
            else str(
                matchup.actual.winner_id
            )
        )

        if winner_id not in {
            red_id,
            blue_id,
        }:
            continue

        date_key = pd.Timestamp(
            matchup.date
        )

        if (
            cached_population_date != date_key
            or cached_population is None
        ):
            population = history.loc[
                history["date"] < matchup.date
            ].copy()

            prior_counts = pd.to_numeric(
                population["rfs_traj_prior_fight_count"],
                errors="coerce",
            )

            population = population.loc[
                prior_counts > 0
            ].copy()

            cached_population_date = date_key
            cached_population = population

        population = cached_population

        try:
            # Resolve complete parameter bundles first. This ensures the
            # diagnostic only studies fights the simulator itself can resolve.
            red_parameters = resolve_fighter_parameters(
                profile=matchup.red.features,
                prior_fight_count=(
                    matchup.red.prior_fight_count
                ),
                population_history=population,
            )

            blue_parameters = resolve_fighter_parameters(
                profile=matchup.blue.features,
                prior_fight_count=(
                    matchup.blue.prior_fight_count
                ),
                population_history=population,
            )

        except (
            RFSParameterResolutionError,
            RuntimeError,
            ValueError,
            KeyError,
        ):
            skipped_resolver += 1
            continue

        red_win = int(
            winner_id
            == red_id
        )

        eligible_fights += 1
        eligible_by_split[record_split] += 1

        for target in sorted(
            ALL_TARGETS
        ):
            try:
                red_percentile = _precompression_percentile(
                    target=target,
                    profile=matchup.red.features,
                    prior_fight_count=(
                        matchup.red.prior_fight_count
                    ),
                    population_history=population,
                )

                blue_percentile = _precompression_percentile(
                    target=target,
                    profile=matchup.blue.features,
                    prior_fight_count=(
                        matchup.blue.prior_fight_count
                    ),
                    population_history=population,
                )

            except (
                RuntimeError,
                RFSParameterResolutionError,
                ValueError,
                KeyError,
            ):
                missing_parameter_observations += 1
                continue

            if (
                red_percentile is None
                or blue_percentile is None
            ):
                missing_parameter_observations += 1
                continue

            red_final_estimate = (
                red_parameters
                .estimates[
                    target
                ]
                .shrunk_estimate
            )

            blue_final_estimate = (
                blue_parameters
                .estimates[
                    target
                ]
                .shrunk_estimate
            )

            rows.append(
                {
                    "fight_id": matchup.fight_id,
                    "date": matchup.date,
                    "split": _split_name(
                        matchup.date
                    ),
                    "red_fighter": matchup.red.fighter_name,
                    "blue_fighter": matchup.blue.fighter_name,
                    "red_win": red_win,
                    "target": target,
                    "family": _calibration_family(
                        target
                    ),
                    "red_raw_percentile": red_percentile,
                    "blue_raw_percentile": blue_percentile,
                    "raw_gap": (
                        red_percentile
                        - blue_percentile
                    ),
                    "red_final_parameter": red_final_estimate,
                    "blue_final_parameter": blue_final_estimate,
                }
            )

        if (
            eligible_fights % 25
            == 0
        ):
            print(
                f"Eligible fights resolved: {eligible_fights:,} "
                f"(TRAIN={eligible_by_split['TRAIN']}, "
                f"DEVELOPMENT={eligible_by_split['DEVELOPMENT']})"
            )

        if (
            eligible_by_split["TRAIN"] >= TRAIN_FIGHT_LIMIT
            and eligible_by_split["DEVELOPMENT"] >= DEVELOPMENT_FIGHT_LIMIT
        ):
            print(
                "Smoke-test quotas reached: "
                f"TRAIN={eligible_by_split['TRAIN']}, "
                f"DEVELOPMENT={eligible_by_split['DEVELOPMENT']}"
            )
            break

    if not rows:
        raise RuntimeError(
            "No parameter observations were produced."
        )

    raw_df = pd.DataFrame(
        rows
    )

    orientation_df = _target_orientations(
        raw_df
    )

    oriented_df = _add_oriented_outcome(
        raw_df,
        orientation_df,
    )

    bins_df = _curve_bins(
        oriented_df
    )

    fits_df = _fit_family_curves(
        bins_df,
        orientation_df,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oriented_df.to_csv(
        RAW_OUTPUT_PATH,
        index=False,
    )

    orientation_df.to_csv(
        ORIENTATION_OUTPUT_PATH,
        index=False,
    )

    bins_df.to_csv(
        BIN_OUTPUT_PATH,
        index=False,
    )

    fits_df.to_csv(
        FIT_OUTPUT_PATH,
        index=False,
    )

    _plot_curves(
        bins_df,
        fits_df,
    )

    print()
    print("=" * 78)
    print("AUDIT COUNTS")
    print("=" * 78)
    print("Eligible fights              :", eligible_fights)
    print("Parameter observations       :", len(oriented_df))
    print("Skipped loader               :", skipped_loader)
    print("Skipped resolver             :", skipped_resolver)
    print(
        "Missing parameter observations:",
        missing_parameter_observations,
    )
    print("Reserved 2025+ fights        :", holdout_count)

    print()
    print("=" * 78)
    print("TARGET ORIENTATION — TRAIN ONLY")
    print("=" * 78)

    display_orientation = (
        orientation_df
        .sort_values(
            "train_predictive_edge",
            ascending=False,
        )
        [
            [
                "family",
                "target",
                "train_auc",
                "train_predictive_edge",
                "direction_label",
                "curve_eligible",
            ]
        ]
    )

    print(
        display_orientation.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("FAMILY CURVE FITS")
    print("=" * 78)

    print(
        fits_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Interpretation:")
    print("  gamma < 1.0 : effect rises quickly then saturates")
    print("  gamma = 1.0 : approximately linear")
    print("  gamma > 1.0 : small differences matter less; extremes matter more")
    print()
    print(
        "win_edge_amplitude is descriptive historical winner signal, "
        "NOT the simulator compression cap."
    )

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(RAW_OUTPUT_PATH)
    print(ORIENTATION_OUTPUT_PATH)
    print(BIN_OUTPUT_PATH)
    print(FIT_OUTPUT_PATH)

    if PLOT_OUTPUT_PATH.exists():
        print(PLOT_OUTPUT_PATH)

    print()
    print("2025+ outcomes were NOT analyzed.")
    print("COMPRESSION CURVE AUDIT COMPLETE")


if __name__ == "__main__":
    main()
