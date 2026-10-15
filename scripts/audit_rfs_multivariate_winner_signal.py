"""Measure joint winner signal in the 37 RFS simulator targets.

Input
-----
The existing 750-fight raw-percentile audit:
    500 TRAIN fights
    250 DEVELOPMENT fights
    2025+ excluded

Each historical fight becomes one row:

    feature[target] =
        red_precompression_percentile
        - blue_precompression_percentile

    outcome =
        1 if Red won
        0 if Blue won

This diagnostic answers:
- Do the 37 RFS dimensions jointly predict winners?
- Which families carry joint signal?
- Which targets retain influence after controlling for the others?
- Which effects reverse once correlated features are considered?

This does NOT modify the Monte Carlo simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError(
        "This diagnostic requires scikit-learn."
    ) from exc


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

MODEL_OUTPUT = (
    OUTPUT_DIR
    / "historical_multivariate_model_summary_v1.csv"
)

COEFFICIENT_OUTPUT = (
    OUTPUT_DIR
    / "historical_multivariate_coefficients_v1.csv"
)

PREDICTION_OUTPUT = (
    OUTPUT_DIR
    / "historical_multivariate_development_predictions_v1.csv"
)

TUNING_OUTPUT = (
    OUTPUT_DIR
    / "historical_multivariate_tuning_v1.csv"
)


C_GRID = (
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
)


def _build_fight_matrix(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Create one fight row with 37 Red-minus-Blue percentile differences."""

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
            f"Missing required columns: {sorted(missing)}"
        )

    raw = raw.copy()

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce",
    )

    raw["diff"] = (
        pd.to_numeric(
            raw["red_raw_percentile"],
            errors="coerce",
        )
        - pd.to_numeric(
            raw["blue_raw_percentile"],
            errors="coerce",
        )
    )

    family_map = (
        raw[
            [
                "target",
                "family",
            ]
        ]
        .drop_duplicates()
        .set_index("target")["family"]
        .to_dict()
    )

    features = raw.pivot_table(
        index="fight_id",
        columns="target",
        values="diff",
        aggfunc="first",
    )

    metadata = (
        raw[
            [
                "fight_id",
                "date",
                "split",
                "red_win",
            ]
        ]
        .drop_duplicates(
            subset=["fight_id"]
        )
        .set_index("fight_id")
    )

    fights = metadata.join(
        features,
        how="inner",
    )

    fights = fights.loc[
        fights["split"].isin(
            [
                "TRAIN",
                "DEVELOPMENT",
            ]
        )
    ].copy()

    fights["red_win"] = pd.to_numeric(
        fights["red_win"],
        errors="raise",
    ).astype(int)

    return fights, family_map


def _metrics(
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Return winner-prediction metrics."""

    probability = np.asarray(
        probability,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=int,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                actual,
                prediction,
            )
        ),
        "brier": float(
            brier_score_loss(
                actual,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                actual,
                np.column_stack(
                    [
                        1.0 - probability,
                        probability,
                    ]
                ),
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "auc": float(
            roc_auc_score(
                actual,
                probability,
            )
        ),
    }


def _pipeline(
    c_value: float,
) -> Pipeline:
    """Regularized logistic regression with leakage-safe preprocessing."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(c_value),
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def _chronological_inner_split(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use early TRAIN for fitting and late TRAIN for C selection."""

    ordered = train.sort_values(
        [
            "date",
            "fight_id",
        ]
    ).copy()

    desired_index = max(
        1,
        int(
            len(ordered)
            * 0.70
        )
        - 1,
    )

    cutoff = ordered.iloc[
        desired_index
    ]["date"]

    inner_train = ordered.loc[
        ordered["date"] <= cutoff
    ].copy()

    inner_validation = ordered.loc[
        ordered["date"] > cutoff
    ].copy()

    if (
        len(inner_train) < 100
        or len(inner_validation) < 50
    ):
        split_index = int(
            len(ordered)
            * 0.70
        )

        inner_train = ordered.iloc[
            :split_index
        ].copy()

        inner_validation = ordered.iloc[
            split_index:
        ].copy()

    return (
        inner_train,
        inner_validation,
    )


def _fit_model(
    *,
    name: str,
    train: pd.DataFrame,
    development: pd.DataFrame,
    features: list[str],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
]:
    """Tune C chronologically, refit on all TRAIN, score DEVELOPMENT."""

    inner_train, inner_validation = (
        _chronological_inner_split(
            train
        )
    )

    tuning_rows: list[
        dict[str, object]
    ] = []

    best_c: float | None = None
    best_key: tuple[
        float,
        float,
    ] | None = None

    for c_value in C_GRID:
        model = _pipeline(
            c_value
        )

        model.fit(
            inner_train[features],
            inner_train["red_win"],
        )

        validation_probability = (
            model.predict_proba(
                inner_validation[
                    features
                ]
            )[:, 1]
        )

        metric = _metrics(
            inner_validation[
                "red_win"
            ].to_numpy(),
            validation_probability,
        )

        tuning_rows.append(
            {
                "model": name,
                "C": c_value,
                "inner_train_fights": len(
                    inner_train
                ),
                "inner_validation_fights": len(
                    inner_validation
                ),
                **metric,
            }
        )

        # Primary objective = Brier.
        # Log loss breaks ties.
        key = (
            metric["brier"],
            metric["log_loss"],
        )

        if (
            best_key is None
            or key < best_key
        ):
            best_key = key
            best_c = c_value

    if best_c is None:
        raise RuntimeError(
            f"{name}: no valid C selected"
        )

    final_model = _pipeline(
        best_c
    )

    final_model.fit(
        train[features],
        train["red_win"],
    )

    development_probability = (
        final_model.predict_proba(
            development[
                features
            ]
        )[:, 1]
    )

    development_metrics = _metrics(
        development[
            "red_win"
        ].to_numpy(),
        development_probability,
    )

    # Constant 50/50 baseline.
    neutral_probability = np.full(
        len(development),
        0.5,
    )

    neutral_metrics = _metrics(
        development[
            "red_win"
        ].to_numpy(),
        neutral_probability,
    )

    summary = {
        "model": name,
        "features": len(features),
        "selected_C": best_c,
        "train_fights": len(train),
        "development_fights": len(
            development
        ),
        **{
            f"development_{key}": value
            for key, value
            in development_metrics.items()
        },
        "brier_gain_vs_050": (
            neutral_metrics["brier"]
            - development_metrics[
                "brier"
            ]
        ),
        "log_loss_gain_vs_050": (
            neutral_metrics["log_loss"]
            - development_metrics[
                "log_loss"
            ]
        ),
    }

    imputer = final_model.named_steps[
        "imputer"
    ]

    scaler = final_model.named_steps[
        "scaler"
    ]

    logistic = final_model.named_steps[
        "model"
    ]

    # Coefficients are on standardized features, so their magnitudes
    # are comparable within this model.
    coefficients = pd.DataFrame(
        {
            "model": name,
            "target": features,
            "standardized_coefficient": (
                logistic.coef_[0]
            ),
        }
    )

    coefficients[
        "absolute_coefficient"
    ] = coefficients[
        "standardized_coefficient"
    ].abs()

    coefficients[
        "direction"
    ] = np.where(
        coefficients[
            "standardized_coefficient"
        ] > 0,
        "RED_HIGHER_HELPS",
        np.where(
            coefficients[
                "standardized_coefficient"
            ] < 0,
            "BLUE_HIGHER_HELPS",
            "ZERO",
        ),
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    predictions = (
        development
        .reset_index()
        [
            [
                "fight_id",
                "date",
                "red_win",
            ]
        ]
        .copy()
    )

    predictions[
        "model"
    ] = name

    predictions[
        "red_win_probability"
    ] = development_probability

    predictions[
        "predicted_red_win"
    ] = (
        development_probability
        >= 0.5
    ).astype(int)

    return (
        summary,
        coefficients,
        predictions,
        tuning_rows,
    )


def main() -> None:
    """Run multivariate winner-signal audit."""

    print("=" * 78)
    print("RFS MULTIVARIATE WINNER SIGNAL AUDIT")
    print("=" * 78)
    print()
    print("Input:", INPUT_PATH)
    print()

    raw = pd.read_csv(
        INPUT_PATH
    )

    fights, family_map = (
        _build_fight_matrix(
            raw
        )
    )

    targets = sorted(
        family_map
    )

    train = fights.loc[
        fights["split"]
        == "TRAIN"
    ].copy()

    development = fights.loc[
        fights["split"]
        == "DEVELOPMENT"
    ].copy()

    print(
        "TRAIN fights      :",
        len(train),
    )

    print(
        "DEVELOPMENT fights:",
        len(development),
    )

    print(
        "Targets           :",
        len(targets),
    )

    print(
        "TRAIN Red win rate:",
        f"{train['red_win'].mean():.2%}",
    )

    print(
        "DEV Red win rate  :",
        f"{development['red_win'].mean():.2%}",
    )

    print()

    model_sets: dict[
        str,
        list[str],
    ] = {
        "ALL_37": targets,
    }

    for family in sorted(
        set(
            family_map.values()
        )
    ):
        family_targets = [
            target
            for target in targets
            if family_map[target]
            == family
        ]

        model_sets[
            f"FAMILY__{family}"
        ] = family_targets

    summaries = []
    coefficient_frames = []
    prediction_frames = []
    tuning_rows = []

    for (
        model_name,
        features,
    ) in model_sets.items():
        print(
            f"Fitting {model_name} "
            f"({len(features)} features)..."
        )

        (
            summary,
            coefficients,
            predictions,
            tuning,
        ) = _fit_model(
            name=model_name,
            train=train,
            development=development,
            features=features,
        )

        summaries.append(
            summary
        )

        coefficient_frames.append(
            coefficients
        )

        prediction_frames.append(
            predictions
        )

        tuning_rows.extend(
            tuning
        )

    summary_df = pd.DataFrame(
        summaries
    ).sort_values(
        [
            "development_brier",
            "development_log_loss",
        ]
    )

    coefficient_df = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    prediction_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    tuning_df = pd.DataFrame(
        tuning_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        MODEL_OUTPUT,
        index=False,
    )

    coefficient_df.to_csv(
        COEFFICIENT_OUTPUT,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    tuning_df.to_csv(
        TUNING_OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("DEVELOPMENT MODEL PERFORMANCE")
    print("=" * 78)

    display_columns = [
        "model",
        "features",
        "selected_C",
        "development_accuracy",
        "development_auc",
        "development_brier",
        "development_log_loss",
        "brier_gain_vs_050",
        "log_loss_gain_vs_050",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("ALL-37 JOINT COEFFICIENTS")
    print("=" * 78)

    all_coefficients = (
        coefficient_df.loc[
            coefficient_df["model"]
            == "ALL_37"
        ]
        .copy()
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
    )

    all_coefficients[
        "family"
    ] = all_coefficients[
        "target"
    ].map(
        family_map
    )

    print(
        all_coefficients[
            [
                "family",
                "target",
                "standardized_coefficient",
                "absolute_coefficient",
                "direction",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(MODEL_OUTPUT)
    print(COEFFICIENT_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(TUNING_OUTPUT)
    print()
    print("No simulator parameters were changed.")
    print("2025+ remains untouched.")
    print("MULTIVARIATE WINNER SIGNAL AUDIT COMPLETE")


if __name__ == "__main__":
    main()
