"""Audit Last-3 opening-rate inflation caused by short round exposure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_PATH = Path(
    "data/features/round_fighter_state_history.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/"
    "rfs_opening_population_audit_2026-01-24.csv"
)

SNAPSHOT_DATE = pd.Timestamp("2026-01-24")
WINDOW_SIZE = 3

FIGHTER_ID = "fighter_id"
FIGHTER_NAME = "fighter_name"
DATE = "date"
OPPONENT_NAME = "opponent_name"

EXPOSURE = "rfs_open_fight_round1_exposure_seconds"
ATTEMPTED = "rfs_open_fight_round1_sig_attempted"
LANDED = "rfs_open_fight_round1_sig_landed"
KNOCKDOWNS = "rfs_open_fight_round1_kd"

CURRENT_ATTEMPT_RATE = (
    "rfs_open_last3_round1_sig_attempted_per_min"
)
CURRENT_LANDED_RATE = (
    "rfs_open_last3_round1_sig_landed_per_min"
)
CURRENT_KD_RATE = (
    "rfs_open_last3_round1_kd_per_min"
)
CURRENT_KD_PER_LANDED = (
    "rfs_open_last3_round1_kd_per_sig_landed"
)


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Divide safely and return NaN for invalid denominators."""
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0
    ):
        return np.nan

    return float(numerator / denominator)


def summarize_window(window: pd.DataFrame) -> dict[str, float]:
    """Calculate simple-average and exposure-pooled window metrics."""
    valid = window.loc[
        pd.to_numeric(
            window[EXPOSURE],
            errors="coerce",
        ).gt(0)
    ].copy()

    valid = valid.tail(WINDOW_SIZE)

    if valid.empty:
        return {
            "window_count": 0,
            "window_exposure_seconds": np.nan,
            "simple_attempt_rate": np.nan,
            "simple_landed_rate": np.nan,
            "simple_kd_rate": np.nan,
            "simple_kd_per_landed": np.nan,
            "pooled_attempt_rate": np.nan,
            "pooled_landed_rate": np.nan,
            "pooled_kd_rate": np.nan,
            "pooled_kd_per_landed": np.nan,
        }

    exposure = pd.to_numeric(
        valid[EXPOSURE],
        errors="coerce",
    )
    attempted = pd.to_numeric(
        valid[ATTEMPTED],
        errors="coerce",
    )
    landed = pd.to_numeric(
        valid[LANDED],
        errors="coerce",
    )
    knockdowns = pd.to_numeric(
        valid[KNOCKDOWNS],
        errors="coerce",
    )

    exposure_minutes = exposure / 60.0

    per_fight_attempt_rate = attempted / exposure_minutes
    per_fight_landed_rate = landed / exposure_minutes
    per_fight_kd_rate = knockdowns / exposure_minutes
    per_fight_kd_per_landed = knockdowns / landed.replace(
        0,
        np.nan,
    )

    total_exposure_minutes = float(exposure.sum() / 60.0)
    total_attempted = float(attempted.sum())
    total_landed = float(landed.sum())
    total_knockdowns = float(knockdowns.sum())

    return {
        "window_count": int(len(valid)),
        "window_exposure_seconds": float(exposure.sum()),
        "simple_attempt_rate": float(
            per_fight_attempt_rate.mean()
        ),
        "simple_landed_rate": float(
            per_fight_landed_rate.mean()
        ),
        "simple_kd_rate": float(
            per_fight_kd_rate.mean()
        ),
        "simple_kd_per_landed": float(
            per_fight_kd_per_landed.mean()
        ),
        "pooled_attempt_rate": safe_ratio(
            total_attempted,
            total_exposure_minutes,
        ),
        "pooled_landed_rate": safe_ratio(
            total_landed,
            total_exposure_minutes,
        ),
        "pooled_kd_rate": safe_ratio(
            total_knockdowns,
            total_exposure_minutes,
        ),
        "pooled_kd_per_landed": safe_ratio(
            total_knockdowns,
            total_landed,
        ),
    }


def build_row_audit(history: pd.DataFrame) -> pd.DataFrame:
    """Build both prior-only and row-inclusive Last-3 calculations."""
    records: list[dict[str, object]] = []

    for fighter_id, group in history.groupby(
        FIGHTER_ID,
        sort=False,
    ):
        group = group.sort_values(DATE).copy()

        for position, (row_index, row) in enumerate(
            group.iterrows()
        ):
            # Prior-only window excludes the fight represented by this row.
            prior_window = group.iloc[:position]

            # Inclusive window includes the fight represented by this row.
            inclusive_window = group.iloc[: position + 1]

            prior = summarize_window(prior_window)
            inclusive = summarize_window(inclusive_window)

            record: dict[str, object] = {
                "row_index": row_index,
                FIGHTER_ID: str(fighter_id),
                FIGHTER_NAME: row.get(FIGHTER_NAME),
                DATE: row.get(DATE),
                OPPONENT_NAME: row.get(OPPONENT_NAME),
                "stored_last3_attempt_rate": row.get(
                    CURRENT_ATTEMPT_RATE
                ),
                "stored_last3_landed_rate": row.get(
                    CURRENT_LANDED_RATE
                ),
                "stored_last3_kd_rate": row.get(
                    CURRENT_KD_RATE
                ),
                "stored_last3_kd_per_landed": row.get(
                    CURRENT_KD_PER_LANDED
                ),
            }

            for prefix, values in (
                ("prior", prior),
                ("inclusive", inclusive),
            ):
                for name, value in values.items():
                    record[f"{prefix}_{name}"] = value

            records.append(record)

    return pd.DataFrame(records)


def median_absolute_error(
    actual: pd.Series,
    expected: pd.Series,
) -> float:
    """Return median absolute error over valid paired values."""
    paired = pd.DataFrame(
        {
            "actual": pd.to_numeric(
                actual,
                errors="coerce",
            ),
            "expected": pd.to_numeric(
                expected,
                errors="coerce",
            ),
        }
    ).dropna()

    if paired.empty:
        return float("inf")

    return float(
        (paired["actual"] - paired["expected"])
        .abs()
        .median()
    )


def main() -> None:
    history = pd.read_parquet(HISTORY_PATH).copy()
    history[DATE] = pd.to_datetime(
        history[DATE],
        errors="coerce",
    )

    required = {
        FIGHTER_ID,
        FIGHTER_NAME,
        DATE,
        EXPOSURE,
        ATTEMPTED,
        LANDED,
        KNOCKDOWNS,
        CURRENT_ATTEMPT_RATE,
        CURRENT_LANDED_RATE,
        CURRENT_KD_RATE,
        CURRENT_KD_PER_LANDED,
    }

    missing = required - set(history.columns)
    if missing:
        raise RuntimeError(
            f"History is missing required columns: {sorted(missing)}"
        )

    audit = build_row_audit(history)

    prior_error = median_absolute_error(
        audit["stored_last3_attempt_rate"],
        audit["prior_simple_attempt_rate"],
    )
    inclusive_error = median_absolute_error(
        audit["stored_last3_attempt_rate"],
        audit["inclusive_simple_attempt_rate"],
    )

    alignment = (
        "prior"
        if prior_error <= inclusive_error
        else "inclusive"
    )

    print("=" * 88)
    print("LAST-3 ALIGNMENT CHECK")
    print("=" * 88)
    print(
        f"Prior-only median absolute error: {prior_error:.6f}"
    )
    print(
        "Row-inclusive median absolute error: "
        f"{inclusive_error:.6f}"
    )
    print(f"Detected Last-3 alignment: {alignment}")

    # Select the latest state row available before the snapshot date.
    snapshot = (
        audit.loc[audit[DATE].lt(SNAPSHOT_DATE)]
        .sort_values(DATE)
        .groupby(FIGHTER_ID, as_index=False)
        .tail(1)
        .copy()
    )

    selected_columns = {
        f"{alignment}_window_count": "window_count",
        f"{alignment}_window_exposure_seconds": (
            "window_exposure_seconds"
        ),
        f"{alignment}_simple_attempt_rate": (
            "simple_attempt_rate"
        ),
        f"{alignment}_simple_landed_rate": (
            "simple_landed_rate"
        ),
        f"{alignment}_simple_kd_rate": "simple_kd_rate",
        f"{alignment}_simple_kd_per_landed": (
            "simple_kd_per_landed"
        ),
        f"{alignment}_pooled_attempt_rate": (
            "pooled_attempt_rate"
        ),
        f"{alignment}_pooled_landed_rate": (
            "pooled_landed_rate"
        ),
        f"{alignment}_pooled_kd_rate": "pooled_kd_rate",
        f"{alignment}_pooled_kd_per_landed": (
            "pooled_kd_per_landed"
        ),
    }

    for source, destination in selected_columns.items():
        snapshot[destination] = snapshot[source]

    snapshot["attempt_rate_inflation"] = (
        snapshot["stored_last3_attempt_rate"]
        / snapshot["pooled_attempt_rate"]
    )
    snapshot["landed_rate_inflation"] = (
        snapshot["stored_last3_landed_rate"]
        / snapshot["pooled_landed_rate"]
    )
    snapshot["kd_rate_inflation"] = (
        snapshot["stored_last3_kd_rate"]
        / snapshot["pooled_kd_rate"]
    )
    snapshot["kd_per_landed_inflation"] = (
        snapshot["stored_last3_kd_per_landed"]
        / snapshot["pooled_kd_per_landed"]
    )

    snapshot["short_exposure_under_300"] = (
        snapshot["window_exposure_seconds"] < 300
    )
    snapshot["short_exposure_under_600"] = (
        snapshot["window_exposure_seconds"] < 600
    )
    snapshot["material_attempt_inflation"] = (
        snapshot["attempt_rate_inflation"] >= 1.50
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        FIGHTER_ID,
        FIGHTER_NAME,
        DATE,
        OPPONENT_NAME,
        "window_count",
        "window_exposure_seconds",
        "stored_last3_attempt_rate",
        "simple_attempt_rate",
        "pooled_attempt_rate",
        "attempt_rate_inflation",
        "stored_last3_landed_rate",
        "pooled_landed_rate",
        "landed_rate_inflation",
        "stored_last3_kd_rate",
        "pooled_kd_rate",
        "kd_rate_inflation",
        "stored_last3_kd_per_landed",
        "pooled_kd_per_landed",
        "kd_per_landed_inflation",
        "short_exposure_under_300",
        "short_exposure_under_600",
        "material_attempt_inflation",
    ]

    snapshot[output_columns].sort_values(
        "attempt_rate_inflation",
        ascending=False,
    ).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    valid = snapshot.loc[
        snapshot["window_count"].eq(WINDOW_SIZE)
        & np.isfinite(snapshot["attempt_rate_inflation"])
        & snapshot["pooled_attempt_rate"].gt(0)
    ].copy()

    print()
    print("=" * 88)
    print("POPULATION SUMMARY")
    print("=" * 88)
    print(f"Snapshot date: {SNAPSHOT_DATE.date()}")
    print(f"Fighters with three observations: {len(valid):,}")

    if not valid.empty:
        quantiles = valid["attempt_rate_inflation"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        )

        for quantile, value in quantiles.items():
            print(
                f"Inflation p{int(quantile * 100):02d}: "
                f"{value:.3f}x"
            )

        for threshold in (1.25, 1.50, 2.00, 3.00):
            count = int(
                valid["attempt_rate_inflation"]
                .ge(threshold)
                .sum()
            )
            percentage = count / len(valid)

            print(
                f"Inflation >= {threshold:.2f}x: "
                f"{count:,} fighters ({percentage:.1%})"
            )

        short_material = valid.loc[
            valid["window_exposure_seconds"].lt(600)
            & valid["attempt_rate_inflation"].ge(1.50)
        ]

        print(
            "Under 600 sec and >=1.50x inflation: "
            f"{len(short_material):,} fighters "
            f"({len(short_material) / len(valid):.1%})"
        )

    display_columns = [
        FIGHTER_NAME,
        DATE,
        "window_exposure_seconds",
        "stored_last3_attempt_rate",
        "pooled_attempt_rate",
        "attempt_rate_inflation",
        "stored_last3_landed_rate",
        "pooled_landed_rate",
    ]

    print()
    print("=" * 88)
    print("TOP 20 ATTEMPT-RATE INFLATION OUTLIERS")
    print("=" * 88)

    print(
        valid.sort_values(
            "attempt_rate_inflation",
            ascending=False,
        )[display_columns]
        .head(20)
        .to_string(index=False)
    )

    lewis = valid.loc[
        valid[FIGHTER_NAME]
        .astype(str)
        .str.contains(
            "Derrick Lewis",
            case=False,
            na=False,
        )
    ]

    print()
    print("=" * 88)
    print("DERRICK LEWIS")
    print("=" * 88)

    if lewis.empty:
        print("Derrick Lewis was not found in the valid snapshot.")
    else:
        print(
            lewis[display_columns].to_string(index=False)
        )

    print()
    print(f"Saved audit: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
