"""Publish FSR V3 as a safe overlay on the frozen FSR V2 snapshots.

V3 replaces only trait families that survived chronological validation:

- takedown tendency / suppression / paired effectiveness
- standing striking tendency / suppression / paired effectiveness
- ground striking tendency / suppression / attacker-only effectiveness
- attacker-only striking power (native KD-logit posterior mean)

Every untested family remains copied verbatim from frozen FSR V2.  Rebuilt V3
power replaces the old V2 ``striking_power`` field with ``striking_power_v3``
to prevent accidental scale mixing.  The rejected ``ground_striking_defense``
field is also explicitly removed.
"""

from __future__ import annotations

import pandas as pd

from pipeline.common.paths import FSR_V2_LATEST_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.paths import (
    FSR_V3_LATEST_PATH,
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
    POWER_HISTORY_PATH,
    STANDING_EFFECTIVENESS_HISTORY_PATH,
    STANDING_SUPPRESSION_HISTORY_PATH,
    STANDING_TENDENCY_HISTORY_PATH,
    TAKEDOWN_EFFECTIVENESS_HISTORY_PATH,
    TAKEDOWN_SUPPRESSION_HISTORY_PATH,
    TAKEDOWN_TENDENCY_HISTORY_PATH,
)

KEYS = ["event_date", "fight_id", "fighter_id"]

UPDATED_COLUMNS = [
    "takedown_tendency",
    "takedown_suppression",
    "takedown_offense",
    "takedown_defense",
    "takedown_completion_baseline",
    "standing_striking_tendency",
    "standing_striking_suppression",
    "standing_striking_offense",
    "standing_striking_defense",
    "standing_accuracy_baseline",
    "ground_striking_tendency",
    "ground_striking_suppression",
    "ground_striking_offense",
    "ground_accuracy_baseline",
    "ground_striking_burst_baseline",
    "ground_striking_population_slope_15m",
    "striking_power_v3",
]

REPLACED_OR_REJECTED_V2_COLUMNS = [
    "ground_striking_defense",
    "striking_power",
]

HISTORY_PATHS = (
    TAKEDOWN_TENDENCY_HISTORY_PATH,
    TAKEDOWN_SUPPRESSION_HISTORY_PATH,
    TAKEDOWN_EFFECTIVENESS_HISTORY_PATH,
    STANDING_TENDENCY_HISTORY_PATH,
    STANDING_SUPPRESSION_HISTORY_PATH,
    STANDING_EFFECTIVENESS_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    POWER_HISTORY_PATH,
)


def _read_history(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing FSR V3 history: {path}")
    frame = pd.read_parquet(path).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return frame


def _replacement(history, rating_name, extra_columns=None):
    columns = KEYS + ["pre_rating"] + list((extra_columns or {}).keys())
    selected = history[columns].copy().rename(
        columns={"pre_rating": rating_name, **(extra_columns or {})}
    )
    if selected.duplicated(KEYS).any():
        raise ValueError(f"duplicate V3 replacement rows for {rating_name}")
    return selected


def _paired_replacement(
    history: pd.DataFrame,
    offense_trait: str,
    defense_trait: str,
    offense_column: str,
    defense_column: str,
    baseline_column: str,
) -> pd.DataFrame:
    offense = history[history["trait"] == offense_trait][
        KEYS + ["pre_rating", "population_baseline"]
    ].copy()
    defense = history[history["trait"] == defense_trait][
        KEYS + ["pre_rating"]
    ].copy()
    offense = offense.rename(
        columns={
            "pre_rating": offense_column,
            "population_baseline": baseline_column,
        }
    )
    defense = defense.rename(columns={"pre_rating": defense_column})
    if offense.duplicated(KEYS).any() or defense.duplicated(KEYS).any():
        raise ValueError(f"duplicate paired V3 rows for {offense_trait}/{defense_trait}")
    return offense.merge(defense, on=KEYS, how="inner", validate="one_to_one")


def _uncertainty_frame(history: pd.DataFrame) -> pd.DataFrame:
    columns = KEYS + [
        "trait",
        "pre_rating",
        "pre_posterior_sd",
        "variance_multiplier",
        "sampling_enabled",
    ]
    u = history[columns].copy().rename(
        columns={
            "pre_rating": "posterior_mean",
            "pre_posterior_sd": "posterior_sd",
        }
    )
    if "posterior_family" in history.columns:
        u["posterior_family"] = history["posterior_family"].values
    else:
        positive = history["trait"].str.contains("tendency|suppression", regex=True)
        u["posterior_family"] = positive.map({True: "positive_grid", False: "normal_grid"})
    return u


def assemble_prefight() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FSR_V2_PREFIGHT_SNAPSHOTS_PATH.is_file():
        raise FileNotFoundError(
            "FSR V3 publication requires frozen FSR V2 prefight snapshots: "
            f"{FSR_V2_PREFIGHT_SNAPSHOTS_PATH}"
        )

    base = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()
    base["event_date"] = pd.to_datetime(base["event_date"], errors="raise").dt.normalize()
    base["fight_id"] = base["fight_id"].astype(str)
    base["fighter_id"] = base["fighter_id"].astype(str)

    td_tendency = _read_history(TAKEDOWN_TENDENCY_HISTORY_PATH)
    td_suppression = _read_history(TAKEDOWN_SUPPRESSION_HISTORY_PATH)
    td_effectiveness = _read_history(TAKEDOWN_EFFECTIVENESS_HISTORY_PATH)
    standing_tendency = _read_history(STANDING_TENDENCY_HISTORY_PATH)
    standing_suppression = _read_history(STANDING_SUPPRESSION_HISTORY_PATH)
    standing_effectiveness = _read_history(STANDING_EFFECTIVENESS_HISTORY_PATH)
    ground_tendency = _read_history(GROUND_TENDENCY_HISTORY_PATH)
    ground_suppression = _read_history(GROUND_SUPPRESSION_HISTORY_PATH)
    ground_effectiveness = _read_history(GROUND_EFFECTIVENESS_HISTORY_PATH)
    power = _read_history(POWER_HISTORY_PATH)

    # Remove old V2 fields that V3 redefines or rejects.
    drop = UPDATED_COLUMNS + REPLACED_OR_REJECTED_V2_COLUMNS
    base = base.drop(columns=[column for column in drop if column in base.columns])

    replacements = [
        _replacement(td_tendency, "takedown_tendency"),
        _replacement(td_suppression, "takedown_suppression"),
        _paired_replacement(
            td_effectiveness,
            "takedown_offense",
            "takedown_defense",
            "takedown_offense",
            "takedown_defense",
            "takedown_completion_baseline",
        ),
        _replacement(standing_tendency, "standing_striking_tendency"),
        _replacement(standing_suppression, "standing_striking_suppression"),
        _paired_replacement(
            standing_effectiveness,
            "standing_striking_offense",
            "standing_striking_defense",
            "standing_striking_offense",
            "standing_striking_defense",
            "standing_accuracy_baseline",
        ),
        _replacement(
            ground_tendency,
            "ground_striking_tendency",
            {
                "population_burst": "ground_striking_burst_baseline",
                "population_rate_15m": "ground_striking_population_slope_15m",
            },
        ),
        _replacement(ground_suppression, "ground_striking_suppression"),
        _replacement(
            ground_effectiveness,
            "ground_striking_offense",
            {"population_baseline": "ground_accuracy_baseline"},
        ),
        _replacement(power, "striking_power_v3"),
    ]

    out = base
    for replacement in replacements:
        out = out.merge(replacement, on=KEYS, how="left", validate="one_to_one")

    missing = [
        name for name in UPDATED_COLUMNS
        if name not in out.columns or out[name].isna().any()
    ]
    if missing:
        raise ValueError(f"FSR V3 overlay has missing validated fields: {missing}")
    if "ground_striking_defense" in out.columns:
        raise AssertionError("rejected ground_striking_defense leaked into FSR V3")
    if "striking_power" in out.columns:
        raise AssertionError("frozen V2 striking_power leaked into rebuilt FSR V3")

    histories = (
        td_tendency,
        td_suppression,
        td_effectiveness,
        standing_tendency,
        standing_suppression,
        standing_effectiveness,
        ground_tendency,
        ground_suppression,
        ground_effectiveness,
        power,
    )
    uncertainty = pd.concat(
        [_uncertainty_frame(history) for history in histories],
        ignore_index=True,
    )
    uncertainty = uncertainty.sort_values(KEYS + ["trait"]).reset_index(drop=True)
    if uncertainty[KEYS + ["trait"]].duplicated().any():
        raise ValueError("duplicate FSR V3 uncertainty rows")

    return (
        out.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True),
        uncertainty,
    )


def _latest_post_rating(
    history: pd.DataFrame,
    rating_name: str,
    trait: str | None = None,
) -> pd.DataFrame:
    """Return each fighter's state after their most recent observed fight."""
    selected = history if trait is None else history[history["trait"] == trait]
    selected = (
        selected.sort_values(["event_date", "fight_id"])
        .groupby("fighter_id", as_index=False)
        .tail(1)[["fighter_id", "post_rating"]]
        .rename(columns={"post_rating": rating_name})
    )
    if selected["fighter_id"].duplicated().any():
        raise ValueError(f"duplicate latest post-fight state for {rating_name}")
    return selected


def _final_global_value(history: pd.DataFrame, column: str) -> float:
    ordered = history.sort_values(["event_date", "fight_id"])
    if ordered.empty:
        raise ValueError(f"cannot publish final global value from empty {column} history")
    return float(ordered.iloc[-1][column])


def assemble_latest(prefight: pd.DataFrame) -> pd.DataFrame:
    """Overlay true post-most-recent-fight V3 states onto frozen V2 latest.

    Historical prefight snapshots intentionally remain pre-fight for leakage-safe
    replay.  The operational latest table is different: it must include each
    fighter's most recently observed fight.  Therefore validated V3 traits are
    taken from the final ``post_rating`` row for that fighter, not from the last
    prefight snapshot.
    """
    if not FSR_V2_LATEST_PATH.is_file():
        raise FileNotFoundError(f"missing frozen FSR V2 latest profiles: {FSR_V2_LATEST_PATH}")
    base = pd.read_parquet(FSR_V2_LATEST_PATH).copy()
    base["fighter_id"] = base["fighter_id"].astype(str)

    td_tendency = _read_history(TAKEDOWN_TENDENCY_HISTORY_PATH)
    td_suppression = _read_history(TAKEDOWN_SUPPRESSION_HISTORY_PATH)
    td_effectiveness = _read_history(TAKEDOWN_EFFECTIVENESS_HISTORY_PATH)
    standing_tendency = _read_history(STANDING_TENDENCY_HISTORY_PATH)
    standing_suppression = _read_history(STANDING_SUPPRESSION_HISTORY_PATH)
    standing_effectiveness = _read_history(STANDING_EFFECTIVENESS_HISTORY_PATH)
    ground_tendency = _read_history(GROUND_TENDENCY_HISTORY_PATH)
    ground_suppression = _read_history(GROUND_SUPPRESSION_HISTORY_PATH)
    ground_effectiveness = _read_history(GROUND_EFFECTIVENESS_HISTORY_PATH)
    power = _read_history(POWER_HISTORY_PATH)

    latest_v3 = _latest_post_rating(td_tendency, "takedown_tendency")
    for replacement in (
        _latest_post_rating(td_suppression, "takedown_suppression"),
        _latest_post_rating(td_effectiveness, "takedown_offense", "takedown_offense"),
        _latest_post_rating(td_effectiveness, "takedown_defense", "takedown_defense"),
        _latest_post_rating(standing_tendency, "standing_striking_tendency"),
        _latest_post_rating(standing_suppression, "standing_striking_suppression"),
        _latest_post_rating(
            standing_effectiveness,
            "standing_striking_offense",
            "standing_striking_offense",
        ),
        _latest_post_rating(
            standing_effectiveness,
            "standing_striking_defense",
            "standing_striking_defense",
        ),
        _latest_post_rating(ground_tendency, "ground_striking_tendency"),
        _latest_post_rating(ground_suppression, "ground_striking_suppression"),
        _latest_post_rating(ground_effectiveness, "ground_striking_offense"),
        _latest_post_rating(power, "striking_power_v3"),
    ):
        latest_v3 = latest_v3.merge(
            replacement,
            on="fighter_id",
            how="outer",
            validate="one_to_one",
        )

    # Global baselines are population quantities, so publish the final
    # chronological population value consistently for every fighter rather than
    # freezing each fighter to the baseline from their own last appearance.
    latest_v3["takedown_completion_baseline"] = _final_global_value(
        td_effectiveness,
        "population_baseline",
    )
    latest_v3["standing_accuracy_baseline"] = _final_global_value(
        standing_effectiveness,
        "population_baseline",
    )
    latest_v3["ground_accuracy_baseline"] = _final_global_value(
        ground_effectiveness,
        "population_baseline",
    )
    latest_v3["ground_striking_burst_baseline"] = _final_global_value(
        ground_tendency,
        "population_burst",
    )
    latest_v3["ground_striking_population_slope_15m"] = _final_global_value(
        ground_tendency,
        "population_rate_15m",
    )

    base = base.drop(
        columns=[
            column
            for column in UPDATED_COLUMNS + REPLACED_OR_REJECTED_V2_COLUMNS
            if column in base.columns
        ]
    )
    latest = base.merge(latest_v3, on="fighter_id", how="left", validate="one_to_one")
    missing = [
        name for name in UPDATED_COLUMNS
        if name not in latest.columns or latest[name].isna().any()
    ]
    if missing:
        raise ValueError(f"FSR V3 latest overlay has missing validated fields: {missing}")
    if "ground_striking_defense" in latest.columns:
        raise AssertionError("rejected ground_striking_defense leaked into FSR V3 latest")
    if "striking_power" in latest.columns:
        raise AssertionError("frozen V2 striking_power leaked into rebuilt FSR V3 latest")
    return latest


def publish() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in HISTORY_PATHS:
        if not path.is_file():
            raise FileNotFoundError(
                f"canonical FSR V3 publication requires all validated histories; missing {path}"
            )
    prefight, uncertainty = assemble_prefight()
    latest = assemble_latest(prefight)
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH.parent.joinpath("history").mkdir(parents=True, exist_ok=True)
    prefight.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    uncertainty.to_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH, index=False)
    latest.to_parquet(FSR_V3_LATEST_PATH, index=False)
    return prefight, latest, uncertainty


def main() -> None:
    prefight, latest, uncertainty = publish()
    print(
        f"published canonical FSR V3 overlay: prefight={len(prefight):,}, "
        f"latest={len(latest):,}, uncertainty={len(uncertainty):,}"
    )


if __name__ == "__main__":
    main()
