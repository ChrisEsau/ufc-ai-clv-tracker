"""Build and publish the final Event Clock-active FSR V3 trait promotions.

This module deliberately layers only the two active-trait audit winners onto the
existing validated V3 publication:

* escape/retention: V2 semantics, validated eight-entry prior, mean-only;
* knockdown resistance: new native latent posterior, full epistemic variance.

Submission traits remain inherited from V2 because their V3 candidates did not
beat the inherited traits on holdout.  Compatibility-only V2 physical columns
are otherwise untouched.

KD-resistance uncertainty remains in its native history rather than the generic
prefight uncertainty table.  The generic table is consumed by the positive
Gamma path sampler; KD resistance is Normal and is sampled separately at the
canonical C detailed-physics boundary.
"""

from __future__ import annotations

import pandas as pd

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.fsr_v3.paths import (
    ESCAPE_HISTORY_PATH,
    FSR_V3_HISTORY_DIR,
    FSR_V3_LATEST_PATH,
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
    KD_RESISTANCE_HISTORY_PATH,
)
from pipeline.fsr_v3.publish import publish as publish_base_v3
from pipeline.fsr_v3.replay.escape import replay_escape
from pipeline.fsr_v3.replay.kd_resistance import replay_kd_resistance

KEYS = ["event_date", "fight_id", "fighter_id"]


def _write_history(frame: pd.DataFrame, path) -> None:
    FSR_V3_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build_active_trait_histories(
    config: ActiveTraitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or ActiveTraitConfig()
    paired = build_paired_rounds()
    escape = replay_escape(paired, config)
    kd = replay_kd_resistance(config)
    _write_history(escape, ESCAPE_HISTORY_PATH)
    _write_history(kd, KD_RESISTANCE_HISTORY_PATH)
    return escape, kd


def _escape_prefight(history: pd.DataFrame) -> pd.DataFrame:
    ratings = history.pivot_table(
        index=KEYS,
        columns="trait",
        values="pre_rating",
        aggfunc="first",
    ).reset_index()
    baseline = history.groupby(KEYS, as_index=False)[
        "population_duration_baseline_seconds"
    ].first().rename(
        columns={"population_duration_baseline_seconds": "escape_population_mean_seconds"}
    )
    return ratings.merge(baseline, on=KEYS, how="inner", validate="one_to_one")


def _escape_latest(history: pd.DataFrame) -> pd.DataFrame:
    ordered = history.sort_values(["event_date", "fight_id"])
    rows = []
    for trait in ("escape_offense", "escape_defense"):
        part = (
            ordered[ordered["trait"].eq(trait)]
            .groupby("fighter_id", as_index=False)
            .tail(1)[["fighter_id", "post_rating"]]
            .rename(columns={"post_rating": trait})
        )
        rows.append(part)
    latest = rows[0].merge(rows[1], on="fighter_id", how="outer", validate="one_to_one")
    baseline = float(ordered["latest_population_duration_baseline_seconds"].dropna().iloc[-1])
    latest["escape_population_mean_seconds"] = baseline
    return latest


def _kd_prefight(history: pd.DataFrame) -> pd.DataFrame:
    return history[KEYS + ["pre_rating"]].rename(
        columns={"pre_rating": "knockdown_resistance_v3"}
    )


def _kd_latest(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.sort_values(["event_date", "fight_id"])
        .groupby("fighter_id", as_index=False)
        .tail(1)[["fighter_id", "post_rating"]]
        .rename(columns={"post_rating": "knockdown_resistance_v3"})
    )


def _uncertainty_rows(history: pd.DataFrame) -> pd.DataFrame:
    columns = KEYS + [
        "trait",
        "pre_rating",
        "pre_posterior_sd",
        "variance_multiplier",
        "sampling_enabled",
        "posterior_family",
    ]
    return history[columns].copy().rename(
        columns={
            "pre_rating": "posterior_mean",
            "pre_posterior_sd": "posterior_sd",
        }
    )


def overlay_active_traits(
    escape: pd.DataFrame | None = None,
    kd: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    escape = pd.read_parquet(ESCAPE_HISTORY_PATH) if escape is None else escape.copy()
    kd = pd.read_parquet(KD_RESISTANCE_HISTORY_PATH) if kd is None else kd.copy()

    prefight = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    prefight["event_date"] = pd.to_datetime(prefight["event_date"], errors="raise").dt.normalize()
    prefight["fight_id"] = prefight["fight_id"].astype(str)
    prefight["fighter_id"] = prefight["fighter_id"].astype(str)
    prefight = prefight.drop(
        columns=[
            c for c in (
                "escape_offense",
                "escape_defense",
                "escape_population_mean_seconds",
                "knockdown_resistance_v3",
            ) if c in prefight.columns
        ]
    )
    prefight = prefight.merge(_escape_prefight(escape), on=KEYS, how="left", validate="one_to_one")
    prefight = prefight.merge(_kd_prefight(kd), on=KEYS, how="left", validate="one_to_one")
    required = [
        "escape_offense",
        "escape_defense",
        "escape_population_mean_seconds",
        "knockdown_resistance_v3",
    ]
    missing = [c for c in required if c not in prefight or prefight[c].isna().any()]
    if missing:
        raise RuntimeError(f"active V3 prefight overlay has missing fields: {missing}")

    latest = pd.read_parquet(FSR_V3_LATEST_PATH).copy()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    latest = latest.drop(columns=[c for c in required if c in latest.columns])
    latest = latest.merge(_escape_latest(escape), on="fighter_id", how="left", validate="one_to_one")
    latest = latest.merge(_kd_latest(kd), on="fighter_id", how="left", validate="one_to_one")
    missing = [c for c in required if c not in latest or latest[c].isna().any()]
    if missing:
        raise RuntimeError(f"active V3 latest overlay has missing fields: {missing}")

    # Escape is mean-only and can safely join the generic uncertainty table as
    # a non-sampled trait.  KD resistance remains in KD_RESISTANCE_HISTORY_PATH
    # because its Normal posterior is sampled by canonical C at the detailed
    # physics boundary, not by the positive-trait Gamma sampler.
    uncertainty = pd.read_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH).copy()
    key = KEYS + ["trait"]
    uncertainty = uncertainty[
        ~uncertainty["trait"].isin(["escape_offense", "escape_defense"])
    ].copy()
    uncertainty = pd.concat(
        [uncertainty, _uncertainty_rows(escape)],
        ignore_index=True,
    ).sort_values(key).reset_index(drop=True)
    if uncertainty.duplicated(key).any():
        raise RuntimeError("duplicate active V3 uncertainty rows")

    prefight = prefight.sort_values(KEYS).reset_index(drop=True)
    latest = latest.sort_values("fighter_id").reset_index(drop=True)
    prefight.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    latest.to_parquet(FSR_V3_LATEST_PATH, index=False)
    uncertainty.to_parquet(FSR_V3_PREFIGHT_UNCERTAINTY_PATH, index=False)
    return prefight, latest, uncertainty


def publish_canonical_active_v3(
    config: ActiveTraitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    escape, kd = build_active_trait_histories(config)
    publish_base_v3()
    return overlay_active_traits(escape, kd)


def main() -> None:
    prefight, latest, uncertainty = publish_canonical_active_v3()
    print(
        "FSR V3 active-trait publication complete: "
        f"prefight={len(prefight):,}, latest={len(latest):,}, uncertainty={len(uncertainty):,}"
    )
    print("escape prior entries=8; escape c=0; KD resistance rho=.005 sigma=.70 c=1")


if __name__ == "__main__":
    main()
