"""Raw-data source helpers for measurement-only FSR V3 active-trait audits.

The research audits must run in a clean checkout and therefore cannot depend on
locally generated/persisted FSR V2/V3 snapshot parquets.  These helpers rebuild
only the leakage-safe states required as comparators, directly from the frozen
raw round/master sources, without writing canonical artifacts.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v2.physical import build_physical_observations, build_physical_snapshots
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.power import replay_power_from_frames


@lru_cache(maxsize=1)
def raw_master() -> pd.DataFrame:
    frame = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def raw_rounds() -> pd.DataFrame:
    return pd.read_parquet(ROUND_STATS_PATH)


@lru_cache(maxsize=1)
def paired_rounds() -> pd.DataFrame:
    return build_paired_rounds(rounds=raw_rounds(), master=raw_master())


@lru_cache(maxsize=1)
def aggregate_fighter_fights() -> pd.DataFrame:
    return aggregate_fights(paired_rounds())


@lru_cache(maxsize=1)
def physical_observations() -> pd.DataFrame:
    frame = build_physical_observations(raw_rounds(), raw_master()).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["opponent_id"] = frame["opponent_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def legacy_physical_prefight() -> pd.DataFrame:
    frame = build_physical_snapshots(rounds=raw_rounds(), master=raw_master()).prefight.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def v3_power_prefight() -> pd.DataFrame:
    """Rebuild validated V3 POWER states aligned to raw physical observations."""
    obs = physical_observations().copy()
    keys = obs[["date", "fight_id", "fighter_id"]].rename(columns={"date": "event_date"})
    keys = keys.drop_duplicates(["event_date", "fight_id", "fighter_id"])
    history = replay_power_from_frames(keys, obs, config=FSRV3Config())
    return history[["event_date", "fight_id", "fighter_id", "pre_rating", "pre_posterior_sd", "validated_regime"]].rename(
        columns={"pre_rating": "striking_power_v3", "pre_posterior_sd": "striking_power_v3_sd"}
    )


@lru_cache(maxsize=1)
def legacy_submission_states() -> dict[str, pd.DataFrame]:
    """Rebuild the exact frozen V2 submission prefight state from raw rounds."""
    fights = aggregate_fighter_fights()
    engine = ReplayEngine()
    tendency = engine.replay(GROUPS["submission_tendency"], fights).history.copy()
    suppression = engine.replay(GROUPS["submission_suppression"], fights).history.copy()
    effectiveness = engine.replay(GROUPS["submission_effectiveness"], fights).history.copy()
    for frame in (tendency, suppression, effectiveness):
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)
        frame["opponent_id"] = frame["opponent_id"].astype(str)
    return {
        "tendency": tendency,
        "suppression": suppression,
        "effectiveness": effectiveness,
    }


def legacy_submission_attempt_prefight() -> pd.DataFrame:
    states = legacy_submission_states()
    tendency = states["tendency"]
    suppression = states["suppression"]
    t = tendency[tendency["trait"].eq("submission_tendency")][
        ["event_date", "fight_id", "fighter_id", "pre_rating"]
    ].rename(columns={"pre_rating": "submission_tendency"})
    s = suppression[suppression["trait"].eq("submission_suppression")][
        ["event_date", "fight_id", "fighter_id", "pre_rating"]
    ].rename(columns={"pre_rating": "submission_suppression"})
    return t.merge(s, on=["event_date", "fight_id", "fighter_id"], how="inner", validate="one_to_one")


def legacy_submission_conversion_prefight() -> pd.DataFrame:
    eff = legacy_submission_states()["effectiveness"].copy()
    offense = eff[eff["trait"].eq("submission_offense")][
        ["event_date", "fight_id", "fighter_id", "pre_rating", "population_baseline"]
    ].rename(columns={
        "pre_rating": "submission_offense",
        "population_baseline": "submission_conversion_baseline",
    })
    defense = eff[eff["trait"].eq("submission_defense")][
        ["event_date", "fight_id", "fighter_id", "pre_rating"]
    ].rename(columns={"pre_rating": "submission_defense"})
    return offense.merge(
        defense,
        on=["event_date", "fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
