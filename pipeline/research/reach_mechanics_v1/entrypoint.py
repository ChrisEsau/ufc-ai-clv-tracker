from __future__ import annotations

"""CI-safe entrypoint for Reach Mechanics V1.

The canonical FSR V3 publication is intentionally not required on a clean
GitHub Actions checkout.  This wrapper reconstructs only the validated V3
prefight fields needed by the reach-mechanics study, entirely in memory from
frozen raw sources, writes a temporary research snapshot under /tmp, and then
runs the unchanged study.  It never writes data/fsr_v3/ or simulator outputs.

The reconstruction is truncated before the reserved 2024+ outer period, so the
outer holdout is neither used to build the temporary FSR state nor scored by the
study.
"""

from pathlib import Path

import pandas as pd
import yaml

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.ground import (
    build_ground_fighter_fights,
    replay_ground_suppression,
    replay_ground_tendency,
)
from pipeline.fsr_v3.replay.ground_effectiveness import replay_ground_effectiveness
from pipeline.fsr_v3.replay.paired_effectiveness import (
    build_effectiveness_fighter_fights,
    replay_paired_effectiveness,
    standing_effectiveness_spec,
    takedown_effectiveness_spec,
)
from pipeline.fsr_v3.replay.rate_families import (
    build_rate_fighter_fights,
    replay_suppression,
    replay_tendency,
    standing_spec,
    takedown_spec,
)
from pipeline.research.reach_mechanics_v1.run import DEFAULT_CONFIG, run

KEYS = ["event_date", "fight_id", "fighter_id"]
TMP_FSR_PATH = Path("/tmp/reach_mechanics_v1_fsr_prefight.parquet")
TMP_CONFIG_PATH = Path("/tmp/reach_mechanics_v1_runtime_config.yaml")


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="raise").dt.normalize()
    out["fight_id"] = out["fight_id"].astype(str)
    out["fighter_id"] = out["fighter_id"].astype(str)
    return out


def _single_rating(history: pd.DataFrame, name: str, extra: dict[str, str] | None = None) -> pd.DataFrame:
    extra = extra or {}
    cols = KEYS + ["pre_rating", *extra.keys()]
    out = _normalise_keys(history[cols]).rename(columns={"pre_rating": name, **extra})
    if out.duplicated(KEYS).any():
        raise ValueError(f"duplicate temporary FSR rows for {name}")
    return out


def _paired_rating(
    history: pd.DataFrame,
    offense_trait: str,
    defense_trait: str,
    offense_name: str,
    defense_name: str,
    baseline_name: str,
) -> pd.DataFrame:
    offense = history[history["trait"].eq(offense_trait)][
        KEYS + ["pre_rating", "population_baseline"]
    ].copy()
    defense = history[history["trait"].eq(defense_trait)][
        KEYS + ["pre_rating"]
    ].copy()
    offense = _normalise_keys(offense).rename(
        columns={"pre_rating": offense_name, "population_baseline": baseline_name}
    )
    defense = _normalise_keys(defense).rename(columns={"pre_rating": defense_name})
    if offense.duplicated(KEYS).any() or defense.duplicated(KEYS).any():
        raise ValueError(f"duplicate temporary paired FSR rows for {offense_trait}/{defense_trait}")
    return offense.merge(defense, on=KEYS, how="inner", validate="one_to_one")


def build_temporary_prefight(master: pd.DataFrame, outer_start: pd.Timestamp) -> pd.DataFrame:
    # The round parquet already owns event_date.  Drop any master copy before
    # pairing to avoid duplicate/suffixed date columns.
    pair_master = master.drop(columns=["event_date"], errors="ignore").copy()
    paired = build_paired_rounds(master=pair_master)
    paired["event_date"] = pd.to_datetime(paired["event_date"], errors="raise").dt.normalize()
    paired = paired[paired["event_date"].lt(outer_start)].copy()
    if paired.empty:
        raise RuntimeError("no pre-2024 paired round rows available for temporary FSR replay")

    cfg = FSRV3Config()

    td_rate_spec = takedown_spec(cfg)
    td_fights = build_rate_fighter_fights(td_rate_spec, paired_rounds=paired)
    td_tendency = replay_tendency(td_fights, td_rate_spec)
    td_suppression = replay_suppression(td_tendency, td_rate_spec)
    td_eff_spec = takedown_effectiveness_spec(cfg)
    td_eff_fights = build_effectiveness_fighter_fights(td_eff_spec, paired_rounds=paired)
    td_effectiveness = replay_paired_effectiveness(td_eff_fights, td_eff_spec)

    standing_rate_spec = standing_spec(cfg)
    standing_fights = build_rate_fighter_fights(standing_rate_spec, paired_rounds=paired)
    standing_tendency = replay_tendency(standing_fights, standing_rate_spec)
    standing_suppression = replay_suppression(standing_tendency, standing_rate_spec)
    standing_eff_spec = standing_effectiveness_spec(cfg)
    standing_eff_fights = build_effectiveness_fighter_fights(standing_eff_spec, paired_rounds=paired)
    standing_effectiveness = replay_paired_effectiveness(standing_eff_fights, standing_eff_spec)

    ground_fights = build_ground_fighter_fights(paired_rounds=paired)
    ground_tendency = replay_ground_tendency(ground_fights, cfg)
    ground_suppression = replay_ground_suppression(ground_tendency, cfg)
    ground_effectiveness = replay_ground_effectiveness(ground_fights, cfg)

    replacements = [
        _single_rating(td_tendency, "takedown_tendency"),
        _single_rating(td_suppression, "takedown_suppression"),
        _paired_rating(
            td_effectiveness,
            "takedown_offense",
            "takedown_defense",
            "takedown_offense",
            "takedown_defense",
            "takedown_completion_baseline",
        ),
        _single_rating(standing_tendency, "standing_striking_tendency"),
        _single_rating(standing_suppression, "standing_striking_suppression"),
        _paired_rating(
            standing_effectiveness,
            "standing_striking_offense",
            "standing_striking_defense",
            "standing_striking_offense",
            "standing_striking_defense",
            "standing_accuracy_baseline",
        ),
        _single_rating(
            ground_tendency,
            "ground_striking_tendency",
            {"population_burst": "ground_striking_burst_baseline"},
        ),
        _single_rating(ground_suppression, "ground_striking_suppression"),
        _single_rating(
            ground_effectiveness,
            "ground_striking_offense",
            {"population_baseline": "ground_accuracy_baseline"},
        ),
    ]

    out = replacements[0]
    for replacement in replacements[1:]:
        out = out.merge(replacement, on=KEYS, how="inner", validate="one_to_one")

    required = [
        "standing_striking_tendency",
        "standing_striking_suppression",
        "standing_striking_offense",
        "standing_striking_defense",
        "standing_accuracy_baseline",
        "takedown_tendency",
        "takedown_suppression",
        "takedown_offense",
        "takedown_defense",
        "takedown_completion_baseline",
        "ground_striking_tendency",
        "ground_striking_suppression",
        "ground_striking_offense",
        "ground_accuracy_baseline",
        "ground_striking_burst_baseline",
    ]
    missing = [c for c in required if c not in out.columns or out[c].isna().any()]
    if missing:
        raise ValueError(f"temporary FSR reconstruction has missing fields: {missing}")
    if out.duplicated(KEYS).any():
        raise ValueError("duplicate temporary FSR prefight rows")
    if not out["event_date"].lt(outer_start).all():
        raise AssertionError("reserved 2024+ rows entered temporary FSR reconstruction")

    out = out.sort_values(KEYS).reset_index(drop=True)
    print(
        "temporary research-only FSR V3 prefight state built: "
        f"rows={len(out):,} fights={out['fight_id'].nunique():,} "
        f"max_date={out['event_date'].max().date()}"
    )
    return out


def main() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    master = pd.read_parquet(config["inputs"]["master_path"])
    outer_start = pd.Timestamp(config["validation"]["outer_start"])

    temporary = build_temporary_prefight(master, outer_start)
    temporary.to_parquet(TMP_FSR_PATH, index=False)

    runtime_config = dict(config)
    runtime_config["inputs"] = dict(config["inputs"])
    runtime_config["inputs"]["fsr_v3_prefight_path"] = str(TMP_FSR_PATH)
    TMP_CONFIG_PATH.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")

    run(TMP_CONFIG_PATH)


if __name__ == "__main__":
    main()
