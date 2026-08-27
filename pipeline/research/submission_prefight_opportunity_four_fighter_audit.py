"""Decompose prefight submission opportunity for Brito, Carnelossi, Nolan, and Ziam.

Research-only. Production unchanged.
"""
from __future__ import annotations

import json
import pandas as pd

from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

TARGETS = [
    ("Jordan Leavitt", "Joanderson Brito", "Joanderson Brito"),
    ("Ketlen Souza", "Ariane Carnelossi", "Ariane Carnelossi"),
    ("Fares Ziam", "Tom Nolan", "Tom Nolan"),
    ("Fares Ziam", "Tom Nolan", "Fares Ziam"),
]


def main():
    fights = aggregate_fights(build_paired_rounds()).copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id", "opponent_id"):
        fights[c] = fights[c].astype(str)

    engine = ReplayEngine()
    tendency = engine.replay(GROUPS["submission_tendency"], fights).history.copy()
    suppression = engine.replay(GROUPS["submission_suppression"], fights).history.copy()
    tendency["event_date"] = pd.to_datetime(tendency["event_date"]).dt.normalize()
    suppression["event_date"] = pd.to_datetime(suppression["event_date"]).dt.normalize()
    for c in ("fighter_id", "fight_id"):
        tendency[c] = tendency[c].astype(str)
        suppression[c] = suppression[c].astype(str)

    snaps = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    snaps["event_date"] = pd.to_datetime(snaps["event_date"]).dt.normalize()
    for c in ("fight_id", "fighter_id"):
        snaps[c] = snaps[c].astype(str)

    rows = []
    for a, b, target_name in TARGETS:
        m = fights[
            fights["fighter_name"].astype(str).isin([a,b])
            & fights["opponent_name"].astype(str).isin([a,b])
            & fights["event_date"].dt.year.eq(2026)
        ].copy()
        if m.empty:
            raise RuntimeError(f"fight not found: {a} vs {b}")
        date = m["event_date"].max()
        m = m[m["event_date"].eq(date)]
        target = m[m["fighter_name"].astype(str).eq(target_name)].iloc[0]
        opp = m[m["fighter_id"].astype(str).eq(str(target.opponent_id))].iloc[0]

        th = tendency[(tendency["fight_id"].eq(str(target.fight_id))) & (tendency["fighter_id"].eq(str(target.fighter_id)))].iloc[0]
        sh = suppression[(suppression["fight_id"].eq(str(target.fight_id))) & (suppression["fighter_id"].eq(str(opp.fighter_id)))].iloc[0]

        snap = snaps[(snaps["fight_id"].eq(str(target.fight_id))) & (snaps["fighter_id"].eq(str(target.fighter_id)))].iloc[0]
        opp_snap = snaps[(snaps["fight_id"].eq(str(target.fight_id))) & (snaps["fighter_id"].eq(str(opp.fighter_id)))].iloc[0]

        raw_tendency = float(th.fighter_prior_attempts / th.fighter_prior_exposure_seconds) if float(th.fighter_prior_exposure_seconds) > 0 else None
        population_rate = float(th.population_prior_rate)
        prefight_tendency = float(th.pre_rating)

        prior_sup = suppression[(suppression["fighter_id"].eq(str(opp.fighter_id))) & (suppression["event_date"] < date)].copy()
        suppression_actual_hist = float(prior_sup["raw_numerator"].sum())
        suppression_expected_hist = float(prior_sup["raw_denominator"].sum())
        raw_suppression = suppression_actual_hist / suppression_expected_hist if suppression_expected_hist > 0 else None
        prefight_suppression = float(sh.pre_rating)

        matchup_rate = prefight_tendency * prefight_suppression
        scheduled_seconds = 900.0

        rows.append({
            "matchup": f"{a} vs {b}",
            "fighter": target_name,
            "opponent": str(opp.fighter_name),
            "fight_id": str(target.fight_id),
            "date": str(date.date()),
            "attacker_prior_effective_sub_attempts": float(th.fighter_prior_attempts),
            "attacker_prior_exposure_seconds": float(th.fighter_prior_exposure_seconds),
            "attacker_prior_fights": int(fights[(fights["fighter_id"].eq(str(target.fighter_id))) & (fights["event_date"] < date)]["fight_id"].nunique()),
            "attacker_raw_attempt_rate_per_15m": None if raw_tendency is None else raw_tendency * 900.0,
            "population_attempt_rate_per_15m": population_rate * 900.0,
            "attacker_prefight_tendency_per_15m": prefight_tendency * 900.0,
            "tendency_prior_seconds": float(th.population_prior_seconds),
            "defender_prior_actual_attempts_allowed": suppression_actual_hist,
            "defender_prior_expected_attempts_allowed": suppression_expected_hist,
            "defender_raw_suppression_ratio": raw_suppression,
            "defender_prefight_suppression_ratio": prefight_suppression,
            "suppression_prior_expected_attempts": float(sh.suppression_prior_expected_attempts),
            "matchup_attempt_rate_per_15m_before_global_scale": matchup_rate * 900.0,
            "expected_attempts_15m_before_global_scale": matchup_rate * scheduled_seconds,
            "snapshot_submission_tendency": float(snap.submission_tendency),
            "snapshot_opponent_submission_suppression": float(opp_snap.submission_suppression),
        })

    print(json.dumps({"study":"four-fighter prefight submission opportunity decomposition","production_changed":False,"rows":rows}, indent=2))
    print("\nTABLE")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
