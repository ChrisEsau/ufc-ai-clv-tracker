# ============================================================
# pipeline/market/outcome_matcher.py
# ============================================================

"""Fight matching helpers for Market Pipeline V2.

The market pipeline uses provider names only as a bridge to canonical UFCStats
fight and fighter IDs. Once a provider row is matched, downstream artifacts
carry fight_id, red_fighter_id, blue_fighter_id, and outcome_fighter_id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufc_odds_utils import composite_name_score


MATCH_AUDIT_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "provider_market_key",
    "provider_fighter_1",
    "provider_fighter_2",
    "matched_fight_id",
    "event_id",
    "event_name",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "match_type",
    "odds_match_type",
    "odds_match_score",
    "odds_min_single_score",
    "fighter_1_score",
    "fighter_2_score",
    "is_matched",
]


def _score_provider_row_to_live_fight(provider_row: pd.Series, live_row: pd.Series) -> dict:
    """Score provider fighter order against a UFCStats live-card row."""

    fighter_1 = provider_row.get("fighter_1")
    fighter_2 = provider_row.get("fighter_2")
    red_fighter = live_row.get("red_fighter")
    blue_fighter = live_row.get("blue_fighter")

    same_f1_score = composite_name_score(fighter_1, red_fighter)
    same_f2_score = composite_name_score(fighter_2, blue_fighter)
    same_pair_score = (same_f1_score + same_f2_score) / 2
    same_min_score = min(same_f1_score, same_f2_score)

    rev_f1_score = composite_name_score(fighter_1, blue_fighter)
    rev_f2_score = composite_name_score(fighter_2, red_fighter)
    rev_pair_score = (rev_f1_score + rev_f2_score) / 2
    rev_min_score = min(rev_f1_score, rev_f2_score)

    if same_pair_score >= rev_pair_score:
        return {
            "live_row": live_row,
            "match_type": "same_order",
            "odds_match_score": same_pair_score,
            "odds_min_single_score": same_min_score,
            "fighter_1_score": same_f1_score,
            "fighter_2_score": same_f2_score,
            "fighter_1_side": "red",
            "fighter_2_side": "blue",
        }

    return {
        "live_row": live_row,
        "match_type": "reversed_order",
        "odds_match_score": rev_pair_score,
        "odds_min_single_score": rev_min_score,
        "fighter_1_score": rev_f1_score,
        "fighter_2_score": rev_f2_score,
        "fighter_1_side": "blue",
        "fighter_2_side": "red",
    }


def match_provider_row_to_live_card(
    provider_row: pd.Series,
    live_card_df: pd.DataFrame,
    min_single_score: float = 90,
) -> dict | None:
    """Return the best UFCStats live-card match for one provider odds row."""

    candidates = []

    for _, live_row in live_card_df.iterrows():
        candidates.append(_score_provider_row_to_live_fight(provider_row, live_row))

    if not candidates:
        return None

    best = sorted(candidates, key=lambda item: item["odds_match_score"], reverse=True)[0]

    if best["odds_min_single_score"] < min_single_score:
        return None

    live_row = best["live_row"]

    return {
        "fight_id": live_row.get("fight_id"),
        "event_id": live_row.get("event_id"),
        "event_name": live_row.get("event_name"),
        "red_fighter": live_row.get("red_fighter"),
        "blue_fighter": live_row.get("blue_fighter"),
        "red_fighter_id": live_row.get("red_fighter_id"),
        "blue_fighter_id": live_row.get("blue_fighter_id"),
        "match_type": best["match_type"],
        "odds_match_type": "matched",
        "odds_match_score": best["odds_match_score"],
        "odds_min_single_score": best["odds_min_single_score"],
        "fighter_1_score": best["fighter_1_score"],
        "fighter_2_score": best["fighter_2_score"],
        "fighter_1_side": best["fighter_1_side"],
        "fighter_2_side": best["fighter_2_side"],
    }


def build_match_audit_row(
    provider_row: pd.Series,
    match: dict | None,
    snapshot_run_id: str,
    snapshot_timestamp: str,
) -> dict:
    """Build one provider-fight matching audit row."""

    row = {
        "snapshot_run_id": snapshot_run_id,
        "snapshot_timestamp": snapshot_timestamp,
        "source": provider_row.get("source"),
        "bookmaker": provider_row.get("bookmaker"),
        "provider_event_id": provider_row.get("provider_event_id"),
        "provider_market_key": provider_row.get("provider_market_key"),
        "provider_fighter_1": provider_row.get("fighter_1"),
        "provider_fighter_2": provider_row.get("fighter_2"),
        "matched_fight_id": None,
        "event_id": None,
        "event_name": provider_row.get("event_name"),
        "red_fighter": None,
        "blue_fighter": None,
        "red_fighter_id": None,
        "blue_fighter_id": None,
        "match_type": None,
        "odds_match_type": "low_confidence",
        "odds_match_score": np.nan,
        "odds_min_single_score": np.nan,
        "fighter_1_score": np.nan,
        "fighter_2_score": np.nan,
        "is_matched": False,
    }

    if match is None:
        return row

    row.update({
        "matched_fight_id": match.get("fight_id"),
        "event_id": match.get("event_id"),
        "event_name": match.get("event_name"),
        "red_fighter": match.get("red_fighter"),
        "blue_fighter": match.get("blue_fighter"),
        "red_fighter_id": match.get("red_fighter_id"),
        "blue_fighter_id": match.get("blue_fighter_id"),
        "match_type": match.get("match_type"),
        "odds_match_type": match.get("odds_match_type"),
        "odds_match_score": match.get("odds_match_score"),
        "odds_min_single_score": match.get("odds_min_single_score"),
        "fighter_1_score": match.get("fighter_1_score"),
        "fighter_2_score": match.get("fighter_2_score"),
        "is_matched": True,
    })

    return row


def ensure_match_audit_columns(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable audit schema even when no rows are returned."""

    out = audit_df.copy()
    for column in MATCH_AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")

    return out[MATCH_AUDIT_COLUMNS]
