
"""
ufc_clv_utils.py

Shared CLV utility layer for UFC betting pipeline.

Responsibilities:
- official bet ledger helpers
- market snapshot storage
- closing line extraction
- CLV calculation
"""

import os
import hashlib
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# OFFICIAL BET LEDGER HELPERS
# ============================================================

def make_bet_id(row):
    """
    Stable bet ID for one official model bet.
    """
    raw = "|".join([
        str(row.get("event_name", "")),
        str(row.get("fight_id", "")),
        str(row.get("best_side_fighter_id", "")),
        str(row.get("bookmaker", "")),
        str(row.get("best_american_odds", "")),
    ])

    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def build_official_bets_from_action_board(action_board_df):
    """
    Extract official bets from action board and format them for ledger storage.
    """
    official_bets_df = action_board_df[
        action_board_df["is_official_bet"] == True
    ].copy()

    official_bets_df["bet_logged_at"] = datetime.now(timezone.utc).isoformat()
    official_bets_df["bet_id"] = official_bets_df.apply(make_bet_id, axis=1)

    official_bet_cols = [
        "bet_id",
        "snapshot_run_id",
        "bet_logged_at",

        "event_name",
        "commence_time",
        "fight_id",

        "red_fighter_id",
        "blue_fighter_id",
        "red_fighter",
        "blue_fighter",

        "best_side",
        "best_side_fighter_id",
        "best_side_opponent_id",

        "bookmaker",
        "best_american_odds",
        "best_implied_prob",

        "best_prob",
        "best_edge",
        "best_ev",
        "best_confidence",
        "recommended_stake",

        "passes_data_quality_filter",
        "nonzero_feature_count",
        "zero_feature_pct",

        "red_feature_match",
        "blue_feature_match",
        "odds_match_type",
        "odds_match_score",
    ]

    available_cols = [
        col for col in official_bet_cols
        if col in official_bets_df.columns
    ]

    return official_bets_df[available_cols].copy()


def append_official_bets_log(official_bets_df, output_path):
    """
    Append official bets to CSV ledger and dedupe by bet_id.
    """
    if os.path.exists(output_path):
        old_log = pd.read_csv(output_path)

        combined_log = pd.concat(
            [old_log, official_bets_df],
            ignore_index=True
        )

        combined_log = combined_log.drop_duplicates(
            subset=["bet_id"],
            keep="last"
        )
    else:
        combined_log = official_bets_df.copy()

    combined_log.to_csv(output_path, index=False)

    return combined_log


# ============================================================
# MARKET SNAPSHOT HELPERS
# ============================================================

def build_moneyline_market_snapshot_from_matched_card(
    matched_live_df,
    snapshot_timestamp,
):
    """
    Build one market snapshot row per fighter side from a matched live card.
    Requires attach_h2h_odds_to_live_df() to have already run.
    """

    matched_live_df = matched_live_df.copy()

    if "fight_id" not in matched_live_df.columns:
        matched_live_df["fight_id"] = (
            matched_live_df["red_fighter_id"].astype(str)
            + "_vs_"
            + matched_live_df["blue_fighter_id"].astype(str)
        )

    red_rows = pd.DataFrame({
        "snapshot_timestamp": snapshot_timestamp,
        "event_name": matched_live_df["event_name"],
        "commence_time": matched_live_df["commence_time"],
        "fight_id": matched_live_df["fight_id"],

        "fighter_id": matched_live_df["red_fighter_id"],
        "fighter_name": matched_live_df["red_fighter"],

        "opponent_id": matched_live_df["blue_fighter_id"],
        "opponent_name": matched_live_df["blue_fighter"],

        "sportsbook": matched_live_df["bookmaker"],
        "market_type": "moneyline",

        "american_odds": matched_live_df["red_american_odds"],
        "implied_prob": matched_live_df["red_implied_prob"],

        "odds_match_type": matched_live_df["odds_match_type"],
        "odds_match_score": matched_live_df["odds_match_score"],
    })

    blue_rows = pd.DataFrame({
        "snapshot_timestamp": snapshot_timestamp,
        "event_name": matched_live_df["event_name"],
        "commence_time": matched_live_df["commence_time"],
        "fight_id": matched_live_df["fight_id"],

        "fighter_id": matched_live_df["blue_fighter_id"],
        "fighter_name": matched_live_df["blue_fighter"],

        "opponent_id": matched_live_df["red_fighter_id"],
        "opponent_name": matched_live_df["red_fighter"],

        "sportsbook": matched_live_df["bookmaker"],
        "market_type": "moneyline",

        "american_odds": matched_live_df["blue_american_odds"],
        "implied_prob": matched_live_df["blue_implied_prob"],

        "odds_match_type": matched_live_df["odds_match_type"],
        "odds_match_score": matched_live_df["odds_match_score"],
    })

    market_snapshot_df = pd.concat(
        [red_rows, blue_rows],
        ignore_index=True
    )

    market_snapshot_df = market_snapshot_df[
        market_snapshot_df["american_odds"].notna()
    ].copy()

    return market_snapshot_df

# ============================================================
# APPEND MARKET SNAPSHOTS — DEDUPED + PARQUET SAFE
# ============================================================

def append_market_snapshots(
    market_snapshot_df,
    output_path,
):
    """
    Append market snapshots while preventing duplicate unchanged
    market rows.

    Also normalizes timestamp columns to strings before writing
    parquet so GitHub Actions / pyarrow does not fail from mixed
    datetime/string dtypes.
    """

    new_df = market_snapshot_df.copy()

    # --------------------------------------------------------
    # Normalize timestamp columns before any parquet write
    # --------------------------------------------------------

    for col in ["snapshot_timestamp", "commence_time"]:
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str)

    # --------------------------------------------------------
    # If no existing file, create it
    # --------------------------------------------------------

    if not os.path.exists(output_path):
        new_df.to_parquet(
            output_path,
            index=False,
        )

        print(f"New snapshot rows appended: {len(new_df)}")

        return new_df

    # --------------------------------------------------------
    # Load historical snapshots
    # --------------------------------------------------------

    old_df = pd.read_parquet(output_path)

    for col in ["snapshot_timestamp", "commence_time"]:
        if col in old_df.columns:
            old_df[col] = old_df[col].astype(str)

    if old_df.empty:
        new_df.to_parquet(
            output_path,
            index=False,
        )

        print(f"New snapshot rows appended: {len(new_df)}")

        return new_df

    # --------------------------------------------------------
    # Compare newest existing line vs new line
    # --------------------------------------------------------

    old_df["_snapshot_dt"] = pd.to_datetime(
        old_df["snapshot_timestamp"],
        utc=True,
        errors="coerce",
    )

    latest_existing = (
        old_df
        .sort_values("_snapshot_dt")
        .groupby([
            "fight_id",
            "fighter_id",
            "sportsbook",
            "market_type",
        ])
        .tail(1)
        .drop(columns=["_snapshot_dt"])
    )

    compare_cols = [
        "fight_id",
        "fighter_id",
        "sportsbook",
        "market_type",
    ]

    merged = new_df.merge(
        latest_existing[
            compare_cols
            +
            [
                "american_odds",
                "implied_prob",
            ]
        ],
        how="left",
        on=compare_cols,
        suffixes=("", "_old"),
    )

    changed_mask = (
        (merged["american_odds"] != merged["american_odds_old"])
        |
        (merged["implied_prob"] != merged["implied_prob_old"])
        |
        (merged["american_odds_old"].isna())
    )

    changed_rows = merged.loc[
        changed_mask,
        new_df.columns
    ].copy()

    # --------------------------------------------------------
    # If nothing changed, do not rewrite parquet
    # --------------------------------------------------------

    if changed_rows.empty:
        print("New snapshot rows appended: 0")
        return old_df

    # --------------------------------------------------------
    # Append only changed rows
    # --------------------------------------------------------

    combined_df = pd.concat(
        [old_df, changed_rows],
        ignore_index=True,
    )

    for col in ["snapshot_timestamp", "commence_time"]:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str)

    combined_df.to_parquet(
        output_path,
        index=False,
    )

    print(f"New snapshot rows appended: {len(changed_rows)}")

    return combined_df
    
