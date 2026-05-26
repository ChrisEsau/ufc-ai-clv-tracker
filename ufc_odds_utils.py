# ============================================================
# ufc_odds_utils.py
# Shared odds ingestion + matching helpers
# ============================================================

import requests
import numpy as np
import pandas as pd

from ufc_pipeline_utils import (
    normalize_name,
    token_set_score,
    american_to_decimal,
    american_to_implied_prob,
)

from pipeline_config import MIN_ODDS_MATCH_SCORE

# ============================================================
# ODDS API PULL
# ============================================================

def fetch_the_odds_api_events(
    api_key,
    sport="mma_mixed_martial_arts",
    regions="us",
    markets="h2h",
    odds_format="american",
):
    """
    Pull odds from The Odds API using region-based odds.
    """

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Odds API request failed: "
            f"{response.status_code} - {response.text}"
        )

    return response.json()

# ============================================================
# FLATTEN H2H ODDS
# ============================================================

def flatten_h2h_odds(
    odds_json,
    preferred_bookmaker="DraftKings",
):
    """
    Convert The Odds API h2h response into a flat dataframe.
    Optionally filters by bookmaker title, e.g. DraftKings.
    """

    rows = []

    for event in odds_json:

        event_name = (
            event.get("home_team", "")
            + " vs "
            + event.get("away_team", "")
        )

        commence_time = event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):

            bookmaker_name = bookmaker.get("title")

            if (
                preferred_bookmaker is not None
                and bookmaker_name != preferred_bookmaker
            ):
                continue

            for market in bookmaker.get("markets", []):

                if market.get("key") != "h2h":
                    continue

                outcomes = market.get("outcomes", [])

                if len(outcomes) < 2:
                    continue

                fighter_1 = outcomes[0].get("name")
                fighter_2 = outcomes[1].get("name")

                odds_1 = outcomes[0].get("price")
                odds_2 = outcomes[1].get("price")

                rows.append({
                    "event_name": event_name,
                    "commence_time": commence_time,
                    "bookmaker": bookmaker_name,

                    "fighter_1": fighter_1,
                    "fighter_2": fighter_2,

                    "fighter_1_norm": normalize_name(fighter_1),
                    "fighter_2_norm": normalize_name(fighter_2),

                    "fighter_1_american_odds": odds_1,
                    "fighter_2_american_odds": odds_2,

                    "fighter_1_decimal_odds": american_to_decimal(odds_1),
                    "fighter_2_decimal_odds": american_to_decimal(odds_2),

                    "fighter_1_implied_prob": american_to_implied_prob(odds_1),
                    "fighter_2_implied_prob": american_to_implied_prob(odds_2),
                })

    return pd.DataFrame(rows)


# ============================================================
# ODDS MATCHING
# ============================================================
from difflib import SequenceMatcher


def compact_name(name):
    return normalize_name(name).replace(" ", "")


def char_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio() * 100


def composite_name_score(a, b):
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    a_compact = compact_name(a)
    b_compact = compact_name(b)

    if a_norm == b_norm:
        return 100.0

    if a_compact == b_compact:
        return 100.0

    token_score = token_set_score(a_norm, b_norm)
    char_score = char_similarity(a_compact, b_compact)

    return max(token_score, char_score)


def match_live_fight_to_odds_row(
    live_row,
    odds_pool,
    min_single_score=80,
):
    red_name = live_row["red_fighter"]
    blue_name = live_row["blue_fighter"]

    red_id = live_row["red_fighter_id"]
    blue_id = live_row["blue_fighter_id"]

    candidates = []

    for _, odds_row in odds_pool.iterrows():

        f1_name = odds_row["fighter_1"]
        f2_name = odds_row["fighter_2"]

        same_f1_score = composite_name_score(f1_name, red_name)
        same_f2_score = composite_name_score(f2_name, blue_name)

        same_pair_score = (same_f1_score + same_f2_score) / 2
        same_min_score = min(same_f1_score, same_f2_score)

        rev_f1_score = composite_name_score(f1_name, blue_name)
        rev_f2_score = composite_name_score(f2_name, red_name)

        rev_pair_score = (rev_f1_score + rev_f2_score) / 2
        rev_min_score = min(rev_f1_score, rev_f2_score)

        if same_pair_score >= rev_pair_score:
            candidates.append({
                "odds_row": odds_row,
                "match_type": "same_order",
                "pair_score": same_pair_score,
                "min_single_score": same_min_score,
                "odds_fighter_1_score": same_f1_score,
                "odds_fighter_2_score": same_f2_score,
                "odds_fighter_1_id": red_id,
                "odds_fighter_2_id": blue_id,
                "red_odds": odds_row["fighter_1_american_odds"],
                "blue_odds": odds_row["fighter_2_american_odds"],
                "red_decimal_odds": odds_row["fighter_1_decimal_odds"],
                "blue_decimal_odds": odds_row["fighter_2_decimal_odds"],
                "red_implied_prob": odds_row["fighter_1_implied_prob"],
                "blue_implied_prob": odds_row["fighter_2_implied_prob"],
            })
        else:
            candidates.append({
                "odds_row": odds_row,
                "match_type": "reversed_order",
                "pair_score": rev_pair_score,
                "min_single_score": rev_min_score,
                "odds_fighter_1_score": rev_f1_score,
                "odds_fighter_2_score": rev_f2_score,
                "odds_fighter_1_id": blue_id,
                "odds_fighter_2_id": red_id,
                "red_odds": odds_row["fighter_2_american_odds"],
                "blue_odds": odds_row["fighter_1_american_odds"],
                "red_decimal_odds": odds_row["fighter_2_decimal_odds"],
                "blue_decimal_odds": odds_row["fighter_1_decimal_odds"],
                "red_implied_prob": odds_row["fighter_2_implied_prob"],
                "blue_implied_prob": odds_row["fighter_1_implied_prob"],
            })

    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: x["pair_score"], reverse=True)[0]
    odds_row = best["odds_row"]

    if best["min_single_score"] < min_single_score:
        return None

    return {
        "red_american_odds": best["red_odds"],
        "blue_american_odds": best["blue_odds"],
        "red_decimal_odds": best["red_decimal_odds"],
        "blue_decimal_odds": best["blue_decimal_odds"],
        "red_implied_prob": best["red_implied_prob"],
        "blue_implied_prob": best["blue_implied_prob"],

        "bookmaker": odds_row.get("bookmaker"),
        "commence_time": odds_row.get("commence_time"),

        "odds_match_type": best["match_type"],
        "odds_match_score": best["pair_score"],
        "odds_min_single_score": best["min_single_score"],

        "matched_odds_fighter_1": odds_row["fighter_1"],
        "matched_odds_fighter_2": odds_row["fighter_2"],

        "odds_fighter_1_id": best["odds_fighter_1_id"],
        "odds_fighter_2_id": best["odds_fighter_2_id"],

        "odds_fighter_1_score": best["odds_fighter_1_score"],
        "odds_fighter_2_score": best["odds_fighter_2_score"],
    }


def attach_h2h_odds_to_live_df(
    live_df,
    odds_pool,
    min_match_score=80,
):
    rows = []

    for _, row in live_df.iterrows():

        odds_match = match_live_fight_to_odds_row(
            live_row=row,
            odds_pool=odds_pool,
            min_single_score=min_match_score,
        )

        out = row.to_dict()

        if odds_match is None:
            out.update({
                "red_american_odds": np.nan,
                "blue_american_odds": np.nan,
                "red_decimal_odds": np.nan,
                "blue_decimal_odds": np.nan,
                "red_implied_prob": np.nan,
                "blue_implied_prob": np.nan,
                "bookmaker": None,
                "commence_time": None,
                "odds_match_type": "missing",
                "odds_match_score": 0,
                "odds_min_single_score": 0,
                "matched_odds_fighter_1": None,
                "matched_odds_fighter_2": None,
                "odds_fighter_1_id": None,
                "odds_fighter_2_id": None,
                "odds_fighter_1_score": 0,
                "odds_fighter_2_score": 0,
            })
        else:
            out.update(odds_match)

        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# BEST SIDE HELPERS
# ============================================================

def choose_best_side(row):
    """
    Pick the side with higher EV.
    """

    if row["red_ev"] >= row["blue_ev"]:
        return row["red_fighter"]

    return row["blue_fighter"]


def add_best_side_columns(df):
    """
    Add best-side columns after EV is calculated.
    """

    out = df.copy()

    out["best_side"] = out.apply(
        lambda row:
            row["red_fighter"]
            if row["red_ev"] >= row["blue_ev"]
            else row["blue_fighter"],
        axis=1,
    )

    out["best_prob"] = out.apply(
        lambda row:
            row["red_model_prob"]
            if row["red_ev"] >= row["blue_ev"]
            else row["blue_model_prob"],
        axis=1,
    )

    out["best_implied_prob"] = out.apply(
        lambda row:
            row["red_implied_prob"]
            if row["red_ev"] >= row["blue_ev"]
            else row["blue_implied_prob"],
        axis=1,
    )

    out["best_edge"] = (
        out["best_prob"]
        - out["best_implied_prob"]
    )

    out["best_ev"] = out[
        ["red_ev", "blue_ev"]
    ].max(axis=1)

    out["best_american_odds"] = out.apply(
        lambda row:
            row["red_american_odds"]
            if row["red_ev"] >= row["blue_ev"]
            else row["blue_american_odds"],
        axis=1,
    )

    out["best_confidence"] = (
        out["best_prob"] * 100
    )

    return out