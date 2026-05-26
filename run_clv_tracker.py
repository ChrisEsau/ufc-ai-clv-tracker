
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = "."
BASE_PATH = "."

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline_config import *
from ufc_pipeline_utils import *
from ufc_odds_utils import *
from ufc_clv_utils import *


BASE_PATH = PROJECT_ROOT

ACTION_BOARD_PATH = f"{BASE_PATH}/ufc_live_action_board.csv"
LIVE_CARD_PATH = f"{BASE_PATH}/ufc_live_card.parquet"

OFFICIAL_BETS_LOG_PATH = f"{BASE_PATH}/ufc_official_bets_log.csv"
MARKET_SNAPSHOT_PATH = f"{BASE_PATH}/ufc_market_snapshots.parquet"
CLOSING_LINES_PATH = f"{BASE_PATH}/ufc_closing_lines.parquet"
CLV_RESULTS_PATH = f"{BASE_PATH}/ufc_clv_results.parquet"


def main():
    print("Starting UFC CLV tracker...")

    snapshot_timestamp = datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------------
    # Load live card
    # --------------------------------------------------------

    live_card_df = pd.read_parquet(LIVE_CARD_PATH)

    print(f"Live card rows: {len(live_card_df)}")

    # --------------------------------------------------------
    # Pull odds
    # --------------------------------------------------------

    odds_data = fetch_the_odds_api_events(
        api_key=ODDS_API_KEY,
        sport=SPORT,
        regions=REGIONS,
        markets=MARKETS,
        odds_format=ODDS_FORMAT,
    )

    print(f"Odds events returned: {len(odds_data)}")

    # --------------------------------------------------------
    # Flatten odds
    # --------------------------------------------------------

    odds_df = flatten_h2h_odds(
        odds_data,
        preferred_bookmaker=PREFERRED_BOOKMAKER,
    )

    if odds_df.empty:
        raise ValueError(
            f"No odds rows found for bookmaker: {PREFERRED_BOOKMAKER}"
        )

    odds_df["odds_fighter_1"] = odds_df["fighter_1"]
    odds_df["odds_fighter_2"] = odds_df["fighter_2"]

    odds_df["fighter_1_odds"] = odds_df["fighter_1_american_odds"]
    odds_df["fighter_2_odds"] = odds_df["fighter_2_american_odds"]

    print(f"Flattened odds rows: {len(odds_df)}")

    # --------------------------------------------------------
    # Match odds to live card
    # --------------------------------------------------------

    matched_live_df = attach_h2h_odds_to_live_df(
        live_df=live_card_df.copy(),
        odds_pool=odds_df,
        min_match_score=MIN_ODDS_MATCH_SCORE,
    )

    print(f"Matched live card rows: {len(matched_live_df)}")

    # --------------------------------------------------------
    # Build + append market snapshots
    # --------------------------------------------------------

    market_snapshot_df = build_moneyline_market_snapshot_from_matched_card(
        matched_live_df=matched_live_df,
        snapshot_timestamp=snapshot_timestamp,
    )

    combined_snapshots = append_market_snapshots(
        market_snapshot_df=market_snapshot_df,
        output_path=MARKET_SNAPSHOT_PATH,
    )

    print(f"Current market snapshot rows: {len(market_snapshot_df)}")
    print(f"Total historical snapshot rows: {len(combined_snapshots)}")


    # --------------------------------------------------------
    # Extract closing lines
    # --------------------------------------------------------

    market_snapshot_history_df = pd.read_parquet(
        MARKET_SNAPSHOT_PATH
    ).copy()

    market_snapshot_history_df["snapshot_timestamp"] = pd.to_datetime(
        market_snapshot_history_df["snapshot_timestamp"],
        utc=True,
        errors="coerce",
    )

    market_snapshot_history_df["commence_time"] = pd.to_datetime(
        market_snapshot_history_df["commence_time"],
        utc=True,
        errors="coerce",
    )

    pre_fight_df = market_snapshot_history_df[
        market_snapshot_history_df["snapshot_timestamp"]
        <=
        market_snapshot_history_df["commence_time"]
    ].copy()

    pre_fight_df = pre_fight_df.sort_values(
        "snapshot_timestamp",
        ascending=False,
    )

    closing_lines_df = pre_fight_df.drop_duplicates(
        subset=[
            "fight_id",
            "fighter_id",
            "sportsbook",
            "market_type",
        ],
        keep="first",
    )

    closing_lines_df = closing_lines_df.rename(columns={
        "snapshot_timestamp": "closing_timestamp",
        "american_odds": "closing_odds",
        "implied_prob": "closing_implied_prob",
    })

    # --------------------------------------------------------
    # Closing freshness
    # --------------------------------------------------------

    closing_lines_df["minutes_before_fight"] = (
        closing_lines_df["commence_time"]
        -
        closing_lines_df["closing_timestamp"]
    ).dt.total_seconds() / 60

    closing_lines_df["is_true_closing_window"] = (
        closing_lines_df["minutes_before_fight"] <= 60
    )

    closing_lines_df["closing_line_status"] = (
        closing_lines_df["is_true_closing_window"]
        .map({
            True: "true_close_window",
            False: "latest_pre_fight_snapshot",
        })
    )

    closing_lines_df.to_parquet(
        CLOSING_LINES_PATH,
        index=False,
    )

    print(f"Closing lines rows: {len(closing_lines_df)}")

    # --------------------------------------------------------
    # CLV update
    # --------------------------------------------------------

    if Path(OFFICIAL_BETS_LOG_PATH).exists():

        official_bets_df = pd.read_csv(
            OFFICIAL_BETS_LOG_PATH
        ).copy()

        clv_df = official_bets_df.merge(
            closing_lines_df,
            how="left",
            left_on=[
                "fight_id",
                "best_side_fighter_id",
                "bookmaker",
            ],
            right_on=[
                "fight_id",
                "fighter_id",
                "sportsbook",
            ],
        )

        clv_df = clv_df.rename(columns={
            "best_american_odds": "bet_odds",
            "best_implied_prob": "bet_implied_prob",
        })

        clv_df["clv_diff"] = (
            clv_df["closing_implied_prob"]
            -
            clv_df["bet_implied_prob"]
        )

        clv_df["beat_closing_line"] = (
            clv_df["clv_diff"] > 0
        )

        clv_df["odds_movement"] = (
            clv_df["closing_odds"]
            -
            clv_df["bet_odds"]
        )

        clv_df.to_parquet(
            CLV_RESULTS_PATH,
            index=False,
        )

        print(f"CLV rows updated: {len(clv_df)}")

    else:
        print("No official bets log found. Skipping CLV update.")
        
        
    print("UFC CLV tracker completed.")
    
if __name__ == "__main__":
    main()
