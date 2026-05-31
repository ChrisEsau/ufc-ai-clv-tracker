from pipeline.data_maintenance.run_ufcstats_fight_scrape import run_fight_scrape
from pipeline.data_maintenance.run_ufcstats_fight_detail_scrape import run_fight_detail_scrape
from pipeline.data_maintenance.run_staged_master_mapper import run_staged_master_mapper
from pipeline.data_maintenance.run_staged_derived_stats_transformer import (
    run_staged_derived_stats_transformer,
)
from pipeline.data_maintenance.run_fighter_profile_enrichment import (
    run_fighter_profile_enrichment,
)

from pipeline.data_maintenance.run_master_column_validation import (
    run_master_column_validation,
)
from pipeline.data_maintenance.run_append_precheck_validation import (
    run_append_precheck_validation,
)

def run_ingest_single_event(
    event_id: str,
    max_fights: int | None = 1,
    max_fighters: int | None = 1,
):
    print("========== SINGLE EVENT INGEST ==========")
    print("Event ID:", event_id)

    print()
    print("STEP 1: Fight scrape")
    run_fight_scrape(event_id=event_id, max_events=None)

    print()
    print("STEP 2: Fight detail scrape")
    run_fight_detail_scrape(max_fights=max_fights)

    print()
    print("STEP 3: Map staged rows to master schema")
    run_staged_master_mapper()

    print()
    print("STEP 4: Derived stats transformer")
    run_staged_derived_stats_transformer(debug=True)

    print()
    print("STEP 5: Fighter profile enrichment")
    run_fighter_profile_enrichment(max_fighters=max_fighters)

    print()
    print("STEP 6: Master column validation")
    run_master_column_validation()

    print()
    print("STEP 7: Append precheck validation")
    precheck, append_ready = run_append_precheck_validation()

    print()
    print("Append ready:", append_ready)

    print()
    print("========== SINGLE EVENT INGEST COMPLETE ==========")


import os

if __name__ == "__main__":

    event_id = os.getenv("EVENT_ID")

    if not event_id:
        raise ValueError(
            "EVENT_ID environment variable not supplied."
        )

    run_ingest_single_event(
        event_id=event_id,
        max_fights=1,
        max_fighters=1,
    )