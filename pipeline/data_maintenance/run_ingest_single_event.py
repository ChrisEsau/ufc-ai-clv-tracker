import os
from argparse import ArgumentParser

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
from pipeline.data_maintenance.run_staged_final_review import run_staged_final_review



def parse_optional_int(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "" or value.lower() in {"none", "null", "all"}:
        return None

    return int(value)


def run_ingest_single_event(
    event_id: str,
    max_fights: int | None = None,
    max_fighters: int | None = None,
):
    print("========== SINGLE EVENT INGEST ==========")
    print("Event ID:", event_id)
    print("Mode: full")
    print("Max fights:", "all" if max_fights is None else max_fights)
    print("Max fighters:", "all" if max_fighters is None else max_fighters)

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
    _, append_ready = run_append_precheck_validation()

    print()
    print("STEP 8: Final staged review")
    _, final_review_pass = run_staged_final_review()

    print()
    print("Append ready:", append_ready)
    print("Final review pass:", final_review_pass)
    print("Append allowed:", bool(append_ready and final_review_pass))

    print()
    print("========== SINGLE EVENT INGEST COMPLETE ==========")
    print("No append was performed. Review staged artifacts before append.")

    return append_ready, final_review_pass


def parse_args():
    parser = ArgumentParser(description="Stage and review a selected UFCStats event.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID"))
    parser.add_argument("--max-fights", default=os.getenv("MAX_FIGHTS"))
    parser.add_argument("--max-fighters", default=os.getenv("MAX_FIGHTERS"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.event_id:
        raise ValueError("EVENT_ID environment variable or --event-id is required.")

    run_ingest_single_event(
        event_id=args.event_id,
        max_fights=parse_optional_int(args.max_fights),
        max_fighters=parse_optional_int(args.max_fighters),
    )
