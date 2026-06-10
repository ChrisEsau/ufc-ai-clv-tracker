import argparse
import os

import pandas as pd

from pipeline.common.paths import (
    AUDITS_DIR,
    LIVE_CARD_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    UPCOMING_FIGHTS_PATH,
    ensure_data_dirs,
)


LIVE_CARD_COLUMNS = [
    "event_name",
    "event_id",
    "event_date",
    "event_location",
    "event_url",
    "fight_order",
    "fight_id",
    "fight_url",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "red_fighter_url",
    "blue_fighter_url",
    "weight_class",
    "title_fight",
    "total_rounds",
]

LIVE_CARD_BUILD_AUDIT_PATH = AUDITS_DIR / "ufc_live_card_build_audit.parquet"
LIVE_CARD_REJECTED_ROWS_PATH = AUDITS_DIR / "ufc_live_card_rejected_rows.parquet"

REQUIRED_FIGHT_COLUMNS = [
    "fight_id",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
]



def _prepare_live_card(upcoming_fights):
    """Normalize an upcoming-fights slice into the canonical live-card shape.

    This is intentionally strict. Placeholder rows such as ``nan__nan`` should not
    reach live prediction because they create duplicate fights, blank labels, and
    zero-filled feature rows.
    """

    raw_count = len(upcoming_fights)
    live_card = upcoming_fights.copy()

    for column in LIVE_CARD_COLUMNS:
        if column not in live_card.columns:
            live_card[column] = pd.NA

    live_card = live_card[LIVE_CARD_COLUMNS]
    live_card = _normalize_string_columns(live_card)
    live_card = _normalize_fight_context_columns(live_card)

    invalid_mask = _build_invalid_fight_mask(live_card)
    rejected_rows = live_card.loc[invalid_mask].copy()
    live_card = live_card.loc[~invalid_mask].copy()

    duplicate_mask = live_card["fight_id"].duplicated(keep="first") if "fight_id" in live_card.columns else pd.Series(False, index=live_card.index)
    duplicate_rows = live_card.loc[duplicate_mask].copy()
    if duplicate_mask.any():
        live_card = live_card.loc[~duplicate_mask].copy()

    sort_columns = [column for column in ["event_date", "event_name", "fight_order"] if column in live_card.columns]
    if sort_columns:
        live_card = live_card.sort_values(sort_columns, na_position="last")

    live_card = live_card.reset_index(drop=True)
    rejected_rows = rejected_rows.reset_index(drop=True)
    duplicate_rows = duplicate_rows.reset_index(drop=True)

    _write_live_card_build_audit(
        raw_count=raw_count,
        valid_count=len(live_card),
        rejected_rows=rejected_rows,
        duplicate_rows=duplicate_rows,
    )

    if live_card.empty:
        raise ValueError(
            "Live card build produced zero valid fights after filtering invalid rows. "
            f"Rejected rows were written to {LIVE_CARD_REJECTED_ROWS_PATH}."
        )

    return live_card



def _normalize_string_columns(df):
    """Normalize object/string columns and convert obvious placeholder tokens to NA."""

    out = df.copy()
    placeholder_values = {"", "nan", "none", "null", "nat", "<na>"}

    for column in out.columns:
        if out[column].dtype == "object" or str(out[column].dtype).startswith("string"):
            values = out[column].astype("string").str.strip()
            values = values.mask(values.str.lower().isin(placeholder_values), pd.NA)
            out[column] = values

    return out



def _normalize_fight_context_columns(df):
    """Normalize fight-level context columns used by prop models.

    These columns originate from upcoming-fights/live-card data, not fighter state.
    They intentionally remain nullable when upstream data is missing so downstream
    model-contract checks fail loudly rather than silently fabricating context.
    """

    out = df.copy()

    if "total_rounds" in out.columns:
        out["total_rounds"] = pd.to_numeric(out["total_rounds"], errors="coerce")

    if "title_fight" in out.columns:
        title = out["title_fight"]
        if title.dtype == "bool":
            out["title_fight"] = title.astype("Int64")
        else:
            normalized = title.astype("string").str.strip().str.lower()
            bool_map = {
                "true": 1,
                "t": 1,
                "yes": 1,
                "y": 1,
                "1": 1,
                "false": 0,
                "f": 0,
                "no": 0,
                "n": 0,
                "0": 0,
            }
            mapped = normalized.map(bool_map)
            numeric = pd.to_numeric(title, errors="coerce")
            out["title_fight"] = mapped.fillna(numeric).astype("Int64")

    return out



def _build_invalid_fight_mask(live_card):
    """Return rows that cannot represent valid upcoming fights."""

    invalid = pd.Series(False, index=live_card.index)

    for column in REQUIRED_FIGHT_COLUMNS:
        values = live_card[column].astype("string").fillna("").str.strip()
        invalid = invalid | values.eq("") | values.str.lower().isin({"nan", "none", "null", "nat", "<na>"})

    if "fight_id" in live_card.columns:
        fight_id_values = live_card["fight_id"].astype("string").fillna("").str.strip().str.lower()
        invalid = invalid | fight_id_values.eq("nan__nan") | fight_id_values.eq("__")

    return invalid



def _write_live_card_build_audit(*, raw_count, valid_count, rejected_rows, duplicate_rows):
    """Write build audit artifacts for live-card QA."""

    AUDITS_DIR.mkdir(parents=True, exist_ok=True)

    audit = pd.DataFrame(
        [
            {
                "raw_rows": raw_count,
                "valid_rows": valid_count,
                "invalid_rejected_rows": len(rejected_rows),
                "duplicate_rejected_rows": len(duplicate_rows),
                "rejected_rows_path": str(LIVE_CARD_REJECTED_ROWS_PATH),
            }
        ]
    )
    audit.to_parquet(LIVE_CARD_BUILD_AUDIT_PATH, index=False)

    rejected_frames = []
    if len(rejected_rows) > 0:
        rejected = rejected_rows.copy()
        rejected["reject_reason"] = "missing_required_fight_identity"
        rejected_frames.append(rejected)
    if len(duplicate_rows) > 0:
        duplicates = duplicate_rows.copy()
        duplicates["reject_reason"] = "duplicate_fight_id"
        rejected_frames.append(duplicates)

    if rejected_frames:
        pd.concat(rejected_frames, ignore_index=True).to_parquet(LIVE_CARD_REJECTED_ROWS_PATH, index=False)
    else:
        pd.DataFrame(columns=LIVE_CARD_COLUMNS + ["reject_reason"]).to_parquet(LIVE_CARD_REJECTED_ROWS_PATH, index=False)



def _write_live_card(live_card, status_label):
    """Persist the live-card artifact plus event metadata used by the dashboard."""

    live_card.to_parquet(LIVE_CARD_PATH, index=False)

    event_columns = [
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    ]
    selected_events = live_card[event_columns].drop_duplicates().reset_index(drop=True)
    selected_events["selection_scope"] = status_label
    selected_events.to_parquet(SELECTED_LIVE_CARD_EVENT_PATH, index=False)

    print(f"{status_label.title().replace('_', ' ')} live card saved: {LIVE_CARD_PATH} ({len(live_card)} rows)")
    print(f"Live-card event metadata saved: {SELECTED_LIVE_CARD_EVENT_PATH} ({len(selected_events)} event rows)")
    print(f"Live-card build audit saved: {LIVE_CARD_BUILD_AUDIT_PATH}")
    print(f"Live-card rejected rows saved: {LIVE_CARD_REJECTED_ROWS_PATH}")



def build_live_card(event_id):
    ensure_data_dirs()

    if not event_id:
        raise ValueError("event_id is required to build a selected live card.")

    upcoming_fights = pd.read_parquet(UPCOMING_FIGHTS_PATH)

    if "event_id" not in upcoming_fights.columns:
        raise ValueError(f"Upcoming fights artifact is missing event_id: {UPCOMING_FIGHTS_PATH}")

    live_card = upcoming_fights[upcoming_fights["event_id"].astype(str) == str(event_id)].copy()

    if live_card.empty:
        raise ValueError(f"No upcoming fights found for event_id={event_id} in {UPCOMING_FIGHTS_PATH}")

    live_card = _prepare_live_card(live_card)
    _write_live_card(live_card, "selected_event")

    return live_card



def build_all_upcoming_live_card():
    ensure_data_dirs()

    upcoming_fights = pd.read_parquet(UPCOMING_FIGHTS_PATH)

    if upcoming_fights.empty:
        raise ValueError(f"No upcoming fights found in {UPCOMING_FIGHTS_PATH}")

    live_card = _prepare_live_card(upcoming_fights)
    _write_live_card(live_card, "all_upcoming")

    return live_card



def parse_args():
    parser = argparse.ArgumentParser(description="Build data/predictions/ufc_live_card.parquet from upcoming UFCStats fights.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID") or os.getenv("BETTING_EVENT_ID"), help="UFCStats event id to select.")
    parser.add_argument(
        "--all-upcoming",
        action="store_true",
        help="Build the live-card artifact from every fight in the upcoming-fights artifact.",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    if args.all_upcoming:
        build_all_upcoming_live_card()
    else:
        build_live_card(args.event_id)


if __name__ == "__main__":
    main()
