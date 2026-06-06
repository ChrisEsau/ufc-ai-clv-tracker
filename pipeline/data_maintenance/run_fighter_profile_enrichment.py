from argparse import ArgumentParser
from datetime import datetime, timezone

import pandas as pd

from scrapers.ufcstats_fighter_profiles import scrape_fighter_profile

from pipeline.common.paths import (
    STAGED_FIGHT_ROWS_PATH,
    STAGED_MASTER_ROWS_ENRICHED_PATH,
    STAGED_FIGHTER_PROFILES_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    FIGHTER_PROFILE_SCRAPE_AUDIT_PATH,
)

STAGED_FIGHTS_PATH = STAGED_FIGHT_ROWS_PATH
STAGED_MASTER_PATH = STAGED_MASTER_ROWS_ENRICHED_PATH

PROFILE_OUTPUT = STAGED_FIGHTER_PROFILES_PATH
PROFILED_MASTER_OUTPUT = STAGED_MASTER_ROWS_PROFILED_PATH
AUDIT_OUTPUT = FIGHTER_PROFILE_SCRAPE_AUDIT_PATH

MISSING_STRINGS = {"", "nan", "None", "NaN", "NaT", "<NA>"}


def clean_string(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in MISSING_STRINGS:
        return None

    return value


def normalize_name(value):
    value = clean_string(value)

    if value is None:
        return None

    return " ".join(value.lower().split())


def fighter_id_from_url(value):
    value = clean_string(value)

    if value is None:
        return None

    return value.rstrip("/").split("/")[-1]


def is_missing(value):
    return clean_string(value) is None


def build_fighter_queue(staged_fights):
    fighter_rows = []

    for _, row in staged_fights.iterrows():
        for side, name_col, url_col in [
            ("r", "red_fighter", "red_fighter_url"),
            ("b", "blue_fighter", "blue_fighter_url"),
        ]:
            fighter_url = clean_string(row.get(url_col))

            if fighter_url is None:
                continue

            fighter_rows.append(
                {
                    "fighter_role": side,
                    "fighter_name": clean_string(row.get(name_col)),
                    "fighter_name_key": normalize_name(row.get(name_col)),
                    "fighter_url": fighter_url,
                    "fighter_id": fighter_id_from_url(fighter_url),
                }
            )

    if not fighter_rows:
        return pd.DataFrame(
            columns=[
                "fighter_role",
                "fighter_name",
                "fighter_name_key",
                "fighter_url",
                "fighter_id",
            ]
        )

    return (
        pd.DataFrame(fighter_rows)
        .dropna(subset=["fighter_url", "fighter_id"])
        .drop_duplicates(subset=["fighter_id"])
        .reset_index(drop=True)
    )



def build_profile_maps(profiles):
    if profiles.empty:
        return {}, {}

    by_id = {}
    by_name = {}

    for _, row in profiles.iterrows():
        fighter_id = clean_string(row.get("fighter_id"))
        fighter_name_key = normalize_name(row.get("fighter_name"))

        if fighter_id is not None and fighter_id not in by_id:
            by_id[fighter_id] = row

        if fighter_name_key is not None and fighter_name_key not in by_name:
            by_name[fighter_name_key] = row

    return by_id, by_name


def run_fighter_profile_enrichment(max_fighters=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    staged_fights = pd.read_parquet(STAGED_FIGHTS_PATH)
    staged_master = pd.read_parquet(STAGED_MASTER_PATH)

    fighters = build_fighter_queue(staged_fights)

    if max_fighters is not None:
        fighters = fighters.head(max_fighters)

    print("Fighters to scrape:", len(fighters))

    print()
    print("========== FIGHTERS TO SCRAPE ==========")

    if fighters.empty:
        print("None")
    else:
        print(
            fighters[
                [
                    "fighter_name",
                    "fighter_id",
                    "fighter_role",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    profile_dfs = []
    audit_rows = []

    for idx, row in fighters.iterrows():
        fighter_url = row["fighter_url"]
        fighter_id = row["fighter_id"]

        print()
        print(f"[{idx + 1}/{len(fighters)}] {fighter_url}")

        try:
            profile = scrape_fighter_profile(
                fighter_url=fighter_url,
                fighter_id=fighter_id,
            )

            profile["run_id"] = run_id
            profile["run_timestamp"] = run_timestamp

            profile_dfs.append(profile)

            audit_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "fighter_id": fighter_id,
                    "fighter_name": row.get("fighter_name"),
                    "fighter_url": fighter_url,
                    "status": "success",
                    "error": None,
                }
            )

            print("Success")

        except Exception as e:
            audit_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "fighter_id": fighter_id,
                    "fighter_name": row.get("fighter_name"),
                    "fighter_url": fighter_url,
                    "status": "failed",
                    "error": str(e),
                }
            )

            print("FAILED")
            print(e)

    profiles = (
        pd.concat(profile_dfs, ignore_index=True)
        if profile_dfs
        else pd.DataFrame()
    )

    print()
    print("========== PROFILE SAMPLE ==========")

    if not profiles.empty:
        sample_cols = [
            c for c in [
                "fighter_name",
                "fighter_id",
                "height",
                "reach",
                "stance",
                "wins",
                "losses",
                "splm",
                "sapm",
            ]
            if c in profiles.columns
        ]

        print(
            profiles[sample_cols]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("None")

    profiles.to_parquet(PROFILE_OUTPUT, index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_parquet(AUDIT_OUTPUT, index=False)

    profile_cols = [
        "nick_name",
        "wins",
        "losses",
        "draws",
        "height",
        "weight",
        "reach",
        "stance",
        "dob",
        "splm",
        "str_acc",
        "sapm",
        "str_def",
        "td_avg",
        "td_avg_acc",
        "td_def",
        "sub_avg",
    ]

    profiles_by_id, profiles_by_name = build_profile_maps(profiles)

    for side, name_col in [("r", "r_name"), ("b", "b_name")]:
        id_col = f"{side}_id"

        if id_col in staged_master.columns and name_col in staged_master.columns:

            def lookup_id(row):
                existing_id = clean_string(row.get(id_col))

                if existing_id is not None:
                    return existing_id

                name_key = normalize_name(row.get(name_col))
                match = profiles_by_name.get(name_key)

                if match is None:
                    return None

                return match.get("fighter_id")

            staged_master[id_col] = staged_master.apply(lookup_id, axis=1)

        for profile_col in profile_cols:
            target_col = f"{side}_{profile_col}"

            if (
                target_col not in staged_master.columns
                or id_col not in staged_master.columns
            ):
                continue

            def lookup_profile(fid):
                fighter_id = clean_string(fid)

                if fighter_id is None:
                    return None

                match = profiles_by_id.get(fighter_id)

                if match is None:
                    return None

                return match.get(profile_col)

            staged_master[target_col] = staged_master[id_col].apply(lookup_profile)

    if (
        "winner" in staged_master.columns
        and "winner_id" in staged_master.columns
    ):

        def get_winner_id(row):
            winner = normalize_name(row.get("winner"))

            if winner is None:
                return None

            if winner == normalize_name(row.get("r_name")):
                return row.get("r_id")

            if winner == normalize_name(row.get("b_name")):
                return row.get("b_id")

            return None

        staged_master["winner_id"] = staged_master.apply(get_winner_id, axis=1)

    print()
    print("========== PROFILE LOOKUP DEBUG ==========")

    if not profiles.empty:
        print(
            profiles[
                [
                    "fighter_name",
                    "fighter_id",
                    "height",
                    "reach",
                    "stance",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("None")

    print()
    print("========== MASTER DEBUG ==========")

    debug_cols = [
        c for c in [
            "r_name",
            "r_id",
            "b_name",
            "b_id",
            "winner",
            "winner_id",
        ]
        if c in staged_master.columns
    ]

    print(
        staged_master[debug_cols]
        .head()
        .to_string(index=False)
    )

    staged_master.to_parquet(
        PROFILED_MASTER_OUTPUT,
        index=False,
    )

    print()
    print("========== FIGHTER PROFILE ENRICHMENT ==========")
    print("Profiles scraped:", len(profiles))
    print("Audit rows:", len(audit))
    print("Profiled staged rows:", len(staged_master))
    print("Saved:", PROFILE_OUTPUT)
    print("Saved:", PROFILED_MASTER_OUTPUT)
    print("Saved:", AUDIT_OUTPUT)

    return profiles, staged_master, audit


def parse_args():
    parser = ArgumentParser(
        description="Scrape UFCStats fighter profiles for staged red and blue fighters."
    )
    parser.add_argument(
        "--max-fighters",
        type=int,
        default=None,
        help="Optional scrape limit for local validation; defaults to all fighters.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fighter_profile_enrichment(max_fighters=args.max_fighters)
