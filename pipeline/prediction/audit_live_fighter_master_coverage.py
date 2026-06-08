from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, LIVE_CARD_PATH, MASTER_PATH


LIVE_FIGHTER_MASTER_COVERAGE_PATH = AUDITS_DIR / "ufc_live_fighter_master_coverage.parquet"
LIVE_FIGHTERS_NOT_IN_MASTER_PATH = AUDITS_DIR / "ufc_live_fighters_not_in_master.parquet"


class LiveFighterMasterAuditError(RuntimeError):
    """Raised when live fighter master coverage cannot be audited."""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether live-card fighters exist in historical master results.",
    )
    parser.add_argument(
        "--live-card-path",
        default=str(LIVE_CARD_PATH),
        help="Path to ufc_live_card.parquet.",
    )
    parser.add_argument(
        "--master-path",
        default=str(MASTER_PATH),
        help="Path to ufc_master.parquet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of detail rows to print.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    live_card = _read_required_parquet(Path(args.live_card_path), "live card")
    master = _read_required_parquet(Path(args.master_path), "master")

    live_fighters = _build_live_fighter_rows(live_card)
    master_counts = _build_master_fighter_counts(master)

    coverage = live_fighters.merge(master_counts, on="fighter_id", how="left")
    coverage["master_fight_count"] = coverage["master_fight_count"].fillna(0).astype(int)
    coverage["exists_in_master"] = coverage["master_fight_count"] > 0

    _write_outputs(coverage)
    _print_report(coverage, live_card, master, top_n=args.top_n)



def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise LiveFighterMasterAuditError(f"Missing {label}: {path}")
    return pd.read_parquet(path)



def _build_live_fighter_rows(live_card: pd.DataFrame) -> pd.DataFrame:
    required = [
        "event_id",
        "event_name",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
    ]
    missing = [column for column in required if column not in live_card.columns]
    if missing:
        raise LiveFighterMasterAuditError(f"Live card missing required columns: {missing}")

    red = live_card[["event_id", "event_name", "fight_id", "red_fighter", "red_fighter_id"]].copy()
    red = red.rename(columns={"red_fighter": "fighter_name", "red_fighter_id": "fighter_id"})
    red["side"] = "red"

    blue = live_card[["event_id", "event_name", "fight_id", "blue_fighter", "blue_fighter_id"]].copy()
    blue = blue.rename(columns={"blue_fighter": "fighter_name", "blue_fighter_id": "fighter_id"})
    blue["side"] = "blue"

    fighters = pd.concat([red, blue], ignore_index=True)
    fighters["fighter_id"] = _normalize_id_series(fighters["fighter_id"])
    fighters["fighter_name"] = fighters["fighter_name"].astype("string").fillna("").str.strip()
    return fighters



def _build_master_fighter_counts(master: pd.DataFrame) -> pd.DataFrame:
    id_columns = [column for column in ["r_id", "b_id"] if column in master.columns]
    if not id_columns:
        raise LiveFighterMasterAuditError("Master must contain r_id and/or b_id columns.")

    frames = []
    for column in id_columns:
        frame = pd.DataFrame({"fighter_id": _normalize_id_series(master[column])})
        frame = frame.dropna(subset=["fighter_id"])
        frames.append(frame)

    all_fighters = pd.concat(frames, ignore_index=True)
    counts = all_fighters.value_counts("fighter_id").rename("master_fight_count").reset_index()
    return counts



def _write_outputs(coverage: pd.DataFrame) -> None:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(LIVE_FIGHTER_MASTER_COVERAGE_PATH, index=False)
    coverage[~coverage["exists_in_master"]].to_parquet(LIVE_FIGHTERS_NOT_IN_MASTER_PATH, index=False)



def _print_report(coverage: pd.DataFrame, live_card: pd.DataFrame, master: pd.DataFrame, *, top_n: int) -> None:
    print("=" * 80)
    print("LIVE FIGHTER MASTER COVERAGE AUDIT")
    print("=" * 80)
    print(f"Live fights: {live_card['fight_id'].nunique() if 'fight_id' in live_card.columns else 'missing fight_id'}")
    print(f"Live fighter slots: {len(coverage)}")
    print(f"Unique live fighters: {coverage['fighter_id'].nunique(dropna=True)}")
    print(f"Master rows: {len(master)}")

    print("\nExists in master counts:")
    print(coverage["exists_in_master"].value_counts(dropna=False).to_string())

    missing = coverage[~coverage["exists_in_master"]].copy()
    print(f"\nLive fighter slots not in master: {len(missing)}")
    print(f"Unique live fighters not in master: {missing['fighter_id'].nunique(dropna=True)}")

    if len(missing) > 0:
        display_columns = ["event_name", "fight_id", "side", "fighter_name", "fighter_id", "master_fight_count"]
        print(f"\nLive fighters not found in master first {top_n} rows:")
        print(missing[display_columns].head(top_n).to_string(index=False))

    if len(coverage) > 0:
        print("\nLowest master fight counts first rows:")
        display_columns = ["event_name", "fight_id", "side", "fighter_name", "fighter_id", "master_fight_count", "exists_in_master"]
        print(coverage.sort_values(["master_fight_count", "fighter_name"])[display_columns].head(top_n).to_string(index=False))

    print("\nAudit outputs written:")
    print(f"  {LIVE_FIGHTER_MASTER_COVERAGE_PATH}")
    print(f"  {LIVE_FIGHTERS_NOT_IN_MASTER_PATH}")



def _normalize_id_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    values = values.mask(values.str.lower().isin({"", "nan", "none", "null", "nat", "<na>"}), pd.NA)
    return values



if __name__ == "__main__":
    main()
