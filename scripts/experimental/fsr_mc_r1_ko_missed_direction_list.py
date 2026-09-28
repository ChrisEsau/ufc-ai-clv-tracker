"""List historical Round-1 KO/TKO bouts where the MC missed winner direction.

This is a cheap post-hoc diagnostic over the saved mature 2020+ round-recovery
population audit. It does not rerun the Monte Carlo.

Definitions
-----------
- WRONG: the MC assigned higher KO probability to the actual KO loser.
- TIE:   red and blue KO probabilities were equal. Ties are shown separately
         because the 10-path audit creates coarse 0.10 probability increments.
- RIGHT: the MC assigned higher KO probability to the actual KO winner.

The script prints all WRONG R1 KO bouts, prints ties separately, and writes a CSV
containing every historical R1 KO with its direction classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_round_recovery_audit.csv"
)
MASTER_PATH = Path("data/master/ufc_master.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_r1_ko_direction_misses.csv"
)


def _resolve_name_columns(master: pd.DataFrame) -> tuple[str | None, str | None]:
    """Resolve red/blue fighter-name columns without assuming one schema."""
    candidate_pairs = (
        ("r_name", "b_name"),
        ("red_name", "blue_name"),
        ("r_fighter_name", "b_fighter_name"),
        ("red_fighter_name", "blue_fighter_name"),
        ("r_fighter", "b_fighter"),
        ("red_fighter", "blue_fighter"),
    )
    for r_col, b_col in candidate_pairs:
        if r_col in master.columns and b_col in master.columns:
            return r_col, b_col
    return None, None


def _master_names() -> pd.DataFrame:
    """Return one row per fight with corner names when available."""
    if not MASTER_PATH.exists():
        return pd.DataFrame(columns=["bout_id", "r_name", "b_name"])

    master = pd.read_parquet(MASTER_PATH).copy()
    if "fight_id" not in master.columns:
        return pd.DataFrame(columns=["bout_id", "r_name", "b_name"])

    r_col, b_col = _resolve_name_columns(master)
    if r_col is None or b_col is None:
        return pd.DataFrame(columns=["bout_id", "r_name", "b_name"])

    master["fight_id"] = master["fight_id"].astype(str)
    names = (
        master[["fight_id", r_col, b_col]]
        .drop_duplicates("fight_id", keep="last")
        .rename(columns={"fight_id": "bout_id", r_col: "r_name", b_col: "b_name"})
    )
    return names


def _classify(row: pd.Series) -> str:
    winner_id = str(row["winner_id"])
    r_id = str(row["r_id"])
    b_id = str(row["b_id"])
    p_r = float(row["p_r_ko"])
    p_b = float(row["p_b_ko"])

    if np.isclose(p_r, p_b):
        return "TIE"
    predicted_id = r_id if p_r > p_b else b_id
    return "RIGHT" if predicted_id == winner_id else "WRONG"


def _add_direction_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["direction_result"] = out.apply(_classify, axis=1)

    def winner_corner(row: pd.Series) -> str:
        if str(row["winner_id"]) == str(row["r_id"]):
            return "R"
        if str(row["winner_id"]) == str(row["b_id"]):
            return "B"
        return "?"

    out["actual_winner_corner"] = out.apply(winner_corner, axis=1)
    out["actual_winner_name"] = np.where(
        out["actual_winner_corner"].eq("R"),
        out["r_name"],
        np.where(out["actual_winner_corner"].eq("B"), out["b_name"], out["winner_id"]),
    )
    out["actual_loser_name"] = np.where(
        out["actual_winner_corner"].eq("R"),
        out["b_name"],
        np.where(out["actual_winner_corner"].eq("B"), out["r_name"], "unknown"),
    )
    out["p_actual_winner_ko"] = np.where(
        out["actual_winner_corner"].eq("R"), out["p_r_ko"], out["p_b_ko"]
    )
    out["p_actual_loser_ko"] = np.where(
        out["actual_winner_corner"].eq("R"), out["p_b_ko"], out["p_r_ko"]
    )
    out["winner_minus_loser_ko_margin"] = (
        out["p_actual_winner_ko"] - out["p_actual_loser_ko"]
    )
    return out


def _print_subset(title: str, frame: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * len(title))
    if frame.empty:
        print("none")
        return

    display = frame[
        [
            "event_date",
            "actual_winner_name",
            "actual_loser_name",
            "p_actual_winner_ko",
            "p_actual_loser_ko",
            "winner_minus_loser_ko_margin",
            "bout_id",
        ]
    ].copy()
    display["event_date"] = pd.to_datetime(display["event_date"], errors="coerce").dt.date
    print(display.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="List missed historical R1 KO winner-direction calls")
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.audit.exists():
        raise FileNotFoundError(f"Saved recovery audit not found: {args.audit}")

    audit = pd.read_csv(args.audit).copy()
    required = {
        "bout_id", "event_date", "r_id", "b_id", "winner_id",
        "actual_ko_tko", "actual_finish_round", "p_r_ko", "p_b_ko",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"Audit is missing required columns: {missing}")

    for col in ("bout_id", "r_id", "b_id", "winner_id"):
        audit[col] = audit[col].astype(str)

    r1 = audit[
        audit["actual_ko_tko"].eq(1)
        & pd.to_numeric(audit["actual_finish_round"], errors="coerce").eq(1)
    ].copy()

    names = _master_names()
    if not names.empty:
        names["bout_id"] = names["bout_id"].astype(str)
        r1 = r1.merge(names, on="bout_id", how="left", validate="one_to_one")
    else:
        r1["r_name"] = r1["r_id"]
        r1["b_name"] = r1["b_id"]

    r1["r_name"] = r1["r_name"].fillna(r1["r_id"])
    r1["b_name"] = r1["b_name"].fillna(r1["b_id"])
    r1 = _add_direction_fields(r1)

    wrong = r1[r1["direction_result"].eq("WRONG")].copy()
    ties = r1[r1["direction_result"].eq("TIE")].copy()
    right = r1[r1["direction_result"].eq("RIGHT")].copy()

    # Put the largest misses first: most negative winner-minus-loser margin.
    wrong = wrong.sort_values(
        ["winner_minus_loser_ko_margin", "event_date"], ascending=[True, False]
    )
    ties = ties.sort_values("event_date", ascending=False)

    print("\n" + "=" * 118)
    print("HISTORICAL R1 KO/TKO — MC WINNER-DIRECTION MISSES")
    print("=" * 118)
    print(f"R1 KO/TKO bouts: {len(r1):,}")
    print(f"RIGHT: {len(right):,} ({len(right) / len(r1):.2%})")
    print(f"WRONG: {len(wrong):,} ({len(wrong) / len(r1):.2%})")
    print(f"TIE:   {len(ties):,} ({len(ties) / len(r1):.2%})")
    if len(right) + len(wrong):
        print(
            "non-tie direction hit rate: "
            f"{len(right) / (len(right) + len(wrong)):.2%}"
        )

    _print_subset("WRONG R1 KO CALLS", wrong)
    _print_subset("TIED R1 KO CALLS", ties)

    output_cols = [
        "bout_id", "event_date", "r_id", "b_id", "r_name", "b_name", "winner_id",
        "actual_winner_name", "actual_loser_name", "direction_result",
        "p_r_ko", "p_b_ko", "p_actual_winner_ko", "p_actual_loser_ko",
        "winner_minus_loser_ko_margin",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    r1[output_cols].sort_values(
        ["direction_result", "winner_minus_loser_ko_margin", "event_date"]
    ).to_csv(args.output, index=False)

    print(f"\nWrote all {len(r1):,} historical R1 KO/TKO direction rows to {args.output}")
    print("No simulator rerun and no FSR/simulator values were changed.")


if __name__ == "__main__":
    main()
