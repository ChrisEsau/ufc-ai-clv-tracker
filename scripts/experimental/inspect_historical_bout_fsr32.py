"""Inspect the exact leakage-safe FSR-32 profiles used for one historical bout."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_OUTPUT_DIR = Path("data/experimental/single_historical_ko_bout")

HIGHLIGHT_TRAITS = [
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "stamina_capacity",
    "stamina_depletion_resistance",
    "stamina_performance_resilience",
    "distance_striking_pressure",
    "distance_striking_precision",
    "distance_striking_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
    "wrestling_entry",
    "wrestling_conversion",
    "td_defense",
    "control_imposition",
    "control_resistance",
    "submission_pressure",
    "submission_conversion",
    "submission_resistance",
    "reversal_ability",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect exact historical FSR-32 inputs for one bout")
    p.add_argument("--bout-id", required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _select(bout_id: str):
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    selected = cohort.loc[cohort["bout_id"].eq(str(bout_id))]
    if selected.empty:
        raise ValueError(f"bout_id {bout_id} is not in aligned mature 2020+ FSR-32 cohort")
    if len(selected) != 1:
        raise ValueError(f"expected one row for {bout_id}, found {len(selected)}")
    bout = selected.iloc[0]
    red, blue = pairs[str(bout_id)]
    if str(red["fighter_id"]) != str(bout["r_id"]) or str(blue["fighter_id"]) != str(bout["b_id"]):
        raise ValueError("FSR pair is not aligned to historical red/blue corners")
    return bout, red, blue


def _numeric_fsr_columns(red: pd.Series, blue: pd.Series) -> list[str]:
    metadata = {
        "fight_id", "bout_id", "fighter_id", "fighter_name", "name", "fighter",
        "event_date", "fight_date", "date", "opponent_id", "opponent_name",
    }
    cols: list[str] = []
    for col in red.index:
        if col in metadata or col not in blue.index:
            continue
        r = pd.to_numeric(pd.Series([red[col]]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([blue[col]]), errors="coerce").iloc[0]
        if pd.notna(r) or pd.notna(b):
            cols.append(col)
    return cols


def main() -> None:
    args = parse_args()
    bout_id = str(args.bout_id)
    bout, red, blue = _select(bout_id)

    red_name = base._display_name(red)
    blue_name = base._display_name(blue)

    print("\n" + "=" * 108)
    print("HISTORICAL BOUT FSR-32 INPUT AUDIT — EXACT PREFIGHT SNAPSHOTS")
    print("=" * 108)
    print(f"bout_id: {bout_id}")
    print(f"RED : {red_name} [{red['fighter_id']}]")
    print(f"BLUE: {blue_name} [{blue['fighter_id']}]")
    if "event_date" in bout.index:
        print(f"event_date: {bout['event_date']}")
    print("\nKO / DAMAGE / STAMINA HIGHLIGHTS")
    print(f"{'trait':36s} {'RED':>12s} {'BLUE':>12s} {'red-blue':>12s}")
    print("-" * 76)
    rows = []
    for trait in HIGHLIGHT_TRAITS:
        if trait not in red.index or trait not in blue.index:
            continue
        rv = pd.to_numeric(pd.Series([red[trait]]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([blue[trait]]), errors="coerce").iloc[0]
        if pd.isna(rv) and pd.isna(bv):
            continue
        diff = rv - bv if pd.notna(rv) and pd.notna(bv) else float("nan")
        print(f"{trait:36s} {rv:12.4f} {bv:12.4f} {diff:12.4f}")
        rows.append({"trait": trait, "red": rv, "blue": bv, "red_minus_blue": diff})

    out_dir = args.output_dir / bout_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "fsr32_highlights.csv", index=False)

    all_cols = _numeric_fsr_columns(red, blue)
    all_rows = []
    for trait in all_cols:
        rv = pd.to_numeric(pd.Series([red[trait]]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([blue[trait]]), errors="coerce").iloc[0]
        diff = rv - bv if pd.notna(rv) and pd.notna(bv) else float("nan")
        all_rows.append({"trait": trait, "red": rv, "blue": bv, "red_minus_blue": diff})
    pd.DataFrame(all_rows).to_csv(out_dir / "fsr32_all_numeric.csv", index=False)

    print(f"\nhighlights: {out_dir / 'fsr32_highlights.csv'}")
    print(f"all numeric FSR/profile fields: {out_dir / 'fsr32_all_numeric.csv'}")
    print("Leakage-safe prefight rows only; current-fight outcome is not used in these inputs.")


if __name__ == "__main__":
    main()
