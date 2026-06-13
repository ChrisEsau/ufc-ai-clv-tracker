from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, CURRENT_FIGHTER_FEATURES_PATH

DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_fighter_state_profile_audit.parquet"
DEFAULT_PREVIEW_PATH = AUDITS_DIR / "ufc_fighter_state_profile_audit_preview.csv"

KEY_COLUMNS = [
    "fighter_id",
    "fighter_name",
    "name",
    "wins",
    "losses",
    "fights",
    "elo",
    "avg_opponent_elo",
    "best_win_elo",
    "worst_loss_elo",
    "win_pct",
    "splm",
    "sapm",
    "str_acc",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
    "ctrl_per_min",
    "ctrl_against_per_min",
    "finish_rate",
    "ko_rate",
    "sub_win_rate",
    "decision_win_rate",
    "finish_loss_rate",
    "decision_loss_rate",
    "avg_fight_time",
    "win_streak",
    "loss_streak",
    "days_since_last_fight",
    "recent_win_pct",
    "recent_splm",
    "recent_sapm",
    "recent_td_avg",
    "recent_finish_rate",
    "recent_avg_fight_time",
    "ewm_elo",
    "ewm_fights",
    "ewm_wins",
    "ewm_losses",
    "ewm_win_pct",
    "ewm_splm",
    "ewm_sapm",
    "ewm_str_acc",
    "ewm_str_def",
    "ewm_td_avg",
    "ewm_td_acc",
    "ewm_td_def",
    "ewm_sub_avg",
    "ewm_finish_rate",
    "ewm_ko_rate",
    "ewm_sub_win_rate",
    "ewm_decision_win_rate",
    "ewm_avg_opponent_elo",
    "ewm_best_win_elo",
    "ewm_recent_win_pct",
    "ewm_recent_splm",
    "ewm_recent_sapm",
    "ewm_recent_td_avg",
    "ewm_recent_finish_rate",
    "ewm_recent_avg_fight_time",
]

UFCSTATS_LIKE_COLUMNS = [
    "wins",
    "losses",
    "fights",
    "splm",
    "sapm",
    "str_acc",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and save selected fighter-state profiles.")
    parser.add_argument("--fighter-a", default="Michael Chandler")
    parser.add_argument("--fighter-b", default="Mauricio Ruffy")
    parser.add_argument("--fighter-state-path", default=str(CURRENT_FIGHTER_FEATURES_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-path", default=str(DEFAULT_PREVIEW_PATH))
    parser.add_argument("--include-all-columns", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def find_name_column(df: pd.DataFrame) -> str:
    for column in ["fighter_name", "name", "r_name", "b_name"]:
        if column in df.columns:
            return column
    raise ValueError("Fighter state file has no recognizable fighter-name column.")


def select_fighters(df: pd.DataFrame, fighter_names: list[str]) -> pd.DataFrame:
    name_col = find_name_column(df)
    names_norm = [normalize_text(name) for name in fighter_names]
    name_series = df[name_col].astype(str).str.strip().str.lower()
    exact = df[name_series.isin(names_norm)].copy()
    if len(exact) == len(fighter_names):
        return exact

    pieces = [exact]
    found_norm = set(exact[name_col].astype(str).str.strip().str.lower()) if not exact.empty else set()
    for target_raw, target_norm in zip(fighter_names, names_norm):
        if target_norm in found_norm:
            continue
        contains = df[name_series.str.contains(target_norm, na=False)].copy()
        if contains.empty:
            tokens = [token for token in target_norm.split() if token]
            if tokens:
                mask = pd.Series(True, index=df.index)
                for token in tokens:
                    mask &= name_series.str.contains(token, na=False)
                contains = df[mask].copy()
        pieces.append(contains.head(5))

    out = pd.concat(pieces, ignore_index=True).drop_duplicates()
    return out


def build_long_audit(selected: pd.DataFrame, include_all_columns: bool) -> pd.DataFrame:
    name_col = find_name_column(selected)
    columns = list(selected.columns) if include_all_columns else [c for c in KEY_COLUMNS if c in selected.columns]
    rows = []
    for _, row in selected.iterrows():
        for column in columns:
            raw_value = row.get(column)
            numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            rows.append({
                "fighter_name": str(row.get(name_col, "")),
                "fighter_id": str(row.get("fighter_id", row.get("id", row.get("ufcstats_fighter_id", "")))),
                "metric": str(column),
                "value": "" if pd.isna(raw_value) else str(raw_value),
                "numeric_value": None if pd.isna(numeric_value) else float(numeric_value),
                "is_ufcstats_like_metric": column in UFCSTATS_LIKE_COLUMNS,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    state_path = Path(args.fighter_state_path)
    if not state_path.exists():
        raise FileNotFoundError(f"Fighter state file not found: {state_path}")

    state = pd.read_parquet(state_path)
    selected = select_fighters(state, [args.fighter_a, args.fighter_b])
    if selected.empty:
        name_col = find_name_column(state)
        preview_names = state[name_col].dropna().astype(str).sort_values().head(30).tolist()
        raise ValueError(
            "No requested fighters found in fighter state file. "
            f"Requested: {[args.fighter_a, args.fighter_b]}. "
            f"Name column: {name_col}. Sample names: {preview_names}"
        )

    audit = build_long_audit(selected, args.include_all_columns)
    output_path = Path(args.output_path)
    preview_path = Path(args.preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output_path, index=False)
    audit.to_csv(preview_path, index=False)

    wide_cols = [c for c in KEY_COLUMNS if c in selected.columns]
    name_col = find_name_column(selected)
    wide = selected[[c for c in [name_col, "fighter_id", "id", "ufcstats_fighter_id", *wide_cols] if c in selected.columns]].copy()
    wide = wide.loc[:, ~wide.columns.duplicated()].copy()

    print("=" * 80)
    print("FIGHTER STATE PROFILE AUDIT")
    print("=" * 80)
    print("Fighter state path:", state_path)
    print("Requested fighter A:", args.fighter_a)
    print("Requested fighter B:", args.fighter_b)
    print("Matched rows:", len(selected))
    print("State shape:", state.shape)
    print()
    print("========== WIDE PROFILE ==========")
    print(wide.to_string(index=False))
    print()
    print("========== TRANSPOSED PROFILE ==========")
    transpose_cols = [c for c in wide.columns if c != name_col]
    if len(selected) <= 5:
        temp = wide.set_index(name_col)[transpose_cols].T
        print(temp.to_string())
    print()
    print("========== UFCSTATS-LIKE METRICS ONLY ==========")
    ufc_cols = [c for c in [name_col, *UFCSTATS_LIKE_COLUMNS] if c in selected.columns]
    print(selected[ufc_cols].to_string(index=False))
    print()
    print("Saved audit:", output_path)
    print("Saved preview:", preview_path)


if __name__ == "__main__":
    main()
