from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import AUDITS_DIR, MASTER_PATH
from pipeline.features.run_build_rolling_features import prepare_master_for_rolling

DEFAULT_OUTPUT_PATH = AUDITS_DIR / "ufc_fighter_rate_trace.parquet"
DEFAULT_PREVIEW_PATH = AUDITS_DIR / "ufc_fighter_rate_trace_preview.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace fighter rate calculations from master rows.")
    parser.add_argument("--fighter-name", default="Michael Chandler")
    parser.add_argument("--master-path", default=str(MASTER_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-path", default=str(DEFAULT_PREVIEW_PATH))
    return parser.parse_args()


def normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def side_row(row: pd.Series, side: str) -> dict[str, Any]:
    opp = "b" if side == "r" else "r"
    return {
        "date": row.get("date"),
        "event_name": row.get("event_name"),
        "fight_id": row.get("fight_id"),
        "fighter_name": row.get(f"{side}_name"),
        "opponent_name": row.get(f"{opp}_name"),
        "corner": "red" if side == "r" else "blue",
        "method": row.get("method"),
        "finish_round": row.get("finish_round"),
        "match_time_sec": row.get("match_time_sec"),
        "fighter_sig_str_landed": row.get(f"{side}_sig_str_landed"),
        "fighter_sig_str_attempted": row.get(f"{side}_sig_str_atmpted"),
        "opponent_sig_str_landed": row.get(f"{opp}_sig_str_landed"),
        "opponent_sig_str_attempted": row.get(f"{opp}_sig_str_atmpted"),
        "fighter_td_landed": row.get(f"{side}_td_landed"),
        "fighter_td_attempted": row.get(f"{side}_td_atmpted"),
        "opponent_td_landed": row.get(f"{opp}_td_landed"),
        "opponent_td_attempted": row.get(f"{opp}_td_atmpted"),
        "fighter_sub_att": row.get(f"{side}_sub_att"),
        "fighter_ctrl": row.get(f"{side}_ctrl"),
        "opponent_ctrl": row.get(f"{opp}_ctrl"),
    }


def main() -> None:
    args = parse_args()
    master_path = Path(args.master_path)
    if not master_path.exists():
        raise FileNotFoundError(f"Master file not found: {master_path}")

    master = pd.read_parquet(master_path)
    prepared = prepare_master_for_rolling(master)
    target = normalize(args.fighter_name)

    rows: list[dict[str, Any]] = []
    for _, row in prepared.iterrows():
        if normalize(row.get("r_name")) == target:
            rows.append(side_row(row, "r"))
        if normalize(row.get("b_name")) == target:
            rows.append(side_row(row, "b"))

    trace = pd.DataFrame(rows)
    if trace.empty:
        sample = prepared[[c for c in ["r_name", "b_name"] if c in prepared.columns]].head(20)
        raise ValueError(f"No fights found for {args.fighter_name}. Sample names:\n{sample.to_string(index=False)}")

    numeric_cols = [
        "match_time_sec",
        "fighter_sig_str_landed",
        "fighter_sig_str_attempted",
        "opponent_sig_str_landed",
        "opponent_sig_str_attempted",
        "fighter_td_landed",
        "fighter_td_attempted",
        "opponent_td_landed",
        "opponent_td_attempted",
        "fighter_sub_att",
        "fighter_ctrl",
        "opponent_ctrl",
    ]
    for col in numeric_cols:
        trace[col] = pd.to_numeric(trace[col], errors="coerce").fillna(0.0)

    trace = trace.sort_values(["date", "event_name", "fight_id"]).reset_index(drop=True)
    trace["fight_minutes"] = trace["match_time_sec"] / 60.0
    trace["splm_row"] = trace["fighter_sig_str_landed"] / trace["fight_minutes"].replace(0, pd.NA)
    trace["sapm_row"] = trace["opponent_sig_str_landed"] / trace["fight_minutes"].replace(0, pd.NA)
    trace["td_avg_row_per_15"] = trace["fighter_td_landed"] / (trace["match_time_sec"] / 900.0).replace(0, pd.NA)
    trace["sub_avg_row_per_15"] = trace["fighter_sub_att"] / (trace["match_time_sec"] / 900.0).replace(0, pd.NA)

    totals = {
        "fight_count": len(trace),
        "total_match_time_sec": float(trace["match_time_sec"].sum()),
        "total_fight_minutes": float(trace["fight_minutes"].sum()),
        "total_sig_str_landed": float(trace["fighter_sig_str_landed"].sum()),
        "total_sig_str_absorbed": float(trace["opponent_sig_str_landed"].sum()),
        "total_td_landed": float(trace["fighter_td_landed"].sum()),
        "total_sub_att": float(trace["fighter_sub_att"].sum()),
    }
    total_minutes = totals["total_fight_minutes"]
    total_fifteen_units = totals["total_match_time_sec"] / 900.0
    totals.update(
        {
            "computed_splm": totals["total_sig_str_landed"] / total_minutes if total_minutes else 0.0,
            "computed_sapm": totals["total_sig_str_absorbed"] / total_minutes if total_minutes else 0.0,
            "computed_td_avg_per_15": totals["total_td_landed"] / total_fifteen_units if total_fifteen_units else 0.0,
            "computed_sub_avg_per_15": totals["total_sub_att"] / total_fifteen_units if total_fifteen_units else 0.0,
        }
    )

    total_rows = pd.DataFrame([{"row_type": "TOTAL", **totals}])
    trace_out = trace.copy()
    trace_out.insert(0, "row_type", "FIGHT")

    output_path = Path(args.output_path)
    preview_path = Path(args.preview_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([trace_out, total_rows], ignore_index=True, sort=False)
    combined.to_parquet(output_path, index=False)
    combined.to_csv(preview_path, index=False)

    print("=" * 80)
    print("FIGHTER RATE TRACE")
    print("=" * 80)
    print("Fighter:", args.fighter_name)
    print("Master path:", master_path)
    print("Prepared rows:", len(prepared))
    print("Matched fights:", len(trace))
    print()
    print("========== FIGHT ROWS ==========")
    show_cols = [
        "date", "event_name", "fighter_name", "opponent_name", "method", "finish_round",
        "match_time_sec", "fight_minutes", "fighter_sig_str_landed", "opponent_sig_str_landed",
        "fighter_td_landed", "fighter_sub_att", "splm_row", "sapm_row", "td_avg_row_per_15",
    ]
    print(trace[[c for c in show_cols if c in trace.columns]].to_string(index=False))
    print()
    print("========== TOTALS ==========")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print()
    print("Saved trace:", output_path)
    print("Saved preview:", preview_path)


if __name__ == "__main__":
    main()
