from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
VALIDATION_PATH = Path("data/audits/ufc_round_stats_validation.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate standalone UFC round-stats dataset.")
    p.add_argument("--round-stats-path", default=str(ROUND_STATS_PATH))
    p.add_argument("--output-path", default=str(VALIDATION_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ts = datetime.now(timezone.utc).isoformat()

    path = Path(args.round_stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Round stats dataset not found: {path}")

    df = pd.read_parquet(path)

    required = [
        "event_id", "fight_id", "fighter_id", "opponent_id",
        "round", "corner", "fighter_name", "opponent_name",
        "sig_str_landed", "sig_str_attempted",
        "total_str_landed", "total_str_attempted",
        "td_landed", "td_attempted",
        "ctrl_sec",
    ]

    missing_cols = [c for c in required if c not in df.columns]

    numeric_nonnegative = [
        c for c in [
            "round", "kd", "sig_str_landed", "sig_str_attempted",
            "total_str_landed", "total_str_attempted",
            "td_landed", "td_attempted",
            "sub_att", "rev", "ctrl_sec",
            "head_landed", "head_attempted",
            "body_landed", "body_attempted",
            "leg_landed", "leg_attempted",
            "distance_landed", "distance_attempted",
            "clinch_landed", "clinch_attempted",
            "ground_landed", "ground_attempted",
        ]
        if c in df.columns
    ]

    checks = []

    def add_check(name, passed, value, severity="hard"):
        checks.append({
            "run_timestamp": run_ts,
            "check_name": name,
            "passed": bool(passed),
            "value": value,
            "severity": severity,
        })

    add_check("required_columns_present", not missing_cols, ",".join(missing_cols))

    if missing_cols:
        out = pd.DataFrame(checks)
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.output_path, index=False)
        print(out.to_string(index=False))
        raise SystemExit(1)

    add_check("row_count_positive", len(df) > 0, len(df))
    add_check("unique_fighter_round_key", df.duplicated(["fight_id", "fighter_id", "round"]).sum() == 0, int(df.duplicated(["fight_id", "fighter_id", "round"]).sum()))
    add_check("unique_corner_round_key", df.duplicated(["fight_id", "corner", "round"]).sum() == 0, int(df.duplicated(["fight_id", "corner", "round"]).sum()))

    for col in ["event_id", "fight_id", "fighter_id", "opponent_id", "fighter_name", "opponent_name", "corner", "round"]:
        add_check(f"{col}_not_null", df[col].notna().all(), int(df[col].isna().sum()))

    add_check("corner_values_valid", df["corner"].astype(str).isin(["red", "blue"]).all(), sorted(df["corner"].astype(str).unique().tolist()))

    for col in numeric_nonnegative:
        s = pd.to_numeric(df[col], errors="coerce")
        add_check(f"{col}_numeric", s.notna().all(), int(s.isna().sum()))
        add_check(f"{col}_nonnegative", s.ge(0).all(), int((s < 0).sum()))

    landed_attempted_pairs = [
        ("sig_str_landed", "sig_str_attempted"),
        ("total_str_landed", "total_str_attempted"),
        ("td_landed", "td_attempted"),
        ("head_landed", "head_attempted"),
        ("body_landed", "body_attempted"),
        ("leg_landed", "leg_attempted"),
        ("distance_landed", "distance_attempted"),
        ("clinch_landed", "clinch_attempted"),
        ("ground_landed", "ground_attempted"),
    ]

    for landed, attempted in landed_attempted_pairs:
        if landed in df.columns and attempted in df.columns:
            bad = pd.to_numeric(df[landed], errors="coerce") > pd.to_numeric(df[attempted], errors="coerce")
            add_check(f"{landed}_lte_{attempted}", not bad.any(), int(bad.sum()))

    summary = pd.DataFrame(checks)
    summary["value"] = summary["value"].astype(str)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output_path, index=False)

    print("=" * 80)
    print("ROUND STATS VALIDATION")
    print("=" * 80)
    print("Rows:", len(df))
    print("Unique fights:", df["fight_id"].nunique())
    print("Output:", args.output_path)
    print()
    print(summary.to_string(index=False))

    hard_failures = summary[summary["severity"].eq("hard") & ~summary["passed"]]
    if not hard_failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
