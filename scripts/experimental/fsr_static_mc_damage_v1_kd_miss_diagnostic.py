"""Diagnose which actual-KD historical matchups the MC catches vs misses.

This script consumes:
- the 300-bout actual-vs-MC KD validation artifact;
- the leakage-safe FSR-28 pre-fight snapshots;
- the path-level 300-bout MC audit artifact.

It compares actual-KD bouts that the MC rated >=30% (caught) against actual-KD
bouts rated <30% (missed), focusing on whether misses are explained by:
- weak power vs knockdown-resistance matchup edges;
- low simulated significant-strike exposure;
- pressure / precision / defense traits;
- durability / reservoir capacity.

No simulator constants are changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage


ACTUAL_VS_MC_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.parquet"
)
PATH_AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_kd_audit.parquet"
)
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_kd_miss_diagnostic.parquet"
)
DEFAULT_THRESHOLD = 0.30

TRAITS = [
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "distance_striking_pressure",
    "distance_striking_precision",
    "distance_striking_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
]


def _load_validation(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "bout_id",
        "actual_any_kd",
        "actual_total_kd",
        "mc_p_any_kd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Validation artifact missing columns: {missing}")
    frame = frame.copy()
    frame["bout_id"] = frame["bout_id"].astype(str)
    return frame


def _load_path_exposure(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "bout_id",
        "red_fighter_id",
        "blue_fighter_id",
        "red_reservoir_fraction_end",
        "blue_reservoir_fraction_end",
        "red_knockdowns",
        "blue_knockdowns",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Path audit missing columns: {missing}")

    # The historical audit did not store sig landed directly, so exposure here
    # uses reservoir consumption and KD output from the exact same simulated paths.
    frame = frame.copy()
    frame["bout_id"] = frame["bout_id"].astype(str)
    return (
        frame.groupby("bout_id", as_index=False)
        .agg(
            mean_red_res_end=("red_reservoir_fraction_end", "mean"),
            mean_blue_res_end=("blue_reservoir_fraction_end", "mean"),
            mean_red_kd=("red_knockdowns", "mean"),
            mean_blue_kd=("blue_knockdowns", "mean"),
        )
    )


def _load_fsr(path: Path, selected_bouts: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    bout_key = "fight_id" if "fight_id" in frame.columns else "bout_id"
    if bout_key not in frame.columns:
        raise ValueError("FSR-28 artifact has neither fight_id nor bout_id.")

    required = {bout_key, "fighter_id", *TRAITS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR-28 artifact missing diagnostic columns: {missing}")

    frame = frame.copy()
    frame[bout_key] = frame[bout_key].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame = frame[frame[bout_key].isin(selected_bouts)].copy()
    frame = frame.rename(columns={bout_key: "bout_id"})

    counts = frame.groupby("bout_id")["fighter_id"].nunique()
    valid = set(counts.index[counts == 2])
    return frame[frame["bout_id"].isin(valid)].copy()


def _build_bout_features(fsr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bout_id, g in fsr.groupby("bout_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]

        # Acute matchup opportunity from both possible attacker-defender directions.
        edge_a = float(a["striking_power"] - b["knockdown_resistance"])
        edge_b = float(b["striking_power"] - a["knockdown_resistance"])

        row: dict[str, object] = {
            "bout_id": str(bout_id),
            "max_power_minus_opp_kd_res": max(edge_a, edge_b),
            "mean_power_minus_opp_kd_res": (edge_a + edge_b) / 2.0,
            "max_striking_power": max(float(a["striking_power"]), float(b["striking_power"])),
            "min_kd_resistance": min(float(a["knockdown_resistance"]), float(b["knockdown_resistance"])),
            "mean_damage_durability": np.mean([a["damage_durability"], b["damage_durability"]]),
            "max_distance_pressure": max(float(a["distance_striking_pressure"]), float(b["distance_striking_pressure"])),
            "mean_distance_pressure": np.mean([a["distance_striking_pressure"], b["distance_striking_pressure"]]),
            "max_distance_precision": max(float(a["distance_striking_precision"]), float(b["distance_striking_precision"])),
            "min_distance_defense": min(float(a["distance_striking_defense"]), float(b["distance_striking_defense"])),
            "max_clinch_pressure": max(float(a["clinch_striking_pressure"]), float(b["clinch_striking_pressure"])),
            "max_ground_pressure": max(float(a["ground_striking_pressure"]), float(b["ground_striking_pressure"])),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _effect_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in columns:
        caught = pd.to_numeric(frame.loc[frame["diagnostic_group"] == "caught", col], errors="coerce").dropna()
        missed = pd.to_numeric(frame.loc[frame["diagnostic_group"] == "missed", col], errors="coerce").dropna()
        if caught.empty or missed.empty:
            continue
        pooled = np.sqrt((caught.var(ddof=1) + missed.var(ddof=1)) / 2.0)
        effect = (caught.mean() - missed.mean()) / pooled if pooled > 0 else np.nan
        rows.append(
            {
                "feature": col,
                "caught_mean": caught.mean(),
                "missed_mean": missed.mean(),
                "difference": caught.mean() - missed.mean(),
                "std_effect_caught_minus_missed": effect,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "std_effect_caught_minus_missed", key=lambda s: s.abs(), ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose actual-KD matchups caught vs missed by the MC")
    parser.add_argument("--validation", type=Path, default=ACTUAL_VS_MC_PATH)
    parser.add_argument("--path-audit", type=Path, default=PATH_AUDIT_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation = _load_validation(args.validation)
    actual_kd = validation[validation["actual_any_kd"] == 1].copy()
    actual_kd["diagnostic_group"] = np.where(
        actual_kd["mc_p_any_kd"] >= args.threshold, "caught", "missed"
    )

    selected = set(actual_kd["bout_id"])
    fsr = _load_fsr(args.fsr_path, selected)
    features = _build_bout_features(fsr)
    exposure = _load_path_exposure(args.path_audit)

    merged = actual_kd.merge(features, on="bout_id", how="left", validate="one_to_one")
    merged = merged.merge(exposure, on="bout_id", how="left", validate="one_to_one")
    merged["mean_reservoir_depletion"] = 1.0 - (
        merged[["mean_red_res_end", "mean_blue_res_end"]].mean(axis=1)
    )
    merged["mc_expected_total_kd_from_paths"] = merged[["mean_red_kd", "mean_blue_kd"]].sum(axis=1)

    diagnostic_cols = [
        "max_power_minus_opp_kd_res",
        "mean_power_minus_opp_kd_res",
        "max_striking_power",
        "min_kd_resistance",
        "mean_damage_durability",
        "max_distance_pressure",
        "mean_distance_pressure",
        "max_distance_precision",
        "min_distance_defense",
        "max_clinch_pressure",
        "max_ground_pressure",
        "mean_reservoir_depletion",
        "mc_expected_total_kd_from_paths",
    ]

    effects = _effect_table(merged, diagnostic_cols)

    caught = merged[merged["diagnostic_group"] == "caught"]
    missed = merged[merged["diagnostic_group"] == "missed"]

    print("\n" + "=" * 120)
    print("ACTUAL-KD MATCHUP DIAGNOSTIC — MC CAUGHT VS MISSED")
    print("=" * 120)
    print(f"actual-KD fights: {len(merged)}")
    print(f"threshold: p(KD) >= {args.threshold:.0%}")
    print(f"caught: {len(caught)} ({len(caught)/len(merged):.2%})")
    print(f"missed: {len(missed)} ({len(missed)/len(merged):.2%})")

    print("\nFEATURE DIFFERENCES — SORTED BY ABSOLUTE STANDARDIZED EFFECT")
    print(effects.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n20 WORST MISSED ACTUAL-KD FIGHTS")
    display = [c for c in [
        "bout_id", "event_date", "red_name", "blue_name", "mc_p_any_kd",
        "actual_total_kd", "max_power_minus_opp_kd_res", "max_striking_power",
        "min_kd_resistance", "max_distance_pressure", "max_distance_precision",
        "min_distance_defense", "mean_reservoir_depletion",
    ] if c in merged.columns]
    print(
        missed.sort_values("mc_p_any_kd").head(20)[display].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n20 BEST-CAUGHT ACTUAL-KD FIGHTS")
    print(
        caught.sort_values("mc_p_any_kd", ascending=False).head(20)[display].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nINTERPRETATION")
    print("- Large trait effects imply the KD equation / matchup traits separate caught from missed fights.")
    print("- Large reservoir-depletion effects imply the MC is generating materially different damage exposure.")
    print("- If trait edges look similar but exposure differs, investigate phase/striking opportunity generation before retuning KD.")
    print("- If both are similar, the current FSR traits likely lack matchup-specific KD information.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"\n[KD miss diagnostic] wrote {args.output}")


if __name__ == "__main__":
    main()
