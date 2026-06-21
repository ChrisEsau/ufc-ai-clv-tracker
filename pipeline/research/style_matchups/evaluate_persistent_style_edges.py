"""Evaluate persistent UFC style-matchup edges across eras.

Research-only runner. It consumes style_matchup_era_roi_report.csv produced by
pipeline.research.style_matchups.evaluate_style_eras and writes durability-ranked
style matchup reports under data/research/style_matchups/.

The goal is to find matchups that show repeatable signal across eras, not just
one high-ROI pocket in a small sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_MIN_ERAS = 2
DEFAULT_MODERN_ERA = "2021-present"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate persistent style matchup edges across eras.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-eras", type=int, default=DEFAULT_MIN_ERAS)
    parser.add_argument("--modern-era", default=DEFAULT_MODERN_ERA)
    parser.add_argument(
        "--require-modern",
        action="store_true",
        help="Only keep matchups that appear in the modern era.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style research config not found: {p}")
    config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {p}")
    return config


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def build_persistent_report(era_roi: pd.DataFrame, min_eras: int, modern_era: str, require_modern: bool) -> pd.DataFrame:
    required = {"era", "style_matchup_key", "fights", "red_win_rate", "flat_bet_red_roi", "underdog_roi"}
    missing = sorted(required.difference(era_roi.columns))
    if missing:
        raise ValueError(f"Era ROI report missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for matchup, group in era_roi.groupby("style_matchup_key", dropna=False):
        g = group.copy()
        g["fights"] = pd.to_numeric(g["fights"], errors="coerce").fillna(0)
        g["flat_bet_red_roi"] = pd.to_numeric(g["flat_bet_red_roi"], errors="coerce")
        g["underdog_roi"] = pd.to_numeric(g["underdog_roi"], errors="coerce")
        g["red_win_rate"] = pd.to_numeric(g["red_win_rate"], errors="coerce")

        eras_present = sorted(g["era"].dropna().astype(str).unique().tolist())
        modern = g[g["era"].astype(str).eq(modern_era)]
        has_modern = not modern.empty
        if len(eras_present) < min_eras:
            continue
        if require_modern and not has_modern:
            continue

        total_fights = int(g["fights"].sum())
        positive_roi_eras = int(g["flat_bet_red_roi"].gt(0).sum())
        positive_dog_roi_eras = int(g["underdog_roi"].gt(0).sum())
        all_positive_red_roi = positive_roi_eras == len(eras_present)
        all_positive_dog_roi = positive_dog_roi_eras == len(eras_present)

        modern_red_roi = float(modern["flat_bet_red_roi"].iloc[0]) if has_modern else float("nan")
        modern_dog_roi = float(modern["underdog_roi"].iloc[0]) if has_modern else float("nan")
        modern_fights = int(modern["fights"].iloc[0]) if has_modern else 0

        weighted_red_roi = weighted_mean(g["flat_bet_red_roi"], g["fights"])
        weighted_dog_roi = weighted_mean(g["underdog_roi"], g["fights"])
        weighted_red_win_rate = weighted_mean(g["red_win_rate"], g["fights"])
        min_red_roi = float(g["flat_bet_red_roi"].min())
        max_red_roi = float(g["flat_bet_red_roi"].max())

        durability_score = (
            weighted_red_roi
            * (len(eras_present) / 3.0)
            * (total_fights ** 0.5)
        )
        modern_bonus_score = durability_score
        if has_modern and pd.notna(modern_red_roi):
            modern_bonus_score += modern_red_roi * (modern_fights ** 0.5)

        rows.append({
            "style_matchup_key": matchup,
            "eras_present": "|".join(eras_present),
            "era_count": len(eras_present),
            "total_fights": total_fights,
            "weighted_red_win_rate": weighted_red_win_rate,
            "weighted_red_roi": weighted_red_roi,
            "min_red_roi": min_red_roi,
            "max_red_roi": max_red_roi,
            "positive_roi_eras": positive_roi_eras,
            "all_positive_red_roi": all_positive_red_roi,
            "modern_era": modern_era,
            "has_modern": has_modern,
            "modern_fights": modern_fights,
            "modern_red_roi": modern_red_roi,
            "weighted_underdog_roi": weighted_dog_roi,
            "positive_dog_roi_eras": positive_dog_roi_eras,
            "all_positive_dog_roi": all_positive_dog_roi,
            "modern_underdog_roi": modern_dog_roi,
            "durability_score": durability_score,
            "modern_bonus_score": modern_bonus_score,
        })

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(
        ["all_positive_red_roi", "has_modern", "modern_bonus_score", "weighted_red_roi", "total_fights"],
        ascending=[False, False, False, False, False],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    research_dir = Path(str((config.get("outputs") or {}).get("research_dir", "data/research/style_matchups")))
    era_report_path = research_dir / "style_matchup_era_roi_report.csv"
    if not era_report_path.exists():
        raise FileNotFoundError(
            f"Era ROI report not found: {era_report_path}. Run evaluate_style_eras first."
        )

    era_roi = pd.read_csv(era_report_path)
    persistent = build_persistent_report(
        era_roi,
        min_eras=args.min_eras,
        modern_era=args.modern_era,
        require_modern=args.require_modern,
    )

    persistent_path = research_dir / "style_matchup_persistent_edges.csv"
    modern_path = research_dir / "style_matchup_persistent_modern_edges.csv"
    summary_path = research_dir / "style_matchup_persistent_summary.json"

    persistent.to_csv(persistent_path, index=False)
    modern = persistent[persistent["has_modern"].astype(bool)].copy() if not persistent.empty else persistent
    modern.to_csv(modern_path, index=False)

    summary = {
        "input_era_report": str(era_report_path),
        "min_eras": int(args.min_eras),
        "modern_era": args.modern_era,
        "require_modern": bool(args.require_modern),
        "era_report_rows": int(len(era_roi)),
        "persistent_rows": int(len(persistent)),
        "modern_persistent_rows": int(len(modern)),
        "top_persistent": persistent.head(10).to_dict(orient="records") if not persistent.empty else [],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved persistent edge report       : {persistent_path}")
    print(f"Saved modern persistent edge report: {modern_path}")
    print(f"Saved persistent summary          : {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
