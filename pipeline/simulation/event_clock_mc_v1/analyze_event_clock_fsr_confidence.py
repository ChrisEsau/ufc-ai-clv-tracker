from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH


DEFAULT_FULL_CREDIT_FIGHTS = 8.0


def _norm_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tier(score: float) -> str:
    # First-pass evidence tiers. These are diagnostic only and do not alter
    # Event Clock probabilities or betting decisions.
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    if score >= 0.25:
        return "Low"
    return "Very Low"


def build_prior_ufc_fight_counts(master: pd.DataFrame) -> pd.DataFrame:
    """Return leakage-safe prior UFC fight counts for each fighter-fight row.

    Counts are based only on completed master fights strictly before the target
    event date. Fights on the same date do not count toward one another.
    """
    m = master.drop_duplicates("fight_id").copy()
    m["fight_id"] = m["fight_id"].astype(str)
    date_col = "date" if "date" in m.columns else "event_date"
    m["event_date"] = pd.to_datetime(m[date_col], errors="raise").dt.normalize()

    rows: list[dict] = []
    for _, r in m.iterrows():
        for side, name_col in (("red", "r_name"), ("blue", "b_name")):
            rows.append(
                {
                    "fight_id": str(r["fight_id"]),
                    "event_date": r["event_date"],
                    "side": side,
                    "fighter_name": r[name_col],
                    "fighter_key": _norm_name(r[name_col]),
                }
            )
    ff = pd.DataFrame(rows).sort_values(["event_date", "fight_id", "side"]).reset_index(drop=True)

    # Same-date fights are deliberately excluded from prior history by counting
    # only dates strictly less than the current event date.
    date_counts = (
        ff.groupby(["fighter_key", "event_date"], as_index=False)
        .size()
        .rename(columns={"size": "fights_on_date"})
        .sort_values(["fighter_key", "event_date"])
    )
    date_counts["prior_ufc_fights"] = (
        date_counts.groupby("fighter_key")["fights_on_date"].cumsum()
        - date_counts["fights_on_date"]
    )
    ff = ff.merge(
        date_counts[["fighter_key", "event_date", "prior_ufc_fights"]],
        on=["fighter_key", "event_date"],
        how="left",
        validate="many_to_one",
    )
    return ff[["fight_id", "side", "fighter_name", "prior_ufc_fights"]]


def enrich_summary(summary: pd.DataFrame, prior: pd.DataFrame, full_credit_fights: float) -> pd.DataFrame:
    out = summary.copy()
    out["fight_id"] = out["fight_id"].astype(str)

    p = prior.pivot(index="fight_id", columns="side", values="prior_ufc_fights").reset_index()
    p = p.rename(columns={"red": "red_prior_ufc_fights", "blue": "blue_prior_ufc_fights"})
    out = out.merge(p, on="fight_id", how="left", validate="one_to_one")

    for side in ("red", "blue"):
        col = f"{side}_prior_ufc_fights"
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        out[f"{side}_fsr_confidence"] = np.clip(out[col].astype(float) / full_credit_fights, 0.0, 1.0)

    # The less-established fighter determines fight-level evidence confidence.
    out["fight_fsr_confidence"] = out[["red_fsr_confidence", "blue_fsr_confidence"]].min(axis=1)
    out["fight_fsr_confidence_pct"] = 100.0 * out["fight_fsr_confidence"]
    out["confidence_tier"] = out["fight_fsr_confidence"].map(_tier)
    out["min_prior_ufc_fights"] = out[["red_prior_ufc_fights", "blue_prior_ufc_fights"]].min(axis=1)
    return out


def _safe_mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def print_audit(combined: pd.DataFrame) -> None:
    print("=" * 132)
    print("EVENT CLOCK MC — FSR EVIDENCE CONFIDENCE AUDIT")
    print("=" * 132)
    print(f"fights: {len(combined)}")
    print("confidence definition: min(red prior UFC fights, blue prior UFC fights) / 8, capped at 1.0")
    print("prediction probabilities changed: NO")

    cols = [
        c for c in (
            "red", "blue", "red_prior_ufc_fights", "blue_prior_ufc_fights",
            "fight_fsr_confidence_pct", "confidence_tier", "ml_correct",
            "method_correct", "winner_method_correct",
        ) if c in combined.columns
    ]
    print("\nFIGHTS")
    print(combined[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n" + "-" * 132)
    print("PERFORMANCE BY CONFIDENCE TIER")
    print("-" * 132)
    order = ["Very Low", "Low", "Medium", "High"]
    for tier in order:
        g = combined[combined["confidence_tier"] == tier]
        if g.empty:
            continue
        ml = _safe_mean(g, "ml_correct")
        meth = _safe_mean(g, "method_correct")
        joint = _safe_mean(g, "winner_method_correct")
        print(
            f"{tier:<10} fights={len(g):>3}  "
            f"min-history mean={g['min_prior_ufc_fights'].mean():.2f}  "
            f"ML={ml:.1%}  method={meth:.1%}  joint={joint:.1%}"
        )

    print("\n" + "-" * 132)
    print("PERFORMANCE BY MINIMUM PRIOR UFC FIGHTS")
    print("-" * 132)
    bins = [(-1, 1, "0-1"), (1, 3, "2-3"), (3, 5, "4-5"), (5, np.inf, "6+")]
    for lo, hi, label in bins:
        if np.isinf(hi):
            g = combined[combined["min_prior_ufc_fights"] > lo]
        else:
            g = combined[(combined["min_prior_ufc_fights"] > lo) & (combined["min_prior_ufc_fights"] <= hi)]
        if g.empty:
            continue
        print(
            f"{label:<4} fights={len(g):>3}  "
            f"ML={_safe_mean(g, 'ml_correct'):.1%}  "
            f"method={_safe_mean(g, 'method_correct'):.1%}  "
            f"joint={_safe_mean(g, 'winner_method_correct'):.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add leakage-safe FSR evidence confidence to Event Clock summary CSVs and audit accuracy by history depth."
    )
    parser.add_argument("--summary", nargs="+", required=True, type=Path)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--full-credit-fights", type=float, default=DEFAULT_FULL_CREDIT_FIGHTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v1/event_predictions/event_clock_fsr_confidence_audit.csv"),
    )
    args = parser.parse_args()

    if args.full_credit_fights <= 0:
        raise ValueError("--full-credit-fights must be > 0")

    master = pd.read_parquet(args.master)
    prior = build_prior_ufc_fight_counts(master)

    frames = []
    for path in args.summary:
        s = pd.read_csv(path)
        enriched = enrich_summary(s, prior, args.full_credit_fights)
        enriched["source_file"] = str(path)
        frames.append(enriched)

        # Preserve the original simulation artifact and write a confidence-enriched sibling.
        enriched_path = path.with_name(path.stem + "_with_fsr_confidence.csv")
        enriched.to_csv(enriched_path, index=False)
        print(f"confidence summary: {enriched_path}")

    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print_audit(combined)
    print(f"\ncombined audit CSV: {args.output}")


if __name__ == "__main__":
    main()
