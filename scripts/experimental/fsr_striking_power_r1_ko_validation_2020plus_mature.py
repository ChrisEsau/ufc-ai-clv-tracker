"""Validate striking_power FSR directly against modern mature Round-1 KO outcomes.

Primary cohort contract
-----------------------
- UFC bouts dated 2020-01-01 or later.
- Both fighters had at least 3 prior UFC fights before the bout.
- Leakage-safe pre-fight FSR snapshots only.
- Primary directional analysis: actual Round-1 KO/TKO bouts only.

Questions
---------
1. In actual R1 KO bouts, does the eventual winner have higher pre-fight
   striking_power than the loser?
2. Across all eligible fighter-sides, does striking_power discriminate the
   fighter who will score an R1 KO?
3. Across all eligible bouts, does the stronger of the two power ratings
   discriminate whether an R1 KO occurs at all?
4. Is the trait compressed enough that useful ranking signal may still be hard
   for the simulator to translate?

Diagnostic only: no FSR or simulator constants are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_striking_power_r1_ko_validation_2020plus_mature.parquet"
)


def _safe_auc(y: pd.Series, score: pd.Series) -> float:
    mask = y.notna() & score.notna()
    yv = y.loc[mask].astype(int)
    sv = score.loc[mask].astype(float)
    if yv.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(yv, sv))


def _power(row: pd.Series) -> float:
    return float(pd.to_numeric(pd.Series([row.get("striking_power")]), errors="coerce").iloc[0])


def _attach_result_metadata(cohort: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["fight_id", "winner_id", "r_id", "b_id", "r_name", "b_name"] if c in master.columns]
    required = {"fight_id", "winner_id", "r_id", "b_id"}
    missing = sorted(required - set(keep))
    if missing:
        raise ValueError(f"UFC master missing required result columns: {missing}")

    meta = master[keep].copy()
    meta["fight_id"] = meta["fight_id"].astype(str)
    meta["r_id"] = meta["r_id"].astype(str)
    meta["b_id"] = meta["b_id"].astype(str)
    meta = meta.drop_duplicates("fight_id", keep="last")

    out = cohort.merge(
        meta,
        left_on="bout_id",
        right_on="fight_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_master"),
    )
    out["winner_id"] = out["winner_id"].where(out["winner_id"].notna(), None)
    return out


def _build_frame(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, bout in cohort.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        red_power = _power(red)
        blue_power = _power(blue)
        red_id = str(bout["r_id"])
        blue_id = str(bout["b_id"])
        winner_id = None if pd.isna(bout.get("winner_id")) else str(bout.get("winner_id"))

        winner_side = None
        if winner_id == red_id:
            winner_side = "red"
        elif winner_id == blue_id:
            winner_side = "blue"

        winner_power = np.nan
        loser_power = np.nan
        if winner_side == "red":
            winner_power, loser_power = red_power, blue_power
        elif winner_side == "blue":
            winner_power, loser_power = blue_power, red_power

        rows.append(
            {
                "bout_id": bout_id,
                "event_date": bout["event_date"],
                "actual_r1_ko": int(bout["actual_r1_ko"]),
                "actual_ko_tko": int(bout["actual_ko_tko"]),
                "r_id": red_id,
                "b_id": blue_id,
                "winner_id": winner_id,
                "winner_side": winner_side,
                "r_name": bout.get("r_name", None),
                "b_name": bout.get("b_name", None),
                "red_power": red_power,
                "blue_power": blue_power,
                "max_power": max(red_power, blue_power),
                "min_power": min(red_power, blue_power),
                "absolute_power_gap": abs(red_power - blue_power),
                "winner_power": winner_power,
                "loser_power": loser_power,
                "winner_minus_loser_power": winner_power - loser_power if pd.notna(winner_power) else np.nan,
                "r_prior_ufc_fights": int(bout["r_prior_ufc_fights"]),
                "b_prior_ufc_fights": int(bout["b_prior_ufc_fights"]),
            }
        )

    return pd.DataFrame(rows)


def _side_level(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, bout in frame.iterrows():
        for side in ("red", "blue"):
            fighter_id = str(bout[f"{side[0]}_id"])
            is_winner = bout["winner_side"] == side
            rows.append(
                {
                    "bout_id": bout["bout_id"],
                    "event_date": bout["event_date"],
                    "side": side,
                    "fighter_id": fighter_id,
                    "power": float(bout[f"{side}_power"]),
                    "actual_r1_ko_bout": int(bout["actual_r1_ko"]),
                    "is_r1_ko_winner": int(bool(bout["actual_r1_ko"]) and is_winner),
                }
            )
    return pd.DataFrame(rows)


def _print_quantiles(label: str, values: pd.Series) -> None:
    q = values.dropna().quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    print(f"\n{label}")
    print(q.rename_axis("quantile").reset_index(name="power").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _print_summary(frame: pd.DataFrame) -> None:
    primary = frame[frame["actual_r1_ko"].eq(1)].copy()
    resolved = primary.dropna(subset=["winner_power", "loser_power"]).copy()
    sides = _side_level(frame)

    print("\n" + "=" * 116)
    print("STRIKING_POWER FSR VALIDATION — 2020+ MATURE ROUND-1 KO/TKO")
    print("=" * 116)
    print(f"eligible bouts: {len(frame):,}")
    print(f"actual R1 KO bouts: {len(primary):,} ({len(primary)/len(frame):.2%})")
    print(f"R1 KO bouts with resolved winner: {len(resolved):,}")
    print(f"date range: {frame['event_date'].min().date()} -> {frame['event_date'].max().date()}")

    print("\nPRIMARY — ACTUAL R1 KO WINNER VS LOSER POWER")
    print(f"winner power mean:   {resolved['winner_power'].mean():.3f}")
    print(f"winner power median: {resolved['winner_power'].median():.3f}")
    print(f"loser power mean:    {resolved['loser_power'].mean():.3f}")
    print(f"loser power median:  {resolved['loser_power'].median():.3f}")
    print(f"mean winner-minus-loser edge: {resolved['winner_minus_loser_power'].mean():+.3f}")
    print(f"median winner-minus-loser edge: {resolved['winner_minus_loser_power'].median():+.3f}")
    print(f"winner higher power: {(resolved['winner_minus_loser_power'] > 0).mean():.2%}")
    print(f"winner equal power:  {(resolved['winner_minus_loser_power'] == 0).mean():.2%}")
    print(f"winner lower power:  {(resolved['winner_minus_loser_power'] < 0).mean():.2%}")

    _print_quantiles("R1 KO WINNER POWER QUANTILES", resolved["winner_power"])
    _print_quantiles("R1 KO LOSER POWER QUANTILES", resolved["loser_power"])

    # Side-level test: can raw fighter power identify which fighter will score an R1 KO?
    side_auc = _safe_auc(sides["is_r1_ko_winner"], sides["power"])
    print("\nSIDE-LEVEL R1-KO-WINNER SIGNAL")
    print(f"fighter-sides: {len(sides):,}")
    print(f"R1 KO winner sides: {int(sides['is_r1_ko_winner'].sum()):,}")
    print(f"AUC using striking_power alone: {side_auc:.4f}")
    print(
        f"mean power, R1 KO winner sides: "
        f"{sides.loc[sides['is_r1_ko_winner'].eq(1), 'power'].mean():.3f}"
    )
    print(
        f"mean power, all other fighter-sides: "
        f"{sides.loc[sides['is_r1_ko_winner'].eq(0), 'power'].mean():.3f}"
    )

    # Bout-level test: can the strongest power rating in a matchup identify R1 KO occurrence?
    bout_auc_max = _safe_auc(frame["actual_r1_ko"], frame["max_power"])
    bout_auc_gap = _safe_auc(frame["actual_r1_ko"], frame["absolute_power_gap"])
    print("\nBOUT-LEVEL R1 KO OCCURRENCE SIGNAL")
    print(f"AUC using max(red_power, blue_power): {bout_auc_max:.4f}")
    print(f"AUC using absolute power gap:         {bout_auc_gap:.4f}")
    print(
        f"mean max power, R1 KO bouts: "
        f"{frame.loc[frame['actual_r1_ko'].eq(1), 'max_power'].mean():.3f}"
    )
    print(
        f"mean max power, non-R1-KO bouts: "
        f"{frame.loc[frame['actual_r1_ko'].eq(0), 'max_power'].mean():.3f}"
    )

    # Empirical R1-KO-winner rate by power quintile over all fighter sides.
    valid = sides.dropna(subset=["power"]).copy()
    valid["power_quintile"] = pd.qcut(valid["power"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    by_q = (
        valid.groupby("power_quintile", observed=True)
        .agg(
            fighter_sides=("fighter_id", "size"),
            power_mean=("power", "mean"),
            r1_ko_winner_sides=("is_r1_ko_winner", "sum"),
            r1_ko_winner_rate=("is_r1_ko_winner", "mean"),
        )
        .reset_index()
    )
    print("\nR1 KO WINNER RATE BY STRIKING_POWER QUINTILE")
    print(by_q.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLOWEST 15 WINNER POWER EDGES AMONG ACTUAL R1 KOs")
    cols = ["event_date", "bout_id", "r_name", "b_name", "winner_side", "winner_power", "loser_power", "winner_minus_loser_power"]
    print(
        resolved.sort_values("winner_minus_loser_power").head(15)[cols]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nHIGHEST 15 WINNER POWER EDGES AMONG ACTUAL R1 KOs")
    print(
        resolved.sort_values("winner_minus_loser_power", ascending=False).head(15)[cols]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nDECISION GUIDE")
    print("- Winner > loser materially above 50% plus positive side-level AUC -> power FSR has directional value.")
    print("- Bout max-power AUC > 0.50 -> power also helps identify explosive R1-KO matchups.")
    print("- Useful ranking with narrow numeric spread -> simulator mapping may be too insensitive to FSR differences.")
    print("- Weak ranking and narrow spread -> revisit the striking_power FSR construction before changing the tail curve.")
    print("- No FSR values or simulator constants are changed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate striking_power FSR against 2020+ mature R1 KO outcomes")
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=modern.FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    master = modern._load_master(args.master)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(args.fsr_path, candidate)
    cohort = _attach_result_metadata(cohort, master)

    print(
        f"[power R1 validation] eligible={len(cohort):,}; "
        f"actual_R1_KO={int(cohort['actual_r1_ko'].sum()):,}; "
        f"date={cohort['event_date'].min().date()} -> {cohort['event_date'].max().date()}",
        flush=True,
    )

    frame = _build_frame(cohort, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame)
    print(f"\n[power R1 validation] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
