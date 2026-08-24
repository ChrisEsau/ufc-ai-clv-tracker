"""Diagnose why Event Clock V2 overvalues bantamweight underdogs.

Measurement only. No simulator or FSR changes.

Uses the completed i10_b0 market audit plus frozen FSR V3 prefight snapshots to
separate upstream trait compression from downstream MC probability compression.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import load_prefight_snapshots, historical_fighter_rows


def _side_value(row: pd.Series, side: str, suffix: str) -> float:
    return float(row[f"p_{side}_{suffix}"])


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    return float(x[m].corr(y[m])) if int(m.sum()) >= 5 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bet-audit-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    bets = pd.read_csv(args.bet_audit_path)
    ml = bets[bets["market_key"].eq("moneyline")].copy()
    ml["fight_id"] = ml["fight_id"].astype(str)
    ml["event_date"] = pd.to_datetime(ml["event_date"], errors="raise").dt.normalize()

    # One row per priced fight, explicitly orienting all fields favorite -> dog.
    rows = []
    for fight_id, g in ml.groupby("fight_id", sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("market_fair_probability", ascending=False)
        fav = g.iloc[0]
        dog = g.iloc[-1]
        fav_side = str(fav["outcome_side"])
        dog_side = str(dog["outcome_side"])
        rows.append({
            "fight_id": fight_id,
            "event_date": fav["event_date"],
            "red": fav["red"],
            "blue": fav["blue"],
            "favorite": fav["outcome_label"],
            "underdog": dog["outcome_label"],
            "favorite_side": fav_side,
            "underdog_side": dog_side,
            "market_favorite_fair_p": float(fav["market_fair_probability"]),
            "mc_favorite_p": float(fav["model_probability"]),
            "favorite_won": int(bool(fav["won"])),
            "market_dog_fair_p": float(dog["market_fair_probability"]),
            "mc_dog_p": float(dog["model_probability"]),
            "dog_won": int(bool(dog["won"])),
            "dog_dec_p": _side_value(dog, dog_side, "dec"),
            "dog_ko_p": _side_value(dog, dog_side, "ko_tko"),
            "dog_sub_p": _side_value(dog, dog_side, "sub"),
            "actual_method": dog["actual_method"],
            "red_prior_ufc_fights": fav["red_prior_ufc_fights"],
            "blue_prior_ufc_fights": fav["blue_prior_ufc_fights"],
            "fight_evidence_bucket": fav["fight_evidence_bucket"],
        })
    fights = pd.DataFrame(rows)
    fights["compression_pp"] = 100.0 * (fights["market_favorite_fair_p"] - fights["mc_favorite_p"])

    # Attach frozen prefight FSR rows and orient numeric trait deltas favorite - dog.
    fsr = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["event_date"] = pd.to_datetime(fsr["event_date"], errors="raise").dt.normalize()

    id_like = {"fight_id", "fighter_id", "event_date", "fighter_name", "name", "side"}
    numeric_cols = [c for c in fsr.columns if c not in id_like and pd.api.types.is_numeric_dtype(fsr[c])]
    trait_records = []
    for _, r in fights.iterrows():
        frows = fsr[(fsr["fight_id"].eq(r["fight_id"])) & (fsr["event_date"].eq(r["event_date"]))].copy()
        if len(frows) != 2:
            continue
        # Prefer fighter-name matching; fallback to red/blue ordering only if side exists.
        def find_name(name: str):
            for name_col in ("fighter_name", "name"):
                if name_col in frows.columns:
                    m = frows[name_col].astype(str).str.casefold().eq(str(name).casefold())
                    if int(m.sum()) == 1:
                        return frows.loc[m].iloc[0]
            return None
        fav_row = find_name(r["favorite"])
        dog_row = find_name(r["underdog"])
        if fav_row is None or dog_row is None:
            continue
        rec = {"fight_id": r["fight_id"]}
        for c in numeric_cols:
            fv = pd.to_numeric(pd.Series([fav_row[c]]), errors="coerce").iloc[0]
            dv = pd.to_numeric(pd.Series([dog_row[c]]), errors="coerce").iloc[0]
            if pd.notna(fv) and pd.notna(dv):
                rec[f"delta__{c}"] = float(fv - dv)
        trait_records.append(rec)
    trait_delta = pd.DataFrame(trait_records)
    if not trait_delta.empty:
        fights = fights.merge(trait_delta, on="fight_id", how="left")

    # Market-strength buckets expose saturation directly.
    bins = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = ["50-60", "60-70", "70-80", "80-90", "90+"]
    fights["market_favorite_bucket"] = pd.cut(fights["market_favorite_fair_p"], bins=bins, labels=labels, right=False)
    bucket = fights.groupby("market_favorite_bucket", observed=True).agg(
        fights=("fight_id", "size"),
        market_favorite_fair_p=("market_favorite_fair_p", "mean"),
        mc_favorite_p=("mc_favorite_p", "mean"),
        actual_favorite_win_rate=("favorite_won", "mean"),
        mean_compression_pp=("compression_pp", "mean"),
        mean_dog_dec_p=("dog_dec_p", "mean"),
        mean_dog_ko_p=("dog_ko_p", "mean"),
        mean_dog_sub_p=("dog_sub_p", "mean"),
    ).reset_index()

    # Rank traits by association with market strength and MC response.
    trait_rows = []
    for c in [c for c in fights.columns if c.startswith("delta__")]:
        trait_rows.append({
            "trait": c.replace("delta__", ""),
            "n": int(pd.to_numeric(fights[c], errors="coerce").notna().sum()),
            "corr_delta_market_favorite_p": _safe_corr(fights[c], fights["market_favorite_fair_p"]),
            "corr_delta_mc_favorite_p": _safe_corr(fights[c], fights["mc_favorite_p"]),
            "corr_delta_compression": _safe_corr(fights[c], fights["compression_pp"]),
        })
    trait_corr = pd.DataFrame(trait_rows)
    if not trait_corr.empty:
        trait_corr["abs_market_corr"] = trait_corr["corr_delta_market_favorite_p"].abs()
        trait_corr = trait_corr.sort_values(["abs_market_corr", "n"], ascending=[False, False])

    # Dog win-path calibration across the 41 priced fights.
    dog_paths = pd.DataFrame([
        {"path": "DEC", "mean_mc_probability": fights["dog_dec_p"].mean(), "actual_rate": ((fights["dog_won"].eq(1)) & fights["actual_method"].eq("DEC")).mean()},
        {"path": "KO_TKO", "mean_mc_probability": fights["dog_ko_p"].mean(), "actual_rate": ((fights["dog_won"].eq(1)) & fights["actual_method"].eq("KO_TKO")).mean()},
        {"path": "SUB", "mean_mc_probability": fights["dog_sub_p"].mean(), "actual_rate": ((fights["dog_won"].eq(1)) & fights["actual_method"].eq("SUB")).mean()},
    ])
    dog_paths["overstatement_pp"] = 100.0 * (dog_paths["mean_mc_probability"] - dog_paths["actual_rate"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fights.to_csv(args.out_dir / "favorite_dog_fight_level.csv", index=False)
    bucket.to_csv(args.out_dir / "market_strength_buckets.csv", index=False)
    trait_corr.to_csv(args.out_dir / "fsr_trait_gap_correlations.csv", index=False)
    dog_paths.to_csv(args.out_dir / "dog_win_path_calibration.csv", index=False)

    print("BANTAMWEIGHT DOG COMPRESSION DIAGNOSTIC")
    print(f"priced fights: {len(fights)}")
    print(f"mean market favorite fair p: {fights['market_favorite_fair_p'].mean():.4f}")
    print(f"mean MC favorite p:          {fights['mc_favorite_p'].mean():.4f}")
    print(f"actual favorite win rate:    {fights['favorite_won'].mean():.4f}")
    print(f"mean compression:            {fights['compression_pp'].mean():+.2f} pp")
    print("\nMARKET STRENGTH BUCKETS")
    print(bucket.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDOG WIN-PATH CALIBRATION")
    print(dog_paths.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not trait_corr.empty:
        print("\nTOP FSR TRAIT GAPS BY MARKET-STRENGTH ASSOCIATION")
        print(trait_corr.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nNo FSR trait deltas were matched; inspect favorite_dog_fight_level.csv matching keys.")


if __name__ == "__main__":
    main()
