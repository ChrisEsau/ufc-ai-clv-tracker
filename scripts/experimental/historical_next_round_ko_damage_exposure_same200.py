"""Historical pre-round damage exposure -> next-round KO/TKO audit.

Research-only calibration diagnostic using the exact first 200 bouts from the
aligned mature 2020+ FSR-32 cohort used by the current MC calibration.

For every fighter who enters R2 or R3, measure what that fighter absorbed before
that round:
- prior-round significant strikes
- cumulative significant strikes
- prior-round head strikes
- cumulative head strikes
- prior-round knockdowns
- cumulative knockdowns

Then measure whether that fighter is KO/TKO'd in the upcoming round.

The latent simulator damage reservoir is not observable historically; this is an
observable exposure/hazard target for deciding whether reservoir accumulation,
recovery, or KD-related damage is too aggressive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = modern.MASTER_PATH
OUTPUT_DIR = Path("data/experimental")
DETAIL_PATH = OUTPUT_DIR / "historical_next_round_ko_damage_exposure_same200_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "historical_next_round_ko_damage_exposure_same200_summary.csv"
QUINTILE_PATH = OUTPUT_DIR / "historical_next_round_ko_damage_exposure_same200_quintiles.csv"
KD_PATH = OUTPUT_DIR / "historical_next_round_ko_damage_exposure_same200_kd.csv"

BOUTS = 200


def _load_round_stats() -> pd.DataFrame:
    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing round stats: {ROUND_STATS_PATH}")
    df = pd.read_parquet(ROUND_STATS_PATH).copy()
    required = {
        "fight_id", "round", "fighter_id", "corner",
        "sig_str_landed", "head_landed", "kd",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Round stats missing required columns: {missing}")
    df["fight_id"] = df["fight_id"].astype(str)
    df["fighter_id"] = df["fighter_id"].astype(str)
    for col in ("round", "sig_str_landed", "head_landed", "kd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[["round", "sig_str_landed", "head_landed", "kd"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"Invalid numeric round-stat rows: {int(bad.sum())}")
    df["round"] = df["round"].astype(int)
    return df


def _build_absorbed_rows(rs: pd.DataFrame) -> pd.DataFrame:
    counts = rs.groupby(["fight_id", "round"]).size()
    bad = counts[counts.ne(2)]
    if not bad.empty:
        raise ValueError(f"Expected 2 fighter rows per fight-round; bad keys={len(bad)}")

    a = rs[["fight_id", "round", "fighter_id", "corner"]].copy()
    opp = rs[["fight_id", "round", "fighter_id", "sig_str_landed", "head_landed", "kd"]].copy()
    opp = opp.rename(columns={
        "fighter_id": "opponent_id",
        "sig_str_landed": "sig_absorbed",
        "head_landed": "head_absorbed",
        "kd": "kd_absorbed",
    })
    merged = a.merge(opp, on=["fight_id", "round"], how="inner")
    merged = merged[merged["fighter_id"] != merged["opponent_id"]].copy()
    return merged.sort_values(["fight_id", "fighter_id", "round"]).reset_index(drop=True)


def _master_outcomes(ids: set[str]) -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).copy()
    required = {"fight_id", "winner_id", "method", "finish_round"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master missing required columns: {missing}")
    master["fight_id"] = master["fight_id"].astype(str)
    master["winner_id"] = master["winner_id"].astype(str)
    master = master[master["fight_id"].isin(ids)].copy()
    master["actual_ko_tko"] = master["method"].map(modern._is_ko_tko).astype(int)
    master["finish_round"] = pd.to_numeric(master["finish_round"], errors="coerce")
    return master[["fight_id", "winner_id", "actual_ko_tko", "finish_round"]].drop_duplicates("fight_id")


def _build_entry_frame(cohort: pd.DataFrame, absorbed: pd.DataFrame) -> pd.DataFrame:
    bout_ids = set(cohort["bout_id"].astype(str))
    absorbed = absorbed[absorbed["fight_id"].isin(bout_ids)].copy()
    outcomes = _master_outcomes(bout_ids)

    rows: list[dict[str, object]] = []
    for (fight_id, fighter_id), g in absorbed.groupby(["fight_id", "fighter_id"], sort=False):
        g = g.sort_values("round")
        outcome = outcomes[outcomes["fight_id"].eq(fight_id)]
        if outcome.empty:
            continue
        o = outcome.iloc[0]

        for entering_round in (2, 3):
            prior = g[g["round"] < entering_round]
            # Fighter must have completed every prior scheduled round to enter this one.
            if len(prior) != entering_round - 1:
                continue
            prev = prior.iloc[-1]
            ko_loss = int(
                int(o["actual_ko_tko"]) == 1
                and float(o["finish_round"]) == entering_round
                and str(fighter_id) != str(o["winner_id"])
            )
            rows.append({
                "bout_id": fight_id,
                "fighter_id": fighter_id,
                "entering_round": entering_round,
                "prior_round_sig_absorbed": float(prev["sig_absorbed"]),
                "cum_sig_absorbed": float(prior["sig_absorbed"].sum()),
                "prior_round_head_absorbed": float(prev["head_absorbed"]),
                "cum_head_absorbed": float(prior["head_absorbed"].sum()),
                "prior_round_kd_absorbed": float(prev["kd_absorbed"]),
                "cum_kd_absorbed": float(prior["kd_absorbed"].sum()),
                "next_round_ko_loss": ko_loss,
            })
    return pd.DataFrame(rows)


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rnd, g in detail.groupby("entering_round", sort=True):
        rows.append({
            "entering_round": int(rnd),
            "fighter_entries": len(g),
            "bouts_reaching_round": g["bout_id"].nunique(),
            "next_round_ko_losses": int(g["next_round_ko_loss"].sum()),
            "fighter_next_round_ko_loss_rate": float(g["next_round_ko_loss"].mean()),
            "median_prior_sig_absorbed": float(g["prior_round_sig_absorbed"].median()),
            "median_cum_sig_absorbed": float(g["cum_sig_absorbed"].median()),
            "median_prior_head_absorbed": float(g["prior_round_head_absorbed"].median()),
            "median_cum_head_absorbed": float(g["cum_head_absorbed"].median()),
            "median_cum_kd_absorbed": float(g["cum_kd_absorbed"].median()),
            "p75_cum_head_absorbed": float(g["cum_head_absorbed"].quantile(.75)),
            "p90_cum_head_absorbed": float(g["cum_head_absorbed"].quantile(.90)),
        })
    return pd.DataFrame(rows)


def _quintiles(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = ["cum_sig_absorbed", "cum_head_absorbed"]
    rows: list[dict[str, object]] = []
    for rnd, g0 in detail.groupby("entering_round", sort=True):
        for metric in metrics:
            g = g0.copy()
            # rank(method='first') guarantees five similarly sized groups even with ties.
            g["quintile"] = pd.qcut(g[metric].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            for q, h in g.groupby("quintile", observed=True, sort=True):
                rows.append({
                    "entering_round": int(rnd),
                    "metric": metric,
                    "quintile": str(q),
                    "fighter_entries": len(h),
                    "exposure_min": float(h[metric].min()),
                    "exposure_median": float(h[metric].median()),
                    "exposure_max": float(h[metric].max()),
                    "next_round_ko_losses": int(h["next_round_ko_loss"].sum()),
                    "next_round_ko_loss_rate": float(h["next_round_ko_loss"].mean()),
                })
    return pd.DataFrame(rows)


def _kd_table(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    d["cum_kd_group"] = np.select(
        [d["cum_kd_absorbed"].eq(0), d["cum_kd_absorbed"].eq(1)],
        ["0", "1"],
        default="2+",
    )
    rows = []
    for (rnd, kd_group), g in d.groupby(["entering_round", "cum_kd_group"], sort=True):
        rows.append({
            "entering_round": int(rnd),
            "cum_kd_absorbed": kd_group,
            "fighter_entries": len(g),
            "next_round_ko_losses": int(g["next_round_ko_loss"].sum()),
            "next_round_ko_loss_rate": float(g["next_round_ko_loss"].mean()),
            "median_cum_head_absorbed": float(g["cum_head_absorbed"].median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cohort, _pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(BOUTS).reset_index(drop=True)
    rs = _load_round_stats()
    absorbed = _build_absorbed_rows(rs)
    detail = _build_entry_frame(cohort, absorbed)
    if detail.empty:
        raise ValueError("No fighter round-entry rows were constructed")

    summary = _summary(detail)
    quintiles = _quintiles(detail)
    kd = _kd_table(detail)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    quintiles.to_csv(QUINTILE_PATH, index=False)
    kd.to_csv(KD_PATH, index=False)

    print("\n" + "=" * 150)
    print("HISTORICAL NEXT-ROUND KO/TKO vs PRIOR DAMAGE EXPOSURE — EXACT SAME 200 BOUTS")
    print("=" * 150)
    print("Unit: fighter entering R2 or R3; exposure is what that fighter absorbed before the round")
    print("KO/TKO target: that specific fighter is stopped by KO/TKO in the upcoming round")
    print("Reservoir itself is latent/unobserved; these are historical observable calibration targets")

    print("\nROUND-ENTRY SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCUMULATIVE EXPOSURE QUINTILES")
    print(quintiles.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCUMULATIVE KD EXPOSURE")
    print(kd.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOUTPUTS")
    print(DETAIL_PATH)
    print(SUMMARY_PATH)
    print(QUINTILE_PATH)
    print(KD_PATH)
    print("\nResearch only: no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
