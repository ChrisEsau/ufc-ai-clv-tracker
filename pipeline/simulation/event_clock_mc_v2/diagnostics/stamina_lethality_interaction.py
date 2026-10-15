"""Historical-only stamina x round lethality interaction study.

Question
--------
Do fighters with better prefight stamina ability show a flatter decline in
KD/KO lethality per landed significant strike from R1 -> R2 -> R3?

This is measurement only. It does not alter FSR, Event Clock, or calibration.
It uses leakage-safe prefight FSR V3 snapshots and observed UFCStats rounds.

Two inherited stamina proxies are tested separately:
- stamina_performance_resilience
- stamina_depletion_resistance

For each proxy we fit Poisson rate models with log(sig landed) exposure:
  outcome ~ round_index + stamina_z + round_index:stamina_z + division FE
with fighter-clustered robust covariance. Positive interaction means higher
stamina is associated with a flatter (less negative / more positive) round
slope in lethality.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.round_stats.build_round_fighter_state import standardize_round_stats_input
from pipeline.simulation.event_clock_mc_v2.diagnostics.weight_class_audit import select_cohort
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import normalize_method

DIVISIONS = (
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "light heavyweight",
    "heavyweight",
    "women's strawweight",
    "women's flyweight",
    "women's bantamweight",
)
STAMINA_TRAITS = (
    "stamina_performance_resilience",
    "stamina_depletion_resistance",
)


def _cohort_ids(target_n: int) -> set[str]:
    ids: set[str] = set()
    for division in DIVISIONS:
        cohort, _ = select_cohort(division, target_n)
        ids.update(cohort["fight_id"].astype(str))
    return ids


def _winner_id(row: pd.Series) -> str | None:
    winner = str(row.get("winner", ""))
    if winner == str(row.get("r_name", "")):
        return str(row.get("r_id"))
    if winner == str(row.get("b_name", "")):
        return str(row.get("b_id"))
    return None


def build_frame(target_n: int) -> pd.DataFrame:
    fight_ids = _cohort_ids(target_n)

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master = master[master["fight_id"].isin(fight_ids)].copy()
    master["division"] = master["division"].astype(str).str.strip().str.lower()
    master["finish_round"] = pd.to_numeric(master["finish_round"], errors="coerce")
    master["method_norm"] = master["method"].map(normalize_method)
    master["winner_id"] = master.apply(_winner_id, axis=1)

    rounds = pd.read_parquet(ROUND_STATS_PATH)
    rounds = standardize_round_stats_input(rounds)
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    rounds = rounds[
        rounds["fight_id"].isin(fight_ids)
        & pd.to_numeric(rounds["round"], errors="coerce").between(1, 3)
    ].copy()

    meta_cols = ["fight_id", "division", "finish_round", "method_norm", "winner_id"]
    rounds = rounds.merge(master[meta_cols], on="fight_id", how="inner", validate="many_to_one")

    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    required = ["fight_id", "fighter_id", *STAMINA_TRAITS]
    missing = [c for c in required if c not in fsr.columns]
    if missing:
        raise RuntimeError(f"FSR V3 snapshots missing stamina fields: {missing}")
    optional = [c for c in ("age_years",) if c in fsr.columns]
    fsr = fsr[required + optional].copy()
    rounds = rounds.merge(fsr, on=["fight_id", "fighter_id"], how="inner", validate="many_to_one")

    rounds["round"] = pd.to_numeric(rounds["round"], errors="raise").astype(int)
    rounds["round_index"] = rounds["round"] - 1
    rounds["sig_landed"] = pd.to_numeric(rounds["sig_str_landed"], errors="coerce").fillna(0.0)
    rounds["sig_attempted"] = pd.to_numeric(rounds["sig_str_attempted"], errors="coerce").fillna(0.0)
    rounds["kd_count"] = pd.to_numeric(rounds["kd"], errors="coerce").fillna(0.0)
    rounds["ko_win"] = (
        rounds["method_norm"].eq("KO_TKO")
        & rounds["finish_round"].eq(rounds["round"])
        & rounds["winner_id"].eq(rounds["fighter_id"])
    ).astype(int)
    rounds["sex_group"] = np.where(rounds["division"].str.startswith("women's "), "women", "men")

    for trait in STAMINA_TRAITS:
        rounds[trait] = pd.to_numeric(rounds[trait], errors="coerce")
    if "age_years" in rounds:
        rounds["age_years"] = pd.to_numeric(rounds["age_years"], errors="coerce")

    return rounds.reset_index(drop=True)


def descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trait in STAMINA_TRAITS:
        valid = frame.dropna(subset=[trait]).copy()
        valid["stamina_group"] = pd.qcut(valid[trait], 3, labels=["low", "mid", "high"], duplicates="drop")
        for sex_group in ("all", "men", "women"):
            part = valid if sex_group == "all" else valid[valid["sex_group"].eq(sex_group)]
            for (group, rnd), g in part.groupby(["stamina_group", "round"], observed=True):
                landed = float(g["sig_landed"].sum())
                exposure_min = float(g["round_exposure_seconds"].sum()) / 60.0
                rows.append({
                    "trait": trait,
                    "sex_group": sex_group,
                    "stamina_group": str(group),
                    "round": int(rnd),
                    "fighter_rounds": int(len(g)),
                    "unique_fighters": int(g["fighter_id"].nunique()),
                    "mean_stamina": float(g[trait].mean()),
                    "sig_attempts_per_min": float(g["sig_attempted"].sum()) / exposure_min if exposure_min > 0 else np.nan,
                    "sig_landed_per_min": landed / exposure_min if exposure_min > 0 else np.nan,
                    "kd_per_100_sig_landed": float(g["kd_count"].sum()) / landed * 100.0 if landed > 0 else np.nan,
                    "ko_wins_per_100_sig_landed": float(g["ko_win"].sum()) / landed * 100.0 if landed > 0 else np.nan,
                    "ko_win_per_fighter_round": float(g["ko_win"].mean()),
                })
    return pd.DataFrame(rows)


def fit_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trait in STAMINA_TRAITS:
        for sex_group in ("all", "men", "women"):
            base = frame if sex_group == "all" else frame[frame["sex_group"].eq(sex_group)]
            d = base.dropna(subset=[trait]).copy()
            mean = float(d[trait].mean())
            sd = float(d[trait].std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                continue
            d["stamina_z"] = (d[trait] - mean) / sd
            d = d[d["sig_landed"] > 0].copy()
            d["log_landed"] = np.log(d["sig_landed"].astype(float))
            if len(d) < 50:
                continue

            age_term = " + age_years" if "age_years" in d.columns and d["age_years"].notna().mean() > 0.95 else ""
            if age_term:
                d = d.dropna(subset=["age_years"]).copy()
            formula_base = f"round_index + stamina_z + round_index:stamina_z + C(division){age_term}"

            for outcome in ("kd_count", "ko_win"):
                try:
                    model = smf.glm(
                        formula=f"{outcome} ~ {formula_base}",
                        data=d,
                        family=sm.families.Poisson(),
                        offset=d["log_landed"],
                    ).fit(cov_type="cluster", cov_kwds={"groups": d["fighter_id"]})
                    name = "round_index:stamina_z"
                    coef = float(model.params[name])
                    se = float(model.bse[name])
                    p = float(model.pvalues[name])
                    lo, hi = model.conf_int().loc[name].astype(float)
                    round_coef = float(model.params["round_index"])
                    rows.append({
                        "trait": trait,
                        "sex_group": sex_group,
                        "outcome": outcome,
                        "n_fighter_rounds": int(len(d)),
                        "n_fighters": int(d["fighter_id"].nunique()),
                        "stamina_mean": mean,
                        "stamina_sd": sd,
                        "round_log_rate_coef_at_mean_stamina": round_coef,
                        "round_rate_ratio_at_mean_stamina": float(np.exp(round_coef)),
                        "round_x_stamina_coef": coef,
                        "round_x_stamina_rate_ratio": float(np.exp(coef)),
                        "interaction_se": se,
                        "interaction_p": p,
                        "interaction_ci_low": float(lo),
                        "interaction_ci_high": float(hi),
                        "interpretation": (
                            "higher stamina = flatter/more positive lethality slope" if coef > 0
                            else "higher stamina = steeper/more negative lethality slope"
                        ),
                    })
                except Exception as exc:
                    rows.append({
                        "trait": trait,
                        "sex_group": sex_group,
                        "outcome": outcome,
                        "n_fighter_rounds": int(len(d)),
                        "n_fighters": int(d["fighter_id"].nunique()),
                        "error": repr(exc),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics/event_clock_mc_v2/stamina_lethality_interaction"),
    )
    args = parser.parse_args()

    frame = build_frame(args.target_n)
    desc = descriptive(frame)
    models = fit_models(frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "fighter_round_frame.csv", index=False)
    desc.to_csv(args.out_dir / "stamina_group_round_rates.csv", index=False)
    models.to_csv(args.out_dir / "interaction_models.csv", index=False)

    print("=" * 120)
    print("HISTORICAL STAMINA x ROUND LETHALITY INTERACTION")
    print("=" * 120)
    print(f"fighter-round rows: {len(frame):,} | fights: {frame['fight_id'].nunique():,} | fighters: {frame['fighter_id'].nunique():,}")
    print("\nINTERACTION MODELS")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDESCRIPTIVE LOW/MID/HIGH STAMINA BY ROUND")
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nOUTPUT: {args.out_dir}")


if __name__ == "__main__":
    main()
