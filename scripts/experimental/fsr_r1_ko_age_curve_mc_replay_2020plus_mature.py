"""Replay mature 2020+ R1 KO/TKO bouts under shadow age-adjustment curves.

Purpose
-------
Evaluate whether candidate age adjustments to knockdown_resistance and
 damage_durability improve the STATIC shadow Monte Carlo on the fights we care
about, rather than only improving auxiliary logistic regressions.

Primary cohort
--------------
- UFC bouts 2020-01-01+
- both fighters had >=3 prior UFC fights
- all actual round-1 KO/TKO bouts in that cohort (currently 247)

Comparison cohort
-----------------
For each actual R1 KO/TKO bout, select one deterministic age-matched non-R1-KO
bout from the same mature cohort. This gives a balanced diagnostic comparison.
Because the comparison sample is deliberately balanced, Brier/log loss are
reported as diagnostic scores, NOT population calibration estimates.

Candidate curves
----------------
The same curve is applied independently to BOTH knockdown_resistance and
 damage_durability for each fighter. Stored FSR values are never modified.

Variants:
- none
- linear after 30, -1.0 point/year
- linear after 30, -1.5 points/year
- linear after 30, -2.0 points/year
- quadratic after 30, slope 1.0

For actual R1 KO bouts we report:
- mean simulated P(any R1 KO/TKO)
- mean simulated P(actual winner scores R1 KO/TKO)
- winner-direction hit rate (actual winner has higher simulated R1-KO share)
- the same metrics by loser-age band

For the balanced KO/control diagnostic set we report AUC, Brier and log loss
for P(any R1 KO/TKO).

No production artifact, stored FSR, or simulator constant is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.experimental import fsr_age_decay_curve_search_kd_durability_2020plus_mature as curves
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "r1_ko_age_curve_mc_replay_2020plus_mature.csv"
)
DEFAULT_PATHS = 100
DEFAULT_SEED = 20260810

AGE_BINS = [-np.inf, 30.999, 33.999, 36.999, 39.999, np.inf]
AGE_LABELS = ["<=30", "31-33", "34-36", "37-39", "40+"]


@dataclass(frozen=True)
class Variant:
    label: str
    curve: curves.Curve


VARIANTS = (
    Variant("none", curves.Curve("none")),
    Variant("linear_on30_s1", curves.Curve("linear", onset=30.0, slope=1.0)),
    Variant("linear_on30_s1.5", curves.Curve("linear", onset=30.0, slope=1.5)),
    Variant("linear_on30_s2", curves.Curve("linear", onset=30.0, slope=2.0)),
    Variant("quadratic_on30_s1", curves.Curve("quadratic", onset=30.0, slope=1.0)),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS, help="MC paths per bout per variant")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _master_meta() -> pd.DataFrame:
    raw = pd.read_parquet(modern.MASTER_PATH).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    raw["fight_id"] = raw["fight_id"].astype(str)
    raw["r_id"] = raw["r_id"].astype(str)
    raw["b_id"] = raw["b_id"].astype(str)
    raw["winner_id"] = raw["winner_id"].astype(str)
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")

    def age(corner: str) -> pd.Series:
        for col in (f"{corner}_age", f"{corner}_fighter_age", f"{corner}_age_years"):
            if col in raw.columns:
                return pd.to_numeric(raw[col], errors="coerce")
        for col in (
            f"{corner}_dob", f"{corner}_date_of_birth",
            f"{corner}_fighter_dob", f"{corner}_fighter_date_of_birth",
        ):
            if col in raw.columns:
                dob = pd.to_datetime(raw[col], errors="coerce")
                return (raw["event_date"] - dob).dt.days / 365.25
        return pd.Series(np.nan, index=raw.index, dtype=float)

    raw["r_age"] = age("r")
    raw["b_age"] = age("b")
    return raw[["fight_id", "winner_id", "r_age", "b_age"]].rename(columns={"fight_id": "bout_id"})


def _build_cohort() -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    master = modern._load_master(modern.MASTER_PATH)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(modern.FSR_PATH, cohort)
    cohort = cohort.merge(_master_meta(), on="bout_id", how="left", validate="one_to_one")
    cohort["mean_age"] = cohort[["r_age", "b_age"]].mean(axis=1)
    cohort["max_age"] = cohort[["r_age", "b_age"]].max(axis=1)
    return cohort.reset_index(drop=True), pairs


def _match_controls(cohort: pd.DataFrame) -> pd.DataFrame:
    positives = cohort.loc[cohort["actual_r1_ko"].eq(1)].copy()
    controls = cohort.loc[cohort["actual_r1_ko"].eq(0)].copy()
    available = set(controls.index.tolist())
    chosen: list[int] = []

    # Deterministic nearest-neighbour age matching without replacement.
    for _, pos in positives.sort_values(["event_date", "bout_id"]).iterrows():
        candidates = controls.loc[list(available)].copy()
        candidates["distance"] = (
            (candidates["mean_age"] - pos["mean_age"]).abs()
            + 0.5 * (candidates["max_age"] - pos["max_age"]).abs()
        )
        pick = int(candidates.sort_values(["distance", "event_date", "bout_id"]).index[0])
        chosen.append(pick)
        available.remove(pick)

    matched_controls = controls.loc[chosen].copy()
    positives["sample_class"] = "actual_r1_ko"
    matched_controls["sample_class"] = "age_matched_control"
    return pd.concat([positives, matched_controls], ignore_index=True)


def _effective_profile(profile: pd.Series, age: float, curve: curves.Curve) -> pd.Series:
    out = profile.copy()
    age_series = pd.Series([age], dtype=float)
    for trait in ("knockdown_resistance", "damage_durability"):
        original = pd.Series([pd.to_numeric(out.get(trait), errors="coerce")], dtype=float)
        out[trait] = float(curves._apply_curve(original, age_series, curve).iloc[0])
    return out


def _simulate_bout(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
    variant: Variant,
    path_seeds: np.ndarray,
) -> dict[str, object]:
    red, blue = pair
    red_eff = _effective_profile(red, float(bout["r_age"]), variant.curve)
    blue_eff = _effective_profile(blue, float(bout["b_age"]), variant.curve)

    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout["winner_id"])
    r_r1_ko = 0
    b_r1_ko = 0

    for seed in path_seeds:
        sim = ko.StaticFSRMCKOTKOV2(red_eff, blue_eff, rounds=3, seed=int(seed))
        result = sim.run()
        finish = result.finish
        if finish is None or finish.round != 1:
            continue
        if finish.winner == 0:
            r_r1_ko += 1
        else:
            b_r1_ko += 1

    n = len(path_seeds)
    p_r = r_r1_ko / n
    p_b = b_r1_ko / n
    p_any = p_r + p_b
    p_actual_winner = p_r if winner_id == r_id else p_b if winner_id == b_id else np.nan
    p_actual_loser = p_b if winner_id == r_id else p_r if winner_id == b_id else np.nan

    if winner_id == r_id:
        winner_age, loser_age = float(bout["r_age"]), float(bout["b_age"])
    elif winner_id == b_id:
        winner_age, loser_age = float(bout["b_age"]), float(bout["r_age"])
    else:
        winner_age = loser_age = np.nan

    return {
        "variant": variant.label,
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "sample_class": bout["sample_class"],
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "winner_id": winner_id,
        "winner_age": winner_age,
        "loser_age": loser_age,
        "r_age": float(bout["r_age"]),
        "b_age": float(bout["b_age"]),
        "p_r_r1_ko": p_r,
        "p_b_r1_ko": p_b,
        "p_any_r1_ko": p_any,
        "p_actual_winner_r1_ko": p_actual_winner,
        "p_actual_loser_r1_ko": p_actual_loser,
        "winner_direction_hit": int(p_actual_winner > p_actual_loser) if pd.notna(p_actual_winner) else np.nan,
        "winner_direction_tie": int(p_actual_winner == p_actual_loser) if pd.notna(p_actual_winner) else np.nan,
    }


def _print_summary(results: pd.DataFrame) -> None:
    positives = results.loc[results["actual_r1_ko"].eq(1)].copy()
    positives["loser_age_band"] = pd.cut(positives["loser_age"], bins=AGE_BINS, labels=AGE_LABELS)

    print("\n" + "=" * 124)
    print("ACTUAL R1 KO/TKO REPLAY — AGE CURVE VARIANTS")
    print("=" * 124)
    overall = positives.groupby("variant", observed=True).agg(
        bouts=("bout_id", "size"),
        mean_p_any_r1_ko=("p_any_r1_ko", "mean"),
        mean_p_actual_winner_r1_ko=("p_actual_winner_r1_ko", "mean"),
        mean_p_actual_loser_r1_ko=("p_actual_loser_r1_ko", "mean"),
        winner_direction_hit_rate=("winner_direction_hit", "mean"),
        direction_tie_rate=("winner_direction_tie", "mean"),
    ).reset_index()
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nACTUAL R1 KO/TKO — BY ACTUAL LOSER AGE")
    by_age = positives.groupby(["variant", "loser_age_band"], observed=True).agg(
        bouts=("bout_id", "size"),
        mean_loser_age=("loser_age", "mean"),
        mean_p_any_r1_ko=("p_any_r1_ko", "mean"),
        mean_p_actual_winner_r1_ko=("p_actual_winner_r1_ko", "mean"),
        winner_direction_hit_rate=("winner_direction_hit", "mean"),
    ).reset_index()
    print(by_age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBALANCED R1-KO / AGE-MATCHED-CONTROL DIAGNOSTIC")
    rows = []
    for variant, g in results.groupby("variant", observed=True):
        y = g["actual_r1_ko"].astype(int)
        p = g["p_any_r1_ko"].clip(1e-6, 1 - 1e-6)
        rows.append({
            "variant": variant,
            "n": len(g),
            "auc": roc_auc_score(y, p),
            "brier_balanced": brier_score_loss(y, p),
            "logloss_balanced": log_loss(y, p),
            "mean_p_positive": g.loc[y.eq(1), "p_any_r1_ko"].mean(),
            "mean_p_control": g.loc[y.eq(0), "p_any_r1_ko"].mean(),
            "separation": g.loc[y.eq(1), "p_any_r1_ko"].mean() - g.loc[y.eq(0), "p_any_r1_ko"].mean(),
        })
    print(pd.DataFrame(rows).sort_values(["auc", "separation"], ascending=False).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    cohort, pairs = _build_cohort()
    sample = _match_controls(cohort)
    positives = int(sample["actual_r1_ko"].sum())
    controls = len(sample) - positives
    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"actual R1 KO/TKO bouts: {positives:,}")
    print(f"age-matched controls: {controls:,}")
    print(f"variants: {len(VARIANTS)}")
    print(f"paths per bout/variant: {args.paths:,}")
    print("Common random-number seeds are reused across variants for each bout.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    total_jobs = len(sample) * len(VARIANTS)
    job = 0
    total_paths = total_jobs * args.paths
    completed_paths = 0

    for _, bout in sample.iterrows():
        bout_id = str(bout["bout_id"])
        path_seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        pair = pairs[bout_id]
        for variant in VARIANTS:
            rows.append(_simulate_bout(bout, pair, variant, path_seeds))
            job += 1
            completed_paths += args.paths
            if completed_paths % 1000 == 0 or job == total_jobs:
                print(
                    f"[R1 KO age replay] paths {completed_paths:,}/{total_paths:,}; "
                    f"jobs {job:,}/{total_jobs:,}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    _print_summary(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nWrote {len(results):,} bout-variant rows to {args.output}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
