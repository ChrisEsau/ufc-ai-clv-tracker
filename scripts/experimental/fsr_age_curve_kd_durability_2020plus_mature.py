"""Age-interaction audit for leakage-safe knockdown resistance and damage durability.

Purpose
-------
Test whether the predictive meaning/calibration of the existing pre-fight
``knockdown_resistance`` and ``damage_durability`` FSR traits changes with
fighter age in the modern mature cohort.

This study is diagnostic-only:
- UFC bouts 2020-01-01+
- both fighters have >=3 prior UFC fights
- leakage-safe pre-fight FSR snapshots
- current-fight KD outcome comes from RFS history, never from the pre-fight FSR
- no FSR values or simulator constants are changed

Run from repo root:
    PYTHONPATH=. python scripts/experimental/fsr_age_curve_kd_durability_2020plus_mature.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

MASTER_PATH = modern.MASTER_PATH
FSR_PATH = modern.FSR_PATH
RFS_PATH = reservoir.RFS_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/age_curve_kd_durability_2020plus_mature.parquet"
)
AGE_BINS = [-np.inf, 27.999, 30.999, 33.999, 36.999, 39.999, np.inf]
AGE_LABELS = ["<=27", "28-30", "31-33", "34-36", "37-39", "40+"]


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _resolve_corner_age(master: pd.DataFrame, corner: str) -> pd.Series:
    direct = _first_existing(master, (f"{corner}_age", f"{corner}_fighter_age", f"{corner}_age_years"))
    if direct is not None:
        return pd.to_numeric(master[direct], errors="coerce")

    dob_col = _first_existing(
        master,
        (f"{corner}_dob", f"{corner}_date_of_birth", f"{corner}_fighter_dob", f"{corner}_fighter_date_of_birth"),
    )
    if dob_col is None:
        return pd.Series(np.nan, index=master.index, dtype=float)

    dob = pd.to_datetime(master[dob_col], errors="coerce")
    return (master["event_date"] - dob).dt.days / 365.25


def _load_master_side_metadata(master_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(master_path).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    raw["fight_id"] = raw["fight_id"].astype(str)
    raw["r_id"] = raw["r_id"].astype(str)
    raw["b_id"] = raw["b_id"].astype(str)
    raw["winner_id"] = raw["winner_id"].astype(str)
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")
    raw["r_age_calc"] = _resolve_corner_age(raw, "r")
    raw["b_age_calc"] = _resolve_corner_age(raw, "b")
    return raw[["fight_id", "r_age_calc", "b_age_calc", "winner_id"]].copy()


def _load_kd_outcomes(rfs_path: Path) -> pd.DataFrame:
    rfs = pd.read_parquet(rfs_path, columns=["fight_id", "fighter_id", reservoir.KD_COL]).copy()
    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    rfs[reservoir.KD_COL] = pd.to_numeric(rfs[reservoir.KD_COL], errors="coerce")
    if rfs.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("RFS KD outcome source violates fighter-fight grain")
    return rfs.rename(columns={reservoir.KD_COL: "kd_absorbed"})


def _numeric(profile: pd.Series, key: str) -> float:
    return float(pd.to_numeric(profile.get(key), errors="coerce"))


def _prepare_side_frame() -> pd.DataFrame:
    master = modern._load_master(MASTER_PATH)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(FSR_PATH, cohort)

    meta = _load_master_side_metadata(MASTER_PATH).rename(columns={"fight_id": "bout_id"})
    cohort = cohort.merge(meta, on="bout_id", how="left", validate="one_to_one")
    kd_outcomes = _load_kd_outcomes(RFS_PATH)
    kd_map = kd_outcomes.set_index(["fight_id", "fighter_id"])["kd_absorbed"]

    rows: list[dict[str, object]] = []
    for _, bout in cohort.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        winner_id = str(bout["winner_id"])

        for side, fighter_id, age, profile in (
            ("r", str(bout["r_id"]), bout["r_age_calc"], red),
            ("b", str(bout["b_id"]), bout["b_age_calc"], blue),
        ):
            kd_target = kd_map.get((bout_id, fighter_id), np.nan)
            rows.append(
                {
                    "bout_id": bout_id,
                    "event_date": bout["event_date"],
                    "side": side,
                    "fighter_id": fighter_id,
                    "age": pd.to_numeric(age, errors="coerce"),
                    "knockdown_resistance": _numeric(profile, "knockdown_resistance"),
                    "damage_durability": _numeric(profile, "damage_durability"),
                    "kd_absorbed": kd_target,
                    "any_kd_absorbed": np.nan if pd.isna(kd_target) else int(float(kd_target) > 0),
                    "ko_tko_loss": int(bool(bout["actual_ko_tko"]) and fighter_id != winner_id),
                }
            )

    frame = pd.DataFrame(rows)
    frame["age_band"] = pd.cut(frame["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return frame


def _safe_auc(y: pd.Series, x: pd.Series) -> float:
    mask = y.notna() & x.notna()
    yy = y.loc[mask].astype(int)
    xx = x.loc[mask].astype(float)
    if yy.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(yy, xx))


def _band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for band in AGE_LABELS:
        g = frame.loc[frame["age_band"].astype(str).eq(band)].copy()
        if g.empty:
            continue
        out.append(
            {
                "age_band": band,
                "fighter_sides": len(g),
                "mean_age": g["age"].mean(),
                "kd_absorbed_rate": g["any_kd_absorbed"].mean(),
                "ko_tko_loss_rate": g["ko_tko_loss"].mean(),
                "mean_kd_resistance": g["knockdown_resistance"].mean(),
                "mean_durability": g["damage_durability"].mean(),
                "kd_resistance_auc": _safe_auc(g["any_kd_absorbed"], -g["knockdown_resistance"]),
                "durability_auc": _safe_auc(g["ko_tko_loss"], -g["damage_durability"]),
            }
        )
    return pd.DataFrame(out)


def _quartile_calibration(frame: pd.DataFrame, trait: str, target: str) -> pd.DataFrame:
    work = frame[["age", "age_band", trait, target]].dropna().copy()
    work["trait_quartile"] = pd.qcut(work[trait], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return (
        work.groupby(["age_band", "trait_quartile"], observed=True)
        .agg(n=(target, "size"), event_rate=(target, "mean"), trait_mean=(trait, "mean"), age_mean=("age", "mean"))
        .reset_index()
    )


def _oof_model(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, float]:
    work = frame[[target] + features].dropna(subset=[target]).copy()
    y = work[target].astype(int).reset_index(drop=True)
    if y.nunique() < 2:
        return {"auc": np.nan, "logloss": np.nan, "brier": np.nan}
    X = work[features].reset_index(drop=True)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")),
        ]
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20260810)
    prediction_sum = np.zeros(len(work), dtype=float)
    prediction_count = np.zeros(len(work), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_p = model.predict_proba(X.iloc[test_idx])[:, 1]
        prediction_sum[test_idx] += fold_p
        prediction_count[test_idx] += 1

    if np.any(prediction_count == 0):
        raise RuntimeError("Repeated CV left one or more rows without an out-of-fold prediction")

    p = prediction_sum / prediction_count
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["age_over_35"] = np.maximum(work["age"] - 35.0, 0.0)
    work["kd_x_age35"] = (work["knockdown_resistance"] - 50.0) * work["age_over_35"]
    work["dur_x_age35"] = (work["damage_durability"] - 50.0) * work["age_over_35"]

    specs = [
        ("KD trait only", "any_kd_absorbed", ["knockdown_resistance"]),
        ("KD trait + age", "any_kd_absorbed", ["knockdown_resistance", "age", "age_over_35"]),
        ("KD trait + age interaction", "any_kd_absorbed", ["knockdown_resistance", "age", "age_over_35", "kd_x_age35"]),
        ("Durability only", "ko_tko_loss", ["damage_durability"]),
        ("Durability + age", "ko_tko_loss", ["damage_durability", "age", "age_over_35"]),
        ("Durability + age interaction", "ko_tko_loss", ["damage_durability", "age", "age_over_35", "dur_x_age35"]),
    ]
    rows = []
    for name, target, features in specs:
        rows.append({"model": name, "target": target, "features": ", ".join(features), **_oof_model(work, target, features)})
    return pd.DataFrame(rows)


def main() -> None:
    frame = _prepare_side_frame()
    print("\n" + "=" * 118)
    print("AGE CURVE AUDIT — KNOCKDOWN RESISTANCE / DAMAGE DURABILITY")
    print("=" * 118)
    print(f"fighter-side rows: {len(frame):,}")
    print(f"age available: {frame['age'].notna().mean():.2%}")
    print(f"KD target available: {frame['kd_absorbed'].notna().mean():.2%}")

    print("\nAGE-BAND PERFORMANCE")
    print(_band_summary(frame).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nKD RESISTANCE QUARTILES WITHIN AGE BANDS")
    print(_quartile_calibration(frame, "knockdown_resistance", "any_kd_absorbed").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDURABILITY QUARTILES WITHIN AGE BANDS")
    print(_quartile_calibration(frame, "damage_durability", "ko_tko_loss").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOUT-OF-FOLD AGE / TRAIT MODEL COMPARISON")
    print(_model_comparison(frame).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(frame):,} fighter-side rows to {OUTPUT_PATH}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
