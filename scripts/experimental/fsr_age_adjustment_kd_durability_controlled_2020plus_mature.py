"""Controlled age study for KD resistance and damage durability.

Purpose
-------
Estimate whether fighter age adds vulnerability beyond the existing leakage-safe
pre-fight FSR traits after controlling for opponent danger / damaging exposure.

Cohort
------
- UFC bouts 2020-01-01+
- both fighters have >=3 prior UFC fights
- leakage-safe pre-fight FSR snapshots
- current-fight outcomes from canonical RFS/master artifacts

No simulator constants or FSR values are changed.
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
OUTPUT_PATH = Path("data/simulation/rfs_mc_v2_shared_state/age_adjustment_kd_durability_controlled_2020plus_mature.parquet")
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
    dob_col = _first_existing(master, (f"{corner}_dob", f"{corner}_date_of_birth", f"{corner}_fighter_dob", f"{corner}_fighter_date_of_birth"))
    if dob_col is None:
        return pd.Series(np.nan, index=master.index, dtype=float)
    dob = pd.to_datetime(master[dob_col], errors="coerce")
    return (master["event_date"] - dob).dt.days / 365.25


def _load_master_meta() -> pd.DataFrame:
    raw = pd.read_parquet(MASTER_PATH).copy()
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
    return raw[["fight_id", "r_age_calc", "b_age_calc", "winner_id"]]


def _load_rfs_outcomes() -> pd.DataFrame:
    cols = [
        "fight_id", "fighter_id", reservoir.KD_COL, reservoir.SIG_ABS_COL,
        reservoir.HEAD_ABS_COL, reservoir.GROUND_ABS_COL, reservoir.OPP_CTRL_COL,
        reservoir.ROUNDS_COL, reservoir.KO_LOSS_COL,
    ]
    rfs = pd.read_parquet(RFS_PATH, columns=cols).copy()
    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    if rfs.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("RFS outcome source violates fighter-fight grain")
    for col in cols[2:]:
        rfs[col] = pd.to_numeric(rfs[col], errors="coerce")
    rounds = rfs[reservoir.ROUNDS_COL].clip(lower=1).fillna(1.0)
    rfs["damage_exposure"] = (
        (rfs[reservoir.KD_COL].fillna(0) / rounds)
        + (rfs[reservoir.HEAD_ABS_COL].fillna(0) / rounds)
        + (rfs[reservoir.GROUND_ABS_COL].fillna(0) / rounds)
        + (rfs[reservoir.OPP_CTRL_COL].fillna(0) / (rounds * 60.0))
    ) / 4.0
    rfs["any_kd_absorbed"] = (rfs[reservoir.KD_COL].fillna(0) > 0).astype(int)
    rfs["ko_tko_loss"] = (rfs[reservoir.KO_LOSS_COL].fillna(0) >= 0.5).astype(int)
    return rfs


def _num(profile: pd.Series, key: str) -> float:
    return float(pd.to_numeric(profile.get(key), errors="coerce"))


def _prepare_frame() -> pd.DataFrame:
    master = modern._load_master(MASTER_PATH)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(FSR_PATH, cohort)
    meta = _load_master_meta().rename(columns={"fight_id": "bout_id"})
    cohort = cohort.merge(meta, on="bout_id", how="left", validate="one_to_one")
    rfs = _load_rfs_outcomes().set_index(["fight_id", "fighter_id"])

    rows = []
    for _, bout in cohort.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        sides = [
            (str(bout["r_id"]), bout["r_age_calc"], red, blue),
            (str(bout["b_id"]), bout["b_age_calc"], blue, red),
        ]
        for fighter_id, age, profile, opp in sides:
            outcome = rfs.loc[(bout_id, fighter_id)]
            rows.append({
                "bout_id": bout_id,
                "event_date": bout["event_date"],
                "fighter_id": fighter_id,
                "age": pd.to_numeric(age, errors="coerce"),
                "knockdown_resistance": _num(profile, "knockdown_resistance"),
                "damage_durability": _num(profile, "damage_durability"),
                "opponent_striking_power": _num(opp, "striking_power"),
                "opponent_distance_pressure": _num(opp, "distance_striking_pressure"),
                "opponent_distance_precision": _num(opp, "distance_striking_precision"),
                "sig_absorbed": float(outcome[reservoir.SIG_ABS_COL]),
                "damage_exposure": float(outcome["damage_exposure"]),
                "any_kd_absorbed": int(outcome["any_kd_absorbed"]),
                "ko_tko_loss": int(outcome["ko_tko_loss"]),
            })
    frame = pd.DataFrame(rows)
    frame["age_band"] = pd.cut(frame["age"], AGE_BINS, labels=AGE_LABELS)
    frame["age_over_35"] = np.maximum(frame["age"] - 35.0, 0.0)
    frame["age_over_37"] = np.maximum(frame["age"] - 37.0, 0.0)
    frame["kd_per_sig"] = frame["any_kd_absorbed"] / frame["sig_absorbed"].clip(lower=1.0)
    return frame


def _oof(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, float]:
    work = frame[[target] + features].dropna(subset=[target]).copy()
    y = work[target].astype(int).to_numpy()
    X = work[features]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")),
    ])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20260810)
    sums = np.zeros(len(work), dtype=float)
    counts = np.zeros(len(work), dtype=int)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_proba(X.iloc[test_idx])[:, 1]
        sums[test_idx] += p
        counts[test_idx] += 1
    p = sums / counts
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _models(frame: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("KD: resistance + opponent power + exposure", "any_kd_absorbed", ["knockdown_resistance", "opponent_striking_power", "sig_absorbed"]),
        ("KD: + age", "any_kd_absorbed", ["knockdown_resistance", "opponent_striking_power", "sig_absorbed", "age", "age_over_35"]),
        ("KD: + age37 hinge", "any_kd_absorbed", ["knockdown_resistance", "opponent_striking_power", "sig_absorbed", "age", "age_over_35", "age_over_37"]),
        ("KO: durability + power + damage exposure", "ko_tko_loss", ["damage_durability", "opponent_striking_power", "damage_exposure"]),
        ("KO: + age", "ko_tko_loss", ["damage_durability", "opponent_striking_power", "damage_exposure", "age", "age_over_35"]),
        ("KO: + age37 hinge", "ko_tko_loss", ["damage_durability", "opponent_striking_power", "damage_exposure", "age", "age_over_35", "age_over_37"]),
    ]
    out = []
    for name, target, features in specs:
        out.append({"model": name, "target": target, **_oof(frame, target, features)})
    return pd.DataFrame(out)


def _age_rates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("age_band", observed=True).agg(
        n=("fighter_id", "size"),
        mean_age=("age", "mean"),
        kd_rate=("any_kd_absorbed", "mean"),
        mean_sig_absorbed=("sig_absorbed", "mean"),
        kd_per_100_sig=("any_kd_absorbed", lambda s: np.nan),
        ko_loss_rate=("ko_tko_loss", "mean"),
        mean_damage_exposure=("damage_exposure", "mean"),
        mean_kd_resistance=("knockdown_resistance", "mean"),
        mean_durability=("damage_durability", "mean"),
    ).reset_index()


def main() -> None:
    frame = _prepare_frame()
    print("\n" + "=" * 120)
    print("CONTROLLED AGE STUDY — KD RESISTANCE / DAMAGE DURABILITY")
    print("=" * 120)
    print(f"fighter-side rows: {len(frame):,}")
    print("\nCONTROLLED OUT-OF-FOLD MODEL COMPARISON")
    print(_models(frame).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nAGE-BAND RAW CONTEXT")
    summary = frame.groupby("age_band", observed=True).agg(
        n=("fighter_id", "size"),
        mean_age=("age", "mean"),
        kd_rate=("any_kd_absorbed", "mean"),
        mean_sig_absorbed=("sig_absorbed", "mean"),
        ko_loss_rate=("ko_tko_loss", "mean"),
        mean_damage_exposure=("damage_exposure", "mean"),
        mean_kd_resistance=("knockdown_resistance", "mean"),
        mean_durability=("damage_durability", "mean"),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(frame):,} fighter-side rows to {OUTPUT_PATH}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
